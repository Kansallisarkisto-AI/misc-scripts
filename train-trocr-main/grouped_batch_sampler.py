# GroupedBatchSampler code from torchvision, https://github.com/pytorch/vision/blob/main/references/detection/group_by_aspect_ratio.py
# Under BSD 3-Clause License
# Modified by John Mäkelä at National Archives of Finland

# BSD 3-Clause License

# Copyright (c) Soumith Chintala 2016, 
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import bisect
import copy
import math
from collections import defaultdict, Counter
from itertools import chain, repeat
from typing import Iterable, List, Optional

import numpy as np
import torch
import torch.utils.data
from torch.utils.data.sampler import BatchSampler, Sampler


def _repeat_to_at_least(iterable, n):
    repeat_times = math.ceil(n / len(iterable))
    repeated = chain.from_iterable(repeat(iterable, repeat_times))
    return list(repeated)


class GroupedBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    Args:
        sampler (Sampler): Base sampler.
        group_ids (list[int]): If the sampler produces indices in range [0, N),
            `group_ids` must be a list of `N` ints which contains the group id of each sample.
            The group ids must be a continuous set of integers starting from
            0, i.e. they must be in the range [0, num_groups).
        batch_size (int): Size of mini-batch.
    """

    def __init__(self, sampler, group_ids, batch_size, drop_last=True, is_eval=False):
        if not isinstance(sampler, Sampler):
            raise ValueError(f"sampler should be an instance of torch.utils.data.Sampler, but got sampler={sampler}")
        self.sampler = sampler
        self.group_ids = group_ids
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.is_eval = is_eval

    def __iter__(self):
        buffer_per_group = defaultdict(list)
        samples_per_group = defaultdict(list)

        num_batches = 0
        for idx in self.sampler:
            group_id = self.group_ids[idx]
            buffer_per_group[group_id].append(idx)
            samples_per_group[group_id].append(idx)
            if len(buffer_per_group[group_id]) == self.batch_size:  # yield here during training and eval
                yield buffer_per_group[group_id]
                num_batches += 1
                del buffer_per_group[group_id]
            assert len(buffer_per_group[group_id]) < self.batch_size
        
        if self.is_eval:  # yield leftovers here during eval
            for group_id, buf in buffer_per_group.items():
                if len(buf) > 0:
                    yield buf
                    num_batches += 1

        # now we have run out of elements that satisfy
        # the group criteria, let's return the remaining
        # elements so that the size of the sampler is
        # deterministic
        expected_num_batches = len(self)
        num_remaining = expected_num_batches - num_batches

        # repeat samples if not drop_last
        if not self.drop_last:
            if num_remaining > 0:
                # for the remaining batches, take first the buffers with the largest number
                # of elements
                for group_id, _ in sorted(buffer_per_group.items(), key=lambda x: len(x[1]), reverse=True):
                    remaining = self.batch_size - len(buffer_per_group[group_id])
                    samples_from_group_id = _repeat_to_at_least(samples_per_group[group_id], remaining)
                    buffer_per_group[group_id].extend(samples_from_group_id[:remaining])
                    assert len(buffer_per_group[group_id]) == self.batch_size
                    yield buffer_per_group[group_id]
                    num_remaining -= 1
                    if num_remaining == 0:
                        break
            assert num_remaining == 0

    def __len__(self):
        if self.is_eval:
            indices = list(self.sampler)
            counts = Counter(self.group_ids[i] for i in indices)
            return sum(math.ceil(c / self.batch_size) for c in counts.values())

        # possibly approximate
        return len(self.sampler) // self.batch_size
    

class GlobalPaddedGroupedBatchSampler(Sampler[List[int]]):
    """
    Build group-homogeneous batches globally, optionally fill partial batches by repeating
    samples within each group (no drop), then pad the *global* batch list once to a multiple
    of world_size, and finally shard batches by rank.

    DDP-safe: every rank yields the same number of batches, and no per-group 32x padding.
    """

    def __init__(
        self,
        dataset_len: int,
        group_ids: Iterable[int],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        fill_partial_by_repeating: bool = True,  # if False + drop_last=False, yields smaller last batches per group
        deterministic_group_order: bool = False,  # sort group ids
    ):
        self.dataset_len = int(dataset_len)
        self.group_ids = list(group_ids)
        assert len(self.group_ids) == self.dataset_len, "group_ids must have length dataset_len"

        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.fill_partial_by_repeating = bool(fill_partial_by_repeating)
        self.deterministic_group_order = bool(deterministic_group_order)

        self.epoch = 0

        self._built_epoch: Optional[int] = None
        self._batches_for_rank: Optional[List[List[int]]] = None

    def set_epoch(self, epoch: int) -> None:
        """Call this once per epoch (like DistributedSampler.set_epoch)."""
        self.epoch = int(epoch)
        # Force rebuild next iteration
        self._built_epoch = None
        self._batches_for_rank = None

    @staticmethod
    def _dist_info():
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def _build(self):
        rank, world_size = self._dist_info()

        # 1) global index order (same on all ranks)
        indices = list(range(self.dataset_len))
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(self.dataset_len, generator=g).tolist()
            indices = [indices[i] for i in perm]

        # 2) collect per-group indices in that global order
        per_group = defaultdict(list)
        for idx in indices:
            per_group[self.group_ids[idx]].append(idx)

        # 3) build group-homogeneous batches (optionally fill partials)
        group_batches: List[List[int]] = []
        gids = list(per_group.keys())
        if self.deterministic_group_order:
            gids.sort()

        for gid in gids:
            idxs = per_group[gid]
            if not idxs:
                continue

            # full batches
            for i in range(0, len(idxs), self.batch_size):
                chunk = idxs[i : i + self.batch_size]
                if len(chunk) == self.batch_size:
                    group_batches.append(chunk)
                else:
                    # partial
                    if self.drop_last:
                        continue
                    if self.fill_partial_by_repeating:
                        # repeat from the start of this group's idxs (or chunk) to fill
                        needed = self.batch_size - len(chunk)
                        # using idxs gives stable repetition even if chunk is tiny
                        reps = (idxs * ((needed // len(idxs)) + 1))[:needed]
                        group_batches.append(chunk + reps)
                    else:
                        # yield smaller batch (ONLY safe if your training loop can handle variable batch size
                        # and you don't do per-batch collectives that assume fixed shapes)
                        group_batches.append(chunk)

        # 4) pad the *global* batch list ONCE to a multiple of world_size
        #    (very small overhead: at most world_size-1 batches per epoch)
        if len(group_batches) == 0:
            # avoid division by zero / weirdness
            batches_for_rank = []
        else:
            rem = len(group_batches) % world_size
            if rem != 0:
                need = world_size - rem
                group_batches = group_batches + group_batches[-need:]

            # 5) shard by rank
            batches_for_rank = group_batches[rank::world_size]

        self._batches_for_rank = batches_for_rank
        self._built_epoch = self.epoch

    def __iter__(self):
        if self._batches_for_rank is None or self._built_epoch != self.epoch:
            self._build()
        yield from self._batches_for_rank

    def __len__(self) -> int:
        if self._batches_for_rank is None or self._built_epoch != self.epoch:
            self._build()
        return len(self._batches_for_rank)


import math
from collections import defaultdict
import torch
import torch.distributed as dist
from torch.utils.data import Sampler

class DistributedGroupedBatchSampler(Sampler):
    """
    Deterministic grouped batching *then* distribute batches across ranks.

    - Groups samples by group_id (e.g. your bin_id) into batches of batch_size.
    - Optionally includes partial (incomplete) batches.
    - Makes total #batches divisible by world_size by either:
        * drop_remainder_batches=True  -> drops the last few batches
        * drop_remainder_batches=False -> pads by repeating last batches (duplicates!)
    """

    def __init__(
        self,
        dataset_len: int,
        group_ids: list[int],
        batch_size: int,
        include_partial_batches: bool = True,
        drop_remainder_batches: bool = True,
        shuffle: bool = False,
        seed: int = 0,
    ):
        self.dataset_len = dataset_len
        self.group_ids = group_ids
        self.batch_size = batch_size
        self.include_partial_batches = include_partial_batches
        self.drop_remainder_batches = drop_remainder_batches
        self.shuffle = shuffle
        self.seed = seed

        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        self._batches = None  # cache

    def _build_batches(self):
        # global deterministic index order (same on every rank)
        indices = list(range(self.dataset_len))

        if self.shuffle:
            self.seed += 1
            g = torch.Generator()
            g.manual_seed(self.seed)
            perm = torch.randperm(self.dataset_len, generator=g).tolist()
            indices = [indices[i] for i in perm]

        buffers = defaultdict(list)
        batches = []

        for idx in indices:
            gid = self.group_ids[idx]
            buf = buffers[gid]
            buf.append(idx)
            if len(buf) == self.batch_size:
                batches.append(buf)
                buffers[gid] = []

        if self.include_partial_batches:
            for buf in buffers.values():
                if buf:
                    batches.append(buf)

        # Make total number of batches divisible by world_size so each rank gets same count
        if self.world_size > 1:
            rem = len(batches) % self.world_size
            if rem != 0:
                if self.drop_remainder_batches:
                    batches = batches[:-rem]  # drop last few batches
                else:
                    # pad by repeating last batch(es) (duplicates!)
                    need = self.world_size - rem
                    batches.extend(batches[-1:] * need)

        self._batches = batches

    def __iter__(self):
        if self._batches is None:
            self._build_batches()

        # shard by batches, not samples
        for b in self._batches[self.rank :: self.world_size]:
            yield b

    def __len__(self):
        if self._batches is None:
            self._build_batches()
        return len(self._batches) // self.world_size


class SyncGroupedBatchSampler:
    """
    Group -> batch -> pad to world_size -> shard within group,
    so that at step t all ranks process the same group_id.
    """

    def __init__(
        self,
        dataset_len: int,
        group_ids,
        batch_size: int,
        include_partial_batches: bool = False,
        shuffle: bool = False,
        seed: int = 0,
        pad_by_repeating: bool = True,   # True avoids uneven steps; duplicates last batches in group
    ):
        self.dataset_len = dataset_len
        self.group_ids = list(group_ids)
        self.batch_size = batch_size
        self.include_partial_batches = include_partial_batches
        self.shuffle = shuffle
        self.seed = seed
        self.pad_by_repeating = pad_by_repeating

        self._batches_for_rank = None
        self.epoch = 0
    
    def set_epoch(self, epoch: int) -> None:
        """Call this once per epoch"""
        self.epoch = int(epoch)
        # Force rebuild next iteration
        self._batches_for_rank = None

    def _build(self):
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        
        print(f"[rank? {dist.get_rank() if dist.is_initialized() else 'NA'}] dist.init={dist.is_initialized()}  ws={dist.get_world_size() if dist.is_initialized() else 1}")
        
        indices = list(range(self.dataset_len))
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(self.dataset_len, generator=g).tolist()
            indices = [indices[i] for i in perm]

        # collect indices by group in global order
        per_group = defaultdict(list)
        for idx in indices:
            per_group[self.group_ids[idx]].append(idx)

        # build batches per group
        group_batches = {}
        for gid, idxs in per_group.items():
            batches = [idxs[i:i+self.batch_size] for i in range(0, len(idxs), self.batch_size)]
            if not self.include_partial_batches and batches and len(batches[-1]) < self.batch_size:
                batches = batches[:-1]
            group_batches[gid] = batches

        # deterministic group order
        gids_sorted = sorted(group_batches.keys())

        # now create per-rank batch stream, synchronized by group
        batches_for_rank = []
        for gid in gids_sorted:
            batches = group_batches[gid]
            if not batches:
                continue

            # pad within this group to multiple of world_size
            rem = len(batches) % self.world_size
            if rem != 0:
                need = self.world_size - rem
                if self.pad_by_repeating:
                    batches = batches + batches[-need:]
                else:
                    batches = batches[:-rem]  # drop

            # shard within this group in lockstep
            for i in range(self.rank, len(batches), self.world_size):
                batches_for_rank.append(batches[i])

        self._batches_for_rank = batches_for_rank

    def __iter__(self):
        if self._batches_for_rank is None:
            self._build()
        yield from self._batches_for_rank

    def __len__(self):
        if self._batches_for_rank is None:
            self._build()
        return len(self._batches_for_rank)
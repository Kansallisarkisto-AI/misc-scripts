import os
# Set GPU 0 as the default device used for running the code
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, VisionEncoderDecoderConfig
from transformers.models.deit.modeling_deit import DeiTPatchEmbeddings, DeiTEmbeddings
from transformers.models.vit.modeling_vit import ViTPatchEmbeddings, ViTEmbeddings
from transformers.models.swin.modeling_swin import SwinPatchEmbeddings, SwinEmbeddings
import torch.nn.functional as F
import onnxruntime as ort
from pathlib import Path
import torch.nn as nn
from PIL import Image
import numpy as np
import torch
import onnx
import json

def apply_deit_custom_size_patches():
    """Apply patches to DeiT model to support custom image sizes"""
    
    def deit_patch_forward(self, pixel_values, interpolate_pos_encoding=None):
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings
    
    def deit_embeddings_forward(self, pixel_values, bool_masked_pos=None, interpolate_pos_encoding=None):
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        distillation_tokens = self.distillation_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, distillation_tokens, embeddings), dim=1)
        
        patch_size = self.patch_embeddings.patch_size[0]
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
        num_patches = num_patches_h * num_patches_w
        
        pos_embed = self.position_embeddings
        
        if num_patches + 2 != pos_embed.shape[1]:
            special_pos_embed = pos_embed[:, :2, :]
            patch_pos_embed = pos_embed[:, 2:, :]
            
            orig_size = int(patch_pos_embed.shape[1] ** 0.5)
            embed_dim = patch_pos_embed.shape[2]
            
            patch_pos_embed = patch_pos_embed.reshape(1, orig_size, orig_size, embed_dim)
            patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
            patch_pos_embed = F.interpolate(patch_pos_embed,
                                          size=(num_patches_h, num_patches_w),
                                          mode='bicubic',
                                          align_corners=False)
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, num_patches, embed_dim)
            
            pos_embed = torch.cat([special_pos_embed, patch_pos_embed], dim=1)
        
        embeddings = embeddings + pos_embed
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    DeiTPatchEmbeddings.forward = deit_patch_forward
    DeiTEmbeddings.forward = deit_embeddings_forward

def apply_vit_custom_size_patches():
    """Apply patches to ViT model to support custom image sizes"""
    
    def vit_patch_forward(self, pixel_values, interpolate_pos_encoding=None):
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings
    
    def vit_embeddings_forward(self, pixel_values, bool_masked_pos=None, interpolate_pos_encoding=None):
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        
        # Handle position embeddings
        patch_size = self.patch_embeddings.patch_size[0]
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
        num_patches = num_patches_h * num_patches_w
        
        pos_embed = self.position_embeddings
        
        # Check if we need to interpolate position embeddings
        if num_patches + 1 != pos_embed.shape[1]:  # +1 for CLS token
            cls_pos_embed = pos_embed[:, :1, :]
            patch_pos_embed = pos_embed[:, 1:, :]
            
            orig_size = int(patch_pos_embed.shape[1] ** 0.5)
            embed_dim = patch_pos_embed.shape[2]
            
            patch_pos_embed = patch_pos_embed.reshape(1, orig_size, orig_size, embed_dim)
            patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
            patch_pos_embed = F.interpolate(patch_pos_embed,
                                          size=(num_patches_h, num_patches_w),
                                          mode='bicubic',
                                          align_corners=False)
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, num_patches, embed_dim)
            
            pos_embed = torch.cat([cls_pos_embed, patch_pos_embed], dim=1)
        
        embeddings = embeddings + pos_embed
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    ViTPatchEmbeddings.forward = vit_patch_forward
    ViTEmbeddings.forward = vit_embeddings_forward

def apply_swin_custom_size_patches():
    """Apply patches to Swin Transformer to support custom image sizes"""
    
    original_swin_embeddings_forward = SwinEmbeddings.forward
    
    def swin_embeddings_forward(self, pixel_values, **kwargs):
        embeddings, output_dimensions = self.patch_embeddings(pixel_values)
        embeddings = self.norm(embeddings)
        
        batch_size, seq_len, embed_dim = embeddings.shape
        
        if hasattr(self, 'position_embeddings') and self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings
        
        embeddings = self.dropout(embeddings)
        
        return embeddings, output_dimensions
    
    SwinEmbeddings.forward = swin_embeddings_forward

def detect_encoder_type(model):
    """Detect which encoder architecture is being used"""
    encoder_config = model.config.encoder
    model_type = encoder_config.model_type
    
    print(f"Detected encoder type: {model_type}")
    
    # Also check the actual encoder class
    encoder_class_name = model.encoder.__class__.__name__
    print(f"Encoder class: {encoder_class_name}")
    
    return model_type

# Wrapper to remove distillation token from DeiT encoder output
class DeiTEncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    
    def forward(self, pixel_values):
        outputs = self.encoder(pixel_values)
        last_hidden_state = outputs.last_hidden_state
        # Keep CLS token and all patch tokens, remove distillation token
        last_hidden_state = torch.cat([
            last_hidden_state[:, :1, :],  # CLS token
            last_hidden_state[:, 2:, :]   # All patch tokens (skip distillation)
        ], dim=1)
        return last_hidden_state

# Wrapper for ViT encoder (has CLS token, no distillation token)
class ViTEncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    
    def forward(self, pixel_values):
        outputs = self.encoder(pixel_values)
        # ViT: [CLS, patch1, patch2, ...]
        # Keep everything - no distillation token to remove
        return outputs.last_hidden_state

# Wrapper for Swin encoder (no CLS token)
class SwinEncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    
    def forward(self, pixel_values):
        outputs = self.encoder(pixel_values)
        # Swin: [patch1, patch2, ...]
        # No special token handling needed
        return outputs.last_hidden_state

# Wrapper for decoder
class DecoderWrapper(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
    
    def forward(self, input_ids, encoder_hidden_states):
        outputs = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
        )
        return outputs.logits

class TrOCRProcessorCustom(TrOCRProcessor):
    def __init__(self, image_processor, tokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.current_processor = self.image_processor
        self.chat_template = None


def save_processor_compatible(processor, output_path: Path):
    """Save processor in both transformers 4.x and 5.x compatible formats."""
    import json

    # Normal save (produces processor_config.json in 5.x, preprocessor_config.json in 4.x)
    processor.save_pretrained(output_path)

    processor_config_path = output_path / "processor_config.json"
    preprocessor_config_path = output_path / "preprocessor_config.json"

    # Build the flat 4.x-style preprocessor_config.json from the image processor
    ip = processor.image_processor

    processor_type = getattr(ip, "image_processor_type", "ViTImageProcessor")

    # allow BitImageProcessorFast through for DINOv2 models, otherwise replace with ViTImageProcessor as before
    if "BitImageProcessor" not in processor_type:
        processor_type = "ViTImageProcessor"

    flat_config = {
        "image_processor_type": processor_type,  # 4.x class name
        "do_convert_rgb": getattr(ip, "do_convert_rgb", True),
        "do_resize": getattr(ip, "do_resize", True),
        "do_center_crop": getattr(ip, "do_center_crop", False),
        "size": getattr(ip, "size", {"height": 192, "width": 1024}),
        "do_normalize": getattr(ip, "do_normalize", True),
        "image_mean": getattr(ip, "image_mean", [0.5, 0.5, 0.5]),
        "image_std": getattr(ip, "image_std", [0.5, 0.5, 0.5]),
        "do_rescale": getattr(ip, "do_rescale", True),
        "rescale_factor": getattr(ip, "rescale_factor", 1/255),
        "resample": getattr(ip, "resample", 2),
        "data_format": "channels_first",
    }

    # Always write the flat preprocessor_config.json for 4.x compatibility
    with open(preprocessor_config_path, "w", encoding="utf-8") as f:
        json.dump(flat_config, f, indent=2)
    print(f"✓ Saved flat preprocessor_config.json (transformers 4.x compatible)")

    # Remove processor_config.json if 5.x wrote it — having both files causes
    # transformers to merge them and pass image_processor twice to __init__
    if processor_config_path.exists():
        processor_config_path.unlink()
        print("✓ Removed processor_config.json (prevents duplicate argument conflict)")

    print("✓ Processor saved in dual-compatible format")

def export_model(
    processor_path, 
    model_path, 
    output_path, 
    encoder_path, 
    decoder_path, 
    custom_height = 192, 
    custom_width = 1024,
    opset_version = 14):
    
    print("Loading model to detect encoder type...")
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    encoder_type = detect_encoder_type(model)
    
    # Apply appropriate patches and choose wrapper based on encoder type
    if encoder_type == "deit":
        print("Applying DeiT custom size patches...")
        apply_deit_custom_size_patches()
        EncoderWrapper = DeiTEncoderWrapper
    elif encoder_type == "vit":
        print("Applying ViT custom size patches...")
        apply_vit_custom_size_patches()
        EncoderWrapper = ViTEncoderWrapper
    elif encoder_type == "swin":
        print("Applying Swin custom size patches...")
        apply_swin_custom_size_patches()
        EncoderWrapper = SwinEncoderWrapper
    else:
        print(f"Warning: Encoder type '{encoder_type}' not explicitly supported.")
        print("Attempting generic export (may not work with custom image sizes)...")
        print("This should be fine with DINOv2-based models (they don't need the patches)")
        # Generic wrapper - just pass through
        class GenericEncoderWrapper(nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder
            def forward(self, pixel_values):
                return self.encoder(pixel_values).last_hidden_state
        EncoderWrapper = GenericEncoderWrapper

    print("Loading processor and model...")

    if encoder_type == "dinov2":  # load slightly patched processor for DINOv2-based models trained on transformers v5.x
        processor = TrOCRProcessorCustom.from_pretrained(processor_path)
    else:
        processor = TrOCRProcessor.from_pretrained(processor_path)
    
    # Prepare model for export
    model.eval()
    model.config.use_cache = False

    # Wrap encoder and decoder
    wrapped_encoder = EncoderWrapper(model.encoder)
    wrapped_encoder.eval()

    wrapped_decoder = DecoderWrapper(model.decoder)
    wrapped_decoder.eval()

    # Create output directory
    output_path.mkdir(exist_ok=True, parents=True)

    print(f"\nExporting ONNX models to {output_path}...")
    print(f"Target image size: {custom_height}x{custom_width}")
    print(f"ONNX opset version: {opset_version}")

    # Export Encoder with batch size > 1 to force dynamic batch
    print("\nExporting encoder...")
    # Use batch size of 2 to force dynamic batch dimension
    dummy_pixel_values = torch.randn(2, 3, custom_height, custom_width)

    with torch.no_grad():
        # Test forward pass
        try:
            test_output = wrapped_encoder(dummy_pixel_values)
            print(f"  Encoder output shape: {test_output.shape}")
        except Exception as e:
            print(f"  Warning: Encoder forward pass failed: {e}")
            print("  Continuing with export...")
        
        torch.onnx.export(
            wrapped_encoder,
            dummy_pixel_values,
            encoder_path,
            input_names=["pixel_values"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "pixel_values": {0: "batch", 2: "height", 3: "width"},
                "last_hidden_state": {0: "batch", 1: "sequence"}
            },
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False,
        )
    print(f"✓ Encoder exported to {encoder_path}")

    # Export Decoder with batch > 1 and sequence > 1
    print("\nExporting decoder...")
    with torch.no_grad():
        encoder_outputs = wrapped_encoder(dummy_pixel_values)
        print(f"  Encoder outputs shape for decoder: {encoder_outputs.shape}")

    # Use batch size 2 and sequence length 4 to force dynamic shapes
    dummy_decoder_input_ids = torch.tensor([
        [model.config.decoder_start_token_id, 100, 200, 300],
        [model.config.decoder_start_token_id, 150, 250, 350]
    ])

    with torch.no_grad():
        # Test forward pass
        try:
            test_decoder_output = wrapped_decoder(dummy_decoder_input_ids, encoder_outputs)
            print(f"  Decoder output shape: {test_decoder_output.shape}")
        except Exception as e:
            print(f"  Warning: Decoder forward pass failed: {e}")
            print("  Continuing with export...")
        
        torch.onnx.export(
            wrapped_decoder,
            (dummy_decoder_input_ids, encoder_outputs),
            decoder_path,
            input_names=["input_ids", "encoder_hidden_states"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "encoder_hidden_states": {0: "batch", 1: "encoder_sequence"},
                "logits": {0: "batch", 1: "sequence"}
            },
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False,
        )

    print(f"✓ Decoder exported to {decoder_path}")

    # Verify the exported ONNX models
    print("\nVerifying exported ONNX models...")

    encoder_onnx = onnx.load(encoder_path)
    decoder_onnx = onnx.load(decoder_path)

    print("\nEncoder inputs:")
    for inp in encoder_onnx.graph.input:
        print(f"  {inp.name}: {[d.dim_param or d.dim_value for d in inp.type.tensor_type.shape.dim]}")
    print("Encoder outputs:")
    for out in encoder_onnx.graph.output:
        print(f"  {out.name}: {[d.dim_param or d.dim_value for d in out.type.tensor_type.shape.dim]}")

    print("\nDecoder inputs:")
    for inp in decoder_onnx.graph.input:
        print(f"  {inp.name}: {[d.dim_param or d.dim_value for d in inp.type.tensor_type.shape.dim]}")
    print("Decoder outputs:")
    for out in decoder_onnx.graph.output:
        print(f"  {out.name}: {[d.dim_param or d.dim_value for d in out.type.tensor_type.shape.dim]}")

    # Save config and processor
    model.config.save_pretrained(output_path)
    save_processor_compatible(processor, output_path)
    print(f"\n✓ Config and processor saved")

    # Test CUDA or MIGraphX availability
    print("\nTesting CUDA/MIGraphX compatibility...")
    try:
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        enc_session = ort.InferenceSession(
            str(encoder_path),
            sess_opts,
            providers=['MIGraphXExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        dec_session = ort.InferenceSession(
            str(decoder_path),
            sess_opts,
            providers=['MIGraphXExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        enc_provider = enc_session.get_providers()[0]
        dec_provider = dec_session.get_providers()[0]
        
        print(f"✓ Encoder will use: {enc_provider}")
        print(f"✓ Decoder will use: {dec_provider}")
        
        if enc_provider not in ['CUDAExecutionProvider', 'MIGraphXExecutionProvider']:
            print(f"  ⚠️  Warning: Encoder not using CUDA/MIGraphX - performance will be slower")
        if dec_provider not in ['CUDAExecutionProvider', 'MIGraphXExecutionProvider']:
            print(f"  ⚠️  Warning: Decoder not using CUDA/MIGraphX - performance will be slower")
        
    except Exception as e:
        print(f"Warning: Could not test CUDA/MIGraphX compatibility: {e}")

    print("\n" + "="*60)
    print("ONNX Export Complete!")
    print("="*60)
    return model, processor


class ONNXTrOCRModel:
    def __init__(self, model_path, use_gpu=True):
        self.model_path = Path(model_path)
        self.processor = TrOCRProcessor.from_pretrained(model_path)
        self.config = VisionEncoderDecoderConfig.from_pretrained(model_path)
        
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = ['MIGraphXExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        
        self.encoder_session = ort.InferenceSession(
            str(self.model_path / "encoder_model.onnx"),
            sess_opts,
            providers=providers
        )
        self.decoder_session = ort.InferenceSession(
            str(self.model_path / "decoder_model.onnx"),
            sess_opts,
            providers=providers
        )
        
        print(f"✓ Encoder using: {self.encoder_session.get_providers()[0]}")
        print(f"✓ Decoder using: {self.decoder_session.get_providers()[0]}")
    
    def generate(self, pixel_values, max_length=128):
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values.numpy()
        
        batch_size = pixel_values.shape[0]
        
        encoder_outputs = self.encoder_session.run(
            None,
            {"pixel_values": pixel_values}
        )[0]
        
        decoder_input_ids = np.full((batch_size, 1), 
                                   self.config.decoder_start_token_id, 
                                   dtype=np.int64)
        
        return self._greedy_decode(decoder_input_ids, encoder_outputs, max_length)
    
    def _greedy_decode(self, decoder_input_ids, encoder_outputs, max_length):
        """Greedy decoding with per-sequence early stopping"""
        batch_size = decoder_input_ids.shape[0]
        finished = np.zeros(batch_size, dtype=bool)
        
        for step in range(max_length):
            decoder_outputs = self.decoder_session.run(
                None,
                {
                    "input_ids": decoder_input_ids,
                    "encoder_hidden_states": encoder_outputs
                }
            )[0]
            
            next_token_logits = decoder_outputs[:, -1, :]
            next_tokens = np.argmax(next_token_logits, axis=-1)
            
            # Mark finished sequences
            just_finished = (next_tokens == self.config.eos_token_id)
            finished = finished | just_finished
            
            # Force PAD for finished sequences
            next_tokens[finished] = self.config.pad_token_id
            
            next_tokens = next_tokens.reshape(-1, 1)
            decoder_input_ids = np.concatenate([decoder_input_ids, next_tokens], axis=1)
            
            # Stop when all are done
            if np.all(finished):
                break
        
        return decoder_input_ids
    
    def predict(self, images, max_length=128):
        if not isinstance(images, list):
            images = [images]
        
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs.pixel_values.numpy()
        
        generated_ids = self.generate(pixel_values, max_length=max_length)
        
        # Decode individually
        texts = []
        for i in range(generated_ids.shape[0]):
            text = self.processor.decode(generated_ids[i], skip_special_tokens=True)
            texts.append(text)
        
        return texts

def test_model(ort_output_dir, custom_height, custom_width):
    print("\n" + "="*60)
    print("Testing ONNX Model")
    print("="*60)

    onnx_model = ONNXTrOCRModel(ort_output_dir, use_gpu=torch.cuda.is_available())

    print(f"\nTest 1: Single image ({custom_height}x{custom_width})")
    test_image = Image.fromarray(
        np.random.randint(0, 255, (custom_height, custom_width, 3), dtype=np.uint8)
    )
    result = onnx_model.predict(test_image)
    print(f"Result: '{result[0]}'")

    print(f"\nTest 2: Batch of 4 images")
    batch_images = [
        Image.fromarray(np.random.randint(0, 255, (custom_height, custom_width, 3), dtype=np.uint8))
        for _ in range(4)
    ]
    batch_results = onnx_model.predict(batch_images)
    for i, text in enumerate(batch_results):
        print(f"  Image {i+1}: '{text}'")

    print("\n✓ All tests completed!")

if __name__ == "__main__":
    # Custom input image size used in model training
    custom_height = 182
    custom_width = 1022

    model_path = "/scratch/project_462001175/trocr/outputs/proper-cyrillic-synthetic-rurobertalarge-138mil-32node-5e05-dinov2large"
    processor_path = "/scratch/project_462001175/trocr/outputs/proper-cyrillic-synthetic-rurobertalarge-138mil-32node-5e05-dinov2large"
    ort_output_dir = "/scratch/project_462001175/trocr/onnx/proper-cyrillic-synthetic-rurobertalarge-138mil-32node-5e05-dinov2large-onnx"
    
    output_path = Path(ort_output_dir)
    encoder_path = output_path / "encoder_model.onnx"
    decoder_path = output_path / "decoder_model.onnx"

    model, processor = export_model(
        processor_path, 
        model_path, 
        output_path, 
        encoder_path, 
        decoder_path, 
        custom_height, 
        custom_width,
        opset_version=21
    )

    #del model
    #del processor
    #torch.cuda.empty_cache()
    
    #test_model(model, processor, ort_output_dir, custom_height, custom_width)
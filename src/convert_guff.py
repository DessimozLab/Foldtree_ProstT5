import torch
import numpy as np
import struct
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer
import argparse

def save_model_to_gguf(model_path, output_path):
    """
    Save a retrained ProstT5 model in GGUF (GGML GPU-Friendly Format) for Foldseek compatibility.
    
    Args:
        model_path (str): Path to the retrained ProstT5 model
        output_path (str): Output path for the GGUF file
    """
    
    # Load the retrained model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    
    # Set model to evaluation mode
    model.eval()
    
    # GGUF file header
    magic = b'GGUF'
    version = 3  # GGUF v3 format
    
    with open(output_path, 'wb') as f:
        # Write magic number and version
        f.write(magic)
        f.write(struct.pack('<I', version))
        
        # Write tensor count and metadata count
        state_dict = model.state_dict()
        tensor_count = len(state_dict)
        metadata_count = 4  # vocab_size, embed_dim, num_layers, num_heads
        
        f.write(struct.pack('<Q', tensor_count))
        f.write(struct.pack('<Q', metadata_count))
        
        # Write metadata
        metadata = {
            'vocab_size': len(tokenizer.vocab),
            'embed_dim': model.config.d_model,
            'num_layers': model.config.num_layers,
            'num_heads': model.config.num_heads
        }
        
        for key, value in metadata.items():
            key_bytes = key.encode('utf-8')
            f.write(struct.pack('<Q', len(key_bytes)))
            f.write(key_bytes)
            f.write(struct.pack('<I', 4))  # INT32 type
            f.write(struct.pack('<i', value))
        
        # Write tensor info
        for name, tensor in state_dict.items():
            tensor_np = tensor.detach().cpu().float().numpy()
            
            # Write tensor name
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<Q', len(name_bytes)))
            f.write(name_bytes)
            
            # Write tensor dimensions
            f.write(struct.pack('<I', len(tensor_np.shape)))
            for dim in tensor_np.shape:
                f.write(struct.pack('<Q', dim))
            
            # Write tensor type (0 = F32)
            f.write(struct.pack('<I', 0))
            
            # Write tensor offset (will be calculated)
            f.write(struct.pack('<Q', 0))  # Placeholder
        
        # Align to 32-byte boundary
        while f.tell() % 32 != 0:
            f.write(b'\0')
        
        # Write tensor data
        for name, tensor in state_dict.items():
            tensor_np = tensor.detach().cpu().float().numpy()
            f.write(tensor_np.tobytes())

def convert_prostT5_to_gguf(model_dir, output_file):
    """
    Main function to convert ProstT5 model to GGUF format.
    
    Args:
        model_dir (str): Directory containing the retrained ProstT5 model
        output_file (str): Output GGUF file path
    """
    print(f"Converting ProstT5 model from {model_dir} to GGUF format...")
    
    #try:
    save_model_to_gguf(model_dir, output_file)
    print(f"Successfully saved model to {output_file}")
    
    # Verify file was created
    if os.path.exists(output_file):
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"Output file size: {size_mb:.2f} MB")
        
    #except Exception as e:
    #    print(f"Error converting model: {str(e)}")
    #    #print traceback for debugging
    #    import traceback
    #    print(traceback.format_exc())

if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser(description='Convert ProstT5 model to GGUF format')
    parser.add_argument('model_dir', help='Directory containing the retrained ProstT5 model')
    parser.add_argument('output_file', help='Output GGUF file path') 

    args = parser.parse_args()

    convert_prostT5_to_gguf(args.model_dir, args.output_file)
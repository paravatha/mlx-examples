python lora.py --model mlx-community/gemma-3n-E2B-it-lm-4bit \
                --train \
                --iters 10 
                # --save-every 2 \
                # --batch-size 1 \
                # --val-batches 1 \               
                # --adapter-file ./adapters.npz


# python -m mlx_lm.lora \
#     --model /Users/gokdenizgulmez/Library/Mobile\ Documents/com\~apple\~CloudDocs/Transformer\ Models/Safetensors/tiny-random-GemmaForCausalLM \
#     --train \
#     --fine-tune-type full \
#     --data /Users/gokdenizgulmez/Library/Mobile\ Documents/com\~apple\~CloudDocs/Datastes/data_tyni \
#     --iters 10 \
#     --batch-size 1 \
#     --val-batches 1 \
#     --adapter-path /Users/gokdenizgulmez/Desktop/tinyGemma-pretrain
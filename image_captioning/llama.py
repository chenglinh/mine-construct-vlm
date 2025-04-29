import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import os

def load_text(fpaths, by_lines=False):
    with open(fpaths, "r") as fp:
        if by_lines:
            return fp.readlines()
        else:
            return fp.read()

def load_prompt(prompt):
    return load_text(f"prompts/{prompt}.txt")

def main():
    model_name = "Llama-3.2-11B-Vision-Instruct"

    model = MllamaForConditionalGeneration.from_pretrained(
        f"/nfs/turbo/coe-stellayu/clhsieh/Minecraft/ckpt/meta-llama/{model_name}",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(f"meta-llama/{model_name}")

    for image_name in ['barn_house.webp', 'castle_wall.webp', 'greek_house.webp', 'mg_nest.png', 'japanese_house.webp', 'incinerator.png', 'easy-0.png', 'easy-1.png', 'easy-2.png', 'easy-3.png', 'easy-4.png', 'easy-5.png']: # 
    # image_name = 'greek_house.webp' # barn_house.webp, castle_wall.webp, greek_house.webp, mg_nest.png, japanese_house.webp, incinerator.png
        # image_name = f'easy-{i}.png'
        folder = 'easy' if 'easy' in image_name else 'mid'
        img_path = f"/nfs/turbo/coe-stellayu/clhsieh/Minecraft/data/{folder}/{image_name}"
        image_name_without_ext = os.path.splitext(os.path.basename(img_path))[0]
        input_image = Image.open(img_path)

        output_dir = f"output/{model_name}/{folder}"
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = f"{output_dir}/{image_name_without_ext}.txt"
        

        building_description_system = load_prompt("building_description_system")
        building_description_query = load_prompt("building_description_query")
        
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": building_description_system},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": input_image,
                    },
                    {"type": "text", "text": building_description_query},
                ],
            },
        ]
    
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            input_image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)

        output = model.generate(**inputs, max_new_tokens=4800)
        print(processor.decode(output[0]))

        
        # Save the output_text to the file
        with open(output_file_path, "w") as file:
            file.write(processor.decode(output[0]))


if __name__ == "__main__":
    main()

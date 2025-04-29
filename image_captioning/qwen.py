from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import torch

def load_text(fpaths, by_lines=False):
    with open(fpaths, "r") as fp:
        if by_lines:
            return fp.readlines()
        else:
            return fp.read()

def load_prompt(prompt):
    return load_text(f"prompts/{prompt}.txt")


def main(model_name):
    # model_name = 'Qwen2.5-VL-7B-Instruct' # 'Qwen2.5-VL-32B-Instruct-AWQ'  # Qwen2.5-VL-7B-Instruct, Qwen2.5-VL-7B-Instruct-AWQ, Qwen2.5-VL-32B-Instruct, Qwen2.5-VL-32B-Instruct-AWQ
    # torch_dtype = torch.float16 if 'AWQ' in model_name else 'auto'
    # default: Load the model on the available device(s)
    if 'AWQ' in model_name:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            f"/nfs/turbo/coe-stellayu/clhsieh/Minecraft/ckpt/Qwen/{model_name}", torch_dtype=torch.float16, device_map="auto"
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            f"/nfs/turbo/coe-stellayu/clhsieh/Minecraft/ckpt/Qwen/{model_name}", torch_dtype='auto', device_map="auto"
        )
    processor = AutoProcessor.from_pretrained(f"Qwen/{model_name}")

    for image_name in ['barn_house.webp', 'castle_wall.webp', 'greek_house.webp', 'mg_nest.png','japanese_house.webp', 'incinerator.png', 'easy-0.png', 'easy-1.png', 'easy-2.png', 'easy-3.png', 'easy-4.png', 'easy-5.png']: # 
    # image_name = 'greek_house.webp' # barn_house.webp, castle_wall.webp, greek_house.webp, mg_nest.png, japanese_house.webp, incinerator.png
    # image_name = f'easy-{i}.png'
        folder = 'easy' if 'easy' in image_name else 'mid'
        img_path = f"/nfs/turbo/coe-stellayu/clhsieh/Minecraft/data/{folder}/{image_name}"
        image_name_without_ext = os.path.splitext(os.path.basename(img_path))[0]

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
                        "image": img_path,
                    },
                    {"type": "text", "text": building_description_query},
                ],
            },
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=4800)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # print(output_text)

        # Save the output_text as a txt file with the name of image_name without the file type

        # Save the output_text to the file
        with open(output_file_path, "w") as file:
            file.write(output_text[0])


        # Display the image
        # display(Image(filename=img_path))


if __name__ == "__main__":
    model_name_list = ['Qwen2.5-VL-7B-Instruct', 'Qwen2.5-VL-32B-Instruct-AWQ']
    for model_name in model_name_list:
        main(model_name)
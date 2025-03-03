input_file = "data/IIT_HW_Hindi_v1/train.txt"
output_file = "data/processed_IIT_HW_Hindi_v1/filtered_4/filtered_sorted_train.txt"


with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

filtered_lines = [line.strip() for line in lines if line.startswith("HindiSeg/train/4/")]
filtered_lines.sort(key=lambda x: (int(x.split("/")[-2]), int(x.split("/")[-1].split(".")[0])))


with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(filtered_lines))

print(f"Filtered and sorted lines saved to {output_file}")
import json
import os

print("--- Day 52: Preparing the LoRA Fine-Tuning Dataset ---\n")

# 1. Define the System Prompt
# This tells the model exactly what "mode" it is in for these specific examples.
system_prompt = "You are a medical scribe. Convert the doctor's casual notes into formal clinical terminology."

# 2. Construct the Raw Data (Pairs of Casual -> Formal)
# In the real world, you would pull thousands of these from a hospital database.
# For our LoRA test, a small, high-quality batch is enough to teach tone.
raw_data = [
    ("Patient has a really bad headache and a fever.", "Patient presents with severe cephalgia and pyrexia."),
    ("His stomach hurts right below the ribs on the right side.", "Patient reports pain in the right upper quadrant (RUQ) of the abdomen."),
    ("She's breathing super fast and sweating a lot.", "Patient is exhibiting tachypnea and diaphoresis."),
    ("He passed out when he stood up too fast.", "Patient experienced orthostatic syncope."),
    ("Her skin looks kind of yellow.", "Patient presents with jaundice (icterus)."),
    ("He threw up blood this morning.", "Patient reports an episode of hematemesis."),
    ("She has a bloody nose that won't stop.", "Patient is experiencing uncontrolled epistaxis."),
    ("His heart is beating crazy fast.", "Patient is exhibiting tachycardia."),
    ("She has trouble swallowing her food.", "Patient reports experiencing dysphagia."),
    ("He's coughing up thick green stuff.", "Patient is expectorating purulent sputum."),
    ("Her muscles are shrinking and getting weak.", "Patient is exhibiting muscle atrophy and asthenia."),
    ("He has a ringing sound in his ears.", "Patient reports experiencing tinnitus.")
]

print(f"🧬 Total raw examples created: {len(raw_data)}")

# 3. Format the Data for MLX (The 'messages' architecture)
formatted_dataset = []
for casual, formal in raw_data:
    row = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": casual},
            {"role": "assistant", "content": formal}
        ]
    }
    formatted_dataset.append(row)

# 4. Split into Training and Validation Sets
# We reserve the last 2 examples for validation. The AI will NOT train on these.
train_data = formatted_dataset[:-2]
valid_data = formatted_dataset[-2:]

print(f"🎓 Training set size: {len(train_data)} examples")
print(f"🧪 Validation set size: {len(valid_data)} examples")

# 5. Write to JSONL (JSON Lines) Format
# Notice we write each dictionary as a string and manually add a newline character (\n)
dataset_dir = "lora_dataset"
os.makedirs(dataset_dir, exist_ok=True)

train_path = os.path.join(dataset_dir, "train.jsonl")
valid_path = os.path.join(dataset_dir, "valid.jsonl")

with open(train_path, "w") as f:
    for item in train_data:
        f.write(json.dumps(item) + "\n")

with open(valid_path, "w") as f:
    for item in valid_data:
        f.write(json.dumps(item) + "\n")

print("\n" + "=" * 50)
print(f"✅ SUCCESS: Datasets generated in the '{dataset_dir}' folder!")
print("=" * 50)
print("💡 THE ML ENGINEER TAKEAWAY:")
print("Open 'train.jsonl' in your editor. Notice how there are no commas between the lines,")
print("and no opening/closing brackets for the file itself. Every line is a perfectly")
print("isolated JSON object. This allows the MLX training engine to stream millions of rows")
print("into memory one by one without crashing your RAM!")
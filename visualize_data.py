import matplotlib.pyplot as plt
import numpy as np

ear_values = []
frame_counts = []

# Read data from the file
with open(file_path, 'r') as f:
    for line in f:
        # Skip lines that are empty or don't contain valid data
        if not line.strip():
            continue
        
        parts = line.split(", ")
        
        # Ensure the line has both EAR and Frame Count values
        if len(parts) == 2:
            try:
                # Extract EAR and Frame Count values from the line
                ear = float(parts[0].split(": ")[1])
                frame_count = int(parts[1].split(": ")[1])
                
                # Append values to the lists
                ear_values.append(ear)
                frame_counts.append(frame_count)
            except (IndexError, ValueError) as e:
                print(f"Skipping invalid line: {line.strip()} due to error: {e}")

# Now, create the visualization
plt.figure(figsize=(10, 6))

# Plot EAR values
plt.plot(ear_values, label='EAR', color='b', alpha=0.7)
plt.axhline(y=0.07, color='r', linestyle='--', label='EAR Threshold')

# Plot Frame Counts
plt.plot(frame_counts, label='Frame Count', color='g', alpha=0.7)

plt.xlabel('Frame Index')
plt.ylabel('EAR / Frame Count')
plt.title('Driver Sleepiness Detection - EAR and Frame Count over Time')
plt.legend()

# Display the plot
plt.show()
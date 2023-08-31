from torch.utils.data import Dataset
import textwrap


def save_and_print_results(outputs, influences, top_seqs, filename="results.txt"):
    """Prints out the results in a nicely formatted manner to the console and saves to a txt file."""

    # Open the file for writing
    with open(filename, "w") as file:
        for i, output in enumerate(outputs):
            print("\nOutput:", output)
            file.write("\nOutput: " + output + "\n")

            # Loop through the top influential sequences for this output
            for j, (influence, seq) in enumerate(zip(influences[i], top_seqs[i])):
                wrapped_seq = textwrap.fill(seq, width=50)  # Adjust width as needed
                print(
                    f"\nInfluence {j+1}: {influence:.4f}\nTraining Sequence:\n{wrapped_seq}"
                )
                file.write(
                    f"\nInfluence {j+1}: {influence:.4f}\nTraining Sequence:\n{wrapped_seq}\n"
                )

            print("\n" + "-" * 80)
            file.write("\n" + "-" * 80 + "\n")

    print(f"\nResults saved to {filename}")


class StringDataset(Dataset):
    def __init__(self, string_list):
        self.data = string_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def slice_chain_dataset(chain_dataset, limit):
    sliced_data = []

    for i, item in enumerate(chain_dataset):
        if i >= limit:
            break
        sliced_data.append(item)

    return StringDataset(sliced_data)

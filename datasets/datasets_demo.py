# This dataset module leverages most functions and tools from the HuggingFace library with the aim to make these tools accessible to the lay population
# Check out this link for further dataset functionality
# https://colab.research.google.com/github/huggingface/datasets/blob/master/notebooks/Overview.ipynb



import stdatasets as st

# Currently available datasets and metrics
datasets = st.list_datasets()
print(f"?? Currently {len(datasets)} datasets are available on the hub:")
print(datasets)

# Currently available metrics
metrics = st.list_metrics()
print(f"?? Currently {len(metrics)} metrics are available on the hub:")
print(metrics)


# Downloading and loading a dataset
dataset = st.load_dataset('squad', split='validation[:10%]')
print(dataset)

print(f"??Dataset len(dataset): {len(dataset)}")
print("\n??First item 'dataset[0]':")
print(dataset[0])

# Or get slices with several examples:
print("\n??Slice of the two items 'dataset[10:12]':")
print(dataset[10:12])


## Load the CNN dataset
## This dataset appears to have sporadic errors when loading apparently due to an issue revolving aroung being hosted on Google Drive
## This gets into more details about this: https://github.com/huggingface/datasets/issues/873
## Other datasets are available though
#cnn_dataset = st.load_dataset("cnn_dailymail", '3.0.0')
#print(cnn_dataset)



## Load custom datasets
# Currently supports csv, json, pandas, text formats

dataset1 = st.load_dataset('csv', data_files='train.csv')
dataset2 = st.load_dataset('csv', data_files=['train.csv', 'train.csv', 'train.csv'])
dataset3 = st.load_dataset('csv', data_files={'train': ['train.csv', 'train.csv'],
                                              'test': 'train.csv'})


                                              
print(dataset1)                     
print(dataset2)                     
print(dataset3)

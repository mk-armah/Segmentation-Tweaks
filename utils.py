class PARAMS:
    epochs = 500
    train_val_split = 0.70
    train_dir = {"image" :"/content/drive/MyDrive/Coding-Stuffs/Repository/learning_datasets/images_prepped_train",
                 "mask": "/content/drive/MyDrive/Coding-Stuffs/Repository/learning_datasets/annotations_prepped_train"}
    
    test_dir = {"image" :"/content/drive/MyDrive/Coding-Stuffs/Repository/learning_datasets/images_prepped_test",
                "mask": "/content/drive/MyDrive/Coding-Stuffs/Repository/learning_datasets/annotations_prepped_test"}
    
    batch_size = 32
    
    
    
    
    
    
    
    #www.apeer.com for semantic segmentation datalabeling
    # https://github.com/sksq96/pytorch-summary ------> summarizing pytorch models
    
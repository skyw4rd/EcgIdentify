import sys
sys.path.append('..')
import torch
from torchvision import datasets, transforms
import timm
import time
from thop import profile
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from collections import defaultdict
import random
import os
from teacher_model import create_test_teacher_model

def evaluate_per_person(model, dataset, num_images_per_person):
    model.eval()
    
    person_images = defaultdict(list)
    for img_path, label in dataset.samples:
        person_images[label].append(img_path)

    all_person_preds = []
    all_person_targets = []
    total_inference_time = 0
    num_people = len(person_images)

    data_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])

    for person_id, images in person_images.items():
        if len(images) < num_images_per_person:
            selected_images_paths = images
        else:
            selected_images_paths = random.sample(images, num_images_per_person)
        
        person_samples = []
        for img_path in selected_images_paths:
            img = datasets.folder.default_loader(img_path)
            img_tensor = data_transform(img)
            person_samples.append(img_tensor)
            
        if not person_samples:
            num_people -= 1 # Adjust total number of people if skipping
            continue

        data = torch.stack(person_samples).to('cuda')
        
        start_time = time.time()
        with torch.no_grad():
            logits = model.forward(data)
            preds = logits.argmax(dim=1)
        end_time = time.time()
        
        total_inference_time += (end_time - start_time)

        majority_pred = np.bincount(preds.cpu().numpy()).argmax()
        
        all_person_preds.append(majority_pred)
        all_person_targets.append(person_id)
            
    # Calculate metrics
    accuracy = accuracy_score(all_person_targets, all_person_preds)
    report = classification_report(all_person_targets, all_person_preds, output_dict=True, zero_division=0)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']
    
    avg_inference_time_per_person = total_inference_time / num_people if num_people > 0 else 0

    # Calculate FLOPs and Params
    dummy_input = torch.randn(1, 3, 224, 224).to('cuda')
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    
    return accuracy, precision, recall, f1, flops, params, avg_inference_time_per_person

def main():
    datasets_to_test = {
        'ecgid': {
            'path': '../data/ecgid/test', 
            'classes': 89, 
            'images_per_person': 5
        },
        # 'ptb': {
            # 'path': '../data/ptb/test', 
            # 'classes': 289, 
            # 'images_per_person': 5
        # }
    }

    for dataset_name, info in datasets_to_test.items():
        print(f'--- Evaluating on {dataset_name} dataset ---')
        
        dataset = datasets.ImageFolder(root=info['path'])
        
        nb_classes = info['classes']
        model_path = f'../models_para/deit_tiny_patch16_224_{dataset_name}.pth'
        model = timm.create_model('deit_tiny_patch16_224', pretrained=False, num_classes=nb_classes).to('cuda')
        model.load_state_dict(torch.load(model_path), strict=False)
        
        acc, p, r, f1, flops, params, avg_time = evaluate_per_person(model, dataset, info['images_per_person'])
        
        print(f'Person Identification Accuracy (ACC): {acc:.4f}')
        print(f'Precision (P): {p:.4f}')
        print(f'Recall (R): {r:.4f}')
        print(f'F1-score (F1): {f1:.4f}')
        print(f'FLOPs: {flops/1e9:.2f} GFLOPs')
        print(f'Params: {params/1e6:.2f} M')
        print(f'Average inference time per person (Student): {avg_time:.4f}s')

        # --- Evaluate Teacher Model ---
        teacher_model_base_name = 'resnet34' # Using resnet34 as the teacher model base name as per user's request.
        print(f'--- Evaluating Teacher Model ({teacher_model_base_name}) on {dataset_name} dataset ---')
        teacher_model_path = f'../models_para/{teacher_model_base_name}.a1_in1k_{dataset_name}_baseline.pth'
        
        if not os.path.exists(teacher_model_path):
            print(f"Warning: Teacher model weights not found at {teacher_model_path}. Skipping teacher evaluation for {dataset_name}.")
            print("-" * 50) # Separator for clarity
            continue # Skip to the next dataset

        teacher_model = timm.create_model('resnet34.a1_in1k', nb_classes=nb_classes)
        teacher_model.load_state_dict(torch.load(teacher_model_path), strict=False)
        teacher_model.to('cuda')

        _, _, _, _, _, _, teacher_avg_time = evaluate_per_person(teacher_model, dataset, info['images_per_person'])
        
        print(f'Average inference time per person (Teacher): {teacher_avg_time:.4f}s')
        
        if avg_time > 0:
            print(f'Teacher model is {teacher_avg_time / avg_time:.2f} times slower than Student model.')
        print("-" * 50) # Separator for clarity

if __name__ == '__main__':
    main()

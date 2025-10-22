import argparse
import json
import os
import glob
import numpy as np

# ------------------------------------
# Définis ici ton mapping de classes
# ------------------------------------
# Exemple : 
# - "unknown" -> None (ignoré)
# - tu peux fusionner plusieurs labels vers une même classe
MAPPING = {
    "detached": "detached",
    "occluseDetached": "detached",
    "occlusedDetached": "detached",
    "occlusedAttached": "occludedAttached",
    "unknown": "occludedAttached",
    "attachedSide": "attached",
    "attached": "attached"
}
# Mapping fixe pour les IDs
category_ids = {
    "detached": 0,
    "occludedAttached": 1,
    "attached": 2
}


def labelme2coco(labelme_folder, output_json):
    data = {
        "info": {
            "description": "Test"
        },
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Classes finales (après mapping)
    final_labels = sorted(set(v for v in MAPPING.values() if v is not None))

    # Ajout des catégories COCO
    # for i, label in enumerate(final_labels, 1):
    #     data["categories"].append({
    #         "id": i,
    #         "name": label,
    #         "supercategory": "bubble"
    #     })
    # Création des catégories COCO avec IDs fixes
    data["categories"] = [
        {"id": category_ids[label], "name": label, "supercategory": "bubble"}
        for label in sorted(set(MAPPING.values()) if MAPPING else [])
    ]

    ann_id = 1
    for i, json_file in enumerate(glob.glob(os.path.join(labelme_folder, "*.json"))):
        with open(json_file) as f:
            label_data = json.load(f)

        # Image info
        data["images"].append({
            "id": i,
            "file_name": label_data["imagePath"],
            "height": label_data["imageHeight"],
            "width": label_data["imageWidth"]
        })

        # Annotations
        for shape in label_data["shapes"]:
            raw_label = shape["label"]
            label = MAPPING.get(raw_label, None)
            if label is None:  # on ignore cette classe
                continue

            points = np.asarray(shape["points"])
            xmin, ymin = points.min(axis=0)
            xmax, ymax = points.max(axis=0)
            width = xmax - xmin
            height = ymax - ymin
            # calcul de l'aire de la forme
            x = points[:,0]
            y = points[:,1]
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

            segmentation = [points.flatten().tolist()]
            ann = {
                "id": ann_id,
                "image_id": i,
                # "category_id": final_labels.index(label) + 1,
                "category_id": category_ids[label],
                "segmentation": segmentation,
                "bbox": [float(xmin), float(ymin), float(width), float(height)],
                "area": area,
                "iscrowd": 0
            }
            data["annotations"].append(ann)
            ann_id += 1

    with open(output_json, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Conversion terminée : {len(data['images'])} images, {len(data['annotations'])} annotations")
    print(f"Classes retenues : {final_labels}")



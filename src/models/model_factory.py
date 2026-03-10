import torch.nn as nn
import torchvision.models as models
import timm

class MosquitoModelFactory:
    """
    Factory per istanziare modelli di Deep Learning.
    Gestisce il download dei pesi (ImageNet o random) e adatta l'ultimo
    strato (classifier) al numero di classi del nostro task.
    """
    
    @staticmethod
    def create_model(model_name: str, pretrained: bool = True, num_classes: int = 4) -> nn.Module:
        """
        Crea e restituisce il modello richiesto.
        
        Args:
            model_name (str): Nome del modello (es. 'resnet50', 'efficientnet_b0').
            pretrained (bool): Se True, carica i pesi ImageNet. Se False, pesi random.
            num_classes (int): Numero di nodi nell'ultimo strato di output (default 4).
        """
        model_name = model_name.lower().strip()
        print(f"[ModelFactory] Inizializzazione modello: {model_name} (Pretrained: {pretrained})")

        # -------------------------------------------------------------------
        # 1. Famiglia ResNet (Nativi in torchvision)
        # -------------------------------------------------------------------
        if model_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
            # Sostituiamo il fully connected layer (fc)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            return model

        elif model_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            return model

        # -------------------------------------------------------------------
        # 2. Famiglia EfficientNet (Nativi in torchvision)
        # -------------------------------------------------------------------
        elif model_name == "efficientnetb0" or model_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)
            # Nelle EfficientNet, l'ultimo layer si chiama 'classifier' ed è un modulo Sequential
            # Il layer Lineare è l'ultimo elemento [1] di questa sequenza
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
            return model

        # -------------------------------------------------------------------
        # 3. Famiglia MobileNetV2 (Nativi in torchvision)
        # -------------------------------------------------------------------
        elif model_name == "mobilenetv2" or model_name == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            model = models.mobilenet_v2(weights=weights)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
            return model

        # -------------------------------------------------------------------
        # 4. Modelli Esterni tramite TIMM (NASNetMobile, MobileNet V1)
        # -------------------------------------------------------------------
        elif model_name in ["nasnetamobile", "nasnetmobile"]:
            # Usiamo timm che gestisce in automatico la sostituzione dell'head 
            # tramite il parametro num_classes!
            model = timm.create_model('nasnetamobile', pretrained=pretrained, num_classes=num_classes)
            return model

        elif model_name in ["mobilenet", "mobilenetv1", "mobilenet_v1"]:
            # Anche il V1 non è in torchvision, usiamo timm
            model = timm.create_model('mobilenetv1_100', pretrained=pretrained, num_classes=num_classes)
            return model

        else:
            raise ValueError(f"Modello '{model_name}' non supportato dalla Factory. "
                             f"Scegli tra: resnet18, resnet50, efficientnetb0, mobilenetv2, nasnetmobile, mobilenet.")
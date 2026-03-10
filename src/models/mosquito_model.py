import torch
import torch.nn as nn
import timm

class MosquitoNet(nn.Module):
    """
    Rete Neurale Custom per MOSQ-AI.

    - freeze_weights=True  → backbone estratto senza head originale (num_classes=0,
                             global_pool=''), pesi congelati, head custom aggiunta:
                             [GAP → Dropout(0.2) → Linear(in_channels, num_classes)].

    - freeze_weights=False → rete caricata nativamente con num_classes=num_classes,
                             tutti i pesi aggiornabili, nessuna head custom aggiuntiva.
    """
    def __init__(self, model_name: str, pretrained: bool = True, num_classes: int = 4, freeze_weights: bool = False):
        super().__init__()

        timm_name = self._map_to_timm_name(model_name)
        self.freeze_weights = freeze_weights
        print(f"[MosquitoNet] Backbone: {timm_name} | Pretrained: {pretrained} | Freeze: {freeze_weights}")

        if freeze_weights:
            # Backbone puro (solo feature maps), head custom separata
            self.backbone = timm.create_model(
                timm_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool=''
            )
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("[MosquitoNet] Pesi del backbone congelati.")

            # Calcolo dinamico dei canali in uscita dal backbone
            dummy = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                in_channels = self.backbone(dummy).shape[1]

            print(f"[MosquitoNet] Custom Head: {in_channels} → GAP → Dropout(0.2) → FC({num_classes})")
            self.custom_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(p=0.2),
                nn.Linear(in_channels, num_classes)
            )
        else:
            # Rete nativa con la head originale sostituita da num_classes classi
            self.backbone = timm.create_model(
                timm_name,
                pretrained=pretrained,
                num_classes=num_classes
            )
            self.custom_head = None
            print(f"[MosquitoNet] Head nativa del backbone con {num_classes} classi di output.")

    def _map_to_timm_name(self, name: str) -> str:
        """Mappa la stringa utente ai nomi esatti richiesti da timm."""
        name = name.lower().strip()
        mapping = {
            "resnet18": "resnet18",
            "resnet50": "resnet50",
            "efficientnetb0": "efficientnet_b0",
            "mobilenet": "mobilenetv1_100",        # V1 classica
            "mobilenetv2": "mobilenetv2_100",      # V2 classica
            "nasnetmobile": "nasnetamobile"
        }
        if name not in mapping:
            raise ValueError(f"Modello '{name}' non supportato. Usa: {list(mapping.keys())}")
        return mapping[name]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Restituisce i LOGITS grezzi (senza Softmax) richiesti da nn.CrossEntropyLoss.
        """
        if self.custom_head is not None:
            # Modalità freeze: backbone → feature maps → head custom
            return self.custom_head(self.backbone(x))
        else:
            # Modalità fine-tuning: forward nativo della rete
            return self.backbone(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict:
        """
        Aggregatore Finale per l'Inferenza (Uso in produzione/trappola).
        Prende l'immagine, esegue il forward, applica il Softmax e restituisce la classe finale.
        """
        self.eval() # Assicuriamoci che il modello sia in modalità inferenza (disabilita il Dropout)
        
        logits = self.forward(x)
        
        # Applica Softmax per ottenere percentuali [0.0 - 1.0]
        probabilities = torch.softmax(logits, dim=1)
        
        # Argmax per trovare l'indice con la probabilità più alta (la classe vincente)
        predicted_class_idx = torch.argmax(probabilities, dim=1)
        
        return {
            "predicted_class": predicted_class_idx.cpu().numpy(),
            "probabilities": probabilities.cpu().numpy(),
            "logits": logits.cpu().numpy()
        }
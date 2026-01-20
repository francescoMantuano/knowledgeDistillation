# knowledgeDistillation 
 
# conviene utilizzare batch size diversi per teacher/student? NO

# dovrei fare freezing di qualche layer del teacher? MAYBE

# dovrei usare meno classi o uso tutte le 120? TUTTE

# eventualmente utilizzo WeightedRandomSampler per il class imbalance?

# metto anche il tempo di inferenza? SI

# avendo a disposizione una GPU converrebbe utilizzare mixed precision o non dovrei usarla per maggiore generalità?

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

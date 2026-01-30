#definisco l'hook che si agganci allo specifico layer per poter effettuare feature comparison
class FeatureHook:
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        self.features = output

    #per resettare le feature a ogni batch, altrimenti in caso di bug riutilizzerei features vecchie
    def clear(self):
        self.features = None

    def close(self):
        self.hook.remove()
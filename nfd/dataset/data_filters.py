
def get_n(name): 
    name = name.split('/')[-1]
    return int(name.split('-')[0])

def val_line(x):
    n = get_n(x) 
    # return (n>=50) and (n<1000) # test no 0-50
    return (n<1000)

def val_curve(x):
    return True

    

def train_10k(x): 
    n = get_n(x)
    # return (n<350) or ((10000<=n) and (n<=10350))
    return (n<=350)# or ((10000<=n) and (n<=10350))
val_10k = val_line



def train_5k(x): 
    n = get_n(x)
    # return (n<350) or ((10000<=n) and (n<=10350))
    return (n<=170)# or ((10000<=n) and (n<=10350))
val_5k = val_line



def train_10k10kc(x): 
    n = get_n(x)
    return (n<350) or ((10000<=n) and (n<=10350))
    # return (n<=700)# or ((10000<=n) and (n<=10350))
val_10k10kc = val_curve    

def train_30k30kc(x): 
    return True
val_30k30kc = val_curve

def train_5k5kc(x): 
    n = get_n(x)
    # return (n<350) or ((10000<=n) and (n<=10350))
    return (n<=170) or ((10000<=n) and (n<=10170))
val_5k5kc = val_curve

def train_2k2kc(x): 
    n = get_n(x)
    # return (n<350) or ((10000<=n) and (n<=10350))
    return (n<=80) or ((10000<=n) and (n<=10080))
val_2k2kc = val_curve



# === used line ===
def train_2k(x): 
    n = get_n(x)
    # return (n<350) or ((10000<=n) and (n<=10350))
    return (n<=69)# or ((10000<=n) and (n<=10350))
val_2k = val_line


def train_6k(x): 
    n = get_n(x)
    # return (n<350) or ((10000<=n) and (n<=10350))
    return (n<=207)# or ((10000<=n) and (n<=10350))
val_6k = val_line


def train_20k(x): 
    n = get_n(x)
    # return (n<350) or ((10000<=n) and (n<=10350))
    return (n<=684)# or ((10000<=n) and (n<=10350))
val_20k = val_line

def train_60k(x):
    n = get_n(x)
    # return (n<350) or ((10000<=n) and (n<=10350))
    return (n<=2244)# or ((10000<=n) and (n<=10350)) # 1800-2000 is missing!
val_60k = val_line


# =========== Used Curve ==========
def train_2kc(x): 
    n = get_n(x)
    # return (n<350) or ((10000<=n) and (n<=10350))
    return (n<=36) or ((10000<=n) and (n<=10033))
val_2kc = val_curve

def train_6kc(x): 
    n = get_n(x)
    # return (n<350) or ((10000<=n) and (n<=10350))
    return (n<=104) or ((10000<=n) and (n<=10100))
val_6kc = val_curve

def train_20kc(x): 
    n = get_n(x)
    # return (n<350) or ((10000<=n) and (n<=10350))
    return (n<=343) or ((10000<=n) and (n<=10333))
val_20kc = val_curve

def train_40kc(x): 
    n = get_n(x)
    # return (n<350) or ((10000<=n) and (n<=10350))
    return (n<=684) or ((10000<=n) and (n<=10667))
val_40kc = val_curve

def train_60kc(x): 
    n = get_n(x)
    # return (n<350) or ((10000<=n) and (n<=10350))
    return (n<=1024) or ((10000<=n) and (n<=11000))
val_60kc = val_curve
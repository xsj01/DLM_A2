import inspect
from omegaconf import OmegaConf
import pandas as pd
import fcntl
import io

def save_cfg(path, cfg):
    if type(cfg) is dict:
        cfg = OmegaConf.create(cfg)
    # with open('test.yaml', 'w') as f:
    OmegaConf.save(config=cfg, f=path)

def save_codes(path, *args):
    with open(path, "w") as f:
        for obj in args:
            if inspect.ismodule(obj):
                source = inspect.getsource(obj)
            else:
                obj = inspect.getmodule(obj)
                source = inspect.getsource(obj)
            f.write('#  '+'='*60+'\n')
            f.write('#  '+obj.__name__+'\n')
            f.write('#  '+obj.__file__+'\n')
            f.write('#  '+'='*60+'\n\n')
            f.write(source)
            f.write('\n\n')

def save_result_excel(data, filename, sheet_name=0, lock=False):
    def add_one(data,df):
        new_row = pd.DataFrame([data.values()], columns=data.keys())
        df = df.append(new_row, ignore_index=True)
        return df
    def save_all(data,file_obj):
        df = pd.read_excel(file_obj, sheet_name)

        # if lock:
        #     file_obj.seek(0)

        if type(data) is list:
            for dd in data:
                df = add_one(dd, df)
        else:
            df = add_one(data, df)

        df.to_excel(file_obj, index=False)

    # if lock:
    #     with open(filename, 'rb+') as file:
    #         fcntl.flock(file, fcntl.LOCK_EX)

    #         file_obj = io.BytesIO(file.read())
            

    #         save_all(data, file_obj)
            
    #         file.seek(0)
    #         file.write(file_obj.getvalue())
    #         # file.truncate()

    #         fcntl.flock(file, fcntl.LOCK_UN)
    # else:
    save_all(data, filename)

if __name__ == '__main__':
    def test_save_codes():
        from ndf import train
        from nfd.utils import log_utils
        save_codes('./test.log', train, log_utils)
    # test_save_codes()
    def test_save_result_excel():
        data={
            'time':'333',
            'task':'333',
            'model':'333',
            'run_id':'333',
            'loss':'333',
            'model_path':'jbjbjb',
        }
        save_result_excel(data, 'results/eval/pred.xlsx', lock=True)
        pass
    test_save_result_excel()

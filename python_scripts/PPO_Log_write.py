import json
import re
from datetime import datetime

import numpy as np


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.void):
            return None
        if isinstance(obj, datetime):
            return obj.isoformat()
        if 'torch.Tensor' in str(type(obj)):
            try:
                return obj.cpu().detach().numpy().tolist()
            except Exception:
                pass
        return super().default(obj)


class Log_write:
    def __init__(self):
        self.data = {
            'start time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'save time': [],
            'records': [],
        }

    def _normalize_scalar(self, value):
        if hasattr(value, 'item') and callable(value.item):
            try:
                value = value.item()
            except Exception:
                pass
        if isinstance(value, np.ndarray) and value.size == 1:
            return value.item()
        return value

    def add_cycle_record(self, episode_num=None, action_type=None, decision_reward=None,
                         catch_reward=None, tai_reward=None, total_reward=None,
                         loss_discrete=None, loss_continuous=None, **extra_fields):
        record = {
            'episode_num': self._normalize_scalar(episode_num),
            'action_type': action_type,
            'decision_reward': self._normalize_scalar(decision_reward),
            'catch_reward': self._normalize_scalar(catch_reward),
            'tai_reward': self._normalize_scalar(tai_reward),
            'total_reward': self._normalize_scalar(total_reward),
            'loss_discrete': self._normalize_scalar(loss_discrete),
            'loss_continuous': self._normalize_scalar(loss_continuous),
        }
        for key, value in extra_fields.items():
            record[key] = self._normalize_scalar(value)
        self.data['records'].append(record)

    def log_cycle(self, file_path, episode_num=None, action_type=None, decision_reward=None,
                  catch_reward=None, tai_reward=None, total_reward=None,
                  loss_discrete=None, loss_continuous=None, **extra_fields):
        self.add_cycle_record(
            episode_num=episode_num,
            action_type=action_type,
            decision_reward=decision_reward,
            catch_reward=catch_reward,
            tai_reward=tai_reward,
            total_reward=total_reward,
            loss_discrete=loss_discrete,
            loss_continuous=loss_continuous,
            **extra_fields,
        )
        self.save(file_path)

    def add_loss(self, loss_discrete, loss_continuous):
        self.add_cycle_record(loss_discrete=loss_discrete, loss_continuous=loss_continuous)

    def add_reward(self, decision_reward=None, catch_reward=None, tai_reward=None, total_reward=None):
        self.add_cycle_record(
            decision_reward=decision_reward,
            catch_reward=catch_reward,
            tai_reward=tai_reward,
            total_reward=total_reward,
        )

    def add_action_type(self, action_type):
        self.add_cycle_record(action_type=action_type)

    def add(self, **kwargs):
        if not kwargs:
            return
        if not self.data['records']:
            self.data['records'].append({})
        record = self.data['records'][-1]
        for key, value in kwargs.items():
            record[key] = self._normalize_scalar(value)

    def clear(self):
        return

    def reset(self):
        self.__init__()

    def get(self, key):
        return self.data.get(key, [])

    def save_catch(self, file_path):
        self.save(file_path)

    def save_tai(self, file_path):
        self.save(file_path)

    def save(self, file_path):
        self.data['save time'].append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print('保存日志...')

        try:
            data_to_save = json.loads(json.dumps(self.data, cls=CustomJSONEncoder))
        except Exception as e:
            print(f'Error creating deep copy of data for saving: {e}')
            data_to_save = self.data

        try:
            json_data = json.dumps(data_to_save, cls=CustomJSONEncoder, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f'Error during JSON serialization: {e}')
            return

        pattern = r'\[\s*(-?\d+\.?\d*(?:\s*,\s*-?\d+\.?\d*)*)\s*\]'

        def compress_list(match):
            list_content = match.group(0)
            compressed_content = re.sub(r'\s+', ' ', list_content)
            compressed_content = re.sub(r'\s*,\s*', ', ', compressed_content)
            compressed_content = re.sub(r'\[\s+', '[', compressed_content)
            compressed_content = re.sub(r'\s+\]', ']', compressed_content)
            return compressed_content.strip()

        try:
            formatted_json = re.sub(pattern, compress_list, json_data, flags=re.MULTILINE)
        except Exception as e:
            print(f'Error during regex formatting: {e}')
            formatted_json = json_data

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(formatted_json)
            print('日志保存成功')
        except IOError as e:
            print(f'错误：无法写入日志文件: {e}')
        except Exception as e:
            print(f'保存过程中发生意外错误: {e}')

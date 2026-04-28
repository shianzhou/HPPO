import json
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
    def __init__(self, keep_records=True):
        self.keep_records = bool(keep_records)
        self.data = {
            'start time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'save time': [],
            'series': {},
        }
        if self.keep_records:
            self.data['records'] = []

    def _normalize_scalar(self, value):
        if hasattr(value, 'item') and callable(value.item):
            try:
                value = value.item()
            except Exception:
                pass
        if isinstance(value, np.ndarray) and value.size == 1:
            return value.item()
        return value

    def _append_series_record(self, record):
        series = self.data.setdefault('series', {})
        existing_count = 0
        if series:
            first_key = next(iter(series))
            existing_count = len(series[first_key])

        for key in record:
            if key not in series:
                series[key] = [None] * existing_count

        for key, values in series.items():
            values.append(self._normalize_scalar(record.get(key)))

    def _series_length(self):
        series = self.data.get('series', {})
        if not series:
            return 0
        first_key = next(iter(series))
        return len(series[first_key])

    def _is_scalar(self, value):
        return value is None or isinstance(value, (bool, int, float, str))

    def _dump_with_inline_lists(self, obj, indent=4, level=0):
        pad = ' ' * (indent * level)
        next_pad = ' ' * (indent * (level + 1))

        if isinstance(obj, dict):
            if not obj:
                return '{}'
            items = []
            for key, value in obj.items():
                key_text = json.dumps(str(key), ensure_ascii=False)
                value_text = self._dump_with_inline_lists(value, indent, level + 1)
                items.append(f"{next_pad}{key_text}: {value_text}")
            return '{\n' + ',\n'.join(items) + '\n' + pad + '}'

        if isinstance(obj, list):
            if not obj:
                return '[]'
            if all(self._is_scalar(v) for v in obj):
                return json.dumps(obj, ensure_ascii=False, separators=(', ', ': '))
            items = [self._dump_with_inline_lists(v, indent, level + 1) for v in obj]
            return '[\n' + ',\n'.join(f"{next_pad}{item}" for item in items) + '\n' + pad + ']'

        return json.dumps(obj, ensure_ascii=False)

    def add_cycle_record(self, episode_num=None, action_type=None, decision_reward=None,
                         catch_reward=None, tai_reward=None, total_reward=None,
                         loss_discrete=None, loss_continuous=None,
                         loss_decision=None, loss_grab_discrete=None, loss_step_discrete=None,
                         loss_grab_continuous=None, loss_step_continuous=None,
                         loss_value=None, loss_total=None,
                         grab_mask_mean=None, step_mask_mean=None,
                         **extra_fields):
        total_episode_num = extra_fields.get('total_episode_num')
        canonical_episode_num = total_episode_num if total_episode_num is not None else episode_num
        record = {
            'episode_num': self._normalize_scalar(canonical_episode_num),
            'action_type': action_type,
            'decision_reward': self._normalize_scalar(decision_reward),
            'catch_reward': self._normalize_scalar(catch_reward),
            'tai_reward': self._normalize_scalar(tai_reward),
            'total_reward': self._normalize_scalar(total_reward),
            'loss_discrete': self._normalize_scalar(loss_discrete),
            'loss_continuous': self._normalize_scalar(loss_continuous),
            # Branching PPO 细分损失（与原字段同样按标量序列存储）
            'loss_decision': self._normalize_scalar(loss_decision),
            'loss_grab_discrete': self._normalize_scalar(loss_grab_discrete),
            'loss_step_discrete': self._normalize_scalar(loss_step_discrete),
            'loss_grab_continuous': self._normalize_scalar(loss_grab_continuous),
            'loss_step_continuous': self._normalize_scalar(loss_step_continuous),
            'loss_value': self._normalize_scalar(loss_value),
            'loss_total': self._normalize_scalar(loss_total),
            'grab_mask_mean': self._normalize_scalar(grab_mask_mean),
            'step_mask_mean': self._normalize_scalar(step_mask_mean),
        }
        for key, value in extra_fields.items():
            record[key] = self._normalize_scalar(value)
        if self.keep_records:
            self.data['records'].append(record)
        self._append_series_record(record)

    def log_cycle(self, file_path, episode_num=None, action_type=None, decision_reward=None,
                  catch_reward=None, tai_reward=None, total_reward=None,
                  loss_discrete=None, loss_continuous=None,
                  loss_decision=None, loss_grab_discrete=None, loss_step_discrete=None,
                  loss_grab_continuous=None, loss_step_continuous=None,
                  loss_value=None, loss_total=None,
                  grab_mask_mean=None, step_mask_mean=None,
                  **extra_fields):
        self.add_cycle_record(
            episode_num=episode_num,
            action_type=action_type,
            decision_reward=decision_reward,
            catch_reward=catch_reward,
            tai_reward=tai_reward,
            total_reward=total_reward,
            loss_discrete=loss_discrete,
            loss_continuous=loss_continuous,
            loss_decision=loss_decision,
            loss_grab_discrete=loss_grab_discrete,
            loss_step_discrete=loss_step_discrete,
            loss_grab_continuous=loss_grab_continuous,
            loss_step_continuous=loss_step_continuous,
            loss_value=loss_value,
            loss_total=loss_total,
            grab_mask_mean=grab_mask_mean,
            step_mask_mean=step_mask_mean,
            **extra_fields,
        )
        self.save(file_path)

    def add_loss(self, loss_discrete=None, loss_continuous=None,
                 loss_decision=None, loss_grab_discrete=None, loss_step_discrete=None,
                 loss_grab_continuous=None, loss_step_continuous=None,
                 loss_value=None, loss_total=None,
                 grab_mask_mean=None, step_mask_mean=None):
        self.add_cycle_record(
            loss_discrete=loss_discrete,
            loss_continuous=loss_continuous,
            loss_decision=loss_decision,
            loss_grab_discrete=loss_grab_discrete,
            loss_step_discrete=loss_step_discrete,
            loss_grab_continuous=loss_grab_continuous,
            loss_step_continuous=loss_step_continuous,
            loss_value=loss_value,
            loss_total=loss_total,
            grab_mask_mean=grab_mask_mean,
            step_mask_mean=step_mask_mean,
        )

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
        if not self.keep_records:
            self.add_cycle_record(**kwargs)
            return
        if not self.data['records']:
            self.data['records'].append({})
        record = self.data['records'][-1]
        for key, value in kwargs.items():
            record[key] = self._normalize_scalar(value)

    def clear(self):
        return

    def reset(self):
        self.__init__(keep_records=self.keep_records)

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
            formatted_json = self._dump_with_inline_lists(data_to_save, indent=4, level=0)
        except Exception as e:
            print(f'Error during JSON formatting: {e}')
            return

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(formatted_json)
            print('日志保存成功')
        except IOError as e:
            print(f'错误：无法写入日志文件: {e}')
        except Exception as e:
            print(f'保存过程中发生意外错误: {e}')

    def save_series(self, file_path, keys=None, include_meta=True):
        """保存适合画图的列式日志，每个字段对应一个数组。"""
        series_data = dict(self.data.get('series', {}))

        if keys is not None:
            series_data = {key: series_data.get(key, []) for key in keys}

        if include_meta:
            series_data['start time'] = self.data.get('start time')
            series_data['save time'] = list(self.data.get('save time', []))
            series_data['record_count'] = self._series_length()

        try:
            formatted_json = self._dump_with_inline_lists(series_data, indent=4, level=0)
        except Exception as e:
            print(f'Error during series formatting: {e}')
            return

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(formatted_json)
            print('系列日志保存成功')
        except IOError as e:
            print(f'错误：无法写入系列日志文件: {e}')
        except Exception as e:
            print(f'保存系列日志过程中发生意外错误: {e}')

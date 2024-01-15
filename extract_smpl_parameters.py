import os
import pickle as pkl
import numpy as np

if __name__ == '__main__':
    male_path = 'body_models/smpl/male/model.pkl'
    female_path = 'body_models/smpl/female/model.pkl'
    neutral_path = 'body_models/smpl/neutral/model.pkl'

    data_m = pkl.load(open(male_path, 'rb'), encoding='latin1')
    data_f = pkl.load(open(female_path, 'rb'), encoding='latin1')
    data_n = pkl.load(open(neutral_path, 'rb'), encoding='latin1')

    if not os.path.exists('body_models/misc'):
        os.makedirs('body_models/misc')

    np.savez('body_models/misc/faces.npz', faces=data_m['f'].astype(np.int64))
    np.savez('body_models/misc/J_regressors.npz', male=data_m['J_regressor'].toarray(), female=data_f['J_regressor'].toarray(), neutral=data_n['J_regressor'].toarray())
    np.savez('body_models/misc/posedirs_all.npz', male=data_m['posedirs'], female=data_f['posedirs'], neutral=data_n['posedirs'])
    np.savez('body_models/misc/shapedirs_all.npz', male=data_m['shapedirs'], female=data_f['shapedirs'], neutral=data_n['shapedirs'])
    np.savez('body_models/misc/skinning_weights_all.npz', male=data_m['weights'], female=data_f['weights'], neutral=data_n['weights'])
    np.savez('body_models/misc/v_templates.npz', male=data_m['v_template'], female=data_f['v_template'], neutral=data_n['v_template'])
    np.save('body_models/misc/kintree_table.npy', data_m['kintree_table'].astype(np.int32))

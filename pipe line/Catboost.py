class CatBoost_wrapper:
    def __init__(self, features, target, cat_cols=None):
        self.param = {
            'boosting_type': 'gbdt',
            'metric': 'rmse',
            "objective":"rmse",
            
            
            #'objective': 'multiclass','metric': 'multiclass', 'num_class':4,
            #'metric': 'rmse',  'metric': 'tweedie',
            #'objective': 'tweedie',
            #'tweedie_variance_power': trial.suggest_uniform('tweedie_variance_power', 1.01, 1.8),

            'n_estimators': 1400,
            'boost_from_average': False,'verbose': -1,'random_state':2020,
            
            
            'num_leaves': 96,
            'max_depth': 10,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.95,
            'bagging_freq': 5
        }
        
        self.features = features
        self.target = target
        self.cat_cols = cat_cols 
        
    def set_param(self, param):
        self.param = param
    
    def train_predict(self, train_df, test_df):
        """
        Fold のカラム名は　基本Fold
        """        
        models=[]
        test_df[f'{self.target}_catbst_predict']=0
        Folds = set(train_df["Fold"])
        valid = []
        
        for fold in Folds:
            
            train_data = lgb.Dataset(train_df.loc[train_df["Fold"]!=fold, self.features], train_df.loc[train_df["Fold"]!=fold, self.target])
            valid_df = train_df[train_df["Fold"]==fold]
            valid_data = lgb.Dataset(valid_df[self.features], valid_df[self.target])
            
            model = lgb.train(self.param,  train_data,  num_boost_round=3000,  valid_sets=[train_data, valid_data], 
                              verbose_eval=500, early_stopping_rounds=100)
            
            
            valid_df[f'{self.target}_catbst_predict'] = model.predict(valid_df[self.features])
            test_df[f'{self.target}_catbst_predict'] += model.predict(test_df[self.features])/len(Folds)
                
            valid.append(valid_df)
            models.append(model)
            
        train_df = pd.concat(valid, axis=0)
        self.models = models
        return train_df, test_df
from genetic import *
from sklearn.preprocessing import normalize

class GeneticCoder(object):
    def __init__(self, num_param, population_size=1200):
        self.population = np.random.rand(population_size, num_param) - 0.5
        self.population = normalize(self.population)
        self.num_param = num_param
        self.population_size = population_size
        
    def get_group_max(self, data, group_key='user_id', sort_key='score'):
        return data.groupby(group_key).apply(lambda x: x.sort_values(sort_key, 
                                          ascending=False).head(1))
        
    def train(self, data, label, iter=300, 
              group_key='user_id', output_key='sku_id',
              retain_ratio=0.4, hybrid_ratio=0.4):
        train_num = data.drop([group_key, output_key], 1).as_matrix()
        train_pred = data.loc[:, [group_key, output_key]]
        ans_key = pd.DataFrame(label)
        ans_key['ans'] = np.ones(len(ans_key))
        ans_key = ans_key.groupby([group_key, output_key]).count()
        assert train_num.shape[1] == self.num_param, \
            "Input matrix should have same dimension as parameter vector"
        assert hybrid_ratio <= retain_ratio, \
            "Cannot hybrid more vectors than retained ones"
        assert retain_ratio + hybrid_ratio <= 1.0, \
            "Must maintain same number of vectors each round."
        
        scores = np.zeros(self.population_size)
        retain_num = int(self.population_size * retain_ratio)
        hybrid_num = int(self.population_size * hybrid_ratio)
        mut_num = self.population_size - retain_num - hybrid_num
        
        train_scores = np.zeros(iter)
        for _ in range(iter):
            for idx in range(self.population_size):
                train_pred['score'] = train_num.dot(self.population[idx, :].transpose())
                pred_key = train_pred.sort_values('score', ascending=False)
                pred_key = pred_key.groupby(group_key).first()
                #pred_key['score'] = np.ones(len(pred_key))
                pred_key = pred_key.groupby([pred_key.index, output_key])['score'].count()
                test_df = ans_key.join(pred_key, how='left')
                test_df[pd.isnull(test_df)] = 0
                scores[idx] = np.sum(test_df['score'] * test_df['ans'])
            sort_idx = np.argsort(scores)
            train_scores[_] = scores[sort_idx[-1]]
            print scores[sort_idx[-1]]
            self.population[:retain_num, :] = self.population[sort_idx[-retain_num:], :]
            
            hybrid_idx = np.random.randint(0, retain_num, 2 * hybrid_num)
            self.population[retain_num:(retain_num + hybrid_num), 
                            :] = (self.population[hybrid_idx[:hybrid_num], 
                                  :] + self.population[hybrid_idx[hybrid_num:], 
                                  :]) / 2.0
            
            mut_idx = np.random.randint(0, retain_num, mut_num)
            self.population[(retain_num + hybrid_num):, :] = self.population[mut_idx, :]
            mut_add = (np.random.rand(mut_num, self.num_param) - 0.5) * 0.01
            self.population[(retain_num + hybrid_num):, :] += mut_add
            
            self.population = normalize(self.population)
            
        for idx in range(self.population_size):
            train_pred['score'] = train_num.dot(self.population[idx, :].transpose())
            pred_key = train_pred.sort_values('score', ascending=False)
            pred_key = pred_key.groupby(group_key).first()
            #pred_key['score'] = np.ones(len(pred_key))
            pred_key = pred_key.groupby([pred_key.index, output_key])['score'].count()
            test_df = ans_key.join(pred_key, how='left')
            test_df[pd.isnull(test_df)] = 0
            scores[idx] = np.sum(test_df['score'] * test_df['ans'])
        sort_idx = np.argsort(scores)
        self.population = self.population[sort_idx, :]
        return train_scores
    
    def predict(self, data, group_key='user_id', output_key='sku_id'):
        train_num = data.drop([group_key, output_key], 1).as_matrix()
        train_pred = data.loc[:, [group_key, output_key]]
        train_pred['score'] = train_num.dot(self.population[idx, :].transpose())
        pred_key = train_pred.sort_values('score', ascending=False)
        pred_key = pred_key.groupby(group_key).first()
        return pred_key
    
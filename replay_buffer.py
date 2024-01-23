import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, batch_size, n_col, even_sample_label_col=None):
        self.data, self.max_size, self.batch_size = [], max_size, batch_size
        self.n_col = n_col

        self.even_sample_label_col = even_sample_label_col
        if even_sample_label_col is not None:
            self.data_by_label = {}

    @property
    def size(self):
        if self.even_sample_label_col is not None:
            size = 0
            for key in self.data_by_label:
                size += len(self.data_by_label[key])
            return size
        else:
            return len(self.data)

    def full(self):
        return self.size >= self.max_size

    def add(self, experience):
        # replay buffer : (s, a, r, s, g, t)
        # oracle data : (s, g, goal_achieved)
        assert len(experience) == self.n_col, 'Experience form didn\'t match, %s=/%s'%(len(experience), self.n_col)
        
        c = self.even_sample_label_col
        if self.full():
            if c is not None:
                for key in self.data_by_label:
                    d = self.data_by_label[key]
                    self.data_by_label[key] = d[round(len(d)/6):]
            else:
                self.data = self.data[round(self.max_size/6):]

        if c is not None:
            key = experience[c]
            if not key in self.data_by_label:
                self.data_by_label[key] = []
            self.data_by_label[key].append(experience)
        else:
            self.data.append(experience)

        # If replay buffer is filled, remove a percentage of replay buffer.  Only removing a single transition slows down performance
        
    def add_multi(self, experiences):
        for exp in experiences:
            self.add(exp)

    def get_batch(self, full=False):
        result = tuple([[] for _ in range(self.n_col)])

        if self.even_sample_label_col is not None:
            n = len(self.data_by_label)
            for key in self.data_by_label:
                d = self.data_by_label[key]
                dist = np.random.randint(0, high=len(d), size=round(self.batch_size/n))
                for i in dist:
                    for j in range(self.n_col):
                        result[j].append(d[i][j])
        else:
            if full:
                dist = np.arange(0, self.size)
            else:
                dist = np.random.randint(0, high=self.size, size=self.batch_size)
            for i in dist:
                for j in range(self.n_col):
                    result[j].append(self.data[i][j])
        return result

    def empty(self):
        self.data = []

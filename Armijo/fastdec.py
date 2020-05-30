import numpy as np
import math

# ff = open('test.md', 'w+', encoding='utf-8')


def to_str(p):
    x_str = str(p)
    x_str = x_str[1:-1]
    x_str = ','.join(x_str.split())
    x_str = '(' + x_str + ')'
    return x_str


class UnOp:
    def __init__(self, f, method):
        self.f = f
        self.method = method
        if 'G' in self.method:
            self.get_alpha = self.Goldstein
        elif 'A' in self.method:
            self.get_alpha = self.Armijo
        elif 'B' in self.method:
            self.get_alpha = self.Backtracking
        else:
            assert 0, 'not implement'

    def get_d(self, x_0):
        g = self.df(x_0)
        d = g.T @ g
        d = -1 * g / math.sqrt(d)
        return d

    def get_phi(self, x_0, d):
        def phi(alpha):
            return self.f(x_0 + alpha * d)

        return phi

    def get_dphi(self, x_0, d):
        def dphi(alpha):
            return (self.df(x_0 + alpha * d)).T @ d

        return dphi

    def df(self, x):
        ans = []
        p = np.eye(x.shape[0])
        base = self.f(x)
        for i in p:
            ans.append((self.f(x + i * 1e-4) - base) / 1e-4)
        return np.array(ans)

    def Armijo(self, phi, dphi, rho=0.1, beta=0.5, tor=1):
        tbeta = 1
        for m in range(1, 100000):
            tbeta = tbeta * beta
            if phi(tbeta * tor) <= phi(0) + rho * tbeta * tor * dphi(0):
                return tbeta * tor

    def Backtracking(self, phi, dphi, alpha=0.3, beta=0.5):
        return self.Armijo(phi, dphi, rho=alpha, beta=beta, tor=1)

    def Goldstein(self, phi, dphi, rho=0.1):
        a = 0
        b = -1
        k = 1
        alpha = 1

        while True:
            while True:
                if phi(alpha) <= phi(0) + rho * alpha * dphi(0):
                    if phi(alpha) >= phi(0) + (1 - rho) * alpha * dphi(0):
                        return alpha
                    else:
                        if b == -1:
                            alpha = alpha * 1.1
                        else:
                            a = alpha
                            b = b
                            break
                else:
                    a = a
                    b = alpha
                    break
            alpha = (a + b) / 2
            k += 1

    def calculate(self, x_0):
        x = x_0
        k = 1

        alpha = 1
        d = np.ones_like(x_0)
        while np.sqrt((alpha * d).T @ (alpha * d)) > 1e-5:
            d = self.get_d(x)
            phi = self.get_phi(x, d)
            dphi = self.get_dphi(x, d)
            alpha = self.get_alpha(phi, dphi)
            print(f'现在$x$为${to_str(x)}$，计算它的梯度为${to_str(self.df(x))}$,通过{self.method}求得的步长因子为${alpha}$\n')
            x = x + alpha * d
            k = k + 1
        return x


class Gloden:
    def __init__(self, input_shape, f, df, path='./out.md'):
        self.input_shape = input_shape
        self.f = f
        self.df = df
        self.ans = np.zeros(self.input_shape)
        self.file = None
        self.record = path

    def init(self):
        self.ans = np.random.normal(0, 1, size=self.input_shape)

    def check_trace(self, trace):
        if trace is True:
            if self.file is None:
                if self.record is None:
                    return False
                else:
                    self.file = open(self.record, 'w+', encoding='utf-8')
                    return True
            else:
                return True
        return False

    def calculate(self, low, high, x0=None, epsilon=1e-4, trace=False):
        if self.check_trace(trace):
            print('使用黄金分割法', file=self.file)
        if x0 is None:
            self.init()
        else:
            self.ans = x0
        k = 1
        a = low
        b = high
        lam = a + 0.382 * (b - a)
        mu = a + 0.618 * (b - a)
        phi_lam = self.f(lam)
        phi_mu = self.f(mu)
        phi_lam = round(phi_lam, 3)
        phi_mu = round(phi_mu, 3)

        '''
        for record
        '''
        pl = '\\phi(\\lambda)'
        pm = '\\phi(\\mu)'

        if self.check_trace(trace):
            print(f'当前第${k}$轮，$a_{k}={a},b_{k}={b},\\lambda_{k}={lam},\\mu_{k}={mu},{pl}={phi_lam},{pm}={phi_mu}$\n',
                  file=self.file)
        while True:
            if phi_lam > phi_mu:
                if self.check_trace(trace):
                    print(f'因为${pl}={phi_lam}>{phi_mu}={pm}$\n', file=self.file)
                if b - a < epsilon:
                    if self.check_trace(trace):
                        print(f'$b-a={b - a}<\\epsilon={epsilon}$,停止，答案为$x={mu}$\n', file=self.file)
                    self.ans = mu
                    break
                a = lam
                b = b
                k += 1
                lam = mu
                phi_lam = phi_mu
                mu = a + 0.618 * (b - a)
                phi_mu = self.f(mu)
                if self.check_trace(trace):
                    a = round(a, 3)
                    b = round(b, 3)
                    lam = round(lam, 3)
                    mu = round(a + 0.618 * (b - a), 3)
                    phi_mu = self.f(mu)
                    phi_mu = round(phi_mu, 3)
                if self.check_trace(trace):
                    print(
                        f'当前第${k}$轮，$a_{k}={a},b_{k}={b},\\lambda_{k}={lam},\\mu_{k}={mu},{pl}={phi_lam},{pm}={phi_mu}$\n',
                        file=self.file)
                pass
            else:
                if self.check_trace(trace):
                    print(f'因为${pl}={phi_lam}\\leq {phi_mu}={pm}$\n', file=self.file)
                if b - a < epsilon:
                    if self.check_trace(trace):
                        print(f'$b-a={b - a}<\\epsilon={epsilon}$,停止，答案为$x={lam}$\n', file=self.file)
                    self.ans = lam
                    break
                a = a
                b = mu
                k += 1
                mu = lam
                phi_mu = phi_lam
                lam = a + 0.382 * (b - a)
                phi_lam = self.f(lam)
                if self.check_trace(trace):
                    a = round(a, 3)
                    b = round(b, 3)
                    mu = round(mu, 3)
                    lam = round(a + 0.382 * (b - a), 3)
                    phi_lam = round(self.f(lam), 3)
                if self.check_trace(trace):
                    print(
                        f'当前第${k}$轮，$a_{k}={a},b_{k}={b},\\lambda_{k}={lam},\\mu_{k}={mu},{pl}={phi_lam},{pm}={phi_mu}$\n',
                        file=self.file)

        return self.ans, self.f(self.ans)


def get_test_f():
    tests = []
    x_0 = []
    names = []

    def rosen_brock(x):
        assert isinstance(x, np.ndarray) and len(x.shape) == 1
        v = 0
        for i in range(x.shape[0] - 1):
            v += 100 * ((x[i + 1] - x[i] ** 2) ** 2) + (1 - x[i]) ** 2
        return v

    names.extend([f'\n rosenbrock function dim = {i}\n' for i in range(2, 10)])
    tests.extend([rosen_brock for _ in range(2, 10)])
    x_0.extend([np.random.normal(0, 0.01, size=[i]) for i in range(2, 10)])

    def wood(x):
        assert isinstance(x, np.ndarray) and len(x.shape) == 1
        return 100 * ((x[0] * x[0] - x[1]) ** 2) + (x[0] - 1) ** 2 + (x[2] - 1) ** 2 + 90 * (
                (x[2] * x[2] - x[3]) ** 2) + 10.1 * ((x[1] - 1) ** 2 + (x[3] - 1) ** 2) \
               + 19.8 * (x[1] - 1) * (x[3] - 1)

    names.append(f'\n wood function \n')
    tests.append(wood)
    x_0.append(np.random.normal(0, 0.5, size=[4]))

    def powell(x):
        return (x[0] + 10 * x[1]) ** 2 + 5 * ((x[2] - x[3]) ** 2) + (x[1] - 2 * x[2]) ** 4 + 10 * ((x[0] - x[3]) ** 4)

    names.append(f'\n powell function \n')
    tests.append(powell)
    x_0.append(np.random.normal(0, 0.5, size=[4]))

    def tube(x):
        return 100 * ((x[1] - (x[0] ** 3)) ** 2) + (1 - x[0]) ** 2

    names.append(f'\n 立方体 function \n')
    tests.append(tube)
    x_0.append(np.random.normal(0, 0.5, size=[2]))

    def triple(x):
        v = 0
        cos_v = 0
        for i in range(x.shape[0]):
            cos_v += np.cos(x[i])
        for i in range(x.shape[0]):
            v += (x.shape[0] + (i + 1) * (1 - np.cos(x[i])) - np.sin(x[i]) - cos_v) ** 2
        return v

    names.extend([f'\n 三角 function dim = {i}\n' for i in range(2, 10)])
    tests.extend([triple for _ in range(2, 10)])
    x_0.extend([np.random.normal(0, 0.01, size=[i]) for i in range(2, 10)])

    # def luo(x):
    #     if x[0] >= 0:
    #         theta = np.arctan(x[0] / x[1]) / 2 / np.pi
    #     else:
    #         theta = np.arctan(x[1] / x[0]) + np.pi
    #         theta = theta / 2 / np.pi
    #     return 100 * ((x[2] - 10 * theta) ** 2 + (np.sqrt(x[0] * x[0] + x[1] * x[1]) - 1) ** 2) + x[2] * x[2]
    #
    # names.append('\n螺旋形凹谷函数\n')
    # tests.append(luo)
    # x_0.append(np.random.normal(0, 0.01, size=3))
    return tests, x_0, names


if __name__ == '__main__':
    # V = Gloden(1, f, df)
    # a, v = V.calculate(-2, -1, epsilon=0.05, trace=True)
    # print(a, v)
    # print(df(0.5))
    def f(x):
        return np.exp(-x[0])+x[0]*x[0]


    S = UnOp(f, 'Armijo')
    ans =S.calculate(np.array([1]))
    print(f(ans))
    # fs, xs, names = get_test_f()
    # ap=[]
    # for name, f, x in zip(names, fs, xs):
    #     print(f'\\subsection{{{name}}}', file=ff)
    #     S = UnOp(f, 'Armijo')
    #     ans = S.calculate(x)
    #     print('解为', to_str(ans), '最终答案为', f(ans), '标准答案为 0.0\n\n', file=ff)
    #     ap.append(f(ans))
    # print(np.mean(np.array(ap)))

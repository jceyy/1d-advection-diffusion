import numpy as np


class BcHandler:
    def __init__(self, axes=1, axes_names=None):  # Non-periodic directions only!

        if axes_names is None:
            axes_names = ["x"]  # Default axis name

        self.bc = []
        if axes < 1:
            exit(1)

        for i in range(axes):
            self.bc.append(((0, 0), (0, 0), axes_names[i]))

    def set(self, axis_str, bc):
        if len(bc) != 2 or len(bc[0]) != 2 or len(bc[1]) != 2:
            print("Invalid bc")
            exit(1)

        for i in range(len(self.bc)):
            if self.bc[i][-1] == axis_str:
                self.bc[i] = (bc[0], bc[1], axis_str)
                return
            print("Invalid axis")
            exit(1)

    def gen_bc_str(self):
        bc_str = []
        for i in range(len(self.bc)):
            # left
            bc_order = self.bc[i][0][0]  # Derivative order
            bc_value = self.bc[i][0][1]  # BC value
            bc_start, bc_end = "", ""
            if bc_order > 0:
                bc_start, bc_end = "dx(" * bc_order, ")" * bc_order
            bc_str.append("left" + '(' + bc_start + "u" + bc_end + ") = %f" % bc_value)

            # right
            bc_order = self.bc[i][1][0]  # Derivative order
            bc_value = self.bc[i][1][1]  # BC value
            bc_start, bc_end = "", ""
            if bc_order > 0:
                bc_start, bc_end = "dx(" * bc_order, ")" * bc_order
            bc_str.append("right" + '(' + bc_start + "u" + bc_end + ") = %f" % bc_value)

        return bc_str

    def i_j_selector(self, axis):
        mA = self.bc[axis][0][0]
        mB = self.bc[axis][1][0]
        m = max(mA, mB)
        return m, m + 1

    def find_fi_fj(self, i, j, axis, f, xlim):
        # i < j and j = i + 1 necessarily for this algorithm
        f = np.array(f)
        detJ = 2. / (xlim[1] - xlim[0])
        mA = self.bc[axis][0][0]
        mB = self.bc[axis][1][0]
        bA = self.bc[axis][0][1]
        bB = self.bc[axis][1][1]
        Nc = len(f) + 2  # f here is only composed of Nc - 2 independent coefficients

        # Construct K and H such that f = K^(-1) * H
        if mA == 0:
            K11 = 1 if i % 2 == 0 else -1
            K12 = 1 if j % 2 == 0 else -1
        else:
            sign = 1
            if (mA + i) % 2 != 0:
                sign = -1
            K11 = [(i*i - k*k) / (2*k + 1) for k in range(mA)]
            K11 = (detJ ** mA) * sign * np.prod(K11)
            sign = -1 * sign
            K12 = [(j*j - k*k) / (2*k + 1) for k in range(mA)]
            K12 = (detJ ** mA) * sign * np.prod(K12)

        if mB == 0:
            K21 = 1
            K22 = 1
        else:
            K21 = [(i*i - k*k) / (2*k + 1) for k in range(mB)]
            K21 = (detJ ** mB) * np.prod(K21)
            K22 = [(j*j - k*k) / (2*k + 1) for k in range(mB)]
            K22 = (detJ ** mB) * np.prod(K22)

        H1 = []
        H2 = []
        indexes = list(np.linspace(0, Nc - 1, Nc))
        del indexes[j]  # remember that i < j
        del indexes[i]

        for n in indexes:  # from which i and j are deleted
            if mA == 0:
                h1 = 1 if n % 2 == 0 else -1
            else:
                h1 = [(n*n - p*p) / (2*p + 1) for p in range(mA)]

            if mB == 0:
                h2 = 1
            else:
                h2 = [(n*n - p*p) / (2*p + 1) for p in range(mB)]

            sign = 1
            if mA != 0 and (mA + n) % 2 != 0:
                sign = -1
            H1.append((detJ ** mA) * sign * np.prod(h1))
            H2.append((detJ ** mB) * np.prod(h2))

        H1 = bA - np.sum(f * H1)
        H2 = bB - np.sum(f * H2)
        K = np.array([[K11, K12], [K21, K22]])
        K_inv = np.linalg.inv(K)
        H = np.array([H1, H2])
        f = np.matmul(K_inv, H)

        return f

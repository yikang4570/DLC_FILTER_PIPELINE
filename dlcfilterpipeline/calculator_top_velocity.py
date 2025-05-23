# A PYTHON VERSION OF THE MATLAB CALCULATOR TOP VELOCITY CODE
import sys
import numpy as np
from scipy.io import loadmat
import readyplot as rp
import pandas as pd

class calculator_top_velocity:
    def __init__(self, data_file_path):
        self.filepath = data_file_path
        self.mat_data = None
        self.AGATHA = None
        self.BottomCentroidVelocity = None
        self.BottomNoseVelocity = None
        self.TopCentroidVelocity = None
        self.TopNoseVelocity = None
        self.FL = None
        self.FR = None
        self.BL = None
        self.BR = None
        self.col_names = [
            'Left Fore Duty Factor',
            'Right Fore Duty Factor',
            'Fore Duty Factor',
            'ST Imbalance Fore',
            'Left Hind Duty Factor',
            'Right Hind Duty Factor',
            'Hind Duty Factor',
            'ST Imbalance Hind',
            'Fore Limb Temporal Symmetry',
            'Hind Limb Temporal Symmetry',
            'Fore Step Width',
            'Hind Step Width',
            'Fore Stride Length',
            'Hind Stride Length',
            'Spatial Symmetry Fore',
            'Spatial Symmetry Hind',
            'Top Nose Velocity',
            'Top Body Velocity'
                            ]
        self.DF_FL=0
        self.DF_FR=0
        self.DF_F=0
        self.STI_F=0
        self.DF_BL=0
        self.DF_BR=0
        self.DF_B=0
        self.STI_B=0
        self.TS_F=0
        self.TS_B=0
        self.SW_F=0
        self.SW_B=0
        self.SL_F=0
        self.SL_B=0
        self.SS_F=0
        self.SS_B=0
        self.V_F=0
        self.V_B=0
        self.output_rows = []

        self.load_data()
        self.filter_data()
        self.calculate_parameters()
        self.export()

    def load_data(self):
        temp_mat = loadmat(self.filepath, struct_as_record=False, squeeze_me=True)
        self.mat_data = temp_mat['DATA']
        self.AGATHA = np.array(self.mat_data.AGATHA)
        self.BottomCentroidVelocity = self.mat_data.Velocity.BottomCentroidVelocity
        self.BottomNoseVelocity = self.mat_data.Velocity.BottomNoseVelocity
        self.TopCentroidVelocity = self.mat_data.Velocity.TopCentroidVelocity
        self.TopNoseVelocity = self.mat_data.Velocity.TopNoseVelocity

    def filter_data(self):
        self.AGATHA = self.AGATHA[self.AGATHA[:, 3].argsort()]
        self.FL = self.AGATHA[(self.AGATHA[:,1] == 1) & (self.AGATHA[:,8] == 1)]
        self.FR = self.AGATHA[(self.AGATHA[:,1] == 1) & (self.AGATHA[:,8] == 0)]
        self.BL = self.AGATHA[(self.AGATHA[:,1] == 0) & (self.AGATHA[:,8] == 1)]
        self.BR = self.AGATHA[(self.AGATHA[:,1] == 0) & (self.AGATHA[:,8] == 0)]

        fl_br_pairs = self.pair_diagonal_strikes(self.FL, self.BR)
        fr_bl_pairs = self.pair_diagonal_strikes(self.FR, self.BL)

        fl_indices = set(i for i, _ in fl_br_pairs)
        fr_indices = set(i for i, _ in fr_bl_pairs)
        bl_indices = set(j for _, j in fr_bl_pairs)
        br_indices = set(j for _, j in fl_br_pairs)

        self.FL = self.filter_by_pairs(self.FL, fl_indices)
        self.FR = self.filter_by_pairs(self.FR, fr_indices)
        self.BL = self.filter_by_pairs(self.BL, bl_indices)
        self.BR = self.filter_by_pairs(self.BR, br_indices)

        self.readyplot_tool()

    def pair_diagonal_strikes(self,primary, diagonal, tolerance=0.1):
        pairs = []
        for i, (p_start, p_end) in enumerate(zip(primary[:, 3], primary[:, 5])):
            for j, (d_start, d_end) in enumerate(zip(diagonal[:, 3], diagonal[:, 5])):
                p_up = (p_start + p_end) / 2 + (p_end - p_start)*tolerance
                p_down = (p_start + p_end) / 2 - (p_end - p_start)*tolerance
                d_up = (d_start + d_end) / 2 + (d_end - d_start)*tolerance
                d_down = (d_start + d_end) / 2 - (d_end - d_start)*tolerance

                if (((p_down >= d_start or p_up >= d_start) and (p_down <= d_end or p_up <= d_end)) # P middle range in bounds of D
                        and ((d_down >= p_start or d_up >= p_start) and (d_down <= p_end or d_up <= p_end))): # Vice versa
                    pairs.append((i, j))
                    break  # One-to-one pairing; stop once matched
        return pairs

    def filter_by_pairs(self,array, valid_indices):
        return array[sorted(valid_indices)]

    def max_under_threshold(self,arr, threshold):
        mask = arr < threshold
        if np.any(mask): return np.where(mask)[0][np.argmax(arr[mask])]
        else: return None

    def distance(self,xs,ys):
        sum = 0
        for i,v in enumerate(xs): sum += (xs[i] - ys[i])**2
        return np.sqrt(sum)


    def calculate_parameters(self):
        for i in range(min(len(self.FL), len(self.FR))):
            self.duty_factors(i)
            self.temporal_symmetry(i)
            self.step_width(i)
            self.velocities(i)

            row = [self.DF_FL,self.DF_FR,self.DF_F,self.STI_F,self.DF_BL,self.DF_BR,self.DF_B,self.STI_B,
                   self.TS_F,self.TS_B,self.SW_F,self.SW_B,self.SL_F,self.SL_B,self.SS_F,self.SS_B,self.V_F,self.V_B]
            self.output_rows.append(row)

    def duty_factors(self,i):
        self.DF_FL = (self.FL[i,5]-self.FL[i,3])/(self.FL[i+1,3]-self.FL[i,3]) if i+1<len(self.FL) else 0
        self.DF_FR = (self.FR[i,5]-self.FR[i,3])/(self.FR[i+1,3]-self.FR[i,3]) if i+1<len(self.FR) else 0
        self.DF_F = (self.DF_FL + self.DF_FR)/2 if (self.DF_FL != 0 and self.DF_FR != 0) else 0
        self.STI_F = (self.DF_FL - self.DF_FR) if (self.DF_FL != 0 and self.DF_FR != 0) else 0

        self.DF_BL = (self.BL[i,5]-self.BL[i,3])/(self.BL[i+1,3]-self.BL[i,3]) if i+1<len(self.BL) else 0
        self.DF_BR = (self.BR[i,5]-self.BR[i,3])/(self.BR[i+1,3]-self.BR[i,3]) if i+1<len(self.BR) else 0
        self.DF_B = (self.DF_BL + self.DF_BR)/2 if (self.DF_BL != 0 and self.DF_BR != 0) else 0
        self.STI_B = (self.DF_BL - self.DF_BR) if (self.DF_BL != 0 and self.DF_BR != 0) else 0

    def temporal_symmetry(self,i):
        pre_FL_indx = self.max_under_threshold(self.FL[:,3],self.FR[i,3])
        pre_BL_indx = self.max_under_threshold(self.BL[:, 3], self.BR[i, 3])

        pre_FL = self.FL[pre_FL_indx,3] if (not pre_FL_indx is None) else None
        if pre_FL_indx is not None:
            post_FL = self.FL[pre_FL_indx+1,3] if (not pre_FL_indx+1 == len(self.FL)) else None
        pre_BL = self.BL[pre_BL_indx, 3] if (not pre_BL_indx is None) else None
        if pre_BL_indx is not None:
            post_BL = self.BL[pre_BL_indx+1, 3] if (not pre_BL_indx+1 == len(self.BL)) else None

        self.TS_F = (self.FR[i,3]-pre_FL)/(post_FL-pre_FL) if (not pre_FL is None) and (not post_FL is None) else 0
        self.TS_B = (self.BR[i,3]-pre_BL)/(post_BL-pre_BL) if (not pre_BL is None) and (not post_BL is None) else 0

    def step_width(self,i):
        pre_FL_indx = self.max_under_threshold(self.FL[:, 3], self.FR[i, 3])
        if pre_FL_indx is not None:
            if pre_FL_indx + 1 != len(self.FL):
                a = self.distance([self.FL[pre_FL_indx,9],self.FL[pre_FL_indx+1,9]],
                                  [self.FL[pre_FL_indx,10],self.FL[pre_FL_indx+1,10]])
                b = self.distance([self.FL[pre_FL_indx,9],self.FR[i,9]],
                                  [self.FL[pre_FL_indx,10],self.FR[i,10]])
                c = self.distance([self.FL[pre_FL_indx+1,9],self.FR[i,9]],
                                  [self.FL[pre_FL_indx+1,10],self.FR[i,10]])
                s = (a+b+c)/2
                Area = np.sqrt(s*(s-a)*(s-b)*(s-c))
                self.SW_F = (Area*2)/a
                self.SL_F = a
                self.SS_F = np.sqrt(b**2 - self.SW_F**2)/a
        else:
            self.SW_F = 0
            self.SL_F = 0
            self.SS_F = 0

        pre_BL_indx = self.max_under_threshold(self.BL[:, 3], self.BR[i, 3])
        if not pre_BL_indx is not None:
            if pre_BL_indx + 1 != len(self.BL):
                a = self.distance([self.BL[pre_BL_indx,9],self.BL[pre_BL_indx+1,9]],
                                  [self.BL[pre_BL_indx,10],self.BL[pre_BL_indx+1,10]])
                b = self.distance([self.BL[pre_BL_indx,9],self.BR[i,9]],
                                  [self.BL[pre_BL_indx,10],self.BR[i,10]])
                c = self.distance([self.BL[pre_BL_indx+1,9],self.BR[i,9]],
                                  [self.BL[pre_BL_indx+1,10],self.BR[i,10]])
                s = (a+b+c)/2
                Area = np.sqrt(s*(s-a)*(s-b)*(s-c))
                self.SW_B = (Area*2)/a
                self.SL_B = a
                self.SS_B = np.sqrt(b ** 2 - self.SW_B ** 2) / a
        else:
            self.SW_B = 0
            self.SL_B = 0
            self.SS_B = 0

    def velocities(self,i):
        def fit_velocity(x):
            t = np.arange(len(x))
            slope, intercept = np.polyfit(t, x, 1)
            return slope

        if i == 0:
            V_F = abs(fit_velocity(self.TopNoseVelocity[:,0][self.TopNoseVelocity[:,0] != 0]))
            V_B = abs(fit_velocity(self.TopCentroidVelocity[:,0][self.TopCentroidVelocity[:,0] != 0]))
        else:
            V_F,V_B = 0,0

        self.V_F, self.V_B = V_F, V_B

    def export(self):
        df = pd.DataFrame(self.output_rows, columns=self.col_names)
        df.to_excel(self.filepath.split('_DATA.mat')[0]+".xlsx", index=False)

    def multi_print_tool(self,attrs):
        for attr in attrs: print(f"{attr}: {getattr(self, attr)}")

    def readyplot_tool(self):
        max_x = np.max(self.AGATHA[:,4])
        max_y = np.max(self.AGATHA[:, 10])
        max_t = np.max(self.AGATHA[:, 5])
        DF = df = pd.DataFrame(columns=['x', 'y', 't', 'num', 'foot'])
        title = self.filepath.split('/')[-1].split('.')[0]

        for foot in ['BL', 'BR','FL', 'FR']:
            array = getattr(self, foot)
            for i in range(array.shape[0]):
                rowstart = [array[i,2], array[i,10], array[i,3], i, foot]
                DF.loc[len(DF)] = rowstart
                rowend = [array[i,4], array[i,10], array[i,5], i, foot]
                DF.loc[len(DF)] = rowend

        settings = {'dpi':150,'darkmode':True,'folder_name':self.filepath.split('.')[0],'colors':['gold', 'red', 'cyan', 'lime']}
        scattery = rp.scatter(DF, xlab='x', ylab='y', zlab='foot', title=title+'xy', **settings)
        scattery.plot(save=False)
        scattery.set_xlim([0, max_x * 1.05])
        scattery.set_ylim([0, max_y * 1.05])

        scattert = rp.scatter(DF, xlab='x', ylab='t', zlab='foot', title=title+'xt', **settings)
        scattert.plot(save=False)
        scattert.set_xlim([0, max_x * 1.05])
        scattert.set_ylim([0, max_t * 1.05])


        sub = rp.subplots(1, 2)
        fig, axes = sub.plot(scattert, {'title': title+'_XT', 'linewidth': 0, 's': 20},
                             scattery, {'title': title+'_XY', 'linewidth': 0, 's': 20},
                             figsize=(6, 5), dpi=200, folder_name=self.filepath + 'SUB.png')
        axes[1].get_legend().set_visible(False)
        sub.save()

if __name__ == '__main__':
    filename = '/Users/paveyboys/Desktop/PYTHON_TOP_VELOCITY/20250121_4353_01_DATA.mat'
    calculator_top_velocity(filename)



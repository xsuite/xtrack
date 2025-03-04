import numpy as np
import xtrack as xt

#NOTE: Put in an option to specify the spacing. Does nothing yet, something for later.

class wiggler:
    def __init__(self, Period, Amplitude, NumPeriods, Angle_Rad=0, Scheme='121s'):
        # The SchemeLibrary is a list of all the possible schemes that can be used.
        # The scheme determines the order of the dipoles in the wiggler.
        # The 's' and 'a' stand for a symmetric/antisymmetric configuration respectively.
        self.SchemeLibrary = ['121s', '121a']

        self.WigglerPeriod = Period
        self.WigglerAmplitude = Amplitude
        self.WigglerNumPeriods = NumPeriods
        self.Angle_Rad = Angle_Rad
        self.Scheme = Scheme
        self.Spacing = 0
        self.Wiggler = self._build_wiggler_()
        self.WigglerDict = self._build_dict_()


    def _build_wiggler_(self):
        Wiggler = []

        if self.Scheme == '121s':
            for i in range(self.WigglerNumPeriods+1):
                if i!=0 and i!=self.WigglerNumPeriods:
                    Wiggler += [xt.Bend(length=self.WigglerPeriod/4, k0=-self.WigglerAmplitude, h=0, rot_s_rad=self.Angle_Rad),
                                xt.Bend(length=self.WigglerPeriod/4, k0=-self.WigglerAmplitude, h=0, rot_s_rad=self.Angle_Rad),
                                xt.Bend(length=self.WigglerPeriod/4, k0= self.WigglerAmplitude, h=0, rot_s_rad=self.Angle_Rad),
                                xt.Bend(length=self.WigglerPeriod/4, k0= self.WigglerAmplitude, h=0, rot_s_rad=self.Angle_Rad)]

                elif i==0:
                    Wiggler += [xt.Bend(length=self.WigglerPeriod/4, k0= self.WigglerAmplitude, h=0, rot_s_rad=self.Angle_Rad)]

                else:
                    Wiggler += [xt.Bend(length=self.WigglerPeriod/4, k0=-self.WigglerAmplitude, h=0, rot_s_rad=self.Angle_Rad),
                                xt.Bend(length=self.WigglerPeriod/4, k0=-self.WigglerAmplitude, h=0, rot_s_rad=self.Angle_Rad),
                                xt.Bend(length=self.WigglerPeriod/4, k0= self.WigglerAmplitude, h=0, rot_s_rad=self.Angle_Rad)]
                    
        if self.Scheme == '121a':
            for i in range(self.WigglerNumPeriods):
                sign = 1 if i%2==0 else -1
                Wiggler += [xt.Bend(length=self.WigglerPeriod/4, k0=-sign*self.WigglerAmplitude, h=0, rot_s_rad=self.Angle_Rad),
                            xt.Bend(length=self.WigglerPeriod/4, k0= sign*self.WigglerAmplitude, h=0, rot_s_rad=self.Angle_Rad),
                            xt.Bend(length=self.WigglerPeriod/4, k0= sign*self.WigglerAmplitude, h=0, rot_s_rad=self.Angle_Rad),
                            xt.Bend(length=self.WigglerPeriod/4, k0=-sign*self.WigglerAmplitude, h=0, rot_s_rad=self.Angle_Rad)]

        print(f'Wiggler.shape = {Wiggler.__len__()}')

        return Wiggler
        

    def _get_wiggler_names_(self, Wiggler, WigglerNumber='1'):
        WigglerNames = []
        for i in range(len(Wiggler)):
            WigglerNames += ['mwp' + str(i+1) + '.' + WigglerNumber]
        
        print(f'WigglerNames.shape = {WigglerNames.__len__()}')


        return WigglerNames


    def _get_element_positions_(self, Wiggler):
        ElePos = np.zeros(len(Wiggler))
        for i in range(1, len(Wiggler)):
            ElePos[i] = ElePos[i-1] + Wiggler[i-1].length + self.Spacing

        print(f'ElePos.shape = {ElePos.shape}')

        return ElePos


    def _build_dict_(self):
        Wiggler = self._build_wiggler_()
        WigglerNames = self._get_wiggler_names_(Wiggler)
        ElePos = self._get_element_positions_(Wiggler)
        WigglerDict = {
        name: {'element': obj, 'position': pos}
        for name, obj, pos in zip(WigglerNames, Wiggler, ElePos)}

        return WigglerDict
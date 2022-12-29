from time import time
from sympy import *

x, y, z = symbols('x, y, z')
[uk_0, uk_1, uk_2, uk_3, uk_4, uk_5, uk_6, uk_7, uk_8, uk_9, uk_10, uk_11, uk_12, uk_13, uk_14, uk_15, uk_16, uk_17, uk_18, uk_19, uk_20, uk_21, uk_22
] = symbols('uk_:23')
#[uk_0, uk_1, uk_2, uk_3, uk_4, uk_5, uk_6, uk_7, uk_8, uk_9, uk_10, uk_11, uk_12, uk_13, uk_14, uk_15, uk_16, uk_17, uk_18, uk_19, uk_20, uk_21, uk_22, uk_23, uk_24, uk_25, uk_26, uk_27, uk_28, uk_29, uk_30, uk_31, uk_32, uk_33, uk_34, uk_35, uk_36, uk_37, uk_38, uk_39, uk_40, uk_41, uk_42, uk_43, uk_44, uk_45, uk_46, uk_47, uk_48, uk_49, uk_50, uk_51, uk_52, uk_53, uk_54, uk_55, uk_56, uk_57, uk_58, uk_59, uk_60, uk_61, uk_62, uk_63, uk_64, uk_65, uk_66, uk_67, uk_68, uk_69, uk_70, uk_71, uk_72, uk_73, uk_74, uk_75, uk_76, uk_77, uk_78, uk_79, uk_80, uk_81, uk_82, uk_83]

eqt = (uk_0**2 +
      uk_1**2 +
      uk_2**2 +
      uk_3**2 +
      uk_4**2 +
      uk_5**2 +
      uk_6**2 +
      uk_6**2 +
      uk_7**2 +
      uk_8**2 +
      uk_9**2 +
      uk_10**2 +
      uk_11**2 +
      uk_12**2 +
      uk_13**2 +
      uk_14**2 +
      uk_15**2 +
      uk_16**2 +
      uk_17**2 +
      uk_18**2 +
      uk_19**2 +
      uk_20**2 +
      uk_21**2 -
      uk_22**2 
      )

uk_vs = [uk_0, uk_1, uk_2, uk_3, uk_4, uk_5, uk_6, uk_7, uk_8, uk_9, uk_10,
        uk_11, uk_12, uk_13, uk_14, uk_15, uk_16, uk_17, uk_18, uk_19, uk_20,
        uk_21, uk_22]

print(eqt)
print(uk_vs)

identity_m = [{uk: i for i, uk in enumerate(uk_vs)}]

print(identity_m)

st = time()
sols = [eqt.xreplace(d) for d in identity_m]
print('xreplace', sols, time() - st)

st = time()
sols = [eqt.subs(d) for d in identity_m]
print('subs', sols, time() - st)

f= 2 * I
# Use sympy.is_complex method
gfg = f.is_complex
    
print(gfg)
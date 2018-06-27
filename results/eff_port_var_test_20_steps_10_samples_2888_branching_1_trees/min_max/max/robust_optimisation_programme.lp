\ENCODING=ISO-8859-1
\Problem name: 

Maximize
 obj: opt_return
Subject To
 w_0_ADM:                        w_0_ADM - 0.99 b_0_ADM + 1.01 s_0_ADM  = 0.1
 w_0_AXP:                        w_0_AXP - 0.99 b_0_AXP + 1.01 s_0_AXP  = 0.1
 w_0_DIS:                        w_0_DIS - 0.99 b_0_DIS + 1.01 s_0_DIS  = 0.1
 w_0_FDX:                        w_0_FDX - 0.99 b_0_FDX + 1.01 s_0_FDX  = 0.1
 w_0_IBM:                        w_0_IBM - 0.99 b_0_IBM + 1.01 s_0_IBM  = 0.1
 w_0_INTC:                       w_0_INTC - 0.99 b_0_INTC + 1.01 s_0_INTC
                                  = 0.1
 w_0_KO:                         w_0_KO - 0.99 b_0_KO + 1.01 s_0_KO  = 0.1
 w_0_MAT:                        w_0_MAT - 0.99 b_0_MAT + 1.01 s_0_MAT  = 0.1
 w_0_MSFT:                       w_0_MSFT - 0.99 b_0_MSFT + 1.01 s_0_MSFT
                                  = 0.1
 w_0_PG:                         w_0_PG - 0.99 b_0_PG + 1.01 s_0_PG  = 0.1
 w_0_balance_constraint:         b_0_ADM + b_0_AXP + b_0_DIS + b_0_FDX
                                 + b_0_IBM + b_0_INTC + b_0_KO + b_0_MAT
                                 + b_0_MSFT + b_0_PG - s_0_ADM - s_0_AXP
                                 - s_0_DIS - s_0_FDX - s_0_IBM - s_0_INTC
                                 - s_0_KO - s_0_MAT - s_0_MSFT - s_0_PG  = 0
 w_k1_ADM_s0_return_constraint:  1.13907 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.880536 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.848755 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.19888 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s02_return_constraint: 0.850478 w_k1_ADM_s0 - w_k1_ADM_s02
                                 + 0.99 b_k1_ADM_s02 - 1.01 s_k1_ADM_s02  = 0
 w_k1_ADM_s03_return_constraint: 1.00346 w_k1_ADM_s0 - w_k1_ADM_s03
                                 + 0.99 b_k1_ADM_s03 - 1.01 s_k1_ADM_s03  = 0
 w_k1_ADM_s04_return_constraint: 0.721525 w_k1_ADM_s0 - w_k1_ADM_s04
                                 + 0.99 b_k1_ADM_s04 - 1.01 s_k1_ADM_s04  = 0
 w_k1_ADM_s05_return_constraint: 0.962885 w_k1_ADM_s0 - w_k1_ADM_s05
                                 + 0.99 b_k1_ADM_s05 - 1.01 s_k1_ADM_s05  = 0
 w_k1_ADM_s06_return_constraint: 0.834851 w_k1_ADM_s0 - w_k1_ADM_s06
                                 + 0.99 b_k1_ADM_s06 - 1.01 s_k1_ADM_s06  = 0
 w_k1_ADM_s07_return_constraint: 1.17065 w_k1_ADM_s0 - w_k1_ADM_s07
                                 + 0.99 b_k1_ADM_s07 - 1.01 s_k1_ADM_s07  = 0
 w_k1_ADM_s10_return_constraint: 0.910399 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.30526 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s12_return_constraint: 1.47634 w_k1_ADM_s1 - w_k1_ADM_s12
                                 + 0.99 b_k1_ADM_s12 - 1.01 s_k1_ADM_s12  = 0
 w_k1_ADM_s13_return_constraint: 1.17343 w_k1_ADM_s1 - w_k1_ADM_s13
                                 + 0.99 b_k1_ADM_s13 - 1.01 s_k1_ADM_s13  = 0
 w_k1_ADM_s14_return_constraint: 0.97553 w_k1_ADM_s1 - w_k1_ADM_s14
                                 + 0.99 b_k1_ADM_s14 - 1.01 s_k1_ADM_s14  = 0
 w_k1_ADM_s15_return_constraint: 1.08731 w_k1_ADM_s1 - w_k1_ADM_s15
                                 + 0.99 b_k1_ADM_s15 - 1.01 s_k1_ADM_s15  = 0
 w_k1_ADM_s16_return_constraint: 0.929991 w_k1_ADM_s1 - w_k1_ADM_s16
                                 + 0.99 b_k1_ADM_s16 - 1.01 s_k1_ADM_s16  = 0
 w_k1_ADM_s17_return_constraint: 1.01484 w_k1_ADM_s1 - w_k1_ADM_s17
                                 + 0.99 b_k1_ADM_s17 - 1.01 s_k1_ADM_s17  = 0
 w_k1_ADM_s000_return_constraint: 1.13279 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.27355 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.936314 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.22975 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.08689 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.10326 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.33919 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.03711 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.806315 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.893043 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.41638 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.991085 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.816405 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.26593 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.935851 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.12329 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s020_return_constraint: 1.03267 w_k1_ADM_s02 - w_k1_ADM_s020
                                 + 0.99 b_k1_ADM_s020 - 1.01 s_k1_ADM_s020  = 0
 w_k1_ADM_s021_return_constraint: 1.07587 w_k1_ADM_s02 - w_k1_ADM_s021
                                 + 0.99 b_k1_ADM_s021 - 1.01 s_k1_ADM_s021  = 0
 w_k1_ADM_s022_return_constraint: 1.23962 w_k1_ADM_s02 - w_k1_ADM_s022
                                 + 0.99 b_k1_ADM_s022 - 1.01 s_k1_ADM_s022  = 0
 w_k1_ADM_s023_return_constraint: 1.00494 w_k1_ADM_s02 - w_k1_ADM_s023
                                 + 0.99 b_k1_ADM_s023 - 1.01 s_k1_ADM_s023  = 0
 w_k1_ADM_s024_return_constraint: 1.22266 w_k1_ADM_s02 - w_k1_ADM_s024
                                 + 0.99 b_k1_ADM_s024 - 1.01 s_k1_ADM_s024  = 0
 w_k1_ADM_s025_return_constraint: 1.09941 w_k1_ADM_s02 - w_k1_ADM_s025
                                 + 0.99 b_k1_ADM_s025 - 1.01 s_k1_ADM_s025  = 0
 w_k1_ADM_s026_return_constraint: 0.921074 w_k1_ADM_s02 - w_k1_ADM_s026
                                 + 0.99 b_k1_ADM_s026 - 1.01 s_k1_ADM_s026  = 0
 w_k1_ADM_s027_return_constraint: 1.22033 w_k1_ADM_s02 - w_k1_ADM_s027
                                 + 0.99 b_k1_ADM_s027 - 1.01 s_k1_ADM_s027  = 0
 w_k1_ADM_s030_return_constraint: 1.04604 w_k1_ADM_s03 - w_k1_ADM_s030
                                 + 0.99 b_k1_ADM_s030 - 1.01 s_k1_ADM_s030  = 0
 w_k1_ADM_s031_return_constraint: 1.48748 w_k1_ADM_s03 - w_k1_ADM_s031
                                 + 0.99 b_k1_ADM_s031 - 1.01 s_k1_ADM_s031  = 0
 w_k1_ADM_s032_return_constraint: 0.929456 w_k1_ADM_s03 - w_k1_ADM_s032
                                 + 0.99 b_k1_ADM_s032 - 1.01 s_k1_ADM_s032  = 0
 w_k1_ADM_s033_return_constraint: 0.898553 w_k1_ADM_s03 - w_k1_ADM_s033
                                 + 0.99 b_k1_ADM_s033 - 1.01 s_k1_ADM_s033  = 0
 w_k1_ADM_s034_return_constraint: 1.1146 w_k1_ADM_s03 - w_k1_ADM_s034
                                 + 0.99 b_k1_ADM_s034 - 1.01 s_k1_ADM_s034  = 0
 w_k1_ADM_s035_return_constraint: 1.17944 w_k1_ADM_s03 - w_k1_ADM_s035
                                 + 0.99 b_k1_ADM_s035 - 1.01 s_k1_ADM_s035  = 0
 w_k1_ADM_s036_return_constraint: 0.941799 w_k1_ADM_s03 - w_k1_ADM_s036
                                 + 0.99 b_k1_ADM_s036 - 1.01 s_k1_ADM_s036  = 0
 w_k1_ADM_s037_return_constraint: 0.896534 w_k1_ADM_s03 - w_k1_ADM_s037
                                 + 0.99 b_k1_ADM_s037 - 1.01 s_k1_ADM_s037  = 0
 w_k1_ADM_s040_return_constraint: 1.74735 w_k1_ADM_s04 - w_k1_ADM_s040
                                 + 0.99 b_k1_ADM_s040 - 1.01 s_k1_ADM_s040  = 0
 w_k1_ADM_s041_return_constraint: 1.34819 w_k1_ADM_s04 - w_k1_ADM_s041
                                 + 0.99 b_k1_ADM_s041 - 1.01 s_k1_ADM_s041  = 0
 w_k1_ADM_s042_return_constraint: 1.15948 w_k1_ADM_s04 - w_k1_ADM_s042
                                 + 0.99 b_k1_ADM_s042 - 1.01 s_k1_ADM_s042  = 0
 w_k1_ADM_s043_return_constraint: 1.35215 w_k1_ADM_s04 - w_k1_ADM_s043
                                 + 0.99 b_k1_ADM_s043 - 1.01 s_k1_ADM_s043  = 0
 w_k1_ADM_s044_return_constraint: 1.28851 w_k1_ADM_s04 - w_k1_ADM_s044
                                 + 0.99 b_k1_ADM_s044 - 1.01 s_k1_ADM_s044  = 0
 w_k1_ADM_s045_return_constraint: 1.02841 w_k1_ADM_s04 - w_k1_ADM_s045
                                 + 0.99 b_k1_ADM_s045 - 1.01 s_k1_ADM_s045  = 0
 w_k1_ADM_s046_return_constraint: 0.817127 w_k1_ADM_s04 - w_k1_ADM_s046
                                 + 0.99 b_k1_ADM_s046 - 1.01 s_k1_ADM_s046  = 0
 w_k1_ADM_s047_return_constraint: 1.05175 w_k1_ADM_s04 - w_k1_ADM_s047
                                 + 0.99 b_k1_ADM_s047 - 1.01 s_k1_ADM_s047  = 0
 w_k1_ADM_s050_return_constraint: 0.883133 w_k1_ADM_s05 - w_k1_ADM_s050
                                 + 0.99 b_k1_ADM_s050 - 1.01 s_k1_ADM_s050  = 0
 w_k1_ADM_s051_return_constraint: 1.03595 w_k1_ADM_s05 - w_k1_ADM_s051
                                 + 0.99 b_k1_ADM_s051 - 1.01 s_k1_ADM_s051  = 0
 w_k1_ADM_s052_return_constraint: 0.946293 w_k1_ADM_s05 - w_k1_ADM_s052
                                 + 0.99 b_k1_ADM_s052 - 1.01 s_k1_ADM_s052  = 0
 w_k1_ADM_s053_return_constraint: 1.18707 w_k1_ADM_s05 - w_k1_ADM_s053
                                 + 0.99 b_k1_ADM_s053 - 1.01 s_k1_ADM_s053  = 0
 w_k1_ADM_s054_return_constraint: 1.00427 w_k1_ADM_s05 - w_k1_ADM_s054
                                 + 0.99 b_k1_ADM_s054 - 1.01 s_k1_ADM_s054  = 0
 w_k1_ADM_s055_return_constraint: 0.862077 w_k1_ADM_s05 - w_k1_ADM_s055
                                 + 0.99 b_k1_ADM_s055 - 1.01 s_k1_ADM_s055  = 0
 w_k1_ADM_s056_return_constraint: 1.31897 w_k1_ADM_s05 - w_k1_ADM_s056
                                 + 0.99 b_k1_ADM_s056 - 1.01 s_k1_ADM_s056  = 0
 w_k1_ADM_s057_return_constraint: 1.17184 w_k1_ADM_s05 - w_k1_ADM_s057
                                 + 0.99 b_k1_ADM_s057 - 1.01 s_k1_ADM_s057  = 0
 w_k1_ADM_s060_return_constraint: 1.22233 w_k1_ADM_s06 - w_k1_ADM_s060
                                 + 0.99 b_k1_ADM_s060 - 1.01 s_k1_ADM_s060  = 0
 w_k1_ADM_s061_return_constraint: 1.23473 w_k1_ADM_s06 - w_k1_ADM_s061
                                 + 0.99 b_k1_ADM_s061 - 1.01 s_k1_ADM_s061  = 0
 w_k1_ADM_s062_return_constraint: 1.00556 w_k1_ADM_s06 - w_k1_ADM_s062
                                 + 0.99 b_k1_ADM_s062 - 1.01 s_k1_ADM_s062  = 0
 w_k1_ADM_s063_return_constraint: 0.771934 w_k1_ADM_s06 - w_k1_ADM_s063
                                 + 0.99 b_k1_ADM_s063 - 1.01 s_k1_ADM_s063  = 0
 w_k1_ADM_s064_return_constraint: 1.00826 w_k1_ADM_s06 - w_k1_ADM_s064
                                 + 0.99 b_k1_ADM_s064 - 1.01 s_k1_ADM_s064  = 0
 w_k1_ADM_s065_return_constraint: 0.710875 w_k1_ADM_s06 - w_k1_ADM_s065
                                 + 0.99 b_k1_ADM_s065 - 1.01 s_k1_ADM_s065  = 0
 w_k1_ADM_s066_return_constraint: 1.32103 w_k1_ADM_s06 - w_k1_ADM_s066
                                 + 0.99 b_k1_ADM_s066 - 1.01 s_k1_ADM_s066  = 0
 w_k1_ADM_s067_return_constraint: 1.06195 w_k1_ADM_s06 - w_k1_ADM_s067
                                 + 0.99 b_k1_ADM_s067 - 1.01 s_k1_ADM_s067  = 0
 w_k1_ADM_s070_return_constraint: 1.14111 w_k1_ADM_s07 - w_k1_ADM_s070
                                 + 0.99 b_k1_ADM_s070 - 1.01 s_k1_ADM_s070  = 0
 w_k1_ADM_s071_return_constraint: 1.50342 w_k1_ADM_s07 - w_k1_ADM_s071
                                 + 0.99 b_k1_ADM_s071 - 1.01 s_k1_ADM_s071  = 0
 w_k1_ADM_s072_return_constraint: 1.10312 w_k1_ADM_s07 - w_k1_ADM_s072
                                 + 0.99 b_k1_ADM_s072 - 1.01 s_k1_ADM_s072  = 0
 w_k1_ADM_s073_return_constraint: 1.04747 w_k1_ADM_s07 - w_k1_ADM_s073
                                 + 0.99 b_k1_ADM_s073 - 1.01 s_k1_ADM_s073  = 0
 w_k1_ADM_s074_return_constraint: 1.18215 w_k1_ADM_s07 - w_k1_ADM_s074
                                 + 0.99 b_k1_ADM_s074 - 1.01 s_k1_ADM_s074  = 0
 w_k1_ADM_s075_return_constraint: 0.924977 w_k1_ADM_s07 - w_k1_ADM_s075
                                 + 0.99 b_k1_ADM_s075 - 1.01 s_k1_ADM_s075  = 0
 w_k1_ADM_s076_return_constraint: 1.12926 w_k1_ADM_s07 - w_k1_ADM_s076
                                 + 0.99 b_k1_ADM_s076 - 1.01 s_k1_ADM_s076  = 0
 w_k1_ADM_s077_return_constraint: 1.40249 w_k1_ADM_s07 - w_k1_ADM_s077
                                 + 0.99 b_k1_ADM_s077 - 1.01 s_k1_ADM_s077  = 0
 w_k1_ADM_s100_return_constraint: 1.26522 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.13142 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.649333 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.02226 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.2551 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.850939 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.3486 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.884703 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.03469 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.02788 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.886457 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.755681 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.734072 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.763044 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.25714 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.15894 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_ADM_s120_return_constraint: 1.14291 w_k1_ADM_s12 - w_k1_ADM_s120
                                 + 0.99 b_k1_ADM_s120 - 1.01 s_k1_ADM_s120  = 0
 w_k1_ADM_s121_return_constraint: 0.779446 w_k1_ADM_s12 - w_k1_ADM_s121
                                 + 0.99 b_k1_ADM_s121 - 1.01 s_k1_ADM_s121  = 0
 w_k1_ADM_s122_return_constraint: 1.08294 w_k1_ADM_s12 - w_k1_ADM_s122
                                 + 0.99 b_k1_ADM_s122 - 1.01 s_k1_ADM_s122  = 0
 w_k1_ADM_s123_return_constraint: 1.03578 w_k1_ADM_s12 - w_k1_ADM_s123
                                 + 0.99 b_k1_ADM_s123 - 1.01 s_k1_ADM_s123  = 0
 w_k1_ADM_s124_return_constraint: 1.10325 w_k1_ADM_s12 - w_k1_ADM_s124
                                 + 0.99 b_k1_ADM_s124 - 1.01 s_k1_ADM_s124  = 0
 w_k1_ADM_s125_return_constraint: 1.00121 w_k1_ADM_s12 - w_k1_ADM_s125
                                 + 0.99 b_k1_ADM_s125 - 1.01 s_k1_ADM_s125  = 0
 w_k1_ADM_s126_return_constraint: 0.624482 w_k1_ADM_s12 - w_k1_ADM_s126
                                 + 0.99 b_k1_ADM_s126 - 1.01 s_k1_ADM_s126  = 0
 w_k1_ADM_s127_return_constraint: 0.93239 w_k1_ADM_s12 - w_k1_ADM_s127
                                 + 0.99 b_k1_ADM_s127 - 1.01 s_k1_ADM_s127  = 0
 w_k1_ADM_s130_return_constraint: 1.14156 w_k1_ADM_s13 - w_k1_ADM_s130
                                 + 0.99 b_k1_ADM_s130 - 1.01 s_k1_ADM_s130  = 0
 w_k1_ADM_s131_return_constraint: 1.09261 w_k1_ADM_s13 - w_k1_ADM_s131
                                 + 0.99 b_k1_ADM_s131 - 1.01 s_k1_ADM_s131  = 0
 w_k1_ADM_s132_return_constraint: 1.04483 w_k1_ADM_s13 - w_k1_ADM_s132
                                 + 0.99 b_k1_ADM_s132 - 1.01 s_k1_ADM_s132  = 0
 w_k1_ADM_s133_return_constraint: 0.772441 w_k1_ADM_s13 - w_k1_ADM_s133
                                 + 0.99 b_k1_ADM_s133 - 1.01 s_k1_ADM_s133  = 0
 w_k1_ADM_s134_return_constraint: 1.16511 w_k1_ADM_s13 - w_k1_ADM_s134
                                 + 0.99 b_k1_ADM_s134 - 1.01 s_k1_ADM_s134  = 0
 w_k1_ADM_s135_return_constraint: 1.13112 w_k1_ADM_s13 - w_k1_ADM_s135
                                 + 0.99 b_k1_ADM_s135 - 1.01 s_k1_ADM_s135  = 0
 w_k1_ADM_s136_return_constraint: 0.931603 w_k1_ADM_s13 - w_k1_ADM_s136
                                 + 0.99 b_k1_ADM_s136 - 1.01 s_k1_ADM_s136  = 0
 w_k1_ADM_s137_return_constraint: 1.07386 w_k1_ADM_s13 - w_k1_ADM_s137
                                 + 0.99 b_k1_ADM_s137 - 1.01 s_k1_ADM_s137  = 0
 w_k1_ADM_s140_return_constraint: 1.17864 w_k1_ADM_s14 - w_k1_ADM_s140
                                 + 0.99 b_k1_ADM_s140 - 1.01 s_k1_ADM_s140  = 0
 w_k1_ADM_s141_return_constraint: 1.39679 w_k1_ADM_s14 - w_k1_ADM_s141
                                 + 0.99 b_k1_ADM_s141 - 1.01 s_k1_ADM_s141  = 0
 w_k1_ADM_s142_return_constraint: 1.16828 w_k1_ADM_s14 - w_k1_ADM_s142
                                 + 0.99 b_k1_ADM_s142 - 1.01 s_k1_ADM_s142  = 0
 w_k1_ADM_s143_return_constraint: 0.995322 w_k1_ADM_s14 - w_k1_ADM_s143
                                 + 0.99 b_k1_ADM_s143 - 1.01 s_k1_ADM_s143  = 0
 w_k1_ADM_s144_return_constraint: 1.34105 w_k1_ADM_s14 - w_k1_ADM_s144
                                 + 0.99 b_k1_ADM_s144 - 1.01 s_k1_ADM_s144  = 0
 w_k1_ADM_s145_return_constraint: 1.40354 w_k1_ADM_s14 - w_k1_ADM_s145
                                 + 0.99 b_k1_ADM_s145 - 1.01 s_k1_ADM_s145  = 0
 w_k1_ADM_s146_return_constraint: 1.12315 w_k1_ADM_s14 - w_k1_ADM_s146
                                 + 0.99 b_k1_ADM_s146 - 1.01 s_k1_ADM_s146  = 0
 w_k1_ADM_s147_return_constraint: 1.33026 w_k1_ADM_s14 - w_k1_ADM_s147
                                 + 0.99 b_k1_ADM_s147 - 1.01 s_k1_ADM_s147  = 0
 w_k1_ADM_s150_return_constraint: 1.03527 w_k1_ADM_s15 - w_k1_ADM_s150
                                 + 0.99 b_k1_ADM_s150 - 1.01 s_k1_ADM_s150  = 0
 w_k1_ADM_s151_return_constraint: 0.797785 w_k1_ADM_s15 - w_k1_ADM_s151
                                 + 0.99 b_k1_ADM_s151 - 1.01 s_k1_ADM_s151  = 0
 w_k1_ADM_s152_return_constraint: 1.09726 w_k1_ADM_s15 - w_k1_ADM_s152
                                 + 0.99 b_k1_ADM_s152 - 1.01 s_k1_ADM_s152  = 0
 w_k1_ADM_s153_return_constraint: 0.996146 w_k1_ADM_s15 - w_k1_ADM_s153
                                 + 0.99 b_k1_ADM_s153 - 1.01 s_k1_ADM_s153  = 0
 w_k1_ADM_s154_return_constraint: 1.18716 w_k1_ADM_s15 - w_k1_ADM_s154
                                 + 0.99 b_k1_ADM_s154 - 1.01 s_k1_ADM_s154  = 0
 w_k1_ADM_s155_return_constraint: 0.804625 w_k1_ADM_s15 - w_k1_ADM_s155
                                 + 0.99 b_k1_ADM_s155 - 1.01 s_k1_ADM_s155  = 0
 w_k1_ADM_s156_return_constraint: 0.912663 w_k1_ADM_s15 - w_k1_ADM_s156
                                 + 0.99 b_k1_ADM_s156 - 1.01 s_k1_ADM_s156  = 0
 w_k1_ADM_s157_return_constraint: 0.688087 w_k1_ADM_s15 - w_k1_ADM_s157
                                 + 0.99 b_k1_ADM_s157 - 1.01 s_k1_ADM_s157  = 0
 w_k1_ADM_s160_return_constraint: 0.780291 w_k1_ADM_s16 - w_k1_ADM_s160
                                 + 0.99 b_k1_ADM_s160 - 1.01 s_k1_ADM_s160  = 0
 w_k1_ADM_s161_return_constraint: 1.06256 w_k1_ADM_s16 - w_k1_ADM_s161
                                 + 0.99 b_k1_ADM_s161 - 1.01 s_k1_ADM_s161  = 0
 w_k1_ADM_s162_return_constraint: 1.02858 w_k1_ADM_s16 - w_k1_ADM_s162
                                 + 0.99 b_k1_ADM_s162 - 1.01 s_k1_ADM_s162  = 0
 w_k1_ADM_s163_return_constraint: 0.715697 w_k1_ADM_s16 - w_k1_ADM_s163
                                 + 0.99 b_k1_ADM_s163 - 1.01 s_k1_ADM_s163  = 0
 w_k1_ADM_s164_return_constraint: 1.53487 w_k1_ADM_s16 - w_k1_ADM_s164
                                 + 0.99 b_k1_ADM_s164 - 1.01 s_k1_ADM_s164  = 0
 w_k1_ADM_s165_return_constraint: 1.05564 w_k1_ADM_s16 - w_k1_ADM_s165
                                 + 0.99 b_k1_ADM_s165 - 1.01 s_k1_ADM_s165  = 0
 w_k1_ADM_s166_return_constraint: 0.982195 w_k1_ADM_s16 - w_k1_ADM_s166
                                 + 0.99 b_k1_ADM_s166 - 1.01 s_k1_ADM_s166  = 0
 w_k1_ADM_s167_return_constraint: 1.23046 w_k1_ADM_s16 - w_k1_ADM_s167
                                 + 0.99 b_k1_ADM_s167 - 1.01 s_k1_ADM_s167  = 0
 w_k1_ADM_s170_return_constraint: 0.846246 w_k1_ADM_s17 - w_k1_ADM_s170
                                 + 0.99 b_k1_ADM_s170 - 1.01 s_k1_ADM_s170  = 0
 w_k1_ADM_s171_return_constraint: 1.00903 w_k1_ADM_s17 - w_k1_ADM_s171
                                 + 0.99 b_k1_ADM_s171 - 1.01 s_k1_ADM_s171  = 0
 w_k1_ADM_s172_return_constraint: 1.06643 w_k1_ADM_s17 - w_k1_ADM_s172
                                 + 0.99 b_k1_ADM_s172 - 1.01 s_k1_ADM_s172  = 0
 w_k1_ADM_s173_return_constraint: 1.0524 w_k1_ADM_s17 - w_k1_ADM_s173
                                 + 0.99 b_k1_ADM_s173 - 1.01 s_k1_ADM_s173  = 0
 w_k1_ADM_s174_return_constraint: 0.815501 w_k1_ADM_s17 - w_k1_ADM_s174
                                 + 0.99 b_k1_ADM_s174 - 1.01 s_k1_ADM_s174  = 0
 w_k1_ADM_s175_return_constraint: 1.0632 w_k1_ADM_s17 - w_k1_ADM_s175
                                 + 0.99 b_k1_ADM_s175 - 1.01 s_k1_ADM_s175  = 0
 w_k1_ADM_s176_return_constraint: 1.2469 w_k1_ADM_s17 - w_k1_ADM_s176
                                 + 0.99 b_k1_ADM_s176 - 1.01 s_k1_ADM_s176  = 0
 w_k1_ADM_s177_return_constraint: 1.03998 w_k1_ADM_s17 - w_k1_ADM_s177
                                 + 0.99 b_k1_ADM_s177 - 1.01 s_k1_ADM_s177  = 0
 w_k1_AXP_s0_return_constraint:  1.23006 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.890614 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.792726 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.05776 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s02_return_constraint: 0.880529 w_k1_AXP_s0 - w_k1_AXP_s02
                                 + 0.99 b_k1_AXP_s02 - 1.01 s_k1_AXP_s02  = 0
 w_k1_AXP_s03_return_constraint: 0.936967 w_k1_AXP_s0 - w_k1_AXP_s03
                                 + 0.99 b_k1_AXP_s03 - 1.01 s_k1_AXP_s03  = 0
 w_k1_AXP_s04_return_constraint: 0.689689 w_k1_AXP_s0 - w_k1_AXP_s04
                                 + 0.99 b_k1_AXP_s04 - 1.01 s_k1_AXP_s04  = 0
 w_k1_AXP_s05_return_constraint: 1.08707 w_k1_AXP_s0 - w_k1_AXP_s05
                                 + 0.99 b_k1_AXP_s05 - 1.01 s_k1_AXP_s05  = 0
 w_k1_AXP_s06_return_constraint: 0.821505 w_k1_AXP_s0 - w_k1_AXP_s06
                                 + 0.99 b_k1_AXP_s06 - 1.01 s_k1_AXP_s06  = 0
 w_k1_AXP_s07_return_constraint: 1.45579 w_k1_AXP_s0 - w_k1_AXP_s07
                                 + 0.99 b_k1_AXP_s07 - 1.01 s_k1_AXP_s07  = 0
 w_k1_AXP_s10_return_constraint: 0.759335 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.30868 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s12_return_constraint: 1.63627 w_k1_AXP_s1 - w_k1_AXP_s12
                                 + 0.99 b_k1_AXP_s12 - 1.01 s_k1_AXP_s12  = 0
 w_k1_AXP_s13_return_constraint: 1.20814 w_k1_AXP_s1 - w_k1_AXP_s13
                                 + 0.99 b_k1_AXP_s13 - 1.01 s_k1_AXP_s13  = 0
 w_k1_AXP_s14_return_constraint: 1.17539 w_k1_AXP_s1 - w_k1_AXP_s14
                                 + 0.99 b_k1_AXP_s14 - 1.01 s_k1_AXP_s14  = 0
 w_k1_AXP_s15_return_constraint: 0.985026 w_k1_AXP_s1 - w_k1_AXP_s15
                                 + 0.99 b_k1_AXP_s15 - 1.01 s_k1_AXP_s15  = 0
 w_k1_AXP_s16_return_constraint: 0.764292 w_k1_AXP_s1 - w_k1_AXP_s16
                                 + 0.99 b_k1_AXP_s16 - 1.01 s_k1_AXP_s16  = 0
 w_k1_AXP_s17_return_constraint: 0.878126 w_k1_AXP_s1 - w_k1_AXP_s17
                                 + 0.99 b_k1_AXP_s17 - 1.01 s_k1_AXP_s17  = 0
 w_k1_AXP_s000_return_constraint: 1.03272 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.2663 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.870494 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.34864 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.976881 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.03547 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.895548 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.733721 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.967154 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.940582 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.5689 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.33696 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.04427 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.11932 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.03883 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.34286 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s020_return_constraint: 0.895427 w_k1_AXP_s02 - w_k1_AXP_s020
                                 + 0.99 b_k1_AXP_s020 - 1.01 s_k1_AXP_s020  = 0
 w_k1_AXP_s021_return_constraint: 1.21793 w_k1_AXP_s02 - w_k1_AXP_s021
                                 + 0.99 b_k1_AXP_s021 - 1.01 s_k1_AXP_s021  = 0
 w_k1_AXP_s022_return_constraint: 1.17693 w_k1_AXP_s02 - w_k1_AXP_s022
                                 + 0.99 b_k1_AXP_s022 - 1.01 s_k1_AXP_s022  = 0
 w_k1_AXP_s023_return_constraint: 0.90854 w_k1_AXP_s02 - w_k1_AXP_s023
                                 + 0.99 b_k1_AXP_s023 - 1.01 s_k1_AXP_s023  = 0
 w_k1_AXP_s024_return_constraint: 1.30963 w_k1_AXP_s02 - w_k1_AXP_s024
                                 + 0.99 b_k1_AXP_s024 - 1.01 s_k1_AXP_s024  = 0
 w_k1_AXP_s025_return_constraint: 0.799982 w_k1_AXP_s02 - w_k1_AXP_s025
                                 + 0.99 b_k1_AXP_s025 - 1.01 s_k1_AXP_s025  = 0
 w_k1_AXP_s026_return_constraint: 0.823927 w_k1_AXP_s02 - w_k1_AXP_s026
                                 + 0.99 b_k1_AXP_s026 - 1.01 s_k1_AXP_s026  = 0
 w_k1_AXP_s027_return_constraint: 1.07147 w_k1_AXP_s02 - w_k1_AXP_s027
                                 + 0.99 b_k1_AXP_s027 - 1.01 s_k1_AXP_s027  = 0
 w_k1_AXP_s030_return_constraint: 1.13833 w_k1_AXP_s03 - w_k1_AXP_s030
                                 + 0.99 b_k1_AXP_s030 - 1.01 s_k1_AXP_s030  = 0
 w_k1_AXP_s031_return_constraint: 1.44191 w_k1_AXP_s03 - w_k1_AXP_s031
                                 + 0.99 b_k1_AXP_s031 - 1.01 s_k1_AXP_s031  = 0
 w_k1_AXP_s032_return_constraint: 0.720801 w_k1_AXP_s03 - w_k1_AXP_s032
                                 + 0.99 b_k1_AXP_s032 - 1.01 s_k1_AXP_s032  = 0
 w_k1_AXP_s033_return_constraint: 0.914939 w_k1_AXP_s03 - w_k1_AXP_s033
                                 + 0.99 b_k1_AXP_s033 - 1.01 s_k1_AXP_s033  = 0
 w_k1_AXP_s034_return_constraint: 1.2759 w_k1_AXP_s03 - w_k1_AXP_s034
                                 + 0.99 b_k1_AXP_s034 - 1.01 s_k1_AXP_s034  = 0
 w_k1_AXP_s035_return_constraint: 1.16211 w_k1_AXP_s03 - w_k1_AXP_s035
                                 + 0.99 b_k1_AXP_s035 - 1.01 s_k1_AXP_s035  = 0
 w_k1_AXP_s036_return_constraint: 0.943736 w_k1_AXP_s03 - w_k1_AXP_s036
                                 + 0.99 b_k1_AXP_s036 - 1.01 s_k1_AXP_s036  = 0
 w_k1_AXP_s037_return_constraint: 0.976477 w_k1_AXP_s03 - w_k1_AXP_s037
                                 + 0.99 b_k1_AXP_s037 - 1.01 s_k1_AXP_s037  = 0
 w_k1_AXP_s040_return_constraint: 1.42812 w_k1_AXP_s04 - w_k1_AXP_s040
                                 + 0.99 b_k1_AXP_s040 - 1.01 s_k1_AXP_s040  = 0
 w_k1_AXP_s041_return_constraint: 1.09414 w_k1_AXP_s04 - w_k1_AXP_s041
                                 + 0.99 b_k1_AXP_s041 - 1.01 s_k1_AXP_s041  = 0
 w_k1_AXP_s042_return_constraint: 0.98902 w_k1_AXP_s04 - w_k1_AXP_s042
                                 + 0.99 b_k1_AXP_s042 - 1.01 s_k1_AXP_s042  = 0
 w_k1_AXP_s043_return_constraint: 1.2153 w_k1_AXP_s04 - w_k1_AXP_s043
                                 + 0.99 b_k1_AXP_s043 - 1.01 s_k1_AXP_s043  = 0
 w_k1_AXP_s044_return_constraint: 1.20815 w_k1_AXP_s04 - w_k1_AXP_s044
                                 + 0.99 b_k1_AXP_s044 - 1.01 s_k1_AXP_s044  = 0
 w_k1_AXP_s045_return_constraint: 0.734958 w_k1_AXP_s04 - w_k1_AXP_s045
                                 + 0.99 b_k1_AXP_s045 - 1.01 s_k1_AXP_s045  = 0
 w_k1_AXP_s046_return_constraint: 0.69774 w_k1_AXP_s04 - w_k1_AXP_s046
                                 + 0.99 b_k1_AXP_s046 - 1.01 s_k1_AXP_s046  = 0
 w_k1_AXP_s047_return_constraint: 0.945986 w_k1_AXP_s04 - w_k1_AXP_s047
                                 + 0.99 b_k1_AXP_s047 - 1.01 s_k1_AXP_s047  = 0
 w_k1_AXP_s050_return_constraint: 0.727965 w_k1_AXP_s05 - w_k1_AXP_s050
                                 + 0.99 b_k1_AXP_s050 - 1.01 s_k1_AXP_s050  = 0
 w_k1_AXP_s051_return_constraint: 1.06132 w_k1_AXP_s05 - w_k1_AXP_s051
                                 + 0.99 b_k1_AXP_s051 - 1.01 s_k1_AXP_s051  = 0
 w_k1_AXP_s052_return_constraint: 0.896062 w_k1_AXP_s05 - w_k1_AXP_s052
                                 + 0.99 b_k1_AXP_s052 - 1.01 s_k1_AXP_s052  = 0
 w_k1_AXP_s053_return_constraint: 1.17226 w_k1_AXP_s05 - w_k1_AXP_s053
                                 + 0.99 b_k1_AXP_s053 - 1.01 s_k1_AXP_s053  = 0
 w_k1_AXP_s054_return_constraint: 1.18907 w_k1_AXP_s05 - w_k1_AXP_s054
                                 + 0.99 b_k1_AXP_s054 - 1.01 s_k1_AXP_s054  = 0
 w_k1_AXP_s055_return_constraint: 0.85099 w_k1_AXP_s05 - w_k1_AXP_s055
                                 + 0.99 b_k1_AXP_s055 - 1.01 s_k1_AXP_s055  = 0
 w_k1_AXP_s056_return_constraint: 1.42682 w_k1_AXP_s05 - w_k1_AXP_s056
                                 + 0.99 b_k1_AXP_s056 - 1.01 s_k1_AXP_s056  = 0
 w_k1_AXP_s057_return_constraint: 0.949806 w_k1_AXP_s05 - w_k1_AXP_s057
                                 + 0.99 b_k1_AXP_s057 - 1.01 s_k1_AXP_s057  = 0
 w_k1_AXP_s060_return_constraint: 1.13076 w_k1_AXP_s06 - w_k1_AXP_s060
                                 + 0.99 b_k1_AXP_s060 - 1.01 s_k1_AXP_s060  = 0
 w_k1_AXP_s061_return_constraint: 1.05979 w_k1_AXP_s06 - w_k1_AXP_s061
                                 + 0.99 b_k1_AXP_s061 - 1.01 s_k1_AXP_s061  = 0
 w_k1_AXP_s062_return_constraint: 0.786839 w_k1_AXP_s06 - w_k1_AXP_s062
                                 + 0.99 b_k1_AXP_s062 - 1.01 s_k1_AXP_s062  = 0
 w_k1_AXP_s063_return_constraint: 0.769304 w_k1_AXP_s06 - w_k1_AXP_s063
                                 + 0.99 b_k1_AXP_s063 - 1.01 s_k1_AXP_s063  = 0
 w_k1_AXP_s064_return_constraint: 0.948764 w_k1_AXP_s06 - w_k1_AXP_s064
                                 + 0.99 b_k1_AXP_s064 - 1.01 s_k1_AXP_s064  = 0
 w_k1_AXP_s065_return_constraint: 0.797101 w_k1_AXP_s06 - w_k1_AXP_s065
                                 + 0.99 b_k1_AXP_s065 - 1.01 s_k1_AXP_s065  = 0
 w_k1_AXP_s066_return_constraint: 1.46557 w_k1_AXP_s06 - w_k1_AXP_s066
                                 + 0.99 b_k1_AXP_s066 - 1.01 s_k1_AXP_s066  = 0
 w_k1_AXP_s067_return_constraint: 1.26629 w_k1_AXP_s06 - w_k1_AXP_s067
                                 + 0.99 b_k1_AXP_s067 - 1.01 s_k1_AXP_s067  = 0
 w_k1_AXP_s070_return_constraint: 0.990965 w_k1_AXP_s07 - w_k1_AXP_s070
                                 + 0.99 b_k1_AXP_s070 - 1.01 s_k1_AXP_s070  = 0
 w_k1_AXP_s071_return_constraint: 1.59908 w_k1_AXP_s07 - w_k1_AXP_s071
                                 + 0.99 b_k1_AXP_s071 - 1.01 s_k1_AXP_s071  = 0
 w_k1_AXP_s072_return_constraint: 0.858412 w_k1_AXP_s07 - w_k1_AXP_s072
                                 + 0.99 b_k1_AXP_s072 - 1.01 s_k1_AXP_s072  = 0
 w_k1_AXP_s073_return_constraint: 1.06764 w_k1_AXP_s07 - w_k1_AXP_s073
                                 + 0.99 b_k1_AXP_s073 - 1.01 s_k1_AXP_s073  = 0
 w_k1_AXP_s074_return_constraint: 1.31652 w_k1_AXP_s07 - w_k1_AXP_s074
                                 + 0.99 b_k1_AXP_s074 - 1.01 s_k1_AXP_s074  = 0
 w_k1_AXP_s075_return_constraint: 0.730866 w_k1_AXP_s07 - w_k1_AXP_s075
                                 + 0.99 b_k1_AXP_s075 - 1.01 s_k1_AXP_s075  = 0
 w_k1_AXP_s076_return_constraint: 0.71539 w_k1_AXP_s07 - w_k1_AXP_s076
                                 + 0.99 b_k1_AXP_s076 - 1.01 s_k1_AXP_s076  = 0
 w_k1_AXP_s077_return_constraint: 1.29419 w_k1_AXP_s07 - w_k1_AXP_s077
                                 + 0.99 b_k1_AXP_s077 - 1.01 s_k1_AXP_s077  = 0
 w_k1_AXP_s100_return_constraint: 1.73799 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.12062 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.906899 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.751234 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.37739 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.763496 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.31617 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.843959 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.23613 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.50502 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.18271 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.916205 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.983481 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.75688 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.5846 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.18938 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_AXP_s120_return_constraint: 1.28284 w_k1_AXP_s12 - w_k1_AXP_s120
                                 + 0.99 b_k1_AXP_s120 - 1.01 s_k1_AXP_s120  = 0
 w_k1_AXP_s121_return_constraint: 0.845439 w_k1_AXP_s12 - w_k1_AXP_s121
                                 + 0.99 b_k1_AXP_s121 - 1.01 s_k1_AXP_s121  = 0
 w_k1_AXP_s122_return_constraint: 0.889812 w_k1_AXP_s12 - w_k1_AXP_s122
                                 + 0.99 b_k1_AXP_s122 - 1.01 s_k1_AXP_s122  = 0
 w_k1_AXP_s123_return_constraint: 1.24115 w_k1_AXP_s12 - w_k1_AXP_s123
                                 + 0.99 b_k1_AXP_s123 - 1.01 s_k1_AXP_s123  = 0
 w_k1_AXP_s124_return_constraint: 0.906419 w_k1_AXP_s12 - w_k1_AXP_s124
                                 + 0.99 b_k1_AXP_s124 - 1.01 s_k1_AXP_s124  = 0
 w_k1_AXP_s125_return_constraint: 1.19396 w_k1_AXP_s12 - w_k1_AXP_s125
                                 + 0.99 b_k1_AXP_s125 - 1.01 s_k1_AXP_s125  = 0
 w_k1_AXP_s126_return_constraint: 0.683166 w_k1_AXP_s12 - w_k1_AXP_s126
                                 + 0.99 b_k1_AXP_s126 - 1.01 s_k1_AXP_s126  = 0
 w_k1_AXP_s127_return_constraint: 0.887826 w_k1_AXP_s12 - w_k1_AXP_s127
                                 + 0.99 b_k1_AXP_s127 - 1.01 s_k1_AXP_s127  = 0
 w_k1_AXP_s130_return_constraint: 1.21002 w_k1_AXP_s13 - w_k1_AXP_s130
                                 + 0.99 b_k1_AXP_s130 - 1.01 s_k1_AXP_s130  = 0
 w_k1_AXP_s131_return_constraint: 1.66453 w_k1_AXP_s13 - w_k1_AXP_s131
                                 + 0.99 b_k1_AXP_s131 - 1.01 s_k1_AXP_s131  = 0
 w_k1_AXP_s132_return_constraint: 0.916181 w_k1_AXP_s13 - w_k1_AXP_s132
                                 + 0.99 b_k1_AXP_s132 - 1.01 s_k1_AXP_s132  = 0
 w_k1_AXP_s133_return_constraint: 0.618995 w_k1_AXP_s13 - w_k1_AXP_s133
                                 + 0.99 b_k1_AXP_s133 - 1.01 s_k1_AXP_s133  = 0
 w_k1_AXP_s134_return_constraint: 1.05321 w_k1_AXP_s13 - w_k1_AXP_s134
                                 + 0.99 b_k1_AXP_s134 - 1.01 s_k1_AXP_s134  = 0
 w_k1_AXP_s135_return_constraint: 1.4532 w_k1_AXP_s13 - w_k1_AXP_s135
                                 + 0.99 b_k1_AXP_s135 - 1.01 s_k1_AXP_s135  = 0
 w_k1_AXP_s136_return_constraint: 1.01534 w_k1_AXP_s13 - w_k1_AXP_s136
                                 + 0.99 b_k1_AXP_s136 - 1.01 s_k1_AXP_s136  = 0
 w_k1_AXP_s137_return_constraint: 0.691428 w_k1_AXP_s13 - w_k1_AXP_s137
                                 + 0.99 b_k1_AXP_s137 - 1.01 s_k1_AXP_s137  = 0
 w_k1_AXP_s140_return_constraint: 0.906108 w_k1_AXP_s14 - w_k1_AXP_s140
                                 + 0.99 b_k1_AXP_s140 - 1.01 s_k1_AXP_s140  = 0
 w_k1_AXP_s141_return_constraint: 1.118 w_k1_AXP_s14 - w_k1_AXP_s141
                                 + 0.99 b_k1_AXP_s141 - 1.01 s_k1_AXP_s141  = 0
 w_k1_AXP_s142_return_constraint: 1.01226 w_k1_AXP_s14 - w_k1_AXP_s142
                                 + 0.99 b_k1_AXP_s142 - 1.01 s_k1_AXP_s142  = 0
 w_k1_AXP_s143_return_constraint: 0.839188 w_k1_AXP_s14 - w_k1_AXP_s143
                                 + 0.99 b_k1_AXP_s143 - 1.01 s_k1_AXP_s143  = 0
 w_k1_AXP_s144_return_constraint: 1.31481 w_k1_AXP_s14 - w_k1_AXP_s144
                                 + 0.99 b_k1_AXP_s144 - 1.01 s_k1_AXP_s144  = 0
 w_k1_AXP_s145_return_constraint: 1.26455 w_k1_AXP_s14 - w_k1_AXP_s145
                                 + 0.99 b_k1_AXP_s145 - 1.01 s_k1_AXP_s145  = 0
 w_k1_AXP_s146_return_constraint: 0.785702 w_k1_AXP_s14 - w_k1_AXP_s146
                                 + 0.99 b_k1_AXP_s146 - 1.01 s_k1_AXP_s146  = 0
 w_k1_AXP_s147_return_constraint: 1.06233 w_k1_AXP_s14 - w_k1_AXP_s147
                                 + 0.99 b_k1_AXP_s147 - 1.01 s_k1_AXP_s147  = 0
 w_k1_AXP_s150_return_constraint: 1.3443 w_k1_AXP_s15 - w_k1_AXP_s150
                                 + 0.99 b_k1_AXP_s150 - 1.01 s_k1_AXP_s150  = 0
 w_k1_AXP_s151_return_constraint: 1.07939 w_k1_AXP_s15 - w_k1_AXP_s151
                                 + 0.99 b_k1_AXP_s151 - 1.01 s_k1_AXP_s151  = 0
 w_k1_AXP_s152_return_constraint: 1.03013 w_k1_AXP_s15 - w_k1_AXP_s152
                                 + 0.99 b_k1_AXP_s152 - 1.01 s_k1_AXP_s152  = 0
 w_k1_AXP_s153_return_constraint: 1.2593 w_k1_AXP_s15 - w_k1_AXP_s153
                                 + 0.99 b_k1_AXP_s153 - 1.01 s_k1_AXP_s153  = 0
 w_k1_AXP_s154_return_constraint: 1.32399 w_k1_AXP_s15 - w_k1_AXP_s154
                                 + 0.99 b_k1_AXP_s154 - 1.01 s_k1_AXP_s154  = 0
 w_k1_AXP_s155_return_constraint: 1.0044 w_k1_AXP_s15 - w_k1_AXP_s155
                                 + 0.99 b_k1_AXP_s155 - 1.01 s_k1_AXP_s155  = 0
 w_k1_AXP_s156_return_constraint: 0.801428 w_k1_AXP_s15 - w_k1_AXP_s156
                                 + 0.99 b_k1_AXP_s156 - 1.01 s_k1_AXP_s156  = 0
 w_k1_AXP_s157_return_constraint: 0.72244 w_k1_AXP_s15 - w_k1_AXP_s157
                                 + 0.99 b_k1_AXP_s157 - 1.01 s_k1_AXP_s157  = 0
 w_k1_AXP_s160_return_constraint: 0.748651 w_k1_AXP_s16 - w_k1_AXP_s160
                                 + 0.99 b_k1_AXP_s160 - 1.01 s_k1_AXP_s160  = 0
 w_k1_AXP_s161_return_constraint: 1.12834 w_k1_AXP_s16 - w_k1_AXP_s161
                                 + 0.99 b_k1_AXP_s161 - 1.01 s_k1_AXP_s161  = 0
 w_k1_AXP_s162_return_constraint: 1.48857 w_k1_AXP_s16 - w_k1_AXP_s162
                                 + 0.99 b_k1_AXP_s162 - 1.01 s_k1_AXP_s162  = 0
 w_k1_AXP_s163_return_constraint: 0.825182 w_k1_AXP_s16 - w_k1_AXP_s163
                                 + 0.99 b_k1_AXP_s163 - 1.01 s_k1_AXP_s163  = 0
 w_k1_AXP_s164_return_constraint: 1.49013 w_k1_AXP_s16 - w_k1_AXP_s164
                                 + 0.99 b_k1_AXP_s164 - 1.01 s_k1_AXP_s164  = 0
 w_k1_AXP_s165_return_constraint: 0.837866 w_k1_AXP_s16 - w_k1_AXP_s165
                                 + 0.99 b_k1_AXP_s165 - 1.01 s_k1_AXP_s165  = 0
 w_k1_AXP_s166_return_constraint: 0.977444 w_k1_AXP_s16 - w_k1_AXP_s166
                                 + 0.99 b_k1_AXP_s166 - 1.01 s_k1_AXP_s166  = 0
 w_k1_AXP_s167_return_constraint: 1.08351 w_k1_AXP_s16 - w_k1_AXP_s167
                                 + 0.99 b_k1_AXP_s167 - 1.01 s_k1_AXP_s167  = 0
 w_k1_AXP_s170_return_constraint: 0.727884 w_k1_AXP_s17 - w_k1_AXP_s170
                                 + 0.99 b_k1_AXP_s170 - 1.01 s_k1_AXP_s170  = 0
 w_k1_AXP_s171_return_constraint: 1.17634 w_k1_AXP_s17 - w_k1_AXP_s171
                                 + 0.99 b_k1_AXP_s171 - 1.01 s_k1_AXP_s171  = 0
 w_k1_AXP_s172_return_constraint: 1.29614 w_k1_AXP_s17 - w_k1_AXP_s172
                                 + 0.99 b_k1_AXP_s172 - 1.01 s_k1_AXP_s172  = 0
 w_k1_AXP_s173_return_constraint: 1.24252 w_k1_AXP_s17 - w_k1_AXP_s173
                                 + 0.99 b_k1_AXP_s173 - 1.01 s_k1_AXP_s173  = 0
 w_k1_AXP_s174_return_constraint: 0.834465 w_k1_AXP_s17 - w_k1_AXP_s174
                                 + 0.99 b_k1_AXP_s174 - 1.01 s_k1_AXP_s174  = 0
 w_k1_AXP_s175_return_constraint: 1.12296 w_k1_AXP_s17 - w_k1_AXP_s175
                                 + 0.99 b_k1_AXP_s175 - 1.01 s_k1_AXP_s175  = 0
 w_k1_AXP_s176_return_constraint: 1.08137 w_k1_AXP_s17 - w_k1_AXP_s176
                                 + 0.99 b_k1_AXP_s176 - 1.01 s_k1_AXP_s176  = 0
 w_k1_AXP_s177_return_constraint: 0.910439 w_k1_AXP_s17 - w_k1_AXP_s177
                                 + 0.99 b_k1_AXP_s177 - 1.01 s_k1_AXP_s177  = 0
 w_k1_DIS_s0_return_constraint:  1.2913 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.863407 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.921832 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.35233 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s02_return_constraint: 1.03488 w_k1_DIS_s0 - w_k1_DIS_s02
                                 + 0.99 b_k1_DIS_s02 - 1.01 s_k1_DIS_s02  = 0
 w_k1_DIS_s03_return_constraint: 0.884472 w_k1_DIS_s0 - w_k1_DIS_s03
                                 + 0.99 b_k1_DIS_s03 - 1.01 s_k1_DIS_s03  = 0
 w_k1_DIS_s04_return_constraint: 0.59938 w_k1_DIS_s0 - w_k1_DIS_s04
                                 + 0.99 b_k1_DIS_s04 - 1.01 s_k1_DIS_s04  = 0
 w_k1_DIS_s05_return_constraint: 1.17795 w_k1_DIS_s0 - w_k1_DIS_s05
                                 + 0.99 b_k1_DIS_s05 - 1.01 s_k1_DIS_s05  = 0
 w_k1_DIS_s06_return_constraint: 0.747172 w_k1_DIS_s0 - w_k1_DIS_s06
                                 + 0.99 b_k1_DIS_s06 - 1.01 s_k1_DIS_s06  = 0
 w_k1_DIS_s07_return_constraint: 1.55675 w_k1_DIS_s0 - w_k1_DIS_s07
                                 + 0.99 b_k1_DIS_s07 - 1.01 s_k1_DIS_s07  = 0
 w_k1_DIS_s10_return_constraint: 0.663497 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.44852 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s12_return_constraint: 1.84012 w_k1_DIS_s1 - w_k1_DIS_s12
                                 + 0.99 b_k1_DIS_s12 - 1.01 s_k1_DIS_s12  = 0
 w_k1_DIS_s13_return_constraint: 1.25898 w_k1_DIS_s1 - w_k1_DIS_s13
                                 + 0.99 b_k1_DIS_s13 - 1.01 s_k1_DIS_s13  = 0
 w_k1_DIS_s14_return_constraint: 1.09015 w_k1_DIS_s1 - w_k1_DIS_s14
                                 + 0.99 b_k1_DIS_s14 - 1.01 s_k1_DIS_s14  = 0
 w_k1_DIS_s15_return_constraint: 0.945152 w_k1_DIS_s1 - w_k1_DIS_s15
                                 + 0.99 b_k1_DIS_s15 - 1.01 s_k1_DIS_s15  = 0
 w_k1_DIS_s16_return_constraint: 0.838199 w_k1_DIS_s1 - w_k1_DIS_s16
                                 + 0.99 b_k1_DIS_s16 - 1.01 s_k1_DIS_s16  = 0
 w_k1_DIS_s17_return_constraint: 0.859412 w_k1_DIS_s1 - w_k1_DIS_s17
                                 + 0.99 b_k1_DIS_s17 - 1.01 s_k1_DIS_s17  = 0
 w_k1_DIS_s000_return_constraint: 1.06016 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.35136 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.685924 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.20041 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.917399 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.921977 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.16816 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.682746 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.757306 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.846183 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.32808 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.981283 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.694066 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.55907 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.01147 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.53909 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s020_return_constraint: 0.724867 w_k1_DIS_s02 - w_k1_DIS_s020
                                 + 0.99 b_k1_DIS_s020 - 1.01 s_k1_DIS_s020  = 0
 w_k1_DIS_s021_return_constraint: 1.38607 w_k1_DIS_s02 - w_k1_DIS_s021
                                 + 0.99 b_k1_DIS_s021 - 1.01 s_k1_DIS_s021  = 0
 w_k1_DIS_s022_return_constraint: 1.05304 w_k1_DIS_s02 - w_k1_DIS_s022
                                 + 0.99 b_k1_DIS_s022 - 1.01 s_k1_DIS_s022  = 0
 w_k1_DIS_s023_return_constraint: 0.887508 w_k1_DIS_s02 - w_k1_DIS_s023
                                 + 0.99 b_k1_DIS_s023 - 1.01 s_k1_DIS_s023  = 0
 w_k1_DIS_s024_return_constraint: 1.26679 w_k1_DIS_s02 - w_k1_DIS_s024
                                 + 0.99 b_k1_DIS_s024 - 1.01 s_k1_DIS_s024  = 0
 w_k1_DIS_s025_return_constraint: 0.890731 w_k1_DIS_s02 - w_k1_DIS_s025
                                 + 0.99 b_k1_DIS_s025 - 1.01 s_k1_DIS_s025  = 0
 w_k1_DIS_s026_return_constraint: 0.70572 w_k1_DIS_s02 - w_k1_DIS_s026
                                 + 0.99 b_k1_DIS_s026 - 1.01 s_k1_DIS_s026  = 0
 w_k1_DIS_s027_return_constraint: 1.021 w_k1_DIS_s02 - w_k1_DIS_s027
                                 + 0.99 b_k1_DIS_s027 - 1.01 s_k1_DIS_s027  = 0
 w_k1_DIS_s030_return_constraint: 1.3749 w_k1_DIS_s03 - w_k1_DIS_s030
                                 + 0.99 b_k1_DIS_s030 - 1.01 s_k1_DIS_s030  = 0
 w_k1_DIS_s031_return_constraint: 1.75202 w_k1_DIS_s03 - w_k1_DIS_s031
                                 + 0.99 b_k1_DIS_s031 - 1.01 s_k1_DIS_s031  = 0
 w_k1_DIS_s032_return_constraint: 0.712418 w_k1_DIS_s03 - w_k1_DIS_s032
                                 + 0.99 b_k1_DIS_s032 - 1.01 s_k1_DIS_s032  = 0
 w_k1_DIS_s033_return_constraint: 0.876946 w_k1_DIS_s03 - w_k1_DIS_s033
                                 + 0.99 b_k1_DIS_s033 - 1.01 s_k1_DIS_s033  = 0
 w_k1_DIS_s034_return_constraint: 0.970943 w_k1_DIS_s03 - w_k1_DIS_s034
                                 + 0.99 b_k1_DIS_s034 - 1.01 s_k1_DIS_s034  = 0
 w_k1_DIS_s035_return_constraint: 1.34495 w_k1_DIS_s03 - w_k1_DIS_s035
                                 + 0.99 b_k1_DIS_s035 - 1.01 s_k1_DIS_s035  = 0
 w_k1_DIS_s036_return_constraint: 1.20878 w_k1_DIS_s03 - w_k1_DIS_s036
                                 + 0.99 b_k1_DIS_s036 - 1.01 s_k1_DIS_s036  = 0
 w_k1_DIS_s037_return_constraint: 0.903167 w_k1_DIS_s03 - w_k1_DIS_s037
                                 + 0.99 b_k1_DIS_s037 - 1.01 s_k1_DIS_s037  = 0
 w_k1_DIS_s040_return_constraint: 1.74215 w_k1_DIS_s04 - w_k1_DIS_s040
                                 + 0.99 b_k1_DIS_s040 - 1.01 s_k1_DIS_s040  = 0
 w_k1_DIS_s041_return_constraint: 0.993927 w_k1_DIS_s04 - w_k1_DIS_s041
                                 + 0.99 b_k1_DIS_s041 - 1.01 s_k1_DIS_s041  = 0
 w_k1_DIS_s042_return_constraint: 1.15928 w_k1_DIS_s04 - w_k1_DIS_s042
                                 + 0.99 b_k1_DIS_s042 - 1.01 s_k1_DIS_s042  = 0
 w_k1_DIS_s043_return_constraint: 1.14776 w_k1_DIS_s04 - w_k1_DIS_s043
                                 + 0.99 b_k1_DIS_s043 - 1.01 s_k1_DIS_s043  = 0
 w_k1_DIS_s044_return_constraint: 1.40182 w_k1_DIS_s04 - w_k1_DIS_s044
                                 + 0.99 b_k1_DIS_s044 - 1.01 s_k1_DIS_s044  = 0
 w_k1_DIS_s045_return_constraint: 0.820694 w_k1_DIS_s04 - w_k1_DIS_s045
                                 + 0.99 b_k1_DIS_s045 - 1.01 s_k1_DIS_s045  = 0
 w_k1_DIS_s046_return_constraint: 0.640341 w_k1_DIS_s04 - w_k1_DIS_s046
                                 + 0.99 b_k1_DIS_s046 - 1.01 s_k1_DIS_s046  = 0
 w_k1_DIS_s047_return_constraint: 0.865515 w_k1_DIS_s04 - w_k1_DIS_s047
                                 + 0.99 b_k1_DIS_s047 - 1.01 s_k1_DIS_s047  = 0
 w_k1_DIS_s050_return_constraint: 0.535051 w_k1_DIS_s05 - w_k1_DIS_s050
                                 + 0.99 b_k1_DIS_s050 - 1.01 s_k1_DIS_s050  = 0
 w_k1_DIS_s051_return_constraint: 1.06656 w_k1_DIS_s05 - w_k1_DIS_s051
                                 + 0.99 b_k1_DIS_s051 - 1.01 s_k1_DIS_s051  = 0
 w_k1_DIS_s052_return_constraint: 0.697207 w_k1_DIS_s05 - w_k1_DIS_s052
                                 + 0.99 b_k1_DIS_s052 - 1.01 s_k1_DIS_s052  = 0
 w_k1_DIS_s053_return_constraint: 1.01042 w_k1_DIS_s05 - w_k1_DIS_s053
                                 + 0.99 b_k1_DIS_s053 - 1.01 s_k1_DIS_s053  = 0
 w_k1_DIS_s054_return_constraint: 1.04472 w_k1_DIS_s05 - w_k1_DIS_s054
                                 + 0.99 b_k1_DIS_s054 - 1.01 s_k1_DIS_s054  = 0
 w_k1_DIS_s055_return_constraint: 0.745289 w_k1_DIS_s05 - w_k1_DIS_s055
                                 + 0.99 b_k1_DIS_s055 - 1.01 s_k1_DIS_s055  = 0
 w_k1_DIS_s056_return_constraint: 1.73451 w_k1_DIS_s05 - w_k1_DIS_s056
                                 + 0.99 b_k1_DIS_s056 - 1.01 s_k1_DIS_s056  = 0
 w_k1_DIS_s057_return_constraint: 1.0876 w_k1_DIS_s05 - w_k1_DIS_s057
                                 + 0.99 b_k1_DIS_s057 - 1.01 s_k1_DIS_s057  = 0
 w_k1_DIS_s060_return_constraint: 1.17863 w_k1_DIS_s06 - w_k1_DIS_s060
                                 + 0.99 b_k1_DIS_s060 - 1.01 s_k1_DIS_s060  = 0
 w_k1_DIS_s061_return_constraint: 0.89279 w_k1_DIS_s06 - w_k1_DIS_s061
                                 + 0.99 b_k1_DIS_s061 - 1.01 s_k1_DIS_s061  = 0
 w_k1_DIS_s062_return_constraint: 0.847429 w_k1_DIS_s06 - w_k1_DIS_s062
                                 + 0.99 b_k1_DIS_s062 - 1.01 s_k1_DIS_s062  = 0
 w_k1_DIS_s063_return_constraint: 0.624592 w_k1_DIS_s06 - w_k1_DIS_s063
                                 + 0.99 b_k1_DIS_s063 - 1.01 s_k1_DIS_s063  = 0
 w_k1_DIS_s064_return_constraint: 1.03032 w_k1_DIS_s06 - w_k1_DIS_s064
                                 + 0.99 b_k1_DIS_s064 - 1.01 s_k1_DIS_s064  = 0
 w_k1_DIS_s065_return_constraint: 0.809764 w_k1_DIS_s06 - w_k1_DIS_s065
                                 + 0.99 b_k1_DIS_s065 - 1.01 s_k1_DIS_s065  = 0
 w_k1_DIS_s066_return_constraint: 1.43338 w_k1_DIS_s06 - w_k1_DIS_s066
                                 + 0.99 b_k1_DIS_s066 - 1.01 s_k1_DIS_s066  = 0
 w_k1_DIS_s067_return_constraint: 1.13181 w_k1_DIS_s06 - w_k1_DIS_s067
                                 + 0.99 b_k1_DIS_s067 - 1.01 s_k1_DIS_s067  = 0
 w_k1_DIS_s070_return_constraint: 0.839358 w_k1_DIS_s07 - w_k1_DIS_s070
                                 + 0.99 b_k1_DIS_s070 - 1.01 s_k1_DIS_s070  = 0
 w_k1_DIS_s071_return_constraint: 1.78423 w_k1_DIS_s07 - w_k1_DIS_s071
                                 + 0.99 b_k1_DIS_s071 - 1.01 s_k1_DIS_s071  = 0
 w_k1_DIS_s072_return_constraint: 0.811338 w_k1_DIS_s07 - w_k1_DIS_s072
                                 + 0.99 b_k1_DIS_s072 - 1.01 s_k1_DIS_s072  = 0
 w_k1_DIS_s073_return_constraint: 0.99956 w_k1_DIS_s07 - w_k1_DIS_s073
                                 + 0.99 b_k1_DIS_s073 - 1.01 s_k1_DIS_s073  = 0
 w_k1_DIS_s074_return_constraint: 1.21116 w_k1_DIS_s07 - w_k1_DIS_s074
                                 + 0.99 b_k1_DIS_s074 - 1.01 s_k1_DIS_s074  = 0
 w_k1_DIS_s075_return_constraint: 0.566578 w_k1_DIS_s07 - w_k1_DIS_s075
                                 + 0.99 b_k1_DIS_s075 - 1.01 s_k1_DIS_s075  = 0
 w_k1_DIS_s076_return_constraint: 0.654412 w_k1_DIS_s07 - w_k1_DIS_s076
                                 + 0.99 b_k1_DIS_s076 - 1.01 s_k1_DIS_s076  = 0
 w_k1_DIS_s077_return_constraint: 1.3258 w_k1_DIS_s07 - w_k1_DIS_s077
                                 + 0.99 b_k1_DIS_s077 - 1.01 s_k1_DIS_s077  = 0
 w_k1_DIS_s100_return_constraint: 1.40458 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.23892 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.595429 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.734921 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.08963 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.718037 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.46074 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.826479 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.07156 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.25812 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.24853 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.865591 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.812361 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.724283 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.76209 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.20732 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_DIS_s120_return_constraint: 1.51355 w_k1_DIS_s12 - w_k1_DIS_s120
                                 + 0.99 b_k1_DIS_s120 - 1.01 s_k1_DIS_s120  = 0
 w_k1_DIS_s121_return_constraint: 0.788141 w_k1_DIS_s12 - w_k1_DIS_s121
                                 + 0.99 b_k1_DIS_s121 - 1.01 s_k1_DIS_s121  = 0
 w_k1_DIS_s122_return_constraint: 0.929816 w_k1_DIS_s12 - w_k1_DIS_s122
                                 + 0.99 b_k1_DIS_s122 - 1.01 s_k1_DIS_s122  = 0
 w_k1_DIS_s123_return_constraint: 1.25914 w_k1_DIS_s12 - w_k1_DIS_s123
                                 + 0.99 b_k1_DIS_s123 - 1.01 s_k1_DIS_s123  = 0
 w_k1_DIS_s124_return_constraint: 1.14318 w_k1_DIS_s12 - w_k1_DIS_s124
                                 + 0.99 b_k1_DIS_s124 - 1.01 s_k1_DIS_s124  = 0
 w_k1_DIS_s125_return_constraint: 1.11342 w_k1_DIS_s12 - w_k1_DIS_s125
                                 + 0.99 b_k1_DIS_s125 - 1.01 s_k1_DIS_s125  = 0
 w_k1_DIS_s126_return_constraint: 0.746641 w_k1_DIS_s12 - w_k1_DIS_s126
                                 + 0.99 b_k1_DIS_s126 - 1.01 s_k1_DIS_s126  = 0
 w_k1_DIS_s127_return_constraint: 1.05854 w_k1_DIS_s12 - w_k1_DIS_s127
                                 + 0.99 b_k1_DIS_s127 - 1.01 s_k1_DIS_s127  = 0
 w_k1_DIS_s130_return_constraint: 1.20989 w_k1_DIS_s13 - w_k1_DIS_s130
                                 + 0.99 b_k1_DIS_s130 - 1.01 s_k1_DIS_s130  = 0
 w_k1_DIS_s131_return_constraint: 1.29453 w_k1_DIS_s13 - w_k1_DIS_s131
                                 + 0.99 b_k1_DIS_s131 - 1.01 s_k1_DIS_s131  = 0
 w_k1_DIS_s132_return_constraint: 1.02118 w_k1_DIS_s13 - w_k1_DIS_s132
                                 + 0.99 b_k1_DIS_s132 - 1.01 s_k1_DIS_s132  = 0
 w_k1_DIS_s133_return_constraint: 0.581468 w_k1_DIS_s13 - w_k1_DIS_s133
                                 + 0.99 b_k1_DIS_s133 - 1.01 s_k1_DIS_s133  = 0
 w_k1_DIS_s134_return_constraint: 1.08246 w_k1_DIS_s13 - w_k1_DIS_s134
                                 + 0.99 b_k1_DIS_s134 - 1.01 s_k1_DIS_s134  = 0
 w_k1_DIS_s135_return_constraint: 1.62723 w_k1_DIS_s13 - w_k1_DIS_s135
                                 + 0.99 b_k1_DIS_s135 - 1.01 s_k1_DIS_s135  = 0
 w_k1_DIS_s136_return_constraint: 0.777695 w_k1_DIS_s13 - w_k1_DIS_s136
                                 + 0.99 b_k1_DIS_s136 - 1.01 s_k1_DIS_s136  = 0
 w_k1_DIS_s137_return_constraint: 0.742947 w_k1_DIS_s13 - w_k1_DIS_s137
                                 + 0.99 b_k1_DIS_s137 - 1.01 s_k1_DIS_s137  = 0
 w_k1_DIS_s140_return_constraint: 0.846184 w_k1_DIS_s14 - w_k1_DIS_s140
                                 + 0.99 b_k1_DIS_s140 - 1.01 s_k1_DIS_s140  = 0
 w_k1_DIS_s141_return_constraint: 1.1748 w_k1_DIS_s14 - w_k1_DIS_s141
                                 + 0.99 b_k1_DIS_s141 - 1.01 s_k1_DIS_s141  = 0
 w_k1_DIS_s142_return_constraint: 1.03339 w_k1_DIS_s14 - w_k1_DIS_s142
                                 + 0.99 b_k1_DIS_s142 - 1.01 s_k1_DIS_s142  = 0
 w_k1_DIS_s143_return_constraint: 0.744279 w_k1_DIS_s14 - w_k1_DIS_s143
                                 + 0.99 b_k1_DIS_s143 - 1.01 s_k1_DIS_s143  = 0
 w_k1_DIS_s144_return_constraint: 1.27089 w_k1_DIS_s14 - w_k1_DIS_s144
                                 + 0.99 b_k1_DIS_s144 - 1.01 s_k1_DIS_s144  = 0
 w_k1_DIS_s145_return_constraint: 1.32478 w_k1_DIS_s14 - w_k1_DIS_s145
                                 + 0.99 b_k1_DIS_s145 - 1.01 s_k1_DIS_s145  = 0
 w_k1_DIS_s146_return_constraint: 0.694626 w_k1_DIS_s14 - w_k1_DIS_s146
                                 + 0.99 b_k1_DIS_s146 - 1.01 s_k1_DIS_s146  = 0
 w_k1_DIS_s147_return_constraint: 1.36604 w_k1_DIS_s14 - w_k1_DIS_s147
                                 + 0.99 b_k1_DIS_s147 - 1.01 s_k1_DIS_s147  = 0
 w_k1_DIS_s150_return_constraint: 1.40354 w_k1_DIS_s15 - w_k1_DIS_s150
                                 + 0.99 b_k1_DIS_s150 - 1.01 s_k1_DIS_s150  = 0
 w_k1_DIS_s151_return_constraint: 1.06634 w_k1_DIS_s15 - w_k1_DIS_s151
                                 + 0.99 b_k1_DIS_s151 - 1.01 s_k1_DIS_s151  = 0
 w_k1_DIS_s152_return_constraint: 1.21615 w_k1_DIS_s15 - w_k1_DIS_s152
                                 + 0.99 b_k1_DIS_s152 - 1.01 s_k1_DIS_s152  = 0
 w_k1_DIS_s153_return_constraint: 1.16617 w_k1_DIS_s15 - w_k1_DIS_s153
                                 + 0.99 b_k1_DIS_s153 - 1.01 s_k1_DIS_s153  = 0
 w_k1_DIS_s154_return_constraint: 1.57245 w_k1_DIS_s15 - w_k1_DIS_s154
                                 + 0.99 b_k1_DIS_s154 - 1.01 s_k1_DIS_s154  = 0
 w_k1_DIS_s155_return_constraint: 0.874569 w_k1_DIS_s15 - w_k1_DIS_s155
                                 + 0.99 b_k1_DIS_s155 - 1.01 s_k1_DIS_s155  = 0
 w_k1_DIS_s156_return_constraint: 0.947549 w_k1_DIS_s15 - w_k1_DIS_s156
                                 + 0.99 b_k1_DIS_s156 - 1.01 s_k1_DIS_s156  = 0
 w_k1_DIS_s157_return_constraint: 0.579631 w_k1_DIS_s15 - w_k1_DIS_s157
                                 + 0.99 b_k1_DIS_s157 - 1.01 s_k1_DIS_s157  = 0
 w_k1_DIS_s160_return_constraint: 0.655053 w_k1_DIS_s16 - w_k1_DIS_s160
                                 + 0.99 b_k1_DIS_s160 - 1.01 s_k1_DIS_s160  = 0
 w_k1_DIS_s161_return_constraint: 1.16349 w_k1_DIS_s16 - w_k1_DIS_s161
                                 + 0.99 b_k1_DIS_s161 - 1.01 s_k1_DIS_s161  = 0
 w_k1_DIS_s162_return_constraint: 1.25731 w_k1_DIS_s16 - w_k1_DIS_s162
                                 + 0.99 b_k1_DIS_s162 - 1.01 s_k1_DIS_s162  = 0
 w_k1_DIS_s163_return_constraint: 0.752888 w_k1_DIS_s16 - w_k1_DIS_s163
                                 + 0.99 b_k1_DIS_s163 - 1.01 s_k1_DIS_s163  = 0
 w_k1_DIS_s164_return_constraint: 1.74256 w_k1_DIS_s16 - w_k1_DIS_s164
                                 + 0.99 b_k1_DIS_s164 - 1.01 s_k1_DIS_s164  = 0
 w_k1_DIS_s165_return_constraint: 1.01464 w_k1_DIS_s16 - w_k1_DIS_s165
                                 + 0.99 b_k1_DIS_s165 - 1.01 s_k1_DIS_s165  = 0
 w_k1_DIS_s166_return_constraint: 0.852783 w_k1_DIS_s16 - w_k1_DIS_s166
                                 + 0.99 b_k1_DIS_s166 - 1.01 s_k1_DIS_s166  = 0
 w_k1_DIS_s167_return_constraint: 1.14559 w_k1_DIS_s16 - w_k1_DIS_s167
                                 + 0.99 b_k1_DIS_s167 - 1.01 s_k1_DIS_s167  = 0
 w_k1_DIS_s170_return_constraint: 0.762045 w_k1_DIS_s17 - w_k1_DIS_s170
                                 + 0.99 b_k1_DIS_s170 - 1.01 s_k1_DIS_s170  = 0
 w_k1_DIS_s171_return_constraint: 1.21682 w_k1_DIS_s17 - w_k1_DIS_s171
                                 + 0.99 b_k1_DIS_s171 - 1.01 s_k1_DIS_s171  = 0
 w_k1_DIS_s172_return_constraint: 1.6412 w_k1_DIS_s17 - w_k1_DIS_s172
                                 + 0.99 b_k1_DIS_s172 - 1.01 s_k1_DIS_s172  = 0
 w_k1_DIS_s173_return_constraint: 0.956824 w_k1_DIS_s17 - w_k1_DIS_s173
                                 + 0.99 b_k1_DIS_s173 - 1.01 s_k1_DIS_s173  = 0
 w_k1_DIS_s174_return_constraint: 0.684973 w_k1_DIS_s17 - w_k1_DIS_s174
                                 + 0.99 b_k1_DIS_s174 - 1.01 s_k1_DIS_s174  = 0
 w_k1_DIS_s175_return_constraint: 1.172 w_k1_DIS_s17 - w_k1_DIS_s175
                                 + 0.99 b_k1_DIS_s175 - 1.01 s_k1_DIS_s175  = 0
 w_k1_DIS_s176_return_constraint: 1.25275 w_k1_DIS_s17 - w_k1_DIS_s176
                                 + 0.99 b_k1_DIS_s176 - 1.01 s_k1_DIS_s176  = 0
 w_k1_DIS_s177_return_constraint: 0.937688 w_k1_DIS_s17 - w_k1_DIS_s177
                                 + 0.99 b_k1_DIS_s177 - 1.01 s_k1_DIS_s177  = 0
 w_k1_FDX_s0_return_constraint:  1.18097 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.888366 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.931849 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.30675 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s02_return_constraint: 0.901986 w_k1_FDX_s0 - w_k1_FDX_s02
                                 + 0.99 b_k1_FDX_s02 - 1.01 s_k1_FDX_s02  = 0
 w_k1_FDX_s03_return_constraint: 1.04141 w_k1_FDX_s0 - w_k1_FDX_s03
                                 + 0.99 b_k1_FDX_s03 - 1.01 s_k1_FDX_s03  = 0
 w_k1_FDX_s04_return_constraint: 0.782916 w_k1_FDX_s0 - w_k1_FDX_s04
                                 + 0.99 b_k1_FDX_s04 - 1.01 s_k1_FDX_s04  = 0
 w_k1_FDX_s05_return_constraint: 1.06544 w_k1_FDX_s0 - w_k1_FDX_s05
                                 + 0.99 b_k1_FDX_s05 - 1.01 s_k1_FDX_s05  = 0
 w_k1_FDX_s06_return_constraint: 0.792013 w_k1_FDX_s0 - w_k1_FDX_s06
                                 + 0.99 b_k1_FDX_s06 - 1.01 s_k1_FDX_s06  = 0
 w_k1_FDX_s07_return_constraint: 1.2965 w_k1_FDX_s0 - w_k1_FDX_s07
                                 + 0.99 b_k1_FDX_s07 - 1.01 s_k1_FDX_s07  = 0
 w_k1_FDX_s10_return_constraint: 0.856227 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.20313 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s12_return_constraint: 1.51674 w_k1_FDX_s1 - w_k1_FDX_s12
                                 + 0.99 b_k1_FDX_s12 - 1.01 s_k1_FDX_s12  = 0
 w_k1_FDX_s13_return_constraint: 1.26399 w_k1_FDX_s1 - w_k1_FDX_s13
                                 + 0.99 b_k1_FDX_s13 - 1.01 s_k1_FDX_s13  = 0
 w_k1_FDX_s14_return_constraint: 1.01304 w_k1_FDX_s1 - w_k1_FDX_s14
                                 + 0.99 b_k1_FDX_s14 - 1.01 s_k1_FDX_s14  = 0
 w_k1_FDX_s15_return_constraint: 0.976092 w_k1_FDX_s1 - w_k1_FDX_s15
                                 + 0.99 b_k1_FDX_s15 - 1.01 s_k1_FDX_s15  = 0
 w_k1_FDX_s16_return_constraint: 0.919 w_k1_FDX_s1 - w_k1_FDX_s16
                                 + 0.99 b_k1_FDX_s16 - 1.01 s_k1_FDX_s16  = 0
 w_k1_FDX_s17_return_constraint: 0.906969 w_k1_FDX_s1 - w_k1_FDX_s17
                                 + 0.99 b_k1_FDX_s17 - 1.01 s_k1_FDX_s17  = 0
 w_k1_FDX_s000_return_constraint: 1.02576 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.16985 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.832497 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.22303 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.962285 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.04357 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.11021 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.789834 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.855407 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.92762 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.24808 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.11523 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.762639 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.25543 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.932912 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.19189 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s020_return_constraint: 0.808186 w_k1_FDX_s02 - w_k1_FDX_s020
                                 + 0.99 b_k1_FDX_s020 - 1.01 s_k1_FDX_s020  = 0
 w_k1_FDX_s021_return_constraint: 1.19237 w_k1_FDX_s02 - w_k1_FDX_s021
                                 + 0.99 b_k1_FDX_s021 - 1.01 s_k1_FDX_s021  = 0
 w_k1_FDX_s022_return_constraint: 0.94555 w_k1_FDX_s02 - w_k1_FDX_s022
                                 + 0.99 b_k1_FDX_s022 - 1.01 s_k1_FDX_s022  = 0
 w_k1_FDX_s023_return_constraint: 0.852276 w_k1_FDX_s02 - w_k1_FDX_s023
                                 + 0.99 b_k1_FDX_s023 - 1.01 s_k1_FDX_s023  = 0
 w_k1_FDX_s024_return_constraint: 1.14272 w_k1_FDX_s02 - w_k1_FDX_s024
                                 + 0.99 b_k1_FDX_s024 - 1.01 s_k1_FDX_s024  = 0
 w_k1_FDX_s025_return_constraint: 0.898989 w_k1_FDX_s02 - w_k1_FDX_s025
                                 + 0.99 b_k1_FDX_s025 - 1.01 s_k1_FDX_s025  = 0
 w_k1_FDX_s026_return_constraint: 0.829776 w_k1_FDX_s02 - w_k1_FDX_s026
                                 + 0.99 b_k1_FDX_s026 - 1.01 s_k1_FDX_s026  = 0
 w_k1_FDX_s027_return_constraint: 1.09264 w_k1_FDX_s02 - w_k1_FDX_s027
                                 + 0.99 b_k1_FDX_s027 - 1.01 s_k1_FDX_s027  = 0
 w_k1_FDX_s030_return_constraint: 1.10648 w_k1_FDX_s03 - w_k1_FDX_s030
                                 + 0.99 b_k1_FDX_s030 - 1.01 s_k1_FDX_s030  = 0
 w_k1_FDX_s031_return_constraint: 1.44331 w_k1_FDX_s03 - w_k1_FDX_s031
                                 + 0.99 b_k1_FDX_s031 - 1.01 s_k1_FDX_s031  = 0
 w_k1_FDX_s032_return_constraint: 0.798948 w_k1_FDX_s03 - w_k1_FDX_s032
                                 + 0.99 b_k1_FDX_s032 - 1.01 s_k1_FDX_s032  = 0
 w_k1_FDX_s033_return_constraint: 0.903351 w_k1_FDX_s03 - w_k1_FDX_s033
                                 + 0.99 b_k1_FDX_s033 - 1.01 s_k1_FDX_s033  = 0
 w_k1_FDX_s034_return_constraint: 1.07713 w_k1_FDX_s03 - w_k1_FDX_s034
                                 + 0.99 b_k1_FDX_s034 - 1.01 s_k1_FDX_s034  = 0
 w_k1_FDX_s035_return_constraint: 1.18998 w_k1_FDX_s03 - w_k1_FDX_s035
                                 + 0.99 b_k1_FDX_s035 - 1.01 s_k1_FDX_s035  = 0
 w_k1_FDX_s036_return_constraint: 1.06454 w_k1_FDX_s03 - w_k1_FDX_s036
                                 + 0.99 b_k1_FDX_s036 - 1.01 s_k1_FDX_s036  = 0
 w_k1_FDX_s037_return_constraint: 0.884396 w_k1_FDX_s03 - w_k1_FDX_s037
                                 + 0.99 b_k1_FDX_s037 - 1.01 s_k1_FDX_s037  = 0
 w_k1_FDX_s040_return_constraint: 1.33408 w_k1_FDX_s04 - w_k1_FDX_s040
                                 + 0.99 b_k1_FDX_s040 - 1.01 s_k1_FDX_s040  = 0
 w_k1_FDX_s041_return_constraint: 0.921423 w_k1_FDX_s04 - w_k1_FDX_s041
                                 + 0.99 b_k1_FDX_s041 - 1.01 s_k1_FDX_s041  = 0
 w_k1_FDX_s042_return_constraint: 1.12069 w_k1_FDX_s04 - w_k1_FDX_s042
                                 + 0.99 b_k1_FDX_s042 - 1.01 s_k1_FDX_s042  = 0
 w_k1_FDX_s043_return_constraint: 1.06012 w_k1_FDX_s04 - w_k1_FDX_s043
                                 + 0.99 b_k1_FDX_s043 - 1.01 s_k1_FDX_s043  = 0
 w_k1_FDX_s044_return_constraint: 1.12221 w_k1_FDX_s04 - w_k1_FDX_s044
                                 + 0.99 b_k1_FDX_s044 - 1.01 s_k1_FDX_s044  = 0
 w_k1_FDX_s045_return_constraint: 0.862441 w_k1_FDX_s04 - w_k1_FDX_s045
                                 + 0.99 b_k1_FDX_s045 - 1.01 s_k1_FDX_s045  = 0
 w_k1_FDX_s046_return_constraint: 0.665941 w_k1_FDX_s04 - w_k1_FDX_s046
                                 + 0.99 b_k1_FDX_s046 - 1.01 s_k1_FDX_s046  = 0
 w_k1_FDX_s047_return_constraint: 0.828357 w_k1_FDX_s04 - w_k1_FDX_s047
                                 + 0.99 b_k1_FDX_s047 - 1.01 s_k1_FDX_s047  = 0
 w_k1_FDX_s050_return_constraint: 0.699491 w_k1_FDX_s05 - w_k1_FDX_s050
                                 + 0.99 b_k1_FDX_s050 - 1.01 s_k1_FDX_s050  = 0
 w_k1_FDX_s051_return_constraint: 0.989005 w_k1_FDX_s05 - w_k1_FDX_s051
                                 + 0.99 b_k1_FDX_s051 - 1.01 s_k1_FDX_s051  = 0
 w_k1_FDX_s052_return_constraint: 0.806534 w_k1_FDX_s05 - w_k1_FDX_s052
                                 + 0.99 b_k1_FDX_s052 - 1.01 s_k1_FDX_s052  = 0
 w_k1_FDX_s053_return_constraint: 1.03023 w_k1_FDX_s05 - w_k1_FDX_s053
                                 + 0.99 b_k1_FDX_s053 - 1.01 s_k1_FDX_s053  = 0
 w_k1_FDX_s054_return_constraint: 1.13718 w_k1_FDX_s05 - w_k1_FDX_s054
                                 + 0.99 b_k1_FDX_s054 - 1.01 s_k1_FDX_s054  = 0
 w_k1_FDX_s055_return_constraint: 0.867898 w_k1_FDX_s05 - w_k1_FDX_s055
                                 + 0.99 b_k1_FDX_s055 - 1.01 s_k1_FDX_s055  = 0
 w_k1_FDX_s056_return_constraint: 1.36403 w_k1_FDX_s05 - w_k1_FDX_s056
                                 + 0.99 b_k1_FDX_s056 - 1.01 s_k1_FDX_s056  = 0
 w_k1_FDX_s057_return_constraint: 1.01783 w_k1_FDX_s05 - w_k1_FDX_s057
                                 + 0.99 b_k1_FDX_s057 - 1.01 s_k1_FDX_s057  = 0
 w_k1_FDX_s060_return_constraint: 1.2165 w_k1_FDX_s06 - w_k1_FDX_s060
                                 + 0.99 b_k1_FDX_s060 - 1.01 s_k1_FDX_s060  = 0
 w_k1_FDX_s061_return_constraint: 1.0576 w_k1_FDX_s06 - w_k1_FDX_s061
                                 + 0.99 b_k1_FDX_s061 - 1.01 s_k1_FDX_s061  = 0
 w_k1_FDX_s062_return_constraint: 0.896318 w_k1_FDX_s06 - w_k1_FDX_s062
                                 + 0.99 b_k1_FDX_s062 - 1.01 s_k1_FDX_s062  = 0
 w_k1_FDX_s063_return_constraint: 0.800056 w_k1_FDX_s06 - w_k1_FDX_s063
                                 + 0.99 b_k1_FDX_s063 - 1.01 s_k1_FDX_s063  = 0
 w_k1_FDX_s064_return_constraint: 0.993249 w_k1_FDX_s06 - w_k1_FDX_s064
                                 + 0.99 b_k1_FDX_s064 - 1.01 s_k1_FDX_s064  = 0
 w_k1_FDX_s065_return_constraint: 0.7885 w_k1_FDX_s06 - w_k1_FDX_s065
                                 + 0.99 b_k1_FDX_s065 - 1.01 s_k1_FDX_s065  = 0
 w_k1_FDX_s066_return_constraint: 1.3619 w_k1_FDX_s06 - w_k1_FDX_s066
                                 + 0.99 b_k1_FDX_s066 - 1.01 s_k1_FDX_s066  = 0
 w_k1_FDX_s067_return_constraint: 1.02685 w_k1_FDX_s06 - w_k1_FDX_s067
                                 + 0.99 b_k1_FDX_s067 - 1.01 s_k1_FDX_s067  = 0
 w_k1_FDX_s070_return_constraint: 0.989725 w_k1_FDX_s07 - w_k1_FDX_s070
                                 + 0.99 b_k1_FDX_s070 - 1.01 s_k1_FDX_s070  = 0
 w_k1_FDX_s071_return_constraint: 1.50506 w_k1_FDX_s07 - w_k1_FDX_s071
                                 + 0.99 b_k1_FDX_s071 - 1.01 s_k1_FDX_s071  = 0
 w_k1_FDX_s072_return_constraint: 0.885469 w_k1_FDX_s07 - w_k1_FDX_s072
                                 + 0.99 b_k1_FDX_s072 - 1.01 s_k1_FDX_s072  = 0
 w_k1_FDX_s073_return_constraint: 0.871032 w_k1_FDX_s07 - w_k1_FDX_s073
                                 + 0.99 b_k1_FDX_s073 - 1.01 s_k1_FDX_s073  = 0
 w_k1_FDX_s074_return_constraint: 1.17789 w_k1_FDX_s07 - w_k1_FDX_s074
                                 + 0.99 b_k1_FDX_s074 - 1.01 s_k1_FDX_s074  = 0
 w_k1_FDX_s075_return_constraint: 0.667367 w_k1_FDX_s07 - w_k1_FDX_s075
                                 + 0.99 b_k1_FDX_s075 - 1.01 s_k1_FDX_s075  = 0
 w_k1_FDX_s076_return_constraint: 0.874631 w_k1_FDX_s07 - w_k1_FDX_s076
                                 + 0.99 b_k1_FDX_s076 - 1.01 s_k1_FDX_s076  = 0
 w_k1_FDX_s077_return_constraint: 1.18328 w_k1_FDX_s07 - w_k1_FDX_s077
                                 + 0.99 b_k1_FDX_s077 - 1.01 s_k1_FDX_s077  = 0
 w_k1_FDX_s100_return_constraint: 1.23716 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.007 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.650863 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.914274 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.07812 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.740911 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.24706 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.850133 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.12989 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.19615 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.10668 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.878222 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.840826 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.837733 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.40157 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.02479 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_FDX_s120_return_constraint: 1.33774 w_k1_FDX_s12 - w_k1_FDX_s120
                                 + 0.99 b_k1_FDX_s120 - 1.01 s_k1_FDX_s120  = 0
 w_k1_FDX_s121_return_constraint: 0.874122 w_k1_FDX_s12 - w_k1_FDX_s121
                                 + 0.99 b_k1_FDX_s121 - 1.01 s_k1_FDX_s121  = 0
 w_k1_FDX_s122_return_constraint: 1.03617 w_k1_FDX_s12 - w_k1_FDX_s122
                                 + 0.99 b_k1_FDX_s122 - 1.01 s_k1_FDX_s122  = 0
 w_k1_FDX_s123_return_constraint: 1.02158 w_k1_FDX_s12 - w_k1_FDX_s123
                                 + 0.99 b_k1_FDX_s123 - 1.01 s_k1_FDX_s123  = 0
 w_k1_FDX_s124_return_constraint: 1.17236 w_k1_FDX_s12 - w_k1_FDX_s124
                                 + 0.99 b_k1_FDX_s124 - 1.01 s_k1_FDX_s124  = 0
 w_k1_FDX_s125_return_constraint: 1.11613 w_k1_FDX_s12 - w_k1_FDX_s125
                                 + 0.99 b_k1_FDX_s125 - 1.01 s_k1_FDX_s125  = 0
 w_k1_FDX_s126_return_constraint: 0.761312 w_k1_FDX_s12 - w_k1_FDX_s126
                                 + 0.99 b_k1_FDX_s126 - 1.01 s_k1_FDX_s126  = 0
 w_k1_FDX_s127_return_constraint: 1.00018 w_k1_FDX_s12 - w_k1_FDX_s127
                                 + 0.99 b_k1_FDX_s127 - 1.01 s_k1_FDX_s127  = 0
 w_k1_FDX_s130_return_constraint: 1.269 w_k1_FDX_s13 - w_k1_FDX_s130
                                 + 0.99 b_k1_FDX_s130 - 1.01 s_k1_FDX_s130  = 0
 w_k1_FDX_s131_return_constraint: 1.11197 w_k1_FDX_s13 - w_k1_FDX_s131
                                 + 0.99 b_k1_FDX_s131 - 1.01 s_k1_FDX_s131  = 0
 w_k1_FDX_s132_return_constraint: 1.00951 w_k1_FDX_s13 - w_k1_FDX_s132
                                 + 0.99 b_k1_FDX_s132 - 1.01 s_k1_FDX_s132  = 0
 w_k1_FDX_s133_return_constraint: 0.7073 w_k1_FDX_s13 - w_k1_FDX_s133
                                 + 0.99 b_k1_FDX_s133 - 1.01 s_k1_FDX_s133  = 0
 w_k1_FDX_s134_return_constraint: 1.00207 w_k1_FDX_s13 - w_k1_FDX_s134
                                 + 0.99 b_k1_FDX_s134 - 1.01 s_k1_FDX_s134  = 0
 w_k1_FDX_s135_return_constraint: 1.31136 w_k1_FDX_s13 - w_k1_FDX_s135
                                 + 0.99 b_k1_FDX_s135 - 1.01 s_k1_FDX_s135  = 0
 w_k1_FDX_s136_return_constraint: 0.876415 w_k1_FDX_s13 - w_k1_FDX_s136
                                 + 0.99 b_k1_FDX_s136 - 1.01 s_k1_FDX_s136  = 0
 w_k1_FDX_s137_return_constraint: 0.885426 w_k1_FDX_s13 - w_k1_FDX_s137
                                 + 0.99 b_k1_FDX_s137 - 1.01 s_k1_FDX_s137  = 0
 w_k1_FDX_s140_return_constraint: 0.818799 w_k1_FDX_s14 - w_k1_FDX_s140
                                 + 0.99 b_k1_FDX_s140 - 1.01 s_k1_FDX_s140  = 0
 w_k1_FDX_s141_return_constraint: 1.20894 w_k1_FDX_s14 - w_k1_FDX_s141
                                 + 0.99 b_k1_FDX_s141 - 1.01 s_k1_FDX_s141  = 0
 w_k1_FDX_s142_return_constraint: 0.97288 w_k1_FDX_s14 - w_k1_FDX_s142
                                 + 0.99 b_k1_FDX_s142 - 1.01 s_k1_FDX_s142  = 0
 w_k1_FDX_s143_return_constraint: 0.823091 w_k1_FDX_s14 - w_k1_FDX_s143
                                 + 0.99 b_k1_FDX_s143 - 1.01 s_k1_FDX_s143  = 0
 w_k1_FDX_s144_return_constraint: 1.23128 w_k1_FDX_s14 - w_k1_FDX_s144
                                 + 0.99 b_k1_FDX_s144 - 1.01 s_k1_FDX_s144  = 0
 w_k1_FDX_s145_return_constraint: 1.07356 w_k1_FDX_s14 - w_k1_FDX_s145
                                 + 0.99 b_k1_FDX_s145 - 1.01 s_k1_FDX_s145  = 0
 w_k1_FDX_s146_return_constraint: 0.855863 w_k1_FDX_s14 - w_k1_FDX_s146
                                 + 0.99 b_k1_FDX_s146 - 1.01 s_k1_FDX_s146  = 0
 w_k1_FDX_s147_return_constraint: 1.21988 w_k1_FDX_s14 - w_k1_FDX_s147
                                 + 0.99 b_k1_FDX_s147 - 1.01 s_k1_FDX_s147  = 0
 w_k1_FDX_s150_return_constraint: 1.06541 w_k1_FDX_s15 - w_k1_FDX_s150
                                 + 0.99 b_k1_FDX_s150 - 1.01 s_k1_FDX_s150  = 0
 w_k1_FDX_s151_return_constraint: 0.96989 w_k1_FDX_s15 - w_k1_FDX_s151
                                 + 0.99 b_k1_FDX_s151 - 1.01 s_k1_FDX_s151  = 0
 w_k1_FDX_s152_return_constraint: 1.12824 w_k1_FDX_s15 - w_k1_FDX_s152
                                 + 0.99 b_k1_FDX_s152 - 1.01 s_k1_FDX_s152  = 0
 w_k1_FDX_s153_return_constraint: 1.19863 w_k1_FDX_s15 - w_k1_FDX_s153
                                 + 0.99 b_k1_FDX_s153 - 1.01 s_k1_FDX_s153  = 0
 w_k1_FDX_s154_return_constraint: 1.32812 w_k1_FDX_s15 - w_k1_FDX_s154
                                 + 0.99 b_k1_FDX_s154 - 1.01 s_k1_FDX_s154  = 0
 w_k1_FDX_s155_return_constraint: 0.906485 w_k1_FDX_s15 - w_k1_FDX_s155
                                 + 0.99 b_k1_FDX_s155 - 1.01 s_k1_FDX_s155  = 0
 w_k1_FDX_s156_return_constraint: 0.926669 w_k1_FDX_s15 - w_k1_FDX_s156
                                 + 0.99 b_k1_FDX_s156 - 1.01 s_k1_FDX_s156  = 0
 w_k1_FDX_s157_return_constraint: 0.683043 w_k1_FDX_s15 - w_k1_FDX_s157
                                 + 0.99 b_k1_FDX_s157 - 1.01 s_k1_FDX_s157  = 0
 w_k1_FDX_s160_return_constraint: 0.73456 w_k1_FDX_s16 - w_k1_FDX_s160
                                 + 0.99 b_k1_FDX_s160 - 1.01 s_k1_FDX_s160  = 0
 w_k1_FDX_s161_return_constraint: 0.991099 w_k1_FDX_s16 - w_k1_FDX_s161
                                 + 0.99 b_k1_FDX_s161 - 1.01 s_k1_FDX_s161  = 0
 w_k1_FDX_s162_return_constraint: 1.23023 w_k1_FDX_s16 - w_k1_FDX_s162
                                 + 0.99 b_k1_FDX_s162 - 1.01 s_k1_FDX_s162  = 0
 w_k1_FDX_s163_return_constraint: 0.754634 w_k1_FDX_s16 - w_k1_FDX_s163
                                 + 0.99 b_k1_FDX_s163 - 1.01 s_k1_FDX_s163  = 0
 w_k1_FDX_s164_return_constraint: 1.50425 w_k1_FDX_s16 - w_k1_FDX_s164
                                 + 0.99 b_k1_FDX_s164 - 1.01 s_k1_FDX_s164  = 0
 w_k1_FDX_s165_return_constraint: 0.985395 w_k1_FDX_s16 - w_k1_FDX_s165
                                 + 0.99 b_k1_FDX_s165 - 1.01 s_k1_FDX_s165  = 0
 w_k1_FDX_s166_return_constraint: 1.03021 w_k1_FDX_s16 - w_k1_FDX_s166
                                 + 0.99 b_k1_FDX_s166 - 1.01 s_k1_FDX_s166  = 0
 w_k1_FDX_s167_return_constraint: 1.21707 w_k1_FDX_s16 - w_k1_FDX_s167
                                 + 0.99 b_k1_FDX_s167 - 1.01 s_k1_FDX_s167  = 0
 w_k1_FDX_s170_return_constraint: 0.801521 w_k1_FDX_s17 - w_k1_FDX_s170
                                 + 0.99 b_k1_FDX_s170 - 1.01 s_k1_FDX_s170  = 0
 w_k1_FDX_s171_return_constraint: 1.14918 w_k1_FDX_s17 - w_k1_FDX_s171
                                 + 0.99 b_k1_FDX_s171 - 1.01 s_k1_FDX_s171  = 0
 w_k1_FDX_s172_return_constraint: 1.39445 w_k1_FDX_s17 - w_k1_FDX_s172
                                 + 0.99 b_k1_FDX_s172 - 1.01 s_k1_FDX_s172  = 0
 w_k1_FDX_s173_return_constraint: 0.978645 w_k1_FDX_s17 - w_k1_FDX_s173
                                 + 0.99 b_k1_FDX_s173 - 1.01 s_k1_FDX_s173  = 0
 w_k1_FDX_s174_return_constraint: 0.724953 w_k1_FDX_s17 - w_k1_FDX_s174
                                 + 0.99 b_k1_FDX_s174 - 1.01 s_k1_FDX_s174  = 0
 w_k1_FDX_s175_return_constraint: 1.03678 w_k1_FDX_s17 - w_k1_FDX_s175
                                 + 0.99 b_k1_FDX_s175 - 1.01 s_k1_FDX_s175  = 0
 w_k1_FDX_s176_return_constraint: 1.25493 w_k1_FDX_s17 - w_k1_FDX_s176
                                 + 0.99 b_k1_FDX_s176 - 1.01 s_k1_FDX_s176  = 0
 w_k1_FDX_s177_return_constraint: 1.03428 w_k1_FDX_s17 - w_k1_FDX_s177
                                 + 0.99 b_k1_FDX_s177 - 1.01 s_k1_FDX_s177  = 0
 w_k1_IBM_s0_return_constraint:  1.13483 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.949604 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.908583 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.778744 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s02_return_constraint: 1.16869 w_k1_IBM_s0 - w_k1_IBM_s02
                                 + 0.99 b_k1_IBM_s02 - 1.01 s_k1_IBM_s02  = 0
 w_k1_IBM_s03_return_constraint: 0.670583 w_k1_IBM_s0 - w_k1_IBM_s03
                                 + 0.99 b_k1_IBM_s03 - 1.01 s_k1_IBM_s03  = 0
 w_k1_IBM_s04_return_constraint: 0.677919 w_k1_IBM_s0 - w_k1_IBM_s04
                                 + 0.99 b_k1_IBM_s04 - 1.01 s_k1_IBM_s04  = 0
 w_k1_IBM_s05_return_constraint: 1.19119 w_k1_IBM_s0 - w_k1_IBM_s05
                                 + 0.99 b_k1_IBM_s05 - 1.01 s_k1_IBM_s05  = 0
 w_k1_IBM_s06_return_constraint: 1.01356 w_k1_IBM_s0 - w_k1_IBM_s06
                                 + 0.99 b_k1_IBM_s06 - 1.01 s_k1_IBM_s06  = 0
 w_k1_IBM_s07_return_constraint: 1.42957 w_k1_IBM_s0 - w_k1_IBM_s07
                                 + 0.99 b_k1_IBM_s07 - 1.01 s_k1_IBM_s07  = 0
 w_k1_IBM_s10_return_constraint: 0.668142 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.52013 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s12_return_constraint: 1.24273 w_k1_IBM_s1 - w_k1_IBM_s12
                                 + 0.99 b_k1_IBM_s12 - 1.01 s_k1_IBM_s12  = 0
 w_k1_IBM_s13_return_constraint: 0.889969 w_k1_IBM_s1 - w_k1_IBM_s13
                                 + 0.99 b_k1_IBM_s13 - 1.01 s_k1_IBM_s13  = 0
 w_k1_IBM_s14_return_constraint: 1.52312 w_k1_IBM_s1 - w_k1_IBM_s14
                                 + 0.99 b_k1_IBM_s14 - 1.01 s_k1_IBM_s14  = 0
 w_k1_IBM_s15_return_constraint: 1.15025 w_k1_IBM_s1 - w_k1_IBM_s15
                                 + 0.99 b_k1_IBM_s15 - 1.01 s_k1_IBM_s15  = 0
 w_k1_IBM_s16_return_constraint: 0.795112 w_k1_IBM_s1 - w_k1_IBM_s16
                                 + 0.99 b_k1_IBM_s16 - 1.01 s_k1_IBM_s16  = 0
 w_k1_IBM_s17_return_constraint: 0.985193 w_k1_IBM_s1 - w_k1_IBM_s17
                                 + 0.99 b_k1_IBM_s17 - 1.01 s_k1_IBM_s17  = 0
 w_k1_IBM_s000_return_constraint: 1.02173 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.3991 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.967604 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.936143 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.16715 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.776405 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.753499 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.714306 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.878523 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.679376 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.08159 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.911942 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.1695 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.04156 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.29774 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.50906 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s020_return_constraint: 1.02769 w_k1_IBM_s02 - w_k1_IBM_s020
                                 + 0.99 b_k1_IBM_s020 - 1.01 s_k1_IBM_s020  = 0
 w_k1_IBM_s021_return_constraint: 1.00581 w_k1_IBM_s02 - w_k1_IBM_s021
                                 + 0.99 b_k1_IBM_s021 - 1.01 s_k1_IBM_s021  = 0
 w_k1_IBM_s022_return_constraint: 1.66874 w_k1_IBM_s02 - w_k1_IBM_s022
                                 + 0.99 b_k1_IBM_s022 - 1.01 s_k1_IBM_s022  = 0
 w_k1_IBM_s023_return_constraint: 1.16038 w_k1_IBM_s02 - w_k1_IBM_s023
                                 + 0.99 b_k1_IBM_s023 - 1.01 s_k1_IBM_s023  = 0
 w_k1_IBM_s024_return_constraint: 1.29368 w_k1_IBM_s02 - w_k1_IBM_s024
                                 + 0.99 b_k1_IBM_s024 - 1.01 s_k1_IBM_s024  = 0
 w_k1_IBM_s025_return_constraint: 0.948932 w_k1_IBM_s02 - w_k1_IBM_s025
                                 + 0.99 b_k1_IBM_s025 - 1.01 s_k1_IBM_s025  = 0
 w_k1_IBM_s026_return_constraint: 0.698767 w_k1_IBM_s02 - w_k1_IBM_s026
                                 + 0.99 b_k1_IBM_s026 - 1.01 s_k1_IBM_s026  = 0
 w_k1_IBM_s027_return_constraint: 0.730312 w_k1_IBM_s02 - w_k1_IBM_s027
                                 + 0.99 b_k1_IBM_s027 - 1.01 s_k1_IBM_s027  = 0
 w_k1_IBM_s030_return_constraint: 1.47766 w_k1_IBM_s03 - w_k1_IBM_s030
                                 + 0.99 b_k1_IBM_s030 - 1.01 s_k1_IBM_s030  = 0
 w_k1_IBM_s031_return_constraint: 1.18695 w_k1_IBM_s03 - w_k1_IBM_s031
                                 + 0.99 b_k1_IBM_s031 - 1.01 s_k1_IBM_s031  = 0
 w_k1_IBM_s032_return_constraint: 0.70789 w_k1_IBM_s03 - w_k1_IBM_s032
                                 + 0.99 b_k1_IBM_s032 - 1.01 s_k1_IBM_s032  = 0
 w_k1_IBM_s033_return_constraint: 1.01099 w_k1_IBM_s03 - w_k1_IBM_s033
                                 + 0.99 b_k1_IBM_s033 - 1.01 s_k1_IBM_s033  = 0
 w_k1_IBM_s034_return_constraint: 1.09426 w_k1_IBM_s03 - w_k1_IBM_s034
                                 + 0.99 b_k1_IBM_s034 - 1.01 s_k1_IBM_s034  = 0
 w_k1_IBM_s035_return_constraint: 0.962603 w_k1_IBM_s03 - w_k1_IBM_s035
                                 + 0.99 b_k1_IBM_s035 - 1.01 s_k1_IBM_s035  = 0
 w_k1_IBM_s036_return_constraint: 0.855819 w_k1_IBM_s03 - w_k1_IBM_s036
                                 + 0.99 b_k1_IBM_s036 - 1.01 s_k1_IBM_s036  = 0
 w_k1_IBM_s037_return_constraint: 1.46013 w_k1_IBM_s03 - w_k1_IBM_s037
                                 + 0.99 b_k1_IBM_s037 - 1.01 s_k1_IBM_s037  = 0
 w_k1_IBM_s040_return_constraint: 1.18112 w_k1_IBM_s04 - w_k1_IBM_s040
                                 + 0.99 b_k1_IBM_s040 - 1.01 s_k1_IBM_s040  = 0
 w_k1_IBM_s041_return_constraint: 1.52427 w_k1_IBM_s04 - w_k1_IBM_s041
                                 + 0.99 b_k1_IBM_s041 - 1.01 s_k1_IBM_s041  = 0
 w_k1_IBM_s042_return_constraint: 0.811737 w_k1_IBM_s04 - w_k1_IBM_s042
                                 + 0.99 b_k1_IBM_s042 - 1.01 s_k1_IBM_s042  = 0
 w_k1_IBM_s043_return_constraint: 1.01796 w_k1_IBM_s04 - w_k1_IBM_s043
                                 + 0.99 b_k1_IBM_s043 - 1.01 s_k1_IBM_s043  = 0
 w_k1_IBM_s044_return_constraint: 1.3821 w_k1_IBM_s04 - w_k1_IBM_s044
                                 + 0.99 b_k1_IBM_s044 - 1.01 s_k1_IBM_s044  = 0
 w_k1_IBM_s045_return_constraint: 0.668728 w_k1_IBM_s04 - w_k1_IBM_s045
                                 + 0.99 b_k1_IBM_s045 - 1.01 s_k1_IBM_s045  = 0
 w_k1_IBM_s046_return_constraint: 0.879623 w_k1_IBM_s04 - w_k1_IBM_s046
                                 + 0.99 b_k1_IBM_s046 - 1.01 s_k1_IBM_s046  = 0
 w_k1_IBM_s047_return_constraint: 1.13594 w_k1_IBM_s04 - w_k1_IBM_s047
                                 + 0.99 b_k1_IBM_s047 - 1.01 s_k1_IBM_s047  = 0
 w_k1_IBM_s050_return_constraint: 0.683796 w_k1_IBM_s05 - w_k1_IBM_s050
                                 + 0.99 b_k1_IBM_s050 - 1.01 s_k1_IBM_s050  = 0
 w_k1_IBM_s051_return_constraint: 1.51403 w_k1_IBM_s05 - w_k1_IBM_s051
                                 + 0.99 b_k1_IBM_s051 - 1.01 s_k1_IBM_s051  = 0
 w_k1_IBM_s052_return_constraint: 1.00671 w_k1_IBM_s05 - w_k1_IBM_s052
                                 + 0.99 b_k1_IBM_s052 - 1.01 s_k1_IBM_s052  = 0
 w_k1_IBM_s053_return_constraint: 1.0938 w_k1_IBM_s05 - w_k1_IBM_s053
                                 + 0.99 b_k1_IBM_s053 - 1.01 s_k1_IBM_s053  = 0
 w_k1_IBM_s054_return_constraint: 0.716043 w_k1_IBM_s05 - w_k1_IBM_s054
                                 + 0.99 b_k1_IBM_s054 - 1.01 s_k1_IBM_s054  = 0
 w_k1_IBM_s055_return_constraint: 0.730584 w_k1_IBM_s05 - w_k1_IBM_s055
                                 + 0.99 b_k1_IBM_s055 - 1.01 s_k1_IBM_s055  = 0
 w_k1_IBM_s056_return_constraint: 1.22581 w_k1_IBM_s05 - w_k1_IBM_s056
                                 + 0.99 b_k1_IBM_s056 - 1.01 s_k1_IBM_s056  = 0
 w_k1_IBM_s057_return_constraint: 0.891453 w_k1_IBM_s05 - w_k1_IBM_s057
                                 + 0.99 b_k1_IBM_s057 - 1.01 s_k1_IBM_s057  = 0
 w_k1_IBM_s060_return_constraint: 0.775211 w_k1_IBM_s06 - w_k1_IBM_s060
                                 + 0.99 b_k1_IBM_s060 - 1.01 s_k1_IBM_s060  = 0
 w_k1_IBM_s061_return_constraint: 0.752956 w_k1_IBM_s06 - w_k1_IBM_s061
                                 + 0.99 b_k1_IBM_s061 - 1.01 s_k1_IBM_s061  = 0
 w_k1_IBM_s062_return_constraint: 0.632991 w_k1_IBM_s06 - w_k1_IBM_s062
                                 + 0.99 b_k1_IBM_s062 - 1.01 s_k1_IBM_s062  = 0
 w_k1_IBM_s063_return_constraint: 0.725246 w_k1_IBM_s06 - w_k1_IBM_s063
                                 + 0.99 b_k1_IBM_s063 - 1.01 s_k1_IBM_s063  = 0
 w_k1_IBM_s064_return_constraint: 0.98198 w_k1_IBM_s06 - w_k1_IBM_s064
                                 + 0.99 b_k1_IBM_s064 - 1.01 s_k1_IBM_s064  = 0
 w_k1_IBM_s065_return_constraint: 1.13417 w_k1_IBM_s06 - w_k1_IBM_s065
                                 + 0.99 b_k1_IBM_s065 - 1.01 s_k1_IBM_s065  = 0
 w_k1_IBM_s066_return_constraint: 1.09229 w_k1_IBM_s06 - w_k1_IBM_s066
                                 + 0.99 b_k1_IBM_s066 - 1.01 s_k1_IBM_s066  = 0
 w_k1_IBM_s067_return_constraint: 1.60603 w_k1_IBM_s06 - w_k1_IBM_s067
                                 + 0.99 b_k1_IBM_s067 - 1.01 s_k1_IBM_s067  = 0
 w_k1_IBM_s070_return_constraint: 0.838061 w_k1_IBM_s07 - w_k1_IBM_s070
                                 + 0.99 b_k1_IBM_s070 - 1.01 s_k1_IBM_s070  = 0
 w_k1_IBM_s071_return_constraint: 1.38343 w_k1_IBM_s07 - w_k1_IBM_s071
                                 + 0.99 b_k1_IBM_s071 - 1.01 s_k1_IBM_s071  = 0
 w_k1_IBM_s072_return_constraint: 1.03266 w_k1_IBM_s07 - w_k1_IBM_s072
                                 + 0.99 b_k1_IBM_s072 - 1.01 s_k1_IBM_s072  = 0
 w_k1_IBM_s073_return_constraint: 1.5353 w_k1_IBM_s07 - w_k1_IBM_s073
                                 + 0.99 b_k1_IBM_s073 - 1.01 s_k1_IBM_s073  = 0
 w_k1_IBM_s074_return_constraint: 0.823323 w_k1_IBM_s07 - w_k1_IBM_s074
                                 + 0.99 b_k1_IBM_s074 - 1.01 s_k1_IBM_s074  = 0
 w_k1_IBM_s075_return_constraint: 0.979972 w_k1_IBM_s07 - w_k1_IBM_s075
                                 + 0.99 b_k1_IBM_s075 - 1.01 s_k1_IBM_s075  = 0
 w_k1_IBM_s076_return_constraint: 0.62714 w_k1_IBM_s07 - w_k1_IBM_s076
                                 + 0.99 b_k1_IBM_s076 - 1.01 s_k1_IBM_s076  = 0
 w_k1_IBM_s077_return_constraint: 1.47191 w_k1_IBM_s07 - w_k1_IBM_s077
                                 + 0.99 b_k1_IBM_s077 - 1.01 s_k1_IBM_s077  = 0
 w_k1_IBM_s100_return_constraint: 1.53716 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.53714 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.359 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.580685 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.04217 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.07594 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.922013 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.860666 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.883789 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.23022 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.971685 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.02008 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.34103 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.747507 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.4506 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.69701 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_IBM_s120_return_constraint: 1.23019 w_k1_IBM_s12 - w_k1_IBM_s120
                                 + 0.99 b_k1_IBM_s120 - 1.01 s_k1_IBM_s120  = 0
 w_k1_IBM_s121_return_constraint: 0.883866 w_k1_IBM_s12 - w_k1_IBM_s121
                                 + 0.99 b_k1_IBM_s121 - 1.01 s_k1_IBM_s121  = 0
 w_k1_IBM_s122_return_constraint: 0.726835 w_k1_IBM_s12 - w_k1_IBM_s122
                                 + 0.99 b_k1_IBM_s122 - 1.01 s_k1_IBM_s122  = 0
 w_k1_IBM_s123_return_constraint: 1.80985 w_k1_IBM_s12 - w_k1_IBM_s123
                                 + 0.99 b_k1_IBM_s123 - 1.01 s_k1_IBM_s123  = 0
 w_k1_IBM_s124_return_constraint: 0.828606 w_k1_IBM_s12 - w_k1_IBM_s124
                                 + 0.99 b_k1_IBM_s124 - 1.01 s_k1_IBM_s124  = 0
 w_k1_IBM_s125_return_constraint: 1.11464 w_k1_IBM_s12 - w_k1_IBM_s125
                                 + 0.99 b_k1_IBM_s125 - 1.01 s_k1_IBM_s125  = 0
 w_k1_IBM_s126_return_constraint: 1.09637 w_k1_IBM_s12 - w_k1_IBM_s126
                                 + 0.99 b_k1_IBM_s126 - 1.01 s_k1_IBM_s126  = 0
 w_k1_IBM_s127_return_constraint: 1.29557 w_k1_IBM_s12 - w_k1_IBM_s127
                                 + 0.99 b_k1_IBM_s127 - 1.01 s_k1_IBM_s127  = 0
 w_k1_IBM_s130_return_constraint: 0.868986 w_k1_IBM_s13 - w_k1_IBM_s130
                                 + 0.99 b_k1_IBM_s130 - 1.01 s_k1_IBM_s130  = 0
 w_k1_IBM_s131_return_constraint: 1.63557 w_k1_IBM_s13 - w_k1_IBM_s131
                                 + 0.99 b_k1_IBM_s131 - 1.01 s_k1_IBM_s131  = 0
 w_k1_IBM_s132_return_constraint: 0.808235 w_k1_IBM_s13 - w_k1_IBM_s132
                                 + 0.99 b_k1_IBM_s132 - 1.01 s_k1_IBM_s132  = 0
 w_k1_IBM_s133_return_constraint: 0.740511 w_k1_IBM_s13 - w_k1_IBM_s133
                                 + 0.99 b_k1_IBM_s133 - 1.01 s_k1_IBM_s133  = 0
 w_k1_IBM_s134_return_constraint: 1.27071 w_k1_IBM_s13 - w_k1_IBM_s134
                                 + 0.99 b_k1_IBM_s134 - 1.01 s_k1_IBM_s134  = 0
 w_k1_IBM_s135_return_constraint: 1.31817 w_k1_IBM_s13 - w_k1_IBM_s135
                                 + 0.99 b_k1_IBM_s135 - 1.01 s_k1_IBM_s135  = 0
 w_k1_IBM_s136_return_constraint: 1.02354 w_k1_IBM_s13 - w_k1_IBM_s136
                                 + 0.99 b_k1_IBM_s136 - 1.01 s_k1_IBM_s136  = 0
 w_k1_IBM_s137_return_constraint: 0.685168 w_k1_IBM_s13 - w_k1_IBM_s137
                                 + 0.99 b_k1_IBM_s137 - 1.01 s_k1_IBM_s137  = 0
 w_k1_IBM_s140_return_constraint: 1.44966 w_k1_IBM_s14 - w_k1_IBM_s140
                                 + 0.99 b_k1_IBM_s140 - 1.01 s_k1_IBM_s140  = 0
 w_k1_IBM_s141_return_constraint: 0.808562 w_k1_IBM_s14 - w_k1_IBM_s141
                                 + 0.99 b_k1_IBM_s141 - 1.01 s_k1_IBM_s141  = 0
 w_k1_IBM_s142_return_constraint: 1.01681 w_k1_IBM_s14 - w_k1_IBM_s142
                                 + 0.99 b_k1_IBM_s142 - 1.01 s_k1_IBM_s142  = 0
 w_k1_IBM_s143_return_constraint: 1.09669 w_k1_IBM_s14 - w_k1_IBM_s143
                                 + 0.99 b_k1_IBM_s143 - 1.01 s_k1_IBM_s143  = 0
 w_k1_IBM_s144_return_constraint: 1.19596 w_k1_IBM_s14 - w_k1_IBM_s144
                                 + 0.99 b_k1_IBM_s144 - 1.01 s_k1_IBM_s144  = 0
 w_k1_IBM_s145_return_constraint: 1.83957 w_k1_IBM_s14 - w_k1_IBM_s145
                                 + 0.99 b_k1_IBM_s145 - 1.01 s_k1_IBM_s145  = 0
 w_k1_IBM_s146_return_constraint: 0.714011 w_k1_IBM_s14 - w_k1_IBM_s146
                                 + 0.99 b_k1_IBM_s146 - 1.01 s_k1_IBM_s146  = 0
 w_k1_IBM_s147_return_constraint: 0.935445 w_k1_IBM_s14 - w_k1_IBM_s147
                                 + 0.99 b_k1_IBM_s147 - 1.01 s_k1_IBM_s147  = 0
 w_k1_IBM_s150_return_constraint: 1.56801 w_k1_IBM_s15 - w_k1_IBM_s150
                                 + 0.99 b_k1_IBM_s150 - 1.01 s_k1_IBM_s150  = 0
 w_k1_IBM_s151_return_constraint: 1.21955 w_k1_IBM_s15 - w_k1_IBM_s151
                                 + 0.99 b_k1_IBM_s151 - 1.01 s_k1_IBM_s151  = 0
 w_k1_IBM_s152_return_constraint: 0.728904 w_k1_IBM_s15 - w_k1_IBM_s152
                                 + 0.99 b_k1_IBM_s152 - 1.01 s_k1_IBM_s152  = 0
 w_k1_IBM_s153_return_constraint: 0.927988 w_k1_IBM_s15 - w_k1_IBM_s153
                                 + 0.99 b_k1_IBM_s153 - 1.01 s_k1_IBM_s153  = 0
 w_k1_IBM_s154_return_constraint: 1.03124 w_k1_IBM_s15 - w_k1_IBM_s154
                                 + 0.99 b_k1_IBM_s154 - 1.01 s_k1_IBM_s154  = 0
 w_k1_IBM_s155_return_constraint: 1.03966 w_k1_IBM_s15 - w_k1_IBM_s155
                                 + 0.99 b_k1_IBM_s155 - 1.01 s_k1_IBM_s155  = 0
 w_k1_IBM_s156_return_constraint: 0.70331 w_k1_IBM_s15 - w_k1_IBM_s156
                                 + 0.99 b_k1_IBM_s156 - 1.01 s_k1_IBM_s156  = 0
 w_k1_IBM_s157_return_constraint: 0.807356 w_k1_IBM_s15 - w_k1_IBM_s157
                                 + 0.99 b_k1_IBM_s157 - 1.01 s_k1_IBM_s157  = 0
 w_k1_IBM_s160_return_constraint: 1.0039 w_k1_IBM_s16 - w_k1_IBM_s160
                                 + 0.99 b_k1_IBM_s160 - 1.01 s_k1_IBM_s160  = 0
 w_k1_IBM_s161_return_constraint: 1.39948 w_k1_IBM_s16 - w_k1_IBM_s161
                                 + 0.99 b_k1_IBM_s161 - 1.01 s_k1_IBM_s161  = 0
 w_k1_IBM_s162_return_constraint: 1.08554 w_k1_IBM_s16 - w_k1_IBM_s162
                                 + 0.99 b_k1_IBM_s162 - 1.01 s_k1_IBM_s162  = 0
 w_k1_IBM_s163_return_constraint: 1.48487 w_k1_IBM_s16 - w_k1_IBM_s163
                                 + 0.99 b_k1_IBM_s163 - 1.01 s_k1_IBM_s163  = 0
 w_k1_IBM_s164_return_constraint: 1.18212 w_k1_IBM_s16 - w_k1_IBM_s164
                                 + 0.99 b_k1_IBM_s164 - 1.01 s_k1_IBM_s164  = 0
 w_k1_IBM_s165_return_constraint: 0.852431 w_k1_IBM_s16 - w_k1_IBM_s165
                                 + 0.99 b_k1_IBM_s165 - 1.01 s_k1_IBM_s165  = 0
 w_k1_IBM_s166_return_constraint: 0.877173 w_k1_IBM_s16 - w_k1_IBM_s166
                                 + 0.99 b_k1_IBM_s166 - 1.01 s_k1_IBM_s166  = 0
 w_k1_IBM_s167_return_constraint: 0.68593 w_k1_IBM_s16 - w_k1_IBM_s167
                                 + 0.99 b_k1_IBM_s167 - 1.01 s_k1_IBM_s167  = 0
 w_k1_IBM_s170_return_constraint: 0.79098 w_k1_IBM_s17 - w_k1_IBM_s170
                                 + 0.99 b_k1_IBM_s170 - 1.01 s_k1_IBM_s170  = 0
 w_k1_IBM_s171_return_constraint: 1.03339 w_k1_IBM_s17 - w_k1_IBM_s171
                                 + 0.99 b_k1_IBM_s171 - 1.01 s_k1_IBM_s171  = 0
 w_k1_IBM_s172_return_constraint: 1.10862 w_k1_IBM_s17 - w_k1_IBM_s172
                                 + 0.99 b_k1_IBM_s172 - 1.01 s_k1_IBM_s172  = 0
 w_k1_IBM_s173_return_constraint: 1.07253 w_k1_IBM_s17 - w_k1_IBM_s173
                                 + 0.99 b_k1_IBM_s173 - 1.01 s_k1_IBM_s173  = 0
 w_k1_IBM_s174_return_constraint: 1.10661 w_k1_IBM_s17 - w_k1_IBM_s174
                                 + 0.99 b_k1_IBM_s174 - 1.01 s_k1_IBM_s174  = 0
 w_k1_IBM_s175_return_constraint: 1.50229 w_k1_IBM_s17 - w_k1_IBM_s175
                                 + 0.99 b_k1_IBM_s175 - 1.01 s_k1_IBM_s175  = 0
 w_k1_IBM_s176_return_constraint: 0.849544 w_k1_IBM_s17 - w_k1_IBM_s176
                                 + 0.99 b_k1_IBM_s176 - 1.01 s_k1_IBM_s176  = 0
 w_k1_IBM_s177_return_constraint: 0.743971 w_k1_IBM_s17 - w_k1_IBM_s177
                                 + 0.99 b_k1_IBM_s177 - 1.01 s_k1_IBM_s177  = 0
 w_k1_INTC_s0_return_constraint: 1.17305 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.958618 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.90624 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.14672 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s02_return_constraint: 0.86528 w_k1_INTC_s0 - w_k1_INTC_s02
                                 + 0.99 b_k1_INTC_s02 - 1.01 s_k1_INTC_s02  = 0
 w_k1_INTC_s03_return_constraint: 0.966547 w_k1_INTC_s0 - w_k1_INTC_s03
                                 + 0.99 b_k1_INTC_s03 - 1.01 s_k1_INTC_s03  = 0
 w_k1_INTC_s04_return_constraint: 0.756353 w_k1_INTC_s0 - w_k1_INTC_s04
                                 + 0.99 b_k1_INTC_s04 - 1.01 s_k1_INTC_s04  = 0
 w_k1_INTC_s05_return_constraint: 1.03307 w_k1_INTC_s0 - w_k1_INTC_s05
                                 + 0.99 b_k1_INTC_s05 - 1.01 s_k1_INTC_s05  = 0
 w_k1_INTC_s06_return_constraint: 0.847574 w_k1_INTC_s0 - w_k1_INTC_s06
                                 + 0.99 b_k1_INTC_s06 - 1.01 s_k1_INTC_s06  = 0
 w_k1_INTC_s07_return_constraint: 1.08044 w_k1_INTC_s0 - w_k1_INTC_s07
                                 + 0.99 b_k1_INTC_s07 - 1.01 s_k1_INTC_s07  = 0
 w_k1_INTC_s10_return_constraint: 0.821483 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.00429 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s12_return_constraint: 1.37741 w_k1_INTC_s1 - w_k1_INTC_s12
                                 + 0.99 b_k1_INTC_s12 - 1.01 s_k1_INTC_s12  = 0
 w_k1_INTC_s13_return_constraint: 1.03495 w_k1_INTC_s1 - w_k1_INTC_s13
                                 + 0.99 b_k1_INTC_s13 - 1.01 s_k1_INTC_s13  = 0
 w_k1_INTC_s14_return_constraint: 0.957557 w_k1_INTC_s1 - w_k1_INTC_s14
                                 + 0.99 b_k1_INTC_s14 - 1.01 s_k1_INTC_s14  = 0
 w_k1_INTC_s15_return_constraint: 0.995207 w_k1_INTC_s1 - w_k1_INTC_s15
                                 + 0.99 b_k1_INTC_s15 - 1.01 s_k1_INTC_s15  = 0
 w_k1_INTC_s16_return_constraint: 0.834349 w_k1_INTC_s1 - w_k1_INTC_s16
                                 + 0.99 b_k1_INTC_s16 - 1.01 s_k1_INTC_s16  = 0
 w_k1_INTC_s17_return_constraint: 0.909531 w_k1_INTC_s1 - w_k1_INTC_s17
                                 + 0.99 b_k1_INTC_s17 - 1.01 s_k1_INTC_s17  = 0
 w_k1_INTC_s000_return_constraint: 1.10282 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.09243 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.940447 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.17118 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.917592 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.956598 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.09265 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.742718 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.806309 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.983813 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.23093 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.991222 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.763303 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.15431 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.03476 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.18467 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s020_return_constraint: 0.945335 w_k1_INTC_s02 - w_k1_INTC_s020
                                 + 0.99 b_k1_INTC_s020 - 1.01 s_k1_INTC_s020
                                  = 0
 w_k1_INTC_s021_return_constraint: 1.36574 w_k1_INTC_s02 - w_k1_INTC_s021
                                 + 0.99 b_k1_INTC_s021 - 1.01 s_k1_INTC_s021
                                  = 0
 w_k1_INTC_s022_return_constraint: 0.898096 w_k1_INTC_s02 - w_k1_INTC_s022
                                 + 0.99 b_k1_INTC_s022 - 1.01 s_k1_INTC_s022
                                  = 0
 w_k1_INTC_s023_return_constraint: 0.958011 w_k1_INTC_s02 - w_k1_INTC_s023
                                 + 0.99 b_k1_INTC_s023 - 1.01 s_k1_INTC_s023
                                  = 0
 w_k1_INTC_s024_return_constraint: 1.08936 w_k1_INTC_s02 - w_k1_INTC_s024
                                 + 0.99 b_k1_INTC_s024 - 1.01 s_k1_INTC_s024
                                  = 0
 w_k1_INTC_s025_return_constraint: 0.974054 w_k1_INTC_s02 - w_k1_INTC_s025
                                 + 0.99 b_k1_INTC_s025 - 1.01 s_k1_INTC_s025
                                  = 0
 w_k1_INTC_s026_return_constraint: 0.791881 w_k1_INTC_s02 - w_k1_INTC_s026
                                 + 0.99 b_k1_INTC_s026 - 1.01 s_k1_INTC_s026
                                  = 0
 w_k1_INTC_s027_return_constraint: 1.09929 w_k1_INTC_s02 - w_k1_INTC_s027
                                 + 0.99 b_k1_INTC_s027 - 1.01 s_k1_INTC_s027
                                  = 0
 w_k1_INTC_s030_return_constraint: 1.02352 w_k1_INTC_s03 - w_k1_INTC_s030
                                 + 0.99 b_k1_INTC_s030 - 1.01 s_k1_INTC_s030
                                  = 0
 w_k1_INTC_s031_return_constraint: 1.35235 w_k1_INTC_s03 - w_k1_INTC_s031
                                 + 0.99 b_k1_INTC_s031 - 1.01 s_k1_INTC_s031
                                  = 0
 w_k1_INTC_s032_return_constraint: 0.851794 w_k1_INTC_s03 - w_k1_INTC_s032
                                 + 0.99 b_k1_INTC_s032 - 1.01 s_k1_INTC_s032
                                  = 0
 w_k1_INTC_s033_return_constraint: 1.07285 w_k1_INTC_s03 - w_k1_INTC_s033
                                 + 0.99 b_k1_INTC_s033 - 1.01 s_k1_INTC_s033
                                  = 0
 w_k1_INTC_s034_return_constraint: 1.11179 w_k1_INTC_s03 - w_k1_INTC_s034
                                 + 0.99 b_k1_INTC_s034 - 1.01 s_k1_INTC_s034
                                  = 0
 w_k1_INTC_s035_return_constraint: 1.00879 w_k1_INTC_s03 - w_k1_INTC_s035
                                 + 0.99 b_k1_INTC_s035 - 1.01 s_k1_INTC_s035
                                  = 0
 w_k1_INTC_s036_return_constraint: 1.05645 w_k1_INTC_s03 - w_k1_INTC_s036
                                 + 0.99 b_k1_INTC_s036 - 1.01 s_k1_INTC_s036
                                  = 0
 w_k1_INTC_s037_return_constraint: 0.90111 w_k1_INTC_s03 - w_k1_INTC_s037
                                 + 0.99 b_k1_INTC_s037 - 1.01 s_k1_INTC_s037
                                  = 0
 w_k1_INTC_s040_return_constraint: 1.21002 w_k1_INTC_s04 - w_k1_INTC_s040
                                 + 0.99 b_k1_INTC_s040 - 1.01 s_k1_INTC_s040
                                  = 0
 w_k1_INTC_s041_return_constraint: 0.995116 w_k1_INTC_s04 - w_k1_INTC_s041
                                 + 0.99 b_k1_INTC_s041 - 1.01 s_k1_INTC_s041
                                  = 0
 w_k1_INTC_s042_return_constraint: 1.13454 w_k1_INTC_s04 - w_k1_INTC_s042
                                 + 0.99 b_k1_INTC_s042 - 1.01 s_k1_INTC_s042
                                  = 0
 w_k1_INTC_s043_return_constraint: 0.99509 w_k1_INTC_s04 - w_k1_INTC_s043
                                 + 0.99 b_k1_INTC_s043 - 1.01 s_k1_INTC_s043
                                  = 0
 w_k1_INTC_s044_return_constraint: 1.28934 w_k1_INTC_s04 - w_k1_INTC_s044
                                 + 0.99 b_k1_INTC_s044 - 1.01 s_k1_INTC_s044
                                  = 0
 w_k1_INTC_s045_return_constraint: 0.956691 w_k1_INTC_s04 - w_k1_INTC_s045
                                 + 0.99 b_k1_INTC_s045 - 1.01 s_k1_INTC_s045
                                  = 0
 w_k1_INTC_s046_return_constraint: 0.738584 w_k1_INTC_s04 - w_k1_INTC_s046
                                 + 0.99 b_k1_INTC_s046 - 1.01 s_k1_INTC_s046
                                  = 0
 w_k1_INTC_s047_return_constraint: 0.829017 w_k1_INTC_s04 - w_k1_INTC_s047
                                 + 0.99 b_k1_INTC_s047 - 1.01 s_k1_INTC_s047
                                  = 0
 w_k1_INTC_s050_return_constraint: 0.68439 w_k1_INTC_s05 - w_k1_INTC_s050
                                 + 0.99 b_k1_INTC_s050 - 1.01 s_k1_INTC_s050
                                  = 0
 w_k1_INTC_s051_return_constraint: 0.943873 w_k1_INTC_s05 - w_k1_INTC_s051
                                 + 0.99 b_k1_INTC_s051 - 1.01 s_k1_INTC_s051
                                  = 0
 w_k1_INTC_s052_return_constraint: 0.92935 w_k1_INTC_s05 - w_k1_INTC_s052
                                 + 0.99 b_k1_INTC_s052 - 1.01 s_k1_INTC_s052
                                  = 0
 w_k1_INTC_s053_return_constraint: 1.00999 w_k1_INTC_s05 - w_k1_INTC_s053
                                 + 0.99 b_k1_INTC_s053 - 1.01 s_k1_INTC_s053
                                  = 0
 w_k1_INTC_s054_return_constraint: 1.16034 w_k1_INTC_s05 - w_k1_INTC_s054
                                 + 0.99 b_k1_INTC_s054 - 1.01 s_k1_INTC_s054
                                  = 0
 w_k1_INTC_s055_return_constraint: 0.771563 w_k1_INTC_s05 - w_k1_INTC_s055
                                 + 0.99 b_k1_INTC_s055 - 1.01 s_k1_INTC_s055
                                  = 0
 w_k1_INTC_s056_return_constraint: 1.106 w_k1_INTC_s05 - w_k1_INTC_s056
                                 + 0.99 b_k1_INTC_s056 - 1.01 s_k1_INTC_s056
                                  = 0
 w_k1_INTC_s057_return_constraint: 0.886481 w_k1_INTC_s05 - w_k1_INTC_s057
                                 + 0.99 b_k1_INTC_s057 - 1.01 s_k1_INTC_s057
                                  = 0
 w_k1_INTC_s060_return_constraint: 1.09856 w_k1_INTC_s06 - w_k1_INTC_s060
                                 + 0.99 b_k1_INTC_s060 - 1.01 s_k1_INTC_s060
                                  = 0
 w_k1_INTC_s061_return_constraint: 1.14582 w_k1_INTC_s06 - w_k1_INTC_s061
                                 + 0.99 b_k1_INTC_s061 - 1.01 s_k1_INTC_s061
                                  = 0
 w_k1_INTC_s062_return_constraint: 0.974389 w_k1_INTC_s06 - w_k1_INTC_s062
                                 + 0.99 b_k1_INTC_s062 - 1.01 s_k1_INTC_s062
                                  = 0
 w_k1_INTC_s063_return_constraint: 0.87005 w_k1_INTC_s06 - w_k1_INTC_s063
                                 + 0.99 b_k1_INTC_s063 - 1.01 s_k1_INTC_s063
                                  = 0
 w_k1_INTC_s064_return_constraint: 1.07411 w_k1_INTC_s06 - w_k1_INTC_s064
                                 + 0.99 b_k1_INTC_s064 - 1.01 s_k1_INTC_s064
                                  = 0
 w_k1_INTC_s065_return_constraint: 0.843155 w_k1_INTC_s06 - w_k1_INTC_s065
                                 + 0.99 b_k1_INTC_s065 - 1.01 s_k1_INTC_s065
                                  = 0
 w_k1_INTC_s066_return_constraint: 1.14747 w_k1_INTC_s06 - w_k1_INTC_s066
                                 + 0.99 b_k1_INTC_s066 - 1.01 s_k1_INTC_s066
                                  = 0
 w_k1_INTC_s067_return_constraint: 1.21588 w_k1_INTC_s06 - w_k1_INTC_s067
                                 + 0.99 b_k1_INTC_s067 - 1.01 s_k1_INTC_s067
                                  = 0
 w_k1_INTC_s070_return_constraint: 1.22064 w_k1_INTC_s07 - w_k1_INTC_s070
                                 + 0.99 b_k1_INTC_s070 - 1.01 s_k1_INTC_s070
                                  = 0
 w_k1_INTC_s071_return_constraint: 1.32395 w_k1_INTC_s07 - w_k1_INTC_s071
                                 + 0.99 b_k1_INTC_s071 - 1.01 s_k1_INTC_s071
                                  = 0
 w_k1_INTC_s072_return_constraint: 0.886466 w_k1_INTC_s07 - w_k1_INTC_s072
                                 + 0.99 b_k1_INTC_s072 - 1.01 s_k1_INTC_s072
                                  = 0
 w_k1_INTC_s073_return_constraint: 0.999473 w_k1_INTC_s07 - w_k1_INTC_s073
                                 + 0.99 b_k1_INTC_s073 - 1.01 s_k1_INTC_s073
                                  = 0
 w_k1_INTC_s074_return_constraint: 1.21694 w_k1_INTC_s07 - w_k1_INTC_s074
                                 + 0.99 b_k1_INTC_s074 - 1.01 s_k1_INTC_s074
                                  = 0
 w_k1_INTC_s075_return_constraint: 0.736991 w_k1_INTC_s07 - w_k1_INTC_s075
                                 + 0.99 b_k1_INTC_s075 - 1.01 s_k1_INTC_s075
                                  = 0
 w_k1_INTC_s076_return_constraint: 0.933995 w_k1_INTC_s07 - w_k1_INTC_s076
                                 + 0.99 b_k1_INTC_s076 - 1.01 s_k1_INTC_s076
                                  = 0
 w_k1_INTC_s077_return_constraint: 1.09352 w_k1_INTC_s07 - w_k1_INTC_s077
                                 + 0.99 b_k1_INTC_s077 - 1.01 s_k1_INTC_s077
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.27757 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.06416 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.74636 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.878101 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.07494 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.805343 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.20533 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.949779 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.2956 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.39854 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.08982 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.12214 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.0609 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.11263 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.82283 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.948315 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_INTC_s120_return_constraint: 1.03934 w_k1_INTC_s12 - w_k1_INTC_s120
                                 + 0.99 b_k1_INTC_s120 - 1.01 s_k1_INTC_s120
                                  = 0
 w_k1_INTC_s121_return_constraint: 0.792655 w_k1_INTC_s12 - w_k1_INTC_s121
                                 + 0.99 b_k1_INTC_s121 - 1.01 s_k1_INTC_s121
                                  = 0
 w_k1_INTC_s122_return_constraint: 0.836524 w_k1_INTC_s12 - w_k1_INTC_s122
                                 + 0.99 b_k1_INTC_s122 - 1.01 s_k1_INTC_s122
                                  = 0
 w_k1_INTC_s123_return_constraint: 0.980067 w_k1_INTC_s12 - w_k1_INTC_s123
                                 + 0.99 b_k1_INTC_s123 - 1.01 s_k1_INTC_s123
                                  = 0
 w_k1_INTC_s124_return_constraint: 0.90427 w_k1_INTC_s12 - w_k1_INTC_s124
                                 + 0.99 b_k1_INTC_s124 - 1.01 s_k1_INTC_s124
                                  = 0
 w_k1_INTC_s125_return_constraint: 1.01811 w_k1_INTC_s12 - w_k1_INTC_s125
                                 + 0.99 b_k1_INTC_s125 - 1.01 s_k1_INTC_s125
                                  = 0
 w_k1_INTC_s126_return_constraint: 0.919129 w_k1_INTC_s12 - w_k1_INTC_s126
                                 + 0.99 b_k1_INTC_s126 - 1.01 s_k1_INTC_s126
                                  = 0
 w_k1_INTC_s127_return_constraint: 0.946122 w_k1_INTC_s12 - w_k1_INTC_s127
                                 + 0.99 b_k1_INTC_s127 - 1.01 s_k1_INTC_s127
                                  = 0
 w_k1_INTC_s130_return_constraint: 1.10765 w_k1_INTC_s13 - w_k1_INTC_s130
                                 + 0.99 b_k1_INTC_s130 - 1.01 s_k1_INTC_s130
                                  = 0
 w_k1_INTC_s131_return_constraint: 1.33863 w_k1_INTC_s13 - w_k1_INTC_s131
                                 + 0.99 b_k1_INTC_s131 - 1.01 s_k1_INTC_s131
                                  = 0
 w_k1_INTC_s132_return_constraint: 1.12655 w_k1_INTC_s13 - w_k1_INTC_s132
                                 + 0.99 b_k1_INTC_s132 - 1.01 s_k1_INTC_s132
                                  = 0
 w_k1_INTC_s133_return_constraint: 0.885291 w_k1_INTC_s13 - w_k1_INTC_s133
                                 + 0.99 b_k1_INTC_s133 - 1.01 s_k1_INTC_s133
                                  = 0
 w_k1_INTC_s134_return_constraint: 1.06588 w_k1_INTC_s13 - w_k1_INTC_s134
                                 + 0.99 b_k1_INTC_s134 - 1.01 s_k1_INTC_s134
                                  = 0
 w_k1_INTC_s135_return_constraint: 1.23463 w_k1_INTC_s13 - w_k1_INTC_s135
                                 + 0.99 b_k1_INTC_s135 - 1.01 s_k1_INTC_s135
                                  = 0
 w_k1_INTC_s136_return_constraint: 0.97252 w_k1_INTC_s13 - w_k1_INTC_s136
                                 + 0.99 b_k1_INTC_s136 - 1.01 s_k1_INTC_s136
                                  = 0
 w_k1_INTC_s137_return_constraint: 0.841325 w_k1_INTC_s13 - w_k1_INTC_s137
                                 + 0.99 b_k1_INTC_s137 - 1.01 s_k1_INTC_s137
                                  = 0
 w_k1_INTC_s140_return_constraint: 0.904521 w_k1_INTC_s14 - w_k1_INTC_s140
                                 + 0.99 b_k1_INTC_s140 - 1.01 s_k1_INTC_s140
                                  = 0
 w_k1_INTC_s141_return_constraint: 1.15096 w_k1_INTC_s14 - w_k1_INTC_s141
                                 + 0.99 b_k1_INTC_s141 - 1.01 s_k1_INTC_s141
                                  = 0
 w_k1_INTC_s142_return_constraint: 0.941213 w_k1_INTC_s14 - w_k1_INTC_s142
                                 + 0.99 b_k1_INTC_s142 - 1.01 s_k1_INTC_s142
                                  = 0
 w_k1_INTC_s143_return_constraint: 0.997962 w_k1_INTC_s14 - w_k1_INTC_s143
                                 + 0.99 b_k1_INTC_s143 - 1.01 s_k1_INTC_s143
                                  = 0
 w_k1_INTC_s144_return_constraint: 1.39331 w_k1_INTC_s14 - w_k1_INTC_s144
                                 + 0.99 b_k1_INTC_s144 - 1.01 s_k1_INTC_s144
                                  = 0
 w_k1_INTC_s145_return_constraint: 0.919946 w_k1_INTC_s14 - w_k1_INTC_s145
                                 + 0.99 b_k1_INTC_s145 - 1.01 s_k1_INTC_s145
                                  = 0
 w_k1_INTC_s146_return_constraint: 0.752872 w_k1_INTC_s14 - w_k1_INTC_s146
                                 + 0.99 b_k1_INTC_s146 - 1.01 s_k1_INTC_s146
                                  = 0
 w_k1_INTC_s147_return_constraint: 1.29709 w_k1_INTC_s14 - w_k1_INTC_s147
                                 + 0.99 b_k1_INTC_s147 - 1.01 s_k1_INTC_s147
                                  = 0
 w_k1_INTC_s150_return_constraint: 1.2403 w_k1_INTC_s15 - w_k1_INTC_s150
                                 + 0.99 b_k1_INTC_s150 - 1.01 s_k1_INTC_s150
                                  = 0
 w_k1_INTC_s151_return_constraint: 0.904364 w_k1_INTC_s15 - w_k1_INTC_s151
                                 + 0.99 b_k1_INTC_s151 - 1.01 s_k1_INTC_s151
                                  = 0
 w_k1_INTC_s152_return_constraint: 1.07475 w_k1_INTC_s15 - w_k1_INTC_s152
                                 + 0.99 b_k1_INTC_s152 - 1.01 s_k1_INTC_s152
                                  = 0
 w_k1_INTC_s153_return_constraint: 1.21329 w_k1_INTC_s15 - w_k1_INTC_s153
                                 + 0.99 b_k1_INTC_s153 - 1.01 s_k1_INTC_s153
                                  = 0
 w_k1_INTC_s154_return_constraint: 1.16549 w_k1_INTC_s15 - w_k1_INTC_s154
                                 + 0.99 b_k1_INTC_s154 - 1.01 s_k1_INTC_s154
                                  = 0
 w_k1_INTC_s155_return_constraint: 0.843778 w_k1_INTC_s15 - w_k1_INTC_s155
                                 + 0.99 b_k1_INTC_s155 - 1.01 s_k1_INTC_s155
                                  = 0
 w_k1_INTC_s156_return_constraint: 0.868057 w_k1_INTC_s15 - w_k1_INTC_s156
                                 + 0.99 b_k1_INTC_s156 - 1.01 s_k1_INTC_s156
                                  = 0
 w_k1_INTC_s157_return_constraint: 0.778024 w_k1_INTC_s15 - w_k1_INTC_s157
                                 + 0.99 b_k1_INTC_s157 - 1.01 s_k1_INTC_s157
                                  = 0
 w_k1_INTC_s160_return_constraint: 0.914969 w_k1_INTC_s16 - w_k1_INTC_s160
                                 + 0.99 b_k1_INTC_s160 - 1.01 s_k1_INTC_s160
                                  = 0
 w_k1_INTC_s161_return_constraint: 0.998339 w_k1_INTC_s16 - w_k1_INTC_s161
                                 + 0.99 b_k1_INTC_s161 - 1.01 s_k1_INTC_s161
                                  = 0
 w_k1_INTC_s162_return_constraint: 1.30437 w_k1_INTC_s16 - w_k1_INTC_s162
                                 + 0.99 b_k1_INTC_s162 - 1.01 s_k1_INTC_s162
                                  = 0
 w_k1_INTC_s163_return_constraint: 0.92831 w_k1_INTC_s16 - w_k1_INTC_s163
                                 + 0.99 b_k1_INTC_s163 - 1.01 s_k1_INTC_s163
                                  = 0
 w_k1_INTC_s164_return_constraint: 1.47724 w_k1_INTC_s16 - w_k1_INTC_s164
                                 + 0.99 b_k1_INTC_s164 - 1.01 s_k1_INTC_s164
                                  = 0
 w_k1_INTC_s165_return_constraint: 1.06327 w_k1_INTC_s16 - w_k1_INTC_s165
                                 + 0.99 b_k1_INTC_s165 - 1.01 s_k1_INTC_s165
                                  = 0
 w_k1_INTC_s166_return_constraint: 1.04245 w_k1_INTC_s16 - w_k1_INTC_s166
                                 + 0.99 b_k1_INTC_s166 - 1.01 s_k1_INTC_s166
                                  = 0
 w_k1_INTC_s167_return_constraint: 1.18946 w_k1_INTC_s16 - w_k1_INTC_s167
                                 + 0.99 b_k1_INTC_s167 - 1.01 s_k1_INTC_s167
                                  = 0
 w_k1_INTC_s170_return_constraint: 0.891128 w_k1_INTC_s17 - w_k1_INTC_s170
                                 + 0.99 b_k1_INTC_s170 - 1.01 s_k1_INTC_s170
                                  = 0
 w_k1_INTC_s171_return_constraint: 0.931228 w_k1_INTC_s17 - w_k1_INTC_s171
                                 + 0.99 b_k1_INTC_s171 - 1.01 s_k1_INTC_s171
                                  = 0
 w_k1_INTC_s172_return_constraint: 1.26709 w_k1_INTC_s17 - w_k1_INTC_s172
                                 + 0.99 b_k1_INTC_s172 - 1.01 s_k1_INTC_s172
                                  = 0
 w_k1_INTC_s173_return_constraint: 1.11003 w_k1_INTC_s17 - w_k1_INTC_s173
                                 + 0.99 b_k1_INTC_s173 - 1.01 s_k1_INTC_s173
                                  = 0
 w_k1_INTC_s174_return_constraint: 0.829463 w_k1_INTC_s17 - w_k1_INTC_s174
                                 + 0.99 b_k1_INTC_s174 - 1.01 s_k1_INTC_s174
                                  = 0
 w_k1_INTC_s175_return_constraint: 1.06685 w_k1_INTC_s17 - w_k1_INTC_s175
                                 + 0.99 b_k1_INTC_s175 - 1.01 s_k1_INTC_s175
                                  = 0
 w_k1_INTC_s176_return_constraint: 1.13296 w_k1_INTC_s17 - w_k1_INTC_s176
                                 + 0.99 b_k1_INTC_s176 - 1.01 s_k1_INTC_s176
                                  = 0
 w_k1_INTC_s177_return_constraint: 0.976982 w_k1_INTC_s17 - w_k1_INTC_s177
                                 + 0.99 b_k1_INTC_s177 - 1.01 s_k1_INTC_s177
                                  = 0
 w_k1_KO_s0_return_constraint:   1.22732 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.932518 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.904583 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.04948 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s02_return_constraint:  1.03811 w_k1_KO_s0 - w_k1_KO_s02
                                 + 0.99 b_k1_KO_s02 - 1.01 s_k1_KO_s02  = 0
 w_k1_KO_s03_return_constraint:  0.817274 w_k1_KO_s0 - w_k1_KO_s03
                                 + 0.99 b_k1_KO_s03 - 1.01 s_k1_KO_s03  = 0
 w_k1_KO_s04_return_constraint:  0.651559 w_k1_KO_s0 - w_k1_KO_s04
                                 + 0.99 b_k1_KO_s04 - 1.01 s_k1_KO_s04  = 0
 w_k1_KO_s05_return_constraint:  1.05784 w_k1_KO_s0 - w_k1_KO_s05
                                 + 0.99 b_k1_KO_s05 - 1.01 s_k1_KO_s05  = 0
 w_k1_KO_s06_return_constraint:  0.85525 w_k1_KO_s0 - w_k1_KO_s06
                                 + 0.99 b_k1_KO_s06 - 1.01 s_k1_KO_s06  = 0
 w_k1_KO_s07_return_constraint:  1.34287 w_k1_KO_s0 - w_k1_KO_s07
                                 + 0.99 b_k1_KO_s07 - 1.01 s_k1_KO_s07  = 0
 w_k1_KO_s10_return_constraint:  0.7112 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.37209 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s12_return_constraint:  1.37848 w_k1_KO_s1 - w_k1_KO_s12
                                 + 0.99 b_k1_KO_s12 - 1.01 s_k1_KO_s12  = 0
 w_k1_KO_s13_return_constraint:  0.99793 w_k1_KO_s1 - w_k1_KO_s13
                                 + 0.99 b_k1_KO_s13 - 1.01 s_k1_KO_s13  = 0
 w_k1_KO_s14_return_constraint:  1.18927 w_k1_KO_s1 - w_k1_KO_s14
                                 + 0.99 b_k1_KO_s14 - 1.01 s_k1_KO_s14  = 0
 w_k1_KO_s15_return_constraint:  1.04777 w_k1_KO_s1 - w_k1_KO_s15
                                 + 0.99 b_k1_KO_s15 - 1.01 s_k1_KO_s15  = 0
 w_k1_KO_s16_return_constraint:  0.801646 w_k1_KO_s1 - w_k1_KO_s16
                                 + 0.99 b_k1_KO_s16 - 1.01 s_k1_KO_s16  = 0
 w_k1_KO_s17_return_constraint:  0.875095 w_k1_KO_s1 - w_k1_KO_s17
                                 + 0.99 b_k1_KO_s17 - 1.01 s_k1_KO_s17  = 0
 w_k1_KO_s000_return_constraint: 1.07707 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.32038 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.83136 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.06736 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.992631 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.881821 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.02315 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.741109 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.768024 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.731513 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.22567 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.841206 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.791749 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.11658 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.01518 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.36794 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s020_return_constraint: 0.887454 w_k1_KO_s02 - w_k1_KO_s020
                                 + 0.99 b_k1_KO_s020 - 1.01 s_k1_KO_s020  = 0
 w_k1_KO_s021_return_constraint: 1.22682 w_k1_KO_s02 - w_k1_KO_s021
                                 + 0.99 b_k1_KO_s021 - 1.01 s_k1_KO_s021  = 0
 w_k1_KO_s022_return_constraint: 1.26488 w_k1_KO_s02 - w_k1_KO_s022
                                 + 0.99 b_k1_KO_s022 - 1.01 s_k1_KO_s022  = 0
 w_k1_KO_s023_return_constraint: 1.05834 w_k1_KO_s02 - w_k1_KO_s023
                                 + 0.99 b_k1_KO_s023 - 1.01 s_k1_KO_s023  = 0
 w_k1_KO_s024_return_constraint: 1.10595 w_k1_KO_s02 - w_k1_KO_s024
                                 + 0.99 b_k1_KO_s024 - 1.01 s_k1_KO_s024  = 0
 w_k1_KO_s025_return_constraint: 0.877155 w_k1_KO_s02 - w_k1_KO_s025
                                 + 0.99 b_k1_KO_s025 - 1.01 s_k1_KO_s025  = 0
 w_k1_KO_s026_return_constraint: 0.727045 w_k1_KO_s02 - w_k1_KO_s026
                                 + 0.99 b_k1_KO_s026 - 1.01 s_k1_KO_s026  = 0
 w_k1_KO_s027_return_constraint: 0.915696 w_k1_KO_s02 - w_k1_KO_s027
                                 + 0.99 b_k1_KO_s027 - 1.01 s_k1_KO_s027  = 0
 w_k1_KO_s030_return_constraint: 1.37502 w_k1_KO_s03 - w_k1_KO_s030
                                 + 0.99 b_k1_KO_s030 - 1.01 s_k1_KO_s030  = 0
 w_k1_KO_s031_return_constraint: 1.37029 w_k1_KO_s03 - w_k1_KO_s031
                                 + 0.99 b_k1_KO_s031 - 1.01 s_k1_KO_s031  = 0
 w_k1_KO_s032_return_constraint: 0.666347 w_k1_KO_s03 - w_k1_KO_s032
                                 + 0.99 b_k1_KO_s032 - 1.01 s_k1_KO_s032  = 0
 w_k1_KO_s033_return_constraint: 0.822112 w_k1_KO_s03 - w_k1_KO_s033
                                 + 0.99 b_k1_KO_s033 - 1.01 s_k1_KO_s033  = 0
 w_k1_KO_s034_return_constraint: 0.94926 w_k1_KO_s03 - w_k1_KO_s034
                                 + 0.99 b_k1_KO_s034 - 1.01 s_k1_KO_s034  = 0
 w_k1_KO_s035_return_constraint: 1.1182 w_k1_KO_s03 - w_k1_KO_s035
                                 + 0.99 b_k1_KO_s035 - 1.01 s_k1_KO_s035  = 0
 w_k1_KO_s036_return_constraint: 1.02137 w_k1_KO_s03 - w_k1_KO_s036
                                 + 0.99 b_k1_KO_s036 - 1.01 s_k1_KO_s036  = 0
 w_k1_KO_s037_return_constraint: 1.06032 w_k1_KO_s03 - w_k1_KO_s037
                                 + 0.99 b_k1_KO_s037 - 1.01 s_k1_KO_s037  = 0
 w_k1_KO_s040_return_constraint: 1.37351 w_k1_KO_s04 - w_k1_KO_s040
                                 + 0.99 b_k1_KO_s040 - 1.01 s_k1_KO_s040  = 0
 w_k1_KO_s041_return_constraint: 1.19231 w_k1_KO_s04 - w_k1_KO_s041
                                 + 0.99 b_k1_KO_s041 - 1.01 s_k1_KO_s041  = 0
 w_k1_KO_s042_return_constraint: 0.972221 w_k1_KO_s04 - w_k1_KO_s042
                                 + 0.99 b_k1_KO_s042 - 1.01 s_k1_KO_s042  = 0
 w_k1_KO_s043_return_constraint: 0.969609 w_k1_KO_s04 - w_k1_KO_s043
                                 + 0.99 b_k1_KO_s043 - 1.01 s_k1_KO_s043  = 0
 w_k1_KO_s044_return_constraint: 1.25727 w_k1_KO_s04 - w_k1_KO_s044
                                 + 0.99 b_k1_KO_s044 - 1.01 s_k1_KO_s044  = 0
 w_k1_KO_s045_return_constraint: 0.869052 w_k1_KO_s04 - w_k1_KO_s045
                                 + 0.99 b_k1_KO_s045 - 1.01 s_k1_KO_s045  = 0
 w_k1_KO_s046_return_constraint: 0.808205 w_k1_KO_s04 - w_k1_KO_s046
                                 + 0.99 b_k1_KO_s046 - 1.01 s_k1_KO_s046  = 0
 w_k1_KO_s047_return_constraint: 0.938176 w_k1_KO_s04 - w_k1_KO_s047
                                 + 0.99 b_k1_KO_s047 - 1.01 s_k1_KO_s047  = 0
 w_k1_KO_s050_return_constraint: 0.662332 w_k1_KO_s05 - w_k1_KO_s050
                                 + 0.99 b_k1_KO_s050 - 1.01 s_k1_KO_s050  = 0
 w_k1_KO_s051_return_constraint: 1.26283 w_k1_KO_s05 - w_k1_KO_s051
                                 + 0.99 b_k1_KO_s051 - 1.01 s_k1_KO_s051  = 0
 w_k1_KO_s052_return_constraint: 0.86619 w_k1_KO_s05 - w_k1_KO_s052
                                 + 0.99 b_k1_KO_s052 - 1.01 s_k1_KO_s052  = 0
 w_k1_KO_s053_return_constraint: 1.03966 w_k1_KO_s05 - w_k1_KO_s053
                                 + 0.99 b_k1_KO_s053 - 1.01 s_k1_KO_s053  = 0
 w_k1_KO_s054_return_constraint: 0.87549 w_k1_KO_s05 - w_k1_KO_s054
                                 + 0.99 b_k1_KO_s054 - 1.01 s_k1_KO_s054  = 0
 w_k1_KO_s055_return_constraint: 0.826241 w_k1_KO_s05 - w_k1_KO_s055
                                 + 0.99 b_k1_KO_s055 - 1.01 s_k1_KO_s055  = 0
 w_k1_KO_s056_return_constraint: 1.51816 w_k1_KO_s05 - w_k1_KO_s056
                                 + 0.99 b_k1_KO_s056 - 1.01 s_k1_KO_s056  = 0
 w_k1_KO_s057_return_constraint: 0.973781 w_k1_KO_s05 - w_k1_KO_s057
                                 + 0.99 b_k1_KO_s057 - 1.01 s_k1_KO_s057  = 0
 w_k1_KO_s060_return_constraint: 0.890021 w_k1_KO_s06 - w_k1_KO_s060
                                 + 0.99 b_k1_KO_s060 - 1.01 s_k1_KO_s060  = 0
 w_k1_KO_s061_return_constraint: 0.867812 w_k1_KO_s06 - w_k1_KO_s061
                                 + 0.99 b_k1_KO_s061 - 1.01 s_k1_KO_s061  = 0
 w_k1_KO_s062_return_constraint: 0.710056 w_k1_KO_s06 - w_k1_KO_s062
                                 + 0.99 b_k1_KO_s062 - 1.01 s_k1_KO_s062  = 0
 w_k1_KO_s063_return_constraint: 0.701433 w_k1_KO_s06 - w_k1_KO_s063
                                 + 0.99 b_k1_KO_s063 - 1.01 s_k1_KO_s063  = 0
 w_k1_KO_s064_return_constraint: 0.993495 w_k1_KO_s06 - w_k1_KO_s064
                                 + 0.99 b_k1_KO_s064 - 1.01 s_k1_KO_s064  = 0
 w_k1_KO_s065_return_constraint: 0.930359 w_k1_KO_s06 - w_k1_KO_s065
                                 + 0.99 b_k1_KO_s065 - 1.01 s_k1_KO_s065  = 0
 w_k1_KO_s066_return_constraint: 1.22868 w_k1_KO_s06 - w_k1_KO_s066
                                 + 0.99 b_k1_KO_s066 - 1.01 s_k1_KO_s066  = 0
 w_k1_KO_s067_return_constraint: 1.35887 w_k1_KO_s06 - w_k1_KO_s067
                                 + 0.99 b_k1_KO_s067 - 1.01 s_k1_KO_s067  = 0
 w_k1_KO_s070_return_constraint: 0.856224 w_k1_KO_s07 - w_k1_KO_s070
                                 + 0.99 b_k1_KO_s070 - 1.01 s_k1_KO_s070  = 0
 w_k1_KO_s071_return_constraint: 1.43967 w_k1_KO_s07 - w_k1_KO_s071
                                 + 0.99 b_k1_KO_s071 - 1.01 s_k1_KO_s071  = 0
 w_k1_KO_s072_return_constraint: 0.868501 w_k1_KO_s07 - w_k1_KO_s072
                                 + 0.99 b_k1_KO_s072 - 1.01 s_k1_KO_s072  = 0
 w_k1_KO_s073_return_constraint: 1.19125 w_k1_KO_s07 - w_k1_KO_s073
                                 + 0.99 b_k1_KO_s073 - 1.01 s_k1_KO_s073  = 0
 w_k1_KO_s074_return_constraint: 0.947303 w_k1_KO_s07 - w_k1_KO_s074
                                 + 0.99 b_k1_KO_s074 - 1.01 s_k1_KO_s074  = 0
 w_k1_KO_s075_return_constraint: 0.720117 w_k1_KO_s07 - w_k1_KO_s075
                                 + 0.99 b_k1_KO_s075 - 1.01 s_k1_KO_s075  = 0
 w_k1_KO_s076_return_constraint: 0.679708 w_k1_KO_s07 - w_k1_KO_s076
                                 + 0.99 b_k1_KO_s076 - 1.01 s_k1_KO_s076  = 0
 w_k1_KO_s077_return_constraint: 1.23375 w_k1_KO_s07 - w_k1_KO_s077
                                 + 0.99 b_k1_KO_s077 - 1.01 s_k1_KO_s077  = 0
 w_k1_KO_s100_return_constraint: 1.39257 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.25624 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.832925 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.77754 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.0253 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.8872 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.15121 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.917852 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.989878 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.2398 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.01414 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.929966 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.01351 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.830034 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.45634 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.17533 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_KO_s120_return_constraint: 1.30271 w_k1_KO_s12 - w_k1_KO_s120
                                 + 0.99 b_k1_KO_s120 - 1.01 s_k1_KO_s120  = 0
 w_k1_KO_s121_return_constraint: 0.853081 w_k1_KO_s12 - w_k1_KO_s121
                                 + 0.99 b_k1_KO_s121 - 1.01 s_k1_KO_s121  = 0
 w_k1_KO_s122_return_constraint: 0.878779 w_k1_KO_s12 - w_k1_KO_s122
                                 + 0.99 b_k1_KO_s122 - 1.01 s_k1_KO_s122  = 0
 w_k1_KO_s123_return_constraint: 1.28426 w_k1_KO_s12 - w_k1_KO_s123
                                 + 0.99 b_k1_KO_s123 - 1.01 s_k1_KO_s123  = 0
 w_k1_KO_s124_return_constraint: 0.885464 w_k1_KO_s12 - w_k1_KO_s124
                                 + 0.99 b_k1_KO_s124 - 1.01 s_k1_KO_s124  = 0
 w_k1_KO_s125_return_constraint: 1.09581 w_k1_KO_s12 - w_k1_KO_s125
                                 + 0.99 b_k1_KO_s125 - 1.01 s_k1_KO_s125  = 0
 w_k1_KO_s126_return_constraint: 0.881735 w_k1_KO_s12 - w_k1_KO_s126
                                 + 0.99 b_k1_KO_s126 - 1.01 s_k1_KO_s126  = 0
 w_k1_KO_s127_return_constraint: 1.23184 w_k1_KO_s12 - w_k1_KO_s127
                                 + 0.99 b_k1_KO_s127 - 1.01 s_k1_KO_s127  = 0
 w_k1_KO_s130_return_constraint: 1.07245 w_k1_KO_s13 - w_k1_KO_s130
                                 + 0.99 b_k1_KO_s130 - 1.01 s_k1_KO_s130  = 0
 w_k1_KO_s131_return_constraint: 1.43399 w_k1_KO_s13 - w_k1_KO_s131
                                 + 0.99 b_k1_KO_s131 - 1.01 s_k1_KO_s131  = 0
 w_k1_KO_s132_return_constraint: 0.966388 w_k1_KO_s13 - w_k1_KO_s132
                                 + 0.99 b_k1_KO_s132 - 1.01 s_k1_KO_s132  = 0
 w_k1_KO_s133_return_constraint: 0.780457 w_k1_KO_s13 - w_k1_KO_s133
                                 + 0.99 b_k1_KO_s133 - 1.01 s_k1_KO_s133  = 0
 w_k1_KO_s134_return_constraint: 1.31903 w_k1_KO_s13 - w_k1_KO_s134
                                 + 0.99 b_k1_KO_s134 - 1.01 s_k1_KO_s134  = 0
 w_k1_KO_s135_return_constraint: 1.51308 w_k1_KO_s13 - w_k1_KO_s135
                                 + 0.99 b_k1_KO_s135 - 1.01 s_k1_KO_s135  = 0
 w_k1_KO_s136_return_constraint: 0.894967 w_k1_KO_s13 - w_k1_KO_s136
                                 + 0.99 b_k1_KO_s136 - 1.01 s_k1_KO_s136  = 0
 w_k1_KO_s137_return_constraint: 0.856242 w_k1_KO_s13 - w_k1_KO_s137
                                 + 0.99 b_k1_KO_s137 - 1.01 s_k1_KO_s137  = 0
 w_k1_KO_s140_return_constraint: 1.07972 w_k1_KO_s14 - w_k1_KO_s140
                                 + 0.99 b_k1_KO_s140 - 1.01 s_k1_KO_s140  = 0
 w_k1_KO_s141_return_constraint: 1.03687 w_k1_KO_s14 - w_k1_KO_s141
                                 + 0.99 b_k1_KO_s141 - 1.01 s_k1_KO_s141  = 0
 w_k1_KO_s142_return_constraint: 1.05222 w_k1_KO_s14 - w_k1_KO_s142
                                 + 0.99 b_k1_KO_s142 - 1.01 s_k1_KO_s142  = 0
 w_k1_KO_s143_return_constraint: 0.971981 w_k1_KO_s14 - w_k1_KO_s143
                                 + 0.99 b_k1_KO_s143 - 1.01 s_k1_KO_s143  = 0
 w_k1_KO_s144_return_constraint: 1.18127 w_k1_KO_s14 - w_k1_KO_s144
                                 + 0.99 b_k1_KO_s144 - 1.01 s_k1_KO_s144  = 0
 w_k1_KO_s145_return_constraint: 1.39623 w_k1_KO_s14 - w_k1_KO_s145
                                 + 0.99 b_k1_KO_s145 - 1.01 s_k1_KO_s145  = 0
 w_k1_KO_s146_return_constraint: 0.794378 w_k1_KO_s14 - w_k1_KO_s146
                                 + 0.99 b_k1_KO_s146 - 1.01 s_k1_KO_s146  = 0
 w_k1_KO_s147_return_constraint: 1.08639 w_k1_KO_s14 - w_k1_KO_s147
                                 + 0.99 b_k1_KO_s147 - 1.01 s_k1_KO_s147  = 0
 w_k1_KO_s150_return_constraint: 1.34976 w_k1_KO_s15 - w_k1_KO_s150
                                 + 0.99 b_k1_KO_s150 - 1.01 s_k1_KO_s150  = 0
 w_k1_KO_s151_return_constraint: 0.972901 w_k1_KO_s15 - w_k1_KO_s151
                                 + 0.99 b_k1_KO_s151 - 1.01 s_k1_KO_s151  = 0
 w_k1_KO_s152_return_constraint: 0.973778 w_k1_KO_s15 - w_k1_KO_s152
                                 + 0.99 b_k1_KO_s152 - 1.01 s_k1_KO_s152  = 0
 w_k1_KO_s153_return_constraint: 0.969633 w_k1_KO_s15 - w_k1_KO_s153
                                 + 0.99 b_k1_KO_s153 - 1.01 s_k1_KO_s153  = 0
 w_k1_KO_s154_return_constraint: 1.09552 w_k1_KO_s15 - w_k1_KO_s154
                                 + 0.99 b_k1_KO_s154 - 1.01 s_k1_KO_s154  = 0
 w_k1_KO_s155_return_constraint: 1.02979 w_k1_KO_s15 - w_k1_KO_s155
                                 + 0.99 b_k1_KO_s155 - 1.01 s_k1_KO_s155  = 0
 w_k1_KO_s156_return_constraint: 0.782092 w_k1_KO_s15 - w_k1_KO_s156
                                 + 0.99 b_k1_KO_s156 - 1.01 s_k1_KO_s156  = 0
 w_k1_KO_s157_return_constraint: 0.679175 w_k1_KO_s15 - w_k1_KO_s157
                                 + 0.99 b_k1_KO_s157 - 1.01 s_k1_KO_s157  = 0
 w_k1_KO_s160_return_constraint: 0.833433 w_k1_KO_s16 - w_k1_KO_s160
                                 + 0.99 b_k1_KO_s160 - 1.01 s_k1_KO_s160  = 0
 w_k1_KO_s161_return_constraint: 1.2759 w_k1_KO_s16 - w_k1_KO_s161
                                 + 0.99 b_k1_KO_s161 - 1.01 s_k1_KO_s161  = 0
 w_k1_KO_s162_return_constraint: 1.22982 w_k1_KO_s16 - w_k1_KO_s162
                                 + 0.99 b_k1_KO_s162 - 1.01 s_k1_KO_s162  = 0
 w_k1_KO_s163_return_constraint: 1.12445 w_k1_KO_s16 - w_k1_KO_s163
                                 + 0.99 b_k1_KO_s163 - 1.01 s_k1_KO_s163  = 0
 w_k1_KO_s164_return_constraint: 1.42843 w_k1_KO_s16 - w_k1_KO_s164
                                 + 0.99 b_k1_KO_s164 - 1.01 s_k1_KO_s164  = 0
 w_k1_KO_s165_return_constraint: 1.12845 w_k1_KO_s16 - w_k1_KO_s165
                                 + 0.99 b_k1_KO_s165 - 1.01 s_k1_KO_s165  = 0
 w_k1_KO_s166_return_constraint: 0.908661 w_k1_KO_s16 - w_k1_KO_s166
                                 + 0.99 b_k1_KO_s166 - 1.01 s_k1_KO_s166  = 0
 w_k1_KO_s167_return_constraint: 0.91948 w_k1_KO_s16 - w_k1_KO_s167
                                 + 0.99 b_k1_KO_s167 - 1.01 s_k1_KO_s167  = 0
 w_k1_KO_s170_return_constraint: 0.911158 w_k1_KO_s17 - w_k1_KO_s170
                                 + 0.99 b_k1_KO_s170 - 1.01 s_k1_KO_s170  = 0
 w_k1_KO_s171_return_constraint: 1.04563 w_k1_KO_s17 - w_k1_KO_s171
                                 + 0.99 b_k1_KO_s171 - 1.01 s_k1_KO_s171  = 0
 w_k1_KO_s172_return_constraint: 1.3076 w_k1_KO_s17 - w_k1_KO_s172
                                 + 0.99 b_k1_KO_s172 - 1.01 s_k1_KO_s172  = 0
 w_k1_KO_s173_return_constraint: 1.03531 w_k1_KO_s17 - w_k1_KO_s173
                                 + 0.99 b_k1_KO_s173 - 1.01 s_k1_KO_s173  = 0
 w_k1_KO_s174_return_constraint: 0.929957 w_k1_KO_s17 - w_k1_KO_s174
                                 + 0.99 b_k1_KO_s174 - 1.01 s_k1_KO_s174  = 0
 w_k1_KO_s175_return_constraint: 1.44028 w_k1_KO_s17 - w_k1_KO_s175
                                 + 0.99 b_k1_KO_s175 - 1.01 s_k1_KO_s175  = 0
 w_k1_KO_s176_return_constraint: 0.999518 w_k1_KO_s17 - w_k1_KO_s176
                                 + 0.99 b_k1_KO_s176 - 1.01 s_k1_KO_s176  = 0
 w_k1_KO_s177_return_constraint: 0.909054 w_k1_KO_s17 - w_k1_KO_s177
                                 + 0.99 b_k1_KO_s177 - 1.01 s_k1_KO_s177  = 0
 w_k1_MAT_s0_return_constraint:  1.31257 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.939398 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.907218 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.75829 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s02_return_constraint: 0.960061 w_k1_MAT_s0 - w_k1_MAT_s02
                                 + 0.99 b_k1_MAT_s02 - 1.01 s_k1_MAT_s02  = 0
 w_k1_MAT_s03_return_constraint: 0.651196 w_k1_MAT_s0 - w_k1_MAT_s03
                                 + 0.99 b_k1_MAT_s03 - 1.01 s_k1_MAT_s03  = 0
 w_k1_MAT_s04_return_constraint: 0.645319 w_k1_MAT_s0 - w_k1_MAT_s04
                                 + 0.99 b_k1_MAT_s04 - 1.01 s_k1_MAT_s04  = 0
 w_k1_MAT_s05_return_constraint: 0.921307 w_k1_MAT_s0 - w_k1_MAT_s05
                                 + 0.99 b_k1_MAT_s05 - 1.01 s_k1_MAT_s05  = 0
 w_k1_MAT_s06_return_constraint: 0.823255 w_k1_MAT_s0 - w_k1_MAT_s06
                                 + 0.99 b_k1_MAT_s06 - 1.01 s_k1_MAT_s06  = 0
 w_k1_MAT_s07_return_constraint: 1.53471 w_k1_MAT_s0 - w_k1_MAT_s07
                                 + 0.99 b_k1_MAT_s07 - 1.01 s_k1_MAT_s07  = 0
 w_k1_MAT_s10_return_constraint: 0.61233 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.69438 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s12_return_constraint: 1.64902 w_k1_MAT_s1 - w_k1_MAT_s12
                                 + 0.99 b_k1_MAT_s12 - 1.01 s_k1_MAT_s12  = 0
 w_k1_MAT_s13_return_constraint: 0.936295 w_k1_MAT_s1 - w_k1_MAT_s13
                                 + 0.99 b_k1_MAT_s13 - 1.01 s_k1_MAT_s13  = 0
 w_k1_MAT_s14_return_constraint: 1.54933 w_k1_MAT_s1 - w_k1_MAT_s14
                                 + 0.99 b_k1_MAT_s14 - 1.01 s_k1_MAT_s14  = 0
 w_k1_MAT_s15_return_constraint: 1.2213 w_k1_MAT_s1 - w_k1_MAT_s15
                                 + 0.99 b_k1_MAT_s15 - 1.01 s_k1_MAT_s15  = 0
 w_k1_MAT_s16_return_constraint: 0.727406 w_k1_MAT_s1 - w_k1_MAT_s16
                                 + 0.99 b_k1_MAT_s16 - 1.01 s_k1_MAT_s16  = 0
 w_k1_MAT_s17_return_constraint: 0.838423 w_k1_MAT_s1 - w_k1_MAT_s17
                                 + 0.99 b_k1_MAT_s17 - 1.01 s_k1_MAT_s17  = 0
 w_k1_MAT_s000_return_constraint: 0.787973 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.31172 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.723304 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.036 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.13432 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.674062 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.498685 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.493946 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.809431 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.726405 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.23813 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.14994 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.915654 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.971671 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.07483 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.59958 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s020_return_constraint: 0.988112 w_k1_MAT_s02 - w_k1_MAT_s020
                                 + 0.99 b_k1_MAT_s020 - 1.01 s_k1_MAT_s020  = 0
 w_k1_MAT_s021_return_constraint: 1.22882 w_k1_MAT_s02 - w_k1_MAT_s021
                                 + 0.99 b_k1_MAT_s021 - 1.01 s_k1_MAT_s021  = 0
 w_k1_MAT_s022_return_constraint: 1.71991 w_k1_MAT_s02 - w_k1_MAT_s022
                                 + 0.99 b_k1_MAT_s022 - 1.01 s_k1_MAT_s022  = 0
 w_k1_MAT_s023_return_constraint: 1.14982 w_k1_MAT_s02 - w_k1_MAT_s023
                                 + 0.99 b_k1_MAT_s023 - 1.01 s_k1_MAT_s023  = 0
 w_k1_MAT_s024_return_constraint: 1.16685 w_k1_MAT_s02 - w_k1_MAT_s024
                                 + 0.99 b_k1_MAT_s024 - 1.01 s_k1_MAT_s024  = 0
 w_k1_MAT_s025_return_constraint: 0.688609 w_k1_MAT_s02 - w_k1_MAT_s025
                                 + 0.99 b_k1_MAT_s025 - 1.01 s_k1_MAT_s025  = 0
 w_k1_MAT_s026_return_constraint: 0.875987 w_k1_MAT_s02 - w_k1_MAT_s026
                                 + 0.99 b_k1_MAT_s026 - 1.01 s_k1_MAT_s026  = 0
 w_k1_MAT_s027_return_constraint: 0.821566 w_k1_MAT_s02 - w_k1_MAT_s027
                                 + 0.99 b_k1_MAT_s027 - 1.01 s_k1_MAT_s027  = 0
 w_k1_MAT_s030_return_constraint: 1.37658 w_k1_MAT_s03 - w_k1_MAT_s030
                                 + 0.99 b_k1_MAT_s030 - 1.01 s_k1_MAT_s030  = 0
 w_k1_MAT_s031_return_constraint: 1.05446 w_k1_MAT_s03 - w_k1_MAT_s031
                                 + 0.99 b_k1_MAT_s031 - 1.01 s_k1_MAT_s031  = 0
 w_k1_MAT_s032_return_constraint: 0.565254 w_k1_MAT_s03 - w_k1_MAT_s032
                                 + 0.99 b_k1_MAT_s032 - 1.01 s_k1_MAT_s032  = 0
 w_k1_MAT_s033_return_constraint: 0.8938 w_k1_MAT_s03 - w_k1_MAT_s033
                                 + 0.99 b_k1_MAT_s033 - 1.01 s_k1_MAT_s033  = 0
 w_k1_MAT_s034_return_constraint: 0.99845 w_k1_MAT_s03 - w_k1_MAT_s034
                                 + 0.99 b_k1_MAT_s034 - 1.01 s_k1_MAT_s034  = 0
 w_k1_MAT_s035_return_constraint: 0.988205 w_k1_MAT_s03 - w_k1_MAT_s035
                                 + 0.99 b_k1_MAT_s035 - 1.01 s_k1_MAT_s035  = 0
 w_k1_MAT_s036_return_constraint: 0.913842 w_k1_MAT_s03 - w_k1_MAT_s036
                                 + 0.99 b_k1_MAT_s036 - 1.01 s_k1_MAT_s036  = 0
 w_k1_MAT_s037_return_constraint: 1.21671 w_k1_MAT_s03 - w_k1_MAT_s037
                                 + 0.99 b_k1_MAT_s037 - 1.01 s_k1_MAT_s037  = 0
 w_k1_MAT_s040_return_constraint: 1.15503 w_k1_MAT_s04 - w_k1_MAT_s040
                                 + 0.99 b_k1_MAT_s040 - 1.01 s_k1_MAT_s040  = 0
 w_k1_MAT_s041_return_constraint: 1.23397 w_k1_MAT_s04 - w_k1_MAT_s041
                                 + 0.99 b_k1_MAT_s041 - 1.01 s_k1_MAT_s041  = 0
 w_k1_MAT_s042_return_constraint: 0.797701 w_k1_MAT_s04 - w_k1_MAT_s042
                                 + 0.99 b_k1_MAT_s042 - 1.01 s_k1_MAT_s042  = 0
 w_k1_MAT_s043_return_constraint: 0.906045 w_k1_MAT_s04 - w_k1_MAT_s043
                                 + 0.99 b_k1_MAT_s043 - 1.01 s_k1_MAT_s043  = 0
 w_k1_MAT_s044_return_constraint: 1.17169 w_k1_MAT_s04 - w_k1_MAT_s044
                                 + 0.99 b_k1_MAT_s044 - 1.01 s_k1_MAT_s044  = 0
 w_k1_MAT_s045_return_constraint: 0.479113 w_k1_MAT_s04 - w_k1_MAT_s045
                                 + 0.99 b_k1_MAT_s045 - 1.01 s_k1_MAT_s045  = 0
 w_k1_MAT_s046_return_constraint: 0.850603 w_k1_MAT_s04 - w_k1_MAT_s046
                                 + 0.99 b_k1_MAT_s046 - 1.01 s_k1_MAT_s046  = 0
 w_k1_MAT_s047_return_constraint: 1.04675 w_k1_MAT_s04 - w_k1_MAT_s047
                                 + 0.99 b_k1_MAT_s047 - 1.01 s_k1_MAT_s047  = 0
 w_k1_MAT_s050_return_constraint: 0.865649 w_k1_MAT_s05 - w_k1_MAT_s050
                                 + 0.99 b_k1_MAT_s050 - 1.01 s_k1_MAT_s050  = 0
 w_k1_MAT_s051_return_constraint: 1.79493 w_k1_MAT_s05 - w_k1_MAT_s051
                                 + 0.99 b_k1_MAT_s051 - 1.01 s_k1_MAT_s051  = 0
 w_k1_MAT_s052_return_constraint: 1.25925 w_k1_MAT_s05 - w_k1_MAT_s052
                                 + 0.99 b_k1_MAT_s052 - 1.01 s_k1_MAT_s052  = 0
 w_k1_MAT_s053_return_constraint: 1.32624 w_k1_MAT_s05 - w_k1_MAT_s053
                                 + 0.99 b_k1_MAT_s053 - 1.01 s_k1_MAT_s053  = 0
 w_k1_MAT_s054_return_constraint: 0.886153 w_k1_MAT_s05 - w_k1_MAT_s054
                                 + 0.99 b_k1_MAT_s054 - 1.01 s_k1_MAT_s054  = 0
 w_k1_MAT_s055_return_constraint: 1.0274 w_k1_MAT_s05 - w_k1_MAT_s055
                                 + 0.99 b_k1_MAT_s055 - 1.01 s_k1_MAT_s055  = 0
 w_k1_MAT_s056_return_constraint: 1.82787 w_k1_MAT_s05 - w_k1_MAT_s056
                                 + 0.99 b_k1_MAT_s056 - 1.01 s_k1_MAT_s056  = 0
 w_k1_MAT_s057_return_constraint: 0.908362 w_k1_MAT_s05 - w_k1_MAT_s057
                                 + 0.99 b_k1_MAT_s057 - 1.01 s_k1_MAT_s057  = 0
 w_k1_MAT_s060_return_constraint: 0.711943 w_k1_MAT_s06 - w_k1_MAT_s060
                                 + 0.99 b_k1_MAT_s060 - 1.01 s_k1_MAT_s060  = 0
 w_k1_MAT_s061_return_constraint: 0.891274 w_k1_MAT_s06 - w_k1_MAT_s061
                                 + 0.99 b_k1_MAT_s061 - 1.01 s_k1_MAT_s061  = 0
 w_k1_MAT_s062_return_constraint: 0.700992 w_k1_MAT_s06 - w_k1_MAT_s062
                                 + 0.99 b_k1_MAT_s062 - 1.01 s_k1_MAT_s062  = 0
 w_k1_MAT_s063_return_constraint: 0.76056 w_k1_MAT_s06 - w_k1_MAT_s063
                                 + 0.99 b_k1_MAT_s063 - 1.01 s_k1_MAT_s063  = 0
 w_k1_MAT_s064_return_constraint: 0.961669 w_k1_MAT_s06 - w_k1_MAT_s064
                                 + 0.99 b_k1_MAT_s064 - 1.01 s_k1_MAT_s064  = 0
 w_k1_MAT_s065_return_constraint: 0.960043 w_k1_MAT_s06 - w_k1_MAT_s065
                                 + 0.99 b_k1_MAT_s065 - 1.01 s_k1_MAT_s065  = 0
 w_k1_MAT_s066_return_constraint: 1.44681 w_k1_MAT_s06 - w_k1_MAT_s066
                                 + 0.99 b_k1_MAT_s066 - 1.01 s_k1_MAT_s066  = 0
 w_k1_MAT_s067_return_constraint: 1.53679 w_k1_MAT_s06 - w_k1_MAT_s067
                                 + 0.99 b_k1_MAT_s067 - 1.01 s_k1_MAT_s067  = 0
 w_k1_MAT_s070_return_constraint: 0.717587 w_k1_MAT_s07 - w_k1_MAT_s070
                                 + 0.99 b_k1_MAT_s070 - 1.01 s_k1_MAT_s070  = 0
 w_k1_MAT_s071_return_constraint: 1.63463 w_k1_MAT_s07 - w_k1_MAT_s071
                                 + 0.99 b_k1_MAT_s071 - 1.01 s_k1_MAT_s071  = 0
 w_k1_MAT_s072_return_constraint: 0.999296 w_k1_MAT_s07 - w_k1_MAT_s072
                                 + 0.99 b_k1_MAT_s072 - 1.01 s_k1_MAT_s072  = 0
 w_k1_MAT_s073_return_constraint: 1.33441 w_k1_MAT_s07 - w_k1_MAT_s073
                                 + 0.99 b_k1_MAT_s073 - 1.01 s_k1_MAT_s073  = 0
 w_k1_MAT_s074_return_constraint: 0.685269 w_k1_MAT_s07 - w_k1_MAT_s074
                                 + 0.99 b_k1_MAT_s074 - 1.01 s_k1_MAT_s074  = 0
 w_k1_MAT_s075_return_constraint: 0.674676 w_k1_MAT_s07 - w_k1_MAT_s075
                                 + 0.99 b_k1_MAT_s075 - 1.01 s_k1_MAT_s075  = 0
 w_k1_MAT_s076_return_constraint: 0.446769 w_k1_MAT_s07 - w_k1_MAT_s076
                                 + 0.99 b_k1_MAT_s076 - 1.01 s_k1_MAT_s076  = 0
 w_k1_MAT_s077_return_constraint: 1.5592 w_k1_MAT_s07 - w_k1_MAT_s077
                                 + 0.99 b_k1_MAT_s077 - 1.01 s_k1_MAT_s077  = 0
 w_k1_MAT_s100_return_constraint: 1.80844 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.2029 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.44054 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.572205 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.32802 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.766966 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.13429 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.828205 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.04048 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.08347 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.831658 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.963256 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.06337 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.720393 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.72413 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.64332 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MAT_s120_return_constraint: 1.31726 w_k1_MAT_s12 - w_k1_MAT_s120
                                 + 0.99 b_k1_MAT_s120 - 1.01 s_k1_MAT_s120  = 0
 w_k1_MAT_s121_return_constraint: 0.606766 w_k1_MAT_s12 - w_k1_MAT_s121
                                 + 0.99 b_k1_MAT_s121 - 1.01 s_k1_MAT_s121  = 0
 w_k1_MAT_s122_return_constraint: 0.634551 w_k1_MAT_s12 - w_k1_MAT_s122
                                 + 0.99 b_k1_MAT_s122 - 1.01 s_k1_MAT_s122  = 0
 w_k1_MAT_s123_return_constraint: 1.7069 w_k1_MAT_s12 - w_k1_MAT_s123
                                 + 0.99 b_k1_MAT_s123 - 1.01 s_k1_MAT_s123  = 0
 w_k1_MAT_s124_return_constraint: 0.719037 w_k1_MAT_s12 - w_k1_MAT_s124
                                 + 0.99 b_k1_MAT_s124 - 1.01 s_k1_MAT_s124  = 0
 w_k1_MAT_s125_return_constraint: 0.933054 w_k1_MAT_s12 - w_k1_MAT_s125
                                 + 0.99 b_k1_MAT_s125 - 1.01 s_k1_MAT_s125  = 0
 w_k1_MAT_s126_return_constraint: 0.741529 w_k1_MAT_s12 - w_k1_MAT_s126
                                 + 0.99 b_k1_MAT_s126 - 1.01 s_k1_MAT_s126  = 0
 w_k1_MAT_s127_return_constraint: 0.966489 w_k1_MAT_s12 - w_k1_MAT_s127
                                 + 0.99 b_k1_MAT_s127 - 1.01 s_k1_MAT_s127  = 0
 w_k1_MAT_s130_return_constraint: 0.878201 w_k1_MAT_s13 - w_k1_MAT_s130
                                 + 0.99 b_k1_MAT_s130 - 1.01 s_k1_MAT_s130  = 0
 w_k1_MAT_s131_return_constraint: 2.23935 w_k1_MAT_s13 - w_k1_MAT_s131
                                 + 0.99 b_k1_MAT_s131 - 1.01 s_k1_MAT_s131  = 0
 w_k1_MAT_s132_return_constraint: 0.720757 w_k1_MAT_s13 - w_k1_MAT_s132
                                 + 0.99 b_k1_MAT_s132 - 1.01 s_k1_MAT_s132  = 0
 w_k1_MAT_s133_return_constraint: 0.535215 w_k1_MAT_s13 - w_k1_MAT_s133
                                 + 0.99 b_k1_MAT_s133 - 1.01 s_k1_MAT_s133  = 0
 w_k1_MAT_s134_return_constraint: 1.1533 w_k1_MAT_s13 - w_k1_MAT_s134
                                 + 0.99 b_k1_MAT_s134 - 1.01 s_k1_MAT_s134  = 0
 w_k1_MAT_s135_return_constraint: 1.45259 w_k1_MAT_s13 - w_k1_MAT_s135
                                 + 0.99 b_k1_MAT_s135 - 1.01 s_k1_MAT_s135  = 0
 w_k1_MAT_s136_return_constraint: 1.11067 w_k1_MAT_s13 - w_k1_MAT_s136
                                 + 0.99 b_k1_MAT_s136 - 1.01 s_k1_MAT_s136  = 0
 w_k1_MAT_s137_return_constraint: 0.850204 w_k1_MAT_s13 - w_k1_MAT_s137
                                 + 0.99 b_k1_MAT_s137 - 1.01 s_k1_MAT_s137  = 0
 w_k1_MAT_s140_return_constraint: 1.32073 w_k1_MAT_s14 - w_k1_MAT_s140
                                 + 0.99 b_k1_MAT_s140 - 1.01 s_k1_MAT_s140  = 0
 w_k1_MAT_s141_return_constraint: 0.7237 w_k1_MAT_s14 - w_k1_MAT_s141
                                 + 0.99 b_k1_MAT_s141 - 1.01 s_k1_MAT_s141  = 0
 w_k1_MAT_s142_return_constraint: 0.910986 w_k1_MAT_s14 - w_k1_MAT_s142
                                 + 0.99 b_k1_MAT_s142 - 1.01 s_k1_MAT_s142  = 0
 w_k1_MAT_s143_return_constraint: 0.898973 w_k1_MAT_s14 - w_k1_MAT_s143
                                 + 0.99 b_k1_MAT_s143 - 1.01 s_k1_MAT_s143  = 0
 w_k1_MAT_s144_return_constraint: 1.34483 w_k1_MAT_s14 - w_k1_MAT_s144
                                 + 0.99 b_k1_MAT_s144 - 1.01 s_k1_MAT_s144  = 0
 w_k1_MAT_s145_return_constraint: 1.86702 w_k1_MAT_s14 - w_k1_MAT_s145
                                 + 0.99 b_k1_MAT_s145 - 1.01 s_k1_MAT_s145  = 0
 w_k1_MAT_s146_return_constraint: 0.691626 w_k1_MAT_s14 - w_k1_MAT_s146
                                 + 0.99 b_k1_MAT_s146 - 1.01 s_k1_MAT_s146  = 0
 w_k1_MAT_s147_return_constraint: 0.717811 w_k1_MAT_s14 - w_k1_MAT_s147
                                 + 0.99 b_k1_MAT_s147 - 1.01 s_k1_MAT_s147  = 0
 w_k1_MAT_s150_return_constraint: 1.42073 w_k1_MAT_s15 - w_k1_MAT_s150
                                 + 0.99 b_k1_MAT_s150 - 1.01 s_k1_MAT_s150  = 0
 w_k1_MAT_s151_return_constraint: 1.23413 w_k1_MAT_s15 - w_k1_MAT_s151
                                 + 0.99 b_k1_MAT_s151 - 1.01 s_k1_MAT_s151  = 0
 w_k1_MAT_s152_return_constraint: 0.736613 w_k1_MAT_s15 - w_k1_MAT_s152
                                 + 0.99 b_k1_MAT_s152 - 1.01 s_k1_MAT_s152  = 0
 w_k1_MAT_s153_return_constraint: 0.825726 w_k1_MAT_s15 - w_k1_MAT_s153
                                 + 0.99 b_k1_MAT_s153 - 1.01 s_k1_MAT_s153  = 0
 w_k1_MAT_s154_return_constraint: 0.682747 w_k1_MAT_s15 - w_k1_MAT_s154
                                 + 0.99 b_k1_MAT_s154 - 1.01 s_k1_MAT_s154  = 0
 w_k1_MAT_s155_return_constraint: 1.04914 w_k1_MAT_s15 - w_k1_MAT_s155
                                 + 0.99 b_k1_MAT_s155 - 1.01 s_k1_MAT_s155  = 0
 w_k1_MAT_s156_return_constraint: 0.666729 w_k1_MAT_s15 - w_k1_MAT_s156
                                 + 0.99 b_k1_MAT_s156 - 1.01 s_k1_MAT_s156  = 0
 w_k1_MAT_s157_return_constraint: 0.708933 w_k1_MAT_s15 - w_k1_MAT_s157
                                 + 0.99 b_k1_MAT_s157 - 1.01 s_k1_MAT_s157  = 0
 w_k1_MAT_s160_return_constraint: 0.820158 w_k1_MAT_s16 - w_k1_MAT_s160
                                 + 0.99 b_k1_MAT_s160 - 1.01 s_k1_MAT_s160  = 0
 w_k1_MAT_s161_return_constraint: 1.5202 w_k1_MAT_s16 - w_k1_MAT_s161
                                 + 0.99 b_k1_MAT_s161 - 1.01 s_k1_MAT_s161  = 0
 w_k1_MAT_s162_return_constraint: 1.2847 w_k1_MAT_s16 - w_k1_MAT_s162
                                 + 0.99 b_k1_MAT_s162 - 1.01 s_k1_MAT_s162  = 0
 w_k1_MAT_s163_return_constraint: 1.52058 w_k1_MAT_s16 - w_k1_MAT_s163
                                 + 0.99 b_k1_MAT_s163 - 1.01 s_k1_MAT_s163  = 0
 w_k1_MAT_s164_return_constraint: 1.34093 w_k1_MAT_s16 - w_k1_MAT_s164
                                 + 0.99 b_k1_MAT_s164 - 1.01 s_k1_MAT_s164  = 0
 w_k1_MAT_s165_return_constraint: 0.674251 w_k1_MAT_s16 - w_k1_MAT_s165
                                 + 0.99 b_k1_MAT_s165 - 1.01 s_k1_MAT_s165  = 0
 w_k1_MAT_s166_return_constraint: 0.684137 w_k1_MAT_s16 - w_k1_MAT_s166
                                 + 0.99 b_k1_MAT_s166 - 1.01 s_k1_MAT_s166  = 0
 w_k1_MAT_s167_return_constraint: 0.868901 w_k1_MAT_s16 - w_k1_MAT_s167
                                 + 0.99 b_k1_MAT_s167 - 1.01 s_k1_MAT_s167  = 0
 w_k1_MAT_s170_return_constraint: 0.867421 w_k1_MAT_s17 - w_k1_MAT_s170
                                 + 0.99 b_k1_MAT_s170 - 1.01 s_k1_MAT_s170  = 0
 w_k1_MAT_s171_return_constraint: 1.19054 w_k1_MAT_s17 - w_k1_MAT_s171
                                 + 0.99 b_k1_MAT_s171 - 1.01 s_k1_MAT_s171  = 0
 w_k1_MAT_s172_return_constraint: 1.05878 w_k1_MAT_s17 - w_k1_MAT_s172
                                 + 0.99 b_k1_MAT_s172 - 1.01 s_k1_MAT_s172  = 0
 w_k1_MAT_s173_return_constraint: 1.32881 w_k1_MAT_s17 - w_k1_MAT_s173
                                 + 0.99 b_k1_MAT_s173 - 1.01 s_k1_MAT_s173  = 0
 w_k1_MAT_s174_return_constraint: 1.09676 w_k1_MAT_s17 - w_k1_MAT_s174
                                 + 0.99 b_k1_MAT_s174 - 1.01 s_k1_MAT_s174  = 0
 w_k1_MAT_s175_return_constraint: 1.66803 w_k1_MAT_s17 - w_k1_MAT_s175
                                 + 0.99 b_k1_MAT_s175 - 1.01 s_k1_MAT_s175  = 0
 w_k1_MAT_s176_return_constraint: 0.889901 w_k1_MAT_s17 - w_k1_MAT_s176
                                 + 0.99 b_k1_MAT_s176 - 1.01 s_k1_MAT_s176  = 0
 w_k1_MAT_s177_return_constraint: 0.689038 w_k1_MAT_s17 - w_k1_MAT_s177
                                 + 0.99 b_k1_MAT_s177 - 1.01 s_k1_MAT_s177  = 0
 w_k1_MSFT_s0_return_constraint: 1.19116 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.950617 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.988344 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.25159 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s02_return_constraint: 0.926678 w_k1_MSFT_s0 - w_k1_MSFT_s02
                                 + 0.99 b_k1_MSFT_s02 - 1.01 s_k1_MSFT_s02  = 0
 w_k1_MSFT_s03_return_constraint: 1.01325 w_k1_MSFT_s0 - w_k1_MSFT_s03
                                 + 0.99 b_k1_MSFT_s03 - 1.01 s_k1_MSFT_s03  = 0
 w_k1_MSFT_s04_return_constraint: 0.781776 w_k1_MSFT_s0 - w_k1_MSFT_s04
                                 + 0.99 b_k1_MSFT_s04 - 1.01 s_k1_MSFT_s04  = 0
 w_k1_MSFT_s05_return_constraint: 0.997568 w_k1_MSFT_s0 - w_k1_MSFT_s05
                                 + 0.99 b_k1_MSFT_s05 - 1.01 s_k1_MSFT_s05  = 0
 w_k1_MSFT_s06_return_constraint: 0.808983 w_k1_MSFT_s0 - w_k1_MSFT_s06
                                 + 0.99 b_k1_MSFT_s06 - 1.01 s_k1_MSFT_s06  = 0
 w_k1_MSFT_s07_return_constraint: 1.14992 w_k1_MSFT_s0 - w_k1_MSFT_s07
                                 + 0.99 b_k1_MSFT_s07 - 1.01 s_k1_MSFT_s07  = 0
 w_k1_MSFT_s10_return_constraint: 0.789676 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.10361 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s12_return_constraint: 1.36206 w_k1_MSFT_s1 - w_k1_MSFT_s12
                                 + 0.99 b_k1_MSFT_s12 - 1.01 s_k1_MSFT_s12  = 0
 w_k1_MSFT_s13_return_constraint: 1.06789 w_k1_MSFT_s1 - w_k1_MSFT_s13
                                 + 0.99 b_k1_MSFT_s13 - 1.01 s_k1_MSFT_s13  = 0
 w_k1_MSFT_s14_return_constraint: 0.973575 w_k1_MSFT_s1 - w_k1_MSFT_s14
                                 + 0.99 b_k1_MSFT_s14 - 1.01 s_k1_MSFT_s14  = 0
 w_k1_MSFT_s15_return_constraint: 0.957255 w_k1_MSFT_s1 - w_k1_MSFT_s15
                                 + 0.99 b_k1_MSFT_s15 - 1.01 s_k1_MSFT_s15  = 0
 w_k1_MSFT_s16_return_constraint: 0.854297 w_k1_MSFT_s1 - w_k1_MSFT_s16
                                 + 0.99 b_k1_MSFT_s16 - 1.01 s_k1_MSFT_s16  = 0
 w_k1_MSFT_s17_return_constraint: 0.910813 w_k1_MSFT_s1 - w_k1_MSFT_s17
                                 + 0.99 b_k1_MSFT_s17 - 1.01 s_k1_MSFT_s17  = 0
 w_k1_MSFT_s000_return_constraint: 1.11946 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.1498 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.750556 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.10656 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.856128 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.843714 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.13909 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.740804 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.783311 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.908329 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.16399 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.07435 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.678625 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.30229 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.01811 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.16765 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s020_return_constraint: 0.895933 w_k1_MSFT_s02 - w_k1_MSFT_s020
                                 + 0.99 b_k1_MSFT_s020 - 1.01 s_k1_MSFT_s020
                                  = 0
 w_k1_MSFT_s021_return_constraint: 1.27782 w_k1_MSFT_s02 - w_k1_MSFT_s021
                                 + 0.99 b_k1_MSFT_s021 - 1.01 s_k1_MSFT_s021
                                  = 0
 w_k1_MSFT_s022_return_constraint: 1.10156 w_k1_MSFT_s02 - w_k1_MSFT_s022
                                 + 0.99 b_k1_MSFT_s022 - 1.01 s_k1_MSFT_s022
                                  = 0
 w_k1_MSFT_s023_return_constraint: 0.923391 w_k1_MSFT_s02 - w_k1_MSFT_s023
                                 + 0.99 b_k1_MSFT_s023 - 1.01 s_k1_MSFT_s023
                                  = 0
 w_k1_MSFT_s024_return_constraint: 0.965656 w_k1_MSFT_s02 - w_k1_MSFT_s024
                                 + 0.99 b_k1_MSFT_s024 - 1.01 s_k1_MSFT_s024
                                  = 0
 w_k1_MSFT_s025_return_constraint: 0.922689 w_k1_MSFT_s02 - w_k1_MSFT_s025
                                 + 0.99 b_k1_MSFT_s025 - 1.01 s_k1_MSFT_s025
                                  = 0
 w_k1_MSFT_s026_return_constraint: 0.828281 w_k1_MSFT_s02 - w_k1_MSFT_s026
                                 + 0.99 b_k1_MSFT_s026 - 1.01 s_k1_MSFT_s026
                                  = 0
 w_k1_MSFT_s027_return_constraint: 1.02128 w_k1_MSFT_s02 - w_k1_MSFT_s027
                                 + 0.99 b_k1_MSFT_s027 - 1.01 s_k1_MSFT_s027
                                  = 0
 w_k1_MSFT_s030_return_constraint: 1.13596 w_k1_MSFT_s03 - w_k1_MSFT_s030
                                 + 0.99 b_k1_MSFT_s030 - 1.01 s_k1_MSFT_s030
                                  = 0
 w_k1_MSFT_s031_return_constraint: 1.27488 w_k1_MSFT_s03 - w_k1_MSFT_s031
                                 + 0.99 b_k1_MSFT_s031 - 1.01 s_k1_MSFT_s031
                                  = 0
 w_k1_MSFT_s032_return_constraint: 0.76879 w_k1_MSFT_s03 - w_k1_MSFT_s032
                                 + 0.99 b_k1_MSFT_s032 - 1.01 s_k1_MSFT_s032
                                  = 0
 w_k1_MSFT_s033_return_constraint: 0.897093 w_k1_MSFT_s03 - w_k1_MSFT_s033
                                 + 0.99 b_k1_MSFT_s033 - 1.01 s_k1_MSFT_s033
                                  = 0
 w_k1_MSFT_s034_return_constraint: 0.87649 w_k1_MSFT_s03 - w_k1_MSFT_s034
                                 + 0.99 b_k1_MSFT_s034 - 1.01 s_k1_MSFT_s034
                                  = 0
 w_k1_MSFT_s035_return_constraint: 1.07042 w_k1_MSFT_s03 - w_k1_MSFT_s035
                                 + 0.99 b_k1_MSFT_s035 - 1.01 s_k1_MSFT_s035
                                  = 0
 w_k1_MSFT_s036_return_constraint: 1.10651 w_k1_MSFT_s03 - w_k1_MSFT_s036
                                 + 0.99 b_k1_MSFT_s036 - 1.01 s_k1_MSFT_s036
                                  = 0
 w_k1_MSFT_s037_return_constraint: 0.845151 w_k1_MSFT_s03 - w_k1_MSFT_s037
                                 + 0.99 b_k1_MSFT_s037 - 1.01 s_k1_MSFT_s037
                                  = 0
 w_k1_MSFT_s040_return_constraint: 1.34516 w_k1_MSFT_s04 - w_k1_MSFT_s040
                                 + 0.99 b_k1_MSFT_s040 - 1.01 s_k1_MSFT_s040
                                  = 0
 w_k1_MSFT_s041_return_constraint: 0.827258 w_k1_MSFT_s04 - w_k1_MSFT_s041
                                 + 0.99 b_k1_MSFT_s041 - 1.01 s_k1_MSFT_s041
                                  = 0
 w_k1_MSFT_s042_return_constraint: 1.0817 w_k1_MSFT_s04 - w_k1_MSFT_s042
                                 + 0.99 b_k1_MSFT_s042 - 1.01 s_k1_MSFT_s042
                                  = 0
 w_k1_MSFT_s043_return_constraint: 0.950471 w_k1_MSFT_s04 - w_k1_MSFT_s043
                                 + 0.99 b_k1_MSFT_s043 - 1.01 s_k1_MSFT_s043
                                  = 0
 w_k1_MSFT_s044_return_constraint: 1.19951 w_k1_MSFT_s04 - w_k1_MSFT_s044
                                 + 0.99 b_k1_MSFT_s044 - 1.01 s_k1_MSFT_s044
                                  = 0
 w_k1_MSFT_s045_return_constraint: 0.863911 w_k1_MSFT_s04 - w_k1_MSFT_s045
                                 + 0.99 b_k1_MSFT_s045 - 1.01 s_k1_MSFT_s045
                                  = 0
 w_k1_MSFT_s046_return_constraint: 0.703192 w_k1_MSFT_s04 - w_k1_MSFT_s046
                                 + 0.99 b_k1_MSFT_s046 - 1.01 s_k1_MSFT_s046
                                  = 0
 w_k1_MSFT_s047_return_constraint: 0.795434 w_k1_MSFT_s04 - w_k1_MSFT_s047
                                 + 0.99 b_k1_MSFT_s047 - 1.01 s_k1_MSFT_s047
                                  = 0
 w_k1_MSFT_s050_return_constraint: 0.750646 w_k1_MSFT_s05 - w_k1_MSFT_s050
                                 + 0.99 b_k1_MSFT_s050 - 1.01 s_k1_MSFT_s050
                                  = 0
 w_k1_MSFT_s051_return_constraint: 0.964494 w_k1_MSFT_s05 - w_k1_MSFT_s051
                                 + 0.99 b_k1_MSFT_s051 - 1.01 s_k1_MSFT_s051
                                  = 0
 w_k1_MSFT_s052_return_constraint: 0.883166 w_k1_MSFT_s05 - w_k1_MSFT_s052
                                 + 0.99 b_k1_MSFT_s052 - 1.01 s_k1_MSFT_s052
                                  = 0
 w_k1_MSFT_s053_return_constraint: 1.09573 w_k1_MSFT_s05 - w_k1_MSFT_s053
                                 + 0.99 b_k1_MSFT_s053 - 1.01 s_k1_MSFT_s053
                                  = 0
 w_k1_MSFT_s054_return_constraint: 1.10186 w_k1_MSFT_s05 - w_k1_MSFT_s054
                                 + 0.99 b_k1_MSFT_s054 - 1.01 s_k1_MSFT_s054
                                  = 0
 w_k1_MSFT_s055_return_constraint: 0.90544 w_k1_MSFT_s05 - w_k1_MSFT_s055
                                 + 0.99 b_k1_MSFT_s055 - 1.01 s_k1_MSFT_s055
                                  = 0
 w_k1_MSFT_s056_return_constraint: 1.36241 w_k1_MSFT_s05 - w_k1_MSFT_s056
                                 + 0.99 b_k1_MSFT_s056 - 1.01 s_k1_MSFT_s056
                                  = 0
 w_k1_MSFT_s057_return_constraint: 1.0051 w_k1_MSFT_s05 - w_k1_MSFT_s057
                                 + 0.99 b_k1_MSFT_s057 - 1.01 s_k1_MSFT_s057
                                  = 0
 w_k1_MSFT_s060_return_constraint: 1.08029 w_k1_MSFT_s06 - w_k1_MSFT_s060
                                 + 0.99 b_k1_MSFT_s060 - 1.01 s_k1_MSFT_s060
                                  = 0
 w_k1_MSFT_s061_return_constraint: 1.00261 w_k1_MSFT_s06 - w_k1_MSFT_s061
                                 + 0.99 b_k1_MSFT_s061 - 1.01 s_k1_MSFT_s061
                                  = 0
 w_k1_MSFT_s062_return_constraint: 0.929983 w_k1_MSFT_s06 - w_k1_MSFT_s062
                                 + 0.99 b_k1_MSFT_s062 - 1.01 s_k1_MSFT_s062
                                  = 0
 w_k1_MSFT_s063_return_constraint: 0.722514 w_k1_MSFT_s06 - w_k1_MSFT_s063
                                 + 0.99 b_k1_MSFT_s063 - 1.01 s_k1_MSFT_s063
                                  = 0
 w_k1_MSFT_s064_return_constraint: 1.06858 w_k1_MSFT_s06 - w_k1_MSFT_s064
                                 + 0.99 b_k1_MSFT_s064 - 1.01 s_k1_MSFT_s064
                                  = 0
 w_k1_MSFT_s065_return_constraint: 0.873913 w_k1_MSFT_s06 - w_k1_MSFT_s065
                                 + 0.99 b_k1_MSFT_s065 - 1.01 s_k1_MSFT_s065
                                  = 0
 w_k1_MSFT_s066_return_constraint: 1.23718 w_k1_MSFT_s06 - w_k1_MSFT_s066
                                 + 0.99 b_k1_MSFT_s066 - 1.01 s_k1_MSFT_s066
                                  = 0
 w_k1_MSFT_s067_return_constraint: 1.11403 w_k1_MSFT_s06 - w_k1_MSFT_s067
                                 + 0.99 b_k1_MSFT_s067 - 1.01 s_k1_MSFT_s067
                                  = 0
 w_k1_MSFT_s070_return_constraint: 1.09665 w_k1_MSFT_s07 - w_k1_MSFT_s070
                                 + 0.99 b_k1_MSFT_s070 - 1.01 s_k1_MSFT_s070
                                  = 0
 w_k1_MSFT_s071_return_constraint: 1.48908 w_k1_MSFT_s07 - w_k1_MSFT_s071
                                 + 0.99 b_k1_MSFT_s071 - 1.01 s_k1_MSFT_s071
                                  = 0
 w_k1_MSFT_s072_return_constraint: 0.890825 w_k1_MSFT_s07 - w_k1_MSFT_s072
                                 + 0.99 b_k1_MSFT_s072 - 1.01 s_k1_MSFT_s072
                                  = 0
 w_k1_MSFT_s073_return_constraint: 1.00045 w_k1_MSFT_s07 - w_k1_MSFT_s073
                                 + 0.99 b_k1_MSFT_s073 - 1.01 s_k1_MSFT_s073
                                  = 0
 w_k1_MSFT_s074_return_constraint: 1.14219 w_k1_MSFT_s07 - w_k1_MSFT_s074
                                 + 0.99 b_k1_MSFT_s074 - 1.01 s_k1_MSFT_s074
                                  = 0
 w_k1_MSFT_s075_return_constraint: 0.710478 w_k1_MSFT_s07 - w_k1_MSFT_s075
                                 + 0.99 b_k1_MSFT_s075 - 1.01 s_k1_MSFT_s075
                                  = 0
 w_k1_MSFT_s076_return_constraint: 0.89461 w_k1_MSFT_s07 - w_k1_MSFT_s076
                                 + 0.99 b_k1_MSFT_s076 - 1.01 s_k1_MSFT_s076
                                  = 0
 w_k1_MSFT_s077_return_constraint: 1.15417 w_k1_MSFT_s07 - w_k1_MSFT_s077
                                 + 0.99 b_k1_MSFT_s077 - 1.01 s_k1_MSFT_s077
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.25751 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.978108 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.674532 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.957053 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.04068 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.791646 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.20787 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.887923 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.16424 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.23919 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.09556 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.868261 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.900078 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.910665 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.60118 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.887249 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_MSFT_s120_return_constraint: 1.36129 w_k1_MSFT_s12 - w_k1_MSFT_s120
                                 + 0.99 b_k1_MSFT_s120 - 1.01 s_k1_MSFT_s120
                                  = 0
 w_k1_MSFT_s121_return_constraint: 0.798182 w_k1_MSFT_s12 - w_k1_MSFT_s121
                                 + 0.99 b_k1_MSFT_s121 - 1.01 s_k1_MSFT_s121
                                  = 0
 w_k1_MSFT_s122_return_constraint: 0.965782 w_k1_MSFT_s12 - w_k1_MSFT_s122
                                 + 0.99 b_k1_MSFT_s122 - 1.01 s_k1_MSFT_s122
                                  = 0
 w_k1_MSFT_s123_return_constraint: 1.06545 w_k1_MSFT_s12 - w_k1_MSFT_s123
                                 + 0.99 b_k1_MSFT_s123 - 1.01 s_k1_MSFT_s123
                                  = 0
 w_k1_MSFT_s124_return_constraint: 1.12002 w_k1_MSFT_s12 - w_k1_MSFT_s124
                                 + 0.99 b_k1_MSFT_s124 - 1.01 s_k1_MSFT_s124
                                  = 0
 w_k1_MSFT_s125_return_constraint: 1.08573 w_k1_MSFT_s12 - w_k1_MSFT_s125
                                 + 0.99 b_k1_MSFT_s125 - 1.01 s_k1_MSFT_s125
                                  = 0
 w_k1_MSFT_s126_return_constraint: 0.800454 w_k1_MSFT_s12 - w_k1_MSFT_s126
                                 + 0.99 b_k1_MSFT_s126 - 1.01 s_k1_MSFT_s126
                                  = 0
 w_k1_MSFT_s127_return_constraint: 1.06349 w_k1_MSFT_s12 - w_k1_MSFT_s127
                                 + 0.99 b_k1_MSFT_s127 - 1.01 s_k1_MSFT_s127
                                  = 0
 w_k1_MSFT_s130_return_constraint: 1.31704 w_k1_MSFT_s13 - w_k1_MSFT_s130
                                 + 0.99 b_k1_MSFT_s130 - 1.01 s_k1_MSFT_s130
                                  = 0
 w_k1_MSFT_s131_return_constraint: 1.10694 w_k1_MSFT_s13 - w_k1_MSFT_s131
                                 + 0.99 b_k1_MSFT_s131 - 1.01 s_k1_MSFT_s131
                                  = 0
 w_k1_MSFT_s132_return_constraint: 1.03796 w_k1_MSFT_s13 - w_k1_MSFT_s132
                                 + 0.99 b_k1_MSFT_s132 - 1.01 s_k1_MSFT_s132
                                  = 0
 w_k1_MSFT_s133_return_constraint: 0.76143 w_k1_MSFT_s13 - w_k1_MSFT_s133
                                 + 0.99 b_k1_MSFT_s133 - 1.01 s_k1_MSFT_s133
                                  = 0
 w_k1_MSFT_s134_return_constraint: 1.19726 w_k1_MSFT_s13 - w_k1_MSFT_s134
                                 + 0.99 b_k1_MSFT_s134 - 1.01 s_k1_MSFT_s134
                                  = 0
 w_k1_MSFT_s135_return_constraint: 1.3478 w_k1_MSFT_s13 - w_k1_MSFT_s135
                                 + 0.99 b_k1_MSFT_s135 - 1.01 s_k1_MSFT_s135
                                  = 0
 w_k1_MSFT_s136_return_constraint: 0.95012 w_k1_MSFT_s13 - w_k1_MSFT_s136
                                 + 0.99 b_k1_MSFT_s136 - 1.01 s_k1_MSFT_s136
                                  = 0
 w_k1_MSFT_s137_return_constraint: 0.959088 w_k1_MSFT_s13 - w_k1_MSFT_s137
                                 + 0.99 b_k1_MSFT_s137 - 1.01 s_k1_MSFT_s137
                                  = 0
 w_k1_MSFT_s140_return_constraint: 0.863421 w_k1_MSFT_s14 - w_k1_MSFT_s140
                                 + 0.99 b_k1_MSFT_s140 - 1.01 s_k1_MSFT_s140
                                  = 0
 w_k1_MSFT_s141_return_constraint: 1.07442 w_k1_MSFT_s14 - w_k1_MSFT_s141
                                 + 0.99 b_k1_MSFT_s141 - 1.01 s_k1_MSFT_s141
                                  = 0
 w_k1_MSFT_s142_return_constraint: 0.910584 w_k1_MSFT_s14 - w_k1_MSFT_s142
                                 + 0.99 b_k1_MSFT_s142 - 1.01 s_k1_MSFT_s142
                                  = 0
 w_k1_MSFT_s143_return_constraint: 0.877913 w_k1_MSFT_s14 - w_k1_MSFT_s143
                                 + 0.99 b_k1_MSFT_s143 - 1.01 s_k1_MSFT_s143
                                  = 0
 w_k1_MSFT_s144_return_constraint: 1.20402 w_k1_MSFT_s14 - w_k1_MSFT_s144
                                 + 0.99 b_k1_MSFT_s144 - 1.01 s_k1_MSFT_s144
                                  = 0
 w_k1_MSFT_s145_return_constraint: 0.961908 w_k1_MSFT_s14 - w_k1_MSFT_s145
                                 + 0.99 b_k1_MSFT_s145 - 1.01 s_k1_MSFT_s145
                                  = 0
 w_k1_MSFT_s146_return_constraint: 0.791504 w_k1_MSFT_s14 - w_k1_MSFT_s146
                                 + 0.99 b_k1_MSFT_s146 - 1.01 s_k1_MSFT_s146
                                  = 0
 w_k1_MSFT_s147_return_constraint: 1.398 w_k1_MSFT_s14 - w_k1_MSFT_s147
                                 + 0.99 b_k1_MSFT_s147 - 1.01 s_k1_MSFT_s147
                                  = 0
 w_k1_MSFT_s150_return_constraint: 1.07039 w_k1_MSFT_s15 - w_k1_MSFT_s150
                                 + 0.99 b_k1_MSFT_s150 - 1.01 s_k1_MSFT_s150
                                  = 0
 w_k1_MSFT_s151_return_constraint: 0.931706 w_k1_MSFT_s15 - w_k1_MSFT_s151
                                 + 0.99 b_k1_MSFT_s151 - 1.01 s_k1_MSFT_s151
                                  = 0
 w_k1_MSFT_s152_return_constraint: 1.0573 w_k1_MSFT_s15 - w_k1_MSFT_s152
                                 + 0.99 b_k1_MSFT_s152 - 1.01 s_k1_MSFT_s152
                                  = 0
 w_k1_MSFT_s153_return_constraint: 1.13899 w_k1_MSFT_s15 - w_k1_MSFT_s153
                                 + 0.99 b_k1_MSFT_s153 - 1.01 s_k1_MSFT_s153
                                  = 0
 w_k1_MSFT_s154_return_constraint: 1.19708 w_k1_MSFT_s15 - w_k1_MSFT_s154
                                 + 0.99 b_k1_MSFT_s154 - 1.01 s_k1_MSFT_s154
                                  = 0
 w_k1_MSFT_s155_return_constraint: 0.871226 w_k1_MSFT_s15 - w_k1_MSFT_s155
                                 + 0.99 b_k1_MSFT_s155 - 1.01 s_k1_MSFT_s155
                                  = 0
 w_k1_MSFT_s156_return_constraint: 0.883767 w_k1_MSFT_s15 - w_k1_MSFT_s156
                                 + 0.99 b_k1_MSFT_s156 - 1.01 s_k1_MSFT_s156
                                  = 0
 w_k1_MSFT_s157_return_constraint: 0.646122 w_k1_MSFT_s15 - w_k1_MSFT_s157
                                 + 0.99 b_k1_MSFT_s157 - 1.01 s_k1_MSFT_s157
                                  = 0
 w_k1_MSFT_s160_return_constraint: 0.752183 w_k1_MSFT_s16 - w_k1_MSFT_s160
                                 + 0.99 b_k1_MSFT_s160 - 1.01 s_k1_MSFT_s160
                                  = 0
 w_k1_MSFT_s161_return_constraint: 1.04254 w_k1_MSFT_s16 - w_k1_MSFT_s161
                                 + 0.99 b_k1_MSFT_s161 - 1.01 s_k1_MSFT_s161
                                  = 0
 w_k1_MSFT_s162_return_constraint: 1.26693 w_k1_MSFT_s16 - w_k1_MSFT_s162
                                 + 0.99 b_k1_MSFT_s162 - 1.01 s_k1_MSFT_s162
                                  = 0
 w_k1_MSFT_s163_return_constraint: 0.944312 w_k1_MSFT_s16 - w_k1_MSFT_s163
                                 + 0.99 b_k1_MSFT_s163 - 1.01 s_k1_MSFT_s163
                                  = 0
 w_k1_MSFT_s164_return_constraint: 1.43455 w_k1_MSFT_s16 - w_k1_MSFT_s164
                                 + 0.99 b_k1_MSFT_s164 - 1.01 s_k1_MSFT_s164
                                  = 0
 w_k1_MSFT_s165_return_constraint: 1.0785 w_k1_MSFT_s16 - w_k1_MSFT_s165
                                 + 0.99 b_k1_MSFT_s165 - 1.01 s_k1_MSFT_s165
                                  = 0
 w_k1_MSFT_s166_return_constraint: 0.941818 w_k1_MSFT_s16 - w_k1_MSFT_s166
                                 + 0.99 b_k1_MSFT_s166 - 1.01 s_k1_MSFT_s166
                                  = 0
 w_k1_MSFT_s167_return_constraint: 1.22511 w_k1_MSFT_s16 - w_k1_MSFT_s167
                                 + 0.99 b_k1_MSFT_s167 - 1.01 s_k1_MSFT_s167
                                  = 0
 w_k1_MSFT_s170_return_constraint: 0.857508 w_k1_MSFT_s17 - w_k1_MSFT_s170
                                 + 0.99 b_k1_MSFT_s170 - 1.01 s_k1_MSFT_s170
                                  = 0
 w_k1_MSFT_s171_return_constraint: 1.05647 w_k1_MSFT_s17 - w_k1_MSFT_s171
                                 + 0.99 b_k1_MSFT_s171 - 1.01 s_k1_MSFT_s171
                                  = 0
 w_k1_MSFT_s172_return_constraint: 1.24605 w_k1_MSFT_s17 - w_k1_MSFT_s172
                                 + 0.99 b_k1_MSFT_s172 - 1.01 s_k1_MSFT_s172
                                  = 0
 w_k1_MSFT_s173_return_constraint: 0.871908 w_k1_MSFT_s17 - w_k1_MSFT_s173
                                 + 0.99 b_k1_MSFT_s173 - 1.01 s_k1_MSFT_s173
                                  = 0
 w_k1_MSFT_s174_return_constraint: 0.743471 w_k1_MSFT_s17 - w_k1_MSFT_s174
                                 + 0.99 b_k1_MSFT_s174 - 1.01 s_k1_MSFT_s174
                                  = 0
 w_k1_MSFT_s175_return_constraint: 1.12709 w_k1_MSFT_s17 - w_k1_MSFT_s175
                                 + 0.99 b_k1_MSFT_s175 - 1.01 s_k1_MSFT_s175
                                  = 0
 w_k1_MSFT_s176_return_constraint: 1.12834 w_k1_MSFT_s17 - w_k1_MSFT_s176
                                 + 0.99 b_k1_MSFT_s176 - 1.01 s_k1_MSFT_s176
                                  = 0
 w_k1_MSFT_s177_return_constraint: 0.932301 w_k1_MSFT_s17 - w_k1_MSFT_s177
                                 + 0.99 b_k1_MSFT_s177 - 1.01 s_k1_MSFT_s177
                                  = 0
 w_k1_PG_s0_return_constraint:   1.13929 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.960103 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.945143 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.01889 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s02_return_constraint:  1.04003 w_k1_PG_s0 - w_k1_PG_s02
                                 + 0.99 b_k1_PG_s02 - 1.01 s_k1_PG_s02  = 0
 w_k1_PG_s03_return_constraint:  0.918323 w_k1_PG_s0 - w_k1_PG_s03
                                 + 0.99 b_k1_PG_s03 - 1.01 s_k1_PG_s03  = 0
 w_k1_PG_s04_return_constraint:  0.811814 w_k1_PG_s0 - w_k1_PG_s04
                                 + 0.99 b_k1_PG_s04 - 1.01 s_k1_PG_s04  = 0
 w_k1_PG_s05_return_constraint:  1.04435 w_k1_PG_s0 - w_k1_PG_s05
                                 + 0.99 b_k1_PG_s05 - 1.01 s_k1_PG_s05  = 0
 w_k1_PG_s06_return_constraint:  0.862595 w_k1_PG_s0 - w_k1_PG_s06
                                 + 0.99 b_k1_PG_s06 - 1.01 s_k1_PG_s06  = 0
 w_k1_PG_s07_return_constraint:  1.18285 w_k1_PG_s0 - w_k1_PG_s07
                                 + 0.99 b_k1_PG_s07 - 1.01 s_k1_PG_s07  = 0
 w_k1_PG_s10_return_constraint:  0.810709 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.12394 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s12_return_constraint:  1.26465 w_k1_PG_s1 - w_k1_PG_s12
                                 + 0.99 b_k1_PG_s12 - 1.01 s_k1_PG_s12  = 0
 w_k1_PG_s13_return_constraint:  1.12257 w_k1_PG_s1 - w_k1_PG_s13
                                 + 0.99 b_k1_PG_s13 - 1.01 s_k1_PG_s13  = 0
 w_k1_PG_s14_return_constraint:  0.977987 w_k1_PG_s1 - w_k1_PG_s14
                                 + 0.99 b_k1_PG_s14 - 1.01 s_k1_PG_s14  = 0
 w_k1_PG_s15_return_constraint:  0.997322 w_k1_PG_s1 - w_k1_PG_s15
                                 + 0.99 b_k1_PG_s15 - 1.01 s_k1_PG_s15  = 0
 w_k1_PG_s16_return_constraint:  0.876278 w_k1_PG_s1 - w_k1_PG_s16
                                 + 0.99 b_k1_PG_s16 - 1.01 s_k1_PG_s16  = 0
 w_k1_PG_s17_return_constraint:  0.844743 w_k1_PG_s1 - w_k1_PG_s17
                                 + 0.99 b_k1_PG_s17 - 1.01 s_k1_PG_s17  = 0
 w_k1_PG_s000_return_constraint: 1.04935 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.15772 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.939369 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.04712 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.940542 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.916925 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.05175 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.842491 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.868739 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.940443 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.24601 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.05988 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.905471 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.12486 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.961196 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.19237 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s020_return_constraint: 0.83197 w_k1_PG_s02 - w_k1_PG_s020
                                 + 0.99 b_k1_PG_s020 - 1.01 s_k1_PG_s020  = 0
 w_k1_PG_s021_return_constraint: 1.02541 w_k1_PG_s02 - w_k1_PG_s021
                                 + 0.99 b_k1_PG_s021 - 1.01 s_k1_PG_s021  = 0
 w_k1_PG_s022_return_constraint: 0.904471 w_k1_PG_s02 - w_k1_PG_s022
                                 + 0.99 b_k1_PG_s022 - 1.01 s_k1_PG_s022  = 0
 w_k1_PG_s023_return_constraint: 0.836091 w_k1_PG_s02 - w_k1_PG_s023
                                 + 0.99 b_k1_PG_s023 - 1.01 s_k1_PG_s023  = 0
 w_k1_PG_s024_return_constraint: 0.963289 w_k1_PG_s02 - w_k1_PG_s024
                                 + 0.99 b_k1_PG_s024 - 1.01 s_k1_PG_s024  = 0
 w_k1_PG_s025_return_constraint: 0.86064 w_k1_PG_s02 - w_k1_PG_s025
                                 + 0.99 b_k1_PG_s025 - 1.01 s_k1_PG_s025  = 0
 w_k1_PG_s026_return_constraint: 0.888861 w_k1_PG_s02 - w_k1_PG_s026
                                 + 0.99 b_k1_PG_s026 - 1.01 s_k1_PG_s026  = 0
 w_k1_PG_s027_return_constraint: 0.898979 w_k1_PG_s02 - w_k1_PG_s027
                                 + 0.99 b_k1_PG_s027 - 1.01 s_k1_PG_s027  = 0
 w_k1_PG_s030_return_constraint: 1.03232 w_k1_PG_s03 - w_k1_PG_s030
                                 + 0.99 b_k1_PG_s030 - 1.01 s_k1_PG_s030  = 0
 w_k1_PG_s031_return_constraint: 1.33942 w_k1_PG_s03 - w_k1_PG_s031
                                 + 0.99 b_k1_PG_s031 - 1.01 s_k1_PG_s031  = 0
 w_k1_PG_s032_return_constraint: 0.824119 w_k1_PG_s03 - w_k1_PG_s032
                                 + 0.99 b_k1_PG_s032 - 1.01 s_k1_PG_s032  = 0
 w_k1_PG_s033_return_constraint: 0.938257 w_k1_PG_s03 - w_k1_PG_s033
                                 + 0.99 b_k1_PG_s033 - 1.01 s_k1_PG_s033  = 0
 w_k1_PG_s034_return_constraint: 1.0217 w_k1_PG_s03 - w_k1_PG_s034
                                 + 0.99 b_k1_PG_s034 - 1.01 s_k1_PG_s034  = 0
 w_k1_PG_s035_return_constraint: 1.18707 w_k1_PG_s03 - w_k1_PG_s035
                                 + 0.99 b_k1_PG_s035 - 1.01 s_k1_PG_s035  = 0
 w_k1_PG_s036_return_constraint: 0.960508 w_k1_PG_s03 - w_k1_PG_s036
                                 + 0.99 b_k1_PG_s036 - 1.01 s_k1_PG_s036  = 0
 w_k1_PG_s037_return_constraint: 1.02923 w_k1_PG_s03 - w_k1_PG_s037
                                 + 0.99 b_k1_PG_s037 - 1.01 s_k1_PG_s037  = 0
 w_k1_PG_s040_return_constraint: 1.2583 w_k1_PG_s04 - w_k1_PG_s040
                                 + 0.99 b_k1_PG_s040 - 1.01 s_k1_PG_s040  = 0
 w_k1_PG_s041_return_constraint: 0.929503 w_k1_PG_s04 - w_k1_PG_s041
                                 + 0.99 b_k1_PG_s041 - 1.01 s_k1_PG_s041  = 0
 w_k1_PG_s042_return_constraint: 1.09012 w_k1_PG_s04 - w_k1_PG_s042
                                 + 0.99 b_k1_PG_s042 - 1.01 s_k1_PG_s042  = 0
 w_k1_PG_s043_return_constraint: 0.981394 w_k1_PG_s04 - w_k1_PG_s043
                                 + 0.99 b_k1_PG_s043 - 1.01 s_k1_PG_s043  = 0
 w_k1_PG_s044_return_constraint: 1.05406 w_k1_PG_s04 - w_k1_PG_s044
                                 + 0.99 b_k1_PG_s044 - 1.01 s_k1_PG_s044  = 0
 w_k1_PG_s045_return_constraint: 0.86158 w_k1_PG_s04 - w_k1_PG_s045
                                 + 0.99 b_k1_PG_s045 - 1.01 s_k1_PG_s045  = 0
 w_k1_PG_s046_return_constraint: 0.771949 w_k1_PG_s04 - w_k1_PG_s046
                                 + 0.99 b_k1_PG_s046 - 1.01 s_k1_PG_s046  = 0
 w_k1_PG_s047_return_constraint: 0.882761 w_k1_PG_s04 - w_k1_PG_s047
                                 + 0.99 b_k1_PG_s047 - 1.01 s_k1_PG_s047  = 0
 w_k1_PG_s050_return_constraint: 0.693886 w_k1_PG_s05 - w_k1_PG_s050
                                 + 0.99 b_k1_PG_s050 - 1.01 s_k1_PG_s050  = 0
 w_k1_PG_s051_return_constraint: 1.02088 w_k1_PG_s05 - w_k1_PG_s051
                                 + 0.99 b_k1_PG_s051 - 1.01 s_k1_PG_s051  = 0
 w_k1_PG_s052_return_constraint: 0.886052 w_k1_PG_s05 - w_k1_PG_s052
                                 + 0.99 b_k1_PG_s052 - 1.01 s_k1_PG_s052  = 0
 w_k1_PG_s053_return_constraint: 1.07517 w_k1_PG_s05 - w_k1_PG_s053
                                 + 0.99 b_k1_PG_s053 - 1.01 s_k1_PG_s053  = 0
 w_k1_PG_s054_return_constraint: 1.02988 w_k1_PG_s05 - w_k1_PG_s054
                                 + 0.99 b_k1_PG_s054 - 1.01 s_k1_PG_s054  = 0
 w_k1_PG_s055_return_constraint: 0.929434 w_k1_PG_s05 - w_k1_PG_s055
                                 + 0.99 b_k1_PG_s055 - 1.01 s_k1_PG_s055  = 0
 w_k1_PG_s056_return_constraint: 1.30701 w_k1_PG_s05 - w_k1_PG_s056
                                 + 0.99 b_k1_PG_s056 - 1.01 s_k1_PG_s056  = 0
 w_k1_PG_s057_return_constraint: 1.00949 w_k1_PG_s05 - w_k1_PG_s057
                                 + 0.99 b_k1_PG_s057 - 1.01 s_k1_PG_s057  = 0
 w_k1_PG_s060_return_constraint: 1.03898 w_k1_PG_s06 - w_k1_PG_s060
                                 + 0.99 b_k1_PG_s060 - 1.01 s_k1_PG_s060  = 0
 w_k1_PG_s061_return_constraint: 0.950706 w_k1_PG_s06 - w_k1_PG_s061
                                 + 0.99 b_k1_PG_s061 - 1.01 s_k1_PG_s061  = 0
 w_k1_PG_s062_return_constraint: 0.861869 w_k1_PG_s06 - w_k1_PG_s062
                                 + 0.99 b_k1_PG_s062 - 1.01 s_k1_PG_s062  = 0
 w_k1_PG_s063_return_constraint: 0.785687 w_k1_PG_s06 - w_k1_PG_s063
                                 + 0.99 b_k1_PG_s063 - 1.01 s_k1_PG_s063  = 0
 w_k1_PG_s064_return_constraint: 0.937772 w_k1_PG_s06 - w_k1_PG_s064
                                 + 0.99 b_k1_PG_s064 - 1.01 s_k1_PG_s064  = 0
 w_k1_PG_s065_return_constraint: 0.870211 w_k1_PG_s06 - w_k1_PG_s065
                                 + 0.99 b_k1_PG_s065 - 1.01 s_k1_PG_s065  = 0
 w_k1_PG_s066_return_constraint: 1.19656 w_k1_PG_s06 - w_k1_PG_s066
                                 + 0.99 b_k1_PG_s066 - 1.01 s_k1_PG_s066  = 0
 w_k1_PG_s067_return_constraint: 1.19465 w_k1_PG_s06 - w_k1_PG_s067
                                 + 0.99 b_k1_PG_s067 - 1.01 s_k1_PG_s067  = 0
 w_k1_PG_s070_return_constraint: 0.965283 w_k1_PG_s07 - w_k1_PG_s070
                                 + 0.99 b_k1_PG_s070 - 1.01 s_k1_PG_s070  = 0
 w_k1_PG_s071_return_constraint: 1.36532 w_k1_PG_s07 - w_k1_PG_s071
                                 + 0.99 b_k1_PG_s071 - 1.01 s_k1_PG_s071  = 0
 w_k1_PG_s072_return_constraint: 0.960717 w_k1_PG_s07 - w_k1_PG_s072
                                 + 0.99 b_k1_PG_s072 - 1.01 s_k1_PG_s072  = 0
 w_k1_PG_s073_return_constraint: 1.05726 w_k1_PG_s07 - w_k1_PG_s073
                                 + 0.99 b_k1_PG_s073 - 1.01 s_k1_PG_s073  = 0
 w_k1_PG_s074_return_constraint: 1.14004 w_k1_PG_s07 - w_k1_PG_s074
                                 + 0.99 b_k1_PG_s074 - 1.01 s_k1_PG_s074  = 0
 w_k1_PG_s075_return_constraint: 0.835955 w_k1_PG_s07 - w_k1_PG_s075
                                 + 0.99 b_k1_PG_s075 - 1.01 s_k1_PG_s075  = 0
 w_k1_PG_s076_return_constraint: 0.838494 w_k1_PG_s07 - w_k1_PG_s076
                                 + 0.99 b_k1_PG_s076 - 1.01 s_k1_PG_s076  = 0
 w_k1_PG_s077_return_constraint: 1.08503 w_k1_PG_s07 - w_k1_PG_s077
                                 + 0.99 b_k1_PG_s077 - 1.01 s_k1_PG_s077  = 0
 w_k1_PG_s100_return_constraint: 1.1169 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.08098 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.861626 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.868027 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.00165 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.921804 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.15378 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.926613 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.02251 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.14557 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.11243 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.930096 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.962395 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.945199 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.42827 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.16351 w_k1_PG_s11 - w_k1_PG_s117
                                 + 0.99 b_k1_PG_s117 - 1.01 s_k1_PG_s117  = 0
 w_k1_PG_s120_return_constraint: 1.23585 w_k1_PG_s12 - w_k1_PG_s120
                                 + 0.99 b_k1_PG_s120 - 1.01 s_k1_PG_s120  = 0
 w_k1_PG_s121_return_constraint: 0.847392 w_k1_PG_s12 - w_k1_PG_s121
                                 + 0.99 b_k1_PG_s121 - 1.01 s_k1_PG_s121  = 0
 w_k1_PG_s122_return_constraint: 0.937201 w_k1_PG_s12 - w_k1_PG_s122
                                 + 0.99 b_k1_PG_s122 - 1.01 s_k1_PG_s122  = 0
 w_k1_PG_s123_return_constraint: 1.21232 w_k1_PG_s12 - w_k1_PG_s123
                                 + 0.99 b_k1_PG_s123 - 1.01 s_k1_PG_s123  = 0
 w_k1_PG_s124_return_constraint: 0.997105 w_k1_PG_s12 - w_k1_PG_s124
                                 + 0.99 b_k1_PG_s124 - 1.01 s_k1_PG_s124  = 0
 w_k1_PG_s125_return_constraint: 0.981179 w_k1_PG_s12 - w_k1_PG_s125
                                 + 0.99 b_k1_PG_s125 - 1.01 s_k1_PG_s125  = 0
 w_k1_PG_s126_return_constraint: 0.851937 w_k1_PG_s12 - w_k1_PG_s126
                                 + 0.99 b_k1_PG_s126 - 1.01 s_k1_PG_s126  = 0
 w_k1_PG_s127_return_constraint: 1.06347 w_k1_PG_s12 - w_k1_PG_s127
                                 + 0.99 b_k1_PG_s127 - 1.01 s_k1_PG_s127  = 0
 w_k1_PG_s130_return_constraint: 0.977042 w_k1_PG_s13 - w_k1_PG_s130
                                 + 0.99 b_k1_PG_s130 - 1.01 s_k1_PG_s130  = 0
 w_k1_PG_s131_return_constraint: 1.23692 w_k1_PG_s13 - w_k1_PG_s131
                                 + 0.99 b_k1_PG_s131 - 1.01 s_k1_PG_s131  = 0
 w_k1_PG_s132_return_constraint: 0.82 w_k1_PG_s13 - w_k1_PG_s132
                                 + 0.99 b_k1_PG_s132 - 1.01 s_k1_PG_s132  = 0
 w_k1_PG_s133_return_constraint: 0.766962 w_k1_PG_s13 - w_k1_PG_s133
                                 + 0.99 b_k1_PG_s133 - 1.01 s_k1_PG_s133  = 0
 w_k1_PG_s134_return_constraint: 0.964425 w_k1_PG_s13 - w_k1_PG_s134
                                 + 0.99 b_k1_PG_s134 - 1.01 s_k1_PG_s134  = 0
 w_k1_PG_s135_return_constraint: 1.14752 w_k1_PG_s13 - w_k1_PG_s135
                                 + 0.99 b_k1_PG_s135 - 1.01 s_k1_PG_s135  = 0
 w_k1_PG_s136_return_constraint: 0.887466 w_k1_PG_s13 - w_k1_PG_s136
                                 + 0.99 b_k1_PG_s136 - 1.01 s_k1_PG_s136  = 0
 w_k1_PG_s137_return_constraint: 0.834791 w_k1_PG_s13 - w_k1_PG_s137
                                 + 0.99 b_k1_PG_s137 - 1.01 s_k1_PG_s137  = 0
 w_k1_PG_s140_return_constraint: 1.01763 w_k1_PG_s14 - w_k1_PG_s140
                                 + 0.99 b_k1_PG_s140 - 1.01 s_k1_PG_s140  = 0
 w_k1_PG_s141_return_constraint: 1.13087 w_k1_PG_s14 - w_k1_PG_s141
                                 + 0.99 b_k1_PG_s141 - 1.01 s_k1_PG_s141  = 0
 w_k1_PG_s142_return_constraint: 1.08011 w_k1_PG_s14 - w_k1_PG_s142
                                 + 0.99 b_k1_PG_s142 - 1.01 s_k1_PG_s142  = 0
 w_k1_PG_s143_return_constraint: 1.03439 w_k1_PG_s14 - w_k1_PG_s143
                                 + 0.99 b_k1_PG_s143 - 1.01 s_k1_PG_s143  = 0
 w_k1_PG_s144_return_constraint: 1.33751 w_k1_PG_s14 - w_k1_PG_s144
                                 + 0.99 b_k1_PG_s144 - 1.01 s_k1_PG_s144  = 0
 w_k1_PG_s145_return_constraint: 1.28611 w_k1_PG_s14 - w_k1_PG_s145
                                 + 0.99 b_k1_PG_s145 - 1.01 s_k1_PG_s145  = 0
 w_k1_PG_s146_return_constraint: 0.953437 w_k1_PG_s14 - w_k1_PG_s146
                                 + 0.99 b_k1_PG_s146 - 1.01 s_k1_PG_s146  = 0
 w_k1_PG_s147_return_constraint: 1.15085 w_k1_PG_s14 - w_k1_PG_s147
                                 + 0.99 b_k1_PG_s147 - 1.01 s_k1_PG_s147  = 0
 w_k1_PG_s150_return_constraint: 1.10719 w_k1_PG_s15 - w_k1_PG_s150
                                 + 0.99 b_k1_PG_s150 - 1.01 s_k1_PG_s150  = 0
 w_k1_PG_s151_return_constraint: 0.971261 w_k1_PG_s15 - w_k1_PG_s151
                                 + 0.99 b_k1_PG_s151 - 1.01 s_k1_PG_s151  = 0
 w_k1_PG_s152_return_constraint: 1.05519 w_k1_PG_s15 - w_k1_PG_s152
                                 + 0.99 b_k1_PG_s152 - 1.01 s_k1_PG_s152  = 0
 w_k1_PG_s153_return_constraint: 1.082 w_k1_PG_s15 - w_k1_PG_s153
                                 + 0.99 b_k1_PG_s153 - 1.01 s_k1_PG_s153  = 0
 w_k1_PG_s154_return_constraint: 1.14997 w_k1_PG_s15 - w_k1_PG_s154
                                 + 0.99 b_k1_PG_s154 - 1.01 s_k1_PG_s154  = 0
 w_k1_PG_s155_return_constraint: 0.944448 w_k1_PG_s15 - w_k1_PG_s155
                                 + 0.99 b_k1_PG_s155 - 1.01 s_k1_PG_s155  = 0
 w_k1_PG_s156_return_constraint: 0.89706 w_k1_PG_s15 - w_k1_PG_s156
                                 + 0.99 b_k1_PG_s156 - 1.01 s_k1_PG_s156  = 0
 w_k1_PG_s157_return_constraint: 0.79604 w_k1_PG_s15 - w_k1_PG_s157
                                 + 0.99 b_k1_PG_s157 - 1.01 s_k1_PG_s157  = 0
 w_k1_PG_s160_return_constraint: 0.835996 w_k1_PG_s16 - w_k1_PG_s160
                                 + 0.99 b_k1_PG_s160 - 1.01 s_k1_PG_s160  = 0
 w_k1_PG_s161_return_constraint: 1.03907 w_k1_PG_s16 - w_k1_PG_s161
                                 + 0.99 b_k1_PG_s161 - 1.01 s_k1_PG_s161  = 0
 w_k1_PG_s162_return_constraint: 1.14412 w_k1_PG_s16 - w_k1_PG_s162
                                 + 0.99 b_k1_PG_s162 - 1.01 s_k1_PG_s162  = 0
 w_k1_PG_s163_return_constraint: 1.00237 w_k1_PG_s16 - w_k1_PG_s163
                                 + 0.99 b_k1_PG_s163 - 1.01 s_k1_PG_s163  = 0
 w_k1_PG_s164_return_constraint: 1.15096 w_k1_PG_s16 - w_k1_PG_s164
                                 + 0.99 b_k1_PG_s164 - 1.01 s_k1_PG_s164  = 0
 w_k1_PG_s165_return_constraint: 0.984345 w_k1_PG_s16 - w_k1_PG_s165
                                 + 0.99 b_k1_PG_s165 - 1.01 s_k1_PG_s165  = 0
 w_k1_PG_s166_return_constraint: 0.934509 w_k1_PG_s16 - w_k1_PG_s166
                                 + 0.99 b_k1_PG_s166 - 1.01 s_k1_PG_s166  = 0
 w_k1_PG_s167_return_constraint: 1.04105 w_k1_PG_s16 - w_k1_PG_s167
                                 + 0.99 b_k1_PG_s167 - 1.01 s_k1_PG_s167  = 0
 w_k1_PG_s170_return_constraint: 0.86662 w_k1_PG_s17 - w_k1_PG_s170
                                 + 0.99 b_k1_PG_s170 - 1.01 s_k1_PG_s170  = 0
 w_k1_PG_s171_return_constraint: 1.18073 w_k1_PG_s17 - w_k1_PG_s171
                                 + 0.99 b_k1_PG_s171 - 1.01 s_k1_PG_s171  = 0
 w_k1_PG_s172_return_constraint: 1.32145 w_k1_PG_s17 - w_k1_PG_s172
                                 + 0.99 b_k1_PG_s172 - 1.01 s_k1_PG_s172  = 0
 w_k1_PG_s173_return_constraint: 1.11926 w_k1_PG_s17 - w_k1_PG_s173
                                 + 0.99 b_k1_PG_s173 - 1.01 s_k1_PG_s173  = 0
 w_k1_PG_s174_return_constraint: 0.865292 w_k1_PG_s17 - w_k1_PG_s174
                                 + 0.99 b_k1_PG_s174 - 1.01 s_k1_PG_s174  = 0
 w_k1_PG_s175_return_constraint: 1.24375 w_k1_PG_s17 - w_k1_PG_s175
                                 + 0.99 b_k1_PG_s175 - 1.01 s_k1_PG_s175  = 0
 w_k1_PG_s176_return_constraint: 1.15346 w_k1_PG_s17 - w_k1_PG_s176
                                 + 0.99 b_k1_PG_s176 - 1.01 s_k1_PG_s176  = 0
 w_k1_PG_s177_return_constraint: 1.07936 w_k1_PG_s17 - w_k1_PG_s177
                                 + 0.99 b_k1_PG_s177 - 1.01 s_k1_PG_s177  = 0
 k1_s0_balance_constraint:       b_k1_ADM_s0 + b_k1_AXP_s0 + b_k1_DIS_s0
                                 + b_k1_FDX_s0 + b_k1_IBM_s0 + b_k1_INTC_s0
                                 + b_k1_KO_s0 + b_k1_MAT_s0 + b_k1_MSFT_s0
                                 + b_k1_PG_s0 - s_k1_ADM_s0 - s_k1_AXP_s0
                                 - s_k1_DIS_s0 - s_k1_FDX_s0 - s_k1_IBM_s0
                                 - s_k1_INTC_s0 - s_k1_KO_s0 - s_k1_MAT_s0
                                 - s_k1_MSFT_s0 - s_k1_PG_s0  = 0
 k1_s1_balance_constraint:       b_k1_ADM_s1 + b_k1_AXP_s1 + b_k1_DIS_s1
                                 + b_k1_FDX_s1 + b_k1_IBM_s1 + b_k1_INTC_s1
                                 + b_k1_KO_s1 + b_k1_MAT_s1 + b_k1_MSFT_s1
                                 + b_k1_PG_s1 - s_k1_ADM_s1 - s_k1_AXP_s1
                                 - s_k1_DIS_s1 - s_k1_FDX_s1 - s_k1_IBM_s1
                                 - s_k1_INTC_s1 - s_k1_KO_s1 - s_k1_MAT_s1
                                 - s_k1_MSFT_s1 - s_k1_PG_s1  = 0
 k1_s00_balance_constraint:      b_k1_ADM_s00 + b_k1_AXP_s00 + b_k1_DIS_s00
                                 + b_k1_FDX_s00 + b_k1_IBM_s00 + b_k1_INTC_s00
                                 + b_k1_KO_s00 + b_k1_MAT_s00 + b_k1_MSFT_s00
                                 + b_k1_PG_s00 - s_k1_ADM_s00 - s_k1_AXP_s00
                                 - s_k1_DIS_s00 - s_k1_FDX_s00 - s_k1_IBM_s00
                                 - s_k1_INTC_s00 - s_k1_KO_s00 - s_k1_MAT_s00
                                 - s_k1_MSFT_s00 - s_k1_PG_s00  = 0
 k1_s01_balance_constraint:      b_k1_ADM_s01 + b_k1_AXP_s01 + b_k1_DIS_s01
                                 + b_k1_FDX_s01 + b_k1_IBM_s01 + b_k1_INTC_s01
                                 + b_k1_KO_s01 + b_k1_MAT_s01 + b_k1_MSFT_s01
                                 + b_k1_PG_s01 - s_k1_ADM_s01 - s_k1_AXP_s01
                                 - s_k1_DIS_s01 - s_k1_FDX_s01 - s_k1_IBM_s01
                                 - s_k1_INTC_s01 - s_k1_KO_s01 - s_k1_MAT_s01
                                 - s_k1_MSFT_s01 - s_k1_PG_s01  = 0
 k1_s02_balance_constraint:      b_k1_ADM_s02 + b_k1_AXP_s02 + b_k1_DIS_s02
                                 + b_k1_FDX_s02 + b_k1_IBM_s02 + b_k1_INTC_s02
                                 + b_k1_KO_s02 + b_k1_MAT_s02 + b_k1_MSFT_s02
                                 + b_k1_PG_s02 - s_k1_ADM_s02 - s_k1_AXP_s02
                                 - s_k1_DIS_s02 - s_k1_FDX_s02 - s_k1_IBM_s02
                                 - s_k1_INTC_s02 - s_k1_KO_s02 - s_k1_MAT_s02
                                 - s_k1_MSFT_s02 - s_k1_PG_s02  = 0
 k1_s03_balance_constraint:      b_k1_ADM_s03 + b_k1_AXP_s03 + b_k1_DIS_s03
                                 + b_k1_FDX_s03 + b_k1_IBM_s03 + b_k1_INTC_s03
                                 + b_k1_KO_s03 + b_k1_MAT_s03 + b_k1_MSFT_s03
                                 + b_k1_PG_s03 - s_k1_ADM_s03 - s_k1_AXP_s03
                                 - s_k1_DIS_s03 - s_k1_FDX_s03 - s_k1_IBM_s03
                                 - s_k1_INTC_s03 - s_k1_KO_s03 - s_k1_MAT_s03
                                 - s_k1_MSFT_s03 - s_k1_PG_s03  = 0
 k1_s04_balance_constraint:      b_k1_ADM_s04 + b_k1_AXP_s04 + b_k1_DIS_s04
                                 + b_k1_FDX_s04 + b_k1_IBM_s04 + b_k1_INTC_s04
                                 + b_k1_KO_s04 + b_k1_MAT_s04 + b_k1_MSFT_s04
                                 + b_k1_PG_s04 - s_k1_ADM_s04 - s_k1_AXP_s04
                                 - s_k1_DIS_s04 - s_k1_FDX_s04 - s_k1_IBM_s04
                                 - s_k1_INTC_s04 - s_k1_KO_s04 - s_k1_MAT_s04
                                 - s_k1_MSFT_s04 - s_k1_PG_s04  = 0
 k1_s05_balance_constraint:      b_k1_ADM_s05 + b_k1_AXP_s05 + b_k1_DIS_s05
                                 + b_k1_FDX_s05 + b_k1_IBM_s05 + b_k1_INTC_s05
                                 + b_k1_KO_s05 + b_k1_MAT_s05 + b_k1_MSFT_s05
                                 + b_k1_PG_s05 - s_k1_ADM_s05 - s_k1_AXP_s05
                                 - s_k1_DIS_s05 - s_k1_FDX_s05 - s_k1_IBM_s05
                                 - s_k1_INTC_s05 - s_k1_KO_s05 - s_k1_MAT_s05
                                 - s_k1_MSFT_s05 - s_k1_PG_s05  = 0
 k1_s06_balance_constraint:      b_k1_ADM_s06 + b_k1_AXP_s06 + b_k1_DIS_s06
                                 + b_k1_FDX_s06 + b_k1_IBM_s06 + b_k1_INTC_s06
                                 + b_k1_KO_s06 + b_k1_MAT_s06 + b_k1_MSFT_s06
                                 + b_k1_PG_s06 - s_k1_ADM_s06 - s_k1_AXP_s06
                                 - s_k1_DIS_s06 - s_k1_FDX_s06 - s_k1_IBM_s06
                                 - s_k1_INTC_s06 - s_k1_KO_s06 - s_k1_MAT_s06
                                 - s_k1_MSFT_s06 - s_k1_PG_s06  = 0
 k1_s07_balance_constraint:      b_k1_ADM_s07 + b_k1_AXP_s07 + b_k1_DIS_s07
                                 + b_k1_FDX_s07 + b_k1_IBM_s07 + b_k1_INTC_s07
                                 + b_k1_KO_s07 + b_k1_MAT_s07 + b_k1_MSFT_s07
                                 + b_k1_PG_s07 - s_k1_ADM_s07 - s_k1_AXP_s07
                                 - s_k1_DIS_s07 - s_k1_FDX_s07 - s_k1_IBM_s07
                                 - s_k1_INTC_s07 - s_k1_KO_s07 - s_k1_MAT_s07
                                 - s_k1_MSFT_s07 - s_k1_PG_s07  = 0
 k1_s10_balance_constraint:      b_k1_ADM_s10 + b_k1_AXP_s10 + b_k1_DIS_s10
                                 + b_k1_FDX_s10 + b_k1_IBM_s10 + b_k1_INTC_s10
                                 + b_k1_KO_s10 + b_k1_MAT_s10 + b_k1_MSFT_s10
                                 + b_k1_PG_s10 - s_k1_ADM_s10 - s_k1_AXP_s10
                                 - s_k1_DIS_s10 - s_k1_FDX_s10 - s_k1_IBM_s10
                                 - s_k1_INTC_s10 - s_k1_KO_s10 - s_k1_MAT_s10
                                 - s_k1_MSFT_s10 - s_k1_PG_s10  = 0
 k1_s11_balance_constraint:      b_k1_ADM_s11 + b_k1_AXP_s11 + b_k1_DIS_s11
                                 + b_k1_FDX_s11 + b_k1_IBM_s11 + b_k1_INTC_s11
                                 + b_k1_KO_s11 + b_k1_MAT_s11 + b_k1_MSFT_s11
                                 + b_k1_PG_s11 - s_k1_ADM_s11 - s_k1_AXP_s11
                                 - s_k1_DIS_s11 - s_k1_FDX_s11 - s_k1_IBM_s11
                                 - s_k1_INTC_s11 - s_k1_KO_s11 - s_k1_MAT_s11
                                 - s_k1_MSFT_s11 - s_k1_PG_s11  = 0
 k1_s12_balance_constraint:      b_k1_ADM_s12 + b_k1_AXP_s12 + b_k1_DIS_s12
                                 + b_k1_FDX_s12 + b_k1_IBM_s12 + b_k1_INTC_s12
                                 + b_k1_KO_s12 + b_k1_MAT_s12 + b_k1_MSFT_s12
                                 + b_k1_PG_s12 - s_k1_ADM_s12 - s_k1_AXP_s12
                                 - s_k1_DIS_s12 - s_k1_FDX_s12 - s_k1_IBM_s12
                                 - s_k1_INTC_s12 - s_k1_KO_s12 - s_k1_MAT_s12
                                 - s_k1_MSFT_s12 - s_k1_PG_s12  = 0
 k1_s13_balance_constraint:      b_k1_ADM_s13 + b_k1_AXP_s13 + b_k1_DIS_s13
                                 + b_k1_FDX_s13 + b_k1_IBM_s13 + b_k1_INTC_s13
                                 + b_k1_KO_s13 + b_k1_MAT_s13 + b_k1_MSFT_s13
                                 + b_k1_PG_s13 - s_k1_ADM_s13 - s_k1_AXP_s13
                                 - s_k1_DIS_s13 - s_k1_FDX_s13 - s_k1_IBM_s13
                                 - s_k1_INTC_s13 - s_k1_KO_s13 - s_k1_MAT_s13
                                 - s_k1_MSFT_s13 - s_k1_PG_s13  = 0
 k1_s14_balance_constraint:      b_k1_ADM_s14 + b_k1_AXP_s14 + b_k1_DIS_s14
                                 + b_k1_FDX_s14 + b_k1_IBM_s14 + b_k1_INTC_s14
                                 + b_k1_KO_s14 + b_k1_MAT_s14 + b_k1_MSFT_s14
                                 + b_k1_PG_s14 - s_k1_ADM_s14 - s_k1_AXP_s14
                                 - s_k1_DIS_s14 - s_k1_FDX_s14 - s_k1_IBM_s14
                                 - s_k1_INTC_s14 - s_k1_KO_s14 - s_k1_MAT_s14
                                 - s_k1_MSFT_s14 - s_k1_PG_s14  = 0
 k1_s15_balance_constraint:      b_k1_ADM_s15 + b_k1_AXP_s15 + b_k1_DIS_s15
                                 + b_k1_FDX_s15 + b_k1_IBM_s15 + b_k1_INTC_s15
                                 + b_k1_KO_s15 + b_k1_MAT_s15 + b_k1_MSFT_s15
                                 + b_k1_PG_s15 - s_k1_ADM_s15 - s_k1_AXP_s15
                                 - s_k1_DIS_s15 - s_k1_FDX_s15 - s_k1_IBM_s15
                                 - s_k1_INTC_s15 - s_k1_KO_s15 - s_k1_MAT_s15
                                 - s_k1_MSFT_s15 - s_k1_PG_s15  = 0
 k1_s16_balance_constraint:      b_k1_ADM_s16 + b_k1_AXP_s16 + b_k1_DIS_s16
                                 + b_k1_FDX_s16 + b_k1_IBM_s16 + b_k1_INTC_s16
                                 + b_k1_KO_s16 + b_k1_MAT_s16 + b_k1_MSFT_s16
                                 + b_k1_PG_s16 - s_k1_ADM_s16 - s_k1_AXP_s16
                                 - s_k1_DIS_s16 - s_k1_FDX_s16 - s_k1_IBM_s16
                                 - s_k1_INTC_s16 - s_k1_KO_s16 - s_k1_MAT_s16
                                 - s_k1_MSFT_s16 - s_k1_PG_s16  = 0
 k1_s17_balance_constraint:      b_k1_ADM_s17 + b_k1_AXP_s17 + b_k1_DIS_s17
                                 + b_k1_FDX_s17 + b_k1_IBM_s17 + b_k1_INTC_s17
                                 + b_k1_KO_s17 + b_k1_MAT_s17 + b_k1_MSFT_s17
                                 + b_k1_PG_s17 - s_k1_ADM_s17 - s_k1_AXP_s17
                                 - s_k1_DIS_s17 - s_k1_FDX_s17 - s_k1_IBM_s17
                                 - s_k1_INTC_s17 - s_k1_KO_s17 - s_k1_MAT_s17
                                 - s_k1_MSFT_s17 - s_k1_PG_s17  = 0
 k1_s000_balance_constraint:     b_k1_ADM_s000 + b_k1_AXP_s000 + b_k1_DIS_s000
                                 + b_k1_FDX_s000 + b_k1_IBM_s000
                                 + b_k1_INTC_s000 + b_k1_KO_s000
                                 + b_k1_MAT_s000 + b_k1_MSFT_s000
                                 + b_k1_PG_s000 - s_k1_ADM_s000 - s_k1_AXP_s000
                                 - s_k1_DIS_s000 - s_k1_FDX_s000
                                 - s_k1_IBM_s000 - s_k1_INTC_s000
                                 - s_k1_KO_s000 - s_k1_MAT_s000
                                 - s_k1_MSFT_s000 - s_k1_PG_s000  = 0
 k1_s001_balance_constraint:     b_k1_ADM_s001 + b_k1_AXP_s001 + b_k1_DIS_s001
                                 + b_k1_FDX_s001 + b_k1_IBM_s001
                                 + b_k1_INTC_s001 + b_k1_KO_s001
                                 + b_k1_MAT_s001 + b_k1_MSFT_s001
                                 + b_k1_PG_s001 - s_k1_ADM_s001 - s_k1_AXP_s001
                                 - s_k1_DIS_s001 - s_k1_FDX_s001
                                 - s_k1_IBM_s001 - s_k1_INTC_s001
                                 - s_k1_KO_s001 - s_k1_MAT_s001
                                 - s_k1_MSFT_s001 - s_k1_PG_s001  = 0
 k1_s002_balance_constraint:     b_k1_ADM_s002 + b_k1_AXP_s002 + b_k1_DIS_s002
                                 + b_k1_FDX_s002 + b_k1_IBM_s002
                                 + b_k1_INTC_s002 + b_k1_KO_s002
                                 + b_k1_MAT_s002 + b_k1_MSFT_s002
                                 + b_k1_PG_s002 - s_k1_ADM_s002 - s_k1_AXP_s002
                                 - s_k1_DIS_s002 - s_k1_FDX_s002
                                 - s_k1_IBM_s002 - s_k1_INTC_s002
                                 - s_k1_KO_s002 - s_k1_MAT_s002
                                 - s_k1_MSFT_s002 - s_k1_PG_s002  = 0
 k1_s003_balance_constraint:     b_k1_ADM_s003 + b_k1_AXP_s003 + b_k1_DIS_s003
                                 + b_k1_FDX_s003 + b_k1_IBM_s003
                                 + b_k1_INTC_s003 + b_k1_KO_s003
                                 + b_k1_MAT_s003 + b_k1_MSFT_s003
                                 + b_k1_PG_s003 - s_k1_ADM_s003 - s_k1_AXP_s003
                                 - s_k1_DIS_s003 - s_k1_FDX_s003
                                 - s_k1_IBM_s003 - s_k1_INTC_s003
                                 - s_k1_KO_s003 - s_k1_MAT_s003
                                 - s_k1_MSFT_s003 - s_k1_PG_s003  = 0
 k1_s004_balance_constraint:     b_k1_ADM_s004 + b_k1_AXP_s004 + b_k1_DIS_s004
                                 + b_k1_FDX_s004 + b_k1_IBM_s004
                                 + b_k1_INTC_s004 + b_k1_KO_s004
                                 + b_k1_MAT_s004 + b_k1_MSFT_s004
                                 + b_k1_PG_s004 - s_k1_ADM_s004 - s_k1_AXP_s004
                                 - s_k1_DIS_s004 - s_k1_FDX_s004
                                 - s_k1_IBM_s004 - s_k1_INTC_s004
                                 - s_k1_KO_s004 - s_k1_MAT_s004
                                 - s_k1_MSFT_s004 - s_k1_PG_s004  = 0
 k1_s005_balance_constraint:     b_k1_ADM_s005 + b_k1_AXP_s005 + b_k1_DIS_s005
                                 + b_k1_FDX_s005 + b_k1_IBM_s005
                                 + b_k1_INTC_s005 + b_k1_KO_s005
                                 + b_k1_MAT_s005 + b_k1_MSFT_s005
                                 + b_k1_PG_s005 - s_k1_ADM_s005 - s_k1_AXP_s005
                                 - s_k1_DIS_s005 - s_k1_FDX_s005
                                 - s_k1_IBM_s005 - s_k1_INTC_s005
                                 - s_k1_KO_s005 - s_k1_MAT_s005
                                 - s_k1_MSFT_s005 - s_k1_PG_s005  = 0
 k1_s006_balance_constraint:     b_k1_ADM_s006 + b_k1_AXP_s006 + b_k1_DIS_s006
                                 + b_k1_FDX_s006 + b_k1_IBM_s006
                                 + b_k1_INTC_s006 + b_k1_KO_s006
                                 + b_k1_MAT_s006 + b_k1_MSFT_s006
                                 + b_k1_PG_s006 - s_k1_ADM_s006 - s_k1_AXP_s006
                                 - s_k1_DIS_s006 - s_k1_FDX_s006
                                 - s_k1_IBM_s006 - s_k1_INTC_s006
                                 - s_k1_KO_s006 - s_k1_MAT_s006
                                 - s_k1_MSFT_s006 - s_k1_PG_s006  = 0
 k1_s007_balance_constraint:     b_k1_ADM_s007 + b_k1_AXP_s007 + b_k1_DIS_s007
                                 + b_k1_FDX_s007 + b_k1_IBM_s007
                                 + b_k1_INTC_s007 + b_k1_KO_s007
                                 + b_k1_MAT_s007 + b_k1_MSFT_s007
                                 + b_k1_PG_s007 - s_k1_ADM_s007 - s_k1_AXP_s007
                                 - s_k1_DIS_s007 - s_k1_FDX_s007
                                 - s_k1_IBM_s007 - s_k1_INTC_s007
                                 - s_k1_KO_s007 - s_k1_MAT_s007
                                 - s_k1_MSFT_s007 - s_k1_PG_s007  = 0
 k1_s010_balance_constraint:     b_k1_ADM_s010 + b_k1_AXP_s010 + b_k1_DIS_s010
                                 + b_k1_FDX_s010 + b_k1_IBM_s010
                                 + b_k1_INTC_s010 + b_k1_KO_s010
                                 + b_k1_MAT_s010 + b_k1_MSFT_s010
                                 + b_k1_PG_s010 - s_k1_ADM_s010 - s_k1_AXP_s010
                                 - s_k1_DIS_s010 - s_k1_FDX_s010
                                 - s_k1_IBM_s010 - s_k1_INTC_s010
                                 - s_k1_KO_s010 - s_k1_MAT_s010
                                 - s_k1_MSFT_s010 - s_k1_PG_s010  = 0
 k1_s011_balance_constraint:     b_k1_ADM_s011 + b_k1_AXP_s011 + b_k1_DIS_s011
                                 + b_k1_FDX_s011 + b_k1_IBM_s011
                                 + b_k1_INTC_s011 + b_k1_KO_s011
                                 + b_k1_MAT_s011 + b_k1_MSFT_s011
                                 + b_k1_PG_s011 - s_k1_ADM_s011 - s_k1_AXP_s011
                                 - s_k1_DIS_s011 - s_k1_FDX_s011
                                 - s_k1_IBM_s011 - s_k1_INTC_s011
                                 - s_k1_KO_s011 - s_k1_MAT_s011
                                 - s_k1_MSFT_s011 - s_k1_PG_s011  = 0
 k1_s012_balance_constraint:     b_k1_ADM_s012 + b_k1_AXP_s012 + b_k1_DIS_s012
                                 + b_k1_FDX_s012 + b_k1_IBM_s012
                                 + b_k1_INTC_s012 + b_k1_KO_s012
                                 + b_k1_MAT_s012 + b_k1_MSFT_s012
                                 + b_k1_PG_s012 - s_k1_ADM_s012 - s_k1_AXP_s012
                                 - s_k1_DIS_s012 - s_k1_FDX_s012
                                 - s_k1_IBM_s012 - s_k1_INTC_s012
                                 - s_k1_KO_s012 - s_k1_MAT_s012
                                 - s_k1_MSFT_s012 - s_k1_PG_s012  = 0
 k1_s013_balance_constraint:     b_k1_ADM_s013 + b_k1_AXP_s013 + b_k1_DIS_s013
                                 + b_k1_FDX_s013 + b_k1_IBM_s013
                                 + b_k1_INTC_s013 + b_k1_KO_s013
                                 + b_k1_MAT_s013 + b_k1_MSFT_s013
                                 + b_k1_PG_s013 - s_k1_ADM_s013 - s_k1_AXP_s013
                                 - s_k1_DIS_s013 - s_k1_FDX_s013
                                 - s_k1_IBM_s013 - s_k1_INTC_s013
                                 - s_k1_KO_s013 - s_k1_MAT_s013
                                 - s_k1_MSFT_s013 - s_k1_PG_s013  = 0
 k1_s014_balance_constraint:     b_k1_ADM_s014 + b_k1_AXP_s014 + b_k1_DIS_s014
                                 + b_k1_FDX_s014 + b_k1_IBM_s014
                                 + b_k1_INTC_s014 + b_k1_KO_s014
                                 + b_k1_MAT_s014 + b_k1_MSFT_s014
                                 + b_k1_PG_s014 - s_k1_ADM_s014 - s_k1_AXP_s014
                                 - s_k1_DIS_s014 - s_k1_FDX_s014
                                 - s_k1_IBM_s014 - s_k1_INTC_s014
                                 - s_k1_KO_s014 - s_k1_MAT_s014
                                 - s_k1_MSFT_s014 - s_k1_PG_s014  = 0
 k1_s015_balance_constraint:     b_k1_ADM_s015 + b_k1_AXP_s015 + b_k1_DIS_s015
                                 + b_k1_FDX_s015 + b_k1_IBM_s015
                                 + b_k1_INTC_s015 + b_k1_KO_s015
                                 + b_k1_MAT_s015 + b_k1_MSFT_s015
                                 + b_k1_PG_s015 - s_k1_ADM_s015 - s_k1_AXP_s015
                                 - s_k1_DIS_s015 - s_k1_FDX_s015
                                 - s_k1_IBM_s015 - s_k1_INTC_s015
                                 - s_k1_KO_s015 - s_k1_MAT_s015
                                 - s_k1_MSFT_s015 - s_k1_PG_s015  = 0
 k1_s016_balance_constraint:     b_k1_ADM_s016 + b_k1_AXP_s016 + b_k1_DIS_s016
                                 + b_k1_FDX_s016 + b_k1_IBM_s016
                                 + b_k1_INTC_s016 + b_k1_KO_s016
                                 + b_k1_MAT_s016 + b_k1_MSFT_s016
                                 + b_k1_PG_s016 - s_k1_ADM_s016 - s_k1_AXP_s016
                                 - s_k1_DIS_s016 - s_k1_FDX_s016
                                 - s_k1_IBM_s016 - s_k1_INTC_s016
                                 - s_k1_KO_s016 - s_k1_MAT_s016
                                 - s_k1_MSFT_s016 - s_k1_PG_s016  = 0
 k1_s017_balance_constraint:     b_k1_ADM_s017 + b_k1_AXP_s017 + b_k1_DIS_s017
                                 + b_k1_FDX_s017 + b_k1_IBM_s017
                                 + b_k1_INTC_s017 + b_k1_KO_s017
                                 + b_k1_MAT_s017 + b_k1_MSFT_s017
                                 + b_k1_PG_s017 - s_k1_ADM_s017 - s_k1_AXP_s017
                                 - s_k1_DIS_s017 - s_k1_FDX_s017
                                 - s_k1_IBM_s017 - s_k1_INTC_s017
                                 - s_k1_KO_s017 - s_k1_MAT_s017
                                 - s_k1_MSFT_s017 - s_k1_PG_s017  = 0
 k1_s020_balance_constraint:     b_k1_ADM_s020 + b_k1_AXP_s020 + b_k1_DIS_s020
                                 + b_k1_FDX_s020 + b_k1_IBM_s020
                                 + b_k1_INTC_s020 + b_k1_KO_s020
                                 + b_k1_MAT_s020 + b_k1_MSFT_s020
                                 + b_k1_PG_s020 - s_k1_ADM_s020 - s_k1_AXP_s020
                                 - s_k1_DIS_s020 - s_k1_FDX_s020
                                 - s_k1_IBM_s020 - s_k1_INTC_s020
                                 - s_k1_KO_s020 - s_k1_MAT_s020
                                 - s_k1_MSFT_s020 - s_k1_PG_s020  = 0
 k1_s021_balance_constraint:     b_k1_ADM_s021 + b_k1_AXP_s021 + b_k1_DIS_s021
                                 + b_k1_FDX_s021 + b_k1_IBM_s021
                                 + b_k1_INTC_s021 + b_k1_KO_s021
                                 + b_k1_MAT_s021 + b_k1_MSFT_s021
                                 + b_k1_PG_s021 - s_k1_ADM_s021 - s_k1_AXP_s021
                                 - s_k1_DIS_s021 - s_k1_FDX_s021
                                 - s_k1_IBM_s021 - s_k1_INTC_s021
                                 - s_k1_KO_s021 - s_k1_MAT_s021
                                 - s_k1_MSFT_s021 - s_k1_PG_s021  = 0
 k1_s022_balance_constraint:     b_k1_ADM_s022 + b_k1_AXP_s022 + b_k1_DIS_s022
                                 + b_k1_FDX_s022 + b_k1_IBM_s022
                                 + b_k1_INTC_s022 + b_k1_KO_s022
                                 + b_k1_MAT_s022 + b_k1_MSFT_s022
                                 + b_k1_PG_s022 - s_k1_ADM_s022 - s_k1_AXP_s022
                                 - s_k1_DIS_s022 - s_k1_FDX_s022
                                 - s_k1_IBM_s022 - s_k1_INTC_s022
                                 - s_k1_KO_s022 - s_k1_MAT_s022
                                 - s_k1_MSFT_s022 - s_k1_PG_s022  = 0
 k1_s023_balance_constraint:     b_k1_ADM_s023 + b_k1_AXP_s023 + b_k1_DIS_s023
                                 + b_k1_FDX_s023 + b_k1_IBM_s023
                                 + b_k1_INTC_s023 + b_k1_KO_s023
                                 + b_k1_MAT_s023 + b_k1_MSFT_s023
                                 + b_k1_PG_s023 - s_k1_ADM_s023 - s_k1_AXP_s023
                                 - s_k1_DIS_s023 - s_k1_FDX_s023
                                 - s_k1_IBM_s023 - s_k1_INTC_s023
                                 - s_k1_KO_s023 - s_k1_MAT_s023
                                 - s_k1_MSFT_s023 - s_k1_PG_s023  = 0
 k1_s024_balance_constraint:     b_k1_ADM_s024 + b_k1_AXP_s024 + b_k1_DIS_s024
                                 + b_k1_FDX_s024 + b_k1_IBM_s024
                                 + b_k1_INTC_s024 + b_k1_KO_s024
                                 + b_k1_MAT_s024 + b_k1_MSFT_s024
                                 + b_k1_PG_s024 - s_k1_ADM_s024 - s_k1_AXP_s024
                                 - s_k1_DIS_s024 - s_k1_FDX_s024
                                 - s_k1_IBM_s024 - s_k1_INTC_s024
                                 - s_k1_KO_s024 - s_k1_MAT_s024
                                 - s_k1_MSFT_s024 - s_k1_PG_s024  = 0
 k1_s025_balance_constraint:     b_k1_ADM_s025 + b_k1_AXP_s025 + b_k1_DIS_s025
                                 + b_k1_FDX_s025 + b_k1_IBM_s025
                                 + b_k1_INTC_s025 + b_k1_KO_s025
                                 + b_k1_MAT_s025 + b_k1_MSFT_s025
                                 + b_k1_PG_s025 - s_k1_ADM_s025 - s_k1_AXP_s025
                                 - s_k1_DIS_s025 - s_k1_FDX_s025
                                 - s_k1_IBM_s025 - s_k1_INTC_s025
                                 - s_k1_KO_s025 - s_k1_MAT_s025
                                 - s_k1_MSFT_s025 - s_k1_PG_s025  = 0
 k1_s026_balance_constraint:     b_k1_ADM_s026 + b_k1_AXP_s026 + b_k1_DIS_s026
                                 + b_k1_FDX_s026 + b_k1_IBM_s026
                                 + b_k1_INTC_s026 + b_k1_KO_s026
                                 + b_k1_MAT_s026 + b_k1_MSFT_s026
                                 + b_k1_PG_s026 - s_k1_ADM_s026 - s_k1_AXP_s026
                                 - s_k1_DIS_s026 - s_k1_FDX_s026
                                 - s_k1_IBM_s026 - s_k1_INTC_s026
                                 - s_k1_KO_s026 - s_k1_MAT_s026
                                 - s_k1_MSFT_s026 - s_k1_PG_s026  = 0
 k1_s027_balance_constraint:     b_k1_ADM_s027 + b_k1_AXP_s027 + b_k1_DIS_s027
                                 + b_k1_FDX_s027 + b_k1_IBM_s027
                                 + b_k1_INTC_s027 + b_k1_KO_s027
                                 + b_k1_MAT_s027 + b_k1_MSFT_s027
                                 + b_k1_PG_s027 - s_k1_ADM_s027 - s_k1_AXP_s027
                                 - s_k1_DIS_s027 - s_k1_FDX_s027
                                 - s_k1_IBM_s027 - s_k1_INTC_s027
                                 - s_k1_KO_s027 - s_k1_MAT_s027
                                 - s_k1_MSFT_s027 - s_k1_PG_s027  = 0
 k1_s030_balance_constraint:     b_k1_ADM_s030 + b_k1_AXP_s030 + b_k1_DIS_s030
                                 + b_k1_FDX_s030 + b_k1_IBM_s030
                                 + b_k1_INTC_s030 + b_k1_KO_s030
                                 + b_k1_MAT_s030 + b_k1_MSFT_s030
                                 + b_k1_PG_s030 - s_k1_ADM_s030 - s_k1_AXP_s030
                                 - s_k1_DIS_s030 - s_k1_FDX_s030
                                 - s_k1_IBM_s030 - s_k1_INTC_s030
                                 - s_k1_KO_s030 - s_k1_MAT_s030
                                 - s_k1_MSFT_s030 - s_k1_PG_s030  = 0
 k1_s031_balance_constraint:     b_k1_ADM_s031 + b_k1_AXP_s031 + b_k1_DIS_s031
                                 + b_k1_FDX_s031 + b_k1_IBM_s031
                                 + b_k1_INTC_s031 + b_k1_KO_s031
                                 + b_k1_MAT_s031 + b_k1_MSFT_s031
                                 + b_k1_PG_s031 - s_k1_ADM_s031 - s_k1_AXP_s031
                                 - s_k1_DIS_s031 - s_k1_FDX_s031
                                 - s_k1_IBM_s031 - s_k1_INTC_s031
                                 - s_k1_KO_s031 - s_k1_MAT_s031
                                 - s_k1_MSFT_s031 - s_k1_PG_s031  = 0
 k1_s032_balance_constraint:     b_k1_ADM_s032 + b_k1_AXP_s032 + b_k1_DIS_s032
                                 + b_k1_FDX_s032 + b_k1_IBM_s032
                                 + b_k1_INTC_s032 + b_k1_KO_s032
                                 + b_k1_MAT_s032 + b_k1_MSFT_s032
                                 + b_k1_PG_s032 - s_k1_ADM_s032 - s_k1_AXP_s032
                                 - s_k1_DIS_s032 - s_k1_FDX_s032
                                 - s_k1_IBM_s032 - s_k1_INTC_s032
                                 - s_k1_KO_s032 - s_k1_MAT_s032
                                 - s_k1_MSFT_s032 - s_k1_PG_s032  = 0
 k1_s033_balance_constraint:     b_k1_ADM_s033 + b_k1_AXP_s033 + b_k1_DIS_s033
                                 + b_k1_FDX_s033 + b_k1_IBM_s033
                                 + b_k1_INTC_s033 + b_k1_KO_s033
                                 + b_k1_MAT_s033 + b_k1_MSFT_s033
                                 + b_k1_PG_s033 - s_k1_ADM_s033 - s_k1_AXP_s033
                                 - s_k1_DIS_s033 - s_k1_FDX_s033
                                 - s_k1_IBM_s033 - s_k1_INTC_s033
                                 - s_k1_KO_s033 - s_k1_MAT_s033
                                 - s_k1_MSFT_s033 - s_k1_PG_s033  = 0
 k1_s034_balance_constraint:     b_k1_ADM_s034 + b_k1_AXP_s034 + b_k1_DIS_s034
                                 + b_k1_FDX_s034 + b_k1_IBM_s034
                                 + b_k1_INTC_s034 + b_k1_KO_s034
                                 + b_k1_MAT_s034 + b_k1_MSFT_s034
                                 + b_k1_PG_s034 - s_k1_ADM_s034 - s_k1_AXP_s034
                                 - s_k1_DIS_s034 - s_k1_FDX_s034
                                 - s_k1_IBM_s034 - s_k1_INTC_s034
                                 - s_k1_KO_s034 - s_k1_MAT_s034
                                 - s_k1_MSFT_s034 - s_k1_PG_s034  = 0
 k1_s035_balance_constraint:     b_k1_ADM_s035 + b_k1_AXP_s035 + b_k1_DIS_s035
                                 + b_k1_FDX_s035 + b_k1_IBM_s035
                                 + b_k1_INTC_s035 + b_k1_KO_s035
                                 + b_k1_MAT_s035 + b_k1_MSFT_s035
                                 + b_k1_PG_s035 - s_k1_ADM_s035 - s_k1_AXP_s035
                                 - s_k1_DIS_s035 - s_k1_FDX_s035
                                 - s_k1_IBM_s035 - s_k1_INTC_s035
                                 - s_k1_KO_s035 - s_k1_MAT_s035
                                 - s_k1_MSFT_s035 - s_k1_PG_s035  = 0
 k1_s036_balance_constraint:     b_k1_ADM_s036 + b_k1_AXP_s036 + b_k1_DIS_s036
                                 + b_k1_FDX_s036 + b_k1_IBM_s036
                                 + b_k1_INTC_s036 + b_k1_KO_s036
                                 + b_k1_MAT_s036 + b_k1_MSFT_s036
                                 + b_k1_PG_s036 - s_k1_ADM_s036 - s_k1_AXP_s036
                                 - s_k1_DIS_s036 - s_k1_FDX_s036
                                 - s_k1_IBM_s036 - s_k1_INTC_s036
                                 - s_k1_KO_s036 - s_k1_MAT_s036
                                 - s_k1_MSFT_s036 - s_k1_PG_s036  = 0
 k1_s037_balance_constraint:     b_k1_ADM_s037 + b_k1_AXP_s037 + b_k1_DIS_s037
                                 + b_k1_FDX_s037 + b_k1_IBM_s037
                                 + b_k1_INTC_s037 + b_k1_KO_s037
                                 + b_k1_MAT_s037 + b_k1_MSFT_s037
                                 + b_k1_PG_s037 - s_k1_ADM_s037 - s_k1_AXP_s037
                                 - s_k1_DIS_s037 - s_k1_FDX_s037
                                 - s_k1_IBM_s037 - s_k1_INTC_s037
                                 - s_k1_KO_s037 - s_k1_MAT_s037
                                 - s_k1_MSFT_s037 - s_k1_PG_s037  = 0
 k1_s040_balance_constraint:     b_k1_ADM_s040 + b_k1_AXP_s040 + b_k1_DIS_s040
                                 + b_k1_FDX_s040 + b_k1_IBM_s040
                                 + b_k1_INTC_s040 + b_k1_KO_s040
                                 + b_k1_MAT_s040 + b_k1_MSFT_s040
                                 + b_k1_PG_s040 - s_k1_ADM_s040 - s_k1_AXP_s040
                                 - s_k1_DIS_s040 - s_k1_FDX_s040
                                 - s_k1_IBM_s040 - s_k1_INTC_s040
                                 - s_k1_KO_s040 - s_k1_MAT_s040
                                 - s_k1_MSFT_s040 - s_k1_PG_s040  = 0
 k1_s041_balance_constraint:     b_k1_ADM_s041 + b_k1_AXP_s041 + b_k1_DIS_s041
                                 + b_k1_FDX_s041 + b_k1_IBM_s041
                                 + b_k1_INTC_s041 + b_k1_KO_s041
                                 + b_k1_MAT_s041 + b_k1_MSFT_s041
                                 + b_k1_PG_s041 - s_k1_ADM_s041 - s_k1_AXP_s041
                                 - s_k1_DIS_s041 - s_k1_FDX_s041
                                 - s_k1_IBM_s041 - s_k1_INTC_s041
                                 - s_k1_KO_s041 - s_k1_MAT_s041
                                 - s_k1_MSFT_s041 - s_k1_PG_s041  = 0
 k1_s042_balance_constraint:     b_k1_ADM_s042 + b_k1_AXP_s042 + b_k1_DIS_s042
                                 + b_k1_FDX_s042 + b_k1_IBM_s042
                                 + b_k1_INTC_s042 + b_k1_KO_s042
                                 + b_k1_MAT_s042 + b_k1_MSFT_s042
                                 + b_k1_PG_s042 - s_k1_ADM_s042 - s_k1_AXP_s042
                                 - s_k1_DIS_s042 - s_k1_FDX_s042
                                 - s_k1_IBM_s042 - s_k1_INTC_s042
                                 - s_k1_KO_s042 - s_k1_MAT_s042
                                 - s_k1_MSFT_s042 - s_k1_PG_s042  = 0
 k1_s043_balance_constraint:     b_k1_ADM_s043 + b_k1_AXP_s043 + b_k1_DIS_s043
                                 + b_k1_FDX_s043 + b_k1_IBM_s043
                                 + b_k1_INTC_s043 + b_k1_KO_s043
                                 + b_k1_MAT_s043 + b_k1_MSFT_s043
                                 + b_k1_PG_s043 - s_k1_ADM_s043 - s_k1_AXP_s043
                                 - s_k1_DIS_s043 - s_k1_FDX_s043
                                 - s_k1_IBM_s043 - s_k1_INTC_s043
                                 - s_k1_KO_s043 - s_k1_MAT_s043
                                 - s_k1_MSFT_s043 - s_k1_PG_s043  = 0
 k1_s044_balance_constraint:     b_k1_ADM_s044 + b_k1_AXP_s044 + b_k1_DIS_s044
                                 + b_k1_FDX_s044 + b_k1_IBM_s044
                                 + b_k1_INTC_s044 + b_k1_KO_s044
                                 + b_k1_MAT_s044 + b_k1_MSFT_s044
                                 + b_k1_PG_s044 - s_k1_ADM_s044 - s_k1_AXP_s044
                                 - s_k1_DIS_s044 - s_k1_FDX_s044
                                 - s_k1_IBM_s044 - s_k1_INTC_s044
                                 - s_k1_KO_s044 - s_k1_MAT_s044
                                 - s_k1_MSFT_s044 - s_k1_PG_s044  = 0
 k1_s045_balance_constraint:     b_k1_ADM_s045 + b_k1_AXP_s045 + b_k1_DIS_s045
                                 + b_k1_FDX_s045 + b_k1_IBM_s045
                                 + b_k1_INTC_s045 + b_k1_KO_s045
                                 + b_k1_MAT_s045 + b_k1_MSFT_s045
                                 + b_k1_PG_s045 - s_k1_ADM_s045 - s_k1_AXP_s045
                                 - s_k1_DIS_s045 - s_k1_FDX_s045
                                 - s_k1_IBM_s045 - s_k1_INTC_s045
                                 - s_k1_KO_s045 - s_k1_MAT_s045
                                 - s_k1_MSFT_s045 - s_k1_PG_s045  = 0
 k1_s046_balance_constraint:     b_k1_ADM_s046 + b_k1_AXP_s046 + b_k1_DIS_s046
                                 + b_k1_FDX_s046 + b_k1_IBM_s046
                                 + b_k1_INTC_s046 + b_k1_KO_s046
                                 + b_k1_MAT_s046 + b_k1_MSFT_s046
                                 + b_k1_PG_s046 - s_k1_ADM_s046 - s_k1_AXP_s046
                                 - s_k1_DIS_s046 - s_k1_FDX_s046
                                 - s_k1_IBM_s046 - s_k1_INTC_s046
                                 - s_k1_KO_s046 - s_k1_MAT_s046
                                 - s_k1_MSFT_s046 - s_k1_PG_s046  = 0
 k1_s047_balance_constraint:     b_k1_ADM_s047 + b_k1_AXP_s047 + b_k1_DIS_s047
                                 + b_k1_FDX_s047 + b_k1_IBM_s047
                                 + b_k1_INTC_s047 + b_k1_KO_s047
                                 + b_k1_MAT_s047 + b_k1_MSFT_s047
                                 + b_k1_PG_s047 - s_k1_ADM_s047 - s_k1_AXP_s047
                                 - s_k1_DIS_s047 - s_k1_FDX_s047
                                 - s_k1_IBM_s047 - s_k1_INTC_s047
                                 - s_k1_KO_s047 - s_k1_MAT_s047
                                 - s_k1_MSFT_s047 - s_k1_PG_s047  = 0
 k1_s050_balance_constraint:     b_k1_ADM_s050 + b_k1_AXP_s050 + b_k1_DIS_s050
                                 + b_k1_FDX_s050 + b_k1_IBM_s050
                                 + b_k1_INTC_s050 + b_k1_KO_s050
                                 + b_k1_MAT_s050 + b_k1_MSFT_s050
                                 + b_k1_PG_s050 - s_k1_ADM_s050 - s_k1_AXP_s050
                                 - s_k1_DIS_s050 - s_k1_FDX_s050
                                 - s_k1_IBM_s050 - s_k1_INTC_s050
                                 - s_k1_KO_s050 - s_k1_MAT_s050
                                 - s_k1_MSFT_s050 - s_k1_PG_s050  = 0
 k1_s051_balance_constraint:     b_k1_ADM_s051 + b_k1_AXP_s051 + b_k1_DIS_s051
                                 + b_k1_FDX_s051 + b_k1_IBM_s051
                                 + b_k1_INTC_s051 + b_k1_KO_s051
                                 + b_k1_MAT_s051 + b_k1_MSFT_s051
                                 + b_k1_PG_s051 - s_k1_ADM_s051 - s_k1_AXP_s051
                                 - s_k1_DIS_s051 - s_k1_FDX_s051
                                 - s_k1_IBM_s051 - s_k1_INTC_s051
                                 - s_k1_KO_s051 - s_k1_MAT_s051
                                 - s_k1_MSFT_s051 - s_k1_PG_s051  = 0
 k1_s052_balance_constraint:     b_k1_ADM_s052 + b_k1_AXP_s052 + b_k1_DIS_s052
                                 + b_k1_FDX_s052 + b_k1_IBM_s052
                                 + b_k1_INTC_s052 + b_k1_KO_s052
                                 + b_k1_MAT_s052 + b_k1_MSFT_s052
                                 + b_k1_PG_s052 - s_k1_ADM_s052 - s_k1_AXP_s052
                                 - s_k1_DIS_s052 - s_k1_FDX_s052
                                 - s_k1_IBM_s052 - s_k1_INTC_s052
                                 - s_k1_KO_s052 - s_k1_MAT_s052
                                 - s_k1_MSFT_s052 - s_k1_PG_s052  = 0
 k1_s053_balance_constraint:     b_k1_ADM_s053 + b_k1_AXP_s053 + b_k1_DIS_s053
                                 + b_k1_FDX_s053 + b_k1_IBM_s053
                                 + b_k1_INTC_s053 + b_k1_KO_s053
                                 + b_k1_MAT_s053 + b_k1_MSFT_s053
                                 + b_k1_PG_s053 - s_k1_ADM_s053 - s_k1_AXP_s053
                                 - s_k1_DIS_s053 - s_k1_FDX_s053
                                 - s_k1_IBM_s053 - s_k1_INTC_s053
                                 - s_k1_KO_s053 - s_k1_MAT_s053
                                 - s_k1_MSFT_s053 - s_k1_PG_s053  = 0
 k1_s054_balance_constraint:     b_k1_ADM_s054 + b_k1_AXP_s054 + b_k1_DIS_s054
                                 + b_k1_FDX_s054 + b_k1_IBM_s054
                                 + b_k1_INTC_s054 + b_k1_KO_s054
                                 + b_k1_MAT_s054 + b_k1_MSFT_s054
                                 + b_k1_PG_s054 - s_k1_ADM_s054 - s_k1_AXP_s054
                                 - s_k1_DIS_s054 - s_k1_FDX_s054
                                 - s_k1_IBM_s054 - s_k1_INTC_s054
                                 - s_k1_KO_s054 - s_k1_MAT_s054
                                 - s_k1_MSFT_s054 - s_k1_PG_s054  = 0
 k1_s055_balance_constraint:     b_k1_ADM_s055 + b_k1_AXP_s055 + b_k1_DIS_s055
                                 + b_k1_FDX_s055 + b_k1_IBM_s055
                                 + b_k1_INTC_s055 + b_k1_KO_s055
                                 + b_k1_MAT_s055 + b_k1_MSFT_s055
                                 + b_k1_PG_s055 - s_k1_ADM_s055 - s_k1_AXP_s055
                                 - s_k1_DIS_s055 - s_k1_FDX_s055
                                 - s_k1_IBM_s055 - s_k1_INTC_s055
                                 - s_k1_KO_s055 - s_k1_MAT_s055
                                 - s_k1_MSFT_s055 - s_k1_PG_s055  = 0
 k1_s056_balance_constraint:     b_k1_ADM_s056 + b_k1_AXP_s056 + b_k1_DIS_s056
                                 + b_k1_FDX_s056 + b_k1_IBM_s056
                                 + b_k1_INTC_s056 + b_k1_KO_s056
                                 + b_k1_MAT_s056 + b_k1_MSFT_s056
                                 + b_k1_PG_s056 - s_k1_ADM_s056 - s_k1_AXP_s056
                                 - s_k1_DIS_s056 - s_k1_FDX_s056
                                 - s_k1_IBM_s056 - s_k1_INTC_s056
                                 - s_k1_KO_s056 - s_k1_MAT_s056
                                 - s_k1_MSFT_s056 - s_k1_PG_s056  = 0
 k1_s057_balance_constraint:     b_k1_ADM_s057 + b_k1_AXP_s057 + b_k1_DIS_s057
                                 + b_k1_FDX_s057 + b_k1_IBM_s057
                                 + b_k1_INTC_s057 + b_k1_KO_s057
                                 + b_k1_MAT_s057 + b_k1_MSFT_s057
                                 + b_k1_PG_s057 - s_k1_ADM_s057 - s_k1_AXP_s057
                                 - s_k1_DIS_s057 - s_k1_FDX_s057
                                 - s_k1_IBM_s057 - s_k1_INTC_s057
                                 - s_k1_KO_s057 - s_k1_MAT_s057
                                 - s_k1_MSFT_s057 - s_k1_PG_s057  = 0
 k1_s060_balance_constraint:     b_k1_ADM_s060 + b_k1_AXP_s060 + b_k1_DIS_s060
                                 + b_k1_FDX_s060 + b_k1_IBM_s060
                                 + b_k1_INTC_s060 + b_k1_KO_s060
                                 + b_k1_MAT_s060 + b_k1_MSFT_s060
                                 + b_k1_PG_s060 - s_k1_ADM_s060 - s_k1_AXP_s060
                                 - s_k1_DIS_s060 - s_k1_FDX_s060
                                 - s_k1_IBM_s060 - s_k1_INTC_s060
                                 - s_k1_KO_s060 - s_k1_MAT_s060
                                 - s_k1_MSFT_s060 - s_k1_PG_s060  = 0
 k1_s061_balance_constraint:     b_k1_ADM_s061 + b_k1_AXP_s061 + b_k1_DIS_s061
                                 + b_k1_FDX_s061 + b_k1_IBM_s061
                                 + b_k1_INTC_s061 + b_k1_KO_s061
                                 + b_k1_MAT_s061 + b_k1_MSFT_s061
                                 + b_k1_PG_s061 - s_k1_ADM_s061 - s_k1_AXP_s061
                                 - s_k1_DIS_s061 - s_k1_FDX_s061
                                 - s_k1_IBM_s061 - s_k1_INTC_s061
                                 - s_k1_KO_s061 - s_k1_MAT_s061
                                 - s_k1_MSFT_s061 - s_k1_PG_s061  = 0
 k1_s062_balance_constraint:     b_k1_ADM_s062 + b_k1_AXP_s062 + b_k1_DIS_s062
                                 + b_k1_FDX_s062 + b_k1_IBM_s062
                                 + b_k1_INTC_s062 + b_k1_KO_s062
                                 + b_k1_MAT_s062 + b_k1_MSFT_s062
                                 + b_k1_PG_s062 - s_k1_ADM_s062 - s_k1_AXP_s062
                                 - s_k1_DIS_s062 - s_k1_FDX_s062
                                 - s_k1_IBM_s062 - s_k1_INTC_s062
                                 - s_k1_KO_s062 - s_k1_MAT_s062
                                 - s_k1_MSFT_s062 - s_k1_PG_s062  = 0
 k1_s063_balance_constraint:     b_k1_ADM_s063 + b_k1_AXP_s063 + b_k1_DIS_s063
                                 + b_k1_FDX_s063 + b_k1_IBM_s063
                                 + b_k1_INTC_s063 + b_k1_KO_s063
                                 + b_k1_MAT_s063 + b_k1_MSFT_s063
                                 + b_k1_PG_s063 - s_k1_ADM_s063 - s_k1_AXP_s063
                                 - s_k1_DIS_s063 - s_k1_FDX_s063
                                 - s_k1_IBM_s063 - s_k1_INTC_s063
                                 - s_k1_KO_s063 - s_k1_MAT_s063
                                 - s_k1_MSFT_s063 - s_k1_PG_s063  = 0
 k1_s064_balance_constraint:     b_k1_ADM_s064 + b_k1_AXP_s064 + b_k1_DIS_s064
                                 + b_k1_FDX_s064 + b_k1_IBM_s064
                                 + b_k1_INTC_s064 + b_k1_KO_s064
                                 + b_k1_MAT_s064 + b_k1_MSFT_s064
                                 + b_k1_PG_s064 - s_k1_ADM_s064 - s_k1_AXP_s064
                                 - s_k1_DIS_s064 - s_k1_FDX_s064
                                 - s_k1_IBM_s064 - s_k1_INTC_s064
                                 - s_k1_KO_s064 - s_k1_MAT_s064
                                 - s_k1_MSFT_s064 - s_k1_PG_s064  = 0
 k1_s065_balance_constraint:     b_k1_ADM_s065 + b_k1_AXP_s065 + b_k1_DIS_s065
                                 + b_k1_FDX_s065 + b_k1_IBM_s065
                                 + b_k1_INTC_s065 + b_k1_KO_s065
                                 + b_k1_MAT_s065 + b_k1_MSFT_s065
                                 + b_k1_PG_s065 - s_k1_ADM_s065 - s_k1_AXP_s065
                                 - s_k1_DIS_s065 - s_k1_FDX_s065
                                 - s_k1_IBM_s065 - s_k1_INTC_s065
                                 - s_k1_KO_s065 - s_k1_MAT_s065
                                 - s_k1_MSFT_s065 - s_k1_PG_s065  = 0
 k1_s066_balance_constraint:     b_k1_ADM_s066 + b_k1_AXP_s066 + b_k1_DIS_s066
                                 + b_k1_FDX_s066 + b_k1_IBM_s066
                                 + b_k1_INTC_s066 + b_k1_KO_s066
                                 + b_k1_MAT_s066 + b_k1_MSFT_s066
                                 + b_k1_PG_s066 - s_k1_ADM_s066 - s_k1_AXP_s066
                                 - s_k1_DIS_s066 - s_k1_FDX_s066
                                 - s_k1_IBM_s066 - s_k1_INTC_s066
                                 - s_k1_KO_s066 - s_k1_MAT_s066
                                 - s_k1_MSFT_s066 - s_k1_PG_s066  = 0
 k1_s067_balance_constraint:     b_k1_ADM_s067 + b_k1_AXP_s067 + b_k1_DIS_s067
                                 + b_k1_FDX_s067 + b_k1_IBM_s067
                                 + b_k1_INTC_s067 + b_k1_KO_s067
                                 + b_k1_MAT_s067 + b_k1_MSFT_s067
                                 + b_k1_PG_s067 - s_k1_ADM_s067 - s_k1_AXP_s067
                                 - s_k1_DIS_s067 - s_k1_FDX_s067
                                 - s_k1_IBM_s067 - s_k1_INTC_s067
                                 - s_k1_KO_s067 - s_k1_MAT_s067
                                 - s_k1_MSFT_s067 - s_k1_PG_s067  = 0
 k1_s070_balance_constraint:     b_k1_ADM_s070 + b_k1_AXP_s070 + b_k1_DIS_s070
                                 + b_k1_FDX_s070 + b_k1_IBM_s070
                                 + b_k1_INTC_s070 + b_k1_KO_s070
                                 + b_k1_MAT_s070 + b_k1_MSFT_s070
                                 + b_k1_PG_s070 - s_k1_ADM_s070 - s_k1_AXP_s070
                                 - s_k1_DIS_s070 - s_k1_FDX_s070
                                 - s_k1_IBM_s070 - s_k1_INTC_s070
                                 - s_k1_KO_s070 - s_k1_MAT_s070
                                 - s_k1_MSFT_s070 - s_k1_PG_s070  = 0
 k1_s071_balance_constraint:     b_k1_ADM_s071 + b_k1_AXP_s071 + b_k1_DIS_s071
                                 + b_k1_FDX_s071 + b_k1_IBM_s071
                                 + b_k1_INTC_s071 + b_k1_KO_s071
                                 + b_k1_MAT_s071 + b_k1_MSFT_s071
                                 + b_k1_PG_s071 - s_k1_ADM_s071 - s_k1_AXP_s071
                                 - s_k1_DIS_s071 - s_k1_FDX_s071
                                 - s_k1_IBM_s071 - s_k1_INTC_s071
                                 - s_k1_KO_s071 - s_k1_MAT_s071
                                 - s_k1_MSFT_s071 - s_k1_PG_s071  = 0
 k1_s072_balance_constraint:     b_k1_ADM_s072 + b_k1_AXP_s072 + b_k1_DIS_s072
                                 + b_k1_FDX_s072 + b_k1_IBM_s072
                                 + b_k1_INTC_s072 + b_k1_KO_s072
                                 + b_k1_MAT_s072 + b_k1_MSFT_s072
                                 + b_k1_PG_s072 - s_k1_ADM_s072 - s_k1_AXP_s072
                                 - s_k1_DIS_s072 - s_k1_FDX_s072
                                 - s_k1_IBM_s072 - s_k1_INTC_s072
                                 - s_k1_KO_s072 - s_k1_MAT_s072
                                 - s_k1_MSFT_s072 - s_k1_PG_s072  = 0
 k1_s073_balance_constraint:     b_k1_ADM_s073 + b_k1_AXP_s073 + b_k1_DIS_s073
                                 + b_k1_FDX_s073 + b_k1_IBM_s073
                                 + b_k1_INTC_s073 + b_k1_KO_s073
                                 + b_k1_MAT_s073 + b_k1_MSFT_s073
                                 + b_k1_PG_s073 - s_k1_ADM_s073 - s_k1_AXP_s073
                                 - s_k1_DIS_s073 - s_k1_FDX_s073
                                 - s_k1_IBM_s073 - s_k1_INTC_s073
                                 - s_k1_KO_s073 - s_k1_MAT_s073
                                 - s_k1_MSFT_s073 - s_k1_PG_s073  = 0
 k1_s074_balance_constraint:     b_k1_ADM_s074 + b_k1_AXP_s074 + b_k1_DIS_s074
                                 + b_k1_FDX_s074 + b_k1_IBM_s074
                                 + b_k1_INTC_s074 + b_k1_KO_s074
                                 + b_k1_MAT_s074 + b_k1_MSFT_s074
                                 + b_k1_PG_s074 - s_k1_ADM_s074 - s_k1_AXP_s074
                                 - s_k1_DIS_s074 - s_k1_FDX_s074
                                 - s_k1_IBM_s074 - s_k1_INTC_s074
                                 - s_k1_KO_s074 - s_k1_MAT_s074
                                 - s_k1_MSFT_s074 - s_k1_PG_s074  = 0
 k1_s075_balance_constraint:     b_k1_ADM_s075 + b_k1_AXP_s075 + b_k1_DIS_s075
                                 + b_k1_FDX_s075 + b_k1_IBM_s075
                                 + b_k1_INTC_s075 + b_k1_KO_s075
                                 + b_k1_MAT_s075 + b_k1_MSFT_s075
                                 + b_k1_PG_s075 - s_k1_ADM_s075 - s_k1_AXP_s075
                                 - s_k1_DIS_s075 - s_k1_FDX_s075
                                 - s_k1_IBM_s075 - s_k1_INTC_s075
                                 - s_k1_KO_s075 - s_k1_MAT_s075
                                 - s_k1_MSFT_s075 - s_k1_PG_s075  = 0
 k1_s076_balance_constraint:     b_k1_ADM_s076 + b_k1_AXP_s076 + b_k1_DIS_s076
                                 + b_k1_FDX_s076 + b_k1_IBM_s076
                                 + b_k1_INTC_s076 + b_k1_KO_s076
                                 + b_k1_MAT_s076 + b_k1_MSFT_s076
                                 + b_k1_PG_s076 - s_k1_ADM_s076 - s_k1_AXP_s076
                                 - s_k1_DIS_s076 - s_k1_FDX_s076
                                 - s_k1_IBM_s076 - s_k1_INTC_s076
                                 - s_k1_KO_s076 - s_k1_MAT_s076
                                 - s_k1_MSFT_s076 - s_k1_PG_s076  = 0
 k1_s077_balance_constraint:     b_k1_ADM_s077 + b_k1_AXP_s077 + b_k1_DIS_s077
                                 + b_k1_FDX_s077 + b_k1_IBM_s077
                                 + b_k1_INTC_s077 + b_k1_KO_s077
                                 + b_k1_MAT_s077 + b_k1_MSFT_s077
                                 + b_k1_PG_s077 - s_k1_ADM_s077 - s_k1_AXP_s077
                                 - s_k1_DIS_s077 - s_k1_FDX_s077
                                 - s_k1_IBM_s077 - s_k1_INTC_s077
                                 - s_k1_KO_s077 - s_k1_MAT_s077
                                 - s_k1_MSFT_s077 - s_k1_PG_s077  = 0
 k1_s100_balance_constraint:     b_k1_ADM_s100 + b_k1_AXP_s100 + b_k1_DIS_s100
                                 + b_k1_FDX_s100 + b_k1_IBM_s100
                                 + b_k1_INTC_s100 + b_k1_KO_s100
                                 + b_k1_MAT_s100 + b_k1_MSFT_s100
                                 + b_k1_PG_s100 - s_k1_ADM_s100 - s_k1_AXP_s100
                                 - s_k1_DIS_s100 - s_k1_FDX_s100
                                 - s_k1_IBM_s100 - s_k1_INTC_s100
                                 - s_k1_KO_s100 - s_k1_MAT_s100
                                 - s_k1_MSFT_s100 - s_k1_PG_s100  = 0
 k1_s101_balance_constraint:     b_k1_ADM_s101 + b_k1_AXP_s101 + b_k1_DIS_s101
                                 + b_k1_FDX_s101 + b_k1_IBM_s101
                                 + b_k1_INTC_s101 + b_k1_KO_s101
                                 + b_k1_MAT_s101 + b_k1_MSFT_s101
                                 + b_k1_PG_s101 - s_k1_ADM_s101 - s_k1_AXP_s101
                                 - s_k1_DIS_s101 - s_k1_FDX_s101
                                 - s_k1_IBM_s101 - s_k1_INTC_s101
                                 - s_k1_KO_s101 - s_k1_MAT_s101
                                 - s_k1_MSFT_s101 - s_k1_PG_s101  = 0
 k1_s102_balance_constraint:     b_k1_ADM_s102 + b_k1_AXP_s102 + b_k1_DIS_s102
                                 + b_k1_FDX_s102 + b_k1_IBM_s102
                                 + b_k1_INTC_s102 + b_k1_KO_s102
                                 + b_k1_MAT_s102 + b_k1_MSFT_s102
                                 + b_k1_PG_s102 - s_k1_ADM_s102 - s_k1_AXP_s102
                                 - s_k1_DIS_s102 - s_k1_FDX_s102
                                 - s_k1_IBM_s102 - s_k1_INTC_s102
                                 - s_k1_KO_s102 - s_k1_MAT_s102
                                 - s_k1_MSFT_s102 - s_k1_PG_s102  = 0
 k1_s103_balance_constraint:     b_k1_ADM_s103 + b_k1_AXP_s103 + b_k1_DIS_s103
                                 + b_k1_FDX_s103 + b_k1_IBM_s103
                                 + b_k1_INTC_s103 + b_k1_KO_s103
                                 + b_k1_MAT_s103 + b_k1_MSFT_s103
                                 + b_k1_PG_s103 - s_k1_ADM_s103 - s_k1_AXP_s103
                                 - s_k1_DIS_s103 - s_k1_FDX_s103
                                 - s_k1_IBM_s103 - s_k1_INTC_s103
                                 - s_k1_KO_s103 - s_k1_MAT_s103
                                 - s_k1_MSFT_s103 - s_k1_PG_s103  = 0
 k1_s104_balance_constraint:     b_k1_ADM_s104 + b_k1_AXP_s104 + b_k1_DIS_s104
                                 + b_k1_FDX_s104 + b_k1_IBM_s104
                                 + b_k1_INTC_s104 + b_k1_KO_s104
                                 + b_k1_MAT_s104 + b_k1_MSFT_s104
                                 + b_k1_PG_s104 - s_k1_ADM_s104 - s_k1_AXP_s104
                                 - s_k1_DIS_s104 - s_k1_FDX_s104
                                 - s_k1_IBM_s104 - s_k1_INTC_s104
                                 - s_k1_KO_s104 - s_k1_MAT_s104
                                 - s_k1_MSFT_s104 - s_k1_PG_s104  = 0
 k1_s105_balance_constraint:     b_k1_ADM_s105 + b_k1_AXP_s105 + b_k1_DIS_s105
                                 + b_k1_FDX_s105 + b_k1_IBM_s105
                                 + b_k1_INTC_s105 + b_k1_KO_s105
                                 + b_k1_MAT_s105 + b_k1_MSFT_s105
                                 + b_k1_PG_s105 - s_k1_ADM_s105 - s_k1_AXP_s105
                                 - s_k1_DIS_s105 - s_k1_FDX_s105
                                 - s_k1_IBM_s105 - s_k1_INTC_s105
                                 - s_k1_KO_s105 - s_k1_MAT_s105
                                 - s_k1_MSFT_s105 - s_k1_PG_s105  = 0
 k1_s106_balance_constraint:     b_k1_ADM_s106 + b_k1_AXP_s106 + b_k1_DIS_s106
                                 + b_k1_FDX_s106 + b_k1_IBM_s106
                                 + b_k1_INTC_s106 + b_k1_KO_s106
                                 + b_k1_MAT_s106 + b_k1_MSFT_s106
                                 + b_k1_PG_s106 - s_k1_ADM_s106 - s_k1_AXP_s106
                                 - s_k1_DIS_s106 - s_k1_FDX_s106
                                 - s_k1_IBM_s106 - s_k1_INTC_s106
                                 - s_k1_KO_s106 - s_k1_MAT_s106
                                 - s_k1_MSFT_s106 - s_k1_PG_s106  = 0
 k1_s107_balance_constraint:     b_k1_ADM_s107 + b_k1_AXP_s107 + b_k1_DIS_s107
                                 + b_k1_FDX_s107 + b_k1_IBM_s107
                                 + b_k1_INTC_s107 + b_k1_KO_s107
                                 + b_k1_MAT_s107 + b_k1_MSFT_s107
                                 + b_k1_PG_s107 - s_k1_ADM_s107 - s_k1_AXP_s107
                                 - s_k1_DIS_s107 - s_k1_FDX_s107
                                 - s_k1_IBM_s107 - s_k1_INTC_s107
                                 - s_k1_KO_s107 - s_k1_MAT_s107
                                 - s_k1_MSFT_s107 - s_k1_PG_s107  = 0
 k1_s110_balance_constraint:     b_k1_ADM_s110 + b_k1_AXP_s110 + b_k1_DIS_s110
                                 + b_k1_FDX_s110 + b_k1_IBM_s110
                                 + b_k1_INTC_s110 + b_k1_KO_s110
                                 + b_k1_MAT_s110 + b_k1_MSFT_s110
                                 + b_k1_PG_s110 - s_k1_ADM_s110 - s_k1_AXP_s110
                                 - s_k1_DIS_s110 - s_k1_FDX_s110
                                 - s_k1_IBM_s110 - s_k1_INTC_s110
                                 - s_k1_KO_s110 - s_k1_MAT_s110
                                 - s_k1_MSFT_s110 - s_k1_PG_s110  = 0
 k1_s111_balance_constraint:     b_k1_ADM_s111 + b_k1_AXP_s111 + b_k1_DIS_s111
                                 + b_k1_FDX_s111 + b_k1_IBM_s111
                                 + b_k1_INTC_s111 + b_k1_KO_s111
                                 + b_k1_MAT_s111 + b_k1_MSFT_s111
                                 + b_k1_PG_s111 - s_k1_ADM_s111 - s_k1_AXP_s111
                                 - s_k1_DIS_s111 - s_k1_FDX_s111
                                 - s_k1_IBM_s111 - s_k1_INTC_s111
                                 - s_k1_KO_s111 - s_k1_MAT_s111
                                 - s_k1_MSFT_s111 - s_k1_PG_s111  = 0
 k1_s112_balance_constraint:     b_k1_ADM_s112 + b_k1_AXP_s112 + b_k1_DIS_s112
                                 + b_k1_FDX_s112 + b_k1_IBM_s112
                                 + b_k1_INTC_s112 + b_k1_KO_s112
                                 + b_k1_MAT_s112 + b_k1_MSFT_s112
                                 + b_k1_PG_s112 - s_k1_ADM_s112 - s_k1_AXP_s112
                                 - s_k1_DIS_s112 - s_k1_FDX_s112
                                 - s_k1_IBM_s112 - s_k1_INTC_s112
                                 - s_k1_KO_s112 - s_k1_MAT_s112
                                 - s_k1_MSFT_s112 - s_k1_PG_s112  = 0
 k1_s113_balance_constraint:     b_k1_ADM_s113 + b_k1_AXP_s113 + b_k1_DIS_s113
                                 + b_k1_FDX_s113 + b_k1_IBM_s113
                                 + b_k1_INTC_s113 + b_k1_KO_s113
                                 + b_k1_MAT_s113 + b_k1_MSFT_s113
                                 + b_k1_PG_s113 - s_k1_ADM_s113 - s_k1_AXP_s113
                                 - s_k1_DIS_s113 - s_k1_FDX_s113
                                 - s_k1_IBM_s113 - s_k1_INTC_s113
                                 - s_k1_KO_s113 - s_k1_MAT_s113
                                 - s_k1_MSFT_s113 - s_k1_PG_s113  = 0
 k1_s114_balance_constraint:     b_k1_ADM_s114 + b_k1_AXP_s114 + b_k1_DIS_s114
                                 + b_k1_FDX_s114 + b_k1_IBM_s114
                                 + b_k1_INTC_s114 + b_k1_KO_s114
                                 + b_k1_MAT_s114 + b_k1_MSFT_s114
                                 + b_k1_PG_s114 - s_k1_ADM_s114 - s_k1_AXP_s114
                                 - s_k1_DIS_s114 - s_k1_FDX_s114
                                 - s_k1_IBM_s114 - s_k1_INTC_s114
                                 - s_k1_KO_s114 - s_k1_MAT_s114
                                 - s_k1_MSFT_s114 - s_k1_PG_s114  = 0
 k1_s115_balance_constraint:     b_k1_ADM_s115 + b_k1_AXP_s115 + b_k1_DIS_s115
                                 + b_k1_FDX_s115 + b_k1_IBM_s115
                                 + b_k1_INTC_s115 + b_k1_KO_s115
                                 + b_k1_MAT_s115 + b_k1_MSFT_s115
                                 + b_k1_PG_s115 - s_k1_ADM_s115 - s_k1_AXP_s115
                                 - s_k1_DIS_s115 - s_k1_FDX_s115
                                 - s_k1_IBM_s115 - s_k1_INTC_s115
                                 - s_k1_KO_s115 - s_k1_MAT_s115
                                 - s_k1_MSFT_s115 - s_k1_PG_s115  = 0
 k1_s116_balance_constraint:     b_k1_ADM_s116 + b_k1_AXP_s116 + b_k1_DIS_s116
                                 + b_k1_FDX_s116 + b_k1_IBM_s116
                                 + b_k1_INTC_s116 + b_k1_KO_s116
                                 + b_k1_MAT_s116 + b_k1_MSFT_s116
                                 + b_k1_PG_s116 - s_k1_ADM_s116 - s_k1_AXP_s116
                                 - s_k1_DIS_s116 - s_k1_FDX_s116
                                 - s_k1_IBM_s116 - s_k1_INTC_s116
                                 - s_k1_KO_s116 - s_k1_MAT_s116
                                 - s_k1_MSFT_s116 - s_k1_PG_s116  = 0
 k1_s117_balance_constraint:     b_k1_ADM_s117 + b_k1_AXP_s117 + b_k1_DIS_s117
                                 + b_k1_FDX_s117 + b_k1_IBM_s117
                                 + b_k1_INTC_s117 + b_k1_KO_s117
                                 + b_k1_MAT_s117 + b_k1_MSFT_s117
                                 + b_k1_PG_s117 - s_k1_ADM_s117 - s_k1_AXP_s117
                                 - s_k1_DIS_s117 - s_k1_FDX_s117
                                 - s_k1_IBM_s117 - s_k1_INTC_s117
                                 - s_k1_KO_s117 - s_k1_MAT_s117
                                 - s_k1_MSFT_s117 - s_k1_PG_s117  = 0
 k1_s120_balance_constraint:     b_k1_ADM_s120 + b_k1_AXP_s120 + b_k1_DIS_s120
                                 + b_k1_FDX_s120 + b_k1_IBM_s120
                                 + b_k1_INTC_s120 + b_k1_KO_s120
                                 + b_k1_MAT_s120 + b_k1_MSFT_s120
                                 + b_k1_PG_s120 - s_k1_ADM_s120 - s_k1_AXP_s120
                                 - s_k1_DIS_s120 - s_k1_FDX_s120
                                 - s_k1_IBM_s120 - s_k1_INTC_s120
                                 - s_k1_KO_s120 - s_k1_MAT_s120
                                 - s_k1_MSFT_s120 - s_k1_PG_s120  = 0
 k1_s121_balance_constraint:     b_k1_ADM_s121 + b_k1_AXP_s121 + b_k1_DIS_s121
                                 + b_k1_FDX_s121 + b_k1_IBM_s121
                                 + b_k1_INTC_s121 + b_k1_KO_s121
                                 + b_k1_MAT_s121 + b_k1_MSFT_s121
                                 + b_k1_PG_s121 - s_k1_ADM_s121 - s_k1_AXP_s121
                                 - s_k1_DIS_s121 - s_k1_FDX_s121
                                 - s_k1_IBM_s121 - s_k1_INTC_s121
                                 - s_k1_KO_s121 - s_k1_MAT_s121
                                 - s_k1_MSFT_s121 - s_k1_PG_s121  = 0
 k1_s122_balance_constraint:     b_k1_ADM_s122 + b_k1_AXP_s122 + b_k1_DIS_s122
                                 + b_k1_FDX_s122 + b_k1_IBM_s122
                                 + b_k1_INTC_s122 + b_k1_KO_s122
                                 + b_k1_MAT_s122 + b_k1_MSFT_s122
                                 + b_k1_PG_s122 - s_k1_ADM_s122 - s_k1_AXP_s122
                                 - s_k1_DIS_s122 - s_k1_FDX_s122
                                 - s_k1_IBM_s122 - s_k1_INTC_s122
                                 - s_k1_KO_s122 - s_k1_MAT_s122
                                 - s_k1_MSFT_s122 - s_k1_PG_s122  = 0
 k1_s123_balance_constraint:     b_k1_ADM_s123 + b_k1_AXP_s123 + b_k1_DIS_s123
                                 + b_k1_FDX_s123 + b_k1_IBM_s123
                                 + b_k1_INTC_s123 + b_k1_KO_s123
                                 + b_k1_MAT_s123 + b_k1_MSFT_s123
                                 + b_k1_PG_s123 - s_k1_ADM_s123 - s_k1_AXP_s123
                                 - s_k1_DIS_s123 - s_k1_FDX_s123
                                 - s_k1_IBM_s123 - s_k1_INTC_s123
                                 - s_k1_KO_s123 - s_k1_MAT_s123
                                 - s_k1_MSFT_s123 - s_k1_PG_s123  = 0
 k1_s124_balance_constraint:     b_k1_ADM_s124 + b_k1_AXP_s124 + b_k1_DIS_s124
                                 + b_k1_FDX_s124 + b_k1_IBM_s124
                                 + b_k1_INTC_s124 + b_k1_KO_s124
                                 + b_k1_MAT_s124 + b_k1_MSFT_s124
                                 + b_k1_PG_s124 - s_k1_ADM_s124 - s_k1_AXP_s124
                                 - s_k1_DIS_s124 - s_k1_FDX_s124
                                 - s_k1_IBM_s124 - s_k1_INTC_s124
                                 - s_k1_KO_s124 - s_k1_MAT_s124
                                 - s_k1_MSFT_s124 - s_k1_PG_s124  = 0
 k1_s125_balance_constraint:     b_k1_ADM_s125 + b_k1_AXP_s125 + b_k1_DIS_s125
                                 + b_k1_FDX_s125 + b_k1_IBM_s125
                                 + b_k1_INTC_s125 + b_k1_KO_s125
                                 + b_k1_MAT_s125 + b_k1_MSFT_s125
                                 + b_k1_PG_s125 - s_k1_ADM_s125 - s_k1_AXP_s125
                                 - s_k1_DIS_s125 - s_k1_FDX_s125
                                 - s_k1_IBM_s125 - s_k1_INTC_s125
                                 - s_k1_KO_s125 - s_k1_MAT_s125
                                 - s_k1_MSFT_s125 - s_k1_PG_s125  = 0
 k1_s126_balance_constraint:     b_k1_ADM_s126 + b_k1_AXP_s126 + b_k1_DIS_s126
                                 + b_k1_FDX_s126 + b_k1_IBM_s126
                                 + b_k1_INTC_s126 + b_k1_KO_s126
                                 + b_k1_MAT_s126 + b_k1_MSFT_s126
                                 + b_k1_PG_s126 - s_k1_ADM_s126 - s_k1_AXP_s126
                                 - s_k1_DIS_s126 - s_k1_FDX_s126
                                 - s_k1_IBM_s126 - s_k1_INTC_s126
                                 - s_k1_KO_s126 - s_k1_MAT_s126
                                 - s_k1_MSFT_s126 - s_k1_PG_s126  = 0
 k1_s127_balance_constraint:     b_k1_ADM_s127 + b_k1_AXP_s127 + b_k1_DIS_s127
                                 + b_k1_FDX_s127 + b_k1_IBM_s127
                                 + b_k1_INTC_s127 + b_k1_KO_s127
                                 + b_k1_MAT_s127 + b_k1_MSFT_s127
                                 + b_k1_PG_s127 - s_k1_ADM_s127 - s_k1_AXP_s127
                                 - s_k1_DIS_s127 - s_k1_FDX_s127
                                 - s_k1_IBM_s127 - s_k1_INTC_s127
                                 - s_k1_KO_s127 - s_k1_MAT_s127
                                 - s_k1_MSFT_s127 - s_k1_PG_s127  = 0
 k1_s130_balance_constraint:     b_k1_ADM_s130 + b_k1_AXP_s130 + b_k1_DIS_s130
                                 + b_k1_FDX_s130 + b_k1_IBM_s130
                                 + b_k1_INTC_s130 + b_k1_KO_s130
                                 + b_k1_MAT_s130 + b_k1_MSFT_s130
                                 + b_k1_PG_s130 - s_k1_ADM_s130 - s_k1_AXP_s130
                                 - s_k1_DIS_s130 - s_k1_FDX_s130
                                 - s_k1_IBM_s130 - s_k1_INTC_s130
                                 - s_k1_KO_s130 - s_k1_MAT_s130
                                 - s_k1_MSFT_s130 - s_k1_PG_s130  = 0
 k1_s131_balance_constraint:     b_k1_ADM_s131 + b_k1_AXP_s131 + b_k1_DIS_s131
                                 + b_k1_FDX_s131 + b_k1_IBM_s131
                                 + b_k1_INTC_s131 + b_k1_KO_s131
                                 + b_k1_MAT_s131 + b_k1_MSFT_s131
                                 + b_k1_PG_s131 - s_k1_ADM_s131 - s_k1_AXP_s131
                                 - s_k1_DIS_s131 - s_k1_FDX_s131
                                 - s_k1_IBM_s131 - s_k1_INTC_s131
                                 - s_k1_KO_s131 - s_k1_MAT_s131
                                 - s_k1_MSFT_s131 - s_k1_PG_s131  = 0
 k1_s132_balance_constraint:     b_k1_ADM_s132 + b_k1_AXP_s132 + b_k1_DIS_s132
                                 + b_k1_FDX_s132 + b_k1_IBM_s132
                                 + b_k1_INTC_s132 + b_k1_KO_s132
                                 + b_k1_MAT_s132 + b_k1_MSFT_s132
                                 + b_k1_PG_s132 - s_k1_ADM_s132 - s_k1_AXP_s132
                                 - s_k1_DIS_s132 - s_k1_FDX_s132
                                 - s_k1_IBM_s132 - s_k1_INTC_s132
                                 - s_k1_KO_s132 - s_k1_MAT_s132
                                 - s_k1_MSFT_s132 - s_k1_PG_s132  = 0
 k1_s133_balance_constraint:     b_k1_ADM_s133 + b_k1_AXP_s133 + b_k1_DIS_s133
                                 + b_k1_FDX_s133 + b_k1_IBM_s133
                                 + b_k1_INTC_s133 + b_k1_KO_s133
                                 + b_k1_MAT_s133 + b_k1_MSFT_s133
                                 + b_k1_PG_s133 - s_k1_ADM_s133 - s_k1_AXP_s133
                                 - s_k1_DIS_s133 - s_k1_FDX_s133
                                 - s_k1_IBM_s133 - s_k1_INTC_s133
                                 - s_k1_KO_s133 - s_k1_MAT_s133
                                 - s_k1_MSFT_s133 - s_k1_PG_s133  = 0
 k1_s134_balance_constraint:     b_k1_ADM_s134 + b_k1_AXP_s134 + b_k1_DIS_s134
                                 + b_k1_FDX_s134 + b_k1_IBM_s134
                                 + b_k1_INTC_s134 + b_k1_KO_s134
                                 + b_k1_MAT_s134 + b_k1_MSFT_s134
                                 + b_k1_PG_s134 - s_k1_ADM_s134 - s_k1_AXP_s134
                                 - s_k1_DIS_s134 - s_k1_FDX_s134
                                 - s_k1_IBM_s134 - s_k1_INTC_s134
                                 - s_k1_KO_s134 - s_k1_MAT_s134
                                 - s_k1_MSFT_s134 - s_k1_PG_s134  = 0
 k1_s135_balance_constraint:     b_k1_ADM_s135 + b_k1_AXP_s135 + b_k1_DIS_s135
                                 + b_k1_FDX_s135 + b_k1_IBM_s135
                                 + b_k1_INTC_s135 + b_k1_KO_s135
                                 + b_k1_MAT_s135 + b_k1_MSFT_s135
                                 + b_k1_PG_s135 - s_k1_ADM_s135 - s_k1_AXP_s135
                                 - s_k1_DIS_s135 - s_k1_FDX_s135
                                 - s_k1_IBM_s135 - s_k1_INTC_s135
                                 - s_k1_KO_s135 - s_k1_MAT_s135
                                 - s_k1_MSFT_s135 - s_k1_PG_s135  = 0
 k1_s136_balance_constraint:     b_k1_ADM_s136 + b_k1_AXP_s136 + b_k1_DIS_s136
                                 + b_k1_FDX_s136 + b_k1_IBM_s136
                                 + b_k1_INTC_s136 + b_k1_KO_s136
                                 + b_k1_MAT_s136 + b_k1_MSFT_s136
                                 + b_k1_PG_s136 - s_k1_ADM_s136 - s_k1_AXP_s136
                                 - s_k1_DIS_s136 - s_k1_FDX_s136
                                 - s_k1_IBM_s136 - s_k1_INTC_s136
                                 - s_k1_KO_s136 - s_k1_MAT_s136
                                 - s_k1_MSFT_s136 - s_k1_PG_s136  = 0
 k1_s137_balance_constraint:     b_k1_ADM_s137 + b_k1_AXP_s137 + b_k1_DIS_s137
                                 + b_k1_FDX_s137 + b_k1_IBM_s137
                                 + b_k1_INTC_s137 + b_k1_KO_s137
                                 + b_k1_MAT_s137 + b_k1_MSFT_s137
                                 + b_k1_PG_s137 - s_k1_ADM_s137 - s_k1_AXP_s137
                                 - s_k1_DIS_s137 - s_k1_FDX_s137
                                 - s_k1_IBM_s137 - s_k1_INTC_s137
                                 - s_k1_KO_s137 - s_k1_MAT_s137
                                 - s_k1_MSFT_s137 - s_k1_PG_s137  = 0
 k1_s140_balance_constraint:     b_k1_ADM_s140 + b_k1_AXP_s140 + b_k1_DIS_s140
                                 + b_k1_FDX_s140 + b_k1_IBM_s140
                                 + b_k1_INTC_s140 + b_k1_KO_s140
                                 + b_k1_MAT_s140 + b_k1_MSFT_s140
                                 + b_k1_PG_s140 - s_k1_ADM_s140 - s_k1_AXP_s140
                                 - s_k1_DIS_s140 - s_k1_FDX_s140
                                 - s_k1_IBM_s140 - s_k1_INTC_s140
                                 - s_k1_KO_s140 - s_k1_MAT_s140
                                 - s_k1_MSFT_s140 - s_k1_PG_s140  = 0
 k1_s141_balance_constraint:     b_k1_ADM_s141 + b_k1_AXP_s141 + b_k1_DIS_s141
                                 + b_k1_FDX_s141 + b_k1_IBM_s141
                                 + b_k1_INTC_s141 + b_k1_KO_s141
                                 + b_k1_MAT_s141 + b_k1_MSFT_s141
                                 + b_k1_PG_s141 - s_k1_ADM_s141 - s_k1_AXP_s141
                                 - s_k1_DIS_s141 - s_k1_FDX_s141
                                 - s_k1_IBM_s141 - s_k1_INTC_s141
                                 - s_k1_KO_s141 - s_k1_MAT_s141
                                 - s_k1_MSFT_s141 - s_k1_PG_s141  = 0
 k1_s142_balance_constraint:     b_k1_ADM_s142 + b_k1_AXP_s142 + b_k1_DIS_s142
                                 + b_k1_FDX_s142 + b_k1_IBM_s142
                                 + b_k1_INTC_s142 + b_k1_KO_s142
                                 + b_k1_MAT_s142 + b_k1_MSFT_s142
                                 + b_k1_PG_s142 - s_k1_ADM_s142 - s_k1_AXP_s142
                                 - s_k1_DIS_s142 - s_k1_FDX_s142
                                 - s_k1_IBM_s142 - s_k1_INTC_s142
                                 - s_k1_KO_s142 - s_k1_MAT_s142
                                 - s_k1_MSFT_s142 - s_k1_PG_s142  = 0
 k1_s143_balance_constraint:     b_k1_ADM_s143 + b_k1_AXP_s143 + b_k1_DIS_s143
                                 + b_k1_FDX_s143 + b_k1_IBM_s143
                                 + b_k1_INTC_s143 + b_k1_KO_s143
                                 + b_k1_MAT_s143 + b_k1_MSFT_s143
                                 + b_k1_PG_s143 - s_k1_ADM_s143 - s_k1_AXP_s143
                                 - s_k1_DIS_s143 - s_k1_FDX_s143
                                 - s_k1_IBM_s143 - s_k1_INTC_s143
                                 - s_k1_KO_s143 - s_k1_MAT_s143
                                 - s_k1_MSFT_s143 - s_k1_PG_s143  = 0
 k1_s144_balance_constraint:     b_k1_ADM_s144 + b_k1_AXP_s144 + b_k1_DIS_s144
                                 + b_k1_FDX_s144 + b_k1_IBM_s144
                                 + b_k1_INTC_s144 + b_k1_KO_s144
                                 + b_k1_MAT_s144 + b_k1_MSFT_s144
                                 + b_k1_PG_s144 - s_k1_ADM_s144 - s_k1_AXP_s144
                                 - s_k1_DIS_s144 - s_k1_FDX_s144
                                 - s_k1_IBM_s144 - s_k1_INTC_s144
                                 - s_k1_KO_s144 - s_k1_MAT_s144
                                 - s_k1_MSFT_s144 - s_k1_PG_s144  = 0
 k1_s145_balance_constraint:     b_k1_ADM_s145 + b_k1_AXP_s145 + b_k1_DIS_s145
                                 + b_k1_FDX_s145 + b_k1_IBM_s145
                                 + b_k1_INTC_s145 + b_k1_KO_s145
                                 + b_k1_MAT_s145 + b_k1_MSFT_s145
                                 + b_k1_PG_s145 - s_k1_ADM_s145 - s_k1_AXP_s145
                                 - s_k1_DIS_s145 - s_k1_FDX_s145
                                 - s_k1_IBM_s145 - s_k1_INTC_s145
                                 - s_k1_KO_s145 - s_k1_MAT_s145
                                 - s_k1_MSFT_s145 - s_k1_PG_s145  = 0
 k1_s146_balance_constraint:     b_k1_ADM_s146 + b_k1_AXP_s146 + b_k1_DIS_s146
                                 + b_k1_FDX_s146 + b_k1_IBM_s146
                                 + b_k1_INTC_s146 + b_k1_KO_s146
                                 + b_k1_MAT_s146 + b_k1_MSFT_s146
                                 + b_k1_PG_s146 - s_k1_ADM_s146 - s_k1_AXP_s146
                                 - s_k1_DIS_s146 - s_k1_FDX_s146
                                 - s_k1_IBM_s146 - s_k1_INTC_s146
                                 - s_k1_KO_s146 - s_k1_MAT_s146
                                 - s_k1_MSFT_s146 - s_k1_PG_s146  = 0
 k1_s147_balance_constraint:     b_k1_ADM_s147 + b_k1_AXP_s147 + b_k1_DIS_s147
                                 + b_k1_FDX_s147 + b_k1_IBM_s147
                                 + b_k1_INTC_s147 + b_k1_KO_s147
                                 + b_k1_MAT_s147 + b_k1_MSFT_s147
                                 + b_k1_PG_s147 - s_k1_ADM_s147 - s_k1_AXP_s147
                                 - s_k1_DIS_s147 - s_k1_FDX_s147
                                 - s_k1_IBM_s147 - s_k1_INTC_s147
                                 - s_k1_KO_s147 - s_k1_MAT_s147
                                 - s_k1_MSFT_s147 - s_k1_PG_s147  = 0
 k1_s150_balance_constraint:     b_k1_ADM_s150 + b_k1_AXP_s150 + b_k1_DIS_s150
                                 + b_k1_FDX_s150 + b_k1_IBM_s150
                                 + b_k1_INTC_s150 + b_k1_KO_s150
                                 + b_k1_MAT_s150 + b_k1_MSFT_s150
                                 + b_k1_PG_s150 - s_k1_ADM_s150 - s_k1_AXP_s150
                                 - s_k1_DIS_s150 - s_k1_FDX_s150
                                 - s_k1_IBM_s150 - s_k1_INTC_s150
                                 - s_k1_KO_s150 - s_k1_MAT_s150
                                 - s_k1_MSFT_s150 - s_k1_PG_s150  = 0
 k1_s151_balance_constraint:     b_k1_ADM_s151 + b_k1_AXP_s151 + b_k1_DIS_s151
                                 + b_k1_FDX_s151 + b_k1_IBM_s151
                                 + b_k1_INTC_s151 + b_k1_KO_s151
                                 + b_k1_MAT_s151 + b_k1_MSFT_s151
                                 + b_k1_PG_s151 - s_k1_ADM_s151 - s_k1_AXP_s151
                                 - s_k1_DIS_s151 - s_k1_FDX_s151
                                 - s_k1_IBM_s151 - s_k1_INTC_s151
                                 - s_k1_KO_s151 - s_k1_MAT_s151
                                 - s_k1_MSFT_s151 - s_k1_PG_s151  = 0
 k1_s152_balance_constraint:     b_k1_ADM_s152 + b_k1_AXP_s152 + b_k1_DIS_s152
                                 + b_k1_FDX_s152 + b_k1_IBM_s152
                                 + b_k1_INTC_s152 + b_k1_KO_s152
                                 + b_k1_MAT_s152 + b_k1_MSFT_s152
                                 + b_k1_PG_s152 - s_k1_ADM_s152 - s_k1_AXP_s152
                                 - s_k1_DIS_s152 - s_k1_FDX_s152
                                 - s_k1_IBM_s152 - s_k1_INTC_s152
                                 - s_k1_KO_s152 - s_k1_MAT_s152
                                 - s_k1_MSFT_s152 - s_k1_PG_s152  = 0
 k1_s153_balance_constraint:     b_k1_ADM_s153 + b_k1_AXP_s153 + b_k1_DIS_s153
                                 + b_k1_FDX_s153 + b_k1_IBM_s153
                                 + b_k1_INTC_s153 + b_k1_KO_s153
                                 + b_k1_MAT_s153 + b_k1_MSFT_s153
                                 + b_k1_PG_s153 - s_k1_ADM_s153 - s_k1_AXP_s153
                                 - s_k1_DIS_s153 - s_k1_FDX_s153
                                 - s_k1_IBM_s153 - s_k1_INTC_s153
                                 - s_k1_KO_s153 - s_k1_MAT_s153
                                 - s_k1_MSFT_s153 - s_k1_PG_s153  = 0
 k1_s154_balance_constraint:     b_k1_ADM_s154 + b_k1_AXP_s154 + b_k1_DIS_s154
                                 + b_k1_FDX_s154 + b_k1_IBM_s154
                                 + b_k1_INTC_s154 + b_k1_KO_s154
                                 + b_k1_MAT_s154 + b_k1_MSFT_s154
                                 + b_k1_PG_s154 - s_k1_ADM_s154 - s_k1_AXP_s154
                                 - s_k1_DIS_s154 - s_k1_FDX_s154
                                 - s_k1_IBM_s154 - s_k1_INTC_s154
                                 - s_k1_KO_s154 - s_k1_MAT_s154
                                 - s_k1_MSFT_s154 - s_k1_PG_s154  = 0
 k1_s155_balance_constraint:     b_k1_ADM_s155 + b_k1_AXP_s155 + b_k1_DIS_s155
                                 + b_k1_FDX_s155 + b_k1_IBM_s155
                                 + b_k1_INTC_s155 + b_k1_KO_s155
                                 + b_k1_MAT_s155 + b_k1_MSFT_s155
                                 + b_k1_PG_s155 - s_k1_ADM_s155 - s_k1_AXP_s155
                                 - s_k1_DIS_s155 - s_k1_FDX_s155
                                 - s_k1_IBM_s155 - s_k1_INTC_s155
                                 - s_k1_KO_s155 - s_k1_MAT_s155
                                 - s_k1_MSFT_s155 - s_k1_PG_s155  = 0
 k1_s156_balance_constraint:     b_k1_ADM_s156 + b_k1_AXP_s156 + b_k1_DIS_s156
                                 + b_k1_FDX_s156 + b_k1_IBM_s156
                                 + b_k1_INTC_s156 + b_k1_KO_s156
                                 + b_k1_MAT_s156 + b_k1_MSFT_s156
                                 + b_k1_PG_s156 - s_k1_ADM_s156 - s_k1_AXP_s156
                                 - s_k1_DIS_s156 - s_k1_FDX_s156
                                 - s_k1_IBM_s156 - s_k1_INTC_s156
                                 - s_k1_KO_s156 - s_k1_MAT_s156
                                 - s_k1_MSFT_s156 - s_k1_PG_s156  = 0
 k1_s157_balance_constraint:     b_k1_ADM_s157 + b_k1_AXP_s157 + b_k1_DIS_s157
                                 + b_k1_FDX_s157 + b_k1_IBM_s157
                                 + b_k1_INTC_s157 + b_k1_KO_s157
                                 + b_k1_MAT_s157 + b_k1_MSFT_s157
                                 + b_k1_PG_s157 - s_k1_ADM_s157 - s_k1_AXP_s157
                                 - s_k1_DIS_s157 - s_k1_FDX_s157
                                 - s_k1_IBM_s157 - s_k1_INTC_s157
                                 - s_k1_KO_s157 - s_k1_MAT_s157
                                 - s_k1_MSFT_s157 - s_k1_PG_s157  = 0
 k1_s160_balance_constraint:     b_k1_ADM_s160 + b_k1_AXP_s160 + b_k1_DIS_s160
                                 + b_k1_FDX_s160 + b_k1_IBM_s160
                                 + b_k1_INTC_s160 + b_k1_KO_s160
                                 + b_k1_MAT_s160 + b_k1_MSFT_s160
                                 + b_k1_PG_s160 - s_k1_ADM_s160 - s_k1_AXP_s160
                                 - s_k1_DIS_s160 - s_k1_FDX_s160
                                 - s_k1_IBM_s160 - s_k1_INTC_s160
                                 - s_k1_KO_s160 - s_k1_MAT_s160
                                 - s_k1_MSFT_s160 - s_k1_PG_s160  = 0
 k1_s161_balance_constraint:     b_k1_ADM_s161 + b_k1_AXP_s161 + b_k1_DIS_s161
                                 + b_k1_FDX_s161 + b_k1_IBM_s161
                                 + b_k1_INTC_s161 + b_k1_KO_s161
                                 + b_k1_MAT_s161 + b_k1_MSFT_s161
                                 + b_k1_PG_s161 - s_k1_ADM_s161 - s_k1_AXP_s161
                                 - s_k1_DIS_s161 - s_k1_FDX_s161
                                 - s_k1_IBM_s161 - s_k1_INTC_s161
                                 - s_k1_KO_s161 - s_k1_MAT_s161
                                 - s_k1_MSFT_s161 - s_k1_PG_s161  = 0
 k1_s162_balance_constraint:     b_k1_ADM_s162 + b_k1_AXP_s162 + b_k1_DIS_s162
                                 + b_k1_FDX_s162 + b_k1_IBM_s162
                                 + b_k1_INTC_s162 + b_k1_KO_s162
                                 + b_k1_MAT_s162 + b_k1_MSFT_s162
                                 + b_k1_PG_s162 - s_k1_ADM_s162 - s_k1_AXP_s162
                                 - s_k1_DIS_s162 - s_k1_FDX_s162
                                 - s_k1_IBM_s162 - s_k1_INTC_s162
                                 - s_k1_KO_s162 - s_k1_MAT_s162
                                 - s_k1_MSFT_s162 - s_k1_PG_s162  = 0
 k1_s163_balance_constraint:     b_k1_ADM_s163 + b_k1_AXP_s163 + b_k1_DIS_s163
                                 + b_k1_FDX_s163 + b_k1_IBM_s163
                                 + b_k1_INTC_s163 + b_k1_KO_s163
                                 + b_k1_MAT_s163 + b_k1_MSFT_s163
                                 + b_k1_PG_s163 - s_k1_ADM_s163 - s_k1_AXP_s163
                                 - s_k1_DIS_s163 - s_k1_FDX_s163
                                 - s_k1_IBM_s163 - s_k1_INTC_s163
                                 - s_k1_KO_s163 - s_k1_MAT_s163
                                 - s_k1_MSFT_s163 - s_k1_PG_s163  = 0
 k1_s164_balance_constraint:     b_k1_ADM_s164 + b_k1_AXP_s164 + b_k1_DIS_s164
                                 + b_k1_FDX_s164 + b_k1_IBM_s164
                                 + b_k1_INTC_s164 + b_k1_KO_s164
                                 + b_k1_MAT_s164 + b_k1_MSFT_s164
                                 + b_k1_PG_s164 - s_k1_ADM_s164 - s_k1_AXP_s164
                                 - s_k1_DIS_s164 - s_k1_FDX_s164
                                 - s_k1_IBM_s164 - s_k1_INTC_s164
                                 - s_k1_KO_s164 - s_k1_MAT_s164
                                 - s_k1_MSFT_s164 - s_k1_PG_s164  = 0
 k1_s165_balance_constraint:     b_k1_ADM_s165 + b_k1_AXP_s165 + b_k1_DIS_s165
                                 + b_k1_FDX_s165 + b_k1_IBM_s165
                                 + b_k1_INTC_s165 + b_k1_KO_s165
                                 + b_k1_MAT_s165 + b_k1_MSFT_s165
                                 + b_k1_PG_s165 - s_k1_ADM_s165 - s_k1_AXP_s165
                                 - s_k1_DIS_s165 - s_k1_FDX_s165
                                 - s_k1_IBM_s165 - s_k1_INTC_s165
                                 - s_k1_KO_s165 - s_k1_MAT_s165
                                 - s_k1_MSFT_s165 - s_k1_PG_s165  = 0
 k1_s166_balance_constraint:     b_k1_ADM_s166 + b_k1_AXP_s166 + b_k1_DIS_s166
                                 + b_k1_FDX_s166 + b_k1_IBM_s166
                                 + b_k1_INTC_s166 + b_k1_KO_s166
                                 + b_k1_MAT_s166 + b_k1_MSFT_s166
                                 + b_k1_PG_s166 - s_k1_ADM_s166 - s_k1_AXP_s166
                                 - s_k1_DIS_s166 - s_k1_FDX_s166
                                 - s_k1_IBM_s166 - s_k1_INTC_s166
                                 - s_k1_KO_s166 - s_k1_MAT_s166
                                 - s_k1_MSFT_s166 - s_k1_PG_s166  = 0
 k1_s167_balance_constraint:     b_k1_ADM_s167 + b_k1_AXP_s167 + b_k1_DIS_s167
                                 + b_k1_FDX_s167 + b_k1_IBM_s167
                                 + b_k1_INTC_s167 + b_k1_KO_s167
                                 + b_k1_MAT_s167 + b_k1_MSFT_s167
                                 + b_k1_PG_s167 - s_k1_ADM_s167 - s_k1_AXP_s167
                                 - s_k1_DIS_s167 - s_k1_FDX_s167
                                 - s_k1_IBM_s167 - s_k1_INTC_s167
                                 - s_k1_KO_s167 - s_k1_MAT_s167
                                 - s_k1_MSFT_s167 - s_k1_PG_s167  = 0
 k1_s170_balance_constraint:     b_k1_ADM_s170 + b_k1_AXP_s170 + b_k1_DIS_s170
                                 + b_k1_FDX_s170 + b_k1_IBM_s170
                                 + b_k1_INTC_s170 + b_k1_KO_s170
                                 + b_k1_MAT_s170 + b_k1_MSFT_s170
                                 + b_k1_PG_s170 - s_k1_ADM_s170 - s_k1_AXP_s170
                                 - s_k1_DIS_s170 - s_k1_FDX_s170
                                 - s_k1_IBM_s170 - s_k1_INTC_s170
                                 - s_k1_KO_s170 - s_k1_MAT_s170
                                 - s_k1_MSFT_s170 - s_k1_PG_s170  = 0
 k1_s171_balance_constraint:     b_k1_ADM_s171 + b_k1_AXP_s171 + b_k1_DIS_s171
                                 + b_k1_FDX_s171 + b_k1_IBM_s171
                                 + b_k1_INTC_s171 + b_k1_KO_s171
                                 + b_k1_MAT_s171 + b_k1_MSFT_s171
                                 + b_k1_PG_s171 - s_k1_ADM_s171 - s_k1_AXP_s171
                                 - s_k1_DIS_s171 - s_k1_FDX_s171
                                 - s_k1_IBM_s171 - s_k1_INTC_s171
                                 - s_k1_KO_s171 - s_k1_MAT_s171
                                 - s_k1_MSFT_s171 - s_k1_PG_s171  = 0
 k1_s172_balance_constraint:     b_k1_ADM_s172 + b_k1_AXP_s172 + b_k1_DIS_s172
                                 + b_k1_FDX_s172 + b_k1_IBM_s172
                                 + b_k1_INTC_s172 + b_k1_KO_s172
                                 + b_k1_MAT_s172 + b_k1_MSFT_s172
                                 + b_k1_PG_s172 - s_k1_ADM_s172 - s_k1_AXP_s172
                                 - s_k1_DIS_s172 - s_k1_FDX_s172
                                 - s_k1_IBM_s172 - s_k1_INTC_s172
                                 - s_k1_KO_s172 - s_k1_MAT_s172
                                 - s_k1_MSFT_s172 - s_k1_PG_s172  = 0
 k1_s173_balance_constraint:     b_k1_ADM_s173 + b_k1_AXP_s173 + b_k1_DIS_s173
                                 + b_k1_FDX_s173 + b_k1_IBM_s173
                                 + b_k1_INTC_s173 + b_k1_KO_s173
                                 + b_k1_MAT_s173 + b_k1_MSFT_s173
                                 + b_k1_PG_s173 - s_k1_ADM_s173 - s_k1_AXP_s173
                                 - s_k1_DIS_s173 - s_k1_FDX_s173
                                 - s_k1_IBM_s173 - s_k1_INTC_s173
                                 - s_k1_KO_s173 - s_k1_MAT_s173
                                 - s_k1_MSFT_s173 - s_k1_PG_s173  = 0
 k1_s174_balance_constraint:     b_k1_ADM_s174 + b_k1_AXP_s174 + b_k1_DIS_s174
                                 + b_k1_FDX_s174 + b_k1_IBM_s174
                                 + b_k1_INTC_s174 + b_k1_KO_s174
                                 + b_k1_MAT_s174 + b_k1_MSFT_s174
                                 + b_k1_PG_s174 - s_k1_ADM_s174 - s_k1_AXP_s174
                                 - s_k1_DIS_s174 - s_k1_FDX_s174
                                 - s_k1_IBM_s174 - s_k1_INTC_s174
                                 - s_k1_KO_s174 - s_k1_MAT_s174
                                 - s_k1_MSFT_s174 - s_k1_PG_s174  = 0
 k1_s175_balance_constraint:     b_k1_ADM_s175 + b_k1_AXP_s175 + b_k1_DIS_s175
                                 + b_k1_FDX_s175 + b_k1_IBM_s175
                                 + b_k1_INTC_s175 + b_k1_KO_s175
                                 + b_k1_MAT_s175 + b_k1_MSFT_s175
                                 + b_k1_PG_s175 - s_k1_ADM_s175 - s_k1_AXP_s175
                                 - s_k1_DIS_s175 - s_k1_FDX_s175
                                 - s_k1_IBM_s175 - s_k1_INTC_s175
                                 - s_k1_KO_s175 - s_k1_MAT_s175
                                 - s_k1_MSFT_s175 - s_k1_PG_s175  = 0
 k1_s176_balance_constraint:     b_k1_ADM_s176 + b_k1_AXP_s176 + b_k1_DIS_s176
                                 + b_k1_FDX_s176 + b_k1_IBM_s176
                                 + b_k1_INTC_s176 + b_k1_KO_s176
                                 + b_k1_MAT_s176 + b_k1_MSFT_s176
                                 + b_k1_PG_s176 - s_k1_ADM_s176 - s_k1_AXP_s176
                                 - s_k1_DIS_s176 - s_k1_FDX_s176
                                 - s_k1_IBM_s176 - s_k1_INTC_s176
                                 - s_k1_KO_s176 - s_k1_MAT_s176
                                 - s_k1_MSFT_s176 - s_k1_PG_s176  = 0
 k1_s177_balance_constraint:     b_k1_ADM_s177 + b_k1_AXP_s177 + b_k1_DIS_s177
                                 + b_k1_FDX_s177 + b_k1_IBM_s177
                                 + b_k1_INTC_s177 + b_k1_KO_s177
                                 + b_k1_MAT_s177 + b_k1_MSFT_s177
                                 + b_k1_PG_s177 - s_k1_ADM_s177 - s_k1_AXP_s177
                                 - s_k1_DIS_s177 - s_k1_FDX_s177
                                 - s_k1_IBM_s177 - s_k1_INTC_s177
                                 - s_k1_KO_s177 - s_k1_MAT_s177
                                 - s_k1_MSFT_s177 - s_k1_PG_s177  = 0
 k1_return_constraint:           opt_return - 0.00332514109910791 w_k1_ADM_s000
                                 - 0.00493561240300334 w_k1_ADM_s001
                                 - 0.00349821283988356 w_k1_ADM_s002
                                 - 0.00406856575087739 w_k1_ADM_s003
                                 - 0.00306280302736255 w_k1_ADM_s004
                                 - 0.00229770389909392 w_k1_ADM_s005
                                 - 0.000907473374646728 w_k1_ADM_s006
                                 - 0.00195073660355908 w_k1_ADM_s007
                                 - 0.00621833524377228 w_k1_ADM_s010
                                 - 0.00662791927608081 w_k1_ADM_s011
                                 - 0.00318140081605023 w_k1_ADM_s012
                                 - 0.00440294209895736 w_k1_ADM_s013
                                 - 0.00156681847972958 w_k1_ADM_s014
                                 - 0.00305009283394256 w_k1_ADM_s015
                                 - 0.00272501336498476 w_k1_ADM_s016
                                 - 0.00186579588131203 w_k1_ADM_s017
                                 - 0.00273393523298989 w_k1_ADM_s020
                                 - 0.00645440803039008 w_k1_ADM_s021
                                 - 0.0037138458092465 w_k1_ADM_s022
                                 - 0.00343654773896111 w_k1_ADM_s023
                                 - 0.00711515920803853 w_k1_ADM_s024
                                 - 0.00285938271201873 w_k1_ADM_s025
                                 - 0.00484051008558628 w_k1_ADM_s026
                                 - 0.00777015846597449 w_k1_ADM_s027
                                 - 0.00584573425595834 w_k1_ADM_s030
                                 - 0.00279379688949927 w_k1_ADM_s031
                                 - 0.00908903746542857 w_k1_ADM_s032
                                 - 0.0150941612846594 w_k1_ADM_s033
                                 - 0.00367656510501436 w_k1_ADM_s034
                                 - 0.00762897586577895 w_k1_ADM_s035
                                 - 0.00592494273561202 w_k1_ADM_s036
                                 - 0.0037948239149229 w_k1_ADM_s037
                                 - 0.0032917208462964 w_k1_ADM_s040
                                 - 0.00359283172604265 w_k1_ADM_s041
                                 - 0.00999350739383103 w_k1_ADM_s042
                                 - 0.00552523340215233 w_k1_ADM_s043
                                 - 0.00545098874155047 w_k1_ADM_s044
                                 - 0.00715202609454136 w_k1_ADM_s045
                                 - 0.00525952861373695 w_k1_ADM_s046
                                 - 0.00737066847915222 w_k1_ADM_s047
                                 - 0.00463139756832746 w_k1_ADM_s050
                                 - 0.0200600603482544 w_k1_ADM_s051
                                 - 0.0107571244439092 w_k1_ADM_s052
                                 - 0.0202453104378492 w_k1_ADM_s053
                                 - 0.0211415832970278 w_k1_ADM_s054
                                 - 0.0114432070934705 w_k1_ADM_s055
                                 - 0.0129579407575958 w_k1_ADM_s056
                                 - 0.0121935067538652 w_k1_ADM_s057
                                 - 0.00687980014066564 w_k1_ADM_s060
                                 - 0.00608187940353442 w_k1_ADM_s061
                                 - 0.00401319100976117 w_k1_ADM_s062
                                 - 0.00656979049203644 w_k1_ADM_s063
                                 - 0.0150180138938815 w_k1_ADM_s064
                                 - 0.00742150611071495 w_k1_ADM_s065
                                 - 0.0109743428625001 w_k1_ADM_s066
                                 - 0.0090084074918465 w_k1_ADM_s067
                                 - 0.0076120941962821 w_k1_ADM_s070
                                 - 0.00474637388827811 w_k1_ADM_s071
                                 - 0.0104146368670093 w_k1_ADM_s072
                                 - 0.00880461239922553 w_k1_ADM_s073
                                 - 0.0163881757795983 w_k1_ADM_s074
                                 - 0.00356920078557303 w_k1_ADM_s075
                                 - 0.00669146235803799 w_k1_ADM_s076
                                 - 0.00575068595937512 w_k1_ADM_s077
                                 - 0.00709838255272038 w_k1_ADM_s100
                                 - 0.0102088287676422 w_k1_ADM_s101
                                 - 0.00597500265728758 w_k1_ADM_s102
                                 - 0.00823528834096448 w_k1_ADM_s103
                                 - 0.0140306648454872 w_k1_ADM_s104
                                 - 0.0143170587038547 w_k1_ADM_s105
                                 - 0.0169220097108832 w_k1_ADM_s106
                                 - 0.0238570374803567 w_k1_ADM_s107
                                 - 0.0122188560412143 w_k1_ADM_s110
                                 - 0.00467408894807954 w_k1_ADM_s111
                                 - 0.00939648010440941 w_k1_ADM_s112
                                 - 0.0110221183666951 w_k1_ADM_s113
                                 - 0.00364459522639201 w_k1_ADM_s114
                                 - 0.0193274273426054 w_k1_ADM_s115
                                 - 0.0037180232135661 w_k1_ADM_s116
                                 - 0.00575872957606755 w_k1_ADM_s117
                                 - 0.00267032147638217 w_k1_ADM_s120
                                 - 0.0066209485288361 w_k1_ADM_s121
                                 - 0.00193930628736105 w_k1_ADM_s122
                                 - 0.00156692887891904 w_k1_ADM_s123
                                 - 0.00219929576342457 w_k1_ADM_s124
                                 - 0.00274730729921163 w_k1_ADM_s125
                                 - 0.00292838260913309 w_k1_ADM_s126
                                 - 0.0024892525706666 w_k1_ADM_s127
                                 - 0.0161276747237419 w_k1_ADM_s130
                                 - 0.00635556867429909 w_k1_ADM_s131
                                 - 0.0100106244965467 w_k1_ADM_s132
                                 - 0.00639768788433807 w_k1_ADM_s133
                                 - 0.0176602337856238 w_k1_ADM_s134
                                 - 0.0079147997613205 w_k1_ADM_s135
                                 - 0.0171557814169813 w_k1_ADM_s136
                                 - 0.00816475135840565 w_k1_ADM_s137
                                 - 0.00287073833466797 w_k1_ADM_s140
                                 - 0.00321660472129426 w_k1_ADM_s141
                                 - 0.00734881381792833 w_k1_ADM_s142
                                 - 0.00825138592025094 w_k1_ADM_s143
                                 - 0.00588907716417939 w_k1_ADM_s144
                                 - 0.00286855849909746 w_k1_ADM_s145
                                 - 0.0086338715666003 w_k1_ADM_s146
                                 - 0.00210495805646607 w_k1_ADM_s147
                                 - 0.0139226376509549 w_k1_ADM_s150
                                 - 0.0208436031797756 w_k1_ADM_s151
                                 - 0.00899212447719229 w_k1_ADM_s152
                                 - 0.00691930468093833 w_k1_ADM_s153
                                 - 0.0106296752383474 w_k1_ADM_s154
                                 - 0.0246060075474004 w_k1_ADM_s155
                                 - 0.01298899757878 w_k1_ADM_s156
                                 - 0.0122920729592749 w_k1_ADM_s157
                                 - 0.012598001461421 w_k1_ADM_s160
                                 - 0.011607857032257 w_k1_ADM_s161
                                 - 0.00781466540595483 w_k1_ADM_s162
                                 - 0.00347188005793622 w_k1_ADM_s163
                                 - 0.00613701577872353 w_k1_ADM_s164
                                 - 0.0144456696556354 w_k1_ADM_s165
                                 - 0.0120489487548629 w_k1_ADM_s166
                                 - 0.00505425094084384 w_k1_ADM_s167
                                 - 0.00964142861426103 w_k1_ADM_s170
                                 - 0.00555672144426855 w_k1_ADM_s171
                                 - 0.00675189208405069 w_k1_ADM_s172
                                 - 0.00529650721224094 w_k1_ADM_s173
                                 - 0.00954159278765297 w_k1_ADM_s174
                                 - 0.00896560188199605 w_k1_ADM_s175
                                 - 0.00558447353997792 w_k1_ADM_s176
                                 - 0.0100343311662364 w_k1_ADM_s177
                                 - 0.00309001786375912 w_k1_AXP_s000
                                 - 0.00502591418462027 w_k1_AXP_s001
                                 - 0.00311777316455463 w_k1_AXP_s002
                                 - 0.00349553681715963 w_k1_AXP_s003
                                 - 0.00326178450224944 w_k1_AXP_s004
                                 - 0.00219256469559166 w_k1_AXP_s005
                                 - 0.00114336502548182 w_k1_AXP_s006
                                 - 0.00235371974017009 w_k1_AXP_s007
                                 - 0.00589195116431804 w_k1_AXP_s010
                                 - 0.00604658027699895 w_k1_AXP_s011
                                 - 0.0037911954216152 w_k1_AXP_s012
                                 - 0.00377498014550588 w_k1_AXP_s013
                                 - 0.00151720981714506 w_k1_AXP_s014
                                 - 0.00349510231100443 w_k1_AXP_s015
                                 - 0.00259493916470979 w_k1_AXP_s016
                                 - 0.00228332173912941 w_k1_AXP_s017
                                 - 0.00316548846296473 w_k1_AXP_s020
                                 - 0.00472316604251687 w_k1_AXP_s021
                                 - 0.00405820655471327 w_k1_AXP_s022
                                 - 0.00352177610947416 w_k1_AXP_s023
                                 - 0.00730925900242235 w_k1_AXP_s024
                                 - 0.00290481326240488 w_k1_AXP_s025
                                 - 0.0045692073155044 w_k1_AXP_s026
                                 - 0.00784518425396779 w_k1_AXP_s027
                                 - 0.00530295031807513 w_k1_AXP_s030
                                 - 0.00303042323789326 w_k1_AXP_s031
                                 - 0.00932877912040411 w_k1_AXP_s032
                                 - 0.0149872692490268 w_k1_AXP_s033
                                 - 0.00345447213570387 w_k1_AXP_s034
                                 - 0.00892881402987968 w_k1_AXP_s035
                                 - 0.00559822409092753 w_k1_AXP_s036
                                 - 0.00382934350936102 w_k1_AXP_s037
                                 - 0.00383487638291902 w_k1_AXP_s040
                                 - 0.00433812028561357 w_k1_AXP_s041
                                 - 0.00930295646027085 w_k1_AXP_s042
                                 - 0.00589693850824935 w_k1_AXP_s043
                                 - 0.0056800576616697 w_k1_AXP_s044
                                 - 0.00849419667011303 w_k1_AXP_s045
                                 - 0.00516045984615076 w_k1_AXP_s046
                                 - 0.0071816229249298 w_k1_AXP_s047
                                 - 0.0043535565953454 w_k1_AXP_s050
                                 - 0.02074396878508 w_k1_AXP_s051
                                 - 0.0118555268219653 w_k1_AXP_s052
                                 - 0.0218282186762164 w_k1_AXP_s053
                                 - 0.0164826676851045 w_k1_AXP_s054
                                 - 0.0099658262775277 w_k1_AXP_s055
                                 - 0.0119873089564252 w_k1_AXP_s056
                                 - 0.0130083128677708 w_k1_AXP_s057
                                 - 0.00788038850653303 w_k1_AXP_s060
                                 - 0.0058829779487828 w_k1_AXP_s061
                                 - 0.00476166055871923 w_k1_AXP_s062
                                 - 0.00575364336797858 w_k1_AXP_s063
                                 - 0.0166003803025915 w_k1_AXP_s064
                                 - 0.00650329393878379 w_k1_AXP_s065
                                 - 0.010755460560495 w_k1_AXP_s066
                                 - 0.00852523800610536 w_k1_AXP_s067
                                 - 0.00816785698188303 w_k1_AXP_s070
                                 - 0.00395217001251138 w_k1_AXP_s071
                                 - 0.012735240668704 w_k1_AXP_s072
                                 - 0.00900272632697753 w_k1_AXP_s073
                                 - 0.0137898859181066 w_k1_AXP_s074
                                 - 0.00427167222338075 w_k1_AXP_s075
                                 - 0.00912224690210381 w_k1_AXP_s076
                                 - 0.0070873365643329 w_k1_AXP_s077
                                 - 0.00739109836111482 w_k1_AXP_s100
                                 - 0.0111214956159672 w_k1_AXP_s101
                                 - 0.00551742046084432 w_k1_AXP_s102
                                 - 0.00915042549471821 w_k1_AXP_s103
                                 - 0.0148085093238858 w_k1_AXP_s104
                                 - 0.0145259427953694 w_k1_AXP_s105
                                 - 0.0172820051833629 w_k1_AXP_s106
                                 - 0.0235261156149583 w_k1_AXP_s107
                                 - 0.0117747856864046 w_k1_AXP_s110
                                 - 0.00451276045736089 w_k1_AXP_s111
                                 - 0.0069648928845565 w_k1_AXP_s112
                                 - 0.0104162787364281 w_k1_AXP_s113
                                 - 0.00371683511023488 w_k1_AXP_s114
                                 - 0.0181996607333899 w_k1_AXP_s115
                                 - 0.00362428289996084 w_k1_AXP_s116
                                 - 0.00711696830119338 w_k1_AXP_s117
                                 - 0.00311985009083297 w_k1_AXP_s120
                                 - 0.00594547586465286 w_k1_AXP_s121
                                 - 0.00211865043127071 w_k1_AXP_s122
                                 - 0.00147683129150436 w_k1_AXP_s123
                                 - 0.00266674576606193 w_k1_AXP_s124
                                 - 0.00267632163673916 w_k1_AXP_s125
                                 - 0.00230519042038966 w_k1_AXP_s126
                                 - 0.00282009856108055 w_k1_AXP_s127
                                 - 0.0155953385550713 w_k1_AXP_s130
                                 - 0.0055436851246558 w_k1_AXP_s131
                                 - 0.00983817808692928 w_k1_AXP_s132
                                 - 0.00669283628686064 w_k1_AXP_s133
                                 - 0.0179476291122862 w_k1_AXP_s134
                                 - 0.00597137746030613 w_k1_AXP_s135
                                 - 0.0170292396939951 w_k1_AXP_s136
                                 - 0.0107754386926279 w_k1_AXP_s137
                                 - 0.00310404226739705 w_k1_AXP_s140
                                 - 0.00399660768868085 w_k1_AXP_s141
                                 - 0.0091478236216514 w_k1_AXP_s142
                                 - 0.00813723139314412 w_k1_AXP_s143
                                 - 0.00591908278724231 w_k1_AXP_s144
                                 - 0.00356996604091175 w_k1_AXP_s145
                                 - 0.00927206400205793 w_k1_AXP_s146
                                 - 0.00217829522673608 w_k1_AXP_s147
                                 - 0.0137099816041474 w_k1_AXP_s150
                                 - 0.0174243682628259 w_k1_AXP_s151
                                 - 0.00895434127825177 w_k1_AXP_s152
                                 - 0.00600254424489049 w_k1_AXP_s153
                                 - 0.0109939159648241 w_k1_AXP_s154
                                 - 0.0218109388199507 w_k1_AXP_s155
                                 - 0.0144713206413128 w_k1_AXP_s156
                                 - 0.0110781964816166 w_k1_AXP_s157
                                 - 0.0121350708959379 w_k1_AXP_s160
                                 - 0.0119830974152353 w_k1_AXP_s161
                                 - 0.00634976402240895 w_k1_AXP_s162
                                 - 0.00352066218095012 w_k1_AXP_s163
                                 - 0.00700595375237096 w_k1_AXP_s164
                                 - 0.0170091953027334 w_k1_AXP_s165
                                 - 0.0131225470654752 w_k1_AXP_s166
                                 - 0.00532806110492348 w_k1_AXP_s167
                                 - 0.0104227253308406 w_k1_AXP_s170
                                 - 0.00580081800551822 w_k1_AXP_s171
                                 - 0.00573927668133162 w_k1_AXP_s172
                                 - 0.00574017956777358 w_k1_AXP_s173
                                 - 0.0101409451362584 w_k1_AXP_s174
                                 - 0.0101420826768723 w_k1_AXP_s175
                                 - 0.00715631742370501 w_k1_AXP_s176
                                 - 0.0113982681768066 w_k1_AXP_s177
                                 - 0.0036227411383175 w_k1_DIS_s000
                                 - 0.00483633825386752 w_k1_DIS_s001
                                 - 0.00347918700879508 w_k1_DIS_s002
                                 - 0.00388552668946068 w_k1_DIS_s003
                                 - 0.00295188985369841 w_k1_DIS_s004
                                 - 0.00213678488000856 w_k1_DIS_s005
                                 - 0.0010680509898406 w_k1_DIS_s006
                                 - 0.00231728484047317 w_k1_DIS_s007
                                 - 0.00632258066179614 w_k1_DIS_s010
                                 - 0.0062094074248551 w_k1_DIS_s011
                                 - 0.00399974805930937 w_k1_DIS_s012
                                 - 0.00392737057082088 w_k1_DIS_s013
                                 - 0.00163080166244368 w_k1_DIS_s014
                                 - 0.00320258444988375 w_k1_DIS_s015
                                 - 0.00268784248638508 w_k1_DIS_s016
                                 - 0.00212858528802838 w_k1_DIS_s017
                                 - 0.00332302416785098 w_k1_DIS_s020
                                 - 0.00540506649505107 w_k1_DIS_s021
                                 - 0.00411703494448307 w_k1_DIS_s022
                                 - 0.00322726372868943 w_k1_DIS_s023
                                 - 0.00685797443290816 w_k1_DIS_s024
                                 - 0.00319477082257865 w_k1_DIS_s025
                                 - 0.00464873489369021 w_k1_DIS_s026
                                 - 0.00844235373282669 w_k1_DIS_s027
                                 - 0.00536799882234024 w_k1_DIS_s030
                                 - 0.00331804022046159 w_k1_DIS_s031
                                 - 0.00987883602253851 w_k1_DIS_s032
                                 - 0.0143763802637317 w_k1_DIS_s033
                                 - 0.0034386510383228 w_k1_DIS_s034
                                 - 0.00879898616324311 w_k1_DIS_s035
                                 - 0.00555782989893462 w_k1_DIS_s036
                                 - 0.00383994234033086 w_k1_DIS_s037
                                 - 0.00390714611848241 w_k1_DIS_s040
                                 - 0.00454109488746455 w_k1_DIS_s041
                                 - 0.00953171718452177 w_k1_DIS_s042
                                 - 0.00559739994648697 w_k1_DIS_s043
                                 - 0.00481978397669972 w_k1_DIS_s044
                                 - 0.00767859582740105 w_k1_DIS_s045
                                 - 0.00519821538395709 w_k1_DIS_s046
                                 - 0.00800258430410911 w_k1_DIS_s047
                                 - 0.00459290852888676 w_k1_DIS_s050
                                 - 0.0189882720784712 w_k1_DIS_s051
                                 - 0.0119965024307089 w_k1_DIS_s052
                                 - 0.0235082385622001 w_k1_DIS_s053
                                 - 0.0174033793542554 w_k1_DIS_s054
                                 - 0.0103994275862323 w_k1_DIS_s055
                                 - 0.0115312314429823 w_k1_DIS_s056
                                 - 0.0123511486303086 w_k1_DIS_s057
                                 - 0.00783400690305233 w_k1_DIS_s060
                                 - 0.00624179140511344 w_k1_DIS_s061
                                 - 0.00488552778861237 w_k1_DIS_s062
                                 - 0.0065213991450523 w_k1_DIS_s063
                                 - 0.0159243467526948 w_k1_DIS_s064
                                 - 0.00652737511974736 w_k1_DIS_s065
                                 - 0.0121584308942747 w_k1_DIS_s066
                                 - 0.00992458390354858 w_k1_DIS_s067
                                 - 0.00820017233843385 w_k1_DIS_s070
                                 - 0.00469817926812883 w_k1_DIS_s071
                                 - 0.0127649178122418 w_k1_DIS_s072
                                 - 0.00906374241440369 w_k1_DIS_s073
                                 - 0.0149879709385346 w_k1_DIS_s074
                                 - 0.00457784695487183 w_k1_DIS_s075
                                 - 0.00839037725462508 w_k1_DIS_s076
                                 - 0.00658418120018401 w_k1_DIS_s077
                                 - 0.00736467121854487 w_k1_DIS_s100
                                 - 0.0120202714868249 w_k1_DIS_s101
                                 - 0.00648651124345144 w_k1_DIS_s102
                                 - 0.0100371431532415 w_k1_DIS_s103
                                 - 0.0152943585271369 w_k1_DIS_s104
                                 - 0.0143538401277132 w_k1_DIS_s105
                                 - 0.0173296288308646 w_k1_DIS_s106
                                 - 0.0248600571608394 w_k1_DIS_s107
                                 - 0.0127052203998612 w_k1_DIS_s110
                                 - 0.00513459329433142 w_k1_DIS_s111
                                 - 0.00787849006918412 w_k1_DIS_s112
                                 - 0.0104573368564002 w_k1_DIS_s113
                                 - 0.00390391416347942 w_k1_DIS_s114
                                 - 0.018010919647316 w_k1_DIS_s115
                                 - 0.00404819060663983 w_k1_DIS_s116
                                 - 0.00685827893301599 w_k1_DIS_s117
                                 - 0.00286939594529879 w_k1_DIS_s120
                                 - 0.00597749257545735 w_k1_DIS_s121
                                 - 0.00222030870348173 w_k1_DIS_s122
                                 - 0.00163970868048631 w_k1_DIS_s123
                                 - 0.0024315710319631 w_k1_DIS_s124
                                 - 0.00288486740319642 w_k1_DIS_s125
                                 - 0.00210129571260255 w_k1_DIS_s126
                                 - 0.00258264522471586 w_k1_DIS_s127
                                 - 0.0153991136986125 w_k1_DIS_s130
                                 - 0.00613215389544781 w_k1_DIS_s131
                                 - 0.0107588754621219 w_k1_DIS_s132
                                 - 0.00636674155869713 w_k1_DIS_s133
                                 - 0.0183056215977225 w_k1_DIS_s134
                                 - 0.007563799356216 w_k1_DIS_s135
                                 - 0.019255161949537 w_k1_DIS_s136
                                 - 0.0109251577055621 w_k1_DIS_s137
                                 - 0.00344230133966671 w_k1_DIS_s140
                                 - 0.00355681302820786 w_k1_DIS_s141
                                 - 0.00813469021196697 w_k1_DIS_s142
                                 - 0.00939972334300503 w_k1_DIS_s143
                                 - 0.00671802393462684 w_k1_DIS_s144
                                 - 0.00344954526293066 w_k1_DIS_s145
                                 - 0.0105843201357591 w_k1_DIS_s146
                                 - 0.0023852266600613 w_k1_DIS_s147
                                 - 0.0138297722152754 w_k1_DIS_s150
                                 - 0.0173360981815102 w_k1_DIS_s151
                                 - 0.00934670850793811 w_k1_DIS_s152
                                 - 0.0071835549349148 w_k1_DIS_s153
                                 - 0.0116751843702844 w_k1_DIS_s154
                                 - 0.0241324804323661 w_k1_DIS_s155
                                 - 0.0127272302377797 w_k1_DIS_s156
                                 - 0.01210556441369 w_k1_DIS_s157
                                 - 0.0122511890578931 w_k1_DIS_s160
                                 - 0.011531523453159 w_k1_DIS_s161
                                 - 0.00704518250671445 w_k1_DIS_s162
                                 - 0.00338095395441091 w_k1_DIS_s163
                                 - 0.00674110558558774 w_k1_DIS_s164
                                 - 0.0164462273184434 w_k1_DIS_s165
                                 - 0.0116905555250112 w_k1_DIS_s166
                                 - 0.00517787935295642 w_k1_DIS_s167
                                 - 0.0100160712985689 w_k1_DIS_s170
                                 - 0.00533094443371868 w_k1_DIS_s171
                                 - 0.00670435895348071 w_k1_DIS_s172
                                 - 0.00585551680733901 w_k1_DIS_s173
                                 - 0.011636406322 w_k1_DIS_s174
                                 - 0.0111171138834278 w_k1_DIS_s175
                                 - 0.00643000613042078 w_k1_DIS_s176
                                 - 0.0126174388888666 w_k1_DIS_s177
                                 - 0.0033609240595714 w_k1_FDX_s000
                                 - 0.00484042701958973 w_k1_FDX_s001
                                 - 0.00305484987843965 w_k1_FDX_s002
                                 - 0.0038056059452967 w_k1_FDX_s003
                                 - 0.00276658111932811 w_k1_FDX_s004
                                 - 0.00210120336921677 w_k1_FDX_s005
                                 - 0.00106322907096028 w_k1_FDX_s006
                                 - 0.00233780149820333 w_k1_FDX_s007
                                 - 0.00593949088973108 w_k1_FDX_s010
                                 - 0.00648946366069393 w_k1_FDX_s011
                                 - 0.00408660225427634 w_k1_FDX_s012
                                 - 0.00372112987837403 w_k1_FDX_s013
                                 - 0.00160465090269491 w_k1_FDX_s014
                                 - 0.00299600451623229 w_k1_FDX_s015
                                 - 0.00261320854238386 w_k1_FDX_s016
                                 - 0.00199850572670529 w_k1_FDX_s017
                                 - 0.00305719372438647 w_k1_FDX_s020
                                 - 0.00539911180185192 w_k1_FDX_s021
                                 - 0.00400726046166248 w_k1_FDX_s022
                                 - 0.0034280176940724 w_k1_FDX_s023
                                 - 0.00710691306126722 w_k1_FDX_s024
                                 - 0.00296187808309776 w_k1_FDX_s025
                                 - 0.00481410755103105 w_k1_FDX_s026
                                 - 0.00817464319910244 w_k1_FDX_s027
                                 - 0.00518143597300228 w_k1_FDX_s030
                                 - 0.00308699081836435 w_k1_FDX_s031
                                 - 0.00973365765797178 w_k1_FDX_s032
                                 - 0.0143371559151488 w_k1_FDX_s033
                                 - 0.00340252249384767 w_k1_FDX_s034
                                 - 0.00840917460635642 w_k1_FDX_s035
                                 - 0.00550251979107407 w_k1_FDX_s036
                                 - 0.00353795173798994 w_k1_FDX_s037
                                 - 0.00375258236279952 w_k1_FDX_s040
                                 - 0.00414563801404337 w_k1_FDX_s041
                                 - 0.00927575931108496 w_k1_FDX_s042
                                 - 0.00592866676418843 w_k1_FDX_s043
                                 - 0.0052691778642106 w_k1_FDX_s044
                                 - 0.00753909914186145 w_k1_FDX_s045
                                 - 0.00520249244500112 w_k1_FDX_s046
                                 - 0.00802315122013558 w_k1_FDX_s047
                                 - 0.00473152639566326 w_k1_FDX_s050
                                 - 0.0182825823836225 w_k1_FDX_s051
                                 - 0.0116229756595113 w_k1_FDX_s052
                                 - 0.0210010530340403 w_k1_FDX_s053
                                 - 0.0177065849149557 w_k1_FDX_s054
                                 - 0.0100890684395138 w_k1_FDX_s055
                                 - 0.0119082836709825 w_k1_FDX_s056
                                 - 0.0128634378512541 w_k1_FDX_s057
                                 - 0.00776599135403022 w_k1_FDX_s060
                                 - 0.0064340716606466 w_k1_FDX_s061
                                 - 0.00471469306667894 w_k1_FDX_s062
                                 - 0.00604001765577153 w_k1_FDX_s063
                                 - 0.0159885554160192 w_k1_FDX_s064
                                 - 0.00662085644043672 w_k1_FDX_s065
                                 - 0.0113154151056286 w_k1_FDX_s066
                                 - 0.00923970654467824 w_k1_FDX_s067
                                 - 0.00807105461856934 w_k1_FDX_s070
                                 - 0.00417736257381855 w_k1_FDX_s071
                                 - 0.0121453092711645 w_k1_FDX_s072
                                 - 0.0096432412524104 w_k1_FDX_s073
                                 - 0.0158247125126404 w_k1_FDX_s074
                                 - 0.00436744978265 w_k1_FDX_s075
                                 - 0.00797757656189567 w_k1_FDX_s076
                                 - 0.0061594347433374 w_k1_FDX_s077
                                 - 0.00723516261894966 w_k1_FDX_s100
                                 - 0.011093799412481 w_k1_FDX_s101
                                 - 0.00578982569581778 w_k1_FDX_s102
                                 - 0.00929291665222604 w_k1_FDX_s103
                                 - 0.0151941718204893 w_k1_FDX_s104
                                 - 0.0139948390612931 w_k1_FDX_s105
                                 - 0.0181277240057874 w_k1_FDX_s106
                                 - 0.0242021335360836 w_k1_FDX_s107
                                 - 0.0129307049264167 w_k1_FDX_s110
                                 - 0.00482826291367289 w_k1_FDX_s111
                                 - 0.00763829826624938 w_k1_FDX_s112
                                 - 0.0102574680923825 w_k1_FDX_s113
                                 - 0.00361031958100864 w_k1_FDX_s114
                                 - 0.0172388638521127 w_k1_FDX_s115
                                 - 0.00373505749929395 w_k1_FDX_s116
                                 - 0.00675849700491321 w_k1_FDX_s117
                                 - 0.00266412387402406 w_k1_FDX_s120
                                 - 0.00599849528916271 w_k1_FDX_s121
                                 - 0.00223016040222467 w_k1_FDX_s122
                                 - 0.00164696116019035 w_k1_FDX_s123
                                 - 0.00235694902461892 w_k1_FDX_s124
                                 - 0.00274179117628566 w_k1_FDX_s125
                                 - 0.00207651274509263 w_k1_FDX_s126
                                 - 0.00247012559284567 w_k1_FDX_s127
                                 - 0.0151950640191547 w_k1_FDX_s130
                                 - 0.00609166042581679 w_k1_FDX_s131
                                 - 0.010529924598218 w_k1_FDX_s132
                                 - 0.00644805014607868 w_k1_FDX_s133
                                 - 0.0179706715753417 w_k1_FDX_s134
                                 - 0.00693772933381564 w_k1_FDX_s135
                                 - 0.0171038702039442 w_k1_FDX_s136
                                 - 0.0101890096852737 w_k1_FDX_s137
                                 - 0.0032985941924352 w_k1_FDX_s140
                                 - 0.00362871626469134 w_k1_FDX_s141
                                 - 0.00853365847290139 w_k1_FDX_s142
                                 - 0.00855255216120294 w_k1_FDX_s143
                                 - 0.00617976854008361 w_k1_FDX_s144
                                 - 0.00334205575245418 w_k1_FDX_s145
                                 - 0.00958955368084537 w_k1_FDX_s146
                                 - 0.00226692795798316 w_k1_FDX_s147
                                 - 0.0141959405641993 w_k1_FDX_s150
                                 - 0.0175191252871303 w_k1_FDX_s151
                                 - 0.00942614152667447 w_k1_FDX_s152
                                 - 0.0066955108573671 w_k1_FDX_s153
                                 - 0.0109446759190978 w_k1_FDX_s154
                                 - 0.022543614040788 w_k1_FDX_s155
                                 - 0.0133896143808286 w_k1_FDX_s156
                                 - 0.0115486528869103 w_k1_FDX_s157
                                 - 0.0120914300349634 w_k1_FDX_s160
                                 - 0.0116754736852778 w_k1_FDX_s161
                                 - 0.00679081849015352 w_k1_FDX_s162
                                 - 0.00331539436087403 w_k1_FDX_s163
                                 - 0.00673619112734276 w_k1_FDX_s164
                                 - 0.0156608288549005 w_k1_FDX_s165
                                 - 0.0119306325921722 w_k1_FDX_s166
                                 - 0.00538933638730332 w_k1_FDX_s167
                                 - 0.010211388950206 w_k1_FDX_s170
                                 - 0.00543096671844426 w_k1_FDX_s171
                                 - 0.00591986434760065 w_k1_FDX_s172
                                 - 0.00600116838897344 w_k1_FDX_s173
                                 - 0.0109581641556476 w_k1_FDX_s174
                                 - 0.0104908150172272 w_k1_FDX_s175
                                 - 0.00629895593651388 w_k1_FDX_s176
                                 - 0.0115134332561064 w_k1_FDX_s177
                                 - 0.00331509280531014 w_k1_IBM_s000
                                 - 0.00513597191552501 w_k1_IBM_s001
                                 - 0.00323022709032187 w_k1_IBM_s002
                                 - 0.00388865759712947 w_k1_IBM_s003
                                 - 0.00310104007972993 w_k1_IBM_s004
                                 - 0.00212307351410388 w_k1_IBM_s005
                                 - 0.000960316844780748 w_k1_IBM_s006
                                 - 0.00222971678964628 w_k1_IBM_s007
                                 - 0.00661306689764596 w_k1_IBM_s010
                                 - 0.0067726919227243 w_k1_IBM_s011
                                 - 0.00414118630052097 w_k1_IBM_s012
                                 - 0.00394121302489257 w_k1_IBM_s013
                                 - 0.00160433557891635 w_k1_IBM_s014
                                 - 0.00303933681823248 w_k1_IBM_s015
                                 - 0.00258083113972408 w_k1_IBM_s016
                                 - 0.00223675193932455 w_k1_IBM_s017
                                 - 0.00294871365940993 w_k1_IBM_s020
                                 - 0.00529956196396653 w_k1_IBM_s021
                                 - 0.00412921161941865 w_k1_IBM_s022
                                 - 0.00368189452808029 w_k1_IBM_s023
                                 - 0.0069640089360377 w_k1_IBM_s024
                                 - 0.00311513517290165 w_k1_IBM_s025
                                 - 0.00496898024374063 w_k1_IBM_s026
                                 - 0.00811834776955044 w_k1_IBM_s027
                                 - 0.00547966786811737 w_k1_IBM_s030
                                 - 0.00339539235737088 w_k1_IBM_s031
                                 - 0.0107539343278575 w_k1_IBM_s032
                                 - 0.0138521437401509 w_k1_IBM_s033
                                 - 0.00354075092861194 w_k1_IBM_s034
                                 - 0.00917754091446073 w_k1_IBM_s035
                                 - 0.00585430452188738 w_k1_IBM_s036
                                 - 0.00370800339996137 w_k1_IBM_s037
                                 - 0.00364149229435513 w_k1_IBM_s040
                                 - 0.00434634526255602 w_k1_IBM_s041
                                 - 0.00900056854700574 w_k1_IBM_s042
                                 - 0.00558176833320366 w_k1_IBM_s043
                                 - 0.0051577977355559 w_k1_IBM_s044
                                 - 0.0075844067137264 w_k1_IBM_s045
                                 - 0.00514780548837024 w_k1_IBM_s046
                                 - 0.00802776289474523 w_k1_IBM_s047
                                 - 0.00477455498768209 w_k1_IBM_s050
                                 - 0.0196677372015789 w_k1_IBM_s051
                                 - 0.0119029511465004 w_k1_IBM_s052
                                 - 0.0212410093260656 w_k1_IBM_s053
                                 - 0.0180341690369288 w_k1_IBM_s054
                                 - 0.0102831064388922 w_k1_IBM_s055
                                 - 0.0129272653870828 w_k1_IBM_s056
                                 - 0.0130558295743239 w_k1_IBM_s057
                                 - 0.00786439548153085 w_k1_IBM_s060
                                 - 0.00635723688082631 w_k1_IBM_s061
                                 - 0.00482277871168323 w_k1_IBM_s062
                                 - 0.00634600209879471 w_k1_IBM_s063
                                 - 0.0158845596731466 w_k1_IBM_s064
                                 - 0.00672587104265753 w_k1_IBM_s065
                                 - 0.0116141831786056 w_k1_IBM_s066
                                 - 0.00935682956786281 w_k1_IBM_s067
                                 - 0.00806029423938163 w_k1_IBM_s070
                                 - 0.00449088638544579 w_k1_IBM_s071
                                 - 0.0122273613174498 w_k1_IBM_s072
                                 - 0.0094887054459086 w_k1_IBM_s073
                                 - 0.0152739996822314 w_k1_IBM_s074
                                 - 0.00423687674598288 w_k1_IBM_s075
                                 - 0.00775453458862499 w_k1_IBM_s076
                                 - 0.00645469491767915 w_k1_IBM_s077
                                 - 0.00733297738991192 w_k1_IBM_s100
                                 - 0.0108355357148369 w_k1_IBM_s101
                                 - 0.00614084244096699 w_k1_IBM_s102
                                 - 0.009054763240898 w_k1_IBM_s103
                                 - 0.0159315090518143 w_k1_IBM_s104
                                 - 0.0138028969771163 w_k1_IBM_s105
                                 - 0.0166244399676635 w_k1_IBM_s106
                                 - 0.0225067749528624 w_k1_IBM_s107
                                 - 0.0130879221162076 w_k1_IBM_s110
                                 - 0.0050263531206014 w_k1_IBM_s111
                                 - 0.00770658611172757 w_k1_IBM_s112
                                 - 0.0105212238630784 w_k1_IBM_s113
                                 - 0.00345239667892213 w_k1_IBM_s114
                                 - 0.0165152508029983 w_k1_IBM_s115
                                 - 0.00420932835135819 w_k1_IBM_s116
                                 - 0.00689720522571857 w_k1_IBM_s117
                                 - 0.0029677954311041 w_k1_IBM_s120
                                 - 0.00587375903793818 w_k1_IBM_s121
                                 - 0.002075251816076 w_k1_IBM_s122
                                 - 0.0015559426673066 w_k1_IBM_s123
                                 - 0.00241137777829348 w_k1_IBM_s124
                                 - 0.00266914364475347 w_k1_IBM_s125
                                 - 0.00218360176325385 w_k1_IBM_s126
                                 - 0.00252739926654499 w_k1_IBM_s127
                                 - 0.015765649710685 w_k1_IBM_s130
                                 - 0.00575394859137025 w_k1_IBM_s131
                                 - 0.0106137692984512 w_k1_IBM_s132
                                 - 0.00641887962220903 w_k1_IBM_s133
                                 - 0.0180765568420829 w_k1_IBM_s134
                                 - 0.00709882606341432 w_k1_IBM_s135
                                 - 0.017634217164117 w_k1_IBM_s136
                                 - 0.0109774544346678 w_k1_IBM_s137
                                 - 0.0031401526518127 w_k1_IBM_s140
                                 - 0.00330012659656084 w_k1_IBM_s141
                                 - 0.00844004571718814 w_k1_IBM_s142
                                 - 0.00862465520271163 w_k1_IBM_s143
                                 - 0.00624155021012871 w_k1_IBM_s144
                                 - 0.00334138093820689 w_k1_IBM_s145
                                 - 0.00993881559056227 w_k1_IBM_s146
                                 - 0.00221854172190534 w_k1_IBM_s147
                                 - 0.0138781999612266 w_k1_IBM_s150
                                 - 0.0172256376217168 w_k1_IBM_s151
                                 - 0.00932407066246357 w_k1_IBM_s152
                                 - 0.00642586067027165 w_k1_IBM_s153
                                 - 0.0119798858057822 w_k1_IBM_s154
                                 - 0.0229533268105015 w_k1_IBM_s155
                                 - 0.0142822358697407 w_k1_IBM_s156
                                 - 0.0117581605801972 w_k1_IBM_s157
                                 - 0.011445360264949 w_k1_IBM_s160
                                 - 0.0112523660167665 w_k1_IBM_s161
                                 - 0.00697002457933768 w_k1_IBM_s162
                                 - 0.00344891751319781 w_k1_IBM_s163
                                 - 0.00688731613142484 w_k1_IBM_s164
                                 - 0.0165240259101178 w_k1_IBM_s165
                                 - 0.0119764210698965 w_k1_IBM_s166
                                 - 0.00499456859603047 w_k1_IBM_s167
                                 - 0.01045958134455 w_k1_IBM_s170
                                 - 0.00564023038719766 w_k1_IBM_s171
                                 - 0.00618310344751536 w_k1_IBM_s172
                                 - 0.00609536077786519 w_k1_IBM_s173
                                 - 0.0103147007978823 w_k1_IBM_s174
                                 - 0.0102576451910549 w_k1_IBM_s175
                                 - 0.00587762448160151 w_k1_IBM_s176
                                 - 0.012372497199703 w_k1_IBM_s177
                                 - 0.00314766958561383 w_k1_INTC_s000
                                 - 0.0051463060061552 w_k1_INTC_s001
                                 - 0.00294587352025968 w_k1_INTC_s002
                                 - 0.00358587837821603 w_k1_INTC_s003
                                 - 0.00303681971729432 w_k1_INTC_s004
                                 - 0.0021354212876767 w_k1_INTC_s005
                                 - 0.00107638923298366 w_k1_INTC_s006
                                 - 0.00268830873655843 w_k1_INTC_s007
                                 - 0.00623751118590362 w_k1_INTC_s010
                                 - 0.00615458904241034 w_k1_INTC_s011
                                 - 0.00404385999930057 w_k1_INTC_s012
                                 - 0.00389853643704881 w_k1_INTC_s013
                                 - 0.00177491262576427 w_k1_INTC_s014
                                 - 0.00292286835887815 w_k1_INTC_s015
                                 - 0.00259577794378091 w_k1_INTC_s016
                                 - 0.00214201583331525 w_k1_INTC_s017
                                 - 0.00284201616037957 w_k1_INTC_s020
                                 - 0.00482140562823743 w_k1_INTC_s021
                                 - 0.00440675041889744 w_k1_INTC_s022
                                 - 0.0036034855405261 w_k1_INTC_s023
                                 - 0.00758571584716744 w_k1_INTC_s024
                                 - 0.00293097438608061 w_k1_INTC_s025
                                 - 0.00558458538374214 w_k1_INTC_s026
                                 - 0.00796254974746901 w_k1_INTC_s027
                                 - 0.00549723487449445 w_k1_INTC_s030
                                 - 0.00293308669746046 w_k1_INTC_s031
                                 - 0.00903309328795673 w_k1_INTC_s032
                                 - 0.0122559351233967 w_k1_INTC_s033
                                 - 0.0030864559380832 w_k1_INTC_s034
                                 - 0.00929173220314864 w_k1_INTC_s035
                                 - 0.0058087333552809 w_k1_INTC_s036
                                 - 0.00380271658306353 w_k1_INTC_s037
                                 - 0.00396157997363173 w_k1_INTC_s040
                                 - 0.00450924857295805 w_k1_INTC_s041
                                 - 0.00870093980213311 w_k1_INTC_s042
                                 - 0.0061201104367172 w_k1_INTC_s043
                                 - 0.00453213930436085 w_k1_INTC_s044
                                 - 0.00738211989842729 w_k1_INTC_s045
                                 - 0.00557167006555274 w_k1_INTC_s046
                                 - 0.00894829127112563 w_k1_INTC_s047
                                 - 0.00569974956827205 w_k1_INTC_s050
                                 - 0.0196221856304343 w_k1_INTC_s051
                                 - 0.00962249645878895 w_k1_INTC_s052
                                 - 0.0223218215887858 w_k1_INTC_s053
                                 - 0.0166881787534153 w_k1_INTC_s054
                                 - 0.0118665944715807 w_k1_INTC_s055
                                 - 0.0140181535456264 w_k1_INTC_s056
                                 - 0.0140323117589594 w_k1_INTC_s057
                                 - 0.00776636339329883 w_k1_INTC_s060
                                 - 0.00536347139776284 w_k1_INTC_s061
                                 - 0.00420829886391543 w_k1_INTC_s062
                                 - 0.00597312271757498 w_k1_INTC_s063
                                 - 0.0161332375880919 w_k1_INTC_s064
                                 - 0.00673547639621433 w_k1_INTC_s065
                                 - 0.0125350157348389 w_k1_INTC_s066
                                 - 0.00799028292178819 w_k1_INTC_s067
                                 - 0.00698807764424657 w_k1_INTC_s070
                                 - 0.00450418320288451 w_k1_INTC_s071
                                 - 0.0138652824938395 w_k1_INTC_s072
                                 - 0.0101917815925522 w_k1_INTC_s073
                                 - 0.0144253892815057 w_k1_INTC_s074
                                 - 0.00502674648944994 w_k1_INTC_s075
                                 - 0.00862645571642093 w_k1_INTC_s076
                                 - 0.00743104551647183 w_k1_INTC_s077
                                 - 0.00663618396902481 w_k1_INTC_s100
                                 - 0.0105210782800715 w_k1_INTC_s101
                                 - 0.006116493769937 w_k1_INTC_s102
                                 - 0.00894313347839584 w_k1_INTC_s103
                                 - 0.0153496761011691 w_k1_INTC_s104
                                 - 0.013402464545352 w_k1_INTC_s105
                                 - 0.0160273057605903 w_k1_INTC_s106
                                 - 0.0227832350101436 w_k1_INTC_s107
                                 - 0.0118893443015321 w_k1_INTC_s110
                                 - 0.0045222712115409 w_k1_INTC_s111
                                 - 0.0074697365595565 w_k1_INTC_s112
                                 - 0.00953266482450303 w_k1_INTC_s113
                                 - 0.00337195820475187 w_k1_INTC_s114
                                 - 0.0143105941475556 w_k1_INTC_s115
                                 - 0.00311145987046951 w_k1_INTC_s116
                                 - 0.00789941271984445 w_k1_INTC_s117
                                 - 0.00319482000673442 w_k1_INTC_s120
                                 - 0.00623754938779625 w_k1_INTC_s121
                                 - 0.00252879342155855 w_k1_INTC_s122
                                 - 0.00157486835257634 w_k1_INTC_s123
                                 - 0.0027810313714819 w_k1_INTC_s124
                                 - 0.00278913397822282 w_k1_INTC_s125
                                 - 0.00166583432554141 w_k1_INTC_s126
                                 - 0.00258151173721544 w_k1_INTC_s127
                                 - 0.0170571837839528 w_k1_INTC_s130
                                 - 0.00507144686645344 w_k1_INTC_s131
                                 - 0.0102413478835684 w_k1_INTC_s132
                                 - 0.00599511508601982 w_k1_INTC_s133
                                 - 0.0182259239048979 w_k1_INTC_s134
                                 - 0.00742684511487398 w_k1_INTC_s135
                                 - 0.0176214039876369 w_k1_INTC_s136
                                 - 0.011992681464868 w_k1_INTC_s137
                                 - 0.00337901642300435 w_k1_INTC_s140
                                 - 0.00338314997295752 w_k1_INTC_s141
                                 - 0.00953781185306647 w_k1_INTC_s142
                                 - 0.00759466988878273 w_k1_INTC_s143
                                 - 0.0051292145251253 w_k1_INTC_s144
                                 - 0.00426951023727261 w_k1_INTC_s145
                                 - 0.0121541679691579 w_k1_INTC_s146
                                 - 0.00229857142499272 w_k1_INTC_s147
                                 - 0.0118665979075498 w_k1_INTC_s150
                                 - 0.0180516261427366 w_k1_INTC_s151
                                 - 0.00828960636976643 w_k1_INTC_s152
                                 - 0.00560710988142799 w_k1_INTC_s153
                                 - 0.010918663781551 w_k1_INTC_s154
                                 - 0.0232877037879668 w_k1_INTC_s155
                                 - 0.0135986351391016 w_k1_INTC_s156
                                 - 0.0105159020613347 w_k1_INTC_s157
                                 - 0.0114647362515503 w_k1_INTC_s160
                                 - 0.0130325021636788 w_k1_INTC_s161
                                 - 0.00615076126199579 w_k1_INTC_s162
                                 - 0.00336576562517798 w_k1_INTC_s163
                                 - 0.00629281122361217 w_k1_INTC_s164
                                 - 0.0156541321022772 w_k1_INTC_s165
                                 - 0.0123290048459156 w_k1_INTC_s166
                                 - 0.00516468593547978 w_k1_INTC_s167
                                 - 0.00946926727008613 w_k1_INTC_s170
                                 - 0.00633418164056316 w_k1_INTC_s171
                                 - 0.00557381323322454 w_k1_INTC_s172
                                 - 0.00489517850863084 w_k1_INTC_s173
                                 - 0.0101783496066971 w_k1_INTC_s174
                                 - 0.00954775443107831 w_k1_INTC_s175
                                 - 0.00669145967186427 w_k1_INTC_s176
                                 - 0.0113158281196321 w_k1_INTC_s177
                                 - 0.00359396722094905 w_k1_KO_s000
                                 - 0.00490127421714913 w_k1_KO_s001
                                 - 0.00339658043871096 w_k1_KO_s002
                                 - 0.00374301198210118 w_k1_KO_s003
                                 - 0.0031207611048103 w_k1_KO_s004
                                 - 0.00212848537953203 w_k1_KO_s005
                                 - 0.000921678014372875 w_k1_KO_s006
                                 - 0.00223038153516416 w_k1_KO_s007
                                 - 0.00622298920676165 w_k1_KO_s010
                                 - 0.00653630032873668 w_k1_KO_s011
                                 - 0.00341485805684011 w_k1_KO_s012
                                 - 0.00401040772492471 w_k1_KO_s013
                                 - 0.00167822743022722 w_k1_KO_s014
                                 - 0.00312837557919525 w_k1_KO_s015
                                 - 0.00285540157925895 w_k1_KO_s016
                                 - 0.00210010360212558 w_k1_KO_s017
                                 - 0.00297014446059043 w_k1_KO_s020
                                 - 0.00500505329431347 w_k1_KO_s021
                                 - 0.00401734863925703 w_k1_KO_s022
                                 - 0.0032596907206608 w_k1_KO_s023
                                 - 0.00730485265170731 w_k1_KO_s024
                                 - 0.00329160186118769 w_k1_KO_s025
                                 - 0.00509836589897117 w_k1_KO_s026
                                 - 0.00783764514746892 w_k1_KO_s027
                                 - 0.00479209026623466 w_k1_KO_s030
                                 - 0.0031160335903549 w_k1_KO_s031
                                 - 0.0112018495866064 w_k1_KO_s032
                                 - 0.0152555479030544 w_k1_KO_s033
                                 - 0.00341293034988562 w_k1_KO_s034
                                 - 0.00788811127926428 w_k1_KO_s035
                                 - 0.00555794929295318 w_k1_KO_s036
                                 - 0.00372731232249203 w_k1_KO_s037
                                 - 0.00349437279610153 w_k1_KO_s040
                                 - 0.0041750483548408 w_k1_KO_s041
                                 - 0.00947821417107663 w_k1_KO_s042
                                 - 0.00633760053526147 w_k1_KO_s043
                                 - 0.00544721555068654 w_k1_KO_s044
                                 - 0.00696793258300528 w_k1_KO_s045
                                 - 0.0051991575313755 w_k1_KO_s046
                                 - 0.00855980822946476 w_k1_KO_s047
                                 - 0.00475763508765625 w_k1_KO_s050
                                 - 0.0194951644765723 w_k1_KO_s051
                                 - 0.0121402180219121 w_k1_KO_s052
                                 - 0.0223959442936708 w_k1_KO_s053
                                 - 0.0189815966356795 w_k1_KO_s054
                                 - 0.0101513556817521 w_k1_KO_s055
                                 - 0.01091936027221 w_k1_KO_s056
                                 - 0.0129594031821169 w_k1_KO_s057
                                 - 0.00804353944226847 w_k1_KO_s060
                                 - 0.00626596163281767 w_k1_KO_s061
                                 - 0.00519978511441001 w_k1_KO_s062
                                 - 0.00624168350244292 w_k1_KO_s063
                                 - 0.0158383566775544 w_k1_KO_s064
                                 - 0.00665867043435365 w_k1_KO_s065
                                 - 0.0111584490285995 w_k1_KO_s066
                                 - 0.00854292674904717 w_k1_KO_s067
                                 - 0.00806167185040396 w_k1_KO_s070
                                 - 0.00432809379811089 w_k1_KO_s071
                                 - 0.0126933590966472 w_k1_KO_s072
                                 - 0.00891808699116786 w_k1_KO_s073
                                 - 0.0149857128387365 w_k1_KO_s074
                                 - 0.00469218512327926 w_k1_KO_s075
                                 - 0.00762821716169818 w_k1_KO_s076
                                 - 0.00648528337771393 w_k1_KO_s077
                                 - 0.00691807235227977 w_k1_KO_s100
                                 - 0.0115842792974869 w_k1_KO_s101
                                 - 0.00690234847614264 w_k1_KO_s102
                                 - 0.00844546934855761 w_k1_KO_s103
                                 - 0.0158205645642658 w_k1_KO_s104
                                 - 0.013211373509775 w_k1_KO_s105
                                 - 0.0158119587872021 w_k1_KO_s106
                                 - 0.022024914991556 w_k1_KO_s107
                                 - 0.0123195800353623 w_k1_KO_s110
                                 - 0.00456486340459986 w_k1_KO_s111
                                 - 0.00757676829152264 w_k1_KO_s112
                                 - 0.0104694252117224 w_k1_KO_s113
                                 - 0.00377823327471597 w_k1_KO_s114
                                 - 0.0154703382999681 w_k1_KO_s115
                                 - 0.00404910200682416 w_k1_KO_s116
                                 - 0.00766998461865694 w_k1_KO_s117
                                 - 0.00288328563550478 w_k1_KO_s120
                                 - 0.00627322947490697 w_k1_KO_s121
                                 - 0.00210237782582131 w_k1_KO_s122
                                 - 0.00163366665473509 w_k1_KO_s123
                                 - 0.00257361312108927 w_k1_KO_s124
                                 - 0.00263929251541566 w_k1_KO_s125
                                 - 0.00233593749568972 w_k1_KO_s126
                                 - 0.00243331545401988 w_k1_KO_s127
                                 - 0.0147655845656846 w_k1_KO_s130
                                 - 0.00558956637902588 w_k1_KO_s131
                                 - 0.0107019692383062 w_k1_KO_s132
                                 - 0.00579889931576736 w_k1_KO_s133
                                 - 0.0160038675075259 w_k1_KO_s134
                                 - 0.00692356627613032 w_k1_KO_s135
                                 - 0.018146491225393 w_k1_KO_s136
                                 - 0.0101430773092533 w_k1_KO_s137
                                 - 0.00347928701779057 w_k1_KO_s140
                                 - 0.00316061034967803 w_k1_KO_s141
                                 - 0.00785583030445401 w_k1_KO_s142
                                 - 0.00828643642821725 w_k1_KO_s143
                                 - 0.00622729761067575 w_k1_KO_s144
                                 - 0.00348035936822668 w_k1_KO_s145
                                 - 0.0096381991157558 w_k1_KO_s146
                                 - 0.00234461109529816 w_k1_KO_s147
                                 - 0.0138336208769008 w_k1_KO_s150
                                 - 0.0184837306682202 w_k1_KO_s151
                                 - 0.00919031250987408 w_k1_KO_s152
                                 - 0.00658931398225038 w_k1_KO_s153
                                 - 0.0126460981583478 w_k1_KO_s154
                                 - 0.0200631550851857 w_k1_KO_s155
                                 - 0.0138425992197689 w_k1_KO_s156
                                 - 0.0121964969945249 w_k1_KO_s157
                                 - 0.0130469237702333 w_k1_KO_s160
                                 - 0.0112601806936431 w_k1_KO_s161
                                 - 0.00687844723275003 w_k1_KO_s162
                                 - 0.00330299978984473 w_k1_KO_s163
                                 - 0.00647981237949819 w_k1_KO_s164
                                 - 0.0140779759482916 w_k1_KO_s165
                                 - 0.0120122645244081 w_k1_KO_s166
                                 - 0.00539080792012553 w_k1_KO_s167
                                 - 0.00956816216737062 w_k1_KO_s170
                                 - 0.00598592222212862 w_k1_KO_s171
                                 - 0.00667654279946648 w_k1_KO_s172
                                 - 0.00602921478821667 w_k1_KO_s173
                                 - 0.0102609133744421 w_k1_KO_s174
                                 - 0.0100998546740154 w_k1_KO_s175
                                 - 0.00659595760939863 w_k1_KO_s176
                                 - 0.0124791666870201 w_k1_KO_s177
                                 - 0.00375932175503098 w_k1_MAT_s000
                                 - 0.00507127285998036 w_k1_MAT_s001
                                 - 0.00374851617900296 w_k1_MAT_s002
                                 - 0.00339917489987513 w_k1_MAT_s003
                                 - 0.00290068031883587 w_k1_MAT_s004
                                 - 0.00204469859735304 w_k1_MAT_s005
                                 - 0.00125774391065423 w_k1_MAT_s006
                                 - 0.00238846280679225 w_k1_MAT_s007
                                 - 0.00736414822274678 w_k1_MAT_s010
                                 - 0.00557195666439711 w_k1_MAT_s011
                                 - 0.00460683988972473 w_k1_MAT_s012
                                 - 0.00356765330263254 w_k1_MAT_s013
                                 - 0.00210763011432244 w_k1_MAT_s014
                                 - 0.00297537974796652 w_k1_MAT_s015
                                 - 0.002911132887221 w_k1_MAT_s016
                                 - 0.00218712596527669 w_k1_MAT_s017
                                 - 0.00318403916428692 w_k1_MAT_s020
                                 - 0.00438382613263294 w_k1_MAT_s021
                                 - 0.00457645569505556 w_k1_MAT_s022
                                 - 0.00337098662321684 w_k1_MAT_s023
                                 - 0.00834547202486015 w_k1_MAT_s024
                                 - 0.00356650788643805 w_k1_MAT_s025
                                 - 0.00442334122643428 w_k1_MAT_s026
                                 - 0.0083044085822871 w_k1_MAT_s027
                                 - 0.00492593578329904 w_k1_MAT_s030
                                 - 0.00371332279052396 w_k1_MAT_s031
                                 - 0.0122042681850841 w_k1_MAT_s032
                                 - 0.01620664864824 w_k1_MAT_s033
                                 - 0.00414531235832003 w_k1_MAT_s034
                                 - 0.00910465292688196 w_k1_MAT_s035
                                 - 0.00512165417938083 w_k1_MAT_s036
                                 - 0.00446778183990845 w_k1_MAT_s037
                                 - 0.00341175694323442 w_k1_MAT_s040
                                 - 0.00437707169718342 w_k1_MAT_s041
                                 - 0.00813285692734676 w_k1_MAT_s042
                                 - 0.00626233252501793 w_k1_MAT_s043
                                 - 0.00549744776467498 w_k1_MAT_s044
                                 - 0.0092532927401167 w_k1_MAT_s045
                                 - 0.0042456508141572 w_k1_MAT_s046
                                 - 0.00745712665012573 w_k1_MAT_s047
                                 - 0.00381722937917413 w_k1_MAT_s050
                                 - 0.0193828939412873 w_k1_MAT_s051
                                 - 0.0123052751931849 w_k1_MAT_s052
                                 - 0.0219999667789775 w_k1_MAT_s053
                                 - 0.019695080515281 w_k1_MAT_s054
                                 - 0.00983315580982676 w_k1_MAT_s055
                                 - 0.0102125451994617 w_k1_MAT_s056
                                 - 0.0138516718163976 w_k1_MAT_s057
                                 - 0.00973289577844465 w_k1_MAT_s060
                                 - 0.00586370159416208 w_k1_MAT_s061
                                 - 0.00431742244521395 w_k1_MAT_s062
                                 - 0.00609474039891518 w_k1_MAT_s063
                                 - 0.0175992503097565 w_k1_MAT_s064
                                 - 0.00818331679835483 w_k1_MAT_s065
                                 - 0.00965800882100415 w_k1_MAT_s066
                                 - 0.0101812336770249 w_k1_MAT_s067
                                 - 0.00862917684333072 w_k1_MAT_s070
                                 - 0.00343844113815563 w_k1_MAT_s071
                                 - 0.0115520133372286 w_k1_MAT_s072
                                 - 0.00906643562281962 w_k1_MAT_s073
                                 - 0.0204585881379809 w_k1_MAT_s074
                                 - 0.00500005642738245 w_k1_MAT_s075
                                 - 0.00871604967259666 w_k1_MAT_s076
                                 - 0.00556476286211248 w_k1_MAT_s077
                                 - 0.00885919064689737 w_k1_MAT_s100
                                 - 0.0145815493906461 w_k1_MAT_s101
                                 - 0.00625484524261375 w_k1_MAT_s102
                                 - 0.00893929070761259 w_k1_MAT_s103
                                 - 0.0147478310781692 w_k1_MAT_s104
                                 - 0.0203221796000878 w_k1_MAT_s105
                                 - 0.0144487860442035 w_k1_MAT_s106
                                 - 0.0249579753305186 w_k1_MAT_s107
                                 - 0.0111424001253292 w_k1_MAT_s110
                                 - 0.00550458474681732 w_k1_MAT_s111
                                 - 0.00800938240129484 w_k1_MAT_s112
                                 - 0.00976265730654242 w_k1_MAT_s113
                                 - 0.00450740752644694 w_k1_MAT_s114
                                 - 0.0146254486210796 w_k1_MAT_s115
                                 - 0.00315015293538681 w_k1_MAT_s116
                                 - 0.00662679284705418 w_k1_MAT_s117
                                 - 0.00274672264442462 w_k1_MAT_s120
                                 - 0.00711562665760895 w_k1_MAT_s121
                                 - 0.00198051921875904 w_k1_MAT_s122
                                 - 0.00143968519614398 w_k1_MAT_s123
                                 - 0.0021900465160428 w_k1_MAT_s124
                                 - 0.0029017542750075 w_k1_MAT_s125
                                 - 0.0027951623393845 w_k1_MAT_s126
                                 - 0.00305031379340681 w_k1_MAT_s127
                                 - 0.0167638406290955 w_k1_MAT_s130
                                 - 0.00520534561599333 w_k1_MAT_s131
                                 - 0.0104080490957288 w_k1_MAT_s132
                                 - 0.00811812187331809 w_k1_MAT_s133
                                 - 0.0194962953125122 w_k1_MAT_s134
                                 - 0.00612426972931786 w_k1_MAT_s135
                                 - 0.0178961771932829 w_k1_MAT_s136
                                 - 0.00897007768793209 w_k1_MAT_s137
                                 - 0.00333441225720451 w_k1_MAT_s140
                                 - 0.00384914738709598 w_k1_MAT_s141
                                 - 0.00958946666217579 w_k1_MAT_s142
                                 - 0.00948945404880952 w_k1_MAT_s143
                                 - 0.00580200611352882 w_k1_MAT_s144
                                 - 0.0034526851965636 w_k1_MAT_s145
                                 - 0.0100300725939691 w_k1_MAT_s146
                                 - 0.00277314201147195 w_k1_MAT_s147
                                 - 0.0162368556904419 w_k1_MAT_s150
                                 - 0.0170405977232187 w_k1_MAT_s151
                                 - 0.00906231442925829 w_k1_MAT_s152
                                 - 0.00679247235222381 w_k1_MAT_s153
                                 - 0.0185246476127069 w_k1_MAT_s154
                                 - 0.020066332102677 w_k1_MAT_s155
                                 - 0.0129123972938872 w_k1_MAT_s156
                                 - 0.0121597517597059 w_k1_MAT_s157
                                 - 0.0123981151895848 w_k1_MAT_s160
                                 - 0.00954713506181209 w_k1_MAT_s161
                                 - 0.00703734598041768 w_k1_MAT_s162
                                 - 0.00300578526690714 w_k1_MAT_s163
                                 - 0.00625705005595686 w_k1_MAT_s164
                                 - 0.0202779867966163 w_k1_MAT_s165
                                 - 0.0157116153707383 w_k1_MAT_s166
                                 - 0.00423787914773179 w_k1_MAT_s167
                                 - 0.00921528296887322 w_k1_MAT_s170
                                 - 0.00619329504271103 w_k1_MAT_s171
                                 - 0.00757512320835 w_k1_MAT_s172
                                 - 0.00650560582381529 w_k1_MAT_s173
                                 - 0.0108980903747018 w_k1_MAT_s174
                                 - 0.0100173014481006 w_k1_MAT_s175
                                 - 0.00669341425676915 w_k1_MAT_s176
                                 - 0.0153505690380734 w_k1_MAT_s177
                                 - 0.00320930407761409 w_k1_MSFT_s000
                                 - 0.00458984650032635 w_k1_MSFT_s001
                                 - 0.00346622321063214 w_k1_MSFT_s002
                                 - 0.00370538525809254 w_k1_MSFT_s003
                                 - 0.00314902721256604 w_k1_MSFT_s004
                                 - 0.00226713496702561 w_k1_MSFT_s005
                                 - 0.000985035592828134 w_k1_MSFT_s006
                                 - 0.00236927574494267 w_k1_MSFT_s007
                                 - 0.00630211589809379 w_k1_MSFT_s010
                                 - 0.00631647091451496 w_k1_MSFT_s011
                                 - 0.00380862853212829 w_k1_MSFT_s012
                                 - 0.00351000260695446 w_k1_MSFT_s013
                                 - 0.00176725764537206 w_k1_MSFT_s014
                                 - 0.00267956678923204 w_k1_MSFT_s015
                                 - 0.00244188943658533 w_k1_MSFT_s016
                                 - 0.00207861217317052 w_k1_MSFT_s017
                                 - 0.00270193283475687 w_k1_MSFT_s020
                                 - 0.00529982644106597 w_k1_MSFT_s021
                                 - 0.00363735421204387 w_k1_MSFT_s022
                                 - 0.00347564803424105 w_k1_MSFT_s023
                                 - 0.00816181946840339 w_k1_MSFT_s024
                                 - 0.00317958127838048 w_k1_MSFT_s025
                                 - 0.00491835987600581 w_k1_MSFT_s026
                                 - 0.00843197161064295 w_k1_MSFT_s027
                                 - 0.00482419295622871 w_k1_MSFT_s030
                                 - 0.0032952994325878 w_k1_MSFT_s031
                                 - 0.00983102272297957 w_k1_MSFT_s032
                                 - 0.0137676308539768 w_k1_MSFT_s033
                                 - 0.00335151825261874 w_k1_MSFT_s034
                                 - 0.00817585716410326 w_k1_MSFT_s035
                                 - 0.00536896742004503 w_k1_MSFT_s036
                                 - 0.00372771244771432 w_k1_MSFT_s037
                                 - 0.00343371712450747 w_k1_MSFT_s040
                                 - 0.00459732438512903 w_k1_MSFT_s041
                                 - 0.00911674664788497 w_k1_MSFT_s042
                                 - 0.00588101330955481 w_k1_MSFT_s043
                                 - 0.00450339878476596 w_k1_MSFT_s044
                                 - 0.0077417230976174 w_k1_MSFT_s045
                                 - 0.0052406278842535 w_k1_MSFT_s046
                                 - 0.00869857024833482 w_k1_MSFT_s047
                                 - 0.00465625257010697 w_k1_MSFT_s050
                                 - 0.0195803491265168 w_k1_MSFT_s051
                                 - 0.0105741920681013 w_k1_MSFT_s052
                                 - 0.0220698542234179 w_k1_MSFT_s053
                                 - 0.0176219481114595 w_k1_MSFT_s054
                                 - 0.0106026177169097 w_k1_MSFT_s055
                                 - 0.0112798378019331 w_k1_MSFT_s056
                                 - 0.0133698981870843 w_k1_MSFT_s057
                                 - 0.00813344635397006 w_k1_MSFT_s060
                                 - 0.00622954721398476 w_k1_MSFT_s061
                                 - 0.00465362698805053 w_k1_MSFT_s062
                                 - 0.00673749579554966 w_k1_MSFT_s063
                                 - 0.0153623931863921 w_k1_MSFT_s064
                                 - 0.00652320098181589 w_k1_MSFT_s065
                                 - 0.0115283158799423 w_k1_MSFT_s066
                                 - 0.00853818127770515 w_k1_MSFT_s067
                                 - 0.0076029727856659 w_k1_MSFT_s070
                                 - 0.00454630870604476 w_k1_MSFT_s071
                                 - 0.0127941685158281 w_k1_MSFT_s072
                                 - 0.00902731671740813 w_k1_MSFT_s073
                                 - 0.0155533664113103 w_k1_MSFT_s074
                                 - 0.00456929139317172 w_k1_MSFT_s075
                                 - 0.00771284426140919 w_k1_MSFT_s076
                                 - 0.00645075342627966 w_k1_MSFT_s077
                                 - 0.00671906880128409 w_k1_MSFT_s100
                                 - 0.0121804132654245 w_k1_MSFT_s101
                                 - 0.00661301020500051 w_k1_MSFT_s102
                                 - 0.0083748060994612 w_k1_MSFT_s103
                                 - 0.0153932415992756 w_k1_MSFT_s104
                                 - 0.0137652663529314 w_k1_MSFT_s105
                                 - 0.0170772933026472 w_k1_MSFT_s106
                                 - 0.0243826832973259 w_k1_MSFT_s107
                                 - 0.0120338705065503 w_k1_MSFT_s110
                                 - 0.00446249176913578 w_k1_MSFT_s111
                                 - 0.00733434841859909 w_k1_MSFT_s112
                                 - 0.0111362597302168 w_k1_MSFT_s113
                                 - 0.00359793700317193 w_k1_MSFT_s114
                                 - 0.0165093495051132 w_k1_MSFT_s115
                                 - 0.00347295767237141 w_k1_MSFT_s116
                                 - 0.00798418543119322 w_k1_MSFT_s117
                                 - 0.00251945074847195 w_k1_MSFT_s120
                                 - 0.00630376396090674 w_k1_MSFT_s121
                                 - 0.0023237115091809 w_k1_MSFT_s122
                                 - 0.00158090170065598 w_k1_MSFT_s123
                                 - 0.00242435529285239 w_k1_MSFT_s124
                                 - 0.00260506706348633 w_k1_MSFT_s125
                                 - 0.00221764660617766 w_k1_MSFT_s126
                                 - 0.00254100360654452 w_k1_MSFT_s127
                                 - 0.014020358519421 w_k1_MSFT_s130
                                 - 0.00602110843504741 w_k1_MSFT_s131
                                 - 0.0110681234163148 w_k1_MSFT_s132
                                 - 0.00658396485732007 w_k1_MSFT_s133
                                 - 0.0160356427923812 w_k1_MSFT_s134
                                 - 0.00734128888500392 w_k1_MSFT_s135
                                 - 0.0167306590069971 w_k1_MSFT_s136
                                 - 0.0101487774485889 w_k1_MSFT_s137
                                 - 0.0033235130660287 w_k1_MSFT_s140
                                 - 0.00362021450757914 w_k1_MSFT_s141
                                 - 0.00924543794631672 w_k1_MSFT_s142
                                 - 0.00881791727720296 w_k1_MSFT_s143
                                 - 0.00575556774521949 w_k1_MSFT_s144
                                 - 0.00391545605996013 w_k1_MSFT_s145
                                 - 0.0111445078445188 w_k1_MSFT_s146
                                 - 0.00209061722030302 w_k1_MSFT_s147
                                 - 0.0142593013726632 w_k1_MSFT_s150
                                 - 0.0174000264466548 w_k1_MSFT_s151
                                 - 0.00991158424693605 w_k1_MSFT_s152
                                 - 0.00635844565108434 w_k1_MSFT_s153
                                 - 0.0117142975067614 w_k1_MSFT_s154
                                 - 0.0225069365351853 w_k1_MSFT_s155
                                 - 0.0138084316132885 w_k1_MSFT_s156
                                 - 0.0130030006353049 w_k1_MSFT_s157
                                 - 0.0129164916666227 w_k1_MSFT_s160
                                 - 0.0122631645544497 w_k1_MSFT_s161
                                 - 0.00657888918324487 w_k1_MSFT_s162
                                 - 0.00299789711514356 w_k1_MSFT_s163
                                 - 0.0068311135650463 w_k1_MSFT_s164
                                 - 0.0153745747772678 w_k1_MSFT_s165
                                 - 0.0129589148454641 w_k1_MSFT_s166
                                 - 0.0050775049634284 w_k1_MSFT_s167
                                 - 0.00961676639501285 w_k1_MSFT_s170
                                 - 0.00556817426629007 w_k1_MSFT_s171
                                 - 0.00654668660854177 w_k1_MSFT_s172
                                 - 0.00594536113448744 w_k1_MSFT_s173
                                 - 0.010817331704054 w_k1_MSFT_s174
                                 - 0.00944626244977207 w_k1_MSFT_s175
                                 - 0.0065384982827857 w_k1_MSFT_s176
                                 - 0.0121349719756216 w_k1_MSFT_s177
                                 - 0.00350047403984379 w_k1_PG_s000
                                 - 0.00515557750431525 w_k1_PG_s001
                                 - 0.00306351914090824 w_k1_PG_s002
                                 - 0.00394880308344519 w_k1_PG_s003
                                 - 0.00305235869325742 w_k1_PG_s004
                                 - 0.00226649885267218 w_k1_PG_s005
                                 - 0.000962381306372078 w_k1_PG_s006
                                 - 0.00230391757951871 w_k1_PG_s007
                                 - 0.00667126488298031 w_k1_PG_s010
                                 - 0.00631299403915387 w_k1_PG_s011
                                 - 0.00362952425806294 w_k1_PG_s012
                                 - 0.0037301543730946 w_k1_PG_s013
                                 - 0.00157888411832815 w_k1_PG_s014
                                 - 0.00321302550229522 w_k1_PG_s015
                                 - 0.00283240421006907 w_k1_PG_s016
                                 - 0.00220459678947361 w_k1_PG_s017
                                 - 0.00300707208284716 w_k1_PG_s020
                                 - 0.0053159479640329 w_k1_PG_s021
                                 - 0.00418685222209917 w_k1_PG_s022
                                 - 0.00370729691435052 w_k1_PG_s023
                                 - 0.00732005102177334 w_k1_PG_s024
                                 - 0.00306423336376804 w_k1_PG_s025
                                 - 0.00426065981211168 w_k1_PG_s026
                                 - 0.00765625903433551 w_k1_PG_s027
                                 - 0.00529811644072599 w_k1_PG_s030
                                 - 0.00277807444551721 w_k1_PG_s031
                                 - 0.010048134709641 w_k1_PG_s032
                                 - 0.0141391056153721 w_k1_PG_s033
                                 - 0.00338538766600053 w_k1_PG_s034
                                 - 0.00731657078667933 w_k1_PG_s035
                                 - 0.00587031560346881 w_k1_PG_s036
                                 - 0.00362927768223565 w_k1_PG_s037
                                 - 0.00347747690750485 w_k1_PG_s040
                                 - 0.0047001891786255 w_k1_PG_s041
                                 - 0.00847464015471279 w_k1_PG_s042
                                 - 0.00593865014211271 w_k1_PG_s043
                                 - 0.00573996640122684 w_k1_PG_s044
                                 - 0.00750475633590828 w_k1_PG_s045
                                 - 0.00523933716640595 w_k1_PG_s046
                                 - 0.00830010471289442 w_k1_PG_s047
                                 - 0.00504613771954518 w_k1_PG_s050
                                 - 0.0210740010726503 w_k1_PG_s051
                                 - 0.0117628179044461 w_k1_PG_s052
                                 - 0.0214476816061895 w_k1_PG_s053
                                 - 0.0169216900751398 w_k1_PG_s054
                                 - 0.00967964748661245 w_k1_PG_s055
                                 - 0.0111249385674368 w_k1_PG_s056
                                 - 0.0118436974109861 w_k1_PG_s057
                                 - 0.00750459263834658 w_k1_PG_s060
                                 - 0.00639939883988807 w_k1_PG_s061
                                 - 0.00491363150416058 w_k1_PG_s062
                                 - 0.00640606289327466 w_k1_PG_s063
                                 - 0.0172024264739869 w_k1_PG_s064
                                 - 0.0071406765597783 w_k1_PG_s065
                                 - 0.0108761062431397 w_k1_PG_s066
                                 - 0.008607362428393 w_k1_PG_s067
                                 - 0.00811337530102588 w_k1_PG_s070
                                 - 0.00403905258333745 w_k1_PG_s071
                                 - 0.0118011874975587 w_k1_PG_s072
                                 - 0.00892104046139745 w_k1_PG_s073
                                 - 0.0140590240741881 w_k1_PG_s074
                                 - 0.00410985054538689 w_k1_PG_s075
                                 - 0.0078137015358991 w_k1_PG_s076
                                 - 0.00682115895933098 w_k1_PG_s077
                                 - 0.00818027182103997 w_k1_PG_s100
                                 - 0.0113112963273965 w_k1_PG_s101
                                 - 0.00602933775695558 w_k1_PG_s102
                                 - 0.00901316687406622 w_k1_PG_s103
                                 - 0.0153932613354722 w_k1_PG_s104
                                 - 0.013399288935002 w_k1_PG_s105
                                 - 0.0160646679232774 w_k1_PG_s106
                                 - 0.0238063271506298 w_k1_PG_s107
                                 - 0.0128620335147628 w_k1_PG_s110
                                 - 0.00489841186464362 w_k1_PG_s111
                                 - 0.00697173279301966 w_k1_PG_s112
                                 - 0.0100310937309152 w_k1_PG_s113
                                 - 0.00379990784835946 w_k1_PG_s114
                                 - 0.0158017868526474 w_k1_PG_s115
                                 - 0.00367858759104975 w_k1_PG_s116
                                 - 0.00710340113510003 w_k1_PG_s117
                                 - 0.00244205163703948 w_k1_PG_s120
                                 - 0.00659902511062566 w_k1_PG_s121
                                 - 0.00211266545362983 w_k1_PG_s122
                                 - 0.00149876099834336 w_k1_PG_s123
                                 - 0.00237964786736171 w_k1_PG_s124
                                 - 0.00278500634823329 w_k1_PG_s125
                                 - 0.00226834164762788 w_k1_PG_s126
                                 - 0.00242001207335668 w_k1_PG_s127
                                 - 0.0158480512752677 w_k1_PG_s130
                                 - 0.00542152701092578 w_k1_PG_s131
                                 - 0.0118806146198612 w_k1_PG_s132
                                 - 0.00607086705062275 w_k1_PG_s133
                                 - 0.0181154289629003 w_k1_PG_s134
                                 - 0.00694244494575285 w_k1_PG_s135
                                 - 0.0173189038671221 w_k1_PG_s136
                                 - 0.010102487082397 w_k1_PG_s137
                                 - 0.00320899971984235 w_k1_PG_s140
                                 - 0.0034861079025902 w_k1_PG_s141
                                 - 0.00823243420599541 w_k1_PG_s142
                                 - 0.00808690362934114 w_k1_PG_s143
                                 - 0.00554125778644892 w_k1_PG_s144
                                 - 0.00332877623119053 w_k1_PG_s145
                                 - 0.00948759375846412 w_k1_PG_s146
                                 - 0.00238758618263482 w_k1_PG_s147
                                 - 0.0144467743303999 w_k1_PG_s150
                                 - 0.0176924406054782 w_k1_PG_s151
                                 - 0.00922052402955192 w_k1_PG_s152
                                 - 0.00605758131951384 w_k1_PG_s153
                                 - 0.0117588389762829 w_k1_PG_s154
                                 - 0.0231041423096779 w_k1_PG_s155
                                 - 0.0132845448047703 w_k1_PG_s156
                                 - 0.0108840820112756 w_k1_PG_s157
                                 - 0.0123305883201686 w_k1_PG_s160
                                 - 0.0119069958947869 w_k1_PG_s161
                                 - 0.00669542653979041 w_k1_PG_s162
                                 - 0.00307924503019047 w_k1_PG_s163
                                 - 0.00761615887752909 w_k1_PG_s164
                                 - 0.0158592882070368 w_k1_PG_s165
                                 - 0.0119596427659806 w_k1_PG_s166
                                 - 0.00519308931722525 w_k1_PG_s167
                                 - 0.0109491528451578 w_k1_PG_s170
                                 - 0.00559609280088198 w_k1_PG_s171
                                 - 0.00601953925473425 w_k1_PG_s172
                                 - 0.00589800904351258 w_k1_PG_s173
                                 - 0.0110315915671252 w_k1_PG_s174
                                 - 0.0102324204324906 w_k1_PG_s175
                                 - 0.0062531277953918 w_k1_PG_s176
                                 - 0.0112550101552918 w_k1_PG_s177 <= 0
Bounds
 0 <= w_0_ADM <= 1
 0 <= w_0_AXP <= 1
 0 <= w_0_DIS <= 1
 0 <= w_0_FDX <= 1
 0 <= w_0_IBM <= 1
 0 <= w_0_INTC <= 1
 0 <= w_0_KO <= 1
 0 <= w_0_MAT <= 1
 0 <= w_0_MSFT <= 1
 0 <= w_0_PG <= 1
 0 <= w_k1_ADM_s0 <= 1
 0 <= w_k1_ADM_s1 <= 1
 0 <= w_k1_ADM_s00 <= 1
 0 <= w_k1_ADM_s01 <= 1
 0 <= w_k1_ADM_s02 <= 1
 0 <= w_k1_ADM_s03 <= 1
 0 <= w_k1_ADM_s04 <= 1
 0 <= w_k1_ADM_s05 <= 1
 0 <= w_k1_ADM_s06 <= 1
 0 <= w_k1_ADM_s07 <= 1
 0 <= w_k1_ADM_s10 <= 1
 0 <= w_k1_ADM_s11 <= 1
 0 <= w_k1_ADM_s12 <= 1
 0 <= w_k1_ADM_s13 <= 1
 0 <= w_k1_ADM_s14 <= 1
 0 <= w_k1_ADM_s15 <= 1
 0 <= w_k1_ADM_s16 <= 1
 0 <= w_k1_ADM_s17 <= 1
 0 <= w_k1_ADM_s000 <= 1
 0 <= w_k1_ADM_s001 <= 1
 0 <= w_k1_ADM_s002 <= 1
 0 <= w_k1_ADM_s003 <= 1
 0 <= w_k1_ADM_s004 <= 1
 0 <= w_k1_ADM_s005 <= 1
 0 <= w_k1_ADM_s006 <= 1
 0 <= w_k1_ADM_s007 <= 1
 0 <= w_k1_ADM_s010 <= 1
 0 <= w_k1_ADM_s011 <= 1
 0 <= w_k1_ADM_s012 <= 1
 0 <= w_k1_ADM_s013 <= 1
 0 <= w_k1_ADM_s014 <= 1
 0 <= w_k1_ADM_s015 <= 1
 0 <= w_k1_ADM_s016 <= 1
 0 <= w_k1_ADM_s017 <= 1
 0 <= w_k1_ADM_s020 <= 1
 0 <= w_k1_ADM_s021 <= 1
 0 <= w_k1_ADM_s022 <= 1
 0 <= w_k1_ADM_s023 <= 1
 0 <= w_k1_ADM_s024 <= 1
 0 <= w_k1_ADM_s025 <= 1
 0 <= w_k1_ADM_s026 <= 1
 0 <= w_k1_ADM_s027 <= 1
 0 <= w_k1_ADM_s030 <= 1
 0 <= w_k1_ADM_s031 <= 1
 0 <= w_k1_ADM_s032 <= 1
 0 <= w_k1_ADM_s033 <= 1
 0 <= w_k1_ADM_s034 <= 1
 0 <= w_k1_ADM_s035 <= 1
 0 <= w_k1_ADM_s036 <= 1
 0 <= w_k1_ADM_s037 <= 1
 0 <= w_k1_ADM_s040 <= 1
 0 <= w_k1_ADM_s041 <= 1
 0 <= w_k1_ADM_s042 <= 1
 0 <= w_k1_ADM_s043 <= 1
 0 <= w_k1_ADM_s044 <= 1
 0 <= w_k1_ADM_s045 <= 1
 0 <= w_k1_ADM_s046 <= 1
 0 <= w_k1_ADM_s047 <= 1
 0 <= w_k1_ADM_s050 <= 1
 0 <= w_k1_ADM_s051 <= 1
 0 <= w_k1_ADM_s052 <= 1
 0 <= w_k1_ADM_s053 <= 1
 0 <= w_k1_ADM_s054 <= 1
 0 <= w_k1_ADM_s055 <= 1
 0 <= w_k1_ADM_s056 <= 1
 0 <= w_k1_ADM_s057 <= 1
 0 <= w_k1_ADM_s060 <= 1
 0 <= w_k1_ADM_s061 <= 1
 0 <= w_k1_ADM_s062 <= 1
 0 <= w_k1_ADM_s063 <= 1
 0 <= w_k1_ADM_s064 <= 1
 0 <= w_k1_ADM_s065 <= 1
 0 <= w_k1_ADM_s066 <= 1
 0 <= w_k1_ADM_s067 <= 1
 0 <= w_k1_ADM_s070 <= 1
 0 <= w_k1_ADM_s071 <= 1
 0 <= w_k1_ADM_s072 <= 1
 0 <= w_k1_ADM_s073 <= 1
 0 <= w_k1_ADM_s074 <= 1
 0 <= w_k1_ADM_s075 <= 1
 0 <= w_k1_ADM_s076 <= 1
 0 <= w_k1_ADM_s077 <= 1
 0 <= w_k1_ADM_s100 <= 1
 0 <= w_k1_ADM_s101 <= 1
 0 <= w_k1_ADM_s102 <= 1
 0 <= w_k1_ADM_s103 <= 1
 0 <= w_k1_ADM_s104 <= 1
 0 <= w_k1_ADM_s105 <= 1
 0 <= w_k1_ADM_s106 <= 1
 0 <= w_k1_ADM_s107 <= 1
 0 <= w_k1_ADM_s110 <= 1
 0 <= w_k1_ADM_s111 <= 1
 0 <= w_k1_ADM_s112 <= 1
 0 <= w_k1_ADM_s113 <= 1
 0 <= w_k1_ADM_s114 <= 1
 0 <= w_k1_ADM_s115 <= 1
 0 <= w_k1_ADM_s116 <= 1
 0 <= w_k1_ADM_s117 <= 1
 0 <= w_k1_ADM_s120 <= 1
 0 <= w_k1_ADM_s121 <= 1
 0 <= w_k1_ADM_s122 <= 1
 0 <= w_k1_ADM_s123 <= 1
 0 <= w_k1_ADM_s124 <= 1
 0 <= w_k1_ADM_s125 <= 1
 0 <= w_k1_ADM_s126 <= 1
 0 <= w_k1_ADM_s127 <= 1
 0 <= w_k1_ADM_s130 <= 1
 0 <= w_k1_ADM_s131 <= 1
 0 <= w_k1_ADM_s132 <= 1
 0 <= w_k1_ADM_s133 <= 1
 0 <= w_k1_ADM_s134 <= 1
 0 <= w_k1_ADM_s135 <= 1
 0 <= w_k1_ADM_s136 <= 1
 0 <= w_k1_ADM_s137 <= 1
 0 <= w_k1_ADM_s140 <= 1
 0 <= w_k1_ADM_s141 <= 1
 0 <= w_k1_ADM_s142 <= 1
 0 <= w_k1_ADM_s143 <= 1
 0 <= w_k1_ADM_s144 <= 1
 0 <= w_k1_ADM_s145 <= 1
 0 <= w_k1_ADM_s146 <= 1
 0 <= w_k1_ADM_s147 <= 1
 0 <= w_k1_ADM_s150 <= 1
 0 <= w_k1_ADM_s151 <= 1
 0 <= w_k1_ADM_s152 <= 1
 0 <= w_k1_ADM_s153 <= 1
 0 <= w_k1_ADM_s154 <= 1
 0 <= w_k1_ADM_s155 <= 1
 0 <= w_k1_ADM_s156 <= 1
 0 <= w_k1_ADM_s157 <= 1
 0 <= w_k1_ADM_s160 <= 1
 0 <= w_k1_ADM_s161 <= 1
 0 <= w_k1_ADM_s162 <= 1
 0 <= w_k1_ADM_s163 <= 1
 0 <= w_k1_ADM_s164 <= 1
 0 <= w_k1_ADM_s165 <= 1
 0 <= w_k1_ADM_s166 <= 1
 0 <= w_k1_ADM_s167 <= 1
 0 <= w_k1_ADM_s170 <= 1
 0 <= w_k1_ADM_s171 <= 1
 0 <= w_k1_ADM_s172 <= 1
 0 <= w_k1_ADM_s173 <= 1
 0 <= w_k1_ADM_s174 <= 1
 0 <= w_k1_ADM_s175 <= 1
 0 <= w_k1_ADM_s176 <= 1
 0 <= w_k1_ADM_s177 <= 1
 0 <= w_k1_AXP_s0 <= 1
 0 <= w_k1_AXP_s1 <= 1
 0 <= w_k1_AXP_s00 <= 1
 0 <= w_k1_AXP_s01 <= 1
 0 <= w_k1_AXP_s02 <= 1
 0 <= w_k1_AXP_s03 <= 1
 0 <= w_k1_AXP_s04 <= 1
 0 <= w_k1_AXP_s05 <= 1
 0 <= w_k1_AXP_s06 <= 1
 0 <= w_k1_AXP_s07 <= 1
 0 <= w_k1_AXP_s10 <= 1
 0 <= w_k1_AXP_s11 <= 1
 0 <= w_k1_AXP_s12 <= 1
 0 <= w_k1_AXP_s13 <= 1
 0 <= w_k1_AXP_s14 <= 1
 0 <= w_k1_AXP_s15 <= 1
 0 <= w_k1_AXP_s16 <= 1
 0 <= w_k1_AXP_s17 <= 1
 0 <= w_k1_AXP_s000 <= 1
 0 <= w_k1_AXP_s001 <= 1
 0 <= w_k1_AXP_s002 <= 1
 0 <= w_k1_AXP_s003 <= 1
 0 <= w_k1_AXP_s004 <= 1
 0 <= w_k1_AXP_s005 <= 1
 0 <= w_k1_AXP_s006 <= 1
 0 <= w_k1_AXP_s007 <= 1
 0 <= w_k1_AXP_s010 <= 1
 0 <= w_k1_AXP_s011 <= 1
 0 <= w_k1_AXP_s012 <= 1
 0 <= w_k1_AXP_s013 <= 1
 0 <= w_k1_AXP_s014 <= 1
 0 <= w_k1_AXP_s015 <= 1
 0 <= w_k1_AXP_s016 <= 1
 0 <= w_k1_AXP_s017 <= 1
 0 <= w_k1_AXP_s020 <= 1
 0 <= w_k1_AXP_s021 <= 1
 0 <= w_k1_AXP_s022 <= 1
 0 <= w_k1_AXP_s023 <= 1
 0 <= w_k1_AXP_s024 <= 1
 0 <= w_k1_AXP_s025 <= 1
 0 <= w_k1_AXP_s026 <= 1
 0 <= w_k1_AXP_s027 <= 1
 0 <= w_k1_AXP_s030 <= 1
 0 <= w_k1_AXP_s031 <= 1
 0 <= w_k1_AXP_s032 <= 1
 0 <= w_k1_AXP_s033 <= 1
 0 <= w_k1_AXP_s034 <= 1
 0 <= w_k1_AXP_s035 <= 1
 0 <= w_k1_AXP_s036 <= 1
 0 <= w_k1_AXP_s037 <= 1
 0 <= w_k1_AXP_s040 <= 1
 0 <= w_k1_AXP_s041 <= 1
 0 <= w_k1_AXP_s042 <= 1
 0 <= w_k1_AXP_s043 <= 1
 0 <= w_k1_AXP_s044 <= 1
 0 <= w_k1_AXP_s045 <= 1
 0 <= w_k1_AXP_s046 <= 1
 0 <= w_k1_AXP_s047 <= 1
 0 <= w_k1_AXP_s050 <= 1
 0 <= w_k1_AXP_s051 <= 1
 0 <= w_k1_AXP_s052 <= 1
 0 <= w_k1_AXP_s053 <= 1
 0 <= w_k1_AXP_s054 <= 1
 0 <= w_k1_AXP_s055 <= 1
 0 <= w_k1_AXP_s056 <= 1
 0 <= w_k1_AXP_s057 <= 1
 0 <= w_k1_AXP_s060 <= 1
 0 <= w_k1_AXP_s061 <= 1
 0 <= w_k1_AXP_s062 <= 1
 0 <= w_k1_AXP_s063 <= 1
 0 <= w_k1_AXP_s064 <= 1
 0 <= w_k1_AXP_s065 <= 1
 0 <= w_k1_AXP_s066 <= 1
 0 <= w_k1_AXP_s067 <= 1
 0 <= w_k1_AXP_s070 <= 1
 0 <= w_k1_AXP_s071 <= 1
 0 <= w_k1_AXP_s072 <= 1
 0 <= w_k1_AXP_s073 <= 1
 0 <= w_k1_AXP_s074 <= 1
 0 <= w_k1_AXP_s075 <= 1
 0 <= w_k1_AXP_s076 <= 1
 0 <= w_k1_AXP_s077 <= 1
 0 <= w_k1_AXP_s100 <= 1
 0 <= w_k1_AXP_s101 <= 1
 0 <= w_k1_AXP_s102 <= 1
 0 <= w_k1_AXP_s103 <= 1
 0 <= w_k1_AXP_s104 <= 1
 0 <= w_k1_AXP_s105 <= 1
 0 <= w_k1_AXP_s106 <= 1
 0 <= w_k1_AXP_s107 <= 1
 0 <= w_k1_AXP_s110 <= 1
 0 <= w_k1_AXP_s111 <= 1
 0 <= w_k1_AXP_s112 <= 1
 0 <= w_k1_AXP_s113 <= 1
 0 <= w_k1_AXP_s114 <= 1
 0 <= w_k1_AXP_s115 <= 1
 0 <= w_k1_AXP_s116 <= 1
 0 <= w_k1_AXP_s117 <= 1
 0 <= w_k1_AXP_s120 <= 1
 0 <= w_k1_AXP_s121 <= 1
 0 <= w_k1_AXP_s122 <= 1
 0 <= w_k1_AXP_s123 <= 1
 0 <= w_k1_AXP_s124 <= 1
 0 <= w_k1_AXP_s125 <= 1
 0 <= w_k1_AXP_s126 <= 1
 0 <= w_k1_AXP_s127 <= 1
 0 <= w_k1_AXP_s130 <= 1
 0 <= w_k1_AXP_s131 <= 1
 0 <= w_k1_AXP_s132 <= 1
 0 <= w_k1_AXP_s133 <= 1
 0 <= w_k1_AXP_s134 <= 1
 0 <= w_k1_AXP_s135 <= 1
 0 <= w_k1_AXP_s136 <= 1
 0 <= w_k1_AXP_s137 <= 1
 0 <= w_k1_AXP_s140 <= 1
 0 <= w_k1_AXP_s141 <= 1
 0 <= w_k1_AXP_s142 <= 1
 0 <= w_k1_AXP_s143 <= 1
 0 <= w_k1_AXP_s144 <= 1
 0 <= w_k1_AXP_s145 <= 1
 0 <= w_k1_AXP_s146 <= 1
 0 <= w_k1_AXP_s147 <= 1
 0 <= w_k1_AXP_s150 <= 1
 0 <= w_k1_AXP_s151 <= 1
 0 <= w_k1_AXP_s152 <= 1
 0 <= w_k1_AXP_s153 <= 1
 0 <= w_k1_AXP_s154 <= 1
 0 <= w_k1_AXP_s155 <= 1
 0 <= w_k1_AXP_s156 <= 1
 0 <= w_k1_AXP_s157 <= 1
 0 <= w_k1_AXP_s160 <= 1
 0 <= w_k1_AXP_s161 <= 1
 0 <= w_k1_AXP_s162 <= 1
 0 <= w_k1_AXP_s163 <= 1
 0 <= w_k1_AXP_s164 <= 1
 0 <= w_k1_AXP_s165 <= 1
 0 <= w_k1_AXP_s166 <= 1
 0 <= w_k1_AXP_s167 <= 1
 0 <= w_k1_AXP_s170 <= 1
 0 <= w_k1_AXP_s171 <= 1
 0 <= w_k1_AXP_s172 <= 1
 0 <= w_k1_AXP_s173 <= 1
 0 <= w_k1_AXP_s174 <= 1
 0 <= w_k1_AXP_s175 <= 1
 0 <= w_k1_AXP_s176 <= 1
 0 <= w_k1_AXP_s177 <= 1
 0 <= w_k1_DIS_s0 <= 1
 0 <= w_k1_DIS_s1 <= 1
 0 <= w_k1_DIS_s00 <= 1
 0 <= w_k1_DIS_s01 <= 1
 0 <= w_k1_DIS_s02 <= 1
 0 <= w_k1_DIS_s03 <= 1
 0 <= w_k1_DIS_s04 <= 1
 0 <= w_k1_DIS_s05 <= 1
 0 <= w_k1_DIS_s06 <= 1
 0 <= w_k1_DIS_s07 <= 1
 0 <= w_k1_DIS_s10 <= 1
 0 <= w_k1_DIS_s11 <= 1
 0 <= w_k1_DIS_s12 <= 1
 0 <= w_k1_DIS_s13 <= 1
 0 <= w_k1_DIS_s14 <= 1
 0 <= w_k1_DIS_s15 <= 1
 0 <= w_k1_DIS_s16 <= 1
 0 <= w_k1_DIS_s17 <= 1
 0 <= w_k1_DIS_s000 <= 1
 0 <= w_k1_DIS_s001 <= 1
 0 <= w_k1_DIS_s002 <= 1
 0 <= w_k1_DIS_s003 <= 1
 0 <= w_k1_DIS_s004 <= 1
 0 <= w_k1_DIS_s005 <= 1
 0 <= w_k1_DIS_s006 <= 1
 0 <= w_k1_DIS_s007 <= 1
 0 <= w_k1_DIS_s010 <= 1
 0 <= w_k1_DIS_s011 <= 1
 0 <= w_k1_DIS_s012 <= 1
 0 <= w_k1_DIS_s013 <= 1
 0 <= w_k1_DIS_s014 <= 1
 0 <= w_k1_DIS_s015 <= 1
 0 <= w_k1_DIS_s016 <= 1
 0 <= w_k1_DIS_s017 <= 1
 0 <= w_k1_DIS_s020 <= 1
 0 <= w_k1_DIS_s021 <= 1
 0 <= w_k1_DIS_s022 <= 1
 0 <= w_k1_DIS_s023 <= 1
 0 <= w_k1_DIS_s024 <= 1
 0 <= w_k1_DIS_s025 <= 1
 0 <= w_k1_DIS_s026 <= 1
 0 <= w_k1_DIS_s027 <= 1
 0 <= w_k1_DIS_s030 <= 1
 0 <= w_k1_DIS_s031 <= 1
 0 <= w_k1_DIS_s032 <= 1
 0 <= w_k1_DIS_s033 <= 1
 0 <= w_k1_DIS_s034 <= 1
 0 <= w_k1_DIS_s035 <= 1
 0 <= w_k1_DIS_s036 <= 1
 0 <= w_k1_DIS_s037 <= 1
 0 <= w_k1_DIS_s040 <= 1
 0 <= w_k1_DIS_s041 <= 1
 0 <= w_k1_DIS_s042 <= 1
 0 <= w_k1_DIS_s043 <= 1
 0 <= w_k1_DIS_s044 <= 1
 0 <= w_k1_DIS_s045 <= 1
 0 <= w_k1_DIS_s046 <= 1
 0 <= w_k1_DIS_s047 <= 1
 0 <= w_k1_DIS_s050 <= 1
 0 <= w_k1_DIS_s051 <= 1
 0 <= w_k1_DIS_s052 <= 1
 0 <= w_k1_DIS_s053 <= 1
 0 <= w_k1_DIS_s054 <= 1
 0 <= w_k1_DIS_s055 <= 1
 0 <= w_k1_DIS_s056 <= 1
 0 <= w_k1_DIS_s057 <= 1
 0 <= w_k1_DIS_s060 <= 1
 0 <= w_k1_DIS_s061 <= 1
 0 <= w_k1_DIS_s062 <= 1
 0 <= w_k1_DIS_s063 <= 1
 0 <= w_k1_DIS_s064 <= 1
 0 <= w_k1_DIS_s065 <= 1
 0 <= w_k1_DIS_s066 <= 1
 0 <= w_k1_DIS_s067 <= 1
 0 <= w_k1_DIS_s070 <= 1
 0 <= w_k1_DIS_s071 <= 1
 0 <= w_k1_DIS_s072 <= 1
 0 <= w_k1_DIS_s073 <= 1
 0 <= w_k1_DIS_s074 <= 1
 0 <= w_k1_DIS_s075 <= 1
 0 <= w_k1_DIS_s076 <= 1
 0 <= w_k1_DIS_s077 <= 1
 0 <= w_k1_DIS_s100 <= 1
 0 <= w_k1_DIS_s101 <= 1
 0 <= w_k1_DIS_s102 <= 1
 0 <= w_k1_DIS_s103 <= 1
 0 <= w_k1_DIS_s104 <= 1
 0 <= w_k1_DIS_s105 <= 1
 0 <= w_k1_DIS_s106 <= 1
 0 <= w_k1_DIS_s107 <= 1
 0 <= w_k1_DIS_s110 <= 1
 0 <= w_k1_DIS_s111 <= 1
 0 <= w_k1_DIS_s112 <= 1
 0 <= w_k1_DIS_s113 <= 1
 0 <= w_k1_DIS_s114 <= 1
 0 <= w_k1_DIS_s115 <= 1
 0 <= w_k1_DIS_s116 <= 1
 0 <= w_k1_DIS_s117 <= 1
 0 <= w_k1_DIS_s120 <= 1
 0 <= w_k1_DIS_s121 <= 1
 0 <= w_k1_DIS_s122 <= 1
 0 <= w_k1_DIS_s123 <= 1
 0 <= w_k1_DIS_s124 <= 1
 0 <= w_k1_DIS_s125 <= 1
 0 <= w_k1_DIS_s126 <= 1
 0 <= w_k1_DIS_s127 <= 1
 0 <= w_k1_DIS_s130 <= 1
 0 <= w_k1_DIS_s131 <= 1
 0 <= w_k1_DIS_s132 <= 1
 0 <= w_k1_DIS_s133 <= 1
 0 <= w_k1_DIS_s134 <= 1
 0 <= w_k1_DIS_s135 <= 1
 0 <= w_k1_DIS_s136 <= 1
 0 <= w_k1_DIS_s137 <= 1
 0 <= w_k1_DIS_s140 <= 1
 0 <= w_k1_DIS_s141 <= 1
 0 <= w_k1_DIS_s142 <= 1
 0 <= w_k1_DIS_s143 <= 1
 0 <= w_k1_DIS_s144 <= 1
 0 <= w_k1_DIS_s145 <= 1
 0 <= w_k1_DIS_s146 <= 1
 0 <= w_k1_DIS_s147 <= 1
 0 <= w_k1_DIS_s150 <= 1
 0 <= w_k1_DIS_s151 <= 1
 0 <= w_k1_DIS_s152 <= 1
 0 <= w_k1_DIS_s153 <= 1
 0 <= w_k1_DIS_s154 <= 1
 0 <= w_k1_DIS_s155 <= 1
 0 <= w_k1_DIS_s156 <= 1
 0 <= w_k1_DIS_s157 <= 1
 0 <= w_k1_DIS_s160 <= 1
 0 <= w_k1_DIS_s161 <= 1
 0 <= w_k1_DIS_s162 <= 1
 0 <= w_k1_DIS_s163 <= 1
 0 <= w_k1_DIS_s164 <= 1
 0 <= w_k1_DIS_s165 <= 1
 0 <= w_k1_DIS_s166 <= 1
 0 <= w_k1_DIS_s167 <= 1
 0 <= w_k1_DIS_s170 <= 1
 0 <= w_k1_DIS_s171 <= 1
 0 <= w_k1_DIS_s172 <= 1
 0 <= w_k1_DIS_s173 <= 1
 0 <= w_k1_DIS_s174 <= 1
 0 <= w_k1_DIS_s175 <= 1
 0 <= w_k1_DIS_s176 <= 1
 0 <= w_k1_DIS_s177 <= 1
 0 <= w_k1_FDX_s0 <= 1
 0 <= w_k1_FDX_s1 <= 1
 0 <= w_k1_FDX_s00 <= 1
 0 <= w_k1_FDX_s01 <= 1
 0 <= w_k1_FDX_s02 <= 1
 0 <= w_k1_FDX_s03 <= 1
 0 <= w_k1_FDX_s04 <= 1
 0 <= w_k1_FDX_s05 <= 1
 0 <= w_k1_FDX_s06 <= 1
 0 <= w_k1_FDX_s07 <= 1
 0 <= w_k1_FDX_s10 <= 1
 0 <= w_k1_FDX_s11 <= 1
 0 <= w_k1_FDX_s12 <= 1
 0 <= w_k1_FDX_s13 <= 1
 0 <= w_k1_FDX_s14 <= 1
 0 <= w_k1_FDX_s15 <= 1
 0 <= w_k1_FDX_s16 <= 1
 0 <= w_k1_FDX_s17 <= 1
 0 <= w_k1_FDX_s000 <= 1
 0 <= w_k1_FDX_s001 <= 1
 0 <= w_k1_FDX_s002 <= 1
 0 <= w_k1_FDX_s003 <= 1
 0 <= w_k1_FDX_s004 <= 1
 0 <= w_k1_FDX_s005 <= 1
 0 <= w_k1_FDX_s006 <= 1
 0 <= w_k1_FDX_s007 <= 1
 0 <= w_k1_FDX_s010 <= 1
 0 <= w_k1_FDX_s011 <= 1
 0 <= w_k1_FDX_s012 <= 1
 0 <= w_k1_FDX_s013 <= 1
 0 <= w_k1_FDX_s014 <= 1
 0 <= w_k1_FDX_s015 <= 1
 0 <= w_k1_FDX_s016 <= 1
 0 <= w_k1_FDX_s017 <= 1
 0 <= w_k1_FDX_s020 <= 1
 0 <= w_k1_FDX_s021 <= 1
 0 <= w_k1_FDX_s022 <= 1
 0 <= w_k1_FDX_s023 <= 1
 0 <= w_k1_FDX_s024 <= 1
 0 <= w_k1_FDX_s025 <= 1
 0 <= w_k1_FDX_s026 <= 1
 0 <= w_k1_FDX_s027 <= 1
 0 <= w_k1_FDX_s030 <= 1
 0 <= w_k1_FDX_s031 <= 1
 0 <= w_k1_FDX_s032 <= 1
 0 <= w_k1_FDX_s033 <= 1
 0 <= w_k1_FDX_s034 <= 1
 0 <= w_k1_FDX_s035 <= 1
 0 <= w_k1_FDX_s036 <= 1
 0 <= w_k1_FDX_s037 <= 1
 0 <= w_k1_FDX_s040 <= 1
 0 <= w_k1_FDX_s041 <= 1
 0 <= w_k1_FDX_s042 <= 1
 0 <= w_k1_FDX_s043 <= 1
 0 <= w_k1_FDX_s044 <= 1
 0 <= w_k1_FDX_s045 <= 1
 0 <= w_k1_FDX_s046 <= 1
 0 <= w_k1_FDX_s047 <= 1
 0 <= w_k1_FDX_s050 <= 1
 0 <= w_k1_FDX_s051 <= 1
 0 <= w_k1_FDX_s052 <= 1
 0 <= w_k1_FDX_s053 <= 1
 0 <= w_k1_FDX_s054 <= 1
 0 <= w_k1_FDX_s055 <= 1
 0 <= w_k1_FDX_s056 <= 1
 0 <= w_k1_FDX_s057 <= 1
 0 <= w_k1_FDX_s060 <= 1
 0 <= w_k1_FDX_s061 <= 1
 0 <= w_k1_FDX_s062 <= 1
 0 <= w_k1_FDX_s063 <= 1
 0 <= w_k1_FDX_s064 <= 1
 0 <= w_k1_FDX_s065 <= 1
 0 <= w_k1_FDX_s066 <= 1
 0 <= w_k1_FDX_s067 <= 1
 0 <= w_k1_FDX_s070 <= 1
 0 <= w_k1_FDX_s071 <= 1
 0 <= w_k1_FDX_s072 <= 1
 0 <= w_k1_FDX_s073 <= 1
 0 <= w_k1_FDX_s074 <= 1
 0 <= w_k1_FDX_s075 <= 1
 0 <= w_k1_FDX_s076 <= 1
 0 <= w_k1_FDX_s077 <= 1
 0 <= w_k1_FDX_s100 <= 1
 0 <= w_k1_FDX_s101 <= 1
 0 <= w_k1_FDX_s102 <= 1
 0 <= w_k1_FDX_s103 <= 1
 0 <= w_k1_FDX_s104 <= 1
 0 <= w_k1_FDX_s105 <= 1
 0 <= w_k1_FDX_s106 <= 1
 0 <= w_k1_FDX_s107 <= 1
 0 <= w_k1_FDX_s110 <= 1
 0 <= w_k1_FDX_s111 <= 1
 0 <= w_k1_FDX_s112 <= 1
 0 <= w_k1_FDX_s113 <= 1
 0 <= w_k1_FDX_s114 <= 1
 0 <= w_k1_FDX_s115 <= 1
 0 <= w_k1_FDX_s116 <= 1
 0 <= w_k1_FDX_s117 <= 1
 0 <= w_k1_FDX_s120 <= 1
 0 <= w_k1_FDX_s121 <= 1
 0 <= w_k1_FDX_s122 <= 1
 0 <= w_k1_FDX_s123 <= 1
 0 <= w_k1_FDX_s124 <= 1
 0 <= w_k1_FDX_s125 <= 1
 0 <= w_k1_FDX_s126 <= 1
 0 <= w_k1_FDX_s127 <= 1
 0 <= w_k1_FDX_s130 <= 1
 0 <= w_k1_FDX_s131 <= 1
 0 <= w_k1_FDX_s132 <= 1
 0 <= w_k1_FDX_s133 <= 1
 0 <= w_k1_FDX_s134 <= 1
 0 <= w_k1_FDX_s135 <= 1
 0 <= w_k1_FDX_s136 <= 1
 0 <= w_k1_FDX_s137 <= 1
 0 <= w_k1_FDX_s140 <= 1
 0 <= w_k1_FDX_s141 <= 1
 0 <= w_k1_FDX_s142 <= 1
 0 <= w_k1_FDX_s143 <= 1
 0 <= w_k1_FDX_s144 <= 1
 0 <= w_k1_FDX_s145 <= 1
 0 <= w_k1_FDX_s146 <= 1
 0 <= w_k1_FDX_s147 <= 1
 0 <= w_k1_FDX_s150 <= 1
 0 <= w_k1_FDX_s151 <= 1
 0 <= w_k1_FDX_s152 <= 1
 0 <= w_k1_FDX_s153 <= 1
 0 <= w_k1_FDX_s154 <= 1
 0 <= w_k1_FDX_s155 <= 1
 0 <= w_k1_FDX_s156 <= 1
 0 <= w_k1_FDX_s157 <= 1
 0 <= w_k1_FDX_s160 <= 1
 0 <= w_k1_FDX_s161 <= 1
 0 <= w_k1_FDX_s162 <= 1
 0 <= w_k1_FDX_s163 <= 1
 0 <= w_k1_FDX_s164 <= 1
 0 <= w_k1_FDX_s165 <= 1
 0 <= w_k1_FDX_s166 <= 1
 0 <= w_k1_FDX_s167 <= 1
 0 <= w_k1_FDX_s170 <= 1
 0 <= w_k1_FDX_s171 <= 1
 0 <= w_k1_FDX_s172 <= 1
 0 <= w_k1_FDX_s173 <= 1
 0 <= w_k1_FDX_s174 <= 1
 0 <= w_k1_FDX_s175 <= 1
 0 <= w_k1_FDX_s176 <= 1
 0 <= w_k1_FDX_s177 <= 1
 0 <= w_k1_IBM_s0 <= 1
 0 <= w_k1_IBM_s1 <= 1
 0 <= w_k1_IBM_s00 <= 1
 0 <= w_k1_IBM_s01 <= 1
 0 <= w_k1_IBM_s02 <= 1
 0 <= w_k1_IBM_s03 <= 1
 0 <= w_k1_IBM_s04 <= 1
 0 <= w_k1_IBM_s05 <= 1
 0 <= w_k1_IBM_s06 <= 1
 0 <= w_k1_IBM_s07 <= 1
 0 <= w_k1_IBM_s10 <= 1
 0 <= w_k1_IBM_s11 <= 1
 0 <= w_k1_IBM_s12 <= 1
 0 <= w_k1_IBM_s13 <= 1
 0 <= w_k1_IBM_s14 <= 1
 0 <= w_k1_IBM_s15 <= 1
 0 <= w_k1_IBM_s16 <= 1
 0 <= w_k1_IBM_s17 <= 1
 0 <= w_k1_IBM_s000 <= 1
 0 <= w_k1_IBM_s001 <= 1
 0 <= w_k1_IBM_s002 <= 1
 0 <= w_k1_IBM_s003 <= 1
 0 <= w_k1_IBM_s004 <= 1
 0 <= w_k1_IBM_s005 <= 1
 0 <= w_k1_IBM_s006 <= 1
 0 <= w_k1_IBM_s007 <= 1
 0 <= w_k1_IBM_s010 <= 1
 0 <= w_k1_IBM_s011 <= 1
 0 <= w_k1_IBM_s012 <= 1
 0 <= w_k1_IBM_s013 <= 1
 0 <= w_k1_IBM_s014 <= 1
 0 <= w_k1_IBM_s015 <= 1
 0 <= w_k1_IBM_s016 <= 1
 0 <= w_k1_IBM_s017 <= 1
 0 <= w_k1_IBM_s020 <= 1
 0 <= w_k1_IBM_s021 <= 1
 0 <= w_k1_IBM_s022 <= 1
 0 <= w_k1_IBM_s023 <= 1
 0 <= w_k1_IBM_s024 <= 1
 0 <= w_k1_IBM_s025 <= 1
 0 <= w_k1_IBM_s026 <= 1
 0 <= w_k1_IBM_s027 <= 1
 0 <= w_k1_IBM_s030 <= 1
 0 <= w_k1_IBM_s031 <= 1
 0 <= w_k1_IBM_s032 <= 1
 0 <= w_k1_IBM_s033 <= 1
 0 <= w_k1_IBM_s034 <= 1
 0 <= w_k1_IBM_s035 <= 1
 0 <= w_k1_IBM_s036 <= 1
 0 <= w_k1_IBM_s037 <= 1
 0 <= w_k1_IBM_s040 <= 1
 0 <= w_k1_IBM_s041 <= 1
 0 <= w_k1_IBM_s042 <= 1
 0 <= w_k1_IBM_s043 <= 1
 0 <= w_k1_IBM_s044 <= 1
 0 <= w_k1_IBM_s045 <= 1
 0 <= w_k1_IBM_s046 <= 1
 0 <= w_k1_IBM_s047 <= 1
 0 <= w_k1_IBM_s050 <= 1
 0 <= w_k1_IBM_s051 <= 1
 0 <= w_k1_IBM_s052 <= 1
 0 <= w_k1_IBM_s053 <= 1
 0 <= w_k1_IBM_s054 <= 1
 0 <= w_k1_IBM_s055 <= 1
 0 <= w_k1_IBM_s056 <= 1
 0 <= w_k1_IBM_s057 <= 1
 0 <= w_k1_IBM_s060 <= 1
 0 <= w_k1_IBM_s061 <= 1
 0 <= w_k1_IBM_s062 <= 1
 0 <= w_k1_IBM_s063 <= 1
 0 <= w_k1_IBM_s064 <= 1
 0 <= w_k1_IBM_s065 <= 1
 0 <= w_k1_IBM_s066 <= 1
 0 <= w_k1_IBM_s067 <= 1
 0 <= w_k1_IBM_s070 <= 1
 0 <= w_k1_IBM_s071 <= 1
 0 <= w_k1_IBM_s072 <= 1
 0 <= w_k1_IBM_s073 <= 1
 0 <= w_k1_IBM_s074 <= 1
 0 <= w_k1_IBM_s075 <= 1
 0 <= w_k1_IBM_s076 <= 1
 0 <= w_k1_IBM_s077 <= 1
 0 <= w_k1_IBM_s100 <= 1
 0 <= w_k1_IBM_s101 <= 1
 0 <= w_k1_IBM_s102 <= 1
 0 <= w_k1_IBM_s103 <= 1
 0 <= w_k1_IBM_s104 <= 1
 0 <= w_k1_IBM_s105 <= 1
 0 <= w_k1_IBM_s106 <= 1
 0 <= w_k1_IBM_s107 <= 1
 0 <= w_k1_IBM_s110 <= 1
 0 <= w_k1_IBM_s111 <= 1
 0 <= w_k1_IBM_s112 <= 1
 0 <= w_k1_IBM_s113 <= 1
 0 <= w_k1_IBM_s114 <= 1
 0 <= w_k1_IBM_s115 <= 1
 0 <= w_k1_IBM_s116 <= 1
 0 <= w_k1_IBM_s117 <= 1
 0 <= w_k1_IBM_s120 <= 1
 0 <= w_k1_IBM_s121 <= 1
 0 <= w_k1_IBM_s122 <= 1
 0 <= w_k1_IBM_s123 <= 1
 0 <= w_k1_IBM_s124 <= 1
 0 <= w_k1_IBM_s125 <= 1
 0 <= w_k1_IBM_s126 <= 1
 0 <= w_k1_IBM_s127 <= 1
 0 <= w_k1_IBM_s130 <= 1
 0 <= w_k1_IBM_s131 <= 1
 0 <= w_k1_IBM_s132 <= 1
 0 <= w_k1_IBM_s133 <= 1
 0 <= w_k1_IBM_s134 <= 1
 0 <= w_k1_IBM_s135 <= 1
 0 <= w_k1_IBM_s136 <= 1
 0 <= w_k1_IBM_s137 <= 1
 0 <= w_k1_IBM_s140 <= 1
 0 <= w_k1_IBM_s141 <= 1
 0 <= w_k1_IBM_s142 <= 1
 0 <= w_k1_IBM_s143 <= 1
 0 <= w_k1_IBM_s144 <= 1
 0 <= w_k1_IBM_s145 <= 1
 0 <= w_k1_IBM_s146 <= 1
 0 <= w_k1_IBM_s147 <= 1
 0 <= w_k1_IBM_s150 <= 1
 0 <= w_k1_IBM_s151 <= 1
 0 <= w_k1_IBM_s152 <= 1
 0 <= w_k1_IBM_s153 <= 1
 0 <= w_k1_IBM_s154 <= 1
 0 <= w_k1_IBM_s155 <= 1
 0 <= w_k1_IBM_s156 <= 1
 0 <= w_k1_IBM_s157 <= 1
 0 <= w_k1_IBM_s160 <= 1
 0 <= w_k1_IBM_s161 <= 1
 0 <= w_k1_IBM_s162 <= 1
 0 <= w_k1_IBM_s163 <= 1
 0 <= w_k1_IBM_s164 <= 1
 0 <= w_k1_IBM_s165 <= 1
 0 <= w_k1_IBM_s166 <= 1
 0 <= w_k1_IBM_s167 <= 1
 0 <= w_k1_IBM_s170 <= 1
 0 <= w_k1_IBM_s171 <= 1
 0 <= w_k1_IBM_s172 <= 1
 0 <= w_k1_IBM_s173 <= 1
 0 <= w_k1_IBM_s174 <= 1
 0 <= w_k1_IBM_s175 <= 1
 0 <= w_k1_IBM_s176 <= 1
 0 <= w_k1_IBM_s177 <= 1
 0 <= w_k1_INTC_s0 <= 1
 0 <= w_k1_INTC_s1 <= 1
 0 <= w_k1_INTC_s00 <= 1
 0 <= w_k1_INTC_s01 <= 1
 0 <= w_k1_INTC_s02 <= 1
 0 <= w_k1_INTC_s03 <= 1
 0 <= w_k1_INTC_s04 <= 1
 0 <= w_k1_INTC_s05 <= 1
 0 <= w_k1_INTC_s06 <= 1
 0 <= w_k1_INTC_s07 <= 1
 0 <= w_k1_INTC_s10 <= 1
 0 <= w_k1_INTC_s11 <= 1
 0 <= w_k1_INTC_s12 <= 1
 0 <= w_k1_INTC_s13 <= 1
 0 <= w_k1_INTC_s14 <= 1
 0 <= w_k1_INTC_s15 <= 1
 0 <= w_k1_INTC_s16 <= 1
 0 <= w_k1_INTC_s17 <= 1
 0 <= w_k1_INTC_s000 <= 1
 0 <= w_k1_INTC_s001 <= 1
 0 <= w_k1_INTC_s002 <= 1
 0 <= w_k1_INTC_s003 <= 1
 0 <= w_k1_INTC_s004 <= 1
 0 <= w_k1_INTC_s005 <= 1
 0 <= w_k1_INTC_s006 <= 1
 0 <= w_k1_INTC_s007 <= 1
 0 <= w_k1_INTC_s010 <= 1
 0 <= w_k1_INTC_s011 <= 1
 0 <= w_k1_INTC_s012 <= 1
 0 <= w_k1_INTC_s013 <= 1
 0 <= w_k1_INTC_s014 <= 1
 0 <= w_k1_INTC_s015 <= 1
 0 <= w_k1_INTC_s016 <= 1
 0 <= w_k1_INTC_s017 <= 1
 0 <= w_k1_INTC_s020 <= 1
 0 <= w_k1_INTC_s021 <= 1
 0 <= w_k1_INTC_s022 <= 1
 0 <= w_k1_INTC_s023 <= 1
 0 <= w_k1_INTC_s024 <= 1
 0 <= w_k1_INTC_s025 <= 1
 0 <= w_k1_INTC_s026 <= 1
 0 <= w_k1_INTC_s027 <= 1
 0 <= w_k1_INTC_s030 <= 1
 0 <= w_k1_INTC_s031 <= 1
 0 <= w_k1_INTC_s032 <= 1
 0 <= w_k1_INTC_s033 <= 1
 0 <= w_k1_INTC_s034 <= 1
 0 <= w_k1_INTC_s035 <= 1
 0 <= w_k1_INTC_s036 <= 1
 0 <= w_k1_INTC_s037 <= 1
 0 <= w_k1_INTC_s040 <= 1
 0 <= w_k1_INTC_s041 <= 1
 0 <= w_k1_INTC_s042 <= 1
 0 <= w_k1_INTC_s043 <= 1
 0 <= w_k1_INTC_s044 <= 1
 0 <= w_k1_INTC_s045 <= 1
 0 <= w_k1_INTC_s046 <= 1
 0 <= w_k1_INTC_s047 <= 1
 0 <= w_k1_INTC_s050 <= 1
 0 <= w_k1_INTC_s051 <= 1
 0 <= w_k1_INTC_s052 <= 1
 0 <= w_k1_INTC_s053 <= 1
 0 <= w_k1_INTC_s054 <= 1
 0 <= w_k1_INTC_s055 <= 1
 0 <= w_k1_INTC_s056 <= 1
 0 <= w_k1_INTC_s057 <= 1
 0 <= w_k1_INTC_s060 <= 1
 0 <= w_k1_INTC_s061 <= 1
 0 <= w_k1_INTC_s062 <= 1
 0 <= w_k1_INTC_s063 <= 1
 0 <= w_k1_INTC_s064 <= 1
 0 <= w_k1_INTC_s065 <= 1
 0 <= w_k1_INTC_s066 <= 1
 0 <= w_k1_INTC_s067 <= 1
 0 <= w_k1_INTC_s070 <= 1
 0 <= w_k1_INTC_s071 <= 1
 0 <= w_k1_INTC_s072 <= 1
 0 <= w_k1_INTC_s073 <= 1
 0 <= w_k1_INTC_s074 <= 1
 0 <= w_k1_INTC_s075 <= 1
 0 <= w_k1_INTC_s076 <= 1
 0 <= w_k1_INTC_s077 <= 1
 0 <= w_k1_INTC_s100 <= 1
 0 <= w_k1_INTC_s101 <= 1
 0 <= w_k1_INTC_s102 <= 1
 0 <= w_k1_INTC_s103 <= 1
 0 <= w_k1_INTC_s104 <= 1
 0 <= w_k1_INTC_s105 <= 1
 0 <= w_k1_INTC_s106 <= 1
 0 <= w_k1_INTC_s107 <= 1
 0 <= w_k1_INTC_s110 <= 1
 0 <= w_k1_INTC_s111 <= 1
 0 <= w_k1_INTC_s112 <= 1
 0 <= w_k1_INTC_s113 <= 1
 0 <= w_k1_INTC_s114 <= 1
 0 <= w_k1_INTC_s115 <= 1
 0 <= w_k1_INTC_s116 <= 1
 0 <= w_k1_INTC_s117 <= 1
 0 <= w_k1_INTC_s120 <= 1
 0 <= w_k1_INTC_s121 <= 1
 0 <= w_k1_INTC_s122 <= 1
 0 <= w_k1_INTC_s123 <= 1
 0 <= w_k1_INTC_s124 <= 1
 0 <= w_k1_INTC_s125 <= 1
 0 <= w_k1_INTC_s126 <= 1
 0 <= w_k1_INTC_s127 <= 1
 0 <= w_k1_INTC_s130 <= 1
 0 <= w_k1_INTC_s131 <= 1
 0 <= w_k1_INTC_s132 <= 1
 0 <= w_k1_INTC_s133 <= 1
 0 <= w_k1_INTC_s134 <= 1
 0 <= w_k1_INTC_s135 <= 1
 0 <= w_k1_INTC_s136 <= 1
 0 <= w_k1_INTC_s137 <= 1
 0 <= w_k1_INTC_s140 <= 1
 0 <= w_k1_INTC_s141 <= 1
 0 <= w_k1_INTC_s142 <= 1
 0 <= w_k1_INTC_s143 <= 1
 0 <= w_k1_INTC_s144 <= 1
 0 <= w_k1_INTC_s145 <= 1
 0 <= w_k1_INTC_s146 <= 1
 0 <= w_k1_INTC_s147 <= 1
 0 <= w_k1_INTC_s150 <= 1
 0 <= w_k1_INTC_s151 <= 1
 0 <= w_k1_INTC_s152 <= 1
 0 <= w_k1_INTC_s153 <= 1
 0 <= w_k1_INTC_s154 <= 1
 0 <= w_k1_INTC_s155 <= 1
 0 <= w_k1_INTC_s156 <= 1
 0 <= w_k1_INTC_s157 <= 1
 0 <= w_k1_INTC_s160 <= 1
 0 <= w_k1_INTC_s161 <= 1
 0 <= w_k1_INTC_s162 <= 1
 0 <= w_k1_INTC_s163 <= 1
 0 <= w_k1_INTC_s164 <= 1
 0 <= w_k1_INTC_s165 <= 1
 0 <= w_k1_INTC_s166 <= 1
 0 <= w_k1_INTC_s167 <= 1
 0 <= w_k1_INTC_s170 <= 1
 0 <= w_k1_INTC_s171 <= 1
 0 <= w_k1_INTC_s172 <= 1
 0 <= w_k1_INTC_s173 <= 1
 0 <= w_k1_INTC_s174 <= 1
 0 <= w_k1_INTC_s175 <= 1
 0 <= w_k1_INTC_s176 <= 1
 0 <= w_k1_INTC_s177 <= 1
 0 <= w_k1_KO_s0 <= 1
 0 <= w_k1_KO_s1 <= 1
 0 <= w_k1_KO_s00 <= 1
 0 <= w_k1_KO_s01 <= 1
 0 <= w_k1_KO_s02 <= 1
 0 <= w_k1_KO_s03 <= 1
 0 <= w_k1_KO_s04 <= 1
 0 <= w_k1_KO_s05 <= 1
 0 <= w_k1_KO_s06 <= 1
 0 <= w_k1_KO_s07 <= 1
 0 <= w_k1_KO_s10 <= 1
 0 <= w_k1_KO_s11 <= 1
 0 <= w_k1_KO_s12 <= 1
 0 <= w_k1_KO_s13 <= 1
 0 <= w_k1_KO_s14 <= 1
 0 <= w_k1_KO_s15 <= 1
 0 <= w_k1_KO_s16 <= 1
 0 <= w_k1_KO_s17 <= 1
 0 <= w_k1_KO_s000 <= 1
 0 <= w_k1_KO_s001 <= 1
 0 <= w_k1_KO_s002 <= 1
 0 <= w_k1_KO_s003 <= 1
 0 <= w_k1_KO_s004 <= 1
 0 <= w_k1_KO_s005 <= 1
 0 <= w_k1_KO_s006 <= 1
 0 <= w_k1_KO_s007 <= 1
 0 <= w_k1_KO_s010 <= 1
 0 <= w_k1_KO_s011 <= 1
 0 <= w_k1_KO_s012 <= 1
 0 <= w_k1_KO_s013 <= 1
 0 <= w_k1_KO_s014 <= 1
 0 <= w_k1_KO_s015 <= 1
 0 <= w_k1_KO_s016 <= 1
 0 <= w_k1_KO_s017 <= 1
 0 <= w_k1_KO_s020 <= 1
 0 <= w_k1_KO_s021 <= 1
 0 <= w_k1_KO_s022 <= 1
 0 <= w_k1_KO_s023 <= 1
 0 <= w_k1_KO_s024 <= 1
 0 <= w_k1_KO_s025 <= 1
 0 <= w_k1_KO_s026 <= 1
 0 <= w_k1_KO_s027 <= 1
 0 <= w_k1_KO_s030 <= 1
 0 <= w_k1_KO_s031 <= 1
 0 <= w_k1_KO_s032 <= 1
 0 <= w_k1_KO_s033 <= 1
 0 <= w_k1_KO_s034 <= 1
 0 <= w_k1_KO_s035 <= 1
 0 <= w_k1_KO_s036 <= 1
 0 <= w_k1_KO_s037 <= 1
 0 <= w_k1_KO_s040 <= 1
 0 <= w_k1_KO_s041 <= 1
 0 <= w_k1_KO_s042 <= 1
 0 <= w_k1_KO_s043 <= 1
 0 <= w_k1_KO_s044 <= 1
 0 <= w_k1_KO_s045 <= 1
 0 <= w_k1_KO_s046 <= 1
 0 <= w_k1_KO_s047 <= 1
 0 <= w_k1_KO_s050 <= 1
 0 <= w_k1_KO_s051 <= 1
 0 <= w_k1_KO_s052 <= 1
 0 <= w_k1_KO_s053 <= 1
 0 <= w_k1_KO_s054 <= 1
 0 <= w_k1_KO_s055 <= 1
 0 <= w_k1_KO_s056 <= 1
 0 <= w_k1_KO_s057 <= 1
 0 <= w_k1_KO_s060 <= 1
 0 <= w_k1_KO_s061 <= 1
 0 <= w_k1_KO_s062 <= 1
 0 <= w_k1_KO_s063 <= 1
 0 <= w_k1_KO_s064 <= 1
 0 <= w_k1_KO_s065 <= 1
 0 <= w_k1_KO_s066 <= 1
 0 <= w_k1_KO_s067 <= 1
 0 <= w_k1_KO_s070 <= 1
 0 <= w_k1_KO_s071 <= 1
 0 <= w_k1_KO_s072 <= 1
 0 <= w_k1_KO_s073 <= 1
 0 <= w_k1_KO_s074 <= 1
 0 <= w_k1_KO_s075 <= 1
 0 <= w_k1_KO_s076 <= 1
 0 <= w_k1_KO_s077 <= 1
 0 <= w_k1_KO_s100 <= 1
 0 <= w_k1_KO_s101 <= 1
 0 <= w_k1_KO_s102 <= 1
 0 <= w_k1_KO_s103 <= 1
 0 <= w_k1_KO_s104 <= 1
 0 <= w_k1_KO_s105 <= 1
 0 <= w_k1_KO_s106 <= 1
 0 <= w_k1_KO_s107 <= 1
 0 <= w_k1_KO_s110 <= 1
 0 <= w_k1_KO_s111 <= 1
 0 <= w_k1_KO_s112 <= 1
 0 <= w_k1_KO_s113 <= 1
 0 <= w_k1_KO_s114 <= 1
 0 <= w_k1_KO_s115 <= 1
 0 <= w_k1_KO_s116 <= 1
 0 <= w_k1_KO_s117 <= 1
 0 <= w_k1_KO_s120 <= 1
 0 <= w_k1_KO_s121 <= 1
 0 <= w_k1_KO_s122 <= 1
 0 <= w_k1_KO_s123 <= 1
 0 <= w_k1_KO_s124 <= 1
 0 <= w_k1_KO_s125 <= 1
 0 <= w_k1_KO_s126 <= 1
 0 <= w_k1_KO_s127 <= 1
 0 <= w_k1_KO_s130 <= 1
 0 <= w_k1_KO_s131 <= 1
 0 <= w_k1_KO_s132 <= 1
 0 <= w_k1_KO_s133 <= 1
 0 <= w_k1_KO_s134 <= 1
 0 <= w_k1_KO_s135 <= 1
 0 <= w_k1_KO_s136 <= 1
 0 <= w_k1_KO_s137 <= 1
 0 <= w_k1_KO_s140 <= 1
 0 <= w_k1_KO_s141 <= 1
 0 <= w_k1_KO_s142 <= 1
 0 <= w_k1_KO_s143 <= 1
 0 <= w_k1_KO_s144 <= 1
 0 <= w_k1_KO_s145 <= 1
 0 <= w_k1_KO_s146 <= 1
 0 <= w_k1_KO_s147 <= 1
 0 <= w_k1_KO_s150 <= 1
 0 <= w_k1_KO_s151 <= 1
 0 <= w_k1_KO_s152 <= 1
 0 <= w_k1_KO_s153 <= 1
 0 <= w_k1_KO_s154 <= 1
 0 <= w_k1_KO_s155 <= 1
 0 <= w_k1_KO_s156 <= 1
 0 <= w_k1_KO_s157 <= 1
 0 <= w_k1_KO_s160 <= 1
 0 <= w_k1_KO_s161 <= 1
 0 <= w_k1_KO_s162 <= 1
 0 <= w_k1_KO_s163 <= 1
 0 <= w_k1_KO_s164 <= 1
 0 <= w_k1_KO_s165 <= 1
 0 <= w_k1_KO_s166 <= 1
 0 <= w_k1_KO_s167 <= 1
 0 <= w_k1_KO_s170 <= 1
 0 <= w_k1_KO_s171 <= 1
 0 <= w_k1_KO_s172 <= 1
 0 <= w_k1_KO_s173 <= 1
 0 <= w_k1_KO_s174 <= 1
 0 <= w_k1_KO_s175 <= 1
 0 <= w_k1_KO_s176 <= 1
 0 <= w_k1_KO_s177 <= 1
 0 <= w_k1_MAT_s0 <= 1
 0 <= w_k1_MAT_s1 <= 1
 0 <= w_k1_MAT_s00 <= 1
 0 <= w_k1_MAT_s01 <= 1
 0 <= w_k1_MAT_s02 <= 1
 0 <= w_k1_MAT_s03 <= 1
 0 <= w_k1_MAT_s04 <= 1
 0 <= w_k1_MAT_s05 <= 1
 0 <= w_k1_MAT_s06 <= 1
 0 <= w_k1_MAT_s07 <= 1
 0 <= w_k1_MAT_s10 <= 1
 0 <= w_k1_MAT_s11 <= 1
 0 <= w_k1_MAT_s12 <= 1
 0 <= w_k1_MAT_s13 <= 1
 0 <= w_k1_MAT_s14 <= 1
 0 <= w_k1_MAT_s15 <= 1
 0 <= w_k1_MAT_s16 <= 1
 0 <= w_k1_MAT_s17 <= 1
 0 <= w_k1_MAT_s000 <= 1
 0 <= w_k1_MAT_s001 <= 1
 0 <= w_k1_MAT_s002 <= 1
 0 <= w_k1_MAT_s003 <= 1
 0 <= w_k1_MAT_s004 <= 1
 0 <= w_k1_MAT_s005 <= 1
 0 <= w_k1_MAT_s006 <= 1
 0 <= w_k1_MAT_s007 <= 1
 0 <= w_k1_MAT_s010 <= 1
 0 <= w_k1_MAT_s011 <= 1
 0 <= w_k1_MAT_s012 <= 1
 0 <= w_k1_MAT_s013 <= 1
 0 <= w_k1_MAT_s014 <= 1
 0 <= w_k1_MAT_s015 <= 1
 0 <= w_k1_MAT_s016 <= 1
 0 <= w_k1_MAT_s017 <= 1
 0 <= w_k1_MAT_s020 <= 1
 0 <= w_k1_MAT_s021 <= 1
 0 <= w_k1_MAT_s022 <= 1
 0 <= w_k1_MAT_s023 <= 1
 0 <= w_k1_MAT_s024 <= 1
 0 <= w_k1_MAT_s025 <= 1
 0 <= w_k1_MAT_s026 <= 1
 0 <= w_k1_MAT_s027 <= 1
 0 <= w_k1_MAT_s030 <= 1
 0 <= w_k1_MAT_s031 <= 1
 0 <= w_k1_MAT_s032 <= 1
 0 <= w_k1_MAT_s033 <= 1
 0 <= w_k1_MAT_s034 <= 1
 0 <= w_k1_MAT_s035 <= 1
 0 <= w_k1_MAT_s036 <= 1
 0 <= w_k1_MAT_s037 <= 1
 0 <= w_k1_MAT_s040 <= 1
 0 <= w_k1_MAT_s041 <= 1
 0 <= w_k1_MAT_s042 <= 1
 0 <= w_k1_MAT_s043 <= 1
 0 <= w_k1_MAT_s044 <= 1
 0 <= w_k1_MAT_s045 <= 1
 0 <= w_k1_MAT_s046 <= 1
 0 <= w_k1_MAT_s047 <= 1
 0 <= w_k1_MAT_s050 <= 1
 0 <= w_k1_MAT_s051 <= 1
 0 <= w_k1_MAT_s052 <= 1
 0 <= w_k1_MAT_s053 <= 1
 0 <= w_k1_MAT_s054 <= 1
 0 <= w_k1_MAT_s055 <= 1
 0 <= w_k1_MAT_s056 <= 1
 0 <= w_k1_MAT_s057 <= 1
 0 <= w_k1_MAT_s060 <= 1
 0 <= w_k1_MAT_s061 <= 1
 0 <= w_k1_MAT_s062 <= 1
 0 <= w_k1_MAT_s063 <= 1
 0 <= w_k1_MAT_s064 <= 1
 0 <= w_k1_MAT_s065 <= 1
 0 <= w_k1_MAT_s066 <= 1
 0 <= w_k1_MAT_s067 <= 1
 0 <= w_k1_MAT_s070 <= 1
 0 <= w_k1_MAT_s071 <= 1
 0 <= w_k1_MAT_s072 <= 1
 0 <= w_k1_MAT_s073 <= 1
 0 <= w_k1_MAT_s074 <= 1
 0 <= w_k1_MAT_s075 <= 1
 0 <= w_k1_MAT_s076 <= 1
 0 <= w_k1_MAT_s077 <= 1
 0 <= w_k1_MAT_s100 <= 1
 0 <= w_k1_MAT_s101 <= 1
 0 <= w_k1_MAT_s102 <= 1
 0 <= w_k1_MAT_s103 <= 1
 0 <= w_k1_MAT_s104 <= 1
 0 <= w_k1_MAT_s105 <= 1
 0 <= w_k1_MAT_s106 <= 1
 0 <= w_k1_MAT_s107 <= 1
 0 <= w_k1_MAT_s110 <= 1
 0 <= w_k1_MAT_s111 <= 1
 0 <= w_k1_MAT_s112 <= 1
 0 <= w_k1_MAT_s113 <= 1
 0 <= w_k1_MAT_s114 <= 1
 0 <= w_k1_MAT_s115 <= 1
 0 <= w_k1_MAT_s116 <= 1
 0 <= w_k1_MAT_s117 <= 1
 0 <= w_k1_MAT_s120 <= 1
 0 <= w_k1_MAT_s121 <= 1
 0 <= w_k1_MAT_s122 <= 1
 0 <= w_k1_MAT_s123 <= 1
 0 <= w_k1_MAT_s124 <= 1
 0 <= w_k1_MAT_s125 <= 1
 0 <= w_k1_MAT_s126 <= 1
 0 <= w_k1_MAT_s127 <= 1
 0 <= w_k1_MAT_s130 <= 1
 0 <= w_k1_MAT_s131 <= 1
 0 <= w_k1_MAT_s132 <= 1
 0 <= w_k1_MAT_s133 <= 1
 0 <= w_k1_MAT_s134 <= 1
 0 <= w_k1_MAT_s135 <= 1
 0 <= w_k1_MAT_s136 <= 1
 0 <= w_k1_MAT_s137 <= 1
 0 <= w_k1_MAT_s140 <= 1
 0 <= w_k1_MAT_s141 <= 1
 0 <= w_k1_MAT_s142 <= 1
 0 <= w_k1_MAT_s143 <= 1
 0 <= w_k1_MAT_s144 <= 1
 0 <= w_k1_MAT_s145 <= 1
 0 <= w_k1_MAT_s146 <= 1
 0 <= w_k1_MAT_s147 <= 1
 0 <= w_k1_MAT_s150 <= 1
 0 <= w_k1_MAT_s151 <= 1
 0 <= w_k1_MAT_s152 <= 1
 0 <= w_k1_MAT_s153 <= 1
 0 <= w_k1_MAT_s154 <= 1
 0 <= w_k1_MAT_s155 <= 1
 0 <= w_k1_MAT_s156 <= 1
 0 <= w_k1_MAT_s157 <= 1
 0 <= w_k1_MAT_s160 <= 1
 0 <= w_k1_MAT_s161 <= 1
 0 <= w_k1_MAT_s162 <= 1
 0 <= w_k1_MAT_s163 <= 1
 0 <= w_k1_MAT_s164 <= 1
 0 <= w_k1_MAT_s165 <= 1
 0 <= w_k1_MAT_s166 <= 1
 0 <= w_k1_MAT_s167 <= 1
 0 <= w_k1_MAT_s170 <= 1
 0 <= w_k1_MAT_s171 <= 1
 0 <= w_k1_MAT_s172 <= 1
 0 <= w_k1_MAT_s173 <= 1
 0 <= w_k1_MAT_s174 <= 1
 0 <= w_k1_MAT_s175 <= 1
 0 <= w_k1_MAT_s176 <= 1
 0 <= w_k1_MAT_s177 <= 1
 0 <= w_k1_MSFT_s0 <= 1
 0 <= w_k1_MSFT_s1 <= 1
 0 <= w_k1_MSFT_s00 <= 1
 0 <= w_k1_MSFT_s01 <= 1
 0 <= w_k1_MSFT_s02 <= 1
 0 <= w_k1_MSFT_s03 <= 1
 0 <= w_k1_MSFT_s04 <= 1
 0 <= w_k1_MSFT_s05 <= 1
 0 <= w_k1_MSFT_s06 <= 1
 0 <= w_k1_MSFT_s07 <= 1
 0 <= w_k1_MSFT_s10 <= 1
 0 <= w_k1_MSFT_s11 <= 1
 0 <= w_k1_MSFT_s12 <= 1
 0 <= w_k1_MSFT_s13 <= 1
 0 <= w_k1_MSFT_s14 <= 1
 0 <= w_k1_MSFT_s15 <= 1
 0 <= w_k1_MSFT_s16 <= 1
 0 <= w_k1_MSFT_s17 <= 1
 0 <= w_k1_MSFT_s000 <= 1
 0 <= w_k1_MSFT_s001 <= 1
 0 <= w_k1_MSFT_s002 <= 1
 0 <= w_k1_MSFT_s003 <= 1
 0 <= w_k1_MSFT_s004 <= 1
 0 <= w_k1_MSFT_s005 <= 1
 0 <= w_k1_MSFT_s006 <= 1
 0 <= w_k1_MSFT_s007 <= 1
 0 <= w_k1_MSFT_s010 <= 1
 0 <= w_k1_MSFT_s011 <= 1
 0 <= w_k1_MSFT_s012 <= 1
 0 <= w_k1_MSFT_s013 <= 1
 0 <= w_k1_MSFT_s014 <= 1
 0 <= w_k1_MSFT_s015 <= 1
 0 <= w_k1_MSFT_s016 <= 1
 0 <= w_k1_MSFT_s017 <= 1
 0 <= w_k1_MSFT_s020 <= 1
 0 <= w_k1_MSFT_s021 <= 1
 0 <= w_k1_MSFT_s022 <= 1
 0 <= w_k1_MSFT_s023 <= 1
 0 <= w_k1_MSFT_s024 <= 1
 0 <= w_k1_MSFT_s025 <= 1
 0 <= w_k1_MSFT_s026 <= 1
 0 <= w_k1_MSFT_s027 <= 1
 0 <= w_k1_MSFT_s030 <= 1
 0 <= w_k1_MSFT_s031 <= 1
 0 <= w_k1_MSFT_s032 <= 1
 0 <= w_k1_MSFT_s033 <= 1
 0 <= w_k1_MSFT_s034 <= 1
 0 <= w_k1_MSFT_s035 <= 1
 0 <= w_k1_MSFT_s036 <= 1
 0 <= w_k1_MSFT_s037 <= 1
 0 <= w_k1_MSFT_s040 <= 1
 0 <= w_k1_MSFT_s041 <= 1
 0 <= w_k1_MSFT_s042 <= 1
 0 <= w_k1_MSFT_s043 <= 1
 0 <= w_k1_MSFT_s044 <= 1
 0 <= w_k1_MSFT_s045 <= 1
 0 <= w_k1_MSFT_s046 <= 1
 0 <= w_k1_MSFT_s047 <= 1
 0 <= w_k1_MSFT_s050 <= 1
 0 <= w_k1_MSFT_s051 <= 1
 0 <= w_k1_MSFT_s052 <= 1
 0 <= w_k1_MSFT_s053 <= 1
 0 <= w_k1_MSFT_s054 <= 1
 0 <= w_k1_MSFT_s055 <= 1
 0 <= w_k1_MSFT_s056 <= 1
 0 <= w_k1_MSFT_s057 <= 1
 0 <= w_k1_MSFT_s060 <= 1
 0 <= w_k1_MSFT_s061 <= 1
 0 <= w_k1_MSFT_s062 <= 1
 0 <= w_k1_MSFT_s063 <= 1
 0 <= w_k1_MSFT_s064 <= 1
 0 <= w_k1_MSFT_s065 <= 1
 0 <= w_k1_MSFT_s066 <= 1
 0 <= w_k1_MSFT_s067 <= 1
 0 <= w_k1_MSFT_s070 <= 1
 0 <= w_k1_MSFT_s071 <= 1
 0 <= w_k1_MSFT_s072 <= 1
 0 <= w_k1_MSFT_s073 <= 1
 0 <= w_k1_MSFT_s074 <= 1
 0 <= w_k1_MSFT_s075 <= 1
 0 <= w_k1_MSFT_s076 <= 1
 0 <= w_k1_MSFT_s077 <= 1
 0 <= w_k1_MSFT_s100 <= 1
 0 <= w_k1_MSFT_s101 <= 1
 0 <= w_k1_MSFT_s102 <= 1
 0 <= w_k1_MSFT_s103 <= 1
 0 <= w_k1_MSFT_s104 <= 1
 0 <= w_k1_MSFT_s105 <= 1
 0 <= w_k1_MSFT_s106 <= 1
 0 <= w_k1_MSFT_s107 <= 1
 0 <= w_k1_MSFT_s110 <= 1
 0 <= w_k1_MSFT_s111 <= 1
 0 <= w_k1_MSFT_s112 <= 1
 0 <= w_k1_MSFT_s113 <= 1
 0 <= w_k1_MSFT_s114 <= 1
 0 <= w_k1_MSFT_s115 <= 1
 0 <= w_k1_MSFT_s116 <= 1
 0 <= w_k1_MSFT_s117 <= 1
 0 <= w_k1_MSFT_s120 <= 1
 0 <= w_k1_MSFT_s121 <= 1
 0 <= w_k1_MSFT_s122 <= 1
 0 <= w_k1_MSFT_s123 <= 1
 0 <= w_k1_MSFT_s124 <= 1
 0 <= w_k1_MSFT_s125 <= 1
 0 <= w_k1_MSFT_s126 <= 1
 0 <= w_k1_MSFT_s127 <= 1
 0 <= w_k1_MSFT_s130 <= 1
 0 <= w_k1_MSFT_s131 <= 1
 0 <= w_k1_MSFT_s132 <= 1
 0 <= w_k1_MSFT_s133 <= 1
 0 <= w_k1_MSFT_s134 <= 1
 0 <= w_k1_MSFT_s135 <= 1
 0 <= w_k1_MSFT_s136 <= 1
 0 <= w_k1_MSFT_s137 <= 1
 0 <= w_k1_MSFT_s140 <= 1
 0 <= w_k1_MSFT_s141 <= 1
 0 <= w_k1_MSFT_s142 <= 1
 0 <= w_k1_MSFT_s143 <= 1
 0 <= w_k1_MSFT_s144 <= 1
 0 <= w_k1_MSFT_s145 <= 1
 0 <= w_k1_MSFT_s146 <= 1
 0 <= w_k1_MSFT_s147 <= 1
 0 <= w_k1_MSFT_s150 <= 1
 0 <= w_k1_MSFT_s151 <= 1
 0 <= w_k1_MSFT_s152 <= 1
 0 <= w_k1_MSFT_s153 <= 1
 0 <= w_k1_MSFT_s154 <= 1
 0 <= w_k1_MSFT_s155 <= 1
 0 <= w_k1_MSFT_s156 <= 1
 0 <= w_k1_MSFT_s157 <= 1
 0 <= w_k1_MSFT_s160 <= 1
 0 <= w_k1_MSFT_s161 <= 1
 0 <= w_k1_MSFT_s162 <= 1
 0 <= w_k1_MSFT_s163 <= 1
 0 <= w_k1_MSFT_s164 <= 1
 0 <= w_k1_MSFT_s165 <= 1
 0 <= w_k1_MSFT_s166 <= 1
 0 <= w_k1_MSFT_s167 <= 1
 0 <= w_k1_MSFT_s170 <= 1
 0 <= w_k1_MSFT_s171 <= 1
 0 <= w_k1_MSFT_s172 <= 1
 0 <= w_k1_MSFT_s173 <= 1
 0 <= w_k1_MSFT_s174 <= 1
 0 <= w_k1_MSFT_s175 <= 1
 0 <= w_k1_MSFT_s176 <= 1
 0 <= w_k1_MSFT_s177 <= 1
 0 <= w_k1_PG_s0 <= 1
 0 <= w_k1_PG_s1 <= 1
 0 <= w_k1_PG_s00 <= 1
 0 <= w_k1_PG_s01 <= 1
 0 <= w_k1_PG_s02 <= 1
 0 <= w_k1_PG_s03 <= 1
 0 <= w_k1_PG_s04 <= 1
 0 <= w_k1_PG_s05 <= 1
 0 <= w_k1_PG_s06 <= 1
 0 <= w_k1_PG_s07 <= 1
 0 <= w_k1_PG_s10 <= 1
 0 <= w_k1_PG_s11 <= 1
 0 <= w_k1_PG_s12 <= 1
 0 <= w_k1_PG_s13 <= 1
 0 <= w_k1_PG_s14 <= 1
 0 <= w_k1_PG_s15 <= 1
 0 <= w_k1_PG_s16 <= 1
 0 <= w_k1_PG_s17 <= 1
 0 <= w_k1_PG_s000 <= 1
 0 <= w_k1_PG_s001 <= 1
 0 <= w_k1_PG_s002 <= 1
 0 <= w_k1_PG_s003 <= 1
 0 <= w_k1_PG_s004 <= 1
 0 <= w_k1_PG_s005 <= 1
 0 <= w_k1_PG_s006 <= 1
 0 <= w_k1_PG_s007 <= 1
 0 <= w_k1_PG_s010 <= 1
 0 <= w_k1_PG_s011 <= 1
 0 <= w_k1_PG_s012 <= 1
 0 <= w_k1_PG_s013 <= 1
 0 <= w_k1_PG_s014 <= 1
 0 <= w_k1_PG_s015 <= 1
 0 <= w_k1_PG_s016 <= 1
 0 <= w_k1_PG_s017 <= 1
 0 <= w_k1_PG_s020 <= 1
 0 <= w_k1_PG_s021 <= 1
 0 <= w_k1_PG_s022 <= 1
 0 <= w_k1_PG_s023 <= 1
 0 <= w_k1_PG_s024 <= 1
 0 <= w_k1_PG_s025 <= 1
 0 <= w_k1_PG_s026 <= 1
 0 <= w_k1_PG_s027 <= 1
 0 <= w_k1_PG_s030 <= 1
 0 <= w_k1_PG_s031 <= 1
 0 <= w_k1_PG_s032 <= 1
 0 <= w_k1_PG_s033 <= 1
 0 <= w_k1_PG_s034 <= 1
 0 <= w_k1_PG_s035 <= 1
 0 <= w_k1_PG_s036 <= 1
 0 <= w_k1_PG_s037 <= 1
 0 <= w_k1_PG_s040 <= 1
 0 <= w_k1_PG_s041 <= 1
 0 <= w_k1_PG_s042 <= 1
 0 <= w_k1_PG_s043 <= 1
 0 <= w_k1_PG_s044 <= 1
 0 <= w_k1_PG_s045 <= 1
 0 <= w_k1_PG_s046 <= 1
 0 <= w_k1_PG_s047 <= 1
 0 <= w_k1_PG_s050 <= 1
 0 <= w_k1_PG_s051 <= 1
 0 <= w_k1_PG_s052 <= 1
 0 <= w_k1_PG_s053 <= 1
 0 <= w_k1_PG_s054 <= 1
 0 <= w_k1_PG_s055 <= 1
 0 <= w_k1_PG_s056 <= 1
 0 <= w_k1_PG_s057 <= 1
 0 <= w_k1_PG_s060 <= 1
 0 <= w_k1_PG_s061 <= 1
 0 <= w_k1_PG_s062 <= 1
 0 <= w_k1_PG_s063 <= 1
 0 <= w_k1_PG_s064 <= 1
 0 <= w_k1_PG_s065 <= 1
 0 <= w_k1_PG_s066 <= 1
 0 <= w_k1_PG_s067 <= 1
 0 <= w_k1_PG_s070 <= 1
 0 <= w_k1_PG_s071 <= 1
 0 <= w_k1_PG_s072 <= 1
 0 <= w_k1_PG_s073 <= 1
 0 <= w_k1_PG_s074 <= 1
 0 <= w_k1_PG_s075 <= 1
 0 <= w_k1_PG_s076 <= 1
 0 <= w_k1_PG_s077 <= 1
 0 <= w_k1_PG_s100 <= 1
 0 <= w_k1_PG_s101 <= 1
 0 <= w_k1_PG_s102 <= 1
 0 <= w_k1_PG_s103 <= 1
 0 <= w_k1_PG_s104 <= 1
 0 <= w_k1_PG_s105 <= 1
 0 <= w_k1_PG_s106 <= 1
 0 <= w_k1_PG_s107 <= 1
 0 <= w_k1_PG_s110 <= 1
 0 <= w_k1_PG_s111 <= 1
 0 <= w_k1_PG_s112 <= 1
 0 <= w_k1_PG_s113 <= 1
 0 <= w_k1_PG_s114 <= 1
 0 <= w_k1_PG_s115 <= 1
 0 <= w_k1_PG_s116 <= 1
 0 <= w_k1_PG_s117 <= 1
 0 <= w_k1_PG_s120 <= 1
 0 <= w_k1_PG_s121 <= 1
 0 <= w_k1_PG_s122 <= 1
 0 <= w_k1_PG_s123 <= 1
 0 <= w_k1_PG_s124 <= 1
 0 <= w_k1_PG_s125 <= 1
 0 <= w_k1_PG_s126 <= 1
 0 <= w_k1_PG_s127 <= 1
 0 <= w_k1_PG_s130 <= 1
 0 <= w_k1_PG_s131 <= 1
 0 <= w_k1_PG_s132 <= 1
 0 <= w_k1_PG_s133 <= 1
 0 <= w_k1_PG_s134 <= 1
 0 <= w_k1_PG_s135 <= 1
 0 <= w_k1_PG_s136 <= 1
 0 <= w_k1_PG_s137 <= 1
 0 <= w_k1_PG_s140 <= 1
 0 <= w_k1_PG_s141 <= 1
 0 <= w_k1_PG_s142 <= 1
 0 <= w_k1_PG_s143 <= 1
 0 <= w_k1_PG_s144 <= 1
 0 <= w_k1_PG_s145 <= 1
 0 <= w_k1_PG_s146 <= 1
 0 <= w_k1_PG_s147 <= 1
 0 <= w_k1_PG_s150 <= 1
 0 <= w_k1_PG_s151 <= 1
 0 <= w_k1_PG_s152 <= 1
 0 <= w_k1_PG_s153 <= 1
 0 <= w_k1_PG_s154 <= 1
 0 <= w_k1_PG_s155 <= 1
 0 <= w_k1_PG_s156 <= 1
 0 <= w_k1_PG_s157 <= 1
 0 <= w_k1_PG_s160 <= 1
 0 <= w_k1_PG_s161 <= 1
 0 <= w_k1_PG_s162 <= 1
 0 <= w_k1_PG_s163 <= 1
 0 <= w_k1_PG_s164 <= 1
 0 <= w_k1_PG_s165 <= 1
 0 <= w_k1_PG_s166 <= 1
 0 <= w_k1_PG_s167 <= 1
 0 <= w_k1_PG_s170 <= 1
 0 <= w_k1_PG_s171 <= 1
 0 <= w_k1_PG_s172 <= 1
 0 <= w_k1_PG_s173 <= 1
 0 <= w_k1_PG_s174 <= 1
 0 <= w_k1_PG_s175 <= 1
 0 <= w_k1_PG_s176 <= 1
 0 <= w_k1_PG_s177 <= 1
 0 <= b_0_ADM <= 0.2
 0 <= b_0_AXP <= 0.2
 0 <= b_0_DIS <= 0.2
 0 <= b_0_FDX <= 0.2
 0 <= b_0_IBM <= 0.2
 0 <= b_0_INTC <= 0.2
 0 <= b_0_KO <= 0.2
 0 <= b_0_MAT <= 0.2
 0 <= b_0_MSFT <= 0.2
 0 <= b_0_PG <= 0.2
 0 <= b_k1_ADM_s0 <= 0.2
 0 <= b_k1_ADM_s1 <= 0.2
 0 <= b_k1_ADM_s00 <= 0.2
 0 <= b_k1_ADM_s01 <= 0.2
 0 <= b_k1_ADM_s02 <= 0.2
 0 <= b_k1_ADM_s03 <= 0.2
 0 <= b_k1_ADM_s04 <= 0.2
 0 <= b_k1_ADM_s05 <= 0.2
 0 <= b_k1_ADM_s06 <= 0.2
 0 <= b_k1_ADM_s07 <= 0.2
 0 <= b_k1_ADM_s10 <= 0.2
 0 <= b_k1_ADM_s11 <= 0.2
 0 <= b_k1_ADM_s12 <= 0.2
 0 <= b_k1_ADM_s13 <= 0.2
 0 <= b_k1_ADM_s14 <= 0.2
 0 <= b_k1_ADM_s15 <= 0.2
 0 <= b_k1_ADM_s16 <= 0.2
 0 <= b_k1_ADM_s17 <= 0.2
 0 <= b_k1_ADM_s000 <= 0.2
 0 <= b_k1_ADM_s001 <= 0.2
 0 <= b_k1_ADM_s002 <= 0.2
 0 <= b_k1_ADM_s003 <= 0.2
 0 <= b_k1_ADM_s004 <= 0.2
 0 <= b_k1_ADM_s005 <= 0.2
 0 <= b_k1_ADM_s006 <= 0.2
 0 <= b_k1_ADM_s007 <= 0.2
 0 <= b_k1_ADM_s010 <= 0.2
 0 <= b_k1_ADM_s011 <= 0.2
 0 <= b_k1_ADM_s012 <= 0.2
 0 <= b_k1_ADM_s013 <= 0.2
 0 <= b_k1_ADM_s014 <= 0.2
 0 <= b_k1_ADM_s015 <= 0.2
 0 <= b_k1_ADM_s016 <= 0.2
 0 <= b_k1_ADM_s017 <= 0.2
 0 <= b_k1_ADM_s020 <= 0.2
 0 <= b_k1_ADM_s021 <= 0.2
 0 <= b_k1_ADM_s022 <= 0.2
 0 <= b_k1_ADM_s023 <= 0.2
 0 <= b_k1_ADM_s024 <= 0.2
 0 <= b_k1_ADM_s025 <= 0.2
 0 <= b_k1_ADM_s026 <= 0.2
 0 <= b_k1_ADM_s027 <= 0.2
 0 <= b_k1_ADM_s030 <= 0.2
 0 <= b_k1_ADM_s031 <= 0.2
 0 <= b_k1_ADM_s032 <= 0.2
 0 <= b_k1_ADM_s033 <= 0.2
 0 <= b_k1_ADM_s034 <= 0.2
 0 <= b_k1_ADM_s035 <= 0.2
 0 <= b_k1_ADM_s036 <= 0.2
 0 <= b_k1_ADM_s037 <= 0.2
 0 <= b_k1_ADM_s040 <= 0.2
 0 <= b_k1_ADM_s041 <= 0.2
 0 <= b_k1_ADM_s042 <= 0.2
 0 <= b_k1_ADM_s043 <= 0.2
 0 <= b_k1_ADM_s044 <= 0.2
 0 <= b_k1_ADM_s045 <= 0.2
 0 <= b_k1_ADM_s046 <= 0.2
 0 <= b_k1_ADM_s047 <= 0.2
 0 <= b_k1_ADM_s050 <= 0.2
 0 <= b_k1_ADM_s051 <= 0.2
 0 <= b_k1_ADM_s052 <= 0.2
 0 <= b_k1_ADM_s053 <= 0.2
 0 <= b_k1_ADM_s054 <= 0.2
 0 <= b_k1_ADM_s055 <= 0.2
 0 <= b_k1_ADM_s056 <= 0.2
 0 <= b_k1_ADM_s057 <= 0.2
 0 <= b_k1_ADM_s060 <= 0.2
 0 <= b_k1_ADM_s061 <= 0.2
 0 <= b_k1_ADM_s062 <= 0.2
 0 <= b_k1_ADM_s063 <= 0.2
 0 <= b_k1_ADM_s064 <= 0.2
 0 <= b_k1_ADM_s065 <= 0.2
 0 <= b_k1_ADM_s066 <= 0.2
 0 <= b_k1_ADM_s067 <= 0.2
 0 <= b_k1_ADM_s070 <= 0.2
 0 <= b_k1_ADM_s071 <= 0.2
 0 <= b_k1_ADM_s072 <= 0.2
 0 <= b_k1_ADM_s073 <= 0.2
 0 <= b_k1_ADM_s074 <= 0.2
 0 <= b_k1_ADM_s075 <= 0.2
 0 <= b_k1_ADM_s076 <= 0.2
 0 <= b_k1_ADM_s077 <= 0.2
 0 <= b_k1_ADM_s100 <= 0.2
 0 <= b_k1_ADM_s101 <= 0.2
 0 <= b_k1_ADM_s102 <= 0.2
 0 <= b_k1_ADM_s103 <= 0.2
 0 <= b_k1_ADM_s104 <= 0.2
 0 <= b_k1_ADM_s105 <= 0.2
 0 <= b_k1_ADM_s106 <= 0.2
 0 <= b_k1_ADM_s107 <= 0.2
 0 <= b_k1_ADM_s110 <= 0.2
 0 <= b_k1_ADM_s111 <= 0.2
 0 <= b_k1_ADM_s112 <= 0.2
 0 <= b_k1_ADM_s113 <= 0.2
 0 <= b_k1_ADM_s114 <= 0.2
 0 <= b_k1_ADM_s115 <= 0.2
 0 <= b_k1_ADM_s116 <= 0.2
 0 <= b_k1_ADM_s117 <= 0.2
 0 <= b_k1_ADM_s120 <= 0.2
 0 <= b_k1_ADM_s121 <= 0.2
 0 <= b_k1_ADM_s122 <= 0.2
 0 <= b_k1_ADM_s123 <= 0.2
 0 <= b_k1_ADM_s124 <= 0.2
 0 <= b_k1_ADM_s125 <= 0.2
 0 <= b_k1_ADM_s126 <= 0.2
 0 <= b_k1_ADM_s127 <= 0.2
 0 <= b_k1_ADM_s130 <= 0.2
 0 <= b_k1_ADM_s131 <= 0.2
 0 <= b_k1_ADM_s132 <= 0.2
 0 <= b_k1_ADM_s133 <= 0.2
 0 <= b_k1_ADM_s134 <= 0.2
 0 <= b_k1_ADM_s135 <= 0.2
 0 <= b_k1_ADM_s136 <= 0.2
 0 <= b_k1_ADM_s137 <= 0.2
 0 <= b_k1_ADM_s140 <= 0.2
 0 <= b_k1_ADM_s141 <= 0.2
 0 <= b_k1_ADM_s142 <= 0.2
 0 <= b_k1_ADM_s143 <= 0.2
 0 <= b_k1_ADM_s144 <= 0.2
 0 <= b_k1_ADM_s145 <= 0.2
 0 <= b_k1_ADM_s146 <= 0.2
 0 <= b_k1_ADM_s147 <= 0.2
 0 <= b_k1_ADM_s150 <= 0.2
 0 <= b_k1_ADM_s151 <= 0.2
 0 <= b_k1_ADM_s152 <= 0.2
 0 <= b_k1_ADM_s153 <= 0.2
 0 <= b_k1_ADM_s154 <= 0.2
 0 <= b_k1_ADM_s155 <= 0.2
 0 <= b_k1_ADM_s156 <= 0.2
 0 <= b_k1_ADM_s157 <= 0.2
 0 <= b_k1_ADM_s160 <= 0.2
 0 <= b_k1_ADM_s161 <= 0.2
 0 <= b_k1_ADM_s162 <= 0.2
 0 <= b_k1_ADM_s163 <= 0.2
 0 <= b_k1_ADM_s164 <= 0.2
 0 <= b_k1_ADM_s165 <= 0.2
 0 <= b_k1_ADM_s166 <= 0.2
 0 <= b_k1_ADM_s167 <= 0.2
 0 <= b_k1_ADM_s170 <= 0.2
 0 <= b_k1_ADM_s171 <= 0.2
 0 <= b_k1_ADM_s172 <= 0.2
 0 <= b_k1_ADM_s173 <= 0.2
 0 <= b_k1_ADM_s174 <= 0.2
 0 <= b_k1_ADM_s175 <= 0.2
 0 <= b_k1_ADM_s176 <= 0.2
 0 <= b_k1_ADM_s177 <= 0.2
 0 <= b_k1_AXP_s0 <= 0.2
 0 <= b_k1_AXP_s1 <= 0.2
 0 <= b_k1_AXP_s00 <= 0.2
 0 <= b_k1_AXP_s01 <= 0.2
 0 <= b_k1_AXP_s02 <= 0.2
 0 <= b_k1_AXP_s03 <= 0.2
 0 <= b_k1_AXP_s04 <= 0.2
 0 <= b_k1_AXP_s05 <= 0.2
 0 <= b_k1_AXP_s06 <= 0.2
 0 <= b_k1_AXP_s07 <= 0.2
 0 <= b_k1_AXP_s10 <= 0.2
 0 <= b_k1_AXP_s11 <= 0.2
 0 <= b_k1_AXP_s12 <= 0.2
 0 <= b_k1_AXP_s13 <= 0.2
 0 <= b_k1_AXP_s14 <= 0.2
 0 <= b_k1_AXP_s15 <= 0.2
 0 <= b_k1_AXP_s16 <= 0.2
 0 <= b_k1_AXP_s17 <= 0.2
 0 <= b_k1_AXP_s000 <= 0.2
 0 <= b_k1_AXP_s001 <= 0.2
 0 <= b_k1_AXP_s002 <= 0.2
 0 <= b_k1_AXP_s003 <= 0.2
 0 <= b_k1_AXP_s004 <= 0.2
 0 <= b_k1_AXP_s005 <= 0.2
 0 <= b_k1_AXP_s006 <= 0.2
 0 <= b_k1_AXP_s007 <= 0.2
 0 <= b_k1_AXP_s010 <= 0.2
 0 <= b_k1_AXP_s011 <= 0.2
 0 <= b_k1_AXP_s012 <= 0.2
 0 <= b_k1_AXP_s013 <= 0.2
 0 <= b_k1_AXP_s014 <= 0.2
 0 <= b_k1_AXP_s015 <= 0.2
 0 <= b_k1_AXP_s016 <= 0.2
 0 <= b_k1_AXP_s017 <= 0.2
 0 <= b_k1_AXP_s020 <= 0.2
 0 <= b_k1_AXP_s021 <= 0.2
 0 <= b_k1_AXP_s022 <= 0.2
 0 <= b_k1_AXP_s023 <= 0.2
 0 <= b_k1_AXP_s024 <= 0.2
 0 <= b_k1_AXP_s025 <= 0.2
 0 <= b_k1_AXP_s026 <= 0.2
 0 <= b_k1_AXP_s027 <= 0.2
 0 <= b_k1_AXP_s030 <= 0.2
 0 <= b_k1_AXP_s031 <= 0.2
 0 <= b_k1_AXP_s032 <= 0.2
 0 <= b_k1_AXP_s033 <= 0.2
 0 <= b_k1_AXP_s034 <= 0.2
 0 <= b_k1_AXP_s035 <= 0.2
 0 <= b_k1_AXP_s036 <= 0.2
 0 <= b_k1_AXP_s037 <= 0.2
 0 <= b_k1_AXP_s040 <= 0.2
 0 <= b_k1_AXP_s041 <= 0.2
 0 <= b_k1_AXP_s042 <= 0.2
 0 <= b_k1_AXP_s043 <= 0.2
 0 <= b_k1_AXP_s044 <= 0.2
 0 <= b_k1_AXP_s045 <= 0.2
 0 <= b_k1_AXP_s046 <= 0.2
 0 <= b_k1_AXP_s047 <= 0.2
 0 <= b_k1_AXP_s050 <= 0.2
 0 <= b_k1_AXP_s051 <= 0.2
 0 <= b_k1_AXP_s052 <= 0.2
 0 <= b_k1_AXP_s053 <= 0.2
 0 <= b_k1_AXP_s054 <= 0.2
 0 <= b_k1_AXP_s055 <= 0.2
 0 <= b_k1_AXP_s056 <= 0.2
 0 <= b_k1_AXP_s057 <= 0.2
 0 <= b_k1_AXP_s060 <= 0.2
 0 <= b_k1_AXP_s061 <= 0.2
 0 <= b_k1_AXP_s062 <= 0.2
 0 <= b_k1_AXP_s063 <= 0.2
 0 <= b_k1_AXP_s064 <= 0.2
 0 <= b_k1_AXP_s065 <= 0.2
 0 <= b_k1_AXP_s066 <= 0.2
 0 <= b_k1_AXP_s067 <= 0.2
 0 <= b_k1_AXP_s070 <= 0.2
 0 <= b_k1_AXP_s071 <= 0.2
 0 <= b_k1_AXP_s072 <= 0.2
 0 <= b_k1_AXP_s073 <= 0.2
 0 <= b_k1_AXP_s074 <= 0.2
 0 <= b_k1_AXP_s075 <= 0.2
 0 <= b_k1_AXP_s076 <= 0.2
 0 <= b_k1_AXP_s077 <= 0.2
 0 <= b_k1_AXP_s100 <= 0.2
 0 <= b_k1_AXP_s101 <= 0.2
 0 <= b_k1_AXP_s102 <= 0.2
 0 <= b_k1_AXP_s103 <= 0.2
 0 <= b_k1_AXP_s104 <= 0.2
 0 <= b_k1_AXP_s105 <= 0.2
 0 <= b_k1_AXP_s106 <= 0.2
 0 <= b_k1_AXP_s107 <= 0.2
 0 <= b_k1_AXP_s110 <= 0.2
 0 <= b_k1_AXP_s111 <= 0.2
 0 <= b_k1_AXP_s112 <= 0.2
 0 <= b_k1_AXP_s113 <= 0.2
 0 <= b_k1_AXP_s114 <= 0.2
 0 <= b_k1_AXP_s115 <= 0.2
 0 <= b_k1_AXP_s116 <= 0.2
 0 <= b_k1_AXP_s117 <= 0.2
 0 <= b_k1_AXP_s120 <= 0.2
 0 <= b_k1_AXP_s121 <= 0.2
 0 <= b_k1_AXP_s122 <= 0.2
 0 <= b_k1_AXP_s123 <= 0.2
 0 <= b_k1_AXP_s124 <= 0.2
 0 <= b_k1_AXP_s125 <= 0.2
 0 <= b_k1_AXP_s126 <= 0.2
 0 <= b_k1_AXP_s127 <= 0.2
 0 <= b_k1_AXP_s130 <= 0.2
 0 <= b_k1_AXP_s131 <= 0.2
 0 <= b_k1_AXP_s132 <= 0.2
 0 <= b_k1_AXP_s133 <= 0.2
 0 <= b_k1_AXP_s134 <= 0.2
 0 <= b_k1_AXP_s135 <= 0.2
 0 <= b_k1_AXP_s136 <= 0.2
 0 <= b_k1_AXP_s137 <= 0.2
 0 <= b_k1_AXP_s140 <= 0.2
 0 <= b_k1_AXP_s141 <= 0.2
 0 <= b_k1_AXP_s142 <= 0.2
 0 <= b_k1_AXP_s143 <= 0.2
 0 <= b_k1_AXP_s144 <= 0.2
 0 <= b_k1_AXP_s145 <= 0.2
 0 <= b_k1_AXP_s146 <= 0.2
 0 <= b_k1_AXP_s147 <= 0.2
 0 <= b_k1_AXP_s150 <= 0.2
 0 <= b_k1_AXP_s151 <= 0.2
 0 <= b_k1_AXP_s152 <= 0.2
 0 <= b_k1_AXP_s153 <= 0.2
 0 <= b_k1_AXP_s154 <= 0.2
 0 <= b_k1_AXP_s155 <= 0.2
 0 <= b_k1_AXP_s156 <= 0.2
 0 <= b_k1_AXP_s157 <= 0.2
 0 <= b_k1_AXP_s160 <= 0.2
 0 <= b_k1_AXP_s161 <= 0.2
 0 <= b_k1_AXP_s162 <= 0.2
 0 <= b_k1_AXP_s163 <= 0.2
 0 <= b_k1_AXP_s164 <= 0.2
 0 <= b_k1_AXP_s165 <= 0.2
 0 <= b_k1_AXP_s166 <= 0.2
 0 <= b_k1_AXP_s167 <= 0.2
 0 <= b_k1_AXP_s170 <= 0.2
 0 <= b_k1_AXP_s171 <= 0.2
 0 <= b_k1_AXP_s172 <= 0.2
 0 <= b_k1_AXP_s173 <= 0.2
 0 <= b_k1_AXP_s174 <= 0.2
 0 <= b_k1_AXP_s175 <= 0.2
 0 <= b_k1_AXP_s176 <= 0.2
 0 <= b_k1_AXP_s177 <= 0.2
 0 <= b_k1_DIS_s0 <= 0.2
 0 <= b_k1_DIS_s1 <= 0.2
 0 <= b_k1_DIS_s00 <= 0.2
 0 <= b_k1_DIS_s01 <= 0.2
 0 <= b_k1_DIS_s02 <= 0.2
 0 <= b_k1_DIS_s03 <= 0.2
 0 <= b_k1_DIS_s04 <= 0.2
 0 <= b_k1_DIS_s05 <= 0.2
 0 <= b_k1_DIS_s06 <= 0.2
 0 <= b_k1_DIS_s07 <= 0.2
 0 <= b_k1_DIS_s10 <= 0.2
 0 <= b_k1_DIS_s11 <= 0.2
 0 <= b_k1_DIS_s12 <= 0.2
 0 <= b_k1_DIS_s13 <= 0.2
 0 <= b_k1_DIS_s14 <= 0.2
 0 <= b_k1_DIS_s15 <= 0.2
 0 <= b_k1_DIS_s16 <= 0.2
 0 <= b_k1_DIS_s17 <= 0.2
 0 <= b_k1_DIS_s000 <= 0.2
 0 <= b_k1_DIS_s001 <= 0.2
 0 <= b_k1_DIS_s002 <= 0.2
 0 <= b_k1_DIS_s003 <= 0.2
 0 <= b_k1_DIS_s004 <= 0.2
 0 <= b_k1_DIS_s005 <= 0.2
 0 <= b_k1_DIS_s006 <= 0.2
 0 <= b_k1_DIS_s007 <= 0.2
 0 <= b_k1_DIS_s010 <= 0.2
 0 <= b_k1_DIS_s011 <= 0.2
 0 <= b_k1_DIS_s012 <= 0.2
 0 <= b_k1_DIS_s013 <= 0.2
 0 <= b_k1_DIS_s014 <= 0.2
 0 <= b_k1_DIS_s015 <= 0.2
 0 <= b_k1_DIS_s016 <= 0.2
 0 <= b_k1_DIS_s017 <= 0.2
 0 <= b_k1_DIS_s020 <= 0.2
 0 <= b_k1_DIS_s021 <= 0.2
 0 <= b_k1_DIS_s022 <= 0.2
 0 <= b_k1_DIS_s023 <= 0.2
 0 <= b_k1_DIS_s024 <= 0.2
 0 <= b_k1_DIS_s025 <= 0.2
 0 <= b_k1_DIS_s026 <= 0.2
 0 <= b_k1_DIS_s027 <= 0.2
 0 <= b_k1_DIS_s030 <= 0.2
 0 <= b_k1_DIS_s031 <= 0.2
 0 <= b_k1_DIS_s032 <= 0.2
 0 <= b_k1_DIS_s033 <= 0.2
 0 <= b_k1_DIS_s034 <= 0.2
 0 <= b_k1_DIS_s035 <= 0.2
 0 <= b_k1_DIS_s036 <= 0.2
 0 <= b_k1_DIS_s037 <= 0.2
 0 <= b_k1_DIS_s040 <= 0.2
 0 <= b_k1_DIS_s041 <= 0.2
 0 <= b_k1_DIS_s042 <= 0.2
 0 <= b_k1_DIS_s043 <= 0.2
 0 <= b_k1_DIS_s044 <= 0.2
 0 <= b_k1_DIS_s045 <= 0.2
 0 <= b_k1_DIS_s046 <= 0.2
 0 <= b_k1_DIS_s047 <= 0.2
 0 <= b_k1_DIS_s050 <= 0.2
 0 <= b_k1_DIS_s051 <= 0.2
 0 <= b_k1_DIS_s052 <= 0.2
 0 <= b_k1_DIS_s053 <= 0.2
 0 <= b_k1_DIS_s054 <= 0.2
 0 <= b_k1_DIS_s055 <= 0.2
 0 <= b_k1_DIS_s056 <= 0.2
 0 <= b_k1_DIS_s057 <= 0.2
 0 <= b_k1_DIS_s060 <= 0.2
 0 <= b_k1_DIS_s061 <= 0.2
 0 <= b_k1_DIS_s062 <= 0.2
 0 <= b_k1_DIS_s063 <= 0.2
 0 <= b_k1_DIS_s064 <= 0.2
 0 <= b_k1_DIS_s065 <= 0.2
 0 <= b_k1_DIS_s066 <= 0.2
 0 <= b_k1_DIS_s067 <= 0.2
 0 <= b_k1_DIS_s070 <= 0.2
 0 <= b_k1_DIS_s071 <= 0.2
 0 <= b_k1_DIS_s072 <= 0.2
 0 <= b_k1_DIS_s073 <= 0.2
 0 <= b_k1_DIS_s074 <= 0.2
 0 <= b_k1_DIS_s075 <= 0.2
 0 <= b_k1_DIS_s076 <= 0.2
 0 <= b_k1_DIS_s077 <= 0.2
 0 <= b_k1_DIS_s100 <= 0.2
 0 <= b_k1_DIS_s101 <= 0.2
 0 <= b_k1_DIS_s102 <= 0.2
 0 <= b_k1_DIS_s103 <= 0.2
 0 <= b_k1_DIS_s104 <= 0.2
 0 <= b_k1_DIS_s105 <= 0.2
 0 <= b_k1_DIS_s106 <= 0.2
 0 <= b_k1_DIS_s107 <= 0.2
 0 <= b_k1_DIS_s110 <= 0.2
 0 <= b_k1_DIS_s111 <= 0.2
 0 <= b_k1_DIS_s112 <= 0.2
 0 <= b_k1_DIS_s113 <= 0.2
 0 <= b_k1_DIS_s114 <= 0.2
 0 <= b_k1_DIS_s115 <= 0.2
 0 <= b_k1_DIS_s116 <= 0.2
 0 <= b_k1_DIS_s117 <= 0.2
 0 <= b_k1_DIS_s120 <= 0.2
 0 <= b_k1_DIS_s121 <= 0.2
 0 <= b_k1_DIS_s122 <= 0.2
 0 <= b_k1_DIS_s123 <= 0.2
 0 <= b_k1_DIS_s124 <= 0.2
 0 <= b_k1_DIS_s125 <= 0.2
 0 <= b_k1_DIS_s126 <= 0.2
 0 <= b_k1_DIS_s127 <= 0.2
 0 <= b_k1_DIS_s130 <= 0.2
 0 <= b_k1_DIS_s131 <= 0.2
 0 <= b_k1_DIS_s132 <= 0.2
 0 <= b_k1_DIS_s133 <= 0.2
 0 <= b_k1_DIS_s134 <= 0.2
 0 <= b_k1_DIS_s135 <= 0.2
 0 <= b_k1_DIS_s136 <= 0.2
 0 <= b_k1_DIS_s137 <= 0.2
 0 <= b_k1_DIS_s140 <= 0.2
 0 <= b_k1_DIS_s141 <= 0.2
 0 <= b_k1_DIS_s142 <= 0.2
 0 <= b_k1_DIS_s143 <= 0.2
 0 <= b_k1_DIS_s144 <= 0.2
 0 <= b_k1_DIS_s145 <= 0.2
 0 <= b_k1_DIS_s146 <= 0.2
 0 <= b_k1_DIS_s147 <= 0.2
 0 <= b_k1_DIS_s150 <= 0.2
 0 <= b_k1_DIS_s151 <= 0.2
 0 <= b_k1_DIS_s152 <= 0.2
 0 <= b_k1_DIS_s153 <= 0.2
 0 <= b_k1_DIS_s154 <= 0.2
 0 <= b_k1_DIS_s155 <= 0.2
 0 <= b_k1_DIS_s156 <= 0.2
 0 <= b_k1_DIS_s157 <= 0.2
 0 <= b_k1_DIS_s160 <= 0.2
 0 <= b_k1_DIS_s161 <= 0.2
 0 <= b_k1_DIS_s162 <= 0.2
 0 <= b_k1_DIS_s163 <= 0.2
 0 <= b_k1_DIS_s164 <= 0.2
 0 <= b_k1_DIS_s165 <= 0.2
 0 <= b_k1_DIS_s166 <= 0.2
 0 <= b_k1_DIS_s167 <= 0.2
 0 <= b_k1_DIS_s170 <= 0.2
 0 <= b_k1_DIS_s171 <= 0.2
 0 <= b_k1_DIS_s172 <= 0.2
 0 <= b_k1_DIS_s173 <= 0.2
 0 <= b_k1_DIS_s174 <= 0.2
 0 <= b_k1_DIS_s175 <= 0.2
 0 <= b_k1_DIS_s176 <= 0.2
 0 <= b_k1_DIS_s177 <= 0.2
 0 <= b_k1_FDX_s0 <= 0.2
 0 <= b_k1_FDX_s1 <= 0.2
 0 <= b_k1_FDX_s00 <= 0.2
 0 <= b_k1_FDX_s01 <= 0.2
 0 <= b_k1_FDX_s02 <= 0.2
 0 <= b_k1_FDX_s03 <= 0.2
 0 <= b_k1_FDX_s04 <= 0.2
 0 <= b_k1_FDX_s05 <= 0.2
 0 <= b_k1_FDX_s06 <= 0.2
 0 <= b_k1_FDX_s07 <= 0.2
 0 <= b_k1_FDX_s10 <= 0.2
 0 <= b_k1_FDX_s11 <= 0.2
 0 <= b_k1_FDX_s12 <= 0.2
 0 <= b_k1_FDX_s13 <= 0.2
 0 <= b_k1_FDX_s14 <= 0.2
 0 <= b_k1_FDX_s15 <= 0.2
 0 <= b_k1_FDX_s16 <= 0.2
 0 <= b_k1_FDX_s17 <= 0.2
 0 <= b_k1_FDX_s000 <= 0.2
 0 <= b_k1_FDX_s001 <= 0.2
 0 <= b_k1_FDX_s002 <= 0.2
 0 <= b_k1_FDX_s003 <= 0.2
 0 <= b_k1_FDX_s004 <= 0.2
 0 <= b_k1_FDX_s005 <= 0.2
 0 <= b_k1_FDX_s006 <= 0.2
 0 <= b_k1_FDX_s007 <= 0.2
 0 <= b_k1_FDX_s010 <= 0.2
 0 <= b_k1_FDX_s011 <= 0.2
 0 <= b_k1_FDX_s012 <= 0.2
 0 <= b_k1_FDX_s013 <= 0.2
 0 <= b_k1_FDX_s014 <= 0.2
 0 <= b_k1_FDX_s015 <= 0.2
 0 <= b_k1_FDX_s016 <= 0.2
 0 <= b_k1_FDX_s017 <= 0.2
 0 <= b_k1_FDX_s020 <= 0.2
 0 <= b_k1_FDX_s021 <= 0.2
 0 <= b_k1_FDX_s022 <= 0.2
 0 <= b_k1_FDX_s023 <= 0.2
 0 <= b_k1_FDX_s024 <= 0.2
 0 <= b_k1_FDX_s025 <= 0.2
 0 <= b_k1_FDX_s026 <= 0.2
 0 <= b_k1_FDX_s027 <= 0.2
 0 <= b_k1_FDX_s030 <= 0.2
 0 <= b_k1_FDX_s031 <= 0.2
 0 <= b_k1_FDX_s032 <= 0.2
 0 <= b_k1_FDX_s033 <= 0.2
 0 <= b_k1_FDX_s034 <= 0.2
 0 <= b_k1_FDX_s035 <= 0.2
 0 <= b_k1_FDX_s036 <= 0.2
 0 <= b_k1_FDX_s037 <= 0.2
 0 <= b_k1_FDX_s040 <= 0.2
 0 <= b_k1_FDX_s041 <= 0.2
 0 <= b_k1_FDX_s042 <= 0.2
 0 <= b_k1_FDX_s043 <= 0.2
 0 <= b_k1_FDX_s044 <= 0.2
 0 <= b_k1_FDX_s045 <= 0.2
 0 <= b_k1_FDX_s046 <= 0.2
 0 <= b_k1_FDX_s047 <= 0.2
 0 <= b_k1_FDX_s050 <= 0.2
 0 <= b_k1_FDX_s051 <= 0.2
 0 <= b_k1_FDX_s052 <= 0.2
 0 <= b_k1_FDX_s053 <= 0.2
 0 <= b_k1_FDX_s054 <= 0.2
 0 <= b_k1_FDX_s055 <= 0.2
 0 <= b_k1_FDX_s056 <= 0.2
 0 <= b_k1_FDX_s057 <= 0.2
 0 <= b_k1_FDX_s060 <= 0.2
 0 <= b_k1_FDX_s061 <= 0.2
 0 <= b_k1_FDX_s062 <= 0.2
 0 <= b_k1_FDX_s063 <= 0.2
 0 <= b_k1_FDX_s064 <= 0.2
 0 <= b_k1_FDX_s065 <= 0.2
 0 <= b_k1_FDX_s066 <= 0.2
 0 <= b_k1_FDX_s067 <= 0.2
 0 <= b_k1_FDX_s070 <= 0.2
 0 <= b_k1_FDX_s071 <= 0.2
 0 <= b_k1_FDX_s072 <= 0.2
 0 <= b_k1_FDX_s073 <= 0.2
 0 <= b_k1_FDX_s074 <= 0.2
 0 <= b_k1_FDX_s075 <= 0.2
 0 <= b_k1_FDX_s076 <= 0.2
 0 <= b_k1_FDX_s077 <= 0.2
 0 <= b_k1_FDX_s100 <= 0.2
 0 <= b_k1_FDX_s101 <= 0.2
 0 <= b_k1_FDX_s102 <= 0.2
 0 <= b_k1_FDX_s103 <= 0.2
 0 <= b_k1_FDX_s104 <= 0.2
 0 <= b_k1_FDX_s105 <= 0.2
 0 <= b_k1_FDX_s106 <= 0.2
 0 <= b_k1_FDX_s107 <= 0.2
 0 <= b_k1_FDX_s110 <= 0.2
 0 <= b_k1_FDX_s111 <= 0.2
 0 <= b_k1_FDX_s112 <= 0.2
 0 <= b_k1_FDX_s113 <= 0.2
 0 <= b_k1_FDX_s114 <= 0.2
 0 <= b_k1_FDX_s115 <= 0.2
 0 <= b_k1_FDX_s116 <= 0.2
 0 <= b_k1_FDX_s117 <= 0.2
 0 <= b_k1_FDX_s120 <= 0.2
 0 <= b_k1_FDX_s121 <= 0.2
 0 <= b_k1_FDX_s122 <= 0.2
 0 <= b_k1_FDX_s123 <= 0.2
 0 <= b_k1_FDX_s124 <= 0.2
 0 <= b_k1_FDX_s125 <= 0.2
 0 <= b_k1_FDX_s126 <= 0.2
 0 <= b_k1_FDX_s127 <= 0.2
 0 <= b_k1_FDX_s130 <= 0.2
 0 <= b_k1_FDX_s131 <= 0.2
 0 <= b_k1_FDX_s132 <= 0.2
 0 <= b_k1_FDX_s133 <= 0.2
 0 <= b_k1_FDX_s134 <= 0.2
 0 <= b_k1_FDX_s135 <= 0.2
 0 <= b_k1_FDX_s136 <= 0.2
 0 <= b_k1_FDX_s137 <= 0.2
 0 <= b_k1_FDX_s140 <= 0.2
 0 <= b_k1_FDX_s141 <= 0.2
 0 <= b_k1_FDX_s142 <= 0.2
 0 <= b_k1_FDX_s143 <= 0.2
 0 <= b_k1_FDX_s144 <= 0.2
 0 <= b_k1_FDX_s145 <= 0.2
 0 <= b_k1_FDX_s146 <= 0.2
 0 <= b_k1_FDX_s147 <= 0.2
 0 <= b_k1_FDX_s150 <= 0.2
 0 <= b_k1_FDX_s151 <= 0.2
 0 <= b_k1_FDX_s152 <= 0.2
 0 <= b_k1_FDX_s153 <= 0.2
 0 <= b_k1_FDX_s154 <= 0.2
 0 <= b_k1_FDX_s155 <= 0.2
 0 <= b_k1_FDX_s156 <= 0.2
 0 <= b_k1_FDX_s157 <= 0.2
 0 <= b_k1_FDX_s160 <= 0.2
 0 <= b_k1_FDX_s161 <= 0.2
 0 <= b_k1_FDX_s162 <= 0.2
 0 <= b_k1_FDX_s163 <= 0.2
 0 <= b_k1_FDX_s164 <= 0.2
 0 <= b_k1_FDX_s165 <= 0.2
 0 <= b_k1_FDX_s166 <= 0.2
 0 <= b_k1_FDX_s167 <= 0.2
 0 <= b_k1_FDX_s170 <= 0.2
 0 <= b_k1_FDX_s171 <= 0.2
 0 <= b_k1_FDX_s172 <= 0.2
 0 <= b_k1_FDX_s173 <= 0.2
 0 <= b_k1_FDX_s174 <= 0.2
 0 <= b_k1_FDX_s175 <= 0.2
 0 <= b_k1_FDX_s176 <= 0.2
 0 <= b_k1_FDX_s177 <= 0.2
 0 <= b_k1_IBM_s0 <= 0.2
 0 <= b_k1_IBM_s1 <= 0.2
 0 <= b_k1_IBM_s00 <= 0.2
 0 <= b_k1_IBM_s01 <= 0.2
 0 <= b_k1_IBM_s02 <= 0.2
 0 <= b_k1_IBM_s03 <= 0.2
 0 <= b_k1_IBM_s04 <= 0.2
 0 <= b_k1_IBM_s05 <= 0.2
 0 <= b_k1_IBM_s06 <= 0.2
 0 <= b_k1_IBM_s07 <= 0.2
 0 <= b_k1_IBM_s10 <= 0.2
 0 <= b_k1_IBM_s11 <= 0.2
 0 <= b_k1_IBM_s12 <= 0.2
 0 <= b_k1_IBM_s13 <= 0.2
 0 <= b_k1_IBM_s14 <= 0.2
 0 <= b_k1_IBM_s15 <= 0.2
 0 <= b_k1_IBM_s16 <= 0.2
 0 <= b_k1_IBM_s17 <= 0.2
 0 <= b_k1_IBM_s000 <= 0.2
 0 <= b_k1_IBM_s001 <= 0.2
 0 <= b_k1_IBM_s002 <= 0.2
 0 <= b_k1_IBM_s003 <= 0.2
 0 <= b_k1_IBM_s004 <= 0.2
 0 <= b_k1_IBM_s005 <= 0.2
 0 <= b_k1_IBM_s006 <= 0.2
 0 <= b_k1_IBM_s007 <= 0.2
 0 <= b_k1_IBM_s010 <= 0.2
 0 <= b_k1_IBM_s011 <= 0.2
 0 <= b_k1_IBM_s012 <= 0.2
 0 <= b_k1_IBM_s013 <= 0.2
 0 <= b_k1_IBM_s014 <= 0.2
 0 <= b_k1_IBM_s015 <= 0.2
 0 <= b_k1_IBM_s016 <= 0.2
 0 <= b_k1_IBM_s017 <= 0.2
 0 <= b_k1_IBM_s020 <= 0.2
 0 <= b_k1_IBM_s021 <= 0.2
 0 <= b_k1_IBM_s022 <= 0.2
 0 <= b_k1_IBM_s023 <= 0.2
 0 <= b_k1_IBM_s024 <= 0.2
 0 <= b_k1_IBM_s025 <= 0.2
 0 <= b_k1_IBM_s026 <= 0.2
 0 <= b_k1_IBM_s027 <= 0.2
 0 <= b_k1_IBM_s030 <= 0.2
 0 <= b_k1_IBM_s031 <= 0.2
 0 <= b_k1_IBM_s032 <= 0.2
 0 <= b_k1_IBM_s033 <= 0.2
 0 <= b_k1_IBM_s034 <= 0.2
 0 <= b_k1_IBM_s035 <= 0.2
 0 <= b_k1_IBM_s036 <= 0.2
 0 <= b_k1_IBM_s037 <= 0.2
 0 <= b_k1_IBM_s040 <= 0.2
 0 <= b_k1_IBM_s041 <= 0.2
 0 <= b_k1_IBM_s042 <= 0.2
 0 <= b_k1_IBM_s043 <= 0.2
 0 <= b_k1_IBM_s044 <= 0.2
 0 <= b_k1_IBM_s045 <= 0.2
 0 <= b_k1_IBM_s046 <= 0.2
 0 <= b_k1_IBM_s047 <= 0.2
 0 <= b_k1_IBM_s050 <= 0.2
 0 <= b_k1_IBM_s051 <= 0.2
 0 <= b_k1_IBM_s052 <= 0.2
 0 <= b_k1_IBM_s053 <= 0.2
 0 <= b_k1_IBM_s054 <= 0.2
 0 <= b_k1_IBM_s055 <= 0.2
 0 <= b_k1_IBM_s056 <= 0.2
 0 <= b_k1_IBM_s057 <= 0.2
 0 <= b_k1_IBM_s060 <= 0.2
 0 <= b_k1_IBM_s061 <= 0.2
 0 <= b_k1_IBM_s062 <= 0.2
 0 <= b_k1_IBM_s063 <= 0.2
 0 <= b_k1_IBM_s064 <= 0.2
 0 <= b_k1_IBM_s065 <= 0.2
 0 <= b_k1_IBM_s066 <= 0.2
 0 <= b_k1_IBM_s067 <= 0.2
 0 <= b_k1_IBM_s070 <= 0.2
 0 <= b_k1_IBM_s071 <= 0.2
 0 <= b_k1_IBM_s072 <= 0.2
 0 <= b_k1_IBM_s073 <= 0.2
 0 <= b_k1_IBM_s074 <= 0.2
 0 <= b_k1_IBM_s075 <= 0.2
 0 <= b_k1_IBM_s076 <= 0.2
 0 <= b_k1_IBM_s077 <= 0.2
 0 <= b_k1_IBM_s100 <= 0.2
 0 <= b_k1_IBM_s101 <= 0.2
 0 <= b_k1_IBM_s102 <= 0.2
 0 <= b_k1_IBM_s103 <= 0.2
 0 <= b_k1_IBM_s104 <= 0.2
 0 <= b_k1_IBM_s105 <= 0.2
 0 <= b_k1_IBM_s106 <= 0.2
 0 <= b_k1_IBM_s107 <= 0.2
 0 <= b_k1_IBM_s110 <= 0.2
 0 <= b_k1_IBM_s111 <= 0.2
 0 <= b_k1_IBM_s112 <= 0.2
 0 <= b_k1_IBM_s113 <= 0.2
 0 <= b_k1_IBM_s114 <= 0.2
 0 <= b_k1_IBM_s115 <= 0.2
 0 <= b_k1_IBM_s116 <= 0.2
 0 <= b_k1_IBM_s117 <= 0.2
 0 <= b_k1_IBM_s120 <= 0.2
 0 <= b_k1_IBM_s121 <= 0.2
 0 <= b_k1_IBM_s122 <= 0.2
 0 <= b_k1_IBM_s123 <= 0.2
 0 <= b_k1_IBM_s124 <= 0.2
 0 <= b_k1_IBM_s125 <= 0.2
 0 <= b_k1_IBM_s126 <= 0.2
 0 <= b_k1_IBM_s127 <= 0.2
 0 <= b_k1_IBM_s130 <= 0.2
 0 <= b_k1_IBM_s131 <= 0.2
 0 <= b_k1_IBM_s132 <= 0.2
 0 <= b_k1_IBM_s133 <= 0.2
 0 <= b_k1_IBM_s134 <= 0.2
 0 <= b_k1_IBM_s135 <= 0.2
 0 <= b_k1_IBM_s136 <= 0.2
 0 <= b_k1_IBM_s137 <= 0.2
 0 <= b_k1_IBM_s140 <= 0.2
 0 <= b_k1_IBM_s141 <= 0.2
 0 <= b_k1_IBM_s142 <= 0.2
 0 <= b_k1_IBM_s143 <= 0.2
 0 <= b_k1_IBM_s144 <= 0.2
 0 <= b_k1_IBM_s145 <= 0.2
 0 <= b_k1_IBM_s146 <= 0.2
 0 <= b_k1_IBM_s147 <= 0.2
 0 <= b_k1_IBM_s150 <= 0.2
 0 <= b_k1_IBM_s151 <= 0.2
 0 <= b_k1_IBM_s152 <= 0.2
 0 <= b_k1_IBM_s153 <= 0.2
 0 <= b_k1_IBM_s154 <= 0.2
 0 <= b_k1_IBM_s155 <= 0.2
 0 <= b_k1_IBM_s156 <= 0.2
 0 <= b_k1_IBM_s157 <= 0.2
 0 <= b_k1_IBM_s160 <= 0.2
 0 <= b_k1_IBM_s161 <= 0.2
 0 <= b_k1_IBM_s162 <= 0.2
 0 <= b_k1_IBM_s163 <= 0.2
 0 <= b_k1_IBM_s164 <= 0.2
 0 <= b_k1_IBM_s165 <= 0.2
 0 <= b_k1_IBM_s166 <= 0.2
 0 <= b_k1_IBM_s167 <= 0.2
 0 <= b_k1_IBM_s170 <= 0.2
 0 <= b_k1_IBM_s171 <= 0.2
 0 <= b_k1_IBM_s172 <= 0.2
 0 <= b_k1_IBM_s173 <= 0.2
 0 <= b_k1_IBM_s174 <= 0.2
 0 <= b_k1_IBM_s175 <= 0.2
 0 <= b_k1_IBM_s176 <= 0.2
 0 <= b_k1_IBM_s177 <= 0.2
 0 <= b_k1_INTC_s0 <= 0.2
 0 <= b_k1_INTC_s1 <= 0.2
 0 <= b_k1_INTC_s00 <= 0.2
 0 <= b_k1_INTC_s01 <= 0.2
 0 <= b_k1_INTC_s02 <= 0.2
 0 <= b_k1_INTC_s03 <= 0.2
 0 <= b_k1_INTC_s04 <= 0.2
 0 <= b_k1_INTC_s05 <= 0.2
 0 <= b_k1_INTC_s06 <= 0.2
 0 <= b_k1_INTC_s07 <= 0.2
 0 <= b_k1_INTC_s10 <= 0.2
 0 <= b_k1_INTC_s11 <= 0.2
 0 <= b_k1_INTC_s12 <= 0.2
 0 <= b_k1_INTC_s13 <= 0.2
 0 <= b_k1_INTC_s14 <= 0.2
 0 <= b_k1_INTC_s15 <= 0.2
 0 <= b_k1_INTC_s16 <= 0.2
 0 <= b_k1_INTC_s17 <= 0.2
 0 <= b_k1_INTC_s000 <= 0.2
 0 <= b_k1_INTC_s001 <= 0.2
 0 <= b_k1_INTC_s002 <= 0.2
 0 <= b_k1_INTC_s003 <= 0.2
 0 <= b_k1_INTC_s004 <= 0.2
 0 <= b_k1_INTC_s005 <= 0.2
 0 <= b_k1_INTC_s006 <= 0.2
 0 <= b_k1_INTC_s007 <= 0.2
 0 <= b_k1_INTC_s010 <= 0.2
 0 <= b_k1_INTC_s011 <= 0.2
 0 <= b_k1_INTC_s012 <= 0.2
 0 <= b_k1_INTC_s013 <= 0.2
 0 <= b_k1_INTC_s014 <= 0.2
 0 <= b_k1_INTC_s015 <= 0.2
 0 <= b_k1_INTC_s016 <= 0.2
 0 <= b_k1_INTC_s017 <= 0.2
 0 <= b_k1_INTC_s020 <= 0.2
 0 <= b_k1_INTC_s021 <= 0.2
 0 <= b_k1_INTC_s022 <= 0.2
 0 <= b_k1_INTC_s023 <= 0.2
 0 <= b_k1_INTC_s024 <= 0.2
 0 <= b_k1_INTC_s025 <= 0.2
 0 <= b_k1_INTC_s026 <= 0.2
 0 <= b_k1_INTC_s027 <= 0.2
 0 <= b_k1_INTC_s030 <= 0.2
 0 <= b_k1_INTC_s031 <= 0.2
 0 <= b_k1_INTC_s032 <= 0.2
 0 <= b_k1_INTC_s033 <= 0.2
 0 <= b_k1_INTC_s034 <= 0.2
 0 <= b_k1_INTC_s035 <= 0.2
 0 <= b_k1_INTC_s036 <= 0.2
 0 <= b_k1_INTC_s037 <= 0.2
 0 <= b_k1_INTC_s040 <= 0.2
 0 <= b_k1_INTC_s041 <= 0.2
 0 <= b_k1_INTC_s042 <= 0.2
 0 <= b_k1_INTC_s043 <= 0.2
 0 <= b_k1_INTC_s044 <= 0.2
 0 <= b_k1_INTC_s045 <= 0.2
 0 <= b_k1_INTC_s046 <= 0.2
 0 <= b_k1_INTC_s047 <= 0.2
 0 <= b_k1_INTC_s050 <= 0.2
 0 <= b_k1_INTC_s051 <= 0.2
 0 <= b_k1_INTC_s052 <= 0.2
 0 <= b_k1_INTC_s053 <= 0.2
 0 <= b_k1_INTC_s054 <= 0.2
 0 <= b_k1_INTC_s055 <= 0.2
 0 <= b_k1_INTC_s056 <= 0.2
 0 <= b_k1_INTC_s057 <= 0.2
 0 <= b_k1_INTC_s060 <= 0.2
 0 <= b_k1_INTC_s061 <= 0.2
 0 <= b_k1_INTC_s062 <= 0.2
 0 <= b_k1_INTC_s063 <= 0.2
 0 <= b_k1_INTC_s064 <= 0.2
 0 <= b_k1_INTC_s065 <= 0.2
 0 <= b_k1_INTC_s066 <= 0.2
 0 <= b_k1_INTC_s067 <= 0.2
 0 <= b_k1_INTC_s070 <= 0.2
 0 <= b_k1_INTC_s071 <= 0.2
 0 <= b_k1_INTC_s072 <= 0.2
 0 <= b_k1_INTC_s073 <= 0.2
 0 <= b_k1_INTC_s074 <= 0.2
 0 <= b_k1_INTC_s075 <= 0.2
 0 <= b_k1_INTC_s076 <= 0.2
 0 <= b_k1_INTC_s077 <= 0.2
 0 <= b_k1_INTC_s100 <= 0.2
 0 <= b_k1_INTC_s101 <= 0.2
 0 <= b_k1_INTC_s102 <= 0.2
 0 <= b_k1_INTC_s103 <= 0.2
 0 <= b_k1_INTC_s104 <= 0.2
 0 <= b_k1_INTC_s105 <= 0.2
 0 <= b_k1_INTC_s106 <= 0.2
 0 <= b_k1_INTC_s107 <= 0.2
 0 <= b_k1_INTC_s110 <= 0.2
 0 <= b_k1_INTC_s111 <= 0.2
 0 <= b_k1_INTC_s112 <= 0.2
 0 <= b_k1_INTC_s113 <= 0.2
 0 <= b_k1_INTC_s114 <= 0.2
 0 <= b_k1_INTC_s115 <= 0.2
 0 <= b_k1_INTC_s116 <= 0.2
 0 <= b_k1_INTC_s117 <= 0.2
 0 <= b_k1_INTC_s120 <= 0.2
 0 <= b_k1_INTC_s121 <= 0.2
 0 <= b_k1_INTC_s122 <= 0.2
 0 <= b_k1_INTC_s123 <= 0.2
 0 <= b_k1_INTC_s124 <= 0.2
 0 <= b_k1_INTC_s125 <= 0.2
 0 <= b_k1_INTC_s126 <= 0.2
 0 <= b_k1_INTC_s127 <= 0.2
 0 <= b_k1_INTC_s130 <= 0.2
 0 <= b_k1_INTC_s131 <= 0.2
 0 <= b_k1_INTC_s132 <= 0.2
 0 <= b_k1_INTC_s133 <= 0.2
 0 <= b_k1_INTC_s134 <= 0.2
 0 <= b_k1_INTC_s135 <= 0.2
 0 <= b_k1_INTC_s136 <= 0.2
 0 <= b_k1_INTC_s137 <= 0.2
 0 <= b_k1_INTC_s140 <= 0.2
 0 <= b_k1_INTC_s141 <= 0.2
 0 <= b_k1_INTC_s142 <= 0.2
 0 <= b_k1_INTC_s143 <= 0.2
 0 <= b_k1_INTC_s144 <= 0.2
 0 <= b_k1_INTC_s145 <= 0.2
 0 <= b_k1_INTC_s146 <= 0.2
 0 <= b_k1_INTC_s147 <= 0.2
 0 <= b_k1_INTC_s150 <= 0.2
 0 <= b_k1_INTC_s151 <= 0.2
 0 <= b_k1_INTC_s152 <= 0.2
 0 <= b_k1_INTC_s153 <= 0.2
 0 <= b_k1_INTC_s154 <= 0.2
 0 <= b_k1_INTC_s155 <= 0.2
 0 <= b_k1_INTC_s156 <= 0.2
 0 <= b_k1_INTC_s157 <= 0.2
 0 <= b_k1_INTC_s160 <= 0.2
 0 <= b_k1_INTC_s161 <= 0.2
 0 <= b_k1_INTC_s162 <= 0.2
 0 <= b_k1_INTC_s163 <= 0.2
 0 <= b_k1_INTC_s164 <= 0.2
 0 <= b_k1_INTC_s165 <= 0.2
 0 <= b_k1_INTC_s166 <= 0.2
 0 <= b_k1_INTC_s167 <= 0.2
 0 <= b_k1_INTC_s170 <= 0.2
 0 <= b_k1_INTC_s171 <= 0.2
 0 <= b_k1_INTC_s172 <= 0.2
 0 <= b_k1_INTC_s173 <= 0.2
 0 <= b_k1_INTC_s174 <= 0.2
 0 <= b_k1_INTC_s175 <= 0.2
 0 <= b_k1_INTC_s176 <= 0.2
 0 <= b_k1_INTC_s177 <= 0.2
 0 <= b_k1_KO_s0 <= 0.2
 0 <= b_k1_KO_s1 <= 0.2
 0 <= b_k1_KO_s00 <= 0.2
 0 <= b_k1_KO_s01 <= 0.2
 0 <= b_k1_KO_s02 <= 0.2
 0 <= b_k1_KO_s03 <= 0.2
 0 <= b_k1_KO_s04 <= 0.2
 0 <= b_k1_KO_s05 <= 0.2
 0 <= b_k1_KO_s06 <= 0.2
 0 <= b_k1_KO_s07 <= 0.2
 0 <= b_k1_KO_s10 <= 0.2
 0 <= b_k1_KO_s11 <= 0.2
 0 <= b_k1_KO_s12 <= 0.2
 0 <= b_k1_KO_s13 <= 0.2
 0 <= b_k1_KO_s14 <= 0.2
 0 <= b_k1_KO_s15 <= 0.2
 0 <= b_k1_KO_s16 <= 0.2
 0 <= b_k1_KO_s17 <= 0.2
 0 <= b_k1_KO_s000 <= 0.2
 0 <= b_k1_KO_s001 <= 0.2
 0 <= b_k1_KO_s002 <= 0.2
 0 <= b_k1_KO_s003 <= 0.2
 0 <= b_k1_KO_s004 <= 0.2
 0 <= b_k1_KO_s005 <= 0.2
 0 <= b_k1_KO_s006 <= 0.2
 0 <= b_k1_KO_s007 <= 0.2
 0 <= b_k1_KO_s010 <= 0.2
 0 <= b_k1_KO_s011 <= 0.2
 0 <= b_k1_KO_s012 <= 0.2
 0 <= b_k1_KO_s013 <= 0.2
 0 <= b_k1_KO_s014 <= 0.2
 0 <= b_k1_KO_s015 <= 0.2
 0 <= b_k1_KO_s016 <= 0.2
 0 <= b_k1_KO_s017 <= 0.2
 0 <= b_k1_KO_s020 <= 0.2
 0 <= b_k1_KO_s021 <= 0.2
 0 <= b_k1_KO_s022 <= 0.2
 0 <= b_k1_KO_s023 <= 0.2
 0 <= b_k1_KO_s024 <= 0.2
 0 <= b_k1_KO_s025 <= 0.2
 0 <= b_k1_KO_s026 <= 0.2
 0 <= b_k1_KO_s027 <= 0.2
 0 <= b_k1_KO_s030 <= 0.2
 0 <= b_k1_KO_s031 <= 0.2
 0 <= b_k1_KO_s032 <= 0.2
 0 <= b_k1_KO_s033 <= 0.2
 0 <= b_k1_KO_s034 <= 0.2
 0 <= b_k1_KO_s035 <= 0.2
 0 <= b_k1_KO_s036 <= 0.2
 0 <= b_k1_KO_s037 <= 0.2
 0 <= b_k1_KO_s040 <= 0.2
 0 <= b_k1_KO_s041 <= 0.2
 0 <= b_k1_KO_s042 <= 0.2
 0 <= b_k1_KO_s043 <= 0.2
 0 <= b_k1_KO_s044 <= 0.2
 0 <= b_k1_KO_s045 <= 0.2
 0 <= b_k1_KO_s046 <= 0.2
 0 <= b_k1_KO_s047 <= 0.2
 0 <= b_k1_KO_s050 <= 0.2
 0 <= b_k1_KO_s051 <= 0.2
 0 <= b_k1_KO_s052 <= 0.2
 0 <= b_k1_KO_s053 <= 0.2
 0 <= b_k1_KO_s054 <= 0.2
 0 <= b_k1_KO_s055 <= 0.2
 0 <= b_k1_KO_s056 <= 0.2
 0 <= b_k1_KO_s057 <= 0.2
 0 <= b_k1_KO_s060 <= 0.2
 0 <= b_k1_KO_s061 <= 0.2
 0 <= b_k1_KO_s062 <= 0.2
 0 <= b_k1_KO_s063 <= 0.2
 0 <= b_k1_KO_s064 <= 0.2
 0 <= b_k1_KO_s065 <= 0.2
 0 <= b_k1_KO_s066 <= 0.2
 0 <= b_k1_KO_s067 <= 0.2
 0 <= b_k1_KO_s070 <= 0.2
 0 <= b_k1_KO_s071 <= 0.2
 0 <= b_k1_KO_s072 <= 0.2
 0 <= b_k1_KO_s073 <= 0.2
 0 <= b_k1_KO_s074 <= 0.2
 0 <= b_k1_KO_s075 <= 0.2
 0 <= b_k1_KO_s076 <= 0.2
 0 <= b_k1_KO_s077 <= 0.2
 0 <= b_k1_KO_s100 <= 0.2
 0 <= b_k1_KO_s101 <= 0.2
 0 <= b_k1_KO_s102 <= 0.2
 0 <= b_k1_KO_s103 <= 0.2
 0 <= b_k1_KO_s104 <= 0.2
 0 <= b_k1_KO_s105 <= 0.2
 0 <= b_k1_KO_s106 <= 0.2
 0 <= b_k1_KO_s107 <= 0.2
 0 <= b_k1_KO_s110 <= 0.2
 0 <= b_k1_KO_s111 <= 0.2
 0 <= b_k1_KO_s112 <= 0.2
 0 <= b_k1_KO_s113 <= 0.2
 0 <= b_k1_KO_s114 <= 0.2
 0 <= b_k1_KO_s115 <= 0.2
 0 <= b_k1_KO_s116 <= 0.2
 0 <= b_k1_KO_s117 <= 0.2
 0 <= b_k1_KO_s120 <= 0.2
 0 <= b_k1_KO_s121 <= 0.2
 0 <= b_k1_KO_s122 <= 0.2
 0 <= b_k1_KO_s123 <= 0.2
 0 <= b_k1_KO_s124 <= 0.2
 0 <= b_k1_KO_s125 <= 0.2
 0 <= b_k1_KO_s126 <= 0.2
 0 <= b_k1_KO_s127 <= 0.2
 0 <= b_k1_KO_s130 <= 0.2
 0 <= b_k1_KO_s131 <= 0.2
 0 <= b_k1_KO_s132 <= 0.2
 0 <= b_k1_KO_s133 <= 0.2
 0 <= b_k1_KO_s134 <= 0.2
 0 <= b_k1_KO_s135 <= 0.2
 0 <= b_k1_KO_s136 <= 0.2
 0 <= b_k1_KO_s137 <= 0.2
 0 <= b_k1_KO_s140 <= 0.2
 0 <= b_k1_KO_s141 <= 0.2
 0 <= b_k1_KO_s142 <= 0.2
 0 <= b_k1_KO_s143 <= 0.2
 0 <= b_k1_KO_s144 <= 0.2
 0 <= b_k1_KO_s145 <= 0.2
 0 <= b_k1_KO_s146 <= 0.2
 0 <= b_k1_KO_s147 <= 0.2
 0 <= b_k1_KO_s150 <= 0.2
 0 <= b_k1_KO_s151 <= 0.2
 0 <= b_k1_KO_s152 <= 0.2
 0 <= b_k1_KO_s153 <= 0.2
 0 <= b_k1_KO_s154 <= 0.2
 0 <= b_k1_KO_s155 <= 0.2
 0 <= b_k1_KO_s156 <= 0.2
 0 <= b_k1_KO_s157 <= 0.2
 0 <= b_k1_KO_s160 <= 0.2
 0 <= b_k1_KO_s161 <= 0.2
 0 <= b_k1_KO_s162 <= 0.2
 0 <= b_k1_KO_s163 <= 0.2
 0 <= b_k1_KO_s164 <= 0.2
 0 <= b_k1_KO_s165 <= 0.2
 0 <= b_k1_KO_s166 <= 0.2
 0 <= b_k1_KO_s167 <= 0.2
 0 <= b_k1_KO_s170 <= 0.2
 0 <= b_k1_KO_s171 <= 0.2
 0 <= b_k1_KO_s172 <= 0.2
 0 <= b_k1_KO_s173 <= 0.2
 0 <= b_k1_KO_s174 <= 0.2
 0 <= b_k1_KO_s175 <= 0.2
 0 <= b_k1_KO_s176 <= 0.2
 0 <= b_k1_KO_s177 <= 0.2
 0 <= b_k1_MAT_s0 <= 0.2
 0 <= b_k1_MAT_s1 <= 0.2
 0 <= b_k1_MAT_s00 <= 0.2
 0 <= b_k1_MAT_s01 <= 0.2
 0 <= b_k1_MAT_s02 <= 0.2
 0 <= b_k1_MAT_s03 <= 0.2
 0 <= b_k1_MAT_s04 <= 0.2
 0 <= b_k1_MAT_s05 <= 0.2
 0 <= b_k1_MAT_s06 <= 0.2
 0 <= b_k1_MAT_s07 <= 0.2
 0 <= b_k1_MAT_s10 <= 0.2
 0 <= b_k1_MAT_s11 <= 0.2
 0 <= b_k1_MAT_s12 <= 0.2
 0 <= b_k1_MAT_s13 <= 0.2
 0 <= b_k1_MAT_s14 <= 0.2
 0 <= b_k1_MAT_s15 <= 0.2
 0 <= b_k1_MAT_s16 <= 0.2
 0 <= b_k1_MAT_s17 <= 0.2
 0 <= b_k1_MAT_s000 <= 0.2
 0 <= b_k1_MAT_s001 <= 0.2
 0 <= b_k1_MAT_s002 <= 0.2
 0 <= b_k1_MAT_s003 <= 0.2
 0 <= b_k1_MAT_s004 <= 0.2
 0 <= b_k1_MAT_s005 <= 0.2
 0 <= b_k1_MAT_s006 <= 0.2
 0 <= b_k1_MAT_s007 <= 0.2
 0 <= b_k1_MAT_s010 <= 0.2
 0 <= b_k1_MAT_s011 <= 0.2
 0 <= b_k1_MAT_s012 <= 0.2
 0 <= b_k1_MAT_s013 <= 0.2
 0 <= b_k1_MAT_s014 <= 0.2
 0 <= b_k1_MAT_s015 <= 0.2
 0 <= b_k1_MAT_s016 <= 0.2
 0 <= b_k1_MAT_s017 <= 0.2
 0 <= b_k1_MAT_s020 <= 0.2
 0 <= b_k1_MAT_s021 <= 0.2
 0 <= b_k1_MAT_s022 <= 0.2
 0 <= b_k1_MAT_s023 <= 0.2
 0 <= b_k1_MAT_s024 <= 0.2
 0 <= b_k1_MAT_s025 <= 0.2
 0 <= b_k1_MAT_s026 <= 0.2
 0 <= b_k1_MAT_s027 <= 0.2
 0 <= b_k1_MAT_s030 <= 0.2
 0 <= b_k1_MAT_s031 <= 0.2
 0 <= b_k1_MAT_s032 <= 0.2
 0 <= b_k1_MAT_s033 <= 0.2
 0 <= b_k1_MAT_s034 <= 0.2
 0 <= b_k1_MAT_s035 <= 0.2
 0 <= b_k1_MAT_s036 <= 0.2
 0 <= b_k1_MAT_s037 <= 0.2
 0 <= b_k1_MAT_s040 <= 0.2
 0 <= b_k1_MAT_s041 <= 0.2
 0 <= b_k1_MAT_s042 <= 0.2
 0 <= b_k1_MAT_s043 <= 0.2
 0 <= b_k1_MAT_s044 <= 0.2
 0 <= b_k1_MAT_s045 <= 0.2
 0 <= b_k1_MAT_s046 <= 0.2
 0 <= b_k1_MAT_s047 <= 0.2
 0 <= b_k1_MAT_s050 <= 0.2
 0 <= b_k1_MAT_s051 <= 0.2
 0 <= b_k1_MAT_s052 <= 0.2
 0 <= b_k1_MAT_s053 <= 0.2
 0 <= b_k1_MAT_s054 <= 0.2
 0 <= b_k1_MAT_s055 <= 0.2
 0 <= b_k1_MAT_s056 <= 0.2
 0 <= b_k1_MAT_s057 <= 0.2
 0 <= b_k1_MAT_s060 <= 0.2
 0 <= b_k1_MAT_s061 <= 0.2
 0 <= b_k1_MAT_s062 <= 0.2
 0 <= b_k1_MAT_s063 <= 0.2
 0 <= b_k1_MAT_s064 <= 0.2
 0 <= b_k1_MAT_s065 <= 0.2
 0 <= b_k1_MAT_s066 <= 0.2
 0 <= b_k1_MAT_s067 <= 0.2
 0 <= b_k1_MAT_s070 <= 0.2
 0 <= b_k1_MAT_s071 <= 0.2
 0 <= b_k1_MAT_s072 <= 0.2
 0 <= b_k1_MAT_s073 <= 0.2
 0 <= b_k1_MAT_s074 <= 0.2
 0 <= b_k1_MAT_s075 <= 0.2
 0 <= b_k1_MAT_s076 <= 0.2
 0 <= b_k1_MAT_s077 <= 0.2
 0 <= b_k1_MAT_s100 <= 0.2
 0 <= b_k1_MAT_s101 <= 0.2
 0 <= b_k1_MAT_s102 <= 0.2
 0 <= b_k1_MAT_s103 <= 0.2
 0 <= b_k1_MAT_s104 <= 0.2
 0 <= b_k1_MAT_s105 <= 0.2
 0 <= b_k1_MAT_s106 <= 0.2
 0 <= b_k1_MAT_s107 <= 0.2
 0 <= b_k1_MAT_s110 <= 0.2
 0 <= b_k1_MAT_s111 <= 0.2
 0 <= b_k1_MAT_s112 <= 0.2
 0 <= b_k1_MAT_s113 <= 0.2
 0 <= b_k1_MAT_s114 <= 0.2
 0 <= b_k1_MAT_s115 <= 0.2
 0 <= b_k1_MAT_s116 <= 0.2
 0 <= b_k1_MAT_s117 <= 0.2
 0 <= b_k1_MAT_s120 <= 0.2
 0 <= b_k1_MAT_s121 <= 0.2
 0 <= b_k1_MAT_s122 <= 0.2
 0 <= b_k1_MAT_s123 <= 0.2
 0 <= b_k1_MAT_s124 <= 0.2
 0 <= b_k1_MAT_s125 <= 0.2
 0 <= b_k1_MAT_s126 <= 0.2
 0 <= b_k1_MAT_s127 <= 0.2
 0 <= b_k1_MAT_s130 <= 0.2
 0 <= b_k1_MAT_s131 <= 0.2
 0 <= b_k1_MAT_s132 <= 0.2
 0 <= b_k1_MAT_s133 <= 0.2
 0 <= b_k1_MAT_s134 <= 0.2
 0 <= b_k1_MAT_s135 <= 0.2
 0 <= b_k1_MAT_s136 <= 0.2
 0 <= b_k1_MAT_s137 <= 0.2
 0 <= b_k1_MAT_s140 <= 0.2
 0 <= b_k1_MAT_s141 <= 0.2
 0 <= b_k1_MAT_s142 <= 0.2
 0 <= b_k1_MAT_s143 <= 0.2
 0 <= b_k1_MAT_s144 <= 0.2
 0 <= b_k1_MAT_s145 <= 0.2
 0 <= b_k1_MAT_s146 <= 0.2
 0 <= b_k1_MAT_s147 <= 0.2
 0 <= b_k1_MAT_s150 <= 0.2
 0 <= b_k1_MAT_s151 <= 0.2
 0 <= b_k1_MAT_s152 <= 0.2
 0 <= b_k1_MAT_s153 <= 0.2
 0 <= b_k1_MAT_s154 <= 0.2
 0 <= b_k1_MAT_s155 <= 0.2
 0 <= b_k1_MAT_s156 <= 0.2
 0 <= b_k1_MAT_s157 <= 0.2
 0 <= b_k1_MAT_s160 <= 0.2
 0 <= b_k1_MAT_s161 <= 0.2
 0 <= b_k1_MAT_s162 <= 0.2
 0 <= b_k1_MAT_s163 <= 0.2
 0 <= b_k1_MAT_s164 <= 0.2
 0 <= b_k1_MAT_s165 <= 0.2
 0 <= b_k1_MAT_s166 <= 0.2
 0 <= b_k1_MAT_s167 <= 0.2
 0 <= b_k1_MAT_s170 <= 0.2
 0 <= b_k1_MAT_s171 <= 0.2
 0 <= b_k1_MAT_s172 <= 0.2
 0 <= b_k1_MAT_s173 <= 0.2
 0 <= b_k1_MAT_s174 <= 0.2
 0 <= b_k1_MAT_s175 <= 0.2
 0 <= b_k1_MAT_s176 <= 0.2
 0 <= b_k1_MAT_s177 <= 0.2
 0 <= b_k1_MSFT_s0 <= 0.2
 0 <= b_k1_MSFT_s1 <= 0.2
 0 <= b_k1_MSFT_s00 <= 0.2
 0 <= b_k1_MSFT_s01 <= 0.2
 0 <= b_k1_MSFT_s02 <= 0.2
 0 <= b_k1_MSFT_s03 <= 0.2
 0 <= b_k1_MSFT_s04 <= 0.2
 0 <= b_k1_MSFT_s05 <= 0.2
 0 <= b_k1_MSFT_s06 <= 0.2
 0 <= b_k1_MSFT_s07 <= 0.2
 0 <= b_k1_MSFT_s10 <= 0.2
 0 <= b_k1_MSFT_s11 <= 0.2
 0 <= b_k1_MSFT_s12 <= 0.2
 0 <= b_k1_MSFT_s13 <= 0.2
 0 <= b_k1_MSFT_s14 <= 0.2
 0 <= b_k1_MSFT_s15 <= 0.2
 0 <= b_k1_MSFT_s16 <= 0.2
 0 <= b_k1_MSFT_s17 <= 0.2
 0 <= b_k1_MSFT_s000 <= 0.2
 0 <= b_k1_MSFT_s001 <= 0.2
 0 <= b_k1_MSFT_s002 <= 0.2
 0 <= b_k1_MSFT_s003 <= 0.2
 0 <= b_k1_MSFT_s004 <= 0.2
 0 <= b_k1_MSFT_s005 <= 0.2
 0 <= b_k1_MSFT_s006 <= 0.2
 0 <= b_k1_MSFT_s007 <= 0.2
 0 <= b_k1_MSFT_s010 <= 0.2
 0 <= b_k1_MSFT_s011 <= 0.2
 0 <= b_k1_MSFT_s012 <= 0.2
 0 <= b_k1_MSFT_s013 <= 0.2
 0 <= b_k1_MSFT_s014 <= 0.2
 0 <= b_k1_MSFT_s015 <= 0.2
 0 <= b_k1_MSFT_s016 <= 0.2
 0 <= b_k1_MSFT_s017 <= 0.2
 0 <= b_k1_MSFT_s020 <= 0.2
 0 <= b_k1_MSFT_s021 <= 0.2
 0 <= b_k1_MSFT_s022 <= 0.2
 0 <= b_k1_MSFT_s023 <= 0.2
 0 <= b_k1_MSFT_s024 <= 0.2
 0 <= b_k1_MSFT_s025 <= 0.2
 0 <= b_k1_MSFT_s026 <= 0.2
 0 <= b_k1_MSFT_s027 <= 0.2
 0 <= b_k1_MSFT_s030 <= 0.2
 0 <= b_k1_MSFT_s031 <= 0.2
 0 <= b_k1_MSFT_s032 <= 0.2
 0 <= b_k1_MSFT_s033 <= 0.2
 0 <= b_k1_MSFT_s034 <= 0.2
 0 <= b_k1_MSFT_s035 <= 0.2
 0 <= b_k1_MSFT_s036 <= 0.2
 0 <= b_k1_MSFT_s037 <= 0.2
 0 <= b_k1_MSFT_s040 <= 0.2
 0 <= b_k1_MSFT_s041 <= 0.2
 0 <= b_k1_MSFT_s042 <= 0.2
 0 <= b_k1_MSFT_s043 <= 0.2
 0 <= b_k1_MSFT_s044 <= 0.2
 0 <= b_k1_MSFT_s045 <= 0.2
 0 <= b_k1_MSFT_s046 <= 0.2
 0 <= b_k1_MSFT_s047 <= 0.2
 0 <= b_k1_MSFT_s050 <= 0.2
 0 <= b_k1_MSFT_s051 <= 0.2
 0 <= b_k1_MSFT_s052 <= 0.2
 0 <= b_k1_MSFT_s053 <= 0.2
 0 <= b_k1_MSFT_s054 <= 0.2
 0 <= b_k1_MSFT_s055 <= 0.2
 0 <= b_k1_MSFT_s056 <= 0.2
 0 <= b_k1_MSFT_s057 <= 0.2
 0 <= b_k1_MSFT_s060 <= 0.2
 0 <= b_k1_MSFT_s061 <= 0.2
 0 <= b_k1_MSFT_s062 <= 0.2
 0 <= b_k1_MSFT_s063 <= 0.2
 0 <= b_k1_MSFT_s064 <= 0.2
 0 <= b_k1_MSFT_s065 <= 0.2
 0 <= b_k1_MSFT_s066 <= 0.2
 0 <= b_k1_MSFT_s067 <= 0.2
 0 <= b_k1_MSFT_s070 <= 0.2
 0 <= b_k1_MSFT_s071 <= 0.2
 0 <= b_k1_MSFT_s072 <= 0.2
 0 <= b_k1_MSFT_s073 <= 0.2
 0 <= b_k1_MSFT_s074 <= 0.2
 0 <= b_k1_MSFT_s075 <= 0.2
 0 <= b_k1_MSFT_s076 <= 0.2
 0 <= b_k1_MSFT_s077 <= 0.2
 0 <= b_k1_MSFT_s100 <= 0.2
 0 <= b_k1_MSFT_s101 <= 0.2
 0 <= b_k1_MSFT_s102 <= 0.2
 0 <= b_k1_MSFT_s103 <= 0.2
 0 <= b_k1_MSFT_s104 <= 0.2
 0 <= b_k1_MSFT_s105 <= 0.2
 0 <= b_k1_MSFT_s106 <= 0.2
 0 <= b_k1_MSFT_s107 <= 0.2
 0 <= b_k1_MSFT_s110 <= 0.2
 0 <= b_k1_MSFT_s111 <= 0.2
 0 <= b_k1_MSFT_s112 <= 0.2
 0 <= b_k1_MSFT_s113 <= 0.2
 0 <= b_k1_MSFT_s114 <= 0.2
 0 <= b_k1_MSFT_s115 <= 0.2
 0 <= b_k1_MSFT_s116 <= 0.2
 0 <= b_k1_MSFT_s117 <= 0.2
 0 <= b_k1_MSFT_s120 <= 0.2
 0 <= b_k1_MSFT_s121 <= 0.2
 0 <= b_k1_MSFT_s122 <= 0.2
 0 <= b_k1_MSFT_s123 <= 0.2
 0 <= b_k1_MSFT_s124 <= 0.2
 0 <= b_k1_MSFT_s125 <= 0.2
 0 <= b_k1_MSFT_s126 <= 0.2
 0 <= b_k1_MSFT_s127 <= 0.2
 0 <= b_k1_MSFT_s130 <= 0.2
 0 <= b_k1_MSFT_s131 <= 0.2
 0 <= b_k1_MSFT_s132 <= 0.2
 0 <= b_k1_MSFT_s133 <= 0.2
 0 <= b_k1_MSFT_s134 <= 0.2
 0 <= b_k1_MSFT_s135 <= 0.2
 0 <= b_k1_MSFT_s136 <= 0.2
 0 <= b_k1_MSFT_s137 <= 0.2
 0 <= b_k1_MSFT_s140 <= 0.2
 0 <= b_k1_MSFT_s141 <= 0.2
 0 <= b_k1_MSFT_s142 <= 0.2
 0 <= b_k1_MSFT_s143 <= 0.2
 0 <= b_k1_MSFT_s144 <= 0.2
 0 <= b_k1_MSFT_s145 <= 0.2
 0 <= b_k1_MSFT_s146 <= 0.2
 0 <= b_k1_MSFT_s147 <= 0.2
 0 <= b_k1_MSFT_s150 <= 0.2
 0 <= b_k1_MSFT_s151 <= 0.2
 0 <= b_k1_MSFT_s152 <= 0.2
 0 <= b_k1_MSFT_s153 <= 0.2
 0 <= b_k1_MSFT_s154 <= 0.2
 0 <= b_k1_MSFT_s155 <= 0.2
 0 <= b_k1_MSFT_s156 <= 0.2
 0 <= b_k1_MSFT_s157 <= 0.2
 0 <= b_k1_MSFT_s160 <= 0.2
 0 <= b_k1_MSFT_s161 <= 0.2
 0 <= b_k1_MSFT_s162 <= 0.2
 0 <= b_k1_MSFT_s163 <= 0.2
 0 <= b_k1_MSFT_s164 <= 0.2
 0 <= b_k1_MSFT_s165 <= 0.2
 0 <= b_k1_MSFT_s166 <= 0.2
 0 <= b_k1_MSFT_s167 <= 0.2
 0 <= b_k1_MSFT_s170 <= 0.2
 0 <= b_k1_MSFT_s171 <= 0.2
 0 <= b_k1_MSFT_s172 <= 0.2
 0 <= b_k1_MSFT_s173 <= 0.2
 0 <= b_k1_MSFT_s174 <= 0.2
 0 <= b_k1_MSFT_s175 <= 0.2
 0 <= b_k1_MSFT_s176 <= 0.2
 0 <= b_k1_MSFT_s177 <= 0.2
 0 <= b_k1_PG_s0 <= 0.2
 0 <= b_k1_PG_s1 <= 0.2
 0 <= b_k1_PG_s00 <= 0.2
 0 <= b_k1_PG_s01 <= 0.2
 0 <= b_k1_PG_s02 <= 0.2
 0 <= b_k1_PG_s03 <= 0.2
 0 <= b_k1_PG_s04 <= 0.2
 0 <= b_k1_PG_s05 <= 0.2
 0 <= b_k1_PG_s06 <= 0.2
 0 <= b_k1_PG_s07 <= 0.2
 0 <= b_k1_PG_s10 <= 0.2
 0 <= b_k1_PG_s11 <= 0.2
 0 <= b_k1_PG_s12 <= 0.2
 0 <= b_k1_PG_s13 <= 0.2
 0 <= b_k1_PG_s14 <= 0.2
 0 <= b_k1_PG_s15 <= 0.2
 0 <= b_k1_PG_s16 <= 0.2
 0 <= b_k1_PG_s17 <= 0.2
 0 <= b_k1_PG_s000 <= 0.2
 0 <= b_k1_PG_s001 <= 0.2
 0 <= b_k1_PG_s002 <= 0.2
 0 <= b_k1_PG_s003 <= 0.2
 0 <= b_k1_PG_s004 <= 0.2
 0 <= b_k1_PG_s005 <= 0.2
 0 <= b_k1_PG_s006 <= 0.2
 0 <= b_k1_PG_s007 <= 0.2
 0 <= b_k1_PG_s010 <= 0.2
 0 <= b_k1_PG_s011 <= 0.2
 0 <= b_k1_PG_s012 <= 0.2
 0 <= b_k1_PG_s013 <= 0.2
 0 <= b_k1_PG_s014 <= 0.2
 0 <= b_k1_PG_s015 <= 0.2
 0 <= b_k1_PG_s016 <= 0.2
 0 <= b_k1_PG_s017 <= 0.2
 0 <= b_k1_PG_s020 <= 0.2
 0 <= b_k1_PG_s021 <= 0.2
 0 <= b_k1_PG_s022 <= 0.2
 0 <= b_k1_PG_s023 <= 0.2
 0 <= b_k1_PG_s024 <= 0.2
 0 <= b_k1_PG_s025 <= 0.2
 0 <= b_k1_PG_s026 <= 0.2
 0 <= b_k1_PG_s027 <= 0.2
 0 <= b_k1_PG_s030 <= 0.2
 0 <= b_k1_PG_s031 <= 0.2
 0 <= b_k1_PG_s032 <= 0.2
 0 <= b_k1_PG_s033 <= 0.2
 0 <= b_k1_PG_s034 <= 0.2
 0 <= b_k1_PG_s035 <= 0.2
 0 <= b_k1_PG_s036 <= 0.2
 0 <= b_k1_PG_s037 <= 0.2
 0 <= b_k1_PG_s040 <= 0.2
 0 <= b_k1_PG_s041 <= 0.2
 0 <= b_k1_PG_s042 <= 0.2
 0 <= b_k1_PG_s043 <= 0.2
 0 <= b_k1_PG_s044 <= 0.2
 0 <= b_k1_PG_s045 <= 0.2
 0 <= b_k1_PG_s046 <= 0.2
 0 <= b_k1_PG_s047 <= 0.2
 0 <= b_k1_PG_s050 <= 0.2
 0 <= b_k1_PG_s051 <= 0.2
 0 <= b_k1_PG_s052 <= 0.2
 0 <= b_k1_PG_s053 <= 0.2
 0 <= b_k1_PG_s054 <= 0.2
 0 <= b_k1_PG_s055 <= 0.2
 0 <= b_k1_PG_s056 <= 0.2
 0 <= b_k1_PG_s057 <= 0.2
 0 <= b_k1_PG_s060 <= 0.2
 0 <= b_k1_PG_s061 <= 0.2
 0 <= b_k1_PG_s062 <= 0.2
 0 <= b_k1_PG_s063 <= 0.2
 0 <= b_k1_PG_s064 <= 0.2
 0 <= b_k1_PG_s065 <= 0.2
 0 <= b_k1_PG_s066 <= 0.2
 0 <= b_k1_PG_s067 <= 0.2
 0 <= b_k1_PG_s070 <= 0.2
 0 <= b_k1_PG_s071 <= 0.2
 0 <= b_k1_PG_s072 <= 0.2
 0 <= b_k1_PG_s073 <= 0.2
 0 <= b_k1_PG_s074 <= 0.2
 0 <= b_k1_PG_s075 <= 0.2
 0 <= b_k1_PG_s076 <= 0.2
 0 <= b_k1_PG_s077 <= 0.2
 0 <= b_k1_PG_s100 <= 0.2
 0 <= b_k1_PG_s101 <= 0.2
 0 <= b_k1_PG_s102 <= 0.2
 0 <= b_k1_PG_s103 <= 0.2
 0 <= b_k1_PG_s104 <= 0.2
 0 <= b_k1_PG_s105 <= 0.2
 0 <= b_k1_PG_s106 <= 0.2
 0 <= b_k1_PG_s107 <= 0.2
 0 <= b_k1_PG_s110 <= 0.2
 0 <= b_k1_PG_s111 <= 0.2
 0 <= b_k1_PG_s112 <= 0.2
 0 <= b_k1_PG_s113 <= 0.2
 0 <= b_k1_PG_s114 <= 0.2
 0 <= b_k1_PG_s115 <= 0.2
 0 <= b_k1_PG_s116 <= 0.2
 0 <= b_k1_PG_s117 <= 0.2
 0 <= b_k1_PG_s120 <= 0.2
 0 <= b_k1_PG_s121 <= 0.2
 0 <= b_k1_PG_s122 <= 0.2
 0 <= b_k1_PG_s123 <= 0.2
 0 <= b_k1_PG_s124 <= 0.2
 0 <= b_k1_PG_s125 <= 0.2
 0 <= b_k1_PG_s126 <= 0.2
 0 <= b_k1_PG_s127 <= 0.2
 0 <= b_k1_PG_s130 <= 0.2
 0 <= b_k1_PG_s131 <= 0.2
 0 <= b_k1_PG_s132 <= 0.2
 0 <= b_k1_PG_s133 <= 0.2
 0 <= b_k1_PG_s134 <= 0.2
 0 <= b_k1_PG_s135 <= 0.2
 0 <= b_k1_PG_s136 <= 0.2
 0 <= b_k1_PG_s137 <= 0.2
 0 <= b_k1_PG_s140 <= 0.2
 0 <= b_k1_PG_s141 <= 0.2
 0 <= b_k1_PG_s142 <= 0.2
 0 <= b_k1_PG_s143 <= 0.2
 0 <= b_k1_PG_s144 <= 0.2
 0 <= b_k1_PG_s145 <= 0.2
 0 <= b_k1_PG_s146 <= 0.2
 0 <= b_k1_PG_s147 <= 0.2
 0 <= b_k1_PG_s150 <= 0.2
 0 <= b_k1_PG_s151 <= 0.2
 0 <= b_k1_PG_s152 <= 0.2
 0 <= b_k1_PG_s153 <= 0.2
 0 <= b_k1_PG_s154 <= 0.2
 0 <= b_k1_PG_s155 <= 0.2
 0 <= b_k1_PG_s156 <= 0.2
 0 <= b_k1_PG_s157 <= 0.2
 0 <= b_k1_PG_s160 <= 0.2
 0 <= b_k1_PG_s161 <= 0.2
 0 <= b_k1_PG_s162 <= 0.2
 0 <= b_k1_PG_s163 <= 0.2
 0 <= b_k1_PG_s164 <= 0.2
 0 <= b_k1_PG_s165 <= 0.2
 0 <= b_k1_PG_s166 <= 0.2
 0 <= b_k1_PG_s167 <= 0.2
 0 <= b_k1_PG_s170 <= 0.2
 0 <= b_k1_PG_s171 <= 0.2
 0 <= b_k1_PG_s172 <= 0.2
 0 <= b_k1_PG_s173 <= 0.2
 0 <= b_k1_PG_s174 <= 0.2
 0 <= b_k1_PG_s175 <= 0.2
 0 <= b_k1_PG_s176 <= 0.2
 0 <= b_k1_PG_s177 <= 0.2
 0 <= s_0_ADM <= 0.2
 0 <= s_0_AXP <= 0.2
 0 <= s_0_DIS <= 0.2
 0 <= s_0_FDX <= 0.2
 0 <= s_0_IBM <= 0.2
 0 <= s_0_INTC <= 0.2
 0 <= s_0_KO <= 0.2
 0 <= s_0_MAT <= 0.2
 0 <= s_0_MSFT <= 0.2
 0 <= s_0_PG <= 0.2
 0 <= s_k1_ADM_s0 <= 0.2
 0 <= s_k1_ADM_s1 <= 0.2
 0 <= s_k1_ADM_s00 <= 0.2
 0 <= s_k1_ADM_s01 <= 0.2
 0 <= s_k1_ADM_s02 <= 0.2
 0 <= s_k1_ADM_s03 <= 0.2
 0 <= s_k1_ADM_s04 <= 0.2
 0 <= s_k1_ADM_s05 <= 0.2
 0 <= s_k1_ADM_s06 <= 0.2
 0 <= s_k1_ADM_s07 <= 0.2
 0 <= s_k1_ADM_s10 <= 0.2
 0 <= s_k1_ADM_s11 <= 0.2
 0 <= s_k1_ADM_s12 <= 0.2
 0 <= s_k1_ADM_s13 <= 0.2
 0 <= s_k1_ADM_s14 <= 0.2
 0 <= s_k1_ADM_s15 <= 0.2
 0 <= s_k1_ADM_s16 <= 0.2
 0 <= s_k1_ADM_s17 <= 0.2
 0 <= s_k1_ADM_s000 <= 0.2
 0 <= s_k1_ADM_s001 <= 0.2
 0 <= s_k1_ADM_s002 <= 0.2
 0 <= s_k1_ADM_s003 <= 0.2
 0 <= s_k1_ADM_s004 <= 0.2
 0 <= s_k1_ADM_s005 <= 0.2
 0 <= s_k1_ADM_s006 <= 0.2
 0 <= s_k1_ADM_s007 <= 0.2
 0 <= s_k1_ADM_s010 <= 0.2
 0 <= s_k1_ADM_s011 <= 0.2
 0 <= s_k1_ADM_s012 <= 0.2
 0 <= s_k1_ADM_s013 <= 0.2
 0 <= s_k1_ADM_s014 <= 0.2
 0 <= s_k1_ADM_s015 <= 0.2
 0 <= s_k1_ADM_s016 <= 0.2
 0 <= s_k1_ADM_s017 <= 0.2
 0 <= s_k1_ADM_s020 <= 0.2
 0 <= s_k1_ADM_s021 <= 0.2
 0 <= s_k1_ADM_s022 <= 0.2
 0 <= s_k1_ADM_s023 <= 0.2
 0 <= s_k1_ADM_s024 <= 0.2
 0 <= s_k1_ADM_s025 <= 0.2
 0 <= s_k1_ADM_s026 <= 0.2
 0 <= s_k1_ADM_s027 <= 0.2
 0 <= s_k1_ADM_s030 <= 0.2
 0 <= s_k1_ADM_s031 <= 0.2
 0 <= s_k1_ADM_s032 <= 0.2
 0 <= s_k1_ADM_s033 <= 0.2
 0 <= s_k1_ADM_s034 <= 0.2
 0 <= s_k1_ADM_s035 <= 0.2
 0 <= s_k1_ADM_s036 <= 0.2
 0 <= s_k1_ADM_s037 <= 0.2
 0 <= s_k1_ADM_s040 <= 0.2
 0 <= s_k1_ADM_s041 <= 0.2
 0 <= s_k1_ADM_s042 <= 0.2
 0 <= s_k1_ADM_s043 <= 0.2
 0 <= s_k1_ADM_s044 <= 0.2
 0 <= s_k1_ADM_s045 <= 0.2
 0 <= s_k1_ADM_s046 <= 0.2
 0 <= s_k1_ADM_s047 <= 0.2
 0 <= s_k1_ADM_s050 <= 0.2
 0 <= s_k1_ADM_s051 <= 0.2
 0 <= s_k1_ADM_s052 <= 0.2
 0 <= s_k1_ADM_s053 <= 0.2
 0 <= s_k1_ADM_s054 <= 0.2
 0 <= s_k1_ADM_s055 <= 0.2
 0 <= s_k1_ADM_s056 <= 0.2
 0 <= s_k1_ADM_s057 <= 0.2
 0 <= s_k1_ADM_s060 <= 0.2
 0 <= s_k1_ADM_s061 <= 0.2
 0 <= s_k1_ADM_s062 <= 0.2
 0 <= s_k1_ADM_s063 <= 0.2
 0 <= s_k1_ADM_s064 <= 0.2
 0 <= s_k1_ADM_s065 <= 0.2
 0 <= s_k1_ADM_s066 <= 0.2
 0 <= s_k1_ADM_s067 <= 0.2
 0 <= s_k1_ADM_s070 <= 0.2
 0 <= s_k1_ADM_s071 <= 0.2
 0 <= s_k1_ADM_s072 <= 0.2
 0 <= s_k1_ADM_s073 <= 0.2
 0 <= s_k1_ADM_s074 <= 0.2
 0 <= s_k1_ADM_s075 <= 0.2
 0 <= s_k1_ADM_s076 <= 0.2
 0 <= s_k1_ADM_s077 <= 0.2
 0 <= s_k1_ADM_s100 <= 0.2
 0 <= s_k1_ADM_s101 <= 0.2
 0 <= s_k1_ADM_s102 <= 0.2
 0 <= s_k1_ADM_s103 <= 0.2
 0 <= s_k1_ADM_s104 <= 0.2
 0 <= s_k1_ADM_s105 <= 0.2
 0 <= s_k1_ADM_s106 <= 0.2
 0 <= s_k1_ADM_s107 <= 0.2
 0 <= s_k1_ADM_s110 <= 0.2
 0 <= s_k1_ADM_s111 <= 0.2
 0 <= s_k1_ADM_s112 <= 0.2
 0 <= s_k1_ADM_s113 <= 0.2
 0 <= s_k1_ADM_s114 <= 0.2
 0 <= s_k1_ADM_s115 <= 0.2
 0 <= s_k1_ADM_s116 <= 0.2
 0 <= s_k1_ADM_s117 <= 0.2
 0 <= s_k1_ADM_s120 <= 0.2
 0 <= s_k1_ADM_s121 <= 0.2
 0 <= s_k1_ADM_s122 <= 0.2
 0 <= s_k1_ADM_s123 <= 0.2
 0 <= s_k1_ADM_s124 <= 0.2
 0 <= s_k1_ADM_s125 <= 0.2
 0 <= s_k1_ADM_s126 <= 0.2
 0 <= s_k1_ADM_s127 <= 0.2
 0 <= s_k1_ADM_s130 <= 0.2
 0 <= s_k1_ADM_s131 <= 0.2
 0 <= s_k1_ADM_s132 <= 0.2
 0 <= s_k1_ADM_s133 <= 0.2
 0 <= s_k1_ADM_s134 <= 0.2
 0 <= s_k1_ADM_s135 <= 0.2
 0 <= s_k1_ADM_s136 <= 0.2
 0 <= s_k1_ADM_s137 <= 0.2
 0 <= s_k1_ADM_s140 <= 0.2
 0 <= s_k1_ADM_s141 <= 0.2
 0 <= s_k1_ADM_s142 <= 0.2
 0 <= s_k1_ADM_s143 <= 0.2
 0 <= s_k1_ADM_s144 <= 0.2
 0 <= s_k1_ADM_s145 <= 0.2
 0 <= s_k1_ADM_s146 <= 0.2
 0 <= s_k1_ADM_s147 <= 0.2
 0 <= s_k1_ADM_s150 <= 0.2
 0 <= s_k1_ADM_s151 <= 0.2
 0 <= s_k1_ADM_s152 <= 0.2
 0 <= s_k1_ADM_s153 <= 0.2
 0 <= s_k1_ADM_s154 <= 0.2
 0 <= s_k1_ADM_s155 <= 0.2
 0 <= s_k1_ADM_s156 <= 0.2
 0 <= s_k1_ADM_s157 <= 0.2
 0 <= s_k1_ADM_s160 <= 0.2
 0 <= s_k1_ADM_s161 <= 0.2
 0 <= s_k1_ADM_s162 <= 0.2
 0 <= s_k1_ADM_s163 <= 0.2
 0 <= s_k1_ADM_s164 <= 0.2
 0 <= s_k1_ADM_s165 <= 0.2
 0 <= s_k1_ADM_s166 <= 0.2
 0 <= s_k1_ADM_s167 <= 0.2
 0 <= s_k1_ADM_s170 <= 0.2
 0 <= s_k1_ADM_s171 <= 0.2
 0 <= s_k1_ADM_s172 <= 0.2
 0 <= s_k1_ADM_s173 <= 0.2
 0 <= s_k1_ADM_s174 <= 0.2
 0 <= s_k1_ADM_s175 <= 0.2
 0 <= s_k1_ADM_s176 <= 0.2
 0 <= s_k1_ADM_s177 <= 0.2
 0 <= s_k1_AXP_s0 <= 0.2
 0 <= s_k1_AXP_s1 <= 0.2
 0 <= s_k1_AXP_s00 <= 0.2
 0 <= s_k1_AXP_s01 <= 0.2
 0 <= s_k1_AXP_s02 <= 0.2
 0 <= s_k1_AXP_s03 <= 0.2
 0 <= s_k1_AXP_s04 <= 0.2
 0 <= s_k1_AXP_s05 <= 0.2
 0 <= s_k1_AXP_s06 <= 0.2
 0 <= s_k1_AXP_s07 <= 0.2
 0 <= s_k1_AXP_s10 <= 0.2
 0 <= s_k1_AXP_s11 <= 0.2
 0 <= s_k1_AXP_s12 <= 0.2
 0 <= s_k1_AXP_s13 <= 0.2
 0 <= s_k1_AXP_s14 <= 0.2
 0 <= s_k1_AXP_s15 <= 0.2
 0 <= s_k1_AXP_s16 <= 0.2
 0 <= s_k1_AXP_s17 <= 0.2
 0 <= s_k1_AXP_s000 <= 0.2
 0 <= s_k1_AXP_s001 <= 0.2
 0 <= s_k1_AXP_s002 <= 0.2
 0 <= s_k1_AXP_s003 <= 0.2
 0 <= s_k1_AXP_s004 <= 0.2
 0 <= s_k1_AXP_s005 <= 0.2
 0 <= s_k1_AXP_s006 <= 0.2
 0 <= s_k1_AXP_s007 <= 0.2
 0 <= s_k1_AXP_s010 <= 0.2
 0 <= s_k1_AXP_s011 <= 0.2
 0 <= s_k1_AXP_s012 <= 0.2
 0 <= s_k1_AXP_s013 <= 0.2
 0 <= s_k1_AXP_s014 <= 0.2
 0 <= s_k1_AXP_s015 <= 0.2
 0 <= s_k1_AXP_s016 <= 0.2
 0 <= s_k1_AXP_s017 <= 0.2
 0 <= s_k1_AXP_s020 <= 0.2
 0 <= s_k1_AXP_s021 <= 0.2
 0 <= s_k1_AXP_s022 <= 0.2
 0 <= s_k1_AXP_s023 <= 0.2
 0 <= s_k1_AXP_s024 <= 0.2
 0 <= s_k1_AXP_s025 <= 0.2
 0 <= s_k1_AXP_s026 <= 0.2
 0 <= s_k1_AXP_s027 <= 0.2
 0 <= s_k1_AXP_s030 <= 0.2
 0 <= s_k1_AXP_s031 <= 0.2
 0 <= s_k1_AXP_s032 <= 0.2
 0 <= s_k1_AXP_s033 <= 0.2
 0 <= s_k1_AXP_s034 <= 0.2
 0 <= s_k1_AXP_s035 <= 0.2
 0 <= s_k1_AXP_s036 <= 0.2
 0 <= s_k1_AXP_s037 <= 0.2
 0 <= s_k1_AXP_s040 <= 0.2
 0 <= s_k1_AXP_s041 <= 0.2
 0 <= s_k1_AXP_s042 <= 0.2
 0 <= s_k1_AXP_s043 <= 0.2
 0 <= s_k1_AXP_s044 <= 0.2
 0 <= s_k1_AXP_s045 <= 0.2
 0 <= s_k1_AXP_s046 <= 0.2
 0 <= s_k1_AXP_s047 <= 0.2
 0 <= s_k1_AXP_s050 <= 0.2
 0 <= s_k1_AXP_s051 <= 0.2
 0 <= s_k1_AXP_s052 <= 0.2
 0 <= s_k1_AXP_s053 <= 0.2
 0 <= s_k1_AXP_s054 <= 0.2
 0 <= s_k1_AXP_s055 <= 0.2
 0 <= s_k1_AXP_s056 <= 0.2
 0 <= s_k1_AXP_s057 <= 0.2
 0 <= s_k1_AXP_s060 <= 0.2
 0 <= s_k1_AXP_s061 <= 0.2
 0 <= s_k1_AXP_s062 <= 0.2
 0 <= s_k1_AXP_s063 <= 0.2
 0 <= s_k1_AXP_s064 <= 0.2
 0 <= s_k1_AXP_s065 <= 0.2
 0 <= s_k1_AXP_s066 <= 0.2
 0 <= s_k1_AXP_s067 <= 0.2
 0 <= s_k1_AXP_s070 <= 0.2
 0 <= s_k1_AXP_s071 <= 0.2
 0 <= s_k1_AXP_s072 <= 0.2
 0 <= s_k1_AXP_s073 <= 0.2
 0 <= s_k1_AXP_s074 <= 0.2
 0 <= s_k1_AXP_s075 <= 0.2
 0 <= s_k1_AXP_s076 <= 0.2
 0 <= s_k1_AXP_s077 <= 0.2
 0 <= s_k1_AXP_s100 <= 0.2
 0 <= s_k1_AXP_s101 <= 0.2
 0 <= s_k1_AXP_s102 <= 0.2
 0 <= s_k1_AXP_s103 <= 0.2
 0 <= s_k1_AXP_s104 <= 0.2
 0 <= s_k1_AXP_s105 <= 0.2
 0 <= s_k1_AXP_s106 <= 0.2
 0 <= s_k1_AXP_s107 <= 0.2
 0 <= s_k1_AXP_s110 <= 0.2
 0 <= s_k1_AXP_s111 <= 0.2
 0 <= s_k1_AXP_s112 <= 0.2
 0 <= s_k1_AXP_s113 <= 0.2
 0 <= s_k1_AXP_s114 <= 0.2
 0 <= s_k1_AXP_s115 <= 0.2
 0 <= s_k1_AXP_s116 <= 0.2
 0 <= s_k1_AXP_s117 <= 0.2
 0 <= s_k1_AXP_s120 <= 0.2
 0 <= s_k1_AXP_s121 <= 0.2
 0 <= s_k1_AXP_s122 <= 0.2
 0 <= s_k1_AXP_s123 <= 0.2
 0 <= s_k1_AXP_s124 <= 0.2
 0 <= s_k1_AXP_s125 <= 0.2
 0 <= s_k1_AXP_s126 <= 0.2
 0 <= s_k1_AXP_s127 <= 0.2
 0 <= s_k1_AXP_s130 <= 0.2
 0 <= s_k1_AXP_s131 <= 0.2
 0 <= s_k1_AXP_s132 <= 0.2
 0 <= s_k1_AXP_s133 <= 0.2
 0 <= s_k1_AXP_s134 <= 0.2
 0 <= s_k1_AXP_s135 <= 0.2
 0 <= s_k1_AXP_s136 <= 0.2
 0 <= s_k1_AXP_s137 <= 0.2
 0 <= s_k1_AXP_s140 <= 0.2
 0 <= s_k1_AXP_s141 <= 0.2
 0 <= s_k1_AXP_s142 <= 0.2
 0 <= s_k1_AXP_s143 <= 0.2
 0 <= s_k1_AXP_s144 <= 0.2
 0 <= s_k1_AXP_s145 <= 0.2
 0 <= s_k1_AXP_s146 <= 0.2
 0 <= s_k1_AXP_s147 <= 0.2
 0 <= s_k1_AXP_s150 <= 0.2
 0 <= s_k1_AXP_s151 <= 0.2
 0 <= s_k1_AXP_s152 <= 0.2
 0 <= s_k1_AXP_s153 <= 0.2
 0 <= s_k1_AXP_s154 <= 0.2
 0 <= s_k1_AXP_s155 <= 0.2
 0 <= s_k1_AXP_s156 <= 0.2
 0 <= s_k1_AXP_s157 <= 0.2
 0 <= s_k1_AXP_s160 <= 0.2
 0 <= s_k1_AXP_s161 <= 0.2
 0 <= s_k1_AXP_s162 <= 0.2
 0 <= s_k1_AXP_s163 <= 0.2
 0 <= s_k1_AXP_s164 <= 0.2
 0 <= s_k1_AXP_s165 <= 0.2
 0 <= s_k1_AXP_s166 <= 0.2
 0 <= s_k1_AXP_s167 <= 0.2
 0 <= s_k1_AXP_s170 <= 0.2
 0 <= s_k1_AXP_s171 <= 0.2
 0 <= s_k1_AXP_s172 <= 0.2
 0 <= s_k1_AXP_s173 <= 0.2
 0 <= s_k1_AXP_s174 <= 0.2
 0 <= s_k1_AXP_s175 <= 0.2
 0 <= s_k1_AXP_s176 <= 0.2
 0 <= s_k1_AXP_s177 <= 0.2
 0 <= s_k1_DIS_s0 <= 0.2
 0 <= s_k1_DIS_s1 <= 0.2
 0 <= s_k1_DIS_s00 <= 0.2
 0 <= s_k1_DIS_s01 <= 0.2
 0 <= s_k1_DIS_s02 <= 0.2
 0 <= s_k1_DIS_s03 <= 0.2
 0 <= s_k1_DIS_s04 <= 0.2
 0 <= s_k1_DIS_s05 <= 0.2
 0 <= s_k1_DIS_s06 <= 0.2
 0 <= s_k1_DIS_s07 <= 0.2
 0 <= s_k1_DIS_s10 <= 0.2
 0 <= s_k1_DIS_s11 <= 0.2
 0 <= s_k1_DIS_s12 <= 0.2
 0 <= s_k1_DIS_s13 <= 0.2
 0 <= s_k1_DIS_s14 <= 0.2
 0 <= s_k1_DIS_s15 <= 0.2
 0 <= s_k1_DIS_s16 <= 0.2
 0 <= s_k1_DIS_s17 <= 0.2
 0 <= s_k1_DIS_s000 <= 0.2
 0 <= s_k1_DIS_s001 <= 0.2
 0 <= s_k1_DIS_s002 <= 0.2
 0 <= s_k1_DIS_s003 <= 0.2
 0 <= s_k1_DIS_s004 <= 0.2
 0 <= s_k1_DIS_s005 <= 0.2
 0 <= s_k1_DIS_s006 <= 0.2
 0 <= s_k1_DIS_s007 <= 0.2
 0 <= s_k1_DIS_s010 <= 0.2
 0 <= s_k1_DIS_s011 <= 0.2
 0 <= s_k1_DIS_s012 <= 0.2
 0 <= s_k1_DIS_s013 <= 0.2
 0 <= s_k1_DIS_s014 <= 0.2
 0 <= s_k1_DIS_s015 <= 0.2
 0 <= s_k1_DIS_s016 <= 0.2
 0 <= s_k1_DIS_s017 <= 0.2
 0 <= s_k1_DIS_s020 <= 0.2
 0 <= s_k1_DIS_s021 <= 0.2
 0 <= s_k1_DIS_s022 <= 0.2
 0 <= s_k1_DIS_s023 <= 0.2
 0 <= s_k1_DIS_s024 <= 0.2
 0 <= s_k1_DIS_s025 <= 0.2
 0 <= s_k1_DIS_s026 <= 0.2
 0 <= s_k1_DIS_s027 <= 0.2
 0 <= s_k1_DIS_s030 <= 0.2
 0 <= s_k1_DIS_s031 <= 0.2
 0 <= s_k1_DIS_s032 <= 0.2
 0 <= s_k1_DIS_s033 <= 0.2
 0 <= s_k1_DIS_s034 <= 0.2
 0 <= s_k1_DIS_s035 <= 0.2
 0 <= s_k1_DIS_s036 <= 0.2
 0 <= s_k1_DIS_s037 <= 0.2
 0 <= s_k1_DIS_s040 <= 0.2
 0 <= s_k1_DIS_s041 <= 0.2
 0 <= s_k1_DIS_s042 <= 0.2
 0 <= s_k1_DIS_s043 <= 0.2
 0 <= s_k1_DIS_s044 <= 0.2
 0 <= s_k1_DIS_s045 <= 0.2
 0 <= s_k1_DIS_s046 <= 0.2
 0 <= s_k1_DIS_s047 <= 0.2
 0 <= s_k1_DIS_s050 <= 0.2
 0 <= s_k1_DIS_s051 <= 0.2
 0 <= s_k1_DIS_s052 <= 0.2
 0 <= s_k1_DIS_s053 <= 0.2
 0 <= s_k1_DIS_s054 <= 0.2
 0 <= s_k1_DIS_s055 <= 0.2
 0 <= s_k1_DIS_s056 <= 0.2
 0 <= s_k1_DIS_s057 <= 0.2
 0 <= s_k1_DIS_s060 <= 0.2
 0 <= s_k1_DIS_s061 <= 0.2
 0 <= s_k1_DIS_s062 <= 0.2
 0 <= s_k1_DIS_s063 <= 0.2
 0 <= s_k1_DIS_s064 <= 0.2
 0 <= s_k1_DIS_s065 <= 0.2
 0 <= s_k1_DIS_s066 <= 0.2
 0 <= s_k1_DIS_s067 <= 0.2
 0 <= s_k1_DIS_s070 <= 0.2
 0 <= s_k1_DIS_s071 <= 0.2
 0 <= s_k1_DIS_s072 <= 0.2
 0 <= s_k1_DIS_s073 <= 0.2
 0 <= s_k1_DIS_s074 <= 0.2
 0 <= s_k1_DIS_s075 <= 0.2
 0 <= s_k1_DIS_s076 <= 0.2
 0 <= s_k1_DIS_s077 <= 0.2
 0 <= s_k1_DIS_s100 <= 0.2
 0 <= s_k1_DIS_s101 <= 0.2
 0 <= s_k1_DIS_s102 <= 0.2
 0 <= s_k1_DIS_s103 <= 0.2
 0 <= s_k1_DIS_s104 <= 0.2
 0 <= s_k1_DIS_s105 <= 0.2
 0 <= s_k1_DIS_s106 <= 0.2
 0 <= s_k1_DIS_s107 <= 0.2
 0 <= s_k1_DIS_s110 <= 0.2
 0 <= s_k1_DIS_s111 <= 0.2
 0 <= s_k1_DIS_s112 <= 0.2
 0 <= s_k1_DIS_s113 <= 0.2
 0 <= s_k1_DIS_s114 <= 0.2
 0 <= s_k1_DIS_s115 <= 0.2
 0 <= s_k1_DIS_s116 <= 0.2
 0 <= s_k1_DIS_s117 <= 0.2
 0 <= s_k1_DIS_s120 <= 0.2
 0 <= s_k1_DIS_s121 <= 0.2
 0 <= s_k1_DIS_s122 <= 0.2
 0 <= s_k1_DIS_s123 <= 0.2
 0 <= s_k1_DIS_s124 <= 0.2
 0 <= s_k1_DIS_s125 <= 0.2
 0 <= s_k1_DIS_s126 <= 0.2
 0 <= s_k1_DIS_s127 <= 0.2
 0 <= s_k1_DIS_s130 <= 0.2
 0 <= s_k1_DIS_s131 <= 0.2
 0 <= s_k1_DIS_s132 <= 0.2
 0 <= s_k1_DIS_s133 <= 0.2
 0 <= s_k1_DIS_s134 <= 0.2
 0 <= s_k1_DIS_s135 <= 0.2
 0 <= s_k1_DIS_s136 <= 0.2
 0 <= s_k1_DIS_s137 <= 0.2
 0 <= s_k1_DIS_s140 <= 0.2
 0 <= s_k1_DIS_s141 <= 0.2
 0 <= s_k1_DIS_s142 <= 0.2
 0 <= s_k1_DIS_s143 <= 0.2
 0 <= s_k1_DIS_s144 <= 0.2
 0 <= s_k1_DIS_s145 <= 0.2
 0 <= s_k1_DIS_s146 <= 0.2
 0 <= s_k1_DIS_s147 <= 0.2
 0 <= s_k1_DIS_s150 <= 0.2
 0 <= s_k1_DIS_s151 <= 0.2
 0 <= s_k1_DIS_s152 <= 0.2
 0 <= s_k1_DIS_s153 <= 0.2
 0 <= s_k1_DIS_s154 <= 0.2
 0 <= s_k1_DIS_s155 <= 0.2
 0 <= s_k1_DIS_s156 <= 0.2
 0 <= s_k1_DIS_s157 <= 0.2
 0 <= s_k1_DIS_s160 <= 0.2
 0 <= s_k1_DIS_s161 <= 0.2
 0 <= s_k1_DIS_s162 <= 0.2
 0 <= s_k1_DIS_s163 <= 0.2
 0 <= s_k1_DIS_s164 <= 0.2
 0 <= s_k1_DIS_s165 <= 0.2
 0 <= s_k1_DIS_s166 <= 0.2
 0 <= s_k1_DIS_s167 <= 0.2
 0 <= s_k1_DIS_s170 <= 0.2
 0 <= s_k1_DIS_s171 <= 0.2
 0 <= s_k1_DIS_s172 <= 0.2
 0 <= s_k1_DIS_s173 <= 0.2
 0 <= s_k1_DIS_s174 <= 0.2
 0 <= s_k1_DIS_s175 <= 0.2
 0 <= s_k1_DIS_s176 <= 0.2
 0 <= s_k1_DIS_s177 <= 0.2
 0 <= s_k1_FDX_s0 <= 0.2
 0 <= s_k1_FDX_s1 <= 0.2
 0 <= s_k1_FDX_s00 <= 0.2
 0 <= s_k1_FDX_s01 <= 0.2
 0 <= s_k1_FDX_s02 <= 0.2
 0 <= s_k1_FDX_s03 <= 0.2
 0 <= s_k1_FDX_s04 <= 0.2
 0 <= s_k1_FDX_s05 <= 0.2
 0 <= s_k1_FDX_s06 <= 0.2
 0 <= s_k1_FDX_s07 <= 0.2
 0 <= s_k1_FDX_s10 <= 0.2
 0 <= s_k1_FDX_s11 <= 0.2
 0 <= s_k1_FDX_s12 <= 0.2
 0 <= s_k1_FDX_s13 <= 0.2
 0 <= s_k1_FDX_s14 <= 0.2
 0 <= s_k1_FDX_s15 <= 0.2
 0 <= s_k1_FDX_s16 <= 0.2
 0 <= s_k1_FDX_s17 <= 0.2
 0 <= s_k1_FDX_s000 <= 0.2
 0 <= s_k1_FDX_s001 <= 0.2
 0 <= s_k1_FDX_s002 <= 0.2
 0 <= s_k1_FDX_s003 <= 0.2
 0 <= s_k1_FDX_s004 <= 0.2
 0 <= s_k1_FDX_s005 <= 0.2
 0 <= s_k1_FDX_s006 <= 0.2
 0 <= s_k1_FDX_s007 <= 0.2
 0 <= s_k1_FDX_s010 <= 0.2
 0 <= s_k1_FDX_s011 <= 0.2
 0 <= s_k1_FDX_s012 <= 0.2
 0 <= s_k1_FDX_s013 <= 0.2
 0 <= s_k1_FDX_s014 <= 0.2
 0 <= s_k1_FDX_s015 <= 0.2
 0 <= s_k1_FDX_s016 <= 0.2
 0 <= s_k1_FDX_s017 <= 0.2
 0 <= s_k1_FDX_s020 <= 0.2
 0 <= s_k1_FDX_s021 <= 0.2
 0 <= s_k1_FDX_s022 <= 0.2
 0 <= s_k1_FDX_s023 <= 0.2
 0 <= s_k1_FDX_s024 <= 0.2
 0 <= s_k1_FDX_s025 <= 0.2
 0 <= s_k1_FDX_s026 <= 0.2
 0 <= s_k1_FDX_s027 <= 0.2
 0 <= s_k1_FDX_s030 <= 0.2
 0 <= s_k1_FDX_s031 <= 0.2
 0 <= s_k1_FDX_s032 <= 0.2
 0 <= s_k1_FDX_s033 <= 0.2
 0 <= s_k1_FDX_s034 <= 0.2
 0 <= s_k1_FDX_s035 <= 0.2
 0 <= s_k1_FDX_s036 <= 0.2
 0 <= s_k1_FDX_s037 <= 0.2
 0 <= s_k1_FDX_s040 <= 0.2
 0 <= s_k1_FDX_s041 <= 0.2
 0 <= s_k1_FDX_s042 <= 0.2
 0 <= s_k1_FDX_s043 <= 0.2
 0 <= s_k1_FDX_s044 <= 0.2
 0 <= s_k1_FDX_s045 <= 0.2
 0 <= s_k1_FDX_s046 <= 0.2
 0 <= s_k1_FDX_s047 <= 0.2
 0 <= s_k1_FDX_s050 <= 0.2
 0 <= s_k1_FDX_s051 <= 0.2
 0 <= s_k1_FDX_s052 <= 0.2
 0 <= s_k1_FDX_s053 <= 0.2
 0 <= s_k1_FDX_s054 <= 0.2
 0 <= s_k1_FDX_s055 <= 0.2
 0 <= s_k1_FDX_s056 <= 0.2
 0 <= s_k1_FDX_s057 <= 0.2
 0 <= s_k1_FDX_s060 <= 0.2
 0 <= s_k1_FDX_s061 <= 0.2
 0 <= s_k1_FDX_s062 <= 0.2
 0 <= s_k1_FDX_s063 <= 0.2
 0 <= s_k1_FDX_s064 <= 0.2
 0 <= s_k1_FDX_s065 <= 0.2
 0 <= s_k1_FDX_s066 <= 0.2
 0 <= s_k1_FDX_s067 <= 0.2
 0 <= s_k1_FDX_s070 <= 0.2
 0 <= s_k1_FDX_s071 <= 0.2
 0 <= s_k1_FDX_s072 <= 0.2
 0 <= s_k1_FDX_s073 <= 0.2
 0 <= s_k1_FDX_s074 <= 0.2
 0 <= s_k1_FDX_s075 <= 0.2
 0 <= s_k1_FDX_s076 <= 0.2
 0 <= s_k1_FDX_s077 <= 0.2
 0 <= s_k1_FDX_s100 <= 0.2
 0 <= s_k1_FDX_s101 <= 0.2
 0 <= s_k1_FDX_s102 <= 0.2
 0 <= s_k1_FDX_s103 <= 0.2
 0 <= s_k1_FDX_s104 <= 0.2
 0 <= s_k1_FDX_s105 <= 0.2
 0 <= s_k1_FDX_s106 <= 0.2
 0 <= s_k1_FDX_s107 <= 0.2
 0 <= s_k1_FDX_s110 <= 0.2
 0 <= s_k1_FDX_s111 <= 0.2
 0 <= s_k1_FDX_s112 <= 0.2
 0 <= s_k1_FDX_s113 <= 0.2
 0 <= s_k1_FDX_s114 <= 0.2
 0 <= s_k1_FDX_s115 <= 0.2
 0 <= s_k1_FDX_s116 <= 0.2
 0 <= s_k1_FDX_s117 <= 0.2
 0 <= s_k1_FDX_s120 <= 0.2
 0 <= s_k1_FDX_s121 <= 0.2
 0 <= s_k1_FDX_s122 <= 0.2
 0 <= s_k1_FDX_s123 <= 0.2
 0 <= s_k1_FDX_s124 <= 0.2
 0 <= s_k1_FDX_s125 <= 0.2
 0 <= s_k1_FDX_s126 <= 0.2
 0 <= s_k1_FDX_s127 <= 0.2
 0 <= s_k1_FDX_s130 <= 0.2
 0 <= s_k1_FDX_s131 <= 0.2
 0 <= s_k1_FDX_s132 <= 0.2
 0 <= s_k1_FDX_s133 <= 0.2
 0 <= s_k1_FDX_s134 <= 0.2
 0 <= s_k1_FDX_s135 <= 0.2
 0 <= s_k1_FDX_s136 <= 0.2
 0 <= s_k1_FDX_s137 <= 0.2
 0 <= s_k1_FDX_s140 <= 0.2
 0 <= s_k1_FDX_s141 <= 0.2
 0 <= s_k1_FDX_s142 <= 0.2
 0 <= s_k1_FDX_s143 <= 0.2
 0 <= s_k1_FDX_s144 <= 0.2
 0 <= s_k1_FDX_s145 <= 0.2
 0 <= s_k1_FDX_s146 <= 0.2
 0 <= s_k1_FDX_s147 <= 0.2
 0 <= s_k1_FDX_s150 <= 0.2
 0 <= s_k1_FDX_s151 <= 0.2
 0 <= s_k1_FDX_s152 <= 0.2
 0 <= s_k1_FDX_s153 <= 0.2
 0 <= s_k1_FDX_s154 <= 0.2
 0 <= s_k1_FDX_s155 <= 0.2
 0 <= s_k1_FDX_s156 <= 0.2
 0 <= s_k1_FDX_s157 <= 0.2
 0 <= s_k1_FDX_s160 <= 0.2
 0 <= s_k1_FDX_s161 <= 0.2
 0 <= s_k1_FDX_s162 <= 0.2
 0 <= s_k1_FDX_s163 <= 0.2
 0 <= s_k1_FDX_s164 <= 0.2
 0 <= s_k1_FDX_s165 <= 0.2
 0 <= s_k1_FDX_s166 <= 0.2
 0 <= s_k1_FDX_s167 <= 0.2
 0 <= s_k1_FDX_s170 <= 0.2
 0 <= s_k1_FDX_s171 <= 0.2
 0 <= s_k1_FDX_s172 <= 0.2
 0 <= s_k1_FDX_s173 <= 0.2
 0 <= s_k1_FDX_s174 <= 0.2
 0 <= s_k1_FDX_s175 <= 0.2
 0 <= s_k1_FDX_s176 <= 0.2
 0 <= s_k1_FDX_s177 <= 0.2
 0 <= s_k1_IBM_s0 <= 0.2
 0 <= s_k1_IBM_s1 <= 0.2
 0 <= s_k1_IBM_s00 <= 0.2
 0 <= s_k1_IBM_s01 <= 0.2
 0 <= s_k1_IBM_s02 <= 0.2
 0 <= s_k1_IBM_s03 <= 0.2
 0 <= s_k1_IBM_s04 <= 0.2
 0 <= s_k1_IBM_s05 <= 0.2
 0 <= s_k1_IBM_s06 <= 0.2
 0 <= s_k1_IBM_s07 <= 0.2
 0 <= s_k1_IBM_s10 <= 0.2
 0 <= s_k1_IBM_s11 <= 0.2
 0 <= s_k1_IBM_s12 <= 0.2
 0 <= s_k1_IBM_s13 <= 0.2
 0 <= s_k1_IBM_s14 <= 0.2
 0 <= s_k1_IBM_s15 <= 0.2
 0 <= s_k1_IBM_s16 <= 0.2
 0 <= s_k1_IBM_s17 <= 0.2
 0 <= s_k1_IBM_s000 <= 0.2
 0 <= s_k1_IBM_s001 <= 0.2
 0 <= s_k1_IBM_s002 <= 0.2
 0 <= s_k1_IBM_s003 <= 0.2
 0 <= s_k1_IBM_s004 <= 0.2
 0 <= s_k1_IBM_s005 <= 0.2
 0 <= s_k1_IBM_s006 <= 0.2
 0 <= s_k1_IBM_s007 <= 0.2
 0 <= s_k1_IBM_s010 <= 0.2
 0 <= s_k1_IBM_s011 <= 0.2
 0 <= s_k1_IBM_s012 <= 0.2
 0 <= s_k1_IBM_s013 <= 0.2
 0 <= s_k1_IBM_s014 <= 0.2
 0 <= s_k1_IBM_s015 <= 0.2
 0 <= s_k1_IBM_s016 <= 0.2
 0 <= s_k1_IBM_s017 <= 0.2
 0 <= s_k1_IBM_s020 <= 0.2
 0 <= s_k1_IBM_s021 <= 0.2
 0 <= s_k1_IBM_s022 <= 0.2
 0 <= s_k1_IBM_s023 <= 0.2
 0 <= s_k1_IBM_s024 <= 0.2
 0 <= s_k1_IBM_s025 <= 0.2
 0 <= s_k1_IBM_s026 <= 0.2
 0 <= s_k1_IBM_s027 <= 0.2
 0 <= s_k1_IBM_s030 <= 0.2
 0 <= s_k1_IBM_s031 <= 0.2
 0 <= s_k1_IBM_s032 <= 0.2
 0 <= s_k1_IBM_s033 <= 0.2
 0 <= s_k1_IBM_s034 <= 0.2
 0 <= s_k1_IBM_s035 <= 0.2
 0 <= s_k1_IBM_s036 <= 0.2
 0 <= s_k1_IBM_s037 <= 0.2
 0 <= s_k1_IBM_s040 <= 0.2
 0 <= s_k1_IBM_s041 <= 0.2
 0 <= s_k1_IBM_s042 <= 0.2
 0 <= s_k1_IBM_s043 <= 0.2
 0 <= s_k1_IBM_s044 <= 0.2
 0 <= s_k1_IBM_s045 <= 0.2
 0 <= s_k1_IBM_s046 <= 0.2
 0 <= s_k1_IBM_s047 <= 0.2
 0 <= s_k1_IBM_s050 <= 0.2
 0 <= s_k1_IBM_s051 <= 0.2
 0 <= s_k1_IBM_s052 <= 0.2
 0 <= s_k1_IBM_s053 <= 0.2
 0 <= s_k1_IBM_s054 <= 0.2
 0 <= s_k1_IBM_s055 <= 0.2
 0 <= s_k1_IBM_s056 <= 0.2
 0 <= s_k1_IBM_s057 <= 0.2
 0 <= s_k1_IBM_s060 <= 0.2
 0 <= s_k1_IBM_s061 <= 0.2
 0 <= s_k1_IBM_s062 <= 0.2
 0 <= s_k1_IBM_s063 <= 0.2
 0 <= s_k1_IBM_s064 <= 0.2
 0 <= s_k1_IBM_s065 <= 0.2
 0 <= s_k1_IBM_s066 <= 0.2
 0 <= s_k1_IBM_s067 <= 0.2
 0 <= s_k1_IBM_s070 <= 0.2
 0 <= s_k1_IBM_s071 <= 0.2
 0 <= s_k1_IBM_s072 <= 0.2
 0 <= s_k1_IBM_s073 <= 0.2
 0 <= s_k1_IBM_s074 <= 0.2
 0 <= s_k1_IBM_s075 <= 0.2
 0 <= s_k1_IBM_s076 <= 0.2
 0 <= s_k1_IBM_s077 <= 0.2
 0 <= s_k1_IBM_s100 <= 0.2
 0 <= s_k1_IBM_s101 <= 0.2
 0 <= s_k1_IBM_s102 <= 0.2
 0 <= s_k1_IBM_s103 <= 0.2
 0 <= s_k1_IBM_s104 <= 0.2
 0 <= s_k1_IBM_s105 <= 0.2
 0 <= s_k1_IBM_s106 <= 0.2
 0 <= s_k1_IBM_s107 <= 0.2
 0 <= s_k1_IBM_s110 <= 0.2
 0 <= s_k1_IBM_s111 <= 0.2
 0 <= s_k1_IBM_s112 <= 0.2
 0 <= s_k1_IBM_s113 <= 0.2
 0 <= s_k1_IBM_s114 <= 0.2
 0 <= s_k1_IBM_s115 <= 0.2
 0 <= s_k1_IBM_s116 <= 0.2
 0 <= s_k1_IBM_s117 <= 0.2
 0 <= s_k1_IBM_s120 <= 0.2
 0 <= s_k1_IBM_s121 <= 0.2
 0 <= s_k1_IBM_s122 <= 0.2
 0 <= s_k1_IBM_s123 <= 0.2
 0 <= s_k1_IBM_s124 <= 0.2
 0 <= s_k1_IBM_s125 <= 0.2
 0 <= s_k1_IBM_s126 <= 0.2
 0 <= s_k1_IBM_s127 <= 0.2
 0 <= s_k1_IBM_s130 <= 0.2
 0 <= s_k1_IBM_s131 <= 0.2
 0 <= s_k1_IBM_s132 <= 0.2
 0 <= s_k1_IBM_s133 <= 0.2
 0 <= s_k1_IBM_s134 <= 0.2
 0 <= s_k1_IBM_s135 <= 0.2
 0 <= s_k1_IBM_s136 <= 0.2
 0 <= s_k1_IBM_s137 <= 0.2
 0 <= s_k1_IBM_s140 <= 0.2
 0 <= s_k1_IBM_s141 <= 0.2
 0 <= s_k1_IBM_s142 <= 0.2
 0 <= s_k1_IBM_s143 <= 0.2
 0 <= s_k1_IBM_s144 <= 0.2
 0 <= s_k1_IBM_s145 <= 0.2
 0 <= s_k1_IBM_s146 <= 0.2
 0 <= s_k1_IBM_s147 <= 0.2
 0 <= s_k1_IBM_s150 <= 0.2
 0 <= s_k1_IBM_s151 <= 0.2
 0 <= s_k1_IBM_s152 <= 0.2
 0 <= s_k1_IBM_s153 <= 0.2
 0 <= s_k1_IBM_s154 <= 0.2
 0 <= s_k1_IBM_s155 <= 0.2
 0 <= s_k1_IBM_s156 <= 0.2
 0 <= s_k1_IBM_s157 <= 0.2
 0 <= s_k1_IBM_s160 <= 0.2
 0 <= s_k1_IBM_s161 <= 0.2
 0 <= s_k1_IBM_s162 <= 0.2
 0 <= s_k1_IBM_s163 <= 0.2
 0 <= s_k1_IBM_s164 <= 0.2
 0 <= s_k1_IBM_s165 <= 0.2
 0 <= s_k1_IBM_s166 <= 0.2
 0 <= s_k1_IBM_s167 <= 0.2
 0 <= s_k1_IBM_s170 <= 0.2
 0 <= s_k1_IBM_s171 <= 0.2
 0 <= s_k1_IBM_s172 <= 0.2
 0 <= s_k1_IBM_s173 <= 0.2
 0 <= s_k1_IBM_s174 <= 0.2
 0 <= s_k1_IBM_s175 <= 0.2
 0 <= s_k1_IBM_s176 <= 0.2
 0 <= s_k1_IBM_s177 <= 0.2
 0 <= s_k1_INTC_s0 <= 0.2
 0 <= s_k1_INTC_s1 <= 0.2
 0 <= s_k1_INTC_s00 <= 0.2
 0 <= s_k1_INTC_s01 <= 0.2
 0 <= s_k1_INTC_s02 <= 0.2
 0 <= s_k1_INTC_s03 <= 0.2
 0 <= s_k1_INTC_s04 <= 0.2
 0 <= s_k1_INTC_s05 <= 0.2
 0 <= s_k1_INTC_s06 <= 0.2
 0 <= s_k1_INTC_s07 <= 0.2
 0 <= s_k1_INTC_s10 <= 0.2
 0 <= s_k1_INTC_s11 <= 0.2
 0 <= s_k1_INTC_s12 <= 0.2
 0 <= s_k1_INTC_s13 <= 0.2
 0 <= s_k1_INTC_s14 <= 0.2
 0 <= s_k1_INTC_s15 <= 0.2
 0 <= s_k1_INTC_s16 <= 0.2
 0 <= s_k1_INTC_s17 <= 0.2
 0 <= s_k1_INTC_s000 <= 0.2
 0 <= s_k1_INTC_s001 <= 0.2
 0 <= s_k1_INTC_s002 <= 0.2
 0 <= s_k1_INTC_s003 <= 0.2
 0 <= s_k1_INTC_s004 <= 0.2
 0 <= s_k1_INTC_s005 <= 0.2
 0 <= s_k1_INTC_s006 <= 0.2
 0 <= s_k1_INTC_s007 <= 0.2
 0 <= s_k1_INTC_s010 <= 0.2
 0 <= s_k1_INTC_s011 <= 0.2
 0 <= s_k1_INTC_s012 <= 0.2
 0 <= s_k1_INTC_s013 <= 0.2
 0 <= s_k1_INTC_s014 <= 0.2
 0 <= s_k1_INTC_s015 <= 0.2
 0 <= s_k1_INTC_s016 <= 0.2
 0 <= s_k1_INTC_s017 <= 0.2
 0 <= s_k1_INTC_s020 <= 0.2
 0 <= s_k1_INTC_s021 <= 0.2
 0 <= s_k1_INTC_s022 <= 0.2
 0 <= s_k1_INTC_s023 <= 0.2
 0 <= s_k1_INTC_s024 <= 0.2
 0 <= s_k1_INTC_s025 <= 0.2
 0 <= s_k1_INTC_s026 <= 0.2
 0 <= s_k1_INTC_s027 <= 0.2
 0 <= s_k1_INTC_s030 <= 0.2
 0 <= s_k1_INTC_s031 <= 0.2
 0 <= s_k1_INTC_s032 <= 0.2
 0 <= s_k1_INTC_s033 <= 0.2
 0 <= s_k1_INTC_s034 <= 0.2
 0 <= s_k1_INTC_s035 <= 0.2
 0 <= s_k1_INTC_s036 <= 0.2
 0 <= s_k1_INTC_s037 <= 0.2
 0 <= s_k1_INTC_s040 <= 0.2
 0 <= s_k1_INTC_s041 <= 0.2
 0 <= s_k1_INTC_s042 <= 0.2
 0 <= s_k1_INTC_s043 <= 0.2
 0 <= s_k1_INTC_s044 <= 0.2
 0 <= s_k1_INTC_s045 <= 0.2
 0 <= s_k1_INTC_s046 <= 0.2
 0 <= s_k1_INTC_s047 <= 0.2
 0 <= s_k1_INTC_s050 <= 0.2
 0 <= s_k1_INTC_s051 <= 0.2
 0 <= s_k1_INTC_s052 <= 0.2
 0 <= s_k1_INTC_s053 <= 0.2
 0 <= s_k1_INTC_s054 <= 0.2
 0 <= s_k1_INTC_s055 <= 0.2
 0 <= s_k1_INTC_s056 <= 0.2
 0 <= s_k1_INTC_s057 <= 0.2
 0 <= s_k1_INTC_s060 <= 0.2
 0 <= s_k1_INTC_s061 <= 0.2
 0 <= s_k1_INTC_s062 <= 0.2
 0 <= s_k1_INTC_s063 <= 0.2
 0 <= s_k1_INTC_s064 <= 0.2
 0 <= s_k1_INTC_s065 <= 0.2
 0 <= s_k1_INTC_s066 <= 0.2
 0 <= s_k1_INTC_s067 <= 0.2
 0 <= s_k1_INTC_s070 <= 0.2
 0 <= s_k1_INTC_s071 <= 0.2
 0 <= s_k1_INTC_s072 <= 0.2
 0 <= s_k1_INTC_s073 <= 0.2
 0 <= s_k1_INTC_s074 <= 0.2
 0 <= s_k1_INTC_s075 <= 0.2
 0 <= s_k1_INTC_s076 <= 0.2
 0 <= s_k1_INTC_s077 <= 0.2
 0 <= s_k1_INTC_s100 <= 0.2
 0 <= s_k1_INTC_s101 <= 0.2
 0 <= s_k1_INTC_s102 <= 0.2
 0 <= s_k1_INTC_s103 <= 0.2
 0 <= s_k1_INTC_s104 <= 0.2
 0 <= s_k1_INTC_s105 <= 0.2
 0 <= s_k1_INTC_s106 <= 0.2
 0 <= s_k1_INTC_s107 <= 0.2
 0 <= s_k1_INTC_s110 <= 0.2
 0 <= s_k1_INTC_s111 <= 0.2
 0 <= s_k1_INTC_s112 <= 0.2
 0 <= s_k1_INTC_s113 <= 0.2
 0 <= s_k1_INTC_s114 <= 0.2
 0 <= s_k1_INTC_s115 <= 0.2
 0 <= s_k1_INTC_s116 <= 0.2
 0 <= s_k1_INTC_s117 <= 0.2
 0 <= s_k1_INTC_s120 <= 0.2
 0 <= s_k1_INTC_s121 <= 0.2
 0 <= s_k1_INTC_s122 <= 0.2
 0 <= s_k1_INTC_s123 <= 0.2
 0 <= s_k1_INTC_s124 <= 0.2
 0 <= s_k1_INTC_s125 <= 0.2
 0 <= s_k1_INTC_s126 <= 0.2
 0 <= s_k1_INTC_s127 <= 0.2
 0 <= s_k1_INTC_s130 <= 0.2
 0 <= s_k1_INTC_s131 <= 0.2
 0 <= s_k1_INTC_s132 <= 0.2
 0 <= s_k1_INTC_s133 <= 0.2
 0 <= s_k1_INTC_s134 <= 0.2
 0 <= s_k1_INTC_s135 <= 0.2
 0 <= s_k1_INTC_s136 <= 0.2
 0 <= s_k1_INTC_s137 <= 0.2
 0 <= s_k1_INTC_s140 <= 0.2
 0 <= s_k1_INTC_s141 <= 0.2
 0 <= s_k1_INTC_s142 <= 0.2
 0 <= s_k1_INTC_s143 <= 0.2
 0 <= s_k1_INTC_s144 <= 0.2
 0 <= s_k1_INTC_s145 <= 0.2
 0 <= s_k1_INTC_s146 <= 0.2
 0 <= s_k1_INTC_s147 <= 0.2
 0 <= s_k1_INTC_s150 <= 0.2
 0 <= s_k1_INTC_s151 <= 0.2
 0 <= s_k1_INTC_s152 <= 0.2
 0 <= s_k1_INTC_s153 <= 0.2
 0 <= s_k1_INTC_s154 <= 0.2
 0 <= s_k1_INTC_s155 <= 0.2
 0 <= s_k1_INTC_s156 <= 0.2
 0 <= s_k1_INTC_s157 <= 0.2
 0 <= s_k1_INTC_s160 <= 0.2
 0 <= s_k1_INTC_s161 <= 0.2
 0 <= s_k1_INTC_s162 <= 0.2
 0 <= s_k1_INTC_s163 <= 0.2
 0 <= s_k1_INTC_s164 <= 0.2
 0 <= s_k1_INTC_s165 <= 0.2
 0 <= s_k1_INTC_s166 <= 0.2
 0 <= s_k1_INTC_s167 <= 0.2
 0 <= s_k1_INTC_s170 <= 0.2
 0 <= s_k1_INTC_s171 <= 0.2
 0 <= s_k1_INTC_s172 <= 0.2
 0 <= s_k1_INTC_s173 <= 0.2
 0 <= s_k1_INTC_s174 <= 0.2
 0 <= s_k1_INTC_s175 <= 0.2
 0 <= s_k1_INTC_s176 <= 0.2
 0 <= s_k1_INTC_s177 <= 0.2
 0 <= s_k1_KO_s0 <= 0.2
 0 <= s_k1_KO_s1 <= 0.2
 0 <= s_k1_KO_s00 <= 0.2
 0 <= s_k1_KO_s01 <= 0.2
 0 <= s_k1_KO_s02 <= 0.2
 0 <= s_k1_KO_s03 <= 0.2
 0 <= s_k1_KO_s04 <= 0.2
 0 <= s_k1_KO_s05 <= 0.2
 0 <= s_k1_KO_s06 <= 0.2
 0 <= s_k1_KO_s07 <= 0.2
 0 <= s_k1_KO_s10 <= 0.2
 0 <= s_k1_KO_s11 <= 0.2
 0 <= s_k1_KO_s12 <= 0.2
 0 <= s_k1_KO_s13 <= 0.2
 0 <= s_k1_KO_s14 <= 0.2
 0 <= s_k1_KO_s15 <= 0.2
 0 <= s_k1_KO_s16 <= 0.2
 0 <= s_k1_KO_s17 <= 0.2
 0 <= s_k1_KO_s000 <= 0.2
 0 <= s_k1_KO_s001 <= 0.2
 0 <= s_k1_KO_s002 <= 0.2
 0 <= s_k1_KO_s003 <= 0.2
 0 <= s_k1_KO_s004 <= 0.2
 0 <= s_k1_KO_s005 <= 0.2
 0 <= s_k1_KO_s006 <= 0.2
 0 <= s_k1_KO_s007 <= 0.2
 0 <= s_k1_KO_s010 <= 0.2
 0 <= s_k1_KO_s011 <= 0.2
 0 <= s_k1_KO_s012 <= 0.2
 0 <= s_k1_KO_s013 <= 0.2
 0 <= s_k1_KO_s014 <= 0.2
 0 <= s_k1_KO_s015 <= 0.2
 0 <= s_k1_KO_s016 <= 0.2
 0 <= s_k1_KO_s017 <= 0.2
 0 <= s_k1_KO_s020 <= 0.2
 0 <= s_k1_KO_s021 <= 0.2
 0 <= s_k1_KO_s022 <= 0.2
 0 <= s_k1_KO_s023 <= 0.2
 0 <= s_k1_KO_s024 <= 0.2
 0 <= s_k1_KO_s025 <= 0.2
 0 <= s_k1_KO_s026 <= 0.2
 0 <= s_k1_KO_s027 <= 0.2
 0 <= s_k1_KO_s030 <= 0.2
 0 <= s_k1_KO_s031 <= 0.2
 0 <= s_k1_KO_s032 <= 0.2
 0 <= s_k1_KO_s033 <= 0.2
 0 <= s_k1_KO_s034 <= 0.2
 0 <= s_k1_KO_s035 <= 0.2
 0 <= s_k1_KO_s036 <= 0.2
 0 <= s_k1_KO_s037 <= 0.2
 0 <= s_k1_KO_s040 <= 0.2
 0 <= s_k1_KO_s041 <= 0.2
 0 <= s_k1_KO_s042 <= 0.2
 0 <= s_k1_KO_s043 <= 0.2
 0 <= s_k1_KO_s044 <= 0.2
 0 <= s_k1_KO_s045 <= 0.2
 0 <= s_k1_KO_s046 <= 0.2
 0 <= s_k1_KO_s047 <= 0.2
 0 <= s_k1_KO_s050 <= 0.2
 0 <= s_k1_KO_s051 <= 0.2
 0 <= s_k1_KO_s052 <= 0.2
 0 <= s_k1_KO_s053 <= 0.2
 0 <= s_k1_KO_s054 <= 0.2
 0 <= s_k1_KO_s055 <= 0.2
 0 <= s_k1_KO_s056 <= 0.2
 0 <= s_k1_KO_s057 <= 0.2
 0 <= s_k1_KO_s060 <= 0.2
 0 <= s_k1_KO_s061 <= 0.2
 0 <= s_k1_KO_s062 <= 0.2
 0 <= s_k1_KO_s063 <= 0.2
 0 <= s_k1_KO_s064 <= 0.2
 0 <= s_k1_KO_s065 <= 0.2
 0 <= s_k1_KO_s066 <= 0.2
 0 <= s_k1_KO_s067 <= 0.2
 0 <= s_k1_KO_s070 <= 0.2
 0 <= s_k1_KO_s071 <= 0.2
 0 <= s_k1_KO_s072 <= 0.2
 0 <= s_k1_KO_s073 <= 0.2
 0 <= s_k1_KO_s074 <= 0.2
 0 <= s_k1_KO_s075 <= 0.2
 0 <= s_k1_KO_s076 <= 0.2
 0 <= s_k1_KO_s077 <= 0.2
 0 <= s_k1_KO_s100 <= 0.2
 0 <= s_k1_KO_s101 <= 0.2
 0 <= s_k1_KO_s102 <= 0.2
 0 <= s_k1_KO_s103 <= 0.2
 0 <= s_k1_KO_s104 <= 0.2
 0 <= s_k1_KO_s105 <= 0.2
 0 <= s_k1_KO_s106 <= 0.2
 0 <= s_k1_KO_s107 <= 0.2
 0 <= s_k1_KO_s110 <= 0.2
 0 <= s_k1_KO_s111 <= 0.2
 0 <= s_k1_KO_s112 <= 0.2
 0 <= s_k1_KO_s113 <= 0.2
 0 <= s_k1_KO_s114 <= 0.2
 0 <= s_k1_KO_s115 <= 0.2
 0 <= s_k1_KO_s116 <= 0.2
 0 <= s_k1_KO_s117 <= 0.2
 0 <= s_k1_KO_s120 <= 0.2
 0 <= s_k1_KO_s121 <= 0.2
 0 <= s_k1_KO_s122 <= 0.2
 0 <= s_k1_KO_s123 <= 0.2
 0 <= s_k1_KO_s124 <= 0.2
 0 <= s_k1_KO_s125 <= 0.2
 0 <= s_k1_KO_s126 <= 0.2
 0 <= s_k1_KO_s127 <= 0.2
 0 <= s_k1_KO_s130 <= 0.2
 0 <= s_k1_KO_s131 <= 0.2
 0 <= s_k1_KO_s132 <= 0.2
 0 <= s_k1_KO_s133 <= 0.2
 0 <= s_k1_KO_s134 <= 0.2
 0 <= s_k1_KO_s135 <= 0.2
 0 <= s_k1_KO_s136 <= 0.2
 0 <= s_k1_KO_s137 <= 0.2
 0 <= s_k1_KO_s140 <= 0.2
 0 <= s_k1_KO_s141 <= 0.2
 0 <= s_k1_KO_s142 <= 0.2
 0 <= s_k1_KO_s143 <= 0.2
 0 <= s_k1_KO_s144 <= 0.2
 0 <= s_k1_KO_s145 <= 0.2
 0 <= s_k1_KO_s146 <= 0.2
 0 <= s_k1_KO_s147 <= 0.2
 0 <= s_k1_KO_s150 <= 0.2
 0 <= s_k1_KO_s151 <= 0.2
 0 <= s_k1_KO_s152 <= 0.2
 0 <= s_k1_KO_s153 <= 0.2
 0 <= s_k1_KO_s154 <= 0.2
 0 <= s_k1_KO_s155 <= 0.2
 0 <= s_k1_KO_s156 <= 0.2
 0 <= s_k1_KO_s157 <= 0.2
 0 <= s_k1_KO_s160 <= 0.2
 0 <= s_k1_KO_s161 <= 0.2
 0 <= s_k1_KO_s162 <= 0.2
 0 <= s_k1_KO_s163 <= 0.2
 0 <= s_k1_KO_s164 <= 0.2
 0 <= s_k1_KO_s165 <= 0.2
 0 <= s_k1_KO_s166 <= 0.2
 0 <= s_k1_KO_s167 <= 0.2
 0 <= s_k1_KO_s170 <= 0.2
 0 <= s_k1_KO_s171 <= 0.2
 0 <= s_k1_KO_s172 <= 0.2
 0 <= s_k1_KO_s173 <= 0.2
 0 <= s_k1_KO_s174 <= 0.2
 0 <= s_k1_KO_s175 <= 0.2
 0 <= s_k1_KO_s176 <= 0.2
 0 <= s_k1_KO_s177 <= 0.2
 0 <= s_k1_MAT_s0 <= 0.2
 0 <= s_k1_MAT_s1 <= 0.2
 0 <= s_k1_MAT_s00 <= 0.2
 0 <= s_k1_MAT_s01 <= 0.2
 0 <= s_k1_MAT_s02 <= 0.2
 0 <= s_k1_MAT_s03 <= 0.2
 0 <= s_k1_MAT_s04 <= 0.2
 0 <= s_k1_MAT_s05 <= 0.2
 0 <= s_k1_MAT_s06 <= 0.2
 0 <= s_k1_MAT_s07 <= 0.2
 0 <= s_k1_MAT_s10 <= 0.2
 0 <= s_k1_MAT_s11 <= 0.2
 0 <= s_k1_MAT_s12 <= 0.2
 0 <= s_k1_MAT_s13 <= 0.2
 0 <= s_k1_MAT_s14 <= 0.2
 0 <= s_k1_MAT_s15 <= 0.2
 0 <= s_k1_MAT_s16 <= 0.2
 0 <= s_k1_MAT_s17 <= 0.2
 0 <= s_k1_MAT_s000 <= 0.2
 0 <= s_k1_MAT_s001 <= 0.2
 0 <= s_k1_MAT_s002 <= 0.2
 0 <= s_k1_MAT_s003 <= 0.2
 0 <= s_k1_MAT_s004 <= 0.2
 0 <= s_k1_MAT_s005 <= 0.2
 0 <= s_k1_MAT_s006 <= 0.2
 0 <= s_k1_MAT_s007 <= 0.2
 0 <= s_k1_MAT_s010 <= 0.2
 0 <= s_k1_MAT_s011 <= 0.2
 0 <= s_k1_MAT_s012 <= 0.2
 0 <= s_k1_MAT_s013 <= 0.2
 0 <= s_k1_MAT_s014 <= 0.2
 0 <= s_k1_MAT_s015 <= 0.2
 0 <= s_k1_MAT_s016 <= 0.2
 0 <= s_k1_MAT_s017 <= 0.2
 0 <= s_k1_MAT_s020 <= 0.2
 0 <= s_k1_MAT_s021 <= 0.2
 0 <= s_k1_MAT_s022 <= 0.2
 0 <= s_k1_MAT_s023 <= 0.2
 0 <= s_k1_MAT_s024 <= 0.2
 0 <= s_k1_MAT_s025 <= 0.2
 0 <= s_k1_MAT_s026 <= 0.2
 0 <= s_k1_MAT_s027 <= 0.2
 0 <= s_k1_MAT_s030 <= 0.2
 0 <= s_k1_MAT_s031 <= 0.2
 0 <= s_k1_MAT_s032 <= 0.2
 0 <= s_k1_MAT_s033 <= 0.2
 0 <= s_k1_MAT_s034 <= 0.2
 0 <= s_k1_MAT_s035 <= 0.2
 0 <= s_k1_MAT_s036 <= 0.2
 0 <= s_k1_MAT_s037 <= 0.2
 0 <= s_k1_MAT_s040 <= 0.2
 0 <= s_k1_MAT_s041 <= 0.2
 0 <= s_k1_MAT_s042 <= 0.2
 0 <= s_k1_MAT_s043 <= 0.2
 0 <= s_k1_MAT_s044 <= 0.2
 0 <= s_k1_MAT_s045 <= 0.2
 0 <= s_k1_MAT_s046 <= 0.2
 0 <= s_k1_MAT_s047 <= 0.2
 0 <= s_k1_MAT_s050 <= 0.2
 0 <= s_k1_MAT_s051 <= 0.2
 0 <= s_k1_MAT_s052 <= 0.2
 0 <= s_k1_MAT_s053 <= 0.2
 0 <= s_k1_MAT_s054 <= 0.2
 0 <= s_k1_MAT_s055 <= 0.2
 0 <= s_k1_MAT_s056 <= 0.2
 0 <= s_k1_MAT_s057 <= 0.2
 0 <= s_k1_MAT_s060 <= 0.2
 0 <= s_k1_MAT_s061 <= 0.2
 0 <= s_k1_MAT_s062 <= 0.2
 0 <= s_k1_MAT_s063 <= 0.2
 0 <= s_k1_MAT_s064 <= 0.2
 0 <= s_k1_MAT_s065 <= 0.2
 0 <= s_k1_MAT_s066 <= 0.2
 0 <= s_k1_MAT_s067 <= 0.2
 0 <= s_k1_MAT_s070 <= 0.2
 0 <= s_k1_MAT_s071 <= 0.2
 0 <= s_k1_MAT_s072 <= 0.2
 0 <= s_k1_MAT_s073 <= 0.2
 0 <= s_k1_MAT_s074 <= 0.2
 0 <= s_k1_MAT_s075 <= 0.2
 0 <= s_k1_MAT_s076 <= 0.2
 0 <= s_k1_MAT_s077 <= 0.2
 0 <= s_k1_MAT_s100 <= 0.2
 0 <= s_k1_MAT_s101 <= 0.2
 0 <= s_k1_MAT_s102 <= 0.2
 0 <= s_k1_MAT_s103 <= 0.2
 0 <= s_k1_MAT_s104 <= 0.2
 0 <= s_k1_MAT_s105 <= 0.2
 0 <= s_k1_MAT_s106 <= 0.2
 0 <= s_k1_MAT_s107 <= 0.2
 0 <= s_k1_MAT_s110 <= 0.2
 0 <= s_k1_MAT_s111 <= 0.2
 0 <= s_k1_MAT_s112 <= 0.2
 0 <= s_k1_MAT_s113 <= 0.2
 0 <= s_k1_MAT_s114 <= 0.2
 0 <= s_k1_MAT_s115 <= 0.2
 0 <= s_k1_MAT_s116 <= 0.2
 0 <= s_k1_MAT_s117 <= 0.2
 0 <= s_k1_MAT_s120 <= 0.2
 0 <= s_k1_MAT_s121 <= 0.2
 0 <= s_k1_MAT_s122 <= 0.2
 0 <= s_k1_MAT_s123 <= 0.2
 0 <= s_k1_MAT_s124 <= 0.2
 0 <= s_k1_MAT_s125 <= 0.2
 0 <= s_k1_MAT_s126 <= 0.2
 0 <= s_k1_MAT_s127 <= 0.2
 0 <= s_k1_MAT_s130 <= 0.2
 0 <= s_k1_MAT_s131 <= 0.2
 0 <= s_k1_MAT_s132 <= 0.2
 0 <= s_k1_MAT_s133 <= 0.2
 0 <= s_k1_MAT_s134 <= 0.2
 0 <= s_k1_MAT_s135 <= 0.2
 0 <= s_k1_MAT_s136 <= 0.2
 0 <= s_k1_MAT_s137 <= 0.2
 0 <= s_k1_MAT_s140 <= 0.2
 0 <= s_k1_MAT_s141 <= 0.2
 0 <= s_k1_MAT_s142 <= 0.2
 0 <= s_k1_MAT_s143 <= 0.2
 0 <= s_k1_MAT_s144 <= 0.2
 0 <= s_k1_MAT_s145 <= 0.2
 0 <= s_k1_MAT_s146 <= 0.2
 0 <= s_k1_MAT_s147 <= 0.2
 0 <= s_k1_MAT_s150 <= 0.2
 0 <= s_k1_MAT_s151 <= 0.2
 0 <= s_k1_MAT_s152 <= 0.2
 0 <= s_k1_MAT_s153 <= 0.2
 0 <= s_k1_MAT_s154 <= 0.2
 0 <= s_k1_MAT_s155 <= 0.2
 0 <= s_k1_MAT_s156 <= 0.2
 0 <= s_k1_MAT_s157 <= 0.2
 0 <= s_k1_MAT_s160 <= 0.2
 0 <= s_k1_MAT_s161 <= 0.2
 0 <= s_k1_MAT_s162 <= 0.2
 0 <= s_k1_MAT_s163 <= 0.2
 0 <= s_k1_MAT_s164 <= 0.2
 0 <= s_k1_MAT_s165 <= 0.2
 0 <= s_k1_MAT_s166 <= 0.2
 0 <= s_k1_MAT_s167 <= 0.2
 0 <= s_k1_MAT_s170 <= 0.2
 0 <= s_k1_MAT_s171 <= 0.2
 0 <= s_k1_MAT_s172 <= 0.2
 0 <= s_k1_MAT_s173 <= 0.2
 0 <= s_k1_MAT_s174 <= 0.2
 0 <= s_k1_MAT_s175 <= 0.2
 0 <= s_k1_MAT_s176 <= 0.2
 0 <= s_k1_MAT_s177 <= 0.2
 0 <= s_k1_MSFT_s0 <= 0.2
 0 <= s_k1_MSFT_s1 <= 0.2
 0 <= s_k1_MSFT_s00 <= 0.2
 0 <= s_k1_MSFT_s01 <= 0.2
 0 <= s_k1_MSFT_s02 <= 0.2
 0 <= s_k1_MSFT_s03 <= 0.2
 0 <= s_k1_MSFT_s04 <= 0.2
 0 <= s_k1_MSFT_s05 <= 0.2
 0 <= s_k1_MSFT_s06 <= 0.2
 0 <= s_k1_MSFT_s07 <= 0.2
 0 <= s_k1_MSFT_s10 <= 0.2
 0 <= s_k1_MSFT_s11 <= 0.2
 0 <= s_k1_MSFT_s12 <= 0.2
 0 <= s_k1_MSFT_s13 <= 0.2
 0 <= s_k1_MSFT_s14 <= 0.2
 0 <= s_k1_MSFT_s15 <= 0.2
 0 <= s_k1_MSFT_s16 <= 0.2
 0 <= s_k1_MSFT_s17 <= 0.2
 0 <= s_k1_MSFT_s000 <= 0.2
 0 <= s_k1_MSFT_s001 <= 0.2
 0 <= s_k1_MSFT_s002 <= 0.2
 0 <= s_k1_MSFT_s003 <= 0.2
 0 <= s_k1_MSFT_s004 <= 0.2
 0 <= s_k1_MSFT_s005 <= 0.2
 0 <= s_k1_MSFT_s006 <= 0.2
 0 <= s_k1_MSFT_s007 <= 0.2
 0 <= s_k1_MSFT_s010 <= 0.2
 0 <= s_k1_MSFT_s011 <= 0.2
 0 <= s_k1_MSFT_s012 <= 0.2
 0 <= s_k1_MSFT_s013 <= 0.2
 0 <= s_k1_MSFT_s014 <= 0.2
 0 <= s_k1_MSFT_s015 <= 0.2
 0 <= s_k1_MSFT_s016 <= 0.2
 0 <= s_k1_MSFT_s017 <= 0.2
 0 <= s_k1_MSFT_s020 <= 0.2
 0 <= s_k1_MSFT_s021 <= 0.2
 0 <= s_k1_MSFT_s022 <= 0.2
 0 <= s_k1_MSFT_s023 <= 0.2
 0 <= s_k1_MSFT_s024 <= 0.2
 0 <= s_k1_MSFT_s025 <= 0.2
 0 <= s_k1_MSFT_s026 <= 0.2
 0 <= s_k1_MSFT_s027 <= 0.2
 0 <= s_k1_MSFT_s030 <= 0.2
 0 <= s_k1_MSFT_s031 <= 0.2
 0 <= s_k1_MSFT_s032 <= 0.2
 0 <= s_k1_MSFT_s033 <= 0.2
 0 <= s_k1_MSFT_s034 <= 0.2
 0 <= s_k1_MSFT_s035 <= 0.2
 0 <= s_k1_MSFT_s036 <= 0.2
 0 <= s_k1_MSFT_s037 <= 0.2
 0 <= s_k1_MSFT_s040 <= 0.2
 0 <= s_k1_MSFT_s041 <= 0.2
 0 <= s_k1_MSFT_s042 <= 0.2
 0 <= s_k1_MSFT_s043 <= 0.2
 0 <= s_k1_MSFT_s044 <= 0.2
 0 <= s_k1_MSFT_s045 <= 0.2
 0 <= s_k1_MSFT_s046 <= 0.2
 0 <= s_k1_MSFT_s047 <= 0.2
 0 <= s_k1_MSFT_s050 <= 0.2
 0 <= s_k1_MSFT_s051 <= 0.2
 0 <= s_k1_MSFT_s052 <= 0.2
 0 <= s_k1_MSFT_s053 <= 0.2
 0 <= s_k1_MSFT_s054 <= 0.2
 0 <= s_k1_MSFT_s055 <= 0.2
 0 <= s_k1_MSFT_s056 <= 0.2
 0 <= s_k1_MSFT_s057 <= 0.2
 0 <= s_k1_MSFT_s060 <= 0.2
 0 <= s_k1_MSFT_s061 <= 0.2
 0 <= s_k1_MSFT_s062 <= 0.2
 0 <= s_k1_MSFT_s063 <= 0.2
 0 <= s_k1_MSFT_s064 <= 0.2
 0 <= s_k1_MSFT_s065 <= 0.2
 0 <= s_k1_MSFT_s066 <= 0.2
 0 <= s_k1_MSFT_s067 <= 0.2
 0 <= s_k1_MSFT_s070 <= 0.2
 0 <= s_k1_MSFT_s071 <= 0.2
 0 <= s_k1_MSFT_s072 <= 0.2
 0 <= s_k1_MSFT_s073 <= 0.2
 0 <= s_k1_MSFT_s074 <= 0.2
 0 <= s_k1_MSFT_s075 <= 0.2
 0 <= s_k1_MSFT_s076 <= 0.2
 0 <= s_k1_MSFT_s077 <= 0.2
 0 <= s_k1_MSFT_s100 <= 0.2
 0 <= s_k1_MSFT_s101 <= 0.2
 0 <= s_k1_MSFT_s102 <= 0.2
 0 <= s_k1_MSFT_s103 <= 0.2
 0 <= s_k1_MSFT_s104 <= 0.2
 0 <= s_k1_MSFT_s105 <= 0.2
 0 <= s_k1_MSFT_s106 <= 0.2
 0 <= s_k1_MSFT_s107 <= 0.2
 0 <= s_k1_MSFT_s110 <= 0.2
 0 <= s_k1_MSFT_s111 <= 0.2
 0 <= s_k1_MSFT_s112 <= 0.2
 0 <= s_k1_MSFT_s113 <= 0.2
 0 <= s_k1_MSFT_s114 <= 0.2
 0 <= s_k1_MSFT_s115 <= 0.2
 0 <= s_k1_MSFT_s116 <= 0.2
 0 <= s_k1_MSFT_s117 <= 0.2
 0 <= s_k1_MSFT_s120 <= 0.2
 0 <= s_k1_MSFT_s121 <= 0.2
 0 <= s_k1_MSFT_s122 <= 0.2
 0 <= s_k1_MSFT_s123 <= 0.2
 0 <= s_k1_MSFT_s124 <= 0.2
 0 <= s_k1_MSFT_s125 <= 0.2
 0 <= s_k1_MSFT_s126 <= 0.2
 0 <= s_k1_MSFT_s127 <= 0.2
 0 <= s_k1_MSFT_s130 <= 0.2
 0 <= s_k1_MSFT_s131 <= 0.2
 0 <= s_k1_MSFT_s132 <= 0.2
 0 <= s_k1_MSFT_s133 <= 0.2
 0 <= s_k1_MSFT_s134 <= 0.2
 0 <= s_k1_MSFT_s135 <= 0.2
 0 <= s_k1_MSFT_s136 <= 0.2
 0 <= s_k1_MSFT_s137 <= 0.2
 0 <= s_k1_MSFT_s140 <= 0.2
 0 <= s_k1_MSFT_s141 <= 0.2
 0 <= s_k1_MSFT_s142 <= 0.2
 0 <= s_k1_MSFT_s143 <= 0.2
 0 <= s_k1_MSFT_s144 <= 0.2
 0 <= s_k1_MSFT_s145 <= 0.2
 0 <= s_k1_MSFT_s146 <= 0.2
 0 <= s_k1_MSFT_s147 <= 0.2
 0 <= s_k1_MSFT_s150 <= 0.2
 0 <= s_k1_MSFT_s151 <= 0.2
 0 <= s_k1_MSFT_s152 <= 0.2
 0 <= s_k1_MSFT_s153 <= 0.2
 0 <= s_k1_MSFT_s154 <= 0.2
 0 <= s_k1_MSFT_s155 <= 0.2
 0 <= s_k1_MSFT_s156 <= 0.2
 0 <= s_k1_MSFT_s157 <= 0.2
 0 <= s_k1_MSFT_s160 <= 0.2
 0 <= s_k1_MSFT_s161 <= 0.2
 0 <= s_k1_MSFT_s162 <= 0.2
 0 <= s_k1_MSFT_s163 <= 0.2
 0 <= s_k1_MSFT_s164 <= 0.2
 0 <= s_k1_MSFT_s165 <= 0.2
 0 <= s_k1_MSFT_s166 <= 0.2
 0 <= s_k1_MSFT_s167 <= 0.2
 0 <= s_k1_MSFT_s170 <= 0.2
 0 <= s_k1_MSFT_s171 <= 0.2
 0 <= s_k1_MSFT_s172 <= 0.2
 0 <= s_k1_MSFT_s173 <= 0.2
 0 <= s_k1_MSFT_s174 <= 0.2
 0 <= s_k1_MSFT_s175 <= 0.2
 0 <= s_k1_MSFT_s176 <= 0.2
 0 <= s_k1_MSFT_s177 <= 0.2
 0 <= s_k1_PG_s0 <= 0.2
 0 <= s_k1_PG_s1 <= 0.2
 0 <= s_k1_PG_s00 <= 0.2
 0 <= s_k1_PG_s01 <= 0.2
 0 <= s_k1_PG_s02 <= 0.2
 0 <= s_k1_PG_s03 <= 0.2
 0 <= s_k1_PG_s04 <= 0.2
 0 <= s_k1_PG_s05 <= 0.2
 0 <= s_k1_PG_s06 <= 0.2
 0 <= s_k1_PG_s07 <= 0.2
 0 <= s_k1_PG_s10 <= 0.2
 0 <= s_k1_PG_s11 <= 0.2
 0 <= s_k1_PG_s12 <= 0.2
 0 <= s_k1_PG_s13 <= 0.2
 0 <= s_k1_PG_s14 <= 0.2
 0 <= s_k1_PG_s15 <= 0.2
 0 <= s_k1_PG_s16 <= 0.2
 0 <= s_k1_PG_s17 <= 0.2
 0 <= s_k1_PG_s000 <= 0.2
 0 <= s_k1_PG_s001 <= 0.2
 0 <= s_k1_PG_s002 <= 0.2
 0 <= s_k1_PG_s003 <= 0.2
 0 <= s_k1_PG_s004 <= 0.2
 0 <= s_k1_PG_s005 <= 0.2
 0 <= s_k1_PG_s006 <= 0.2
 0 <= s_k1_PG_s007 <= 0.2
 0 <= s_k1_PG_s010 <= 0.2
 0 <= s_k1_PG_s011 <= 0.2
 0 <= s_k1_PG_s012 <= 0.2
 0 <= s_k1_PG_s013 <= 0.2
 0 <= s_k1_PG_s014 <= 0.2
 0 <= s_k1_PG_s015 <= 0.2
 0 <= s_k1_PG_s016 <= 0.2
 0 <= s_k1_PG_s017 <= 0.2
 0 <= s_k1_PG_s020 <= 0.2
 0 <= s_k1_PG_s021 <= 0.2
 0 <= s_k1_PG_s022 <= 0.2
 0 <= s_k1_PG_s023 <= 0.2
 0 <= s_k1_PG_s024 <= 0.2
 0 <= s_k1_PG_s025 <= 0.2
 0 <= s_k1_PG_s026 <= 0.2
 0 <= s_k1_PG_s027 <= 0.2
 0 <= s_k1_PG_s030 <= 0.2
 0 <= s_k1_PG_s031 <= 0.2
 0 <= s_k1_PG_s032 <= 0.2
 0 <= s_k1_PG_s033 <= 0.2
 0 <= s_k1_PG_s034 <= 0.2
 0 <= s_k1_PG_s035 <= 0.2
 0 <= s_k1_PG_s036 <= 0.2
 0 <= s_k1_PG_s037 <= 0.2
 0 <= s_k1_PG_s040 <= 0.2
 0 <= s_k1_PG_s041 <= 0.2
 0 <= s_k1_PG_s042 <= 0.2
 0 <= s_k1_PG_s043 <= 0.2
 0 <= s_k1_PG_s044 <= 0.2
 0 <= s_k1_PG_s045 <= 0.2
 0 <= s_k1_PG_s046 <= 0.2
 0 <= s_k1_PG_s047 <= 0.2
 0 <= s_k1_PG_s050 <= 0.2
 0 <= s_k1_PG_s051 <= 0.2
 0 <= s_k1_PG_s052 <= 0.2
 0 <= s_k1_PG_s053 <= 0.2
 0 <= s_k1_PG_s054 <= 0.2
 0 <= s_k1_PG_s055 <= 0.2
 0 <= s_k1_PG_s056 <= 0.2
 0 <= s_k1_PG_s057 <= 0.2
 0 <= s_k1_PG_s060 <= 0.2
 0 <= s_k1_PG_s061 <= 0.2
 0 <= s_k1_PG_s062 <= 0.2
 0 <= s_k1_PG_s063 <= 0.2
 0 <= s_k1_PG_s064 <= 0.2
 0 <= s_k1_PG_s065 <= 0.2
 0 <= s_k1_PG_s066 <= 0.2
 0 <= s_k1_PG_s067 <= 0.2
 0 <= s_k1_PG_s070 <= 0.2
 0 <= s_k1_PG_s071 <= 0.2
 0 <= s_k1_PG_s072 <= 0.2
 0 <= s_k1_PG_s073 <= 0.2
 0 <= s_k1_PG_s074 <= 0.2
 0 <= s_k1_PG_s075 <= 0.2
 0 <= s_k1_PG_s076 <= 0.2
 0 <= s_k1_PG_s077 <= 0.2
 0 <= s_k1_PG_s100 <= 0.2
 0 <= s_k1_PG_s101 <= 0.2
 0 <= s_k1_PG_s102 <= 0.2
 0 <= s_k1_PG_s103 <= 0.2
 0 <= s_k1_PG_s104 <= 0.2
 0 <= s_k1_PG_s105 <= 0.2
 0 <= s_k1_PG_s106 <= 0.2
 0 <= s_k1_PG_s107 <= 0.2
 0 <= s_k1_PG_s110 <= 0.2
 0 <= s_k1_PG_s111 <= 0.2
 0 <= s_k1_PG_s112 <= 0.2
 0 <= s_k1_PG_s113 <= 0.2
 0 <= s_k1_PG_s114 <= 0.2
 0 <= s_k1_PG_s115 <= 0.2
 0 <= s_k1_PG_s116 <= 0.2
 0 <= s_k1_PG_s117 <= 0.2
 0 <= s_k1_PG_s120 <= 0.2
 0 <= s_k1_PG_s121 <= 0.2
 0 <= s_k1_PG_s122 <= 0.2
 0 <= s_k1_PG_s123 <= 0.2
 0 <= s_k1_PG_s124 <= 0.2
 0 <= s_k1_PG_s125 <= 0.2
 0 <= s_k1_PG_s126 <= 0.2
 0 <= s_k1_PG_s127 <= 0.2
 0 <= s_k1_PG_s130 <= 0.2
 0 <= s_k1_PG_s131 <= 0.2
 0 <= s_k1_PG_s132 <= 0.2
 0 <= s_k1_PG_s133 <= 0.2
 0 <= s_k1_PG_s134 <= 0.2
 0 <= s_k1_PG_s135 <= 0.2
 0 <= s_k1_PG_s136 <= 0.2
 0 <= s_k1_PG_s137 <= 0.2
 0 <= s_k1_PG_s140 <= 0.2
 0 <= s_k1_PG_s141 <= 0.2
 0 <= s_k1_PG_s142 <= 0.2
 0 <= s_k1_PG_s143 <= 0.2
 0 <= s_k1_PG_s144 <= 0.2
 0 <= s_k1_PG_s145 <= 0.2
 0 <= s_k1_PG_s146 <= 0.2
 0 <= s_k1_PG_s147 <= 0.2
 0 <= s_k1_PG_s150 <= 0.2
 0 <= s_k1_PG_s151 <= 0.2
 0 <= s_k1_PG_s152 <= 0.2
 0 <= s_k1_PG_s153 <= 0.2
 0 <= s_k1_PG_s154 <= 0.2
 0 <= s_k1_PG_s155 <= 0.2
 0 <= s_k1_PG_s156 <= 0.2
 0 <= s_k1_PG_s157 <= 0.2
 0 <= s_k1_PG_s160 <= 0.2
 0 <= s_k1_PG_s161 <= 0.2
 0 <= s_k1_PG_s162 <= 0.2
 0 <= s_k1_PG_s163 <= 0.2
 0 <= s_k1_PG_s164 <= 0.2
 0 <= s_k1_PG_s165 <= 0.2
 0 <= s_k1_PG_s166 <= 0.2
 0 <= s_k1_PG_s167 <= 0.2
 0 <= s_k1_PG_s170 <= 0.2
 0 <= s_k1_PG_s171 <= 0.2
 0 <= s_k1_PG_s172 <= 0.2
 0 <= s_k1_PG_s173 <= 0.2
 0 <= s_k1_PG_s174 <= 0.2
 0 <= s_k1_PG_s175 <= 0.2
 0 <= s_k1_PG_s176 <= 0.2
 0 <= s_k1_PG_s177 <= 0.2
End

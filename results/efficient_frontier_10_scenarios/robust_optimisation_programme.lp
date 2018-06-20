\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 obj: wcvar
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
 w_k10_ADM_s0_return_constraint: 0.93777 w_0_ADM - w_k10_ADM_s0
                                 + 0.99 b_k10_ADM_s0 - 1.01 s_k10_ADM_s0  = 0
 w_k10_ADM_s1_return_constraint: 1.07441 w_0_ADM - w_k10_ADM_s1
                                 + 0.99 b_k10_ADM_s1 - 1.01 s_k10_ADM_s1  = 0
 w_k10_ADM_s00_return_constraint: 0.990211 w_k10_ADM_s0 - w_k10_ADM_s00
                                 + 0.99 b_k10_ADM_s00 - 1.01 s_k10_ADM_s00  = 0
 w_k10_ADM_s01_return_constraint: 0.886403 w_k10_ADM_s0 - w_k10_ADM_s01
                                 + 0.99 b_k10_ADM_s01 - 1.01 s_k10_ADM_s01  = 0
 w_k10_ADM_s02_return_constraint: 1.2651 w_k10_ADM_s0 - w_k10_ADM_s02
                                 + 0.99 b_k10_ADM_s02 - 1.01 s_k10_ADM_s02  = 0
 w_k10_ADM_s03_return_constraint: 1.22314 w_k10_ADM_s0 - w_k10_ADM_s03
                                 + 0.99 b_k10_ADM_s03 - 1.01 s_k10_ADM_s03  = 0
 w_k10_ADM_s10_return_constraint: 0.985893 w_k10_ADM_s1 - w_k10_ADM_s10
                                 + 0.99 b_k10_ADM_s10 - 1.01 s_k10_ADM_s10  = 0
 w_k10_ADM_s11_return_constraint: 0.860283 w_k10_ADM_s1 - w_k10_ADM_s11
                                 + 0.99 b_k10_ADM_s11 - 1.01 s_k10_ADM_s11  = 0
 w_k10_ADM_s12_return_constraint: 1.07939 w_k10_ADM_s1 - w_k10_ADM_s12
                                 + 0.99 b_k10_ADM_s12 - 1.01 s_k10_ADM_s12  = 0
 w_k10_ADM_s13_return_constraint: 1.10456 w_k10_ADM_s1 - w_k10_ADM_s13
                                 + 0.99 b_k10_ADM_s13 - 1.01 s_k10_ADM_s13  = 0
 w_k10_ADM_s000_return_constraint: 1.28684 w_k10_ADM_s00 - w_k10_ADM_s000
                                 + 0.99 b_k10_ADM_s000 - 1.01 s_k10_ADM_s000
                                  = 0
 w_k10_ADM_s001_return_constraint: 1.1956 w_k10_ADM_s00 - w_k10_ADM_s001
                                 + 0.99 b_k10_ADM_s001 - 1.01 s_k10_ADM_s001
                                  = 0
 w_k10_ADM_s002_return_constraint: 1.39445 w_k10_ADM_s00 - w_k10_ADM_s002
                                 + 0.99 b_k10_ADM_s002 - 1.01 s_k10_ADM_s002
                                  = 0
 w_k10_ADM_s003_return_constraint: 0.893075 w_k10_ADM_s00 - w_k10_ADM_s003
                                 + 0.99 b_k10_ADM_s003 - 1.01 s_k10_ADM_s003
                                  = 0
 w_k10_ADM_s004_return_constraint: 1.36326 w_k10_ADM_s00 - w_k10_ADM_s004
                                 + 0.99 b_k10_ADM_s004 - 1.01 s_k10_ADM_s004
                                  = 0
 w_k10_ADM_s005_return_constraint: 1.02751 w_k10_ADM_s00 - w_k10_ADM_s005
                                 + 0.99 b_k10_ADM_s005 - 1.01 s_k10_ADM_s005
                                  = 0
 w_k10_ADM_s006_return_constraint: 0.891426 w_k10_ADM_s00 - w_k10_ADM_s006
                                 + 0.99 b_k10_ADM_s006 - 1.01 s_k10_ADM_s006
                                  = 0
 w_k10_ADM_s007_return_constraint: 1.06017 w_k10_ADM_s00 - w_k10_ADM_s007
                                 + 0.99 b_k10_ADM_s007 - 1.01 s_k10_ADM_s007
                                  = 0
 w_k10_ADM_s010_return_constraint: 0.892337 w_k10_ADM_s01 - w_k10_ADM_s010
                                 + 0.99 b_k10_ADM_s010 - 1.01 s_k10_ADM_s010
                                  = 0
 w_k10_ADM_s011_return_constraint: 1.00656 w_k10_ADM_s01 - w_k10_ADM_s011
                                 + 0.99 b_k10_ADM_s011 - 1.01 s_k10_ADM_s011
                                  = 0
 w_k10_ADM_s012_return_constraint: 0.902934 w_k10_ADM_s01 - w_k10_ADM_s012
                                 + 0.99 b_k10_ADM_s012 - 1.01 s_k10_ADM_s012
                                  = 0
 w_k10_ADM_s013_return_constraint: 0.702489 w_k10_ADM_s01 - w_k10_ADM_s013
                                 + 0.99 b_k10_ADM_s013 - 1.01 s_k10_ADM_s013
                                  = 0
 w_k10_ADM_s014_return_constraint: 1.21667 w_k10_ADM_s01 - w_k10_ADM_s014
                                 + 0.99 b_k10_ADM_s014 - 1.01 s_k10_ADM_s014
                                  = 0
 w_k10_ADM_s015_return_constraint: 0.885968 w_k10_ADM_s01 - w_k10_ADM_s015
                                 + 0.99 b_k10_ADM_s015 - 1.01 s_k10_ADM_s015
                                  = 0
 w_k10_ADM_s016_return_constraint: 0.727534 w_k10_ADM_s01 - w_k10_ADM_s016
                                 + 0.99 b_k10_ADM_s016 - 1.01 s_k10_ADM_s016
                                  = 0
 w_k10_ADM_s017_return_constraint: 1.18834 w_k10_ADM_s01 - w_k10_ADM_s017
                                 + 0.99 b_k10_ADM_s017 - 1.01 s_k10_ADM_s017
                                  = 0
 w_k10_ADM_s020_return_constraint: 1.02404 w_k10_ADM_s02 - w_k10_ADM_s020
                                 + 0.99 b_k10_ADM_s020 - 1.01 s_k10_ADM_s020
                                  = 0
 w_k10_ADM_s021_return_constraint: 0.935416 w_k10_ADM_s02 - w_k10_ADM_s021
                                 + 0.99 b_k10_ADM_s021 - 1.01 s_k10_ADM_s021
                                  = 0
 w_k10_ADM_s022_return_constraint: 1.14861 w_k10_ADM_s02 - w_k10_ADM_s022
                                 + 0.99 b_k10_ADM_s022 - 1.01 s_k10_ADM_s022
                                  = 0
 w_k10_ADM_s023_return_constraint: 0.959403 w_k10_ADM_s02 - w_k10_ADM_s023
                                 + 0.99 b_k10_ADM_s023 - 1.01 s_k10_ADM_s023
                                  = 0
 w_k10_ADM_s024_return_constraint: 1.03028 w_k10_ADM_s02 - w_k10_ADM_s024
                                 + 0.99 b_k10_ADM_s024 - 1.01 s_k10_ADM_s024
                                  = 0
 w_k10_ADM_s025_return_constraint: 1.26014 w_k10_ADM_s02 - w_k10_ADM_s025
                                 + 0.99 b_k10_ADM_s025 - 1.01 s_k10_ADM_s025
                                  = 0
 w_k10_ADM_s026_return_constraint: 1.27864 w_k10_ADM_s02 - w_k10_ADM_s026
                                 + 0.99 b_k10_ADM_s026 - 1.01 s_k10_ADM_s026
                                  = 0
 w_k10_ADM_s027_return_constraint: 0.810723 w_k10_ADM_s02 - w_k10_ADM_s027
                                 + 0.99 b_k10_ADM_s027 - 1.01 s_k10_ADM_s027
                                  = 0
 w_k10_ADM_s030_return_constraint: 0.751848 w_k10_ADM_s03 - w_k10_ADM_s030
                                 + 0.99 b_k10_ADM_s030 - 1.01 s_k10_ADM_s030
                                  = 0
 w_k10_ADM_s031_return_constraint: 0.691412 w_k10_ADM_s03 - w_k10_ADM_s031
                                 + 0.99 b_k10_ADM_s031 - 1.01 s_k10_ADM_s031
                                  = 0
 w_k10_ADM_s032_return_constraint: 1.09198 w_k10_ADM_s03 - w_k10_ADM_s032
                                 + 0.99 b_k10_ADM_s032 - 1.01 s_k10_ADM_s032
                                  = 0
 w_k10_ADM_s033_return_constraint: 1.38624 w_k10_ADM_s03 - w_k10_ADM_s033
                                 + 0.99 b_k10_ADM_s033 - 1.01 s_k10_ADM_s033
                                  = 0
 w_k10_ADM_s034_return_constraint: 0.840747 w_k10_ADM_s03 - w_k10_ADM_s034
                                 + 0.99 b_k10_ADM_s034 - 1.01 s_k10_ADM_s034
                                  = 0
 w_k10_ADM_s035_return_constraint: 1.10977 w_k10_ADM_s03 - w_k10_ADM_s035
                                 + 0.99 b_k10_ADM_s035 - 1.01 s_k10_ADM_s035
                                  = 0
 w_k10_ADM_s036_return_constraint: 0.905942 w_k10_ADM_s03 - w_k10_ADM_s036
                                 + 0.99 b_k10_ADM_s036 - 1.01 s_k10_ADM_s036
                                  = 0
 w_k10_ADM_s037_return_constraint: 1.3112 w_k10_ADM_s03 - w_k10_ADM_s037
                                 + 0.99 b_k10_ADM_s037 - 1.01 s_k10_ADM_s037
                                  = 0
 w_k10_ADM_s100_return_constraint: 1.27075 w_k10_ADM_s10 - w_k10_ADM_s100
                                 + 0.99 b_k10_ADM_s100 - 1.01 s_k10_ADM_s100
                                  = 0
 w_k10_ADM_s101_return_constraint: 0.728572 w_k10_ADM_s10 - w_k10_ADM_s101
                                 + 0.99 b_k10_ADM_s101 - 1.01 s_k10_ADM_s101
                                  = 0
 w_k10_ADM_s102_return_constraint: 1.19765 w_k10_ADM_s10 - w_k10_ADM_s102
                                 + 0.99 b_k10_ADM_s102 - 1.01 s_k10_ADM_s102
                                  = 0
 w_k10_ADM_s103_return_constraint: 1.44194 w_k10_ADM_s10 - w_k10_ADM_s103
                                 + 0.99 b_k10_ADM_s103 - 1.01 s_k10_ADM_s103
                                  = 0
 w_k10_ADM_s104_return_constraint: 1.00518 w_k10_ADM_s10 - w_k10_ADM_s104
                                 + 0.99 b_k10_ADM_s104 - 1.01 s_k10_ADM_s104
                                  = 0
 w_k10_ADM_s105_return_constraint: 1.04582 w_k10_ADM_s10 - w_k10_ADM_s105
                                 + 0.99 b_k10_ADM_s105 - 1.01 s_k10_ADM_s105
                                  = 0
 w_k10_ADM_s106_return_constraint: 1.13636 w_k10_ADM_s10 - w_k10_ADM_s106
                                 + 0.99 b_k10_ADM_s106 - 1.01 s_k10_ADM_s106
                                  = 0
 w_k10_ADM_s107_return_constraint: 0.778549 w_k10_ADM_s10 - w_k10_ADM_s107
                                 + 0.99 b_k10_ADM_s107 - 1.01 s_k10_ADM_s107
                                  = 0
 w_k10_ADM_s110_return_constraint: 0.731358 w_k10_ADM_s11 - w_k10_ADM_s110
                                 + 0.99 b_k10_ADM_s110 - 1.01 s_k10_ADM_s110
                                  = 0
 w_k10_ADM_s111_return_constraint: 0.779587 w_k10_ADM_s11 - w_k10_ADM_s111
                                 + 0.99 b_k10_ADM_s111 - 1.01 s_k10_ADM_s111
                                  = 0
 w_k10_ADM_s112_return_constraint: 1.07826 w_k10_ADM_s11 - w_k10_ADM_s112
                                 + 0.99 b_k10_ADM_s112 - 1.01 s_k10_ADM_s112
                                  = 0
 w_k10_ADM_s113_return_constraint: 0.986009 w_k10_ADM_s11 - w_k10_ADM_s113
                                 + 0.99 b_k10_ADM_s113 - 1.01 s_k10_ADM_s113
                                  = 0
 w_k10_ADM_s114_return_constraint: 0.862321 w_k10_ADM_s11 - w_k10_ADM_s114
                                 + 0.99 b_k10_ADM_s114 - 1.01 s_k10_ADM_s114
                                  = 0
 w_k10_ADM_s115_return_constraint: 1.19294 w_k10_ADM_s11 - w_k10_ADM_s115
                                 + 0.99 b_k10_ADM_s115 - 1.01 s_k10_ADM_s115
                                  = 0
 w_k10_ADM_s116_return_constraint: 0.8569 w_k10_ADM_s11 - w_k10_ADM_s116
                                 + 0.99 b_k10_ADM_s116 - 1.01 s_k10_ADM_s116
                                  = 0
 w_k10_ADM_s117_return_constraint: 1.13701 w_k10_ADM_s11 - w_k10_ADM_s117
                                 + 0.99 b_k10_ADM_s117 - 1.01 s_k10_ADM_s117
                                  = 0
 w_k10_ADM_s120_return_constraint: 1.11306 w_k10_ADM_s12 - w_k10_ADM_s120
                                 + 0.99 b_k10_ADM_s120 - 1.01 s_k10_ADM_s120
                                  = 0
 w_k10_ADM_s121_return_constraint: 0.83866 w_k10_ADM_s12 - w_k10_ADM_s121
                                 + 0.99 b_k10_ADM_s121 - 1.01 s_k10_ADM_s121
                                  = 0
 w_k10_ADM_s122_return_constraint: 0.885113 w_k10_ADM_s12 - w_k10_ADM_s122
                                 + 0.99 b_k10_ADM_s122 - 1.01 s_k10_ADM_s122
                                  = 0
 w_k10_ADM_s123_return_constraint: 0.813728 w_k10_ADM_s12 - w_k10_ADM_s123
                                 + 0.99 b_k10_ADM_s123 - 1.01 s_k10_ADM_s123
                                  = 0
 w_k10_ADM_s124_return_constraint: 1.02296 w_k10_ADM_s12 - w_k10_ADM_s124
                                 + 0.99 b_k10_ADM_s124 - 1.01 s_k10_ADM_s124
                                  = 0
 w_k10_ADM_s125_return_constraint: 1.19907 w_k10_ADM_s12 - w_k10_ADM_s125
                                 + 0.99 b_k10_ADM_s125 - 1.01 s_k10_ADM_s125
                                  = 0
 w_k10_ADM_s126_return_constraint: 0.915066 w_k10_ADM_s12 - w_k10_ADM_s126
                                 + 0.99 b_k10_ADM_s126 - 1.01 s_k10_ADM_s126
                                  = 0
 w_k10_ADM_s127_return_constraint: 0.780681 w_k10_ADM_s12 - w_k10_ADM_s127
                                 + 0.99 b_k10_ADM_s127 - 1.01 s_k10_ADM_s127
                                  = 0
 w_k10_ADM_s130_return_constraint: 1.41281 w_k10_ADM_s13 - w_k10_ADM_s130
                                 + 0.99 b_k10_ADM_s130 - 1.01 s_k10_ADM_s130
                                  = 0
 w_k10_ADM_s131_return_constraint: 1.0818 w_k10_ADM_s13 - w_k10_ADM_s131
                                 + 0.99 b_k10_ADM_s131 - 1.01 s_k10_ADM_s131
                                  = 0
 w_k10_ADM_s132_return_constraint: 1.06489 w_k10_ADM_s13 - w_k10_ADM_s132
                                 + 0.99 b_k10_ADM_s132 - 1.01 s_k10_ADM_s132
                                  = 0
 w_k10_ADM_s133_return_constraint: 0.805792 w_k10_ADM_s13 - w_k10_ADM_s133
                                 + 0.99 b_k10_ADM_s133 - 1.01 s_k10_ADM_s133
                                  = 0
 w_k10_ADM_s134_return_constraint: 1.09472 w_k10_ADM_s13 - w_k10_ADM_s134
                                 + 0.99 b_k10_ADM_s134 - 1.01 s_k10_ADM_s134
                                  = 0
 w_k10_ADM_s135_return_constraint: 1.36643 w_k10_ADM_s13 - w_k10_ADM_s135
                                 + 0.99 b_k10_ADM_s135 - 1.01 s_k10_ADM_s135
                                  = 0
 w_k10_ADM_s136_return_constraint: 1.4681 w_k10_ADM_s13 - w_k10_ADM_s136
                                 + 0.99 b_k10_ADM_s136 - 1.01 s_k10_ADM_s136
                                  = 0
 w_k10_ADM_s137_return_constraint: 0.799761 w_k10_ADM_s13 - w_k10_ADM_s137
                                 + 0.99 b_k10_ADM_s137 - 1.01 s_k10_ADM_s137
                                  = 0
 w_k10_AXP_s0_return_constraint: 1.0174 w_0_AXP - w_k10_AXP_s0
                                 + 0.99 b_k10_AXP_s0 - 1.01 s_k10_AXP_s0  = 0
 w_k10_AXP_s1_return_constraint: 1.15555 w_0_AXP - w_k10_AXP_s1
                                 + 0.99 b_k10_AXP_s1 - 1.01 s_k10_AXP_s1  = 0
 w_k10_AXP_s00_return_constraint: 0.863619 w_k10_AXP_s0 - w_k10_AXP_s00
                                 + 0.99 b_k10_AXP_s00 - 1.01 s_k10_AXP_s00  = 0
 w_k10_AXP_s01_return_constraint: 0.751631 w_k10_AXP_s0 - w_k10_AXP_s01
                                 + 0.99 b_k10_AXP_s01 - 1.01 s_k10_AXP_s01  = 0
 w_k10_AXP_s02_return_constraint: 1.25071 w_k10_AXP_s0 - w_k10_AXP_s02
                                 + 0.99 b_k10_AXP_s02 - 1.01 s_k10_AXP_s02  = 0
 w_k10_AXP_s03_return_constraint: 1.34171 w_k10_AXP_s0 - w_k10_AXP_s03
                                 + 0.99 b_k10_AXP_s03 - 1.01 s_k10_AXP_s03  = 0
 w_k10_AXP_s10_return_constraint: 1.21294 w_k10_AXP_s1 - w_k10_AXP_s10
                                 + 0.99 b_k10_AXP_s10 - 1.01 s_k10_AXP_s10  = 0
 w_k10_AXP_s11_return_constraint: 0.774112 w_k10_AXP_s1 - w_k10_AXP_s11
                                 + 0.99 b_k10_AXP_s11 - 1.01 s_k10_AXP_s11  = 0
 w_k10_AXP_s12_return_constraint: 1.08992 w_k10_AXP_s1 - w_k10_AXP_s12
                                 + 0.99 b_k10_AXP_s12 - 1.01 s_k10_AXP_s12  = 0
 w_k10_AXP_s13_return_constraint: 1.50806 w_k10_AXP_s1 - w_k10_AXP_s13
                                 + 0.99 b_k10_AXP_s13 - 1.01 s_k10_AXP_s13  = 0
 w_k10_AXP_s000_return_constraint: 1.31001 w_k10_AXP_s00 - w_k10_AXP_s000
                                 + 0.99 b_k10_AXP_s000 - 1.01 s_k10_AXP_s000
                                  = 0
 w_k10_AXP_s001_return_constraint: 1.24398 w_k10_AXP_s00 - w_k10_AXP_s001
                                 + 0.99 b_k10_AXP_s001 - 1.01 s_k10_AXP_s001
                                  = 0
 w_k10_AXP_s002_return_constraint: 1.54678 w_k10_AXP_s00 - w_k10_AXP_s002
                                 + 0.99 b_k10_AXP_s002 - 1.01 s_k10_AXP_s002
                                  = 0
 w_k10_AXP_s003_return_constraint: 0.687035 w_k10_AXP_s00 - w_k10_AXP_s003
                                 + 0.99 b_k10_AXP_s003 - 1.01 s_k10_AXP_s003
                                  = 0
 w_k10_AXP_s004_return_constraint: 1.21401 w_k10_AXP_s00 - w_k10_AXP_s004
                                 + 0.99 b_k10_AXP_s004 - 1.01 s_k10_AXP_s004
                                  = 0
 w_k10_AXP_s005_return_constraint: 0.954268 w_k10_AXP_s00 - w_k10_AXP_s005
                                 + 0.99 b_k10_AXP_s005 - 1.01 s_k10_AXP_s005
                                  = 0
 w_k10_AXP_s006_return_constraint: 1.00047 w_k10_AXP_s00 - w_k10_AXP_s006
                                 + 0.99 b_k10_AXP_s006 - 1.01 s_k10_AXP_s006
                                  = 0
 w_k10_AXP_s007_return_constraint: 0.857876 w_k10_AXP_s00 - w_k10_AXP_s007
                                 + 0.99 b_k10_AXP_s007 - 1.01 s_k10_AXP_s007
                                  = 0
 w_k10_AXP_s010_return_constraint: 0.769272 w_k10_AXP_s01 - w_k10_AXP_s010
                                 + 0.99 b_k10_AXP_s010 - 1.01 s_k10_AXP_s010
                                  = 0
 w_k10_AXP_s011_return_constraint: 0.916689 w_k10_AXP_s01 - w_k10_AXP_s011
                                 + 0.99 b_k10_AXP_s011 - 1.01 s_k10_AXP_s011
                                  = 0
 w_k10_AXP_s012_return_constraint: 0.936141 w_k10_AXP_s01 - w_k10_AXP_s012
                                 + 0.99 b_k10_AXP_s012 - 1.01 s_k10_AXP_s012
                                  = 0
 w_k10_AXP_s013_return_constraint: 0.614902 w_k10_AXP_s01 - w_k10_AXP_s013
                                 + 0.99 b_k10_AXP_s013 - 1.01 s_k10_AXP_s013
                                  = 0
 w_k10_AXP_s014_return_constraint: 1.47811 w_k10_AXP_s01 - w_k10_AXP_s014
                                 + 0.99 b_k10_AXP_s014 - 1.01 s_k10_AXP_s014
                                  = 0
 w_k10_AXP_s015_return_constraint: 0.758213 w_k10_AXP_s01 - w_k10_AXP_s015
                                 + 0.99 b_k10_AXP_s015 - 1.01 s_k10_AXP_s015
                                  = 0
 w_k10_AXP_s016_return_constraint: 0.716731 w_k10_AXP_s01 - w_k10_AXP_s016
                                 + 0.99 b_k10_AXP_s016 - 1.01 s_k10_AXP_s016
                                  = 0
 w_k10_AXP_s017_return_constraint: 1.46073 w_k10_AXP_s01 - w_k10_AXP_s017
                                 + 0.99 b_k10_AXP_s017 - 1.01 s_k10_AXP_s017
                                  = 0
 w_k10_AXP_s020_return_constraint: 1.00791 w_k10_AXP_s02 - w_k10_AXP_s020
                                 + 0.99 b_k10_AXP_s020 - 1.01 s_k10_AXP_s020
                                  = 0
 w_k10_AXP_s021_return_constraint: 1.00445 w_k10_AXP_s02 - w_k10_AXP_s021
                                 + 0.99 b_k10_AXP_s021 - 1.01 s_k10_AXP_s021
                                  = 0
 w_k10_AXP_s022_return_constraint: 1.1698 w_k10_AXP_s02 - w_k10_AXP_s022
                                 + 0.99 b_k10_AXP_s022 - 1.01 s_k10_AXP_s022
                                  = 0
 w_k10_AXP_s023_return_constraint: 0.721138 w_k10_AXP_s02 - w_k10_AXP_s023
                                 + 0.99 b_k10_AXP_s023 - 1.01 s_k10_AXP_s023
                                  = 0
 w_k10_AXP_s024_return_constraint: 0.953685 w_k10_AXP_s02 - w_k10_AXP_s024
                                 + 0.99 b_k10_AXP_s024 - 1.01 s_k10_AXP_s024
                                  = 0
 w_k10_AXP_s025_return_constraint: 1.58936 w_k10_AXP_s02 - w_k10_AXP_s025
                                 + 0.99 b_k10_AXP_s025 - 1.01 s_k10_AXP_s025
                                  = 0
 w_k10_AXP_s026_return_constraint: 1.48761 w_k10_AXP_s02 - w_k10_AXP_s026
                                 + 0.99 b_k10_AXP_s026 - 1.01 s_k10_AXP_s026
                                  = 0
 w_k10_AXP_s027_return_constraint: 0.736462 w_k10_AXP_s02 - w_k10_AXP_s027
                                 + 0.99 b_k10_AXP_s027 - 1.01 s_k10_AXP_s027
                                  = 0
 w_k10_AXP_s030_return_constraint: 0.703781 w_k10_AXP_s03 - w_k10_AXP_s030
                                 + 0.99 b_k10_AXP_s030 - 1.01 s_k10_AXP_s030
                                  = 0
 w_k10_AXP_s031_return_constraint: 0.682119 w_k10_AXP_s03 - w_k10_AXP_s031
                                 + 0.99 b_k10_AXP_s031 - 1.01 s_k10_AXP_s031
                                  = 0
 w_k10_AXP_s032_return_constraint: 1.07034 w_k10_AXP_s03 - w_k10_AXP_s032
                                 + 0.99 b_k10_AXP_s032 - 1.01 s_k10_AXP_s032
                                  = 0
 w_k10_AXP_s033_return_constraint: 1.58525 w_k10_AXP_s03 - w_k10_AXP_s033
                                 + 0.99 b_k10_AXP_s033 - 1.01 s_k10_AXP_s033
                                  = 0
 w_k10_AXP_s034_return_constraint: 0.968361 w_k10_AXP_s03 - w_k10_AXP_s034
                                 + 0.99 b_k10_AXP_s034 - 1.01 s_k10_AXP_s034
                                  = 0
 w_k10_AXP_s035_return_constraint: 1.06981 w_k10_AXP_s03 - w_k10_AXP_s035
                                 + 0.99 b_k10_AXP_s035 - 1.01 s_k10_AXP_s035
                                  = 0
 w_k10_AXP_s036_return_constraint: 0.82505 w_k10_AXP_s03 - w_k10_AXP_s036
                                 + 0.99 b_k10_AXP_s036 - 1.01 s_k10_AXP_s036
                                  = 0
 w_k10_AXP_s037_return_constraint: 1.81822 w_k10_AXP_s03 - w_k10_AXP_s037
                                 + 0.99 b_k10_AXP_s037 - 1.01 s_k10_AXP_s037
                                  = 0
 w_k10_AXP_s100_return_constraint: 1.42248 w_k10_AXP_s10 - w_k10_AXP_s100
                                 + 0.99 b_k10_AXP_s100 - 1.01 s_k10_AXP_s100
                                  = 0
 w_k10_AXP_s101_return_constraint: 0.692935 w_k10_AXP_s10 - w_k10_AXP_s101
                                 + 0.99 b_k10_AXP_s101 - 1.01 s_k10_AXP_s101
                                  = 0
 w_k10_AXP_s102_return_constraint: 1.22241 w_k10_AXP_s10 - w_k10_AXP_s102
                                 + 0.99 b_k10_AXP_s102 - 1.01 s_k10_AXP_s102
                                  = 0
 w_k10_AXP_s103_return_constraint: 1.46794 w_k10_AXP_s10 - w_k10_AXP_s103
                                 + 0.99 b_k10_AXP_s103 - 1.01 s_k10_AXP_s103
                                  = 0
 w_k10_AXP_s104_return_constraint: 0.932428 w_k10_AXP_s10 - w_k10_AXP_s104
                                 + 0.99 b_k10_AXP_s104 - 1.01 s_k10_AXP_s104
                                  = 0
 w_k10_AXP_s105_return_constraint: 1.07893 w_k10_AXP_s10 - w_k10_AXP_s105
                                 + 0.99 b_k10_AXP_s105 - 1.01 s_k10_AXP_s105
                                  = 0
 w_k10_AXP_s106_return_constraint: 0.820453 w_k10_AXP_s10 - w_k10_AXP_s106
                                 + 0.99 b_k10_AXP_s106 - 1.01 s_k10_AXP_s106
                                  = 0
 w_k10_AXP_s107_return_constraint: 0.851505 w_k10_AXP_s10 - w_k10_AXP_s107
                                 + 0.99 b_k10_AXP_s107 - 1.01 s_k10_AXP_s107
                                  = 0
 w_k10_AXP_s110_return_constraint: 0.701904 w_k10_AXP_s11 - w_k10_AXP_s110
                                 + 0.99 b_k10_AXP_s110 - 1.01 s_k10_AXP_s110
                                  = 0
 w_k10_AXP_s111_return_constraint: 1.07422 w_k10_AXP_s11 - w_k10_AXP_s111
                                 + 0.99 b_k10_AXP_s111 - 1.01 s_k10_AXP_s111
                                  = 0
 w_k10_AXP_s112_return_constraint: 0.956959 w_k10_AXP_s11 - w_k10_AXP_s112
                                 + 0.99 b_k10_AXP_s112 - 1.01 s_k10_AXP_s112
                                  = 0
 w_k10_AXP_s113_return_constraint: 1.49272 w_k10_AXP_s11 - w_k10_AXP_s113
                                 + 0.99 b_k10_AXP_s113 - 1.01 s_k10_AXP_s113
                                  = 0
 w_k10_AXP_s114_return_constraint: 0.723388 w_k10_AXP_s11 - w_k10_AXP_s114
                                 + 0.99 b_k10_AXP_s114 - 1.01 s_k10_AXP_s114
                                  = 0
 w_k10_AXP_s115_return_constraint: 1.21811 w_k10_AXP_s11 - w_k10_AXP_s115
                                 + 0.99 b_k10_AXP_s115 - 1.01 s_k10_AXP_s115
                                  = 0
 w_k10_AXP_s116_return_constraint: 0.890728 w_k10_AXP_s11 - w_k10_AXP_s116
                                 + 0.99 b_k10_AXP_s116 - 1.01 s_k10_AXP_s116
                                  = 0
 w_k10_AXP_s117_return_constraint: 1.20104 w_k10_AXP_s11 - w_k10_AXP_s117
                                 + 0.99 b_k10_AXP_s117 - 1.01 s_k10_AXP_s117
                                  = 0
 w_k10_AXP_s120_return_constraint: 1.304 w_k10_AXP_s12 - w_k10_AXP_s120
                                 + 0.99 b_k10_AXP_s120 - 1.01 s_k10_AXP_s120
                                  = 0
 w_k10_AXP_s121_return_constraint: 0.788178 w_k10_AXP_s12 - w_k10_AXP_s121
                                 + 0.99 b_k10_AXP_s121 - 1.01 s_k10_AXP_s121
                                  = 0
 w_k10_AXP_s122_return_constraint: 0.748503 w_k10_AXP_s12 - w_k10_AXP_s122
                                 + 0.99 b_k10_AXP_s122 - 1.01 s_k10_AXP_s122
                                  = 0
 w_k10_AXP_s123_return_constraint: 1.08089 w_k10_AXP_s12 - w_k10_AXP_s123
                                 + 0.99 b_k10_AXP_s123 - 1.01 s_k10_AXP_s123
                                  = 0
 w_k10_AXP_s124_return_constraint: 1.13294 w_k10_AXP_s12 - w_k10_AXP_s124
                                 + 0.99 b_k10_AXP_s124 - 1.01 s_k10_AXP_s124
                                  = 0
 w_k10_AXP_s125_return_constraint: 1.53018 w_k10_AXP_s12 - w_k10_AXP_s125
                                 + 0.99 b_k10_AXP_s125 - 1.01 s_k10_AXP_s125
                                  = 0
 w_k10_AXP_s126_return_constraint: 0.825666 w_k10_AXP_s12 - w_k10_AXP_s126
                                 + 0.99 b_k10_AXP_s126 - 1.01 s_k10_AXP_s126
                                  = 0
 w_k10_AXP_s127_return_constraint: 0.751662 w_k10_AXP_s12 - w_k10_AXP_s127
                                 + 0.99 b_k10_AXP_s127 - 1.01 s_k10_AXP_s127
                                  = 0
 w_k10_AXP_s130_return_constraint: 1.26423 w_k10_AXP_s13 - w_k10_AXP_s130
                                 + 0.99 b_k10_AXP_s130 - 1.01 s_k10_AXP_s130
                                  = 0
 w_k10_AXP_s131_return_constraint: 1.50908 w_k10_AXP_s13 - w_k10_AXP_s131
                                 + 0.99 b_k10_AXP_s131 - 1.01 s_k10_AXP_s131
                                  = 0
 w_k10_AXP_s132_return_constraint: 0.785374 w_k10_AXP_s13 - w_k10_AXP_s132
                                 + 0.99 b_k10_AXP_s132 - 1.01 s_k10_AXP_s132
                                  = 0
 w_k10_AXP_s133_return_constraint: 0.79418 w_k10_AXP_s13 - w_k10_AXP_s133
                                 + 0.99 b_k10_AXP_s133 - 1.01 s_k10_AXP_s133
                                  = 0
 w_k10_AXP_s134_return_constraint: 0.873399 w_k10_AXP_s13 - w_k10_AXP_s134
                                 + 0.99 b_k10_AXP_s134 - 1.01 s_k10_AXP_s134
                                  = 0
 w_k10_AXP_s135_return_constraint: 1.04026 w_k10_AXP_s13 - w_k10_AXP_s135
                                 + 0.99 b_k10_AXP_s135 - 1.01 s_k10_AXP_s135
                                  = 0
 w_k10_AXP_s136_return_constraint: 1.15564 w_k10_AXP_s13 - w_k10_AXP_s136
                                 + 0.99 b_k10_AXP_s136 - 1.01 s_k10_AXP_s136
                                  = 0
 w_k10_AXP_s137_return_constraint: 0.560207 w_k10_AXP_s13 - w_k10_AXP_s137
                                 + 0.99 b_k10_AXP_s137 - 1.01 s_k10_AXP_s137
                                  = 0
 w_k10_DIS_s0_return_constraint: 0.999809 w_0_DIS - w_k10_DIS_s0
                                 + 0.99 b_k10_DIS_s0 - 1.01 s_k10_DIS_s0  = 0
 w_k10_DIS_s1_return_constraint: 1.12628 w_0_DIS - w_k10_DIS_s1
                                 + 0.99 b_k10_DIS_s1 - 1.01 s_k10_DIS_s1  = 0
 w_k10_DIS_s00_return_constraint: 0.791205 w_k10_DIS_s0 - w_k10_DIS_s00
                                 + 0.99 b_k10_DIS_s00 - 1.01 s_k10_DIS_s00  = 0
 w_k10_DIS_s01_return_constraint: 0.708748 w_k10_DIS_s0 - w_k10_DIS_s01
                                 + 0.99 b_k10_DIS_s01 - 1.01 s_k10_DIS_s01  = 0
 w_k10_DIS_s02_return_constraint: 1.40002 w_k10_DIS_s0 - w_k10_DIS_s02
                                 + 0.99 b_k10_DIS_s02 - 1.01 s_k10_DIS_s02  = 0
 w_k10_DIS_s03_return_constraint: 1.38667 w_k10_DIS_s0 - w_k10_DIS_s03
                                 + 0.99 b_k10_DIS_s03 - 1.01 s_k10_DIS_s03  = 0
 w_k10_DIS_s10_return_constraint: 1.29625 w_k10_DIS_s1 - w_k10_DIS_s10
                                 + 0.99 b_k10_DIS_s10 - 1.01 s_k10_DIS_s10  = 0
 w_k10_DIS_s11_return_constraint: 0.753911 w_k10_DIS_s1 - w_k10_DIS_s11
                                 + 0.99 b_k10_DIS_s11 - 1.01 s_k10_DIS_s11  = 0
 w_k10_DIS_s12_return_constraint: 1.21717 w_k10_DIS_s1 - w_k10_DIS_s12
                                 + 0.99 b_k10_DIS_s12 - 1.01 s_k10_DIS_s12  = 0
 w_k10_DIS_s13_return_constraint: 1.52254 w_k10_DIS_s1 - w_k10_DIS_s13
                                 + 0.99 b_k10_DIS_s13 - 1.01 s_k10_DIS_s13  = 0
 w_k10_DIS_s000_return_constraint: 1.3253 w_k10_DIS_s00 - w_k10_DIS_s000
                                 + 0.99 b_k10_DIS_s000 - 1.01 s_k10_DIS_s000
                                  = 0
 w_k10_DIS_s001_return_constraint: 1.30025 w_k10_DIS_s00 - w_k10_DIS_s001
                                 + 0.99 b_k10_DIS_s001 - 1.01 s_k10_DIS_s001
                                  = 0
 w_k10_DIS_s002_return_constraint: 1.74223 w_k10_DIS_s00 - w_k10_DIS_s002
                                 + 0.99 b_k10_DIS_s002 - 1.01 s_k10_DIS_s002
                                  = 0
 w_k10_DIS_s003_return_constraint: 0.701747 w_k10_DIS_s00 - w_k10_DIS_s003
                                 + 0.99 b_k10_DIS_s003 - 1.01 s_k10_DIS_s003
                                  = 0
 w_k10_DIS_s004_return_constraint: 1.2834 w_k10_DIS_s00 - w_k10_DIS_s004
                                 + 0.99 b_k10_DIS_s004 - 1.01 s_k10_DIS_s004
                                  = 0
 w_k10_DIS_s005_return_constraint: 0.931321 w_k10_DIS_s00 - w_k10_DIS_s005
                                 + 0.99 b_k10_DIS_s005 - 1.01 s_k10_DIS_s005
                                  = 0
 w_k10_DIS_s006_return_constraint: 0.885719 w_k10_DIS_s00 - w_k10_DIS_s006
                                 + 0.99 b_k10_DIS_s006 - 1.01 s_k10_DIS_s006
                                  = 0
 w_k10_DIS_s007_return_constraint: 1.21688 w_k10_DIS_s00 - w_k10_DIS_s007
                                 + 0.99 b_k10_DIS_s007 - 1.01 s_k10_DIS_s007
                                  = 0
 w_k10_DIS_s010_return_constraint: 0.896941 w_k10_DIS_s01 - w_k10_DIS_s010
                                 + 0.99 b_k10_DIS_s010 - 1.01 s_k10_DIS_s010
                                  = 0
 w_k10_DIS_s011_return_constraint: 1.20567 w_k10_DIS_s01 - w_k10_DIS_s011
                                 + 0.99 b_k10_DIS_s011 - 1.01 s_k10_DIS_s011
                                  = 0
 w_k10_DIS_s012_return_constraint: 0.911629 w_k10_DIS_s01 - w_k10_DIS_s012
                                 + 0.99 b_k10_DIS_s012 - 1.01 s_k10_DIS_s012
                                  = 0
 w_k10_DIS_s013_return_constraint: 0.4888 w_k10_DIS_s01 - w_k10_DIS_s013
                                 + 0.99 b_k10_DIS_s013 - 1.01 s_k10_DIS_s013
                                  = 0
 w_k10_DIS_s014_return_constraint: 1.28488 w_k10_DIS_s01 - w_k10_DIS_s014
                                 + 0.99 b_k10_DIS_s014 - 1.01 s_k10_DIS_s014
                                  = 0
 w_k10_DIS_s015_return_constraint: 0.939653 w_k10_DIS_s01 - w_k10_DIS_s015
                                 + 0.99 b_k10_DIS_s015 - 1.01 s_k10_DIS_s015
                                  = 0
 w_k10_DIS_s016_return_constraint: 0.635489 w_k10_DIS_s01 - w_k10_DIS_s016
                                 + 0.99 b_k10_DIS_s016 - 1.01 s_k10_DIS_s016
                                  = 0
 w_k10_DIS_s017_return_constraint: 1.91443 w_k10_DIS_s01 - w_k10_DIS_s017
                                 + 0.99 b_k10_DIS_s017 - 1.01 s_k10_DIS_s017
                                  = 0
 w_k10_DIS_s020_return_constraint: 0.775392 w_k10_DIS_s02 - w_k10_DIS_s020
                                 + 0.99 b_k10_DIS_s020 - 1.01 s_k10_DIS_s020
                                  = 0
 w_k10_DIS_s021_return_constraint: 1.08037 w_k10_DIS_s02 - w_k10_DIS_s021
                                 + 0.99 b_k10_DIS_s021 - 1.01 s_k10_DIS_s021
                                  = 0
 w_k10_DIS_s022_return_constraint: 1.20498 w_k10_DIS_s02 - w_k10_DIS_s022
                                 + 0.99 b_k10_DIS_s022 - 1.01 s_k10_DIS_s022
                                  = 0
 w_k10_DIS_s023_return_constraint: 0.725376 w_k10_DIS_s02 - w_k10_DIS_s023
                                 + 0.99 b_k10_DIS_s023 - 1.01 s_k10_DIS_s023
                                  = 0
 w_k10_DIS_s024_return_constraint: 1.23812 w_k10_DIS_s02 - w_k10_DIS_s024
                                 + 0.99 b_k10_DIS_s024 - 1.01 s_k10_DIS_s024
                                  = 0
 w_k10_DIS_s025_return_constraint: 1.90843 w_k10_DIS_s02 - w_k10_DIS_s025
                                 + 0.99 b_k10_DIS_s025 - 1.01 s_k10_DIS_s025
                                  = 0
 w_k10_DIS_s026_return_constraint: 1.2203 w_k10_DIS_s02 - w_k10_DIS_s026
                                 + 0.99 b_k10_DIS_s026 - 1.01 s_k10_DIS_s026
                                  = 0
 w_k10_DIS_s027_return_constraint: 0.689363 w_k10_DIS_s02 - w_k10_DIS_s027
                                 + 0.99 b_k10_DIS_s027 - 1.01 s_k10_DIS_s027
                                  = 0
 w_k10_DIS_s030_return_constraint: 0.66199 w_k10_DIS_s03 - w_k10_DIS_s030
                                 + 0.99 b_k10_DIS_s030 - 1.01 s_k10_DIS_s030
                                  = 0
 w_k10_DIS_s031_return_constraint: 0.672741 w_k10_DIS_s03 - w_k10_DIS_s031
                                 + 0.99 b_k10_DIS_s031 - 1.01 s_k10_DIS_s031
                                  = 0
 w_k10_DIS_s032_return_constraint: 0.915127 w_k10_DIS_s03 - w_k10_DIS_s032
                                 + 0.99 b_k10_DIS_s032 - 1.01 s_k10_DIS_s032
                                  = 0
 w_k10_DIS_s033_return_constraint: 1.55279 w_k10_DIS_s03 - w_k10_DIS_s033
                                 + 0.99 b_k10_DIS_s033 - 1.01 s_k10_DIS_s033
                                  = 0
 w_k10_DIS_s034_return_constraint: 0.877079 w_k10_DIS_s03 - w_k10_DIS_s034
                                 + 0.99 b_k10_DIS_s034 - 1.01 s_k10_DIS_s034
                                  = 0
 w_k10_DIS_s035_return_constraint: 1.46693 w_k10_DIS_s03 - w_k10_DIS_s035
                                 + 0.99 b_k10_DIS_s035 - 1.01 s_k10_DIS_s035
                                  = 0
 w_k10_DIS_s036_return_constraint: 1.03466 w_k10_DIS_s03 - w_k10_DIS_s036
                                 + 0.99 b_k10_DIS_s036 - 1.01 s_k10_DIS_s036
                                  = 0
 w_k10_DIS_s037_return_constraint: 2.06687 w_k10_DIS_s03 - w_k10_DIS_s037
                                 + 0.99 b_k10_DIS_s037 - 1.01 s_k10_DIS_s037
                                  = 0
 w_k10_DIS_s100_return_constraint: 1.68029 w_k10_DIS_s10 - w_k10_DIS_s100
                                 + 0.99 b_k10_DIS_s100 - 1.01 s_k10_DIS_s100
                                  = 0
 w_k10_DIS_s101_return_constraint: 0.585583 w_k10_DIS_s10 - w_k10_DIS_s101
                                 + 0.99 b_k10_DIS_s101 - 1.01 s_k10_DIS_s101
                                  = 0
 w_k10_DIS_s102_return_constraint: 1.11031 w_k10_DIS_s10 - w_k10_DIS_s102
                                 + 0.99 b_k10_DIS_s102 - 1.01 s_k10_DIS_s102
                                  = 0
 w_k10_DIS_s103_return_constraint: 1.54162 w_k10_DIS_s10 - w_k10_DIS_s103
                                 + 0.99 b_k10_DIS_s103 - 1.01 s_k10_DIS_s103
                                  = 0
 w_k10_DIS_s104_return_constraint: 0.976209 w_k10_DIS_s10 - w_k10_DIS_s104
                                 + 0.99 b_k10_DIS_s104 - 1.01 s_k10_DIS_s104
                                  = 0
 w_k10_DIS_s105_return_constraint: 1.11814 w_k10_DIS_s10 - w_k10_DIS_s105
                                 + 0.99 b_k10_DIS_s105 - 1.01 s_k10_DIS_s105
                                  = 0
 w_k10_DIS_s106_return_constraint: 0.828107 w_k10_DIS_s10 - w_k10_DIS_s106
                                 + 0.99 b_k10_DIS_s106 - 1.01 s_k10_DIS_s106
                                  = 0
 w_k10_DIS_s107_return_constraint: 0.699128 w_k10_DIS_s10 - w_k10_DIS_s107
                                 + 0.99 b_k10_DIS_s107 - 1.01 s_k10_DIS_s107
                                  = 0
 w_k10_DIS_s110_return_constraint: 0.69441 w_k10_DIS_s11 - w_k10_DIS_s110
                                 + 0.99 b_k10_DIS_s110 - 1.01 s_k10_DIS_s110
                                  = 0
 w_k10_DIS_s111_return_constraint: 0.894092 w_k10_DIS_s11 - w_k10_DIS_s111
                                 + 0.99 b_k10_DIS_s111 - 1.01 s_k10_DIS_s111
                                  = 0
 w_k10_DIS_s112_return_constraint: 0.913315 w_k10_DIS_s11 - w_k10_DIS_s112
                                 + 0.99 b_k10_DIS_s112 - 1.01 s_k10_DIS_s112
                                  = 0
 w_k10_DIS_s113_return_constraint: 1.59815 w_k10_DIS_s11 - w_k10_DIS_s113
                                 + 0.99 b_k10_DIS_s113 - 1.01 s_k10_DIS_s113
                                  = 0
 w_k10_DIS_s114_return_constraint: 0.953183 w_k10_DIS_s11 - w_k10_DIS_s114
                                 + 0.99 b_k10_DIS_s114 - 1.01 s_k10_DIS_s114
                                  = 0
 w_k10_DIS_s115_return_constraint: 1.07692 w_k10_DIS_s11 - w_k10_DIS_s115
                                 + 0.99 b_k10_DIS_s115 - 1.01 s_k10_DIS_s115
                                  = 0
 w_k10_DIS_s116_return_constraint: 0.900613 w_k10_DIS_s11 - w_k10_DIS_s116
                                 + 0.99 b_k10_DIS_s116 - 1.01 s_k10_DIS_s116
                                  = 0
 w_k10_DIS_s117_return_constraint: 1.49819 w_k10_DIS_s11 - w_k10_DIS_s117
                                 + 0.99 b_k10_DIS_s117 - 1.01 s_k10_DIS_s117
                                  = 0
 w_k10_DIS_s120_return_constraint: 1.42071 w_k10_DIS_s12 - w_k10_DIS_s120
                                 + 0.99 b_k10_DIS_s120 - 1.01 s_k10_DIS_s120
                                  = 0
 w_k10_DIS_s121_return_constraint: 1.01337 w_k10_DIS_s12 - w_k10_DIS_s121
                                 + 0.99 b_k10_DIS_s121 - 1.01 s_k10_DIS_s121
                                  = 0
 w_k10_DIS_s122_return_constraint: 0.643417 w_k10_DIS_s12 - w_k10_DIS_s122
                                 + 0.99 b_k10_DIS_s122 - 1.01 s_k10_DIS_s122
                                  = 0
 w_k10_DIS_s123_return_constraint: 1.09926 w_k10_DIS_s12 - w_k10_DIS_s123
                                 + 0.99 b_k10_DIS_s123 - 1.01 s_k10_DIS_s123
                                  = 0
 w_k10_DIS_s124_return_constraint: 1.04744 w_k10_DIS_s12 - w_k10_DIS_s124
                                 + 0.99 b_k10_DIS_s124 - 1.01 s_k10_DIS_s124
                                  = 0
 w_k10_DIS_s125_return_constraint: 1.65514 w_k10_DIS_s12 - w_k10_DIS_s125
                                 + 0.99 b_k10_DIS_s125 - 1.01 s_k10_DIS_s125
                                  = 0
 w_k10_DIS_s126_return_constraint: 0.756793 w_k10_DIS_s12 - w_k10_DIS_s126
                                 + 0.99 b_k10_DIS_s126 - 1.01 s_k10_DIS_s126
                                  = 0
 w_k10_DIS_s127_return_constraint: 0.72489 w_k10_DIS_s12 - w_k10_DIS_s127
                                 + 0.99 b_k10_DIS_s127 - 1.01 s_k10_DIS_s127
                                  = 0
 w_k10_DIS_s130_return_constraint: 0.975073 w_k10_DIS_s13 - w_k10_DIS_s130
                                 + 0.99 b_k10_DIS_s130 - 1.01 s_k10_DIS_s130
                                  = 0
 w_k10_DIS_s131_return_constraint: 1.92831 w_k10_DIS_s13 - w_k10_DIS_s131
                                 + 0.99 b_k10_DIS_s131 - 1.01 s_k10_DIS_s131
                                  = 0
 w_k10_DIS_s132_return_constraint: 0.8748 w_k10_DIS_s13 - w_k10_DIS_s132
                                 + 0.99 b_k10_DIS_s132 - 1.01 s_k10_DIS_s132
                                  = 0
 w_k10_DIS_s133_return_constraint: 0.62325 w_k10_DIS_s13 - w_k10_DIS_s133
                                 + 0.99 b_k10_DIS_s133 - 1.01 s_k10_DIS_s133
                                  = 0
 w_k10_DIS_s134_return_constraint: 1.39684 w_k10_DIS_s13 - w_k10_DIS_s134
                                 + 0.99 b_k10_DIS_s134 - 1.01 s_k10_DIS_s134
                                  = 0
 w_k10_DIS_s135_return_constraint: 1.35365 w_k10_DIS_s13 - w_k10_DIS_s135
                                 + 0.99 b_k10_DIS_s135 - 1.01 s_k10_DIS_s135
                                  = 0
 w_k10_DIS_s136_return_constraint: 1.13657 w_k10_DIS_s13 - w_k10_DIS_s136
                                 + 0.99 b_k10_DIS_s136 - 1.01 s_k10_DIS_s136
                                  = 0
 w_k10_DIS_s137_return_constraint: 0.732764 w_k10_DIS_s13 - w_k10_DIS_s137
                                 + 0.99 b_k10_DIS_s137 - 1.01 s_k10_DIS_s137
                                  = 0
 w_k10_FDX_s0_return_constraint: 1.00542 w_0_FDX - w_k10_FDX_s0
                                 + 0.99 b_k10_FDX_s0 - 1.01 s_k10_FDX_s0  = 0
 w_k10_FDX_s1_return_constraint: 1.02281 w_0_FDX - w_k10_FDX_s1
                                 + 0.99 b_k10_FDX_s1 - 1.01 s_k10_FDX_s1  = 0
 w_k10_FDX_s00_return_constraint: 0.95546 w_k10_FDX_s0 - w_k10_FDX_s00
                                 + 0.99 b_k10_FDX_s00 - 1.01 s_k10_FDX_s00  = 0
 w_k10_FDX_s01_return_constraint: 0.745975 w_k10_FDX_s0 - w_k10_FDX_s01
                                 + 0.99 b_k10_FDX_s01 - 1.01 s_k10_FDX_s01  = 0
 w_k10_FDX_s02_return_constraint: 1.26676 w_k10_FDX_s0 - w_k10_FDX_s02
                                 + 0.99 b_k10_FDX_s02 - 1.01 s_k10_FDX_s02  = 0
 w_k10_FDX_s03_return_constraint: 1.13591 w_k10_FDX_s0 - w_k10_FDX_s03
                                 + 0.99 b_k10_FDX_s03 - 1.01 s_k10_FDX_s03  = 0
 w_k10_FDX_s10_return_constraint: 1.16945 w_k10_FDX_s1 - w_k10_FDX_s10
                                 + 0.99 b_k10_FDX_s10 - 1.01 s_k10_FDX_s10  = 0
 w_k10_FDX_s11_return_constraint: 0.817991 w_k10_FDX_s1 - w_k10_FDX_s11
                                 + 0.99 b_k10_FDX_s11 - 1.01 s_k10_FDX_s11  = 0
 w_k10_FDX_s12_return_constraint: 1.21143 w_k10_FDX_s1 - w_k10_FDX_s12
                                 + 0.99 b_k10_FDX_s12 - 1.01 s_k10_FDX_s12  = 0
 w_k10_FDX_s13_return_constraint: 1.33125 w_k10_FDX_s1 - w_k10_FDX_s13
                                 + 0.99 b_k10_FDX_s13 - 1.01 s_k10_FDX_s13  = 0
 w_k10_FDX_s000_return_constraint: 1.17439 w_k10_FDX_s00 - w_k10_FDX_s000
                                 + 0.99 b_k10_FDX_s000 - 1.01 s_k10_FDX_s000
                                  = 0
 w_k10_FDX_s001_return_constraint: 1.11293 w_k10_FDX_s00 - w_k10_FDX_s001
                                 + 0.99 b_k10_FDX_s001 - 1.01 s_k10_FDX_s001
                                  = 0
 w_k10_FDX_s002_return_constraint: 1.43998 w_k10_FDX_s00 - w_k10_FDX_s002
                                 + 0.99 b_k10_FDX_s002 - 1.01 s_k10_FDX_s002
                                  = 0
 w_k10_FDX_s003_return_constraint: 0.779375 w_k10_FDX_s00 - w_k10_FDX_s003
                                 + 0.99 b_k10_FDX_s003 - 1.01 s_k10_FDX_s003
                                  = 0
 w_k10_FDX_s004_return_constraint: 1.28619 w_k10_FDX_s00 - w_k10_FDX_s004
                                 + 0.99 b_k10_FDX_s004 - 1.01 s_k10_FDX_s004
                                  = 0
 w_k10_FDX_s005_return_constraint: 0.994327 w_k10_FDX_s00 - w_k10_FDX_s005
                                 + 0.99 b_k10_FDX_s005 - 1.01 s_k10_FDX_s005
                                  = 0
 w_k10_FDX_s006_return_constraint: 0.85455 w_k10_FDX_s00 - w_k10_FDX_s006
                                 + 0.99 b_k10_FDX_s006 - 1.01 s_k10_FDX_s006
                                  = 0
 w_k10_FDX_s007_return_constraint: 1.0272 w_k10_FDX_s00 - w_k10_FDX_s007
                                 + 0.99 b_k10_FDX_s007 - 1.01 s_k10_FDX_s007
                                  = 0
 w_k10_FDX_s010_return_constraint: 1.01055 w_k10_FDX_s01 - w_k10_FDX_s010
                                 + 0.99 b_k10_FDX_s010 - 1.01 s_k10_FDX_s010
                                  = 0
 w_k10_FDX_s011_return_constraint: 1.11685 w_k10_FDX_s01 - w_k10_FDX_s011
                                 + 0.99 b_k10_FDX_s011 - 1.01 s_k10_FDX_s011
                                  = 0
 w_k10_FDX_s012_return_constraint: 0.914016 w_k10_FDX_s01 - w_k10_FDX_s012
                                 + 0.99 b_k10_FDX_s012 - 1.01 s_k10_FDX_s012
                                  = 0
 w_k10_FDX_s013_return_constraint: 0.735449 w_k10_FDX_s01 - w_k10_FDX_s013
                                 + 0.99 b_k10_FDX_s013 - 1.01 s_k10_FDX_s013
                                  = 0
 w_k10_FDX_s014_return_constraint: 1.26254 w_k10_FDX_s01 - w_k10_FDX_s014
                                 + 0.99 b_k10_FDX_s014 - 1.01 s_k10_FDX_s014
                                  = 0
 w_k10_FDX_s015_return_constraint: 0.861559 w_k10_FDX_s01 - w_k10_FDX_s015
                                 + 0.99 b_k10_FDX_s015 - 1.01 s_k10_FDX_s015
                                  = 0
 w_k10_FDX_s016_return_constraint: 0.733023 w_k10_FDX_s01 - w_k10_FDX_s016
                                 + 0.99 b_k10_FDX_s016 - 1.01 s_k10_FDX_s016
                                  = 0
 w_k10_FDX_s017_return_constraint: 1.39005 w_k10_FDX_s01 - w_k10_FDX_s017
                                 + 0.99 b_k10_FDX_s017 - 1.01 s_k10_FDX_s017
                                  = 0
 w_k10_FDX_s020_return_constraint: 0.898362 w_k10_FDX_s02 - w_k10_FDX_s020
                                 + 0.99 b_k10_FDX_s020 - 1.01 s_k10_FDX_s020
                                  = 0
 w_k10_FDX_s021_return_constraint: 0.927005 w_k10_FDX_s02 - w_k10_FDX_s021
                                 + 0.99 b_k10_FDX_s021 - 1.01 s_k10_FDX_s021
                                  = 0
 w_k10_FDX_s022_return_constraint: 1.2824 w_k10_FDX_s02 - w_k10_FDX_s022
                                 + 0.99 b_k10_FDX_s022 - 1.01 s_k10_FDX_s022
                                  = 0
 w_k10_FDX_s023_return_constraint: 0.909888 w_k10_FDX_s02 - w_k10_FDX_s023
                                 + 0.99 b_k10_FDX_s023 - 1.01 s_k10_FDX_s023
                                  = 0
 w_k10_FDX_s024_return_constraint: 1.07221 w_k10_FDX_s02 - w_k10_FDX_s024
                                 + 0.99 b_k10_FDX_s024 - 1.01 s_k10_FDX_s024
                                  = 0
 w_k10_FDX_s025_return_constraint: 1.42257 w_k10_FDX_s02 - w_k10_FDX_s025
                                 + 0.99 b_k10_FDX_s025 - 1.01 s_k10_FDX_s025
                                  = 0
 w_k10_FDX_s026_return_constraint: 1.21795 w_k10_FDX_s02 - w_k10_FDX_s026
                                 + 0.99 b_k10_FDX_s026 - 1.01 s_k10_FDX_s026
                                  = 0
 w_k10_FDX_s027_return_constraint: 0.739707 w_k10_FDX_s02 - w_k10_FDX_s027
                                 + 0.99 b_k10_FDX_s027 - 1.01 s_k10_FDX_s027
                                  = 0
 w_k10_FDX_s030_return_constraint: 0.791482 w_k10_FDX_s03 - w_k10_FDX_s030
                                 + 0.99 b_k10_FDX_s030 - 1.01 s_k10_FDX_s030
                                  = 0
 w_k10_FDX_s031_return_constraint: 0.681756 w_k10_FDX_s03 - w_k10_FDX_s031
                                 + 0.99 b_k10_FDX_s031 - 1.01 s_k10_FDX_s031
                                  = 0
 w_k10_FDX_s032_return_constraint: 1.08072 w_k10_FDX_s03 - w_k10_FDX_s032
                                 + 0.99 b_k10_FDX_s032 - 1.01 s_k10_FDX_s032
                                  = 0
 w_k10_FDX_s033_return_constraint: 1.42559 w_k10_FDX_s03 - w_k10_FDX_s033
                                 + 0.99 b_k10_FDX_s033 - 1.01 s_k10_FDX_s033
                                  = 0
 w_k10_FDX_s034_return_constraint: 0.855671 w_k10_FDX_s03 - w_k10_FDX_s034
                                 + 0.99 b_k10_FDX_s034 - 1.01 s_k10_FDX_s034
                                  = 0
 w_k10_FDX_s035_return_constraint: 1.20775 w_k10_FDX_s03 - w_k10_FDX_s035
                                 + 0.99 b_k10_FDX_s035 - 1.01 s_k10_FDX_s035
                                  = 0
 w_k10_FDX_s036_return_constraint: 0.992483 w_k10_FDX_s03 - w_k10_FDX_s036
                                 + 0.99 b_k10_FDX_s036 - 1.01 s_k10_FDX_s036
                                  = 0
 w_k10_FDX_s037_return_constraint: 1.55232 w_k10_FDX_s03 - w_k10_FDX_s037
                                 + 0.99 b_k10_FDX_s037 - 1.01 s_k10_FDX_s037
                                  = 0
 w_k10_FDX_s100_return_constraint: 1.35185 w_k10_FDX_s10 - w_k10_FDX_s100
                                 + 0.99 b_k10_FDX_s100 - 1.01 s_k10_FDX_s100
                                  = 0
 w_k10_FDX_s101_return_constraint: 0.669137 w_k10_FDX_s10 - w_k10_FDX_s101
                                 + 0.99 b_k10_FDX_s101 - 1.01 s_k10_FDX_s101
                                  = 0
 w_k10_FDX_s102_return_constraint: 1.07763 w_k10_FDX_s10 - w_k10_FDX_s102
                                 + 0.99 b_k10_FDX_s102 - 1.01 s_k10_FDX_s102
                                  = 0
 w_k10_FDX_s103_return_constraint: 1.27051 w_k10_FDX_s10 - w_k10_FDX_s103
                                 + 0.99 b_k10_FDX_s103 - 1.01 s_k10_FDX_s103
                                  = 0
 w_k10_FDX_s104_return_constraint: 0.888974 w_k10_FDX_s10 - w_k10_FDX_s104
                                 + 0.99 b_k10_FDX_s104 - 1.01 s_k10_FDX_s104
                                  = 0
 w_k10_FDX_s105_return_constraint: 1.10824 w_k10_FDX_s10 - w_k10_FDX_s105
                                 + 0.99 b_k10_FDX_s105 - 1.01 s_k10_FDX_s105
                                  = 0
 w_k10_FDX_s106_return_constraint: 0.857046 w_k10_FDX_s10 - w_k10_FDX_s106
                                 + 0.99 b_k10_FDX_s106 - 1.01 s_k10_FDX_s106
                                  = 0
 w_k10_FDX_s107_return_constraint: 0.763293 w_k10_FDX_s10 - w_k10_FDX_s107
                                 + 0.99 b_k10_FDX_s107 - 1.01 s_k10_FDX_s107
                                  = 0
 w_k10_FDX_s110_return_constraint: 0.776845 w_k10_FDX_s11 - w_k10_FDX_s110
                                 + 0.99 b_k10_FDX_s110 - 1.01 s_k10_FDX_s110
                                  = 0
 w_k10_FDX_s111_return_constraint: 0.926286 w_k10_FDX_s11 - w_k10_FDX_s111
                                 + 0.99 b_k10_FDX_s111 - 1.01 s_k10_FDX_s111
                                  = 0
 w_k10_FDX_s112_return_constraint: 1.08773 w_k10_FDX_s11 - w_k10_FDX_s112
                                 + 0.99 b_k10_FDX_s112 - 1.01 s_k10_FDX_s112
                                  = 0
 w_k10_FDX_s113_return_constraint: 1.21306 w_k10_FDX_s11 - w_k10_FDX_s113
                                 + 0.99 b_k10_FDX_s113 - 1.01 s_k10_FDX_s113
                                  = 0
 w_k10_FDX_s114_return_constraint: 0.880454 w_k10_FDX_s11 - w_k10_FDX_s114
                                 + 0.99 b_k10_FDX_s114 - 1.01 s_k10_FDX_s114
                                  = 0
 w_k10_FDX_s115_return_constraint: 1.13256 w_k10_FDX_s11 - w_k10_FDX_s115
                                 + 0.99 b_k10_FDX_s115 - 1.01 s_k10_FDX_s115
                                  = 0
 w_k10_FDX_s116_return_constraint: 0.855824 w_k10_FDX_s11 - w_k10_FDX_s116
                                 + 0.99 b_k10_FDX_s116 - 1.01 s_k10_FDX_s116
                                  = 0
 w_k10_FDX_s117_return_constraint: 1.32654 w_k10_FDX_s11 - w_k10_FDX_s117
                                 + 0.99 b_k10_FDX_s117 - 1.01 s_k10_FDX_s117
                                  = 0
 w_k10_FDX_s120_return_constraint: 1.34778 w_k10_FDX_s12 - w_k10_FDX_s120
                                 + 0.99 b_k10_FDX_s120 - 1.01 s_k10_FDX_s120
                                  = 0
 w_k10_FDX_s121_return_constraint: 0.889012 w_k10_FDX_s12 - w_k10_FDX_s121
                                 + 0.99 b_k10_FDX_s121 - 1.01 s_k10_FDX_s121
                                  = 0
 w_k10_FDX_s122_return_constraint: 0.757292 w_k10_FDX_s12 - w_k10_FDX_s122
                                 + 0.99 b_k10_FDX_s122 - 1.01 s_k10_FDX_s122
                                  = 0
 w_k10_FDX_s123_return_constraint: 0.945606 w_k10_FDX_s12 - w_k10_FDX_s123
                                 + 0.99 b_k10_FDX_s123 - 1.01 s_k10_FDX_s123
                                  = 0
 w_k10_FDX_s124_return_constraint: 1.05705 w_k10_FDX_s12 - w_k10_FDX_s124
                                 + 0.99 b_k10_FDX_s124 - 1.01 s_k10_FDX_s124
                                  = 0
 w_k10_FDX_s125_return_constraint: 1.25378 w_k10_FDX_s12 - w_k10_FDX_s125
                                 + 0.99 b_k10_FDX_s125 - 1.01 s_k10_FDX_s125
                                  = 0
 w_k10_FDX_s126_return_constraint: 0.914261 w_k10_FDX_s12 - w_k10_FDX_s126
                                 + 0.99 b_k10_FDX_s126 - 1.01 s_k10_FDX_s126
                                  = 0
 w_k10_FDX_s127_return_constraint: 0.84691 w_k10_FDX_s12 - w_k10_FDX_s127
                                 + 0.99 b_k10_FDX_s127 - 1.01 s_k10_FDX_s127
                                  = 0
 w_k10_FDX_s130_return_constraint: 1.14682 w_k10_FDX_s13 - w_k10_FDX_s130
                                 + 0.99 b_k10_FDX_s130 - 1.01 s_k10_FDX_s130
                                  = 0
 w_k10_FDX_s131_return_constraint: 1.59551 w_k10_FDX_s13 - w_k10_FDX_s131
                                 + 0.99 b_k10_FDX_s131 - 1.01 s_k10_FDX_s131
                                  = 0
 w_k10_FDX_s132_return_constraint: 0.951474 w_k10_FDX_s13 - w_k10_FDX_s132
                                 + 0.99 b_k10_FDX_s132 - 1.01 s_k10_FDX_s132
                                  = 0
 w_k10_FDX_s133_return_constraint: 0.741998 w_k10_FDX_s13 - w_k10_FDX_s133
                                 + 0.99 b_k10_FDX_s133 - 1.01 s_k10_FDX_s133
                                  = 0
 w_k10_FDX_s134_return_constraint: 1.05398 w_k10_FDX_s13 - w_k10_FDX_s134
                                 + 0.99 b_k10_FDX_s134 - 1.01 s_k10_FDX_s134
                                  = 0
 w_k10_FDX_s135_return_constraint: 1.25144 w_k10_FDX_s13 - w_k10_FDX_s135
                                 + 0.99 b_k10_FDX_s135 - 1.01 s_k10_FDX_s135
                                  = 0
 w_k10_FDX_s136_return_constraint: 1.13041 w_k10_FDX_s13 - w_k10_FDX_s136
                                 + 0.99 b_k10_FDX_s136 - 1.01 s_k10_FDX_s136
                                  = 0
 w_k10_FDX_s137_return_constraint: 0.715887 w_k10_FDX_s13 - w_k10_FDX_s137
                                 + 0.99 b_k10_FDX_s137 - 1.01 s_k10_FDX_s137
                                  = 0
 w_k10_IBM_s0_return_constraint: 0.898914 w_0_IBM - w_k10_IBM_s0
                                 + 0.99 b_k10_IBM_s0 - 1.01 s_k10_IBM_s0  = 0
 w_k10_IBM_s1_return_constraint: 1.34446 w_0_IBM - w_k10_IBM_s1
                                 + 0.99 b_k10_IBM_s1 - 1.01 s_k10_IBM_s1  = 0
 w_k10_IBM_s00_return_constraint: 0.755289 w_k10_IBM_s0 - w_k10_IBM_s00
                                 + 0.99 b_k10_IBM_s00 - 1.01 s_k10_IBM_s00  = 0
 w_k10_IBM_s01_return_constraint: 1.00851 w_k10_IBM_s0 - w_k10_IBM_s01
                                 + 0.99 b_k10_IBM_s01 - 1.01 s_k10_IBM_s01  = 0
 w_k10_IBM_s02_return_constraint: 1.09659 w_k10_IBM_s0 - w_k10_IBM_s02
                                 + 0.99 b_k10_IBM_s02 - 1.01 s_k10_IBM_s02  = 0
 w_k10_IBM_s03_return_constraint: 1.5967 w_k10_IBM_s0 - w_k10_IBM_s03
                                 + 0.99 b_k10_IBM_s03 - 1.01 s_k10_IBM_s03  = 0
 w_k10_IBM_s10_return_constraint: 1.33755 w_k10_IBM_s1 - w_k10_IBM_s10
                                 + 0.99 b_k10_IBM_s10 - 1.01 s_k10_IBM_s10  = 0
 w_k10_IBM_s11_return_constraint: 0.866908 w_k10_IBM_s1 - w_k10_IBM_s11
                                 + 0.99 b_k10_IBM_s11 - 1.01 s_k10_IBM_s11  = 0
 w_k10_IBM_s12_return_constraint: 0.823234 w_k10_IBM_s1 - w_k10_IBM_s12
                                 + 0.99 b_k10_IBM_s12 - 1.01 s_k10_IBM_s12  = 0
 w_k10_IBM_s13_return_constraint: 1.074 w_k10_IBM_s1 - w_k10_IBM_s13
                                 + 0.99 b_k10_IBM_s13 - 1.01 s_k10_IBM_s13  = 0
 w_k10_IBM_s000_return_constraint: 1.10875 w_k10_IBM_s00 - w_k10_IBM_s000
                                 + 0.99 b_k10_IBM_s000 - 1.01 s_k10_IBM_s000
                                  = 0
 w_k10_IBM_s001_return_constraint: 1.62706 w_k10_IBM_s00 - w_k10_IBM_s001
                                 + 0.99 b_k10_IBM_s001 - 1.01 s_k10_IBM_s001
                                  = 0
 w_k10_IBM_s002_return_constraint: 1.36889 w_k10_IBM_s00 - w_k10_IBM_s002
                                 + 0.99 b_k10_IBM_s002 - 1.01 s_k10_IBM_s002
                                  = 0
 w_k10_IBM_s003_return_constraint: 0.699345 w_k10_IBM_s00 - w_k10_IBM_s003
                                 + 0.99 b_k10_IBM_s003 - 1.01 s_k10_IBM_s003
                                  = 0
 w_k10_IBM_s004_return_constraint: 0.896089 w_k10_IBM_s00 - w_k10_IBM_s004
                                 + 0.99 b_k10_IBM_s004 - 1.01 s_k10_IBM_s004
                                  = 0
 w_k10_IBM_s005_return_constraint: 0.875707 w_k10_IBM_s00 - w_k10_IBM_s005
                                 + 0.99 b_k10_IBM_s005 - 1.01 s_k10_IBM_s005
                                  = 0
 w_k10_IBM_s006_return_constraint: 1.18114 w_k10_IBM_s00 - w_k10_IBM_s006
                                 + 0.99 b_k10_IBM_s006 - 1.01 s_k10_IBM_s006
                                  = 0
 w_k10_IBM_s007_return_constraint: 1.09534 w_k10_IBM_s00 - w_k10_IBM_s007
                                 + 0.99 b_k10_IBM_s007 - 1.01 s_k10_IBM_s007
                                  = 0
 w_k10_IBM_s010_return_constraint: 0.733544 w_k10_IBM_s01 - w_k10_IBM_s010
                                 + 0.99 b_k10_IBM_s010 - 1.01 s_k10_IBM_s010
                                  = 0
 w_k10_IBM_s011_return_constraint: 1.05992 w_k10_IBM_s01 - w_k10_IBM_s011
                                 + 0.99 b_k10_IBM_s011 - 1.01 s_k10_IBM_s011
                                  = 0
 w_k10_IBM_s012_return_constraint: 1.40999 w_k10_IBM_s01 - w_k10_IBM_s012
                                 + 0.99 b_k10_IBM_s012 - 1.01 s_k10_IBM_s012
                                  = 0
 w_k10_IBM_s013_return_constraint: 0.576813 w_k10_IBM_s01 - w_k10_IBM_s013
                                 + 0.99 b_k10_IBM_s013 - 1.01 s_k10_IBM_s013
                                  = 0
 w_k10_IBM_s014_return_constraint: 1.16221 w_k10_IBM_s01 - w_k10_IBM_s014
                                 + 0.99 b_k10_IBM_s014 - 1.01 s_k10_IBM_s014
                                  = 0
 w_k10_IBM_s015_return_constraint: 0.941492 w_k10_IBM_s01 - w_k10_IBM_s015
                                 + 0.99 b_k10_IBM_s015 - 1.01 s_k10_IBM_s015
                                  = 0
 w_k10_IBM_s016_return_constraint: 0.904488 w_k10_IBM_s01 - w_k10_IBM_s016
                                 + 0.99 b_k10_IBM_s016 - 1.01 s_k10_IBM_s016
                                  = 0
 w_k10_IBM_s017_return_constraint: 1.50501 w_k10_IBM_s01 - w_k10_IBM_s017
                                 + 0.99 b_k10_IBM_s017 - 1.01 s_k10_IBM_s017
                                  = 0
 w_k10_IBM_s020_return_constraint: 0.906057 w_k10_IBM_s02 - w_k10_IBM_s020
                                 + 0.99 b_k10_IBM_s020 - 1.01 s_k10_IBM_s020
                                  = 0
 w_k10_IBM_s021_return_constraint: 1.54919 w_k10_IBM_s02 - w_k10_IBM_s021
                                 + 0.99 b_k10_IBM_s021 - 1.01 s_k10_IBM_s021
                                  = 0
 w_k10_IBM_s022_return_constraint: 0.808809 w_k10_IBM_s02 - w_k10_IBM_s022
                                 + 0.99 b_k10_IBM_s022 - 1.01 s_k10_IBM_s022
                                  = 0
 w_k10_IBM_s023_return_constraint: 0.653847 w_k10_IBM_s02 - w_k10_IBM_s023
                                 + 0.99 b_k10_IBM_s023 - 1.01 s_k10_IBM_s023
                                  = 0
 w_k10_IBM_s024_return_constraint: 1.02819 w_k10_IBM_s02 - w_k10_IBM_s024
                                 + 0.99 b_k10_IBM_s024 - 1.01 s_k10_IBM_s024
                                  = 0
 w_k10_IBM_s025_return_constraint: 1.54182 w_k10_IBM_s02 - w_k10_IBM_s025
                                 + 0.99 b_k10_IBM_s025 - 1.01 s_k10_IBM_s025
                                  = 0
 w_k10_IBM_s026_return_constraint: 1.1995 w_k10_IBM_s02 - w_k10_IBM_s026
                                 + 0.99 b_k10_IBM_s026 - 1.01 s_k10_IBM_s026
                                  = 0
 w_k10_IBM_s027_return_constraint: 0.943342 w_k10_IBM_s02 - w_k10_IBM_s027
                                 + 0.99 b_k10_IBM_s027 - 1.01 s_k10_IBM_s027
                                  = 0
 w_k10_IBM_s030_return_constraint: 0.702113 w_k10_IBM_s03 - w_k10_IBM_s030
                                 + 0.99 b_k10_IBM_s030 - 1.01 s_k10_IBM_s030
                                  = 0
 w_k10_IBM_s031_return_constraint: 1.03761 w_k10_IBM_s03 - w_k10_IBM_s031
                                 + 0.99 b_k10_IBM_s031 - 1.01 s_k10_IBM_s031
                                  = 0
 w_k10_IBM_s032_return_constraint: 0.966671 w_k10_IBM_s03 - w_k10_IBM_s032
                                 + 0.99 b_k10_IBM_s032 - 1.01 s_k10_IBM_s032
                                  = 0
 w_k10_IBM_s033_return_constraint: 1.13017 w_k10_IBM_s03 - w_k10_IBM_s033
                                 + 0.99 b_k10_IBM_s033 - 1.01 s_k10_IBM_s033
                                  = 0
 w_k10_IBM_s034_return_constraint: 1.48462 w_k10_IBM_s03 - w_k10_IBM_s034
                                 + 0.99 b_k10_IBM_s034 - 1.01 s_k10_IBM_s034
                                  = 0
 w_k10_IBM_s035_return_constraint: 1.39036 w_k10_IBM_s03 - w_k10_IBM_s035
                                 + 0.99 b_k10_IBM_s035 - 1.01 s_k10_IBM_s035
                                  = 0
 w_k10_IBM_s036_return_constraint: 1.01904 w_k10_IBM_s03 - w_k10_IBM_s036
                                 + 0.99 b_k10_IBM_s036 - 1.01 s_k10_IBM_s036
                                  = 0
 w_k10_IBM_s037_return_constraint: 1.57717 w_k10_IBM_s03 - w_k10_IBM_s037
                                 + 0.99 b_k10_IBM_s037 - 1.01 s_k10_IBM_s037
                                  = 0
 w_k10_IBM_s100_return_constraint: 1.13767 w_k10_IBM_s10 - w_k10_IBM_s100
                                 + 0.99 b_k10_IBM_s100 - 1.01 s_k10_IBM_s100
                                  = 0
 w_k10_IBM_s101_return_constraint: 0.857091 w_k10_IBM_s10 - w_k10_IBM_s101
                                 + 0.99 b_k10_IBM_s101 - 1.01 s_k10_IBM_s101
                                  = 0
 w_k10_IBM_s102_return_constraint: 1.13211 w_k10_IBM_s10 - w_k10_IBM_s102
                                 + 0.99 b_k10_IBM_s102 - 1.01 s_k10_IBM_s102
                                  = 0
 w_k10_IBM_s103_return_constraint: 1.63439 w_k10_IBM_s10 - w_k10_IBM_s103
                                 + 0.99 b_k10_IBM_s103 - 1.01 s_k10_IBM_s103
                                  = 0
 w_k10_IBM_s104_return_constraint: 1.16753 w_k10_IBM_s10 - w_k10_IBM_s104
                                 + 0.99 b_k10_IBM_s104 - 1.01 s_k10_IBM_s104
                                  = 0
 w_k10_IBM_s105_return_constraint: 0.776324 w_k10_IBM_s10 - w_k10_IBM_s105
                                 + 0.99 b_k10_IBM_s105 - 1.01 s_k10_IBM_s105
                                  = 0
 w_k10_IBM_s106_return_constraint: 0.843984 w_k10_IBM_s10 - w_k10_IBM_s106
                                 + 0.99 b_k10_IBM_s106 - 1.01 s_k10_IBM_s106
                                  = 0
 w_k10_IBM_s107_return_constraint: 1.31143 w_k10_IBM_s10 - w_k10_IBM_s107
                                 + 0.99 b_k10_IBM_s107 - 1.01 s_k10_IBM_s107
                                  = 0
 w_k10_IBM_s110_return_constraint: 0.773327 w_k10_IBM_s11 - w_k10_IBM_s110
                                 + 0.99 b_k10_IBM_s110 - 1.01 s_k10_IBM_s110
                                  = 0
 w_k10_IBM_s111_return_constraint: 1.35371 w_k10_IBM_s11 - w_k10_IBM_s111
                                 + 0.99 b_k10_IBM_s111 - 1.01 s_k10_IBM_s111
                                  = 0
 w_k10_IBM_s112_return_constraint: 0.63794 w_k10_IBM_s11 - w_k10_IBM_s112
                                 + 0.99 b_k10_IBM_s112 - 1.01 s_k10_IBM_s112
                                  = 0
 w_k10_IBM_s113_return_constraint: 1.72139 w_k10_IBM_s11 - w_k10_IBM_s113
                                 + 0.99 b_k10_IBM_s113 - 1.01 s_k10_IBM_s113
                                  = 0
 w_k10_IBM_s114_return_constraint: 0.913127 w_k10_IBM_s11 - w_k10_IBM_s114
                                 + 0.99 b_k10_IBM_s114 - 1.01 s_k10_IBM_s114
                                  = 0
 w_k10_IBM_s115_return_constraint: 1.14327 w_k10_IBM_s11 - w_k10_IBM_s115
                                 + 0.99 b_k10_IBM_s115 - 1.01 s_k10_IBM_s115
                                  = 0
 w_k10_IBM_s116_return_constraint: 1.25062 w_k10_IBM_s11 - w_k10_IBM_s116
                                 + 0.99 b_k10_IBM_s116 - 1.01 s_k10_IBM_s116
                                  = 0
 w_k10_IBM_s117_return_constraint: 1.03062 w_k10_IBM_s11 - w_k10_IBM_s117
                                 + 0.99 b_k10_IBM_s117 - 1.01 s_k10_IBM_s117
                                  = 0
 w_k10_IBM_s120_return_constraint: 0.96982 w_k10_IBM_s12 - w_k10_IBM_s120
                                 + 0.99 b_k10_IBM_s120 - 1.01 s_k10_IBM_s120
                                  = 0
 w_k10_IBM_s121_return_constraint: 0.971733 w_k10_IBM_s12 - w_k10_IBM_s121
                                 + 0.99 b_k10_IBM_s121 - 1.01 s_k10_IBM_s121
                                  = 0
 w_k10_IBM_s122_return_constraint: 0.870224 w_k10_IBM_s12 - w_k10_IBM_s122
                                 + 0.99 b_k10_IBM_s122 - 1.01 s_k10_IBM_s122
                                  = 0
 w_k10_IBM_s123_return_constraint: 1.50498 w_k10_IBM_s12 - w_k10_IBM_s123
                                 + 0.99 b_k10_IBM_s123 - 1.01 s_k10_IBM_s123
                                  = 0
 w_k10_IBM_s124_return_constraint: 0.95813 w_k10_IBM_s12 - w_k10_IBM_s124
                                 + 0.99 b_k10_IBM_s124 - 1.01 s_k10_IBM_s124
                                  = 0
 w_k10_IBM_s125_return_constraint: 1.43028 w_k10_IBM_s12 - w_k10_IBM_s125
                                 + 0.99 b_k10_IBM_s125 - 1.01 s_k10_IBM_s125
                                  = 0
 w_k10_IBM_s126_return_constraint: 1.10289 w_k10_IBM_s12 - w_k10_IBM_s126
                                 + 0.99 b_k10_IBM_s126 - 1.01 s_k10_IBM_s126
                                  = 0
 w_k10_IBM_s127_return_constraint: 0.618888 w_k10_IBM_s12 - w_k10_IBM_s127
                                 + 0.99 b_k10_IBM_s127 - 1.01 s_k10_IBM_s127
                                  = 0
 w_k10_IBM_s130_return_constraint: 0.854295 w_k10_IBM_s13 - w_k10_IBM_s130
                                 + 0.99 b_k10_IBM_s130 - 1.01 s_k10_IBM_s130
                                  = 0
 w_k10_IBM_s131_return_constraint: 1.43547 w_k10_IBM_s13 - w_k10_IBM_s131
                                 + 0.99 b_k10_IBM_s131 - 1.01 s_k10_IBM_s131
                                  = 0
 w_k10_IBM_s132_return_constraint: 0.832703 w_k10_IBM_s13 - w_k10_IBM_s132
                                 + 0.99 b_k10_IBM_s132 - 1.01 s_k10_IBM_s132
                                  = 0
 w_k10_IBM_s133_return_constraint: 0.995969 w_k10_IBM_s13 - w_k10_IBM_s133
                                 + 0.99 b_k10_IBM_s133 - 1.01 s_k10_IBM_s133
                                  = 0
 w_k10_IBM_s134_return_constraint: 1.30554 w_k10_IBM_s13 - w_k10_IBM_s134
                                 + 0.99 b_k10_IBM_s134 - 1.01 s_k10_IBM_s134
                                  = 0
 w_k10_IBM_s135_return_constraint: 0.884124 w_k10_IBM_s13 - w_k10_IBM_s135
                                 + 0.99 b_k10_IBM_s135 - 1.01 s_k10_IBM_s135
                                  = 0
 w_k10_IBM_s136_return_constraint: 1.26062 w_k10_IBM_s13 - w_k10_IBM_s136
                                 + 0.99 b_k10_IBM_s136 - 1.01 s_k10_IBM_s136
                                  = 0
 w_k10_IBM_s137_return_constraint: 0.818865 w_k10_IBM_s13 - w_k10_IBM_s137
                                 + 0.99 b_k10_IBM_s137 - 1.01 s_k10_IBM_s137
                                  = 0
 w_k10_INTC_s0_return_constraint: 1.01985 w_0_INTC - w_k10_INTC_s0
                                 + 0.99 b_k10_INTC_s0 - 1.01 s_k10_INTC_s0  = 0
 w_k10_INTC_s1_return_constraint: 1.01853 w_0_INTC - w_k10_INTC_s1
                                 + 0.99 b_k10_INTC_s1 - 1.01 s_k10_INTC_s1  = 0
 w_k10_INTC_s00_return_constraint: 0.948018 w_k10_INTC_s0 - w_k10_INTC_s00
                                 + 0.99 b_k10_INTC_s00 - 1.01 s_k10_INTC_s00
                                  = 0
 w_k10_INTC_s01_return_constraint: 0.883803 w_k10_INTC_s0 - w_k10_INTC_s01
                                 + 0.99 b_k10_INTC_s01 - 1.01 s_k10_INTC_s01
                                  = 0
 w_k10_INTC_s02_return_constraint: 1.08779 w_k10_INTC_s0 - w_k10_INTC_s02
                                 + 0.99 b_k10_INTC_s02 - 1.01 s_k10_INTC_s02
                                  = 0
 w_k10_INTC_s03_return_constraint: 1.15394 w_k10_INTC_s0 - w_k10_INTC_s03
                                 + 0.99 b_k10_INTC_s03 - 1.01 s_k10_INTC_s03
                                  = 0
 w_k10_INTC_s10_return_constraint: 1.16409 w_k10_INTC_s1 - w_k10_INTC_s10
                                 + 0.99 b_k10_INTC_s10 - 1.01 s_k10_INTC_s10
                                  = 0
 w_k10_INTC_s11_return_constraint: 0.859949 w_k10_INTC_s1 - w_k10_INTC_s11
                                 + 0.99 b_k10_INTC_s11 - 1.01 s_k10_INTC_s11
                                  = 0
 w_k10_INTC_s12_return_constraint: 1.32982 w_k10_INTC_s1 - w_k10_INTC_s12
                                 + 0.99 b_k10_INTC_s12 - 1.01 s_k10_INTC_s12
                                  = 0
 w_k10_INTC_s13_return_constraint: 1.20417 w_k10_INTC_s1 - w_k10_INTC_s13
                                 + 0.99 b_k10_INTC_s13 - 1.01 s_k10_INTC_s13
                                  = 0
 w_k10_INTC_s000_return_constraint: 1.1005 w_k10_INTC_s00 - w_k10_INTC_s000
                                 + 0.99 b_k10_INTC_s000 - 1.01 s_k10_INTC_s000
                                  = 0
 w_k10_INTC_s001_return_constraint: 1.12601 w_k10_INTC_s00 - w_k10_INTC_s001
                                 + 0.99 b_k10_INTC_s001 - 1.01 s_k10_INTC_s001
                                  = 0
 w_k10_INTC_s002_return_constraint: 1.22261 w_k10_INTC_s00 - w_k10_INTC_s002
                                 + 0.99 b_k10_INTC_s002 - 1.01 s_k10_INTC_s002
                                  = 0
 w_k10_INTC_s003_return_constraint: 0.767146 w_k10_INTC_s00 - w_k10_INTC_s003
                                 + 0.99 b_k10_INTC_s003 - 1.01 s_k10_INTC_s003
                                  = 0
 w_k10_INTC_s004_return_constraint: 0.936795 w_k10_INTC_s00 - w_k10_INTC_s004
                                 + 0.99 b_k10_INTC_s004 - 1.01 s_k10_INTC_s004
                                  = 0
 w_k10_INTC_s005_return_constraint: 0.94309 w_k10_INTC_s00 - w_k10_INTC_s005
                                 + 0.99 b_k10_INTC_s005 - 1.01 s_k10_INTC_s005
                                  = 0
 w_k10_INTC_s006_return_constraint: 0.866716 w_k10_INTC_s00 - w_k10_INTC_s006
                                 + 0.99 b_k10_INTC_s006 - 1.01 s_k10_INTC_s006
                                  = 0
 w_k10_INTC_s007_return_constraint: 1.03274 w_k10_INTC_s00 - w_k10_INTC_s007
                                 + 0.99 b_k10_INTC_s007 - 1.01 s_k10_INTC_s007
                                  = 0
 w_k10_INTC_s010_return_constraint: 0.962068 w_k10_INTC_s01 - w_k10_INTC_s010
                                 + 0.99 b_k10_INTC_s010 - 1.01 s_k10_INTC_s010
                                  = 0
 w_k10_INTC_s011_return_constraint: 0.978546 w_k10_INTC_s01 - w_k10_INTC_s011
                                 + 0.99 b_k10_INTC_s011 - 1.01 s_k10_INTC_s011
                                  = 0
 w_k10_INTC_s012_return_constraint: 0.798941 w_k10_INTC_s01 - w_k10_INTC_s012
                                 + 0.99 b_k10_INTC_s012 - 1.01 s_k10_INTC_s012
                                  = 0
 w_k10_INTC_s013_return_constraint: 0.689157 w_k10_INTC_s01 - w_k10_INTC_s013
                                 + 0.99 b_k10_INTC_s013 - 1.01 s_k10_INTC_s013
                                  = 0
 w_k10_INTC_s014_return_constraint: 1.10844 w_k10_INTC_s01 - w_k10_INTC_s014
                                 + 0.99 b_k10_INTC_s014 - 1.01 s_k10_INTC_s014
                                  = 0
 w_k10_INTC_s015_return_constraint: 0.82142 w_k10_INTC_s01 - w_k10_INTC_s015
                                 + 0.99 b_k10_INTC_s015 - 1.01 s_k10_INTC_s015
                                  = 0
 w_k10_INTC_s016_return_constraint: 0.733869 w_k10_INTC_s01 - w_k10_INTC_s016
                                 + 0.99 b_k10_INTC_s016 - 1.01 s_k10_INTC_s016
                                  = 0
 w_k10_INTC_s017_return_constraint: 1.15277 w_k10_INTC_s01 - w_k10_INTC_s017
                                 + 0.99 b_k10_INTC_s017 - 1.01 s_k10_INTC_s017
                                  = 0
 w_k10_INTC_s020_return_constraint: 0.935454 w_k10_INTC_s02 - w_k10_INTC_s020
                                 + 0.99 b_k10_INTC_s020 - 1.01 s_k10_INTC_s020
                                  = 0
 w_k10_INTC_s021_return_constraint: 1.07696 w_k10_INTC_s02 - w_k10_INTC_s021
                                 + 0.99 b_k10_INTC_s021 - 1.01 s_k10_INTC_s021
                                  = 0
 w_k10_INTC_s022_return_constraint: 1.25026 w_k10_INTC_s02 - w_k10_INTC_s022
                                 + 0.99 b_k10_INTC_s022 - 1.01 s_k10_INTC_s022
                                  = 0
 w_k10_INTC_s023_return_constraint: 0.981466 w_k10_INTC_s02 - w_k10_INTC_s023
                                 + 0.99 b_k10_INTC_s023 - 1.01 s_k10_INTC_s023
                                  = 0
 w_k10_INTC_s024_return_constraint: 1.08147 w_k10_INTC_s02 - w_k10_INTC_s024
                                 + 0.99 b_k10_INTC_s024 - 1.01 s_k10_INTC_s024
                                  = 0
 w_k10_INTC_s025_return_constraint: 1.45447 w_k10_INTC_s02 - w_k10_INTC_s025
                                 + 0.99 b_k10_INTC_s025 - 1.01 s_k10_INTC_s025
                                  = 0
 w_k10_INTC_s026_return_constraint: 1.14932 w_k10_INTC_s02 - w_k10_INTC_s026
                                 + 0.99 b_k10_INTC_s026 - 1.01 s_k10_INTC_s026
                                  = 0
 w_k10_INTC_s027_return_constraint: 0.931094 w_k10_INTC_s02 - w_k10_INTC_s027
                                 + 0.99 b_k10_INTC_s027 - 1.01 s_k10_INTC_s027
                                  = 0
 w_k10_INTC_s030_return_constraint: 0.664553 w_k10_INTC_s03 - w_k10_INTC_s030
                                 + 0.99 b_k10_INTC_s030 - 1.01 s_k10_INTC_s030
                                  = 0
 w_k10_INTC_s031_return_constraint: 0.731186 w_k10_INTC_s03 - w_k10_INTC_s031
                                 + 0.99 b_k10_INTC_s031 - 1.01 s_k10_INTC_s031
                                  = 0
 w_k10_INTC_s032_return_constraint: 0.968509 w_k10_INTC_s03 - w_k10_INTC_s032
                                 + 0.99 b_k10_INTC_s032 - 1.01 s_k10_INTC_s032
                                  = 0
 w_k10_INTC_s033_return_constraint: 1.28563 w_k10_INTC_s03 - w_k10_INTC_s033
                                 + 0.99 b_k10_INTC_s033 - 1.01 s_k10_INTC_s033
                                  = 0
 w_k10_INTC_s034_return_constraint: 1.05097 w_k10_INTC_s03 - w_k10_INTC_s034
                                 + 0.99 b_k10_INTC_s034 - 1.01 s_k10_INTC_s034
                                  = 0
 w_k10_INTC_s035_return_constraint: 1.18805 w_k10_INTC_s03 - w_k10_INTC_s035
                                 + 0.99 b_k10_INTC_s035 - 1.01 s_k10_INTC_s035
                                  = 0
 w_k10_INTC_s036_return_constraint: 0.930162 w_k10_INTC_s03 - w_k10_INTC_s036
                                 + 0.99 b_k10_INTC_s036 - 1.01 s_k10_INTC_s036
                                  = 0
 w_k10_INTC_s037_return_constraint: 1.47298 w_k10_INTC_s03 - w_k10_INTC_s037
                                 + 0.99 b_k10_INTC_s037 - 1.01 s_k10_INTC_s037
                                  = 0
 w_k10_INTC_s100_return_constraint: 1.32975 w_k10_INTC_s10 - w_k10_INTC_s100
                                 + 0.99 b_k10_INTC_s100 - 1.01 s_k10_INTC_s100
                                  = 0
 w_k10_INTC_s101_return_constraint: 0.758211 w_k10_INTC_s10 - w_k10_INTC_s101
                                 + 0.99 b_k10_INTC_s101 - 1.01 s_k10_INTC_s101
                                  = 0
 w_k10_INTC_s102_return_constraint: 1.07032 w_k10_INTC_s10 - w_k10_INTC_s102
                                 + 0.99 b_k10_INTC_s102 - 1.01 s_k10_INTC_s102
                                  = 0
 w_k10_INTC_s103_return_constraint: 1.22684 w_k10_INTC_s10 - w_k10_INTC_s103
                                 + 0.99 b_k10_INTC_s103 - 1.01 s_k10_INTC_s103
                                  = 0
 w_k10_INTC_s104_return_constraint: 0.881407 w_k10_INTC_s10 - w_k10_INTC_s104
                                 + 0.99 b_k10_INTC_s104 - 1.01 s_k10_INTC_s104
                                  = 0
 w_k10_INTC_s105_return_constraint: 1.14697 w_k10_INTC_s10 - w_k10_INTC_s105
                                 + 0.99 b_k10_INTC_s105 - 1.01 s_k10_INTC_s105
                                  = 0
 w_k10_INTC_s106_return_constraint: 0.878265 w_k10_INTC_s10 - w_k10_INTC_s106
                                 + 0.99 b_k10_INTC_s106 - 1.01 s_k10_INTC_s106
                                  = 0
 w_k10_INTC_s107_return_constraint: 1.03504 w_k10_INTC_s10 - w_k10_INTC_s107
                                 + 0.99 b_k10_INTC_s107 - 1.01 s_k10_INTC_s107
                                  = 0
 w_k10_INTC_s110_return_constraint: 0.817143 w_k10_INTC_s11 - w_k10_INTC_s110
                                 + 0.99 b_k10_INTC_s110 - 1.01 s_k10_INTC_s110
                                  = 0
 w_k10_INTC_s111_return_constraint: 0.913546 w_k10_INTC_s11 - w_k10_INTC_s111
                                 + 0.99 b_k10_INTC_s111 - 1.01 s_k10_INTC_s111
                                  = 0
 w_k10_INTC_s112_return_constraint: 1.13059 w_k10_INTC_s11 - w_k10_INTC_s112
                                 + 0.99 b_k10_INTC_s112 - 1.01 s_k10_INTC_s112
                                  = 0
 w_k10_INTC_s113_return_constraint: 1.45881 w_k10_INTC_s11 - w_k10_INTC_s113
                                 + 0.99 b_k10_INTC_s113 - 1.01 s_k10_INTC_s113
                                  = 0
 w_k10_INTC_s114_return_constraint: 0.954665 w_k10_INTC_s11 - w_k10_INTC_s114
                                 + 0.99 b_k10_INTC_s114 - 1.01 s_k10_INTC_s114
                                  = 0
 w_k10_INTC_s115_return_constraint: 1.06907 w_k10_INTC_s11 - w_k10_INTC_s115
                                 + 0.99 b_k10_INTC_s115 - 1.01 s_k10_INTC_s115
                                  = 0
 w_k10_INTC_s116_return_constraint: 1.02311 w_k10_INTC_s11 - w_k10_INTC_s116
                                 + 0.99 b_k10_INTC_s116 - 1.01 s_k10_INTC_s116
                                  = 0
 w_k10_INTC_s117_return_constraint: 1.20525 w_k10_INTC_s11 - w_k10_INTC_s117
                                 + 0.99 b_k10_INTC_s117 - 1.01 s_k10_INTC_s117
                                  = 0
 w_k10_INTC_s120_return_constraint: 1.16567 w_k10_INTC_s12 - w_k10_INTC_s120
                                 + 0.99 b_k10_INTC_s120 - 1.01 s_k10_INTC_s120
                                  = 0
 w_k10_INTC_s121_return_constraint: 0.820141 w_k10_INTC_s12 - w_k10_INTC_s121
                                 + 0.99 b_k10_INTC_s121 - 1.01 s_k10_INTC_s121
                                  = 0
 w_k10_INTC_s122_return_constraint: 0.773365 w_k10_INTC_s12 - w_k10_INTC_s122
                                 + 0.99 b_k10_INTC_s122 - 1.01 s_k10_INTC_s122
                                  = 0
 w_k10_INTC_s123_return_constraint: 0.933317 w_k10_INTC_s12 - w_k10_INTC_s123
                                 + 0.99 b_k10_INTC_s123 - 1.01 s_k10_INTC_s123
                                  = 0
 w_k10_INTC_s124_return_constraint: 0.983127 w_k10_INTC_s12 - w_k10_INTC_s124
                                 + 0.99 b_k10_INTC_s124 - 1.01 s_k10_INTC_s124
                                  = 0
 w_k10_INTC_s125_return_constraint: 1.06194 w_k10_INTC_s12 - w_k10_INTC_s125
                                 + 0.99 b_k10_INTC_s125 - 1.01 s_k10_INTC_s125
                                  = 0
 w_k10_INTC_s126_return_constraint: 0.807338 w_k10_INTC_s12 - w_k10_INTC_s126
                                 + 0.99 b_k10_INTC_s126 - 1.01 s_k10_INTC_s126
                                  = 0
 w_k10_INTC_s127_return_constraint: 0.766009 w_k10_INTC_s12 - w_k10_INTC_s127
                                 + 0.99 b_k10_INTC_s127 - 1.01 s_k10_INTC_s127
                                  = 0
 w_k10_INTC_s130_return_constraint: 1.03236 w_k10_INTC_s13 - w_k10_INTC_s130
                                 + 0.99 b_k10_INTC_s130 - 1.01 s_k10_INTC_s130
                                  = 0
 w_k10_INTC_s131_return_constraint: 1.40028 w_k10_INTC_s13 - w_k10_INTC_s131
                                 + 0.99 b_k10_INTC_s131 - 1.01 s_k10_INTC_s131
                                  = 0
 w_k10_INTC_s132_return_constraint: 0.93239 w_k10_INTC_s13 - w_k10_INTC_s132
                                 + 0.99 b_k10_INTC_s132 - 1.01 s_k10_INTC_s132
                                  = 0
 w_k10_INTC_s133_return_constraint: 0.820611 w_k10_INTC_s13 - w_k10_INTC_s133
                                 + 0.99 b_k10_INTC_s133 - 1.01 s_k10_INTC_s133
                                  = 0
 w_k10_INTC_s134_return_constraint: 1.26323 w_k10_INTC_s13 - w_k10_INTC_s134
                                 + 0.99 b_k10_INTC_s134 - 1.01 s_k10_INTC_s134
                                  = 0
 w_k10_INTC_s135_return_constraint: 1.4174 w_k10_INTC_s13 - w_k10_INTC_s135
                                 + 0.99 b_k10_INTC_s135 - 1.01 s_k10_INTC_s135
                                  = 0
 w_k10_INTC_s136_return_constraint: 1.1817 w_k10_INTC_s13 - w_k10_INTC_s136
                                 + 0.99 b_k10_INTC_s136 - 1.01 s_k10_INTC_s136
                                  = 0
 w_k10_INTC_s137_return_constraint: 0.923104 w_k10_INTC_s13 - w_k10_INTC_s137
                                 + 0.99 b_k10_INTC_s137 - 1.01 s_k10_INTC_s137
                                  = 0
 w_k10_KO_s0_return_constraint:  0.919594 w_0_KO - w_k10_KO_s0
                                 + 0.99 b_k10_KO_s0 - 1.01 s_k10_KO_s0  = 0
 w_k10_KO_s1_return_constraint:  1.18722 w_0_KO - w_k10_KO_s1
                                 + 0.99 b_k10_KO_s1 - 1.01 s_k10_KO_s1  = 0
 w_k10_KO_s00_return_constraint: 0.845395 w_k10_KO_s0 - w_k10_KO_s00
                                 + 0.99 b_k10_KO_s00 - 1.01 s_k10_KO_s00  = 0
 w_k10_KO_s01_return_constraint: 0.953203 w_k10_KO_s0 - w_k10_KO_s01
                                 + 0.99 b_k10_KO_s01 - 1.01 s_k10_KO_s01  = 0
 w_k10_KO_s02_return_constraint: 1.14332 w_k10_KO_s0 - w_k10_KO_s02
                                 + 0.99 b_k10_KO_s02 - 1.01 s_k10_KO_s02  = 0
 w_k10_KO_s03_return_constraint: 1.45167 w_k10_KO_s0 - w_k10_KO_s03
                                 + 0.99 b_k10_KO_s03 - 1.01 s_k10_KO_s03  = 0
 w_k10_KO_s10_return_constraint: 1.25756 w_k10_KO_s1 - w_k10_KO_s10
                                 + 0.99 b_k10_KO_s10 - 1.01 s_k10_KO_s10  = 0
 w_k10_KO_s11_return_constraint: 0.83917 w_k10_KO_s1 - w_k10_KO_s11
                                 + 0.99 b_k10_KO_s11 - 1.01 s_k10_KO_s11  = 0
 w_k10_KO_s12_return_constraint: 1.0536 w_k10_KO_s1 - w_k10_KO_s12
                                 + 0.99 b_k10_KO_s12 - 1.01 s_k10_KO_s12  = 0
 w_k10_KO_s13_return_constraint: 1.25227 w_k10_KO_s1 - w_k10_KO_s13
                                 + 0.99 b_k10_KO_s13 - 1.01 s_k10_KO_s13  = 0
 w_k10_KO_s000_return_constraint: 1.25724 w_k10_KO_s00 - w_k10_KO_s000
                                 + 0.99 b_k10_KO_s000 - 1.01 s_k10_KO_s000  = 0
 w_k10_KO_s001_return_constraint: 1.425 w_k10_KO_s00 - w_k10_KO_s001
                                 + 0.99 b_k10_KO_s001 - 1.01 s_k10_KO_s001  = 0
 w_k10_KO_s002_return_constraint: 1.45194 w_k10_KO_s00 - w_k10_KO_s002
                                 + 0.99 b_k10_KO_s002 - 1.01 s_k10_KO_s002  = 0
 w_k10_KO_s003_return_constraint: 0.772644 w_k10_KO_s00 - w_k10_KO_s003
                                 + 0.99 b_k10_KO_s003 - 1.01 s_k10_KO_s003  = 0
 w_k10_KO_s004_return_constraint: 1.07762 w_k10_KO_s00 - w_k10_KO_s004
                                 + 0.99 b_k10_KO_s004 - 1.01 s_k10_KO_s004  = 0
 w_k10_KO_s005_return_constraint: 0.919223 w_k10_KO_s00 - w_k10_KO_s005
                                 + 0.99 b_k10_KO_s005 - 1.01 s_k10_KO_s005  = 0
 w_k10_KO_s006_return_constraint: 1.064 w_k10_KO_s00 - w_k10_KO_s006
                                 + 0.99 b_k10_KO_s006 - 1.01 s_k10_KO_s006  = 0
 w_k10_KO_s007_return_constraint: 1.19637 w_k10_KO_s00 - w_k10_KO_s007
                                 + 0.99 b_k10_KO_s007 - 1.01 s_k10_KO_s007  = 0
 w_k10_KO_s010_return_constraint: 0.777283 w_k10_KO_s01 - w_k10_KO_s010
                                 + 0.99 b_k10_KO_s010 - 1.01 s_k10_KO_s010  = 0
 w_k10_KO_s011_return_constraint: 1.00446 w_k10_KO_s01 - w_k10_KO_s011
                                 + 0.99 b_k10_KO_s011 - 1.01 s_k10_KO_s011  = 0
 w_k10_KO_s012_return_constraint: 1.01114 w_k10_KO_s01 - w_k10_KO_s012
                                 + 0.99 b_k10_KO_s012 - 1.01 s_k10_KO_s012  = 0
 w_k10_KO_s013_return_constraint: 0.599254 w_k10_KO_s01 - w_k10_KO_s013
                                 + 0.99 b_k10_KO_s013 - 1.01 s_k10_KO_s013  = 0
 w_k10_KO_s014_return_constraint: 1.06916 w_k10_KO_s01 - w_k10_KO_s014
                                 + 0.99 b_k10_KO_s014 - 1.01 s_k10_KO_s014  = 0
 w_k10_KO_s015_return_constraint: 0.896809 w_k10_KO_s01 - w_k10_KO_s015
                                 + 0.99 b_k10_KO_s015 - 1.01 s_k10_KO_s015  = 0
 w_k10_KO_s016_return_constraint: 0.787835 w_k10_KO_s01 - w_k10_KO_s016
                                 + 0.99 b_k10_KO_s016 - 1.01 s_k10_KO_s016  = 0
 w_k10_KO_s017_return_constraint: 1.54929 w_k10_KO_s01 - w_k10_KO_s017
                                 + 0.99 b_k10_KO_s017 - 1.01 s_k10_KO_s017  = 0
 w_k10_KO_s020_return_constraint: 1.03965 w_k10_KO_s02 - w_k10_KO_s020
                                 + 0.99 b_k10_KO_s020 - 1.01 s_k10_KO_s020  = 0
 w_k10_KO_s021_return_constraint: 1.3756 w_k10_KO_s02 - w_k10_KO_s021
                                 + 0.99 b_k10_KO_s021 - 1.01 s_k10_KO_s021  = 0
 w_k10_KO_s022_return_constraint: 1.07364 w_k10_KO_s02 - w_k10_KO_s022
                                 + 0.99 b_k10_KO_s022 - 1.01 s_k10_KO_s022  = 0
 w_k10_KO_s023_return_constraint: 0.783297 w_k10_KO_s02 - w_k10_KO_s023
                                 + 0.99 b_k10_KO_s023 - 1.01 s_k10_KO_s023  = 0
 w_k10_KO_s024_return_constraint: 1.16442 w_k10_KO_s02 - w_k10_KO_s024
                                 + 0.99 b_k10_KO_s024 - 1.01 s_k10_KO_s024  = 0
 w_k10_KO_s025_return_constraint: 1.57759 w_k10_KO_s02 - w_k10_KO_s025
                                 + 0.99 b_k10_KO_s025 - 1.01 s_k10_KO_s025  = 0
 w_k10_KO_s026_return_constraint: 1.20198 w_k10_KO_s02 - w_k10_KO_s026
                                 + 0.99 b_k10_KO_s026 - 1.01 s_k10_KO_s026  = 0
 w_k10_KO_s027_return_constraint: 1.02747 w_k10_KO_s02 - w_k10_KO_s027
                                 + 0.99 b_k10_KO_s027 - 1.01 s_k10_KO_s027  = 0
 w_k10_KO_s030_return_constraint: 0.705889 w_k10_KO_s03 - w_k10_KO_s030
                                 + 0.99 b_k10_KO_s030 - 1.01 s_k10_KO_s030  = 0
 w_k10_KO_s031_return_constraint: 0.881254 w_k10_KO_s03 - w_k10_KO_s031
                                 + 0.99 b_k10_KO_s031 - 1.01 s_k10_KO_s031  = 0
 w_k10_KO_s032_return_constraint: 0.951837 w_k10_KO_s03 - w_k10_KO_s032
                                 + 0.99 b_k10_KO_s032 - 1.01 s_k10_KO_s032  = 0
 w_k10_KO_s033_return_constraint: 1.24479 w_k10_KO_s03 - w_k10_KO_s033
                                 + 0.99 b_k10_KO_s033 - 1.01 s_k10_KO_s033  = 0
 w_k10_KO_s034_return_constraint: 1.07349 w_k10_KO_s03 - w_k10_KO_s034
                                 + 0.99 b_k10_KO_s034 - 1.01 s_k10_KO_s034  = 0
 w_k10_KO_s035_return_constraint: 1.41636 w_k10_KO_s03 - w_k10_KO_s035
                                 + 0.99 b_k10_KO_s035 - 1.01 s_k10_KO_s035  = 0
 w_k10_KO_s036_return_constraint: 1.03837 w_k10_KO_s03 - w_k10_KO_s036
                                 + 0.99 b_k10_KO_s036 - 1.01 s_k10_KO_s036  = 0
 w_k10_KO_s037_return_constraint: 1.61226 w_k10_KO_s03 - w_k10_KO_s037
                                 + 0.99 b_k10_KO_s037 - 1.01 s_k10_KO_s037  = 0
 w_k10_KO_s100_return_constraint: 1.39 w_k10_KO_s10 - w_k10_KO_s100
                                 + 0.99 b_k10_KO_s100 - 1.01 s_k10_KO_s100  = 0
 w_k10_KO_s101_return_constraint: 0.765969 w_k10_KO_s10 - w_k10_KO_s101
                                 + 0.99 b_k10_KO_s101 - 1.01 s_k10_KO_s101  = 0
 w_k10_KO_s102_return_constraint: 1.05536 w_k10_KO_s10 - w_k10_KO_s102
                                 + 0.99 b_k10_KO_s102 - 1.01 s_k10_KO_s102  = 0
 w_k10_KO_s103_return_constraint: 1.39045 w_k10_KO_s10 - w_k10_KO_s103
                                 + 0.99 b_k10_KO_s103 - 1.01 s_k10_KO_s103  = 0
 w_k10_KO_s104_return_constraint: 1.15799 w_k10_KO_s10 - w_k10_KO_s104
                                 + 0.99 b_k10_KO_s104 - 1.01 s_k10_KO_s104  = 0
 w_k10_KO_s105_return_constraint: 0.862656 w_k10_KO_s10 - w_k10_KO_s105
                                 + 0.99 b_k10_KO_s105 - 1.01 s_k10_KO_s105  = 0
 w_k10_KO_s106_return_constraint: 0.944068 w_k10_KO_s10 - w_k10_KO_s106
                                 + 0.99 b_k10_KO_s106 - 1.01 s_k10_KO_s106  = 0
 w_k10_KO_s107_return_constraint: 1.00069 w_k10_KO_s10 - w_k10_KO_s107
                                 + 0.99 b_k10_KO_s107 - 1.01 s_k10_KO_s107  = 0
 w_k10_KO_s110_return_constraint: 0.65971 w_k10_KO_s11 - w_k10_KO_s110
                                 + 0.99 b_k10_KO_s110 - 1.01 s_k10_KO_s110  = 0
 w_k10_KO_s111_return_constraint: 0.940895 w_k10_KO_s11 - w_k10_KO_s111
                                 + 0.99 b_k10_KO_s111 - 1.01 s_k10_KO_s111  = 0
 w_k10_KO_s112_return_constraint: 0.839253 w_k10_KO_s11 - w_k10_KO_s112
                                 + 0.99 b_k10_KO_s112 - 1.01 s_k10_KO_s112  = 0
 w_k10_KO_s113_return_constraint: 1.47185 w_k10_KO_s11 - w_k10_KO_s113
                                 + 0.99 b_k10_KO_s113 - 1.01 s_k10_KO_s113  = 0
 w_k10_KO_s114_return_constraint: 1.04088 w_k10_KO_s11 - w_k10_KO_s114
                                 + 0.99 b_k10_KO_s114 - 1.01 s_k10_KO_s114  = 0
 w_k10_KO_s115_return_constraint: 1.10114 w_k10_KO_s11 - w_k10_KO_s115
                                 + 0.99 b_k10_KO_s115 - 1.01 s_k10_KO_s115  = 0
 w_k10_KO_s116_return_constraint: 0.97199 w_k10_KO_s11 - w_k10_KO_s116
                                 + 0.99 b_k10_KO_s116 - 1.01 s_k10_KO_s116  = 0
 w_k10_KO_s117_return_constraint: 1.21857 w_k10_KO_s11 - w_k10_KO_s117
                                 + 0.99 b_k10_KO_s117 - 1.01 s_k10_KO_s117  = 0
 w_k10_KO_s120_return_constraint: 1.04449 w_k10_KO_s12 - w_k10_KO_s120
                                 + 0.99 b_k10_KO_s120 - 1.01 s_k10_KO_s120  = 0
 w_k10_KO_s121_return_constraint: 0.837003 w_k10_KO_s12 - w_k10_KO_s121
                                 + 0.99 b_k10_KO_s121 - 1.01 s_k10_KO_s121  = 0
 w_k10_KO_s122_return_constraint: 0.719535 w_k10_KO_s12 - w_k10_KO_s122
                                 + 0.99 b_k10_KO_s122 - 1.01 s_k10_KO_s122  = 0
 w_k10_KO_s123_return_constraint: 1.14814 w_k10_KO_s12 - w_k10_KO_s123
                                 + 0.99 b_k10_KO_s123 - 1.01 s_k10_KO_s123  = 0
 w_k10_KO_s124_return_constraint: 0.933758 w_k10_KO_s12 - w_k10_KO_s124
                                 + 0.99 b_k10_KO_s124 - 1.01 s_k10_KO_s124  = 0
 w_k10_KO_s125_return_constraint: 1.45374 w_k10_KO_s12 - w_k10_KO_s125
                                 + 0.99 b_k10_KO_s125 - 1.01 s_k10_KO_s125  = 0
 w_k10_KO_s126_return_constraint: 0.869932 w_k10_KO_s12 - w_k10_KO_s126
                                 + 0.99 b_k10_KO_s126 - 1.01 s_k10_KO_s126  = 0
 w_k10_KO_s127_return_constraint: 0.685885 w_k10_KO_s12 - w_k10_KO_s127
                                 + 0.99 b_k10_KO_s127 - 1.01 s_k10_KO_s127  = 0
 w_k10_KO_s130_return_constraint: 0.842933 w_k10_KO_s13 - w_k10_KO_s130
                                 + 0.99 b_k10_KO_s130 - 1.01 s_k10_KO_s130  = 0
 w_k10_KO_s131_return_constraint: 1.34723 w_k10_KO_s13 - w_k10_KO_s131
                                 + 0.99 b_k10_KO_s131 - 1.01 s_k10_KO_s131  = 0
 w_k10_KO_s132_return_constraint: 0.795705 w_k10_KO_s13 - w_k10_KO_s132
                                 + 0.99 b_k10_KO_s132 - 1.01 s_k10_KO_s132  = 0
 w_k10_KO_s133_return_constraint: 0.774419 w_k10_KO_s13 - w_k10_KO_s133
                                 + 0.99 b_k10_KO_s133 - 1.01 s_k10_KO_s133  = 0
 w_k10_KO_s134_return_constraint: 1.34655 w_k10_KO_s13 - w_k10_KO_s134
                                 + 0.99 b_k10_KO_s134 - 1.01 s_k10_KO_s134  = 0
 w_k10_KO_s135_return_constraint: 1.04083 w_k10_KO_s13 - w_k10_KO_s135
                                 + 0.99 b_k10_KO_s135 - 1.01 s_k10_KO_s135  = 0
 w_k10_KO_s136_return_constraint: 1.18177 w_k10_KO_s13 - w_k10_KO_s136
                                 + 0.99 b_k10_KO_s136 - 1.01 s_k10_KO_s136  = 0
 w_k10_KO_s137_return_constraint: 0.794348 w_k10_KO_s13 - w_k10_KO_s137
                                 + 0.99 b_k10_KO_s137 - 1.01 s_k10_KO_s137  = 0
 w_k10_MAT_s0_return_constraint: 0.871699 w_0_MAT - w_k10_MAT_s0
                                 + 0.99 b_k10_MAT_s0 - 1.01 s_k10_MAT_s0  = 0
 w_k10_MAT_s1_return_constraint: 1.40107 w_0_MAT - w_k10_MAT_s1
                                 + 0.99 b_k10_MAT_s1 - 1.01 s_k10_MAT_s1  = 0
 w_k10_MAT_s00_return_constraint: 0.832156 w_k10_MAT_s0 - w_k10_MAT_s00
                                 + 0.99 b_k10_MAT_s00 - 1.01 s_k10_MAT_s00  = 0
 w_k10_MAT_s01_return_constraint: 0.946163 w_k10_MAT_s0 - w_k10_MAT_s01
                                 + 0.99 b_k10_MAT_s01 - 1.01 s_k10_MAT_s01  = 0
 w_k10_MAT_s02_return_constraint: 1.17108 w_k10_MAT_s0 - w_k10_MAT_s02
                                 + 0.99 b_k10_MAT_s02 - 1.01 s_k10_MAT_s02  = 0
 w_k10_MAT_s03_return_constraint: 1.74321 w_k10_MAT_s0 - w_k10_MAT_s03
                                 + 0.99 b_k10_MAT_s03 - 1.01 s_k10_MAT_s03  = 0
 w_k10_MAT_s10_return_constraint: 1.29801 w_k10_MAT_s1 - w_k10_MAT_s10
                                 + 0.99 b_k10_MAT_s10 - 1.01 s_k10_MAT_s10  = 0
 w_k10_MAT_s11_return_constraint: 0.725882 w_k10_MAT_s1 - w_k10_MAT_s11
                                 + 0.99 b_k10_MAT_s11 - 1.01 s_k10_MAT_s11  = 0
 w_k10_MAT_s12_return_constraint: 0.834921 w_k10_MAT_s1 - w_k10_MAT_s12
                                 + 0.99 b_k10_MAT_s12 - 1.01 s_k10_MAT_s12  = 0
 w_k10_MAT_s13_return_constraint: 1.21402 w_k10_MAT_s1 - w_k10_MAT_s13
                                 + 0.99 b_k10_MAT_s13 - 1.01 s_k10_MAT_s13  = 0
 w_k10_MAT_s000_return_constraint: 1.52422 w_k10_MAT_s00 - w_k10_MAT_s000
                                 + 0.99 b_k10_MAT_s000 - 1.01 s_k10_MAT_s000
                                  = 0
 w_k10_MAT_s001_return_constraint: 1.52143 w_k10_MAT_s00 - w_k10_MAT_s001
                                 + 0.99 b_k10_MAT_s001 - 1.01 s_k10_MAT_s001
                                  = 0
 w_k10_MAT_s002_return_constraint: 1.40052 w_k10_MAT_s00 - w_k10_MAT_s002
                                 + 0.99 b_k10_MAT_s002 - 1.01 s_k10_MAT_s002
                                  = 0
 w_k10_MAT_s003_return_constraint: 0.634523 w_k10_MAT_s00 - w_k10_MAT_s003
                                 + 0.99 b_k10_MAT_s003 - 1.01 s_k10_MAT_s003
                                  = 0
 w_k10_MAT_s004_return_constraint: 1.08976 w_k10_MAT_s00 - w_k10_MAT_s004
                                 + 0.99 b_k10_MAT_s004 - 1.01 s_k10_MAT_s004
                                  = 0
 w_k10_MAT_s005_return_constraint: 0.862777 w_k10_MAT_s00 - w_k10_MAT_s005
                                 + 0.99 b_k10_MAT_s005 - 1.01 s_k10_MAT_s005
                                  = 0
 w_k10_MAT_s006_return_constraint: 1.26142 w_k10_MAT_s00 - w_k10_MAT_s006
                                 + 0.99 b_k10_MAT_s006 - 1.01 s_k10_MAT_s006
                                  = 0
 w_k10_MAT_s007_return_constraint: 0.639035 w_k10_MAT_s00 - w_k10_MAT_s007
                                 + 0.99 b_k10_MAT_s007 - 1.01 s_k10_MAT_s007
                                  = 0
 w_k10_MAT_s010_return_constraint: 0.580534 w_k10_MAT_s01 - w_k10_MAT_s010
                                 + 0.99 b_k10_MAT_s010 - 1.01 s_k10_MAT_s010
                                  = 0
 w_k10_MAT_s011_return_constraint: 1.05555 w_k10_MAT_s01 - w_k10_MAT_s011
                                 + 0.99 b_k10_MAT_s011 - 1.01 s_k10_MAT_s011
                                  = 0
 w_k10_MAT_s012_return_constraint: 1.2765 w_k10_MAT_s01 - w_k10_MAT_s012
                                 + 0.99 b_k10_MAT_s012 - 1.01 s_k10_MAT_s012
                                  = 0
 w_k10_MAT_s013_return_constraint: 0.62618 w_k10_MAT_s01 - w_k10_MAT_s013
                                 + 0.99 b_k10_MAT_s013 - 1.01 s_k10_MAT_s013
                                  = 0
 w_k10_MAT_s014_return_constraint: 1.4499 w_k10_MAT_s01 - w_k10_MAT_s014
                                 + 0.99 b_k10_MAT_s014 - 1.01 s_k10_MAT_s014
                                  = 0
 w_k10_MAT_s015_return_constraint: 0.797065 w_k10_MAT_s01 - w_k10_MAT_s015
                                 + 0.99 b_k10_MAT_s015 - 1.01 s_k10_MAT_s015
                                  = 0
 w_k10_MAT_s016_return_constraint: 0.744184 w_k10_MAT_s01 - w_k10_MAT_s016
                                 + 0.99 b_k10_MAT_s016 - 1.01 s_k10_MAT_s016
                                  = 0
 w_k10_MAT_s017_return_constraint: 1.80145 w_k10_MAT_s01 - w_k10_MAT_s017
                                 + 0.99 b_k10_MAT_s017 - 1.01 s_k10_MAT_s017
                                  = 0
 w_k10_MAT_s020_return_constraint: 1.17086 w_k10_MAT_s02 - w_k10_MAT_s020
                                 + 0.99 b_k10_MAT_s020 - 1.01 s_k10_MAT_s020
                                  = 0
 w_k10_MAT_s021_return_constraint: 1.55707 w_k10_MAT_s02 - w_k10_MAT_s021
                                 + 0.99 b_k10_MAT_s021 - 1.01 s_k10_MAT_s021
                                  = 0
 w_k10_MAT_s022_return_constraint: 0.715315 w_k10_MAT_s02 - w_k10_MAT_s022
                                 + 0.99 b_k10_MAT_s022 - 1.01 s_k10_MAT_s022
                                  = 0
 w_k10_MAT_s023_return_constraint: 0.61042 w_k10_MAT_s02 - w_k10_MAT_s023
                                 + 0.99 b_k10_MAT_s023 - 1.01 s_k10_MAT_s023
                                  = 0
 w_k10_MAT_s024_return_constraint: 1.09367 w_k10_MAT_s02 - w_k10_MAT_s024
                                 + 0.99 b_k10_MAT_s024 - 1.01 s_k10_MAT_s024
                                  = 0
 w_k10_MAT_s025_return_constraint: 1.70313 w_k10_MAT_s02 - w_k10_MAT_s025
                                 + 0.99 b_k10_MAT_s025 - 1.01 s_k10_MAT_s025
                                  = 0
 w_k10_MAT_s026_return_constraint: 1.20393 w_k10_MAT_s02 - w_k10_MAT_s026
                                 + 0.99 b_k10_MAT_s026 - 1.01 s_k10_MAT_s026
                                  = 0
 w_k10_MAT_s027_return_constraint: 0.831736 w_k10_MAT_s02 - w_k10_MAT_s027
                                 + 0.99 b_k10_MAT_s027 - 1.01 s_k10_MAT_s027
                                  = 0
 w_k10_MAT_s030_return_constraint: 0.608302 w_k10_MAT_s03 - w_k10_MAT_s030
                                 + 0.99 b_k10_MAT_s030 - 1.01 s_k10_MAT_s030
                                  = 0
 w_k10_MAT_s031_return_constraint: 0.878668 w_k10_MAT_s03 - w_k10_MAT_s031
                                 + 0.99 b_k10_MAT_s031 - 1.01 s_k10_MAT_s031
                                  = 0
 w_k10_MAT_s032_return_constraint: 0.968418 w_k10_MAT_s03 - w_k10_MAT_s032
                                 + 0.99 b_k10_MAT_s032 - 1.01 s_k10_MAT_s032
                                  = 0
 w_k10_MAT_s033_return_constraint: 1.23306 w_k10_MAT_s03 - w_k10_MAT_s033
                                 + 0.99 b_k10_MAT_s033 - 1.01 s_k10_MAT_s033
                                  = 0
 w_k10_MAT_s034_return_constraint: 1.26868 w_k10_MAT_s03 - w_k10_MAT_s034
                                 + 0.99 b_k10_MAT_s034 - 1.01 s_k10_MAT_s034
                                  = 0
 w_k10_MAT_s035_return_constraint: 1.5319 w_k10_MAT_s03 - w_k10_MAT_s035
                                 + 0.99 b_k10_MAT_s035 - 1.01 s_k10_MAT_s035
                                  = 0
 w_k10_MAT_s036_return_constraint: 0.886289 w_k10_MAT_s03 - w_k10_MAT_s036
                                 + 0.99 b_k10_MAT_s036 - 1.01 s_k10_MAT_s036
                                  = 0
 w_k10_MAT_s037_return_constraint: 1.94831 w_k10_MAT_s03 - w_k10_MAT_s037
                                 + 0.99 b_k10_MAT_s037 - 1.01 s_k10_MAT_s037
                                  = 0
 w_k10_MAT_s100_return_constraint: 1.3415 w_k10_MAT_s10 - w_k10_MAT_s100
                                 + 0.99 b_k10_MAT_s100 - 1.01 s_k10_MAT_s100
                                  = 0
 w_k10_MAT_s101_return_constraint: 0.819197 w_k10_MAT_s10 - w_k10_MAT_s101
                                 + 0.99 b_k10_MAT_s101 - 1.01 s_k10_MAT_s101
                                  = 0
 w_k10_MAT_s102_return_constraint: 1.45182 w_k10_MAT_s10 - w_k10_MAT_s102
                                 + 0.99 b_k10_MAT_s102 - 1.01 s_k10_MAT_s102
                                  = 0
 w_k10_MAT_s103_return_constraint: 1.82013 w_k10_MAT_s10 - w_k10_MAT_s103
                                 + 0.99 b_k10_MAT_s103 - 1.01 s_k10_MAT_s103
                                  = 0
 w_k10_MAT_s104_return_constraint: 1.02846 w_k10_MAT_s10 - w_k10_MAT_s104
                                 + 0.99 b_k10_MAT_s104 - 1.01 s_k10_MAT_s104
                                  = 0
 w_k10_MAT_s105_return_constraint: 0.788664 w_k10_MAT_s10 - w_k10_MAT_s105
                                 + 0.99 b_k10_MAT_s105 - 1.01 s_k10_MAT_s105
                                  = 0
 w_k10_MAT_s106_return_constraint: 0.862542 w_k10_MAT_s10 - w_k10_MAT_s106
                                 + 0.99 b_k10_MAT_s106 - 1.01 s_k10_MAT_s106
                                  = 0
 w_k10_MAT_s107_return_constraint: 1.23682 w_k10_MAT_s10 - w_k10_MAT_s107
                                 + 0.99 b_k10_MAT_s107 - 1.01 s_k10_MAT_s107
                                  = 0
 w_k10_MAT_s110_return_constraint: 0.954934 w_k10_MAT_s11 - w_k10_MAT_s110
                                 + 0.99 b_k10_MAT_s110 - 1.01 s_k10_MAT_s110
                                  = 0
 w_k10_MAT_s111_return_constraint: 1.88465 w_k10_MAT_s11 - w_k10_MAT_s111
                                 + 0.99 b_k10_MAT_s111 - 1.01 s_k10_MAT_s111
                                  = 0
 w_k10_MAT_s112_return_constraint: 0.782575 w_k10_MAT_s11 - w_k10_MAT_s112
                                 + 0.99 b_k10_MAT_s112 - 1.01 s_k10_MAT_s112
                                  = 0
 w_k10_MAT_s113_return_constraint: 1.84949 w_k10_MAT_s11 - w_k10_MAT_s113
                                 + 0.99 b_k10_MAT_s113 - 1.01 s_k10_MAT_s113
                                  = 0
 w_k10_MAT_s114_return_constraint: 1.07495 w_k10_MAT_s11 - w_k10_MAT_s114
                                 + 0.99 b_k10_MAT_s114 - 1.01 s_k10_MAT_s114
                                  = 0
 w_k10_MAT_s115_return_constraint: 1.24841 w_k10_MAT_s11 - w_k10_MAT_s115
                                 + 0.99 b_k10_MAT_s115 - 1.01 s_k10_MAT_s115
                                  = 0
 w_k10_MAT_s116_return_constraint: 1.27625 w_k10_MAT_s11 - w_k10_MAT_s116
                                 + 0.99 b_k10_MAT_s116 - 1.01 s_k10_MAT_s116
                                  = 0
 w_k10_MAT_s117_return_constraint: 1.12936 w_k10_MAT_s11 - w_k10_MAT_s117
                                 + 0.99 b_k10_MAT_s117 - 1.01 s_k10_MAT_s117
                                  = 0
 w_k10_MAT_s120_return_constraint: 1.09234 w_k10_MAT_s12 - w_k10_MAT_s120
                                 + 0.99 b_k10_MAT_s120 - 1.01 s_k10_MAT_s120
                                  = 0
 w_k10_MAT_s121_return_constraint: 1.00704 w_k10_MAT_s12 - w_k10_MAT_s121
                                 + 0.99 b_k10_MAT_s121 - 1.01 s_k10_MAT_s121
                                  = 0
 w_k10_MAT_s122_return_constraint: 0.842593 w_k10_MAT_s12 - w_k10_MAT_s122
                                 + 0.99 b_k10_MAT_s122 - 1.01 s_k10_MAT_s122
                                  = 0
 w_k10_MAT_s123_return_constraint: 1.41771 w_k10_MAT_s12 - w_k10_MAT_s123
                                 + 0.99 b_k10_MAT_s123 - 1.01 s_k10_MAT_s123
                                  = 0
 w_k10_MAT_s124_return_constraint: 0.86197 w_k10_MAT_s12 - w_k10_MAT_s124
                                 + 0.99 b_k10_MAT_s124 - 1.01 s_k10_MAT_s124
                                  = 0
 w_k10_MAT_s125_return_constraint: 1.74036 w_k10_MAT_s12 - w_k10_MAT_s125
                                 + 0.99 b_k10_MAT_s125 - 1.01 s_k10_MAT_s125
                                  = 0
 w_k10_MAT_s126_return_constraint: 0.70611 w_k10_MAT_s12 - w_k10_MAT_s126
                                 + 0.99 b_k10_MAT_s126 - 1.01 s_k10_MAT_s126
                                  = 0
 w_k10_MAT_s127_return_constraint: 0.716311 w_k10_MAT_s12 - w_k10_MAT_s127
                                 + 0.99 b_k10_MAT_s127 - 1.01 s_k10_MAT_s127
                                  = 0
 w_k10_MAT_s130_return_constraint: 1.02662 w_k10_MAT_s13 - w_k10_MAT_s130
                                 + 0.99 b_k10_MAT_s130 - 1.01 s_k10_MAT_s130
                                  = 0
 w_k10_MAT_s131_return_constraint: 1.94153 w_k10_MAT_s13 - w_k10_MAT_s131
                                 + 0.99 b_k10_MAT_s131 - 1.01 s_k10_MAT_s131
                                  = 0
 w_k10_MAT_s132_return_constraint: 0.835138 w_k10_MAT_s13 - w_k10_MAT_s132
                                 + 0.99 b_k10_MAT_s132 - 1.01 s_k10_MAT_s132
                                  = 0
 w_k10_MAT_s133_return_constraint: 1.19047 w_k10_MAT_s13 - w_k10_MAT_s133
                                 + 0.99 b_k10_MAT_s133 - 1.01 s_k10_MAT_s133
                                  = 0
 w_k10_MAT_s134_return_constraint: 1.11937 w_k10_MAT_s13 - w_k10_MAT_s134
                                 + 0.99 b_k10_MAT_s134 - 1.01 s_k10_MAT_s134
                                  = 0
 w_k10_MAT_s135_return_constraint: 0.692915 w_k10_MAT_s13 - w_k10_MAT_s135
                                 + 0.99 b_k10_MAT_s135 - 1.01 s_k10_MAT_s135
                                  = 0
 w_k10_MAT_s136_return_constraint: 1.28852 w_k10_MAT_s13 - w_k10_MAT_s136
                                 + 0.99 b_k10_MAT_s136 - 1.01 s_k10_MAT_s136
                                  = 0
 w_k10_MAT_s137_return_constraint: 0.655607 w_k10_MAT_s13 - w_k10_MAT_s137
                                 + 0.99 b_k10_MAT_s137 - 1.01 s_k10_MAT_s137
                                  = 0
 w_k10_MSFT_s0_return_constraint: 0.97829 w_0_MSFT - w_k10_MSFT_s0
                                 + 0.99 b_k10_MSFT_s0 - 1.01 s_k10_MSFT_s0  = 0
 w_k10_MSFT_s1_return_constraint: 1.03188 w_0_MSFT - w_k10_MSFT_s1
                                 + 0.99 b_k10_MSFT_s1 - 1.01 s_k10_MSFT_s1  = 0
 w_k10_MSFT_s00_return_constraint: 0.939139 w_k10_MSFT_s0 - w_k10_MSFT_s00
                                 + 0.99 b_k10_MSFT_s00 - 1.01 s_k10_MSFT_s00
                                  = 0
 w_k10_MSFT_s01_return_constraint: 0.860955 w_k10_MSFT_s0 - w_k10_MSFT_s01
                                 + 0.99 b_k10_MSFT_s01 - 1.01 s_k10_MSFT_s01
                                  = 0
 w_k10_MSFT_s02_return_constraint: 1.28775 w_k10_MSFT_s0 - w_k10_MSFT_s02
                                 + 0.99 b_k10_MSFT_s02 - 1.01 s_k10_MSFT_s02
                                  = 0
 w_k10_MSFT_s03_return_constraint: 1.21041 w_k10_MSFT_s0 - w_k10_MSFT_s03
                                 + 0.99 b_k10_MSFT_s03 - 1.01 s_k10_MSFT_s03
                                  = 0
 w_k10_MSFT_s10_return_constraint: 1.04671 w_k10_MSFT_s1 - w_k10_MSFT_s10
                                 + 0.99 b_k10_MSFT_s10 - 1.01 s_k10_MSFT_s10
                                  = 0
 w_k10_MSFT_s11_return_constraint: 0.88223 w_k10_MSFT_s1 - w_k10_MSFT_s11
                                 + 0.99 b_k10_MSFT_s11 - 1.01 s_k10_MSFT_s11
                                  = 0
 w_k10_MSFT_s12_return_constraint: 1.19868 w_k10_MSFT_s1 - w_k10_MSFT_s12
                                 + 0.99 b_k10_MSFT_s12 - 1.01 s_k10_MSFT_s12
                                  = 0
 w_k10_MSFT_s13_return_constraint: 1.24813 w_k10_MSFT_s1 - w_k10_MSFT_s13
                                 + 0.99 b_k10_MSFT_s13 - 1.01 s_k10_MSFT_s13
                                  = 0
 w_k10_MSFT_s000_return_constraint: 1.25552 w_k10_MSFT_s00 - w_k10_MSFT_s000
                                 + 0.99 b_k10_MSFT_s000 - 1.01 s_k10_MSFT_s000
                                  = 0
 w_k10_MSFT_s001_return_constraint: 1.11873 w_k10_MSFT_s00 - w_k10_MSFT_s001
                                 + 0.99 b_k10_MSFT_s001 - 1.01 s_k10_MSFT_s001
                                  = 0
 w_k10_MSFT_s002_return_constraint: 1.43836 w_k10_MSFT_s00 - w_k10_MSFT_s002
                                 + 0.99 b_k10_MSFT_s002 - 1.01 s_k10_MSFT_s002
                                  = 0
 w_k10_MSFT_s003_return_constraint: 0.79385 w_k10_MSFT_s00 - w_k10_MSFT_s003
                                 + 0.99 b_k10_MSFT_s003 - 1.01 s_k10_MSFT_s003
                                  = 0
 w_k10_MSFT_s004_return_constraint: 1.28672 w_k10_MSFT_s00 - w_k10_MSFT_s004
                                 + 0.99 b_k10_MSFT_s004 - 1.01 s_k10_MSFT_s004
                                  = 0
 w_k10_MSFT_s005_return_constraint: 0.975628 w_k10_MSFT_s00 - w_k10_MSFT_s005
                                 + 0.99 b_k10_MSFT_s005 - 1.01 s_k10_MSFT_s005
                                  = 0
 w_k10_MSFT_s006_return_constraint: 0.905466 w_k10_MSFT_s00 - w_k10_MSFT_s006
                                 + 0.99 b_k10_MSFT_s006 - 1.01 s_k10_MSFT_s006
                                  = 0
 w_k10_MSFT_s007_return_constraint: 1.1326 w_k10_MSFT_s00 - w_k10_MSFT_s007
                                 + 0.99 b_k10_MSFT_s007 - 1.01 s_k10_MSFT_s007
                                  = 0
 w_k10_MSFT_s010_return_constraint: 0.932739 w_k10_MSFT_s01 - w_k10_MSFT_s010
                                 + 0.99 b_k10_MSFT_s010 - 1.01 s_k10_MSFT_s010
                                  = 0
 w_k10_MSFT_s011_return_constraint: 1.04808 w_k10_MSFT_s01 - w_k10_MSFT_s011
                                 + 0.99 b_k10_MSFT_s011 - 1.01 s_k10_MSFT_s011
                                  = 0
 w_k10_MSFT_s012_return_constraint: 0.951222 w_k10_MSFT_s01 - w_k10_MSFT_s012
                                 + 0.99 b_k10_MSFT_s012 - 1.01 s_k10_MSFT_s012
                                  = 0
 w_k10_MSFT_s013_return_constraint: 0.725052 w_k10_MSFT_s01 - w_k10_MSFT_s013
                                 + 0.99 b_k10_MSFT_s013 - 1.01 s_k10_MSFT_s013
                                  = 0
 w_k10_MSFT_s014_return_constraint: 1.17347 w_k10_MSFT_s01 - w_k10_MSFT_s014
                                 + 0.99 b_k10_MSFT_s014 - 1.01 s_k10_MSFT_s014
                                  = 0
 w_k10_MSFT_s015_return_constraint: 0.862111 w_k10_MSFT_s01 - w_k10_MSFT_s015
                                 + 0.99 b_k10_MSFT_s015 - 1.01 s_k10_MSFT_s015
                                  = 0
 w_k10_MSFT_s016_return_constraint: 0.783948 w_k10_MSFT_s01 - w_k10_MSFT_s016
                                 + 0.99 b_k10_MSFT_s016 - 1.01 s_k10_MSFT_s016
                                  = 0
 w_k10_MSFT_s017_return_constraint: 1.38672 w_k10_MSFT_s01 - w_k10_MSFT_s017
                                 + 0.99 b_k10_MSFT_s017 - 1.01 s_k10_MSFT_s017
                                  = 0
 w_k10_MSFT_s020_return_constraint: 0.848836 w_k10_MSFT_s02 - w_k10_MSFT_s020
                                 + 0.99 b_k10_MSFT_s020 - 1.01 s_k10_MSFT_s020
                                  = 0
 w_k10_MSFT_s021_return_constraint: 0.97596 w_k10_MSFT_s02 - w_k10_MSFT_s021
                                 + 0.99 b_k10_MSFT_s021 - 1.01 s_k10_MSFT_s021
                                  = 0
 w_k10_MSFT_s022_return_constraint: 1.15248 w_k10_MSFT_s02 - w_k10_MSFT_s022
                                 + 0.99 b_k10_MSFT_s022 - 1.01 s_k10_MSFT_s022
                                  = 0
 w_k10_MSFT_s023_return_constraint: 0.846422 w_k10_MSFT_s02 - w_k10_MSFT_s023
                                 + 0.99 b_k10_MSFT_s023 - 1.01 s_k10_MSFT_s023
                                  = 0
 w_k10_MSFT_s024_return_constraint: 1.11561 w_k10_MSFT_s02 - w_k10_MSFT_s024
                                 + 0.99 b_k10_MSFT_s024 - 1.01 s_k10_MSFT_s024
                                  = 0
 w_k10_MSFT_s025_return_constraint: 1.34544 w_k10_MSFT_s02 - w_k10_MSFT_s025
                                 + 0.99 b_k10_MSFT_s025 - 1.01 s_k10_MSFT_s025
                                  = 0
 w_k10_MSFT_s026_return_constraint: 1.01031 w_k10_MSFT_s02 - w_k10_MSFT_s026
                                 + 0.99 b_k10_MSFT_s026 - 1.01 s_k10_MSFT_s026
                                  = 0
 w_k10_MSFT_s027_return_constraint: 0.83566 w_k10_MSFT_s02 - w_k10_MSFT_s027
                                 + 0.99 b_k10_MSFT_s027 - 1.01 s_k10_MSFT_s027
                                  = 0
 w_k10_MSFT_s030_return_constraint: 0.77042 w_k10_MSFT_s03 - w_k10_MSFT_s030
                                 + 0.99 b_k10_MSFT_s030 - 1.01 s_k10_MSFT_s030
                                  = 0
 w_k10_MSFT_s031_return_constraint: 0.681762 w_k10_MSFT_s03 - w_k10_MSFT_s031
                                 + 0.99 b_k10_MSFT_s031 - 1.01 s_k10_MSFT_s031
                                  = 0
 w_k10_MSFT_s032_return_constraint: 0.943522 w_k10_MSFT_s03 - w_k10_MSFT_s032
                                 + 0.99 b_k10_MSFT_s032 - 1.01 s_k10_MSFT_s032
                                  = 0
 w_k10_MSFT_s033_return_constraint: 1.27709 w_k10_MSFT_s03 - w_k10_MSFT_s033
                                 + 0.99 b_k10_MSFT_s033 - 1.01 s_k10_MSFT_s033
                                  = 0
 w_k10_MSFT_s034_return_constraint: 0.924174 w_k10_MSFT_s03 - w_k10_MSFT_s034
                                 + 0.99 b_k10_MSFT_s034 - 1.01 s_k10_MSFT_s034
                                  = 0
 w_k10_MSFT_s035_return_constraint: 1.18214 w_k10_MSFT_s03 - w_k10_MSFT_s035
                                 + 0.99 b_k10_MSFT_s035 - 1.01 s_k10_MSFT_s035
                                  = 0
 w_k10_MSFT_s036_return_constraint: 0.983047 w_k10_MSFT_s03 - w_k10_MSFT_s036
                                 + 0.99 b_k10_MSFT_s036 - 1.01 s_k10_MSFT_s036
                                  = 0
 w_k10_MSFT_s037_return_constraint: 1.49666 w_k10_MSFT_s03 - w_k10_MSFT_s037
                                 + 0.99 b_k10_MSFT_s037 - 1.01 s_k10_MSFT_s037
                                  = 0
 w_k10_MSFT_s100_return_constraint: 1.54107 w_k10_MSFT_s10 - w_k10_MSFT_s100
                                 + 0.99 b_k10_MSFT_s100 - 1.01 s_k10_MSFT_s100
                                  = 0
 w_k10_MSFT_s101_return_constraint: 0.811639 w_k10_MSFT_s10 - w_k10_MSFT_s101
                                 + 0.99 b_k10_MSFT_s101 - 1.01 s_k10_MSFT_s101
                                  = 0
 w_k10_MSFT_s102_return_constraint: 1.20378 w_k10_MSFT_s10 - w_k10_MSFT_s102
                                 + 0.99 b_k10_MSFT_s102 - 1.01 s_k10_MSFT_s102
                                  = 0
 w_k10_MSFT_s103_return_constraint: 1.31203 w_k10_MSFT_s10 - w_k10_MSFT_s103
                                 + 0.99 b_k10_MSFT_s103 - 1.01 s_k10_MSFT_s103
                                  = 0
 w_k10_MSFT_s104_return_constraint: 1.03025 w_k10_MSFT_s10 - w_k10_MSFT_s104
                                 + 0.99 b_k10_MSFT_s104 - 1.01 s_k10_MSFT_s104
                                  = 0
 w_k10_MSFT_s105_return_constraint: 1.18498 w_k10_MSFT_s10 - w_k10_MSFT_s105
                                 + 0.99 b_k10_MSFT_s105 - 1.01 s_k10_MSFT_s105
                                  = 0
 w_k10_MSFT_s106_return_constraint: 0.949803 w_k10_MSFT_s10 - w_k10_MSFT_s106
                                 + 0.99 b_k10_MSFT_s106 - 1.01 s_k10_MSFT_s106
                                  = 0
 w_k10_MSFT_s107_return_constraint: 0.979169 w_k10_MSFT_s10 - w_k10_MSFT_s107
                                 + 0.99 b_k10_MSFT_s107 - 1.01 s_k10_MSFT_s107
                                  = 0
 w_k10_MSFT_s110_return_constraint: 0.770055 w_k10_MSFT_s11 - w_k10_MSFT_s110
                                 + 0.99 b_k10_MSFT_s110 - 1.01 s_k10_MSFT_s110
                                  = 0
 w_k10_MSFT_s111_return_constraint: 0.810809 w_k10_MSFT_s11 - w_k10_MSFT_s111
                                 + 0.99 b_k10_MSFT_s111 - 1.01 s_k10_MSFT_s111
                                  = 0
 w_k10_MSFT_s112_return_constraint: 0.992757 w_k10_MSFT_s11 - w_k10_MSFT_s112
                                 + 0.99 b_k10_MSFT_s112 - 1.01 s_k10_MSFT_s112
                                  = 0
 w_k10_MSFT_s113_return_constraint: 1.25755 w_k10_MSFT_s11 - w_k10_MSFT_s113
                                 + 0.99 b_k10_MSFT_s113 - 1.01 s_k10_MSFT_s113
                                  = 0
 w_k10_MSFT_s114_return_constraint: 0.955012 w_k10_MSFT_s11 - w_k10_MSFT_s114
                                 + 0.99 b_k10_MSFT_s114 - 1.01 s_k10_MSFT_s114
                                  = 0
 w_k10_MSFT_s115_return_constraint: 0.95658 w_k10_MSFT_s11 - w_k10_MSFT_s115
                                 + 0.99 b_k10_MSFT_s115 - 1.01 s_k10_MSFT_s115
                                  = 0
 w_k10_MSFT_s116_return_constraint: 0.851907 w_k10_MSFT_s11 - w_k10_MSFT_s116
                                 + 0.99 b_k10_MSFT_s116 - 1.01 s_k10_MSFT_s116
                                  = 0
 w_k10_MSFT_s117_return_constraint: 1.23819 w_k10_MSFT_s11 - w_k10_MSFT_s117
                                 + 0.99 b_k10_MSFT_s117 - 1.01 s_k10_MSFT_s117
                                  = 0
 w_k10_MSFT_s120_return_constraint: 1.33255 w_k10_MSFT_s12 - w_k10_MSFT_s120
                                 + 0.99 b_k10_MSFT_s120 - 1.01 s_k10_MSFT_s120
                                  = 0
 w_k10_MSFT_s121_return_constraint: 0.931157 w_k10_MSFT_s12 - w_k10_MSFT_s121
                                 + 0.99 b_k10_MSFT_s121 - 1.01 s_k10_MSFT_s121
                                  = 0
 w_k10_MSFT_s122_return_constraint: 0.753341 w_k10_MSFT_s12 - w_k10_MSFT_s122
                                 + 0.99 b_k10_MSFT_s122 - 1.01 s_k10_MSFT_s122
                                  = 0
 w_k10_MSFT_s123_return_constraint: 0.910669 w_k10_MSFT_s12 - w_k10_MSFT_s123
                                 + 0.99 b_k10_MSFT_s123 - 1.01 s_k10_MSFT_s123
                                  = 0
 w_k10_MSFT_s124_return_constraint: 1.03618 w_k10_MSFT_s12 - w_k10_MSFT_s124
                                 + 0.99 b_k10_MSFT_s124 - 1.01 s_k10_MSFT_s124
                                  = 0
 w_k10_MSFT_s125_return_constraint: 1.30009 w_k10_MSFT_s12 - w_k10_MSFT_s125
                                 + 0.99 b_k10_MSFT_s125 - 1.01 s_k10_MSFT_s125
                                  = 0
 w_k10_MSFT_s126_return_constraint: 0.828012 w_k10_MSFT_s12 - w_k10_MSFT_s126
                                 + 0.99 b_k10_MSFT_s126 - 1.01 s_k10_MSFT_s126
                                  = 0
 w_k10_MSFT_s127_return_constraint: 0.858826 w_k10_MSFT_s12 - w_k10_MSFT_s127
                                 + 0.99 b_k10_MSFT_s127 - 1.01 s_k10_MSFT_s127
                                  = 0
 w_k10_MSFT_s130_return_constraint: 1.01958 w_k10_MSFT_s13 - w_k10_MSFT_s130
                                 + 0.99 b_k10_MSFT_s130 - 1.01 s_k10_MSFT_s130
                                  = 0
 w_k10_MSFT_s131_return_constraint: 1.45762 w_k10_MSFT_s13 - w_k10_MSFT_s131
                                 + 0.99 b_k10_MSFT_s131 - 1.01 s_k10_MSFT_s131
                                  = 0
 w_k10_MSFT_s132_return_constraint: 0.995624 w_k10_MSFT_s13 - w_k10_MSFT_s132
                                 + 0.99 b_k10_MSFT_s132 - 1.01 s_k10_MSFT_s132
                                  = 0
 w_k10_MSFT_s133_return_constraint: 0.716886 w_k10_MSFT_s13 - w_k10_MSFT_s133
                                 + 0.99 b_k10_MSFT_s133 - 1.01 s_k10_MSFT_s133
                                  = 0
 w_k10_MSFT_s134_return_constraint: 1.2557 w_k10_MSFT_s13 - w_k10_MSFT_s134
                                 + 0.99 b_k10_MSFT_s134 - 1.01 s_k10_MSFT_s134
                                  = 0
 w_k10_MSFT_s135_return_constraint: 1.36941 w_k10_MSFT_s13 - w_k10_MSFT_s135
                                 + 0.99 b_k10_MSFT_s135 - 1.01 s_k10_MSFT_s135
                                  = 0
 w_k10_MSFT_s136_return_constraint: 1.09632 w_k10_MSFT_s13 - w_k10_MSFT_s136
                                 + 0.99 b_k10_MSFT_s136 - 1.01 s_k10_MSFT_s136
                                  = 0
 w_k10_MSFT_s137_return_constraint: 0.809541 w_k10_MSFT_s13 - w_k10_MSFT_s137
                                 + 0.99 b_k10_MSFT_s137 - 1.01 s_k10_MSFT_s137
                                  = 0
 w_k10_PG_s0_return_constraint:  0.966331 w_0_PG - w_k10_PG_s0
                                 + 0.99 b_k10_PG_s0 - 1.01 s_k10_PG_s0  = 0
 w_k10_PG_s1_return_constraint:  1.0694 w_0_PG - w_k10_PG_s1 + 0.99 b_k10_PG_s1
                                 - 1.01 s_k10_PG_s1  = 0
 w_k10_PG_s00_return_constraint: 0.96388 w_k10_PG_s0 - w_k10_PG_s00
                                 + 0.99 b_k10_PG_s00 - 1.01 s_k10_PG_s00  = 0
 w_k10_PG_s01_return_constraint: 0.876195 w_k10_PG_s0 - w_k10_PG_s01
                                 + 0.99 b_k10_PG_s01 - 1.01 s_k10_PG_s01  = 0
 w_k10_PG_s02_return_constraint: 1.12495 w_k10_PG_s0 - w_k10_PG_s02
                                 + 0.99 b_k10_PG_s02 - 1.01 s_k10_PG_s02  = 0
 w_k10_PG_s03_return_constraint: 1.18537 w_k10_PG_s0 - w_k10_PG_s03
                                 + 0.99 b_k10_PG_s03 - 1.01 s_k10_PG_s03  = 0
 w_k10_PG_s10_return_constraint: 1.12355 w_k10_PG_s1 - w_k10_PG_s10
                                 + 0.99 b_k10_PG_s10 - 1.01 s_k10_PG_s10  = 0
 w_k10_PG_s11_return_constraint: 0.827341 w_k10_PG_s1 - w_k10_PG_s11
                                 + 0.99 b_k10_PG_s11 - 1.01 s_k10_PG_s11  = 0
 w_k10_PG_s12_return_constraint: 1.07718 w_k10_PG_s1 - w_k10_PG_s12
                                 + 0.99 b_k10_PG_s12 - 1.01 s_k10_PG_s12  = 0
 w_k10_PG_s13_return_constraint: 1.18768 w_k10_PG_s1 - w_k10_PG_s13
                                 + 0.99 b_k10_PG_s13 - 1.01 s_k10_PG_s13  = 0
 w_k10_PG_s000_return_constraint: 1.09961 w_k10_PG_s00 - w_k10_PG_s000
                                 + 0.99 b_k10_PG_s000 - 1.01 s_k10_PG_s000  = 0
 w_k10_PG_s001_return_constraint: 1.09391 w_k10_PG_s00 - w_k10_PG_s001
                                 + 0.99 b_k10_PG_s001 - 1.01 s_k10_PG_s001  = 0
 w_k10_PG_s002_return_constraint: 1.25284 w_k10_PG_s00 - w_k10_PG_s002
                                 + 0.99 b_k10_PG_s002 - 1.01 s_k10_PG_s002  = 0
 w_k10_PG_s003_return_constraint: 0.809233 w_k10_PG_s00 - w_k10_PG_s003
                                 + 0.99 b_k10_PG_s003 - 1.01 s_k10_PG_s003  = 0
 w_k10_PG_s004_return_constraint: 1.05282 w_k10_PG_s00 - w_k10_PG_s004
                                 + 0.99 b_k10_PG_s004 - 1.01 s_k10_PG_s004  = 0
 w_k10_PG_s005_return_constraint: 0.888752 w_k10_PG_s00 - w_k10_PG_s005
                                 + 0.99 b_k10_PG_s005 - 1.01 s_k10_PG_s005  = 0
 w_k10_PG_s006_return_constraint: 0.888421 w_k10_PG_s00 - w_k10_PG_s006
                                 + 0.99 b_k10_PG_s006 - 1.01 s_k10_PG_s006  = 0
 w_k10_PG_s007_return_constraint: 1.00294 w_k10_PG_s00 - w_k10_PG_s007
                                 + 0.99 b_k10_PG_s007 - 1.01 s_k10_PG_s007  = 0
 w_k10_PG_s010_return_constraint: 0.935392 w_k10_PG_s01 - w_k10_PG_s010
                                 + 0.99 b_k10_PG_s010 - 1.01 s_k10_PG_s010  = 0
 w_k10_PG_s011_return_constraint: 1.00566 w_k10_PG_s01 - w_k10_PG_s011
                                 + 0.99 b_k10_PG_s011 - 1.01 s_k10_PG_s011  = 0
 w_k10_PG_s012_return_constraint: 0.973504 w_k10_PG_s01 - w_k10_PG_s012
                                 + 0.99 b_k10_PG_s012 - 1.01 s_k10_PG_s012  = 0
 w_k10_PG_s013_return_constraint: 0.800812 w_k10_PG_s01 - w_k10_PG_s013
                                 + 0.99 b_k10_PG_s013 - 1.01 s_k10_PG_s013  = 0
 w_k10_PG_s014_return_constraint: 1.15781 w_k10_PG_s01 - w_k10_PG_s014
                                 + 0.99 b_k10_PG_s014 - 1.01 s_k10_PG_s014  = 0
 w_k10_PG_s015_return_constraint: 0.956424 w_k10_PG_s01 - w_k10_PG_s015
                                 + 0.99 b_k10_PG_s015 - 1.01 s_k10_PG_s015  = 0
 w_k10_PG_s016_return_constraint: 0.851003 w_k10_PG_s01 - w_k10_PG_s016
                                 + 0.99 b_k10_PG_s016 - 1.01 s_k10_PG_s016  = 0
 w_k10_PG_s017_return_constraint: 1.32113 w_k10_PG_s01 - w_k10_PG_s017
                                 + 0.99 b_k10_PG_s017 - 1.01 s_k10_PG_s017  = 0
 w_k10_PG_s020_return_constraint: 1.003 w_k10_PG_s02 - w_k10_PG_s020
                                 + 0.99 b_k10_PG_s020 - 1.01 s_k10_PG_s020  = 0
 w_k10_PG_s021_return_constraint: 1.0652 w_k10_PG_s02 - w_k10_PG_s021
                                 + 0.99 b_k10_PG_s021 - 1.01 s_k10_PG_s021  = 0
 w_k10_PG_s022_return_constraint: 1.08451 w_k10_PG_s02 - w_k10_PG_s022
                                 + 0.99 b_k10_PG_s022 - 1.01 s_k10_PG_s022  = 0
 w_k10_PG_s023_return_constraint: 0.932506 w_k10_PG_s02 - w_k10_PG_s023
                                 + 0.99 b_k10_PG_s023 - 1.01 s_k10_PG_s023  = 0
 w_k10_PG_s024_return_constraint: 1.11562 w_k10_PG_s02 - w_k10_PG_s024
                                 + 0.99 b_k10_PG_s024 - 1.01 s_k10_PG_s024  = 0
 w_k10_PG_s025_return_constraint: 1.23663 w_k10_PG_s02 - w_k10_PG_s025
                                 + 0.99 b_k10_PG_s025 - 1.01 s_k10_PG_s025  = 0
 w_k10_PG_s026_return_constraint: 1.15233 w_k10_PG_s02 - w_k10_PG_s026
                                 + 0.99 b_k10_PG_s026 - 1.01 s_k10_PG_s026  = 0
 w_k10_PG_s027_return_constraint: 0.902865 w_k10_PG_s02 - w_k10_PG_s027
                                 + 0.99 b_k10_PG_s027 - 1.01 s_k10_PG_s027  = 0
 w_k10_PG_s030_return_constraint: 0.816338 w_k10_PG_s03 - w_k10_PG_s030
                                 + 0.99 b_k10_PG_s030 - 1.01 s_k10_PG_s030  = 0
 w_k10_PG_s031_return_constraint: 0.901987 w_k10_PG_s03 - w_k10_PG_s031
                                 + 0.99 b_k10_PG_s031 - 1.01 s_k10_PG_s031  = 0
 w_k10_PG_s032_return_constraint: 0.980759 w_k10_PG_s03 - w_k10_PG_s032
                                 + 0.99 b_k10_PG_s032 - 1.01 s_k10_PG_s032  = 0
 w_k10_PG_s033_return_constraint: 1.25045 w_k10_PG_s03 - w_k10_PG_s033
                                 + 0.99 b_k10_PG_s033 - 1.01 s_k10_PG_s033  = 0
 w_k10_PG_s034_return_constraint: 1.0725 w_k10_PG_s03 - w_k10_PG_s034
                                 + 0.99 b_k10_PG_s034 - 1.01 s_k10_PG_s034  = 0
 w_k10_PG_s035_return_constraint: 1.10757 w_k10_PG_s03 - w_k10_PG_s035
                                 + 0.99 b_k10_PG_s035 - 1.01 s_k10_PG_s035  = 0
 w_k10_PG_s036_return_constraint: 0.960341 w_k10_PG_s03 - w_k10_PG_s036
                                 + 0.99 b_k10_PG_s036 - 1.01 s_k10_PG_s036  = 0
 w_k10_PG_s037_return_constraint: 1.53707 w_k10_PG_s03 - w_k10_PG_s037
                                 + 0.99 b_k10_PG_s037 - 1.01 s_k10_PG_s037  = 0
 w_k10_PG_s100_return_constraint: 1.23377 w_k10_PG_s10 - w_k10_PG_s100
                                 + 0.99 b_k10_PG_s100 - 1.01 s_k10_PG_s100  = 0
 w_k10_PG_s101_return_constraint: 0.82725 w_k10_PG_s10 - w_k10_PG_s101
                                 + 0.99 b_k10_PG_s101 - 1.01 s_k10_PG_s101  = 0
 w_k10_PG_s102_return_constraint: 1.1673 w_k10_PG_s10 - w_k10_PG_s102
                                 + 0.99 b_k10_PG_s102 - 1.01 s_k10_PG_s102  = 0
 w_k10_PG_s103_return_constraint: 1.11441 w_k10_PG_s10 - w_k10_PG_s103
                                 + 0.99 b_k10_PG_s103 - 1.01 s_k10_PG_s103  = 0
 w_k10_PG_s104_return_constraint: 1.0386 w_k10_PG_s10 - w_k10_PG_s104
                                 + 0.99 b_k10_PG_s104 - 1.01 s_k10_PG_s104  = 0
 w_k10_PG_s105_return_constraint: 0.980509 w_k10_PG_s10 - w_k10_PG_s105
                                 + 0.99 b_k10_PG_s105 - 1.01 s_k10_PG_s105  = 0
 w_k10_PG_s106_return_constraint: 0.874055 w_k10_PG_s10 - w_k10_PG_s106
                                 + 0.99 b_k10_PG_s106 - 1.01 s_k10_PG_s106  = 0
 w_k10_PG_s107_return_constraint: 0.945376 w_k10_PG_s10 - w_k10_PG_s107
                                 + 0.99 b_k10_PG_s107 - 1.01 s_k10_PG_s107  = 0
 w_k10_PG_s110_return_constraint: 0.870214 w_k10_PG_s11 - w_k10_PG_s110
                                 + 0.99 b_k10_PG_s110 - 1.01 s_k10_PG_s110  = 0
 w_k10_PG_s111_return_constraint: 1.11709 w_k10_PG_s11 - w_k10_PG_s111
                                 + 0.99 b_k10_PG_s111 - 1.01 s_k10_PG_s111  = 0
 w_k10_PG_s112_return_constraint: 1.03641 w_k10_PG_s11 - w_k10_PG_s112
                                 + 0.99 b_k10_PG_s112 - 1.01 s_k10_PG_s112  = 0
 w_k10_PG_s113_return_constraint: 1.41381 w_k10_PG_s11 - w_k10_PG_s113
                                 + 0.99 b_k10_PG_s113 - 1.01 s_k10_PG_s113  = 0
 w_k10_PG_s114_return_constraint: 1.11416 w_k10_PG_s11 - w_k10_PG_s114
                                 + 0.99 b_k10_PG_s114 - 1.01 s_k10_PG_s114  = 0
 w_k10_PG_s115_return_constraint: 1.11128 w_k10_PG_s11 - w_k10_PG_s115
                                 + 0.99 b_k10_PG_s115 - 1.01 s_k10_PG_s115  = 0
 w_k10_PG_s116_return_constraint: 0.963292 w_k10_PG_s11 - w_k10_PG_s116
                                 + 0.99 b_k10_PG_s116 - 1.01 s_k10_PG_s116  = 0
 w_k10_PG_s117_return_constraint: 1.33313 w_k10_PG_s11 - w_k10_PG_s117
                                 + 0.99 b_k10_PG_s117 - 1.01 s_k10_PG_s117  = 0
 w_k10_PG_s120_return_constraint: 1.04578 w_k10_PG_s12 - w_k10_PG_s120
                                 + 0.99 b_k10_PG_s120 - 1.01 s_k10_PG_s120  = 0
 w_k10_PG_s121_return_constraint: 0.949316 w_k10_PG_s12 - w_k10_PG_s121
                                 + 0.99 b_k10_PG_s121 - 1.01 s_k10_PG_s121  = 0
 w_k10_PG_s122_return_constraint: 0.809681 w_k10_PG_s12 - w_k10_PG_s122
                                 + 0.99 b_k10_PG_s122 - 1.01 s_k10_PG_s122  = 0
 w_k10_PG_s123_return_constraint: 1.02454 w_k10_PG_s12 - w_k10_PG_s123
                                 + 0.99 b_k10_PG_s123 - 1.01 s_k10_PG_s123  = 0
 w_k10_PG_s124_return_constraint: 0.95825 w_k10_PG_s12 - w_k10_PG_s124
                                 + 0.99 b_k10_PG_s124 - 1.01 s_k10_PG_s124  = 0
 w_k10_PG_s125_return_constraint: 1.24154 w_k10_PG_s12 - w_k10_PG_s125
                                 + 0.99 b_k10_PG_s125 - 1.01 s_k10_PG_s125  = 0
 w_k10_PG_s126_return_constraint: 0.929896 w_k10_PG_s12 - w_k10_PG_s126
                                 + 0.99 b_k10_PG_s126 - 1.01 s_k10_PG_s126  = 0
 w_k10_PG_s127_return_constraint: 0.827595 w_k10_PG_s12 - w_k10_PG_s127
                                 + 0.99 b_k10_PG_s127 - 1.01 s_k10_PG_s127  = 0
 w_k10_PG_s130_return_constraint: 1.00393 w_k10_PG_s13 - w_k10_PG_s130
                                 + 0.99 b_k10_PG_s130 - 1.01 s_k10_PG_s130  = 0
 w_k10_PG_s131_return_constraint: 1.27585 w_k10_PG_s13 - w_k10_PG_s131
                                 + 0.99 b_k10_PG_s131 - 1.01 s_k10_PG_s131  = 0
 w_k10_PG_s132_return_constraint: 0.967515 w_k10_PG_s13 - w_k10_PG_s132
                                 + 0.99 b_k10_PG_s132 - 1.01 s_k10_PG_s132  = 0
 w_k10_PG_s133_return_constraint: 0.922046 w_k10_PG_s13 - w_k10_PG_s133
                                 + 0.99 b_k10_PG_s133 - 1.01 s_k10_PG_s133  = 0
 w_k10_PG_s134_return_constraint: 1.193 w_k10_PG_s13 - w_k10_PG_s134
                                 + 0.99 b_k10_PG_s134 - 1.01 s_k10_PG_s134  = 0
 w_k10_PG_s135_return_constraint: 1.12501 w_k10_PG_s13 - w_k10_PG_s135
                                 + 0.99 b_k10_PG_s135 - 1.01 s_k10_PG_s135  = 0
 w_k10_PG_s136_return_constraint: 1.05175 w_k10_PG_s13 - w_k10_PG_s136
                                 + 0.99 b_k10_PG_s136 - 1.01 s_k10_PG_s136  = 0
 w_k10_PG_s137_return_constraint: 0.823624 w_k10_PG_s13 - w_k10_PG_s137
                                 + 0.99 b_k10_PG_s137 - 1.01 s_k10_PG_s137  = 0
 k10_s0_balance_constraint:      b_k10_ADM_s0 + b_k10_AXP_s0 + b_k10_DIS_s0
                                 + b_k10_FDX_s0 + b_k10_IBM_s0 + b_k10_INTC_s0
                                 + b_k10_KO_s0 + b_k10_MAT_s0 + b_k10_MSFT_s0
                                 + b_k10_PG_s0 - s_k10_ADM_s0 - s_k10_AXP_s0
                                 - s_k10_DIS_s0 - s_k10_FDX_s0 - s_k10_IBM_s0
                                 - s_k10_INTC_s0 - s_k10_KO_s0 - s_k10_MAT_s0
                                 - s_k10_MSFT_s0 - s_k10_PG_s0  = 0
 k10_s1_balance_constraint:      b_k10_ADM_s1 + b_k10_AXP_s1 + b_k10_DIS_s1
                                 + b_k10_FDX_s1 + b_k10_IBM_s1 + b_k10_INTC_s1
                                 + b_k10_KO_s1 + b_k10_MAT_s1 + b_k10_MSFT_s1
                                 + b_k10_PG_s1 - s_k10_ADM_s1 - s_k10_AXP_s1
                                 - s_k10_DIS_s1 - s_k10_FDX_s1 - s_k10_IBM_s1
                                 - s_k10_INTC_s1 - s_k10_KO_s1 - s_k10_MAT_s1
                                 - s_k10_MSFT_s1 - s_k10_PG_s1  = 0
 k10_s00_balance_constraint:     b_k10_ADM_s00 + b_k10_AXP_s00 + b_k10_DIS_s00
                                 + b_k10_FDX_s00 + b_k10_IBM_s00
                                 + b_k10_INTC_s00 + b_k10_KO_s00
                                 + b_k10_MAT_s00 + b_k10_MSFT_s00
                                 + b_k10_PG_s00 - s_k10_ADM_s00 - s_k10_AXP_s00
                                 - s_k10_DIS_s00 - s_k10_FDX_s00
                                 - s_k10_IBM_s00 - s_k10_INTC_s00
                                 - s_k10_KO_s00 - s_k10_MAT_s00
                                 - s_k10_MSFT_s00 - s_k10_PG_s00  = 0
 k10_s01_balance_constraint:     b_k10_ADM_s01 + b_k10_AXP_s01 + b_k10_DIS_s01
                                 + b_k10_FDX_s01 + b_k10_IBM_s01
                                 + b_k10_INTC_s01 + b_k10_KO_s01
                                 + b_k10_MAT_s01 + b_k10_MSFT_s01
                                 + b_k10_PG_s01 - s_k10_ADM_s01 - s_k10_AXP_s01
                                 - s_k10_DIS_s01 - s_k10_FDX_s01
                                 - s_k10_IBM_s01 - s_k10_INTC_s01
                                 - s_k10_KO_s01 - s_k10_MAT_s01
                                 - s_k10_MSFT_s01 - s_k10_PG_s01  = 0
 k10_s02_balance_constraint:     b_k10_ADM_s02 + b_k10_AXP_s02 + b_k10_DIS_s02
                                 + b_k10_FDX_s02 + b_k10_IBM_s02
                                 + b_k10_INTC_s02 + b_k10_KO_s02
                                 + b_k10_MAT_s02 + b_k10_MSFT_s02
                                 + b_k10_PG_s02 - s_k10_ADM_s02 - s_k10_AXP_s02
                                 - s_k10_DIS_s02 - s_k10_FDX_s02
                                 - s_k10_IBM_s02 - s_k10_INTC_s02
                                 - s_k10_KO_s02 - s_k10_MAT_s02
                                 - s_k10_MSFT_s02 - s_k10_PG_s02  = 0
 k10_s03_balance_constraint:     b_k10_ADM_s03 + b_k10_AXP_s03 + b_k10_DIS_s03
                                 + b_k10_FDX_s03 + b_k10_IBM_s03
                                 + b_k10_INTC_s03 + b_k10_KO_s03
                                 + b_k10_MAT_s03 + b_k10_MSFT_s03
                                 + b_k10_PG_s03 - s_k10_ADM_s03 - s_k10_AXP_s03
                                 - s_k10_DIS_s03 - s_k10_FDX_s03
                                 - s_k10_IBM_s03 - s_k10_INTC_s03
                                 - s_k10_KO_s03 - s_k10_MAT_s03
                                 - s_k10_MSFT_s03 - s_k10_PG_s03  = 0
 k10_s10_balance_constraint:     b_k10_ADM_s10 + b_k10_AXP_s10 + b_k10_DIS_s10
                                 + b_k10_FDX_s10 + b_k10_IBM_s10
                                 + b_k10_INTC_s10 + b_k10_KO_s10
                                 + b_k10_MAT_s10 + b_k10_MSFT_s10
                                 + b_k10_PG_s10 - s_k10_ADM_s10 - s_k10_AXP_s10
                                 - s_k10_DIS_s10 - s_k10_FDX_s10
                                 - s_k10_IBM_s10 - s_k10_INTC_s10
                                 - s_k10_KO_s10 - s_k10_MAT_s10
                                 - s_k10_MSFT_s10 - s_k10_PG_s10  = 0
 k10_s11_balance_constraint:     b_k10_ADM_s11 + b_k10_AXP_s11 + b_k10_DIS_s11
                                 + b_k10_FDX_s11 + b_k10_IBM_s11
                                 + b_k10_INTC_s11 + b_k10_KO_s11
                                 + b_k10_MAT_s11 + b_k10_MSFT_s11
                                 + b_k10_PG_s11 - s_k10_ADM_s11 - s_k10_AXP_s11
                                 - s_k10_DIS_s11 - s_k10_FDX_s11
                                 - s_k10_IBM_s11 - s_k10_INTC_s11
                                 - s_k10_KO_s11 - s_k10_MAT_s11
                                 - s_k10_MSFT_s11 - s_k10_PG_s11  = 0
 k10_s12_balance_constraint:     b_k10_ADM_s12 + b_k10_AXP_s12 + b_k10_DIS_s12
                                 + b_k10_FDX_s12 + b_k10_IBM_s12
                                 + b_k10_INTC_s12 + b_k10_KO_s12
                                 + b_k10_MAT_s12 + b_k10_MSFT_s12
                                 + b_k10_PG_s12 - s_k10_ADM_s12 - s_k10_AXP_s12
                                 - s_k10_DIS_s12 - s_k10_FDX_s12
                                 - s_k10_IBM_s12 - s_k10_INTC_s12
                                 - s_k10_KO_s12 - s_k10_MAT_s12
                                 - s_k10_MSFT_s12 - s_k10_PG_s12  = 0
 k10_s13_balance_constraint:     b_k10_ADM_s13 + b_k10_AXP_s13 + b_k10_DIS_s13
                                 + b_k10_FDX_s13 + b_k10_IBM_s13
                                 + b_k10_INTC_s13 + b_k10_KO_s13
                                 + b_k10_MAT_s13 + b_k10_MSFT_s13
                                 + b_k10_PG_s13 - s_k10_ADM_s13 - s_k10_AXP_s13
                                 - s_k10_DIS_s13 - s_k10_FDX_s13
                                 - s_k10_IBM_s13 - s_k10_INTC_s13
                                 - s_k10_KO_s13 - s_k10_MAT_s13
                                 - s_k10_MSFT_s13 - s_k10_PG_s13  = 0
 k10_s000_balance_constraint:    b_k10_ADM_s000 + b_k10_AXP_s000
                                 + b_k10_DIS_s000 + b_k10_FDX_s000
                                 + b_k10_IBM_s000 + b_k10_INTC_s000
                                 + b_k10_KO_s000 + b_k10_MAT_s000
                                 + b_k10_MSFT_s000 + b_k10_PG_s000
                                 - s_k10_ADM_s000 - s_k10_AXP_s000
                                 - s_k10_DIS_s000 - s_k10_FDX_s000
                                 - s_k10_IBM_s000 - s_k10_INTC_s000
                                 - s_k10_KO_s000 - s_k10_MAT_s000
                                 - s_k10_MSFT_s000 - s_k10_PG_s000  = 0
 k10_s001_balance_constraint:    b_k10_ADM_s001 + b_k10_AXP_s001
                                 + b_k10_DIS_s001 + b_k10_FDX_s001
                                 + b_k10_IBM_s001 + b_k10_INTC_s001
                                 + b_k10_KO_s001 + b_k10_MAT_s001
                                 + b_k10_MSFT_s001 + b_k10_PG_s001
                                 - s_k10_ADM_s001 - s_k10_AXP_s001
                                 - s_k10_DIS_s001 - s_k10_FDX_s001
                                 - s_k10_IBM_s001 - s_k10_INTC_s001
                                 - s_k10_KO_s001 - s_k10_MAT_s001
                                 - s_k10_MSFT_s001 - s_k10_PG_s001  = 0
 k10_s002_balance_constraint:    b_k10_ADM_s002 + b_k10_AXP_s002
                                 + b_k10_DIS_s002 + b_k10_FDX_s002
                                 + b_k10_IBM_s002 + b_k10_INTC_s002
                                 + b_k10_KO_s002 + b_k10_MAT_s002
                                 + b_k10_MSFT_s002 + b_k10_PG_s002
                                 - s_k10_ADM_s002 - s_k10_AXP_s002
                                 - s_k10_DIS_s002 - s_k10_FDX_s002
                                 - s_k10_IBM_s002 - s_k10_INTC_s002
                                 - s_k10_KO_s002 - s_k10_MAT_s002
                                 - s_k10_MSFT_s002 - s_k10_PG_s002  = 0
 k10_s003_balance_constraint:    b_k10_ADM_s003 + b_k10_AXP_s003
                                 + b_k10_DIS_s003 + b_k10_FDX_s003
                                 + b_k10_IBM_s003 + b_k10_INTC_s003
                                 + b_k10_KO_s003 + b_k10_MAT_s003
                                 + b_k10_MSFT_s003 + b_k10_PG_s003
                                 - s_k10_ADM_s003 - s_k10_AXP_s003
                                 - s_k10_DIS_s003 - s_k10_FDX_s003
                                 - s_k10_IBM_s003 - s_k10_INTC_s003
                                 - s_k10_KO_s003 - s_k10_MAT_s003
                                 - s_k10_MSFT_s003 - s_k10_PG_s003  = 0
 k10_s004_balance_constraint:    b_k10_ADM_s004 + b_k10_AXP_s004
                                 + b_k10_DIS_s004 + b_k10_FDX_s004
                                 + b_k10_IBM_s004 + b_k10_INTC_s004
                                 + b_k10_KO_s004 + b_k10_MAT_s004
                                 + b_k10_MSFT_s004 + b_k10_PG_s004
                                 - s_k10_ADM_s004 - s_k10_AXP_s004
                                 - s_k10_DIS_s004 - s_k10_FDX_s004
                                 - s_k10_IBM_s004 - s_k10_INTC_s004
                                 - s_k10_KO_s004 - s_k10_MAT_s004
                                 - s_k10_MSFT_s004 - s_k10_PG_s004  = 0
 k10_s005_balance_constraint:    b_k10_ADM_s005 + b_k10_AXP_s005
                                 + b_k10_DIS_s005 + b_k10_FDX_s005
                                 + b_k10_IBM_s005 + b_k10_INTC_s005
                                 + b_k10_KO_s005 + b_k10_MAT_s005
                                 + b_k10_MSFT_s005 + b_k10_PG_s005
                                 - s_k10_ADM_s005 - s_k10_AXP_s005
                                 - s_k10_DIS_s005 - s_k10_FDX_s005
                                 - s_k10_IBM_s005 - s_k10_INTC_s005
                                 - s_k10_KO_s005 - s_k10_MAT_s005
                                 - s_k10_MSFT_s005 - s_k10_PG_s005  = 0
 k10_s006_balance_constraint:    b_k10_ADM_s006 + b_k10_AXP_s006
                                 + b_k10_DIS_s006 + b_k10_FDX_s006
                                 + b_k10_IBM_s006 + b_k10_INTC_s006
                                 + b_k10_KO_s006 + b_k10_MAT_s006
                                 + b_k10_MSFT_s006 + b_k10_PG_s006
                                 - s_k10_ADM_s006 - s_k10_AXP_s006
                                 - s_k10_DIS_s006 - s_k10_FDX_s006
                                 - s_k10_IBM_s006 - s_k10_INTC_s006
                                 - s_k10_KO_s006 - s_k10_MAT_s006
                                 - s_k10_MSFT_s006 - s_k10_PG_s006  = 0
 k10_s007_balance_constraint:    b_k10_ADM_s007 + b_k10_AXP_s007
                                 + b_k10_DIS_s007 + b_k10_FDX_s007
                                 + b_k10_IBM_s007 + b_k10_INTC_s007
                                 + b_k10_KO_s007 + b_k10_MAT_s007
                                 + b_k10_MSFT_s007 + b_k10_PG_s007
                                 - s_k10_ADM_s007 - s_k10_AXP_s007
                                 - s_k10_DIS_s007 - s_k10_FDX_s007
                                 - s_k10_IBM_s007 - s_k10_INTC_s007
                                 - s_k10_KO_s007 - s_k10_MAT_s007
                                 - s_k10_MSFT_s007 - s_k10_PG_s007  = 0
 k10_s010_balance_constraint:    b_k10_ADM_s010 + b_k10_AXP_s010
                                 + b_k10_DIS_s010 + b_k10_FDX_s010
                                 + b_k10_IBM_s010 + b_k10_INTC_s010
                                 + b_k10_KO_s010 + b_k10_MAT_s010
                                 + b_k10_MSFT_s010 + b_k10_PG_s010
                                 - s_k10_ADM_s010 - s_k10_AXP_s010
                                 - s_k10_DIS_s010 - s_k10_FDX_s010
                                 - s_k10_IBM_s010 - s_k10_INTC_s010
                                 - s_k10_KO_s010 - s_k10_MAT_s010
                                 - s_k10_MSFT_s010 - s_k10_PG_s010  = 0
 k10_s011_balance_constraint:    b_k10_ADM_s011 + b_k10_AXP_s011
                                 + b_k10_DIS_s011 + b_k10_FDX_s011
                                 + b_k10_IBM_s011 + b_k10_INTC_s011
                                 + b_k10_KO_s011 + b_k10_MAT_s011
                                 + b_k10_MSFT_s011 + b_k10_PG_s011
                                 - s_k10_ADM_s011 - s_k10_AXP_s011
                                 - s_k10_DIS_s011 - s_k10_FDX_s011
                                 - s_k10_IBM_s011 - s_k10_INTC_s011
                                 - s_k10_KO_s011 - s_k10_MAT_s011
                                 - s_k10_MSFT_s011 - s_k10_PG_s011  = 0
 k10_s012_balance_constraint:    b_k10_ADM_s012 + b_k10_AXP_s012
                                 + b_k10_DIS_s012 + b_k10_FDX_s012
                                 + b_k10_IBM_s012 + b_k10_INTC_s012
                                 + b_k10_KO_s012 + b_k10_MAT_s012
                                 + b_k10_MSFT_s012 + b_k10_PG_s012
                                 - s_k10_ADM_s012 - s_k10_AXP_s012
                                 - s_k10_DIS_s012 - s_k10_FDX_s012
                                 - s_k10_IBM_s012 - s_k10_INTC_s012
                                 - s_k10_KO_s012 - s_k10_MAT_s012
                                 - s_k10_MSFT_s012 - s_k10_PG_s012  = 0
 k10_s013_balance_constraint:    b_k10_ADM_s013 + b_k10_AXP_s013
                                 + b_k10_DIS_s013 + b_k10_FDX_s013
                                 + b_k10_IBM_s013 + b_k10_INTC_s013
                                 + b_k10_KO_s013 + b_k10_MAT_s013
                                 + b_k10_MSFT_s013 + b_k10_PG_s013
                                 - s_k10_ADM_s013 - s_k10_AXP_s013
                                 - s_k10_DIS_s013 - s_k10_FDX_s013
                                 - s_k10_IBM_s013 - s_k10_INTC_s013
                                 - s_k10_KO_s013 - s_k10_MAT_s013
                                 - s_k10_MSFT_s013 - s_k10_PG_s013  = 0
 k10_s014_balance_constraint:    b_k10_ADM_s014 + b_k10_AXP_s014
                                 + b_k10_DIS_s014 + b_k10_FDX_s014
                                 + b_k10_IBM_s014 + b_k10_INTC_s014
                                 + b_k10_KO_s014 + b_k10_MAT_s014
                                 + b_k10_MSFT_s014 + b_k10_PG_s014
                                 - s_k10_ADM_s014 - s_k10_AXP_s014
                                 - s_k10_DIS_s014 - s_k10_FDX_s014
                                 - s_k10_IBM_s014 - s_k10_INTC_s014
                                 - s_k10_KO_s014 - s_k10_MAT_s014
                                 - s_k10_MSFT_s014 - s_k10_PG_s014  = 0
 k10_s015_balance_constraint:    b_k10_ADM_s015 + b_k10_AXP_s015
                                 + b_k10_DIS_s015 + b_k10_FDX_s015
                                 + b_k10_IBM_s015 + b_k10_INTC_s015
                                 + b_k10_KO_s015 + b_k10_MAT_s015
                                 + b_k10_MSFT_s015 + b_k10_PG_s015
                                 - s_k10_ADM_s015 - s_k10_AXP_s015
                                 - s_k10_DIS_s015 - s_k10_FDX_s015
                                 - s_k10_IBM_s015 - s_k10_INTC_s015
                                 - s_k10_KO_s015 - s_k10_MAT_s015
                                 - s_k10_MSFT_s015 - s_k10_PG_s015  = 0
 k10_s016_balance_constraint:    b_k10_ADM_s016 + b_k10_AXP_s016
                                 + b_k10_DIS_s016 + b_k10_FDX_s016
                                 + b_k10_IBM_s016 + b_k10_INTC_s016
                                 + b_k10_KO_s016 + b_k10_MAT_s016
                                 + b_k10_MSFT_s016 + b_k10_PG_s016
                                 - s_k10_ADM_s016 - s_k10_AXP_s016
                                 - s_k10_DIS_s016 - s_k10_FDX_s016
                                 - s_k10_IBM_s016 - s_k10_INTC_s016
                                 - s_k10_KO_s016 - s_k10_MAT_s016
                                 - s_k10_MSFT_s016 - s_k10_PG_s016  = 0
 k10_s017_balance_constraint:    b_k10_ADM_s017 + b_k10_AXP_s017
                                 + b_k10_DIS_s017 + b_k10_FDX_s017
                                 + b_k10_IBM_s017 + b_k10_INTC_s017
                                 + b_k10_KO_s017 + b_k10_MAT_s017
                                 + b_k10_MSFT_s017 + b_k10_PG_s017
                                 - s_k10_ADM_s017 - s_k10_AXP_s017
                                 - s_k10_DIS_s017 - s_k10_FDX_s017
                                 - s_k10_IBM_s017 - s_k10_INTC_s017
                                 - s_k10_KO_s017 - s_k10_MAT_s017
                                 - s_k10_MSFT_s017 - s_k10_PG_s017  = 0
 k10_s020_balance_constraint:    b_k10_ADM_s020 + b_k10_AXP_s020
                                 + b_k10_DIS_s020 + b_k10_FDX_s020
                                 + b_k10_IBM_s020 + b_k10_INTC_s020
                                 + b_k10_KO_s020 + b_k10_MAT_s020
                                 + b_k10_MSFT_s020 + b_k10_PG_s020
                                 - s_k10_ADM_s020 - s_k10_AXP_s020
                                 - s_k10_DIS_s020 - s_k10_FDX_s020
                                 - s_k10_IBM_s020 - s_k10_INTC_s020
                                 - s_k10_KO_s020 - s_k10_MAT_s020
                                 - s_k10_MSFT_s020 - s_k10_PG_s020  = 0
 k10_s021_balance_constraint:    b_k10_ADM_s021 + b_k10_AXP_s021
                                 + b_k10_DIS_s021 + b_k10_FDX_s021
                                 + b_k10_IBM_s021 + b_k10_INTC_s021
                                 + b_k10_KO_s021 + b_k10_MAT_s021
                                 + b_k10_MSFT_s021 + b_k10_PG_s021
                                 - s_k10_ADM_s021 - s_k10_AXP_s021
                                 - s_k10_DIS_s021 - s_k10_FDX_s021
                                 - s_k10_IBM_s021 - s_k10_INTC_s021
                                 - s_k10_KO_s021 - s_k10_MAT_s021
                                 - s_k10_MSFT_s021 - s_k10_PG_s021  = 0
 k10_s022_balance_constraint:    b_k10_ADM_s022 + b_k10_AXP_s022
                                 + b_k10_DIS_s022 + b_k10_FDX_s022
                                 + b_k10_IBM_s022 + b_k10_INTC_s022
                                 + b_k10_KO_s022 + b_k10_MAT_s022
                                 + b_k10_MSFT_s022 + b_k10_PG_s022
                                 - s_k10_ADM_s022 - s_k10_AXP_s022
                                 - s_k10_DIS_s022 - s_k10_FDX_s022
                                 - s_k10_IBM_s022 - s_k10_INTC_s022
                                 - s_k10_KO_s022 - s_k10_MAT_s022
                                 - s_k10_MSFT_s022 - s_k10_PG_s022  = 0
 k10_s023_balance_constraint:    b_k10_ADM_s023 + b_k10_AXP_s023
                                 + b_k10_DIS_s023 + b_k10_FDX_s023
                                 + b_k10_IBM_s023 + b_k10_INTC_s023
                                 + b_k10_KO_s023 + b_k10_MAT_s023
                                 + b_k10_MSFT_s023 + b_k10_PG_s023
                                 - s_k10_ADM_s023 - s_k10_AXP_s023
                                 - s_k10_DIS_s023 - s_k10_FDX_s023
                                 - s_k10_IBM_s023 - s_k10_INTC_s023
                                 - s_k10_KO_s023 - s_k10_MAT_s023
                                 - s_k10_MSFT_s023 - s_k10_PG_s023  = 0
 k10_s024_balance_constraint:    b_k10_ADM_s024 + b_k10_AXP_s024
                                 + b_k10_DIS_s024 + b_k10_FDX_s024
                                 + b_k10_IBM_s024 + b_k10_INTC_s024
                                 + b_k10_KO_s024 + b_k10_MAT_s024
                                 + b_k10_MSFT_s024 + b_k10_PG_s024
                                 - s_k10_ADM_s024 - s_k10_AXP_s024
                                 - s_k10_DIS_s024 - s_k10_FDX_s024
                                 - s_k10_IBM_s024 - s_k10_INTC_s024
                                 - s_k10_KO_s024 - s_k10_MAT_s024
                                 - s_k10_MSFT_s024 - s_k10_PG_s024  = 0
 k10_s025_balance_constraint:    b_k10_ADM_s025 + b_k10_AXP_s025
                                 + b_k10_DIS_s025 + b_k10_FDX_s025
                                 + b_k10_IBM_s025 + b_k10_INTC_s025
                                 + b_k10_KO_s025 + b_k10_MAT_s025
                                 + b_k10_MSFT_s025 + b_k10_PG_s025
                                 - s_k10_ADM_s025 - s_k10_AXP_s025
                                 - s_k10_DIS_s025 - s_k10_FDX_s025
                                 - s_k10_IBM_s025 - s_k10_INTC_s025
                                 - s_k10_KO_s025 - s_k10_MAT_s025
                                 - s_k10_MSFT_s025 - s_k10_PG_s025  = 0
 k10_s026_balance_constraint:    b_k10_ADM_s026 + b_k10_AXP_s026
                                 + b_k10_DIS_s026 + b_k10_FDX_s026
                                 + b_k10_IBM_s026 + b_k10_INTC_s026
                                 + b_k10_KO_s026 + b_k10_MAT_s026
                                 + b_k10_MSFT_s026 + b_k10_PG_s026
                                 - s_k10_ADM_s026 - s_k10_AXP_s026
                                 - s_k10_DIS_s026 - s_k10_FDX_s026
                                 - s_k10_IBM_s026 - s_k10_INTC_s026
                                 - s_k10_KO_s026 - s_k10_MAT_s026
                                 - s_k10_MSFT_s026 - s_k10_PG_s026  = 0
 k10_s027_balance_constraint:    b_k10_ADM_s027 + b_k10_AXP_s027
                                 + b_k10_DIS_s027 + b_k10_FDX_s027
                                 + b_k10_IBM_s027 + b_k10_INTC_s027
                                 + b_k10_KO_s027 + b_k10_MAT_s027
                                 + b_k10_MSFT_s027 + b_k10_PG_s027
                                 - s_k10_ADM_s027 - s_k10_AXP_s027
                                 - s_k10_DIS_s027 - s_k10_FDX_s027
                                 - s_k10_IBM_s027 - s_k10_INTC_s027
                                 - s_k10_KO_s027 - s_k10_MAT_s027
                                 - s_k10_MSFT_s027 - s_k10_PG_s027  = 0
 k10_s030_balance_constraint:    b_k10_ADM_s030 + b_k10_AXP_s030
                                 + b_k10_DIS_s030 + b_k10_FDX_s030
                                 + b_k10_IBM_s030 + b_k10_INTC_s030
                                 + b_k10_KO_s030 + b_k10_MAT_s030
                                 + b_k10_MSFT_s030 + b_k10_PG_s030
                                 - s_k10_ADM_s030 - s_k10_AXP_s030
                                 - s_k10_DIS_s030 - s_k10_FDX_s030
                                 - s_k10_IBM_s030 - s_k10_INTC_s030
                                 - s_k10_KO_s030 - s_k10_MAT_s030
                                 - s_k10_MSFT_s030 - s_k10_PG_s030  = 0
 k10_s031_balance_constraint:    b_k10_ADM_s031 + b_k10_AXP_s031
                                 + b_k10_DIS_s031 + b_k10_FDX_s031
                                 + b_k10_IBM_s031 + b_k10_INTC_s031
                                 + b_k10_KO_s031 + b_k10_MAT_s031
                                 + b_k10_MSFT_s031 + b_k10_PG_s031
                                 - s_k10_ADM_s031 - s_k10_AXP_s031
                                 - s_k10_DIS_s031 - s_k10_FDX_s031
                                 - s_k10_IBM_s031 - s_k10_INTC_s031
                                 - s_k10_KO_s031 - s_k10_MAT_s031
                                 - s_k10_MSFT_s031 - s_k10_PG_s031  = 0
 k10_s032_balance_constraint:    b_k10_ADM_s032 + b_k10_AXP_s032
                                 + b_k10_DIS_s032 + b_k10_FDX_s032
                                 + b_k10_IBM_s032 + b_k10_INTC_s032
                                 + b_k10_KO_s032 + b_k10_MAT_s032
                                 + b_k10_MSFT_s032 + b_k10_PG_s032
                                 - s_k10_ADM_s032 - s_k10_AXP_s032
                                 - s_k10_DIS_s032 - s_k10_FDX_s032
                                 - s_k10_IBM_s032 - s_k10_INTC_s032
                                 - s_k10_KO_s032 - s_k10_MAT_s032
                                 - s_k10_MSFT_s032 - s_k10_PG_s032  = 0
 k10_s033_balance_constraint:    b_k10_ADM_s033 + b_k10_AXP_s033
                                 + b_k10_DIS_s033 + b_k10_FDX_s033
                                 + b_k10_IBM_s033 + b_k10_INTC_s033
                                 + b_k10_KO_s033 + b_k10_MAT_s033
                                 + b_k10_MSFT_s033 + b_k10_PG_s033
                                 - s_k10_ADM_s033 - s_k10_AXP_s033
                                 - s_k10_DIS_s033 - s_k10_FDX_s033
                                 - s_k10_IBM_s033 - s_k10_INTC_s033
                                 - s_k10_KO_s033 - s_k10_MAT_s033
                                 - s_k10_MSFT_s033 - s_k10_PG_s033  = 0
 k10_s034_balance_constraint:    b_k10_ADM_s034 + b_k10_AXP_s034
                                 + b_k10_DIS_s034 + b_k10_FDX_s034
                                 + b_k10_IBM_s034 + b_k10_INTC_s034
                                 + b_k10_KO_s034 + b_k10_MAT_s034
                                 + b_k10_MSFT_s034 + b_k10_PG_s034
                                 - s_k10_ADM_s034 - s_k10_AXP_s034
                                 - s_k10_DIS_s034 - s_k10_FDX_s034
                                 - s_k10_IBM_s034 - s_k10_INTC_s034
                                 - s_k10_KO_s034 - s_k10_MAT_s034
                                 - s_k10_MSFT_s034 - s_k10_PG_s034  = 0
 k10_s035_balance_constraint:    b_k10_ADM_s035 + b_k10_AXP_s035
                                 + b_k10_DIS_s035 + b_k10_FDX_s035
                                 + b_k10_IBM_s035 + b_k10_INTC_s035
                                 + b_k10_KO_s035 + b_k10_MAT_s035
                                 + b_k10_MSFT_s035 + b_k10_PG_s035
                                 - s_k10_ADM_s035 - s_k10_AXP_s035
                                 - s_k10_DIS_s035 - s_k10_FDX_s035
                                 - s_k10_IBM_s035 - s_k10_INTC_s035
                                 - s_k10_KO_s035 - s_k10_MAT_s035
                                 - s_k10_MSFT_s035 - s_k10_PG_s035  = 0
 k10_s036_balance_constraint:    b_k10_ADM_s036 + b_k10_AXP_s036
                                 + b_k10_DIS_s036 + b_k10_FDX_s036
                                 + b_k10_IBM_s036 + b_k10_INTC_s036
                                 + b_k10_KO_s036 + b_k10_MAT_s036
                                 + b_k10_MSFT_s036 + b_k10_PG_s036
                                 - s_k10_ADM_s036 - s_k10_AXP_s036
                                 - s_k10_DIS_s036 - s_k10_FDX_s036
                                 - s_k10_IBM_s036 - s_k10_INTC_s036
                                 - s_k10_KO_s036 - s_k10_MAT_s036
                                 - s_k10_MSFT_s036 - s_k10_PG_s036  = 0
 k10_s037_balance_constraint:    b_k10_ADM_s037 + b_k10_AXP_s037
                                 + b_k10_DIS_s037 + b_k10_FDX_s037
                                 + b_k10_IBM_s037 + b_k10_INTC_s037
                                 + b_k10_KO_s037 + b_k10_MAT_s037
                                 + b_k10_MSFT_s037 + b_k10_PG_s037
                                 - s_k10_ADM_s037 - s_k10_AXP_s037
                                 - s_k10_DIS_s037 - s_k10_FDX_s037
                                 - s_k10_IBM_s037 - s_k10_INTC_s037
                                 - s_k10_KO_s037 - s_k10_MAT_s037
                                 - s_k10_MSFT_s037 - s_k10_PG_s037  = 0
 k10_s100_balance_constraint:    b_k10_ADM_s100 + b_k10_AXP_s100
                                 + b_k10_DIS_s100 + b_k10_FDX_s100
                                 + b_k10_IBM_s100 + b_k10_INTC_s100
                                 + b_k10_KO_s100 + b_k10_MAT_s100
                                 + b_k10_MSFT_s100 + b_k10_PG_s100
                                 - s_k10_ADM_s100 - s_k10_AXP_s100
                                 - s_k10_DIS_s100 - s_k10_FDX_s100
                                 - s_k10_IBM_s100 - s_k10_INTC_s100
                                 - s_k10_KO_s100 - s_k10_MAT_s100
                                 - s_k10_MSFT_s100 - s_k10_PG_s100  = 0
 k10_s101_balance_constraint:    b_k10_ADM_s101 + b_k10_AXP_s101
                                 + b_k10_DIS_s101 + b_k10_FDX_s101
                                 + b_k10_IBM_s101 + b_k10_INTC_s101
                                 + b_k10_KO_s101 + b_k10_MAT_s101
                                 + b_k10_MSFT_s101 + b_k10_PG_s101
                                 - s_k10_ADM_s101 - s_k10_AXP_s101
                                 - s_k10_DIS_s101 - s_k10_FDX_s101
                                 - s_k10_IBM_s101 - s_k10_INTC_s101
                                 - s_k10_KO_s101 - s_k10_MAT_s101
                                 - s_k10_MSFT_s101 - s_k10_PG_s101  = 0
 k10_s102_balance_constraint:    b_k10_ADM_s102 + b_k10_AXP_s102
                                 + b_k10_DIS_s102 + b_k10_FDX_s102
                                 + b_k10_IBM_s102 + b_k10_INTC_s102
                                 + b_k10_KO_s102 + b_k10_MAT_s102
                                 + b_k10_MSFT_s102 + b_k10_PG_s102
                                 - s_k10_ADM_s102 - s_k10_AXP_s102
                                 - s_k10_DIS_s102 - s_k10_FDX_s102
                                 - s_k10_IBM_s102 - s_k10_INTC_s102
                                 - s_k10_KO_s102 - s_k10_MAT_s102
                                 - s_k10_MSFT_s102 - s_k10_PG_s102  = 0
 k10_s103_balance_constraint:    b_k10_ADM_s103 + b_k10_AXP_s103
                                 + b_k10_DIS_s103 + b_k10_FDX_s103
                                 + b_k10_IBM_s103 + b_k10_INTC_s103
                                 + b_k10_KO_s103 + b_k10_MAT_s103
                                 + b_k10_MSFT_s103 + b_k10_PG_s103
                                 - s_k10_ADM_s103 - s_k10_AXP_s103
                                 - s_k10_DIS_s103 - s_k10_FDX_s103
                                 - s_k10_IBM_s103 - s_k10_INTC_s103
                                 - s_k10_KO_s103 - s_k10_MAT_s103
                                 - s_k10_MSFT_s103 - s_k10_PG_s103  = 0
 k10_s104_balance_constraint:    b_k10_ADM_s104 + b_k10_AXP_s104
                                 + b_k10_DIS_s104 + b_k10_FDX_s104
                                 + b_k10_IBM_s104 + b_k10_INTC_s104
                                 + b_k10_KO_s104 + b_k10_MAT_s104
                                 + b_k10_MSFT_s104 + b_k10_PG_s104
                                 - s_k10_ADM_s104 - s_k10_AXP_s104
                                 - s_k10_DIS_s104 - s_k10_FDX_s104
                                 - s_k10_IBM_s104 - s_k10_INTC_s104
                                 - s_k10_KO_s104 - s_k10_MAT_s104
                                 - s_k10_MSFT_s104 - s_k10_PG_s104  = 0
 k10_s105_balance_constraint:    b_k10_ADM_s105 + b_k10_AXP_s105
                                 + b_k10_DIS_s105 + b_k10_FDX_s105
                                 + b_k10_IBM_s105 + b_k10_INTC_s105
                                 + b_k10_KO_s105 + b_k10_MAT_s105
                                 + b_k10_MSFT_s105 + b_k10_PG_s105
                                 - s_k10_ADM_s105 - s_k10_AXP_s105
                                 - s_k10_DIS_s105 - s_k10_FDX_s105
                                 - s_k10_IBM_s105 - s_k10_INTC_s105
                                 - s_k10_KO_s105 - s_k10_MAT_s105
                                 - s_k10_MSFT_s105 - s_k10_PG_s105  = 0
 k10_s106_balance_constraint:    b_k10_ADM_s106 + b_k10_AXP_s106
                                 + b_k10_DIS_s106 + b_k10_FDX_s106
                                 + b_k10_IBM_s106 + b_k10_INTC_s106
                                 + b_k10_KO_s106 + b_k10_MAT_s106
                                 + b_k10_MSFT_s106 + b_k10_PG_s106
                                 - s_k10_ADM_s106 - s_k10_AXP_s106
                                 - s_k10_DIS_s106 - s_k10_FDX_s106
                                 - s_k10_IBM_s106 - s_k10_INTC_s106
                                 - s_k10_KO_s106 - s_k10_MAT_s106
                                 - s_k10_MSFT_s106 - s_k10_PG_s106  = 0
 k10_s107_balance_constraint:    b_k10_ADM_s107 + b_k10_AXP_s107
                                 + b_k10_DIS_s107 + b_k10_FDX_s107
                                 + b_k10_IBM_s107 + b_k10_INTC_s107
                                 + b_k10_KO_s107 + b_k10_MAT_s107
                                 + b_k10_MSFT_s107 + b_k10_PG_s107
                                 - s_k10_ADM_s107 - s_k10_AXP_s107
                                 - s_k10_DIS_s107 - s_k10_FDX_s107
                                 - s_k10_IBM_s107 - s_k10_INTC_s107
                                 - s_k10_KO_s107 - s_k10_MAT_s107
                                 - s_k10_MSFT_s107 - s_k10_PG_s107  = 0
 k10_s110_balance_constraint:    b_k10_ADM_s110 + b_k10_AXP_s110
                                 + b_k10_DIS_s110 + b_k10_FDX_s110
                                 + b_k10_IBM_s110 + b_k10_INTC_s110
                                 + b_k10_KO_s110 + b_k10_MAT_s110
                                 + b_k10_MSFT_s110 + b_k10_PG_s110
                                 - s_k10_ADM_s110 - s_k10_AXP_s110
                                 - s_k10_DIS_s110 - s_k10_FDX_s110
                                 - s_k10_IBM_s110 - s_k10_INTC_s110
                                 - s_k10_KO_s110 - s_k10_MAT_s110
                                 - s_k10_MSFT_s110 - s_k10_PG_s110  = 0
 k10_s111_balance_constraint:    b_k10_ADM_s111 + b_k10_AXP_s111
                                 + b_k10_DIS_s111 + b_k10_FDX_s111
                                 + b_k10_IBM_s111 + b_k10_INTC_s111
                                 + b_k10_KO_s111 + b_k10_MAT_s111
                                 + b_k10_MSFT_s111 + b_k10_PG_s111
                                 - s_k10_ADM_s111 - s_k10_AXP_s111
                                 - s_k10_DIS_s111 - s_k10_FDX_s111
                                 - s_k10_IBM_s111 - s_k10_INTC_s111
                                 - s_k10_KO_s111 - s_k10_MAT_s111
                                 - s_k10_MSFT_s111 - s_k10_PG_s111  = 0
 k10_s112_balance_constraint:    b_k10_ADM_s112 + b_k10_AXP_s112
                                 + b_k10_DIS_s112 + b_k10_FDX_s112
                                 + b_k10_IBM_s112 + b_k10_INTC_s112
                                 + b_k10_KO_s112 + b_k10_MAT_s112
                                 + b_k10_MSFT_s112 + b_k10_PG_s112
                                 - s_k10_ADM_s112 - s_k10_AXP_s112
                                 - s_k10_DIS_s112 - s_k10_FDX_s112
                                 - s_k10_IBM_s112 - s_k10_INTC_s112
                                 - s_k10_KO_s112 - s_k10_MAT_s112
                                 - s_k10_MSFT_s112 - s_k10_PG_s112  = 0
 k10_s113_balance_constraint:    b_k10_ADM_s113 + b_k10_AXP_s113
                                 + b_k10_DIS_s113 + b_k10_FDX_s113
                                 + b_k10_IBM_s113 + b_k10_INTC_s113
                                 + b_k10_KO_s113 + b_k10_MAT_s113
                                 + b_k10_MSFT_s113 + b_k10_PG_s113
                                 - s_k10_ADM_s113 - s_k10_AXP_s113
                                 - s_k10_DIS_s113 - s_k10_FDX_s113
                                 - s_k10_IBM_s113 - s_k10_INTC_s113
                                 - s_k10_KO_s113 - s_k10_MAT_s113
                                 - s_k10_MSFT_s113 - s_k10_PG_s113  = 0
 k10_s114_balance_constraint:    b_k10_ADM_s114 + b_k10_AXP_s114
                                 + b_k10_DIS_s114 + b_k10_FDX_s114
                                 + b_k10_IBM_s114 + b_k10_INTC_s114
                                 + b_k10_KO_s114 + b_k10_MAT_s114
                                 + b_k10_MSFT_s114 + b_k10_PG_s114
                                 - s_k10_ADM_s114 - s_k10_AXP_s114
                                 - s_k10_DIS_s114 - s_k10_FDX_s114
                                 - s_k10_IBM_s114 - s_k10_INTC_s114
                                 - s_k10_KO_s114 - s_k10_MAT_s114
                                 - s_k10_MSFT_s114 - s_k10_PG_s114  = 0
 k10_s115_balance_constraint:    b_k10_ADM_s115 + b_k10_AXP_s115
                                 + b_k10_DIS_s115 + b_k10_FDX_s115
                                 + b_k10_IBM_s115 + b_k10_INTC_s115
                                 + b_k10_KO_s115 + b_k10_MAT_s115
                                 + b_k10_MSFT_s115 + b_k10_PG_s115
                                 - s_k10_ADM_s115 - s_k10_AXP_s115
                                 - s_k10_DIS_s115 - s_k10_FDX_s115
                                 - s_k10_IBM_s115 - s_k10_INTC_s115
                                 - s_k10_KO_s115 - s_k10_MAT_s115
                                 - s_k10_MSFT_s115 - s_k10_PG_s115  = 0
 k10_s116_balance_constraint:    b_k10_ADM_s116 + b_k10_AXP_s116
                                 + b_k10_DIS_s116 + b_k10_FDX_s116
                                 + b_k10_IBM_s116 + b_k10_INTC_s116
                                 + b_k10_KO_s116 + b_k10_MAT_s116
                                 + b_k10_MSFT_s116 + b_k10_PG_s116
                                 - s_k10_ADM_s116 - s_k10_AXP_s116
                                 - s_k10_DIS_s116 - s_k10_FDX_s116
                                 - s_k10_IBM_s116 - s_k10_INTC_s116
                                 - s_k10_KO_s116 - s_k10_MAT_s116
                                 - s_k10_MSFT_s116 - s_k10_PG_s116  = 0
 k10_s117_balance_constraint:    b_k10_ADM_s117 + b_k10_AXP_s117
                                 + b_k10_DIS_s117 + b_k10_FDX_s117
                                 + b_k10_IBM_s117 + b_k10_INTC_s117
                                 + b_k10_KO_s117 + b_k10_MAT_s117
                                 + b_k10_MSFT_s117 + b_k10_PG_s117
                                 - s_k10_ADM_s117 - s_k10_AXP_s117
                                 - s_k10_DIS_s117 - s_k10_FDX_s117
                                 - s_k10_IBM_s117 - s_k10_INTC_s117
                                 - s_k10_KO_s117 - s_k10_MAT_s117
                                 - s_k10_MSFT_s117 - s_k10_PG_s117  = 0
 k10_s120_balance_constraint:    b_k10_ADM_s120 + b_k10_AXP_s120
                                 + b_k10_DIS_s120 + b_k10_FDX_s120
                                 + b_k10_IBM_s120 + b_k10_INTC_s120
                                 + b_k10_KO_s120 + b_k10_MAT_s120
                                 + b_k10_MSFT_s120 + b_k10_PG_s120
                                 - s_k10_ADM_s120 - s_k10_AXP_s120
                                 - s_k10_DIS_s120 - s_k10_FDX_s120
                                 - s_k10_IBM_s120 - s_k10_INTC_s120
                                 - s_k10_KO_s120 - s_k10_MAT_s120
                                 - s_k10_MSFT_s120 - s_k10_PG_s120  = 0
 k10_s121_balance_constraint:    b_k10_ADM_s121 + b_k10_AXP_s121
                                 + b_k10_DIS_s121 + b_k10_FDX_s121
                                 + b_k10_IBM_s121 + b_k10_INTC_s121
                                 + b_k10_KO_s121 + b_k10_MAT_s121
                                 + b_k10_MSFT_s121 + b_k10_PG_s121
                                 - s_k10_ADM_s121 - s_k10_AXP_s121
                                 - s_k10_DIS_s121 - s_k10_FDX_s121
                                 - s_k10_IBM_s121 - s_k10_INTC_s121
                                 - s_k10_KO_s121 - s_k10_MAT_s121
                                 - s_k10_MSFT_s121 - s_k10_PG_s121  = 0
 k10_s122_balance_constraint:    b_k10_ADM_s122 + b_k10_AXP_s122
                                 + b_k10_DIS_s122 + b_k10_FDX_s122
                                 + b_k10_IBM_s122 + b_k10_INTC_s122
                                 + b_k10_KO_s122 + b_k10_MAT_s122
                                 + b_k10_MSFT_s122 + b_k10_PG_s122
                                 - s_k10_ADM_s122 - s_k10_AXP_s122
                                 - s_k10_DIS_s122 - s_k10_FDX_s122
                                 - s_k10_IBM_s122 - s_k10_INTC_s122
                                 - s_k10_KO_s122 - s_k10_MAT_s122
                                 - s_k10_MSFT_s122 - s_k10_PG_s122  = 0
 k10_s123_balance_constraint:    b_k10_ADM_s123 + b_k10_AXP_s123
                                 + b_k10_DIS_s123 + b_k10_FDX_s123
                                 + b_k10_IBM_s123 + b_k10_INTC_s123
                                 + b_k10_KO_s123 + b_k10_MAT_s123
                                 + b_k10_MSFT_s123 + b_k10_PG_s123
                                 - s_k10_ADM_s123 - s_k10_AXP_s123
                                 - s_k10_DIS_s123 - s_k10_FDX_s123
                                 - s_k10_IBM_s123 - s_k10_INTC_s123
                                 - s_k10_KO_s123 - s_k10_MAT_s123
                                 - s_k10_MSFT_s123 - s_k10_PG_s123  = 0
 k10_s124_balance_constraint:    b_k10_ADM_s124 + b_k10_AXP_s124
                                 + b_k10_DIS_s124 + b_k10_FDX_s124
                                 + b_k10_IBM_s124 + b_k10_INTC_s124
                                 + b_k10_KO_s124 + b_k10_MAT_s124
                                 + b_k10_MSFT_s124 + b_k10_PG_s124
                                 - s_k10_ADM_s124 - s_k10_AXP_s124
                                 - s_k10_DIS_s124 - s_k10_FDX_s124
                                 - s_k10_IBM_s124 - s_k10_INTC_s124
                                 - s_k10_KO_s124 - s_k10_MAT_s124
                                 - s_k10_MSFT_s124 - s_k10_PG_s124  = 0
 k10_s125_balance_constraint:    b_k10_ADM_s125 + b_k10_AXP_s125
                                 + b_k10_DIS_s125 + b_k10_FDX_s125
                                 + b_k10_IBM_s125 + b_k10_INTC_s125
                                 + b_k10_KO_s125 + b_k10_MAT_s125
                                 + b_k10_MSFT_s125 + b_k10_PG_s125
                                 - s_k10_ADM_s125 - s_k10_AXP_s125
                                 - s_k10_DIS_s125 - s_k10_FDX_s125
                                 - s_k10_IBM_s125 - s_k10_INTC_s125
                                 - s_k10_KO_s125 - s_k10_MAT_s125
                                 - s_k10_MSFT_s125 - s_k10_PG_s125  = 0
 k10_s126_balance_constraint:    b_k10_ADM_s126 + b_k10_AXP_s126
                                 + b_k10_DIS_s126 + b_k10_FDX_s126
                                 + b_k10_IBM_s126 + b_k10_INTC_s126
                                 + b_k10_KO_s126 + b_k10_MAT_s126
                                 + b_k10_MSFT_s126 + b_k10_PG_s126
                                 - s_k10_ADM_s126 - s_k10_AXP_s126
                                 - s_k10_DIS_s126 - s_k10_FDX_s126
                                 - s_k10_IBM_s126 - s_k10_INTC_s126
                                 - s_k10_KO_s126 - s_k10_MAT_s126
                                 - s_k10_MSFT_s126 - s_k10_PG_s126  = 0
 k10_s127_balance_constraint:    b_k10_ADM_s127 + b_k10_AXP_s127
                                 + b_k10_DIS_s127 + b_k10_FDX_s127
                                 + b_k10_IBM_s127 + b_k10_INTC_s127
                                 + b_k10_KO_s127 + b_k10_MAT_s127
                                 + b_k10_MSFT_s127 + b_k10_PG_s127
                                 - s_k10_ADM_s127 - s_k10_AXP_s127
                                 - s_k10_DIS_s127 - s_k10_FDX_s127
                                 - s_k10_IBM_s127 - s_k10_INTC_s127
                                 - s_k10_KO_s127 - s_k10_MAT_s127
                                 - s_k10_MSFT_s127 - s_k10_PG_s127  = 0
 k10_s130_balance_constraint:    b_k10_ADM_s130 + b_k10_AXP_s130
                                 + b_k10_DIS_s130 + b_k10_FDX_s130
                                 + b_k10_IBM_s130 + b_k10_INTC_s130
                                 + b_k10_KO_s130 + b_k10_MAT_s130
                                 + b_k10_MSFT_s130 + b_k10_PG_s130
                                 - s_k10_ADM_s130 - s_k10_AXP_s130
                                 - s_k10_DIS_s130 - s_k10_FDX_s130
                                 - s_k10_IBM_s130 - s_k10_INTC_s130
                                 - s_k10_KO_s130 - s_k10_MAT_s130
                                 - s_k10_MSFT_s130 - s_k10_PG_s130  = 0
 k10_s131_balance_constraint:    b_k10_ADM_s131 + b_k10_AXP_s131
                                 + b_k10_DIS_s131 + b_k10_FDX_s131
                                 + b_k10_IBM_s131 + b_k10_INTC_s131
                                 + b_k10_KO_s131 + b_k10_MAT_s131
                                 + b_k10_MSFT_s131 + b_k10_PG_s131
                                 - s_k10_ADM_s131 - s_k10_AXP_s131
                                 - s_k10_DIS_s131 - s_k10_FDX_s131
                                 - s_k10_IBM_s131 - s_k10_INTC_s131
                                 - s_k10_KO_s131 - s_k10_MAT_s131
                                 - s_k10_MSFT_s131 - s_k10_PG_s131  = 0
 k10_s132_balance_constraint:    b_k10_ADM_s132 + b_k10_AXP_s132
                                 + b_k10_DIS_s132 + b_k10_FDX_s132
                                 + b_k10_IBM_s132 + b_k10_INTC_s132
                                 + b_k10_KO_s132 + b_k10_MAT_s132
                                 + b_k10_MSFT_s132 + b_k10_PG_s132
                                 - s_k10_ADM_s132 - s_k10_AXP_s132
                                 - s_k10_DIS_s132 - s_k10_FDX_s132
                                 - s_k10_IBM_s132 - s_k10_INTC_s132
                                 - s_k10_KO_s132 - s_k10_MAT_s132
                                 - s_k10_MSFT_s132 - s_k10_PG_s132  = 0
 k10_s133_balance_constraint:    b_k10_ADM_s133 + b_k10_AXP_s133
                                 + b_k10_DIS_s133 + b_k10_FDX_s133
                                 + b_k10_IBM_s133 + b_k10_INTC_s133
                                 + b_k10_KO_s133 + b_k10_MAT_s133
                                 + b_k10_MSFT_s133 + b_k10_PG_s133
                                 - s_k10_ADM_s133 - s_k10_AXP_s133
                                 - s_k10_DIS_s133 - s_k10_FDX_s133
                                 - s_k10_IBM_s133 - s_k10_INTC_s133
                                 - s_k10_KO_s133 - s_k10_MAT_s133
                                 - s_k10_MSFT_s133 - s_k10_PG_s133  = 0
 k10_s134_balance_constraint:    b_k10_ADM_s134 + b_k10_AXP_s134
                                 + b_k10_DIS_s134 + b_k10_FDX_s134
                                 + b_k10_IBM_s134 + b_k10_INTC_s134
                                 + b_k10_KO_s134 + b_k10_MAT_s134
                                 + b_k10_MSFT_s134 + b_k10_PG_s134
                                 - s_k10_ADM_s134 - s_k10_AXP_s134
                                 - s_k10_DIS_s134 - s_k10_FDX_s134
                                 - s_k10_IBM_s134 - s_k10_INTC_s134
                                 - s_k10_KO_s134 - s_k10_MAT_s134
                                 - s_k10_MSFT_s134 - s_k10_PG_s134  = 0
 k10_s135_balance_constraint:    b_k10_ADM_s135 + b_k10_AXP_s135
                                 + b_k10_DIS_s135 + b_k10_FDX_s135
                                 + b_k10_IBM_s135 + b_k10_INTC_s135
                                 + b_k10_KO_s135 + b_k10_MAT_s135
                                 + b_k10_MSFT_s135 + b_k10_PG_s135
                                 - s_k10_ADM_s135 - s_k10_AXP_s135
                                 - s_k10_DIS_s135 - s_k10_FDX_s135
                                 - s_k10_IBM_s135 - s_k10_INTC_s135
                                 - s_k10_KO_s135 - s_k10_MAT_s135
                                 - s_k10_MSFT_s135 - s_k10_PG_s135  = 0
 k10_s136_balance_constraint:    b_k10_ADM_s136 + b_k10_AXP_s136
                                 + b_k10_DIS_s136 + b_k10_FDX_s136
                                 + b_k10_IBM_s136 + b_k10_INTC_s136
                                 + b_k10_KO_s136 + b_k10_MAT_s136
                                 + b_k10_MSFT_s136 + b_k10_PG_s136
                                 - s_k10_ADM_s136 - s_k10_AXP_s136
                                 - s_k10_DIS_s136 - s_k10_FDX_s136
                                 - s_k10_IBM_s136 - s_k10_INTC_s136
                                 - s_k10_KO_s136 - s_k10_MAT_s136
                                 - s_k10_MSFT_s136 - s_k10_PG_s136  = 0
 k10_s137_balance_constraint:    b_k10_ADM_s137 + b_k10_AXP_s137
                                 + b_k10_DIS_s137 + b_k10_FDX_s137
                                 + b_k10_IBM_s137 + b_k10_INTC_s137
                                 + b_k10_KO_s137 + b_k10_MAT_s137
                                 + b_k10_MSFT_s137 + b_k10_PG_s137
                                 - s_k10_ADM_s137 - s_k10_AXP_s137
                                 - s_k10_DIS_s137 - s_k10_FDX_s137
                                 - s_k10_IBM_s137 - s_k10_INTC_s137
                                 - s_k10_KO_s137 - s_k10_MAT_s137
                                 - s_k10_MSFT_s137 - s_k10_PG_s137  = 0
 k10_wcvar_constraint:           wcvar - var_10 - 0.104000268839527 z_k10_s0000
                                 - 0.299000489100445 z_k10_s0001
                                 - 0.077000133549297 z_k10_s0002
                                 - 0.253000046161012 z_k10_s0003
                                 - 0.140000568726795 z_k10_s0004
                                 - 0.121000022077006 z_k10_s0005
                                 - 0.0810001668701045 z_k10_s0006
                                 - 0.11999980462129 z_k10_s0007
                                 - 0.27899933722743 z_k10_s0010
                                 - 0.25299892399231 z_k10_s0011
                                 - 0.46699839062918 z_k10_s0012
                                 - 0.472999201374047 z_k10_s0013
                                 - 0.131999701430146 z_k10_s0014
                                 - 0.194999611771681 z_k10_s0015
                                 - 0.213999466636869 z_k10_s0016
                                 - 0.312000308832481 z_k10_s0017
                                 - 0.0730002313742072 z_k10_s0020
                                 - 0.288000053200569 z_k10_s0021
                                 - 0.144000365800988 z_k10_s0022
                                 - 0.348001265619846 z_k10_s0023
                                 - 0.141000305180024 z_k10_s0024
                                 - 0.258001142994448 z_k10_s0025
                                 - 0.166000414620569 z_k10_s0026
                                 - 0.278000416465195 z_k10_s0027
                                 - 0.308001368375398 z_k10_s0030
                                 - 0.260001224431472 z_k10_s0031
                                 - 0.584000861847346 z_k10_s0032
                                 - 0.261001116246502 z_k10_s0033
                                 - 0.276001018278376 z_k10_s0034
                                 - 0.766001757069537 z_k10_s0035
                                 - 1.03700560348241 z_k10_s0036
                                 - 0.320001213760576 z_k10_s0037
                                 - 0.223999865619938 z_k10_s0040
                                 - 0.190999959225122 z_k10_s0041
                                 - 0.226999857110375 z_k10_s0042
                                 - 0.366999960998055 z_k10_s0043
                                 - 0.101000064211814 z_k10_s0044
                                 - 0.234000839255896 z_k10_s0045
                                 - 0.0790000264151532 z_k10_s0046
                                 - 0.080000123778749 z_k10_s0047
                                 - 0.159000213016517 z_k10_s0050
                                 - 0.535001626425568 z_k10_s0051
                                 - 0.431002459964146 z_k10_s0052
                                 - 0.56000113326659 z_k10_s0053
                                 - 0.249000569149043 z_k10_s0054
                                 - 0.195000416369666 z_k10_s0055
                                 - 0.231000619722815 z_k10_s0056
                                 - 0.685001712478624 z_k10_s0057
                                 - 0.277000737002388 z_k10_s0060
                                 - 0.329000563759515 z_k10_s0061
                                 - 0.685999318430427 z_k10_s0062
                                 - 1.02400449606637 z_k10_s0063
                                 - 0.384000958673201 z_k10_s0064
                                 - 0.518000474939556 z_k10_s0065
                                 - 0.86100013415477 z_k10_s0066
                                 - 0.770003589187105 z_k10_s0067
                                 - 0.281000475416988 z_k10_s0070
                                 - 0.275000627145782 z_k10_s0071
                                 - 0.513001422217557 z_k10_s0072
                                 - 0.34100097343317 z_k10_s0073
                                 - 0.362000442382391 z_k10_s0074
                                 - 0.458999908770814 z_k10_s0075
                                 - 0.128000313892991 z_k10_s0076
                                 - 0.520000285532001 z_k10_s0077
                                 - 0.213999179438549 z_k10_s0100
                                 - 0.901998087022273 z_k10_s0101
                                 - 0.389998615667979 z_k10_s0102
                                 - 0.466999239240185 z_k10_s0103
                                 - 0.537999152666317 z_k10_s0104
                                 - 0.186999379768847 z_k10_s0105
                                 - 0.511999471354294 z_k10_s0106
                                 - 0.185999261411168 z_k10_s0107
                                 - 0.747996453824257 z_k10_s0110
                                 - 0.354000640618634 z_k10_s0111
                                 - 0.570997875009871 z_k10_s0112
                                 - 0.152999385468275 z_k10_s0113
                                 - 0.401000105818393 z_k10_s0114
                                 - 0.489997462775872 z_k10_s0115
                                 - 0.425997524058993 z_k10_s0116
                                 - 0.339999021372819 z_k10_s0117
                                 - 0.301999991479401 z_k10_s0120
                                 - 0.251000208653489 z_k10_s0121
                                 - 0.257000302868347 z_k10_s0122
                                 - 0.325000691969813 z_k10_s0123
                                 - 0.277999614619972 z_k10_s0124
                                 - 0.157999969923651 z_k10_s0125
                                 - 0.289999803049687 z_k10_s0126
                                 - 0.17499996539898 z_k10_s0127
                                 - 0.464997396742055 z_k10_s0130
                                 - 0.139999381358085 z_k10_s0131
                                 - 0.33799825452576 z_k10_s0132
                                 - 0.10099942232898 z_k10_s0133
                                 - 0.187999039241497 z_k10_s0134
                                 - 0.277997734176165 z_k10_s0135
                                 - 0.17199915328036 z_k10_s0136
                                 - 0.213998569529747 z_k10_s0137
                                 - 0.349999752322233 z_k10_s0140
                                 - 0.185999438015613 z_k10_s0141
                                 - 0.479998055590414 z_k10_s0142
                                 - 0.278000299282774 z_k10_s0143
                                 - 0.27399976233813 z_k10_s0144
                                 - 0.460998696342146 z_k10_s0145
                                 - 0.341998678432946 z_k10_s0146
                                 - 0.181999411669176 z_k10_s0147
                                 - 0.123999830642946 z_k10_s0150
                                 - 0.217998950889146 z_k10_s0151
                                 - 0.370999185915891 z_k10_s0152
                                 - 0.0949999344586451 z_k10_s0153
                                 - 0.0959999016719257 z_k10_s0154
                                 - 0.169999508252682 z_k10_s0155
                                 - 0.194000076570078 z_k10_s0156
                                 - 0.425999923643815 z_k10_s0157
                                 - 0.250999866339776 z_k10_s0160
                                 - 0.136999922308969 z_k10_s0161
                                 - 0.199000225623422 z_k10_s0162
                                 - 0.242999098461687 z_k10_s0163
                                 - 0.0889998332330246 z_k10_s0164
                                 - 0.0639998413350475 z_k10_s0165
                                 - 0.0529999447757738 z_k10_s0166
                                 - 0.153000269067097 z_k10_s0167
                                 - 0.204000118659571 z_k10_s0170
                                 - 0.116999792761124 z_k10_s0171
                                 - 0.231000070835215 z_k10_s0172
                                 - 0.114999780303697 z_k10_s0173
                                 - 0.103999783787722 z_k10_s0174
                                 - 0.381999643373359 z_k10_s0175
                                 - 0.131999814191952 z_k10_s0176
                                 - 0.154999885452488 z_k10_s0177
                                 - 0.242999423614862 z_k10_s0200
                                 - 0.416999668340109 z_k10_s0201
                                 - 0.480999463807514 z_k10_s0202
                                 - 0.64899892690945 z_k10_s0203
                                 - 0.456999540507237 z_k10_s0204
                                 - 0.665999873578282 z_k10_s0205
                                 - 0.624999003609174 z_k10_s0206
                                 - 0.466000512742643 z_k10_s0207
                                 - 0.617998206759274 z_k10_s0210
                                 - 0.131999609413657 z_k10_s0211
                                 - 0.349999102073624 z_k10_s0212
                                 - 0.167999361236043 z_k10_s0213
                                 - 0.199999227200093 z_k10_s0214
                                 - 0.596998935852571 z_k10_s0215
                                 - 0.26799910468565 z_k10_s0216
                                 - 0.263999738225556 z_k10_s0217
                                 - 0.148999943765151 z_k10_s0220
                                 - 0.256999430921028 z_k10_s0221
                                 - 0.279999060404184 z_k10_s0222
                                 - 0.489000638056224 z_k10_s0223
                                 - 0.520999206902216 z_k10_s0224
                                 - 0.157999936128146 z_k10_s0225
                                 - 0.109999976858839 z_k10_s0226
                                 - 0.142000019905054 z_k10_s0227
                                 - 0.164000047934493 z_k10_s0230
                                 - 0.26599896247453 z_k10_s0231
                                 - 0.536000363902923 z_k10_s0232
                                 - 0.378000173200218 z_k10_s0233
                                 - 0.18300000576838 z_k10_s0234
                                 - 0.399000679711211 z_k10_s0235
                                 - 0.107999889771533 z_k10_s0236
                                 - 0.201999963602266 z_k10_s0237
                                 - 0.348999433830002 z_k10_s0240
                                 - 0.469999466120225 z_k10_s0241
                                 - 0.394000496605484 z_k10_s0242
                                 - 0.116999771621588 z_k10_s0243
                                 - 0.219999628101174 z_k10_s0244
                                 - 0.237999554612256 z_k10_s0245
                                 - 0.505998321944167 z_k10_s0246
                                 - 0.198999630738393 z_k10_s0247
                                 - 0.133999940415402 z_k10_s0250
                                 - 0.288999866004117 z_k10_s0251
                                 - 0.317999145053338 z_k10_s0252
                                 - 0.0780000289865446 z_k10_s0253
                                 - 0.156999749057692 z_k10_s0254
                                 - 0.224999630999881 z_k10_s0255
                                 - 0.118999866948794 z_k10_s0256
                                 - 0.150999543151139 z_k10_s0257
                                 - 0.836997946884661 z_k10_s0260
                                 - 0.243999152157837 z_k10_s0261
                                 - 0.248999254629549 z_k10_s0262
                                 - 0.459998129232839 z_k10_s0263
                                 - 0.351997643798454 z_k10_s0264
                                 - 0.385998473525682 z_k10_s0265
                                 - 0.368998058662068 z_k10_s0266
                                 - 0.425997299505611 z_k10_s0267
                                 - 0.61100091231981 z_k10_s0270
                                 - 0.432999754617163 z_k10_s0271
                                 - 0.701001771253577 z_k10_s0272
                                 - 0.41299990806502 z_k10_s0273
                                 - 0.452000870535984 z_k10_s0274
                                 - 0.533001643576374 z_k10_s0275
                                 - 0.405999939636782 z_k10_s0276
                                 - 0.877997557826668 z_k10_s0277
                                 - 0.346999523642686 z_k10_s0300
                                 - 0.131999638126159 z_k10_s0301
                                 - 0.110999791596779 z_k10_s0302
                                 - 0.152999333955881 z_k10_s0303
                                 - 0.188999975061333 z_k10_s0304
                                 - 0.114999812597499 z_k10_s0305
                                 - 0.159000194305076 z_k10_s0306
                                 - 0.300999960282863 z_k10_s0307
                                 - 0.0829996630903812 z_k10_s0310
                                 - 0.0669996338300625 z_k10_s0311
                                 - 0.157999166751337 z_k10_s0312
                                 - 0.0399997835522652 z_k10_s0313
                                 - 0.137999492174103 z_k10_s0314
                                 - 0.0439997908673448 z_k10_s0315
                                 - 0.0769999055164766 z_k10_s0316
                                 - 0.116999254670945 z_k10_s0317
                                 - 0.0739997416873301 z_k10_s0320
                                 - 0.0969995650770299 z_k10_s0321
                                 - 0.23999877693346 z_k10_s0322
                                 - 0.0919996499573396 z_k10_s0323
                                 - 0.0999996091884549 z_k10_s0324
                                 - 0.125999951887633 z_k10_s0325
                                 - 0.34299914303042 z_k10_s0326
                                 - 0.115999527650686 z_k10_s0327
                                 - 0.0549996827054275 z_k10_s0330
                                 - 0.120998910954303 z_k10_s0331
                                 - 0.0549996827054275 z_k10_s0332
                                 - 0.0519997142305001 z_k10_s0333
                                 - 0.12499959878313 z_k10_s0334
                                 - 0.139999050160129 z_k10_s0335
                                 - 0.140999222117336 z_k10_s0336
                                 - 0.0929995701194422 z_k10_s0337
                                 - 0.0479998866802717 z_k10_s0340
                                 - 0.0570001036510027 z_k10_s0341
                                 - 0.0799997286339834 z_k10_s0342
                                 - 0.0169999423347215 z_k10_s0343
                                 - 0.0620000866906267 z_k10_s0344
                                 - 0.0770000358094986 z_k10_s0345
                                 - 0.0799997286339834 z_k10_s0346
                                 - 0.0739998479861979 z_k10_s0347
                                 - 0.0629996518705434 z_k10_s0350
                                 - 0.107999815348894 z_k10_s0351
                                 - 0.07699995917454 z_k10_s0352
                                 - 0.1179997875664 z_k10_s0353
                                 - 0.0879998709138819 z_k10_s0354
                                 - 0.028999977130682 z_k10_s0355
                                 - 0.0399999465699385 z_k10_s0356
                                 - 0.0539999076744469 z_k10_s0357
                                 - 0.388000097274018 z_k10_s0360
                                 - 0.0869998899353068 z_k10_s0361
                                 - 0.108999865778713 z_k10_s0362
                                 - 0.245999468231124 z_k10_s0363
                                 - 0.0909999436885859 z_k10_s0364
                                 - 0.288000352440183 z_k10_s0365
                                 - 0.286000005763915 z_k10_s0366
                                 - 0.103999958486929 z_k10_s0367
                                 - 0.0219999494651646 z_k10_s0370
                                 - 0.0390000812788017 z_k10_s0371
                                 - 0.0279999454465529 z_k10_s0372
                                 - 0.0469999327209492 z_k10_s0373
                                 - 0.0430000069998755 z_k10_s0374
                                 - 0.0679998112560306 z_k10_s0375
                                 - 0.069999953116197 z_k10_s0376
                                 - 0.0409998651397091 z_k10_s0377
                                 - 0.0690003053039653 z_k10_s1000
                                 - 0.0570002795077361 z_k10_s1001
                                 - 0.135000850834891 z_k10_s1002
                                 - 0.210001124186786 z_k10_s1003
                                 - 0.10400040296806 z_k10_s1004
                                 - 0.04900017260988 z_k10_s1005
                                 - 0.110000460716359 z_k10_s1006
                                 - 0.163000821325722 z_k10_s1007
                                 - 0.0989995680222642 z_k10_s1010
                                 - 0.0850000125918082 z_k10_s1011
                                 - 0.133000043938945 z_k10_s1012
                                 - 0.271000244424452 z_k10_s1013
                                 - 0.127000248483034 z_k10_s1014
                                 - 0.0890000397290673 z_k10_s1015
                                 - 0.0969999959035946 z_k10_s1016
                                 - 0.0799999541452365 z_k10_s1017
                                 - 0.0830003940073248 z_k10_s1020
                                 - 0.121000730981219 z_k10_s1021
                                 - 0.0500001776487387 z_k10_s1022
                                 - 0.119000757535412 z_k10_s1023
                                 - 0.0480002042029313 z_k10_s1024
                                 - 0.105000008410811 z_k10_s1025
                                 - 0.176000468242897 z_k10_s1026
                                 - 0.233001113954331 z_k10_s1027
                                 - 0.181001137067432 z_k10_s1030
                                 - 0.115000362908546 z_k10_s1031
                                 - 0.19100087736245 z_k10_s1032
                                 - 0.0400002052859552 z_k10_s1033
                                 - 0.14600061052808 z_k10_s1034
                                 - 0.151000959177852 z_k10_s1035
                                 - 0.0430002230749132 z_k10_s1036
                                 - 0.0900004379682861 z_k10_s1037
                                 - 0.0909999471929246 z_k10_s1040
                                 - 0.0809999207321299 z_k10_s1041
                                 - 0.0580000800725324 z_k10_s1042
                                 - 0.163999993556572 z_k10_s1043
                                 - 0.158999980326175 z_k10_s1044
                                 - 0.08200036377867 z_k10_s1045
                                 - 0.0530000668421351 z_k10_s1046
                                 - 0.0460000483195788 z_k10_s1047
                                 - 0.0480001436407563 z_k10_s1050
                                 - 0.169000592299911 z_k10_s1051
                                 - 0.184000531906197 z_k10_s1052
                                 - 0.0820001564819558 z_k10_s1053
                                 - 0.17300072592854 z_k10_s1054
                                 - 0.157000191414026 z_k10_s1055
                                 - 0.116000506223798 z_k10_s1056
                                 - 0.194000865977768 z_k10_s1057
                                 - 0.161999892832188 z_k10_s1060
                                 - 0.320999574848993 z_k10_s1061
                                 - 0.109000066026578 z_k10_s1062
                                 - 0.199999875160725 z_k10_s1063
                                 - 0.117999892072832 z_k10_s1064
                                 - 0.414999594798523 z_k10_s1065
                                 - 0.172999993822014 z_k10_s1066
                                 - 0.51799903759436 z_k10_s1067
                                 - 0.0430000215202756 z_k10_s1070
                                 - 0.0280000209899471 z_k10_s1071
                                 - 0.133999804737594 z_k10_s1072
                                 - 0.115000204065858 z_k10_s1073
                                 - 0.048000001697051 z_k10_s1074
                                 - 0.058000022050604 z_k10_s1075
                                 - 0.127000204490121 z_k10_s1076
                                 - 0.0469999816616951 z_k10_s1077
                                 - 0.584998981535986 z_k10_s1100
                                 - 0.194000048065991 z_k10_s1101
                                 - 0.427000028234381 z_k10_s1102
                                 - 0.464000735444771 z_k10_s1103
                                 - 0.447000794159125 z_k10_s1104
                                 - 0.205000043474368 z_k10_s1105
                                 - 0.266000250293676 z_k10_s1106
                                 - 0.25100010170035 z_k10_s1107
                                 - 0.271999522307295 z_k10_s1110
                                 - 0.115999966477983 z_k10_s1111
                                 - 0.0909998771836628 z_k10_s1112
                                 - 0.178999426636247 z_k10_s1113
                                 - 0.0789999500583017 z_k10_s1114
                                 - 0.284999684476254 z_k10_s1115
                                 - 0.0699999732643048 z_k10_s1116
                                 - 0.165999264467288 z_k10_s1117
                                 - 0.409998804488243 z_k10_s1120
                                 - 0.109999932352686 z_k10_s1121
                                 - 0.214999879284784 z_k10_s1122
                                 - 0.393000625756443 z_k10_s1123
                                 - 0.214999879284784 z_k10_s1124
                                 - 0.353000742391736 z_k10_s1125
                                 - 0.442000735977951 z_k10_s1126
                                 - 0.393000625756443 z_k10_s1127
                                 - 0.247999283563511 z_k10_s1130
                                 - 0.270999393055177 z_k10_s1131
                                 - 0.069999985444421 z_k10_s1132
                                 - 0.125999440520294 z_k10_s1133
                                 - 0.177000368349209 z_k10_s1134
                                 - 0.141000355057805 z_k10_s1135
                                 - 0.0869999715209297 z_k10_s1136
                                 - 0.0919999531669707 z_k10_s1137
                                 - 0.0719999351110109 z_k10_s1140
                                 - 0.0800000045122142 z_k10_s1141
                                 - 0.0830000182251687 z_k10_s1142
                                 - 0.183000146990404 z_k10_s1143
                                 - 0.184999696465828 z_k10_s1144
                                 - 0.149999700647982 z_k10_s1145
                                 - 0.03999995300612 z_k10_s1146
                                 - 0.192000089629293 z_k10_s1147
                                 - 0.158000004131583 z_k10_s1150
                                 - 0.229000218553375 z_k10_s1151
                                 - 0.365999813875687 z_k10_s1152
                                 - 0.389999976457704 z_k10_s1153
                                 - 0.11499997938881 z_k10_s1154
                                 - 0.262999560377877 z_k10_s1155
                                 - 0.125999965055565 z_k10_s1156
                                 - 0.129999992152568 z_k10_s1157
                                 - 0.110999701492864 z_k10_s1160
                                 - 0.187999239999317 z_k10_s1161
                                 - 0.109999676922721 z_k10_s1162
                                 - 0.140999598099096 z_k10_s1163
                                 - 0.135999643347992 z_k10_s1164
                                 - 0.484999309048472 z_k10_s1165
                                 - 0.248999225881533 z_k10_s1166
                                 - 0.260999856922459 z_k10_s1167
                                 - 0.152000443732426 z_k10_s1170
                                 - 0.267000507897763 z_k10_s1171
                                 - 0.137000224232099 z_k10_s1172
                                 - 0.518000457927745 z_k10_s1173
                                 - 0.3820005720631 z_k10_s1174
                                 - 0.277001463566562 z_k10_s1175
                                 - 0.141000363699044 z_k10_s1176
                                 - 0.554000499127378 z_k10_s1177
                                 - 0.0819997726757212 z_k10_s1200
                                 - 0.0590001159497224 z_k10_s1201
                                 - 0.047000001075164 z_k10_s1202
                                 - 0.076999772561342 z_k10_s1203
                                 - 0.0619997722182045 z_k10_s1204
                                 - 0.047000001075164 z_k10_s1205
                                 - 0.140000003202616 z_k10_s1206
                                 - 0.0590001159497224 z_k10_s1207
                                 - 0.138999417714884 z_k10_s1210
                                 - 0.0479997744671107 z_k10_s1211
                                 - 0.0639997239227132 z_k10_s1212
                                 - 0.077999651983979 z_k10_s1213
                                 - 0.0409998473863261 z_k10_s1214
                                 - 0.202999511136081 z_k10_s1215
                                 - 0.0639997239227132 z_k10_s1216
                                 - 0.101999317518444 z_k10_s1217
                                 - 0.0670001153221141 z_k10_s1220
                                 - 0.124999800974339 z_k10_s1221
                                 - 0.0499999758897026 z_k10_s1222
                                 - 0.0809999820313058 z_k10_s1223
                                 - 0.0799998726235768 z_k10_s1224
                                 - 0.0730002167688167 z_k10_s1225
                                 - 0.0489999774819079 z_k10_s1226
                                 - 0.0300000077338084 z_k10_s1227
                                 - 0.0920003780167566 z_k10_s1230
                                 - 0.0510000519330941 z_k10_s1231
                                 - 0.156999875538938 z_k10_s1232
                                 - 0.0720000733173092 z_k10_s1233
                                 - 0.0469999909932046 z_k10_s1234
                                 - 0.0800000245969713 z_k10_s1235
                                 - 0.221000509376936 z_k10_s1236
                                 - 0.132999851099835 z_k10_s1237
                                 - 0.203999572911215 z_k10_s1240
                                 - 0.243999952304522 z_k10_s1241
                                 - 0.119999957182595 z_k10_s1242
                                 - 0.177999916804228 z_k10_s1243
                                 - 0.111999763204202 z_k10_s1244
                                 - 0.0929997601419796 z_k10_s1245
                                 - 0.110999694669503 z_k10_s1246
                                 - 0.118999652448432 z_k10_s1247
                                 - 0.0379999218298145 z_k10_s1250
                                 - 0.0459997328207226 z_k10_s1251
                                 - 0.0669999266076738 z_k10_s1252
                                 - 0.0279999402167108 z_k10_s1253
                                 - 0.070999686853482 z_k10_s1254
                                 - 0.0229999494101589 z_k10_s1255
                                 - 0.104999723937792 z_k10_s1256
                                 - 0.0369998987685649 z_k10_s1257
                                 - 0.0399999776546882 z_k10_s1260
                                 - 0.0349999488229165 z_k10_s1261
                                 - 0.0169999715282811 z_k10_s1262
                                 - 0.0259999981255216 z_k10_s1263
                                 - 0.0389998706885397 z_k10_s1264
                                 - 0.0259999981255216 z_k10_s1265
                                 - 0.0310000269572933 z_k10_s1266
                                 - 0.0389998706885397 z_k10_s1267
                                 - 0.0549999679286762 z_k10_s1270
                                 - 0.0459999822585238 z_k10_s1271
                                 - 0.0789997632158479 z_k10_s1272
                                 - 0.099999840779471 z_k10_s1273
                                 - 0.0499999758897026 z_k10_s1274
                                 - 0.057000131244167 z_k10_s1275
                                 - 0.0359999981805767 z_k10_s1276
                                 - 0.13200001182877 z_k10_s1277
                                 - 0.0639998901290378 z_k10_s1300
                                 - 0.0330000603477013 z_k10_s1301
                                 - 0.0459999210302459 z_k10_s1302
                                 - 0.0159999959322427 z_k10_s1303
                                 - 0.0169999942155089 z_k10_s1304
                                 - 0.0159999959322427 z_k10_s1305
                                 - 0.027000070648104 z_k10_s1306
                                 - 0.0149999976489765 z_k10_s1307
                                 - 0.0220000746805374 z_k10_s1310
                                 - 0.0179999973931911 z_k10_s1311
                                 - 0.0379999896300719 z_k10_s1312
                                 - 0.0330000025208475 z_k10_s1313
                                 - 0.0330000025208475 z_k10_s1314
                                 - 0.0420000231174348 z_k10_s1315
                                 - 0.0099999961184404 z_k10_s1316
                                 - 0.0230000283023988 z_k10_s1317
                                 - 0.105999732911434 z_k10_s1320
                                 - 0.0589998871563158 z_k10_s1321
                                 - 0.0539999151057557 z_k10_s1322
                                 - 0.129999544514195 z_k10_s1323
                                 - 0.0919997298099755 z_k10_s1324
                                 - 0.0619998975066152 z_k10_s1325
                                 - 0.100000119010286 z_k10_s1326
                                 - 0.0749995807584008 z_k10_s1327
                                 - 0.0459999899205149 z_k10_s1330
                                 - 0.0210000598986431 z_k10_s1331
                                 - 0.0420000507971287 z_k10_s1332
                                 - 0.050000274044688 z_k10_s1333
                                 - 0.0610002791343935 z_k10_s1334
                                 - 0.0300000609265767 z_k10_s1335
                                 - 0.0550000254485273 z_k10_s1336
                                 - 0.0400000812354357 z_k10_s1337
                                 - 0.0329999768629779 z_k10_s1340
                                 - 0.011000000587661 z_k10_s1341
                                 - 0.0339999610709437 z_k10_s1342
                                 - 0.020999992067351 z_k10_s1343
                                 - 0.011000000587661 z_k10_s1344
                                 - 0.0410000995267567 z_k10_s1345
                                 - 0.0440000521506544 z_k10_s1346
                                 - 0.0539998942303133 z_k10_s1347
                                 - 0.0699998215680115 z_k10_s1350
                                 - 0.0389998189879696 z_k10_s1351
                                 - 0.0519999826504135 z_k10_s1352
                                 - 0.061000070109053 z_k10_s1353
                                 - 0.0199999778194045 z_k10_s1354
                                 - 0.0410000250897123 z_k10_s1355
                                 - 0.0269999599762057 z_k10_s1356
                                 - 0.0259999913252067 z_k10_s1357
                                 - 0.0560000868498789 z_k10_s1360
                                 - 0.0370000308650693 z_k10_s1361
                                 - 0.0919999016102492 z_k10_s1362
                                 - 0.108000119281856 z_k10_s1363
                                 - 0.0970003071329991 z_k10_s1364
                                 - 0.0440000585963228 z_k10_s1365
                                 - 0.131000094684418 z_k10_s1366
                                 - 0.110000146490807 z_k10_s1367
                                 - 0.039000007588075 z_k10_s1370
                                 - 0.0270000739611928 z_k10_s1371
                                 - 0.0150000479340177 z_k10_s1372
                                 - 0.0400000867572503 z_k10_s1373
                                 - 0.0370001572507005 z_k10_s1374
                                 - 0.0140000303650376 z_k10_s1375
                                 - 0.0670000683181502 z_k10_s1376
                                 - 0.0690002266565008 z_k10_s1377 >= 0
 z_k10_s0000_constraint:         var_10 + z_k10_s0000 + 1.19513 w_k10_ADM_s000
                                 + 1.00936 w_k10_AXP_s000
                                 + 0.994826 w_k10_DIS_s000
                                 + 1.12343 w_k10_FDX_s000
                                 + 0.820759 w_k10_IBM_s000
                                 + 1.04916 w_k10_INTC_s000
                                 + 0.91816 w_k10_KO_s000
                                 + 0.556592 w_k10_MAT_s000
                                 + 1.07671 w_k10_MSFT_s000
                                 + 0.952698 w_k10_PG_s000 >= 1
 z_k10_s0001_constraint:         var_10 + z_k10_s0001 + 1.02378 w_k10_ADM_s000
                                 + 0.923988 w_k10_AXP_s000
                                 + 0.978058 w_k10_DIS_s000
                                 + 0.923561 w_k10_FDX_s000
                                 + 1.01936 w_k10_IBM_s000
                                 + 1.0713 w_k10_INTC_s000
                                 + 0.934716 w_k10_KO_s000
                                 + 0.922998 w_k10_MAT_s000
                                 + 0.976521 w_k10_MSFT_s000
                                 + 1.01211 w_k10_PG_s000 >= 1
 z_k10_s0002_constraint:         var_10 + z_k10_s0002 + 1.23163 w_k10_ADM_s000
                                 + 1.54041 w_k10_AXP_s000
                                 + 1.67499 w_k10_DIS_s000
                                 + 1.34363 w_k10_FDX_s000
                                 + 1.10102 w_k10_IBM_s000
                                 + 1.34619 w_k10_INTC_s000
                                 + 1.32378 w_k10_KO_s000
                                 + 0.955116 w_k10_MAT_s000
                                 + 1.35617 w_k10_MSFT_s000
                                 + 1.2513 w_k10_PG_s000 >= 1
 z_k10_s0003_constraint:         var_10 + z_k10_s0003 + 0.790993 w_k10_ADM_s000
                                 + 0.894348 w_k10_AXP_s000
                                 + 0.652926 w_k10_DIS_s000
                                 + 0.764679 w_k10_FDX_s000
                                 + 0.956877 w_k10_IBM_s000
                                 + 0.682614 w_k10_INTC_s000
                                 + 0.854426 w_k10_KO_s000
                                 + 0.956894 w_k10_MAT_s000
                                 + 0.73322 w_k10_MSFT_s000
                                 + 0.959982 w_k10_PG_s000 >= 1
 z_k10_s0004_constraint:         var_10 + z_k10_s0004 + 1.01885 w_k10_ADM_s000
                                 + 1.2742 w_k10_AXP_s000
                                 + 1.13769 w_k10_DIS_s000
                                 + 1.07511 w_k10_FDX_s000
                                 + 1.38843 w_k10_IBM_s000
                                 + 1.15155 w_k10_INTC_s000
                                 + 1.15797 w_k10_KO_s000
                                 + 0.979521 w_k10_MAT_s000
                                 + 0.985062 w_k10_MSFT_s000
                                 + 1.13231 w_k10_PG_s000 >= 1
 z_k10_s0005_constraint:         var_10 + z_k10_s0005 + 0.83432 w_k10_ADM_s000
                                 + 1.31489 w_k10_AXP_s000
                                 + 1.02281 w_k10_DIS_s000
                                 + 1.07096 w_k10_FDX_s000
                                 + 0.890337 w_k10_IBM_s000
                                 + 1.00302 w_k10_INTC_s000
                                 + 0.844202 w_k10_KO_s000
                                 + 0.815383 w_k10_MAT_s000
                                 + 0.796186 w_k10_MSFT_s000
                                 + 0.993057 w_k10_PG_s000 >= 1
 z_k10_s0006_constraint:         var_10 + z_k10_s0006 + 0.874306 w_k10_ADM_s000
                                 + 1.37175 w_k10_AXP_s000
                                 + 1.1788 w_k10_DIS_s000
                                 + 1.14866 w_k10_FDX_s000
                                 + 1.19091 w_k10_IBM_s000
                                 + 0.918366 w_k10_INTC_s000
                                 + 1.06434 w_k10_KO_s000
                                 + 1.14775 w_k10_MAT_s000
                                 + 0.97104 w_k10_MSFT_s000
                                 + 1.13238 w_k10_PG_s000 >= 1
 z_k10_s0007_constraint:         var_10 + z_k10_s0007 + 0.91785 w_k10_ADM_s000
                                 + 1.03019 w_k10_AXP_s000
                                 + 1.53323 w_k10_DIS_s000
                                 + 1.15329 w_k10_FDX_s000
                                 + 1.32876 w_k10_IBM_s000
                                 + 1.26048 w_k10_INTC_s000
                                 + 1.25803 w_k10_KO_s000
                                 + 1.22775 w_k10_MAT_s000
                                 + 1.31626 w_k10_MSFT_s000
                                 + 1.16252 w_k10_PG_s000 >= 1
 z_k10_s0010_constraint:         var_10 + z_k10_s0010 + 1.00054 w_k10_ADM_s001
                                 + 1.05322 w_k10_AXP_s001
                                 + 0.935663 w_k10_DIS_s001
                                 + 0.994063 w_k10_FDX_s001
                                 + 1.08948 w_k10_IBM_s001
                                 + 0.970969 w_k10_INTC_s001
                                 + 0.979783 w_k10_KO_s001
                                 + 1.10109 w_k10_MAT_s001
                                 + 1.02034 w_k10_MSFT_s001
                                 + 1.03878 w_k10_PG_s001 >= 1
 z_k10_s0011_constraint:         var_10 + z_k10_s0011 + 0.992763 w_k10_ADM_s001
                                 + 0.946484 w_k10_AXP_s001
                                 + 1.05469 w_k10_DIS_s001
                                 + 1.10651 w_k10_FDX_s001
                                 + 0.830341 w_k10_IBM_s001
                                 + 1.03489 w_k10_INTC_s001
                                 + 0.899803 w_k10_KO_s001
                                 + 0.771094 w_k10_MAT_s001
                                 + 0.998352 w_k10_MSFT_s001
                                 + 1.0078 w_k10_PG_s001 >= 1
 z_k10_s0012_constraint:         var_10 + z_k10_s0012 + 1.0524 w_k10_ADM_s001
                                 + 0.954907 w_k10_AXP_s001
                                 + 0.766367 w_k10_DIS_s001
                                 + 0.938618 w_k10_FDX_s001
                                 + 0.768928 w_k10_IBM_s001
                                 + 0.896804 w_k10_INTC_s001
                                 + 0.818201 w_k10_KO_s001
                                 + 0.757142 w_k10_MAT_s001
                                 + 0.945511 w_k10_MSFT_s001
                                 + 0.95676 w_k10_PG_s001 >= 1
 z_k10_s0013_constraint:         var_10 + z_k10_s0013 + 0.749284 w_k10_ADM_s001
                                 + 0.911352 w_k10_AXP_s001
                                 + 0.727507 w_k10_DIS_s001
                                 + 0.758049 w_k10_FDX_s001
                                 + 1.08959 w_k10_IBM_s001
                                 + 0.981506 w_k10_INTC_s001
                                 + 0.802631 w_k10_KO_s001
                                 + 1.10426 w_k10_MAT_s001
                                 + 0.873239 w_k10_MSFT_s001
                                 + 0.822132 w_k10_PG_s001 >= 1
 z_k10_s0014_constraint:         var_10 + z_k10_s0014 + 1.18385 w_k10_ADM_s001
                                 + 1.22302 w_k10_AXP_s001
                                 + 1.64882 w_k10_DIS_s001
                                 + 1.43056 w_k10_FDX_s001
                                 + 1.11403 w_k10_IBM_s001
                                 + 1.05897 w_k10_INTC_s001
                                 + 1.26583 w_k10_KO_s001
                                 + 1.39319 w_k10_MAT_s001
                                 + 1.33673 w_k10_MSFT_s001
                                 + 1.33113 w_k10_PG_s001 >= 1
 z_k10_s0015_constraint:         var_10 + z_k10_s0015 + 0.918138 w_k10_ADM_s001
                                 + 1.1283 w_k10_AXP_s001
                                 + 1.23635 w_k10_DIS_s001
                                 + 0.967825 w_k10_FDX_s001
                                 + 1.56754 w_k10_IBM_s001
                                 + 0.807885 w_k10_INTC_s001
                                 + 1.14496 w_k10_KO_s001
                                 + 1.34435 w_k10_MAT_s001
                                 + 1.05894 w_k10_MSFT_s001
                                 + 1.11118 w_k10_PG_s001 >= 1
 z_k10_s0016_constraint:         var_10 + z_k10_s0016 + 1.17365 w_k10_ADM_s001
                                 + 1.3827 w_k10_AXP_s001
                                 + 1.51918 w_k10_DIS_s001
                                 + 1.25002 w_k10_FDX_s001
                                 + 1.62463 w_k10_IBM_s001
                                 + 1.11903 w_k10_INTC_s001
                                 + 1.31729 w_k10_KO_s001
                                 + 1.74458 w_k10_MAT_s001
                                 + 1.4141 w_k10_MSFT_s001
                                 + 1.23276 w_k10_PG_s001 >= 1
 z_k10_s0017_constraint:         var_10 + z_k10_s0017 + 1.18988 w_k10_ADM_s001
                                 + 1.32681 w_k10_AXP_s001
                                 + 1.151 w_k10_DIS_s001
                                 + 1.10402 w_k10_FDX_s001
                                 + 1.16039 w_k10_IBM_s001
                                 + 0.962906 w_k10_INTC_s001
                                 + 1.1328 w_k10_KO_s001
                                 + 1.49954 w_k10_MAT_s001
                                 + 1.07604 w_k10_MSFT_s001
                                 + 1.16171 w_k10_PG_s001 >= 1
 z_k10_s0020_constraint:         var_10 + z_k10_s0020 + 1.24931 w_k10_ADM_s002
                                 + 1.71439 w_k10_AXP_s002
                                 + 1.37081 w_k10_DIS_s002
                                 + 1.29003 w_k10_FDX_s002
                                 + 1.36693 w_k10_IBM_s002
                                 + 1.13178 w_k10_INTC_s002
                                 + 1.18736 w_k10_KO_s002
                                 + 1.59292 w_k10_MAT_s002
                                 + 1.12641 w_k10_MSFT_s002
                                 + 1.22825 w_k10_PG_s002 >= 1
 z_k10_s0021_constraint:         var_10 + z_k10_s0021 + 0.853782 w_k10_ADM_s002
                                 + 0.804381 w_k10_AXP_s002
                                 + 0.856158 w_k10_DIS_s002
                                 + 0.848699 w_k10_FDX_s002
                                 + 0.960432 w_k10_IBM_s002
                                 + 0.914165 w_k10_INTC_s002
                                 + 0.950013 w_k10_KO_s002
                                 + 0.926916 w_k10_MAT_s002
                                 + 0.841231 w_k10_MSFT_s002
                                 + 0.849633 w_k10_PG_s002 >= 1
 z_k10_s0022_constraint:         var_10 + z_k10_s0022 + 0.881068 w_k10_ADM_s002
                                 + 1.20132 w_k10_AXP_s002
                                 + 1.34116 w_k10_DIS_s002
                                 + 1.00123 w_k10_FDX_s002
                                 + 1.44003 w_k10_IBM_s002
                                 + 1.11748 w_k10_INTC_s002
                                 + 1.28653 w_k10_KO_s002
                                 + 1.35975 w_k10_MAT_s002
                                 + 1.02321 w_k10_MSFT_s002
                                 + 1.00549 w_k10_PG_s002 >= 1
 z_k10_s0023_constraint:         var_10 + z_k10_s0023 + 1.20407 w_k10_ADM_s002
                                 + 0.88975 w_k10_AXP_s002
                                 + 1.16596 w_k10_DIS_s002
                                 + 1.01334 w_k10_FDX_s002
                                 + 0.842822 w_k10_IBM_s002
                                 + 1.00227 w_k10_INTC_s002
                                 + 0.927758 w_k10_KO_s002
                                 + 0.770631 w_k10_MAT_s002
                                 + 0.983236 w_k10_MSFT_s002
                                 + 0.990406 w_k10_PG_s002 >= 1
 z_k10_s0024_constraint:         var_10 + z_k10_s0024 + 1.4068 w_k10_ADM_s002
                                 + 1.21007 w_k10_AXP_s002
                                 + 1.78149 w_k10_DIS_s002
                                 + 1.3174 w_k10_FDX_s002
                                 + 1.26305 w_k10_IBM_s002
                                 + 1.61277 w_k10_INTC_s002
                                 + 1.28738 w_k10_KO_s002
                                 + 1.14094 w_k10_MAT_s002
                                 + 1.41345 w_k10_MSFT_s002
                                 + 1.20428 w_k10_PG_s002 >= 1
 z_k10_s0025_constraint:         var_10 + z_k10_s0025 + 0.915581 w_k10_ADM_s002
                                 + 0.995687 w_k10_AXP_s002
                                 + 0.994033 w_k10_DIS_s002
                                 + 0.974395 w_k10_FDX_s002
                                 + 1.01057 w_k10_IBM_s002
                                 + 1.03311 w_k10_INTC_s002
                                 + 0.930266 w_k10_KO_s002
                                 + 1.1212 w_k10_MAT_s002
                                 + 0.955495 w_k10_MSFT_s002
                                 + 0.964408 w_k10_PG_s002 >= 1
 z_k10_s0026_constraint:         var_10 + z_k10_s0026 + 1.29741 w_k10_ADM_s002
                                 + 1.27022 w_k10_AXP_s002
                                 + 1.09357 w_k10_DIS_s002
                                 + 1.26398 w_k10_FDX_s002
                                 + 0.859181 w_k10_IBM_s002
                                 + 1.32522 w_k10_INTC_s002
                                 + 0.942899 w_k10_KO_s002
                                 + 0.80491 w_k10_MAT_s002
                                 + 0.963487 w_k10_MSFT_s002
                                 + 1.04114 w_k10_PG_s002 >= 1
 z_k10_s0027_constraint:         var_10 + z_k10_s0027 + 0.841405 w_k10_ADM_s002
                                 + 0.775821 w_k10_AXP_s002
                                 + 0.688806 w_k10_DIS_s002
                                 + 0.768428 w_k10_FDX_s002
                                 + 0.916728 w_k10_IBM_s002
                                 + 0.98338 w_k10_INTC_s002
                                 + 0.856409 w_k10_KO_s002
                                 + 0.548413 w_k10_MAT_s002
                                 + 0.777212 w_k10_MSFT_s002
                                 + 0.809079 w_k10_PG_s002 >= 1
 z_k10_s0030_constraint:         var_10 + z_k10_s0030 + 1.42094 w_k10_ADM_s003
                                 + 1.41424 w_k10_AXP_s003
                                 + 1.30293 w_k10_DIS_s003
                                 + 1.26509 w_k10_FDX_s003
                                 + 0.954791 w_k10_IBM_s003
                                 + 1.24162 w_k10_INTC_s003
                                 + 1.04613 w_k10_KO_s003
                                 + 1.02144 w_k10_MAT_s003
                                 + 1.18472 w_k10_MSFT_s003
                                 + 1.15038 w_k10_PG_s003 >= 1
 z_k10_s0031_constraint:         var_10 + z_k10_s0031 + 1.12719 w_k10_ADM_s003
                                 + 1.355 w_k10_AXP_s003
                                 + 1.73456 w_k10_DIS_s003
                                 + 1.57526 w_k10_FDX_s003
                                 + 1.03869 w_k10_IBM_s003
                                 + 1.34293 w_k10_INTC_s003
                                 + 1.35098 w_k10_KO_s003
                                 + 1.20981 w_k10_MAT_s003
                                 + 1.51089 w_k10_MSFT_s003
                                 + 1.12864 w_k10_PG_s003 >= 1
 z_k10_s0032_constraint:         var_10 + z_k10_s0032 + 0.949463 w_k10_ADM_s003
                                 + 0.939517 w_k10_AXP_s003
                                 + 1.23329 w_k10_DIS_s003
                                 + 1.12413 w_k10_FDX_s003
                                 + 0.90905 w_k10_IBM_s003
                                 + 1.05042 w_k10_INTC_s003
                                 + 0.927266 w_k10_KO_s003
                                 + 0.986141 w_k10_MAT_s003
                                 + 1.19658 w_k10_MSFT_s003
                                 + 0.955078 w_k10_PG_s003 >= 1
 z_k10_s0033_constraint:         var_10 + z_k10_s0033 + 0.960871 w_k10_ADM_s003
                                 + 1.01791 w_k10_AXP_s003
                                 + 0.91849 w_k10_DIS_s003
                                 + 0.849501 w_k10_FDX_s003
                                 + 1.56182 w_k10_IBM_s003
                                 + 0.993026 w_k10_INTC_s003
                                 + 0.994661 w_k10_KO_s003
                                 + 1.55408 w_k10_MAT_s003
                                 + 0.907222 w_k10_MSFT_s003
                                 + 1.03506 w_k10_PG_s003 >= 1
 z_k10_s0034_constraint:         var_10 + z_k10_s0034 + 0.970865 w_k10_ADM_s003
                                 + 1.0494 w_k10_AXP_s003
                                 + 0.947324 w_k10_DIS_s003
                                 + 1.15255 w_k10_FDX_s003
                                 + 0.600953 w_k10_IBM_s003
                                 + 1.07319 w_k10_INTC_s003
                                 + 0.815559 w_k10_KO_s003
                                 + 0.632189 w_k10_MAT_s003
                                 + 1.07413 w_k10_MSFT_s003
                                 + 0.935723 w_k10_PG_s003 >= 1
 z_k10_s0035_constraint:         var_10 + z_k10_s0035 + 0.677154 w_k10_ADM_s003
                                 + 0.699057 w_k10_AXP_s003
                                 + 0.604123 w_k10_DIS_s003
                                 + 0.747168 w_k10_FDX_s003
                                 + 0.79808 w_k10_IBM_s003
                                 + 0.934099 w_k10_INTC_s003
                                 + 0.712322 w_k10_KO_s003
                                 + 0.832133 w_k10_MAT_s003
                                 + 0.821191 w_k10_MSFT_s003
                                 + 0.807136 w_k10_PG_s003 >= 1
 z_k10_s0036_constraint:         var_10 + z_k10_s0036 + 0.84709 w_k10_ADM_s003
                                 + 0.914341 w_k10_AXP_s003
                                 + 0.956905 w_k10_DIS_s003
                                 + 0.93191 w_k10_FDX_s003
                                 + 1.03718 w_k10_IBM_s003
                                 + 1.12628 w_k10_INTC_s003
                                 + 1.00101 w_k10_KO_s003
                                 + 0.884824 w_k10_MAT_s003
                                 + 1.06242 w_k10_MSFT_s003
                                 + 1.06434 w_k10_PG_s003 >= 1
 z_k10_s0037_constraint:         var_10 + z_k10_s0037 + 0.837645 w_k10_ADM_s003
                                 + 1.27903 w_k10_AXP_s003
                                 + 1.39342 w_k10_DIS_s003
                                 + 1.19813 w_k10_FDX_s003
                                 + 1.42972 w_k10_IBM_s003
                                 + 1.1813 w_k10_INTC_s003
                                 + 1.26376 w_k10_KO_s003
                                 + 1.4296 w_k10_MAT_s003
                                 + 1.34582 w_k10_MSFT_s003
                                 + 1.05905 w_k10_PG_s003 >= 1
 z_k10_s0040_constraint:         var_10 + z_k10_s0040 + 0.843127 w_k10_ADM_s004
                                 + 1.2018 w_k10_AXP_s004 + 1.034 w_k10_DIS_s004
                                 + 1.1277 w_k10_FDX_s004
                                 + 0.962016 w_k10_IBM_s004
                                 + 1.25584 w_k10_INTC_s004
                                 + 0.978737 w_k10_KO_s004
                                 + 1.26824 w_k10_MAT_s004
                                 + 1.00915 w_k10_MSFT_s004
                                 + 1.15832 w_k10_PG_s004 >= 1
 z_k10_s0041_constraint:         var_10 + z_k10_s0041 + 0.910303 w_k10_ADM_s004
                                 + 0.792868 w_k10_AXP_s004
                                 + 0.698749 w_k10_DIS_s004
                                 + 0.754501 w_k10_FDX_s004
                                 + 0.991888 w_k10_IBM_s004
                                 + 0.875953 w_k10_INTC_s004
                                 + 0.735057 w_k10_KO_s004
                                 + 0.593625 w_k10_MAT_s004
                                 + 0.69606 w_k10_MSFT_s004
                                 + 0.85661 w_k10_PG_s004 >= 1
 z_k10_s0042_constraint:         var_10 + z_k10_s0042 + 0.795174 w_k10_ADM_s004
                                 + 1.08779 w_k10_AXP_s004
                                 + 0.834822 w_k10_DIS_s004
                                 + 0.916049 w_k10_FDX_s004
                                 + 1.11427 w_k10_IBM_s004
                                 + 1.30688 w_k10_INTC_s004
                                 + 0.904576 w_k10_KO_s004
                                 + 1.23453 w_k10_MAT_s004
                                 + 0.885259 w_k10_MSFT_s004
                                 + 0.988867 w_k10_PG_s004 >= 1
 z_k10_s0043_constraint:         var_10 + z_k10_s0043 + 0.651264 w_k10_ADM_s004
                                 + 0.873696 w_k10_AXP_s004
                                 + 0.90416 w_k10_DIS_s004
                                 + 0.935355 w_k10_FDX_s004
                                 + 0.708583 w_k10_IBM_s004
                                 + 1.14686 w_k10_INTC_s004
                                 + 0.796795 w_k10_KO_s004
                                 + 0.517864 w_k10_MAT_s004
                                 + 0.880855 w_k10_MSFT_s004
                                 + 0.913181 w_k10_PG_s004 >= 1
 z_k10_s0044_constraint:         var_10 + z_k10_s0044 + 1.1159 w_k10_ADM_s004
                                 + 1.28375 w_k10_AXP_s004
                                 + 1.47211 w_k10_DIS_s004
                                 + 1.35953 w_k10_FDX_s004
                                 + 1.06118 w_k10_IBM_s004
                                 + 1.27651 w_k10_INTC_s004
                                 + 1.1924 w_k10_KO_s004
                                 + 0.848362 w_k10_MAT_s004
                                 + 1.07032 w_k10_MSFT_s004
                                 + 1.11458 w_k10_PG_s004 >= 1
 z_k10_s0045_constraint:         var_10 + z_k10_s0045 + 1.07078 w_k10_ADM_s004
                                 + 1.09609 w_k10_AXP_s004
                                 + 1.40052 w_k10_DIS_s004
                                 + 1.1012 w_k10_FDX_s004
                                 + 1.19039 w_k10_IBM_s004
                                 + 1.47982 w_k10_INTC_s004
                                 + 1.1004 w_k10_KO_s004
                                 + 0.882012 w_k10_MAT_s004
                                 + 1.05106 w_k10_MSFT_s004
                                 + 1.07836 w_k10_PG_s004 >= 1
 z_k10_s0046_constraint:         var_10 + z_k10_s0046 + 1.34085 w_k10_ADM_s004
                                 + 1.6717 w_k10_AXP_s004
                                 + 1.76666 w_k10_DIS_s004
                                 + 1.49591 w_k10_FDX_s004
                                 + 1.39508 w_k10_IBM_s004
                                 + 1.87648 w_k10_INTC_s004
                                 + 1.46575 w_k10_KO_s004
                                 + 1.28364 w_k10_MAT_s004
                                 + 1.43678 w_k10_MSFT_s004
                                 + 1.16861 w_k10_PG_s004 >= 1
 z_k10_s0047_constraint:         var_10 + z_k10_s0047 + 0.960674 w_k10_ADM_s004
                                 + 0.795153 w_k10_AXP_s004
                                 + 1.2549 w_k10_DIS_s004
                                 + 1.18454 w_k10_FDX_s004
                                 + 0.684384 w_k10_IBM_s004
                                 + 1.3553 w_k10_INTC_s004
                                 + 0.951484 w_k10_KO_s004
                                 + 0.460874 w_k10_MAT_s004
                                 + 1.25329 w_k10_MSFT_s004
                                 + 0.979915 w_k10_PG_s004 >= 1
 z_k10_s0050_constraint:         var_10 + z_k10_s0050 + 0.725293 w_k10_ADM_s005
                                 + 0.692492 w_k10_AXP_s005
                                 + 0.677932 w_k10_DIS_s005
                                 + 0.805474 w_k10_FDX_s005
                                 + 0.614809 w_k10_IBM_s005
                                 + 0.786801 w_k10_INTC_s005
                                 + 0.658776 w_k10_KO_s005
                                 + 0.454811 w_k10_MAT_s005
                                 + 0.848407 w_k10_MSFT_s005
                                 + 0.797685 w_k10_PG_s005 >= 1
 z_k10_s0051_constraint:         var_10 + z_k10_s0051 + 0.909901 w_k10_ADM_s005
                                 + 0.943859 w_k10_AXP_s005
                                 + 0.870365 w_k10_DIS_s005
                                 + 0.907009 w_k10_FDX_s005
                                 + 0.945244 w_k10_IBM_s005
                                 + 0.888051 w_k10_INTC_s005
                                 + 0.919129 w_k10_KO_s005
                                 + 1.10877 w_k10_MAT_s005
                                 + 0.976778 w_k10_MSFT_s005
                                 + 1.03204 w_k10_PG_s005 >= 1
 z_k10_s0052_constraint:         var_10 + z_k10_s0052 + 1.3487 w_k10_ADM_s005
                                 + 1.32198 w_k10_AXP_s005
                                 + 1.24875 w_k10_DIS_s005
                                 + 1.28314 w_k10_FDX_s005
                                 + 0.961914 w_k10_IBM_s005
                                 + 1.14556 w_k10_INTC_s005
                                 + 1.1034 w_k10_KO_s005
                                 + 1.07168 w_k10_MAT_s005
                                 + 1.29621 w_k10_MSFT_s005
                                 + 1.2091 w_k10_PG_s005 >= 1
 z_k10_s0053_constraint:         var_10 + z_k10_s0053 + 0.988644 w_k10_ADM_s005
                                 + 0.817133 w_k10_AXP_s005
                                 + 1.23669 w_k10_DIS_s005
                                 + 1.07588 w_k10_FDX_s005
                                 + 0.81417 w_k10_IBM_s005
                                 + 1.1198 w_k10_INTC_s005
                                 + 0.972163 w_k10_KO_s005
                                 + 0.717144 w_k10_MAT_s005
                                 + 1.28672 w_k10_MSFT_s005
                                 + 1.11081 w_k10_PG_s005 >= 1
 z_k10_s0054_constraint:         var_10 + z_k10_s0054 + 0.613597 w_k10_ADM_s005
                                 + 0.696779 w_k10_AXP_s005
                                 + 0.754782 w_k10_DIS_s005
                                 + 0.785791 w_k10_FDX_s005
                                 + 0.799757 w_k10_IBM_s005
                                 + 0.902588 w_k10_INTC_s005
                                 + 0.936408 w_k10_KO_s005
                                 + 1.0263 w_k10_MAT_s005
                                 + 0.922571 w_k10_MSFT_s005
                                 + 0.952037 w_k10_PG_s005 >= 1
 z_k10_s0055_constraint:         var_10 + z_k10_s0055 + 1.32302 w_k10_ADM_s005
                                 + 1.74231 w_k10_AXP_s005
                                 + 1.69837 w_k10_DIS_s005
                                 + 1.4599 w_k10_FDX_s005
                                 + 1.32269 w_k10_IBM_s005
                                 + 1.19933 w_k10_INTC_s005
                                 + 1.39608 w_k10_KO_s005
                                 + 1.64398 w_k10_MAT_s005
                                 + 1.3895 w_k10_MSFT_s005
                                 + 1.18609 w_k10_PG_s005 >= 1
 z_k10_s0056_constraint:         var_10 + z_k10_s0056 + 0.856267 w_k10_ADM_s005
                                 + 0.790897 w_k10_AXP_s005
                                 + 0.956654 w_k10_DIS_s005
                                 + 0.794029 w_k10_FDX_s005
                                 + 1.14098 w_k10_IBM_s005
                                 + 0.800541 w_k10_INTC_s005
                                 + 0.975335 w_k10_KO_s005
                                 + 1.33473 w_k10_MAT_s005
                                 + 0.877169 w_k10_MSFT_s005
                                 + 0.996038 w_k10_PG_s005 >= 1
 z_k10_s0057_constraint:         var_10 + z_k10_s0057 + 0.831497 w_k10_ADM_s005
                                 + 1.0842 w_k10_AXP_s005
                                 + 1.11699 w_k10_DIS_s005
                                 + 1.03978 w_k10_FDX_s005
                                 + 1.11009 w_k10_IBM_s005
                                 + 0.975593 w_k10_INTC_s005
                                 + 1.05332 w_k10_KO_s005
                                 + 1.19414 w_k10_MAT_s005
                                 + 0.979888 w_k10_MSFT_s005
                                 + 1.05445 w_k10_PG_s005 >= 1
 z_k10_s0060_constraint:         var_10 + z_k10_s0060 + 1.10816 w_k10_ADM_s006
                                 + 1.57626 w_k10_AXP_s006
                                 + 1.15989 w_k10_DIS_s006
                                 + 1.23045 w_k10_FDX_s006
                                 + 1.24622 w_k10_IBM_s006
                                 + 0.901569 w_k10_INTC_s006
                                 + 1.08229 w_k10_KO_s006
                                 + 1.17864 w_k10_MAT_s006
                                 + 1.03486 w_k10_MSFT_s006
                                 + 1.23426 w_k10_PG_s006 >= 1
 z_k10_s0061_constraint:         var_10 + z_k10_s0061 + 1.19142 w_k10_ADM_s006
                                 + 1.35599 w_k10_AXP_s006
                                 + 1.36331 w_k10_DIS_s006
                                 + 1.13334 w_k10_FDX_s006
                                 + 1.73234 w_k10_IBM_s006
                                 + 1.23658 w_k10_INTC_s006
                                 + 1.30465 w_k10_KO_s006
                                 + 1.45749 w_k10_MAT_s006
                                 + 1.12497 w_k10_MSFT_s006
                                 + 1.20759 w_k10_PG_s006 >= 1
 z_k10_s0062_constraint:         var_10 + z_k10_s0062 + 1.07903 w_k10_ADM_s006
                                 + 1.06507 w_k10_AXP_s006
                                 + 0.922578 w_k10_DIS_s006
                                 + 1.04011 w_k10_FDX_s006
                                 + 0.95238 w_k10_IBM_s006
                                 + 0.919278 w_k10_INTC_s006
                                 + 0.831177 w_k10_KO_s006
                                 + 1.06853 w_k10_MAT_s006
                                 + 0.887636 w_k10_MSFT_s006
                                 + 0.960079 w_k10_PG_s006 >= 1
 z_k10_s0063_constraint:         var_10 + z_k10_s0063 + 1.01405 w_k10_ADM_s006
                                 + 0.780068 w_k10_AXP_s006
                                 + 0.792735 w_k10_DIS_s006
                                 + 0.92267 w_k10_FDX_s006
                                 + 0.716243 w_k10_IBM_s006
                                 + 0.915415 w_k10_INTC_s006
                                 + 0.682018 w_k10_KO_s006
                                 + 0.557481 w_k10_MAT_s006
                                 + 0.925912 w_k10_MSFT_s006
                                 + 0.896771 w_k10_PG_s006 >= 1
 z_k10_s0064_constraint:         var_10 + z_k10_s0064 + 1.18135 w_k10_ADM_s006
                                 + 1.27574 w_k10_AXP_s006
                                 + 1.22461 w_k10_DIS_s006
                                 + 1.31942 w_k10_FDX_s006
                                 + 0.878297 w_k10_IBM_s006
                                 + 1.06754 w_k10_INTC_s006
                                 + 0.983783 w_k10_KO_s006
                                 + 1.10902 w_k10_MAT_s006
                                 + 1.10491 w_k10_MSFT_s006
                                 + 1.26616 w_k10_PG_s006 >= 1
 z_k10_s0065_constraint:         var_10 + z_k10_s0065 + 1.11394 w_k10_ADM_s006
                                 + 1.14325 w_k10_AXP_s006
                                 + 1.38543 w_k10_DIS_s006
                                 + 1.18881 w_k10_FDX_s006
                                 + 1.15187 w_k10_IBM_s006
                                 + 1.24686 w_k10_INTC_s006
                                 + 1.24689 w_k10_KO_s006
                                 + 0.980888 w_k10_MAT_s006
                                 + 1.206 w_k10_MSFT_s006
                                 + 1.25882 w_k10_PG_s006 >= 1
 z_k10_s0066_constraint:         var_10 + z_k10_s0066 + 0.917891 w_k10_ADM_s006
                                 + 0.778012 w_k10_AXP_s006
                                 + 0.935937 w_k10_DIS_s006
                                 + 0.892106 w_k10_FDX_s006
                                 + 1.05832 w_k10_IBM_s006
                                 + 1.0879 w_k10_INTC_s006
                                 + 1.06518 w_k10_KO_s006
                                 + 0.839413 w_k10_MAT_s006
                                 + 0.96156 w_k10_MSFT_s006
                                 + 0.950564 w_k10_PG_s006 >= 1
 z_k10_s0067_constraint:         var_10 + z_k10_s0067 + 0.621548 w_k10_ADM_s006
                                 + 0.678124 w_k10_AXP_s006
                                 + 0.705625 w_k10_DIS_s006
                                 + 0.727989 w_k10_FDX_s006
                                 + 1.06753 w_k10_IBM_s006
                                 + 0.838976 w_k10_INTC_s006
                                 + 0.79213 w_k10_KO_s006
                                 + 0.889289 w_k10_MAT_s006
                                 + 0.869025 w_k10_MSFT_s006
                                 + 0.880639 w_k10_PG_s006 >= 1
 z_k10_s0070_constraint:         var_10 + z_k10_s0070 + 1.39802 w_k10_ADM_s007
                                 + 1.1731 w_k10_AXP_s007
                                 + 1.32957 w_k10_DIS_s007
                                 + 1.24588 w_k10_FDX_s007
                                 + 0.780483 w_k10_IBM_s007
                                 + 0.976949 w_k10_INTC_s007
                                 + 0.985247 w_k10_KO_s007
                                 + 1.36096 w_k10_MAT_s007
                                 + 1.15543 w_k10_MSFT_s007
                                 + 1.12295 w_k10_PG_s007 >= 1
 z_k10_s0071_constraint:         var_10 + z_k10_s0071 + 0.791533 w_k10_ADM_s007
                                 + 0.679203 w_k10_AXP_s007
                                 + 0.685183 w_k10_DIS_s007
                                 + 0.733263 w_k10_FDX_s007
                                 + 0.857989 w_k10_IBM_s007
                                 + 0.729962 w_k10_INTC_s007
                                 + 0.765201 w_k10_KO_s007
                                 + 1.25941 w_k10_MAT_s007
                                 + 0.692323 w_k10_MSFT_s007
                                 + 0.864602 w_k10_PG_s007 >= 1
 z_k10_s0072_constraint:         var_10 + z_k10_s0072 + 1.12567 w_k10_ADM_s007
                                 + 1.32281 w_k10_AXP_s007
                                 + 1.36554 w_k10_DIS_s007
                                 + 1.13831 w_k10_FDX_s007
                                 + 1.32507 w_k10_IBM_s007
                                 + 1.14945 w_k10_INTC_s007
                                 + 1.29613 w_k10_KO_s007
                                 + 2.02128 w_k10_MAT_s007
                                 + 1.17178 w_k10_MSFT_s007
                                 + 1.28489 w_k10_PG_s007 >= 1
 z_k10_s0073_constraint:         var_10 + z_k10_s0073 + 0.775549 w_k10_ADM_s007
                                 + 0.801439 w_k10_AXP_s007
                                 + 0.779319 w_k10_DIS_s007
                                 + 0.87959 w_k10_FDX_s007
                                 + 0.658392 w_k10_IBM_s007
                                 + 0.895254 w_k10_INTC_s007
                                 + 0.702113 w_k10_KO_s007
                                 + 1.17912 w_k10_MAT_s007
                                 + 1.00132 w_k10_MSFT_s007
                                 + 0.863592 w_k10_PG_s007 >= 1
 z_k10_s0074_constraint:         var_10 + z_k10_s0074 + 1.04985 w_k10_ADM_s007
                                 + 1.03539 w_k10_AXP_s007
                                 + 1.0056 w_k10_DIS_s007
                                 + 1.02009 w_k10_FDX_s007
                                 + 0.832144 w_k10_IBM_s007
                                 + 0.961052 w_k10_INTC_s007
                                 + 0.910834 w_k10_KO_s007
                                 + 1.18974 w_k10_MAT_s007
                                 + 1.02854 w_k10_MSFT_s007
                                 + 1.04106 w_k10_PG_s007 >= 1
 z_k10_s0075_constraint:         var_10 + z_k10_s0075 + 0.906584 w_k10_ADM_s007
                                 + 1.05323 w_k10_AXP_s007
                                 + 1.04681 w_k10_DIS_s007
                                 + 0.893532 w_k10_FDX_s007
                                 + 1.23825 w_k10_IBM_s007
                                 + 1.00251 w_k10_INTC_s007
                                 + 0.982757 w_k10_KO_s007
                                 + 1.89391 w_k10_MAT_s007
                                 + 0.945217 w_k10_MSFT_s007
                                 + 1.02472 w_k10_PG_s007 >= 1
 z_k10_s0076_constraint:         var_10 + z_k10_s0076 + 1.21226 w_k10_ADM_s007
                                 + 1.54578 w_k10_AXP_s007
                                 + 1.81555 w_k10_DIS_s007
                                 + 1.44769 w_k10_FDX_s007
                                 + 1.18185 w_k10_IBM_s007
                                 + 1.51021 w_k10_INTC_s007
                                 + 1.35451 w_k10_KO_s007
                                 + 2.0831 w_k10_MAT_s007
                                 + 1.61135 w_k10_MSFT_s007
                                 + 1.09597 w_k10_PG_s007 >= 1
 z_k10_s0077_constraint:         var_10 + z_k10_s0077 + 0.893223 w_k10_ADM_s007
                                 + 0.987804 w_k10_AXP_s007
                                 + 0.979779 w_k10_DIS_s007
                                 + 0.998246 w_k10_FDX_s007
                                 + 0.97288 w_k10_IBM_s007
                                 + 0.86364 w_k10_INTC_s007
                                 + 0.921991 w_k10_KO_s007
                                 + 1.40287 w_k10_MAT_s007
                                 + 0.923286 w_k10_MSFT_s007
                                 + 1.00656 w_k10_PG_s007 >= 1
 z_k10_s0100_constraint:         var_10 + z_k10_s0100 + 1.21291 w_k10_ADM_s010
                                 + 1.49156 w_k10_AXP_s010
                                 + 1.90149 w_k10_DIS_s010
                                 + 1.47003 w_k10_FDX_s010
                                 + 1.22464 w_k10_IBM_s010
                                 + 1.41152 w_k10_INTC_s010
                                 + 1.53668 w_k10_KO_s010
                                 + 1.9905 w_k10_MAT_s010
                                 + 1.50686 w_k10_MSFT_s010
                                 + 1.28365 w_k10_PG_s010 >= 1
 z_k10_s0101_constraint:         var_10 + z_k10_s0101 + 1.07194 w_k10_ADM_s010
                                 + 1.12031 w_k10_AXP_s010
                                 + 0.977508 w_k10_DIS_s010
                                 + 1.0115 w_k10_FDX_s010
                                 + 0.791041 w_k10_IBM_s010
                                 + 0.770542 w_k10_INTC_s010
                                 + 0.882788 w_k10_KO_s010
                                 + 0.946856 w_k10_MAT_s010
                                 + 0.926259 w_k10_MSFT_s010
                                 + 1.00074 w_k10_PG_s010 >= 1
 z_k10_s0102_constraint:         var_10 + z_k10_s0102 + 0.730178 w_k10_ADM_s010
                                 + 0.877926 w_k10_AXP_s010
                                 + 0.653188 w_k10_DIS_s010
                                 + 0.800839 w_k10_FDX_s010
                                 + 0.679184 w_k10_IBM_s010
                                 + 0.755817 w_k10_INTC_s010
                                 + 0.65877 w_k10_KO_s010
                                 + 1.04914 w_k10_MAT_s010
                                 + 0.776877 w_k10_MSFT_s010
                                 + 0.795224 w_k10_PG_s010 >= 1
 z_k10_s0103_constraint:         var_10 + z_k10_s0103 + 0.820882 w_k10_ADM_s010
                                 + 0.863149 w_k10_AXP_s010
                                 + 0.871441 w_k10_DIS_s010
                                 + 0.859936 w_k10_FDX_s010
                                 + 1.12123 w_k10_IBM_s010
                                 + 0.819688 w_k10_INTC_s010
                                 + 0.916047 w_k10_KO_s010
                                 + 1.17849 w_k10_MAT_s010
                                 + 0.857423 w_k10_MSFT_s010
                                 + 0.983778 w_k10_PG_s010 >= 1
 z_k10_s0104_constraint:         var_10 + z_k10_s0104 + 1.15757 w_k10_ADM_s010
                                 + 1.38633 w_k10_AXP_s010
                                 + 1.31576 w_k10_DIS_s010
                                 + 1.13744 w_k10_FDX_s010
                                 + 1.24175 w_k10_IBM_s010
                                 + 1.1025 w_k10_INTC_s010
                                 + 1.20827 w_k10_KO_s010
                                 + 1.89939 w_k10_MAT_s010
                                 + 1.18287 w_k10_MSFT_s010
                                 + 1.18301 w_k10_PG_s010 >= 1
 z_k10_s0105_constraint:         var_10 + z_k10_s0105 + 1.45952 w_k10_ADM_s010
                                 + 1.50595 w_k10_AXP_s010
                                 + 1.20472 w_k10_DIS_s010
                                 + 1.2638 w_k10_FDX_s010
                                 + 1.00449 w_k10_IBM_s010
                                 + 0.943042 w_k10_INTC_s010
                                 + 1.10434 w_k10_KO_s010
                                 + 1.69039 w_k10_MAT_s010
                                 + 1.21298 w_k10_MSFT_s010
                                 + 1.10624 w_k10_PG_s010 >= 1
 z_k10_s0106_constraint:         var_10 + z_k10_s0106 + 1.05216 w_k10_ADM_s010
                                 + 1.42463 w_k10_AXP_s010
                                 + 0.942286 w_k10_DIS_s010
                                 + 0.991764 w_k10_FDX_s010
                                 + 1.27862 w_k10_IBM_s010
                                 + 1.04325 w_k10_INTC_s010
                                 + 1.22023 w_k10_KO_s010
                                 + 1.93669 w_k10_MAT_s010
                                 + 0.902501 w_k10_MSFT_s010
                                 + 1.01178 w_k10_PG_s010 >= 1
 z_k10_s0107_constraint:         var_10 + z_k10_s0107 + 0.708721 w_k10_ADM_s010
                                 + 0.861227 w_k10_AXP_s010
                                 + 0.562037 w_k10_DIS_s010
                                 + 0.667551 w_k10_FDX_s010
                                 + 1.01374 w_k10_IBM_s010
                                 + 0.755654 w_k10_INTC_s010
                                 + 0.73933 w_k10_KO_s010
                                 + 0.984075 w_k10_MAT_s010
                                 + 0.684855 w_k10_MSFT_s010
                                 + 0.846831 w_k10_PG_s010 >= 1
 z_k10_s0110_constraint:         var_10 + z_k10_s0110 + 0.899766 w_k10_ADM_s011
                                 + 1.18141 w_k10_AXP_s011
                                 + 0.824211 w_k10_DIS_s011
                                 + 0.880302 w_k10_FDX_s011
                                 + 1.22429 w_k10_IBM_s011
                                 + 0.938096 w_k10_INTC_s011
                                 + 1.02224 w_k10_KO_s011
                                 + 1.43203 w_k10_MAT_s011
                                 + 1.01128 w_k10_MSFT_s011
                                 + 1.07257 w_k10_PG_s011 >= 1
 z_k10_s0111_constraint:         var_10 + z_k10_s0111 + 1.35336 w_k10_ADM_s011
                                 + 1.46672 w_k10_AXP_s011
                                 + 1.40389 w_k10_DIS_s011
                                 + 1.20471 w_k10_FDX_s011
                                 + 1.51491 w_k10_IBM_s011
                                 + 1.21777 w_k10_INTC_s011
                                 + 1.25191 w_k10_KO_s011
                                 + 1.3618 w_k10_MAT_s011
                                 + 1.20985 w_k10_MSFT_s011
                                 + 1.1566 w_k10_PG_s011 >= 1
 z_k10_s0112_constraint:         var_10 + z_k10_s0112 + 0.76294 w_k10_ADM_s011
                                 + 0.797817 w_k10_AXP_s011
                                 + 0.671235 w_k10_DIS_s011
                                 + 0.824497 w_k10_FDX_s011
                                 + 0.749138 w_k10_IBM_s011
                                 + 0.79643 w_k10_INTC_s011
                                 + 0.79421 w_k10_KO_s011
                                 + 0.696323 w_k10_MAT_s011
                                 + 0.910372 w_k10_MSFT_s011
                                 + 0.960564 w_k10_PG_s011 >= 1
 z_k10_s0113_constraint:         var_10 + z_k10_s0113 + 1.36019 w_k10_ADM_s011
                                 + 1.29076 w_k10_AXP_s011
                                 + 1.70024 w_k10_DIS_s011
                                 + 1.46077 w_k10_FDX_s011
                                 + 0.978103 w_k10_IBM_s011
                                 + 1.37468 w_k10_INTC_s011
                                 + 1.19671 w_k10_KO_s011
                                 + 0.986779 w_k10_MAT_s011
                                 + 1.43067 w_k10_MSFT_s011
                                 + 1.22672 w_k10_PG_s011 >= 1
 z_k10_s0114_constraint:         var_10 + z_k10_s0114 + 0.878797 w_k10_ADM_s011
                                 + 1.05833 w_k10_AXP_s011
                                 + 0.9351 w_k10_DIS_s011
                                 + 0.880574 w_k10_FDX_s011
                                 + 0.962286 w_k10_IBM_s011
                                 + 0.845115 w_k10_INTC_s011
                                 + 0.993413 w_k10_KO_s011
                                 + 1.23187 w_k10_MAT_s011
                                 + 0.919407 w_k10_MSFT_s011
                                 + 1.03321 w_k10_PG_s011 >= 1
 z_k10_s0115_constraint:         var_10 + z_k10_s0115 + 1.17332 w_k10_ADM_s011
                                 + 1.19861 w_k10_AXP_s011
                                 + 1.30961 w_k10_DIS_s011
                                 + 1.09839 w_k10_FDX_s011
                                 + 1.09154 w_k10_IBM_s011
                                 + 1.09636 w_k10_INTC_s011
                                 + 1.27717 w_k10_KO_s011
                                 + 1.17172 w_k10_MAT_s011
                                 + 1.10792 w_k10_MSFT_s011
                                 + 1.23095 w_k10_PG_s011 >= 1
 z_k10_s0116_constraint:         var_10 + z_k10_s0116 + 1.09275 w_k10_ADM_s011
                                 + 1.25153 w_k10_AXP_s011
                                 + 0.97127 w_k10_DIS_s011
                                 + 1.12531 w_k10_FDX_s011
                                 + 0.740935 w_k10_IBM_s011
                                 + 0.951665 w_k10_INTC_s011
                                 + 0.956736 w_k10_KO_s011
                                 + 0.976342 w_k10_MAT_s011
                                 + 1.11165 w_k10_MSFT_s011
                                 + 1.10389 w_k10_PG_s011 >= 1
 z_k10_s0117_constraint:         var_10 + z_k10_s0117 + 1.08238 w_k10_ADM_s011
                                 + 1.08204 w_k10_AXP_s011
                                 + 0.918356 w_k10_DIS_s011
                                 + 1.04912 w_k10_FDX_s011
                                 + 0.856181 w_k10_IBM_s011
                                 + 1.03367 w_k10_INTC_s011
                                 + 0.964815 w_k10_KO_s011
                                 + 0.920612 w_k10_MAT_s011
                                 + 0.928202 w_k10_MSFT_s011
                                 + 0.96326 w_k10_PG_s011 >= 1
 z_k10_s0120_constraint:         var_10 + z_k10_s0120 + 1.34213 w_k10_ADM_s012
                                 + 1.43961 w_k10_AXP_s012
                                 + 0.954695 w_k10_DIS_s012
                                 + 1.2054 w_k10_FDX_s012
                                 + 0.830654 w_k10_IBM_s012
                                 + 1.2059 w_k10_INTC_s012
                                 + 1.03023 w_k10_KO_s012
                                 + 1.04941 w_k10_MAT_s012
                                 + 0.926786 w_k10_MSFT_s012
                                 + 1.10793 w_k10_PG_s012 >= 1
 z_k10_s0121_constraint:         var_10 + z_k10_s0121 + 0.909542 w_k10_ADM_s012
                                 + 1.00154 w_k10_AXP_s012
                                 + 0.704874 w_k10_DIS_s012
                                 + 0.838567 w_k10_FDX_s012
                                 + 1.00596 w_k10_IBM_s012
                                 + 0.894541 w_k10_INTC_s012
                                 + 0.81511 w_k10_KO_s012
                                 + 0.891229 w_k10_MAT_s012
                                 + 0.697014 w_k10_MSFT_s012
                                 + 0.897814 w_k10_PG_s012 >= 1
 z_k10_s0122_constraint:         var_10 + z_k10_s0122 + 1.29413 w_k10_ADM_s012
                                 + 1.4742 w_k10_AXP_s012
                                 + 1.56435 w_k10_DIS_s012
                                 + 1.25272 w_k10_FDX_s012
                                 + 1.41585 w_k10_IBM_s012
                                 + 1.44122 w_k10_INTC_s012
                                 + 1.58021 w_k10_KO_s012
                                 + 1.66757 w_k10_MAT_s012
                                 + 1.15601 w_k10_MSFT_s012
                                 + 1.2096 w_k10_PG_s012 >= 1
 z_k10_s0123_constraint:         var_10 + z_k10_s0123 + 0.945718 w_k10_ADM_s012
                                 + 0.963152 w_k10_AXP_s012
                                 + 0.709624 w_k10_DIS_s012
                                 + 0.870571 w_k10_FDX_s012
                                 + 0.865638 w_k10_IBM_s012
                                 + 1.07813 w_k10_INTC_s012
                                 + 0.821535 w_k10_KO_s012
                                 + 0.86629 w_k10_MAT_s012
                                 + 0.84209 w_k10_MSFT_s012
                                 + 0.855863 w_k10_PG_s012 >= 1
 z_k10_s0124_constraint:         var_10 + z_k10_s0124 + 0.86526 w_k10_ADM_s012
                                 + 1.14739 w_k10_AXP_s012
                                 + 1.14783 w_k10_DIS_s012
                                 + 1.01051 w_k10_FDX_s012
                                 + 1.37636 w_k10_IBM_s012
                                 + 1.15619 w_k10_INTC_s012
                                 + 1.1687 w_k10_KO_s012
                                 + 1.52373 w_k10_MAT_s012
                                 + 1.05559 w_k10_MSFT_s012
                                 + 1.09081 w_k10_PG_s012 >= 1
 z_k10_s0125_constraint:         var_10 + z_k10_s0125 + 0.817957 w_k10_ADM_s012
                                 + 0.915107 w_k10_AXP_s012
                                 + 0.943877 w_k10_DIS_s012
                                 + 0.782274 w_k10_FDX_s012
                                 + 1.24458 w_k10_IBM_s012
                                 + 1.00788 w_k10_INTC_s012
                                 + 1.14907 w_k10_KO_s012
                                 + 1.41497 w_k10_MAT_s012
                                 + 0.775058 w_k10_MSFT_s012
                                 + 1.10163 w_k10_PG_s012 >= 1
 z_k10_s0126_constraint:         var_10 + z_k10_s0126 + 0.829246 w_k10_ADM_s012
                                 + 0.773894 w_k10_AXP_s012
                                 + 0.705388 w_k10_DIS_s012
                                 + 0.892867 w_k10_FDX_s012
                                 + 0.618864 w_k10_IBM_s012
                                 + 0.978488 w_k10_INTC_s012
                                 + 0.903981 w_k10_KO_s012
                                 + 0.667836 w_k10_MAT_s012
                                 + 0.757925 w_k10_MSFT_s012
                                 + 0.939509 w_k10_PG_s012 >= 1
 z_k10_s0127_constraint:         var_10 + z_k10_s0127 + 1.32358 w_k10_ADM_s012
                                 + 1.05466 w_k10_AXP_s012
                                 + 1.20771 w_k10_DIS_s012
                                 + 1.04612 w_k10_FDX_s012
                                 + 0.903143 w_k10_IBM_s012
                                 + 1.2775 w_k10_INTC_s012
                                 + 1.04978 w_k10_KO_s012
                                 + 1.05686 w_k10_MAT_s012
                                 + 1.0012 w_k10_MSFT_s012
                                 + 1.06971 w_k10_PG_s012 >= 1
 z_k10_s0130_constraint:         var_10 + z_k10_s0130 + 0.99544 w_k10_ADM_s013
                                 + 0.905865 w_k10_AXP_s013
                                 + 0.903853 w_k10_DIS_s013
                                 + 0.943193 w_k10_FDX_s013
                                 + 0.758537 w_k10_IBM_s013
                                 + 0.830355 w_k10_INTC_s013
                                 + 0.824873 w_k10_KO_s013
                                 + 0.648003 w_k10_MAT_s013
                                 + 0.874453 w_k10_MSFT_s013
                                 + 0.850078 w_k10_PG_s013 >= 1
 z_k10_s0131_constraint:         var_10 + z_k10_s0131 + 1.4794 w_k10_ADM_s013
                                 + 1.39625 w_k10_AXP_s013
                                 + 1.99892 w_k10_DIS_s013
                                 + 1.51687 w_k10_FDX_s013
                                 + 0.927474 w_k10_IBM_s013
                                 + 1.47446 w_k10_INTC_s013
                                 + 1.32476 w_k10_KO_s013
                                 + 1.16171 w_k10_MAT_s013
                                 + 1.37825 w_k10_MSFT_s013
                                 + 1.2696 w_k10_PG_s013 >= 1
 z_k10_s0132_constraint:         var_10 + z_k10_s0132 + 1.25869 w_k10_ADM_s013
                                 + 1.19834 w_k10_AXP_s013
                                 + 1.32049 w_k10_DIS_s013
                                 + 1.1855 w_k10_FDX_s013
                                 + 0.818103 w_k10_IBM_s013
                                 + 1.33599 w_k10_INTC_s013
                                 + 0.997407 w_k10_KO_s013
                                 + 0.932525 w_k10_MAT_s013
                                 + 1.18261 w_k10_MSFT_s013
                                 + 1.06019 w_k10_PG_s013 >= 1
 z_k10_s0133_constraint:         var_10 + z_k10_s0133 + 0.931271 w_k10_ADM_s013
                                 + 1.04469 w_k10_AXP_s013
                                 + 1.29621 w_k10_DIS_s013
                                 + 1.03596 w_k10_FDX_s013
                                 + 1.18424 w_k10_IBM_s013
                                 + 1.10165 w_k10_INTC_s013
                                 + 1.01582 w_k10_KO_s013
                                 + 1.1549 w_k10_MAT_s013
                                 + 1.10409 w_k10_MSFT_s013
                                 + 1.0604 w_k10_PG_s013 >= 1
 z_k10_s0134_constraint:         var_10 + z_k10_s0134 + 1.32913 w_k10_ADM_s013
                                 + 1.25714 w_k10_AXP_s013
                                 + 1.52755 w_k10_DIS_s013
                                 + 1.15666 w_k10_FDX_s013
                                 + 1.57209 w_k10_IBM_s013
                                 + 1.08944 w_k10_INTC_s013
                                 + 1.22394 w_k10_KO_s013
                                 + 1.26132 w_k10_MAT_s013
                                 + 1.08945 w_k10_MSFT_s013
                                 + 1.1208 w_k10_PG_s013 >= 1
 z_k10_s0135_constraint:         var_10 + z_k10_s0135 + 0.986529 w_k10_ADM_s013
                                 + 0.879211 w_k10_AXP_s013
                                 + 0.936503 w_k10_DIS_s013
                                 + 0.841952 w_k10_FDX_s013
                                 + 1.27332 w_k10_IBM_s013
                                 + 0.906485 w_k10_INTC_s013
                                 + 1.01151 w_k10_KO_s013
                                 + 0.943006 w_k10_MAT_s013
                                 + 0.868464 w_k10_MSFT_s013
                                 + 0.817825 w_k10_PG_s013 >= 1
 z_k10_s0136_constraint:         var_10 + z_k10_s0136 + 0.793674 w_k10_ADM_s013
                                 + 0.633586 w_k10_AXP_s013
                                 + 0.649017 w_k10_DIS_s013
                                 + 0.682982 w_k10_FDX_s013
                                 + 0.974028 w_k10_IBM_s013
                                 + 0.776233 w_k10_INTC_s013
                                 + 0.81769 w_k10_KO_s013
                                 + 0.628399 w_k10_MAT_s013
                                 + 0.645837 w_k10_MSFT_s013
                                 + 0.802443 w_k10_PG_s013 >= 1
 z_k10_s0137_constraint:         var_10 + z_k10_s0137 + 1.06563 w_k10_ADM_s013
                                 + 1.19547 w_k10_AXP_s013
                                 + 1.49602 w_k10_DIS_s013
                                 + 1.16583 w_k10_FDX_s013
                                 + 1.09199 w_k10_IBM_s013
                                 + 1.19523 w_k10_INTC_s013
                                 + 1.08746 w_k10_KO_s013
                                 + 0.802574 w_k10_MAT_s013
                                 + 1.26262 w_k10_MSFT_s013
                                 + 1.04723 w_k10_PG_s013 >= 1
 z_k10_s0140_constraint:         var_10 + z_k10_s0140 + 0.747141 w_k10_ADM_s014
                                 + 0.857754 w_k10_AXP_s014
                                 + 0.858269 w_k10_DIS_s014
                                 + 0.817408 w_k10_FDX_s014
                                 + 1.21025 w_k10_IBM_s014
                                 + 0.817859 w_k10_INTC_s014
                                 + 1.0879 w_k10_KO_s014
                                 + 1.18311 w_k10_MAT_s014
                                 + 0.795606 w_k10_MSFT_s014
                                 + 0.991337 w_k10_PG_s014 >= 1
 z_k10_s0141_constraint:         var_10 + z_k10_s0141 + 1.01416 w_k10_ADM_s014
                                 + 1.0224 w_k10_AXP_s014
                                 + 0.915949 w_k10_DIS_s014
                                 + 0.983331 w_k10_FDX_s014
                                 + 0.863819 w_k10_IBM_s014
                                 + 0.991987 w_k10_INTC_s014
                                 + 0.924921 w_k10_KO_s014
                                 + 0.868111 w_k10_MAT_s014
                                 + 0.820374 w_k10_MSFT_s014
                                 + 1.04646 w_k10_PG_s014 >= 1
 z_k10_s0142_constraint:         var_10 + z_k10_s0142 + 0.978451 w_k10_ADM_s014
                                 + 1.29786 w_k10_AXP_s014
                                 + 1.25745 w_k10_DIS_s014
                                 + 1.05051 w_k10_FDX_s014
                                 + 1.54629 w_k10_IBM_s014
                                 + 1.11423 w_k10_INTC_s014
                                 + 1.39136 w_k10_KO_s014
                                 + 1.46525 w_k10_MAT_s014
                                 + 1.04914 w_k10_MSFT_s014
                                 + 1.28984 w_k10_PG_s014 >= 1
 z_k10_s0143_constraint:         var_10 + z_k10_s0143 + 0.840602 w_k10_ADM_s014
                                 + 0.715587 w_k10_AXP_s014
                                 + 0.936009 w_k10_DIS_s014
                                 + 0.957106 w_k10_FDX_s014
                                 + 0.970297 w_k10_IBM_s014
                                 + 0.87811 w_k10_INTC_s014
                                 + 1.07928 w_k10_KO_s014
                                 + 1.06984 w_k10_MAT_s014
                                 + 0.931601 w_k10_MSFT_s014
                                 + 0.965171 w_k10_PG_s014 >= 1
 z_k10_s0144_constraint:         var_10 + z_k10_s0144 + 0.889521 w_k10_ADM_s014
                                 + 0.889092 w_k10_AXP_s014
                                 + 1.21917 w_k10_DIS_s014
                                 + 1.22761 w_k10_FDX_s014
                                 + 0.779968 w_k10_IBM_s014
                                 + 1.17386 w_k10_INTC_s014
                                 + 1.05902 w_k10_KO_s014
                                 + 0.616992 w_k10_MAT_s014
                                 + 1.11065 w_k10_MSFT_s014
                                 + 1.08611 w_k10_PG_s014 >= 1
 z_k10_s0145_constraint:         var_10 + z_k10_s0145 + 1.06537 w_k10_ADM_s014
                                 + 1.00186 w_k10_AXP_s014
                                 + 1.54239 w_k10_DIS_s014
                                 + 1.29189 w_k10_FDX_s014
                                 + 1.07265 w_k10_IBM_s014
                                 + 1.19772 w_k10_INTC_s014
                                 + 1.27022 w_k10_KO_s014
                                 + 0.915583 w_k10_MAT_s014
                                 + 1.22793 w_k10_MSFT_s014
                                 + 1.20379 w_k10_PG_s014 >= 1
 z_k10_s0146_constraint:         var_10 + z_k10_s0146 + 0.792044 w_k10_ADM_s014
                                 + 0.757166 w_k10_AXP_s014
                                 + 0.809255 w_k10_DIS_s014
                                 + 0.976058 w_k10_FDX_s014
                                 + 0.65456 w_k10_IBM_s014
                                 + 0.903144 w_k10_INTC_s014
                                 + 0.786682 w_k10_KO_s014
                                 + 0.621274 w_k10_MAT_s014
                                 + 0.888564 w_k10_MSFT_s014
                                 + 0.914071 w_k10_PG_s014 >= 1
 z_k10_s0147_constraint:         var_10 + z_k10_s0147 + 0.699944 w_k10_ADM_s014
                                 + 0.809722 w_k10_AXP_s014
                                 + 0.695106 w_k10_DIS_s014
                                 + 0.809696 w_k10_FDX_s014
                                 + 0.869039 w_k10_IBM_s014
                                 + 0.807214 w_k10_INTC_s014
                                 + 0.710834 w_k10_KO_s014
                                 + 0.843542 w_k10_MAT_s014
                                 + 0.71313 w_k10_MSFT_s014
                                 + 0.814323 w_k10_PG_s014 >= 1
 z_k10_s0150_constraint:         var_10 + z_k10_s0150 + 0.940527 w_k10_ADM_s015
                                 + 0.616315 w_k10_AXP_s015
                                 + 0.595589 w_k10_DIS_s015
                                 + 0.788734 w_k10_FDX_s015
                                 + 0.767519 w_k10_IBM_s015
                                 + 0.844554 w_k10_INTC_s015
                                 + 0.722837 w_k10_KO_s015
                                 + 0.507367 w_k10_MAT_s015
                                 + 0.804934 w_k10_MSFT_s015
                                 + 0.793373 w_k10_PG_s015 >= 1
 z_k10_s0151_constraint:         var_10 + z_k10_s0151 + 1.48103 w_k10_ADM_s015
                                 + 1.48001 w_k10_AXP_s015
                                 + 1.33783 w_k10_DIS_s015
                                 + 1.38946 w_k10_FDX_s015
                                 + 1.08735 w_k10_IBM_s015
                                 + 1.26639 w_k10_INTC_s015
                                 + 1.21671 w_k10_KO_s015
                                 + 1.34461 w_k10_MAT_s015
                                 + 1.30318 w_k10_MSFT_s015
                                 + 1.07775 w_k10_PG_s015 >= 1
 z_k10_s0152_constraint:         var_10 + z_k10_s0152 + 1.03687 w_k10_ADM_s015
                                 + 1.19255 w_k10_AXP_s015
                                 + 1.09084 w_k10_DIS_s015
                                 + 1.15296 w_k10_FDX_s015
                                 + 1.28205 w_k10_IBM_s015
                                 + 1.23615 w_k10_INTC_s015
                                 + 1.11477 w_k10_KO_s015
                                 + 1.17246 w_k10_MAT_s015
                                 + 1.0556 w_k10_MSFT_s015
                                 + 1.04958 w_k10_PG_s015 >= 1
 z_k10_s0153_constraint:         var_10 + z_k10_s0153 + 0.78178 w_k10_ADM_s015
                                 + 1.01912 w_k10_AXP_s015
                                 + 0.64687 w_k10_DIS_s015
                                 + 0.90367 w_k10_FDX_s015
                                 + 1.01792 w_k10_IBM_s015
                                 + 0.981247 w_k10_INTC_s015
                                 + 0.824985 w_k10_KO_s015
                                 + 1.34117 w_k10_MAT_s015
                                 + 0.880839 w_k10_MSFT_s015
                                 + 1.05397 w_k10_PG_s015 >= 1
 z_k10_s0154_constraint:         var_10 + z_k10_s0154 + 0.706111 w_k10_ADM_s015
                                 + 0.745365 w_k10_AXP_s015
                                 + 0.833523 w_k10_DIS_s015
                                 + 0.906208 w_k10_FDX_s015
                                 + 1.30792 w_k10_IBM_s015
                                 + 0.949674 w_k10_INTC_s015
                                 + 1.01831 w_k10_KO_s015
                                 + 1.22277 w_k10_MAT_s015
                                 + 0.860602 w_k10_MSFT_s015
                                 + 0.929212 w_k10_PG_s015 >= 1
 z_k10_s0155_constraint:         var_10 + z_k10_s0155 + 0.782836 w_k10_ADM_s015
                                 + 0.640921 w_k10_AXP_s015
                                 + 0.886129 w_k10_DIS_s015
                                 + 0.974695 w_k10_FDX_s015
                                 + 0.673912 w_k10_IBM_s015
                                 + 0.94629 w_k10_INTC_s015
                                 + 0.9037 w_k10_KO_s015
                                 + 0.697936 w_k10_MAT_s015
                                 + 1.0802 w_k10_MSFT_s015
                                 + 0.923484 w_k10_PG_s015 >= 1
 z_k10_s0156_constraint:         var_10 + z_k10_s0156 + 0.833093 w_k10_ADM_s015
                                 + 0.683488 w_k10_AXP_s015
                                 + 0.698689 w_k10_DIS_s015
                                 + 0.817832 w_k10_FDX_s015
                                 + 0.94713 w_k10_IBM_s015
                                 + 0.870623 w_k10_INTC_s015
                                 + 0.876207 w_k10_KO_s015
                                 + 0.890109 w_k10_MAT_s015
                                 + 0.839618 w_k10_MSFT_s015
                                 + 0.821011 w_k10_PG_s015 >= 1
 z_k10_s0157_constraint:         var_10 + z_k10_s0157 + 1.0229 w_k10_ADM_s015
                                 + 1.02126 w_k10_AXP_s015
                                 + 0.964457 w_k10_DIS_s015
                                 + 1.06018 w_k10_FDX_s015
                                 + 0.942237 w_k10_IBM_s015
                                 + 1.01303 w_k10_INTC_s015
                                 + 0.848344 w_k10_KO_s015
                                 + 0.794251 w_k10_MAT_s015
                                 + 0.985148 w_k10_MSFT_s015
                                 + 1.0084 w_k10_PG_s015 >= 1
 z_k10_s0160_constraint:         var_10 + z_k10_s0160 + 0.94761 w_k10_ADM_s016
                                 + 0.723556 w_k10_AXP_s016
                                 + 0.710851 w_k10_DIS_s016
                                 + 0.859311 w_k10_FDX_s016
                                 + 0.816435 w_k10_IBM_s016
                                 + 0.96334 w_k10_INTC_s016
                                 + 0.686193 w_k10_KO_s016
                                 + 0.736257 w_k10_MAT_s016
                                 + 0.868631 w_k10_MSFT_s016
                                 + 0.931448 w_k10_PG_s016 >= 1
 z_k10_s0161_constraint:         var_10 + z_k10_s0161 + 1.30399 w_k10_ADM_s016
                                 + 1.65627 w_k10_AXP_s016
                                 + 1.39088 w_k10_DIS_s016
                                 + 1.30348 w_k10_FDX_s016
                                 + 1.12789 w_k10_IBM_s016
                                 + 1.31441 w_k10_INTC_s016
                                 + 1.15621 w_k10_KO_s016
                                 + 2.07264 w_k10_MAT_s016
                                 + 1.1934 w_k10_MSFT_s016
                                 + 1.32372 w_k10_PG_s016 >= 1
 z_k10_s0162_constraint:         var_10 + z_k10_s0162 + 1.27976 w_k10_ADM_s016
                                 + 1.28101 w_k10_AXP_s016
                                 + 1.28315 w_k10_DIS_s016
                                 + 1.17356 w_k10_FDX_s016
                                 + 1.49231 w_k10_IBM_s016
                                 + 1.19059 w_k10_INTC_s016
                                 + 1.30599 w_k10_KO_s016
                                 + 2.17709 w_k10_MAT_s016
                                 + 1.01596 w_k10_MSFT_s016
                                 + 1.20421 w_k10_PG_s016 >= 1
 z_k10_s0163_constraint:         var_10 + z_k10_s0163 + 0.979818 w_k10_ADM_s016
                                 + 0.951296 w_k10_AXP_s016
                                 + 0.837879 w_k10_DIS_s016
                                 + 0.949743 w_k10_FDX_s016
                                 + 1.00298 w_k10_IBM_s016
                                 + 0.931425 w_k10_INTC_s016
                                 + 0.823639 w_k10_KO_s016
                                 + 1.30837 w_k10_MAT_s016
                                 + 0.757585 w_k10_MSFT_s016
                                 + 0.900958 w_k10_PG_s016 >= 1
 z_k10_s0164_constraint:         var_10 + z_k10_s0164 + 1.27212 w_k10_ADM_s016
                                 + 1.2363 w_k10_AXP_s016
                                 + 1.17417 w_k10_DIS_s016
                                 + 1.18124 w_k10_FDX_s016
                                 + 1.04356 w_k10_IBM_s016
                                 + 1.13401 w_k10_INTC_s016
                                 + 0.909734 w_k10_KO_s016
                                 + 1.10663 w_k10_MAT_s016
                                 + 1.02107 w_k10_MSFT_s016
                                 + 1.00487 w_k10_PG_s016 >= 1
 z_k10_s0165_constraint:         var_10 + z_k10_s0165 + 1.18887 w_k10_ADM_s016
                                 + 1.10323 w_k10_AXP_s016
                                 + 1.14476 w_k10_DIS_s016
                                 + 1.22631 w_k10_FDX_s016
                                 + 0.694903 w_k10_IBM_s016
                                 + 1.23946 w_k10_INTC_s016
                                 + 0.945716 w_k10_KO_s016
                                 + 1.2499 w_k10_MAT_s016
                                 + 1.28924 w_k10_MSFT_s016
                                 + 1.02441 w_k10_PG_s016 >= 1
 z_k10_s0166_constraint:         var_10 + z_k10_s0166 + 0.891399 w_k10_ADM_s016
                                 + 0.690687 w_k10_AXP_s016
                                 + 0.480665 w_k10_DIS_s016
                                 + 0.702539 w_k10_FDX_s016
                                 + 0.617705 w_k10_IBM_s016
                                 + 0.793661 w_k10_INTC_s016
                                 + 0.581856 w_k10_KO_s016
                                 + 0.656186 w_k10_MAT_s016
                                 + 0.638228 w_k10_MSFT_s016
                                 + 0.774542 w_k10_PG_s016 >= 1
 z_k10_s0167_constraint:         var_10 + z_k10_s0167 + 0.721033 w_k10_ADM_s016
                                 + 1.07059 w_k10_AXP_s016
                                 + 1.03624 w_k10_DIS_s016
                                 + 0.883298 w_k10_FDX_s016
                                 + 1.19959 w_k10_IBM_s016
                                 + 0.985916 w_k10_INTC_s016
                                 + 1.03876 w_k10_KO_s016
                                 + 1.44861 w_k10_MAT_s016
                                 + 0.906426 w_k10_MSFT_s016
                                 + 0.914349 w_k10_PG_s016 >= 1
 z_k10_s0170_constraint:         var_10 + z_k10_s0170 + 1.02619 w_k10_ADM_s017
                                 + 0.963984 w_k10_AXP_s017
                                 + 1.1637 w_k10_DIS_s017
                                 + 1.07963 w_k10_FDX_s017
                                 + 0.912297 w_k10_IBM_s017
                                 + 1.05336 w_k10_INTC_s017
                                 + 0.962747 w_k10_KO_s017
                                 + 0.748015 w_k10_MAT_s017
                                 + 0.986078 w_k10_MSFT_s017
                                 + 0.98288 w_k10_PG_s017 >= 1
 z_k10_s0171_constraint:         var_10 + z_k10_s0171 + 0.722178 w_k10_ADM_s017
                                 + 0.726096 w_k10_AXP_s017
                                 + 0.665137 w_k10_DIS_s017
                                 + 0.744112 w_k10_FDX_s017
                                 + 1.12038 w_k10_IBM_s017
                                 + 0.860113 w_k10_INTC_s017
                                 + 0.763102 w_k10_KO_s017
                                 + 0.837477 w_k10_MAT_s017
                                 + 0.706255 w_k10_MSFT_s017
                                 + 0.94956 w_k10_PG_s017 >= 1
 z_k10_s0172_constraint:         var_10 + z_k10_s0172 + 0.838892 w_k10_ADM_s017
                                 + 0.83547 w_k10_AXP_s017
                                 + 0.787699 w_k10_DIS_s017
                                 + 0.867152 w_k10_FDX_s017
                                 + 1.58313 w_k10_IBM_s017
                                 + 0.890058 w_k10_INTC_s017
                                 + 1.12659 w_k10_KO_s017
                                 + 1.49394 w_k10_MAT_s017
                                 + 0.817116 w_k10_MSFT_s017
                                 + 0.932728 w_k10_PG_s017 >= 1
 z_k10_s0173_constraint:         var_10 + z_k10_s0173 + 0.936462 w_k10_ADM_s017
                                 + 0.87416 w_k10_AXP_s017
                                 + 0.907548 w_k10_DIS_s017
                                 + 0.934054 w_k10_FDX_s017
                                 + 0.818638 w_k10_IBM_s017
                                 + 0.954879 w_k10_INTC_s017
                                 + 0.81331 w_k10_KO_s017
                                 + 0.752227 w_k10_MAT_s017
                                 + 0.82965 w_k10_MSFT_s017
                                 + 0.844502 w_k10_PG_s017 >= 1
 z_k10_s0174_constraint:         var_10 + z_k10_s0174 + 1.07028 w_k10_ADM_s017
                                 + 0.850144 w_k10_AXP_s017
                                 + 0.782463 w_k10_DIS_s017
                                 + 1.05069 w_k10_FDX_s017
                                 + 0.632797 w_k10_IBM_s017
                                 + 0.957441 w_k10_INTC_s017
                                 + 0.82631 w_k10_KO_s017
                                 + 0.736143 w_k10_MAT_s017
                                 + 0.962091 w_k10_MSFT_s017
                                 + 0.990574 w_k10_PG_s017 >= 1
 z_k10_s0175_constraint:         var_10 + z_k10_s0175 + 0.950603 w_k10_ADM_s017
                                 + 1.25055 w_k10_AXP_s017
                                 + 1.25496 w_k10_DIS_s017
                                 + 1.18447 w_k10_FDX_s017
                                 + 1.31351 w_k10_IBM_s017
                                 + 1.03962 w_k10_INTC_s017
                                 + 0.997828 w_k10_KO_s017
                                 + 1.29844 w_k10_MAT_s017
                                 + 0.959539 w_k10_MSFT_s017
                                 + 1.14223 w_k10_PG_s017 >= 1
 z_k10_s0176_constraint:         var_10 + z_k10_s0176 + 0.852088 w_k10_ADM_s017
                                 + 0.78394 w_k10_AXP_s017
                                 + 0.613297 w_k10_DIS_s017
                                 + 0.83086 w_k10_FDX_s017
                                 + 0.799769 w_k10_IBM_s017
                                 + 0.770185 w_k10_INTC_s017
                                 + 0.751953 w_k10_KO_s017
                                 + 0.8485 w_k10_MAT_s017
                                 + 0.736309 w_k10_MSFT_s017
                                 + 0.80458 w_k10_PG_s017 >= 1
 z_k10_s0177_constraint:         var_10 + z_k10_s0177 + 1.59087 w_k10_ADM_s017
                                 + 1.27144 w_k10_AXP_s017
                                 + 1.32911 w_k10_DIS_s017
                                 + 1.44601 w_k10_FDX_s017
                                 + 0.766706 w_k10_IBM_s017
                                 + 1.26001 w_k10_INTC_s017
                                 + 0.983363 w_k10_KO_s017
                                 + 0.740421 w_k10_MAT_s017
                                 + 1.23588 w_k10_MSFT_s017
                                 + 1.13024 w_k10_PG_s017 >= 1
 z_k10_s0200_constraint:         var_10 + z_k10_s0200 + 1.17013 w_k10_ADM_s020
                                 + 0.905651 w_k10_AXP_s020
                                 + 1.12172 w_k10_DIS_s020
                                 + 1.01346 w_k10_FDX_s020
                                 + 0.865288 w_k10_IBM_s020
                                 + 0.892519 w_k10_INTC_s020
                                 + 0.838054 w_k10_KO_s020
                                 + 0.553212 w_k10_MAT_s020
                                 + 0.976289 w_k10_MSFT_s020
                                 + 0.980533 w_k10_PG_s020 >= 1
 z_k10_s0201_constraint:         var_10 + z_k10_s0201 + 1.26053 w_k10_ADM_s020
                                 + 1.37176 w_k10_AXP_s020
                                 + 1.71323 w_k10_DIS_s020
                                 + 1.40737 w_k10_FDX_s020
                                 + 1.3396 w_k10_IBM_s020
                                 + 1.45006 w_k10_INTC_s020
                                 + 1.22158 w_k10_KO_s020
                                 + 0.944912 w_k10_MAT_s020
                                 + 1.29843 w_k10_MSFT_s020
                                 + 1.15562 w_k10_PG_s020 >= 1
 z_k10_s0202_constraint:         var_10 + z_k10_s0202 + 1.32426 w_k10_ADM_s020
                                 + 1.14585 w_k10_AXP_s020
                                 + 1.29961 w_k10_DIS_s020
                                 + 1.24553 w_k10_FDX_s020
                                 + 0.838792 w_k10_IBM_s020
                                 + 1.09182 w_k10_INTC_s020
                                 + 1.0257 w_k10_KO_s020
                                 + 0.838048 w_k10_MAT_s020
                                 + 1.17063 w_k10_MSFT_s020
                                 + 1.1099 w_k10_PG_s020 >= 1
 z_k10_s0203_constraint:         var_10 + z_k10_s0203 + 0.933244 w_k10_ADM_s020
                                 + 0.941749 w_k10_AXP_s020
                                 + 0.910048 w_k10_DIS_s020
                                 + 0.854425 w_k10_FDX_s020
                                 + 1.40509 w_k10_IBM_s020
                                 + 1.20898 w_k10_INTC_s020
                                 + 1.05595 w_k10_KO_s020
                                 + 0.929247 w_k10_MAT_s020
                                 + 0.919191 w_k10_MSFT_s020
                                 + 1.0364 w_k10_PG_s020 >= 1
 z_k10_s0204_constraint:         var_10 + z_k10_s0204 + 1.07563 w_k10_ADM_s020
                                 + 1.15148 w_k10_AXP_s020
                                 + 1.39554 w_k10_DIS_s020
                                 + 1.14025 w_k10_FDX_s020
                                 + 1.16192 w_k10_IBM_s020
                                 + 1.17062 w_k10_INTC_s020
                                 + 1.11281 w_k10_KO_s020
                                 + 1.05958 w_k10_MAT_s020
                                 + 1.28704 w_k10_MSFT_s020
                                 + 1.05632 w_k10_PG_s020 >= 1
 z_k10_s0205_constraint:         var_10 + z_k10_s0205 + 1.07348 w_k10_ADM_s020
                                 + 0.752606 w_k10_AXP_s020
                                 + 0.848927 w_k10_DIS_s020
                                 + 0.952912 w_k10_FDX_s020
                                 + 0.813542 w_k10_IBM_s020
                                 + 0.924253 w_k10_INTC_s020
                                 + 0.718944 w_k10_KO_s020
                                 + 0.549126 w_k10_MAT_s020
                                 + 0.938708 w_k10_MSFT_s020
                                 + 0.930198 w_k10_PG_s020 >= 1
 z_k10_s0206_constraint:         var_10 + z_k10_s0206 + 0.843245 w_k10_ADM_s020
                                 + 0.934231 w_k10_AXP_s020
                                 + 0.724482 w_k10_DIS_s020
                                 + 0.831497 w_k10_FDX_s020
                                 + 0.879041 w_k10_IBM_s020
                                 + 0.812529 w_k10_INTC_s020
                                 + 0.795007 w_k10_KO_s020
                                 + 0.792421 w_k10_MAT_s020
                                 + 0.776957 w_k10_MSFT_s020
                                 + 0.887856 w_k10_PG_s020 >= 1
 z_k10_s0207_constraint:         var_10 + z_k10_s0207 + 1.09508 w_k10_ADM_s020
                                 + 1.13159 w_k10_AXP_s020
                                 + 1.03044 w_k10_DIS_s020
                                 + 1.09475 w_k10_FDX_s020
                                 + 1.13646 w_k10_IBM_s020
                                 + 0.900055 w_k10_INTC_s020
                                 + 0.895813 w_k10_KO_s020
                                 + 1.08072 w_k10_MAT_s020
                                 + 0.98499 w_k10_MSFT_s020
                                 + 1.03779 w_k10_PG_s020 >= 1
 z_k10_s0210_constraint:         var_10 + z_k10_s0210 + 1.04687 w_k10_ADM_s021
                                 + 1.10856 w_k10_AXP_s021
                                 + 0.88993 w_k10_DIS_s021
                                 + 0.907989 w_k10_FDX_s021
                                 + 1.31194 w_k10_IBM_s021
                                 + 0.872838 w_k10_INTC_s021
                                 + 1.05955 w_k10_KO_s021
                                 + 1.14044 w_k10_MAT_s021
                                 + 0.90211 w_k10_MSFT_s021
                                 + 1.08087 w_k10_PG_s021 >= 1
 z_k10_s0211_constraint:         var_10 + z_k10_s0211 + 0.687142 w_k10_ADM_s021
                                 + 0.767669 w_k10_AXP_s021
                                 + 0.5452 w_k10_DIS_s021
                                 + 0.705064 w_k10_FDX_s021
                                 + 1.02475 w_k10_IBM_s021
                                 + 0.912842 w_k10_INTC_s021
                                 + 0.786357 w_k10_KO_s021
                                 + 0.883336 w_k10_MAT_s021
                                 + 0.801486 w_k10_MSFT_s021
                                 + 0.952689 w_k10_PG_s021 >= 1
 z_k10_s0212_constraint:         var_10 + z_k10_s0212 + 0.883591 w_k10_ADM_s021
                                 + 0.831196 w_k10_AXP_s021
                                 + 0.631184 w_k10_DIS_s021
                                 + 0.838351 w_k10_FDX_s021
                                 + 0.735528 w_k10_IBM_s021
                                 + 1.00806 w_k10_INTC_s021
                                 + 0.702458 w_k10_KO_s021
                                 + 0.57082 w_k10_MAT_s021
                                 + 0.790287 w_k10_MSFT_s021
                                 + 0.947172 w_k10_PG_s021 >= 1
 z_k10_s0213_constraint:         var_10 + z_k10_s0213 + 0.943025 w_k10_ADM_s021
                                 + 0.934063 w_k10_AXP_s021
                                 + 1.10067 w_k10_DIS_s021
                                 + 1.01515 w_k10_FDX_s021
                                 + 0.885942 w_k10_IBM_s021
                                 + 1.11439 w_k10_INTC_s021
                                 + 1.06624 w_k10_KO_s021
                                 + 0.843878 w_k10_MAT_s021
                                 + 1.22086 w_k10_MSFT_s021
                                 + 0.959281 w_k10_PG_s021 >= 1
 z_k10_s0214_constraint:         var_10 + z_k10_s0214 + 0.902642 w_k10_ADM_s021
                                 + 0.790462 w_k10_AXP_s021
                                 + 0.793544 w_k10_DIS_s021
                                 + 0.946732 w_k10_FDX_s021
                                 + 0.682802 w_k10_IBM_s021
                                 + 0.892808 w_k10_INTC_s021
                                 + 0.740408 w_k10_KO_s021
                                 + 0.704335 w_k10_MAT_s021
                                 + 0.802693 w_k10_MSFT_s021
                                 + 0.960961 w_k10_PG_s021 >= 1
 z_k10_s0215_constraint:         var_10 + z_k10_s0215 + 1.37652 w_k10_ADM_s021
                                 + 1.45449 w_k10_AXP_s021
                                 + 1.39134 w_k10_DIS_s021
                                 + 1.31448 w_k10_FDX_s021
                                 + 1.20426 w_k10_IBM_s021
                                 + 1.11344 w_k10_INTC_s021
                                 + 1.21425 w_k10_KO_s021
                                 + 1.64046 w_k10_MAT_s021
                                 + 1.20463 w_k10_MSFT_s021
                                 + 1.17918 w_k10_PG_s021 >= 1
 z_k10_s0216_constraint:         var_10 + z_k10_s0216 + 0.804443 w_k10_ADM_s021
                                 + 0.834372 w_k10_AXP_s021
                                 + 0.917442 w_k10_DIS_s021
                                 + 0.925031 w_k10_FDX_s021
                                 + 0.885593 w_k10_IBM_s021
                                 + 0.927944 w_k10_INTC_s021
                                 + 0.841429 w_k10_KO_s021
                                 + 1.02371 w_k10_MAT_s021
                                 + 1.00676 w_k10_MSFT_s021
                                 + 0.951934 w_k10_PG_s021 >= 1
 z_k10_s0217_constraint:         var_10 + z_k10_s0217 + 1.37738 w_k10_ADM_s021
                                 + 1.07378 w_k10_AXP_s021
                                 + 0.94923 w_k10_DIS_s021
                                 + 1.14463 w_k10_FDX_s021
                                 + 0.770419 w_k10_IBM_s021
                                 + 0.927322 w_k10_INTC_s021
                                 + 0.885904 w_k10_KO_s021
                                 + 0.759742 w_k10_MAT_s021
                                 + 1.00134 w_k10_MSFT_s021
                                 + 1.02647 w_k10_PG_s021 >= 1
 z_k10_s0220_constraint:         var_10 + z_k10_s0220 + 0.879687 w_k10_ADM_s022
                                 + 0.705974 w_k10_AXP_s022
                                 + 1.15181 w_k10_DIS_s022
                                 + 0.938294 w_k10_FDX_s022
                                 + 0.826006 w_k10_IBM_s022
                                 + 0.998665 w_k10_INTC_s022
                                 + 0.901218 w_k10_KO_s022
                                 + 0.815766 w_k10_MAT_s022
                                 + 0.980659 w_k10_MSFT_s022
                                 + 1.05668 w_k10_PG_s022 >= 1
 z_k10_s0221_constraint:         var_10 + z_k10_s0221 + 1.18613 w_k10_ADM_s022
                                 + 0.994335 w_k10_AXP_s022
                                 + 1.04479 w_k10_DIS_s022
                                 + 1.12733 w_k10_FDX_s022
                                 + 0.808296 w_k10_IBM_s022
                                 + 0.861903 w_k10_INTC_s022
                                 + 0.844564 w_k10_KO_s022
                                 + 1.20253 w_k10_MAT_s022
                                 + 1.02728 w_k10_MSFT_s022
                                 + 1.01872 w_k10_PG_s022 >= 1
 z_k10_s0222_constraint:         var_10 + z_k10_s0222 + 1.24451 w_k10_ADM_s022
                                 + 1.31928 w_k10_AXP_s022
                                 + 1.39269 w_k10_DIS_s022
                                 + 1.23392 w_k10_FDX_s022
                                 + 1.37815 w_k10_IBM_s022
                                 + 1.27081 w_k10_INTC_s022
                                 + 1.1739 w_k10_KO_s022
                                 + 1.25734 w_k10_MAT_s022
                                 + 1.25727 w_k10_MSFT_s022
                                 + 1.23286 w_k10_PG_s022 >= 1
 z_k10_s0223_constraint:         var_10 + z_k10_s0223 + 0.938926 w_k10_ADM_s022
                                 + 0.61454 w_k10_AXP_s022
                                 + 0.687417 w_k10_DIS_s022
                                 + 0.770405 w_k10_FDX_s022
                                 + 0.69123 w_k10_IBM_s022
                                 + 0.815806 w_k10_INTC_s022
                                 + 0.77225 w_k10_KO_s022
                                 + 0.878115 w_k10_MAT_s022
                                 + 0.788108 w_k10_MSFT_s022
                                 + 0.823544 w_k10_PG_s022 >= 1
 z_k10_s0224_constraint:         var_10 + z_k10_s0224 + 1.04461 w_k10_ADM_s022
                                 + 1.05571 w_k10_AXP_s022
                                 + 0.883629 w_k10_DIS_s022
                                 + 0.971319 w_k10_FDX_s022
                                 + 0.93323 w_k10_IBM_s022
                                 + 1.05296 w_k10_INTC_s022
                                 + 0.901846 w_k10_KO_s022
                                 + 1.46932 w_k10_MAT_s022
                                 + 0.996161 w_k10_MSFT_s022
                                 + 1.01564 w_k10_PG_s022 >= 1
 z_k10_s0225_constraint:         var_10 + z_k10_s0225 + 1.45141 w_k10_ADM_s022
                                 + 1.44688 w_k10_AXP_s022
                                 + 1.26561 w_k10_DIS_s022
                                 + 1.33747 w_k10_FDX_s022
                                 + 0.92666 w_k10_IBM_s022
                                 + 1.3151 w_k10_INTC_s022
                                 + 1.03581 w_k10_KO_s022
                                 + 0.935675 w_k10_MAT_s022
                                 + 1.21344 w_k10_MSFT_s022
                                 + 1.17033 w_k10_PG_s022 >= 1
 z_k10_s0226_constraint:         var_10 + z_k10_s0226 + 1.3554 w_k10_ADM_s022
                                 + 1.4533 w_k10_AXP_s022
                                 + 2.00096 w_k10_DIS_s022
                                 + 1.59239 w_k10_FDX_s022
                                 + 1.27369 w_k10_IBM_s022
                                 + 1.26847 w_k10_INTC_s022
                                 + 1.47807 w_k10_KO_s022
                                 + 1.76842 w_k10_MAT_s022
                                 + 1.41919 w_k10_MSFT_s022
                                 + 1.3428 w_k10_PG_s022 >= 1
 z_k10_s0227_constraint:         var_10 + z_k10_s0227 + 1.11589 w_k10_ADM_s022
                                 + 0.859694 w_k10_AXP_s022
                                 + 1.49281 w_k10_DIS_s022
                                 + 1.11479 w_k10_FDX_s022
                                 + 1.12264 w_k10_IBM_s022
                                 + 1.07946 w_k10_INTC_s022
                                 + 1.1617 w_k10_KO_s022
                                 + 1.09858 w_k10_MAT_s022
                                 + 1.21935 w_k10_MSFT_s022
                                 + 1.19521 w_k10_PG_s022 >= 1
 z_k10_s0230_constraint:         var_10 + z_k10_s0230 + 1.08914 w_k10_ADM_s023
                                 + 1.40936 w_k10_AXP_s023
                                 + 1.88738 w_k10_DIS_s023
                                 + 1.32802 w_k10_FDX_s023
                                 + 1.34299 w_k10_IBM_s023
                                 + 1.19587 w_k10_INTC_s023
                                 + 1.49372 w_k10_KO_s023
                                 + 1.47251 w_k10_MAT_s023
                                 + 1.43938 w_k10_MSFT_s023
                                 + 1.2508 w_k10_PG_s023 >= 1
 z_k10_s0231_constraint:         var_10 + z_k10_s0231 + 1.12627 w_k10_ADM_s023
                                 + 1.31442 w_k10_AXP_s023
                                 + 1.21252 w_k10_DIS_s023
                                 + 1.10962 w_k10_FDX_s023
                                 + 1.31518 w_k10_IBM_s023
                                 + 1.0695 w_k10_INTC_s023
                                 + 1.39638 w_k10_KO_s023
                                 + 1.55626 w_k10_MAT_s023
                                 + 1.21898 w_k10_MSFT_s023
                                 + 1.17966 w_k10_PG_s023 >= 1
 z_k10_s0232_constraint:         var_10 + z_k10_s0232 + 0.852133 w_k10_ADM_s023
                                 + 0.933992 w_k10_AXP_s023
                                 + 0.965721 w_k10_DIS_s023
                                 + 0.996775 w_k10_FDX_s023
                                 + 0.899033 w_k10_IBM_s023
                                 + 0.811348 w_k10_INTC_s023
                                 + 0.859103 w_k10_KO_s023
                                 + 1.19325 w_k10_MAT_s023
                                 + 0.994509 w_k10_MSFT_s023
                                 + 0.884459 w_k10_PG_s023 >= 1
 z_k10_s0233_constraint:         var_10 + z_k10_s0233 + 0.916081 w_k10_ADM_s023
                                 + 1.15062 w_k10_AXP_s023
                                 + 0.811321 w_k10_DIS_s023
                                 + 0.918834 w_k10_FDX_s023
                                 + 0.92772 w_k10_IBM_s023
                                 + 0.881092 w_k10_INTC_s023
                                 + 0.902761 w_k10_KO_s023
                                 + 1.02958 w_k10_MAT_s023
                                 + 0.970918 w_k10_MSFT_s023
                                 + 1.00717 w_k10_PG_s023 >= 1
 z_k10_s0234_constraint:         var_10 + z_k10_s0234 + 0.935966 w_k10_ADM_s023
                                 + 1.06021 w_k10_AXP_s023
                                 + 1.02206 w_k10_DIS_s023
                                 + 0.965599 w_k10_FDX_s023
                                 + 1.39361 w_k10_IBM_s023
                                 + 1.09084 w_k10_INTC_s023
                                 + 1.13845 w_k10_KO_s023
                                 + 1.40927 w_k10_MAT_s023
                                 + 1.04036 w_k10_MSFT_s023
                                 + 0.939622 w_k10_PG_s023 >= 1
 z_k10_s0235_constraint:         var_10 + z_k10_s0235 + 0.70168 w_k10_ADM_s023
                                 + 0.75141 w_k10_AXP_s023
                                 + 0.681895 w_k10_DIS_s023
                                 + 0.712795 w_k10_FDX_s023
                                 + 0.765174 w_k10_IBM_s023
                                 + 0.879663 w_k10_INTC_s023
                                 + 0.757224 w_k10_KO_s023
                                 + 0.589458 w_k10_MAT_s023
                                 + 0.858108 w_k10_MSFT_s023
                                 + 0.787783 w_k10_PG_s023 >= 1
 z_k10_s0236_constraint:         var_10 + z_k10_s0236 + 1.18876 w_k10_ADM_s023
                                 + 1.15543 w_k10_AXP_s023
                                 + 1.35118 w_k10_DIS_s023
                                 + 1.30156 w_k10_FDX_s023
                                 + 0.888432 w_k10_IBM_s023
                                 + 1.2003 w_k10_INTC_s023
                                 + 1.06103 w_k10_KO_s023
                                 + 0.947422 w_k10_MAT_s023
                                 + 1.12799 w_k10_MSFT_s023
                                 + 0.898008 w_k10_PG_s023 >= 1
 z_k10_s0237_constraint:         var_10 + z_k10_s0237 + 1.20466 w_k10_ADM_s023
                                 + 1.08393 w_k10_AXP_s023
                                 + 1.35418 w_k10_DIS_s023
                                 + 1.20009 w_k10_FDX_s023
                                 + 1.00527 w_k10_IBM_s023
                                 + 1.11736 w_k10_INTC_s023
                                 + 1.07789 w_k10_KO_s023
                                 + 0.919285 w_k10_MAT_s023
                                 + 1.36574 w_k10_MSFT_s023
                                 + 1.15631 w_k10_PG_s023 >= 1
 z_k10_s0240_constraint:         var_10 + z_k10_s0240 + 1.13601 w_k10_ADM_s024
                                 + 1.09622 w_k10_AXP_s024
                                 + 1.15044 w_k10_DIS_s024
                                 + 1.09818 w_k10_FDX_s024
                                 + 0.850862 w_k10_IBM_s024
                                 + 1.22982 w_k10_INTC_s024
                                 + 1.08309 w_k10_KO_s024
                                 + 0.855209 w_k10_MAT_s024
                                 + 1.18346 w_k10_MSFT_s024
                                 + 1.08232 w_k10_PG_s024 >= 1
 z_k10_s0241_constraint:         var_10 + z_k10_s0241 + 0.925419 w_k10_ADM_s024
                                 + 1.06946 w_k10_AXP_s024
                                 + 1.01729 w_k10_DIS_s024
                                 + 1.01715 w_k10_FDX_s024
                                 + 1.1462 w_k10_IBM_s024
                                 + 1.19194 w_k10_INTC_s024
                                 + 1.15657 w_k10_KO_s024
                                 + 0.996358 w_k10_MAT_s024
                                 + 0.986572 w_k10_MSFT_s024
                                 + 1.13959 w_k10_PG_s024 >= 1
 z_k10_s0242_constraint:         var_10 + z_k10_s0242 + 1.43029 w_k10_ADM_s024
                                 + 1.23506 w_k10_AXP_s024
                                 + 1.63224 w_k10_DIS_s024
                                 + 1.39581 w_k10_FDX_s024
                                 + 1.23943 w_k10_IBM_s024
                                 + 1.36327 w_k10_INTC_s024
                                 + 1.40198 w_k10_KO_s024
                                 + 1.15805 w_k10_MAT_s024
                                 + 1.33128 w_k10_MSFT_s024
                                 + 1.32295 w_k10_PG_s024 >= 1
 z_k10_s0243_constraint:         var_10 + z_k10_s0243 + 1.20837 w_k10_ADM_s024
                                 + 1.29496 w_k10_AXP_s024
                                 + 1.05988 w_k10_DIS_s024
                                 + 1.14026 w_k10_FDX_s024
                                 + 1.26357 w_k10_IBM_s024
                                 + 1.0749 w_k10_INTC_s024
                                 + 1.08341 w_k10_KO_s024
                                 + 1.27082 w_k10_MAT_s024
                                 + 0.908907 w_k10_MSFT_s024
                                 + 1.08052 w_k10_PG_s024 >= 1
 z_k10_s0244_constraint:         var_10 + z_k10_s0244 + 0.99913 w_k10_ADM_s024
                                 + 0.862496 w_k10_AXP_s024
                                 + 0.878579 w_k10_DIS_s024
                                 + 0.841352 w_k10_FDX_s024
                                 + 0.978382 w_k10_IBM_s024
                                 + 1.01974 w_k10_INTC_s024
                                 + 0.965244 w_k10_KO_s024
                                 + 0.507759 w_k10_MAT_s024
                                 + 0.838448 w_k10_MSFT_s024
                                 + 0.936587 w_k10_PG_s024 >= 1
 z_k10_s0245_constraint:         var_10 + z_k10_s0245 + 0.93763 w_k10_ADM_s024
                                 + 0.989959 w_k10_AXP_s024
                                 + 0.735739 w_k10_DIS_s024
                                 + 0.869457 w_k10_FDX_s024
                                 + 1.24203 w_k10_IBM_s024
                                 + 0.977868 w_k10_INTC_s024
                                 + 0.998218 w_k10_KO_s024
                                 + 1.16681 w_k10_MAT_s024
                                 + 0.851429 w_k10_MSFT_s024
                                 + 0.930924 w_k10_PG_s024 >= 1
 z_k10_s0246_constraint:         var_10 + z_k10_s0246 + 0.921854 w_k10_ADM_s024
                                 + 0.760256 w_k10_AXP_s024
                                 + 0.695883 w_k10_DIS_s024
                                 + 0.824259 w_k10_FDX_s024
                                 + 0.839048 w_k10_IBM_s024
                                 + 0.901486 w_k10_INTC_s024
                                 + 0.918538 w_k10_KO_s024
                                 + 0.729503 w_k10_MAT_s024
                                 + 0.892947 w_k10_MSFT_s024
                                 + 0.877291 w_k10_PG_s024 >= 1
 z_k10_s0247_constraint:         var_10 + z_k10_s0247 + 0.873431 w_k10_ADM_s024
                                 + 0.962309 w_k10_AXP_s024
                                 + 0.958277 w_k10_DIS_s024
                                 + 0.847178 w_k10_FDX_s024
                                 + 1.58898 w_k10_IBM_s024
                                 + 1.02239 w_k10_INTC_s024
                                 + 1.18122 w_k10_KO_s024
                                 + 1.26845 w_k10_MAT_s024
                                 + 0.884163 w_k10_MSFT_s024
                                 + 1.04447 w_k10_PG_s024 >= 1
 z_k10_s0250_constraint:         var_10 + z_k10_s0250 + 1.75544 w_k10_ADM_s025
                                 + 1.63947 w_k10_AXP_s025
                                 + 1.74781 w_k10_DIS_s025
                                 + 1.61918 w_k10_FDX_s025
                                 + 1.05577 w_k10_IBM_s025
                                 + 1.15692 w_k10_INTC_s025
                                 + 1.21131 w_k10_KO_s025
                                 + 1.22546 w_k10_MAT_s025
                                 + 1.30045 w_k10_MSFT_s025
                                 + 1.39725 w_k10_PG_s025 >= 1
 z_k10_s0251_constraint:         var_10 + z_k10_s0251 + 0.901077 w_k10_ADM_s025
                                 + 0.734948 w_k10_AXP_s025
                                 + 0.855728 w_k10_DIS_s025
                                 + 0.956431 w_k10_FDX_s025
                                 + 0.669927 w_k10_IBM_s025
                                 + 0.848474 w_k10_INTC_s025
                                 + 0.860718 w_k10_KO_s025
                                 + 0.742106 w_k10_MAT_s025
                                 + 1.09509 w_k10_MSFT_s025
                                 + 0.959077 w_k10_PG_s025 >= 1
 z_k10_s0252_constraint:         var_10 + z_k10_s0252 + 0.815851 w_k10_ADM_s025
                                 + 0.798559 w_k10_AXP_s025
                                 + 0.89453 w_k10_DIS_s025
                                 + 0.798056 w_k10_FDX_s025
                                 + 1.08732 w_k10_IBM_s025
                                 + 0.72446 w_k10_INTC_s025
                                 + 1.05887 w_k10_KO_s025
                                 + 1.09701 w_k10_MAT_s025
                                 + 0.822749 w_k10_MSFT_s025
                                 + 1.03706 w_k10_PG_s025 >= 1
 z_k10_s0253_constraint:         var_10 + z_k10_s0253 + 1.11303 w_k10_ADM_s025
                                 + 1.09372 w_k10_AXP_s025
                                 + 1.05572 w_k10_DIS_s025
                                 + 1.22221 w_k10_FDX_s025
                                 + 1.01092 w_k10_IBM_s025
                                 + 0.947217 w_k10_INTC_s025
                                 + 1.12763 w_k10_KO_s025
                                 + 1.28935 w_k10_MAT_s025
                                 + 0.989144 w_k10_MSFT_s025
                                 + 1.03289 w_k10_PG_s025 >= 1
 z_k10_s0254_constraint:         var_10 + z_k10_s0254 + 1.09661 w_k10_ADM_s025
                                 + 0.807134 w_k10_AXP_s025
                                 + 1.13925 w_k10_DIS_s025
                                 + 1.00419 w_k10_FDX_s025
                                 + 0.82565 w_k10_IBM_s025
                                 + 0.903907 w_k10_INTC_s025
                                 + 1.09747 w_k10_KO_s025
                                 + 1.01264 w_k10_MAT_s025
                                 + 1.06094 w_k10_MSFT_s025
                                 + 1.17876 w_k10_PG_s025 >= 1
 z_k10_s0255_constraint:         var_10 + z_k10_s0255 + 1.17078 w_k10_ADM_s025
                                 + 1.14883 w_k10_AXP_s025
                                 + 0.856685 w_k10_DIS_s025
                                 + 0.880836 w_k10_FDX_s025
                                 + 1.18509 w_k10_IBM_s025
                                 + 0.967392 w_k10_INTC_s025
                                 + 1.05547 w_k10_KO_s025
                                 + 1.25051 w_k10_MAT_s025
                                 + 0.917698 w_k10_MSFT_s025
                                 + 1.13864 w_k10_PG_s025 >= 1
 z_k10_s0256_constraint:         var_10 + z_k10_s0256 + 1.19391 w_k10_ADM_s025
                                 + 1.41751 w_k10_AXP_s025
                                 + 1.55084 w_k10_DIS_s025
                                 + 1.24712 w_k10_FDX_s025
                                 + 1.47716 w_k10_IBM_s025
                                 + 1.23692 w_k10_INTC_s025
                                 + 1.40909 w_k10_KO_s025
                                 + 1.60124 w_k10_MAT_s025
                                 + 1.27127 w_k10_MSFT_s025
                                 + 1.31852 w_k10_PG_s025 >= 1
 z_k10_s0257_constraint:         var_10 + z_k10_s0257 + 1.38397 w_k10_ADM_s025
                                 + 1.15558 w_k10_AXP_s025
                                 + 1.26145 w_k10_DIS_s025
                                 + 1.23669 w_k10_FDX_s025
                                 + 0.970705 w_k10_IBM_s025
                                 + 1.21203 w_k10_INTC_s025
                                 + 1.12607 w_k10_KO_s025
                                 + 0.827221 w_k10_MAT_s025
                                 + 1.27108 w_k10_MSFT_s025
                                 + 1.25337 w_k10_PG_s025 >= 1
 z_k10_s0260_constraint:         var_10 + z_k10_s0260 + 1.05845 w_k10_ADM_s026
                                 + 1.42338 w_k10_AXP_s026
                                 + 1.43132 w_k10_DIS_s026
                                 + 1.26722 w_k10_FDX_s026
                                 + 1.1093 w_k10_IBM_s026
                                 + 1.25336 w_k10_INTC_s026
                                 + 1.30096 w_k10_KO_s026
                                 + 1.85576 w_k10_MAT_s026
                                 + 1.29672 w_k10_MSFT_s026
                                 + 1.20336 w_k10_PG_s026 >= 1
 z_k10_s0261_constraint:         var_10 + z_k10_s0261 + 0.809712 w_k10_ADM_s026
                                 + 1.09069 w_k10_AXP_s026
                                 + 0.817466 w_k10_DIS_s026
                                 + 0.908539 w_k10_FDX_s026
                                 + 1.06412 w_k10_IBM_s026
                                 + 1.06022 w_k10_INTC_s026
                                 + 0.903649 w_k10_KO_s026
                                 + 1.36758 w_k10_MAT_s026
                                 + 0.888779 w_k10_MSFT_s026
                                 + 0.895682 w_k10_PG_s026 >= 1
 z_k10_s0262_constraint:         var_10 + z_k10_s0262 + 0.776544 w_k10_ADM_s026
                                 + 0.89423 w_k10_AXP_s026
                                 + 1.04013 w_k10_DIS_s026
                                 + 0.873189 w_k10_FDX_s026
                                 + 1.42373 w_k10_IBM_s026
                                 + 1.03083 w_k10_INTC_s026
                                 + 1.19149 w_k10_KO_s026
                                 + 1.7948 w_k10_MAT_s026
                                 + 1.03046 w_k10_MSFT_s026
                                 + 0.919281 w_k10_PG_s026 >= 1
 z_k10_s0263_constraint:         var_10 + z_k10_s0263 + 1.06896 w_k10_ADM_s026
                                 + 0.969802 w_k10_AXP_s026
                                 + 0.918943 w_k10_DIS_s026
                                 + 1.04216 w_k10_FDX_s026
                                 + 0.672667 w_k10_IBM_s026
                                 + 0.959387 w_k10_INTC_s026
                                 + 0.84929 w_k10_KO_s026
                                 + 1.24495 w_k10_MAT_s026
                                 + 1.0745 w_k10_MSFT_s026
                                 + 1.00924 w_k10_PG_s026 >= 1
 z_k10_s0264_constraint:         var_10 + z_k10_s0264 + 1.29573 w_k10_ADM_s026
                                 + 1.1288 w_k10_AXP_s026
                                 + 1.31578 w_k10_DIS_s026
                                 + 1.23272 w_k10_FDX_s026
                                 + 1.28311 w_k10_IBM_s026
                                 + 1.47749 w_k10_INTC_s026
                                 + 1.30925 w_k10_KO_s026
                                 + 1.43447 w_k10_MAT_s026
                                 + 1.38377 w_k10_MSFT_s026
                                 + 1.20494 w_k10_PG_s026 >= 1
 z_k10_s0265_constraint:         var_10 + z_k10_s0265 + 1.05095 w_k10_ADM_s026
                                 + 0.813204 w_k10_AXP_s026
                                 + 1.07324 w_k10_DIS_s026
                                 + 0.98149 w_k10_FDX_s026
                                 + 0.767186 w_k10_IBM_s026
                                 + 1.01805 w_k10_INTC_s026
                                 + 1.10812 w_k10_KO_s026
                                 + 0.790877 w_k10_MAT_s026
                                 + 1.12041 w_k10_MSFT_s026
                                 + 0.976827 w_k10_PG_s026 >= 1
 z_k10_s0266_constraint:         var_10 + z_k10_s0266 + 0.668593 w_k10_ADM_s026
                                 + 0.7388 w_k10_AXP_s026
                                 + 0.626912 w_k10_DIS_s026
                                 + 0.728984 w_k10_FDX_s026
                                 + 0.893423 w_k10_IBM_s026
                                 + 0.838879 w_k10_INTC_s026
                                 + 0.670984 w_k10_KO_s026
                                 + 0.973467 w_k10_MAT_s026
                                 + 0.733431 w_k10_MSFT_s026
                                 + 0.785118 w_k10_PG_s026 >= 1
 z_k10_s0267_constraint:         var_10 + z_k10_s0267 + 0.824882 w_k10_ADM_s026
                                 + 0.790255 w_k10_AXP_s026
                                 + 0.838468 w_k10_DIS_s026
                                 + 0.896959 w_k10_FDX_s026
                                 + 0.90952 w_k10_IBM_s026
                                 + 0.845876 w_k10_INTC_s026
                                 + 0.858908 w_k10_KO_s026
                                 + 0.860273 w_k10_MAT_s026
                                 + 0.934634 w_k10_MSFT_s026
                                 + 0.960408 w_k10_PG_s026 >= 1
 z_k10_s0270_constraint:         var_10 + z_k10_s0270 + 1.30596 w_k10_ADM_s027
                                 + 1.13265 w_k10_AXP_s027
                                 + 1.13405 w_k10_DIS_s027
                                 + 1.1363 w_k10_FDX_s027
                                 + 1.13118 w_k10_IBM_s027
                                 + 0.898219 w_k10_INTC_s027
                                 + 0.970843 w_k10_KO_s027
                                 + 1.25642 w_k10_MAT_s027
                                 + 0.993191 w_k10_MSFT_s027
                                 + 1.12257 w_k10_PG_s027 >= 1
 z_k10_s0271_constraint:         var_10 + z_k10_s0271 + 0.842934 w_k10_ADM_s027
                                 + 0.871879 w_k10_AXP_s027
                                 + 0.773796 w_k10_DIS_s027
                                 + 0.904575 w_k10_FDX_s027
                                 + 1.09898 w_k10_IBM_s027
                                 + 0.783799 w_k10_INTC_s027
                                 + 0.819337 w_k10_KO_s027
                                 + 0.941256 w_k10_MAT_s027
                                 + 0.807036 w_k10_MSFT_s027
                                 + 0.901802 w_k10_PG_s027 >= 1
 z_k10_s0272_constraint:         var_10 + z_k10_s0272 + 0.715536 w_k10_ADM_s027
                                 + 0.806407 w_k10_AXP_s027
                                 + 0.794481 w_k10_DIS_s027
                                 + 0.934432 w_k10_FDX_s027
                                 + 0.66254 w_k10_IBM_s027
                                 + 0.841539 w_k10_INTC_s027
                                 + 0.674196 w_k10_KO_s027
                                 + 0.762362 w_k10_MAT_s027
                                 + 0.876083 w_k10_MSFT_s027
                                 + 0.819708 w_k10_PG_s027 >= 1
 z_k10_s0273_constraint:         var_10 + z_k10_s0273 + 0.957475 w_k10_ADM_s027
                                 + 0.819628 w_k10_AXP_s027
                                 + 1.01425 w_k10_DIS_s027
                                 + 0.938658 w_k10_FDX_s027
                                 + 1.42215 w_k10_IBM_s027
                                 + 1.22022 w_k10_INTC_s027
                                 + 1.11782 w_k10_KO_s027
                                 + 1.16624 w_k10_MAT_s027
                                 + 1.0337 w_k10_MSFT_s027
                                 + 1.06155 w_k10_PG_s027 >= 1
 z_k10_s0274_constraint:         var_10 + z_k10_s0274 + 1.37704 w_k10_ADM_s027
                                 + 1.57438 w_k10_AXP_s027
                                 + 1.62636 w_k10_DIS_s027
                                 + 1.39179 w_k10_FDX_s027
                                 + 1.52244 w_k10_IBM_s027
                                 + 1.16903 w_k10_INTC_s027
                                 + 1.37762 w_k10_KO_s027
                                 + 1.59618 w_k10_MAT_s027
                                 + 1.32574 w_k10_MSFT_s027
                                 + 1.09803 w_k10_PG_s027 >= 1
 z_k10_s0275_constraint:         var_10 + z_k10_s0275 + 0.804563 w_k10_ADM_s027
                                 + 0.716877 w_k10_AXP_s027
                                 + 0.719278 w_k10_DIS_s027
                                 + 0.740922 w_k10_FDX_s027
                                 + 0.826202 w_k10_IBM_s027
                                 + 0.845206 w_k10_INTC_s027
                                 + 0.750358 w_k10_KO_s027
                                 + 0.603434 w_k10_MAT_s027
                                 + 0.75054 w_k10_MSFT_s027
                                 + 0.823738 w_k10_PG_s027 >= 1
 z_k10_s0276_constraint:         var_10 + z_k10_s0276 + 0.81036 w_k10_ADM_s027
                                 + 1.20365 w_k10_AXP_s027
                                 + 1.06632 w_k10_DIS_s027
                                 + 0.937347 w_k10_FDX_s027
                                 + 1.30261 w_k10_IBM_s027
                                 + 0.887573 w_k10_INTC_s027
                                 + 0.891138 w_k10_KO_s027
                                 + 1.48206 w_k10_MAT_s027
                                 + 0.842079 w_k10_MSFT_s027
                                 + 0.919593 w_k10_PG_s027 >= 1
 z_k10_s0277_constraint:         var_10 + z_k10_s0277 + 1.19902 w_k10_ADM_s027
                                 + 1.08499 w_k10_AXP_s027
                                 + 1.32291 w_k10_DIS_s027
                                 + 1.18864 w_k10_FDX_s027
                                 + 0.930368 w_k10_IBM_s027
                                 + 0.910473 w_k10_INTC_s027
                                 + 0.979736 w_k10_KO_s027
                                 + 1.13057 w_k10_MAT_s027
                                 + 1.03895 w_k10_MSFT_s027
                                 + 1.03612 w_k10_PG_s027 >= 1
 z_k10_s0300_constraint:         var_10 + z_k10_s0300 + 1.28758 w_k10_ADM_s030
                                 + 1.07639 w_k10_AXP_s030
                                 + 1.13951 w_k10_DIS_s030
                                 + 1.17539 w_k10_FDX_s030
                                 + 0.898692 w_k10_IBM_s030
                                 + 1.29248 w_k10_INTC_s030
                                 + 1.14629 w_k10_KO_s030
                                 + 0.895031 w_k10_MAT_s030
                                 + 1.20287 w_k10_MSFT_s030
                                 + 1.02524 w_k10_PG_s030 >= 1
 z_k10_s0301_constraint:         var_10 + z_k10_s0301 + 1.29356 w_k10_ADM_s030
                                 + 1.57088 w_k10_AXP_s030
                                 + 1.72617 w_k10_DIS_s030
                                 + 1.5498 w_k10_FDX_s030
                                 + 1.17353 w_k10_IBM_s030
                                 + 2.18859 w_k10_INTC_s030
                                 + 1.36547 w_k10_KO_s030
                                 + 1.04703 w_k10_MAT_s030
                                 + 1.52823 w_k10_MSFT_s030
                                 + 1.27429 w_k10_PG_s030 >= 1
 z_k10_s0302_constraint:         var_10 + z_k10_s0302 + 0.933416 w_k10_ADM_s030
                                 + 0.675497 w_k10_AXP_s030
                                 + 0.62351 w_k10_DIS_s030
                                 + 0.775664 w_k10_FDX_s030
                                 + 0.722792 w_k10_IBM_s030
                                 + 0.776204 w_k10_INTC_s030
                                 + 0.784914 w_k10_KO_s030
                                 + 0.85278 w_k10_MAT_s030
                                 + 0.697542 w_k10_MSFT_s030
                                 + 0.849508 w_k10_PG_s030 >= 1
 z_k10_s0303_constraint:         var_10 + z_k10_s0303 + 0.789818 w_k10_ADM_s030
                                 + 0.901969 w_k10_AXP_s030
                                 + 0.76703 w_k10_DIS_s030
                                 + 0.874746 w_k10_FDX_s030
                                 + 0.926809 w_k10_IBM_s030
                                 + 1.13257 w_k10_INTC_s030
                                 + 0.928152 w_k10_KO_s030
                                 + 0.969597 w_k10_MAT_s030
                                 + 0.893296 w_k10_MSFT_s030
                                 + 0.902808 w_k10_PG_s030 >= 1
 z_k10_s0304_constraint:         var_10 + z_k10_s0304 + 1.05843 w_k10_ADM_s030
                                 + 0.933803 w_k10_AXP_s030
                                 + 1.29623 w_k10_DIS_s030
                                 + 1.14782 w_k10_FDX_s030
                                 + 1.22671 w_k10_IBM_s030
                                 + 1.33472 w_k10_INTC_s030
                                 + 1.22523 w_k10_KO_s030
                                 + 1.16011 w_k10_MAT_s030
                                 + 1.09839 w_k10_MSFT_s030
                                 + 1.23962 w_k10_PG_s030 >= 1
 z_k10_s0305_constraint:         var_10 + z_k10_s0305 + 0.743526 w_k10_ADM_s030
                                 + 0.782558 w_k10_AXP_s030
                                 + 0.599009 w_k10_DIS_s030
                                 + 0.711418 w_k10_FDX_s030
                                 + 0.962228 w_k10_IBM_s030
                                 + 0.994479 w_k10_INTC_s030
                                 + 0.73742 w_k10_KO_s030
                                 + 0.95926 w_k10_MAT_s030
                                 + 0.751323 w_k10_MSFT_s030
                                 + 0.814591 w_k10_PG_s030 >= 1
 z_k10_s0306_constraint:         var_10 + z_k10_s0306 + 1.04269 w_k10_ADM_s030
                                 + 0.959934 w_k10_AXP_s030
                                 + 1.0107 w_k10_DIS_s030
                                 + 1.02343 w_k10_FDX_s030
                                 + 0.976803 w_k10_IBM_s030
                                 + 1.42667 w_k10_INTC_s030
                                 + 0.946375 w_k10_KO_s030
                                 + 1.0383 w_k10_MAT_s030
                                 + 1.06106 w_k10_MSFT_s030
                                 + 1.0266 w_k10_PG_s030 >= 1
 z_k10_s0307_constraint:         var_10 + z_k10_s0307 + 1.12527 w_k10_ADM_s030
                                 + 0.960875 w_k10_AXP_s030
                                 + 1.01232 w_k10_DIS_s030
                                 + 0.993647 w_k10_FDX_s030
                                 + 1.38916 w_k10_IBM_s030
                                 + 1.16128 w_k10_INTC_s030
                                 + 1.1725 w_k10_KO_s030
                                 + 1.59413 w_k10_MAT_s030
                                 + 1.08511 w_k10_MSFT_s030
                                 + 0.999287 w_k10_PG_s030 >= 1
 z_k10_s0310_constraint:         var_10 + z_k10_s0310 + 0.96323 w_k10_ADM_s031
                                 + 1.15635 w_k10_AXP_s031
                                 + 0.83075 w_k10_DIS_s031
                                 + 1.02115 w_k10_FDX_s031
                                 + 0.978719 w_k10_IBM_s031
                                 + 1.00092 w_k10_INTC_s031
                                 + 0.890554 w_k10_KO_s031
                                 + 1.25294 w_k10_MAT_s031
                                 + 1.09558 w_k10_MSFT_s031
                                 + 0.885989 w_k10_PG_s031 >= 1
 z_k10_s0311_constraint:         var_10 + z_k10_s0311 + 0.663053 w_k10_ADM_s031
                                 + 0.96744 w_k10_AXP_s031
                                 + 0.768602 w_k10_DIS_s031
                                 + 0.923955 w_k10_FDX_s031
                                 + 1.46285 w_k10_IBM_s031
                                 + 1.19098 w_k10_INTC_s031
                                 + 0.97998 w_k10_KO_s031
                                 + 1.62525 w_k10_MAT_s031
                                 + 1.03625 w_k10_MSFT_s031
                                 + 0.807319 w_k10_PG_s031 >= 1
 z_k10_s0312_constraint:         var_10 + z_k10_s0312 + 0.882961 w_k10_ADM_s031
                                 + 0.868234 w_k10_AXP_s031
                                 + 0.638311 w_k10_DIS_s031
                                 + 0.820879 w_k10_FDX_s031
                                 + 0.850346 w_k10_IBM_s031
                                 + 0.864228 w_k10_INTC_s031
                                 + 0.762646 w_k10_KO_s031
                                 + 0.766483 w_k10_MAT_s031
                                 + 0.806656 w_k10_MSFT_s031
                                 + 0.760543 w_k10_PG_s031 >= 1
 z_k10_s0313_constraint:         var_10 + z_k10_s0313 + 1.17982 w_k10_ADM_s031
                                 + 1.26679 w_k10_AXP_s031
                                 + 1.06347 w_k10_DIS_s031
                                 + 1.16226 w_k10_FDX_s031
                                 + 1.18792 w_k10_IBM_s031
                                 + 1.22011 w_k10_INTC_s031
                                 + 1.21421 w_k10_KO_s031
                                 + 1.3613 w_k10_MAT_s031
                                 + 1.28728 w_k10_MSFT_s031
                                 + 1.15762 w_k10_PG_s031 >= 1
 z_k10_s0314_constraint:         var_10 + z_k10_s0314 + 1.38825 w_k10_ADM_s031
                                 + 1.38933 w_k10_AXP_s031
                                 + 1.36578 w_k10_DIS_s031
                                 + 1.46364 w_k10_FDX_s031
                                 + 1.30914 w_k10_IBM_s031
                                 + 1.40409 w_k10_INTC_s031
                                 + 1.21904 w_k10_KO_s031
                                 + 1.33448 w_k10_MAT_s031
                                 + 1.47746 w_k10_MSFT_s031
                                 + 1.27295 w_k10_PG_s031 >= 1
 z_k10_s0315_constraint:         var_10 + z_k10_s0315 + 1.3529 w_k10_ADM_s031
                                 + 0.782338 w_k10_AXP_s031
                                 + 1.17446 w_k10_DIS_s031
                                 + 1.10797 w_k10_FDX_s031
                                 + 0.951028 w_k10_IBM_s031
                                 + 0.841971 w_k10_INTC_s031
                                 + 1.11653 w_k10_KO_s031
                                 + 1.18582 w_k10_MAT_s031
                                 + 1.15324 w_k10_MSFT_s031
                                 + 1.05769 w_k10_PG_s031 >= 1
 z_k10_s0316_constraint:         var_10 + z_k10_s0316 + 1.20539 w_k10_ADM_s031
                                 + 1.15265 w_k10_AXP_s031
                                 + 1.00086 w_k10_DIS_s031
                                 + 1.05271 w_k10_FDX_s031
                                 + 1.12056 w_k10_IBM_s031
                                 + 1.03308 w_k10_INTC_s031
                                 + 0.890806 w_k10_KO_s031
                                 + 0.917617 w_k10_MAT_s031
                                 + 0.941263 w_k10_MSFT_s031
                                 + 0.886721 w_k10_PG_s031 >= 1
 z_k10_s0317_constraint:         var_10 + z_k10_s0317 + 1.04015 w_k10_ADM_s031
                                 + 1.06576 w_k10_AXP_s031
                                 + 0.866732 w_k10_DIS_s031
                                 + 1.08553 w_k10_FDX_s031
                                 + 0.683447 w_k10_IBM_s031
                                 + 1.26076 w_k10_INTC_s031
                                 + 0.73536 w_k10_KO_s031
                                 + 0.563829 w_k10_MAT_s031
                                 + 1.15774 w_k10_MSFT_s031
                                 + 0.910205 w_k10_PG_s031 >= 1
 z_k10_s0320_constraint:         var_10 + z_k10_s0320 + 1.00585 w_k10_ADM_s032
                                 + 1.15389 w_k10_AXP_s032
                                 + 1.53189 w_k10_DIS_s032
                                 + 1.21509 w_k10_FDX_s032
                                 + 1.20164 w_k10_IBM_s032
                                 + 1.101 w_k10_INTC_s032
                                 + 1.36687 w_k10_KO_s032
                                 + 1.44733 w_k10_MAT_s032
                                 + 1.30381 w_k10_MSFT_s032
                                 + 1.27493 w_k10_PG_s032 >= 1
 z_k10_s0321_constraint:         var_10 + z_k10_s0321 + 0.963169 w_k10_ADM_s032
                                 + 1.02202 w_k10_AXP_s032
                                 + 1.36916 w_k10_DIS_s032
                                 + 0.962347 w_k10_FDX_s032
                                 + 1.63049 w_k10_IBM_s032
                                 + 1.10088 w_k10_INTC_s032
                                 + 1.28308 w_k10_KO_s032
                                 + 1.81661 w_k10_MAT_s032
                                 + 0.986038 w_k10_MSFT_s032
                                 + 1.24839 w_k10_PG_s032 >= 1
 z_k10_s0322_constraint:         var_10 + z_k10_s0322 + 0.932402 w_k10_ADM_s032
                                 + 1.12352 w_k10_AXP_s032
                                 + 1.01306 w_k10_DIS_s032
                                 + 0.955141 w_k10_FDX_s032
                                 + 1.2335 w_k10_IBM_s032
                                 + 0.892607 w_k10_INTC_s032
                                 + 0.960166 w_k10_KO_s032
                                 + 1.67597 w_k10_MAT_s032
                                 + 0.895864 w_k10_MSFT_s032
                                 + 1.04551 w_k10_PG_s032 >= 1
 z_k10_s0323_constraint:         var_10 + z_k10_s0323 + 0.706319 w_k10_ADM_s032
                                 + 0.609077 w_k10_AXP_s032
                                 + 0.617323 w_k10_DIS_s032
                                 + 0.679738 w_k10_FDX_s032
                                 + 0.859067 w_k10_IBM_s032
                                 + 0.813897 w_k10_INTC_s032
                                 + 0.815614 w_k10_KO_s032
                                 + 0.89829 w_k10_MAT_s032
                                 + 0.79198 w_k10_MSFT_s032
                                 + 0.869519 w_k10_PG_s032 >= 1
 z_k10_s0324_constraint:         var_10 + z_k10_s0324 + 1.02102 w_k10_ADM_s032
                                 + 0.917627 w_k10_AXP_s032
                                 + 1.27178 w_k10_DIS_s032
                                 + 1.10502 w_k10_FDX_s032
                                 + 0.928097 w_k10_IBM_s032
                                 + 1.04265 w_k10_INTC_s032
                                 + 1.06249 w_k10_KO_s032
                                 + 1.1431 w_k10_MAT_s032
                                 + 1.23065 w_k10_MSFT_s032
                                 + 1.14943 w_k10_PG_s032 >= 1
 z_k10_s0325_constraint:         var_10 + z_k10_s0325 + 1.52805 w_k10_ADM_s032
                                 + 1.3533 w_k10_AXP_s032
                                 + 1.60888 w_k10_DIS_s032
                                 + 1.42205 w_k10_FDX_s032
                                 + 1.07909 w_k10_IBM_s032
                                 + 1.46685 w_k10_INTC_s032
                                 + 1.3672 w_k10_KO_s032
                                 + 0.966948 w_k10_MAT_s032
                                 + 1.51303 w_k10_MSFT_s032
                                 + 1.32456 w_k10_PG_s032 >= 1
 z_k10_s0326_constraint:         var_10 + z_k10_s0326 + 0.985474 w_k10_ADM_s032
                                 + 1.02509 w_k10_AXP_s032
                                 + 0.906697 w_k10_DIS_s032
                                 + 0.908515 w_k10_FDX_s032
                                 + 0.864752 w_k10_IBM_s032
                                 + 0.913145 w_k10_INTC_s032
                                 + 0.872009 w_k10_KO_s032
                                 + 0.91653 w_k10_MAT_s032
                                 + 0.913763 w_k10_MSFT_s032
                                 + 1.03616 w_k10_PG_s032 >= 1
 z_k10_s0327_constraint:         var_10 + z_k10_s0327 + 1.18989 w_k10_ADM_s032
                                 + 0.836958 w_k10_AXP_s032
                                 + 0.955251 w_k10_DIS_s032
                                 + 1.16655 w_k10_FDX_s032
                                 + 0.698288 w_k10_IBM_s032
                                 + 0.989292 w_k10_INTC_s032
                                 + 0.874042 w_k10_KO_s032
                                 + 0.788107 w_k10_MAT_s032
                                 + 0.98598 w_k10_MSFT_s032
                                 + 1.06331 w_k10_PG_s032 >= 1
 z_k10_s0330_constraint:         var_10 + z_k10_s0330 + 0.471359 w_k10_ADM_s033
                                 + 0.554743 w_k10_AXP_s033
                                 + 0.53644 w_k10_DIS_s033
                                 + 0.63572 w_k10_FDX_s033
                                 + 0.889806 w_k10_IBM_s033
                                 + 0.613949 w_k10_INTC_s033
                                 + 0.664367 w_k10_KO_s033
                                 + 0.985011 w_k10_MAT_s033
                                 + 0.577694 w_k10_MSFT_s033
                                 + 0.712346 w_k10_PG_s033 >= 1
 z_k10_s0331_constraint:         var_10 + z_k10_s0331 + 1.37896 w_k10_ADM_s033
                                 + 1.37747 w_k10_AXP_s033
                                 + 1.6717 w_k10_DIS_s033
                                 + 1.43387 w_k10_FDX_s033
                                 + 1.0561 w_k10_IBM_s033
                                 + 1.11768 w_k10_INTC_s033
                                 + 1.39921 w_k10_KO_s033
                                 + 1.16987 w_k10_MAT_s033
                                 + 1.41828 w_k10_MSFT_s033
                                 + 1.25153 w_k10_PG_s033 >= 1
 z_k10_s0332_constraint:         var_10 + z_k10_s0332 + 0.88239 w_k10_ADM_s033
                                 + 0.820867 w_k10_AXP_s033
                                 + 0.628957 w_k10_DIS_s033
                                 + 0.831436 w_k10_FDX_s033
                                 + 0.794119 w_k10_IBM_s033
                                 + 1.06461 w_k10_INTC_s033
                                 + 0.71972 w_k10_KO_s033
                                 + 0.582887 w_k10_MAT_s033
                                 + 0.742811 w_k10_MSFT_s033
                                 + 0.748169 w_k10_PG_s033 >= 1
 z_k10_s0333_constraint:         var_10 + z_k10_s0333 + 0.658872 w_k10_ADM_s033
                                 + 0.688476 w_k10_AXP_s033
                                 + 1.06068 w_k10_DIS_s033
                                 + 0.989174 w_k10_FDX_s033
                                 + 0.850838 w_k10_IBM_s033
                                 + 0.940133 w_k10_INTC_s033
                                 + 1.00934 w_k10_KO_s033
                                 + 0.812553 w_k10_MAT_s033
                                 + 1.10388 w_k10_MSFT_s033
                                 + 0.886142 w_k10_PG_s033 >= 1
 z_k10_s0334_constraint:         var_10 + z_k10_s0334 + 0.812198 w_k10_ADM_s033
                                 + 0.707695 w_k10_AXP_s033
                                 + 0.902185 w_k10_DIS_s033
                                 + 0.913125 w_k10_FDX_s033
                                 + 0.746393 w_k10_IBM_s033
                                 + 0.93906 w_k10_INTC_s033
                                 + 0.907252 w_k10_KO_s033
                                 + 0.498729 w_k10_MAT_s033
                                 + 0.999011 w_k10_MSFT_s033
                                 + 0.768639 w_k10_PG_s033 >= 1
 z_k10_s0335_constraint:         var_10 + z_k10_s0335 + 1.13318 w_k10_ADM_s033
                                 + 1.03926 w_k10_AXP_s033
                                 + 1.09188 w_k10_DIS_s033
                                 + 1.14678 w_k10_FDX_s033
                                 + 1.00752 w_k10_IBM_s033
                                 + 0.975459 w_k10_INTC_s033
                                 + 0.989517 w_k10_KO_s033
                                 + 1.43092 w_k10_MAT_s033
                                 + 1.19519 w_k10_MSFT_s033
                                 + 1.10231 w_k10_PG_s033 >= 1
 z_k10_s0336_constraint:         var_10 + z_k10_s0336 + 0.970298 w_k10_ADM_s033
                                 + 1.02304 w_k10_AXP_s033
                                 + 0.863784 w_k10_DIS_s033
                                 + 0.933802 w_k10_FDX_s033
                                 + 1.46451 w_k10_IBM_s033
                                 + 0.775228 w_k10_INTC_s033
                                 + 1.09741 w_k10_KO_s033
                                 + 1.47802 w_k10_MAT_s033
                                 + 0.868556 w_k10_MSFT_s033
                                 + 0.994333 w_k10_PG_s033 >= 1
 z_k10_s0337_constraint:         var_10 + z_k10_s0337 + 0.879443 w_k10_ADM_s033
                                 + 0.919359 w_k10_AXP_s033
                                 + 0.843391 w_k10_DIS_s033
                                 + 0.980228 w_k10_FDX_s033
                                 + 1.07021 w_k10_IBM_s033
                                 + 0.882256 w_k10_INTC_s033
                                 + 0.978071 w_k10_KO_s033
                                 + 1.26307 w_k10_MAT_s033
                                 + 0.936067 w_k10_MSFT_s033
                                 + 0.89417 w_k10_PG_s033 >= 1
 z_k10_s0340_constraint:         var_10 + z_k10_s0340 + 0.7938 w_k10_ADM_s034
                                 + 0.776542 w_k10_AXP_s034
                                 + 0.607684 w_k10_DIS_s034
                                 + 0.815414 w_k10_FDX_s034
                                 + 0.808072 w_k10_IBM_s034
                                 + 0.844094 w_k10_INTC_s034
                                 + 0.737126 w_k10_KO_s034
                                 + 0.787123 w_k10_MAT_s034
                                 + 0.736781 w_k10_MSFT_s034
                                 + 0.845146 w_k10_PG_s034 >= 1
 z_k10_s0341_constraint:         var_10 + z_k10_s0341 + 1.07404 w_k10_ADM_s034
                                 + 1.16609 w_k10_AXP_s034
                                 + 1.49384 w_k10_DIS_s034
                                 + 0.956155 w_k10_FDX_s034
                                 + 1.71815 w_k10_IBM_s034
                                 + 1.01239 w_k10_INTC_s034
                                 + 1.83003 w_k10_KO_s034
                                 + 1.94211 w_k10_MAT_s034
                                 + 1.0927 w_k10_MSFT_s034
                                 + 1.09257 w_k10_PG_s034 >= 1
 z_k10_s0342_constraint:         var_10 + z_k10_s0342 + 1.06421 w_k10_ADM_s034
                                 + 1.24583 w_k10_AXP_s034
                                 + 0.987465 w_k10_DIS_s034
                                 + 1.06894 w_k10_FDX_s034
                                 + 1.20364 w_k10_IBM_s034
                                 + 0.95052 w_k10_INTC_s034
                                 + 1.06643 w_k10_KO_s034
                                 + 1.64187 w_k10_MAT_s034
                                 + 0.885813 w_k10_MSFT_s034
                                 + 0.973066 w_k10_PG_s034 >= 1
 z_k10_s0343_constraint:         var_10 + z_k10_s0343 + 1.06651 w_k10_ADM_s034
                                 + 1.67109 w_k10_AXP_s034
                                 + 1.50457 w_k10_DIS_s034
                                 + 1.50911 w_k10_FDX_s034
                                 + 0.941205 w_k10_IBM_s034
                                 + 1.21737 w_k10_INTC_s034
                                 + 1.09054 w_k10_KO_s034
                                 + 1.35909 w_k10_MAT_s034
                                 + 1.16899 w_k10_MSFT_s034
                                 + 1.15991 w_k10_PG_s034 >= 1
 z_k10_s0344_constraint:         var_10 + z_k10_s0344 + 1.0159 w_k10_ADM_s034
                                 + 0.942805 w_k10_AXP_s034
                                 + 1.02093 w_k10_DIS_s034
                                 + 1.07998 w_k10_FDX_s034
                                 + 0.710051 w_k10_IBM_s034
                                 + 0.931461 w_k10_INTC_s034
                                 + 0.859672 w_k10_KO_s034
                                 + 0.755634 w_k10_MAT_s034
                                 + 0.887578 w_k10_MSFT_s034
                                 + 0.913907 w_k10_PG_s034 >= 1
 z_k10_s0345_constraint:         var_10 + z_k10_s0345 + 1.0008 w_k10_ADM_s034
                                 + 0.827781 w_k10_AXP_s034
                                 + 0.881377 w_k10_DIS_s034
                                 + 0.896966 w_k10_FDX_s034
                                 + 1.14192 w_k10_IBM_s034
                                 + 0.884724 w_k10_INTC_s034
                                 + 1.0179 w_k10_KO_s034
                                 + 0.99202 w_k10_MAT_s034
                                 + 0.905232 w_k10_MSFT_s034
                                 + 0.809405 w_k10_PG_s034 >= 1
 z_k10_s0346_constraint:         var_10 + z_k10_s0346 + 0.830454 w_k10_ADM_s034
                                 + 0.792637 w_k10_AXP_s034
                                 + 0.866532 w_k10_DIS_s034
                                 + 0.927264 w_k10_FDX_s034
                                 + 0.822514 w_k10_IBM_s034
                                 + 0.785066 w_k10_INTC_s034
                                 + 0.786901 w_k10_KO_s034
                                 + 0.767022 w_k10_MAT_s034
                                 + 0.781794 w_k10_MSFT_s034
                                 + 0.820305 w_k10_PG_s034 >= 1
 z_k10_s0347_constraint:         var_10 + z_k10_s0347 + 1.0654 w_k10_ADM_s034
                                 + 0.985577 w_k10_AXP_s034
                                 + 1.41454 w_k10_DIS_s034
                                 + 1.21655 w_k10_FDX_s034
                                 + 1.12473 w_k10_IBM_s034
                                 + 0.942493 w_k10_INTC_s034
                                 + 1.25462 w_k10_KO_s034
                                 + 1.43697 w_k10_MAT_s034
                                 + 1.2227 w_k10_MSFT_s034
                                 + 1.10984 w_k10_PG_s034 >= 1
 z_k10_s0350_constraint:         var_10 + z_k10_s0350 + 1.0729 w_k10_ADM_s035
                                 + 0.985418 w_k10_AXP_s035
                                 + 0.750875 w_k10_DIS_s035
                                 + 0.945281 w_k10_FDX_s035
                                 + 0.866271 w_k10_IBM_s035
                                 + 0.86965 w_k10_INTC_s035
                                 + 0.650164 w_k10_KO_s035
                                 + 0.681739 w_k10_MAT_s035
                                 + 0.850825 w_k10_MSFT_s035
                                 + 0.890378 w_k10_PG_s035 >= 1
 z_k10_s0351_constraint:         var_10 + z_k10_s0351 + 1.09071 w_k10_ADM_s035
                                 + 1.419 w_k10_AXP_s035
                                 + 1.15849 w_k10_DIS_s035
                                 + 0.989027 w_k10_FDX_s035
                                 + 1.51757 w_k10_IBM_s035
                                 + 0.885651 w_k10_INTC_s035
                                 + 1.00788 w_k10_KO_s035
                                 + 1.00839 w_k10_MAT_s035
                                 + 0.927453 w_k10_MSFT_s035
                                 + 1.13087 w_k10_PG_s035 >= 1
 z_k10_s0352_constraint:         var_10 + z_k10_s0352 + 1.03925 w_k10_ADM_s035
                                 + 1.19404 w_k10_AXP_s035
                                 + 1.09655 w_k10_DIS_s035
                                 + 1.07181 w_k10_FDX_s035
                                 + 1.11709 w_k10_IBM_s035
                                 + 0.943919 w_k10_INTC_s035
                                 + 1.06304 w_k10_KO_s035
                                 + 0.979142 w_k10_MAT_s035
                                 + 1.11677 w_k10_MSFT_s035
                                 + 1.25898 w_k10_PG_s035 >= 1
 z_k10_s0353_constraint:         var_10 + z_k10_s0353 + 0.779757 w_k10_ADM_s035
                                 + 0.995439 w_k10_AXP_s035
                                 + 0.887781 w_k10_DIS_s035
                                 + 0.835787 w_k10_FDX_s035
                                 + 0.977551 w_k10_IBM_s035
                                 + 1.02318 w_k10_INTC_s035
                                 + 0.937595 w_k10_KO_s035
                                 + 0.841757 w_k10_MAT_s035
                                 + 0.932299 w_k10_MSFT_s035
                                 + 0.993257 w_k10_PG_s035 >= 1
 z_k10_s0354_constraint:         var_10 + z_k10_s0354 + 1.3417 w_k10_ADM_s035
                                 + 1.23297 w_k10_AXP_s035
                                 + 1.28303 w_k10_DIS_s035
                                 + 1.28936 w_k10_FDX_s035
                                 + 0.870965 w_k10_IBM_s035
                                 + 1.31587 w_k10_INTC_s035
                                 + 1.05328 w_k10_KO_s035
                                 + 0.649856 w_k10_MAT_s035
                                 + 1.22521 w_k10_MSFT_s035
                                 + 1.19858 w_k10_PG_s035 >= 1
 z_k10_s0355_constraint:         var_10 + z_k10_s0355 + 1.52503 w_k10_ADM_s035
                                 + 1.09589 w_k10_AXP_s035
                                 + 1.6167 w_k10_DIS_s035
                                 + 1.18012 w_k10_FDX_s035
                                 + 1.23472 w_k10_IBM_s035
                                 + 1.00119 w_k10_INTC_s035
                                 + 1.10771 w_k10_KO_s035
                                 + 0.862097 w_k10_MAT_s035
                                 + 1.22617 w_k10_MSFT_s035
                                 + 1.21689 w_k10_PG_s035 >= 1
 z_k10_s0356_constraint:         var_10 + z_k10_s0356 + 1.27311 w_k10_ADM_s035
                                 + 0.760075 w_k10_AXP_s035
                                 + 0.92772 w_k10_DIS_s035
                                 + 0.959689 w_k10_FDX_s035
                                 + 0.644994 w_k10_IBM_s035
                                 + 0.833036 w_k10_INTC_s035
                                 + 0.749755 w_k10_KO_s035
                                 + 0.599817 w_k10_MAT_s035
                                 + 1.0175 w_k10_MSFT_s035
                                 + 0.958964 w_k10_PG_s035 >= 1
 z_k10_s0357_constraint:         var_10 + z_k10_s0357 + 0.854321 w_k10_ADM_s035
                                 + 0.857689 w_k10_AXP_s035
                                 + 0.608402 w_k10_DIS_s035
                                 + 0.765504 w_k10_FDX_s035
                                 + 0.711329 w_k10_IBM_s035
                                 + 0.793829 w_k10_INTC_s035
                                 + 0.687855 w_k10_KO_s035
                                 + 0.592935 w_k10_MAT_s035
                                 + 0.778531 w_k10_MSFT_s035
                                 + 0.872729 w_k10_PG_s035 >= 1
 z_k10_s0360_constraint:         var_10 + z_k10_s0360 + 1.15306 w_k10_ADM_s036
                                 + 1.08947 w_k10_AXP_s036
                                 + 1.14015 w_k10_DIS_s036
                                 + 1.04669 w_k10_FDX_s036
                                 + 0.984053 w_k10_IBM_s036
                                 + 0.961399 w_k10_INTC_s036
                                 + 1.16896 w_k10_KO_s036
                                 + 1.26958 w_k10_MAT_s036
                                 + 1.06464 w_k10_MSFT_s036
                                 + 1.08632 w_k10_PG_s036 >= 1
 z_k10_s0361_constraint:         var_10 + z_k10_s0361 + 1.24684 w_k10_ADM_s036
                                 + 1.13644 w_k10_AXP_s036
                                 + 1.08514 w_k10_DIS_s036
                                 + 1.14156 w_k10_FDX_s036
                                 + 0.785243 w_k10_IBM_s036
                                 + 1.01941 w_k10_INTC_s036
                                 + 0.94052 w_k10_KO_s036
                                 + 1.07503 w_k10_MAT_s036
                                 + 1.1201 w_k10_MSFT_s036
                                 + 1.0604 w_k10_PG_s036 >= 1
 z_k10_s0362_constraint:         var_10 + z_k10_s0362 + 0.906937 w_k10_ADM_s036
                                 + 0.814195 w_k10_AXP_s036
                                 + 0.745661 w_k10_DIS_s036
                                 + 0.875569 w_k10_FDX_s036
                                 + 0.638048 w_k10_IBM_s036
                                 + 0.896375 w_k10_INTC_s036
                                 + 0.760139 w_k10_KO_s036
                                 + 0.961723 w_k10_MAT_s036
                                 + 0.907471 w_k10_MSFT_s036
                                 + 0.884949 w_k10_PG_s036 >= 1
 z_k10_s0363_constraint:         var_10 + z_k10_s0363 + 0.812329 w_k10_ADM_s036
                                 + 0.742157 w_k10_AXP_s036
                                 + 0.558865 w_k10_DIS_s036
                                 + 0.70962 w_k10_FDX_s036
                                 + 0.933921 w_k10_IBM_s036
                                 + 0.897591 w_k10_INTC_s036
                                 + 0.770552 w_k10_KO_s036
                                 + 0.943767 w_k10_MAT_s036
                                 + 0.75047 w_k10_MSFT_s036
                                 + 0.905489 w_k10_PG_s036 >= 1
 z_k10_s0364_constraint:         var_10 + z_k10_s0364 + 0.971739 w_k10_ADM_s036
                                 + 1.28622 w_k10_AXP_s036 + 1.4 w_k10_DIS_s036
                                 + 1.29239 w_k10_FDX_s036
                                 + 0.993612 w_k10_IBM_s036
                                 + 1.22171 w_k10_INTC_s036
                                 + 1.07083 w_k10_KO_s036
                                 + 1.30568 w_k10_MAT_s036
                                 + 1.32019 w_k10_MSFT_s036
                                 + 1.20287 w_k10_PG_s036 >= 1
 z_k10_s0365_constraint:         var_10 + z_k10_s0365 + 0.932959 w_k10_ADM_s036
                                 + 1.27657 w_k10_AXP_s036
                                 + 1.2928 w_k10_DIS_s036
                                 + 1.07861 w_k10_FDX_s036
                                 + 1.33154 w_k10_IBM_s036
                                 + 1.16439 w_k10_INTC_s036
                                 + 1.30483 w_k10_KO_s036
                                 + 1.35015 w_k10_MAT_s036
                                 + 1.01745 w_k10_MSFT_s036
                                 + 1.28441 w_k10_PG_s036 >= 1
 z_k10_s0366_constraint:         var_10 + z_k10_s0366 + 0.885934 w_k10_ADM_s036
                                 + 0.830871 w_k10_AXP_s036
                                 + 0.755413 w_k10_DIS_s036
                                 + 0.88037 w_k10_FDX_s036
                                 + 0.89755 w_k10_IBM_s036
                                 + 0.89886 w_k10_INTC_s036
                                 + 0.736082 w_k10_KO_s036
                                 + 0.829966 w_k10_MAT_s036
                                 + 0.867113 w_k10_MSFT_s036
                                 + 0.899458 w_k10_PG_s036 >= 1
 z_k10_s0367_constraint:         var_10 + z_k10_s0367 + 0.991018 w_k10_ADM_s036
                                 + 1.22001 w_k10_AXP_s036
                                 + 1.05529 w_k10_DIS_s036
                                 + 0.928234 w_k10_FDX_s036
                                 + 1.75914 w_k10_IBM_s036
                                 + 0.967273 w_k10_INTC_s036
                                 + 1.19172 w_k10_KO_s036 + 2.034 w_k10_MAT_s036
                                 + 0.931274 w_k10_MSFT_s036
                                 + 1.07954 w_k10_PG_s036 >= 1
 z_k10_s0370_constraint:         var_10 + z_k10_s0370 + 1.38967 w_k10_ADM_s037
                                 + 1.46195 w_k10_AXP_s037
                                 + 1.60604 w_k10_DIS_s037
                                 + 1.31216 w_k10_FDX_s037
                                 + 1.54399 w_k10_IBM_s037
                                 + 1.19896 w_k10_INTC_s037
                                 + 1.59429 w_k10_KO_s037
                                 + 1.74542 w_k10_MAT_s037
                                 + 1.34495 w_k10_MSFT_s037
                                 + 1.20682 w_k10_PG_s037 >= 1
 z_k10_s0371_constraint:         var_10 + z_k10_s0371 + 0.847512 w_k10_ADM_s037
                                 + 0.691671 w_k10_AXP_s037
                                 + 0.655954 w_k10_DIS_s037
                                 + 0.846886 w_k10_FDX_s037
                                 + 0.68903 w_k10_IBM_s037
                                 + 0.731936 w_k10_INTC_s037
                                 + 0.691591 w_k10_KO_s037
                                 + 0.83037 w_k10_MAT_s037
                                 + 0.74546 w_k10_MSFT_s037
                                 + 0.707844 w_k10_PG_s037 >= 1
 z_k10_s0372_constraint:         var_10 + z_k10_s0372 + 0.825124 w_k10_ADM_s037
                                 + 0.809799 w_k10_AXP_s037
                                 + 0.700917 w_k10_DIS_s037
                                 + 0.748289 w_k10_FDX_s037
                                 + 1.32117 w_k10_IBM_s037
                                 + 0.729499 w_k10_INTC_s037
                                 + 0.908434 w_k10_KO_s037
                                 + 1.34701 w_k10_MAT_s037
                                 + 0.618279 w_k10_MSFT_s037
                                 + 0.892118 w_k10_PG_s037 >= 1
 z_k10_s0373_constraint:         var_10 + z_k10_s0373 + 1.19205 w_k10_ADM_s037
                                 + 1.15298 w_k10_AXP_s037
                                 + 1.10209 w_k10_DIS_s037
                                 + 1.02494 w_k10_FDX_s037
                                 + 1.47528 w_k10_IBM_s037
                                 + 0.961944 w_k10_INTC_s037
                                 + 1.24148 w_k10_KO_s037
                                 + 1.16159 w_k10_MAT_s037
                                 + 0.876111 w_k10_MSFT_s037
                                 + 1.00139 w_k10_PG_s037 >= 1
 z_k10_s0374_constraint:         var_10 + z_k10_s0374 + 1.47879 w_k10_ADM_s037
                                 + 1.52391 w_k10_AXP_s037
                                 + 1.87692 w_k10_DIS_s037
                                 + 1.49536 w_k10_FDX_s037
                                 + 1.13015 w_k10_IBM_s037
                                 + 1.09576 w_k10_INTC_s037
                                 + 1.17346 w_k10_KO_s037
                                 + 1.01931 w_k10_MAT_s037
                                 + 1.2532 w_k10_MSFT_s037
                                 + 1.04447 w_k10_PG_s037 >= 1
 z_k10_s0375_constraint:         var_10 + z_k10_s0375 + 0.986887 w_k10_ADM_s037
                                 + 1.01985 w_k10_AXP_s037
                                 + 0.966859 w_k10_DIS_s037
                                 + 1.06446 w_k10_FDX_s037
                                 + 0.879874 w_k10_IBM_s037
                                 + 0.906696 w_k10_INTC_s037
                                 + 0.904078 w_k10_KO_s037
                                 + 0.919355 w_k10_MAT_s037
                                 + 0.955757 w_k10_MSFT_s037
                                 + 0.894211 w_k10_PG_s037 >= 1
 z_k10_s0376_constraint:         var_10 + z_k10_s0376 + 0.960681 w_k10_ADM_s037
                                 + 0.970517 w_k10_AXP_s037
                                 + 0.878232 w_k10_DIS_s037
                                 + 0.890611 w_k10_FDX_s037
                                 + 1.12009 w_k10_IBM_s037
                                 + 0.954175 w_k10_INTC_s037
                                 + 0.995504 w_k10_KO_s037
                                 + 1.15402 w_k10_MAT_s037
                                 + 0.862166 w_k10_MSFT_s037
                                 + 0.991353 w_k10_PG_s037 >= 1
 z_k10_s0377_constraint:         var_10 + z_k10_s0377 + 1.05115 w_k10_ADM_s037
                                 + 0.826226 w_k10_AXP_s037
                                 + 0.556814 w_k10_DIS_s037
                                 + 0.756887 w_k10_FDX_s037
                                 + 0.95963 w_k10_IBM_s037
                                 + 0.658373 w_k10_INTC_s037
                                 + 0.788618 w_k10_KO_s037
                                 + 1.09809 w_k10_MAT_s037
                                 + 0.68024 w_k10_MSFT_s037
                                 + 0.854631 w_k10_PG_s037 >= 1
 z_k10_s1000_constraint:         var_10 + z_k10_s1000 + 1.04646 w_k10_ADM_s100
                                 + 0.978895 w_k10_AXP_s100
                                 + 0.656014 w_k10_DIS_s100
                                 + 1.00648 w_k10_FDX_s100
                                 + 0.641338 w_k10_IBM_s100
                                 + 0.981867 w_k10_INTC_s100
                                 + 0.592674 w_k10_KO_s100
                                 + 0.569081 w_k10_MAT_s100
                                 + 0.954187 w_k10_MSFT_s100
                                 + 0.842464 w_k10_PG_s100 >= 1
 z_k10_s1001_constraint:         var_10 + z_k10_s1001 + 0.987481 w_k10_ADM_s100
                                 + 0.620677 w_k10_AXP_s100
                                 + 0.752245 w_k10_DIS_s100
                                 + 0.77551 w_k10_FDX_s100
                                 + 1.02875 w_k10_IBM_s100
                                 + 0.757225 w_k10_INTC_s100
                                 + 0.849785 w_k10_KO_s100
                                 + 0.746363 w_k10_MAT_s100
                                 + 0.906553 w_k10_MSFT_s100
                                 + 0.953247 w_k10_PG_s100 >= 1
 z_k10_s1002_constraint:         var_10 + z_k10_s1002 + 0.87276 w_k10_ADM_s100
                                 + 0.864387 w_k10_AXP_s100
                                 + 0.904077 w_k10_DIS_s100
                                 + 0.931597 w_k10_FDX_s100
                                 + 0.963371 w_k10_IBM_s100
                                 + 0.982831 w_k10_INTC_s100
                                 + 0.821782 w_k10_KO_s100
                                 + 0.82324 w_k10_MAT_s100
                                 + 0.807187 w_k10_MSFT_s100
                                 + 0.894155 w_k10_PG_s100 >= 1
 z_k10_s1003_constraint:         var_10 + z_k10_s1003 + 1.07261 w_k10_ADM_s100
                                 + 1.01345 w_k10_AXP_s100
                                 + 1.08437 w_k10_DIS_s100
                                 + 1.1127 w_k10_FDX_s100
                                 + 0.82915 w_k10_IBM_s100
                                 + 1.0223 w_k10_INTC_s100
                                 + 0.86368 w_k10_KO_s100
                                 + 0.696672 w_k10_MAT_s100
                                 + 0.945698 w_k10_MSFT_s100
                                 + 1.02092 w_k10_PG_s100 >= 1
 z_k10_s1004_constraint:         var_10 + z_k10_s1004 + 1.3302 w_k10_ADM_s100
                                 + 1.24914 w_k10_AXP_s100
                                 + 1.67916 w_k10_DIS_s100
                                 + 1.43752 w_k10_FDX_s100
                                 + 0.925568 w_k10_IBM_s100
                                 + 1.17374 w_k10_INTC_s100
                                 + 1.12708 w_k10_KO_s100
                                 + 1.03416 w_k10_MAT_s100
                                 + 1.22017 w_k10_MSFT_s100
                                 + 1.19056 w_k10_PG_s100 >= 1
 z_k10_s1005_constraint:         var_10 + z_k10_s1005 + 1.01626 w_k10_ADM_s100
                                 + 0.623304 w_k10_AXP_s100
                                 + 0.536949 w_k10_DIS_s100
                                 + 0.768346 w_k10_FDX_s100
                                 + 0.868686 w_k10_IBM_s100
                                 + 0.677348 w_k10_INTC_s100
                                 + 0.623673 w_k10_KO_s100
                                 + 0.680383 w_k10_MAT_s100
                                 + 0.68487 w_k10_MSFT_s100
                                 + 0.760152 w_k10_PG_s100 >= 1
 z_k10_s1006_constraint:         var_10 + z_k10_s1006 + 1.23316 w_k10_ADM_s100
                                 + 0.775159 w_k10_AXP_s100
                                 + 0.703524 w_k10_DIS_s100
                                 + 0.868288 w_k10_FDX_s100
                                 + 0.917907 w_k10_IBM_s100
                                 + 0.701604 w_k10_INTC_s100
                                 + 0.840381 w_k10_KO_s100
                                 + 1.04946 w_k10_MAT_s100
                                 + 0.749887 w_k10_MSFT_s100
                                 + 0.89985 w_k10_PG_s100 >= 1
 z_k10_s1007_constraint:         var_10 + z_k10_s1007 + 1.47614 w_k10_ADM_s100
                                 + 1.2686 w_k10_AXP_s100
                                 + 1.2845 w_k10_DIS_s100
                                 + 1.09234 w_k10_FDX_s100
                                 + 1.51147 w_k10_IBM_s100
                                 + 0.981416 w_k10_INTC_s100
                                 + 1.04633 w_k10_KO_s100
                                 + 1.24316 w_k10_MAT_s100
                                 + 0.968942 w_k10_MSFT_s100
                                 + 1.1128 w_k10_PG_s100 >= 1
 z_k10_s1010_constraint:         var_10 + z_k10_s1010 + 1.01479 w_k10_ADM_s101
                                 + 0.806141 w_k10_AXP_s101
                                 + 0.839707 w_k10_DIS_s101
                                 + 0.788535 w_k10_FDX_s101
                                 + 1.17485 w_k10_IBM_s101
                                 + 0.966487 w_k10_INTC_s101
                                 + 1.02079 w_k10_KO_s101
                                 + 1.13201 w_k10_MAT_s101
                                 + 0.83844 w_k10_MSFT_s101
                                 + 0.885344 w_k10_PG_s101 >= 1
 z_k10_s1011_constraint:         var_10 + z_k10_s1011 + 0.985304 w_k10_ADM_s101
                                 + 0.610451 w_k10_AXP_s101
                                 + 0.705452 w_k10_DIS_s101
                                 + 0.804571 w_k10_FDX_s101
                                 + 0.686366 w_k10_IBM_s101
                                 + 0.897022 w_k10_INTC_s101
                                 + 0.668341 w_k10_KO_s101
                                 + 0.531128 w_k10_MAT_s101
                                 + 0.888797 w_k10_MSFT_s101
                                 + 0.759826 w_k10_PG_s101 >= 1
 z_k10_s1012_constraint:         var_10 + z_k10_s1012 + 1.56055 w_k10_ADM_s101
                                 + 1.474 w_k10_AXP_s101
                                 + 1.28814 w_k10_DIS_s101
                                 + 1.22349 w_k10_FDX_s101
                                 + 1.01348 w_k10_IBM_s101
                                 + 1.10533 w_k10_INTC_s101
                                 + 1.0225 w_k10_KO_s101
                                 + 0.910681 w_k10_MAT_s101
                                 + 1.07931 w_k10_MSFT_s101
                                 + 1.02333 w_k10_PG_s101 >= 1
 z_k10_s1013_constraint:         var_10 + z_k10_s1013 + 1.43499 w_k10_ADM_s101
                                 + 1.30791 w_k10_AXP_s101
                                 + 1.43109 w_k10_DIS_s101
                                 + 1.10847 w_k10_FDX_s101
                                 + 1.43775 w_k10_IBM_s101
                                 + 1.04199 w_k10_INTC_s101
                                 + 1.36607 w_k10_KO_s101
                                 + 1.39696 w_k10_MAT_s101
                                 + 1.16909 w_k10_MSFT_s101
                                 + 1.28008 w_k10_PG_s101 >= 1
 z_k10_s1014_constraint:         var_10 + z_k10_s1014 + 0.996985 w_k10_ADM_s101
                                 + 0.912855 w_k10_AXP_s101
                                 + 1.06583 w_k10_DIS_s101
                                 + 1.06173 w_k10_FDX_s101
                                 + 0.963754 w_k10_IBM_s101
                                 + 0.869886 w_k10_INTC_s101
                                 + 0.873978 w_k10_KO_s101
                                 + 0.903565 w_k10_MAT_s101
                                 + 0.935703 w_k10_MSFT_s101
                                 + 0.909359 w_k10_PG_s101 >= 1
 z_k10_s1015_constraint:         var_10 + z_k10_s1015 + 1.1219 w_k10_ADM_s101
                                 + 0.870705 w_k10_AXP_s101
                                 + 0.895489 w_k10_DIS_s101
                                 + 1.06209 w_k10_FDX_s101
                                 + 0.583405 w_k10_IBM_s101
                                 + 0.824218 w_k10_INTC_s101
                                 + 0.711046 w_k10_KO_s101
                                 + 0.592176 w_k10_MAT_s101
                                 + 0.799064 w_k10_MSFT_s101
                                 + 0.970268 w_k10_PG_s101 >= 1
 z_k10_s1016_constraint:         var_10 + z_k10_s1016 + 1.1927 w_k10_ADM_s101
                                 + 1.04699 w_k10_AXP_s101
                                 + 0.678835 w_k10_DIS_s101
                                 + 0.964696 w_k10_FDX_s101
                                 + 0.867695 w_k10_IBM_s101
                                 + 0.946153 w_k10_INTC_s101
                                 + 0.767548 w_k10_KO_s101
                                 + 0.846238 w_k10_MAT_s101
                                 + 0.887531 w_k10_MSFT_s101
                                 + 0.963515 w_k10_PG_s101 >= 1
 z_k10_s1017_constraint:         var_10 + z_k10_s1017 + 0.871607 w_k10_ADM_s101
                                 + 0.734865 w_k10_AXP_s101
                                 + 0.835583 w_k10_DIS_s101
                                 + 0.796172 w_k10_FDX_s101
                                 + 0.889268 w_k10_IBM_s101
                                 + 0.830905 w_k10_INTC_s101
                                 + 0.731209 w_k10_KO_s101
                                 + 0.778701 w_k10_MAT_s101
                                 + 0.817278 w_k10_MSFT_s101
                                 + 0.762119 w_k10_PG_s101 >= 1
 z_k10_s1020_constraint:         var_10 + z_k10_s1020 + 1.08643 w_k10_ADM_s102
                                 + 1.46207 w_k10_AXP_s102
                                 + 0.98699 w_k10_DIS_s102
                                 + 1.11023 w_k10_FDX_s102
                                 + 1.07613 w_k10_IBM_s102
                                 + 0.942779 w_k10_INTC_s102
                                 + 1.0057 w_k10_KO_s102
                                 + 1.24604 w_k10_MAT_s102
                                 + 1.03966 w_k10_MSFT_s102
                                 + 1.07732 w_k10_PG_s102 >= 1
 z_k10_s1021_constraint:         var_10 + z_k10_s1021 + 1.27184 w_k10_ADM_s102
                                 + 1.2113 w_k10_AXP_s102
                                 + 1.41572 w_k10_DIS_s102
                                 + 1.14737 w_k10_FDX_s102
                                 + 1.17492 w_k10_IBM_s102
                                 + 1.07436 w_k10_INTC_s102
                                 + 1.24187 w_k10_KO_s102
                                 + 1.36772 w_k10_MAT_s102
                                 + 1.19104 w_k10_MSFT_s102
                                 + 1.08945 w_k10_PG_s102 >= 1
 z_k10_s1022_constraint:         var_10 + z_k10_s1022 + 1.1886 w_k10_ADM_s102
                                 + 1.24452 w_k10_AXP_s102
                                 + 1.37154 w_k10_DIS_s102
                                 + 1.21649 w_k10_FDX_s102
                                 + 0.820269 w_k10_IBM_s102
                                 + 1.001 w_k10_INTC_s102
                                 + 1.00915 w_k10_KO_s102
                                 + 0.960231 w_k10_MAT_s102
                                 + 1.26523 w_k10_MSFT_s102
                                 + 1.10201 w_k10_PG_s102 >= 1
 z_k10_s1023_constraint:         var_10 + z_k10_s1023 + 1.2068 w_k10_ADM_s102
                                 + 0.858222 w_k10_AXP_s102
                                 + 0.882619 w_k10_DIS_s102
                                 + 0.875534 w_k10_FDX_s102
                                 + 1.12441 w_k10_IBM_s102
                                 + 0.932597 w_k10_INTC_s102
                                 + 1.0471 w_k10_KO_s102
                                 + 0.910978 w_k10_MAT_s102
                                 + 0.982338 w_k10_MSFT_s102
                                 + 0.841727 w_k10_PG_s102 >= 1
 z_k10_s1024_constraint:         var_10 + z_k10_s1024 + 0.974804 w_k10_ADM_s102
                                 + 1.07072 w_k10_AXP_s102
                                 + 0.986269 w_k10_DIS_s102
                                 + 0.881994 w_k10_FDX_s102
                                 + 1.35833 w_k10_IBM_s102
                                 + 0.905596 w_k10_INTC_s102
                                 + 0.985533 w_k10_KO_s102
                                 + 0.840191 w_k10_MAT_s102
                                 + 0.696875 w_k10_MSFT_s102
                                 + 0.915066 w_k10_PG_s102 >= 1
 z_k10_s1025_constraint:         var_10 + z_k10_s1025 + 0.982749 w_k10_ADM_s102
                                 + 1.34851 w_k10_AXP_s102
                                 + 1.12754 w_k10_DIS_s102
                                 + 1.1067 w_k10_FDX_s102
                                 + 1.59743 w_k10_IBM_s102
                                 + 1.01802 w_k10_INTC_s102
                                 + 1.12081 w_k10_KO_s102
                                 + 1.71252 w_k10_MAT_s102
                                 + 1.04038 w_k10_MSFT_s102
                                 + 0.9834 w_k10_PG_s102 >= 1
 z_k10_s1026_constraint:         var_10 + z_k10_s1026 + 0.963664 w_k10_ADM_s102
                                 + 0.800406 w_k10_AXP_s102
                                 + 1.00122 w_k10_DIS_s102
                                 + 0.923355 w_k10_FDX_s102
                                 + 0.940872 w_k10_IBM_s102
                                 + 0.939661 w_k10_INTC_s102
                                 + 1.09612 w_k10_KO_s102
                                 + 0.928529 w_k10_MAT_s102
                                 + 1.02428 w_k10_MSFT_s102
                                 + 0.948601 w_k10_PG_s102 >= 1
 z_k10_s1027_constraint:         var_10 + z_k10_s1027 + 0.770025 w_k10_ADM_s102
                                 + 0.800078 w_k10_AXP_s102
                                 + 0.729529 w_k10_DIS_s102
                                 + 0.824862 w_k10_FDX_s102
                                 + 0.770613 w_k10_IBM_s102
                                 + 0.809661 w_k10_INTC_s102
                                 + 0.787443 w_k10_KO_s102
                                 + 0.658164 w_k10_MAT_s102
                                 + 0.83454 w_k10_MSFT_s102
                                 + 0.836057 w_k10_PG_s102 >= 1
 z_k10_s1030_constraint:         var_10 + z_k10_s1030 + 1.41842 w_k10_ADM_s103
                                 + 1.42599 w_k10_AXP_s103
                                 + 1.53883 w_k10_DIS_s103
                                 + 1.48288 w_k10_FDX_s103
                                 + 1.0744 w_k10_IBM_s103
                                 + 1.09026 w_k10_INTC_s103
                                 + 1.24075 w_k10_KO_s103
                                 + 1.4193 w_k10_MAT_s103
                                 + 1.20519 w_k10_MSFT_s103
                                 + 1.32662 w_k10_PG_s103 >= 1
 z_k10_s1031_constraint:         var_10 + z_k10_s1031 + 0.695344 w_k10_ADM_s103
                                 + 0.88967 w_k10_AXP_s103
                                 + 1.0438 w_k10_DIS_s103
                                 + 1.03142 w_k10_FDX_s103
                                 + 0.677659 w_k10_IBM_s103
                                 + 1.11671 w_k10_INTC_s103
                                 + 0.806076 w_k10_KO_s103
                                 + 0.553253 w_k10_MAT_s103
                                 + 1.03717 w_k10_MSFT_s103
                                 + 0.899975 w_k10_PG_s103 >= 1
 z_k10_s1032_constraint:         var_10 + z_k10_s1032 + 0.917867 w_k10_ADM_s103
                                 + 0.85586 w_k10_AXP_s103
                                 + 1.05066 w_k10_DIS_s103
                                 + 1.0036 w_k10_FDX_s103
                                 + 0.853358 w_k10_IBM_s103
                                 + 0.834335 w_k10_INTC_s103
                                 + 0.911768 w_k10_KO_s103
                                 + 0.780055 w_k10_MAT_s103
                                 + 0.947934 w_k10_MSFT_s103
                                 + 1.15789 w_k10_PG_s103 >= 1
 z_k10_s1033_constraint:         var_10 + z_k10_s1033 + 0.680084 w_k10_ADM_s103
                                 + 0.775875 w_k10_AXP_s103
                                 + 0.666594 w_k10_DIS_s103
                                 + 0.648251 w_k10_FDX_s103
                                 + 1.15214 w_k10_IBM_s103
                                 + 0.811063 w_k10_INTC_s103
                                 + 0.873546 w_k10_KO_s103
                                 + 1.14428 w_k10_MAT_s103
                                 + 0.775787 w_k10_MSFT_s103
                                 + 0.829061 w_k10_PG_s103 >= 1
 z_k10_s1034_constraint:         var_10 + z_k10_s1034 + 0.610897 w_k10_ADM_s103
                                 + 0.71349 w_k10_AXP_s103
                                 + 0.62931 w_k10_DIS_s103
                                 + 0.628407 w_k10_FDX_s103
                                 + 0.832115 w_k10_IBM_s103
                                 + 0.837146 w_k10_INTC_s103
                                 + 0.746037 w_k10_KO_s103
                                 + 0.813229 w_k10_MAT_s103
                                 + 0.812159 w_k10_MSFT_s103
                                 + 0.84152 w_k10_PG_s103 >= 1
 z_k10_s1035_constraint:         var_10 + z_k10_s1035 + 0.834273 w_k10_ADM_s103
                                 + 0.818657 w_k10_AXP_s103
                                 + 0.9423 w_k10_DIS_s103
                                 + 0.81287 w_k10_FDX_s103
                                 + 1.10171 w_k10_IBM_s103
                                 + 0.81742 w_k10_INTC_s103
                                 + 0.949906 w_k10_KO_s103
                                 + 1.08909 w_k10_MAT_s103
                                 + 0.948008 w_k10_MSFT_s103
                                 + 1.03553 w_k10_PG_s103 >= 1
 z_k10_s1036_constraint:         var_10 + z_k10_s1036 + 0.740084 w_k10_ADM_s103
                                 + 0.814675 w_k10_AXP_s103
                                 + 0.909421 w_k10_DIS_s103
                                 + 0.810316 w_k10_FDX_s103
                                 + 1.37164 w_k10_IBM_s103
                                 + 1.11131 w_k10_INTC_s103
                                 + 1.17508 w_k10_KO_s103
                                 + 1.10063 w_k10_MAT_s103
                                 + 1.006 w_k10_MSFT_s103
                                 + 1.12652 w_k10_PG_s103 >= 1
 z_k10_s1037_constraint:         var_10 + z_k10_s1037 + 0.96744 w_k10_ADM_s103
                                 + 1.31639 w_k10_AXP_s103
                                 + 1.29465 w_k10_DIS_s103
                                 + 0.904486 w_k10_FDX_s103
                                 + 1.75503 w_k10_IBM_s103
                                 + 0.857841 w_k10_INTC_s103
                                 + 1.38751 w_k10_KO_s103
                                 + 2.1774 w_k10_MAT_s103
                                 + 0.965469 w_k10_MSFT_s103
                                 + 1.3653 w_k10_PG_s103 >= 1
 z_k10_s1040_constraint:         var_10 + z_k10_s1040 + 0.855762 w_k10_ADM_s104
                                 + 0.682467 w_k10_AXP_s104
                                 + 1.13898 w_k10_DIS_s104
                                 + 0.967032 w_k10_FDX_s104
                                 + 0.825402 w_k10_IBM_s104
                                 + 0.961262 w_k10_INTC_s104
                                 + 0.973161 w_k10_KO_s104
                                 + 0.820946 w_k10_MAT_s104
                                 + 1.0303 w_k10_MSFT_s104
                                 + 0.955094 w_k10_PG_s104 >= 1
 z_k10_s1041_constraint:         var_10 + z_k10_s1041 + 0.981612 w_k10_ADM_s104
                                 + 0.824215 w_k10_AXP_s104
                                 + 0.768236 w_k10_DIS_s104
                                 + 0.846352 w_k10_FDX_s104
                                 + 0.998148 w_k10_IBM_s104
                                 + 1.13535 w_k10_INTC_s104
                                 + 0.727185 w_k10_KO_s104
                                 + 0.818282 w_k10_MAT_s104
                                 + 0.896681 w_k10_MSFT_s104
                                 + 0.8893 w_k10_PG_s104 >= 1
 z_k10_s1042_constraint:         var_10 + z_k10_s1042 + 0.945772 w_k10_ADM_s104
                                 + 0.936197 w_k10_AXP_s104
                                 + 0.785673 w_k10_DIS_s104
                                 + 0.858788 w_k10_FDX_s104
                                 + 1.29354 w_k10_IBM_s104
                                 + 1.04451 w_k10_INTC_s104
                                 + 0.951388 w_k10_KO_s104
                                 + 1.43575 w_k10_MAT_s104
                                 + 0.979596 w_k10_MSFT_s104
                                 + 0.935011 w_k10_PG_s104 >= 1
 z_k10_s1043_constraint:         var_10 + z_k10_s1043 + 1.25379 w_k10_ADM_s104
                                 + 0.962934 w_k10_AXP_s104
                                 + 1.20069 w_k10_DIS_s104
                                 + 1.06998 w_k10_FDX_s104
                                 + 1.17623 w_k10_IBM_s104
                                 + 1.19959 w_k10_INTC_s104
                                 + 1.07868 w_k10_KO_s104
                                 + 1.12336 w_k10_MAT_s104
                                 + 1.18311 w_k10_MSFT_s104
                                 + 1.02234 w_k10_PG_s104 >= 1
 z_k10_s1044_constraint:         var_10 + z_k10_s1044 + 0.93761 w_k10_ADM_s104
                                 + 0.8917 w_k10_AXP_s104
                                 + 0.79532 w_k10_DIS_s104
                                 + 0.864221 w_k10_FDX_s104
                                 + 0.817151 w_k10_IBM_s104
                                 + 1.03164 w_k10_INTC_s104
                                 + 0.82306 w_k10_KO_s104
                                 + 1.00033 w_k10_MAT_s104
                                 + 0.832643 w_k10_MSFT_s104
                                 + 0.868531 w_k10_PG_s104 >= 1
 z_k10_s1045_constraint:         var_10 + z_k10_s1045 + 1.11876 w_k10_ADM_s104
                                 + 1.40722 w_k10_AXP_s104
                                 + 1.63276 w_k10_DIS_s104
                                 + 1.29018 w_k10_FDX_s104
                                 + 1.63057 w_k10_IBM_s104
                                 + 1.40811 w_k10_INTC_s104
                                 + 1.44836 w_k10_KO_s104
                                 + 1.85792 w_k10_MAT_s104
                                 + 1.37482 w_k10_MSFT_s104
                                 + 1.26808 w_k10_PG_s104 >= 1
 z_k10_s1046_constraint:         var_10 + z_k10_s1046 + 1.27569 w_k10_ADM_s104
                                 + 1.09187 w_k10_AXP_s104
                                 + 1.09651 w_k10_DIS_s104
                                 + 1.16433 w_k10_FDX_s104
                                 + 0.772944 w_k10_IBM_s104
                                 + 1.3268 w_k10_INTC_s104
                                 + 0.873839 w_k10_KO_s104
                                 + 0.660119 w_k10_MAT_s104
                                 + 1.08034 w_k10_MSFT_s104
                                 + 0.983405 w_k10_PG_s104 >= 1
 z_k10_s1047_constraint:         var_10 + z_k10_s1047 + 1.4708 w_k10_ADM_s104
                                 + 1.34006 w_k10_AXP_s104
                                 + 1.62995 w_k10_DIS_s104
                                 + 1.40298 w_k10_FDX_s104
                                 + 1.08198 w_k10_IBM_s104
                                 + 1.34217 w_k10_INTC_s104
                                 + 1.13448 w_k10_KO_s104
                                 + 1.16795 w_k10_MAT_s104
                                 + 1.40255 w_k10_MSFT_s104
                                 + 1.10788 w_k10_PG_s104 >= 1
 z_k10_s1050_constraint:         var_10 + z_k10_s1050 + 1.16026 w_k10_ADM_s105
                                 + 1.53767 w_k10_AXP_s105
                                 + 1.39609 w_k10_DIS_s105
                                 + 1.15676 w_k10_FDX_s105
                                 + 1.81699 w_k10_IBM_s105
                                 + 1.07395 w_k10_INTC_s105
                                 + 1.48345 w_k10_KO_s105
                                 + 2.33807 w_k10_MAT_s105
                                 + 1.18371 w_k10_MSFT_s105
                                 + 1.30845 w_k10_PG_s105 >= 1
 z_k10_s1051_constraint:         var_10 + z_k10_s1051 + 0.956893 w_k10_ADM_s105
                                 + 0.724402 w_k10_AXP_s105
                                 + 0.855845 w_k10_DIS_s105
                                 + 0.80174 w_k10_FDX_s105
                                 + 1.16076 w_k10_IBM_s105
                                 + 0.736368 w_k10_INTC_s105
                                 + 1.07439 w_k10_KO_s105
                                 + 1.10684 w_k10_MAT_s105
                                 + 0.886105 w_k10_MSFT_s105
                                 + 0.894508 w_k10_PG_s105 >= 1
 z_k10_s1052_constraint:         var_10 + z_k10_s1052 + 0.928435 w_k10_ADM_s105
                                 + 0.747454 w_k10_AXP_s105
                                 + 0.938588 w_k10_DIS_s105
                                 + 0.979329 w_k10_FDX_s105
                                 + 0.714824 w_k10_IBM_s105
                                 + 1.07172 w_k10_INTC_s105
                                 + 0.829485 w_k10_KO_s105
                                 + 0.516892 w_k10_MAT_s105
                                 + 0.982619 w_k10_MSFT_s105
                                 + 0.887521 w_k10_PG_s105 >= 1
 z_k10_s1053_constraint:         var_10 + z_k10_s1053 + 1.10861 w_k10_ADM_s105
                                 + 1.10596 w_k10_AXP_s105
                                 + 1.18436 w_k10_DIS_s105
                                 + 0.963694 w_k10_FDX_s105
                                 + 1.27076 w_k10_IBM_s105
                                 + 1.09881 w_k10_INTC_s105
                                 + 1.2473 w_k10_KO_s105
                                 + 1.33142 w_k10_MAT_s105
                                 + 1.09807 w_k10_MSFT_s105
                                 + 1.09928 w_k10_PG_s105 >= 1
 z_k10_s1054_constraint:         var_10 + z_k10_s1054 + 1.05344 w_k10_ADM_s105
                                 + 0.914692 w_k10_AXP_s105
                                 + 0.681203 w_k10_DIS_s105
                                 + 0.931677 w_k10_FDX_s105
                                 + 0.770173 w_k10_IBM_s105
                                 + 0.792226 w_k10_INTC_s105
                                 + 0.899471 w_k10_KO_s105
                                 + 0.90815 w_k10_MAT_s105
                                 + 0.990567 w_k10_MSFT_s105
                                 + 0.890959 w_k10_PG_s105 >= 1
 z_k10_s1055_constraint:         var_10 + z_k10_s1055 + 1.09769 w_k10_ADM_s105
                                 + 1.17852 w_k10_AXP_s105
                                 + 1.58605 w_k10_DIS_s105
                                 + 1.18609 w_k10_FDX_s105
                                 + 1.27439 w_k10_IBM_s105
                                 + 1.02685 w_k10_INTC_s105
                                 + 1.65467 w_k10_KO_s105
                                 + 1.88668 w_k10_MAT_s105
                                 + 1.29058 w_k10_MSFT_s105
                                 + 1.17998 w_k10_PG_s105 >= 1
 z_k10_s1056_constraint:         var_10 + z_k10_s1056 + 1.5582 w_k10_ADM_s105
                                 + 1.38746 w_k10_AXP_s105
                                 + 1.4274 w_k10_DIS_s105
                                 + 1.31929 w_k10_FDX_s105
                                 + 0.883802 w_k10_IBM_s105
                                 + 1.26678 w_k10_INTC_s105
                                 + 1.36088 w_k10_KO_s105
                                 + 0.973572 w_k10_MAT_s105
                                 + 1.38458 w_k10_MSFT_s105
                                 + 1.32362 w_k10_PG_s105 >= 1
 z_k10_s1057_constraint:         var_10 + z_k10_s1057 + 1.09894 w_k10_ADM_s105
                                 + 1.07662 w_k10_AXP_s105
                                 + 0.973962 w_k10_DIS_s105
                                 + 0.997473 w_k10_FDX_s105
                                 + 0.948198 w_k10_IBM_s105
                                 + 0.883058 w_k10_INTC_s105
                                 + 1.09665 w_k10_KO_s105
                                 + 1.15394 w_k10_MAT_s105
                                 + 1.04026 w_k10_MSFT_s105
                                 + 1.10844 w_k10_PG_s105 >= 1
 z_k10_s1060_constraint:         var_10 + z_k10_s1060 + 1.07945 w_k10_ADM_s106
                                 + 1.14773 w_k10_AXP_s106
                                 + 1.2304 w_k10_DIS_s106
                                 + 1.12838 w_k10_FDX_s106
                                 + 1.1602 w_k10_IBM_s106
                                 + 0.927448 w_k10_INTC_s106
                                 + 1.03906 w_k10_KO_s106
                                 + 1.15368 w_k10_MAT_s106
                                 + 0.943274 w_k10_MSFT_s106
                                 + 1.04162 w_k10_PG_s106 >= 1
 z_k10_s1061_constraint:         var_10 + z_k10_s1061 + 1.01994 w_k10_ADM_s106
                                 + 1.407 w_k10_AXP_s106
                                 + 1.42509 w_k10_DIS_s106
                                 + 1.24252 w_k10_FDX_s106
                                 + 1.44524 w_k10_IBM_s106
                                 + 1.49812 w_k10_INTC_s106
                                 + 1.12644 w_k10_KO_s106
                                 + 1.39426 w_k10_MAT_s106
                                 + 1.42436 w_k10_MSFT_s106
                                 + 1.27694 w_k10_PG_s106 >= 1
 z_k10_s1062_constraint:         var_10 + z_k10_s1062 + 0.741848 w_k10_ADM_s106
                                 + 1.07854 w_k10_AXP_s106
                                 + 0.955673 w_k10_DIS_s106
                                 + 0.948964 w_k10_FDX_s106
                                 + 1.26415 w_k10_IBM_s106
                                 + 1.12751 w_k10_INTC_s106
                                 + 1.00094 w_k10_KO_s106
                                 + 1.32835 w_k10_MAT_s106
                                 + 1.04778 w_k10_MSFT_s106
                                 + 1.04214 w_k10_PG_s106 >= 1
 z_k10_s1063_constraint:         var_10 + z_k10_s1063 + 0.799838 w_k10_ADM_s106
                                 + 0.835858 w_k10_AXP_s106
                                 + 1.12098 w_k10_DIS_s106
                                 + 0.989359 w_k10_FDX_s106
                                 + 0.975604 w_k10_IBM_s106
                                 + 1.14482 w_k10_INTC_s106
                                 + 1.0282 w_k10_KO_s106
                                 + 0.807073 w_k10_MAT_s106
                                 + 1.07617 w_k10_MSFT_s106
                                 + 1.10974 w_k10_PG_s106 >= 1
 z_k10_s1064_constraint:         var_10 + z_k10_s1064 + 0.693232 w_k10_ADM_s106
                                 + 0.825733 w_k10_AXP_s106
                                 + 0.907025 w_k10_DIS_s106
                                 + 0.807356 w_k10_FDX_s106
                                 + 1.2236 w_k10_IBM_s106
                                 + 0.832876 w_k10_INTC_s106
                                 + 0.90682 w_k10_KO_s106
                                 + 1.10904 w_k10_MAT_s106
                                 + 0.812727 w_k10_MSFT_s106
                                 + 1.03892 w_k10_PG_s106 >= 1
 z_k10_s1065_constraint:         var_10 + z_k10_s1065 + 0.92759 w_k10_ADM_s106
                                 + 1.02274 w_k10_AXP_s106
                                 + 0.934001 w_k10_DIS_s106
                                 + 1.06163 w_k10_FDX_s106
                                 + 0.721937 w_k10_IBM_s106
                                 + 1.08208 w_k10_INTC_s106
                                 + 0.80514 w_k10_KO_s106
                                 + 0.91753 w_k10_MAT_s106
                                 + 1.10866 w_k10_MSFT_s106
                                 + 1.04616 w_k10_PG_s106 >= 1
 z_k10_s1066_constraint:         var_10 + z_k10_s1066 + 0.798792 w_k10_ADM_s106
                                 + 1.34166 w_k10_AXP_s106
                                 + 1.30664 w_k10_DIS_s106
                                 + 1.27231 w_k10_FDX_s106
                                 + 0.962821 w_k10_IBM_s106
                                 + 1.46718 w_k10_INTC_s106
                                 + 1.04624 w_k10_KO_s106
                                 + 1.28548 w_k10_MAT_s106
                                 + 1.31386 w_k10_MSFT_s106
                                 + 1.19732 w_k10_PG_s106 >= 1
 z_k10_s1067_constraint:         var_10 + z_k10_s1067 + 0.675814 w_k10_ADM_s106
                                 + 0.68292 w_k10_AXP_s106
                                 + 0.708784 w_k10_DIS_s106
                                 + 0.836691 w_k10_FDX_s106
                                 + 0.772189 w_k10_IBM_s106
                                 + 0.936697 w_k10_INTC_s106
                                 + 0.732932 w_k10_KO_s106
                                 + 0.670973 w_k10_MAT_s106
                                 + 0.990026 w_k10_MSFT_s106
                                 + 0.884732 w_k10_PG_s106 >= 1
 z_k10_s1070_constraint:         var_10 + z_k10_s1070 + 0.91168 w_k10_ADM_s107
                                 + 0.626543 w_k10_AXP_s107
                                 + 0.518452 w_k10_DIS_s107
                                 + 0.740456 w_k10_FDX_s107
                                 + 0.659268 w_k10_IBM_s107
                                 + 0.751965 w_k10_INTC_s107
                                 + 0.6083 w_k10_KO_s107
                                 + 0.606495 w_k10_MAT_s107
                                 + 0.727881 w_k10_MSFT_s107
                                 + 0.798505 w_k10_PG_s107 >= 1
 z_k10_s1071_constraint:         var_10 + z_k10_s1071 + 1.71202 w_k10_ADM_s107
                                 + 1.21251 w_k10_AXP_s107
                                 + 1.45231 w_k10_DIS_s107
                                 + 1.30933 w_k10_FDX_s107
                                 + 0.794947 w_k10_IBM_s107
                                 + 0.91343 w_k10_INTC_s107
                                 + 1.01174 w_k10_KO_s107
                                 + 0.742219 w_k10_MAT_s107
                                 + 0.980407 w_k10_MSFT_s107
                                 + 0.874267 w_k10_PG_s107 >= 1
 z_k10_s1072_constraint:         var_10 + z_k10_s1072 + 1.60745 w_k10_ADM_s107
                                 + 1.20863 w_k10_AXP_s107
                                 + 1.26716 w_k10_DIS_s107
                                 + 1.16731 w_k10_FDX_s107
                                 + 1.40654 w_k10_IBM_s107
                                 + 0.890064 w_k10_INTC_s107
                                 + 1.25665 w_k10_KO_s107
                                 + 1.90085 w_k10_MAT_s107
                                 + 1.02408 w_k10_MSFT_s107
                                 + 1.1211 w_k10_PG_s107 >= 1
 z_k10_s1073_constraint:         var_10 + z_k10_s1073 + 0.852517 w_k10_ADM_s107
                                 + 1.03148 w_k10_AXP_s107
                                 + 1.03902 w_k10_DIS_s107
                                 + 1.00816 w_k10_FDX_s107
                                 + 0.923099 w_k10_IBM_s107
                                 + 0.913666 w_k10_INTC_s107
                                 + 0.943287 w_k10_KO_s107
                                 + 1.00745 w_k10_MAT_s107
                                 + 0.998584 w_k10_MSFT_s107
                                 + 0.951884 w_k10_PG_s107 >= 1
 z_k10_s1074_constraint:         var_10 + z_k10_s1074 + 1.26091 w_k10_ADM_s107
                                 + 0.909487 w_k10_AXP_s107
                                 + 1.03858 w_k10_DIS_s107
                                 + 1.0824 w_k10_FDX_s107
                                 + 0.627604 w_k10_IBM_s107
                                 + 0.924309 w_k10_INTC_s107
                                 + 0.834897 w_k10_KO_s107
                                 + 0.589948 w_k10_MAT_s107
                                 + 0.958028 w_k10_MSFT_s107
                                 + 1.00128 w_k10_PG_s107 >= 1
 z_k10_s1075_constraint:         var_10 + z_k10_s1075 + 0.89573 w_k10_ADM_s107
                                 + 0.704887 w_k10_AXP_s107
                                 + 0.971555 w_k10_DIS_s107
                                 + 0.89393 w_k10_FDX_s107
                                 + 0.956631 w_k10_IBM_s107
                                 + 0.831007 w_k10_INTC_s107
                                 + 0.93596 w_k10_KO_s107
                                 + 0.809111 w_k10_MAT_s107
                                 + 0.86066 w_k10_MSFT_s107
                                 + 0.972679 w_k10_PG_s107 >= 1
 z_k10_s1076_constraint:         var_10 + z_k10_s1076 + 0.836144 w_k10_ADM_s107
                                 + 0.926431 w_k10_AXP_s107
                                 + 0.943705 w_k10_DIS_s107
                                 + 0.823549 w_k10_FDX_s107
                                 + 1.17346 w_k10_IBM_s107
                                 + 0.69084 w_k10_INTC_s107
                                 + 0.969875 w_k10_KO_s107
                                 + 1.41829 w_k10_MAT_s107
                                 + 0.781674 w_k10_MSFT_s107
                                 + 0.95089 w_k10_PG_s107 >= 1
 z_k10_s1077_constraint:         var_10 + z_k10_s1077 + 1.21359 w_k10_ADM_s107
                                 + 1.00989 w_k10_AXP_s107
                                 + 1.19767 w_k10_DIS_s107
                                 + 1.09035 w_k10_FDX_s107
                                 + 0.754633 w_k10_IBM_s107
                                 + 0.92241 w_k10_INTC_s107
                                 + 0.947053 w_k10_KO_s107
                                 + 1.13865 w_k10_MAT_s107
                                 + 1.06184 w_k10_MSFT_s107
                                 + 1.0649 w_k10_PG_s107 >= 1
 z_k10_s1100_constraint:         var_10 + z_k10_s1100 + 1.31841 w_k10_ADM_s110
                                 + 1.18237 w_k10_AXP_s110
                                 + 1.03373 w_k10_DIS_s110
                                 + 1.13196 w_k10_FDX_s110
                                 + 1.07776 w_k10_IBM_s110
                                 + 1.08694 w_k10_INTC_s110
                                 + 1.27373 w_k10_KO_s110
                                 + 0.82649 w_k10_MAT_s110
                                 + 1.01567 w_k10_MSFT_s110
                                 + 1.05665 w_k10_PG_s110 >= 1
 z_k10_s1101_constraint:         var_10 + z_k10_s1101 + 1.44506 w_k10_ADM_s110
                                 + 1.40173 w_k10_AXP_s110
                                 + 1.51695 w_k10_DIS_s110
                                 + 1.32677 w_k10_FDX_s110
                                 + 1.2356 w_k10_IBM_s110
                                 + 1.24065 w_k10_INTC_s110
                                 + 1.46509 w_k10_KO_s110
                                 + 0.962567 w_k10_MAT_s110
                                 + 1.39308 w_k10_MSFT_s110
                                 + 1.18603 w_k10_PG_s110 >= 1
 z_k10_s1102_constraint:         var_10 + z_k10_s1102 + 1.20375 w_k10_ADM_s110
                                 + 0.978309 w_k10_AXP_s110
                                 + 0.926568 w_k10_DIS_s110
                                 + 1.05113 w_k10_FDX_s110
                                 + 0.698305 w_k10_IBM_s110
                                 + 0.917123 w_k10_INTC_s110
                                 + 1.01295 w_k10_KO_s110
                                 + 0.693731 w_k10_MAT_s110
                                 + 1.09136 w_k10_MSFT_s110
                                 + 1.05198 w_k10_PG_s110 >= 1
 z_k10_s1103_constraint:         var_10 + z_k10_s1103 + 0.889038 w_k10_ADM_s110
                                 + 0.688384 w_k10_AXP_s110
                                 + 0.695841 w_k10_DIS_s110
                                 + 0.781312 w_k10_FDX_s110
                                 + 1.06581 w_k10_IBM_s110
                                 + 0.853486 w_k10_INTC_s110
                                 + 1.08767 w_k10_KO_s110
                                 + 0.938636 w_k10_MAT_s110
                                 + 0.83958 w_k10_MSFT_s110
                                 + 0.957001 w_k10_PG_s110 >= 1
 z_k10_s1104_constraint:         var_10 + z_k10_s1104 + 1.24135 w_k10_ADM_s110
                                 + 1.26499 w_k10_AXP_s110
                                 + 1.21876 w_k10_DIS_s110
                                 + 1.14015 w_k10_FDX_s110
                                 + 1.53881 w_k10_IBM_s110
                                 + 1.05114 w_k10_INTC_s110
                                 + 1.51848 w_k10_KO_s110
                                 + 1.26238 w_k10_MAT_s110
                                 + 1.07877 w_k10_MSFT_s110
                                 + 1.19156 w_k10_PG_s110 >= 1
 z_k10_s1105_constraint:         var_10 + z_k10_s1105 + 1.30394 w_k10_ADM_s110
                                 + 0.934948 w_k10_AXP_s110
                                 + 0.943247 w_k10_DIS_s110
                                 + 1.05723 w_k10_FDX_s110
                                 + 0.841198 w_k10_IBM_s110
                                 + 0.956405 w_k10_INTC_s110
                                 + 1.12324 w_k10_KO_s110
                                 + 0.597574 w_k10_MAT_s110
                                 + 1.04787 w_k10_MSFT_s110
                                 + 0.965866 w_k10_PG_s110 >= 1
 z_k10_s1106_constraint:         var_10 + z_k10_s1106 + 0.832404 w_k10_ADM_s110
                                 + 1.09485 w_k10_AXP_s110
                                 + 0.773927 w_k10_DIS_s110
                                 + 0.908657 w_k10_FDX_s110
                                 + 0.936821 w_k10_IBM_s110
                                 + 0.933179 w_k10_INTC_s110
                                 + 0.960219 w_k10_KO_s110
                                 + 0.732388 w_k10_MAT_s110
                                 + 0.783679 w_k10_MSFT_s110
                                 + 0.892221 w_k10_PG_s110 >= 1
 z_k10_s1107_constraint:         var_10 + z_k10_s1107 + 0.802354 w_k10_ADM_s110
                                 + 0.61317 w_k10_AXP_s110
                                 + 0.437669 w_k10_DIS_s110
                                 + 0.706752 w_k10_FDX_s110
                                 + 0.651162 w_k10_IBM_s110
                                 + 0.720057 w_k10_INTC_s110
                                 + 0.689345 w_k10_KO_s110
                                 + 0.432721 w_k10_MAT_s110
                                 + 0.709755 w_k10_MSFT_s110
                                 + 0.786427 w_k10_PG_s110 >= 1
 z_k10_s1110_constraint:         var_10 + z_k10_s1110 + 0.927877 w_k10_ADM_s111
                                 + 0.917951 w_k10_AXP_s111
                                 + 0.749638 w_k10_DIS_s111
                                 + 0.8666 w_k10_FDX_s111
                                 + 0.895599 w_k10_IBM_s111
                                 + 1.006 w_k10_INTC_s111 + 1.0486 w_k10_KO_s111
                                 + 0.762896 w_k10_MAT_s111
                                 + 1.00475 w_k10_MSFT_s111
                                 + 0.881219 w_k10_PG_s111 >= 1
 z_k10_s1111_constraint:         var_10 + z_k10_s1111 + 1.68662 w_k10_ADM_s111
                                 + 1.22324 w_k10_AXP_s111
                                 + 1.69865 w_k10_DIS_s111
                                 + 1.2681 w_k10_FDX_s111
                                 + 1.35352 w_k10_IBM_s111
                                 + 1.12339 w_k10_INTC_s111
                                 + 1.47658 w_k10_KO_s111
                                 + 1.02558 w_k10_MAT_s111
                                 + 1.30427 w_k10_MSFT_s111
                                 + 1.13868 w_k10_PG_s111 >= 1
 z_k10_s1112_constraint:         var_10 + z_k10_s1112 + 1.1624 w_k10_ADM_s111
                                 + 0.907867 w_k10_AXP_s111
                                 + 0.69269 w_k10_DIS_s111
                                 + 0.722309 w_k10_FDX_s111
                                 + 1.19389 w_k10_IBM_s111
                                 + 0.875077 w_k10_INTC_s111
                                 + 1.02952 w_k10_KO_s111
                                 + 0.838947 w_k10_MAT_s111
                                 + 0.909374 w_k10_MSFT_s111
                                 + 0.923774 w_k10_PG_s111 >= 1
 z_k10_s1113_constraint:         var_10 + z_k10_s1113 + 1.52691 w_k10_ADM_s111
                                 + 1.11201 w_k10_AXP_s111
                                 + 1.26447 w_k10_DIS_s111
                                 + 1.0736 w_k10_FDX_s111
                                 + 0.966925 w_k10_IBM_s111
                                 + 1.13086 w_k10_INTC_s111
                                 + 1.35161 w_k10_KO_s111
                                 + 0.594668 w_k10_MAT_s111
                                 + 1.25117 w_k10_MSFT_s111
                                 + 1.11839 w_k10_PG_s111 >= 1
 z_k10_s1114_constraint:         var_10 + z_k10_s1114 + 1.34393 w_k10_ADM_s111
                                 + 1.36951 w_k10_AXP_s111
                                 + 1.62805 w_k10_DIS_s111
                                 + 1.39634 w_k10_FDX_s111
                                 + 0.963318 w_k10_IBM_s111
                                 + 1.3092 w_k10_INTC_s111
                                 + 1.26342 w_k10_KO_s111
                                 + 1.00574 w_k10_MAT_s111
                                 + 1.41776 w_k10_MSFT_s111
                                 + 1.29623 w_k10_PG_s111 >= 1
 z_k10_s1115_constraint:         var_10 + z_k10_s1115 + 1.19206 w_k10_ADM_s111
                                 + 1.04546 w_k10_AXP_s111
                                 + 0.934356 w_k10_DIS_s111
                                 + 1.01356 w_k10_FDX_s111
                                 + 0.700126 w_k10_IBM_s111
                                 + 0.871302 w_k10_INTC_s111
                                 + 0.879444 w_k10_KO_s111
                                 + 0.696105 w_k10_MAT_s111
                                 + 1.01744 w_k10_MSFT_s111
                                 + 0.852514 w_k10_PG_s111 >= 1
 z_k10_s1116_constraint:         var_10 + z_k10_s1116 + 0.871818 w_k10_ADM_s111
                                 + 1.27658 w_k10_AXP_s111
                                 + 1.00235 w_k10_DIS_s111
                                 + 1.04287 w_k10_FDX_s111
                                 + 1.25854 w_k10_IBM_s111
                                 + 0.952904 w_k10_INTC_s111
                                 + 1.26655 w_k10_KO_s111
                                 + 1.38161 w_k10_MAT_s111
                                 + 1.07441 w_k10_MSFT_s111
                                 + 1.20213 w_k10_PG_s111 >= 1
 z_k10_s1117_constraint:         var_10 + z_k10_s1117 + 1.18028 w_k10_ADM_s111
                                 + 1.17296 w_k10_AXP_s111
                                 + 1.10696 w_k10_DIS_s111
                                 + 1.0456 w_k10_FDX_s111
                                 + 1.38802 w_k10_IBM_s111
                                 + 1.00335 w_k10_INTC_s111
                                 + 1.46238 w_k10_KO_s111
                                 + 1.0842 w_k10_MAT_s111
                                 + 1.10401 w_k10_MSFT_s111
                                 + 1.02103 w_k10_PG_s111 >= 1
 z_k10_s1120_constraint:         var_10 + z_k10_s1120 + 0.981764 w_k10_ADM_s112
                                 + 1.13849 w_k10_AXP_s112
                                 + 1.16688 w_k10_DIS_s112
                                 + 1.17248 w_k10_FDX_s112
                                 + 0.825262 w_k10_IBM_s112
                                 + 0.908235 w_k10_INTC_s112
                                 + 0.973778 w_k10_KO_s112
                                 + 0.787245 w_k10_MAT_s112
                                 + 1.13538 w_k10_MSFT_s112
                                 + 0.983178 w_k10_PG_s112 >= 1
 z_k10_s1121_constraint:         var_10 + z_k10_s1121 + 0.677194 w_k10_ADM_s112
                                 + 0.701423 w_k10_AXP_s112
                                 + 0.69359 w_k10_DIS_s112
                                 + 0.797938 w_k10_FDX_s112
                                 + 0.584712 w_k10_IBM_s112
                                 + 0.755164 w_k10_INTC_s112
                                 + 0.599777 w_k10_KO_s112
                                 + 0.510784 w_k10_MAT_s112
                                 + 0.778522 w_k10_MSFT_s112
                                 + 0.839263 w_k10_PG_s112 >= 1
 z_k10_s1122_constraint:         var_10 + z_k10_s1122 + 1.51489 w_k10_ADM_s112
                                 + 1.52148 w_k10_AXP_s112
                                 + 1.77167 w_k10_DIS_s112
                                 + 1.4971 w_k10_FDX_s112
                                 + 1.07518 w_k10_IBM_s112
                                 + 1.32199 w_k10_INTC_s112
                                 + 1.20331 w_k10_KO_s112
                                 + 0.844196 w_k10_MAT_s112
                                 + 1.41626 w_k10_MSFT_s112
                                 + 1.13118 w_k10_PG_s112 >= 1
 z_k10_s1123_constraint:         var_10 + z_k10_s1123 + 1.07011 w_k10_ADM_s112
                                 + 1.25593 w_k10_AXP_s112
                                 + 1.24697 w_k10_DIS_s112
                                 + 1.12625 w_k10_FDX_s112
                                 + 1.58658 w_k10_IBM_s112
                                 + 0.957098 w_k10_INTC_s112
                                 + 1.22801 w_k10_KO_s112
                                 + 1.5364 w_k10_MAT_s112
                                 + 1.00311 w_k10_MSFT_s112
                                 + 1.08464 w_k10_PG_s112 >= 1
 z_k10_s1124_constraint:         var_10 + z_k10_s1124 + 0.848357 w_k10_ADM_s112
                                 + 0.676967 w_k10_AXP_s112
                                 + 0.568599 w_k10_DIS_s112
                                 + 0.631845 w_k10_FDX_s112
                                 + 0.821995 w_k10_IBM_s112
                                 + 0.63678 w_k10_INTC_s112
                                 + 0.625371 w_k10_KO_s112
                                 + 0.58414 w_k10_MAT_s112
                                 + 0.640362 w_k10_MSFT_s112
                                 + 0.758144 w_k10_PG_s112 >= 1
 z_k10_s1125_constraint:         var_10 + z_k10_s1125 + 0.858108 w_k10_ADM_s112
                                 + 0.987977 w_k10_AXP_s112
                                 + 0.733113 w_k10_DIS_s112
                                 + 0.844498 w_k10_FDX_s112
                                 + 0.878963 w_k10_IBM_s112
                                 + 0.723505 w_k10_INTC_s112
                                 + 0.734225 w_k10_KO_s112
                                 + 0.849209 w_k10_MAT_s112
                                 + 0.809571 w_k10_MSFT_s112
                                 + 0.86033 w_k10_PG_s112 >= 1
 z_k10_s1126_constraint:         var_10 + z_k10_s1126 + 1.12254 w_k10_ADM_s112
                                 + 1.02763 w_k10_AXP_s112
                                 + 1.27582 w_k10_DIS_s112
                                 + 1.08001 w_k10_FDX_s112
                                 + 1.20235 w_k10_IBM_s112
                                 + 0.974404 w_k10_INTC_s112
                                 + 1.29454 w_k10_KO_s112
                                 + 0.718783 w_k10_MAT_s112
                                 + 0.988563 w_k10_MSFT_s112
                                 + 1.10723 w_k10_PG_s112 >= 1
 z_k10_s1127_constraint:         var_10 + z_k10_s1127 + 0.873614 w_k10_ADM_s112
                                 + 0.832724 w_k10_AXP_s112
                                 + 0.914173 w_k10_DIS_s112
                                 + 0.932958 w_k10_FDX_s112
                                 + 0.892106 w_k10_IBM_s112
                                 + 0.832679 w_k10_INTC_s112
                                 + 0.872812 w_k10_KO_s112
                                 + 0.712339 w_k10_MAT_s112
                                 + 0.853243 w_k10_MSFT_s112
                                 + 0.953227 w_k10_PG_s112 >= 1
 z_k10_s1130_constraint:         var_10 + z_k10_s1130 + 0.867421 w_k10_ADM_s113
                                 + 0.985727 w_k10_AXP_s113
                                 + 0.857989 w_k10_DIS_s113
                                 + 1.04329 w_k10_FDX_s113
                                 + 0.679316 w_k10_IBM_s113
                                 + 0.956821 w_k10_INTC_s113
                                 + 0.855162 w_k10_KO_s113
                                 + 0.923343 w_k10_MAT_s113
                                 + 1.05449 w_k10_MSFT_s113
                                 + 0.914254 w_k10_PG_s113 >= 1
 z_k10_s1131_constraint:         var_10 + z_k10_s1131 + 1.28513 w_k10_ADM_s113
                                 + 0.835093 w_k10_AXP_s113
                                 + 0.64713 w_k10_DIS_s113
                                 + 0.770245 w_k10_FDX_s113
                                 + 1.09279 w_k10_IBM_s113
                                 + 0.735203 w_k10_INTC_s113
                                 + 0.915546 w_k10_KO_s113
                                 + 1.13575 w_k10_MAT_s113
                                 + 0.83334 w_k10_MSFT_s113
                                 + 0.850832 w_k10_PG_s113 >= 1
 z_k10_s1132_constraint:         var_10 + z_k10_s1132 + 1.18778 w_k10_ADM_s113
                                 + 1.28704 w_k10_AXP_s113
                                 + 1.18929 w_k10_DIS_s113
                                 + 1.20095 w_k10_FDX_s113
                                 + 1.28169 w_k10_IBM_s113
                                 + 1.28558 w_k10_INTC_s113
                                 + 1.24974 w_k10_KO_s113
                                 + 1.3042 w_k10_MAT_s113
                                 + 1.08817 w_k10_MSFT_s113
                                 + 1.04896 w_k10_PG_s113 >= 1
 z_k10_s1133_constraint:         var_10 + z_k10_s1133 + 1.72203 w_k10_ADM_s113
                                 + 1.50744 w_k10_AXP_s113
                                 + 1.19705 w_k10_DIS_s113
                                 + 1.33757 w_k10_FDX_s113
                                 + 0.904648 w_k10_IBM_s113
                                 + 0.919631 w_k10_INTC_s113
                                 + 0.976894 w_k10_KO_s113
                                 + 1.21823 w_k10_MAT_s113
                                 + 1.10218 w_k10_MSFT_s113
                                 + 0.958296 w_k10_PG_s113 >= 1
 z_k10_s1134_constraint:         var_10 + z_k10_s1134 + 1.16948 w_k10_ADM_s113
                                 + 0.778635 w_k10_AXP_s113
                                 + 0.895533 w_k10_DIS_s113
                                 + 0.98387 w_k10_FDX_s113
                                 + 0.973336 w_k10_IBM_s113
                                 + 0.807471 w_k10_INTC_s113
                                 + 1.00564 w_k10_KO_s113
                                 + 0.959798 w_k10_MAT_s113
                                 + 0.939795 w_k10_MSFT_s113
                                 + 0.940805 w_k10_PG_s113 >= 1
 z_k10_s1135_constraint:         var_10 + z_k10_s1135 + 1.34594 w_k10_ADM_s113
                                 + 1.14502 w_k10_AXP_s113
                                 + 1.31954 w_k10_DIS_s113
                                 + 1.25136 w_k10_FDX_s113
                                 + 1.03031 w_k10_IBM_s113
                                 + 1.02921 w_k10_INTC_s113
                                 + 1.18134 w_k10_KO_s113
                                 + 1.04167 w_k10_MAT_s113
                                 + 1.01827 w_k10_MSFT_s113
                                 + 1.05495 w_k10_PG_s113 >= 1
 z_k10_s1136_constraint:         var_10 + z_k10_s1136 + 0.960839 w_k10_ADM_s113
                                 + 1.15652 w_k10_AXP_s113
                                 + 1.09513 w_k10_DIS_s113
                                 + 1.09616 w_k10_FDX_s113
                                 + 1.74256 w_k10_IBM_s113
                                 + 1.07933 w_k10_INTC_s113
                                 + 1.2314 w_k10_KO_s113
                                 + 1.69603 w_k10_MAT_s113
                                 + 0.97048 w_k10_MSFT_s113
                                 + 1.05279 w_k10_PG_s113 >= 1
 z_k10_s1137_constraint:         var_10 + z_k10_s1137 + 1.13655 w_k10_ADM_s113
                                 + 1.10896 w_k10_AXP_s113
                                 + 1.30392 w_k10_DIS_s113
                                 + 1.1337 w_k10_FDX_s113
                                 + 1.53396 w_k10_IBM_s113
                                 + 0.964416 w_k10_INTC_s113
                                 + 1.42276 w_k10_KO_s113
                                 + 1.66453 w_k10_MAT_s113
                                 + 1.12821 w_k10_MSFT_s113
                                 + 1.18247 w_k10_PG_s113 >= 1
 z_k10_s1140_constraint:         var_10 + z_k10_s1140 + 0.984882 w_k10_ADM_s114
                                 + 1.3283 w_k10_AXP_s114
                                 + 1.51065 w_k10_DIS_s114
                                 + 1.51622 w_k10_FDX_s114
                                 + 0.994482 w_k10_IBM_s114
                                 + 1.52559 w_k10_INTC_s114
                                 + 1.11395 w_k10_KO_s114
                                 + 0.757602 w_k10_MAT_s114
                                 + 1.48682 w_k10_MSFT_s114
                                 + 1.07865 w_k10_PG_s114 >= 1
 z_k10_s1141_constraint:         var_10 + z_k10_s1141 + 1.20191 w_k10_ADM_s114
                                 + 1.61789 w_k10_AXP_s114
                                 + 1.69512 w_k10_DIS_s114
                                 + 1.30637 w_k10_FDX_s114
                                 + 1.47651 w_k10_IBM_s114
                                 + 0.995429 w_k10_INTC_s114
                                 + 1.1892 w_k10_KO_s114
                                 + 1.23831 w_k10_MAT_s114
                                 + 1.10805 w_k10_MSFT_s114
                                 + 1.16306 w_k10_PG_s114 >= 1
 z_k10_s1142_constraint:         var_10 + z_k10_s1142 + 0.697008 w_k10_ADM_s114
                                 + 0.84276 w_k10_AXP_s114
                                 + 0.702691 w_k10_DIS_s114
                                 + 0.751578 w_k10_FDX_s114
                                 + 1.00477 w_k10_IBM_s114
                                 + 0.871775 w_k10_INTC_s114
                                 + 0.750451 w_k10_KO_s114
                                 + 0.596489 w_k10_MAT_s114
                                 + 0.80883 w_k10_MSFT_s114
                                 + 0.799393 w_k10_PG_s114 >= 1
 z_k10_s1143_constraint:         var_10 + z_k10_s1143 + 1.1927 w_k10_ADM_s114
                                 + 1.34148 w_k10_AXP_s114
                                 + 1.22262 w_k10_DIS_s114
                                 + 1.05559 w_k10_FDX_s114
                                 + 1.18719 w_k10_IBM_s114
                                 + 0.96771 w_k10_INTC_s114
                                 + 1.07533 w_k10_KO_s114
                                 + 1.26705 w_k10_MAT_s114
                                 + 1.21725 w_k10_MSFT_s114
                                 + 1.00814 w_k10_PG_s114 >= 1
 z_k10_s1144_constraint:         var_10 + z_k10_s1144 + 0.862337 w_k10_ADM_s114
                                 + 0.996095 w_k10_AXP_s114
                                 + 0.730954 w_k10_DIS_s114
                                 + 0.968835 w_k10_FDX_s114
                                 + 0.684583 w_k10_IBM_s114
                                 + 1.06801 w_k10_INTC_s114
                                 + 0.675095 w_k10_KO_s114
                                 + 0.513609 w_k10_MAT_s114
                                 + 0.950871 w_k10_MSFT_s114
                                 + 0.86334 w_k10_PG_s114 >= 1
 z_k10_s1145_constraint:         var_10 + z_k10_s1145 + 1.00806 w_k10_ADM_s114
                                 + 1.21875 w_k10_AXP_s114
                                 + 0.881386 w_k10_DIS_s114
                                 + 1.08251 w_k10_FDX_s114
                                 + 1.01483 w_k10_IBM_s114
                                 + 0.924415 w_k10_INTC_s114
                                 + 0.951948 w_k10_KO_s114
                                 + 0.879184 w_k10_MAT_s114
                                 + 0.887023 w_k10_MSFT_s114
                                 + 0.954191 w_k10_PG_s114 >= 1
 z_k10_s1146_constraint:         var_10 + z_k10_s1146 + 0.5564 w_k10_ADM_s114
                                 + 0.773437 w_k10_AXP_s114
                                 + 0.745728 w_k10_DIS_s114
                                 + 0.814674 w_k10_FDX_s114
                                 + 1.54129 w_k10_IBM_s114
                                 + 0.827811 w_k10_INTC_s114
                                 + 0.997951 w_k10_KO_s114
                                 + 1.73473 w_k10_MAT_s114
                                 + 0.794072 w_k10_MSFT_s114
                                 + 1.00184 w_k10_PG_s114 >= 1
 z_k10_s1147_constraint:         var_10 + z_k10_s1147 + 0.959049 w_k10_ADM_s114
                                 + 1.00968 w_k10_AXP_s114
                                 + 1.03429 w_k10_DIS_s114
                                 + 1.0342 w_k10_FDX_s114
                                 + 0.989233 w_k10_IBM_s114
                                 + 1.04171 w_k10_INTC_s114
                                 + 0.975541 w_k10_KO_s114
                                 + 0.772652 w_k10_MAT_s114
                                 + 1.12428 w_k10_MSFT_s114
                                 + 0.934674 w_k10_PG_s114 >= 1
 z_k10_s1150_constraint:         var_10 + z_k10_s1150 + 0.922533 w_k10_ADM_s115
                                 + 1.04723 w_k10_AXP_s115
                                 + 0.723753 w_k10_DIS_s115
                                 + 0.858509 w_k10_FDX_s115
                                 + 0.807035 w_k10_IBM_s115
                                 + 0.851771 w_k10_INTC_s115
                                 + 0.844661 w_k10_KO_s115
                                 + 1.17906 w_k10_MAT_s115
                                 + 0.960012 w_k10_MSFT_s115
                                 + 0.923418 w_k10_PG_s115 >= 1
 z_k10_s1151_constraint:         var_10 + z_k10_s1151 + 1.02412 w_k10_ADM_s115
                                 + 1.33435 w_k10_AXP_s115
                                 + 1.10803 w_k10_DIS_s115
                                 + 0.973137 w_k10_FDX_s115
                                 + 1.4745 w_k10_IBM_s115
                                 + 1.09887 w_k10_INTC_s115
                                 + 1.04958 w_k10_KO_s115
                                 + 1.45523 w_k10_MAT_s115
                                 + 1.01522 w_k10_MSFT_s115
                                 + 1.07408 w_k10_PG_s115 >= 1
 z_k10_s1152_constraint:         var_10 + z_k10_s1152 + 0.90098 w_k10_ADM_s115
                                 + 0.924348 w_k10_AXP_s115
                                 + 0.95068 w_k10_DIS_s115
                                 + 0.947331 w_k10_FDX_s115
                                 + 0.747811 w_k10_IBM_s115
                                 + 0.972898 w_k10_INTC_s115
                                 + 0.915771 w_k10_KO_s115
                                 + 0.818355 w_k10_MAT_s115
                                 + 0.937098 w_k10_MSFT_s115
                                 + 0.876239 w_k10_PG_s115 >= 1
 z_k10_s1153_constraint:         var_10 + z_k10_s1153 + 0.897059 w_k10_ADM_s115
                                 + 1.02522 w_k10_AXP_s115
                                 + 1.11118 w_k10_DIS_s115
                                 + 1.00738 w_k10_FDX_s115
                                 + 1.07931 w_k10_IBM_s115
                                 + 1.2328 w_k10_INTC_s115
                                 + 1.00461 w_k10_KO_s115
                                 + 1.22815 w_k10_MAT_s115
                                 + 1.2108 w_k10_MSFT_s115
                                 + 1.13477 w_k10_PG_s115 >= 1
 z_k10_s1154_constraint:         var_10 + z_k10_s1154 + 0.682107 w_k10_ADM_s115
                                 + 0.70519 w_k10_AXP_s115
                                 + 0.559476 w_k10_DIS_s115
                                 + 0.698254 w_k10_FDX_s115
                                 + 0.766933 w_k10_IBM_s115
                                 + 0.693057 w_k10_INTC_s115
                                 + 0.643017 w_k10_KO_s115
                                 + 0.710384 w_k10_MAT_s115
                                 + 0.713956 w_k10_MSFT_s115
                                 + 0.810316 w_k10_PG_s115 >= 1
 z_k10_s1155_constraint:         var_10 + z_k10_s1155 + 1.11177 w_k10_ADM_s115
                                 + 1.30165 w_k10_AXP_s115
                                 + 1.46753 w_k10_DIS_s115
                                 + 1.21597 w_k10_FDX_s115
                                 + 1.28227 w_k10_IBM_s115
                                 + 1.23613 w_k10_INTC_s115
                                 + 1.31761 w_k10_KO_s115
                                 + 1.80684 w_k10_MAT_s115
                                 + 1.27333 w_k10_MSFT_s115
                                 + 1.22199 w_k10_PG_s115 >= 1
 z_k10_s1156_constraint:         var_10 + z_k10_s1156 + 0.797165 w_k10_ADM_s115
                                 + 1.34519 w_k10_AXP_s115
                                 + 1.26895 w_k10_DIS_s115
                                 + 1.27934 w_k10_FDX_s115
                                 + 0.915035 w_k10_IBM_s115
                                 + 1.33713 w_k10_INTC_s115
                                 + 0.992148 w_k10_KO_s115
                                 + 1.00476 w_k10_MAT_s115
                                 + 1.23194 w_k10_MSFT_s115
                                 + 1.07669 w_k10_PG_s115 >= 1
 z_k10_s1157_constraint:         var_10 + z_k10_s1157 + 0.780493 w_k10_ADM_s115
                                 + 0.87751 w_k10_AXP_s115
                                 + 0.962115 w_k10_DIS_s115
                                 + 0.912518 w_k10_FDX_s115
                                 + 0.920022 w_k10_IBM_s115
                                 + 1.15118 w_k10_INTC_s115
                                 + 1.00292 w_k10_KO_s115
                                 + 0.851191 w_k10_MAT_s115
                                 + 1.12462 w_k10_MSFT_s115
                                 + 1.0245 w_k10_PG_s115 >= 1
 z_k10_s1160_constraint:         var_10 + z_k10_s1160 + 1.13845 w_k10_ADM_s116
                                 + 1.41423 w_k10_AXP_s116
                                 + 1.10229 w_k10_DIS_s116
                                 + 1.26277 w_k10_FDX_s116
                                 + 0.946013 w_k10_IBM_s116
                                 + 1.1592 w_k10_INTC_s116
                                 + 1.11252 w_k10_KO_s116
                                 + 0.921714 w_k10_MAT_s116
                                 + 1.1682 w_k10_MSFT_s116
                                 + 1.3158 w_k10_PG_s116 >= 1
 z_k10_s1161_constraint:         var_10 + z_k10_s1161 + 1.11836 w_k10_ADM_s116
                                 + 1.1692 w_k10_AXP_s116
                                 + 1.63043 w_k10_DIS_s116
                                 + 1.33492 w_k10_FDX_s116
                                 + 1.08427 w_k10_IBM_s116
                                 + 1.16949 w_k10_INTC_s116
                                 + 1.29953 w_k10_KO_s116
                                 + 1.11594 w_k10_MAT_s116
                                 + 1.25606 w_k10_MSFT_s116
                                 + 1.22387 w_k10_PG_s116 >= 1
 z_k10_s1162_constraint:         var_10 + z_k10_s1162 + 1.1613 w_k10_ADM_s116
                                 + 1.11871 w_k10_AXP_s116
                                 + 1.55043 w_k10_DIS_s116
                                 + 1.3482 w_k10_FDX_s116
                                 + 1.54333 w_k10_IBM_s116
                                 + 1.20926 w_k10_INTC_s116
                                 + 1.50532 w_k10_KO_s116
                                 + 1.56564 w_k10_MAT_s116
                                 + 1.4551 w_k10_MSFT_s116
                                 + 1.39083 w_k10_PG_s116 >= 1
 z_k10_s1163_constraint:         var_10 + z_k10_s1163 + 1.21655 w_k10_ADM_s116
                                 + 0.921155 w_k10_AXP_s116
                                 + 0.606195 w_k10_DIS_s116
                                 + 0.989889 w_k10_FDX_s116
                                 + 0.771125 w_k10_IBM_s116
                                 + 0.787076 w_k10_INTC_s116
                                 + 0.810055 w_k10_KO_s116
                                 + 0.703655 w_k10_MAT_s116
                                 + 0.862737 w_k10_MSFT_s116
                                 + 0.961224 w_k10_PG_s116 >= 1
 z_k10_s1164_constraint:         var_10 + z_k10_s1164 + 0.896093 w_k10_ADM_s116
                                 + 1.10212 w_k10_AXP_s116
                                 + 1.14302 w_k10_DIS_s116
                                 + 1.00671 w_k10_FDX_s116
                                 + 1.57764 w_k10_IBM_s116
                                 + 1.13864 w_k10_INTC_s116
                                 + 1.34279 w_k10_KO_s116
                                 + 1.09892 w_k10_MAT_s116
                                 + 0.974745 w_k10_MSFT_s116
                                 + 1.08097 w_k10_PG_s116 >= 1
 z_k10_s1165_constraint:         var_10 + z_k10_s1165 + 0.772719 w_k10_ADM_s116
                                 + 0.826643 w_k10_AXP_s116
                                 + 0.771605 w_k10_DIS_s116
                                 + 0.936651 w_k10_FDX_s116
                                 + 0.892192 w_k10_IBM_s116
                                 + 0.878893 w_k10_INTC_s116
                                 + 0.969546 w_k10_KO_s116
                                 + 0.825605 w_k10_MAT_s116
                                 + 0.975763 w_k10_MSFT_s116
                                 + 0.947316 w_k10_PG_s116 >= 1
 z_k10_s1166_constraint:         var_10 + z_k10_s1166 + 1.23437 w_k10_ADM_s116
                                 + 0.961644 w_k10_AXP_s116
                                 + 1.15874 w_k10_DIS_s116
                                 + 1.15551 w_k10_FDX_s116
                                 + 0.677406 w_k10_IBM_s116
                                 + 1.0685 w_k10_INTC_s116
                                 + 0.990307 w_k10_KO_s116
                                 + 0.760707 w_k10_MAT_s116
                                 + 1.13881 w_k10_MSFT_s116
                                 + 1.0056 w_k10_PG_s116 >= 1
 z_k10_s1167_constraint:         var_10 + z_k10_s1167 + 0.93107 w_k10_ADM_s116
                                 + 0.904506 w_k10_AXP_s116
                                 + 0.77662 w_k10_DIS_s116
                                 + 0.883211 w_k10_FDX_s116
                                 + 1.17326 w_k10_IBM_s116
                                 + 0.931594 w_k10_INTC_s116
                                 + 1.07613 w_k10_KO_s116
                                 + 1.1877 w_k10_MAT_s116
                                 + 1.04895 w_k10_MSFT_s116
                                 + 1.15243 w_k10_PG_s116 >= 1
 z_k10_s1170_constraint:         var_10 + z_k10_s1170 + 1.06548 w_k10_ADM_s117
                                 + 1.23517 w_k10_AXP_s117
                                 + 1.22961 w_k10_DIS_s117
                                 + 1.2188 w_k10_FDX_s117
                                 + 0.676249 w_k10_IBM_s117
                                 + 1.29299 w_k10_INTC_s117
                                 + 0.915732 w_k10_KO_s117
                                 + 0.579781 w_k10_MAT_s117
                                 + 1.15272 w_k10_MSFT_s117
                                 + 1.01891 w_k10_PG_s117 >= 1
 z_k10_s1171_constraint:         var_10 + z_k10_s1171 + 0.829505 w_k10_ADM_s117
                                 + 0.833424 w_k10_AXP_s117
                                 + 0.765704 w_k10_DIS_s117
                                 + 0.760735 w_k10_FDX_s117
                                 + 1.06362 w_k10_IBM_s117
                                 + 0.890046 w_k10_INTC_s117
                                 + 0.897459 w_k10_KO_s117
                                 + 0.9885 w_k10_MAT_s117
                                 + 0.743629 w_k10_MSFT_s117
                                 + 0.726148 w_k10_PG_s117 >= 1
 z_k10_s1172_constraint:         var_10 + z_k10_s1172 + 1.14042 w_k10_ADM_s117
                                 + 1.03253 w_k10_AXP_s117
                                 + 1.164 w_k10_DIS_s117
                                 + 0.971894 w_k10_FDX_s117
                                 + 0.821808 w_k10_IBM_s117
                                 + 1.00985 w_k10_INTC_s117
                                 + 0.987788 w_k10_KO_s117
                                 + 0.978672 w_k10_MAT_s117
                                 + 1.07466 w_k10_MSFT_s117
                                 + 0.970345 w_k10_PG_s117 >= 1
 z_k10_s1173_constraint:         var_10 + z_k10_s1173 + 1.06635 w_k10_ADM_s117
                                 + 1.39185 w_k10_AXP_s117
                                 + 1.2762 w_k10_DIS_s117
                                 + 1.26353 w_k10_FDX_s117
                                 + 1.0925 w_k10_IBM_s117
                                 + 1.19045 w_k10_INTC_s117
                                 + 1.05953 w_k10_KO_s117
                                 + 1.3842 w_k10_MAT_s117
                                 + 1.08219 w_k10_MSFT_s117
                                 + 1.00105 w_k10_PG_s117 >= 1
 z_k10_s1174_constraint:         var_10 + z_k10_s1174 + 1.21989 w_k10_ADM_s117
                                 + 1.46945 w_k10_AXP_s117
                                 + 1.29834 w_k10_DIS_s117
                                 + 1.14749 w_k10_FDX_s117
                                 + 1.46334 w_k10_IBM_s117
                                 + 1.27652 w_k10_INTC_s117
                                 + 1.33809 w_k10_KO_s117
                                 + 1.81799 w_k10_MAT_s117
                                 + 1.18547 w_k10_MSFT_s117
                                 + 1.0756 w_k10_PG_s117 >= 1
 z_k10_s1175_constraint:         var_10 + z_k10_s1175 + 0.778057 w_k10_ADM_s117
                                 + 0.850157 w_k10_AXP_s117
                                 + 0.66306 w_k10_DIS_s117
                                 + 0.769367 w_k10_FDX_s117
                                 + 0.741196 w_k10_IBM_s117
                                 + 1.05279 w_k10_INTC_s117
                                 + 0.693835 w_k10_KO_s117
                                 + 0.763976 w_k10_MAT_s117
                                 + 0.817925 w_k10_MSFT_s117
                                 + 0.794711 w_k10_PG_s117 >= 1
 z_k10_s1176_constraint:         var_10 + z_k10_s1176 + 1.06756 w_k10_ADM_s117
                                 + 0.844785 w_k10_AXP_s117
                                 + 0.828973 w_k10_DIS_s117
                                 + 0.908584 w_k10_FDX_s117
                                 + 0.638933 w_k10_IBM_s117
                                 + 0.955826 w_k10_INTC_s117
                                 + 0.74642 w_k10_KO_s117
                                 + 0.482188 w_k10_MAT_s117
                                 + 0.99735 w_k10_MSFT_s117
                                 + 0.826061 w_k10_PG_s117 >= 1
 z_k10_s1177_constraint:         var_10 + z_k10_s1177 + 1.05989 w_k10_ADM_s117
                                 + 0.890827 w_k10_AXP_s117
                                 + 0.866114 w_k10_DIS_s117
                                 + 0.891251 w_k10_FDX_s117
                                 + 1.00519 w_k10_IBM_s117
                                 + 0.915088 w_k10_INTC_s117
                                 + 0.908378 w_k10_KO_s117
                                 + 0.924276 w_k10_MAT_s117
                                 + 0.918483 w_k10_MSFT_s117
                                 + 0.915439 w_k10_PG_s117 >= 1
 z_k10_s1200_constraint:         var_10 + z_k10_s1200 + 1.07268 w_k10_ADM_s120
                                 + 0.652752 w_k10_AXP_s120
                                 + 0.626979 w_k10_DIS_s120
                                 + 0.757346 w_k10_FDX_s120
                                 + 0.825928 w_k10_IBM_s120
                                 + 0.658452 w_k10_INTC_s120
                                 + 0.809217 w_k10_KO_s120
                                 + 0.754286 w_k10_MAT_s120
                                 + 0.723327 w_k10_MSFT_s120
                                 + 0.877672 w_k10_PG_s120 >= 1
 z_k10_s1201_constraint:         var_10 + z_k10_s1201 + 1.02484 w_k10_ADM_s120
                                 + 0.788283 w_k10_AXP_s120
                                 + 0.949537 w_k10_DIS_s120
                                 + 0.879918 w_k10_FDX_s120
                                 + 1.17458 w_k10_IBM_s120
                                 + 0.801165 w_k10_INTC_s120
                                 + 1.15206 w_k10_KO_s120
                                 + 1.01795 w_k10_MAT_s120
                                 + 0.862383 w_k10_MSFT_s120
                                 + 1.05329 w_k10_PG_s120 >= 1
 z_k10_s1202_constraint:         var_10 + z_k10_s1202 + 1.22118 w_k10_ADM_s120
                                 + 1.40978 w_k10_AXP_s120
                                 + 1.81828 w_k10_DIS_s120
                                 + 1.29629 w_k10_FDX_s120
                                 + 1.33676 w_k10_IBM_s120
                                 + 1.05907 w_k10_INTC_s120
                                 + 1.46105 w_k10_KO_s120
                                 + 1.7196 w_k10_MAT_s120
                                 + 1.31533 w_k10_MSFT_s120
                                 + 1.31507 w_k10_PG_s120 >= 1
 z_k10_s1203_constraint:         var_10 + z_k10_s1203 + 1.05456 w_k10_ADM_s120
                                 + 1.17171 w_k10_AXP_s120
                                 + 0.89991 w_k10_DIS_s120
                                 + 0.96074 w_k10_FDX_s120
                                 + 0.88172 w_k10_IBM_s120
                                 + 0.835618 w_k10_INTC_s120
                                 + 0.885288 w_k10_KO_s120
                                 + 0.981129 w_k10_MAT_s120
                                 + 0.930067 w_k10_MSFT_s120
                                 + 0.976798 w_k10_PG_s120 >= 1
 z_k10_s1204_constraint:         var_10 + z_k10_s1204 + 0.979203 w_k10_ADM_s120
                                 + 1.29178 w_k10_AXP_s120
                                 + 0.909568 w_k10_DIS_s120
                                 + 1.02414 w_k10_FDX_s120
                                 + 1.48462 w_k10_IBM_s120
                                 + 0.877176 w_k10_INTC_s120
                                 + 1.07051 w_k10_KO_s120
                                 + 1.35314 w_k10_MAT_s120
                                 + 0.785267 w_k10_MSFT_s120
                                 + 1.04774 w_k10_PG_s120 >= 1
 z_k10_s1205_constraint:         var_10 + z_k10_s1205 + 1.09998 w_k10_ADM_s120
                                 + 0.831416 w_k10_AXP_s120
                                 + 1.19116 w_k10_DIS_s120
                                 + 1.00744 w_k10_FDX_s120
                                 + 0.903904 w_k10_IBM_s120
                                 + 0.832686 w_k10_INTC_s120
                                 + 1.10643 w_k10_KO_s120
                                 + 0.968912 w_k10_MAT_s120
                                 + 1.05106 w_k10_MSFT_s120
                                 + 1.16246 w_k10_PG_s120 >= 1
 z_k10_s1206_constraint:         var_10 + z_k10_s1206 + 1.31341 w_k10_ADM_s120
                                 + 1.49336 w_k10_AXP_s120
                                 + 1.3777 w_k10_DIS_s120
                                 + 1.27668 w_k10_FDX_s120
                                 + 1.18349 w_k10_IBM_s120
                                 + 1.26903 w_k10_INTC_s120
                                 + 1.26461 w_k10_KO_s120
                                 + 0.975257 w_k10_MAT_s120
                                 + 1.14843 w_k10_MSFT_s120
                                 + 1.28763 w_k10_PG_s120 >= 1
 z_k10_s1207_constraint:         var_10 + z_k10_s1207 + 1.09063 w_k10_ADM_s120
                                 + 0.710003 w_k10_AXP_s120
                                 + 0.924772 w_k10_DIS_s120
                                 + 0.885541 w_k10_FDX_s120
                                 + 0.826747 w_k10_IBM_s120
                                 + 0.850201 w_k10_INTC_s120
                                 + 0.891601 w_k10_KO_s120
                                 + 0.825893 w_k10_MAT_s120
                                 + 0.908556 w_k10_MSFT_s120
                                 + 0.874382 w_k10_PG_s120 >= 1
 z_k10_s1210_constraint:         var_10 + z_k10_s1210 + 1.32855 w_k10_ADM_s121
                                 + 1.01148 w_k10_AXP_s121
                                 + 0.981794 w_k10_DIS_s121
                                 + 1.06411 w_k10_FDX_s121
                                 + 0.82614 w_k10_IBM_s121
                                 + 0.821063 w_k10_INTC_s121
                                 + 1.07813 w_k10_KO_s121
                                 + 0.778912 w_k10_MAT_s121
                                 + 0.880002 w_k10_MSFT_s121
                                 + 1.01149 w_k10_PG_s121 >= 1
 z_k10_s1211_constraint:         var_10 + z_k10_s1211 + 0.990216 w_k10_ADM_s121
                                 + 0.744528 w_k10_AXP_s121
                                 + 0.692334 w_k10_DIS_s121
                                 + 0.917119 w_k10_FDX_s121
                                 + 0.658563 w_k10_IBM_s121
                                 + 0.810226 w_k10_INTC_s121
                                 + 0.86591 w_k10_KO_s121
                                 + 0.592024 w_k10_MAT_s121
                                 + 0.982526 w_k10_MSFT_s121
                                 + 0.804572 w_k10_PG_s121 >= 1
 z_k10_s1212_constraint:         var_10 + z_k10_s1212 + 1.07686 w_k10_ADM_s121
                                 + 0.891641 w_k10_AXP_s121
                                 + 0.781246 w_k10_DIS_s121
                                 + 0.840037 w_k10_FDX_s121
                                 + 1.17368 w_k10_IBM_s121
                                 + 0.959509 w_k10_INTC_s121
                                 + 1.14018 w_k10_KO_s121
                                 + 0.693357 w_k10_MAT_s121
                                 + 0.997406 w_k10_MSFT_s121
                                 + 0.908996 w_k10_PG_s121 >= 1
 z_k10_s1213_constraint:         var_10 + z_k10_s1213 + 0.99024 w_k10_ADM_s121
                                 + 0.71787 w_k10_AXP_s121
                                 + 0.901432 w_k10_DIS_s121
                                 + 0.910894 w_k10_FDX_s121
                                 + 0.86775 w_k10_IBM_s121
                                 + 0.982883 w_k10_INTC_s121
                                 + 0.971573 w_k10_KO_s121
                                 + 0.57336 w_k10_MAT_s121
                                 + 0.928494 w_k10_MSFT_s121
                                 + 0.856872 w_k10_PG_s121 >= 1
 z_k10_s1214_constraint:         var_10 + z_k10_s1214 + 0.994524 w_k10_ADM_s121
                                 + 0.840704 w_k10_AXP_s121
                                 + 0.695058 w_k10_DIS_s121
                                 + 0.815398 w_k10_FDX_s121
                                 + 0.993073 w_k10_IBM_s121
                                 + 0.989444 w_k10_INTC_s121
                                 + 1.0061 w_k10_KO_s121
                                 + 0.817719 w_k10_MAT_s121
                                 + 0.922896 w_k10_MSFT_s121
                                 + 0.848843 w_k10_PG_s121 >= 1
 z_k10_s1215_constraint:         var_10 + z_k10_s1215 + 1.25906 w_k10_ADM_s121
                                 + 1.07136 w_k10_AXP_s121
                                 + 1.08413 w_k10_DIS_s121
                                 + 1.00113 w_k10_FDX_s121
                                 + 1.27651 w_k10_IBM_s121
                                 + 1.19046 w_k10_INTC_s121
                                 + 1.33686 w_k10_KO_s121
                                 + 1.12715 w_k10_MAT_s121
                                 + 1.13966 w_k10_MSFT_s121
                                 + 1.0512 w_k10_PG_s121 >= 1
 z_k10_s1216_constraint:         var_10 + z_k10_s1216 + 0.977092 w_k10_ADM_s121
                                 + 0.559662 w_k10_AXP_s121
                                 + 0.620821 w_k10_DIS_s121
                                 + 0.738633 w_k10_FDX_s121
                                 + 0.765548 w_k10_IBM_s121
                                 + 0.786872 w_k10_INTC_s121
                                 + 0.864905 w_k10_KO_s121
                                 + 0.435849 w_k10_MAT_s121
                                 + 0.759795 w_k10_MSFT_s121
                                 + 0.775922 w_k10_PG_s121 >= 1
 z_k10_s1217_constraint:         var_10 + z_k10_s1217 + 1.37766 w_k10_ADM_s121
                                 + 1.48968 w_k10_AXP_s121
                                 + 1.43178 w_k10_DIS_s121
                                 + 1.31935 w_k10_FDX_s121
                                 + 1.14554 w_k10_IBM_s121
                                 + 1.37172 w_k10_INTC_s121
                                 + 1.60303 w_k10_KO_s121
                                 + 0.927038 w_k10_MAT_s121
                                 + 1.36141 w_k10_MSFT_s121
                                 + 1.18366 w_k10_PG_s121 >= 1
 z_k10_s1220_constraint:         var_10 + z_k10_s1220 + 0.881535 w_k10_ADM_s122
                                 + 1.28067 w_k10_AXP_s122
                                 + 1.60288 w_k10_DIS_s122
                                 + 1.26882 w_k10_FDX_s122
                                 + 1.0308 w_k10_IBM_s122
                                 + 1.13463 w_k10_INTC_s122
                                 + 1.1364 w_k10_KO_s122
                                 + 0.845103 w_k10_MAT_s122
                                 + 1.06386 w_k10_MSFT_s122
                                 + 1.05859 w_k10_PG_s122 >= 1
 z_k10_s1221_constraint:         var_10 + z_k10_s1221 + 0.750459 w_k10_ADM_s122
                                 + 0.739071 w_k10_AXP_s122
                                 + 0.850897 w_k10_DIS_s122
                                 + 0.871254 w_k10_FDX_s122
                                 + 0.84019 w_k10_IBM_s122
                                 + 0.891677 w_k10_INTC_s122
                                 + 0.855367 w_k10_KO_s122
                                 + 0.581577 w_k10_MAT_s122
                                 + 0.898264 w_k10_MSFT_s122
                                 + 0.884091 w_k10_PG_s122 >= 1
 z_k10_s1222_constraint:         var_10 + z_k10_s1222 + 1.12128 w_k10_ADM_s122
                                 + 1.12961 w_k10_AXP_s122
                                 + 0.997716 w_k10_DIS_s122
                                 + 0.882073 w_k10_FDX_s122
                                 + 1.55151 w_k10_IBM_s122
                                 + 0.80723 w_k10_INTC_s122
                                 + 1.3407 w_k10_KO_s122
                                 + 1.37085 w_k10_MAT_s122
                                 + 0.868792 w_k10_MSFT_s122
                                 + 1.08566 w_k10_PG_s122 >= 1
 z_k10_s1223_constraint:         var_10 + z_k10_s1223 + 0.939147 w_k10_ADM_s122
                                 + 0.9822 w_k10_AXP_s122
                                 + 1.01601 w_k10_DIS_s122
                                 + 1.10522 w_k10_FDX_s122
                                 + 0.787297 w_k10_IBM_s122
                                 + 1.13321 w_k10_INTC_s122
                                 + 1.0502 w_k10_KO_s122
                                 + 1.02784 w_k10_MAT_s122
                                 + 1.17388 w_k10_MSFT_s122
                                 + 1.07132 w_k10_PG_s122 >= 1
 z_k10_s1224_constraint:         var_10 + z_k10_s1224 + 0.888749 w_k10_ADM_s122
                                 + 1.00739 w_k10_AXP_s122
                                 + 1.09431 w_k10_DIS_s122
                                 + 0.936055 w_k10_FDX_s122
                                 + 1.16187 w_k10_IBM_s122
                                 + 0.909659 w_k10_INTC_s122
                                 + 1.27618 w_k10_KO_s122
                                 + 1.1958 w_k10_MAT_s122
                                 + 1.12339 w_k10_MSFT_s122
                                 + 1.1223 w_k10_PG_s122 >= 1
 z_k10_s1225_constraint:         var_10 + z_k10_s1225 + 0.822332 w_k10_ADM_s122
                                 + 0.780941 w_k10_AXP_s122
                                 + 0.648783 w_k10_DIS_s122
                                 + 0.777008 w_k10_FDX_s122
                                 + 1.06566 w_k10_IBM_s122
                                 + 0.758785 w_k10_INTC_s122
                                 + 1.0525 w_k10_KO_s122
                                 + 0.894623 w_k10_MAT_s122
                                 + 0.859127 w_k10_MSFT_s122
                                 + 0.965665 w_k10_PG_s122 >= 1
 z_k10_s1226_constraint:         var_10 + z_k10_s1226 + 1.0748 w_k10_ADM_s122
                                 + 1.56582 w_k10_AXP_s122
                                 + 1.51394 w_k10_DIS_s122
                                 + 1.24837 w_k10_FDX_s122
                                 + 1.52833 w_k10_IBM_s122
                                 + 1.38858 w_k10_INTC_s122
                                 + 1.47915 w_k10_KO_s122
                                 + 1.67817 w_k10_MAT_s122
                                 + 1.16563 w_k10_MSFT_s122
                                 + 1.22569 w_k10_PG_s122 >= 1
 z_k10_s1227_constraint:         var_10 + z_k10_s1227 + 0.814066 w_k10_ADM_s122
                                 + 0.718379 w_k10_AXP_s122
                                 + 0.49075 w_k10_DIS_s122
                                 + 0.662486 w_k10_FDX_s122
                                 + 0.723667 w_k10_IBM_s122
                                 + 0.612597 w_k10_INTC_s122
                                 + 0.658405 w_k10_KO_s122
                                 + 0.679877 w_k10_MAT_s122
                                 + 0.610254 w_k10_MSFT_s122
                                 + 0.775939 w_k10_PG_s122 >= 1
 z_k10_s1230_constraint:         var_10 + z_k10_s1230 + 1.18513 w_k10_ADM_s123
                                 + 0.895291 w_k10_AXP_s123
                                 + 1.05415 w_k10_DIS_s123
                                 + 1.03108 w_k10_FDX_s123
                                 + 0.738398 w_k10_IBM_s123
                                 + 0.841829 w_k10_INTC_s123
                                 + 0.927142 w_k10_KO_s123
                                 + 0.621196 w_k10_MAT_s123
                                 + 1.05124 w_k10_MSFT_s123
                                 + 0.826612 w_k10_PG_s123 >= 1
 z_k10_s1231_constraint:         var_10 + z_k10_s1231 + 1.44878 w_k10_ADM_s123
                                 + 0.959204 w_k10_AXP_s123
                                 + 1.30687 w_k10_DIS_s123
                                 + 1.18743 w_k10_FDX_s123
                                 + 0.860578 w_k10_IBM_s123
                                 + 1.08188 w_k10_INTC_s123
                                 + 1.16009 w_k10_KO_s123
                                 + 0.578169 w_k10_MAT_s123
                                 + 1.35396 w_k10_MSFT_s123
                                 + 1.18403 w_k10_PG_s123 >= 1
 z_k10_s1232_constraint:         var_10 + z_k10_s1232 + 1.33537 w_k10_ADM_s123
                                 + 1.2338 w_k10_AXP_s123
                                 + 1.19429 w_k10_DIS_s123
                                 + 1.10089 w_k10_FDX_s123
                                 + 1.35026 w_k10_IBM_s123
                                 + 1.17657 w_k10_INTC_s123
                                 + 1.37957 w_k10_KO_s123
                                 + 1.17193 w_k10_MAT_s123
                                 + 1.26171 w_k10_MSFT_s123
                                 + 1.09035 w_k10_PG_s123 >= 1
 z_k10_s1233_constraint:         var_10 + z_k10_s1233 + 1.07819 w_k10_ADM_s123
                                 + 0.755471 w_k10_AXP_s123
                                 + 0.600231 w_k10_DIS_s123
                                 + 0.675831 w_k10_FDX_s123
                                 + 0.831373 w_k10_IBM_s123
                                 + 0.771835 w_k10_INTC_s123
                                 + 0.687546 w_k10_KO_s123
                                 + 0.547027 w_k10_MAT_s123
                                 + 0.68012 w_k10_MSFT_s123
                                 + 0.738039 w_k10_PG_s123 >= 1
 z_k10_s1234_constraint:         var_10 + z_k10_s1234 + 1.33248 w_k10_ADM_s123
                                 + 1.08615 w_k10_AXP_s123
                                 + 0.791472 w_k10_DIS_s123
                                 + 1.07597 w_k10_FDX_s123
                                 + 0.705846 w_k10_IBM_s123
                                 + 0.844253 w_k10_INTC_s123
                                 + 0.868972 w_k10_KO_s123
                                 + 0.968087 w_k10_MAT_s123
                                 + 1.0596 w_k10_MSFT_s123
                                 + 1.07571 w_k10_PG_s123 >= 1
 z_k10_s1235_constraint:         var_10 + z_k10_s1235 + 1.18236 w_k10_ADM_s123
                                 + 0.710763 w_k10_AXP_s123
                                 + 0.727702 w_k10_DIS_s123
                                 + 0.909912 w_k10_FDX_s123
                                 + 0.659881 w_k10_IBM_s123
                                 + 0.909935 w_k10_INTC_s123
                                 + 0.802099 w_k10_KO_s123
                                 + 0.635573 w_k10_MAT_s123
                                 + 1.03813 w_k10_MSFT_s123
                                 + 0.952367 w_k10_PG_s123 >= 1
 z_k10_s1236_constraint:         var_10 + z_k10_s1236 + 1.11302 w_k10_ADM_s123
                                 + 0.928216 w_k10_AXP_s123
                                 + 0.925904 w_k10_DIS_s123
                                 + 0.947405 w_k10_FDX_s123
                                 + 1.05385 w_k10_IBM_s123
                                 + 0.906989 w_k10_INTC_s123
                                 + 1.02119 w_k10_KO_s123
                                 + 0.958107 w_k10_MAT_s123
                                 + 0.97523 w_k10_MSFT_s123
                                 + 0.955378 w_k10_PG_s123 >= 1
 z_k10_s1237_constraint:         var_10 + z_k10_s1237 + 1.34551 w_k10_ADM_s123
                                 + 1.26887 w_k10_AXP_s123
                                 + 1.33969 w_k10_DIS_s123
                                 + 1.33647 w_k10_FDX_s123
                                 + 1.01666 w_k10_IBM_s123
                                 + 0.970241 w_k10_INTC_s123
                                 + 1.16399 w_k10_KO_s123
                                 + 1.25675 w_k10_MAT_s123
                                 + 1.27254 w_k10_MSFT_s123
                                 + 1.20858 w_k10_PG_s123 >= 1
 z_k10_s1240_constraint:         var_10 + z_k10_s1240 + 1.15961 w_k10_ADM_s124
                                 + 1.18252 w_k10_AXP_s124
                                 + 0.918184 w_k10_DIS_s124
                                 + 1.0907 w_k10_FDX_s124
                                 + 1.26874 w_k10_IBM_s124
                                 + 0.980039 w_k10_INTC_s124
                                 + 1.00918 w_k10_KO_s124
                                 + 1.46438 w_k10_MAT_s124
                                 + 0.924113 w_k10_MSFT_s124
                                 + 1.05706 w_k10_PG_s124 >= 1
 z_k10_s1241_constraint:         var_10 + z_k10_s1241 + 1.09006 w_k10_ADM_s124
                                 + 0.907677 w_k10_AXP_s124
                                 + 0.868262 w_k10_DIS_s124
                                 + 0.991314 w_k10_FDX_s124
                                 + 0.913635 w_k10_IBM_s124
                                 + 1.01794 w_k10_INTC_s124
                                 + 1.05184 w_k10_KO_s124
                                 + 0.906535 w_k10_MAT_s124
                                 + 0.948421 w_k10_MSFT_s124
                                 + 0.978816 w_k10_PG_s124 >= 1
 z_k10_s1242_constraint:         var_10 + z_k10_s1242 + 1.11434 w_k10_ADM_s124
                                 + 0.755223 w_k10_AXP_s124
                                 + 0.736822 w_k10_DIS_s124
                                 + 0.797617 w_k10_FDX_s124
                                 + 1.18595 w_k10_IBM_s124
                                 + 0.831054 w_k10_INTC_s124
                                 + 1.03928 w_k10_KO_s124
                                 + 0.994136 w_k10_MAT_s124
                                 + 0.739837 w_k10_MSFT_s124
                                 + 1.00633 w_k10_PG_s124 >= 1
 z_k10_s1243_constraint:         var_10 + z_k10_s1243 + 1.3367 w_k10_ADM_s124
                                 + 0.964123 w_k10_AXP_s124
                                 + 1.28405 w_k10_DIS_s124
                                 + 1.20768 w_k10_FDX_s124
                                 + 0.868817 w_k10_IBM_s124
                                 + 1.15635 w_k10_INTC_s124
                                 + 1.11879 w_k10_KO_s124
                                 + 1.01641 w_k10_MAT_s124
                                 + 1.24663 w_k10_MSFT_s124
                                 + 1.16406 w_k10_PG_s124 >= 1
 z_k10_s1244_constraint:         var_10 + z_k10_s1244 + 1.09123 w_k10_ADM_s124
                                 + 0.727559 w_k10_AXP_s124
                                 + 0.708384 w_k10_DIS_s124
                                 + 0.836431 w_k10_FDX_s124
                                 + 0.680499 w_k10_IBM_s124
                                 + 0.77359 w_k10_INTC_s124
                                 + 0.743123 w_k10_KO_s124
                                 + 0.854602 w_k10_MAT_s124
                                 + 0.796174 w_k10_MSFT_s124
                                 + 0.92702 w_k10_PG_s124 >= 1
 z_k10_s1245_constraint:         var_10 + z_k10_s1245 + 1.20521 w_k10_ADM_s124
                                 + 1.53417 w_k10_AXP_s124
                                 + 1.489 w_k10_DIS_s124 + 1.3942 w_k10_FDX_s124
                                 + 1.2552 w_k10_IBM_s124
                                 + 1.19345 w_k10_INTC_s124
                                 + 1.23965 w_k10_KO_s124
                                 + 1.23296 w_k10_MAT_s124
                                 + 1.19141 w_k10_MSFT_s124
                                 + 1.07061 w_k10_PG_s124 >= 1
 z_k10_s1246_constraint:         var_10 + z_k10_s1246 + 0.683099 w_k10_ADM_s124
                                 + 0.797237 w_k10_AXP_s124
                                 + 0.588373 w_k10_DIS_s124
                                 + 0.777202 w_k10_FDX_s124
                                 + 0.864794 w_k10_IBM_s124
                                 + 0.851389 w_k10_INTC_s124
                                 + 0.770149 w_k10_KO_s124
                                 + 1.08412 w_k10_MAT_s124
                                 + 0.759215 w_k10_MSFT_s124
                                 + 0.873354 w_k10_PG_s124 >= 1
 z_k10_s1247_constraint:         var_10 + z_k10_s1247 + 1.00679 w_k10_ADM_s124
                                 + 1.03527 w_k10_AXP_s124
                                 + 1.38209 w_k10_DIS_s124
                                 + 1.0557 w_k10_FDX_s124
                                 + 1.55629 w_k10_IBM_s124
                                 + 0.968113 w_k10_INTC_s124
                                 + 1.31208 w_k10_KO_s124
                                 + 2.01008 w_k10_MAT_s124
                                 + 1.10781 w_k10_MSFT_s124
                                 + 1.23845 w_k10_PG_s124 >= 1
 z_k10_s1250_constraint:         var_10 + z_k10_s1250 + 0.867494 w_k10_ADM_s125
                                 + 0.819679 w_k10_AXP_s125
                                 + 1.10737 w_k10_DIS_s125
                                 + 0.964371 w_k10_FDX_s125
                                 + 1.06532 w_k10_IBM_s125
                                 + 1.10445 w_k10_INTC_s125
                                 + 0.964387 w_k10_KO_s125
                                 + 0.837622 w_k10_MAT_s125
                                 + 0.976395 w_k10_MSFT_s125
                                 + 1.04322 w_k10_PG_s125 >= 1
 z_k10_s1251_constraint:         var_10 + z_k10_s1251 + 1.16897 w_k10_ADM_s125
                                 + 0.827902 w_k10_AXP_s125
                                 + 1.40619 w_k10_DIS_s125
                                 + 1.02497 w_k10_FDX_s125
                                 + 1.08351 w_k10_IBM_s125
                                 + 1.18563 w_k10_INTC_s125
                                 + 1.21106 w_k10_KO_s125
                                 + 1.06377 w_k10_MAT_s125
                                 + 1.22846 w_k10_MSFT_s125
                                 + 1.09697 w_k10_PG_s125 >= 1
 z_k10_s1252_constraint:         var_10 + z_k10_s1252 + 1.11126 w_k10_ADM_s125
                                 + 1.01323 w_k10_AXP_s125
                                 + 1.03062 w_k10_DIS_s125
                                 + 1.17341 w_k10_FDX_s125
                                 + 0.772103 w_k10_IBM_s125
                                 + 0.965608 w_k10_INTC_s125
                                 + 0.786043 w_k10_KO_s125
                                 + 0.84375 w_k10_MAT_s125
                                 + 0.942024 w_k10_MSFT_s125
                                 + 1.00601 w_k10_PG_s125 >= 1
 z_k10_s1253_constraint:         var_10 + z_k10_s1253 + 0.87815 w_k10_ADM_s125
                                 + 0.579938 w_k10_AXP_s125
                                 + 0.628554 w_k10_DIS_s125
                                 + 0.733395 w_k10_FDX_s125
                                 + 1.07694 w_k10_IBM_s125
                                 + 0.956224 w_k10_INTC_s125
                                 + 0.786623 w_k10_KO_s125
                                 + 0.568147 w_k10_MAT_s125
                                 + 0.86334 w_k10_MSFT_s125
                                 + 0.853206 w_k10_PG_s125 >= 1
 z_k10_s1254_constraint:         var_10 + z_k10_s1254 + 0.900553 w_k10_ADM_s125
                                 + 0.658178 w_k10_AXP_s125
                                 + 0.765441 w_k10_DIS_s125
                                 + 0.759956 w_k10_FDX_s125
                                 + 0.739891 w_k10_IBM_s125
                                 + 0.862115 w_k10_INTC_s125
                                 + 0.786836 w_k10_KO_s125
                                 + 0.523404 w_k10_MAT_s125
                                 + 0.77372 w_k10_MSFT_s125
                                 + 0.791418 w_k10_PG_s125 >= 1
 z_k10_s1255_constraint:         var_10 + z_k10_s1255 + 0.691811 w_k10_ADM_s125
                                 + 1.2572 w_k10_AXP_s125
                                 + 1.68662 w_k10_DIS_s125
                                 + 1.2815 w_k10_FDX_s125
                                 + 1.67703 w_k10_IBM_s125
                                 + 1.07757 w_k10_INTC_s125
                                 + 1.23596 w_k10_KO_s125
                                 + 1.7821 w_k10_MAT_s125
                                 + 1.19143 w_k10_MSFT_s125
                                 + 0.954208 w_k10_PG_s125 >= 1
 z_k10_s1256_constraint:         var_10 + z_k10_s1256 + 1.23092 w_k10_ADM_s125
                                 + 0.870124 w_k10_AXP_s125
                                 + 0.997678 w_k10_DIS_s125
                                 + 1.07414 w_k10_FDX_s125
                                 + 1.27316 w_k10_IBM_s125
                                 + 1.01752 w_k10_INTC_s125
                                 + 1.149 w_k10_KO_s125 + 1.06467 w_k10_MAT_s125
                                 + 0.98354 w_k10_MSFT_s125
                                 + 1.04354 w_k10_PG_s125 >= 1
 z_k10_s1257_constraint:         var_10 + z_k10_s1257 + 1.40455 w_k10_ADM_s125
                                 + 1.31374 w_k10_AXP_s125
                                 + 1.53438 w_k10_DIS_s125
                                 + 1.56297 w_k10_FDX_s125
                                 + 1.09357 w_k10_IBM_s125
                                 + 1.36109 w_k10_INTC_s125
                                 + 1.10993 w_k10_KO_s125
                                 + 0.868131 w_k10_MAT_s125
                                 + 1.43453 w_k10_MSFT_s125
                                 + 1.10923 w_k10_PG_s125 >= 1
 z_k10_s1260_constraint:         var_10 + z_k10_s1260 + 0.839325 w_k10_ADM_s126
                                 + 0.974842 w_k10_AXP_s126
                                 + 1.02727 w_k10_DIS_s126
                                 + 0.88275 w_k10_FDX_s126
                                 + 0.9353 w_k10_IBM_s126
                                 + 0.810974 w_k10_INTC_s126
                                 + 0.972043 w_k10_KO_s126
                                 + 1.20909 w_k10_MAT_s126
                                 + 0.848031 w_k10_MSFT_s126
                                 + 0.84284 w_k10_PG_s126 >= 1
 z_k10_s1261_constraint:         var_10 + z_k10_s1261 + 0.829508 w_k10_ADM_s126
                                 + 0.889894 w_k10_AXP_s126
                                 + 0.958672 w_k10_DIS_s126
                                 + 1.0497 w_k10_FDX_s126
                                 + 0.718795 w_k10_IBM_s126
                                 + 1.026 w_k10_INTC_s126
                                 + 0.756314 w_k10_KO_s126
                                 + 0.846771 w_k10_MAT_s126
                                 + 1.06583 w_k10_MSFT_s126
                                 + 0.881526 w_k10_PG_s126 >= 1
 z_k10_s1262_constraint:         var_10 + z_k10_s1262 + 0.98634 w_k10_ADM_s126
                                 + 1.07198 w_k10_AXP_s126
                                 + 1.20553 w_k10_DIS_s126
                                 + 1.1706 w_k10_FDX_s126
                                 + 0.661348 w_k10_IBM_s126
                                 + 1.09577 w_k10_INTC_s126
                                 + 0.894298 w_k10_KO_s126
                                 + 1.12311 w_k10_MAT_s126
                                 + 1.26225 w_k10_MSFT_s126
                                 + 0.990152 w_k10_PG_s126 >= 1
 z_k10_s1263_constraint:         var_10 + z_k10_s1263 + 0.614481 w_k10_ADM_s126
                                 + 0.959344 w_k10_AXP_s126
                                 + 0.847451 w_k10_DIS_s126
                                 + 0.823112 w_k10_FDX_s126
                                 + 1.25206 w_k10_IBM_s126
                                 + 0.926836 w_k10_INTC_s126
                                 + 1.11803 w_k10_KO_s126
                                 + 1.98298 w_k10_MAT_s126
                                 + 0.88144 w_k10_MSFT_s126
                                 + 1.01972 w_k10_PG_s126 >= 1
 z_k10_s1264_constraint:         var_10 + z_k10_s1264 + 1.07578 w_k10_ADM_s126
                                 + 1.15194 w_k10_AXP_s126
                                 + 1.14019 w_k10_DIS_s126
                                 + 1.02123 w_k10_FDX_s126
                                 + 1.27278 w_k10_IBM_s126
                                 + 1.19617 w_k10_INTC_s126
                                 + 1.05053 w_k10_KO_s126
                                 + 1.34657 w_k10_MAT_s126
                                 + 1.1759 w_k10_MSFT_s126
                                 + 0.961386 w_k10_PG_s126 >= 1
 z_k10_s1265_constraint:         var_10 + z_k10_s1265 + 1.02747 w_k10_ADM_s126
                                 + 1.49089 w_k10_AXP_s126
                                 + 1.35466 w_k10_DIS_s126
                                 + 1.13354 w_k10_FDX_s126
                                 + 0.89736 w_k10_IBM_s126
                                 + 1.01635 w_k10_INTC_s126
                                 + 1.11614 w_k10_KO_s126
                                 + 1.69467 w_k10_MAT_s126
                                 + 1.19839 w_k10_MSFT_s126
                                 + 1.06443 w_k10_PG_s126 >= 1
 z_k10_s1266_constraint:         var_10 + z_k10_s1266 + 1.06874 w_k10_ADM_s126
                                 + 1.6315 w_k10_AXP_s126
                                 + 1.54965 w_k10_DIS_s126
                                 + 1.21988 w_k10_FDX_s126
                                 + 1.35274 w_k10_IBM_s126
                                 + 1.47385 w_k10_INTC_s126
                                 + 1.32903 w_k10_KO_s126
                                 + 1.61135 w_k10_MAT_s126
                                 + 1.19179 w_k10_MSFT_s126
                                 + 1.2038 w_k10_PG_s126 >= 1
 z_k10_s1267_constraint:         var_10 + z_k10_s1267 + 0.932673 w_k10_ADM_s126
                                 + 0.796694 w_k10_AXP_s126
                                 + 0.640389 w_k10_DIS_s126
                                 + 0.772463 w_k10_FDX_s126
                                 + 0.74999 w_k10_IBM_s126
                                 + 0.746126 w_k10_INTC_s126
                                 + 0.79406 w_k10_KO_s126
                                 + 1.27314 w_k10_MAT_s126
                                 + 0.775178 w_k10_MSFT_s126
                                 + 0.738293 w_k10_PG_s126 >= 1
 z_k10_s1270_constraint:         var_10 + z_k10_s1270 + 1.49332 w_k10_ADM_s127
                                 + 1.91824 w_k10_AXP_s127
                                 + 1.79552 w_k10_DIS_s127
                                 + 1.41246 w_k10_FDX_s127
                                 + 1.28469 w_k10_IBM_s127
                                 + 1.0984 w_k10_INTC_s127
                                 + 1.43281 w_k10_KO_s127
                                 + 1.92913 w_k10_MAT_s127
                                 + 1.2806 w_k10_MSFT_s127
                                 + 1.35836 w_k10_PG_s127 >= 1
 z_k10_s1271_constraint:         var_10 + z_k10_s1271 + 1.31311 w_k10_ADM_s127
                                 + 1.07094 w_k10_AXP_s127
                                 + 1.21269 w_k10_DIS_s127
                                 + 1.21725 w_k10_FDX_s127
                                 + 0.777562 w_k10_IBM_s127
                                 + 1.03674 w_k10_INTC_s127
                                 + 1.0681 w_k10_KO_s127
                                 + 0.920529 w_k10_MAT_s127
                                 + 1.04757 w_k10_MSFT_s127
                                 + 0.942019 w_k10_PG_s127 >= 1
 z_k10_s1272_constraint:         var_10 + z_k10_s1272 + 0.774513 w_k10_ADM_s127
                                 + 0.73627 w_k10_AXP_s127
                                 + 0.485884 w_k10_DIS_s127
                                 + 0.736257 w_k10_FDX_s127
                                 + 0.890932 w_k10_IBM_s127
                                 + 0.716901 w_k10_INTC_s127
                                 + 0.658331 w_k10_KO_s127
                                 + 0.791219 w_k10_MAT_s127
                                 + 0.741332 w_k10_MSFT_s127
                                 + 0.724839 w_k10_PG_s127 >= 1
 z_k10_s1273_constraint:         var_10 + z_k10_s1273 + 1.01384 w_k10_ADM_s127
                                 + 0.752667 w_k10_AXP_s127
                                 + 0.843489 w_k10_DIS_s127
                                 + 0.940152 w_k10_FDX_s127
                                 + 0.906172 w_k10_IBM_s127
                                 + 0.824943 w_k10_INTC_s127
                                 + 0.907184 w_k10_KO_s127
                                 + 0.782244 w_k10_MAT_s127
                                 + 0.951935 w_k10_MSFT_s127
                                 + 0.980944 w_k10_PG_s127 >= 1
 z_k10_s1274_constraint:         var_10 + z_k10_s1274 + 1.08265 w_k10_ADM_s127
                                 + 1.15584 w_k10_AXP_s127
                                 + 0.763573 w_k10_DIS_s127
                                 + 0.969965 w_k10_FDX_s127
                                 + 0.957538 w_k10_IBM_s127
                                 + 0.890327 w_k10_INTC_s127
                                 + 0.888567 w_k10_KO_s127
                                 + 0.808584 w_k10_MAT_s127
                                 + 0.861565 w_k10_MSFT_s127
                                 + 0.986036 w_k10_PG_s127 >= 1
 z_k10_s1275_constraint:         var_10 + z_k10_s1275 + 1.30308 w_k10_ADM_s127
                                 + 1.30381 w_k10_AXP_s127
                                 + 1.45368 w_k10_DIS_s127
                                 + 1.38958 w_k10_FDX_s127
                                 + 0.988561 w_k10_IBM_s127
                                 + 1.3091 w_k10_INTC_s127
                                 + 1.20691 w_k10_KO_s127
                                 + 0.742731 w_k10_MAT_s127
                                 + 1.30078 w_k10_MSFT_s127
                                 + 1.17318 w_k10_PG_s127 >= 1
 z_k10_s1276_constraint:         var_10 + z_k10_s1276 + 1.4404 w_k10_ADM_s127
                                 + 0.960678 w_k10_AXP_s127
                                 + 1.49423 w_k10_DIS_s127
                                 + 1.10459 w_k10_FDX_s127
                                 + 1.50124 w_k10_IBM_s127
                                 + 1.05954 w_k10_INTC_s127
                                 + 1.3866 w_k10_KO_s127
                                 + 1.17823 w_k10_MAT_s127
                                 + 1.24626 w_k10_MSFT_s127
                                 + 1.15378 w_k10_PG_s127 >= 1
 z_k10_s1277_constraint:         var_10 + z_k10_s1277 + 1.3491 w_k10_ADM_s127
                                 + 1.06825 w_k10_AXP_s127
                                 + 0.975149 w_k10_DIS_s127
                                 + 1.07997 w_k10_FDX_s127
                                 + 1.07055 w_k10_IBM_s127
                                 + 1.00842 w_k10_INTC_s127
                                 + 1.06528 w_k10_KO_s127
                                 + 0.917005 w_k10_MAT_s127
                                 + 0.90274 w_k10_MSFT_s127
                                 + 0.877866 w_k10_PG_s127 >= 1
 z_k10_s1300_constraint:         var_10 + z_k10_s1300 + 0.865518 w_k10_ADM_s130
                                 + 1.03737 w_k10_AXP_s130
                                 + 1.05112 w_k10_DIS_s130
                                 + 1.00342 w_k10_FDX_s130
                                 + 1.16758 w_k10_IBM_s130
                                 + 1.02691 w_k10_INTC_s130
                                 + 1.13413 w_k10_KO_s130
                                 + 1.29073 w_k10_MAT_s130
                                 + 1.04585 w_k10_MSFT_s130
                                 + 1.05495 w_k10_PG_s130 >= 1
 z_k10_s1301_constraint:         var_10 + z_k10_s1301 + 0.951429 w_k10_ADM_s130
                                 + 1.13232 w_k10_AXP_s130
                                 + 1.2094 w_k10_DIS_s130
                                 + 1.11569 w_k10_FDX_s130
                                 + 0.913489 w_k10_IBM_s130
                                 + 1.02933 w_k10_INTC_s130
                                 + 1.00669 w_k10_KO_s130
                                 + 1.07753 w_k10_MAT_s130
                                 + 1.00249 w_k10_MSFT_s130
                                 + 0.935646 w_k10_PG_s130 >= 1
 z_k10_s1302_constraint:         var_10 + z_k10_s1302 + 0.811496 w_k10_ADM_s130
                                 + 0.848991 w_k10_AXP_s130
                                 + 0.789325 w_k10_DIS_s130
                                 + 0.908935 w_k10_FDX_s130
                                 + 0.822701 w_k10_IBM_s130
                                 + 1.02289 w_k10_INTC_s130
                                 + 0.776231 w_k10_KO_s130
                                 + 0.656006 w_k10_MAT_s130
                                 + 0.865809 w_k10_MSFT_s130
                                 + 0.786241 w_k10_PG_s130 >= 1
 z_k10_s1303_constraint:         var_10 + z_k10_s1303 + 0.794294 w_k10_ADM_s130
                                 + 1.17976 w_k10_AXP_s130
                                 + 1.051 w_k10_DIS_s130
                                 + 0.88076 w_k10_FDX_s130
                                 + 1.70082 w_k10_IBM_s130
                                 + 0.908519 w_k10_INTC_s130
                                 + 1.21384 w_k10_KO_s130
                                 + 1.92923 w_k10_MAT_s130
                                 + 0.816189 w_k10_MSFT_s130
                                 + 1.02983 w_k10_PG_s130 >= 1
 z_k10_s1304_constraint:         var_10 + z_k10_s1304 + 1.01959 w_k10_ADM_s130
                                 + 1.43758 w_k10_AXP_s130
                                 + 1.62546 w_k10_DIS_s130
                                 + 1.29699 w_k10_FDX_s130
                                 + 1.12328 w_k10_IBM_s130
                                 + 1.03856 w_k10_INTC_s130
                                 + 1.47751 w_k10_KO_s130
                                 + 1.48566 w_k10_MAT_s130
                                 + 1.34498 w_k10_MSFT_s130
                                 + 1.403 w_k10_PG_s130 >= 1
 z_k10_s1305_constraint:         var_10 + z_k10_s1305 + 1.23002 w_k10_ADM_s130
                                 + 1.35184 w_k10_AXP_s130
                                 + 1.66483 w_k10_DIS_s130
                                 + 1.44025 w_k10_FDX_s130
                                 + 1.43972 w_k10_IBM_s130
                                 + 1.34297 w_k10_INTC_s130
                                 + 1.29144 w_k10_KO_s130
                                 + 1.44204 w_k10_MAT_s130
                                 + 1.26625 w_k10_MSFT_s130
                                 + 1.19637 w_k10_PG_s130 >= 1
 z_k10_s1306_constraint:         var_10 + z_k10_s1306 + 0.746713 w_k10_ADM_s130
                                 + 0.7679 w_k10_AXP_s130
                                 + 0.716901 w_k10_DIS_s130
                                 + 0.897404 w_k10_FDX_s130
                                 + 0.821707 w_k10_IBM_s130
                                 + 0.958939 w_k10_INTC_s130
                                 + 1.02112 w_k10_KO_s130
                                 + 0.909651 w_k10_MAT_s130
                                 + 0.939197 w_k10_MSFT_s130
                                 + 0.928025 w_k10_PG_s130 >= 1
 z_k10_s1307_constraint:         var_10 + z_k10_s1307 + 0.730002 w_k10_ADM_s130
                                 + 0.75933 w_k10_AXP_s130
                                 + 0.583986 w_k10_DIS_s130
                                 + 0.721591 w_k10_FDX_s130
                                 + 1.10164 w_k10_IBM_s130
                                 + 0.718954 w_k10_INTC_s130
                                 + 0.779561 w_k10_KO_s130
                                 + 0.931384 w_k10_MAT_s130
                                 + 0.61524 w_k10_MSFT_s130
                                 + 0.750998 w_k10_PG_s130 >= 1
 z_k10_s1310_constraint:         var_10 + z_k10_s1310 + 1.43143 w_k10_ADM_s131
                                 + 0.813299 w_k10_AXP_s131
                                 + 0.736468 w_k10_DIS_s131
                                 + 0.841744 w_k10_FDX_s131
                                 + 1.02931 w_k10_IBM_s131
                                 + 1.21217 w_k10_INTC_s131
                                 + 1.11321 w_k10_KO_s131
                                 + 0.529496 w_k10_MAT_s131
                                 + 1.03639 w_k10_MSFT_s131
                                 + 1.00084 w_k10_PG_s131 >= 1
 z_k10_s1311_constraint:         var_10 + z_k10_s1311 + 1.02389 w_k10_ADM_s131
                                 + 0.987202 w_k10_AXP_s131
                                 + 0.797699 w_k10_DIS_s131
                                 + 0.792333 w_k10_FDX_s131
                                 + 0.754228 w_k10_IBM_s131
                                 + 0.944347 w_k10_INTC_s131
                                 + 0.86004 w_k10_KO_s131
                                 + 0.511002 w_k10_MAT_s131
                                 + 0.783818 w_k10_MSFT_s131
                                 + 0.775658 w_k10_PG_s131 >= 1
 z_k10_s1312_constraint:         var_10 + z_k10_s1312 + 1.38685 w_k10_ADM_s131
                                 + 1.05556 w_k10_AXP_s131
                                 + 0.821475 w_k10_DIS_s131
                                 + 0.878492 w_k10_FDX_s131
                                 + 1.1942 w_k10_IBM_s131
                                 + 0.898663 w_k10_INTC_s131
                                 + 0.993854 w_k10_KO_s131
                                 + 0.684702 w_k10_MAT_s131
                                 + 0.909024 w_k10_MSFT_s131
                                 + 0.913951 w_k10_PG_s131 >= 1
 z_k10_s1313_constraint:         var_10 + z_k10_s1313 + 2.10398 w_k10_ADM_s131
                                 + 0.954034 w_k10_AXP_s131
                                 + 1.45505 w_k10_DIS_s131
                                 + 1.2535 w_k10_FDX_s131
                                 + 0.883265 w_k10_IBM_s131
                                 + 1.34336 w_k10_INTC_s131
                                 + 1.31281 w_k10_KO_s131
                                 + 0.534695 w_k10_MAT_s131
                                 + 1.43616 w_k10_MSFT_s131
                                 + 1.24212 w_k10_PG_s131 >= 1
 z_k10_s1314_constraint:         var_10 + z_k10_s1314 + 1.40231 w_k10_ADM_s131
                                 + 0.699008 w_k10_AXP_s131
                                 + 0.860892 w_k10_DIS_s131
                                 + 0.846027 w_k10_FDX_s131
                                 + 0.688865 w_k10_IBM_s131
                                 + 0.857935 w_k10_INTC_s131
                                 + 0.943281 w_k10_KO_s131
                                 + 0.631072 w_k10_MAT_s131
                                 + 0.954192 w_k10_MSFT_s131
                                 + 0.8153 w_k10_PG_s131 >= 1
 z_k10_s1315_constraint:         var_10 + z_k10_s1315 + 1.38496 w_k10_ADM_s131
                                 + 1.34402 w_k10_AXP_s131
                                 + 1.51851 w_k10_DIS_s131
                                 + 1.212 w_k10_FDX_s131
                                 + 1.43572 w_k10_IBM_s131
                                 + 1.20117 w_k10_INTC_s131
                                 + 1.3623 w_k10_KO_s131
                                 + 0.987224 w_k10_MAT_s131
                                 + 1.2038 w_k10_MSFT_s131
                                 + 1.15244 w_k10_PG_s131 >= 1
 z_k10_s1316_constraint:         var_10 + z_k10_s1316 + 1.41333 w_k10_ADM_s131
                                 + 0.641175 w_k10_AXP_s131
                                 + 0.693689 w_k10_DIS_s131
                                 + 0.827966 w_k10_FDX_s131
                                 + 0.580619 w_k10_IBM_s131
                                 + 0.880356 w_k10_INTC_s131
                                 + 0.669123 w_k10_KO_s131
                                 + 0.293643 w_k10_MAT_s131
                                 + 0.755498 w_k10_MSFT_s131
                                 + 0.79306 w_k10_PG_s131 >= 1
 z_k10_s1317_constraint:         var_10 + z_k10_s1317 + 1.72262 w_k10_ADM_s131
                                 + 0.848088 w_k10_AXP_s131
                                 + 1.23556 w_k10_DIS_s131
                                 + 0.892944 w_k10_FDX_s131
                                 + 0.906535 w_k10_IBM_s131
                                 + 0.883985 w_k10_INTC_s131
                                 + 1.13186 w_k10_KO_s131
                                 + 0.564233 w_k10_MAT_s131
                                 + 1.03384 w_k10_MSFT_s131
                                 + 0.994407 w_k10_PG_s131 >= 1
 z_k10_s1320_constraint:         var_10 + z_k10_s1320 + 0.828273 w_k10_ADM_s132
                                 + 0.847909 w_k10_AXP_s132
                                 + 1.02417 w_k10_DIS_s132
                                 + 0.983606 w_k10_FDX_s132
                                 + 0.922915 w_k10_IBM_s132
                                 + 1.03523 w_k10_INTC_s132
                                 + 1.02342 w_k10_KO_s132
                                 + 0.876727 w_k10_MAT_s132
                                 + 0.928268 w_k10_MSFT_s132
                                 + 0.854878 w_k10_PG_s132 >= 1
 z_k10_s1321_constraint:         var_10 + z_k10_s1321 + 1.28909 w_k10_ADM_s132
                                 + 1.60332 w_k10_AXP_s132
                                 + 1.31915 w_k10_DIS_s132
                                 + 1.27782 w_k10_FDX_s132
                                 + 1.09871 w_k10_IBM_s132
                                 + 1.2457 w_k10_INTC_s132
                                 + 1.37247 w_k10_KO_s132
                                 + 1.63912 w_k10_MAT_s132
                                 + 1.20169 w_k10_MSFT_s132
                                 + 1.21233 w_k10_PG_s132 >= 1
 z_k10_s1322_constraint:         var_10 + z_k10_s1322 + 0.784021 w_k10_ADM_s132
                                 + 0.933013 w_k10_AXP_s132
                                 + 1.28253 w_k10_DIS_s132
                                 + 0.936686 w_k10_FDX_s132
                                 + 1.3617 w_k10_IBM_s132
                                 + 1.05587 w_k10_INTC_s132
                                 + 1.44422 w_k10_KO_s132
                                 + 1.35849 w_k10_MAT_s132
                                 + 1.23922 w_k10_MSFT_s132
                                 + 1.06563 w_k10_PG_s132 >= 1
 z_k10_s1323_constraint:         var_10 + z_k10_s1323 + 0.88136 w_k10_ADM_s132
                                 + 0.776856 w_k10_AXP_s132
                                 + 0.628587 w_k10_DIS_s132
                                 + 0.782105 w_k10_FDX_s132
                                 + 0.799609 w_k10_IBM_s132
                                 + 0.961115 w_k10_INTC_s132
                                 + 0.758843 w_k10_KO_s132
                                 + 0.492119 w_k10_MAT_s132
                                 + 0.729371 w_k10_MSFT_s132
                                 + 0.760239 w_k10_PG_s132 >= 1
 z_k10_s1324_constraint:         var_10 + z_k10_s1324 + 0.809966 w_k10_ADM_s132
                                 + 1.30331 w_k10_AXP_s132
                                 + 0.959469 w_k10_DIS_s132
                                 + 1.06746 w_k10_FDX_s132
                                 + 1.11139 w_k10_IBM_s132
                                 + 0.992996 w_k10_INTC_s132
                                 + 1.19309 w_k10_KO_s132
                                 + 1.41663 w_k10_MAT_s132
                                 + 1.11018 w_k10_MSFT_s132
                                 + 1.01503 w_k10_PG_s132 >= 1
 z_k10_s1325_constraint:         var_10 + z_k10_s1325 + 1.29431 w_k10_ADM_s132
                                 + 1.37807 w_k10_AXP_s132
                                 + 1.46229 w_k10_DIS_s132
                                 + 1.08125 w_k10_FDX_s132
                                 + 1.55613 w_k10_IBM_s132
                                 + 1.18684 w_k10_INTC_s132
                                 + 1.57435 w_k10_KO_s132
                                 + 1.28992 w_k10_MAT_s132
                                 + 1.20761 w_k10_MSFT_s132
                                 + 1.14309 w_k10_PG_s132 >= 1
 z_k10_s1326_constraint:         var_10 + z_k10_s1326 + 0.993751 w_k10_ADM_s132
                                 + 0.912674 w_k10_AXP_s132
                                 + 0.902964 w_k10_DIS_s132
                                 + 1.03757 w_k10_FDX_s132
                                 + 0.788286 w_k10_IBM_s132
                                 + 1.0214 w_k10_INTC_s132
                                 + 0.909155 w_k10_KO_s132
                                 + 0.571276 w_k10_MAT_s132
                                 + 1.02089 w_k10_MSFT_s132
                                 + 0.909519 w_k10_PG_s132 >= 1
 z_k10_s1327_constraint:         var_10 + z_k10_s1327 + 0.891974 w_k10_ADM_s132
                                 + 0.941626 w_k10_AXP_s132
                                 + 0.903699 w_k10_DIS_s132
                                 + 0.882373 w_k10_FDX_s132
                                 + 1.28977 w_k10_IBM_s132
                                 + 1.02811 w_k10_INTC_s132
                                 + 1.08216 w_k10_KO_s132
                                 + 1.10403 w_k10_MAT_s132
                                 + 0.820004 w_k10_MSFT_s132
                                 + 0.827389 w_k10_PG_s132 >= 1
 z_k10_s1330_constraint:         var_10 + z_k10_s1330 + 1.20838 w_k10_ADM_s133
                                 + 1.20163 w_k10_AXP_s133
                                 + 1.03701 w_k10_DIS_s133
                                 + 1.06271 w_k10_FDX_s133
                                 + 1.26767 w_k10_IBM_s133
                                 + 0.904732 w_k10_INTC_s133
                                 + 1.24432 w_k10_KO_s133
                                 + 1.13113 w_k10_MAT_s133
                                 + 1.16031 w_k10_MSFT_s133
                                 + 0.925884 w_k10_PG_s133 >= 1
 z_k10_s1331_constraint:         var_10 + z_k10_s1331 + 1.66776 w_k10_ADM_s133
                                 + 1.54719 w_k10_AXP_s133
                                 + 1.5382 w_k10_DIS_s133
                                 + 1.20261 w_k10_FDX_s133
                                 + 1.70861 w_k10_IBM_s133
                                 + 1.13557 w_k10_INTC_s133
                                 + 1.61896 w_k10_KO_s133
                                 + 1.93622 w_k10_MAT_s133
                                 + 1.16775 w_k10_MSFT_s133
                                 + 1.20666 w_k10_PG_s133 >= 1
 z_k10_s1332_constraint:         var_10 + z_k10_s1332 + 1.85948 w_k10_ADM_s133
                                 + 1.64561 w_k10_AXP_s133
                                 + 1.5409 w_k10_DIS_s133
                                 + 1.30815 w_k10_FDX_s133
                                 + 1.29291 w_k10_IBM_s133
                                 + 0.997069 w_k10_INTC_s133
                                 + 1.37969 w_k10_KO_s133
                                 + 1.63563 w_k10_MAT_s133
                                 + 1.28489 w_k10_MSFT_s133
                                 + 1.20772 w_k10_PG_s133 >= 1
 z_k10_s1333_constraint:         var_10 + z_k10_s1333 + 1.11657 w_k10_ADM_s133
                                 + 0.760767 w_k10_AXP_s133
                                 + 0.768656 w_k10_DIS_s133
                                 + 0.79111 w_k10_FDX_s133
                                 + 0.969122 w_k10_IBM_s133
                                 + 0.759208 w_k10_INTC_s133
                                 + 0.919207 w_k10_KO_s133
                                 + 1.05527 w_k10_MAT_s133
                                 + 0.88905 w_k10_MSFT_s133
                                 + 0.823596 w_k10_PG_s133 >= 1
 z_k10_s1334_constraint:         var_10 + z_k10_s1334 + 1.2164 w_k10_ADM_s133
                                 + 1.08406 w_k10_AXP_s133
                                 + 1.0456 w_k10_DIS_s133
                                 + 1.14896 w_k10_FDX_s133
                                 + 0.942226 w_k10_IBM_s133
                                 + 0.913602 w_k10_INTC_s133
                                 + 0.977678 w_k10_KO_s133
                                 + 1.0694 w_k10_MAT_s133
                                 + 0.977359 w_k10_MSFT_s133
                                 + 0.865389 w_k10_PG_s133 >= 1
 z_k10_s1335_constraint:         var_10 + z_k10_s1335 + 1.1476 w_k10_ADM_s133
                                 + 1.33249 w_k10_AXP_s133
                                 + 1.39668 w_k10_DIS_s133
                                 + 1.45553 w_k10_FDX_s133
                                 + 0.768343 w_k10_IBM_s133
                                 + 1.41414 w_k10_INTC_s133
                                 + 1.06646 w_k10_KO_s133
                                 + 0.588756 w_k10_MAT_s133
                                 + 1.15583 w_k10_MSFT_s133
                                 + 1.03732 w_k10_PG_s133 >= 1
 z_k10_s1336_constraint:         var_10 + z_k10_s1336 + 0.935653 w_k10_ADM_s133
                                 + 0.79152 w_k10_AXP_s133
                                 + 0.703192 w_k10_DIS_s133
                                 + 0.892799 w_k10_FDX_s133
                                 + 0.653163 w_k10_IBM_s133
                                 + 0.689243 w_k10_INTC_s133
                                 + 0.691408 w_k10_KO_s133
                                 + 0.486244 w_k10_MAT_s133
                                 + 0.693435 w_k10_MSFT_s133
                                 + 0.733855 w_k10_PG_s133 >= 1
 z_k10_s1337_constraint:         var_10 + z_k10_s1337 + 1.03844 w_k10_ADM_s133
                                 + 0.9867 w_k10_AXP_s133
                                 + 0.88537 w_k10_DIS_s133
                                 + 0.962433 w_k10_FDX_s133
                                 + 0.852584 w_k10_IBM_s133
                                 + 0.845488 w_k10_INTC_s133
                                 + 0.865113 w_k10_KO_s133
                                 + 0.835789 w_k10_MAT_s133
                                 + 0.979325 w_k10_MSFT_s133
                                 + 0.926885 w_k10_PG_s133 >= 1
 z_k10_s1340_constraint:         var_10 + z_k10_s1340 + 1.02237 w_k10_ADM_s134
                                 + 1.13826 w_k10_AXP_s134
                                 + 1.01581 w_k10_DIS_s134
                                 + 0.991973 w_k10_FDX_s134
                                 + 1.41333 w_k10_IBM_s134
                                 + 1.16836 w_k10_INTC_s134
                                 + 1.1523 w_k10_KO_s134
                                 + 1.16403 w_k10_MAT_s134
                                 + 0.905432 w_k10_MSFT_s134
                                 + 1.04172 w_k10_PG_s134 >= 1
 z_k10_s1341_constraint:         var_10 + z_k10_s1341 + 1.479 w_k10_ADM_s134
                                 + 1.17572 w_k10_AXP_s134
                                 + 1.34498 w_k10_DIS_s134
                                 + 1.18681 w_k10_FDX_s134
                                 + 0.92122 w_k10_IBM_s134
                                 + 0.833267 w_k10_INTC_s134
                                 + 1.0112 w_k10_KO_s134
                                 + 1.08962 w_k10_MAT_s134
                                 + 0.986908 w_k10_MSFT_s134
                                 + 1.19986 w_k10_PG_s134 >= 1
 z_k10_s1342_constraint:         var_10 + z_k10_s1342 + 1.21411 w_k10_ADM_s134
                                 + 1.14048 w_k10_AXP_s134
                                 + 1.16179 w_k10_DIS_s134
                                 + 1.16594 w_k10_FDX_s134
                                 + 1.00901 w_k10_IBM_s134
                                 + 0.873514 w_k10_INTC_s134
                                 + 0.843698 w_k10_KO_s134
                                 + 0.977583 w_k10_MAT_s134
                                 + 0.968472 w_k10_MSFT_s134
                                 + 0.881587 w_k10_PG_s134 >= 1
 z_k10_s1343_constraint:         var_10 + z_k10_s1343 + 0.685774 w_k10_ADM_s134
                                 + 0.738218 w_k10_AXP_s134
                                 + 0.708364 w_k10_DIS_s134
                                 + 0.835774 w_k10_FDX_s134
                                 + 0.702039 w_k10_IBM_s134
                                 + 0.591954 w_k10_INTC_s134
                                 + 0.664093 w_k10_KO_s134
                                 + 0.776065 w_k10_MAT_s134
                                 + 0.619189 w_k10_MSFT_s134
                                 + 0.852645 w_k10_PG_s134 >= 1
 z_k10_s1344_constraint:         var_10 + z_k10_s1344 + 0.92991 w_k10_ADM_s134
                                 + 1.31245 w_k10_AXP_s134
                                 + 1.40989 w_k10_DIS_s134
                                 + 1.49672 w_k10_FDX_s134
                                 + 0.891707 w_k10_IBM_s134
                                 + 1.1596 w_k10_INTC_s134
                                 + 1.01947 w_k10_KO_s134
                                 + 1.3312 w_k10_MAT_s134
                                 + 1.29644 w_k10_MSFT_s134
                                 + 1.09348 w_k10_PG_s134 >= 1
 z_k10_s1345_constraint:         var_10 + z_k10_s1345 + 1.4342 w_k10_ADM_s134
                                 + 1.43284 w_k10_AXP_s134
                                 + 1.42674 w_k10_DIS_s134
                                 + 1.23532 w_k10_FDX_s134
                                 + 1.37445 w_k10_IBM_s134
                                 + 1.19286 w_k10_INTC_s134
                                 + 1.27484 w_k10_KO_s134
                                 + 1.29782 w_k10_MAT_s134
                                 + 1.1017 w_k10_MSFT_s134
                                 + 0.973231 w_k10_PG_s134 >= 1
 z_k10_s1346_constraint:         var_10 + z_k10_s1346 + 1.13991 w_k10_ADM_s134
                                 + 1.00856 w_k10_AXP_s134
                                 + 0.874608 w_k10_DIS_s134
                                 + 0.893369 w_k10_FDX_s134
                                 + 1.00363 w_k10_IBM_s134
                                 + 0.821559 w_k10_INTC_s134
                                 + 0.965775 w_k10_KO_s134
                                 + 1.01084 w_k10_MAT_s134
                                 + 0.86576 w_k10_MSFT_s134
                                 + 0.797446 w_k10_PG_s134 >= 1
 z_k10_s1347_constraint:         var_10 + z_k10_s1347 + 0.835741 w_k10_ADM_s134
                                 + 0.891082 w_k10_AXP_s134
                                 + 1.00354 w_k10_DIS_s134
                                 + 1.13387 w_k10_FDX_s134
                                 + 0.750808 w_k10_IBM_s134
                                 + 0.924958 w_k10_INTC_s134
                                 + 0.859636 w_k10_KO_s134
                                 + 0.882283 w_k10_MAT_s134
                                 + 1.04923 w_k10_MSFT_s134
                                 + 0.983639 w_k10_PG_s134 >= 1
 z_k10_s1350_constraint:         var_10 + z_k10_s1350 + 0.70599 w_k10_ADM_s135
                                 + 0.90526 w_k10_AXP_s135
                                 + 0.965237 w_k10_DIS_s135
                                 + 0.975658 w_k10_FDX_s135
                                 + 1.2277 w_k10_IBM_s135
                                 + 0.714074 w_k10_INTC_s135
                                 + 1.24398 w_k10_KO_s135
                                 + 1.79964 w_k10_MAT_s135
                                 + 0.860407 w_k10_MSFT_s135
                                 + 1.07526 w_k10_PG_s135 >= 1
 z_k10_s1351_constraint:         var_10 + z_k10_s1351 + 1.02142 w_k10_ADM_s135
                                 + 1.08999 w_k10_AXP_s135
                                 + 1.19536 w_k10_DIS_s135
                                 + 1.19293 w_k10_FDX_s135
                                 + 1.13271 w_k10_IBM_s135
                                 + 0.914876 w_k10_INTC_s135
                                 + 1.03572 w_k10_KO_s135
                                 + 1.46936 w_k10_MAT_s135
                                 + 0.893888 w_k10_MSFT_s135
                                 + 1.14709 w_k10_PG_s135 >= 1
 z_k10_s1352_constraint:         var_10 + z_k10_s1352 + 0.979019 w_k10_ADM_s135
                                 + 1.0458 w_k10_AXP_s135
                                 + 0.793913 w_k10_DIS_s135
                                 + 1.01583 w_k10_FDX_s135
                                 + 0.829801 w_k10_IBM_s135
                                 + 0.751737 w_k10_INTC_s135
                                 + 0.948207 w_k10_KO_s135
                                 + 1.30726 w_k10_MAT_s135
                                 + 0.899283 w_k10_MSFT_s135
                                 + 1.04436 w_k10_PG_s135 >= 1
 z_k10_s1353_constraint:         var_10 + z_k10_s1353 + 0.653252 w_k10_ADM_s135
                                 + 0.764939 w_k10_AXP_s135
                                 + 0.618518 w_k10_DIS_s135
                                 + 0.719156 w_k10_FDX_s135
                                 + 0.923215 w_k10_IBM_s135
                                 + 0.726544 w_k10_INTC_s135
                                 + 0.763758 w_k10_KO_s135
                                 + 0.994697 w_k10_MAT_s135
                                 + 0.762428 w_k10_MSFT_s135
                                 + 0.709599 w_k10_PG_s135 >= 1
 z_k10_s1354_constraint:         var_10 + z_k10_s1354 + 1.13703 w_k10_ADM_s135
                                 + 1.56887 w_k10_AXP_s135
                                 + 1.68738 w_k10_DIS_s135
                                 + 1.51566 w_k10_FDX_s135
                                 + 1.55126 w_k10_IBM_s135
                                 + 1.2589 w_k10_INTC_s135
                                 + 1.52333 w_k10_KO_s135
                                 + 2.48546 w_k10_MAT_s135
                                 + 1.38627 w_k10_MSFT_s135
                                 + 1.51437 w_k10_PG_s135 >= 1
 z_k10_s1355_constraint:         var_10 + z_k10_s1355 + 0.939141 w_k10_ADM_s135
                                 + 1.43332 w_k10_AXP_s135
                                 + 1.58907 w_k10_DIS_s135
                                 + 1.29334 w_k10_FDX_s135
                                 + 1.23102 w_k10_IBM_s135
                                 + 0.940789 w_k10_INTC_s135
                                 + 1.10585 w_k10_KO_s135
                                 + 1.54913 w_k10_MAT_s135
                                 + 1.03524 w_k10_MSFT_s135
                                 + 1.10217 w_k10_PG_s135 >= 1
 z_k10_s1356_constraint:         var_10 + z_k10_s1356 + 1.09891 w_k10_ADM_s135
                                 + 0.897431 w_k10_AXP_s135
                                 + 1.20551 w_k10_DIS_s135
                                 + 1.18967 w_k10_FDX_s135
                                 + 0.767717 w_k10_IBM_s135
                                 + 0.871866 w_k10_INTC_s135
                                 + 0.920939 w_k10_KO_s135
                                 + 1.08228 w_k10_MAT_s135
                                 + 1.03893 w_k10_MSFT_s135
                                 + 1.02547 w_k10_PG_s135 >= 1
 z_k10_s1357_constraint:         var_10 + z_k10_s1357 + 1.12572 w_k10_ADM_s135
                                 + 1.93689 w_k10_AXP_s135
                                 + 1.27565 w_k10_DIS_s135
                                 + 1.36703 w_k10_FDX_s135
                                 + 1.39532 w_k10_IBM_s135
                                 + 0.955773 w_k10_INTC_s135
                                 + 1.12065 w_k10_KO_s135 + 2.543 w_k10_MAT_s135
                                 + 0.918896 w_k10_MSFT_s135
                                 + 1.07278 w_k10_PG_s135 >= 1
 z_k10_s1360_constraint:         var_10 + z_k10_s1360 + 0.539673 w_k10_ADM_s136
                                 + 0.649455 w_k10_AXP_s136
                                 + 0.766679 w_k10_DIS_s136
                                 + 0.732153 w_k10_FDX_s136
                                 + 0.963771 w_k10_IBM_s136
                                 + 0.923537 w_k10_INTC_s136
                                 + 0.911222 w_k10_KO_s136
                                 + 0.779722 w_k10_MAT_s136
                                 + 0.841586 w_k10_MSFT_s136
                                 + 0.887547 w_k10_PG_s136 >= 1
 z_k10_s1361_constraint:         var_10 + z_k10_s1361 + 0.870403 w_k10_ADM_s136
                                 + 0.969491 w_k10_AXP_s136
                                 + 0.73414 w_k10_DIS_s136
                                 + 0.741535 w_k10_FDX_s136
                                 + 1.11415 w_k10_IBM_s136
                                 + 0.768189 w_k10_INTC_s136
                                 + 0.876126 w_k10_KO_s136
                                 + 1.02851 w_k10_MAT_s136
                                 + 0.764875 w_k10_MSFT_s136
                                 + 0.934459 w_k10_PG_s136 >= 1
 z_k10_s1362_constraint:         var_10 + z_k10_s1362 + 0.595378 w_k10_ADM_s136
                                 + 1.44637 w_k10_AXP_s136
                                 + 1.2821 w_k10_DIS_s136
                                 + 1.02262 w_k10_FDX_s136
                                 + 1.49952 w_k10_IBM_s136
                                 + 1.11356 w_k10_INTC_s136
                                 + 1.33139 w_k10_KO_s136
                                 + 1.7783 w_k10_MAT_s136
                                 + 1.10064 w_k10_MSFT_s136
                                 + 1.10211 w_k10_PG_s136 >= 1
 z_k10_s1363_constraint:         var_10 + z_k10_s1363 + 0.981639 w_k10_ADM_s136
                                 + 1.21225 w_k10_AXP_s136
                                 + 1.28783 w_k10_DIS_s136
                                 + 1.16029 w_k10_FDX_s136
                                 + 0.870719 w_k10_IBM_s136
                                 + 1.11884 w_k10_INTC_s136
                                 + 1.05421 w_k10_KO_s136
                                 + 0.909822 w_k10_MAT_s136
                                 + 1.13448 w_k10_MSFT_s136
                                 + 1.23389 w_k10_PG_s136 >= 1
 z_k10_s1364_constraint:         var_10 + z_k10_s1364 + 0.789424 w_k10_ADM_s136
                                 + 1.00836 w_k10_AXP_s136
                                 + 0.954882 w_k10_DIS_s136
                                 + 0.927125 w_k10_FDX_s136
                                 + 1.25858 w_k10_IBM_s136
                                 + 0.799089 w_k10_INTC_s136
                                 + 1.06684 w_k10_KO_s136
                                 + 1.33731 w_k10_MAT_s136
                                 + 0.846078 w_k10_MSFT_s136
                                 + 1.01284 w_k10_PG_s136 >= 1
 z_k10_s1365_constraint:         var_10 + z_k10_s1365 + 1.4359 w_k10_ADM_s136
                                 + 1.69228 w_k10_AXP_s136
                                 + 1.64202 w_k10_DIS_s136
                                 + 1.37826 w_k10_FDX_s136
                                 + 1.1248 w_k10_IBM_s136
                                 + 1.11012 w_k10_INTC_s136
                                 + 1.39164 w_k10_KO_s136
                                 + 1.45382 w_k10_MAT_s136
                                 + 1.29948 w_k10_MSFT_s136
                                 + 1.33247 w_k10_PG_s136 >= 1
 z_k10_s1366_constraint:         var_10 + z_k10_s1366 + 0.699487 w_k10_ADM_s136
                                 + 0.784783 w_k10_AXP_s136
                                 + 0.822719 w_k10_DIS_s136
                                 + 0.851679 w_k10_FDX_s136
                                 + 0.641177 w_k10_IBM_s136
                                 + 0.880086 w_k10_INTC_s136
                                 + 0.702367 w_k10_KO_s136
                                 + 0.580131 w_k10_MAT_s136
                                 + 0.869772 w_k10_MSFT_s136
                                 + 0.902571 w_k10_PG_s136 >= 1
 z_k10_s1367_constraint:         var_10 + z_k10_s1367 + 0.818647 w_k10_ADM_s136
                                 + 0.883285 w_k10_AXP_s136
                                 + 0.917397 w_k10_DIS_s136
                                 + 0.881161 w_k10_FDX_s136
                                 + 0.926104 w_k10_IBM_s136
                                 + 0.81013 w_k10_INTC_s136
                                 + 0.861322 w_k10_KO_s136
                                 + 1.02498 w_k10_MAT_s136
                                 + 0.928969 w_k10_MSFT_s136
                                 + 1.05484 w_k10_PG_s136 >= 1
 z_k10_s1370_constraint:         var_10 + z_k10_s1370 + 0.995832 w_k10_ADM_s137
                                 + 1.02911 w_k10_AXP_s137
                                 + 1.07461 w_k10_DIS_s137
                                 + 1.073 w_k10_FDX_s137
                                 + 0.920095 w_k10_IBM_s137
                                 + 1.0068 w_k10_INTC_s137
                                 + 1.0732 w_k10_KO_s137
                                 + 1.22627 w_k10_MAT_s137
                                 + 1.12039 w_k10_MSFT_s137
                                 + 1.10072 w_k10_PG_s137 >= 1
 z_k10_s1371_constraint:         var_10 + z_k10_s1371 + 0.911796 w_k10_ADM_s137
                                 + 0.795247 w_k10_AXP_s137
                                 + 0.564551 w_k10_DIS_s137
                                 + 0.856449 w_k10_FDX_s137
                                 + 0.58409 w_k10_IBM_s137
                                 + 0.904933 w_k10_INTC_s137
                                 + 0.723171 w_k10_KO_s137
                                 + 0.889519 w_k10_MAT_s137
                                 + 0.873978 w_k10_MSFT_s137
                                 + 0.880633 w_k10_PG_s137 >= 1
 z_k10_s1372_constraint:         var_10 + z_k10_s1372 + 1.25643 w_k10_ADM_s137
                                 + 1.52431 w_k10_AXP_s137
                                 + 1.70649 w_k10_DIS_s137
                                 + 1.39531 w_k10_FDX_s137
                                 + 1.26633 w_k10_IBM_s137
                                 + 1.16782 w_k10_INTC_s137
                                 + 1.16631 w_k10_KO_s137
                                 + 1.19231 w_k10_MAT_s137
                                 + 1.19283 w_k10_MSFT_s137
                                 + 1.11688 w_k10_PG_s137 >= 1
 z_k10_s1373_constraint:         var_10 + z_k10_s1373 + 1.4201 w_k10_ADM_s137
                                 + 1.31619 w_k10_AXP_s137
                                 + 1.44775 w_k10_DIS_s137
                                 + 1.1586 w_k10_FDX_s137
                                 + 1.67058 w_k10_IBM_s137
                                 + 0.977534 w_k10_INTC_s137
                                 + 1.52805 w_k10_KO_s137
                                 + 1.58108 w_k10_MAT_s137
                                 + 1.23927 w_k10_MSFT_s137
                                 + 1.29198 w_k10_PG_s137 >= 1
 z_k10_s1374_constraint:         var_10 + z_k10_s1374 + 1.15541 w_k10_ADM_s137
                                 + 1.1455 w_k10_AXP_s137
                                 + 0.882555 w_k10_DIS_s137
                                 + 1.18485 w_k10_FDX_s137
                                 + 1.24038 w_k10_IBM_s137
                                 + 0.949476 w_k10_INTC_s137
                                 + 0.98354 w_k10_KO_s137
                                 + 0.983863 w_k10_MAT_s137
                                 + 1.10074 w_k10_MSFT_s137
                                 + 1.06097 w_k10_PG_s137 >= 1
 z_k10_s1375_constraint:         var_10 + z_k10_s1375 + 1.25609 w_k10_ADM_s137
                                 + 1.57143 w_k10_AXP_s137
                                 + 1.18637 w_k10_DIS_s137
                                 + 1.2911 w_k10_FDX_s137
                                 + 0.916445 w_k10_IBM_s137
                                 + 1.07529 w_k10_INTC_s137
                                 + 0.989332 w_k10_KO_s137
                                 + 1.03543 w_k10_MAT_s137
                                 + 1.22309 w_k10_MSFT_s137
                                 + 1.06323 w_k10_PG_s137 >= 1
 z_k10_s1376_constraint:         var_10 + z_k10_s1376 + 0.717521 w_k10_ADM_s137
                                 + 0.889221 w_k10_AXP_s137
                                 + 0.736446 w_k10_DIS_s137
                                 + 0.776797 w_k10_FDX_s137
                                 + 1.06495 w_k10_IBM_s137
                                 + 0.906259 w_k10_INTC_s137
                                 + 0.848052 w_k10_KO_s137
                                 + 1.09641 w_k10_MAT_s137
                                 + 0.852219 w_k10_MSFT_s137
                                 + 0.898406 w_k10_PG_s137 >= 1
 z_k10_s1377_constraint:         var_10 + z_k10_s1377 + 1.43035 w_k10_ADM_s137
                                 + 1.16068 w_k10_AXP_s137
                                 + 0.816255 w_k10_DIS_s137
                                 + 1.01386 w_k10_FDX_s137
                                 + 0.801988 w_k10_IBM_s137
                                 + 0.788259 w_k10_INTC_s137
                                 + 0.867218 w_k10_KO_s137
                                 + 0.810655 w_k10_MAT_s137
                                 + 1.03212 w_k10_MSFT_s137
                                 + 0.973461 w_k10_PG_s137 >= 1
 k10_return_constraint:          0.0114991457245956 w_k10_ADM_s000
                                 + 0.0233390560913227 w_k10_ADM_s001
                                 + 0.0176684289085026 w_k10_ADM_s002
                                 + 0.034691432341792 w_k10_ADM_s003
                                 + 0.0132829703220255 w_k10_ADM_s004
                                 + 0.0291520556419908 w_k10_ADM_s005
                                 + 0.0477710403876729 w_k10_ADM_s006
                                 + 0.028682636442337 w_k10_ADM_s007
                                 + 0.0346079721307154 w_k10_ADM_s010
                                 + 0.0355669588532545 w_k10_ADM_s011
                                 + 0.0211545604553012 w_k10_ADM_s012
                                 + 0.0207816749076957 w_k10_ADM_s013
                                 + 0.0228660466125482 w_k10_ADM_s014
                                 + 0.0169667889890938 w_k10_ADM_s015
                                 + 0.012561322314341 w_k10_ADM_s016
                                 + 0.0142881361481672 w_k10_ADM_s017
                                 + 0.0429645929831089 w_k10_ADM_s020
                                 + 0.0278687655337764 w_k10_ADM_s021
                                 + 0.0232461913194468 w_k10_ADM_s022
                                 + 0.0210420784388937 w_k10_ADM_s023
                                 + 0.0261956004145577 w_k10_ADM_s024
                                 + 0.0162854281209592 w_k10_ADM_s025
                                 + 0.0322843402198051 w_k10_ADM_s026
                                 + 0.044929614799749 w_k10_ADM_s027
                                 + 0.0163203244013878 w_k10_ADM_s030
                                 + 0.00776688523524653 w_k10_ADM_s031
                                 + 0.0122729529753094 w_k10_ADM_s032
                                 + 0.00754337211386288 w_k10_ADM_s033
                                 + 0.00487912877675753 w_k10_ADM_s034
                                 + 0.00616775022047407 w_k10_ADM_s035
                                 + 0.0156811452328668 w_k10_ADM_s036
                                 + 0.00383796146728267 w_k10_ADM_s037
                                 + 0.0103596385285898 w_k10_ADM_s100
                                 + 0.0119253738580842 w_k10_ADM_s101
                                 + 0.00946109455424615 w_k10_ADM_s102
                                 + 0.00873278506087371 w_k10_ADM_s103
                                 + 0.00793948419524003 w_k10_ADM_s104
                                 + 0.0122767758303701 w_k10_ADM_s105
                                 + 0.0169811285979814 w_k10_ADM_s106
                                 + 0.00676281679269094 w_k10_ADM_s107
                                 + 0.0322312858699312 w_k10_ADM_s110
                                 + 0.0152998421129987 w_k10_ADM_s111
                                 + 0.0254807219571416 w_k10_ADM_s112
                                 + 0.0144844055688694 w_k10_ADM_s113
                                 + 0.0096031444683177 w_k10_ADM_s114
                                 + 0.0163263958102422 w_k10_ADM_s115
                                 + 0.0168289694681944 w_k10_ADM_s116
                                 + 0.0251126681600471 w_k10_ADM_s117
                                 + 0.006476556612516 w_k10_ADM_s120
                                 + 0.00877772833895447 w_k10_ADM_s121
                                 + 0.00493222418533893 w_k10_ADM_s122
                                 + 0.010523489397682 w_k10_ADM_s123
                                 + 0.0130412095687134 w_k10_ADM_s124
                                 + 0.00446844564807872 w_k10_ADM_s125
                                 + 0.00233524569726972 w_k10_ADM_s126
                                 + 0.00663449589250329 w_k10_ADM_s127
                                 + 0.00204952395142997 w_k10_ADM_s130
                                 + 0.0033025140397324 w_k10_ADM_s131
                                 + 0.00641802982117287 w_k10_ADM_s132
                                 + 0.00426163020790568 w_k10_ADM_s133
                                 + 0.00270005506849128 w_k10_ADM_s134
                                 + 0.00300196373547636 w_k10_ADM_s135
                                 + 0.00544656726999613 w_k10_ADM_s136
                                 + 0.00346210728471762 w_k10_ADM_s137
                                 + 0.0129835345761088 w_k10_AXP_s000
                                 + 0.025016344449877 w_k10_AXP_s001
                                 + 0.0169348151410414 w_k10_AXP_s002
                                 + 0.0378482615007756 w_k10_AXP_s003
                                 + 0.0157003686157636 w_k10_AXP_s004
                                 + 0.0308107061050395 w_k10_AXP_s005
                                 + 0.0468629286575818 w_k10_AXP_s006
                                 + 0.0303808059835403 w_k10_AXP_s007
                                 + 0.0399224341802676 w_k10_AXP_s010
                                 + 0.0396869513177483 w_k10_AXP_s011
                                 + 0.0225059929705386 w_k10_AXP_s012
                                 + 0.0197281382136306 w_k10_AXP_s013
                                 + 0.0242406903915603 w_k10_AXP_s014
                                 + 0.0168647962319137 w_k10_AXP_s015
                                 + 0.0127565063970803 w_k10_AXP_s016
                                 + 0.0144180520394792 w_k10_AXP_s017
                                 + 0.0409311975817312 w_k10_AXP_s020
                                 + 0.027677693004713 w_k10_AXP_s021
                                 + 0.0209121177975066 w_k10_AXP_s022
                                 + 0.0235389574502641 w_k10_AXP_s023
                                 + 0.0252489606221408 w_k10_AXP_s024
                                 + 0.0149972174182135 w_k10_AXP_s025
                                 + 0.0344675598674239 w_k10_AXP_s026
                                 + 0.045083907478004 w_k10_AXP_s027
                                 + 0.0150217888768219 w_k10_AXP_s030
                                 + 0.0078824297011997 w_k10_AXP_s031
                                 + 0.0122117115455579 w_k10_AXP_s032
                                 + 0.00741835878563797 w_k10_AXP_s033
                                 + 0.00490352125887802 w_k10_AXP_s034
                                 + 0.00641735422696092 w_k10_AXP_s035
                                 + 0.0164211065158532 w_k10_AXP_s036
                                 + 0.00372691408840097 w_k10_AXP_s037
                                 + 0.0088494519211646 w_k10_AXP_s100
                                 + 0.0103595473133823 w_k10_AXP_s101
                                 + 0.00952555096745808 w_k10_AXP_s102
                                 + 0.00936218205605271 w_k10_AXP_s103
                                 + 0.00717771323493743 w_k10_AXP_s104
                                 + 0.0113753509485406 w_k10_AXP_s105
                                 + 0.0203004313060762 w_k10_AXP_s106
                                 + 0.00591128839436695 w_k10_AXP_s107
                                 + 0.0290302160770787 w_k10_AXP_s110
                                 + 0.0136346102526509 w_k10_AXP_s111
                                 + 0.0264041402085411 w_k10_AXP_s112
                                 + 0.0125270704730344 w_k10_AXP_s113
                                 + 0.0113239435561831 w_k10_AXP_s114
                                 + 0.019161761571117 w_k10_AXP_s115
                                 + 0.0165606180152222 w_k10_AXP_s116
                                 + 0.0268216586703539 w_k10_AXP_s117
                                 + 0.00626642651795663 w_k10_AXP_s120
                                 + 0.00729109543881346 w_k10_AXP_s121
                                 + 0.00550103874973842 w_k10_AXP_s122
                                 + 0.0086119280565523 w_k10_AXP_s123
                                 + 0.0116080097262286 w_k10_AXP_s124
                                 + 0.00368972568785539 w_k10_AXP_s125
                                 + 0.00278642629560764 w_k10_AXP_s126
                                 + 0.00595900827440357 w_k10_AXP_s127
                                 + 0.00239879457163332 w_k10_AXP_s130
                                 + 0.00212690597969606 w_k10_AXP_s131
                                 + 0.00703080806018736 w_k10_AXP_s132
                                 + 0.00384024756719696 w_k10_AXP_s133
                                 + 0.00270452881825938 w_k10_AXP_s134
                                 + 0.00371653611098708 w_k10_AXP_s135
                                 + 0.00708469551181812 w_k10_AXP_s136
                                 + 0.00341168261554518 w_k10_AXP_s137
                                 + 0.0125257433041197 w_k10_DIS_s000
                                 + 0.0237283476055658 w_k10_DIS_s001
                                 + 0.0182619979510764 w_k10_DIS_s002
                                 + 0.0397469489132044 w_k10_DIS_s003
                                 + 0.0160277245292628 w_k10_DIS_s004
                                 + 0.033094518807893 w_k10_DIS_s005
                                 + 0.0475155270398563 w_k10_DIS_s006
                                 + 0.0311469772552651 w_k10_DIS_s007
                                 + 0.0347048040423088 w_k10_DIS_s010
                                 + 0.034995738309832 w_k10_DIS_s011
                                 + 0.0198204854231682 w_k10_DIS_s012
                                 + 0.0225667679065255 w_k10_DIS_s013
                                 + 0.0278290936687426 w_k10_DIS_s014
                                 + 0.0160871577931473 w_k10_DIS_s015
                                 + 0.0118971054888493 w_k10_DIS_s016
                                 + 0.0144927941772175 w_k10_DIS_s017
                                 + 0.0433885805376254 w_k10_DIS_s020
                                 + 0.0251357381364377 w_k10_DIS_s021
                                 + 0.0225865021447399 w_k10_DIS_s022
                                 + 0.0233495091455961 w_k10_DIS_s023
                                 + 0.0255794276105339 w_k10_DIS_s024
                                 + 0.0159496157938363 w_k10_DIS_s025
                                 + 0.0354511144399992 w_k10_DIS_s026
                                 + 0.0471670690457349 w_k10_DIS_s027
                                 + 0.0158911093830865 w_k10_DIS_s030
                                 + 0.00682465262854215 w_k10_DIS_s031
                                 + 0.0129779531402786 w_k10_DIS_s032
                                 + 0.00787388199995179 w_k10_DIS_s033
                                 + 0.00524054212139326 w_k10_DIS_s034
                                 + 0.00591366983986786 w_k10_DIS_s035
                                 + 0.0158106782854528 w_k10_DIS_s036
                                 + 0.00363098152151627 w_k10_DIS_s037
                                 + 0.00925619854245109 w_k10_DIS_s100
                                 + 0.0104999530086246 w_k10_DIS_s101
                                 + 0.0093876236837221 w_k10_DIS_s102
                                 + 0.010157004581388 w_k10_DIS_s103
                                 + 0.00801792084499476 w_k10_DIS_s104
                                 + 0.0120285659013429 w_k10_DIS_s105
                                 + 0.0207297926285842 w_k10_DIS_s106
                                 + 0.00634588111441948 w_k10_DIS_s107
                                 + 0.0267140677112717 w_k10_DIS_s110
                                 + 0.0133914468376341 w_k10_DIS_s111
                                 + 0.0272990517413724 w_k10_DIS_s112
                                 + 0.0118203254462422 w_k10_DIS_s113
                                 + 0.010222860770881 w_k10_DIS_s114
                                 + 0.0188466303746656 w_k10_DIS_s115
                                 + 0.0170579478837069 w_k10_DIS_s116
                                 + 0.0248823336389348 w_k10_DIS_s117
                                 + 0.00622004145588591 w_k10_DIS_s120
                                 + 0.00724359668149411 w_k10_DIS_s121
                                 + 0.00569749158136526 w_k10_DIS_s122
                                 + 0.00872571874645606 w_k10_DIS_s123
                                 + 0.0116373736817869 w_k10_DIS_s124
                                 + 0.00448081341745526 w_k10_DIS_s125
                                 + 0.00269874648057073 w_k10_DIS_s126
                                 + 0.00580821286743216 w_k10_DIS_s127
                                 + 0.00242692970989079 w_k10_DIS_s130
                                 + 0.0023733532162065 w_k10_DIS_s131
                                 + 0.00674370254886073 w_k10_DIS_s132
                                 + 0.00362928457379427 w_k10_DIS_s133
                                 + 0.00269372064735593 w_k10_DIS_s134
                                 + 0.00357813428113083 w_k10_DIS_s135
                                 + 0.00700699055495517 w_k10_DIS_s136
                                 + 0.0029558787643951 w_k10_DIS_s137
                                 + 0.0120144414591184 w_k10_FDX_s000
                                 + 0.0234369561095051 w_k10_FDX_s001
                                 + 0.0169601226825294 w_k10_FDX_s002
                                 + 0.0391766849862406 w_k10_FDX_s003
                                 + 0.0155586716694241 w_k10_FDX_s004
                                 + 0.0314486812926833 w_k10_FDX_s005
                                 + 0.0482315825332852 w_k10_FDX_s006
                                 + 0.0291943346952645 w_k10_FDX_s007
                                 + 0.0342109004696441 w_k10_FDX_s010
                                 + 0.0350661146073738 w_k10_FDX_s011
                                 + 0.0202591975551172 w_k10_FDX_s012
                                 + 0.0197474143855694 w_k10_FDX_s013
                                 + 0.0265240931851933 w_k10_FDX_s014
                                 + 0.0177728913615142 w_k10_FDX_s015
                                 + 0.0121458217073353 w_k10_FDX_s016
                                 + 0.0151057717651323 w_k10_FDX_s017
                                 + 0.0417233504479295 w_k10_FDX_s020
                                 + 0.0264234872488675 w_k10_FDX_s021
                                 + 0.0220259356743359 w_k10_FDX_s022
                                 + 0.0223864107118572 w_k10_FDX_s023
                                 + 0.0252237343383751 w_k10_FDX_s024
                                 + 0.0153348408756274 w_k10_FDX_s025
                                 + 0.0344302231427244 w_k10_FDX_s026
                                 + 0.0457685402244132 w_k10_FDX_s027
                                 + 0.0159293107143821 w_k10_FDX_s030
                                 + 0.00781644895547523 w_k10_FDX_s031
                                 + 0.0121165083158017 w_k10_FDX_s032
                                 + 0.00803141095865782 w_k10_FDX_s033
                                 + 0.0050504105779041 w_k10_FDX_s034
                                 + 0.00574930742609693 w_k10_FDX_s035
                                 + 0.015520040622062 w_k10_FDX_s036
                                 + 0.00361078801473052 w_k10_FDX_s037
                                 + 0.0093380249566143 w_k10_FDX_s100
                                 + 0.00996208291755145 w_k10_FDX_s101
                                 + 0.0090924014012762 w_k10_FDX_s102
                                 + 0.00935375154287842 w_k10_FDX_s103
                                 + 0.00751293858671477 w_k10_FDX_s104
                                 + 0.0114418538475092 w_k10_FDX_s105
                                 + 0.020723133415552 w_k10_FDX_s106
                                 + 0.0060049929217791 w_k10_FDX_s107
                                 + 0.0287642873793278 w_k10_FDX_s110
                                 + 0.0128646293175487 w_k10_FDX_s111
                                 + 0.0261095511068652 w_k10_FDX_s112
                                 + 0.0127032444662962 w_k10_FDX_s113
                                 + 0.0104199523165172 w_k10_FDX_s114
                                 + 0.0177801705636611 w_k10_FDX_s115
                                 + 0.0178843288300909 w_k10_FDX_s116
                                 + 0.0244935538148737 w_k10_FDX_s117
                                 + 0.00590748210317088 w_k10_FDX_s120
                                 + 0.00735248928414192 w_k10_FDX_s121
                                 + 0.00540194812007096 w_k10_FDX_s122
                                 + 0.00887409077240293 w_k10_FDX_s123
                                 + 0.0121030130558025 w_k10_FDX_s124
                                 + 0.00436992977240529 w_k10_FDX_s125
                                 + 0.00250592649496997 w_k10_FDX_s126
                                 + 0.00595883671832364 w_k10_FDX_s127
                                 + 0.00237086353063914 w_k10_FDX_s130
                                 + 0.00215168928056111 w_k10_FDX_s131
                                 + 0.00667085616655211 w_k10_FDX_s132
                                 + 0.00369991849871692 w_k10_FDX_s133
                                 + 0.00270632465214956 w_k10_FDX_s134
                                 + 0.00362515598129315 w_k10_FDX_s135
                                 + 0.00646902646495514 w_k10_FDX_s136
                                 + 0.00316162109275845 w_k10_FDX_s137
                                 + 0.0127504354284193 w_k10_IBM_s000
                                 + 0.0255093622647329 w_k10_IBM_s001
                                 + 0.0171335246137747 w_k10_IBM_s002
                                 + 0.0381292828253284 w_k10_IBM_s003
                                 + 0.0146862475517195 w_k10_IBM_s004
                                 + 0.0295503094928475 w_k10_IBM_s005
                                 + 0.0496906446886512 w_k10_IBM_s006
                                 + 0.0288630689650161 w_k10_IBM_s007
                                 + 0.0346319091079432 w_k10_IBM_s010
                                 + 0.0355691484889848 w_k10_IBM_s011
                                 + 0.0206535208264171 w_k10_IBM_s012
                                 + 0.0192943623217391 w_k10_IBM_s013
                                 + 0.0268644063850266 w_k10_IBM_s014
                                 + 0.0172981630742869 w_k10_IBM_s015
                                 + 0.0125376545209319 w_k10_IBM_s016
                                 + 0.0156901906067717 w_k10_IBM_s017
                                 + 0.0423604411252641 w_k10_IBM_s020
                                 + 0.0264854535302497 w_k10_IBM_s021
                                 + 0.0198684415924646 w_k10_IBM_s022
                                 + 0.0226199792130101 w_k10_IBM_s023
                                 + 0.0272344675750734 w_k10_IBM_s024
                                 + 0.0147833006053512 w_k10_IBM_s025
                                 + 0.0331696820871864 w_k10_IBM_s026
                                 + 0.0469303125762335 w_k10_IBM_s027
                                 + 0.016047353230117 w_k10_IBM_s030
                                 + 0.00749865755962633 w_k10_IBM_s031
                                 + 0.0122853512357032 w_k10_IBM_s032
                                 + 0.00805020132687484 w_k10_IBM_s033
                                 + 0.00529995246885148 w_k10_IBM_s034
                                 + 0.0059650183113157 w_k10_IBM_s035
                                 + 0.0166297234750812 w_k10_IBM_s036
                                 + 0.00393349348558375 w_k10_IBM_s037
                                 + 0.0089323614535784 w_k10_IBM_s100
                                 + 0.010287021234091 w_k10_IBM_s101
                                 + 0.00984382511142672 w_k10_IBM_s102
                                 + 0.00986259475207278 w_k10_IBM_s103
                                 + 0.00778260125326085 w_k10_IBM_s104
                                 + 0.0113890809812886 w_k10_IBM_s105
                                 + 0.019953362291955 w_k10_IBM_s106
                                 + 0.00615346738883785 w_k10_IBM_s107
                                 + 0.029358391076255 w_k10_IBM_s110
                                 + 0.0127647946717906 w_k10_IBM_s111
                                 + 0.0262640643530918 w_k10_IBM_s112
                                 + 0.0127860055614273 w_k10_IBM_s113
                                 + 0.0102083151459635 w_k10_IBM_s114
                                 + 0.0182013287327919 w_k10_IBM_s115
                                 + 0.0170950845564219 w_k10_IBM_s116
                                 + 0.0247655524985529 w_k10_IBM_s117
                                 + 0.00616742588792547 w_k10_IBM_s120
                                 + 0.0075492915464953 w_k10_IBM_s121
                                 + 0.00582774821289238 w_k10_IBM_s122
                                 + 0.00837754254018239 w_k10_IBM_s123
                                 + 0.0125285148234981 w_k10_IBM_s124
                                 + 0.00437455409472876 w_k10_IBM_s125
                                 + 0.00250520325354709 w_k10_IBM_s126
                                 + 0.00567008453351668 w_k10_IBM_s127
                                 + 0.0025076949006483 w_k10_IBM_s130
                                 + 0.00220437672076199 w_k10_IBM_s131
                                 + 0.00714421383015814 w_k10_IBM_s132
                                 + 0.00347506167527073 w_k10_IBM_s133
                                 + 0.00256687081238088 w_k10_IBM_s134
                                 + 0.00368083735880943 w_k10_IBM_s135
                                 + 0.00684628028251079 w_k10_IBM_s136
                                 + 0.00322886130226674 w_k10_IBM_s137
                                 + 0.0121401857935976 w_k10_INTC_s000
                                 + 0.0225300312047938 w_k10_INTC_s001
                                 + 0.0184292026979809 w_k10_INTC_s002
                                 + 0.0416191029608011 w_k10_INTC_s003
                                 + 0.0189804697829343 w_k10_INTC_s004
                                 + 0.0303286182088633 w_k10_INTC_s005
                                 + 0.0486309098460087 w_k10_INTC_s006
                                 + 0.0282066875982416 w_k10_INTC_s007
                                 + 0.031188396571274 w_k10_INTC_s010
                                 + 0.0343082741915623 w_k10_INTC_s011
                                 + 0.0229748790710524 w_k10_INTC_s012
                                 + 0.0200146808126483 w_k10_INTC_s013
                                 + 0.0257927379536393 w_k10_INTC_s014
                                 + 0.0178511436534038 w_k10_INTC_s015
                                 + 0.0125829673970552 w_k10_INTC_s016
                                 + 0.0142460675280219 w_k10_INTC_s017
                                 + 0.0420912878903095 w_k10_INTC_s020
                                 + 0.0253672834087487 w_k10_INTC_s021
                                 + 0.0217425692286642 w_k10_INTC_s022
                                 + 0.0215449318777975 w_k10_INTC_s023
                                 + 0.0276898996383981 w_k10_INTC_s024
                                 + 0.0139428200437076 w_k10_INTC_s025
                                 + 0.0358866901354581 w_k10_INTC_s026
                                 + 0.0412071490026388 w_k10_INTC_s027
                                 + 0.0193983695665174 w_k10_INTC_s030
                                 + 0.0080608798406549 w_k10_INTC_s031
                                 + 0.0119441440330116 w_k10_INTC_s032
                                 + 0.0072174635293872 w_k10_INTC_s033
                                 + 0.00453383161262213 w_k10_INTC_s034
                                 + 0.00564873504262094 w_k10_INTC_s035
                                 + 0.0158441371530262 w_k10_INTC_s036
                                 + 0.00323118112399325 w_k10_INTC_s037
                                 + 0.00850686309100525 w_k10_INTC_s100
                                 + 0.00943397531969679 w_k10_INTC_s101
                                 + 0.00873672893503153 w_k10_INTC_s102
                                 + 0.00888209433241618 w_k10_INTC_s103
                                 + 0.00848309112632191 w_k10_INTC_s104
                                 + 0.0107982848204787 w_k10_INTC_s105
                                 + 0.0226937719631027 w_k10_INTC_s106
                                 + 0.00505906081254665 w_k10_INTC_s107
                                 + 0.0275905834036039 w_k10_INTC_s110
                                 + 0.0127100605914152 w_k10_INTC_s111
                                 + 0.022660471348847 w_k10_INTC_s112
                                 + 0.0111306360799914 w_k10_INTC_s113
                                 + 0.0100828943219258 w_k10_INTC_s114
                                 + 0.0194602866110411 w_k10_INTC_s115
                                 + 0.0168284839933988 w_k10_INTC_s116
                                 + 0.0261016787283154 w_k10_INTC_s117
                                 + 0.00536727546034907 w_k10_INTC_s120
                                 + 0.00763594732830242 w_k10_INTC_s121
                                 + 0.00534213635637819 w_k10_INTC_s122
                                 + 0.00814879687870422 w_k10_INTC_s123
                                 + 0.0116120298723256 w_k10_INTC_s124
                                 + 0.00431171018688422 w_k10_INTC_s125
                                 + 0.00258938463099805 w_k10_INTC_s126
                                 + 0.00537621254627211 w_k10_INTC_s127
                                 + 0.00237097996403631 w_k10_INTC_s130
                                 + 0.00230042403663544 w_k10_INTC_s131
                                 + 0.00709378928110765 w_k10_INTC_s132
                                 + 0.00315184492005286 w_k10_INTC_s133
                                 + 0.00237611514171729 w_k10_INTC_s134
                                 + 0.00281215378372235 w_k10_INTC_s135
                                 + 0.00634186424296285 w_k10_INTC_s136
                                 + 0.00285611584891017 w_k10_INTC_s137
                                 + 0.0119451047325904 w_k10_KO_s000
                                 + 0.0228843961544794 w_k10_KO_s001
                                 + 0.0168453744584999 w_k10_KO_s002
                                 + 0.0368779043123896 w_k10_KO_s003
                                 + 0.0142723529674919 w_k10_KO_s004
                                 + 0.0306869282973578 w_k10_KO_s005
                                 + 0.0454832836037795 w_k10_KO_s006
                                 + 0.028252443528446 w_k10_KO_s007
                                 + 0.0342866506657353 w_k10_KO_s010
                                 + 0.0360416905448826 w_k10_KO_s011
                                 + 0.0214115307405411 w_k10_KO_s012
                                 + 0.0189340234058515 w_k10_KO_s013
                                 + 0.0279484634091687 w_k10_KO_s014
                                 + 0.0162959037232603 w_k10_KO_s015
                                 + 0.0112193161417398 w_k10_KO_s016
                                 + 0.0135824092934467 w_k10_KO_s017
                                 + 0.0379341483267201 w_k10_KO_s020
                                 + 0.025159537534215 w_k10_KO_s021
                                 + 0.020187235919522 w_k10_KO_s022
                                 + 0.0226092322888158 w_k10_KO_s023
                                 + 0.0275049440901669 w_k10_KO_s024
                                 + 0.0158323823510551 w_k10_KO_s025
                                 + 0.0349881219744555 w_k10_KO_s026
                                 + 0.0412686610213476 w_k10_KO_s027
                                 + 0.0162690342162688 w_k10_KO_s030
                                 + 0.00680622090023185 w_k10_KO_s031
                                 + 0.0121008361488168 w_k10_KO_s032
                                 + 0.00795544338597061 w_k10_KO_s033
                                 + 0.0053101872259262 w_k10_KO_s034
                                 + 0.00534247342114377 w_k10_KO_s035
                                 + 0.0161548741775432 w_k10_KO_s036
                                 + 0.0035978637010924 w_k10_KO_s037
                                 + 0.00792419514990026 w_k10_KO_s100
                                 + 0.00971291996029848 w_k10_KO_s101
                                 + 0.00950187620627961 w_k10_KO_s102
                                 + 0.00954130485047597 w_k10_KO_s103
                                 + 0.00727675851114054 w_k10_KO_s104
                                 + 0.0129368929709621 w_k10_KO_s105
                                 + 0.0184645178751204 w_k10_KO_s106
                                 + 0.00593401261461177 w_k10_KO_s107
                                 + 0.0330403781588434 w_k10_KO_s110
                                 + 0.0147398938811421 w_k10_KO_s111
                                 + 0.025153834029524 w_k10_KO_s112
                                 + 0.0125335548107118 w_k10_KO_s113
                                 + 0.00929319565480629 w_k10_KO_s114
                                 + 0.0177664838073672 w_k10_KO_s115
                                 + 0.0182790573696162 w_k10_KO_s116
                                 + 0.0237480969465355 w_k10_KO_s117
                                 + 0.00619187163626373 w_k10_KO_s120
                                 + 0.00871670008122724 w_k10_KO_s121
                                 + 0.00606318221994948 w_k10_KO_s122
                                 + 0.00896061368183434 w_k10_KO_s123
                                 + 0.0122651940177082 w_k10_KO_s124
                                 + 0.00413049394849862 w_k10_KO_s125
                                 + 0.00251782843223772 w_k10_KO_s126
                                 + 0.00574420260827975 w_k10_KO_s127
                                 + 0.00245977447414613 w_k10_KO_s130
                                 + 0.00242129542204222 w_k10_KO_s131
                                 + 0.007455454743243 w_k10_KO_s132
                                 + 0.00359409160171948 w_k10_KO_s133
                                 + 0.0024417786174323 w_k10_KO_s134
                                 + 0.00353175960027957 w_k10_KO_s135
                                 + 0.00671259493819927 w_k10_KO_s136
                                 + 0.00306896910306851 w_k10_KO_s137
                                 + 0.0112559341652893 w_k10_MAT_s000
                                 + 0.0266543205084448 w_k10_MAT_s001
                                 + 0.0158343865244148 w_k10_MAT_s002
                                 + 0.0379762213682995 w_k10_MAT_s003
                                 + 0.012981158774322 w_k10_MAT_s004
                                 + 0.0323144443894677 w_k10_MAT_s005
                                 + 0.0445132870301452 w_k10_MAT_s006
                                 + 0.0446389070732653 w_k10_MAT_s007
                                 + 0.0475213831995275 w_k10_MAT_s010
                                 + 0.0389885324921504 w_k10_MAT_s011
                                 + 0.0227651566264596 w_k10_MAT_s012
                                 + 0.0167490870790252 w_k10_MAT_s013
                                 + 0.0253342662166422 w_k10_MAT_s014
                                 + 0.0166549641511469 w_k10_MAT_s015
                                 + 0.0165482488789426 w_k10_MAT_s016
                                 + 0.0148151531207046 w_k10_MAT_s017
                                 + 0.0338346296150562 w_k10_MAT_s020
                                 + 0.0275809171687978 w_k10_MAT_s021
                                 + 0.024759271135526 w_k10_MAT_s022
                                 + 0.0246532667737293 w_k10_MAT_s023
                                 + 0.0238266853668257 w_k10_MAT_s024
                                 + 0.0158390353551663 w_k10_MAT_s025
                                 + 0.0444242298647155 w_k10_MAT_s026
                                 + 0.0482877154121914 w_k10_MAT_s027
                                 + 0.0166628745896955 w_k10_MAT_s030
                                 + 0.00761397522818525 w_k10_MAT_s031
                                 + 0.0141011954082929 w_k10_MAT_s032
                                 + 0.00858572415330992 w_k10_MAT_s033
                                 + 0.00593867954632206 w_k10_MAT_s034
                                 + 0.00464775415280581 w_k10_MAT_s035
                                 + 0.0187968519189338 w_k10_MAT_s036
                                 + 0.00395243699071003 w_k10_MAT_s037
                                 + 0.00798218791704643 w_k10_MAT_s100
                                 + 0.00968749296023779 w_k10_MAT_s101
                                 + 0.00962254161323336 w_k10_MAT_s102
                                 + 0.0104176168820086 w_k10_MAT_s103
                                 + 0.00808606141260586 w_k10_MAT_s104
                                 + 0.0129368929722536 w_k10_MAT_s105
                                 + 0.0202225021937366 w_k10_MAT_s106
                                 + 0.00726317540040623 w_k10_MAT_s107
                                 + 0.0239220076824813 w_k10_MAT_s110
                                 + 0.0106379607892427 w_k10_MAT_s111
                                 + 0.0218727763848139 w_k10_MAT_s112
                                 + 0.0139901792572501 w_k10_MAT_s113
                                 + 0.0087962425218997 w_k10_MAT_s114
                                 + 0.0209218254092845 w_k10_MAT_s115
                                 + 0.0163281638392875 w_k10_MAT_s116
                                 + 0.0268928522747533 w_k10_MAT_s117
                                 + 0.00592975362972207 w_k10_MAT_s120
                                 + 0.00610570582139255 w_k10_MAT_s121
                                 + 0.00544714624268268 w_k10_MAT_s122
                                 + 0.00785251397916633 w_k10_MAT_s123
                                 + 0.013900605804919 w_k10_MAT_s124
                                 + 0.00375262521559568 w_k10_MAT_s125
                                 + 0.00344832490455798 w_k10_MAT_s126
                                 + 0.00535402688588534 w_k10_MAT_s127
                                 + 0.00266069274176197 w_k10_MAT_s130
                                 + 0.00142713203700813 w_k10_MAT_s131
                                 + 0.00677208253555559 w_k10_MAT_s132
                                 + 0.00357224511499274 w_k10_MAT_s133
                                 + 0.00259908077842785 w_k10_MAT_s134
                                 + 0.00520496345332956 w_k10_MAT_s135
                                 + 0.00726016523022608 w_k10_MAT_s136
                                 + 0.00333263723216223 w_k10_MAT_s137
                                 + 0.0116474145971771 w_k10_MSFT_s000
                                 + 0.0241313193831259 w_k10_MSFT_s001
                                 + 0.0163583293471587 w_k10_MSFT_s002
                                 + 0.0415121258648373 w_k10_MSFT_s003
                                 + 0.0145104463707814 w_k10_MSFT_s004
                                 + 0.0331123439247328 w_k10_MSFT_s005
                                 + 0.0475987921899884 w_k10_MSFT_s006
                                 + 0.0295019028093847 w_k10_MSFT_s007
                                 + 0.0331401961732416 w_k10_MSFT_s010
                                 + 0.0362414184965753 w_k10_MSFT_s011
                                 + 0.0183653559632701 w_k10_MSFT_s012
                                 + 0.0193833395244597 w_k10_MSFT_s013
                                 + 0.0249769044214995 w_k10_MSFT_s014
                                 + 0.0170802004568058 w_k10_MSFT_s015
                                 + 0.0111368639922038 w_k10_MSFT_s016
                                 + 0.0132330986066148 w_k10_MSFT_s017
                                 + 0.040962682124211 w_k10_MSFT_s020
                                 + 0.0255886672607233 w_k10_MSFT_s021
                                 + 0.0218752936965271 w_k10_MSFT_s022
                                 + 0.0239084394956147 w_k10_MSFT_s023
                                 + 0.0252245784085959 w_k10_MSFT_s024
                                 + 0.0154579031034015 w_k10_MSFT_s025
                                 + 0.0364141156639683 w_k10_MSFT_s026
                                 + 0.0425069501035461 w_k10_MSFT_s027
                                 + 0.0162254633351276 w_k10_MSFT_s030
                                 + 0.00801865966367392 w_k10_MSFT_s031
                                 + 0.0122149437388896 w_k10_MSFT_s032
                                 + 0.00803360123893122 w_k10_MSFT_s033
                                 + 0.00466142711318947 w_k10_MSFT_s034
                                 + 0.00575886531460584 w_k10_MSFT_s035
                                 + 0.0155206839298626 w_k10_MSFT_s036
                                 + 0.0032427100116999 w_k10_MSFT_s037
                                 + 0.00825964720836772 w_k10_MSFT_s100
                                 + 0.00960348854322686 w_k10_MSFT_s101
                                 + 0.00927982151489501 w_k10_MSFT_s102
                                 + 0.00941380236132346 w_k10_MSFT_s103
                                 + 0.00784136435062102 w_k10_MSFT_s104
                                 + 0.0121382828922354 w_k10_MSFT_s105
                                 + 0.0223559663230674 w_k10_MSFT_s106
                                 + 0.00555897062031628 w_k10_MSFT_s107
                                 + 0.0280363182041778 w_k10_MSFT_s110
                                 + 0.0139174499866259 w_k10_MSFT_s111
                                 + 0.0244558791991361 w_k10_MSFT_s112
                                 + 0.012005408384888 w_k10_MSFT_s113
                                 + 0.0104217351648458 w_k10_MSFT_s114
                                 + 0.0191777231055005 w_k10_MSFT_s115
                                 + 0.0181066223832824 w_k10_MSFT_s116
                                 + 0.0241045024215142 w_k10_MSFT_s117
                                 + 0.00556100114729618 w_k10_MSFT_s120
                                 + 0.00762415288619516 w_k10_MSFT_s121
                                 + 0.00550096384479441 w_k10_MSFT_s122
                                 + 0.00930449083801976 w_k10_MSFT_s123
                                 + 0.0114668756256156 w_k10_MSFT_s124
                                 + 0.00419586373405596 w_k10_MSFT_s125
                                 + 0.00259796318498076 w_k10_MSFT_s126
                                 + 0.00553629469598791 w_k10_MSFT_s127
                                 + 0.0023061431287186 w_k10_MSFT_s130
                                 + 0.00232226858110508 w_k10_MSFT_s131
                                 + 0.00671628441031347 w_k10_MSFT_s132
                                 + 0.00347921421597536 w_k10_MSFT_s133
                                 + 0.00240848642601175 w_k10_MSFT_s134
                                 + 0.0031047326019473 w_k10_MSFT_s135
                                 + 0.00654585997173245 w_k10_MSFT_s136
                                 + 0.00320922199224894 w_k10_MSFT_s137
                                 + 0.0125083709529778 w_k10_PG_s000
                                 + 0.023991166992052 w_k10_PG_s001
                                 + 0.0164018653713955 w_k10_PG_s002
                                 + 0.0379483651107175 w_k10_PG_s003
                                 + 0.0151830939725291 w_k10_PG_s004
                                 + 0.0324288634633624 w_k10_PG_s005
                                 + 0.0495090673002213 w_k10_PG_s006
                                 + 0.0301785824462786 w_k10_PG_s007
                                 + 0.0346579078615835 w_k10_PG_s010
                                 + 0.0376312466666807 w_k10_PG_s011
                                 + 0.0208592888991367 w_k10_PG_s012
                                 + 0.0183865748716211 w_k10_PG_s013
                                 + 0.0274240407870082 w_k10_PG_s014
                                 + 0.0165789749981134 w_k10_PG_s015
                                 + 0.0120965457024176 w_k10_PG_s016
                                 + 0.0144492615389087 w_k10_PG_s017
                                 + 0.0406741519579045 w_k10_PG_s020
                                 + 0.0270866417595202 w_k10_PG_s021
                                 + 0.021986564163867 w_k10_PG_s022
                                 + 0.0219053650314037 w_k10_PG_s023
                                 + 0.0264036451627192 w_k10_PG_s024
                                 + 0.016621747905516 w_k10_PG_s025
                                 + 0.0341893229776963 w_k10_PG_s026
                                 + 0.0430783790702998 w_k10_PG_s027
                                 + 0.0154836875235096 w_k10_PG_s030
                                 + 0.00691071907782922 w_k10_PG_s031
                                 + 0.0130693683946932 w_k10_PG_s032
                                 + 0.00751600284111177 w_k10_PG_s033
                                 + 0.00467145630428343 w_k10_PG_s034
                                 + 0.0061862324045502 w_k10_PG_s035
                                 + 0.0168184482698001 w_k10_PG_s036
                                 + 0.00336353286305624 w_k10_PG_s037
                                 + 0.00889009179644835 w_k10_PG_s100
                                 + 0.00991511966579972 w_k10_PG_s101
                                 + 0.00885445872883505 w_k10_PG_s102
                                 + 0.0104848446243429 w_k10_PG_s103
                                 + 0.00726003696831001 w_k10_PG_s104
                                 + 0.0117539618742879 w_k10_PG_s105
                                 + 0.0213635581684798 w_k10_PG_s106
                                 + 0.00593799611939741 w_k10_PG_s107
                                 + 0.0290682881294693 w_k10_PG_s110
                                 + 0.0125504010131335 w_k10_PG_s111
                                 + 0.0249560171956225 w_k10_PG_s112
                                 + 0.0116713227792349 w_k10_PG_s113
                                 + 0.00943923984013076 w_k10_PG_s114
                                 + 0.0183854523717508 w_k10_PG_s115
                                 + 0.0182229991199997 w_k10_PG_s116
                                 + 0.022548829877052 w_k10_PG_s117
                                 + 0.00622586757966691 w_k10_PG_s120
                                 + 0.00722813831929288 w_k10_PG_s121
                                 + 0.00566111295468497 w_k10_PG_s122
                                 + 0.00859385612794844 w_k10_PG_s123
                                 + 0.0123014262652744 w_k10_PG_s124
                                 + 0.00410144894048849 w_k10_PG_s125
                                 + 0.0023919249161774 w_k10_PG_s126
                                 + 0.0054698655739025 w_k10_PG_s127
                                 + 0.00230351974092766 w_k10_PG_s130
                                 + 0.00217809883137363 w_k10_PG_s131
                                 + 0.00635778611080236 w_k10_PG_s132
                                 + 0.00321181066727306 w_k10_PG_s133
                                 + 0.00235589574572991 w_k10_PG_s134
                                 + 0.00348652884846385 w_k10_PG_s135
                                 + 0.00710075989808055 w_k10_PG_s136
                                 + 0.00316641412252186 w_k10_PG_s137
                                 >= 1.23224818464593
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
 0 <= w_k10_ADM_s0 <= 1
 0 <= w_k10_ADM_s1 <= 1
 0 <= w_k10_ADM_s00 <= 1
 0 <= w_k10_ADM_s01 <= 1
 0 <= w_k10_ADM_s02 <= 1
 0 <= w_k10_ADM_s03 <= 1
 0 <= w_k10_ADM_s10 <= 1
 0 <= w_k10_ADM_s11 <= 1
 0 <= w_k10_ADM_s12 <= 1
 0 <= w_k10_ADM_s13 <= 1
 0 <= w_k10_ADM_s000 <= 1
 0 <= w_k10_ADM_s001 <= 1
 0 <= w_k10_ADM_s002 <= 1
 0 <= w_k10_ADM_s003 <= 1
 0 <= w_k10_ADM_s004 <= 1
 0 <= w_k10_ADM_s005 <= 1
 0 <= w_k10_ADM_s006 <= 1
 0 <= w_k10_ADM_s007 <= 1
 0 <= w_k10_ADM_s010 <= 1
 0 <= w_k10_ADM_s011 <= 1
 0 <= w_k10_ADM_s012 <= 1
 0 <= w_k10_ADM_s013 <= 1
 0 <= w_k10_ADM_s014 <= 1
 0 <= w_k10_ADM_s015 <= 1
 0 <= w_k10_ADM_s016 <= 1
 0 <= w_k10_ADM_s017 <= 1
 0 <= w_k10_ADM_s020 <= 1
 0 <= w_k10_ADM_s021 <= 1
 0 <= w_k10_ADM_s022 <= 1
 0 <= w_k10_ADM_s023 <= 1
 0 <= w_k10_ADM_s024 <= 1
 0 <= w_k10_ADM_s025 <= 1
 0 <= w_k10_ADM_s026 <= 1
 0 <= w_k10_ADM_s027 <= 1
 0 <= w_k10_ADM_s030 <= 1
 0 <= w_k10_ADM_s031 <= 1
 0 <= w_k10_ADM_s032 <= 1
 0 <= w_k10_ADM_s033 <= 1
 0 <= w_k10_ADM_s034 <= 1
 0 <= w_k10_ADM_s035 <= 1
 0 <= w_k10_ADM_s036 <= 1
 0 <= w_k10_ADM_s037 <= 1
 0 <= w_k10_ADM_s100 <= 1
 0 <= w_k10_ADM_s101 <= 1
 0 <= w_k10_ADM_s102 <= 1
 0 <= w_k10_ADM_s103 <= 1
 0 <= w_k10_ADM_s104 <= 1
 0 <= w_k10_ADM_s105 <= 1
 0 <= w_k10_ADM_s106 <= 1
 0 <= w_k10_ADM_s107 <= 1
 0 <= w_k10_ADM_s110 <= 1
 0 <= w_k10_ADM_s111 <= 1
 0 <= w_k10_ADM_s112 <= 1
 0 <= w_k10_ADM_s113 <= 1
 0 <= w_k10_ADM_s114 <= 1
 0 <= w_k10_ADM_s115 <= 1
 0 <= w_k10_ADM_s116 <= 1
 0 <= w_k10_ADM_s117 <= 1
 0 <= w_k10_ADM_s120 <= 1
 0 <= w_k10_ADM_s121 <= 1
 0 <= w_k10_ADM_s122 <= 1
 0 <= w_k10_ADM_s123 <= 1
 0 <= w_k10_ADM_s124 <= 1
 0 <= w_k10_ADM_s125 <= 1
 0 <= w_k10_ADM_s126 <= 1
 0 <= w_k10_ADM_s127 <= 1
 0 <= w_k10_ADM_s130 <= 1
 0 <= w_k10_ADM_s131 <= 1
 0 <= w_k10_ADM_s132 <= 1
 0 <= w_k10_ADM_s133 <= 1
 0 <= w_k10_ADM_s134 <= 1
 0 <= w_k10_ADM_s135 <= 1
 0 <= w_k10_ADM_s136 <= 1
 0 <= w_k10_ADM_s137 <= 1
 0 <= w_k10_AXP_s0 <= 1
 0 <= w_k10_AXP_s1 <= 1
 0 <= w_k10_AXP_s00 <= 1
 0 <= w_k10_AXP_s01 <= 1
 0 <= w_k10_AXP_s02 <= 1
 0 <= w_k10_AXP_s03 <= 1
 0 <= w_k10_AXP_s10 <= 1
 0 <= w_k10_AXP_s11 <= 1
 0 <= w_k10_AXP_s12 <= 1
 0 <= w_k10_AXP_s13 <= 1
 0 <= w_k10_AXP_s000 <= 1
 0 <= w_k10_AXP_s001 <= 1
 0 <= w_k10_AXP_s002 <= 1
 0 <= w_k10_AXP_s003 <= 1
 0 <= w_k10_AXP_s004 <= 1
 0 <= w_k10_AXP_s005 <= 1
 0 <= w_k10_AXP_s006 <= 1
 0 <= w_k10_AXP_s007 <= 1
 0 <= w_k10_AXP_s010 <= 1
 0 <= w_k10_AXP_s011 <= 1
 0 <= w_k10_AXP_s012 <= 1
 0 <= w_k10_AXP_s013 <= 1
 0 <= w_k10_AXP_s014 <= 1
 0 <= w_k10_AXP_s015 <= 1
 0 <= w_k10_AXP_s016 <= 1
 0 <= w_k10_AXP_s017 <= 1
 0 <= w_k10_AXP_s020 <= 1
 0 <= w_k10_AXP_s021 <= 1
 0 <= w_k10_AXP_s022 <= 1
 0 <= w_k10_AXP_s023 <= 1
 0 <= w_k10_AXP_s024 <= 1
 0 <= w_k10_AXP_s025 <= 1
 0 <= w_k10_AXP_s026 <= 1
 0 <= w_k10_AXP_s027 <= 1
 0 <= w_k10_AXP_s030 <= 1
 0 <= w_k10_AXP_s031 <= 1
 0 <= w_k10_AXP_s032 <= 1
 0 <= w_k10_AXP_s033 <= 1
 0 <= w_k10_AXP_s034 <= 1
 0 <= w_k10_AXP_s035 <= 1
 0 <= w_k10_AXP_s036 <= 1
 0 <= w_k10_AXP_s037 <= 1
 0 <= w_k10_AXP_s100 <= 1
 0 <= w_k10_AXP_s101 <= 1
 0 <= w_k10_AXP_s102 <= 1
 0 <= w_k10_AXP_s103 <= 1
 0 <= w_k10_AXP_s104 <= 1
 0 <= w_k10_AXP_s105 <= 1
 0 <= w_k10_AXP_s106 <= 1
 0 <= w_k10_AXP_s107 <= 1
 0 <= w_k10_AXP_s110 <= 1
 0 <= w_k10_AXP_s111 <= 1
 0 <= w_k10_AXP_s112 <= 1
 0 <= w_k10_AXP_s113 <= 1
 0 <= w_k10_AXP_s114 <= 1
 0 <= w_k10_AXP_s115 <= 1
 0 <= w_k10_AXP_s116 <= 1
 0 <= w_k10_AXP_s117 <= 1
 0 <= w_k10_AXP_s120 <= 1
 0 <= w_k10_AXP_s121 <= 1
 0 <= w_k10_AXP_s122 <= 1
 0 <= w_k10_AXP_s123 <= 1
 0 <= w_k10_AXP_s124 <= 1
 0 <= w_k10_AXP_s125 <= 1
 0 <= w_k10_AXP_s126 <= 1
 0 <= w_k10_AXP_s127 <= 1
 0 <= w_k10_AXP_s130 <= 1
 0 <= w_k10_AXP_s131 <= 1
 0 <= w_k10_AXP_s132 <= 1
 0 <= w_k10_AXP_s133 <= 1
 0 <= w_k10_AXP_s134 <= 1
 0 <= w_k10_AXP_s135 <= 1
 0 <= w_k10_AXP_s136 <= 1
 0 <= w_k10_AXP_s137 <= 1
 0 <= w_k10_DIS_s0 <= 1
 0 <= w_k10_DIS_s1 <= 1
 0 <= w_k10_DIS_s00 <= 1
 0 <= w_k10_DIS_s01 <= 1
 0 <= w_k10_DIS_s02 <= 1
 0 <= w_k10_DIS_s03 <= 1
 0 <= w_k10_DIS_s10 <= 1
 0 <= w_k10_DIS_s11 <= 1
 0 <= w_k10_DIS_s12 <= 1
 0 <= w_k10_DIS_s13 <= 1
 0 <= w_k10_DIS_s000 <= 1
 0 <= w_k10_DIS_s001 <= 1
 0 <= w_k10_DIS_s002 <= 1
 0 <= w_k10_DIS_s003 <= 1
 0 <= w_k10_DIS_s004 <= 1
 0 <= w_k10_DIS_s005 <= 1
 0 <= w_k10_DIS_s006 <= 1
 0 <= w_k10_DIS_s007 <= 1
 0 <= w_k10_DIS_s010 <= 1
 0 <= w_k10_DIS_s011 <= 1
 0 <= w_k10_DIS_s012 <= 1
 0 <= w_k10_DIS_s013 <= 1
 0 <= w_k10_DIS_s014 <= 1
 0 <= w_k10_DIS_s015 <= 1
 0 <= w_k10_DIS_s016 <= 1
 0 <= w_k10_DIS_s017 <= 1
 0 <= w_k10_DIS_s020 <= 1
 0 <= w_k10_DIS_s021 <= 1
 0 <= w_k10_DIS_s022 <= 1
 0 <= w_k10_DIS_s023 <= 1
 0 <= w_k10_DIS_s024 <= 1
 0 <= w_k10_DIS_s025 <= 1
 0 <= w_k10_DIS_s026 <= 1
 0 <= w_k10_DIS_s027 <= 1
 0 <= w_k10_DIS_s030 <= 1
 0 <= w_k10_DIS_s031 <= 1
 0 <= w_k10_DIS_s032 <= 1
 0 <= w_k10_DIS_s033 <= 1
 0 <= w_k10_DIS_s034 <= 1
 0 <= w_k10_DIS_s035 <= 1
 0 <= w_k10_DIS_s036 <= 1
 0 <= w_k10_DIS_s037 <= 1
 0 <= w_k10_DIS_s100 <= 1
 0 <= w_k10_DIS_s101 <= 1
 0 <= w_k10_DIS_s102 <= 1
 0 <= w_k10_DIS_s103 <= 1
 0 <= w_k10_DIS_s104 <= 1
 0 <= w_k10_DIS_s105 <= 1
 0 <= w_k10_DIS_s106 <= 1
 0 <= w_k10_DIS_s107 <= 1
 0 <= w_k10_DIS_s110 <= 1
 0 <= w_k10_DIS_s111 <= 1
 0 <= w_k10_DIS_s112 <= 1
 0 <= w_k10_DIS_s113 <= 1
 0 <= w_k10_DIS_s114 <= 1
 0 <= w_k10_DIS_s115 <= 1
 0 <= w_k10_DIS_s116 <= 1
 0 <= w_k10_DIS_s117 <= 1
 0 <= w_k10_DIS_s120 <= 1
 0 <= w_k10_DIS_s121 <= 1
 0 <= w_k10_DIS_s122 <= 1
 0 <= w_k10_DIS_s123 <= 1
 0 <= w_k10_DIS_s124 <= 1
 0 <= w_k10_DIS_s125 <= 1
 0 <= w_k10_DIS_s126 <= 1
 0 <= w_k10_DIS_s127 <= 1
 0 <= w_k10_DIS_s130 <= 1
 0 <= w_k10_DIS_s131 <= 1
 0 <= w_k10_DIS_s132 <= 1
 0 <= w_k10_DIS_s133 <= 1
 0 <= w_k10_DIS_s134 <= 1
 0 <= w_k10_DIS_s135 <= 1
 0 <= w_k10_DIS_s136 <= 1
 0 <= w_k10_DIS_s137 <= 1
 0 <= w_k10_FDX_s0 <= 1
 0 <= w_k10_FDX_s1 <= 1
 0 <= w_k10_FDX_s00 <= 1
 0 <= w_k10_FDX_s01 <= 1
 0 <= w_k10_FDX_s02 <= 1
 0 <= w_k10_FDX_s03 <= 1
 0 <= w_k10_FDX_s10 <= 1
 0 <= w_k10_FDX_s11 <= 1
 0 <= w_k10_FDX_s12 <= 1
 0 <= w_k10_FDX_s13 <= 1
 0 <= w_k10_FDX_s000 <= 1
 0 <= w_k10_FDX_s001 <= 1
 0 <= w_k10_FDX_s002 <= 1
 0 <= w_k10_FDX_s003 <= 1
 0 <= w_k10_FDX_s004 <= 1
 0 <= w_k10_FDX_s005 <= 1
 0 <= w_k10_FDX_s006 <= 1
 0 <= w_k10_FDX_s007 <= 1
 0 <= w_k10_FDX_s010 <= 1
 0 <= w_k10_FDX_s011 <= 1
 0 <= w_k10_FDX_s012 <= 1
 0 <= w_k10_FDX_s013 <= 1
 0 <= w_k10_FDX_s014 <= 1
 0 <= w_k10_FDX_s015 <= 1
 0 <= w_k10_FDX_s016 <= 1
 0 <= w_k10_FDX_s017 <= 1
 0 <= w_k10_FDX_s020 <= 1
 0 <= w_k10_FDX_s021 <= 1
 0 <= w_k10_FDX_s022 <= 1
 0 <= w_k10_FDX_s023 <= 1
 0 <= w_k10_FDX_s024 <= 1
 0 <= w_k10_FDX_s025 <= 1
 0 <= w_k10_FDX_s026 <= 1
 0 <= w_k10_FDX_s027 <= 1
 0 <= w_k10_FDX_s030 <= 1
 0 <= w_k10_FDX_s031 <= 1
 0 <= w_k10_FDX_s032 <= 1
 0 <= w_k10_FDX_s033 <= 1
 0 <= w_k10_FDX_s034 <= 1
 0 <= w_k10_FDX_s035 <= 1
 0 <= w_k10_FDX_s036 <= 1
 0 <= w_k10_FDX_s037 <= 1
 0 <= w_k10_FDX_s100 <= 1
 0 <= w_k10_FDX_s101 <= 1
 0 <= w_k10_FDX_s102 <= 1
 0 <= w_k10_FDX_s103 <= 1
 0 <= w_k10_FDX_s104 <= 1
 0 <= w_k10_FDX_s105 <= 1
 0 <= w_k10_FDX_s106 <= 1
 0 <= w_k10_FDX_s107 <= 1
 0 <= w_k10_FDX_s110 <= 1
 0 <= w_k10_FDX_s111 <= 1
 0 <= w_k10_FDX_s112 <= 1
 0 <= w_k10_FDX_s113 <= 1
 0 <= w_k10_FDX_s114 <= 1
 0 <= w_k10_FDX_s115 <= 1
 0 <= w_k10_FDX_s116 <= 1
 0 <= w_k10_FDX_s117 <= 1
 0 <= w_k10_FDX_s120 <= 1
 0 <= w_k10_FDX_s121 <= 1
 0 <= w_k10_FDX_s122 <= 1
 0 <= w_k10_FDX_s123 <= 1
 0 <= w_k10_FDX_s124 <= 1
 0 <= w_k10_FDX_s125 <= 1
 0 <= w_k10_FDX_s126 <= 1
 0 <= w_k10_FDX_s127 <= 1
 0 <= w_k10_FDX_s130 <= 1
 0 <= w_k10_FDX_s131 <= 1
 0 <= w_k10_FDX_s132 <= 1
 0 <= w_k10_FDX_s133 <= 1
 0 <= w_k10_FDX_s134 <= 1
 0 <= w_k10_FDX_s135 <= 1
 0 <= w_k10_FDX_s136 <= 1
 0 <= w_k10_FDX_s137 <= 1
 0 <= w_k10_IBM_s0 <= 1
 0 <= w_k10_IBM_s1 <= 1
 0 <= w_k10_IBM_s00 <= 1
 0 <= w_k10_IBM_s01 <= 1
 0 <= w_k10_IBM_s02 <= 1
 0 <= w_k10_IBM_s03 <= 1
 0 <= w_k10_IBM_s10 <= 1
 0 <= w_k10_IBM_s11 <= 1
 0 <= w_k10_IBM_s12 <= 1
 0 <= w_k10_IBM_s13 <= 1
 0 <= w_k10_IBM_s000 <= 1
 0 <= w_k10_IBM_s001 <= 1
 0 <= w_k10_IBM_s002 <= 1
 0 <= w_k10_IBM_s003 <= 1
 0 <= w_k10_IBM_s004 <= 1
 0 <= w_k10_IBM_s005 <= 1
 0 <= w_k10_IBM_s006 <= 1
 0 <= w_k10_IBM_s007 <= 1
 0 <= w_k10_IBM_s010 <= 1
 0 <= w_k10_IBM_s011 <= 1
 0 <= w_k10_IBM_s012 <= 1
 0 <= w_k10_IBM_s013 <= 1
 0 <= w_k10_IBM_s014 <= 1
 0 <= w_k10_IBM_s015 <= 1
 0 <= w_k10_IBM_s016 <= 1
 0 <= w_k10_IBM_s017 <= 1
 0 <= w_k10_IBM_s020 <= 1
 0 <= w_k10_IBM_s021 <= 1
 0 <= w_k10_IBM_s022 <= 1
 0 <= w_k10_IBM_s023 <= 1
 0 <= w_k10_IBM_s024 <= 1
 0 <= w_k10_IBM_s025 <= 1
 0 <= w_k10_IBM_s026 <= 1
 0 <= w_k10_IBM_s027 <= 1
 0 <= w_k10_IBM_s030 <= 1
 0 <= w_k10_IBM_s031 <= 1
 0 <= w_k10_IBM_s032 <= 1
 0 <= w_k10_IBM_s033 <= 1
 0 <= w_k10_IBM_s034 <= 1
 0 <= w_k10_IBM_s035 <= 1
 0 <= w_k10_IBM_s036 <= 1
 0 <= w_k10_IBM_s037 <= 1
 0 <= w_k10_IBM_s100 <= 1
 0 <= w_k10_IBM_s101 <= 1
 0 <= w_k10_IBM_s102 <= 1
 0 <= w_k10_IBM_s103 <= 1
 0 <= w_k10_IBM_s104 <= 1
 0 <= w_k10_IBM_s105 <= 1
 0 <= w_k10_IBM_s106 <= 1
 0 <= w_k10_IBM_s107 <= 1
 0 <= w_k10_IBM_s110 <= 1
 0 <= w_k10_IBM_s111 <= 1
 0 <= w_k10_IBM_s112 <= 1
 0 <= w_k10_IBM_s113 <= 1
 0 <= w_k10_IBM_s114 <= 1
 0 <= w_k10_IBM_s115 <= 1
 0 <= w_k10_IBM_s116 <= 1
 0 <= w_k10_IBM_s117 <= 1
 0 <= w_k10_IBM_s120 <= 1
 0 <= w_k10_IBM_s121 <= 1
 0 <= w_k10_IBM_s122 <= 1
 0 <= w_k10_IBM_s123 <= 1
 0 <= w_k10_IBM_s124 <= 1
 0 <= w_k10_IBM_s125 <= 1
 0 <= w_k10_IBM_s126 <= 1
 0 <= w_k10_IBM_s127 <= 1
 0 <= w_k10_IBM_s130 <= 1
 0 <= w_k10_IBM_s131 <= 1
 0 <= w_k10_IBM_s132 <= 1
 0 <= w_k10_IBM_s133 <= 1
 0 <= w_k10_IBM_s134 <= 1
 0 <= w_k10_IBM_s135 <= 1
 0 <= w_k10_IBM_s136 <= 1
 0 <= w_k10_IBM_s137 <= 1
 0 <= w_k10_INTC_s0 <= 1
 0 <= w_k10_INTC_s1 <= 1
 0 <= w_k10_INTC_s00 <= 1
 0 <= w_k10_INTC_s01 <= 1
 0 <= w_k10_INTC_s02 <= 1
 0 <= w_k10_INTC_s03 <= 1
 0 <= w_k10_INTC_s10 <= 1
 0 <= w_k10_INTC_s11 <= 1
 0 <= w_k10_INTC_s12 <= 1
 0 <= w_k10_INTC_s13 <= 1
 0 <= w_k10_INTC_s000 <= 1
 0 <= w_k10_INTC_s001 <= 1
 0 <= w_k10_INTC_s002 <= 1
 0 <= w_k10_INTC_s003 <= 1
 0 <= w_k10_INTC_s004 <= 1
 0 <= w_k10_INTC_s005 <= 1
 0 <= w_k10_INTC_s006 <= 1
 0 <= w_k10_INTC_s007 <= 1
 0 <= w_k10_INTC_s010 <= 1
 0 <= w_k10_INTC_s011 <= 1
 0 <= w_k10_INTC_s012 <= 1
 0 <= w_k10_INTC_s013 <= 1
 0 <= w_k10_INTC_s014 <= 1
 0 <= w_k10_INTC_s015 <= 1
 0 <= w_k10_INTC_s016 <= 1
 0 <= w_k10_INTC_s017 <= 1
 0 <= w_k10_INTC_s020 <= 1
 0 <= w_k10_INTC_s021 <= 1
 0 <= w_k10_INTC_s022 <= 1
 0 <= w_k10_INTC_s023 <= 1
 0 <= w_k10_INTC_s024 <= 1
 0 <= w_k10_INTC_s025 <= 1
 0 <= w_k10_INTC_s026 <= 1
 0 <= w_k10_INTC_s027 <= 1
 0 <= w_k10_INTC_s030 <= 1
 0 <= w_k10_INTC_s031 <= 1
 0 <= w_k10_INTC_s032 <= 1
 0 <= w_k10_INTC_s033 <= 1
 0 <= w_k10_INTC_s034 <= 1
 0 <= w_k10_INTC_s035 <= 1
 0 <= w_k10_INTC_s036 <= 1
 0 <= w_k10_INTC_s037 <= 1
 0 <= w_k10_INTC_s100 <= 1
 0 <= w_k10_INTC_s101 <= 1
 0 <= w_k10_INTC_s102 <= 1
 0 <= w_k10_INTC_s103 <= 1
 0 <= w_k10_INTC_s104 <= 1
 0 <= w_k10_INTC_s105 <= 1
 0 <= w_k10_INTC_s106 <= 1
 0 <= w_k10_INTC_s107 <= 1
 0 <= w_k10_INTC_s110 <= 1
 0 <= w_k10_INTC_s111 <= 1
 0 <= w_k10_INTC_s112 <= 1
 0 <= w_k10_INTC_s113 <= 1
 0 <= w_k10_INTC_s114 <= 1
 0 <= w_k10_INTC_s115 <= 1
 0 <= w_k10_INTC_s116 <= 1
 0 <= w_k10_INTC_s117 <= 1
 0 <= w_k10_INTC_s120 <= 1
 0 <= w_k10_INTC_s121 <= 1
 0 <= w_k10_INTC_s122 <= 1
 0 <= w_k10_INTC_s123 <= 1
 0 <= w_k10_INTC_s124 <= 1
 0 <= w_k10_INTC_s125 <= 1
 0 <= w_k10_INTC_s126 <= 1
 0 <= w_k10_INTC_s127 <= 1
 0 <= w_k10_INTC_s130 <= 1
 0 <= w_k10_INTC_s131 <= 1
 0 <= w_k10_INTC_s132 <= 1
 0 <= w_k10_INTC_s133 <= 1
 0 <= w_k10_INTC_s134 <= 1
 0 <= w_k10_INTC_s135 <= 1
 0 <= w_k10_INTC_s136 <= 1
 0 <= w_k10_INTC_s137 <= 1
 0 <= w_k10_KO_s0 <= 1
 0 <= w_k10_KO_s1 <= 1
 0 <= w_k10_KO_s00 <= 1
 0 <= w_k10_KO_s01 <= 1
 0 <= w_k10_KO_s02 <= 1
 0 <= w_k10_KO_s03 <= 1
 0 <= w_k10_KO_s10 <= 1
 0 <= w_k10_KO_s11 <= 1
 0 <= w_k10_KO_s12 <= 1
 0 <= w_k10_KO_s13 <= 1
 0 <= w_k10_KO_s000 <= 1
 0 <= w_k10_KO_s001 <= 1
 0 <= w_k10_KO_s002 <= 1
 0 <= w_k10_KO_s003 <= 1
 0 <= w_k10_KO_s004 <= 1
 0 <= w_k10_KO_s005 <= 1
 0 <= w_k10_KO_s006 <= 1
 0 <= w_k10_KO_s007 <= 1
 0 <= w_k10_KO_s010 <= 1
 0 <= w_k10_KO_s011 <= 1
 0 <= w_k10_KO_s012 <= 1
 0 <= w_k10_KO_s013 <= 1
 0 <= w_k10_KO_s014 <= 1
 0 <= w_k10_KO_s015 <= 1
 0 <= w_k10_KO_s016 <= 1
 0 <= w_k10_KO_s017 <= 1
 0 <= w_k10_KO_s020 <= 1
 0 <= w_k10_KO_s021 <= 1
 0 <= w_k10_KO_s022 <= 1
 0 <= w_k10_KO_s023 <= 1
 0 <= w_k10_KO_s024 <= 1
 0 <= w_k10_KO_s025 <= 1
 0 <= w_k10_KO_s026 <= 1
 0 <= w_k10_KO_s027 <= 1
 0 <= w_k10_KO_s030 <= 1
 0 <= w_k10_KO_s031 <= 1
 0 <= w_k10_KO_s032 <= 1
 0 <= w_k10_KO_s033 <= 1
 0 <= w_k10_KO_s034 <= 1
 0 <= w_k10_KO_s035 <= 1
 0 <= w_k10_KO_s036 <= 1
 0 <= w_k10_KO_s037 <= 1
 0 <= w_k10_KO_s100 <= 1
 0 <= w_k10_KO_s101 <= 1
 0 <= w_k10_KO_s102 <= 1
 0 <= w_k10_KO_s103 <= 1
 0 <= w_k10_KO_s104 <= 1
 0 <= w_k10_KO_s105 <= 1
 0 <= w_k10_KO_s106 <= 1
 0 <= w_k10_KO_s107 <= 1
 0 <= w_k10_KO_s110 <= 1
 0 <= w_k10_KO_s111 <= 1
 0 <= w_k10_KO_s112 <= 1
 0 <= w_k10_KO_s113 <= 1
 0 <= w_k10_KO_s114 <= 1
 0 <= w_k10_KO_s115 <= 1
 0 <= w_k10_KO_s116 <= 1
 0 <= w_k10_KO_s117 <= 1
 0 <= w_k10_KO_s120 <= 1
 0 <= w_k10_KO_s121 <= 1
 0 <= w_k10_KO_s122 <= 1
 0 <= w_k10_KO_s123 <= 1
 0 <= w_k10_KO_s124 <= 1
 0 <= w_k10_KO_s125 <= 1
 0 <= w_k10_KO_s126 <= 1
 0 <= w_k10_KO_s127 <= 1
 0 <= w_k10_KO_s130 <= 1
 0 <= w_k10_KO_s131 <= 1
 0 <= w_k10_KO_s132 <= 1
 0 <= w_k10_KO_s133 <= 1
 0 <= w_k10_KO_s134 <= 1
 0 <= w_k10_KO_s135 <= 1
 0 <= w_k10_KO_s136 <= 1
 0 <= w_k10_KO_s137 <= 1
 0 <= w_k10_MAT_s0 <= 1
 0 <= w_k10_MAT_s1 <= 1
 0 <= w_k10_MAT_s00 <= 1
 0 <= w_k10_MAT_s01 <= 1
 0 <= w_k10_MAT_s02 <= 1
 0 <= w_k10_MAT_s03 <= 1
 0 <= w_k10_MAT_s10 <= 1
 0 <= w_k10_MAT_s11 <= 1
 0 <= w_k10_MAT_s12 <= 1
 0 <= w_k10_MAT_s13 <= 1
 0 <= w_k10_MAT_s000 <= 1
 0 <= w_k10_MAT_s001 <= 1
 0 <= w_k10_MAT_s002 <= 1
 0 <= w_k10_MAT_s003 <= 1
 0 <= w_k10_MAT_s004 <= 1
 0 <= w_k10_MAT_s005 <= 1
 0 <= w_k10_MAT_s006 <= 1
 0 <= w_k10_MAT_s007 <= 1
 0 <= w_k10_MAT_s010 <= 1
 0 <= w_k10_MAT_s011 <= 1
 0 <= w_k10_MAT_s012 <= 1
 0 <= w_k10_MAT_s013 <= 1
 0 <= w_k10_MAT_s014 <= 1
 0 <= w_k10_MAT_s015 <= 1
 0 <= w_k10_MAT_s016 <= 1
 0 <= w_k10_MAT_s017 <= 1
 0 <= w_k10_MAT_s020 <= 1
 0 <= w_k10_MAT_s021 <= 1
 0 <= w_k10_MAT_s022 <= 1
 0 <= w_k10_MAT_s023 <= 1
 0 <= w_k10_MAT_s024 <= 1
 0 <= w_k10_MAT_s025 <= 1
 0 <= w_k10_MAT_s026 <= 1
 0 <= w_k10_MAT_s027 <= 1
 0 <= w_k10_MAT_s030 <= 1
 0 <= w_k10_MAT_s031 <= 1
 0 <= w_k10_MAT_s032 <= 1
 0 <= w_k10_MAT_s033 <= 1
 0 <= w_k10_MAT_s034 <= 1
 0 <= w_k10_MAT_s035 <= 1
 0 <= w_k10_MAT_s036 <= 1
 0 <= w_k10_MAT_s037 <= 1
 0 <= w_k10_MAT_s100 <= 1
 0 <= w_k10_MAT_s101 <= 1
 0 <= w_k10_MAT_s102 <= 1
 0 <= w_k10_MAT_s103 <= 1
 0 <= w_k10_MAT_s104 <= 1
 0 <= w_k10_MAT_s105 <= 1
 0 <= w_k10_MAT_s106 <= 1
 0 <= w_k10_MAT_s107 <= 1
 0 <= w_k10_MAT_s110 <= 1
 0 <= w_k10_MAT_s111 <= 1
 0 <= w_k10_MAT_s112 <= 1
 0 <= w_k10_MAT_s113 <= 1
 0 <= w_k10_MAT_s114 <= 1
 0 <= w_k10_MAT_s115 <= 1
 0 <= w_k10_MAT_s116 <= 1
 0 <= w_k10_MAT_s117 <= 1
 0 <= w_k10_MAT_s120 <= 1
 0 <= w_k10_MAT_s121 <= 1
 0 <= w_k10_MAT_s122 <= 1
 0 <= w_k10_MAT_s123 <= 1
 0 <= w_k10_MAT_s124 <= 1
 0 <= w_k10_MAT_s125 <= 1
 0 <= w_k10_MAT_s126 <= 1
 0 <= w_k10_MAT_s127 <= 1
 0 <= w_k10_MAT_s130 <= 1
 0 <= w_k10_MAT_s131 <= 1
 0 <= w_k10_MAT_s132 <= 1
 0 <= w_k10_MAT_s133 <= 1
 0 <= w_k10_MAT_s134 <= 1
 0 <= w_k10_MAT_s135 <= 1
 0 <= w_k10_MAT_s136 <= 1
 0 <= w_k10_MAT_s137 <= 1
 0 <= w_k10_MSFT_s0 <= 1
 0 <= w_k10_MSFT_s1 <= 1
 0 <= w_k10_MSFT_s00 <= 1
 0 <= w_k10_MSFT_s01 <= 1
 0 <= w_k10_MSFT_s02 <= 1
 0 <= w_k10_MSFT_s03 <= 1
 0 <= w_k10_MSFT_s10 <= 1
 0 <= w_k10_MSFT_s11 <= 1
 0 <= w_k10_MSFT_s12 <= 1
 0 <= w_k10_MSFT_s13 <= 1
 0 <= w_k10_MSFT_s000 <= 1
 0 <= w_k10_MSFT_s001 <= 1
 0 <= w_k10_MSFT_s002 <= 1
 0 <= w_k10_MSFT_s003 <= 1
 0 <= w_k10_MSFT_s004 <= 1
 0 <= w_k10_MSFT_s005 <= 1
 0 <= w_k10_MSFT_s006 <= 1
 0 <= w_k10_MSFT_s007 <= 1
 0 <= w_k10_MSFT_s010 <= 1
 0 <= w_k10_MSFT_s011 <= 1
 0 <= w_k10_MSFT_s012 <= 1
 0 <= w_k10_MSFT_s013 <= 1
 0 <= w_k10_MSFT_s014 <= 1
 0 <= w_k10_MSFT_s015 <= 1
 0 <= w_k10_MSFT_s016 <= 1
 0 <= w_k10_MSFT_s017 <= 1
 0 <= w_k10_MSFT_s020 <= 1
 0 <= w_k10_MSFT_s021 <= 1
 0 <= w_k10_MSFT_s022 <= 1
 0 <= w_k10_MSFT_s023 <= 1
 0 <= w_k10_MSFT_s024 <= 1
 0 <= w_k10_MSFT_s025 <= 1
 0 <= w_k10_MSFT_s026 <= 1
 0 <= w_k10_MSFT_s027 <= 1
 0 <= w_k10_MSFT_s030 <= 1
 0 <= w_k10_MSFT_s031 <= 1
 0 <= w_k10_MSFT_s032 <= 1
 0 <= w_k10_MSFT_s033 <= 1
 0 <= w_k10_MSFT_s034 <= 1
 0 <= w_k10_MSFT_s035 <= 1
 0 <= w_k10_MSFT_s036 <= 1
 0 <= w_k10_MSFT_s037 <= 1
 0 <= w_k10_MSFT_s100 <= 1
 0 <= w_k10_MSFT_s101 <= 1
 0 <= w_k10_MSFT_s102 <= 1
 0 <= w_k10_MSFT_s103 <= 1
 0 <= w_k10_MSFT_s104 <= 1
 0 <= w_k10_MSFT_s105 <= 1
 0 <= w_k10_MSFT_s106 <= 1
 0 <= w_k10_MSFT_s107 <= 1
 0 <= w_k10_MSFT_s110 <= 1
 0 <= w_k10_MSFT_s111 <= 1
 0 <= w_k10_MSFT_s112 <= 1
 0 <= w_k10_MSFT_s113 <= 1
 0 <= w_k10_MSFT_s114 <= 1
 0 <= w_k10_MSFT_s115 <= 1
 0 <= w_k10_MSFT_s116 <= 1
 0 <= w_k10_MSFT_s117 <= 1
 0 <= w_k10_MSFT_s120 <= 1
 0 <= w_k10_MSFT_s121 <= 1
 0 <= w_k10_MSFT_s122 <= 1
 0 <= w_k10_MSFT_s123 <= 1
 0 <= w_k10_MSFT_s124 <= 1
 0 <= w_k10_MSFT_s125 <= 1
 0 <= w_k10_MSFT_s126 <= 1
 0 <= w_k10_MSFT_s127 <= 1
 0 <= w_k10_MSFT_s130 <= 1
 0 <= w_k10_MSFT_s131 <= 1
 0 <= w_k10_MSFT_s132 <= 1
 0 <= w_k10_MSFT_s133 <= 1
 0 <= w_k10_MSFT_s134 <= 1
 0 <= w_k10_MSFT_s135 <= 1
 0 <= w_k10_MSFT_s136 <= 1
 0 <= w_k10_MSFT_s137 <= 1
 0 <= w_k10_PG_s0 <= 1
 0 <= w_k10_PG_s1 <= 1
 0 <= w_k10_PG_s00 <= 1
 0 <= w_k10_PG_s01 <= 1
 0 <= w_k10_PG_s02 <= 1
 0 <= w_k10_PG_s03 <= 1
 0 <= w_k10_PG_s10 <= 1
 0 <= w_k10_PG_s11 <= 1
 0 <= w_k10_PG_s12 <= 1
 0 <= w_k10_PG_s13 <= 1
 0 <= w_k10_PG_s000 <= 1
 0 <= w_k10_PG_s001 <= 1
 0 <= w_k10_PG_s002 <= 1
 0 <= w_k10_PG_s003 <= 1
 0 <= w_k10_PG_s004 <= 1
 0 <= w_k10_PG_s005 <= 1
 0 <= w_k10_PG_s006 <= 1
 0 <= w_k10_PG_s007 <= 1
 0 <= w_k10_PG_s010 <= 1
 0 <= w_k10_PG_s011 <= 1
 0 <= w_k10_PG_s012 <= 1
 0 <= w_k10_PG_s013 <= 1
 0 <= w_k10_PG_s014 <= 1
 0 <= w_k10_PG_s015 <= 1
 0 <= w_k10_PG_s016 <= 1
 0 <= w_k10_PG_s017 <= 1
 0 <= w_k10_PG_s020 <= 1
 0 <= w_k10_PG_s021 <= 1
 0 <= w_k10_PG_s022 <= 1
 0 <= w_k10_PG_s023 <= 1
 0 <= w_k10_PG_s024 <= 1
 0 <= w_k10_PG_s025 <= 1
 0 <= w_k10_PG_s026 <= 1
 0 <= w_k10_PG_s027 <= 1
 0 <= w_k10_PG_s030 <= 1
 0 <= w_k10_PG_s031 <= 1
 0 <= w_k10_PG_s032 <= 1
 0 <= w_k10_PG_s033 <= 1
 0 <= w_k10_PG_s034 <= 1
 0 <= w_k10_PG_s035 <= 1
 0 <= w_k10_PG_s036 <= 1
 0 <= w_k10_PG_s037 <= 1
 0 <= w_k10_PG_s100 <= 1
 0 <= w_k10_PG_s101 <= 1
 0 <= w_k10_PG_s102 <= 1
 0 <= w_k10_PG_s103 <= 1
 0 <= w_k10_PG_s104 <= 1
 0 <= w_k10_PG_s105 <= 1
 0 <= w_k10_PG_s106 <= 1
 0 <= w_k10_PG_s107 <= 1
 0 <= w_k10_PG_s110 <= 1
 0 <= w_k10_PG_s111 <= 1
 0 <= w_k10_PG_s112 <= 1
 0 <= w_k10_PG_s113 <= 1
 0 <= w_k10_PG_s114 <= 1
 0 <= w_k10_PG_s115 <= 1
 0 <= w_k10_PG_s116 <= 1
 0 <= w_k10_PG_s117 <= 1
 0 <= w_k10_PG_s120 <= 1
 0 <= w_k10_PG_s121 <= 1
 0 <= w_k10_PG_s122 <= 1
 0 <= w_k10_PG_s123 <= 1
 0 <= w_k10_PG_s124 <= 1
 0 <= w_k10_PG_s125 <= 1
 0 <= w_k10_PG_s126 <= 1
 0 <= w_k10_PG_s127 <= 1
 0 <= w_k10_PG_s130 <= 1
 0 <= w_k10_PG_s131 <= 1
 0 <= w_k10_PG_s132 <= 1
 0 <= w_k10_PG_s133 <= 1
 0 <= w_k10_PG_s134 <= 1
 0 <= w_k10_PG_s135 <= 1
 0 <= w_k10_PG_s136 <= 1
 0 <= w_k10_PG_s137 <= 1
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
 0 <= b_k10_ADM_s0 <= 0.2
 0 <= b_k10_ADM_s1 <= 0.2
 0 <= b_k10_ADM_s00 <= 0.2
 0 <= b_k10_ADM_s01 <= 0.2
 0 <= b_k10_ADM_s02 <= 0.2
 0 <= b_k10_ADM_s03 <= 0.2
 0 <= b_k10_ADM_s10 <= 0.2
 0 <= b_k10_ADM_s11 <= 0.2
 0 <= b_k10_ADM_s12 <= 0.2
 0 <= b_k10_ADM_s13 <= 0.2
 0 <= b_k10_ADM_s000 <= 0.2
 0 <= b_k10_ADM_s001 <= 0.2
 0 <= b_k10_ADM_s002 <= 0.2
 0 <= b_k10_ADM_s003 <= 0.2
 0 <= b_k10_ADM_s004 <= 0.2
 0 <= b_k10_ADM_s005 <= 0.2
 0 <= b_k10_ADM_s006 <= 0.2
 0 <= b_k10_ADM_s007 <= 0.2
 0 <= b_k10_ADM_s010 <= 0.2
 0 <= b_k10_ADM_s011 <= 0.2
 0 <= b_k10_ADM_s012 <= 0.2
 0 <= b_k10_ADM_s013 <= 0.2
 0 <= b_k10_ADM_s014 <= 0.2
 0 <= b_k10_ADM_s015 <= 0.2
 0 <= b_k10_ADM_s016 <= 0.2
 0 <= b_k10_ADM_s017 <= 0.2
 0 <= b_k10_ADM_s020 <= 0.2
 0 <= b_k10_ADM_s021 <= 0.2
 0 <= b_k10_ADM_s022 <= 0.2
 0 <= b_k10_ADM_s023 <= 0.2
 0 <= b_k10_ADM_s024 <= 0.2
 0 <= b_k10_ADM_s025 <= 0.2
 0 <= b_k10_ADM_s026 <= 0.2
 0 <= b_k10_ADM_s027 <= 0.2
 0 <= b_k10_ADM_s030 <= 0.2
 0 <= b_k10_ADM_s031 <= 0.2
 0 <= b_k10_ADM_s032 <= 0.2
 0 <= b_k10_ADM_s033 <= 0.2
 0 <= b_k10_ADM_s034 <= 0.2
 0 <= b_k10_ADM_s035 <= 0.2
 0 <= b_k10_ADM_s036 <= 0.2
 0 <= b_k10_ADM_s037 <= 0.2
 0 <= b_k10_ADM_s100 <= 0.2
 0 <= b_k10_ADM_s101 <= 0.2
 0 <= b_k10_ADM_s102 <= 0.2
 0 <= b_k10_ADM_s103 <= 0.2
 0 <= b_k10_ADM_s104 <= 0.2
 0 <= b_k10_ADM_s105 <= 0.2
 0 <= b_k10_ADM_s106 <= 0.2
 0 <= b_k10_ADM_s107 <= 0.2
 0 <= b_k10_ADM_s110 <= 0.2
 0 <= b_k10_ADM_s111 <= 0.2
 0 <= b_k10_ADM_s112 <= 0.2
 0 <= b_k10_ADM_s113 <= 0.2
 0 <= b_k10_ADM_s114 <= 0.2
 0 <= b_k10_ADM_s115 <= 0.2
 0 <= b_k10_ADM_s116 <= 0.2
 0 <= b_k10_ADM_s117 <= 0.2
 0 <= b_k10_ADM_s120 <= 0.2
 0 <= b_k10_ADM_s121 <= 0.2
 0 <= b_k10_ADM_s122 <= 0.2
 0 <= b_k10_ADM_s123 <= 0.2
 0 <= b_k10_ADM_s124 <= 0.2
 0 <= b_k10_ADM_s125 <= 0.2
 0 <= b_k10_ADM_s126 <= 0.2
 0 <= b_k10_ADM_s127 <= 0.2
 0 <= b_k10_ADM_s130 <= 0.2
 0 <= b_k10_ADM_s131 <= 0.2
 0 <= b_k10_ADM_s132 <= 0.2
 0 <= b_k10_ADM_s133 <= 0.2
 0 <= b_k10_ADM_s134 <= 0.2
 0 <= b_k10_ADM_s135 <= 0.2
 0 <= b_k10_ADM_s136 <= 0.2
 0 <= b_k10_ADM_s137 <= 0.2
 0 <= b_k10_AXP_s0 <= 0.2
 0 <= b_k10_AXP_s1 <= 0.2
 0 <= b_k10_AXP_s00 <= 0.2
 0 <= b_k10_AXP_s01 <= 0.2
 0 <= b_k10_AXP_s02 <= 0.2
 0 <= b_k10_AXP_s03 <= 0.2
 0 <= b_k10_AXP_s10 <= 0.2
 0 <= b_k10_AXP_s11 <= 0.2
 0 <= b_k10_AXP_s12 <= 0.2
 0 <= b_k10_AXP_s13 <= 0.2
 0 <= b_k10_AXP_s000 <= 0.2
 0 <= b_k10_AXP_s001 <= 0.2
 0 <= b_k10_AXP_s002 <= 0.2
 0 <= b_k10_AXP_s003 <= 0.2
 0 <= b_k10_AXP_s004 <= 0.2
 0 <= b_k10_AXP_s005 <= 0.2
 0 <= b_k10_AXP_s006 <= 0.2
 0 <= b_k10_AXP_s007 <= 0.2
 0 <= b_k10_AXP_s010 <= 0.2
 0 <= b_k10_AXP_s011 <= 0.2
 0 <= b_k10_AXP_s012 <= 0.2
 0 <= b_k10_AXP_s013 <= 0.2
 0 <= b_k10_AXP_s014 <= 0.2
 0 <= b_k10_AXP_s015 <= 0.2
 0 <= b_k10_AXP_s016 <= 0.2
 0 <= b_k10_AXP_s017 <= 0.2
 0 <= b_k10_AXP_s020 <= 0.2
 0 <= b_k10_AXP_s021 <= 0.2
 0 <= b_k10_AXP_s022 <= 0.2
 0 <= b_k10_AXP_s023 <= 0.2
 0 <= b_k10_AXP_s024 <= 0.2
 0 <= b_k10_AXP_s025 <= 0.2
 0 <= b_k10_AXP_s026 <= 0.2
 0 <= b_k10_AXP_s027 <= 0.2
 0 <= b_k10_AXP_s030 <= 0.2
 0 <= b_k10_AXP_s031 <= 0.2
 0 <= b_k10_AXP_s032 <= 0.2
 0 <= b_k10_AXP_s033 <= 0.2
 0 <= b_k10_AXP_s034 <= 0.2
 0 <= b_k10_AXP_s035 <= 0.2
 0 <= b_k10_AXP_s036 <= 0.2
 0 <= b_k10_AXP_s037 <= 0.2
 0 <= b_k10_AXP_s100 <= 0.2
 0 <= b_k10_AXP_s101 <= 0.2
 0 <= b_k10_AXP_s102 <= 0.2
 0 <= b_k10_AXP_s103 <= 0.2
 0 <= b_k10_AXP_s104 <= 0.2
 0 <= b_k10_AXP_s105 <= 0.2
 0 <= b_k10_AXP_s106 <= 0.2
 0 <= b_k10_AXP_s107 <= 0.2
 0 <= b_k10_AXP_s110 <= 0.2
 0 <= b_k10_AXP_s111 <= 0.2
 0 <= b_k10_AXP_s112 <= 0.2
 0 <= b_k10_AXP_s113 <= 0.2
 0 <= b_k10_AXP_s114 <= 0.2
 0 <= b_k10_AXP_s115 <= 0.2
 0 <= b_k10_AXP_s116 <= 0.2
 0 <= b_k10_AXP_s117 <= 0.2
 0 <= b_k10_AXP_s120 <= 0.2
 0 <= b_k10_AXP_s121 <= 0.2
 0 <= b_k10_AXP_s122 <= 0.2
 0 <= b_k10_AXP_s123 <= 0.2
 0 <= b_k10_AXP_s124 <= 0.2
 0 <= b_k10_AXP_s125 <= 0.2
 0 <= b_k10_AXP_s126 <= 0.2
 0 <= b_k10_AXP_s127 <= 0.2
 0 <= b_k10_AXP_s130 <= 0.2
 0 <= b_k10_AXP_s131 <= 0.2
 0 <= b_k10_AXP_s132 <= 0.2
 0 <= b_k10_AXP_s133 <= 0.2
 0 <= b_k10_AXP_s134 <= 0.2
 0 <= b_k10_AXP_s135 <= 0.2
 0 <= b_k10_AXP_s136 <= 0.2
 0 <= b_k10_AXP_s137 <= 0.2
 0 <= b_k10_DIS_s0 <= 0.2
 0 <= b_k10_DIS_s1 <= 0.2
 0 <= b_k10_DIS_s00 <= 0.2
 0 <= b_k10_DIS_s01 <= 0.2
 0 <= b_k10_DIS_s02 <= 0.2
 0 <= b_k10_DIS_s03 <= 0.2
 0 <= b_k10_DIS_s10 <= 0.2
 0 <= b_k10_DIS_s11 <= 0.2
 0 <= b_k10_DIS_s12 <= 0.2
 0 <= b_k10_DIS_s13 <= 0.2
 0 <= b_k10_DIS_s000 <= 0.2
 0 <= b_k10_DIS_s001 <= 0.2
 0 <= b_k10_DIS_s002 <= 0.2
 0 <= b_k10_DIS_s003 <= 0.2
 0 <= b_k10_DIS_s004 <= 0.2
 0 <= b_k10_DIS_s005 <= 0.2
 0 <= b_k10_DIS_s006 <= 0.2
 0 <= b_k10_DIS_s007 <= 0.2
 0 <= b_k10_DIS_s010 <= 0.2
 0 <= b_k10_DIS_s011 <= 0.2
 0 <= b_k10_DIS_s012 <= 0.2
 0 <= b_k10_DIS_s013 <= 0.2
 0 <= b_k10_DIS_s014 <= 0.2
 0 <= b_k10_DIS_s015 <= 0.2
 0 <= b_k10_DIS_s016 <= 0.2
 0 <= b_k10_DIS_s017 <= 0.2
 0 <= b_k10_DIS_s020 <= 0.2
 0 <= b_k10_DIS_s021 <= 0.2
 0 <= b_k10_DIS_s022 <= 0.2
 0 <= b_k10_DIS_s023 <= 0.2
 0 <= b_k10_DIS_s024 <= 0.2
 0 <= b_k10_DIS_s025 <= 0.2
 0 <= b_k10_DIS_s026 <= 0.2
 0 <= b_k10_DIS_s027 <= 0.2
 0 <= b_k10_DIS_s030 <= 0.2
 0 <= b_k10_DIS_s031 <= 0.2
 0 <= b_k10_DIS_s032 <= 0.2
 0 <= b_k10_DIS_s033 <= 0.2
 0 <= b_k10_DIS_s034 <= 0.2
 0 <= b_k10_DIS_s035 <= 0.2
 0 <= b_k10_DIS_s036 <= 0.2
 0 <= b_k10_DIS_s037 <= 0.2
 0 <= b_k10_DIS_s100 <= 0.2
 0 <= b_k10_DIS_s101 <= 0.2
 0 <= b_k10_DIS_s102 <= 0.2
 0 <= b_k10_DIS_s103 <= 0.2
 0 <= b_k10_DIS_s104 <= 0.2
 0 <= b_k10_DIS_s105 <= 0.2
 0 <= b_k10_DIS_s106 <= 0.2
 0 <= b_k10_DIS_s107 <= 0.2
 0 <= b_k10_DIS_s110 <= 0.2
 0 <= b_k10_DIS_s111 <= 0.2
 0 <= b_k10_DIS_s112 <= 0.2
 0 <= b_k10_DIS_s113 <= 0.2
 0 <= b_k10_DIS_s114 <= 0.2
 0 <= b_k10_DIS_s115 <= 0.2
 0 <= b_k10_DIS_s116 <= 0.2
 0 <= b_k10_DIS_s117 <= 0.2
 0 <= b_k10_DIS_s120 <= 0.2
 0 <= b_k10_DIS_s121 <= 0.2
 0 <= b_k10_DIS_s122 <= 0.2
 0 <= b_k10_DIS_s123 <= 0.2
 0 <= b_k10_DIS_s124 <= 0.2
 0 <= b_k10_DIS_s125 <= 0.2
 0 <= b_k10_DIS_s126 <= 0.2
 0 <= b_k10_DIS_s127 <= 0.2
 0 <= b_k10_DIS_s130 <= 0.2
 0 <= b_k10_DIS_s131 <= 0.2
 0 <= b_k10_DIS_s132 <= 0.2
 0 <= b_k10_DIS_s133 <= 0.2
 0 <= b_k10_DIS_s134 <= 0.2
 0 <= b_k10_DIS_s135 <= 0.2
 0 <= b_k10_DIS_s136 <= 0.2
 0 <= b_k10_DIS_s137 <= 0.2
 0 <= b_k10_FDX_s0 <= 0.2
 0 <= b_k10_FDX_s1 <= 0.2
 0 <= b_k10_FDX_s00 <= 0.2
 0 <= b_k10_FDX_s01 <= 0.2
 0 <= b_k10_FDX_s02 <= 0.2
 0 <= b_k10_FDX_s03 <= 0.2
 0 <= b_k10_FDX_s10 <= 0.2
 0 <= b_k10_FDX_s11 <= 0.2
 0 <= b_k10_FDX_s12 <= 0.2
 0 <= b_k10_FDX_s13 <= 0.2
 0 <= b_k10_FDX_s000 <= 0.2
 0 <= b_k10_FDX_s001 <= 0.2
 0 <= b_k10_FDX_s002 <= 0.2
 0 <= b_k10_FDX_s003 <= 0.2
 0 <= b_k10_FDX_s004 <= 0.2
 0 <= b_k10_FDX_s005 <= 0.2
 0 <= b_k10_FDX_s006 <= 0.2
 0 <= b_k10_FDX_s007 <= 0.2
 0 <= b_k10_FDX_s010 <= 0.2
 0 <= b_k10_FDX_s011 <= 0.2
 0 <= b_k10_FDX_s012 <= 0.2
 0 <= b_k10_FDX_s013 <= 0.2
 0 <= b_k10_FDX_s014 <= 0.2
 0 <= b_k10_FDX_s015 <= 0.2
 0 <= b_k10_FDX_s016 <= 0.2
 0 <= b_k10_FDX_s017 <= 0.2
 0 <= b_k10_FDX_s020 <= 0.2
 0 <= b_k10_FDX_s021 <= 0.2
 0 <= b_k10_FDX_s022 <= 0.2
 0 <= b_k10_FDX_s023 <= 0.2
 0 <= b_k10_FDX_s024 <= 0.2
 0 <= b_k10_FDX_s025 <= 0.2
 0 <= b_k10_FDX_s026 <= 0.2
 0 <= b_k10_FDX_s027 <= 0.2
 0 <= b_k10_FDX_s030 <= 0.2
 0 <= b_k10_FDX_s031 <= 0.2
 0 <= b_k10_FDX_s032 <= 0.2
 0 <= b_k10_FDX_s033 <= 0.2
 0 <= b_k10_FDX_s034 <= 0.2
 0 <= b_k10_FDX_s035 <= 0.2
 0 <= b_k10_FDX_s036 <= 0.2
 0 <= b_k10_FDX_s037 <= 0.2
 0 <= b_k10_FDX_s100 <= 0.2
 0 <= b_k10_FDX_s101 <= 0.2
 0 <= b_k10_FDX_s102 <= 0.2
 0 <= b_k10_FDX_s103 <= 0.2
 0 <= b_k10_FDX_s104 <= 0.2
 0 <= b_k10_FDX_s105 <= 0.2
 0 <= b_k10_FDX_s106 <= 0.2
 0 <= b_k10_FDX_s107 <= 0.2
 0 <= b_k10_FDX_s110 <= 0.2
 0 <= b_k10_FDX_s111 <= 0.2
 0 <= b_k10_FDX_s112 <= 0.2
 0 <= b_k10_FDX_s113 <= 0.2
 0 <= b_k10_FDX_s114 <= 0.2
 0 <= b_k10_FDX_s115 <= 0.2
 0 <= b_k10_FDX_s116 <= 0.2
 0 <= b_k10_FDX_s117 <= 0.2
 0 <= b_k10_FDX_s120 <= 0.2
 0 <= b_k10_FDX_s121 <= 0.2
 0 <= b_k10_FDX_s122 <= 0.2
 0 <= b_k10_FDX_s123 <= 0.2
 0 <= b_k10_FDX_s124 <= 0.2
 0 <= b_k10_FDX_s125 <= 0.2
 0 <= b_k10_FDX_s126 <= 0.2
 0 <= b_k10_FDX_s127 <= 0.2
 0 <= b_k10_FDX_s130 <= 0.2
 0 <= b_k10_FDX_s131 <= 0.2
 0 <= b_k10_FDX_s132 <= 0.2
 0 <= b_k10_FDX_s133 <= 0.2
 0 <= b_k10_FDX_s134 <= 0.2
 0 <= b_k10_FDX_s135 <= 0.2
 0 <= b_k10_FDX_s136 <= 0.2
 0 <= b_k10_FDX_s137 <= 0.2
 0 <= b_k10_IBM_s0 <= 0.2
 0 <= b_k10_IBM_s1 <= 0.2
 0 <= b_k10_IBM_s00 <= 0.2
 0 <= b_k10_IBM_s01 <= 0.2
 0 <= b_k10_IBM_s02 <= 0.2
 0 <= b_k10_IBM_s03 <= 0.2
 0 <= b_k10_IBM_s10 <= 0.2
 0 <= b_k10_IBM_s11 <= 0.2
 0 <= b_k10_IBM_s12 <= 0.2
 0 <= b_k10_IBM_s13 <= 0.2
 0 <= b_k10_IBM_s000 <= 0.2
 0 <= b_k10_IBM_s001 <= 0.2
 0 <= b_k10_IBM_s002 <= 0.2
 0 <= b_k10_IBM_s003 <= 0.2
 0 <= b_k10_IBM_s004 <= 0.2
 0 <= b_k10_IBM_s005 <= 0.2
 0 <= b_k10_IBM_s006 <= 0.2
 0 <= b_k10_IBM_s007 <= 0.2
 0 <= b_k10_IBM_s010 <= 0.2
 0 <= b_k10_IBM_s011 <= 0.2
 0 <= b_k10_IBM_s012 <= 0.2
 0 <= b_k10_IBM_s013 <= 0.2
 0 <= b_k10_IBM_s014 <= 0.2
 0 <= b_k10_IBM_s015 <= 0.2
 0 <= b_k10_IBM_s016 <= 0.2
 0 <= b_k10_IBM_s017 <= 0.2
 0 <= b_k10_IBM_s020 <= 0.2
 0 <= b_k10_IBM_s021 <= 0.2
 0 <= b_k10_IBM_s022 <= 0.2
 0 <= b_k10_IBM_s023 <= 0.2
 0 <= b_k10_IBM_s024 <= 0.2
 0 <= b_k10_IBM_s025 <= 0.2
 0 <= b_k10_IBM_s026 <= 0.2
 0 <= b_k10_IBM_s027 <= 0.2
 0 <= b_k10_IBM_s030 <= 0.2
 0 <= b_k10_IBM_s031 <= 0.2
 0 <= b_k10_IBM_s032 <= 0.2
 0 <= b_k10_IBM_s033 <= 0.2
 0 <= b_k10_IBM_s034 <= 0.2
 0 <= b_k10_IBM_s035 <= 0.2
 0 <= b_k10_IBM_s036 <= 0.2
 0 <= b_k10_IBM_s037 <= 0.2
 0 <= b_k10_IBM_s100 <= 0.2
 0 <= b_k10_IBM_s101 <= 0.2
 0 <= b_k10_IBM_s102 <= 0.2
 0 <= b_k10_IBM_s103 <= 0.2
 0 <= b_k10_IBM_s104 <= 0.2
 0 <= b_k10_IBM_s105 <= 0.2
 0 <= b_k10_IBM_s106 <= 0.2
 0 <= b_k10_IBM_s107 <= 0.2
 0 <= b_k10_IBM_s110 <= 0.2
 0 <= b_k10_IBM_s111 <= 0.2
 0 <= b_k10_IBM_s112 <= 0.2
 0 <= b_k10_IBM_s113 <= 0.2
 0 <= b_k10_IBM_s114 <= 0.2
 0 <= b_k10_IBM_s115 <= 0.2
 0 <= b_k10_IBM_s116 <= 0.2
 0 <= b_k10_IBM_s117 <= 0.2
 0 <= b_k10_IBM_s120 <= 0.2
 0 <= b_k10_IBM_s121 <= 0.2
 0 <= b_k10_IBM_s122 <= 0.2
 0 <= b_k10_IBM_s123 <= 0.2
 0 <= b_k10_IBM_s124 <= 0.2
 0 <= b_k10_IBM_s125 <= 0.2
 0 <= b_k10_IBM_s126 <= 0.2
 0 <= b_k10_IBM_s127 <= 0.2
 0 <= b_k10_IBM_s130 <= 0.2
 0 <= b_k10_IBM_s131 <= 0.2
 0 <= b_k10_IBM_s132 <= 0.2
 0 <= b_k10_IBM_s133 <= 0.2
 0 <= b_k10_IBM_s134 <= 0.2
 0 <= b_k10_IBM_s135 <= 0.2
 0 <= b_k10_IBM_s136 <= 0.2
 0 <= b_k10_IBM_s137 <= 0.2
 0 <= b_k10_INTC_s0 <= 0.2
 0 <= b_k10_INTC_s1 <= 0.2
 0 <= b_k10_INTC_s00 <= 0.2
 0 <= b_k10_INTC_s01 <= 0.2
 0 <= b_k10_INTC_s02 <= 0.2
 0 <= b_k10_INTC_s03 <= 0.2
 0 <= b_k10_INTC_s10 <= 0.2
 0 <= b_k10_INTC_s11 <= 0.2
 0 <= b_k10_INTC_s12 <= 0.2
 0 <= b_k10_INTC_s13 <= 0.2
 0 <= b_k10_INTC_s000 <= 0.2
 0 <= b_k10_INTC_s001 <= 0.2
 0 <= b_k10_INTC_s002 <= 0.2
 0 <= b_k10_INTC_s003 <= 0.2
 0 <= b_k10_INTC_s004 <= 0.2
 0 <= b_k10_INTC_s005 <= 0.2
 0 <= b_k10_INTC_s006 <= 0.2
 0 <= b_k10_INTC_s007 <= 0.2
 0 <= b_k10_INTC_s010 <= 0.2
 0 <= b_k10_INTC_s011 <= 0.2
 0 <= b_k10_INTC_s012 <= 0.2
 0 <= b_k10_INTC_s013 <= 0.2
 0 <= b_k10_INTC_s014 <= 0.2
 0 <= b_k10_INTC_s015 <= 0.2
 0 <= b_k10_INTC_s016 <= 0.2
 0 <= b_k10_INTC_s017 <= 0.2
 0 <= b_k10_INTC_s020 <= 0.2
 0 <= b_k10_INTC_s021 <= 0.2
 0 <= b_k10_INTC_s022 <= 0.2
 0 <= b_k10_INTC_s023 <= 0.2
 0 <= b_k10_INTC_s024 <= 0.2
 0 <= b_k10_INTC_s025 <= 0.2
 0 <= b_k10_INTC_s026 <= 0.2
 0 <= b_k10_INTC_s027 <= 0.2
 0 <= b_k10_INTC_s030 <= 0.2
 0 <= b_k10_INTC_s031 <= 0.2
 0 <= b_k10_INTC_s032 <= 0.2
 0 <= b_k10_INTC_s033 <= 0.2
 0 <= b_k10_INTC_s034 <= 0.2
 0 <= b_k10_INTC_s035 <= 0.2
 0 <= b_k10_INTC_s036 <= 0.2
 0 <= b_k10_INTC_s037 <= 0.2
 0 <= b_k10_INTC_s100 <= 0.2
 0 <= b_k10_INTC_s101 <= 0.2
 0 <= b_k10_INTC_s102 <= 0.2
 0 <= b_k10_INTC_s103 <= 0.2
 0 <= b_k10_INTC_s104 <= 0.2
 0 <= b_k10_INTC_s105 <= 0.2
 0 <= b_k10_INTC_s106 <= 0.2
 0 <= b_k10_INTC_s107 <= 0.2
 0 <= b_k10_INTC_s110 <= 0.2
 0 <= b_k10_INTC_s111 <= 0.2
 0 <= b_k10_INTC_s112 <= 0.2
 0 <= b_k10_INTC_s113 <= 0.2
 0 <= b_k10_INTC_s114 <= 0.2
 0 <= b_k10_INTC_s115 <= 0.2
 0 <= b_k10_INTC_s116 <= 0.2
 0 <= b_k10_INTC_s117 <= 0.2
 0 <= b_k10_INTC_s120 <= 0.2
 0 <= b_k10_INTC_s121 <= 0.2
 0 <= b_k10_INTC_s122 <= 0.2
 0 <= b_k10_INTC_s123 <= 0.2
 0 <= b_k10_INTC_s124 <= 0.2
 0 <= b_k10_INTC_s125 <= 0.2
 0 <= b_k10_INTC_s126 <= 0.2
 0 <= b_k10_INTC_s127 <= 0.2
 0 <= b_k10_INTC_s130 <= 0.2
 0 <= b_k10_INTC_s131 <= 0.2
 0 <= b_k10_INTC_s132 <= 0.2
 0 <= b_k10_INTC_s133 <= 0.2
 0 <= b_k10_INTC_s134 <= 0.2
 0 <= b_k10_INTC_s135 <= 0.2
 0 <= b_k10_INTC_s136 <= 0.2
 0 <= b_k10_INTC_s137 <= 0.2
 0 <= b_k10_KO_s0 <= 0.2
 0 <= b_k10_KO_s1 <= 0.2
 0 <= b_k10_KO_s00 <= 0.2
 0 <= b_k10_KO_s01 <= 0.2
 0 <= b_k10_KO_s02 <= 0.2
 0 <= b_k10_KO_s03 <= 0.2
 0 <= b_k10_KO_s10 <= 0.2
 0 <= b_k10_KO_s11 <= 0.2
 0 <= b_k10_KO_s12 <= 0.2
 0 <= b_k10_KO_s13 <= 0.2
 0 <= b_k10_KO_s000 <= 0.2
 0 <= b_k10_KO_s001 <= 0.2
 0 <= b_k10_KO_s002 <= 0.2
 0 <= b_k10_KO_s003 <= 0.2
 0 <= b_k10_KO_s004 <= 0.2
 0 <= b_k10_KO_s005 <= 0.2
 0 <= b_k10_KO_s006 <= 0.2
 0 <= b_k10_KO_s007 <= 0.2
 0 <= b_k10_KO_s010 <= 0.2
 0 <= b_k10_KO_s011 <= 0.2
 0 <= b_k10_KO_s012 <= 0.2
 0 <= b_k10_KO_s013 <= 0.2
 0 <= b_k10_KO_s014 <= 0.2
 0 <= b_k10_KO_s015 <= 0.2
 0 <= b_k10_KO_s016 <= 0.2
 0 <= b_k10_KO_s017 <= 0.2
 0 <= b_k10_KO_s020 <= 0.2
 0 <= b_k10_KO_s021 <= 0.2
 0 <= b_k10_KO_s022 <= 0.2
 0 <= b_k10_KO_s023 <= 0.2
 0 <= b_k10_KO_s024 <= 0.2
 0 <= b_k10_KO_s025 <= 0.2
 0 <= b_k10_KO_s026 <= 0.2
 0 <= b_k10_KO_s027 <= 0.2
 0 <= b_k10_KO_s030 <= 0.2
 0 <= b_k10_KO_s031 <= 0.2
 0 <= b_k10_KO_s032 <= 0.2
 0 <= b_k10_KO_s033 <= 0.2
 0 <= b_k10_KO_s034 <= 0.2
 0 <= b_k10_KO_s035 <= 0.2
 0 <= b_k10_KO_s036 <= 0.2
 0 <= b_k10_KO_s037 <= 0.2
 0 <= b_k10_KO_s100 <= 0.2
 0 <= b_k10_KO_s101 <= 0.2
 0 <= b_k10_KO_s102 <= 0.2
 0 <= b_k10_KO_s103 <= 0.2
 0 <= b_k10_KO_s104 <= 0.2
 0 <= b_k10_KO_s105 <= 0.2
 0 <= b_k10_KO_s106 <= 0.2
 0 <= b_k10_KO_s107 <= 0.2
 0 <= b_k10_KO_s110 <= 0.2
 0 <= b_k10_KO_s111 <= 0.2
 0 <= b_k10_KO_s112 <= 0.2
 0 <= b_k10_KO_s113 <= 0.2
 0 <= b_k10_KO_s114 <= 0.2
 0 <= b_k10_KO_s115 <= 0.2
 0 <= b_k10_KO_s116 <= 0.2
 0 <= b_k10_KO_s117 <= 0.2
 0 <= b_k10_KO_s120 <= 0.2
 0 <= b_k10_KO_s121 <= 0.2
 0 <= b_k10_KO_s122 <= 0.2
 0 <= b_k10_KO_s123 <= 0.2
 0 <= b_k10_KO_s124 <= 0.2
 0 <= b_k10_KO_s125 <= 0.2
 0 <= b_k10_KO_s126 <= 0.2
 0 <= b_k10_KO_s127 <= 0.2
 0 <= b_k10_KO_s130 <= 0.2
 0 <= b_k10_KO_s131 <= 0.2
 0 <= b_k10_KO_s132 <= 0.2
 0 <= b_k10_KO_s133 <= 0.2
 0 <= b_k10_KO_s134 <= 0.2
 0 <= b_k10_KO_s135 <= 0.2
 0 <= b_k10_KO_s136 <= 0.2
 0 <= b_k10_KO_s137 <= 0.2
 0 <= b_k10_MAT_s0 <= 0.2
 0 <= b_k10_MAT_s1 <= 0.2
 0 <= b_k10_MAT_s00 <= 0.2
 0 <= b_k10_MAT_s01 <= 0.2
 0 <= b_k10_MAT_s02 <= 0.2
 0 <= b_k10_MAT_s03 <= 0.2
 0 <= b_k10_MAT_s10 <= 0.2
 0 <= b_k10_MAT_s11 <= 0.2
 0 <= b_k10_MAT_s12 <= 0.2
 0 <= b_k10_MAT_s13 <= 0.2
 0 <= b_k10_MAT_s000 <= 0.2
 0 <= b_k10_MAT_s001 <= 0.2
 0 <= b_k10_MAT_s002 <= 0.2
 0 <= b_k10_MAT_s003 <= 0.2
 0 <= b_k10_MAT_s004 <= 0.2
 0 <= b_k10_MAT_s005 <= 0.2
 0 <= b_k10_MAT_s006 <= 0.2
 0 <= b_k10_MAT_s007 <= 0.2
 0 <= b_k10_MAT_s010 <= 0.2
 0 <= b_k10_MAT_s011 <= 0.2
 0 <= b_k10_MAT_s012 <= 0.2
 0 <= b_k10_MAT_s013 <= 0.2
 0 <= b_k10_MAT_s014 <= 0.2
 0 <= b_k10_MAT_s015 <= 0.2
 0 <= b_k10_MAT_s016 <= 0.2
 0 <= b_k10_MAT_s017 <= 0.2
 0 <= b_k10_MAT_s020 <= 0.2
 0 <= b_k10_MAT_s021 <= 0.2
 0 <= b_k10_MAT_s022 <= 0.2
 0 <= b_k10_MAT_s023 <= 0.2
 0 <= b_k10_MAT_s024 <= 0.2
 0 <= b_k10_MAT_s025 <= 0.2
 0 <= b_k10_MAT_s026 <= 0.2
 0 <= b_k10_MAT_s027 <= 0.2
 0 <= b_k10_MAT_s030 <= 0.2
 0 <= b_k10_MAT_s031 <= 0.2
 0 <= b_k10_MAT_s032 <= 0.2
 0 <= b_k10_MAT_s033 <= 0.2
 0 <= b_k10_MAT_s034 <= 0.2
 0 <= b_k10_MAT_s035 <= 0.2
 0 <= b_k10_MAT_s036 <= 0.2
 0 <= b_k10_MAT_s037 <= 0.2
 0 <= b_k10_MAT_s100 <= 0.2
 0 <= b_k10_MAT_s101 <= 0.2
 0 <= b_k10_MAT_s102 <= 0.2
 0 <= b_k10_MAT_s103 <= 0.2
 0 <= b_k10_MAT_s104 <= 0.2
 0 <= b_k10_MAT_s105 <= 0.2
 0 <= b_k10_MAT_s106 <= 0.2
 0 <= b_k10_MAT_s107 <= 0.2
 0 <= b_k10_MAT_s110 <= 0.2
 0 <= b_k10_MAT_s111 <= 0.2
 0 <= b_k10_MAT_s112 <= 0.2
 0 <= b_k10_MAT_s113 <= 0.2
 0 <= b_k10_MAT_s114 <= 0.2
 0 <= b_k10_MAT_s115 <= 0.2
 0 <= b_k10_MAT_s116 <= 0.2
 0 <= b_k10_MAT_s117 <= 0.2
 0 <= b_k10_MAT_s120 <= 0.2
 0 <= b_k10_MAT_s121 <= 0.2
 0 <= b_k10_MAT_s122 <= 0.2
 0 <= b_k10_MAT_s123 <= 0.2
 0 <= b_k10_MAT_s124 <= 0.2
 0 <= b_k10_MAT_s125 <= 0.2
 0 <= b_k10_MAT_s126 <= 0.2
 0 <= b_k10_MAT_s127 <= 0.2
 0 <= b_k10_MAT_s130 <= 0.2
 0 <= b_k10_MAT_s131 <= 0.2
 0 <= b_k10_MAT_s132 <= 0.2
 0 <= b_k10_MAT_s133 <= 0.2
 0 <= b_k10_MAT_s134 <= 0.2
 0 <= b_k10_MAT_s135 <= 0.2
 0 <= b_k10_MAT_s136 <= 0.2
 0 <= b_k10_MAT_s137 <= 0.2
 0 <= b_k10_MSFT_s0 <= 0.2
 0 <= b_k10_MSFT_s1 <= 0.2
 0 <= b_k10_MSFT_s00 <= 0.2
 0 <= b_k10_MSFT_s01 <= 0.2
 0 <= b_k10_MSFT_s02 <= 0.2
 0 <= b_k10_MSFT_s03 <= 0.2
 0 <= b_k10_MSFT_s10 <= 0.2
 0 <= b_k10_MSFT_s11 <= 0.2
 0 <= b_k10_MSFT_s12 <= 0.2
 0 <= b_k10_MSFT_s13 <= 0.2
 0 <= b_k10_MSFT_s000 <= 0.2
 0 <= b_k10_MSFT_s001 <= 0.2
 0 <= b_k10_MSFT_s002 <= 0.2
 0 <= b_k10_MSFT_s003 <= 0.2
 0 <= b_k10_MSFT_s004 <= 0.2
 0 <= b_k10_MSFT_s005 <= 0.2
 0 <= b_k10_MSFT_s006 <= 0.2
 0 <= b_k10_MSFT_s007 <= 0.2
 0 <= b_k10_MSFT_s010 <= 0.2
 0 <= b_k10_MSFT_s011 <= 0.2
 0 <= b_k10_MSFT_s012 <= 0.2
 0 <= b_k10_MSFT_s013 <= 0.2
 0 <= b_k10_MSFT_s014 <= 0.2
 0 <= b_k10_MSFT_s015 <= 0.2
 0 <= b_k10_MSFT_s016 <= 0.2
 0 <= b_k10_MSFT_s017 <= 0.2
 0 <= b_k10_MSFT_s020 <= 0.2
 0 <= b_k10_MSFT_s021 <= 0.2
 0 <= b_k10_MSFT_s022 <= 0.2
 0 <= b_k10_MSFT_s023 <= 0.2
 0 <= b_k10_MSFT_s024 <= 0.2
 0 <= b_k10_MSFT_s025 <= 0.2
 0 <= b_k10_MSFT_s026 <= 0.2
 0 <= b_k10_MSFT_s027 <= 0.2
 0 <= b_k10_MSFT_s030 <= 0.2
 0 <= b_k10_MSFT_s031 <= 0.2
 0 <= b_k10_MSFT_s032 <= 0.2
 0 <= b_k10_MSFT_s033 <= 0.2
 0 <= b_k10_MSFT_s034 <= 0.2
 0 <= b_k10_MSFT_s035 <= 0.2
 0 <= b_k10_MSFT_s036 <= 0.2
 0 <= b_k10_MSFT_s037 <= 0.2
 0 <= b_k10_MSFT_s100 <= 0.2
 0 <= b_k10_MSFT_s101 <= 0.2
 0 <= b_k10_MSFT_s102 <= 0.2
 0 <= b_k10_MSFT_s103 <= 0.2
 0 <= b_k10_MSFT_s104 <= 0.2
 0 <= b_k10_MSFT_s105 <= 0.2
 0 <= b_k10_MSFT_s106 <= 0.2
 0 <= b_k10_MSFT_s107 <= 0.2
 0 <= b_k10_MSFT_s110 <= 0.2
 0 <= b_k10_MSFT_s111 <= 0.2
 0 <= b_k10_MSFT_s112 <= 0.2
 0 <= b_k10_MSFT_s113 <= 0.2
 0 <= b_k10_MSFT_s114 <= 0.2
 0 <= b_k10_MSFT_s115 <= 0.2
 0 <= b_k10_MSFT_s116 <= 0.2
 0 <= b_k10_MSFT_s117 <= 0.2
 0 <= b_k10_MSFT_s120 <= 0.2
 0 <= b_k10_MSFT_s121 <= 0.2
 0 <= b_k10_MSFT_s122 <= 0.2
 0 <= b_k10_MSFT_s123 <= 0.2
 0 <= b_k10_MSFT_s124 <= 0.2
 0 <= b_k10_MSFT_s125 <= 0.2
 0 <= b_k10_MSFT_s126 <= 0.2
 0 <= b_k10_MSFT_s127 <= 0.2
 0 <= b_k10_MSFT_s130 <= 0.2
 0 <= b_k10_MSFT_s131 <= 0.2
 0 <= b_k10_MSFT_s132 <= 0.2
 0 <= b_k10_MSFT_s133 <= 0.2
 0 <= b_k10_MSFT_s134 <= 0.2
 0 <= b_k10_MSFT_s135 <= 0.2
 0 <= b_k10_MSFT_s136 <= 0.2
 0 <= b_k10_MSFT_s137 <= 0.2
 0 <= b_k10_PG_s0 <= 0.2
 0 <= b_k10_PG_s1 <= 0.2
 0 <= b_k10_PG_s00 <= 0.2
 0 <= b_k10_PG_s01 <= 0.2
 0 <= b_k10_PG_s02 <= 0.2
 0 <= b_k10_PG_s03 <= 0.2
 0 <= b_k10_PG_s10 <= 0.2
 0 <= b_k10_PG_s11 <= 0.2
 0 <= b_k10_PG_s12 <= 0.2
 0 <= b_k10_PG_s13 <= 0.2
 0 <= b_k10_PG_s000 <= 0.2
 0 <= b_k10_PG_s001 <= 0.2
 0 <= b_k10_PG_s002 <= 0.2
 0 <= b_k10_PG_s003 <= 0.2
 0 <= b_k10_PG_s004 <= 0.2
 0 <= b_k10_PG_s005 <= 0.2
 0 <= b_k10_PG_s006 <= 0.2
 0 <= b_k10_PG_s007 <= 0.2
 0 <= b_k10_PG_s010 <= 0.2
 0 <= b_k10_PG_s011 <= 0.2
 0 <= b_k10_PG_s012 <= 0.2
 0 <= b_k10_PG_s013 <= 0.2
 0 <= b_k10_PG_s014 <= 0.2
 0 <= b_k10_PG_s015 <= 0.2
 0 <= b_k10_PG_s016 <= 0.2
 0 <= b_k10_PG_s017 <= 0.2
 0 <= b_k10_PG_s020 <= 0.2
 0 <= b_k10_PG_s021 <= 0.2
 0 <= b_k10_PG_s022 <= 0.2
 0 <= b_k10_PG_s023 <= 0.2
 0 <= b_k10_PG_s024 <= 0.2
 0 <= b_k10_PG_s025 <= 0.2
 0 <= b_k10_PG_s026 <= 0.2
 0 <= b_k10_PG_s027 <= 0.2
 0 <= b_k10_PG_s030 <= 0.2
 0 <= b_k10_PG_s031 <= 0.2
 0 <= b_k10_PG_s032 <= 0.2
 0 <= b_k10_PG_s033 <= 0.2
 0 <= b_k10_PG_s034 <= 0.2
 0 <= b_k10_PG_s035 <= 0.2
 0 <= b_k10_PG_s036 <= 0.2
 0 <= b_k10_PG_s037 <= 0.2
 0 <= b_k10_PG_s100 <= 0.2
 0 <= b_k10_PG_s101 <= 0.2
 0 <= b_k10_PG_s102 <= 0.2
 0 <= b_k10_PG_s103 <= 0.2
 0 <= b_k10_PG_s104 <= 0.2
 0 <= b_k10_PG_s105 <= 0.2
 0 <= b_k10_PG_s106 <= 0.2
 0 <= b_k10_PG_s107 <= 0.2
 0 <= b_k10_PG_s110 <= 0.2
 0 <= b_k10_PG_s111 <= 0.2
 0 <= b_k10_PG_s112 <= 0.2
 0 <= b_k10_PG_s113 <= 0.2
 0 <= b_k10_PG_s114 <= 0.2
 0 <= b_k10_PG_s115 <= 0.2
 0 <= b_k10_PG_s116 <= 0.2
 0 <= b_k10_PG_s117 <= 0.2
 0 <= b_k10_PG_s120 <= 0.2
 0 <= b_k10_PG_s121 <= 0.2
 0 <= b_k10_PG_s122 <= 0.2
 0 <= b_k10_PG_s123 <= 0.2
 0 <= b_k10_PG_s124 <= 0.2
 0 <= b_k10_PG_s125 <= 0.2
 0 <= b_k10_PG_s126 <= 0.2
 0 <= b_k10_PG_s127 <= 0.2
 0 <= b_k10_PG_s130 <= 0.2
 0 <= b_k10_PG_s131 <= 0.2
 0 <= b_k10_PG_s132 <= 0.2
 0 <= b_k10_PG_s133 <= 0.2
 0 <= b_k10_PG_s134 <= 0.2
 0 <= b_k10_PG_s135 <= 0.2
 0 <= b_k10_PG_s136 <= 0.2
 0 <= b_k10_PG_s137 <= 0.2
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
 0 <= s_k10_ADM_s0 <= 0.2
 0 <= s_k10_ADM_s1 <= 0.2
 0 <= s_k10_ADM_s00 <= 0.2
 0 <= s_k10_ADM_s01 <= 0.2
 0 <= s_k10_ADM_s02 <= 0.2
 0 <= s_k10_ADM_s03 <= 0.2
 0 <= s_k10_ADM_s10 <= 0.2
 0 <= s_k10_ADM_s11 <= 0.2
 0 <= s_k10_ADM_s12 <= 0.2
 0 <= s_k10_ADM_s13 <= 0.2
 0 <= s_k10_ADM_s000 <= 0.2
 0 <= s_k10_ADM_s001 <= 0.2
 0 <= s_k10_ADM_s002 <= 0.2
 0 <= s_k10_ADM_s003 <= 0.2
 0 <= s_k10_ADM_s004 <= 0.2
 0 <= s_k10_ADM_s005 <= 0.2
 0 <= s_k10_ADM_s006 <= 0.2
 0 <= s_k10_ADM_s007 <= 0.2
 0 <= s_k10_ADM_s010 <= 0.2
 0 <= s_k10_ADM_s011 <= 0.2
 0 <= s_k10_ADM_s012 <= 0.2
 0 <= s_k10_ADM_s013 <= 0.2
 0 <= s_k10_ADM_s014 <= 0.2
 0 <= s_k10_ADM_s015 <= 0.2
 0 <= s_k10_ADM_s016 <= 0.2
 0 <= s_k10_ADM_s017 <= 0.2
 0 <= s_k10_ADM_s020 <= 0.2
 0 <= s_k10_ADM_s021 <= 0.2
 0 <= s_k10_ADM_s022 <= 0.2
 0 <= s_k10_ADM_s023 <= 0.2
 0 <= s_k10_ADM_s024 <= 0.2
 0 <= s_k10_ADM_s025 <= 0.2
 0 <= s_k10_ADM_s026 <= 0.2
 0 <= s_k10_ADM_s027 <= 0.2
 0 <= s_k10_ADM_s030 <= 0.2
 0 <= s_k10_ADM_s031 <= 0.2
 0 <= s_k10_ADM_s032 <= 0.2
 0 <= s_k10_ADM_s033 <= 0.2
 0 <= s_k10_ADM_s034 <= 0.2
 0 <= s_k10_ADM_s035 <= 0.2
 0 <= s_k10_ADM_s036 <= 0.2
 0 <= s_k10_ADM_s037 <= 0.2
 0 <= s_k10_ADM_s100 <= 0.2
 0 <= s_k10_ADM_s101 <= 0.2
 0 <= s_k10_ADM_s102 <= 0.2
 0 <= s_k10_ADM_s103 <= 0.2
 0 <= s_k10_ADM_s104 <= 0.2
 0 <= s_k10_ADM_s105 <= 0.2
 0 <= s_k10_ADM_s106 <= 0.2
 0 <= s_k10_ADM_s107 <= 0.2
 0 <= s_k10_ADM_s110 <= 0.2
 0 <= s_k10_ADM_s111 <= 0.2
 0 <= s_k10_ADM_s112 <= 0.2
 0 <= s_k10_ADM_s113 <= 0.2
 0 <= s_k10_ADM_s114 <= 0.2
 0 <= s_k10_ADM_s115 <= 0.2
 0 <= s_k10_ADM_s116 <= 0.2
 0 <= s_k10_ADM_s117 <= 0.2
 0 <= s_k10_ADM_s120 <= 0.2
 0 <= s_k10_ADM_s121 <= 0.2
 0 <= s_k10_ADM_s122 <= 0.2
 0 <= s_k10_ADM_s123 <= 0.2
 0 <= s_k10_ADM_s124 <= 0.2
 0 <= s_k10_ADM_s125 <= 0.2
 0 <= s_k10_ADM_s126 <= 0.2
 0 <= s_k10_ADM_s127 <= 0.2
 0 <= s_k10_ADM_s130 <= 0.2
 0 <= s_k10_ADM_s131 <= 0.2
 0 <= s_k10_ADM_s132 <= 0.2
 0 <= s_k10_ADM_s133 <= 0.2
 0 <= s_k10_ADM_s134 <= 0.2
 0 <= s_k10_ADM_s135 <= 0.2
 0 <= s_k10_ADM_s136 <= 0.2
 0 <= s_k10_ADM_s137 <= 0.2
 0 <= s_k10_AXP_s0 <= 0.2
 0 <= s_k10_AXP_s1 <= 0.2
 0 <= s_k10_AXP_s00 <= 0.2
 0 <= s_k10_AXP_s01 <= 0.2
 0 <= s_k10_AXP_s02 <= 0.2
 0 <= s_k10_AXP_s03 <= 0.2
 0 <= s_k10_AXP_s10 <= 0.2
 0 <= s_k10_AXP_s11 <= 0.2
 0 <= s_k10_AXP_s12 <= 0.2
 0 <= s_k10_AXP_s13 <= 0.2
 0 <= s_k10_AXP_s000 <= 0.2
 0 <= s_k10_AXP_s001 <= 0.2
 0 <= s_k10_AXP_s002 <= 0.2
 0 <= s_k10_AXP_s003 <= 0.2
 0 <= s_k10_AXP_s004 <= 0.2
 0 <= s_k10_AXP_s005 <= 0.2
 0 <= s_k10_AXP_s006 <= 0.2
 0 <= s_k10_AXP_s007 <= 0.2
 0 <= s_k10_AXP_s010 <= 0.2
 0 <= s_k10_AXP_s011 <= 0.2
 0 <= s_k10_AXP_s012 <= 0.2
 0 <= s_k10_AXP_s013 <= 0.2
 0 <= s_k10_AXP_s014 <= 0.2
 0 <= s_k10_AXP_s015 <= 0.2
 0 <= s_k10_AXP_s016 <= 0.2
 0 <= s_k10_AXP_s017 <= 0.2
 0 <= s_k10_AXP_s020 <= 0.2
 0 <= s_k10_AXP_s021 <= 0.2
 0 <= s_k10_AXP_s022 <= 0.2
 0 <= s_k10_AXP_s023 <= 0.2
 0 <= s_k10_AXP_s024 <= 0.2
 0 <= s_k10_AXP_s025 <= 0.2
 0 <= s_k10_AXP_s026 <= 0.2
 0 <= s_k10_AXP_s027 <= 0.2
 0 <= s_k10_AXP_s030 <= 0.2
 0 <= s_k10_AXP_s031 <= 0.2
 0 <= s_k10_AXP_s032 <= 0.2
 0 <= s_k10_AXP_s033 <= 0.2
 0 <= s_k10_AXP_s034 <= 0.2
 0 <= s_k10_AXP_s035 <= 0.2
 0 <= s_k10_AXP_s036 <= 0.2
 0 <= s_k10_AXP_s037 <= 0.2
 0 <= s_k10_AXP_s100 <= 0.2
 0 <= s_k10_AXP_s101 <= 0.2
 0 <= s_k10_AXP_s102 <= 0.2
 0 <= s_k10_AXP_s103 <= 0.2
 0 <= s_k10_AXP_s104 <= 0.2
 0 <= s_k10_AXP_s105 <= 0.2
 0 <= s_k10_AXP_s106 <= 0.2
 0 <= s_k10_AXP_s107 <= 0.2
 0 <= s_k10_AXP_s110 <= 0.2
 0 <= s_k10_AXP_s111 <= 0.2
 0 <= s_k10_AXP_s112 <= 0.2
 0 <= s_k10_AXP_s113 <= 0.2
 0 <= s_k10_AXP_s114 <= 0.2
 0 <= s_k10_AXP_s115 <= 0.2
 0 <= s_k10_AXP_s116 <= 0.2
 0 <= s_k10_AXP_s117 <= 0.2
 0 <= s_k10_AXP_s120 <= 0.2
 0 <= s_k10_AXP_s121 <= 0.2
 0 <= s_k10_AXP_s122 <= 0.2
 0 <= s_k10_AXP_s123 <= 0.2
 0 <= s_k10_AXP_s124 <= 0.2
 0 <= s_k10_AXP_s125 <= 0.2
 0 <= s_k10_AXP_s126 <= 0.2
 0 <= s_k10_AXP_s127 <= 0.2
 0 <= s_k10_AXP_s130 <= 0.2
 0 <= s_k10_AXP_s131 <= 0.2
 0 <= s_k10_AXP_s132 <= 0.2
 0 <= s_k10_AXP_s133 <= 0.2
 0 <= s_k10_AXP_s134 <= 0.2
 0 <= s_k10_AXP_s135 <= 0.2
 0 <= s_k10_AXP_s136 <= 0.2
 0 <= s_k10_AXP_s137 <= 0.2
 0 <= s_k10_DIS_s0 <= 0.2
 0 <= s_k10_DIS_s1 <= 0.2
 0 <= s_k10_DIS_s00 <= 0.2
 0 <= s_k10_DIS_s01 <= 0.2
 0 <= s_k10_DIS_s02 <= 0.2
 0 <= s_k10_DIS_s03 <= 0.2
 0 <= s_k10_DIS_s10 <= 0.2
 0 <= s_k10_DIS_s11 <= 0.2
 0 <= s_k10_DIS_s12 <= 0.2
 0 <= s_k10_DIS_s13 <= 0.2
 0 <= s_k10_DIS_s000 <= 0.2
 0 <= s_k10_DIS_s001 <= 0.2
 0 <= s_k10_DIS_s002 <= 0.2
 0 <= s_k10_DIS_s003 <= 0.2
 0 <= s_k10_DIS_s004 <= 0.2
 0 <= s_k10_DIS_s005 <= 0.2
 0 <= s_k10_DIS_s006 <= 0.2
 0 <= s_k10_DIS_s007 <= 0.2
 0 <= s_k10_DIS_s010 <= 0.2
 0 <= s_k10_DIS_s011 <= 0.2
 0 <= s_k10_DIS_s012 <= 0.2
 0 <= s_k10_DIS_s013 <= 0.2
 0 <= s_k10_DIS_s014 <= 0.2
 0 <= s_k10_DIS_s015 <= 0.2
 0 <= s_k10_DIS_s016 <= 0.2
 0 <= s_k10_DIS_s017 <= 0.2
 0 <= s_k10_DIS_s020 <= 0.2
 0 <= s_k10_DIS_s021 <= 0.2
 0 <= s_k10_DIS_s022 <= 0.2
 0 <= s_k10_DIS_s023 <= 0.2
 0 <= s_k10_DIS_s024 <= 0.2
 0 <= s_k10_DIS_s025 <= 0.2
 0 <= s_k10_DIS_s026 <= 0.2
 0 <= s_k10_DIS_s027 <= 0.2
 0 <= s_k10_DIS_s030 <= 0.2
 0 <= s_k10_DIS_s031 <= 0.2
 0 <= s_k10_DIS_s032 <= 0.2
 0 <= s_k10_DIS_s033 <= 0.2
 0 <= s_k10_DIS_s034 <= 0.2
 0 <= s_k10_DIS_s035 <= 0.2
 0 <= s_k10_DIS_s036 <= 0.2
 0 <= s_k10_DIS_s037 <= 0.2
 0 <= s_k10_DIS_s100 <= 0.2
 0 <= s_k10_DIS_s101 <= 0.2
 0 <= s_k10_DIS_s102 <= 0.2
 0 <= s_k10_DIS_s103 <= 0.2
 0 <= s_k10_DIS_s104 <= 0.2
 0 <= s_k10_DIS_s105 <= 0.2
 0 <= s_k10_DIS_s106 <= 0.2
 0 <= s_k10_DIS_s107 <= 0.2
 0 <= s_k10_DIS_s110 <= 0.2
 0 <= s_k10_DIS_s111 <= 0.2
 0 <= s_k10_DIS_s112 <= 0.2
 0 <= s_k10_DIS_s113 <= 0.2
 0 <= s_k10_DIS_s114 <= 0.2
 0 <= s_k10_DIS_s115 <= 0.2
 0 <= s_k10_DIS_s116 <= 0.2
 0 <= s_k10_DIS_s117 <= 0.2
 0 <= s_k10_DIS_s120 <= 0.2
 0 <= s_k10_DIS_s121 <= 0.2
 0 <= s_k10_DIS_s122 <= 0.2
 0 <= s_k10_DIS_s123 <= 0.2
 0 <= s_k10_DIS_s124 <= 0.2
 0 <= s_k10_DIS_s125 <= 0.2
 0 <= s_k10_DIS_s126 <= 0.2
 0 <= s_k10_DIS_s127 <= 0.2
 0 <= s_k10_DIS_s130 <= 0.2
 0 <= s_k10_DIS_s131 <= 0.2
 0 <= s_k10_DIS_s132 <= 0.2
 0 <= s_k10_DIS_s133 <= 0.2
 0 <= s_k10_DIS_s134 <= 0.2
 0 <= s_k10_DIS_s135 <= 0.2
 0 <= s_k10_DIS_s136 <= 0.2
 0 <= s_k10_DIS_s137 <= 0.2
 0 <= s_k10_FDX_s0 <= 0.2
 0 <= s_k10_FDX_s1 <= 0.2
 0 <= s_k10_FDX_s00 <= 0.2
 0 <= s_k10_FDX_s01 <= 0.2
 0 <= s_k10_FDX_s02 <= 0.2
 0 <= s_k10_FDX_s03 <= 0.2
 0 <= s_k10_FDX_s10 <= 0.2
 0 <= s_k10_FDX_s11 <= 0.2
 0 <= s_k10_FDX_s12 <= 0.2
 0 <= s_k10_FDX_s13 <= 0.2
 0 <= s_k10_FDX_s000 <= 0.2
 0 <= s_k10_FDX_s001 <= 0.2
 0 <= s_k10_FDX_s002 <= 0.2
 0 <= s_k10_FDX_s003 <= 0.2
 0 <= s_k10_FDX_s004 <= 0.2
 0 <= s_k10_FDX_s005 <= 0.2
 0 <= s_k10_FDX_s006 <= 0.2
 0 <= s_k10_FDX_s007 <= 0.2
 0 <= s_k10_FDX_s010 <= 0.2
 0 <= s_k10_FDX_s011 <= 0.2
 0 <= s_k10_FDX_s012 <= 0.2
 0 <= s_k10_FDX_s013 <= 0.2
 0 <= s_k10_FDX_s014 <= 0.2
 0 <= s_k10_FDX_s015 <= 0.2
 0 <= s_k10_FDX_s016 <= 0.2
 0 <= s_k10_FDX_s017 <= 0.2
 0 <= s_k10_FDX_s020 <= 0.2
 0 <= s_k10_FDX_s021 <= 0.2
 0 <= s_k10_FDX_s022 <= 0.2
 0 <= s_k10_FDX_s023 <= 0.2
 0 <= s_k10_FDX_s024 <= 0.2
 0 <= s_k10_FDX_s025 <= 0.2
 0 <= s_k10_FDX_s026 <= 0.2
 0 <= s_k10_FDX_s027 <= 0.2
 0 <= s_k10_FDX_s030 <= 0.2
 0 <= s_k10_FDX_s031 <= 0.2
 0 <= s_k10_FDX_s032 <= 0.2
 0 <= s_k10_FDX_s033 <= 0.2
 0 <= s_k10_FDX_s034 <= 0.2
 0 <= s_k10_FDX_s035 <= 0.2
 0 <= s_k10_FDX_s036 <= 0.2
 0 <= s_k10_FDX_s037 <= 0.2
 0 <= s_k10_FDX_s100 <= 0.2
 0 <= s_k10_FDX_s101 <= 0.2
 0 <= s_k10_FDX_s102 <= 0.2
 0 <= s_k10_FDX_s103 <= 0.2
 0 <= s_k10_FDX_s104 <= 0.2
 0 <= s_k10_FDX_s105 <= 0.2
 0 <= s_k10_FDX_s106 <= 0.2
 0 <= s_k10_FDX_s107 <= 0.2
 0 <= s_k10_FDX_s110 <= 0.2
 0 <= s_k10_FDX_s111 <= 0.2
 0 <= s_k10_FDX_s112 <= 0.2
 0 <= s_k10_FDX_s113 <= 0.2
 0 <= s_k10_FDX_s114 <= 0.2
 0 <= s_k10_FDX_s115 <= 0.2
 0 <= s_k10_FDX_s116 <= 0.2
 0 <= s_k10_FDX_s117 <= 0.2
 0 <= s_k10_FDX_s120 <= 0.2
 0 <= s_k10_FDX_s121 <= 0.2
 0 <= s_k10_FDX_s122 <= 0.2
 0 <= s_k10_FDX_s123 <= 0.2
 0 <= s_k10_FDX_s124 <= 0.2
 0 <= s_k10_FDX_s125 <= 0.2
 0 <= s_k10_FDX_s126 <= 0.2
 0 <= s_k10_FDX_s127 <= 0.2
 0 <= s_k10_FDX_s130 <= 0.2
 0 <= s_k10_FDX_s131 <= 0.2
 0 <= s_k10_FDX_s132 <= 0.2
 0 <= s_k10_FDX_s133 <= 0.2
 0 <= s_k10_FDX_s134 <= 0.2
 0 <= s_k10_FDX_s135 <= 0.2
 0 <= s_k10_FDX_s136 <= 0.2
 0 <= s_k10_FDX_s137 <= 0.2
 0 <= s_k10_IBM_s0 <= 0.2
 0 <= s_k10_IBM_s1 <= 0.2
 0 <= s_k10_IBM_s00 <= 0.2
 0 <= s_k10_IBM_s01 <= 0.2
 0 <= s_k10_IBM_s02 <= 0.2
 0 <= s_k10_IBM_s03 <= 0.2
 0 <= s_k10_IBM_s10 <= 0.2
 0 <= s_k10_IBM_s11 <= 0.2
 0 <= s_k10_IBM_s12 <= 0.2
 0 <= s_k10_IBM_s13 <= 0.2
 0 <= s_k10_IBM_s000 <= 0.2
 0 <= s_k10_IBM_s001 <= 0.2
 0 <= s_k10_IBM_s002 <= 0.2
 0 <= s_k10_IBM_s003 <= 0.2
 0 <= s_k10_IBM_s004 <= 0.2
 0 <= s_k10_IBM_s005 <= 0.2
 0 <= s_k10_IBM_s006 <= 0.2
 0 <= s_k10_IBM_s007 <= 0.2
 0 <= s_k10_IBM_s010 <= 0.2
 0 <= s_k10_IBM_s011 <= 0.2
 0 <= s_k10_IBM_s012 <= 0.2
 0 <= s_k10_IBM_s013 <= 0.2
 0 <= s_k10_IBM_s014 <= 0.2
 0 <= s_k10_IBM_s015 <= 0.2
 0 <= s_k10_IBM_s016 <= 0.2
 0 <= s_k10_IBM_s017 <= 0.2
 0 <= s_k10_IBM_s020 <= 0.2
 0 <= s_k10_IBM_s021 <= 0.2
 0 <= s_k10_IBM_s022 <= 0.2
 0 <= s_k10_IBM_s023 <= 0.2
 0 <= s_k10_IBM_s024 <= 0.2
 0 <= s_k10_IBM_s025 <= 0.2
 0 <= s_k10_IBM_s026 <= 0.2
 0 <= s_k10_IBM_s027 <= 0.2
 0 <= s_k10_IBM_s030 <= 0.2
 0 <= s_k10_IBM_s031 <= 0.2
 0 <= s_k10_IBM_s032 <= 0.2
 0 <= s_k10_IBM_s033 <= 0.2
 0 <= s_k10_IBM_s034 <= 0.2
 0 <= s_k10_IBM_s035 <= 0.2
 0 <= s_k10_IBM_s036 <= 0.2
 0 <= s_k10_IBM_s037 <= 0.2
 0 <= s_k10_IBM_s100 <= 0.2
 0 <= s_k10_IBM_s101 <= 0.2
 0 <= s_k10_IBM_s102 <= 0.2
 0 <= s_k10_IBM_s103 <= 0.2
 0 <= s_k10_IBM_s104 <= 0.2
 0 <= s_k10_IBM_s105 <= 0.2
 0 <= s_k10_IBM_s106 <= 0.2
 0 <= s_k10_IBM_s107 <= 0.2
 0 <= s_k10_IBM_s110 <= 0.2
 0 <= s_k10_IBM_s111 <= 0.2
 0 <= s_k10_IBM_s112 <= 0.2
 0 <= s_k10_IBM_s113 <= 0.2
 0 <= s_k10_IBM_s114 <= 0.2
 0 <= s_k10_IBM_s115 <= 0.2
 0 <= s_k10_IBM_s116 <= 0.2
 0 <= s_k10_IBM_s117 <= 0.2
 0 <= s_k10_IBM_s120 <= 0.2
 0 <= s_k10_IBM_s121 <= 0.2
 0 <= s_k10_IBM_s122 <= 0.2
 0 <= s_k10_IBM_s123 <= 0.2
 0 <= s_k10_IBM_s124 <= 0.2
 0 <= s_k10_IBM_s125 <= 0.2
 0 <= s_k10_IBM_s126 <= 0.2
 0 <= s_k10_IBM_s127 <= 0.2
 0 <= s_k10_IBM_s130 <= 0.2
 0 <= s_k10_IBM_s131 <= 0.2
 0 <= s_k10_IBM_s132 <= 0.2
 0 <= s_k10_IBM_s133 <= 0.2
 0 <= s_k10_IBM_s134 <= 0.2
 0 <= s_k10_IBM_s135 <= 0.2
 0 <= s_k10_IBM_s136 <= 0.2
 0 <= s_k10_IBM_s137 <= 0.2
 0 <= s_k10_INTC_s0 <= 0.2
 0 <= s_k10_INTC_s1 <= 0.2
 0 <= s_k10_INTC_s00 <= 0.2
 0 <= s_k10_INTC_s01 <= 0.2
 0 <= s_k10_INTC_s02 <= 0.2
 0 <= s_k10_INTC_s03 <= 0.2
 0 <= s_k10_INTC_s10 <= 0.2
 0 <= s_k10_INTC_s11 <= 0.2
 0 <= s_k10_INTC_s12 <= 0.2
 0 <= s_k10_INTC_s13 <= 0.2
 0 <= s_k10_INTC_s000 <= 0.2
 0 <= s_k10_INTC_s001 <= 0.2
 0 <= s_k10_INTC_s002 <= 0.2
 0 <= s_k10_INTC_s003 <= 0.2
 0 <= s_k10_INTC_s004 <= 0.2
 0 <= s_k10_INTC_s005 <= 0.2
 0 <= s_k10_INTC_s006 <= 0.2
 0 <= s_k10_INTC_s007 <= 0.2
 0 <= s_k10_INTC_s010 <= 0.2
 0 <= s_k10_INTC_s011 <= 0.2
 0 <= s_k10_INTC_s012 <= 0.2
 0 <= s_k10_INTC_s013 <= 0.2
 0 <= s_k10_INTC_s014 <= 0.2
 0 <= s_k10_INTC_s015 <= 0.2
 0 <= s_k10_INTC_s016 <= 0.2
 0 <= s_k10_INTC_s017 <= 0.2
 0 <= s_k10_INTC_s020 <= 0.2
 0 <= s_k10_INTC_s021 <= 0.2
 0 <= s_k10_INTC_s022 <= 0.2
 0 <= s_k10_INTC_s023 <= 0.2
 0 <= s_k10_INTC_s024 <= 0.2
 0 <= s_k10_INTC_s025 <= 0.2
 0 <= s_k10_INTC_s026 <= 0.2
 0 <= s_k10_INTC_s027 <= 0.2
 0 <= s_k10_INTC_s030 <= 0.2
 0 <= s_k10_INTC_s031 <= 0.2
 0 <= s_k10_INTC_s032 <= 0.2
 0 <= s_k10_INTC_s033 <= 0.2
 0 <= s_k10_INTC_s034 <= 0.2
 0 <= s_k10_INTC_s035 <= 0.2
 0 <= s_k10_INTC_s036 <= 0.2
 0 <= s_k10_INTC_s037 <= 0.2
 0 <= s_k10_INTC_s100 <= 0.2
 0 <= s_k10_INTC_s101 <= 0.2
 0 <= s_k10_INTC_s102 <= 0.2
 0 <= s_k10_INTC_s103 <= 0.2
 0 <= s_k10_INTC_s104 <= 0.2
 0 <= s_k10_INTC_s105 <= 0.2
 0 <= s_k10_INTC_s106 <= 0.2
 0 <= s_k10_INTC_s107 <= 0.2
 0 <= s_k10_INTC_s110 <= 0.2
 0 <= s_k10_INTC_s111 <= 0.2
 0 <= s_k10_INTC_s112 <= 0.2
 0 <= s_k10_INTC_s113 <= 0.2
 0 <= s_k10_INTC_s114 <= 0.2
 0 <= s_k10_INTC_s115 <= 0.2
 0 <= s_k10_INTC_s116 <= 0.2
 0 <= s_k10_INTC_s117 <= 0.2
 0 <= s_k10_INTC_s120 <= 0.2
 0 <= s_k10_INTC_s121 <= 0.2
 0 <= s_k10_INTC_s122 <= 0.2
 0 <= s_k10_INTC_s123 <= 0.2
 0 <= s_k10_INTC_s124 <= 0.2
 0 <= s_k10_INTC_s125 <= 0.2
 0 <= s_k10_INTC_s126 <= 0.2
 0 <= s_k10_INTC_s127 <= 0.2
 0 <= s_k10_INTC_s130 <= 0.2
 0 <= s_k10_INTC_s131 <= 0.2
 0 <= s_k10_INTC_s132 <= 0.2
 0 <= s_k10_INTC_s133 <= 0.2
 0 <= s_k10_INTC_s134 <= 0.2
 0 <= s_k10_INTC_s135 <= 0.2
 0 <= s_k10_INTC_s136 <= 0.2
 0 <= s_k10_INTC_s137 <= 0.2
 0 <= s_k10_KO_s0 <= 0.2
 0 <= s_k10_KO_s1 <= 0.2
 0 <= s_k10_KO_s00 <= 0.2
 0 <= s_k10_KO_s01 <= 0.2
 0 <= s_k10_KO_s02 <= 0.2
 0 <= s_k10_KO_s03 <= 0.2
 0 <= s_k10_KO_s10 <= 0.2
 0 <= s_k10_KO_s11 <= 0.2
 0 <= s_k10_KO_s12 <= 0.2
 0 <= s_k10_KO_s13 <= 0.2
 0 <= s_k10_KO_s000 <= 0.2
 0 <= s_k10_KO_s001 <= 0.2
 0 <= s_k10_KO_s002 <= 0.2
 0 <= s_k10_KO_s003 <= 0.2
 0 <= s_k10_KO_s004 <= 0.2
 0 <= s_k10_KO_s005 <= 0.2
 0 <= s_k10_KO_s006 <= 0.2
 0 <= s_k10_KO_s007 <= 0.2
 0 <= s_k10_KO_s010 <= 0.2
 0 <= s_k10_KO_s011 <= 0.2
 0 <= s_k10_KO_s012 <= 0.2
 0 <= s_k10_KO_s013 <= 0.2
 0 <= s_k10_KO_s014 <= 0.2
 0 <= s_k10_KO_s015 <= 0.2
 0 <= s_k10_KO_s016 <= 0.2
 0 <= s_k10_KO_s017 <= 0.2
 0 <= s_k10_KO_s020 <= 0.2
 0 <= s_k10_KO_s021 <= 0.2
 0 <= s_k10_KO_s022 <= 0.2
 0 <= s_k10_KO_s023 <= 0.2
 0 <= s_k10_KO_s024 <= 0.2
 0 <= s_k10_KO_s025 <= 0.2
 0 <= s_k10_KO_s026 <= 0.2
 0 <= s_k10_KO_s027 <= 0.2
 0 <= s_k10_KO_s030 <= 0.2
 0 <= s_k10_KO_s031 <= 0.2
 0 <= s_k10_KO_s032 <= 0.2
 0 <= s_k10_KO_s033 <= 0.2
 0 <= s_k10_KO_s034 <= 0.2
 0 <= s_k10_KO_s035 <= 0.2
 0 <= s_k10_KO_s036 <= 0.2
 0 <= s_k10_KO_s037 <= 0.2
 0 <= s_k10_KO_s100 <= 0.2
 0 <= s_k10_KO_s101 <= 0.2
 0 <= s_k10_KO_s102 <= 0.2
 0 <= s_k10_KO_s103 <= 0.2
 0 <= s_k10_KO_s104 <= 0.2
 0 <= s_k10_KO_s105 <= 0.2
 0 <= s_k10_KO_s106 <= 0.2
 0 <= s_k10_KO_s107 <= 0.2
 0 <= s_k10_KO_s110 <= 0.2
 0 <= s_k10_KO_s111 <= 0.2
 0 <= s_k10_KO_s112 <= 0.2
 0 <= s_k10_KO_s113 <= 0.2
 0 <= s_k10_KO_s114 <= 0.2
 0 <= s_k10_KO_s115 <= 0.2
 0 <= s_k10_KO_s116 <= 0.2
 0 <= s_k10_KO_s117 <= 0.2
 0 <= s_k10_KO_s120 <= 0.2
 0 <= s_k10_KO_s121 <= 0.2
 0 <= s_k10_KO_s122 <= 0.2
 0 <= s_k10_KO_s123 <= 0.2
 0 <= s_k10_KO_s124 <= 0.2
 0 <= s_k10_KO_s125 <= 0.2
 0 <= s_k10_KO_s126 <= 0.2
 0 <= s_k10_KO_s127 <= 0.2
 0 <= s_k10_KO_s130 <= 0.2
 0 <= s_k10_KO_s131 <= 0.2
 0 <= s_k10_KO_s132 <= 0.2
 0 <= s_k10_KO_s133 <= 0.2
 0 <= s_k10_KO_s134 <= 0.2
 0 <= s_k10_KO_s135 <= 0.2
 0 <= s_k10_KO_s136 <= 0.2
 0 <= s_k10_KO_s137 <= 0.2
 0 <= s_k10_MAT_s0 <= 0.2
 0 <= s_k10_MAT_s1 <= 0.2
 0 <= s_k10_MAT_s00 <= 0.2
 0 <= s_k10_MAT_s01 <= 0.2
 0 <= s_k10_MAT_s02 <= 0.2
 0 <= s_k10_MAT_s03 <= 0.2
 0 <= s_k10_MAT_s10 <= 0.2
 0 <= s_k10_MAT_s11 <= 0.2
 0 <= s_k10_MAT_s12 <= 0.2
 0 <= s_k10_MAT_s13 <= 0.2
 0 <= s_k10_MAT_s000 <= 0.2
 0 <= s_k10_MAT_s001 <= 0.2
 0 <= s_k10_MAT_s002 <= 0.2
 0 <= s_k10_MAT_s003 <= 0.2
 0 <= s_k10_MAT_s004 <= 0.2
 0 <= s_k10_MAT_s005 <= 0.2
 0 <= s_k10_MAT_s006 <= 0.2
 0 <= s_k10_MAT_s007 <= 0.2
 0 <= s_k10_MAT_s010 <= 0.2
 0 <= s_k10_MAT_s011 <= 0.2
 0 <= s_k10_MAT_s012 <= 0.2
 0 <= s_k10_MAT_s013 <= 0.2
 0 <= s_k10_MAT_s014 <= 0.2
 0 <= s_k10_MAT_s015 <= 0.2
 0 <= s_k10_MAT_s016 <= 0.2
 0 <= s_k10_MAT_s017 <= 0.2
 0 <= s_k10_MAT_s020 <= 0.2
 0 <= s_k10_MAT_s021 <= 0.2
 0 <= s_k10_MAT_s022 <= 0.2
 0 <= s_k10_MAT_s023 <= 0.2
 0 <= s_k10_MAT_s024 <= 0.2
 0 <= s_k10_MAT_s025 <= 0.2
 0 <= s_k10_MAT_s026 <= 0.2
 0 <= s_k10_MAT_s027 <= 0.2
 0 <= s_k10_MAT_s030 <= 0.2
 0 <= s_k10_MAT_s031 <= 0.2
 0 <= s_k10_MAT_s032 <= 0.2
 0 <= s_k10_MAT_s033 <= 0.2
 0 <= s_k10_MAT_s034 <= 0.2
 0 <= s_k10_MAT_s035 <= 0.2
 0 <= s_k10_MAT_s036 <= 0.2
 0 <= s_k10_MAT_s037 <= 0.2
 0 <= s_k10_MAT_s100 <= 0.2
 0 <= s_k10_MAT_s101 <= 0.2
 0 <= s_k10_MAT_s102 <= 0.2
 0 <= s_k10_MAT_s103 <= 0.2
 0 <= s_k10_MAT_s104 <= 0.2
 0 <= s_k10_MAT_s105 <= 0.2
 0 <= s_k10_MAT_s106 <= 0.2
 0 <= s_k10_MAT_s107 <= 0.2
 0 <= s_k10_MAT_s110 <= 0.2
 0 <= s_k10_MAT_s111 <= 0.2
 0 <= s_k10_MAT_s112 <= 0.2
 0 <= s_k10_MAT_s113 <= 0.2
 0 <= s_k10_MAT_s114 <= 0.2
 0 <= s_k10_MAT_s115 <= 0.2
 0 <= s_k10_MAT_s116 <= 0.2
 0 <= s_k10_MAT_s117 <= 0.2
 0 <= s_k10_MAT_s120 <= 0.2
 0 <= s_k10_MAT_s121 <= 0.2
 0 <= s_k10_MAT_s122 <= 0.2
 0 <= s_k10_MAT_s123 <= 0.2
 0 <= s_k10_MAT_s124 <= 0.2
 0 <= s_k10_MAT_s125 <= 0.2
 0 <= s_k10_MAT_s126 <= 0.2
 0 <= s_k10_MAT_s127 <= 0.2
 0 <= s_k10_MAT_s130 <= 0.2
 0 <= s_k10_MAT_s131 <= 0.2
 0 <= s_k10_MAT_s132 <= 0.2
 0 <= s_k10_MAT_s133 <= 0.2
 0 <= s_k10_MAT_s134 <= 0.2
 0 <= s_k10_MAT_s135 <= 0.2
 0 <= s_k10_MAT_s136 <= 0.2
 0 <= s_k10_MAT_s137 <= 0.2
 0 <= s_k10_MSFT_s0 <= 0.2
 0 <= s_k10_MSFT_s1 <= 0.2
 0 <= s_k10_MSFT_s00 <= 0.2
 0 <= s_k10_MSFT_s01 <= 0.2
 0 <= s_k10_MSFT_s02 <= 0.2
 0 <= s_k10_MSFT_s03 <= 0.2
 0 <= s_k10_MSFT_s10 <= 0.2
 0 <= s_k10_MSFT_s11 <= 0.2
 0 <= s_k10_MSFT_s12 <= 0.2
 0 <= s_k10_MSFT_s13 <= 0.2
 0 <= s_k10_MSFT_s000 <= 0.2
 0 <= s_k10_MSFT_s001 <= 0.2
 0 <= s_k10_MSFT_s002 <= 0.2
 0 <= s_k10_MSFT_s003 <= 0.2
 0 <= s_k10_MSFT_s004 <= 0.2
 0 <= s_k10_MSFT_s005 <= 0.2
 0 <= s_k10_MSFT_s006 <= 0.2
 0 <= s_k10_MSFT_s007 <= 0.2
 0 <= s_k10_MSFT_s010 <= 0.2
 0 <= s_k10_MSFT_s011 <= 0.2
 0 <= s_k10_MSFT_s012 <= 0.2
 0 <= s_k10_MSFT_s013 <= 0.2
 0 <= s_k10_MSFT_s014 <= 0.2
 0 <= s_k10_MSFT_s015 <= 0.2
 0 <= s_k10_MSFT_s016 <= 0.2
 0 <= s_k10_MSFT_s017 <= 0.2
 0 <= s_k10_MSFT_s020 <= 0.2
 0 <= s_k10_MSFT_s021 <= 0.2
 0 <= s_k10_MSFT_s022 <= 0.2
 0 <= s_k10_MSFT_s023 <= 0.2
 0 <= s_k10_MSFT_s024 <= 0.2
 0 <= s_k10_MSFT_s025 <= 0.2
 0 <= s_k10_MSFT_s026 <= 0.2
 0 <= s_k10_MSFT_s027 <= 0.2
 0 <= s_k10_MSFT_s030 <= 0.2
 0 <= s_k10_MSFT_s031 <= 0.2
 0 <= s_k10_MSFT_s032 <= 0.2
 0 <= s_k10_MSFT_s033 <= 0.2
 0 <= s_k10_MSFT_s034 <= 0.2
 0 <= s_k10_MSFT_s035 <= 0.2
 0 <= s_k10_MSFT_s036 <= 0.2
 0 <= s_k10_MSFT_s037 <= 0.2
 0 <= s_k10_MSFT_s100 <= 0.2
 0 <= s_k10_MSFT_s101 <= 0.2
 0 <= s_k10_MSFT_s102 <= 0.2
 0 <= s_k10_MSFT_s103 <= 0.2
 0 <= s_k10_MSFT_s104 <= 0.2
 0 <= s_k10_MSFT_s105 <= 0.2
 0 <= s_k10_MSFT_s106 <= 0.2
 0 <= s_k10_MSFT_s107 <= 0.2
 0 <= s_k10_MSFT_s110 <= 0.2
 0 <= s_k10_MSFT_s111 <= 0.2
 0 <= s_k10_MSFT_s112 <= 0.2
 0 <= s_k10_MSFT_s113 <= 0.2
 0 <= s_k10_MSFT_s114 <= 0.2
 0 <= s_k10_MSFT_s115 <= 0.2
 0 <= s_k10_MSFT_s116 <= 0.2
 0 <= s_k10_MSFT_s117 <= 0.2
 0 <= s_k10_MSFT_s120 <= 0.2
 0 <= s_k10_MSFT_s121 <= 0.2
 0 <= s_k10_MSFT_s122 <= 0.2
 0 <= s_k10_MSFT_s123 <= 0.2
 0 <= s_k10_MSFT_s124 <= 0.2
 0 <= s_k10_MSFT_s125 <= 0.2
 0 <= s_k10_MSFT_s126 <= 0.2
 0 <= s_k10_MSFT_s127 <= 0.2
 0 <= s_k10_MSFT_s130 <= 0.2
 0 <= s_k10_MSFT_s131 <= 0.2
 0 <= s_k10_MSFT_s132 <= 0.2
 0 <= s_k10_MSFT_s133 <= 0.2
 0 <= s_k10_MSFT_s134 <= 0.2
 0 <= s_k10_MSFT_s135 <= 0.2
 0 <= s_k10_MSFT_s136 <= 0.2
 0 <= s_k10_MSFT_s137 <= 0.2
 0 <= s_k10_PG_s0 <= 0.2
 0 <= s_k10_PG_s1 <= 0.2
 0 <= s_k10_PG_s00 <= 0.2
 0 <= s_k10_PG_s01 <= 0.2
 0 <= s_k10_PG_s02 <= 0.2
 0 <= s_k10_PG_s03 <= 0.2
 0 <= s_k10_PG_s10 <= 0.2
 0 <= s_k10_PG_s11 <= 0.2
 0 <= s_k10_PG_s12 <= 0.2
 0 <= s_k10_PG_s13 <= 0.2
 0 <= s_k10_PG_s000 <= 0.2
 0 <= s_k10_PG_s001 <= 0.2
 0 <= s_k10_PG_s002 <= 0.2
 0 <= s_k10_PG_s003 <= 0.2
 0 <= s_k10_PG_s004 <= 0.2
 0 <= s_k10_PG_s005 <= 0.2
 0 <= s_k10_PG_s006 <= 0.2
 0 <= s_k10_PG_s007 <= 0.2
 0 <= s_k10_PG_s010 <= 0.2
 0 <= s_k10_PG_s011 <= 0.2
 0 <= s_k10_PG_s012 <= 0.2
 0 <= s_k10_PG_s013 <= 0.2
 0 <= s_k10_PG_s014 <= 0.2
 0 <= s_k10_PG_s015 <= 0.2
 0 <= s_k10_PG_s016 <= 0.2
 0 <= s_k10_PG_s017 <= 0.2
 0 <= s_k10_PG_s020 <= 0.2
 0 <= s_k10_PG_s021 <= 0.2
 0 <= s_k10_PG_s022 <= 0.2
 0 <= s_k10_PG_s023 <= 0.2
 0 <= s_k10_PG_s024 <= 0.2
 0 <= s_k10_PG_s025 <= 0.2
 0 <= s_k10_PG_s026 <= 0.2
 0 <= s_k10_PG_s027 <= 0.2
 0 <= s_k10_PG_s030 <= 0.2
 0 <= s_k10_PG_s031 <= 0.2
 0 <= s_k10_PG_s032 <= 0.2
 0 <= s_k10_PG_s033 <= 0.2
 0 <= s_k10_PG_s034 <= 0.2
 0 <= s_k10_PG_s035 <= 0.2
 0 <= s_k10_PG_s036 <= 0.2
 0 <= s_k10_PG_s037 <= 0.2
 0 <= s_k10_PG_s100 <= 0.2
 0 <= s_k10_PG_s101 <= 0.2
 0 <= s_k10_PG_s102 <= 0.2
 0 <= s_k10_PG_s103 <= 0.2
 0 <= s_k10_PG_s104 <= 0.2
 0 <= s_k10_PG_s105 <= 0.2
 0 <= s_k10_PG_s106 <= 0.2
 0 <= s_k10_PG_s107 <= 0.2
 0 <= s_k10_PG_s110 <= 0.2
 0 <= s_k10_PG_s111 <= 0.2
 0 <= s_k10_PG_s112 <= 0.2
 0 <= s_k10_PG_s113 <= 0.2
 0 <= s_k10_PG_s114 <= 0.2
 0 <= s_k10_PG_s115 <= 0.2
 0 <= s_k10_PG_s116 <= 0.2
 0 <= s_k10_PG_s117 <= 0.2
 0 <= s_k10_PG_s120 <= 0.2
 0 <= s_k10_PG_s121 <= 0.2
 0 <= s_k10_PG_s122 <= 0.2
 0 <= s_k10_PG_s123 <= 0.2
 0 <= s_k10_PG_s124 <= 0.2
 0 <= s_k10_PG_s125 <= 0.2
 0 <= s_k10_PG_s126 <= 0.2
 0 <= s_k10_PG_s127 <= 0.2
 0 <= s_k10_PG_s130 <= 0.2
 0 <= s_k10_PG_s131 <= 0.2
 0 <= s_k10_PG_s132 <= 0.2
 0 <= s_k10_PG_s133 <= 0.2
 0 <= s_k10_PG_s134 <= 0.2
 0 <= s_k10_PG_s135 <= 0.2
 0 <= s_k10_PG_s136 <= 0.2
 0 <= s_k10_PG_s137 <= 0.2
End

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
 w_k1_ADM_s0_return_constraint:  1.24648 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.924899 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.00521 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.18434 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.10081 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.12301 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.68476 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.13352 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.18896 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.12273 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.815301 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.07708 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.888201 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.11742 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.25919 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.864032 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.61138 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.26633 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.970948 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.920499 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.837564 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.988914 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.749575 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.20844 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.49102 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.40341 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.02583 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.972121 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.773365 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.01523 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.09756 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.794245 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.857273 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.954419 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.812405 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.36289 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.67904 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.695403 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.33548 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.92984 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.983861 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.47235 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.10146 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.979493 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.60159 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.28529 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.60162 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.523 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.872856 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.991464 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.04034 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.903173 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.04639 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.822352 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.54283 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.30949 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.779833 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.1086 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.772011 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.10413 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.6986 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.22716 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.40702 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.41012 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.814026 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.76802 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.696768 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.94495 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.20427 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.690497 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.25169 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.881275 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.914277 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.59636 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.725042 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.65574 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.41357 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.908633 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.911377 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.53572 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.41831 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.972349 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.60215 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.54866 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.48192 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.60435 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.816402 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.18149 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.985116 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.32951 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.26186 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.991317 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.7728 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.28968 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.801069 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.078 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.712393 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.17001 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.616575 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.1127 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.25614 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.76425 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.792569 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.861595 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.544235 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.941019 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.26674 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.841362 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.1563 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.89218 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.891839 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.80514 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.650504 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.489922 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.25296 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.93277 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.96227 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.24762 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.28945 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.973637 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.715682 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.37716 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.33232 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.27186 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.955213 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.20145 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.01307 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.09722 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.16104 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.03365 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.44712 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.43318 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.883782 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.04133 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.736548 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.01805 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.724935 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.01234 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.31691 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.24182 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.854725 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.914832 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.801399 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.938474 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.20713 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.822148 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.09842 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.01778 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.831513 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.36941 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.703161 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.745977 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.16173 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.97774 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.95669 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.4602 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.869104 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.07827 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.830631 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.910089 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.40545 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.85258 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.729047 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.64911 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.20616 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.04239 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.04947 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.653864 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.41605 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.803972 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.07473 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.12064 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.08882 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.7637 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.732854 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.986783 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.08706 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.78243 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.01012 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.612415 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.460314 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.965528 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.870201 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.717535 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.37518 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.62036 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.3274 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.70948 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.912216 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.504652 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.13428 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.986597 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.04626 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.26889 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.22067 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.940464 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.835947 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.21946 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.07263 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.12852 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.857071 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.937296 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.03553 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.964815 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.09675 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.04546 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.25749 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.27887 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.900865 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.998602 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.744331 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.897104 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.846934 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.933651 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.07126 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.28834 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.716818 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.981573 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.844835 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.888564 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.09689 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.960429 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.0277 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.00006 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.943809 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.29166 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.755615 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.686455 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.22724 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.951572 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.968864 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.29698 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.15178 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.02635 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.812036 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.04786 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.23774 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.43648 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.92361 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.848007 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.00596 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.963016 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.23351 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.889299 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.53981 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.975972 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.995878 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.17087 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.863049 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.47121 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.725158 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.01173 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.05226 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.42401 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.82162 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.707271 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.504091 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.922361 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.04127 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.798168 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.20237 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.761578 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.07358 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.68598 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.82223 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.51416 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.30334 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.02735 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.988866 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.39188 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.871157 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.944098 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.692766 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.786058 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.59803 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.91146 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.690555 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.605775 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.06916 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.831181 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.13762 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.637916 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.3517 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.87442 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.976869 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.42651 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.09847 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.96597 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.616903 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.26574 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.08252 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.54685 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.889952 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.486719 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.323104 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.984743 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.917751 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.648711 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.7824 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.588344 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.38132 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.87871 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.05748 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.430646 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.25038 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.948551 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.918072 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.2097 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.24471 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.00241 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.815027 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.25131 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.06597 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.20798 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.898072 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.13833 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.998288 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.14935 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.05058 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.06838 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.42047 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.27702 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.918062 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.05507 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.732989 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.919103 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.797731 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.0235 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.16752 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.39125 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.854497 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.924653 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.776296 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.04752 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.16236 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.894614 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.01416 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.0011 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.837184 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.30683 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.737808 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.693251 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.14875 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.968171 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.929279 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.117 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.10355 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.93637 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.886573 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.19914 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.24654 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.35057 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.02874 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.01254 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.04915 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.05267 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.21481 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.04304 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.51166 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.1468 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.983073 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.26494 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.93408 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.18416 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.770655 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.00233 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.13503 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.19313 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.963746 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.872668 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.746674 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.973114 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.10223 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.958715 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.09862 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.956965 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.01777 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.4744 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.882664 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.763249 w_k1_PG_s11 - w_k1_PG_s117
                                 + 0.99 b_k1_PG_s117 - 1.01 s_k1_PG_s117  = 0
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.387998989870058 z_k1_s0000
                                 - 0.698999949742482 z_k1_s0001
                                 - 0.651999447560607 z_k1_s0002
                                 - 0.44499866799052 z_k1_s0003
                                 - 0.257999605890313 z_k1_s0004
                                 - 0.385998986402705 z_k1_s0005
                                 - 1.01899629752802 z_k1_s0006
                                 - 0.35999936202621 z_k1_s0007
                                 - 0.218000280914236 z_k1_s0010
                                 - 0.175999861729244 z_k1_s0011
                                 - 0.291000298926868 z_k1_s0012
                                 - 0.115999878721574 z_k1_s0013
                                 - 0.301999461285672 z_k1_s0014
                                 - 0.264999565717526 z_k1_s0015
                                 - 0.156999861583487 z_k1_s0016
                                 - 0.132999835220448 z_k1_s0017
                                 - 0.141999812345324 z_k1_s0020
                                 - 0.136999955414738 z_k1_s0021
                                 - 0.264000490568055 z_k1_s0022
                                 - 0.0789999121413986 z_k1_s0023
                                 - 0.112999937508529 z_k1_s0024
                                 - 0.346999671694448 z_k1_s0025
                                 - 0.0899999203484113 z_k1_s0026
                                 - 0.295999927243502 z_k1_s0027
                                 - 0.0819999886796516 z_k1_s0030
                                 - 0.0789999931547843 z_k1_s0031
                                 - 0.109999946911746 z_k1_s0032
                                 - 0.123000149519395 z_k1_s0033
                                 - 0.2609997216634 z_k1_s0034
                                 - 0.117999601978223 z_k1_s0035
                                 - 0.227999437890024 z_k1_s0036
                                 - 0.108999948403457 z_k1_s0037
                                 - 0.749000079038001 z_k1_s0040
                                 - 0.201999827802023 z_k1_s0041
                                 - 0.240999925338022 z_k1_s0042
                                 - 0.449998379262264 z_k1_s0043
                                 - 0.465999114969154 z_k1_s0044
                                 - 0.3720010401895 z_k1_s0045
                                 - 0.8489993222075 z_k1_s0046
                                 - 0.240999925338022 z_k1_s0047
                                 - 0.228999754505693 z_k1_s0050
                                 - 0.116999471813346 z_k1_s0051
                                 - 0.167000147014751 z_k1_s0052
                                 - 0.11299946171719 z_k1_s0053
                                 - 0.109999728644806 z_k1_s0054
                                 - 0.076999919851257 z_k1_s0055
                                 - 0.0879998927157376 z_k1_s0056
                                 - 0.196999673736451 z_k1_s0057
                                 - 0.386999561025745 z_k1_s0060
                                 - 1.44099573132387 z_k1_s0061
                                 - 0.625999938164886 z_k1_s0062
                                 - 0.347999732988111 z_k1_s0063
                                 - 0.748000299691774 z_k1_s0064
                                 - 0.499999512506033 z_k1_s0065
                                 - 0.417999260799448 z_k1_s0066
                                 - 0.846997369213417 z_k1_s0067
                                 - 0.0640000412279015 z_k1_s0070
                                 - 0.184999963299377 z_k1_s0071
                                 - 0.122999985709857 z_k1_s0072
                                 - 0.23399958958942 z_k1_s0073
                                 - 0.170999891930763 z_k1_s0074
                                 - 0.217000482713412 z_k1_s0075
                                 - 0.105000036464521 z_k1_s0076
                                 - 0.148000220039543 z_k1_s0077
                                 - 0.0270000626088885 z_k1_s0100
                                 - 0.0190000332062331 z_k1_s0101
                                 - 0.0620000960203151 z_k1_s0102
                                 - 0.0360001030118904 z_k1_s0103
                                 - 0.013000025804271 z_k1_s0104
                                 - 0.0630001070206616 z_k1_s0105
                                 - 0.0360001030118904 z_k1_s0106
                                 - 0.0370001140122369 z_k1_s0107
                                 - 0.0660001191832701 z_k1_s0110
                                 - 0.259999922236236 z_k1_s0111
                                 - 0.117000223006725 z_k1_s0112
                                 - 0.10800013639071 z_k1_s0113
                                 - 0.149000401974568 z_k1_s0114
                                 - 0.244000026252629 z_k1_s0115
                                 - 0.192000041805381 z_k1_s0116
                                 - 0.154000450094576 z_k1_s0117
                                 - 0.0190000498581429 z_k1_s0120
                                 - 0.0560002081504276 z_k1_s0121
                                 - 0.0540001885021757 z_k1_s0122
                                 - 0.0580002277986794 z_k1_s0123
                                 - 0.0360000116679092 z_k1_s0124
                                 - 0.0340000604197821 z_k1_s0125
                                 - 0.0420000706126647 z_k1_s0126
                                 - 0.0430000804367906 z_k1_s0127
                                 - 0.110000316800072 z_k1_s0130
                                 - 0.0400000779272357 z_k1_s0131
                                 - 0.0800001558544714 z_k1_s0132
                                 - 0.028000054549065 z_k1_s0133
                                 - 0.0380000535308391 z_k1_s0134
                                 - 0.0509999866070341 z_k1_s0135
                                 - 0.028000054549065 z_k1_s0136
                                 - 0.0350000784363487 z_k1_s0137
                                 - 0.0240000028896877 z_k1_s0140
                                 - 0.0320000381529433 z_k1_s0141
                                 - 0.083000232943675 z_k1_s0142
                                 - 0.0689999911578389 z_k1_s0143
                                 - 0.0240000028896877 z_k1_s0144
                                 - 0.0370001245050277 z_k1_s0145
                                 - 0.0330000211233338 z_k1_s0146
                                 - 0.0410001935866951 z_k1_s0147
                                 - 0.0320000750940217 z_k1_s0150
                                 - 0.0290000894665194 z_k1_s0151
                                 - 0.114000481648075 z_k1_s0152
                                 - 0.0550001362053705 z_k1_s0153
                                 - 0.0540001409962031 z_k1_s0154
                                 - 0.0670001929157117 z_k1_s0155
                                 - 0.120000567103412 z_k1_s0156
                                 - 0.100000091918402 z_k1_s0157
                                 - 0.0700000444776038 z_k1_s0160
                                 - 0.0890000776701029 z_k1_s0161
                                 - 0.316000707664771 z_k1_s0162
                                 - 0.0720000739084014 z_k1_s0163
                                 - 0.239999835694542 z_k1_s0164
                                 - 0.239999835694542 z_k1_s0165
                                 - 0.101000007854811 z_k1_s0166
                                 - 0.104000052001007 z_k1_s0167
                                 - 0.0960000883538929 z_k1_s0170
                                 - 0.0350000595957267 z_k1_s0171
                                 - 0.0420000545548494 z_k1_s0172
                                 - 0.0690001562544845 z_k1_s0173
                                 - 0.0360000649327524 z_k1_s0174
                                 - 0.0719999602652779 z_k1_s0175
                                 - 0.0530000708620746 z_k1_s0176
                                 - 0.0210000272774247 z_k1_s0177
                                 - 0.166999951167231 z_k1_s1000
                                 - 0.204999799486412 z_k1_s1001
                                 - 0.351000534610367 z_k1_s1002
                                 - 0.150000257893384 z_k1_s1003
                                 - 0.137000243889309 z_k1_s1004
                                 - 0.255000094117998 z_k1_s1005
                                 - 0.221000394569618 z_k1_s1006
                                 - 0.0790001212171501 z_k1_s1007
                                 - 0.137000531076225 z_k1_s1010
                                 - 0.286999990170135 z_k1_s1011
                                 - 0.260001510438488 z_k1_s1012
                                 - 0.310000565133632 z_k1_s1013
                                 - 0.583003298598283 z_k1_s1014
                                 - 0.141000501352006 z_k1_s1015
                                 - 0.247000287412324 z_k1_s1016
                                 - 0.330001564016776 z_k1_s1017
                                 - 0.167000111185961 z_k1_s1020
                                 - 0.358001331878076 z_k1_s1021
                                 - 0.640998827174289 z_k1_s1022
                                 - 0.260000199645817 z_k1_s1023
                                 - 0.233000059576786 z_k1_s1024
                                 - 0.181000019647622 z_k1_s1025
                                 - 0.661001229263275 z_k1_s1026
                                 - 0.454000450900809 z_k1_s1027
                                 - 0.0690000857292795 z_k1_s1030
                                 - 0.133000166979628 z_k1_s1031
                                 - 0.127999684100039 z_k1_s1032
                                 - 0.105000094057549 z_k1_s1033
                                 - 0.281000586596378 z_k1_s1034
                                 - 0.0870001496934967 z_k1_s1035
                                 - 0.189000552024115 z_k1_s1036
                                 - 0.204000804661237 z_k1_s1037
                                 - 0.350999152124057 z_k1_s1040
                                 - 0.18000034334399 z_k1_s1041
                                 - 0.138999778072244 z_k1_s1042
                                 - 0.182999730867725 z_k1_s1043
                                 - 0.122999672619607 z_k1_s1044
                                 - 0.284999138631018 z_k1_s1045
                                 - 0.199999632161607 z_k1_s1046
                                 - 0.224999586181808 z_k1_s1047
                                 - 0.264000179786531 z_k1_s1050
                                 - 0.105000137806085 z_k1_s1051
                                 - 0.100000163301071 z_k1_s1052
                                 - 0.177000197038787 z_k1_s1053
                                 - 0.0690000520896675 z_k1_s1054
                                 - 0.204000036925835 z_k1_s1055
                                 - 0.10800014494912 z_k1_s1056
                                 - 0.0950000765959248 z_k1_s1057
                                 - 0.0830000922077741 z_k1_s1060
                                 - 0.14599976969022 z_k1_s1061
                                 - 0.11500050516794 z_k1_s1062
                                 - 0.054000081556896 z_k1_s1063
                                 - 0.134000487456411 z_k1_s1064
                                 - 0.160999991234087 z_k1_s1065
                                 - 0.0680001391642918 z_k1_s1066
                                 - 0.134000487456411 z_k1_s1067
                                 - 0.109999741293187 z_k1_s1070
                                 - 0.151999624805118 z_k1_s1071
                                 - 0.183000118588142 z_k1_s1072
                                 - 0.142000533779087 z_k1_s1073
                                 - 0.092000111245441 z_k1_s1074
                                 - 0.0980000528614019 z_k1_s1075
                                 - 0.109000416590994 z_k1_s1076
                                 - 0.0880000852347548 z_k1_s1077
                                 - 1.51100064467295 z_k1_s1100
                                 - 2.40000418341509 z_k1_s1101
                                 - 1.19599892878841 z_k1_s1102
                                 - 1.17799936413732 z_k1_s1103
                                 - 3.27599277542479 z_k1_s1104
                                 - 0.951999348023677 z_k1_s1105
                                 - 0.824999604488308 z_k1_s1106
                                 - 1.99699422424691 z_k1_s1107
                                 - 0.517998650345509 z_k1_s1110
                                 - 1.19399746923351 z_k1_s1111
                                 - 0.865000357192586 z_k1_s1112
                                 - 0.911000817552266 z_k1_s1113
                                 - 0.537999062405832 z_k1_s1114
                                 - 0.84299519259531 z_k1_s1115
                                 - 1.08299688812544 z_k1_s1116
                                 - 2.17099753460891 z_k1_s1117
                                 - 1.62899963578955 z_k1_s1120
                                 - 1.52400276029835 z_k1_s1121
                                 - 0.650001091752296 z_k1_s1122
                                 - 1.25700216977739 z_k1_s1123
                                 - 0.413000384487309 z_k1_s1124
                                 - 1.48599820060598 z_k1_s1125
                                 - 0.856001557532154 z_k1_s1126
                                 - 1.2379998899312 z_k1_s1127
                                 - 0.378999857653757 z_k1_s1130
                                 - 0.782998578177448 z_k1_s1131
                                 - 0.261999725547073 z_k1_s1132
                                 - 0.610998433559033 z_k1_s1133
                                 - 0.356999659249228 z_k1_s1134
                                 - 0.405999441832599 z_k1_s1135
                                 - 1.24499694267746 z_k1_s1136
                                 - 0.792001340567943 z_k1_s1137
                                 - 0.758002091981958 z_k1_s1140
                                 - 0.803001469221303 z_k1_s1141
                                 - 0.782998171560862 z_k1_s1142
                                 - 1.49100167038769 z_k1_s1143
                                 - 0.70900076841106 z_k1_s1144
                                 - 0.579999765592764 z_k1_s1145
                                 - 0.543999620401556 z_k1_s1146
                                 - 0.765999550647743 z_k1_s1147
                                 - 0.780997179104386 z_k1_s1150
                                 - 0.606997725502209 z_k1_s1151
                                 - 0.566997518665139 z_k1_s1152
                                 - 1.37799074898914 z_k1_s1153
                                 - 0.645997360438174 z_k1_s1154
                                 - 1.2689909475125 z_k1_s1155
                                 - 1.70398958151795 z_k1_s1156
                                 - 0.864994408505362 z_k1_s1157
                                 - 1.33699909048608 z_k1_s1160
                                 - 1.50300020664483 z_k1_s1161
                                 - 0.620999070490105 z_k1_s1162
                                 - 1.1929943933682 z_k1_s1163
                                 - 0.858000353301747 z_k1_s1164
                                 - 0.744996974409995 z_k1_s1165
                                 - 0.51199926093333 z_k1_s1166
                                 - 0.599998938167593 z_k1_s1167
                                 - 0.480000337398303 z_k1_s1170
                                 - 1.18700265368147 z_k1_s1171
                                 - 0.48799997771489 z_k1_s1172
                                 - 0.391000084975674 z_k1_s1173
                                 - 0.888001027946916 z_k1_s1174
                                 - 0.573000121579185 z_k1_s1175
                                 - 0.705997530542923 z_k1_s1176
                                 - 1.05500195525686 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.887683 w_k1_ADM_s000
                                 + 0.817027 w_k1_AXP_s000
                                 + 0.669092 w_k1_DIS_s000
                                 + 0.792452 w_k1_FDX_s000
                                 + 0.479873 w_k1_IBM_s000
                                 + 0.791842 w_k1_INTC_s000
                                 + 0.614635 w_k1_KO_s000
                                 + 0.478414 w_k1_MAT_s000
                                 + 0.800031 w_k1_MSFT_s000
                                 + 0.785184 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.16419 w_k1_ADM_s000
                                 + 1.33286 w_k1_AXP_s000
                                 + 1.24094 w_k1_DIS_s000
                                 + 1.03566 w_k1_FDX_s000
                                 + 1.76566 w_k1_IBM_s000
                                 + 0.840711 w_k1_INTC_s000
                                 + 1.15133 w_k1_KO_s000 + 1.74282 w_k1_MAT_s000
                                 + 0.873707 w_k1_MSFT_s000
                                 + 0.99434 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.1438 w_k1_ADM_s000
                                 + 1.52069 w_k1_AXP_s000 + 1.276 w_k1_DIS_s000
                                 + 1.17101 w_k1_FDX_s000
                                 + 1.18572 w_k1_IBM_s000
                                 + 1.17421 w_k1_INTC_s000
                                 + 0.972421 w_k1_KO_s000
                                 + 1.45664 w_k1_MAT_s000
                                 + 1.1046 w_k1_MSFT_s000
                                 + 0.983022 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.12045 w_k1_ADM_s000
                                 + 1.26236 w_k1_AXP_s000
                                 + 1.61643 w_k1_DIS_s000
                                 + 1.33526 w_k1_FDX_s000
                                 + 1.06733 w_k1_IBM_s000
                                 + 1.1929 w_k1_INTC_s000 + 1.01928 w_k1_KO_s000
                                 + 0.884498 w_k1_MAT_s000
                                 + 1.16243 w_k1_MSFT_s000
                                 + 1.04082 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.881735 w_k1_ADM_s000
                                 + 0.864634 w_k1_AXP_s000
                                 + 0.695614 w_k1_DIS_s000
                                 + 0.745614 w_k1_FDX_s000
                                 + 1.24954 w_k1_IBM_s000
                                 + 0.647028 w_k1_INTC_s000
                                 + 0.955884 w_k1_KO_s000
                                 + 1.24134 w_k1_MAT_s000
                                 + 0.753902 w_k1_MSFT_s000
                                 + 0.818367 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.25868 w_k1_ADM_s000
                                 + 1.15061 w_k1_AXP_s000
                                 + 0.982556 w_k1_DIS_s000
                                 + 1.20833 w_k1_FDX_s000
                                 + 0.612942 w_k1_IBM_s000
                                 + 0.973614 w_k1_INTC_s000
                                 + 0.646977 w_k1_KO_s000
                                 + 0.538267 w_k1_MAT_s000
                                 + 1.16732 w_k1_MSFT_s000
                                 + 0.956284 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.04023 w_k1_ADM_s000
                                 + 0.84957 w_k1_AXP_s000
                                 + 0.808215 w_k1_DIS_s000
                                 + 0.861052 w_k1_FDX_s000
                                 + 0.829358 w_k1_IBM_s000
                                 + 0.730469 w_k1_INTC_s000
                                 + 0.846688 w_k1_KO_s000
                                 + 0.766035 w_k1_MAT_s000
                                 + 0.850151 w_k1_MSFT_s000
                                 + 0.763831 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.03335 w_k1_ADM_s000
                                 + 1.14707 w_k1_AXP_s000
                                 + 1.14757 w_k1_DIS_s000 + 1.0992 w_k1_FDX_s000
                                 + 0.776229 w_k1_IBM_s000
                                 + 0.908307 w_k1_INTC_s000
                                 + 0.816844 w_k1_KO_s000
                                 + 0.906906 w_k1_MAT_s000
                                 + 1.04532 w_k1_MSFT_s000
                                 + 0.964542 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.80406 w_k1_ADM_s001
                                 + 0.559394 w_k1_AXP_s001
                                 + 0.684953 w_k1_DIS_s001
                                 + 0.744726 w_k1_FDX_s001
                                 + 0.797156 w_k1_IBM_s001
                                 + 0.940074 w_k1_INTC_s001
                                 + 0.890935 w_k1_KO_s001
                                 + 0.783055 w_k1_MAT_s001
                                 + 0.91007 w_k1_MSFT_s001
                                 + 0.820897 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.01898 w_k1_ADM_s001
                                 + 0.807697 w_k1_AXP_s001
                                 + 0.840764 w_k1_DIS_s001
                                 + 0.977139 w_k1_FDX_s001
                                 + 0.620609 w_k1_IBM_s001
                                 + 1.03992 w_k1_INTC_s001
                                 + 0.843665 w_k1_KO_s001
                                 + 0.653902 w_k1_MAT_s001
                                 + 1.04235 w_k1_MSFT_s001
                                 + 0.954528 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.02028 w_k1_ADM_s001
                                 + 0.972449 w_k1_AXP_s001
                                 + 0.991726 w_k1_DIS_s001
                                 + 0.997176 w_k1_FDX_s001
                                 + 1.07713 w_k1_IBM_s001
                                 + 0.950092 w_k1_INTC_s001
                                 + 1.04312 w_k1_KO_s001 + 1.12516 w_k1_MAT_s001
                                 + 1.00497 w_k1_MSFT_s001
                                 + 1.05204 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.15788 w_k1_ADM_s001
                                 + 1.38232 w_k1_AXP_s001
                                 + 1.71751 w_k1_DIS_s001
                                 + 1.38431 w_k1_FDX_s001
                                 + 1.18638 w_k1_IBM_s001
                                 + 1.3926 w_k1_INTC_s001 + 1.54026 w_k1_KO_s001
                                 + 1.64234 w_k1_MAT_s001
                                 + 1.51064 w_k1_MSFT_s001 + 1.3773 w_k1_PG_s001
                                 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.09309 w_k1_ADM_s001
                                 + 1.22826 w_k1_AXP_s001
                                 + 1.09458 w_k1_DIS_s001
                                 + 1.20499 w_k1_FDX_s001
                                 + 0.860288 w_k1_IBM_s001
                                 + 1.00339 w_k1_INTC_s001
                                 + 1.06644 w_k1_KO_s001 + 1.49717 w_k1_MAT_s001
                                 + 1.29184 w_k1_MSFT_s001
                                 + 1.16462 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.99929 w_k1_ADM_s001
                                 + 0.852232 w_k1_AXP_s001
                                 + 0.805544 w_k1_DIS_s001
                                 + 0.818456 w_k1_FDX_s001
                                 + 1.16555 w_k1_IBM_s001
                                 + 0.902366 w_k1_INTC_s001
                                 + 0.987499 w_k1_KO_s001
                                 + 1.62828 w_k1_MAT_s001
                                 + 0.976364 w_k1_MSFT_s001
                                 + 0.971352 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.19555 w_k1_ADM_s001
                                 + 1.03888 w_k1_AXP_s001
                                 + 1.23859 w_k1_DIS_s001
                                 + 1.08498 w_k1_FDX_s001
                                 + 1.65176 w_k1_IBM_s001
                                 + 0.990017 w_k1_INTC_s001
                                 + 1.61251 w_k1_KO_s001 + 1.98173 w_k1_MAT_s001
                                 + 1.3108 w_k1_MSFT_s001 + 1.05035 w_k1_PG_s001
                                 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.13629 w_k1_ADM_s001
                                 + 1.08452 w_k1_AXP_s001
                                 + 1.48067 w_k1_DIS_s001
                                 + 1.15331 w_k1_FDX_s001
                                 + 1.14768 w_k1_IBM_s001
                                 + 1.27932 w_k1_INTC_s001
                                 + 1.23782 w_k1_KO_s001 + 1.02872 w_k1_MAT_s001
                                 + 1.30737 w_k1_MSFT_s001
                                 + 1.02986 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.716241 w_k1_ADM_s002
                                 + 0.939603 w_k1_AXP_s002
                                 + 0.964358 w_k1_DIS_s002
                                 + 0.824267 w_k1_FDX_s002
                                 + 1.47592 w_k1_IBM_s002
                                 + 0.895143 w_k1_INTC_s002
                                 + 1.19919 w_k1_KO_s002 + 1.49565 w_k1_MAT_s002
                                 + 1.09792 w_k1_MSFT_s002
                                 + 1.12457 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.98269 w_k1_ADM_s002
                                 + 1.23935 w_k1_AXP_s002
                                 + 1.40615 w_k1_DIS_s002
                                 + 1.23864 w_k1_FDX_s002
                                 + 0.964155 w_k1_IBM_s002
                                 + 1.32429 w_k1_INTC_s002
                                 + 1.11445 w_k1_KO_s002 + 1.0604 w_k1_MAT_s002
                                 + 1.39783 w_k1_MSFT_s002
                                 + 1.09199 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.760868 w_k1_ADM_s002
                                 + 0.64825 w_k1_AXP_s002
                                 + 0.662701 w_k1_DIS_s002
                                 + 0.741221 w_k1_FDX_s002
                                 + 0.834687 w_k1_IBM_s002
                                 + 0.836432 w_k1_INTC_s002
                                 + 0.855079 w_k1_KO_s002
                                 + 0.906196 w_k1_MAT_s002
                                 + 0.892184 w_k1_MSFT_s002
                                 + 0.86417 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.1101 w_k1_ADM_s002
                                 + 1.60539 w_k1_AXP_s002
                                 + 1.36465 w_k1_DIS_s002
                                 + 1.22233 w_k1_FDX_s002
                                 + 1.64351 w_k1_IBM_s002 + 1.417 w_k1_INTC_s002
                                 + 1.31746 w_k1_KO_s002 + 1.79101 w_k1_MAT_s002
                                 + 1.3485 w_k1_MSFT_s002 + 1.15654 w_k1_PG_s002
                                 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.863014 w_k1_ADM_s002
                                 + 1.01979 w_k1_AXP_s002
                                 + 0.89438 w_k1_DIS_s002
                                 + 0.793755 w_k1_FDX_s002
                                 + 1.23817 w_k1_IBM_s002
                                 + 1.02864 w_k1_INTC_s002
                                 + 1.17076 w_k1_KO_s002 + 1.28506 w_k1_MAT_s002
                                 + 1.00742 w_k1_MSFT_s002
                                 + 1.00143 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.01206 w_k1_ADM_s002
                                 + 1.01077 w_k1_AXP_s002
                                 + 1.02238 w_k1_DIS_s002
                                 + 1.02446 w_k1_FDX_s002
                                 + 0.767954 w_k1_IBM_s002
                                 + 0.99088 w_k1_INTC_s002
                                 + 0.949994 w_k1_KO_s002
                                 + 0.735515 w_k1_MAT_s002
                                 + 1.1315 w_k1_MSFT_s002
                                 + 0.906612 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.50651 w_k1_ADM_s002
                                 + 1.64548 w_k1_AXP_s002
                                 + 1.58745 w_k1_DIS_s002
                                 + 1.52511 w_k1_FDX_s002
                                 + 0.851336 w_k1_IBM_s002
                                 + 1.26957 w_k1_INTC_s002
                                 + 1.25436 w_k1_KO_s002 + 1.16788 w_k1_MAT_s002
                                 + 1.67853 w_k1_MSFT_s002
                                 + 1.16643 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.853209 w_k1_ADM_s002
                                 + 0.945857 w_k1_AXP_s002
                                 + 0.989276 w_k1_DIS_s002
                                 + 1.00015 w_k1_FDX_s002
                                 + 1.15389 w_k1_IBM_s002
                                 + 1.05168 w_k1_INTC_s002
                                 + 1.06323 w_k1_KO_s002
                                 + 0.854849 w_k1_MAT_s002
                                 + 1.15529 w_k1_MSFT_s002
                                 + 1.03898 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.747708 w_k1_ADM_s003
                                 + 0.501652 w_k1_AXP_s003
                                 + 0.753294 w_k1_DIS_s003
                                 + 0.827305 w_k1_FDX_s003
                                 + 0.669271 w_k1_IBM_s003
                                 + 0.780447 w_k1_INTC_s003
                                 + 0.820804 w_k1_KO_s003
                                 + 0.693051 w_k1_MAT_s003
                                 + 0.967893 w_k1_MSFT_s003
                                 + 0.850685 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.700502 w_k1_ADM_s003
                                 + 0.544582 w_k1_AXP_s003
                                 + 0.626197 w_k1_DIS_s003
                                 + 0.647098 w_k1_FDX_s003
                                 + 0.972091 w_k1_IBM_s003
                                 + 0.652379 w_k1_INTC_s003
                                 + 0.885932 w_k1_KO_s003
                                 + 0.721136 w_k1_MAT_s003
                                 + 0.692687 w_k1_MSFT_s003
                                 + 0.79569 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.834309 w_k1_ADM_s003
                                 + 1.13432 w_k1_AXP_s003
                                 + 1.06996 w_k1_DIS_s003
                                 + 0.980711 w_k1_FDX_s003
                                 + 1.31954 w_k1_IBM_s003
                                 + 1.1511 w_k1_INTC_s003 + 1.106 w_k1_KO_s003
                                 + 1.32473 w_k1_MAT_s003
                                 + 1.14874 w_k1_MSFT_s003
                                 + 0.995636 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.02871 w_k1_ADM_s003
                                 + 0.826007 w_k1_AXP_s003
                                 + 0.809321 w_k1_DIS_s003
                                 + 0.875867 w_k1_FDX_s003
                                 + 0.808851 w_k1_IBM_s003
                                 + 0.959021 w_k1_INTC_s003
                                 + 0.826331 w_k1_KO_s003
                                 + 0.599527 w_k1_MAT_s003
                                 + 0.741838 w_k1_MSFT_s003
                                 + 0.825658 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.38353 w_k1_ADM_s003
                                 + 1.29169 w_k1_AXP_s003
                                 + 1.75119 w_k1_DIS_s003 + 1.463 w_k1_FDX_s003
                                 + 1.11136 w_k1_IBM_s003
                                 + 1.38239 w_k1_INTC_s003
                                 + 1.39245 w_k1_KO_s003 + 1.33248 w_k1_MAT_s003
                                 + 1.49436 w_k1_MSFT_s003
                                 + 1.08173 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.994227 w_k1_ADM_s003
                                 + 0.744356 w_k1_AXP_s003
                                 + 0.833597 w_k1_DIS_s003
                                 + 0.719371 w_k1_FDX_s003
                                 + 1.48345 w_k1_IBM_s003
                                 + 0.855038 w_k1_INTC_s003
                                 + 1.1219 w_k1_KO_s003 + 1.37609 w_k1_MAT_s003
                                 + 0.883313 w_k1_MSFT_s003
                                 + 0.948352 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.962475 w_k1_ADM_s003
                                 + 0.915864 w_k1_AXP_s003
                                 + 1.30441 w_k1_DIS_s003
                                 + 1.15071 w_k1_FDX_s003
                                 + 0.813025 w_k1_IBM_s003
                                 + 1.06302 w_k1_INTC_s003
                                 + 1.13738 w_k1_KO_s003
                                 + 0.823326 w_k1_MAT_s003
                                 + 1.207 w_k1_MSFT_s003 + 0.906937 w_k1_PG_s003
                                 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.20418 w_k1_ADM_s003
                                 + 0.727077 w_k1_AXP_s003
                                 + 0.934873 w_k1_DIS_s003
                                 + 0.89896 w_k1_FDX_s003
                                 + 0.97046 w_k1_IBM_s003
                                 + 1.11071 w_k1_INTC_s003
                                 + 1.13698 w_k1_KO_s003
                                 + 0.785013 w_k1_MAT_s003
                                 + 1.1247 w_k1_MSFT_s003 + 1.06824 w_k1_PG_s003
                                 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.1904 w_k1_ADM_s004
                                 + 1.02006 w_k1_AXP_s004 + 1.2201 w_k1_DIS_s004
                                 + 0.996827 w_k1_FDX_s004
                                 + 1.09648 w_k1_IBM_s004
                                 + 1.02521 w_k1_INTC_s004
                                 + 1.00785 w_k1_KO_s004 + 1.10308 w_k1_MAT_s004
                                 + 1.20127 w_k1_MSFT_s004
                                 + 0.970367 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.935235 w_k1_ADM_s004
                                 + 0.59596 w_k1_AXP_s004
                                 + 0.753171 w_k1_DIS_s004
                                 + 0.803395 w_k1_FDX_s004
                                 + 0.566048 w_k1_IBM_s004
                                 + 0.962184 w_k1_INTC_s004
                                 + 0.612665 w_k1_KO_s004
                                 + 0.319931 w_k1_MAT_s004
                                 + 0.897276 w_k1_MSFT_s004
                                 + 0.705479 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.46199 w_k1_ADM_s004
                                 + 1.3771 w_k1_AXP_s004 + 1.82655 w_k1_DIS_s004
                                 + 1.41601 w_k1_FDX_s004
                                 + 0.963327 w_k1_IBM_s004
                                 + 1.49516 w_k1_INTC_s004
                                 + 1.10464 w_k1_KO_s004 + 1.33428 w_k1_MAT_s004
                                 + 1.42698 w_k1_MSFT_s004
                                 + 1.15807 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.2646 w_k1_ADM_s004
                                 + 1.27505 w_k1_AXP_s004
                                 + 1.11641 w_k1_DIS_s004
                                 + 1.18879 w_k1_FDX_s004
                                 + 0.942603 w_k1_IBM_s004
                                 + 0.995856 w_k1_INTC_s004
                                 + 0.840638 w_k1_KO_s004
                                 + 1.35398 w_k1_MAT_s004
                                 + 1.13027 w_k1_MSFT_s004
                                 + 1.01378 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.797113 w_k1_ADM_s004
                                 + 0.707158 w_k1_AXP_s004
                                 + 0.727448 w_k1_DIS_s004
                                 + 0.739758 w_k1_FDX_s004
                                 + 0.996205 w_k1_IBM_s004
                                 + 0.948825 w_k1_INTC_s004
                                 + 0.724576 w_k1_KO_s004
                                 + 0.778417 w_k1_MAT_s004
                                 + 0.905647 w_k1_MSFT_s004
                                 + 0.85719 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.06339 w_k1_ADM_s004
                                 + 1.05351 w_k1_AXP_s004
                                 + 1.12284 w_k1_DIS_s004
                                 + 0.894754 w_k1_FDX_s004
                                 + 1.52798 w_k1_IBM_s004
                                 + 0.916992 w_k1_INTC_s004
                                 + 1.09425 w_k1_KO_s004 + 1.68216 w_k1_MAT_s004
                                 + 0.976058 w_k1_MSFT_s004
                                 + 1.07443 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.992487 w_k1_ADM_s004
                                 + 0.889589 w_k1_AXP_s004
                                 + 0.939559 w_k1_DIS_s004
                                 + 0.974397 w_k1_FDX_s004
                                 + 0.7626 w_k1_IBM_s004
                                 + 1.09934 w_k1_INTC_s004
                                 + 0.673097 w_k1_KO_s004
                                 + 0.49236 w_k1_MAT_s004
                                 + 0.937773 w_k1_MSFT_s004
                                 + 0.852258 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.21484 w_k1_ADM_s004
                                 + 1.13401 w_k1_AXP_s004
                                 + 1.62005 w_k1_DIS_s004
                                 + 1.15256 w_k1_FDX_s004
                                 + 1.49671 w_k1_IBM_s004
                                 + 0.938201 w_k1_INTC_s004
                                 + 1.1639 w_k1_KO_s004 + 1.75572 w_k1_MAT_s004
                                 + 1.22146 w_k1_MSFT_s004
                                 + 1.13486 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.751679 w_k1_ADM_s005
                                 + 0.798029 w_k1_AXP_s005
                                 + 1.02858 w_k1_DIS_s005
                                 + 0.929858 w_k1_FDX_s005
                                 + 1.21708 w_k1_IBM_s005
                                 + 0.997701 w_k1_INTC_s005
                                 + 1.10359 w_k1_KO_s005
                                 + 0.978959 w_k1_MAT_s005
                                 + 0.972271 w_k1_MSFT_s005
                                 + 0.897221 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.15533 w_k1_ADM_s005
                                 + 1.26682 w_k1_AXP_s005
                                 + 0.78891 w_k1_DIS_s005
                                 + 1.03686 w_k1_FDX_s005
                                 + 0.991102 w_k1_IBM_s005
                                 + 1.00655 w_k1_INTC_s005
                                 + 0.921491 w_k1_KO_s005
                                 + 1.25247 w_k1_MAT_s005
                                 + 0.940624 w_k1_MSFT_s005
                                 + 1.10508 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.01699 w_k1_ADM_s005
                                 + 0.971787 w_k1_AXP_s005
                                 + 1.19963 w_k1_DIS_s005
                                 + 1.18934 w_k1_FDX_s005
                                 + 0.925711 w_k1_IBM_s005
                                 + 1.35748 w_k1_INTC_s005
                                 + 0.951616 w_k1_KO_s005
                                 + 0.830711 w_k1_MAT_s005
                                 + 1.18442 w_k1_MSFT_s005
                                 + 1.07894 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.31386 w_k1_ADM_s005
                                 + 1.39483 w_k1_AXP_s005
                                 + 1.13289 w_k1_DIS_s005
                                 + 1.23353 w_k1_FDX_s005
                                 + 1.21469 w_k1_IBM_s005
                                 + 1.22781 w_k1_INTC_s005
                                 + 1.19315 w_k1_KO_s005 + 1.30879 w_k1_MAT_s005
                                 + 1.20674 w_k1_MSFT_s005
                                 + 1.20649 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.984265 w_k1_ADM_s005
                                 + 0.922709 w_k1_AXP_s005
                                 + 0.635895 w_k1_DIS_s005
                                 + 0.856447 w_k1_FDX_s005
                                 + 0.86277 w_k1_IBM_s005
                                 + 0.887483 w_k1_INTC_s005
                                 + 0.818117 w_k1_KO_s005
                                 + 0.995105 w_k1_MAT_s005
                                 + 0.778058 w_k1_MSFT_s005
                                 + 0.962976 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.574526 w_k1_ADM_s005
                                 + 0.461056 w_k1_AXP_s005
                                 + 0.657613 w_k1_DIS_s005
                                 + 0.692081 w_k1_FDX_s005
                                 + 0.876391 w_k1_IBM_s005
                                 + 0.883959 w_k1_INTC_s005
                                 + 0.803201 w_k1_KO_s005
                                 + 0.945308 w_k1_MAT_s005
                                 + 0.820504 w_k1_MSFT_s005
                                 + 0.921467 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.18573 w_k1_ADM_s005
                                 + 1.15468 w_k1_AXP_s005
                                 + 1.65345 w_k1_DIS_s005
                                 + 1.42106 w_k1_FDX_s005
                                 + 1.24188 w_k1_IBM_s005
                                 + 1.69702 w_k1_INTC_s005
                                 + 1.22825 w_k1_KO_s005
                                 + 0.949846 w_k1_MAT_s005
                                 + 1.40492 w_k1_MSFT_s005 + 1.1405 w_k1_PG_s005
                                 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.843888 w_k1_ADM_s005
                                 + 0.807727 w_k1_AXP_s005
                                 + 0.796384 w_k1_DIS_s005
                                 + 0.962811 w_k1_FDX_s005
                                 + 0.798523 w_k1_IBM_s005
                                 + 1.21511 w_k1_INTC_s005
                                 + 0.839984 w_k1_KO_s005
                                 + 0.611316 w_k1_MAT_s005
                                 + 0.931738 w_k1_MSFT_s005
                                 + 0.935794 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.38839 w_k1_ADM_s006
                                 + 1.59371 w_k1_AXP_s006 + 1.3419 w_k1_DIS_s006
                                 + 1.23344 w_k1_FDX_s006
                                 + 1.17871 w_k1_IBM_s006
                                 + 1.19451 w_k1_INTC_s006
                                 + 1.20367 w_k1_KO_s006 + 1.73601 w_k1_MAT_s006
                                 + 1.205 w_k1_MSFT_s006 + 1.26755 w_k1_PG_s006
                                 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.927636 w_k1_ADM_s006
                                 + 0.868492 w_k1_AXP_s006
                                 + 1.03929 w_k1_DIS_s006
                                 + 0.825106 w_k1_FDX_s006
                                 + 1.19002 w_k1_IBM_s006
                                 + 0.844087 w_k1_INTC_s006
                                 + 1.17205 w_k1_KO_s006 + 1.05243 w_k1_MAT_s006
                                 + 0.930308 w_k1_MSFT_s006
                                 + 1.03232 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.04239 w_k1_ADM_s006
                                 + 1.07989 w_k1_AXP_s006 + 1.303 w_k1_DIS_s006
                                 + 1.12687 w_k1_FDX_s006
                                 + 1.59425 w_k1_IBM_s006
                                 + 1.05439 w_k1_INTC_s006
                                 + 1.43186 w_k1_KO_s006 + 1.60491 w_k1_MAT_s006
                                 + 1.10654 w_k1_MSFT_s006
                                 + 1.27159 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.874378 w_k1_ADM_s006
                                 + 0.734454 w_k1_AXP_s006
                                 + 0.720793 w_k1_DIS_s006
                                 + 0.786624 w_k1_FDX_s006
                                 + 0.674257 w_k1_IBM_s006
                                 + 0.837835 w_k1_INTC_s006
                                 + 0.810962 w_k1_KO_s006
                                 + 0.543522 w_k1_MAT_s006
                                 + 0.887543 w_k1_MSFT_s006
                                 + 0.769839 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.35509 w_k1_ADM_s006
                                 + 1.20424 w_k1_AXP_s006
                                 + 1.09393 w_k1_DIS_s006
                                 + 1.12046 w_k1_FDX_s006
                                 + 0.846012 w_k1_IBM_s006
                                 + 0.867718 w_k1_INTC_s006
                                 + 0.910619 w_k1_KO_s006
                                 + 0.944106 w_k1_MAT_s006
                                 + 1.0338 w_k1_MSFT_s006 + 1.00372 w_k1_PG_s006
                                 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.22399 w_k1_ADM_s006
                                 + 1.40621 w_k1_AXP_s006
                                 + 1.30368 w_k1_DIS_s006
                                 + 1.30641 w_k1_FDX_s006
                                 + 0.796416 w_k1_IBM_s006
                                 + 1.117 w_k1_INTC_s006 + 0.998035 w_k1_KO_s006
                                 + 0.9469 w_k1_MAT_s006
                                 + 1.20497 w_k1_MSFT_s006 + 1.0994 w_k1_PG_s006
                                 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.668603 w_k1_ADM_s006
                                 + 0.777316 w_k1_AXP_s006
                                 + 0.599368 w_k1_DIS_s006
                                 + 0.724001 w_k1_FDX_s006
                                 + 0.663129 w_k1_IBM_s006
                                 + 0.790148 w_k1_INTC_s006
                                 + 0.820556 w_k1_KO_s006
                                 + 0.83835 w_k1_MAT_s006
                                 + 0.755623 w_k1_MSFT_s006
                                 + 0.844466 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.946449 w_k1_ADM_s006
                                 + 0.992308 w_k1_AXP_s006
                                 + 0.895779 w_k1_DIS_s006
                                 + 0.919098 w_k1_FDX_s006
                                 + 0.830871 w_k1_IBM_s006
                                 + 0.983074 w_k1_INTC_s006
                                 + 0.866623 w_k1_KO_s006
                                 + 0.685845 w_k1_MAT_s006
                                 + 0.883392 w_k1_MSFT_s006
                                 + 0.883354 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.687477 w_k1_ADM_s007
                                 + 0.801834 w_k1_AXP_s007
                                 + 0.606738 w_k1_DIS_s007
                                 + 0.776084 w_k1_FDX_s007
                                 + 0.69583 w_k1_IBM_s007
                                 + 0.972947 w_k1_INTC_s007
                                 + 0.831686 w_k1_KO_s007
                                 + 0.634639 w_k1_MAT_s007
                                 + 0.681743 w_k1_MSFT_s007
                                 + 0.795712 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.896496 w_k1_ADM_s007
                                 + 1.46736 w_k1_AXP_s007
                                 + 1.23693 w_k1_DIS_s007
                                 + 1.13392 w_k1_FDX_s007
                                 + 1.38976 w_k1_IBM_s007
                                 + 1.14213 w_k1_INTC_s007
                                 + 1.49028 w_k1_KO_s007 + 1.80583 w_k1_MAT_s007
                                 + 1.02272 w_k1_MSFT_s007
                                 + 1.17885 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.703514 w_k1_ADM_s007
                                 + 0.873913 w_k1_AXP_s007
                                 + 0.738155 w_k1_DIS_s007
                                 + 0.749262 w_k1_FDX_s007 + 1.138 w_k1_IBM_s007
                                 + 0.762446 w_k1_INTC_s007
                                 + 1.09408 w_k1_KO_s007 + 1.17992 w_k1_MAT_s007
                                 + 0.852311 w_k1_MSFT_s007
                                 + 1.0607 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.974046 w_k1_ADM_s007
                                 + 1.01887 w_k1_AXP_s007
                                 + 1.05085 w_k1_DIS_s007
                                 + 1.09147 w_k1_FDX_s007
                                 + 0.71302 w_k1_IBM_s007
                                 + 1.27542 w_k1_INTC_s007
                                 + 0.916123 w_k1_KO_s007
                                 + 0.775969 w_k1_MAT_s007
                                 + 1.02153 w_k1_MSFT_s007
                                 + 1.01564 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.783552 w_k1_ADM_s007
                                 + 0.815024 w_k1_AXP_s007
                                 + 0.762349 w_k1_DIS_s007
                                 + 0.886293 w_k1_FDX_s007
                                 + 0.647681 w_k1_IBM_s007
                                 + 0.926105 w_k1_INTC_s007
                                 + 0.877743 w_k1_KO_s007
                                 + 0.619584 w_k1_MAT_s007
                                 + 0.831556 w_k1_MSFT_s007
                                 + 1.0017 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.856043 w_k1_ADM_s007
                                 + 1.07758 w_k1_AXP_s007
                                 + 1.14152 w_k1_DIS_s007
                                 + 0.984566 w_k1_FDX_s007
                                 + 1.02822 w_k1_IBM_s007
                                 + 1.00427 w_k1_INTC_s007
                                 + 1.41054 w_k1_KO_s007 + 1.32183 w_k1_MAT_s007
                                 + 1.07263 w_k1_MSFT_s007
                                 + 1.10879 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.09662 w_k1_ADM_s007
                                 + 1.67146 w_k1_AXP_s007
                                 + 1.59678 w_k1_DIS_s007 + 1.3116 w_k1_FDX_s007
                                 + 1.51056 w_k1_IBM_s007
                                 + 1.22075 w_k1_INTC_s007
                                 + 1.49561 w_k1_KO_s007 + 1.81506 w_k1_MAT_s007
                                 + 1.28522 w_k1_MSFT_s007 + 1.356 w_k1_PG_s007
                                 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.930851 w_k1_ADM_s007
                                 + 1.10528 w_k1_AXP_s007
                                 + 1.09585 w_k1_DIS_s007
                                 + 0.952823 w_k1_FDX_s007
                                 + 1.22504 w_k1_IBM_s007
                                 + 0.983754 w_k1_INTC_s007
                                 + 1.25429 w_k1_KO_s007 + 1.39666 w_k1_MAT_s007
                                 + 1.05085 w_k1_MSFT_s007
                                 + 1.10365 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.779737 w_k1_ADM_s010
                                 + 0.749547 w_k1_AXP_s010
                                 + 0.818779 w_k1_DIS_s010
                                 + 0.833196 w_k1_FDX_s010
                                 + 1.09415 w_k1_IBM_s010
                                 + 0.830354 w_k1_INTC_s010
                                 + 0.925444 w_k1_KO_s010
                                 + 0.812878 w_k1_MAT_s010
                                 + 0.853863 w_k1_MSFT_s010
                                 + 0.942089 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.760297 w_k1_ADM_s010
                                 + 0.712888 w_k1_AXP_s010
                                 + 0.574094 w_k1_DIS_s010
                                 + 0.753062 w_k1_FDX_s010
                                 + 0.805347 w_k1_IBM_s010
                                 + 0.858803 w_k1_INTC_s010
                                 + 0.676033 w_k1_KO_s010
                                 + 0.590844 w_k1_MAT_s010
                                 + 0.896066 w_k1_MSFT_s010
                                 + 0.75536 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.04621 w_k1_ADM_s010
                                 + 0.926111 w_k1_AXP_s010
                                 + 0.924118 w_k1_DIS_s010
                                 + 1.06507 w_k1_FDX_s010
                                 + 0.860944 w_k1_IBM_s010
                                 + 1.05013 w_k1_INTC_s010
                                 + 0.874925 w_k1_KO_s010
                                 + 0.619912 w_k1_MAT_s010
                                 + 1.075 w_k1_MSFT_s010 + 0.989436 w_k1_PG_s010
                                 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.785106 w_k1_ADM_s010
                                 + 0.993623 w_k1_AXP_s010
                                 + 0.903948 w_k1_DIS_s010
                                 + 0.949063 w_k1_FDX_s010
                                 + 1.21408 w_k1_IBM_s010
                                 + 1.0668 w_k1_INTC_s010
                                 + 0.991187 w_k1_KO_s010
                                 + 1.22758 w_k1_MAT_s010
                                 + 1.01208 w_k1_MSFT_s010
                                 + 0.916929 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.69883 w_k1_ADM_s010
                                 + 0.641491 w_k1_AXP_s010
                                 + 0.799442 w_k1_DIS_s010
                                 + 0.734836 w_k1_FDX_s010
                                 + 0.859973 w_k1_IBM_s010
                                 + 0.688754 w_k1_INTC_s010
                                 + 0.804063 w_k1_KO_s010
                                 + 0.701506 w_k1_MAT_s010
                                 + 0.990888 w_k1_MSFT_s010
                                 + 0.99916 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.16436 w_k1_ADM_s010
                                 + 1.21918 w_k1_AXP_s010
                                 + 1.42169 w_k1_DIS_s010
                                 + 1.20532 w_k1_FDX_s010
                                 + 1.04001 w_k1_IBM_s010
                                 + 1.07295 w_k1_INTC_s010
                                 + 1.11715 w_k1_KO_s010 + 1.11535 w_k1_MAT_s010
                                 + 1.13334 w_k1_MSFT_s010
                                 + 1.08403 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.29673 w_k1_ADM_s010
                                 + 1.37182 w_k1_AXP_s010
                                 + 1.65913 w_k1_DIS_s010
                                 + 1.35533 w_k1_FDX_s010
                                 + 1.37826 w_k1_IBM_s010
                                 + 1.5196 w_k1_INTC_s010 + 1.41497 w_k1_KO_s010
                                 + 0.983899 w_k1_MAT_s010
                                 + 1.39279 w_k1_MSFT_s010
                                 + 1.21969 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.868131 w_k1_ADM_s010
                                 + 0.731321 w_k1_AXP_s010
                                 + 1.32585 w_k1_DIS_s010
                                 + 0.99265 w_k1_FDX_s010
                                 + 1.01092 w_k1_IBM_s010
                                 + 0.812717 w_k1_INTC_s010
                                 + 1.09726 w_k1_KO_s010
                                 + 0.887362 w_k1_MAT_s010
                                 + 1.1943 w_k1_MSFT_s010
                                 + 0.964737 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.07675 w_k1_ADM_s011
                                 + 1.54841 w_k1_AXP_s011
                                 + 1.00413 w_k1_DIS_s011
                                 + 1.02666 w_k1_FDX_s011
                                 + 1.61237 w_k1_IBM_s011
                                 + 1.11647 w_k1_INTC_s011
                                 + 1.19663 w_k1_KO_s011 + 1.86688 w_k1_MAT_s011
                                 + 0.909502 w_k1_MSFT_s011
                                 + 1.03611 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.00504 w_k1_ADM_s011
                                 + 0.92202 w_k1_AXP_s011
                                 + 0.755461 w_k1_DIS_s011
                                 + 0.919987 w_k1_FDX_s011
                                 + 0.772568 w_k1_IBM_s011
                                 + 0.810009 w_k1_INTC_s011
                                 + 0.753667 w_k1_KO_s011
                                 + 0.991505 w_k1_MAT_s011
                                 + 0.860533 w_k1_MSFT_s011
                                 + 0.745283 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.63047 w_k1_ADM_s011
                                 + 1.30762 w_k1_AXP_s011
                                 + 1.91665 w_k1_DIS_s011
                                 + 1.49665 w_k1_FDX_s011
                                 + 1.43061 w_k1_IBM_s011
                                 + 1.06904 w_k1_INTC_s011 + 1.341 w_k1_KO_s011
                                 + 1.51948 w_k1_MAT_s011
                                 + 1.3015 w_k1_MSFT_s011 + 1.27091 w_k1_PG_s011
                                 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.10319 w_k1_ADM_s011
                                 + 0.770187 w_k1_AXP_s011
                                 + 1.14416 w_k1_DIS_s011
                                 + 1.01422 w_k1_FDX_s011
                                 + 0.858213 w_k1_IBM_s011
                                 + 0.96003 w_k1_INTC_s011
                                 + 0.991775 w_k1_KO_s011
                                 + 0.836243 w_k1_MAT_s011
                                 + 1.05595 w_k1_MSFT_s011
                                 + 1.10207 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.21438 w_k1_ADM_s011
                                 + 0.889247 w_k1_AXP_s011
                                 + 0.901201 w_k1_DIS_s011
                                 + 1.02804 w_k1_FDX_s011
                                 + 0.975004 w_k1_IBM_s011
                                 + 0.929721 w_k1_INTC_s011
                                 + 1.02901 w_k1_KO_s011 + 1.31666 w_k1_MAT_s011
                                 + 0.986233 w_k1_MSFT_s011
                                 + 0.969859 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.2188 w_k1_ADM_s011
                                 + 1.10189 w_k1_AXP_s011
                                 + 0.972391 w_k1_DIS_s011
                                 + 1.02276 w_k1_FDX_s011
                                 + 1.30265 w_k1_IBM_s011
                                 + 0.903161 w_k1_INTC_s011
                                 + 1.07317 w_k1_KO_s011 + 1.29678 w_k1_MAT_s011
                                 + 0.903527 w_k1_MSFT_s011
                                 + 0.987308 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.3223 w_k1_ADM_s011
                                 + 1.084 w_k1_AXP_s011 + 1.45438 w_k1_DIS_s011
                                 + 1.19715 w_k1_FDX_s011
                                 + 1.23683 w_k1_IBM_s011
                                 + 1.12211 w_k1_INTC_s011
                                 + 1.34484 w_k1_KO_s011 + 1.13061 w_k1_MAT_s011
                                 + 1.07908 w_k1_MSFT_s011
                                 + 1.08531 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.74942 w_k1_ADM_s011
                                 + 0.634559 w_k1_AXP_s011
                                 + 0.636817 w_k1_DIS_s011
                                 + 0.774855 w_k1_FDX_s011
                                 + 0.662394 w_k1_IBM_s011
                                 + 0.730951 w_k1_INTC_s011
                                 + 0.771308 w_k1_KO_s011
                                 + 0.679391 w_k1_MAT_s011
                                 + 0.682251 w_k1_MSFT_s011
                                 + 0.806066 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.882871 w_k1_ADM_s012
                                 + 0.781518 w_k1_AXP_s012
                                 + 1.21704 w_k1_DIS_s012
                                 + 1.05615 w_k1_FDX_s012
                                 + 0.799941 w_k1_IBM_s012
                                 + 1.09889 w_k1_INTC_s012
                                 + 1.03974 w_k1_KO_s012
                                 + 0.983895 w_k1_MAT_s012
                                 + 1.11005 w_k1_MSFT_s012
                                 + 0.859318 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.687164 w_k1_ADM_s012
                                 + 0.683893 w_k1_AXP_s012
                                 + 0.68134 w_k1_DIS_s012
                                 + 0.834143 w_k1_FDX_s012
                                 + 0.764453 w_k1_IBM_s012
                                 + 0.84647 w_k1_INTC_s012
                                 + 0.77235 w_k1_KO_s012
                                 + 0.920341 w_k1_MAT_s012
                                 + 0.733863 w_k1_MSFT_s012
                                 + 0.809044 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.86469 w_k1_ADM_s012
                                 + 0.87473 w_k1_AXP_s012
                                 + 0.832919 w_k1_DIS_s012
                                 + 0.981446 w_k1_FDX_s012
                                 + 0.719914 w_k1_IBM_s012
                                 + 0.951212 w_k1_INTC_s012
                                 + 0.769398 w_k1_KO_s012
                                 + 0.705015 w_k1_MAT_s012
                                 + 0.844648 w_k1_MSFT_s012
                                 + 0.833628 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.752694 w_k1_ADM_s012
                                 + 0.9444 w_k1_AXP_s012 + 1.21802 w_k1_DIS_s012
                                 + 0.959272 w_k1_FDX_s012
                                 + 1.48468 w_k1_IBM_s012
                                 + 1.05908 w_k1_INTC_s012 + 1.3359 w_k1_KO_s012
                                 + 1.60292 w_k1_MAT_s012
                                 + 1.04644 w_k1_MSFT_s012
                                 + 1.00575 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.06053 w_k1_ADM_s012
                                 + 1.48413 w_k1_AXP_s012
                                 + 1.70492 w_k1_DIS_s012
                                 + 1.31677 w_k1_FDX_s012
                                 + 1.34774 w_k1_IBM_s012
                                 + 1.14036 w_k1_INTC_s012
                                 + 1.41298 w_k1_KO_s012 + 2.1161 w_k1_MAT_s012
                                 + 1.1196 w_k1_MSFT_s012 + 1.14492 w_k1_PG_s012
                                 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.06154 w_k1_ADM_s012
                                 + 0.952915 w_k1_AXP_s012
                                 + 1.50299 w_k1_DIS_s012
                                 + 1.18647 w_k1_FDX_s012
                                 + 0.982784 w_k1_IBM_s012
                                 + 1.2543 w_k1_INTC_s012 + 1.27079 w_k1_KO_s012
                                 + 0.909044 w_k1_MAT_s012
                                 + 1.22071 w_k1_MSFT_s012
                                 + 0.999488 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.691576 w_k1_ADM_s012
                                 + 0.661253 w_k1_AXP_s012
                                 + 0.722802 w_k1_DIS_s012
                                 + 0.814442 w_k1_FDX_s012
                                 + 1.02131 w_k1_IBM_s012
                                 + 1.03151 w_k1_INTC_s012
                                 + 0.895696 w_k1_KO_s012
                                 + 1.33221 w_k1_MAT_s012
                                 + 0.928155 w_k1_MSFT_s012
                                 + 0.803163 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.945448 w_k1_ADM_s012
                                 + 1.08759 w_k1_AXP_s012
                                 + 1.15941 w_k1_DIS_s012 + 1.1077 w_k1_FDX_s012
                                 + 0.87738 w_k1_IBM_s012
                                 + 1.22281 w_k1_INTC_s012 + 0.9689 w_k1_KO_s012
                                 + 1.13324 w_k1_MAT_s012
                                 + 1.21728 w_k1_MSFT_s012
                                 + 0.853344 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.18076 w_k1_ADM_s013
                                 + 1.19951 w_k1_AXP_s013
                                 + 1.42836 w_k1_DIS_s013
                                 + 1.21065 w_k1_FDX_s013 + 1.058 w_k1_IBM_s013
                                 + 0.890419 w_k1_INTC_s013
                                 + 1.25789 w_k1_KO_s013
                                 + 0.845874 w_k1_MAT_s013
                                 + 1.07431 w_k1_MSFT_s013
                                 + 1.24435 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.08843 w_k1_ADM_s013
                                 + 1.06248 w_k1_AXP_s013
                                 + 0.724724 w_k1_DIS_s013
                                 + 0.933744 w_k1_FDX_s013
                                 + 0.877633 w_k1_IBM_s013
                                 + 0.853322 w_k1_INTC_s013
                                 + 0.852831 w_k1_KO_s013
                                 + 0.884845 w_k1_MAT_s013
                                 + 0.790837 w_k1_MSFT_s013
                                 + 0.836583 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.71071 w_k1_ADM_s013
                                 + 0.692376 w_k1_AXP_s013
                                 + 0.629468 w_k1_DIS_s013
                                 + 0.707064 w_k1_FDX_s013
                                 + 0.726737 w_k1_IBM_s013
                                 + 0.886327 w_k1_INTC_s013
                                 + 0.846804 w_k1_KO_s013
                                 + 0.915251 w_k1_MAT_s013
                                 + 0.829413 w_k1_MSFT_s013
                                 + 0.966463 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.17972 w_k1_ADM_s013
                                 + 0.697566 w_k1_AXP_s013
                                 + 1.10144 w_k1_DIS_s013
                                 + 0.918389 w_k1_FDX_s013
                                 + 0.841018 w_k1_IBM_s013
                                 + 0.966652 w_k1_INTC_s013
                                 + 1.10061 w_k1_KO_s013
                                 + 0.847976 w_k1_MAT_s013
                                 + 1.0655 w_k1_MSFT_s013
                                 + 0.877068 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.959644 w_k1_ADM_s013
                                 + 0.848086 w_k1_AXP_s013
                                 + 0.879403 w_k1_DIS_s013
                                 + 0.740837 w_k1_FDX_s013
                                 + 0.965088 w_k1_IBM_s013
                                 + 0.710744 w_k1_INTC_s013
                                 + 1.05551 w_k1_KO_s013
                                 + 0.978625 w_k1_MAT_s013
                                 + 0.732895 w_k1_MSFT_s013
                                 + 0.981201 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.14135 w_k1_ADM_s013
                                 + 0.974755 w_k1_AXP_s013
                                 + 0.855386 w_k1_DIS_s013
                                 + 0.960171 w_k1_FDX_s013
                                 + 0.953634 w_k1_IBM_s013
                                 + 0.907617 w_k1_INTC_s013
                                 + 1.03898 w_k1_KO_s013 + 1.04986 w_k1_MAT_s013
                                 + 0.874192 w_k1_MSFT_s013
                                 + 1.00397 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.03017 w_k1_ADM_s013
                                 + 1.28052 w_k1_AXP_s013
                                 + 1.15779 w_k1_DIS_s013
                                 + 1.02367 w_k1_FDX_s013
                                 + 1.67201 w_k1_IBM_s013
                                 + 0.816342 w_k1_INTC_s013
                                 + 1.42453 w_k1_KO_s013 + 2.51595 w_k1_MAT_s013
                                 + 0.928089 w_k1_MSFT_s013
                                 + 1.25661 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.908255 w_k1_ADM_s013
                                 + 0.908617 w_k1_AXP_s013
                                 + 1.35155 w_k1_DIS_s013
                                 + 1.09753 w_k1_FDX_s013
                                 + 0.791425 w_k1_IBM_s013
                                 + 1.07358 w_k1_INTC_s013
                                 + 1.24029 w_k1_KO_s013 + 1.154 w_k1_MAT_s013
                                 + 1.14663 w_k1_MSFT_s013
                                 + 1.20615 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.929307 w_k1_ADM_s014
                                 + 0.863356 w_k1_AXP_s014
                                 + 0.820878 w_k1_DIS_s014
                                 + 0.90789 w_k1_FDX_s014
                                 + 0.924327 w_k1_IBM_s014
                                 + 0.776533 w_k1_INTC_s014
                                 + 0.832861 w_k1_KO_s014
                                 + 0.803519 w_k1_MAT_s014
                                 + 0.794855 w_k1_MSFT_s014
                                 + 0.952299 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.07183 w_k1_ADM_s014
                                 + 0.814206 w_k1_AXP_s014
                                 + 0.778203 w_k1_DIS_s014
                                 + 0.929215 w_k1_FDX_s014
                                 + 0.684527 w_k1_IBM_s014
                                 + 0.925718 w_k1_INTC_s014
                                 + 0.825714 w_k1_KO_s014
                                 + 0.632501 w_k1_MAT_s014
                                 + 0.84557 w_k1_MSFT_s014
                                 + 1.02088 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.36694 w_k1_ADM_s014
                                 + 1.57741 w_k1_AXP_s014
                                 + 1.39086 w_k1_DIS_s014
                                 + 1.12997 w_k1_FDX_s014
                                 + 1.37476 w_k1_IBM_s014
                                 + 1.2594 w_k1_INTC_s014 + 1.20466 w_k1_KO_s014
                                 + 1.49143 w_k1_MAT_s014
                                 + 1.18625 w_k1_MSFT_s014
                                 + 1.13074 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.2493 w_k1_ADM_s014
                                 + 0.900092 w_k1_AXP_s014
                                 + 1.03903 w_k1_DIS_s014
                                 + 1.00024 w_k1_FDX_s014
                                 + 1.06668 w_k1_IBM_s014
                                 + 0.952306 w_k1_INTC_s014 + 1.118 w_k1_KO_s014
                                 + 0.939974 w_k1_MAT_s014
                                 + 1.05442 w_k1_MSFT_s014
                                 + 1.12732 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.06669 w_k1_ADM_s014
                                 + 0.884583 w_k1_AXP_s014
                                 + 0.591292 w_k1_DIS_s014
                                 + 0.761497 w_k1_FDX_s014
                                 + 0.794337 w_k1_IBM_s014
                                 + 0.718881 w_k1_INTC_s014
                                 + 0.782713 w_k1_KO_s014
                                 + 1.08615 w_k1_MAT_s014
                                 + 0.786704 w_k1_MSFT_s014
                                 + 0.97024 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.02455 w_k1_ADM_s014
                                 + 1.01517 w_k1_AXP_s014
                                 + 1.01994 w_k1_DIS_s014
                                 + 0.929519 w_k1_FDX_s014
                                 + 1.31705 w_k1_IBM_s014
                                 + 0.964573 w_k1_INTC_s014
                                 + 1.0634 w_k1_KO_s014 + 1.0515 w_k1_MAT_s014
                                 + 0.826111 w_k1_MSFT_s014
                                 + 0.965213 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.00247 w_k1_ADM_s014
                                 + 1.05565 w_k1_AXP_s014
                                 + 0.886262 w_k1_DIS_s014
                                 + 1.0639 w_k1_FDX_s014
                                 + 0.837458 w_k1_IBM_s014
                                 + 0.929616 w_k1_INTC_s014
                                 + 0.864214 w_k1_KO_s014
                                 + 0.91768 w_k1_MAT_s014
                                 + 0.904155 w_k1_MSFT_s014
                                 + 0.938263 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.13455 w_k1_ADM_s014
                                 + 1.16997 w_k1_AXP_s014
                                 + 0.978511 w_k1_DIS_s014
                                 + 1.15335 w_k1_FDX_s014
                                 + 0.703623 w_k1_IBM_s014
                                 + 1.0484 w_k1_INTC_s014
                                 + 0.834392 w_k1_KO_s014
                                 + 0.879153 w_k1_MAT_s014
                                 + 1.10024 w_k1_MSFT_s014
                                 + 1.08678 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.06097 w_k1_ADM_s015
                                 + 0.862247 w_k1_AXP_s015
                                 + 0.823512 w_k1_DIS_s015
                                 + 0.812178 w_k1_FDX_s015
                                 + 1.19709 w_k1_IBM_s015
                                 + 0.887911 w_k1_INTC_s015
                                 + 0.944479 w_k1_KO_s015
                                 + 1.27065 w_k1_MAT_s015
                                 + 0.928095 w_k1_MSFT_s015
                                 + 0.781052 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.729204 w_k1_ADM_s015
                                 + 0.789281 w_k1_AXP_s015
                                 + 0.715977 w_k1_DIS_s015
                                 + 0.794662 w_k1_FDX_s015
                                 + 0.742986 w_k1_IBM_s015
                                 + 1.01598 w_k1_INTC_s015
                                 + 0.778481 w_k1_KO_s015
                                 + 0.722402 w_k1_MAT_s015
                                 + 0.799098 w_k1_MSFT_s015
                                 + 0.796996 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.21907 w_k1_ADM_s015
                                 + 1.17255 w_k1_AXP_s015
                                 + 1.28129 w_k1_DIS_s015 + 1.1989 w_k1_FDX_s015
                                 + 1.21366 w_k1_IBM_s015
                                 + 1.23266 w_k1_INTC_s015 + 1.1179 w_k1_KO_s015
                                 + 1.02841 w_k1_MAT_s015
                                 + 1.08189 w_k1_MSFT_s015
                                 + 0.970639 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.11875 w_k1_ADM_s015
                                 + 1.2202 w_k1_AXP_s015 + 1.36059 w_k1_DIS_s015
                                 + 1.07009 w_k1_FDX_s015 + 1.5058 w_k1_IBM_s015
                                 + 1.39193 w_k1_INTC_s015
                                 + 1.48885 w_k1_KO_s015 + 1.55262 w_k1_MAT_s015
                                 + 1.28834 w_k1_MSFT_s015
                                 + 1.09702 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.945102 w_k1_ADM_s015
                                 + 0.798473 w_k1_AXP_s015
                                 + 0.650434 w_k1_DIS_s015
                                 + 0.895978 w_k1_FDX_s015
                                 + 0.687357 w_k1_IBM_s015
                                 + 1.04234 w_k1_INTC_s015
                                 + 0.722647 w_k1_KO_s015
                                 + 0.656408 w_k1_MAT_s015
                                 + 0.714242 w_k1_MSFT_s015
                                 + 0.675771 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.899591 w_k1_ADM_s015
                                 + 0.949108 w_k1_AXP_s015
                                 + 0.780111 w_k1_DIS_s015
                                 + 0.834863 w_k1_FDX_s015
                                 + 1.01493 w_k1_IBM_s015
                                 + 0.849266 w_k1_INTC_s015
                                 + 0.831695 w_k1_KO_s015
                                 + 0.910828 w_k1_MAT_s015
                                 + 0.71023 w_k1_MSFT_s015
                                 + 0.810404 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.33859 w_k1_ADM_s015
                                 + 1.10129 w_k1_AXP_s015
                                 + 1.09014 w_k1_DIS_s015
                                 + 1.20418 w_k1_FDX_s015
                                 + 0.984422 w_k1_IBM_s015
                                 + 1.04411 w_k1_INTC_s015
                                 + 0.863094 w_k1_KO_s015 + 1.0749 w_k1_MAT_s015
                                 + 0.879918 w_k1_MSFT_s015
                                 + 0.984466 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.15245 w_k1_ADM_s015
                                 + 0.956891 w_k1_AXP_s015
                                 + 0.949359 w_k1_DIS_s015
                                 + 1.02312 w_k1_FDX_s015
                                 + 0.911835 w_k1_IBM_s015
                                 + 0.919599 w_k1_INTC_s015
                                 + 0.885703 w_k1_KO_s015
                                 + 0.753514 w_k1_MAT_s015
                                 + 0.902301 w_k1_MSFT_s015
                                 + 0.849781 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.01029 w_k1_ADM_s016
                                 + 0.933165 w_k1_AXP_s016
                                 + 1.01464 w_k1_DIS_s016
                                 + 1.08599 w_k1_FDX_s016
                                 + 1.03113 w_k1_IBM_s016
                                 + 1.03761 w_k1_INTC_s016
                                 + 1.12237 w_k1_KO_s016 + 1.33434 w_k1_MAT_s016
                                 + 1.10051 w_k1_MSFT_s016
                                 + 1.09272 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.25382 w_k1_ADM_s016
                                 + 1.18113 w_k1_AXP_s016
                                 + 1.30263 w_k1_DIS_s016
                                 + 1.24125 w_k1_FDX_s016
                                 + 1.17448 w_k1_IBM_s016
                                 + 1.23832 w_k1_INTC_s016
                                 + 1.25926 w_k1_KO_s016 + 1.36239 w_k1_MAT_s016
                                 + 1.14559 w_k1_MSFT_s016 + 1.1899 w_k1_PG_s016
                                 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.944487 w_k1_ADM_s016
                                 + 0.647136 w_k1_AXP_s016
                                 + 0.785833 w_k1_DIS_s016
                                 + 0.892559 w_k1_FDX_s016
                                 + 0.622085 w_k1_IBM_s016
                                 + 0.868433 w_k1_INTC_s016
                                 + 0.832353 w_k1_KO_s016
                                 + 0.686864 w_k1_MAT_s016
                                 + 0.949071 w_k1_MSFT_s016
                                 + 0.820493 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.21328 w_k1_ADM_s016
                                 + 1.0017 w_k1_AXP_s016 + 1.61966 w_k1_DIS_s016
                                 + 1.30611 w_k1_FDX_s016
                                 + 0.988586 w_k1_IBM_s016
                                 + 1.05713 w_k1_INTC_s016 + 1.3466 w_k1_KO_s016
                                 + 0.818376 w_k1_MAT_s016
                                 + 1.56241 w_k1_MSFT_s016
                                 + 1.21825 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.895527 w_k1_ADM_s016
                                 + 0.815634 w_k1_AXP_s016
                                 + 0.772451 w_k1_DIS_s016
                                 + 0.832502 w_k1_FDX_s016 + 1.097 w_k1_IBM_s016
                                 + 1.06638 w_k1_INTC_s016
                                 + 1.12242 w_k1_KO_s016
                                 + 0.900758 w_k1_MAT_s016
                                 + 0.905908 w_k1_MSFT_s016
                                 + 0.880742 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.933157 w_k1_ADM_s016
                                 + 1.10636 w_k1_AXP_s016 + 1.1785 w_k1_DIS_s016
                                 + 1.17363 w_k1_FDX_s016
                                 + 0.776555 w_k1_IBM_s016
                                 + 1.11737 w_k1_INTC_s016
                                 + 1.00939 w_k1_KO_s016
                                 + 0.879243 w_k1_MAT_s016
                                 + 1.21355 w_k1_MSFT_s016
                                 + 1.03654 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.853895 w_k1_ADM_s016
                                 + 1.18867 w_k1_AXP_s016
                                 + 1.36026 w_k1_DIS_s016
                                 + 1.08823 w_k1_FDX_s016
                                 + 1.35679 w_k1_IBM_s016
                                 + 1.40345 w_k1_INTC_s016
                                 + 1.45152 w_k1_KO_s016 + 1.4496 w_k1_MAT_s016
                                 + 1.06167 w_k1_MSFT_s016
                                 + 1.14097 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.27649 w_k1_ADM_s016
                                 + 1.10859 w_k1_AXP_s016
                                 + 1.59332 w_k1_DIS_s016
                                 + 1.20513 w_k1_FDX_s016
                                 + 1.79052 w_k1_IBM_s016
                                 + 1.11559 w_k1_INTC_s016
                                 + 1.71504 w_k1_KO_s016 + 2.07479 w_k1_MAT_s016
                                 + 1.27412 w_k1_MSFT_s016
                                 + 1.16938 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.20332 w_k1_ADM_s017
                                 + 1.52334 w_k1_AXP_s017
                                 + 1.40163 w_k1_DIS_s017
                                 + 1.13359 w_k1_FDX_s017
                                 + 1.36354 w_k1_IBM_s017
                                 + 1.36956 w_k1_INTC_s017
                                 + 1.23642 w_k1_KO_s017 + 1.49343 w_k1_MAT_s017
                                 + 1.28518 w_k1_MSFT_s017 + 1.2774 w_k1_PG_s017
                                 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.01164 w_k1_ADM_s017
                                 + 0.933798 w_k1_AXP_s017
                                 + 0.607356 w_k1_DIS_s017
                                 + 0.841449 w_k1_FDX_s017
                                 + 0.820401 w_k1_IBM_s017
                                 + 1.08024 w_k1_INTC_s017
                                 + 0.752678 w_k1_KO_s017
                                 + 0.658056 w_k1_MAT_s017
                                 + 0.952268 w_k1_MSFT_s017
                                 + 0.929619 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.21532 w_k1_ADM_s017
                                 + 1.14284 w_k1_AXP_s017
                                 + 0.850378 w_k1_DIS_s017
                                 + 0.868209 w_k1_FDX_s017
                                 + 1.10164 w_k1_IBM_s017
                                 + 0.916774 w_k1_INTC_s017
                                 + 0.838829 w_k1_KO_s017
                                 + 0.78645 w_k1_MAT_s017
                                 + 0.734843 w_k1_MSFT_s017
                                 + 0.954623 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.11975 w_k1_ADM_s017
                                 + 1.13246 w_k1_AXP_s017
                                 + 1.09393 w_k1_DIS_s017
                                 + 1.09857 w_k1_FDX_s017
                                 + 1.09096 w_k1_IBM_s017
                                 + 1.07511 w_k1_INTC_s017
                                 + 0.980351 w_k1_KO_s017
                                 + 0.987626 w_k1_MAT_s017
                                 + 1.08764 w_k1_MSFT_s017
                                 + 1.18334 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.01349 w_k1_ADM_s017
                                 + 0.757965 w_k1_AXP_s017
                                 + 0.623972 w_k1_DIS_s017
                                 + 0.707865 w_k1_FDX_s017
                                 + 1.08381 w_k1_IBM_s017
                                 + 0.958658 w_k1_INTC_s017
                                 + 0.758689 w_k1_KO_s017
                                 + 0.893219 w_k1_MAT_s017
                                 + 0.820672 w_k1_MSFT_s017
                                 + 0.826512 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.06768 w_k1_ADM_s017
                                 + 0.964868 w_k1_AXP_s017
                                 + 0.957639 w_k1_DIS_s017
                                 + 1.09132 w_k1_FDX_s017
                                 + 0.796565 w_k1_IBM_s017
                                 + 1.05174 w_k1_INTC_s017
                                 + 0.807416 w_k1_KO_s017
                                 + 0.661301 w_k1_MAT_s017
                                 + 1.2286 w_k1_MSFT_s017
                                 + 0.911028 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.840891 w_k1_ADM_s017
                                 + 0.837426 w_k1_AXP_s017
                                 + 0.620341 w_k1_DIS_s017
                                 + 0.762544 w_k1_FDX_s017
                                 + 0.685072 w_k1_IBM_s017
                                 + 1.14845 w_k1_INTC_s017
                                 + 0.676691 w_k1_KO_s017
                                 + 0.561948 w_k1_MAT_s017
                                 + 0.899771 w_k1_MSFT_s017
                                 + 0.886554 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.854441 w_k1_ADM_s017
                                 + 0.630168 w_k1_AXP_s017
                                 + 0.904276 w_k1_DIS_s017
                                 + 0.779651 w_k1_FDX_s017
                                 + 1.08111 w_k1_IBM_s017
                                 + 0.825454 w_k1_INTC_s017
                                 + 1.01333 w_k1_KO_s017 + 1.05061 w_k1_MAT_s017
                                 + 0.923964 w_k1_MSFT_s017
                                 + 1.07212 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.953478 w_k1_ADM_s100
                                 + 0.806769 w_k1_AXP_s100
                                 + 1.08123 w_k1_DIS_s100
                                 + 0.981953 w_k1_FDX_s100
                                 + 0.972984 w_k1_IBM_s100
                                 + 0.717275 w_k1_INTC_s100
                                 + 0.964139 w_k1_KO_s100
                                 + 1.21036 w_k1_MAT_s100
                                 + 0.870079 w_k1_MSFT_s100
                                 + 0.982748 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.900496 w_k1_ADM_s100
                                 + 0.949272 w_k1_AXP_s100
                                 + 0.80313 w_k1_DIS_s100
                                 + 0.845815 w_k1_FDX_s100
                                 + 1.11429 w_k1_IBM_s100
                                 + 0.765592 w_k1_INTC_s100
                                 + 0.912165 w_k1_KO_s100
                                 + 1.35616 w_k1_MAT_s100
                                 + 0.752179 w_k1_MSFT_s100
                                 + 0.918708 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.42531 w_k1_ADM_s100
                                 + 1.22911 w_k1_AXP_s100
                                 + 1.13029 w_k1_DIS_s100
                                 + 1.12503 w_k1_FDX_s100
                                 + 1.48004 w_k1_IBM_s100
                                 + 0.974734 w_k1_INTC_s100
                                 + 1.20414 w_k1_KO_s100 + 1.39476 w_k1_MAT_s100
                                 + 1.00843 w_k1_MSFT_s100
                                 + 1.21721 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.29334 w_k1_ADM_s100
                                 + 1.22267 w_k1_AXP_s100
                                 + 0.974254 w_k1_DIS_s100
                                 + 1.05954 w_k1_FDX_s100
                                 + 1.00587 w_k1_IBM_s100
                                 + 0.890832 w_k1_INTC_s100
                                 + 0.874444 w_k1_KO_s100
                                 + 1.03016 w_k1_MAT_s100
                                 + 1.00535 w_k1_MSFT_s100
                                 + 1.05176 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.876609 w_k1_ADM_s100
                                 + 0.609032 w_k1_AXP_s100
                                 + 0.687227 w_k1_DIS_s100
                                 + 0.728127 w_k1_FDX_s100
                                 + 0.800562 w_k1_IBM_s100
                                 + 0.677569 w_k1_INTC_s100
                                 + 0.790002 w_k1_KO_s100 + 0.917 w_k1_MAT_s100
                                 + 0.714449 w_k1_MSFT_s100
                                 + 0.897022 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.33197 w_k1_ADM_s100
                                 + 1.17325 w_k1_AXP_s100
                                 + 1.04079 w_k1_DIS_s100
                                 + 1.19783 w_k1_FDX_s100
                                 + 0.830243 w_k1_IBM_s100
                                 + 1.01794 w_k1_INTC_s100
                                 + 0.935122 w_k1_KO_s100
                                 + 0.876683 w_k1_MAT_s100
                                 + 1.00416 w_k1_MSFT_s100
                                 + 1.05315 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.08271 w_k1_ADM_s100
                                 + 0.781318 w_k1_AXP_s100
                                 + 0.61467 w_k1_DIS_s100
                                 + 0.866916 w_k1_FDX_s100
                                 + 0.695937 w_k1_IBM_s100
                                 + 0.70422 w_k1_INTC_s100
                                 + 0.702214 w_k1_KO_s100
                                 + 0.931091 w_k1_MAT_s100
                                 + 0.788791 w_k1_MSFT_s100
                                 + 0.8707 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.27251 w_k1_ADM_s100
                                 + 1.11979 w_k1_AXP_s100
                                 + 1.31613 w_k1_DIS_s100
                                 + 1.21987 w_k1_FDX_s100 + 1.2097 w_k1_IBM_s100
                                 + 1.04665 w_k1_INTC_s100
                                 + 1.23117 w_k1_KO_s100 + 1.37996 w_k1_MAT_s100
                                 + 1.19478 w_k1_MSFT_s100 + 1.1831 w_k1_PG_s100
                                 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.712756 w_k1_ADM_s101
                                 + 1.05041 w_k1_AXP_s101
                                 + 0.937008 w_k1_DIS_s101
                                 + 0.89558 w_k1_FDX_s101
                                 + 1.90579 w_k1_IBM_s101
                                 + 0.979249 w_k1_INTC_s101
                                 + 1.11149 w_k1_KO_s101 + 1.22364 w_k1_MAT_s101
                                 + 0.944653 w_k1_MSFT_s101
                                 + 1.05467 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.31475 w_k1_ADM_s101
                                 + 1.37023 w_k1_AXP_s101
                                 + 1.72559 w_k1_DIS_s101 + 1.3384 w_k1_FDX_s101
                                 + 1.2599 w_k1_IBM_s101
                                 + 1.17809 w_k1_INTC_s101
                                 + 1.38826 w_k1_KO_s101 + 1.33646 w_k1_MAT_s101
                                 + 1.37428 w_k1_MSFT_s101
                                 + 1.16399 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.013 w_k1_ADM_s101
                                 + 0.82947 w_k1_AXP_s101
                                 + 1.13719 w_k1_DIS_s101
                                 + 1.01932 w_k1_FDX_s101
                                 + 1.39002 w_k1_IBM_s101
                                 + 1.05234 w_k1_INTC_s101
                                 + 1.26395 w_k1_KO_s101 + 1.03389 w_k1_MAT_s101
                                 + 1.07715 w_k1_MSFT_s101
                                 + 1.06326 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.05937 w_k1_ADM_s101
                                 + 1.03264 w_k1_AXP_s101 + 1.2624 w_k1_DIS_s101
                                 + 1.14445 w_k1_FDX_s101
                                 + 1.01848 w_k1_IBM_s101
                                 + 1.20301 w_k1_INTC_s101
                                 + 1.12465 w_k1_KO_s101
                                 + 0.709649 w_k1_MAT_s101
                                 + 1.16074 w_k1_MSFT_s101
                                 + 1.07322 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.850375 w_k1_ADM_s101
                                 + 0.669806 w_k1_AXP_s101
                                 + 0.668249 w_k1_DIS_s101
                                 + 0.819036 w_k1_FDX_s101
                                 + 0.961012 w_k1_IBM_s101
                                 + 0.837379 w_k1_INTC_s101
                                 + 0.792434 w_k1_KO_s101
                                 + 0.491728 w_k1_MAT_s101
                                 + 0.777588 w_k1_MSFT_s101
                                 + 0.891827 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.30166 w_k1_ADM_s101
                                 + 1.42917 w_k1_AXP_s101
                                 + 0.979206 w_k1_DIS_s101
                                 + 1.27891 w_k1_FDX_s101
                                 + 0.688536 w_k1_IBM_s101
                                 + 1.21624 w_k1_INTC_s101
                                 + 0.841386 w_k1_KO_s101
                                 + 0.557631 w_k1_MAT_s101
                                 + 1.0999 w_k1_MSFT_s101 + 1.05355 w_k1_PG_s101
                                 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.727309 w_k1_ADM_s101
                                 + 0.827921 w_k1_AXP_s101
                                 + 0.892311 w_k1_DIS_s101
                                 + 1.08113 w_k1_FDX_s101
                                 + 0.683353 w_k1_IBM_s101
                                 + 0.993071 w_k1_INTC_s101
                                 + 0.918326 w_k1_KO_s101
                                 + 0.734254 w_k1_MAT_s101
                                 + 1.04988 w_k1_MSFT_s101
                                 + 0.982184 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.08971 w_k1_ADM_s101
                                 + 1.0882 w_k1_AXP_s101
                                 + 0.810898 w_k1_DIS_s101
                                 + 1.05835 w_k1_FDX_s101
                                 + 1.01578 w_k1_IBM_s101
                                 + 0.910087 w_k1_INTC_s101
                                 + 0.849834 w_k1_KO_s101
                                 + 0.931712 w_k1_MAT_s101
                                 + 0.871777 w_k1_MSFT_s101
                                 + 0.996434 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.799771 w_k1_ADM_s102
                                 + 0.625906 w_k1_AXP_s102
                                 + 0.624096 w_k1_DIS_s102
                                 + 0.864832 w_k1_FDX_s102
                                 + 0.524446 w_k1_IBM_s102
                                 + 0.881273 w_k1_INTC_s102
                                 + 0.7114 w_k1_KO_s102 + 0.494262 w_k1_MAT_s102
                                 + 0.868098 w_k1_MSFT_s102
                                 + 0.787519 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.690422 w_k1_ADM_s102
                                 + 0.600517 w_k1_AXP_s102
                                 + 0.623694 w_k1_DIS_s102
                                 + 0.722021 w_k1_FDX_s102
                                 + 0.908571 w_k1_IBM_s102
                                 + 0.790073 w_k1_INTC_s102
                                 + 0.770823 w_k1_KO_s102
                                 + 0.847433 w_k1_MAT_s102
                                 + 0.832891 w_k1_MSFT_s102
                                 + 0.877348 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.868866 w_k1_ADM_s102
                                 + 0.815295 w_k1_AXP_s102
                                 + 0.862401 w_k1_DIS_s102
                                 + 0.889605 w_k1_FDX_s102
                                 + 0.775175 w_k1_IBM_s102
                                 + 0.921538 w_k1_INTC_s102
                                 + 0.84413 w_k1_KO_s102
                                 + 0.801578 w_k1_MAT_s102
                                 + 0.936744 w_k1_MSFT_s102
                                 + 0.940311 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.11936 w_k1_ADM_s102
                                 + 1.11354 w_k1_AXP_s102
                                 + 1.57143 w_k1_DIS_s102
                                 + 1.21991 w_k1_FDX_s102
                                 + 1.02441 w_k1_IBM_s102
                                 + 1.26968 w_k1_INTC_s102
                                 + 1.28885 w_k1_KO_s102 + 1.27698 w_k1_MAT_s102
                                 + 1.47517 w_k1_MSFT_s102
                                 + 1.13179 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.18223 w_k1_ADM_s102
                                 + 1.20001 w_k1_AXP_s102
                                 + 1.13856 w_k1_DIS_s102
                                 + 1.16399 w_k1_FDX_s102
                                 + 0.743188 w_k1_IBM_s102
                                 + 1.16701 w_k1_INTC_s102
                                 + 0.994567 w_k1_KO_s102
                                 + 0.715385 w_k1_MAT_s102
                                 + 1.28207 w_k1_MSFT_s102
                                 + 1.01565 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.17352 w_k1_ADM_s102
                                 + 1.58715 w_k1_AXP_s102
                                 + 1.78814 w_k1_DIS_s102
                                 + 1.38732 w_k1_FDX_s102
                                 + 1.18304 w_k1_IBM_s102
                                 + 1.10569 w_k1_INTC_s102
                                 + 1.34435 w_k1_KO_s102 + 1.40092 w_k1_MAT_s102
                                 + 1.21095 w_k1_MSFT_s102
                                 + 1.21388 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.899599 w_k1_ADM_s102
                                 + 0.847363 w_k1_AXP_s102
                                 + 0.917248 w_k1_DIS_s102
                                 + 0.851858 w_k1_FDX_s102
                                 + 1.12737 w_k1_IBM_s102
                                 + 0.861068 w_k1_INTC_s102
                                 + 1.07238 w_k1_KO_s102 + 1.16147 w_k1_MAT_s102
                                 + 0.895058 w_k1_MSFT_s102
                                 + 1.07503 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.868527 w_k1_ADM_s102
                                 + 1.20416 w_k1_AXP_s102
                                 + 1.32627 w_k1_DIS_s102
                                 + 1.02722 w_k1_FDX_s102
                                 + 1.58342 w_k1_IBM_s102
                                 + 1.1788 w_k1_INTC_s102 + 1.35251 w_k1_KO_s102
                                 + 1.72678 w_k1_MAT_s102
                                 + 1.14029 w_k1_MSFT_s102
                                 + 1.00631 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.47863 w_k1_ADM_s103
                                 + 2.13684 w_k1_AXP_s103
                                 + 1.89422 w_k1_DIS_s103
                                 + 1.78289 w_k1_FDX_s103
                                 + 1.20653 w_k1_IBM_s103
                                 + 1.08985 w_k1_INTC_s103 + 1.4321 w_k1_KO_s103
                                 + 1.28243 w_k1_MAT_s103
                                 + 1.19278 w_k1_MSFT_s103
                                 + 1.51241 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.765135 w_k1_ADM_s103
                                 + 1.28419 w_k1_AXP_s103
                                 + 0.939724 w_k1_DIS_s103
                                 + 0.983839 w_k1_FDX_s103
                                 + 1.53771 w_k1_IBM_s103
                                 + 0.902542 w_k1_INTC_s103
                                 + 1.28227 w_k1_KO_s103 + 2.50785 w_k1_MAT_s103
                                 + 1.0023 w_k1_MSFT_s103 + 1.0873 w_k1_PG_s103
                                 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.999022 w_k1_ADM_s103
                                 + 1.16444 w_k1_AXP_s103
                                 + 1.09887 w_k1_DIS_s103
                                 + 1.21059 w_k1_FDX_s103
                                 + 0.77158 w_k1_IBM_s103
                                 + 1.0639 w_k1_INTC_s103
                                 + 0.998531 w_k1_KO_s103
                                 + 0.846269 w_k1_MAT_s103
                                 + 0.959244 w_k1_MSFT_s103
                                 + 1.22343 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.789597 w_k1_ADM_s103
                                 + 0.803455 w_k1_AXP_s103
                                 + 0.961788 w_k1_DIS_s103
                                 + 1.0273 w_k1_FDX_s103
                                 + 0.797915 w_k1_IBM_s103
                                 + 0.865644 w_k1_INTC_s103
                                 + 0.956216 w_k1_KO_s103
                                 + 0.928736 w_k1_MAT_s103
                                 + 1.05756 w_k1_MSFT_s103
                                 + 0.986025 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.13676 w_k1_ADM_s103
                                 + 1.32487 w_k1_AXP_s103
                                 + 1.19906 w_k1_DIS_s103
                                 + 1.28162 w_k1_FDX_s103
                                 + 1.23705 w_k1_IBM_s103
                                 + 1.1877 w_k1_INTC_s103 + 1.20303 w_k1_KO_s103
                                 + 0.929942 w_k1_MAT_s103
                                 + 1.04038 w_k1_MSFT_s103
                                 + 1.26823 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.16866 w_k1_ADM_s103
                                 + 1.72586 w_k1_AXP_s103
                                 + 1.02609 w_k1_DIS_s103 + 1.2424 w_k1_FDX_s103
                                 + 0.97899 w_k1_IBM_s103
                                 + 0.92074 w_k1_INTC_s103
                                 + 0.992812 w_k1_KO_s103
                                 + 1.46412 w_k1_MAT_s103
                                 + 0.918893 w_k1_MSFT_s103
                                 + 1.1482 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.83133 w_k1_ADM_s103
                                 + 0.68593 w_k1_AXP_s103
                                 + 0.668667 w_k1_DIS_s103
                                 + 0.75045 w_k1_FDX_s103
                                 + 0.788469 w_k1_IBM_s103
                                 + 0.625634 w_k1_INTC_s103
                                 + 0.726315 w_k1_KO_s103
                                 + 0.784991 w_k1_MAT_s103
                                 + 0.703651 w_k1_MSFT_s103
                                 + 0.807161 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.845291 w_k1_ADM_s103
                                 + 0.925634 w_k1_AXP_s103
                                 + 0.710417 w_k1_DIS_s103
                                 + 0.810889 w_k1_FDX_s103
                                 + 1.12475 w_k1_IBM_s103
                                 + 0.709353 w_k1_INTC_s103
                                 + 0.862515 w_k1_KO_s103
                                 + 0.771413 w_k1_MAT_s103
                                 + 0.781189 w_k1_MSFT_s103
                                 + 0.967072 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.899485 w_k1_ADM_s104
                                 + 0.680162 w_k1_AXP_s104
                                 + 0.826282 w_k1_DIS_s104
                                 + 0.863828 w_k1_FDX_s104
                                 + 0.872438 w_k1_IBM_s104
                                 + 0.963686 w_k1_INTC_s104
                                 + 0.965436 w_k1_KO_s104
                                 + 0.848665 w_k1_MAT_s104
                                 + 0.919091 w_k1_MSFT_s104
                                 + 0.831764 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.964322 w_k1_ADM_s104
                                 + 1.29453 w_k1_AXP_s104
                                 + 1.17581 w_k1_DIS_s104
                                 + 1.01447 w_k1_FDX_s104
                                 + 1.69465 w_k1_IBM_s104
                                 + 1.17598 w_k1_INTC_s104
                                 + 1.21231 w_k1_KO_s104 + 1.63803 w_k1_MAT_s104
                                 + 0.999488 w_k1_MSFT_s104
                                 + 0.974622 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.17269 w_k1_ADM_s104
                                 + 1.22771 w_k1_AXP_s104
                                 + 1.18539 w_k1_DIS_s104
                                 + 1.19978 w_k1_FDX_s104
                                 + 0.86224 w_k1_IBM_s104
                                 + 1.50438 w_k1_INTC_s104
                                 + 1.15672 w_k1_KO_s104 + 1.21463 w_k1_MAT_s104
                                 + 1.25937 w_k1_MSFT_s104
                                 + 1.04603 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.903076 w_k1_ADM_s104
                                 + 1.2587 w_k1_AXP_s104
                                 + 0.756032 w_k1_DIS_s104
                                 + 0.971209 w_k1_FDX_s104
                                 + 0.976981 w_k1_IBM_s104
                                 + 1.12567 w_k1_INTC_s104
                                 + 0.851922 w_k1_KO_s104
                                 + 0.939342 w_k1_MAT_s104
                                 + 0.904149 w_k1_MSFT_s104
                                 + 0.87435 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.32156 w_k1_ADM_s104
                                 + 1.85766 w_k1_AXP_s104 + 1.5631 w_k1_DIS_s104
                                 + 1.43571 w_k1_FDX_s104
                                 + 1.26157 w_k1_IBM_s104
                                 + 1.61542 w_k1_INTC_s104
                                 + 1.34614 w_k1_KO_s104 + 1.35298 w_k1_MAT_s104
                                 + 1.36947 w_k1_MSFT_s104
                                 + 1.07881 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.00685 w_k1_ADM_s104
                                 + 1.22843 w_k1_AXP_s104
                                 + 1.07117 w_k1_DIS_s104
                                 + 0.943764 w_k1_FDX_s104
                                 + 1.20385 w_k1_IBM_s104
                                 + 1.27612 w_k1_INTC_s104
                                 + 1.27199 w_k1_KO_s104 + 1.21014 w_k1_MAT_s104
                                 + 1.00281 w_k1_MSFT_s104
                                 + 1.03303 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.07564 w_k1_ADM_s104
                                 + 1.36522 w_k1_AXP_s104 + 1.0865 w_k1_DIS_s104
                                 + 1.12179 w_k1_FDX_s104
                                 + 0.911621 w_k1_IBM_s104
                                 + 1.34711 w_k1_INTC_s104
                                 + 1.06912 w_k1_KO_s104
                                 + 0.746445 w_k1_MAT_s104
                                 + 1.13776 w_k1_MSFT_s104 + 1.026 w_k1_PG_s104
                                 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.931956 w_k1_ADM_s104
                                 + 0.989421 w_k1_AXP_s104
                                 + 0.719681 w_k1_DIS_s104
                                 + 0.98359 w_k1_FDX_s104
                                 + 0.739218 w_k1_IBM_s104
                                 + 1.12223 w_k1_INTC_s104
                                 + 0.839415 w_k1_KO_s104
                                 + 0.855436 w_k1_MAT_s104
                                 + 0.918089 w_k1_MSFT_s104
                                 + 0.87108 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.982814 w_k1_ADM_s105
                                 + 0.923657 w_k1_AXP_s105
                                 + 0.878107 w_k1_DIS_s105
                                 + 0.964142 w_k1_FDX_s105
                                 + 1.07746 w_k1_IBM_s105
                                 + 0.917636 w_k1_INTC_s105
                                 + 0.894514 w_k1_KO_s105
                                 + 0.998226 w_k1_MAT_s105
                                 + 1.01274 w_k1_MSFT_s105
                                 + 1.00863 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.1309 w_k1_ADM_s105
                                 + 1.40897 w_k1_AXP_s105
                                 + 1.05221 w_k1_DIS_s105
                                 + 1.27549 w_k1_FDX_s105
                                 + 0.988041 w_k1_IBM_s105
                                 + 0.875276 w_k1_INTC_s105
                                 + 0.946732 w_k1_KO_s105
                                 + 1.29422 w_k1_MAT_s105
                                 + 1.04958 w_k1_MSFT_s105
                                 + 0.996741 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.28194 w_k1_ADM_s105
                                 + 1.52931 w_k1_AXP_s105
                                 + 1.32506 w_k1_DIS_s105 + 1.2259 w_k1_FDX_s105
                                 + 1.2758 w_k1_IBM_s105
                                 + 0.930269 w_k1_INTC_s105
                                 + 1.23638 w_k1_KO_s105 + 1.89283 w_k1_MAT_s105
                                 + 1.09151 w_k1_MSFT_s105
                                 + 1.19759 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.14623 w_k1_ADM_s105
                                 + 0.972816 w_k1_AXP_s105
                                 + 0.96268 w_k1_DIS_s105
                                 + 1.07802 w_k1_FDX_s105
                                 + 0.738046 w_k1_IBM_s105
                                 + 0.73042 w_k1_INTC_s105
                                 + 0.962226 w_k1_KO_s105
                                 + 0.966829 w_k1_MAT_s105
                                 + 1.00923 w_k1_MSFT_s105
                                 + 0.944444 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.944435 w_k1_ADM_s105
                                 + 1.01312 w_k1_AXP_s105
                                 + 1.03062 w_k1_DIS_s105
                                 + 0.909094 w_k1_FDX_s105
                                 + 1.60151 w_k1_IBM_s105
                                 + 0.820133 w_k1_INTC_s105
                                 + 1.04355 w_k1_KO_s105 + 1.38791 w_k1_MAT_s105
                                 + 0.889306 w_k1_MSFT_s105
                                 + 1.10856 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.839813 w_k1_ADM_s105
                                 + 0.896876 w_k1_AXP_s105
                                 + 0.702887 w_k1_DIS_s105
                                 + 0.776844 w_k1_FDX_s105
                                 + 0.865763 w_k1_IBM_s105
                                 + 0.774022 w_k1_INTC_s105
                                 + 0.827462 w_k1_KO_s105
                                 + 0.830456 w_k1_MAT_s105
                                 + 0.778159 w_k1_MSFT_s105
                                 + 0.880592 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.16072 w_k1_ADM_s105
                                 + 1.16271 w_k1_AXP_s105 + 1.0974 w_k1_DIS_s105
                                 + 1.0478 w_k1_FDX_s105 + 1.09799 w_k1_IBM_s105
                                 + 0.785959 w_k1_INTC_s105
                                 + 1.12822 w_k1_KO_s105 + 1.48911 w_k1_MAT_s105
                                 + 0.949834 w_k1_MSFT_s105
                                 + 1.01043 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.24096 w_k1_ADM_s105
                                 + 1.369 w_k1_AXP_s105 + 1.70004 w_k1_DIS_s105
                                 + 1.3238 w_k1_FDX_s105 + 1.26046 w_k1_IBM_s105
                                 + 1.09851 w_k1_INTC_s105
                                 + 1.49136 w_k1_KO_s105 + 1.75241 w_k1_MAT_s105
                                 + 1.39399 w_k1_MSFT_s105
                                 + 1.30079 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.30809 w_k1_ADM_s106
                                 + 1.10832 w_k1_AXP_s106
                                 + 1.20355 w_k1_DIS_s106
                                 + 1.15652 w_k1_FDX_s106
                                 + 0.845138 w_k1_IBM_s106
                                 + 0.968636 w_k1_INTC_s106
                                 + 1.06727 w_k1_KO_s106 + 1.21952 w_k1_MAT_s106
                                 + 0.999171 w_k1_MSFT_s106
                                 + 1.1128 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.1317 w_k1_ADM_s106
                                 + 0.986828 w_k1_AXP_s106
                                 + 1.05552 w_k1_DIS_s106
                                 + 1.04321 w_k1_FDX_s106
                                 + 1.03675 w_k1_IBM_s106
                                 + 1.06793 w_k1_INTC_s106
                                 + 1.21129 w_k1_KO_s106 + 1.02328 w_k1_MAT_s106
                                 + 1.09963 w_k1_MSFT_s106 + 1.0356 w_k1_PG_s106
                                 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.05811 w_k1_ADM_s106
                                 + 0.664729 w_k1_AXP_s106
                                 + 0.867956 w_k1_DIS_s106
                                 + 0.822529 w_k1_FDX_s106
                                 + 1.14555 w_k1_IBM_s106
                                 + 0.834988 w_k1_INTC_s106
                                 + 1.14434 w_k1_KO_s106 + 1.07063 w_k1_MAT_s106
                                 + 0.880742 w_k1_MSFT_s106
                                 + 0.852385 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.64932 w_k1_ADM_s106
                                 + 1.6048 w_k1_AXP_s106 + 1.16756 w_k1_DIS_s106
                                 + 1.07661 w_k1_FDX_s106
                                 + 1.64878 w_k1_IBM_s106
                                 + 1.10594 w_k1_INTC_s106 + 1.4597 w_k1_KO_s106
                                 + 2.64094 w_k1_MAT_s106
                                 + 1.01522 w_k1_MSFT_s106 + 1.2599 w_k1_PG_s106
                                 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.896653 w_k1_ADM_s106
                                 + 0.695366 w_k1_AXP_s106
                                 + 0.532152 w_k1_DIS_s106
                                 + 0.709584 w_k1_FDX_s106
                                 + 0.737661 w_k1_IBM_s106
                                 + 0.729405 w_k1_INTC_s106
                                 + 0.777507 w_k1_KO_s106
                                 + 1.05105 w_k1_MAT_s106
                                 + 0.751163 w_k1_MSFT_s106
                                 + 0.794995 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.08708 w_k1_ADM_s106
                                 + 1.00304 w_k1_AXP_s106
                                 + 0.906486 w_k1_DIS_s106
                                 + 0.962171 w_k1_FDX_s106
                                 + 0.818893 w_k1_IBM_s106
                                 + 0.899035 w_k1_INTC_s106
                                 + 0.922205 w_k1_KO_s106
                                 + 1.28103 w_k1_MAT_s106
                                 + 0.948112 w_k1_MSFT_s106
                                 + 0.968427 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.54153 w_k1_ADM_s106
                                 + 1.16859 w_k1_AXP_s106
                                 + 1.40389 w_k1_DIS_s106
                                 + 1.13969 w_k1_FDX_s106
                                 + 1.42169 w_k1_IBM_s106
                                 + 0.8176 w_k1_INTC_s106 + 1.4109 w_k1_KO_s106
                                 + 1.85415 w_k1_MAT_s106
                                 + 1.10857 w_k1_MSFT_s106
                                 + 1.13496 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.36612 w_k1_ADM_s106
                                 + 1.44639 w_k1_AXP_s106
                                 + 1.50808 w_k1_DIS_s106
                                 + 1.39513 w_k1_FDX_s106
                                 + 1.15154 w_k1_IBM_s106
                                 + 1.25695 w_k1_INTC_s106
                                 + 1.48429 w_k1_KO_s106 + 1.876 w_k1_MAT_s106
                                 + 1.47606 w_k1_MSFT_s106
                                 + 1.27007 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.960838 w_k1_ADM_s107
                                 + 0.967564 w_k1_AXP_s107
                                 + 1.21647 w_k1_DIS_s107
                                 + 0.948755 w_k1_FDX_s107
                                 + 1.63382 w_k1_IBM_s107
                                 + 1.02415 w_k1_INTC_s107
                                 + 1.31266 w_k1_KO_s107 + 1.33137 w_k1_MAT_s107
                                 + 0.97518 w_k1_MSFT_s107
                                 + 1.07015 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.0545 w_k1_ADM_s107
                                 + 1.08818 w_k1_AXP_s107
                                 + 1.02362 w_k1_DIS_s107 + 1.1545 w_k1_FDX_s107
                                 + 0.841937 w_k1_IBM_s107
                                 + 1.0635 w_k1_INTC_s107 + 1.07333 w_k1_KO_s107
                                 + 0.838158 w_k1_MAT_s107
                                 + 1.04919 w_k1_MSFT_s107
                                 + 1.07748 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.37211 w_k1_ADM_s107
                                 + 1.23658 w_k1_AXP_s107
                                 + 1.51305 w_k1_DIS_s107
                                 + 1.27609 w_k1_FDX_s107
                                 + 1.39931 w_k1_IBM_s107
                                 + 1.15347 w_k1_INTC_s107
                                 + 1.38046 w_k1_KO_s107
                                 + 0.867871 w_k1_MAT_s107
                                 + 1.02475 w_k1_MSFT_s107
                                 + 1.16957 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.00698 w_k1_ADM_s107
                                 + 0.907521 w_k1_AXP_s107 + 1.073 w_k1_DIS_s107
                                 + 0.853203 w_k1_FDX_s107
                                 + 1.08739 w_k1_IBM_s107
                                 + 0.958713 w_k1_INTC_s107
                                 + 1.19334 w_k1_KO_s107 + 1.05213 w_k1_MAT_s107
                                 + 0.932562 w_k1_MSFT_s107
                                 + 0.991312 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.08598 w_k1_ADM_s107
                                 + 0.891674 w_k1_AXP_s107
                                 + 1.06637 w_k1_DIS_s107
                                 + 0.993845 w_k1_FDX_s107
                                 + 0.707645 w_k1_IBM_s107
                                 + 0.989996 w_k1_INTC_s107
                                 + 0.908809 w_k1_KO_s107
                                 + 0.820943 w_k1_MAT_s107
                                 + 0.912018 w_k1_MSFT_s107
                                 + 1.08206 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.748363 w_k1_ADM_s107
                                 + 0.731788 w_k1_AXP_s107
                                 + 0.816468 w_k1_DIS_s107
                                 + 0.801934 w_k1_FDX_s107
                                 + 0.661939 w_k1_IBM_s107
                                 + 0.871766 w_k1_INTC_s107
                                 + 0.785796 w_k1_KO_s107
                                 + 0.459153 w_k1_MAT_s107
                                 + 0.783677 w_k1_MSFT_s107
                                 + 0.803727 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.914854 w_k1_ADM_s107
                                 + 0.981179 w_k1_AXP_s107
                                 + 0.796074 w_k1_DIS_s107
                                 + 0.918981 w_k1_FDX_s107
                                 + 0.939273 w_k1_IBM_s107
                                 + 1.04604 w_k1_INTC_s107
                                 + 0.855286 w_k1_KO_s107 + 1.0758 w_k1_MAT_s107
                                 + 0.927079 w_k1_MSFT_s107
                                 + 0.991216 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.751191 w_k1_ADM_s107
                                 + 0.657669 w_k1_AXP_s107
                                 + 0.700954 w_k1_DIS_s107
                                 + 0.726045 w_k1_FDX_s107
                                 + 0.885318 w_k1_IBM_s107
                                 + 0.854014 w_k1_INTC_s107
                                 + 0.864478 w_k1_KO_s107
                                 + 0.972362 w_k1_MAT_s107
                                 + 0.735381 w_k1_MSFT_s107
                                 + 0.852519 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.774749 w_k1_ADM_s110
                                 + 0.72861 w_k1_AXP_s110
                                 + 0.618893 w_k1_DIS_s110
                                 + 0.728236 w_k1_FDX_s110
                                 + 0.736456 w_k1_IBM_s110
                                 + 0.781628 w_k1_INTC_s110
                                 + 0.658024 w_k1_KO_s110
                                 + 0.59511 w_k1_MAT_s110
                                 + 0.743194 w_k1_MSFT_s110
                                 + 0.739663 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.989994 w_k1_ADM_s110
                                 + 0.952465 w_k1_AXP_s110
                                 + 0.884588 w_k1_DIS_s110
                                 + 0.970358 w_k1_FDX_s110
                                 + 0.710517 w_k1_IBM_s110
                                 + 1.0966 w_k1_INTC_s110
                                 + 0.876537 w_k1_KO_s110
                                 + 0.756593 w_k1_MAT_s110
                                 + 0.911356 w_k1_MSFT_s110
                                 + 0.975031 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.852307 w_k1_ADM_s110
                                 + 0.926078 w_k1_AXP_s110
                                 + 0.716274 w_k1_DIS_s110
                                 + 0.895138 w_k1_FDX_s110
                                 + 1.07245 w_k1_IBM_s110
                                 + 1.05812 w_k1_INTC_s110
                                 + 0.926557 w_k1_KO_s110
                                 + 1.18222 w_k1_MAT_s110
                                 + 0.833006 w_k1_MSFT_s110
                                 + 0.98786 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.36546 w_k1_ADM_s110
                                 + 1.69812 w_k1_AXP_s110
                                 + 1.40008 w_k1_DIS_s110
                                 + 1.50955 w_k1_FDX_s110
                                 + 0.983591 w_k1_IBM_s110
                                 + 1.2184 w_k1_INTC_s110 + 1.27348 w_k1_KO_s110
                                 + 1.83134 w_k1_MAT_s110
                                 + 1.29038 w_k1_MSFT_s110 + 1.2532 w_k1_PG_s110
                                 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.936546 w_k1_ADM_s110
                                 + 1.13094 w_k1_AXP_s110
                                 + 1.35876 w_k1_DIS_s110
                                 + 1.21064 w_k1_FDX_s110
                                 + 1.22047 w_k1_IBM_s110
                                 + 1.21872 w_k1_INTC_s110
                                 + 1.30375 w_k1_KO_s110 + 1.52489 w_k1_MAT_s110
                                 + 1.24087 w_k1_MSFT_s110
                                 + 1.15376 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.994546 w_k1_ADM_s110
                                 + 1.07074 w_k1_AXP_s110
                                 + 0.836037 w_k1_DIS_s110
                                 + 0.8576 w_k1_FDX_s110 + 1.37544 w_k1_IBM_s110
                                 + 1.00529 w_k1_INTC_s110
                                 + 0.97029 w_k1_KO_s110 + 1.07654 w_k1_MAT_s110
                                 + 0.828276 w_k1_MSFT_s110
                                 + 0.988903 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.94198 w_k1_ADM_s110
                                 + 0.749555 w_k1_AXP_s110
                                 + 0.727056 w_k1_DIS_s110
                                 + 0.780295 w_k1_FDX_s110
                                 + 0.990439 w_k1_IBM_s110
                                 + 0.792345 w_k1_INTC_s110
                                 + 0.921613 w_k1_KO_s110
                                 + 0.797703 w_k1_MAT_s110
                                 + 0.769897 w_k1_MSFT_s110
                                 + 0.943916 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.93409 w_k1_ADM_s110
                                 + 1.07436 w_k1_AXP_s110
                                 + 1.05767 w_k1_DIS_s110
                                 + 0.995199 w_k1_FDX_s110
                                 + 1.14668 w_k1_IBM_s110
                                 + 0.980573 w_k1_INTC_s110
                                 + 1.04322 w_k1_KO_s110 + 1.45742 w_k1_MAT_s110
                                 + 1.00636 w_k1_MSFT_s110
                                 + 1.09832 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.0721 w_k1_ADM_s111
                                 + 0.843308 w_k1_AXP_s111
                                 + 1.0367 w_k1_DIS_s111 + 1.13232 w_k1_FDX_s111
                                 + 0.701369 w_k1_IBM_s111
                                 + 0.983788 w_k1_INTC_s111
                                 + 0.935122 w_k1_KO_s111
                                 + 0.750453 w_k1_MAT_s111
                                 + 1.05373 w_k1_MSFT_s111
                                 + 0.960656 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.82759 w_k1_ADM_s111
                                 + 0.801854 w_k1_AXP_s111
                                 + 0.663472 w_k1_DIS_s111
                                 + 0.850463 w_k1_FDX_s111
                                 + 0.640416 w_k1_IBM_s111
                                 + 0.792949 w_k1_INTC_s111
                                 + 0.730981 w_k1_KO_s111
                                 + 0.644143 w_k1_MAT_s111
                                 + 0.779299 w_k1_MSFT_s111
                                 + 0.808233 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.12766 w_k1_ADM_s111
                                 + 1.3412 w_k1_AXP_s111 + 1.14782 w_k1_DIS_s111
                                 + 1.21703 w_k1_FDX_s111
                                 + 0.997798 w_k1_IBM_s111
                                 + 0.985285 w_k1_INTC_s111
                                 + 1.14164 w_k1_KO_s111 + 1.2326 w_k1_MAT_s111
                                 + 0.997924 w_k1_MSFT_s111
                                 + 1.04167 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.896246 w_k1_ADM_s111
                                 + 1.16349 w_k1_AXP_s111
                                 + 0.897817 w_k1_DIS_s111
                                 + 0.985266 w_k1_FDX_s111
                                 + 1.27484 w_k1_IBM_s111
                                 + 1.05195 w_k1_INTC_s111 + 1.1155 w_k1_KO_s111
                                 + 1.59406 w_k1_MAT_s111
                                 + 1.02931 w_k1_MSFT_s111
                                 + 1.04625 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.19713 w_k1_ADM_s111
                                 + 1.43048 w_k1_AXP_s111
                                 + 1.51172 w_k1_DIS_s111
                                 + 1.37866 w_k1_FDX_s111
                                 + 1.20578 w_k1_IBM_s111
                                 + 1.15998 w_k1_INTC_s111
                                 + 1.25958 w_k1_KO_s111 + 1.10725 w_k1_MAT_s111
                                 + 1.15348 w_k1_MSFT_s111
                                 + 1.14456 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.18267 w_k1_ADM_s111
                                 + 1.45873 w_k1_AXP_s111 + 1.4073 w_k1_DIS_s111
                                 + 1.26405 w_k1_FDX_s111
                                 + 1.68456 w_k1_IBM_s111
                                 + 1.12869 w_k1_INTC_s111
                                 + 1.64726 w_k1_KO_s111 + 2.1334 w_k1_MAT_s111
                                 + 1.12171 w_k1_MSFT_s111 + 1.1535 w_k1_PG_s111
                                 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.01284 w_k1_ADM_s111
                                 + 1.04127 w_k1_AXP_s111
                                 + 1.08268 w_k1_DIS_s111
                                 + 1.11302 w_k1_FDX_s111
                                 + 1.06262 w_k1_IBM_s111
                                 + 0.911478 w_k1_INTC_s111
                                 + 1.0847 w_k1_KO_s111 + 1.14908 w_k1_MAT_s111
                                 + 1.06883 w_k1_MSFT_s111
                                 + 1.11652 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.919694 w_k1_ADM_s111
                                 + 0.825289 w_k1_AXP_s111
                                 + 0.712521 w_k1_DIS_s111
                                 + 0.852803 w_k1_FDX_s111
                                 + 0.987442 w_k1_IBM_s111
                                 + 0.77078 w_k1_INTC_s111
                                 + 0.88106 w_k1_KO_s111
                                 + 0.907192 w_k1_MAT_s111
                                 + 0.843842 w_k1_MSFT_s111
                                 + 0.887247 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.868098 w_k1_ADM_s112
                                 + 1.00023 w_k1_AXP_s112
                                 + 0.871597 w_k1_DIS_s112
                                 + 0.972293 w_k1_FDX_s112
                                 + 0.756078 w_k1_IBM_s112
                                 + 0.855486 w_k1_INTC_s112
                                 + 0.749118 w_k1_KO_s112
                                 + 0.784598 w_k1_MAT_s112
                                 + 0.989278 w_k1_MSFT_s112
                                 + 0.986058 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.04926 w_k1_ADM_s112
                                 + 0.927422 w_k1_AXP_s112
                                 + 1.07736 w_k1_DIS_s112
                                 + 0.997182 w_k1_FDX_s112
                                 + 1.01532 w_k1_IBM_s112
                                 + 0.983154 w_k1_INTC_s112
                                 + 1.0042 w_k1_KO_s112 + 0.96056 w_k1_MAT_s112
                                 + 1.06078 w_k1_MSFT_s112 + 1.0495 w_k1_PG_s112
                                 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.25658 w_k1_ADM_s112
                                 + 1.01224 w_k1_AXP_s112
                                 + 1.02823 w_k1_DIS_s112
                                 + 0.916757 w_k1_FDX_s112
                                 + 1.74114 w_k1_IBM_s112
                                 + 0.997967 w_k1_INTC_s112
                                 + 1.25103 w_k1_KO_s112 + 1.50475 w_k1_MAT_s112
                                 + 0.88451 w_k1_MSFT_s112
                                 + 1.22271 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.19113 w_k1_ADM_s112
                                 + 1.28122 w_k1_AXP_s112
                                 + 1.35582 w_k1_DIS_s112
                                 + 1.22047 w_k1_FDX_s112
                                 + 0.890131 w_k1_IBM_s112
                                 + 1.38521 w_k1_INTC_s112
                                 + 1.09306 w_k1_KO_s112
                                 + 0.832703 w_k1_MAT_s112
                                 + 1.25892 w_k1_MSFT_s112
                                 + 1.10227 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.50864 w_k1_ADM_s112
                                 + 1.81667 w_k1_AXP_s112
                                 + 1.58059 w_k1_DIS_s112
                                 + 1.29735 w_k1_FDX_s112
                                 + 1.38779 w_k1_IBM_s112
                                 + 1.43483 w_k1_INTC_s112
                                 + 1.27576 w_k1_KO_s112 + 1.56764 w_k1_MAT_s112
                                 + 1.43851 w_k1_MSFT_s112
                                 + 1.40646 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.11658 w_k1_ADM_s112
                                 + 1.06613 w_k1_AXP_s112
                                 + 1.31878 w_k1_DIS_s112
                                 + 1.07712 w_k1_FDX_s112
                                 + 1.27119 w_k1_IBM_s112
                                 + 1.12733 w_k1_INTC_s112
                                 + 1.31705 w_k1_KO_s112 + 1.19865 w_k1_MAT_s112
                                 + 1.15088 w_k1_MSFT_s112
                                 + 1.22484 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.840772 w_k1_ADM_s112
                                 + 0.647574 w_k1_AXP_s112
                                 + 0.618076 w_k1_DIS_s112
                                 + 0.746698 w_k1_FDX_s112
                                 + 0.684868 w_k1_IBM_s112
                                 + 0.939174 w_k1_INTC_s112
                                 + 0.640737 w_k1_KO_s112
                                 + 0.535749 w_k1_MAT_s112
                                 + 0.809222 w_k1_MSFT_s112
                                 + 0.852248 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.0592 w_k1_ADM_s112
                                 + 0.728637 w_k1_AXP_s112
                                 + 0.834739 w_k1_DIS_s112
                                 + 0.778743 w_k1_FDX_s112
                                 + 0.921136 w_k1_IBM_s112
                                 + 0.848742 w_k1_INTC_s112
                                 + 0.782757 w_k1_KO_s112
                                 + 0.837194 w_k1_MAT_s112
                                 + 0.94159 w_k1_MSFT_s112
                                 + 0.938804 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.45904 w_k1_ADM_s113
                                 + 1.5004 w_k1_AXP_s113 + 1.56329 w_k1_DIS_s113
                                 + 1.40158 w_k1_FDX_s113
                                 + 1.02128 w_k1_IBM_s113
                                 + 1.42869 w_k1_INTC_s113
                                 + 1.13718 w_k1_KO_s113 + 1.15923 w_k1_MAT_s113
                                 + 1.23819 w_k1_MSFT_s113
                                 + 1.04777 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.792218 w_k1_ADM_s113
                                 + 0.828543 w_k1_AXP_s113
                                 + 0.690723 w_k1_DIS_s113
                                 + 0.814688 w_k1_FDX_s113
                                 + 0.715679 w_k1_IBM_s113
                                 + 0.7861 w_k1_INTC_s113 + 0.73705 w_k1_KO_s113
                                 + 0.908941 w_k1_MAT_s113
                                 + 0.764367 w_k1_MSFT_s113
                                 + 0.803149 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.14927 w_k1_ADM_s113
                                 + 1.11373 w_k1_AXP_s113
                                 + 1.27911 w_k1_DIS_s113
                                 + 1.23455 w_k1_FDX_s113
                                 + 0.81185 w_k1_IBM_s113
                                 + 0.982421 w_k1_INTC_s113
                                 + 0.944619 w_k1_KO_s113
                                 + 0.864818 w_k1_MAT_s113
                                 + 1.07775 w_k1_MSFT_s113
                                 + 1.03048 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.0197 w_k1_ADM_s113
                                 + 1.33583 w_k1_AXP_s113
                                 + 0.998385 w_k1_DIS_s113
                                 + 1.10388 w_k1_FDX_s113
                                 + 1.28298 w_k1_IBM_s113
                                 + 1.09834 w_k1_INTC_s113
                                 + 1.06438 w_k1_KO_s113 + 1.38701 w_k1_MAT_s113
                                 + 0.946472 w_k1_MSFT_s113
                                 + 1.01317 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.980836 w_k1_ADM_s113
                                 + 1.32898 w_k1_AXP_s113
                                 + 1.43018 w_k1_DIS_s113
                                 + 1.15151 w_k1_FDX_s113
                                 + 1.63469 w_k1_IBM_s113
                                 + 1.16079 w_k1_INTC_s113
                                 + 1.40529 w_k1_KO_s113 + 1.80461 w_k1_MAT_s113
                                 + 1.11963 w_k1_MSFT_s113
                                 + 1.20912 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.878114 w_k1_ADM_s113
                                 + 1.12898 w_k1_AXP_s113
                                 + 0.839983 w_k1_DIS_s113
                                 + 0.998271 w_k1_FDX_s113
                                 + 0.875896 w_k1_IBM_s113
                                 + 0.985891 w_k1_INTC_s113
                                 + 0.877704 w_k1_KO_s113
                                 + 0.910781 w_k1_MAT_s113
                                 + 0.816443 w_k1_MSFT_s113
                                 + 0.836441 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.974359 w_k1_ADM_s113
                                 + 0.766016 w_k1_AXP_s113
                                 + 0.871492 w_k1_DIS_s113
                                 + 0.886719 w_k1_FDX_s113
                                 + 0.977172 w_k1_IBM_s113
                                 + 0.765951 w_k1_INTC_s113
                                 + 0.901913 w_k1_KO_s113
                                 + 1.21444 w_k1_MAT_s113
                                 + 0.863885 w_k1_MSFT_s113
                                 + 0.999001 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.940331 w_k1_ADM_s113
                                 + 1.05339 w_k1_AXP_s113
                                 + 1.10007 w_k1_DIS_s113
                                 + 1.06323 w_k1_FDX_s113
                                 + 0.986214 w_k1_IBM_s113
                                 + 1.15846 w_k1_INTC_s113
                                 + 1.17585 w_k1_KO_s113 + 1.24841 w_k1_MAT_s113
                                 + 1.11645 w_k1_MSFT_s113
                                 + 1.01819 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.30436 w_k1_ADM_s114
                                 + 1.22423 w_k1_AXP_s114
                                 + 1.61495 w_k1_DIS_s114
                                 + 1.32077 w_k1_FDX_s114
                                 + 1.22265 w_k1_IBM_s114
                                 + 1.34148 w_k1_INTC_s114
                                 + 1.31986 w_k1_KO_s114 + 1.02621 w_k1_MAT_s114
                                 + 1.38605 w_k1_MSFT_s114 + 1.312 w_k1_PG_s114
                                 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.18645 w_k1_ADM_s114
                                 + 0.911312 w_k1_AXP_s114
                                 + 1.18637 w_k1_DIS_s114
                                 + 1.12949 w_k1_FDX_s114
                                 + 0.830528 w_k1_IBM_s114
                                 + 1.02843 w_k1_INTC_s114
                                 + 0.990025 w_k1_KO_s114
                                 + 0.780463 w_k1_MAT_s114
                                 + 1.26458 w_k1_MSFT_s114
                                 + 1.04777 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.887039 w_k1_ADM_s114
                                 + 0.984458 w_k1_AXP_s114
                                 + 0.641134 w_k1_DIS_s114
                                 + 0.79549 w_k1_FDX_s114
                                 + 0.94356 w_k1_IBM_s114
                                 + 0.859499 w_k1_INTC_s114
                                 + 0.775751 w_k1_KO_s114
                                 + 0.86911 w_k1_MAT_s114
                                 + 0.828742 w_k1_MSFT_s114
                                 + 0.866852 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.768948 w_k1_ADM_s114
                                 + 1.06481 w_k1_AXP_s114
                                 + 1.24225 w_k1_DIS_s114 + 1.0489 w_k1_FDX_s114
                                 + 1.54295 w_k1_IBM_s114
                                 + 1.11826 w_k1_INTC_s114 + 1.2 w_k1_KO_s114
                                 + 1.19656 w_k1_MAT_s114
                                 + 1.0264 w_k1_MSFT_s114 + 1.11417 w_k1_PG_s114
                                 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.903488 w_k1_ADM_s114
                                 + 0.810624 w_k1_AXP_s114
                                 + 0.792269 w_k1_DIS_s114
                                 + 0.843876 w_k1_FDX_s114
                                 + 0.904013 w_k1_IBM_s114
                                 + 0.878771 w_k1_INTC_s114
                                 + 0.898066 w_k1_KO_s114
                                 + 0.672532 w_k1_MAT_s114
                                 + 0.908793 w_k1_MSFT_s114
                                 + 0.909611 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.2406 w_k1_ADM_s114
                                 + 1.19835 w_k1_AXP_s114 + 1.3737 w_k1_DIS_s114
                                 + 1.32325 w_k1_FDX_s114
                                 + 0.968036 w_k1_IBM_s114
                                 + 1.17173 w_k1_INTC_s114
                                 + 1.13427 w_k1_KO_s114
                                 + 0.891145 w_k1_MAT_s114
                                 + 1.32762 w_k1_MSFT_s114
                                 + 1.10405 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.728169 w_k1_ADM_s114
                                 + 1.05053 w_k1_AXP_s114
                                 + 1.09981 w_k1_DIS_s114
                                 + 0.968069 w_k1_FDX_s114
                                 + 1.11534 w_k1_IBM_s114
                                 + 1.02292 w_k1_INTC_s114
                                 + 1.04952 w_k1_KO_s114
                                 + 0.966322 w_k1_MAT_s114
                                 + 1.11889 w_k1_MSFT_s114
                                 + 0.991643 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.919833 w_k1_ADM_s114
                                 + 0.682758 w_k1_AXP_s114
                                 + 0.866686 w_k1_DIS_s114
                                 + 0.958314 w_k1_FDX_s114
                                 + 0.634281 w_k1_IBM_s114
                                 + 0.917973 w_k1_INTC_s114
                                 + 0.760745 w_k1_KO_s114
                                 + 0.54197 w_k1_MAT_s114
                                 + 0.981196 w_k1_MSFT_s114
                                 + 0.823255 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.607985 w_k1_ADM_s115
                                 + 0.890317 w_k1_AXP_s115
                                 + 0.886213 w_k1_DIS_s115
                                 + 0.828916 w_k1_FDX_s115
                                 + 1.26415 w_k1_IBM_s115
                                 + 0.913639 w_k1_INTC_s115
                                 + 0.938521 w_k1_KO_s115
                                 + 1.25457 w_k1_MAT_s115
                                 + 0.859187 w_k1_MSFT_s115
                                 + 0.882577 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.896962 w_k1_ADM_s115
                                 + 1.2441 w_k1_AXP_s115 + 1.29524 w_k1_DIS_s115
                                 + 1.05256 w_k1_FDX_s115
                                 + 1.88202 w_k1_IBM_s115
                                 + 1.0208 w_k1_INTC_s115 + 1.37361 w_k1_KO_s115
                                 + 1.92793 w_k1_MAT_s115
                                 + 1.15844 w_k1_MSFT_s115
                                 + 1.07428 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.971878 w_k1_ADM_s115
                                 + 1.20659 w_k1_AXP_s115
                                 + 1.15133 w_k1_DIS_s115 + 1.0578 w_k1_FDX_s115
                                 + 1.39363 w_k1_IBM_s115
                                 + 1.25141 w_k1_INTC_s115
                                 + 1.15377 w_k1_KO_s115 + 1.28756 w_k1_MAT_s115
                                 + 1.14352 w_k1_MSFT_s115
                                 + 1.09337 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.930639 w_k1_ADM_s115
                                 + 0.909528 w_k1_AXP_s115
                                 + 0.963102 w_k1_DIS_s115
                                 + 1.05074 w_k1_FDX_s115
                                 + 0.604072 w_k1_IBM_s115
                                 + 1.07338 w_k1_INTC_s115
                                 + 0.726341 w_k1_KO_s115
                                 + 0.681091 w_k1_MAT_s115
                                 + 1.00303 w_k1_MSFT_s115
                                 + 0.809275 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.50887 w_k1_ADM_s115
                                 + 1.74963 w_k1_AXP_s115
                                 + 2.05947 w_k1_DIS_s115 + 1.6103 w_k1_FDX_s115
                                 + 1.1688 w_k1_IBM_s115
                                 + 1.24378 w_k1_INTC_s115 + 1.3605 w_k1_KO_s115
                                 + 1.40955 w_k1_MAT_s115
                                 + 1.49542 w_k1_MSFT_s115
                                 + 1.32762 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.11287 w_k1_ADM_s115
                                 + 1.25092 w_k1_AXP_s115
                                 + 1.28368 w_k1_DIS_s115
                                 + 1.28751 w_k1_FDX_s115
                                 + 0.939117 w_k1_IBM_s115
                                 + 1.34566 w_k1_INTC_s115
                                 + 1.09849 w_k1_KO_s115
                                 + 0.921729 w_k1_MAT_s115
                                 + 1.31909 w_k1_MSFT_s115
                                 + 1.00454 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.565885 w_k1_ADM_s115
                                 + 0.74722 w_k1_AXP_s115
                                 + 0.806367 w_k1_DIS_s115
                                 + 0.767489 w_k1_FDX_s115
                                 + 0.940266 w_k1_IBM_s115
                                 + 0.801693 w_k1_INTC_s115
                                 + 0.760581 w_k1_KO_s115
                                 + 0.897504 w_k1_MAT_s115
                                 + 0.878094 w_k1_MSFT_s115
                                 + 0.837903 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.873653 w_k1_ADM_s115
                                 + 1.01508 w_k1_AXP_s115
                                 + 1.02027 w_k1_DIS_s115
                                 + 1.01335 w_k1_FDX_s115
                                 + 0.911833 w_k1_IBM_s115
                                 + 1.06473 w_k1_INTC_s115
                                 + 1.01948 w_k1_KO_s115 + 1.02062 w_k1_MAT_s115
                                 + 1.08841 w_k1_MSFT_s115
                                 + 0.985276 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.05021 w_k1_ADM_s116
                                 + 1.08352 w_k1_AXP_s116
                                 + 1.01674 w_k1_DIS_s116
                                 + 1.10359 w_k1_FDX_s116
                                 + 0.867282 w_k1_IBM_s116
                                 + 1.0929 w_k1_INTC_s116
                                 + 0.964588 w_k1_KO_s116
                                 + 0.707675 w_k1_MAT_s116
                                 + 1.06304 w_k1_MSFT_s116
                                 + 1.06754 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.875202 w_k1_ADM_s116
                                 + 0.75091 w_k1_AXP_s116
                                 + 0.704839 w_k1_DIS_s116
                                 + 0.827957 w_k1_FDX_s116
                                 + 0.979088 w_k1_IBM_s116
                                 + 0.716008 w_k1_INTC_s116
                                 + 0.766038 w_k1_KO_s116
                                 + 0.777051 w_k1_MAT_s116
                                 + 0.755045 w_k1_MSFT_s116
                                 + 0.832816 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.687488 w_k1_ADM_s116
                                 + 0.605459 w_k1_AXP_s116
                                 + 0.53015 w_k1_DIS_s116
                                 + 0.698465 w_k1_FDX_s116
                                 + 0.642661 w_k1_IBM_s116
                                 + 0.861245 w_k1_INTC_s116
                                 + 0.626712 w_k1_KO_s116
                                 + 0.368196 w_k1_MAT_s116
                                 + 0.774224 w_k1_MSFT_s116
                                 + 0.709355 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.96227 w_k1_ADM_s116
                                 + 1.15748 w_k1_AXP_s116
                                 + 0.877462 w_k1_DIS_s116
                                 + 1.04298 w_k1_FDX_s116
                                 + 1.20805 w_k1_IBM_s116
                                 + 1.24457 w_k1_INTC_s116
                                 + 0.973369 w_k1_KO_s116
                                 + 1.14766 w_k1_MAT_s116
                                 + 1.05324 w_k1_MSFT_s116 + 1.1319 w_k1_PG_s116
                                 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.07864 w_k1_ADM_s116
                                 + 1.30527 w_k1_AXP_s116
                                 + 1.50526 w_k1_DIS_s116
                                 + 1.32699 w_k1_FDX_s116
                                 + 1.10859 w_k1_IBM_s116
                                 + 1.38932 w_k1_INTC_s116
                                 + 1.32296 w_k1_KO_s116
                                 + 0.770649 w_k1_MAT_s116
                                 + 1.19819 w_k1_MSFT_s116
                                 + 1.22069 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.805936 w_k1_ADM_s116
                                 + 0.718431 w_k1_AXP_s116
                                 + 0.85059 w_k1_DIS_s116
                                 + 0.960248 w_k1_FDX_s116
                                 + 0.748442 w_k1_IBM_s116
                                 + 0.95514 w_k1_INTC_s116
                                 + 0.963705 w_k1_KO_s116
                                 + 0.632896 w_k1_MAT_s116
                                 + 0.970214 w_k1_MSFT_s116
                                 + 0.910993 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.14904 w_k1_ADM_s116
                                 + 1.64731 w_k1_AXP_s116
                                 + 1.25304 w_k1_DIS_s116 + 1.1836 w_k1_FDX_s116
                                 + 1.61906 w_k1_IBM_s116
                                 + 1.36847 w_k1_INTC_s116
                                 + 1.28968 w_k1_KO_s116 + 1.57955 w_k1_MAT_s116
                                 + 1.1542 w_k1_MSFT_s116 + 1.12061 w_k1_PG_s116
                                 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.38744 w_k1_ADM_s116
                                 + 1.25321 w_k1_AXP_s116
                                 + 1.47541 w_k1_DIS_s116
                                 + 1.15585 w_k1_FDX_s116
                                 + 1.52978 w_k1_IBM_s116
                                 + 1.42202 w_k1_INTC_s116
                                 + 1.36679 w_k1_KO_s116 + 1.20504 w_k1_MAT_s116
                                 + 1.20548 w_k1_MSFT_s116
                                 + 1.24903 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.03572 w_k1_ADM_s117
                                 + 0.920659 w_k1_AXP_s117
                                 + 0.991395 w_k1_DIS_s117
                                 + 1.07493 w_k1_FDX_s117
                                 + 0.651454 w_k1_IBM_s117
                                 + 1.15038 w_k1_INTC_s117
                                 + 0.986296 w_k1_KO_s117
                                 + 1.17664 w_k1_MAT_s117
                                 + 1.02866 w_k1_MSFT_s117
                                 + 1.03292 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.15116 w_k1_ADM_s117
                                 + 1.14524 w_k1_AXP_s117
                                 + 1.23185 w_k1_DIS_s117
                                 + 1.14001 w_k1_FDX_s117 + 1.1141 w_k1_IBM_s117
                                 + 1.16376 w_k1_INTC_s117 + 1.1396 w_k1_KO_s117
                                 + 1.36979 w_k1_MAT_s117
                                 + 1.16331 w_k1_MSFT_s117
                                 + 1.10097 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.760819 w_k1_ADM_s117
                                 + 0.716933 w_k1_AXP_s117
                                 + 0.726301 w_k1_DIS_s117
                                 + 0.749718 w_k1_FDX_s117
                                 + 0.577174 w_k1_IBM_s117
                                 + 0.896719 w_k1_INTC_s117
                                 + 0.860754 w_k1_KO_s117
                                 + 0.643364 w_k1_MAT_s117
                                 + 0.82056 w_k1_MSFT_s117
                                 + 0.87494 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.03051 w_k1_ADM_s117
                                 + 1.43528 w_k1_AXP_s117
                                 + 1.56227 w_k1_DIS_s117 + 1.1618 w_k1_FDX_s117
                                 + 1.72522 w_k1_IBM_s117
                                 + 1.40486 w_k1_INTC_s117
                                 + 1.63785 w_k1_KO_s117 + 2.26713 w_k1_MAT_s117
                                 + 1.1226 w_k1_MSFT_s117 + 1.29681 w_k1_PG_s117
                                 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.967849 w_k1_ADM_s117
                                 + 0.863854 w_k1_AXP_s117
                                 + 1.16292 w_k1_DIS_s117
                                 + 0.982921 w_k1_FDX_s117
                                 + 0.883668 w_k1_IBM_s117
                                 + 1.08103 w_k1_INTC_s117
                                 + 1.14171 w_k1_KO_s117 + 1.24269 w_k1_MAT_s117
                                 + 1.07501 w_k1_MSFT_s117
                                 + 0.993301 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.4659 w_k1_ADM_s117
                                 + 1.57394 w_k1_AXP_s117
                                 + 1.58102 w_k1_DIS_s117
                                 + 1.40958 w_k1_FDX_s117
                                 + 1.08621 w_k1_IBM_s117
                                 + 1.29582 w_k1_INTC_s117
                                 + 1.37864 w_k1_KO_s117 + 1.34912 w_k1_MAT_s117
                                 + 1.27321 w_k1_MSFT_s117
                                 + 1.29435 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.05979 w_k1_ADM_s117
                                 + 0.872127 w_k1_AXP_s117
                                 + 0.963494 w_k1_DIS_s117
                                 + 0.900445 w_k1_FDX_s117
                                 + 1.02079 w_k1_IBM_s117
                                 + 0.959326 w_k1_INTC_s117
                                 + 1.07685 w_k1_KO_s117 + 1.48211 w_k1_MAT_s117
                                 + 0.955884 w_k1_MSFT_s117
                                 + 1.05382 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.834476 w_k1_ADM_s117
                                 + 0.71773 w_k1_AXP_s117
                                 + 0.788506 w_k1_DIS_s117
                                 + 0.750383 w_k1_FDX_s117
                                 + 1.15127 w_k1_IBM_s117
                                 + 0.891117 w_k1_INTC_s117
                                 + 1.13951 w_k1_KO_s117 + 1.26908 w_k1_MAT_s117
                                 + 0.803424 w_k1_MSFT_s117
                                 + 0.968238 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0454787693051773 w_k1_ADM_s000
                                 + 0.0171959294398141 w_k1_ADM_s001
                                 + 0.0136174249923789 w_k1_ADM_s002
                                 + 0.0116407578230133 w_k1_ADM_s003
                                 + 0.0390436771255618 w_k1_ADM_s004
                                 + 0.0104870737084846 w_k1_ADM_s005
                                 + 0.0553756272511381 w_k1_ADM_s006
                                 + 0.010969692039865 w_k1_ADM_s007
                                 + 0.00289870605597276 w_k1_ADM_s010
                                 + 0.0148990898698125 w_k1_ADM_s011
                                 + 0.00289577863905445 w_k1_ADM_s012
                                 + 0.00418619564522652 w_k1_ADM_s013
                                 + 0.00399367340405712 w_k1_ADM_s014
                                 + 0.00642789147305676 w_k1_ADM_s015
                                 + 0.0122600751999456 w_k1_ADM_s016
                                 + 0.00455101908153944 w_k1_ADM_s017
                                 + 0.018376740880191 w_k1_ADM_s100
                                 + 0.0228532569536142 w_k1_ADM_s101
                                 + 0.0270552424988227 w_k1_ADM_s102
                                 + 0.011652366340476 w_k1_ADM_s103
                                 + 0.0169188322689889 w_k1_ADM_s104
                                 + 0.0118902204423949 w_k1_ADM_s105
                                 + 0.0106760293901105 w_k1_ADM_s106
                                 + 0.00999137759519862 w_k1_ADM_s107
                                 + 0.128319357865646 w_k1_ADM_s110
                                 + 0.0806998977688629 w_k1_ADM_s111
                                 + 0.0964053422429144 w_k1_ADM_s112
                                 + 0.0476191434748998 w_k1_ADM_s113
                                 + 0.0624331808910336 w_k1_ADM_s114
                                 + 0.0695966790820724 w_k1_ADM_s115
                                 + 0.0724113426576245 w_k1_ADM_s116
                                 + 0.0596577979650481 w_k1_ADM_s117
                                 + 0.0474777495982025 w_k1_AXP_s000
                                 + 0.0161155447721086 w_k1_AXP_s001
                                 + 0.0149521806793386 w_k1_AXP_s002
                                 + 0.0102356383175289 w_k1_AXP_s003
                                 + 0.0354005887003096 w_k1_AXP_s004
                                 + 0.0104860201327921 w_k1_AXP_s005
                                 + 0.0556913517844715 w_k1_AXP_s006
                                 + 0.0138097484379913 w_k1_AXP_s007
                                 + 0.00288564704605056 w_k1_AXP_s010
                                 + 0.0128530227181001 w_k1_AXP_s011
                                 + 0.00315525026405934 w_k1_AXP_s012
                                 + 0.00398963938226116 w_k1_AXP_s013
                                 + 0.00381403786787519 w_k1_AXP_s014
                                 + 0.00585816305498778 w_k1_AXP_s015
                                 + 0.011436875024233 w_k1_AXP_s016
                                 + 0.00459437448951903 w_k1_AXP_s017
                                 + 0.0158790156065719 w_k1_AXP_s100
                                 + 0.0222856238167753 w_k1_AXP_s101
                                 + 0.0280530976878745 w_k1_AXP_s102
                                 + 0.0139256098533344 w_k1_AXP_s103
                                 + 0.0194700118564268 w_k1_AXP_s104
                                 + 0.0122540355493077 w_k1_AXP_s105
                                 + 0.00927120886447998 w_k1_AXP_s106
                                 + 0.00945570712149089 w_k1_AXP_s107
                                 + 0.139829910851209 w_k1_AXP_s110
                                 + 0.08533014702924 w_k1_AXP_s111
                                 + 0.0910215096726047 w_k1_AXP_s112
                                 + 0.0504617089638937 w_k1_AXP_s113
                                 + 0.063824720790222 w_k1_AXP_s114
                                 + 0.0825691177889199 w_k1_AXP_s115
                                 + 0.0758463698065646 w_k1_AXP_s116
                                 + 0.0575427455273862 w_k1_AXP_s117
                                 + 0.0447371082817353 w_k1_DIS_s000
                                 + 0.017205365133916 w_k1_DIS_s001
                                 + 0.0150386816758052 w_k1_DIS_s002
                                 + 0.0128321187772476 w_k1_DIS_s003
                                 + 0.0395338116832837 w_k1_DIS_s004
                                 + 0.0107917620386533 w_k1_DIS_s005
                                 + 0.0556279802516575 w_k1_DIS_s006
                                 + 0.0131227477178682 w_k1_DIS_s007
                                 + 0.00331597356030301 w_k1_DIS_s010
                                 + 0.0135936428062313 w_k1_DIS_s011
                                 + 0.00369593571972333 w_k1_DIS_s012
                                 + 0.00424071561954248 w_k1_DIS_s013
                                 + 0.00353033086711411 w_k1_DIS_s014
                                 + 0.00581160676037395 w_k1_DIS_s015
                                 + 0.0132321799310391 w_k1_DIS_s016
                                 + 0.00410292303448672 w_k1_DIS_s017
                                 + 0.014874461483125 w_k1_DIS_s100
                                 + 0.0232628932767175 w_k1_DIS_s101
                                 + 0.0308624468387838 w_k1_DIS_s102
                                 + 0.011948382533639 w_k1_DIS_s103
                                 + 0.016815639277047 w_k1_DIS_s104
                                 + 0.0113972846386441 w_k1_DIS_s105
                                 + 0.0093166518751555 w_k1_DIS_s106
                                 + 0.0104523261858003 w_k1_DIS_s107
                                 + 0.135232953251027 w_k1_DIS_s110
                                 + 0.0785904130307786 w_k1_DIS_s111
                                 + 0.0960931990729681 w_k1_DIS_s112
                                 + 0.0488632814651973 w_k1_DIS_s113
                                 + 0.0715163822310053 w_k1_DIS_s114
                                 + 0.086742437820129 w_k1_DIS_s115
                                 + 0.0724678409708227 w_k1_DIS_s116
                                 + 0.063540602920656 w_k1_DIS_s117
                                 + 0.0432098603008436 w_k1_FDX_s000
                                 + 0.0168961367748262 w_k1_FDX_s001
                                 + 0.0145747170914083 w_k1_FDX_s002
                                 + 0.011616455793594 w_k1_FDX_s003
                                 + 0.0356772628951898 w_k1_FDX_s004
                                 + 0.011344832064738 w_k1_FDX_s005
                                 + 0.0521789470451625 w_k1_FDX_s006
                                 + 0.0125095067969066 w_k1_FDX_s007
                                 + 0.00308013713950362 w_k1_FDX_s010
                                 + 0.0134351322181367 w_k1_FDX_s011
                                 + 0.00344996945470466 w_k1_FDX_s012
                                 + 0.0039699889431902 w_k1_FDX_s013
                                 + 0.00349393147127494 w_k1_FDX_s014
                                 + 0.00595698596611067 w_k1_FDX_s015
                                 + 0.0127929615847269 w_k1_FDX_s016
                                 + 0.00411387593228002 w_k1_FDX_s017
                                 + 0.015843544083998 w_k1_FDX_s100
                                 + 0.0240074482571438 w_k1_FDX_s101
                                 + 0.0284207633993584 w_k1_FDX_s102
                                 + 0.0129217447885723 w_k1_FDX_s103
                                 + 0.0172153670062782 w_k1_FDX_s104
                                 + 0.0116198763102796 w_k1_FDX_s105
                                 + 0.00915469267825096 w_k1_FDX_s106
                                 + 0.00968611032814745 w_k1_FDX_s107
                                 + 0.136916936100015 w_k1_FDX_s110
                                 + 0.084164366851615 w_k1_FDX_s111
                                 + 0.0897326238236249 w_k1_FDX_s112
                                 + 0.0492944705392569 w_k1_FDX_s113
                                 + 0.0672139810422862 w_k1_FDX_s114
                                 + 0.0819238598663989 w_k1_FDX_s115
                                 + 0.0755138461796044 w_k1_FDX_s116
                                 + 0.057971797722676 w_k1_FDX_s117
                                 + 0.0435196619817543 w_k1_IBM_s000
                                 + 0.0171471651684873 w_k1_IBM_s001
                                 + 0.0151642892486201 w_k1_IBM_s002
                                 + 0.0113257401379687 w_k1_IBM_s003
                                 + 0.0363273144163983 w_k1_IBM_s004
                                 + 0.0111550320085148 w_k1_IBM_s005
                                 + 0.0541557347441667 w_k1_IBM_s006
                                 + 0.012822517307697 w_k1_IBM_s007
                                 + 0.00305651423706799 w_k1_IBM_s010
                                 + 0.0136995605014856 w_k1_IBM_s011
                                 + 0.00345551589714246 w_k1_IBM_s012
                                 + 0.00392998371880999 w_k1_IBM_s013
                                 + 0.00356074966774315 w_k1_IBM_s014
                                 + 0.00595463322815886 w_k1_IBM_s015
                                 + 0.0121736838830748 w_k1_IBM_s016
                                 + 0.00436541477692655 w_k1_IBM_s017
                                 + 0.0163205112832946 w_k1_IBM_s100
                                 + 0.024611766681446 w_k1_IBM_s101
                                 + 0.030274447001114 w_k1_IBM_s102
                                 + 0.0128156454000314 w_k1_IBM_s103
                                 + 0.017568175276332 w_k1_IBM_s104
                                 + 0.0117185540372398 w_k1_IBM_s105
                                 + 0.00923955614892418 w_k1_IBM_s106
                                 + 0.0102844042526264 w_k1_IBM_s107
                                 + 0.136740418931088 w_k1_IBM_s110
                                 + 0.0851577197924041 w_k1_IBM_s111
                                 + 0.0921839896602588 w_k1_IBM_s112
                                 + 0.0488089988416747 w_k1_IBM_s113
                                 + 0.0692804330641585 w_k1_IBM_s114
                                 + 0.0808997642385661 w_k1_IBM_s115
                                 + 0.0772699500829201 w_k1_IBM_s116
                                 + 0.0593372373666115 w_k1_IBM_s117
                                 + 0.0380539615838109 w_k1_INTC_s000
                                 + 0.0169371237814255 w_k1_INTC_s001
                                 + 0.0152692883557714 w_k1_INTC_s002
                                 + 0.0118524809104533 w_k1_INTC_s003
                                 + 0.0371342928162512 w_k1_INTC_s004
                                 + 0.0126608184390341 w_k1_INTC_s005
                                 + 0.0500071019177807 w_k1_INTC_s006
                                 + 0.0131585679252434 w_k1_INTC_s007
                                 + 0.00303576129029218 w_k1_INTC_s010
                                 + 0.0119996277975935 w_k1_INTC_s011
                                 + 0.00360677639830996 w_k1_INTC_s012
                                 + 0.00363781695079245 w_k1_INTC_s013
                                 + 0.00345103822276036 w_k1_INTC_s014
                                 + 0.00605398172318812 w_k1_INTC_s015
                                 + 0.0131525172325467 w_k1_INTC_s016
                                 + 0.00470413137744387 w_k1_INTC_s017
                                 + 0.0134320857009851 w_k1_INTC_s100
                                 + 0.0232411734558512 w_k1_INTC_s101
                                 + 0.0292722775520096 w_k1_INTC_s102
                                 + 0.0109911221415579 w_k1_INTC_s103
                                 + 0.020493475878767 w_k1_INTC_s104
                                 + 0.00960203661799267 w_k1_INTC_s105
                                 + 0.00858573382528589 w_k1_INTC_s106
                                 + 0.00987215697332308 w_k1_INTC_s107
                                 + 0.140751060801331 w_k1_INTC_s110
                                 + 0.0750301685607853 w_k1_INTC_s111
                                 + 0.0940428000988828 w_k1_INTC_s112
                                 + 0.0477124789722633 w_k1_INTC_s113
                                 + 0.0674527655753264 w_k1_INTC_s114
                                 + 0.0831999780643437 w_k1_INTC_s115
                                 + 0.0801444343421977 w_k1_INTC_s116
                                 + 0.062403409609145 w_k1_INTC_s117
                                 + 0.0378403898592057 w_k1_KO_s000
                                 + 0.0182647202311296 w_k1_KO_s001
                                 + 0.0154233657068312 w_k1_KO_s002
                                 + 0.0123965889637341 w_k1_KO_s003
                                 + 0.0311981447206652 w_k1_KO_s004
                                 + 0.0107968262180124 w_k1_KO_s005
                                 + 0.0559047714035566 w_k1_KO_s006
                                 + 0.0147673003353299 w_k1_KO_s007
                                 + 0.00300131183670344 w_k1_KO_s010
                                 + 0.0133110774552369 w_k1_KO_s011
                                 + 0.00355393320404007 w_k1_KO_s012
                                 + 0.00447437764551753 w_k1_KO_s013
                                 + 0.00344400957085232 w_k1_KO_s014
                                 + 0.00549016425009934 w_k1_KO_s015
                                 + 0.0138722145459633 w_k1_KO_s016
                                 + 0.00390506841819418 w_k1_KO_s017
                                 + 0.0150096423227208 w_k1_KO_s100
                                 + 0.0231587719018061 w_k1_KO_s101
                                 + 0.0306889333178768 w_k1_KO_s102
                                 + 0.0123522618767562 w_k1_KO_s103
                                 + 0.0180455122170026 w_k1_KO_s104
                                 + 0.0113384576146451 w_k1_KO_s105
                                 + 0.0102335346036152 w_k1_KO_s106
                                 + 0.0105953652501127 w_k1_KO_s107
                                 + 0.137447126682955 w_k1_KO_s110
                                 + 0.0851471275010295 w_k1_KO_s111
                                 + 0.0893941442899974 w_k1_KO_s112
                                 + 0.0481811368073896 w_k1_KO_s113
                                 + 0.0664033641770499 w_k1_KO_s114
                                 + 0.0767255617927181 w_k1_KO_s115
                                 + 0.0732486509230143 w_k1_KO_s116
                                 + 0.0665281731893651 w_k1_KO_s117
                                 + 0.0438229230566938 w_k1_MAT_s000
                                 + 0.0213531496121935 w_k1_MAT_s001
                                 + 0.0149696210525299 w_k1_MAT_s002
                                 + 0.0111670127663595 w_k1_MAT_s003
                                 + 0.0365133179806598 w_k1_MAT_s004
                                 + 0.01043606442615 w_k1_MAT_s005
                                 + 0.0549318101739902 w_k1_MAT_s006
                                 + 0.0149147587517546 w_k1_MAT_s007
                                 + 0.00263441148483342 w_k1_MAT_s010
                                 + 0.0148340029267124 w_k1_MAT_s011
                                 + 0.00413043510504261 w_k1_MAT_s012
                                 + 0.00426971349912873 w_k1_MAT_s013
                                 + 0.00359473886193933 w_k1_MAT_s014
                                 + 0.00565056019961219 w_k1_MAT_s015
                                 + 0.0128001714453316 w_k1_MAT_s016
                                 + 0.00399194412594626 w_k1_MAT_s017
                                 + 0.0178820031710405 w_k1_MAT_s100
                                 + 0.0189414055221206 w_k1_MAT_s101
                                 + 0.0320369285596886 w_k1_MAT_s102
                                 + 0.013222972924499 w_k1_MAT_s103
                                 + 0.0178652571759547 w_k1_MAT_s104
                                 + 0.0135231945145666 w_k1_MAT_s105
                                 + 0.0124090728662402 w_k1_MAT_s106
                                 + 0.00905428027262386 w_k1_MAT_s107
                                 + 0.158752471511454 w_k1_MAT_s110
                                 + 0.0928433522183097 w_k1_MAT_s111
                                 + 0.086904802924592 w_k1_MAT_s112
                                 + 0.0573982874496716 w_k1_MAT_s113
                                 + 0.0580368393563046 w_k1_MAT_s114
                                 + 0.0831104457331352 w_k1_MAT_s115
                                 + 0.0637634383662898 w_k1_MAT_s116
                                 + 0.0765294540511076 w_k1_MAT_s117
                                 + 0.0404631677050256 w_k1_MSFT_s000
                                 + 0.0187807603191489 w_k1_MSFT_s001
                                 + 0.0168897732466188 w_k1_MSFT_s002
                                 + 0.0124374269722583 w_k1_MSFT_s003
                                 + 0.0380918836597211 w_k1_MSFT_s004
                                 + 0.0112281141843738 w_k1_MSFT_s005
                                 + 0.0524831070427791 w_k1_MSFT_s006
                                 + 0.0124213797614676 w_k1_MSFT_s007
                                 + 0.00321776656191187 w_k1_MSFT_s010
                                 + 0.0122974437909353 w_k1_MSFT_s011
                                 + 0.00341627935966583 w_k1_MSFT_s012
                                 + 0.00384547721447299 w_k1_MSFT_s013
                                 + 0.00341743086782845 w_k1_MSFT_s014
                                 + 0.00529043437418469 w_k1_MSFT_s015
                                 + 0.013398008873817 w_k1_MSFT_s016
                                 + 0.00447712198773047 w_k1_MSFT_s017
                                 + 0.0142691339189656 w_k1_MSFT_s100
                                 + 0.0231915824772561 w_k1_MSFT_s101
                                 + 0.0305437582229716 w_k1_MSFT_s102
                                 + 0.0111407964892649 w_k1_MSFT_s103
                                 + 0.0173138509350742 w_k1_MSFT_s104
                                 + 0.0112047266532711 w_k1_MSFT_s105
                                 + 0.00926062655233026 w_k1_MSFT_s106
                                 + 0.00913171440145756 w_k1_MSFT_s107
                                 + 0.133250251241022 w_k1_MSFT_s110
                                 + 0.0783290438109509 w_k1_MSFT_s111
                                 + 0.0954825718795773 w_k1_MSFT_s112
                                 + 0.0461938223284525 w_k1_MSFT_s113
                                 + 0.0701998163721241 w_k1_MSFT_s114
                                 + 0.0848241436541195 w_k1_MSFT_s115
                                 + 0.0735850830820308 w_k1_MSFT_s116
                                 + 0.0592060426546458 w_k1_MSFT_s117
                                 + 0.0380962600721826 w_k1_PG_s000
                                 + 0.0172386164574031 w_k1_PG_s001
                                 + 0.014690713778776 w_k1_PG_s002
                                 + 0.0106114724805296 w_k1_PG_s003
                                 + 0.0340081223526476 w_k1_PG_s004
                                 + 0.0111286799839998 w_k1_PG_s005
                                 + 0.0544369603746355 w_k1_PG_s006
                                 + 0.0135475728482143 w_k1_PG_s007
                                 + 0.00295030448956951 w_k1_PG_s010
                                 + 0.0124780402022037 w_k1_PG_s011
                                 + 0.0031061006459747 w_k1_PG_s012
                                 + 0.0043810606210487 w_k1_PG_s013
                                 + 0.00361679784315163 w_k1_PG_s014
                                 + 0.00512999850908565 w_k1_PG_s015
                                 + 0.0122638302444527 w_k1_PG_s016
                                 + 0.00441762535782062 w_k1_PG_s017
                                 + 0.0161479512462951 w_k1_PG_s100
                                 + 0.0232761715087954 w_k1_PG_s101
                                 + 0.0296643154681776 w_k1_PG_s102
                                 + 0.0131520315903126 w_k1_PG_s103
                                 + 0.0160108270506684 w_k1_PG_s104
                                 + 0.0114669564771216 w_k1_PG_s105
                                 + 0.00919433165537042 w_k1_PG_s106
                                 + 0.00997670691174956 w_k1_PG_s107
                                 + 0.138086715126078 w_k1_PG_s110
                                 + 0.0804039209208306 w_k1_PG_s111
                                 + 0.0967879105724435 w_k1_PG_s112
                                 + 0.047364110088312 w_k1_PG_s113
                                 + 0.0663116247099629 w_k1_PG_s114
                                 + 0.0748891459950232 w_k1_PG_s115
                                 + 0.0751908942417017 w_k1_PG_s116
                                 + 0.0612588209804577 w_k1_PG_s117 >= 1.1
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
 0 <= w_k1_ADM_s10 <= 1
 0 <= w_k1_ADM_s11 <= 1
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
 0 <= w_k1_AXP_s0 <= 1
 0 <= w_k1_AXP_s1 <= 1
 0 <= w_k1_AXP_s00 <= 1
 0 <= w_k1_AXP_s01 <= 1
 0 <= w_k1_AXP_s10 <= 1
 0 <= w_k1_AXP_s11 <= 1
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
 0 <= w_k1_DIS_s0 <= 1
 0 <= w_k1_DIS_s1 <= 1
 0 <= w_k1_DIS_s00 <= 1
 0 <= w_k1_DIS_s01 <= 1
 0 <= w_k1_DIS_s10 <= 1
 0 <= w_k1_DIS_s11 <= 1
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
 0 <= w_k1_FDX_s0 <= 1
 0 <= w_k1_FDX_s1 <= 1
 0 <= w_k1_FDX_s00 <= 1
 0 <= w_k1_FDX_s01 <= 1
 0 <= w_k1_FDX_s10 <= 1
 0 <= w_k1_FDX_s11 <= 1
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
 0 <= w_k1_IBM_s0 <= 1
 0 <= w_k1_IBM_s1 <= 1
 0 <= w_k1_IBM_s00 <= 1
 0 <= w_k1_IBM_s01 <= 1
 0 <= w_k1_IBM_s10 <= 1
 0 <= w_k1_IBM_s11 <= 1
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
 0 <= w_k1_INTC_s0 <= 1
 0 <= w_k1_INTC_s1 <= 1
 0 <= w_k1_INTC_s00 <= 1
 0 <= w_k1_INTC_s01 <= 1
 0 <= w_k1_INTC_s10 <= 1
 0 <= w_k1_INTC_s11 <= 1
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
 0 <= w_k1_KO_s0 <= 1
 0 <= w_k1_KO_s1 <= 1
 0 <= w_k1_KO_s00 <= 1
 0 <= w_k1_KO_s01 <= 1
 0 <= w_k1_KO_s10 <= 1
 0 <= w_k1_KO_s11 <= 1
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
 0 <= w_k1_MAT_s0 <= 1
 0 <= w_k1_MAT_s1 <= 1
 0 <= w_k1_MAT_s00 <= 1
 0 <= w_k1_MAT_s01 <= 1
 0 <= w_k1_MAT_s10 <= 1
 0 <= w_k1_MAT_s11 <= 1
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
 0 <= w_k1_MSFT_s0 <= 1
 0 <= w_k1_MSFT_s1 <= 1
 0 <= w_k1_MSFT_s00 <= 1
 0 <= w_k1_MSFT_s01 <= 1
 0 <= w_k1_MSFT_s10 <= 1
 0 <= w_k1_MSFT_s11 <= 1
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
 0 <= w_k1_PG_s0 <= 1
 0 <= w_k1_PG_s1 <= 1
 0 <= w_k1_PG_s00 <= 1
 0 <= w_k1_PG_s01 <= 1
 0 <= w_k1_PG_s10 <= 1
 0 <= w_k1_PG_s11 <= 1
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
 0 <= b_k1_ADM_s10 <= 0.2
 0 <= b_k1_ADM_s11 <= 0.2
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
 0 <= b_k1_AXP_s0 <= 0.2
 0 <= b_k1_AXP_s1 <= 0.2
 0 <= b_k1_AXP_s00 <= 0.2
 0 <= b_k1_AXP_s01 <= 0.2
 0 <= b_k1_AXP_s10 <= 0.2
 0 <= b_k1_AXP_s11 <= 0.2
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
 0 <= b_k1_DIS_s0 <= 0.2
 0 <= b_k1_DIS_s1 <= 0.2
 0 <= b_k1_DIS_s00 <= 0.2
 0 <= b_k1_DIS_s01 <= 0.2
 0 <= b_k1_DIS_s10 <= 0.2
 0 <= b_k1_DIS_s11 <= 0.2
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
 0 <= b_k1_FDX_s0 <= 0.2
 0 <= b_k1_FDX_s1 <= 0.2
 0 <= b_k1_FDX_s00 <= 0.2
 0 <= b_k1_FDX_s01 <= 0.2
 0 <= b_k1_FDX_s10 <= 0.2
 0 <= b_k1_FDX_s11 <= 0.2
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
 0 <= b_k1_IBM_s0 <= 0.2
 0 <= b_k1_IBM_s1 <= 0.2
 0 <= b_k1_IBM_s00 <= 0.2
 0 <= b_k1_IBM_s01 <= 0.2
 0 <= b_k1_IBM_s10 <= 0.2
 0 <= b_k1_IBM_s11 <= 0.2
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
 0 <= b_k1_INTC_s0 <= 0.2
 0 <= b_k1_INTC_s1 <= 0.2
 0 <= b_k1_INTC_s00 <= 0.2
 0 <= b_k1_INTC_s01 <= 0.2
 0 <= b_k1_INTC_s10 <= 0.2
 0 <= b_k1_INTC_s11 <= 0.2
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
 0 <= b_k1_KO_s0 <= 0.2
 0 <= b_k1_KO_s1 <= 0.2
 0 <= b_k1_KO_s00 <= 0.2
 0 <= b_k1_KO_s01 <= 0.2
 0 <= b_k1_KO_s10 <= 0.2
 0 <= b_k1_KO_s11 <= 0.2
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
 0 <= b_k1_MAT_s0 <= 0.2
 0 <= b_k1_MAT_s1 <= 0.2
 0 <= b_k1_MAT_s00 <= 0.2
 0 <= b_k1_MAT_s01 <= 0.2
 0 <= b_k1_MAT_s10 <= 0.2
 0 <= b_k1_MAT_s11 <= 0.2
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
 0 <= b_k1_MSFT_s0 <= 0.2
 0 <= b_k1_MSFT_s1 <= 0.2
 0 <= b_k1_MSFT_s00 <= 0.2
 0 <= b_k1_MSFT_s01 <= 0.2
 0 <= b_k1_MSFT_s10 <= 0.2
 0 <= b_k1_MSFT_s11 <= 0.2
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
 0 <= b_k1_PG_s0 <= 0.2
 0 <= b_k1_PG_s1 <= 0.2
 0 <= b_k1_PG_s00 <= 0.2
 0 <= b_k1_PG_s01 <= 0.2
 0 <= b_k1_PG_s10 <= 0.2
 0 <= b_k1_PG_s11 <= 0.2
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
 0 <= s_k1_ADM_s10 <= 0.2
 0 <= s_k1_ADM_s11 <= 0.2
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
 0 <= s_k1_AXP_s0 <= 0.2
 0 <= s_k1_AXP_s1 <= 0.2
 0 <= s_k1_AXP_s00 <= 0.2
 0 <= s_k1_AXP_s01 <= 0.2
 0 <= s_k1_AXP_s10 <= 0.2
 0 <= s_k1_AXP_s11 <= 0.2
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
 0 <= s_k1_DIS_s0 <= 0.2
 0 <= s_k1_DIS_s1 <= 0.2
 0 <= s_k1_DIS_s00 <= 0.2
 0 <= s_k1_DIS_s01 <= 0.2
 0 <= s_k1_DIS_s10 <= 0.2
 0 <= s_k1_DIS_s11 <= 0.2
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
 0 <= s_k1_FDX_s0 <= 0.2
 0 <= s_k1_FDX_s1 <= 0.2
 0 <= s_k1_FDX_s00 <= 0.2
 0 <= s_k1_FDX_s01 <= 0.2
 0 <= s_k1_FDX_s10 <= 0.2
 0 <= s_k1_FDX_s11 <= 0.2
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
 0 <= s_k1_IBM_s0 <= 0.2
 0 <= s_k1_IBM_s1 <= 0.2
 0 <= s_k1_IBM_s00 <= 0.2
 0 <= s_k1_IBM_s01 <= 0.2
 0 <= s_k1_IBM_s10 <= 0.2
 0 <= s_k1_IBM_s11 <= 0.2
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
 0 <= s_k1_INTC_s0 <= 0.2
 0 <= s_k1_INTC_s1 <= 0.2
 0 <= s_k1_INTC_s00 <= 0.2
 0 <= s_k1_INTC_s01 <= 0.2
 0 <= s_k1_INTC_s10 <= 0.2
 0 <= s_k1_INTC_s11 <= 0.2
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
 0 <= s_k1_KO_s0 <= 0.2
 0 <= s_k1_KO_s1 <= 0.2
 0 <= s_k1_KO_s00 <= 0.2
 0 <= s_k1_KO_s01 <= 0.2
 0 <= s_k1_KO_s10 <= 0.2
 0 <= s_k1_KO_s11 <= 0.2
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
 0 <= s_k1_MAT_s0 <= 0.2
 0 <= s_k1_MAT_s1 <= 0.2
 0 <= s_k1_MAT_s00 <= 0.2
 0 <= s_k1_MAT_s01 <= 0.2
 0 <= s_k1_MAT_s10 <= 0.2
 0 <= s_k1_MAT_s11 <= 0.2
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
 0 <= s_k1_MSFT_s0 <= 0.2
 0 <= s_k1_MSFT_s1 <= 0.2
 0 <= s_k1_MSFT_s00 <= 0.2
 0 <= s_k1_MSFT_s01 <= 0.2
 0 <= s_k1_MSFT_s10 <= 0.2
 0 <= s_k1_MSFT_s11 <= 0.2
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
 0 <= s_k1_PG_s0 <= 0.2
 0 <= s_k1_PG_s1 <= 0.2
 0 <= s_k1_PG_s00 <= 0.2
 0 <= s_k1_PG_s01 <= 0.2
 0 <= s_k1_PG_s10 <= 0.2
 0 <= s_k1_PG_s11 <= 0.2
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
End

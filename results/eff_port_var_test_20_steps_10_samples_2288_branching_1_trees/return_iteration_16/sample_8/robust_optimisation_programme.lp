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
 w_k1_ADM_s0_return_constraint:  1.01115 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.991626 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.1449 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.860189 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.978625 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.0854 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.12718 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.05328 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.951384 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.28965 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.982212 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.62511 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.849822 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.784722 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.858439 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.12885 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.1207 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.26763 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.40358 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.697888 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.039 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.27188 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.10831 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.789561 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.16805 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.32892 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.61492 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.860919 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.99798 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.781771 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.807466 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.819152 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.621957 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.18369 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.913966 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.16325 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.04815 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.857566 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.02283 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  1.13594 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.21308 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.829461 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.961127 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.0196 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.892241 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.17332 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.825159 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.18612 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.58387 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.53915 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.698103 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.91085 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.692915 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.08566 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.980493 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.42962 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.71124 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.734173 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.06274 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.22765 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.01066 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.546311 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.3028 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.82294 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.80581 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.801658 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.0999 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.821462 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.65699 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.1583 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.713713 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.61343 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.939193 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.11007 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.939139 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.84927 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.01671 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  1.14798 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.2972 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.814944 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.990448 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.01955 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.04402 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.9792 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.818985 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.3097 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.58415 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.89868 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.740549 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.646105 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.5941 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.06577 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.852602 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.49994 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.93788 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.609415 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.946267 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.16966 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.993484 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.493336 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.49244 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.62849 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 2.45437 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.79988 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.938368 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.599972 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.681431 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.08091 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.635105 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.74447 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.84504 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.11629 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.18253 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.818067 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.01557 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  1.01647 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.21275 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.858984 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.01697 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.991752 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.08859 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.03814 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.990282 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.25115 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.1695 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.50404 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.832116 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.756358 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.76736 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.02432 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.01412 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.19362 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.46035 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.649694 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.949329 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.29733 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.927508 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.638838 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.44303 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.50753 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.66348 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.844558 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.03365 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.710459 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.768759 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.956364 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.707538 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.36192 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.973671 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.21215 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.10808 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.869632 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.92065 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.37428 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.03176 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.02764 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.864844 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.2363 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.663207 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.15649 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.727477 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.13183 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 2.00698 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.37231 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.764422 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.951615 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.606482 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.965291 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.715606 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.66027 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.20474 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.06035 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.26602 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.733031 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.23166 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.665348 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.857531 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.39932 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.67093 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.737932 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.863947 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.932484 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.624318 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.51787 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.861357 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.30848 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.839361 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.760349 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.965163 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.04343 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.02337 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 1.01467 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.08557 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.842576 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.12235 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.01092 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.12066 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.03514 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.05894 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.20635 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.45465 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.41616 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.915866 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.86201 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.870391 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.00979 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.986978 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.20681 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.48436 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.807307 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.916703 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.15666 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.919481 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.625095 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.09739 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.07746 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.37104 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.846964 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.04495 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.688574 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.876945 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.22206 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.762157 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.29371 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.927147 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.08538 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.11638 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.871205 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   0.996202 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.17351 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.16883 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.926374 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.948856 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.06814 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.811207 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.964012 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.675404 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.09203 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.518 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.43776 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.805124 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.774744 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.666556 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.949687 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.794592 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.35116 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.37339 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.737476 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.00901 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.99637 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.990373 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.639413 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.16648 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.47277 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.99044 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.837635 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.918876 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.762388 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.688707 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.33973 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.790805 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.34893 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.992774 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.965125 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.26247 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.00383 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.03484 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.3679 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.24033 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.878981 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.926806 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.05388 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.582492 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.05973 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.569837 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.886871 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.73238 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.13494 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.664756 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.816577 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.625312 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.952827 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.661861 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.84169 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.46649 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.942991 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.43835 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.895847 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.07685 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.611848 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.11906 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.47614 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.80232 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.696687 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.859596 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.842049 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.61324 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.77763 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.975808 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.56822 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.07526 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.10343 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.15983 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.99372 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 0.997751 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.985641 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.19479 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.967029 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.06864 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.0498 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.06858 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.955953 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.897851 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.16706 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.15828 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.27049 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.884164 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.75461 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.682175 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.994669 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.919927 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.05637 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.34381 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.600895 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.863262 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.02449 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.977195 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.695009 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.36647 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.18957 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.60631 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.864949 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.01847 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.784329 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.833756 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.07752 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.729472 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.3263 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.920318 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.07706 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.22047 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.889088 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.02025 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.10131 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.10354 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.885441 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.954501 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.04248 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.949442 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.932572 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.83914 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.12137 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.29954 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.26094 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.803765 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.826221 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.774798 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.01959 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.922217 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.21989 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.37837 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.792004 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.03559 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.08388 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.05697 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.753026 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.19133 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.21117 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.48115 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.899993 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.05792 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.842173 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.807923 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.988478 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.831668 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.15605 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.978732 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.05519 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.02451 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.917169 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.263001027341024 z_k1_s0000
                                 - 0.195000115858447 z_k1_s0001
                                 - 0.210000197970691 z_k1_s0002
                                 - 0.319000350572658 z_k1_s0003
                                 - 0.194000221404382 z_k1_s0004
                                 - 0.502000305581239 z_k1_s0005
                                 - 0.353001044214126 z_k1_s0006
                                 - 0.342999720671674 z_k1_s0007
                                 - 0.912001984637974 z_k1_s0010
                                 - 0.724999251031516 z_k1_s0011
                                 - 0.795999870871049 z_k1_s0012
                                 - 0.734000666617936 z_k1_s0013
                                 - 0.93099876864852 z_k1_s0014
                                 - 0.460000256489579 z_k1_s0015
                                 - 0.331000254967995 z_k1_s0016
                                 - 0.695001977749725 z_k1_s0017
                                 - 0.219000223856941 z_k1_s0020
                                 - 0.344000369291257 z_k1_s0021
                                 - 0.117999843322174 z_k1_s0022
                                 - 0.206999761207521 z_k1_s0023
                                 - 0.160999986183522 z_k1_s0024
                                 - 0.148999910333866 z_k1_s0025
                                 - 0.345000246678807 z_k1_s0026
                                 - 0.391000408502569 z_k1_s0027
                                 - 0.74099987025958 z_k1_s0030
                                 - 0.744002938335626 z_k1_s0031
                                 - 0.742998087640964 z_k1_s0032
                                 - 0.431000559954392 z_k1_s0033
                                 - 0.699002853226681 z_k1_s0034
                                 - 0.8690006227476 z_k1_s0035
                                 - 0.520000473080797 z_k1_s0036
                                 - 0.995003157854235 z_k1_s0037
                                 - 0.288999786777384 z_k1_s0040
                                 - 0.432999271192434 z_k1_s0041
                                 - 0.160999903319579 z_k1_s0042
                                 - 0.423999007666509 z_k1_s0043
                                 - 0.226999943043129 z_k1_s0044
                                 - 0.208999889191254 z_k1_s0045
                                 - 0.418999649929842 z_k1_s0046
                                 - 0.204000058254613 z_k1_s0047
                                 - 0.407000607101261 z_k1_s0050
                                 - 0.492000315350953 z_k1_s0051
                                 - 0.519998733468651 z_k1_s0052
                                 - 0.168999955502274 z_k1_s0053
                                 - 0.500000702527352 z_k1_s0054
                                 - 0.154000110646435 z_k1_s0055
                                 - 0.256000054246035 z_k1_s0056
                                 - 0.438999218807159 z_k1_s0057
                                 - 0.699999660082033 z_k1_s0060
                                 - 0.56800033075515 z_k1_s0061
                                 - 0.565999983507764 z_k1_s0062
                                 - 0.748000131019099 z_k1_s0063
                                 - 1.07800159953638 z_k1_s0064
                                 - 2.00099987640095 z_k1_s0065
                                 - 0.748000131019099 z_k1_s0066
                                 - 1.45400240544333 z_k1_s0067
                                 - 0.746999180903385 z_k1_s0070
                                 - 0.195000001360384 z_k1_s0071
                                 - 0.611999866884857 z_k1_s0072
                                 - 0.7890014480432 z_k1_s0073
                                 - 0.717998997521532 z_k1_s0074
                                 - 0.258000276569202 z_k1_s0075
                                 - 0.896000738968949 z_k1_s0076
                                 - 0.249999887256869 z_k1_s0077
                                 - 1.09600153672975 z_k1_s0100
                                 - 0.882000538993576 z_k1_s0101
                                 - 0.836000963209731 z_k1_s0102
                                 - 1.73300469650534 z_k1_s0103
                                 - 1.8689994270801 z_k1_s0104
                                 - 1.14900124872001 z_k1_s0105
                                 - 1.23799902262663 z_k1_s0106
                                 - 0.432000754017696 z_k1_s0107
                                 - 0.861998515249275 z_k1_s0110
                                 - 0.581999407170085 z_k1_s0111
                                 - 0.585999459099702 z_k1_s0112
                                 - 0.697999102331378 z_k1_s0113
                                 - 1.03600258498525 z_k1_s0114
                                 - 1.77000034388557 z_k1_s0115
                                 - 2.11899536805723 z_k1_s0116
                                 - 1.40099600607776 z_k1_s0117
                                 - 0.475999550986462 z_k1_s0120
                                 - 0.784998971902248 z_k1_s0121
                                 - 1.13199894520141 z_k1_s0122
                                 - 0.796999581370373 z_k1_s0123
                                 - 0.523997339462267 z_k1_s0124
                                 - 0.317999619717063 z_k1_s0125
                                 - 0.673997208819338 z_k1_s0126
                                 - 0.459999427161806 z_k1_s0127
                                 - 0.399999841173772 z_k1_s0130
                                 - 1.40899722903242 z_k1_s0131
                                 - 0.558000110651291 z_k1_s0132
                                 - 0.529999691845283 z_k1_s0133
                                 - 1.241001230194 z_k1_s0134
                                 - 0.892000779253098 z_k1_s0135
                                 - 0.803996671292377 z_k1_s0136
                                 - 0.680002726434321 z_k1_s0137
                                 - 0.265999741312265 z_k1_s0140
                                 - 0.382999816600036 z_k1_s0141
                                 - 0.330999489027622 z_k1_s0142
                                 - 0.256000166994611 z_k1_s0143
                                 - 0.54000136555919 z_k1_s0144
                                 - 0.197000110929833 z_k1_s0145
                                 - 0.26800087379609 z_k1_s0146
                                 - 0.405999605160825 z_k1_s0147
                                 - 0.358000178348582 z_k1_s0150
                                 - 0.466000407928681 z_k1_s0151
                                 - 0.520999505177338 z_k1_s0152
                                 - 0.157000188074708 z_k1_s0153
                                 - 0.249000179709514 z_k1_s0154
                                 - 0.244000023414072 z_k1_s0155
                                 - 0.411001310680025 z_k1_s0156
                                 - 0.216000196960079 z_k1_s0157
                                 - 1.38199562427538 z_k1_s0160
                                 - 0.412998436386441 z_k1_s0161
                                 - 0.617996892966178 z_k1_s0162
                                 - 0.317998804977425 z_k1_s0163
                                 - 0.378998895224277 z_k1_s0164
                                 - 0.911995619289241 z_k1_s0165
                                 - 0.306999185659514 z_k1_s0166
                                 - 0.934999225921742 z_k1_s0167
                                 - 0.223000212365708 z_k1_s0170
                                 - 0.265999292319322 z_k1_s0171
                                 - 0.264999215994787 z_k1_s0172
                                 - 0.149000026951938 z_k1_s0173
                                 - 0.228000593988379 z_k1_s0174
                                 - 0.50300057322851 z_k1_s0175
                                 - 0.116000029443122 z_k1_s0176
                                 - 0.350999476902697 z_k1_s0177
                                 - 0.295999952658754 z_k1_s1000
                                 - 0.321999682757382 z_k1_s1001
                                 - 0.541998407437229 z_k1_s1002
                                 - 0.600998601737282 z_k1_s1003
                                 - 0.265000072772098 z_k1_s1004
                                 - 0.277000956662923 z_k1_s1005
                                 - 0.153999893875231 z_k1_s1006
                                 - 0.164999961209113 z_k1_s1007
                                 - 0.187999046124796 z_k1_s1010
                                 - 0.141999569408405 z_k1_s1011
                                 - 0.151999660163614 z_k1_s1012
                                 - 0.286000032845588 z_k1_s1013
                                 - 0.181999283630916 z_k1_s1014
                                 - 0.119999629266275 z_k1_s1015
                                 - 0.221999322252589 z_k1_s1016
                                 - 0.329999913129847 z_k1_s1017
                                 - 0.265000013537562 z_k1_s1020
                                 - 0.0750000688125875 z_k1_s1021
                                 - 0.127999841919853 z_k1_s1022
                                 - 0.1080001679702 z_k1_s1023
                                 - 0.111000124802654 z_k1_s1024
                                 - 0.202000001655038 z_k1_s1025
                                 - 0.151000046702577 z_k1_s1026
                                 - 0.1080001679702 z_k1_s1027
                                 - 0.241999963466264 z_k1_s1030
                                 - 0.235000732375481 z_k1_s1031
                                 - 0.108000222134082 z_k1_s1032
                                 - 0.175999528883851 z_k1_s1033
                                 - 0.225999858112951 z_k1_s1034
                                 - 0.0810002073506263 z_k1_s1035
                                 - 0.364000766785269 z_k1_s1036
                                 - 0.197999673744655 z_k1_s1037
                                 - 0.0600000572750695 z_k1_s1040
                                 - 0.109999892604185 z_k1_s1041
                                 - 0.072000097050098 z_k1_s1042
                                 - 0.0990003989440214 z_k1_s1043
                                 - 0.0620000167042167 z_k1_s1044
                                 - 0.132999992439668 z_k1_s1045
                                 - 0.0530000576729817 z_k1_s1046
                                 - 0.118999993235493 z_k1_s1047
                                 - 0.202999802874944 z_k1_s1050
                                 - 0.124000108229145 z_k1_s1051
                                 - 0.135000231749529 z_k1_s1052
                                 - 0.0770000574422865 z_k1_s1053
                                 - 0.15399971128437 z_k1_s1054
                                 - 0.126999866734566 z_k1_s1055
                                 - 0.0480000711887158 z_k1_s1056
                                 - 0.140999748760372 z_k1_s1057
                                 - 0.1840004240006 z_k1_s1060
                                 - 0.629000599250138 z_k1_s1061
                                 - 0.592000219999356 z_k1_s1062
                                 - 0.261000628450912 z_k1_s1063
                                 - 0.269000710451081 z_k1_s1064
                                 - 0.611000414749757 z_k1_s1065
                                 - 0.204000336600365 z_k1_s1066
                                 - 0.174000321500388 z_k1_s1067
                                 - 0.326000257390811 z_k1_s1070
                                 - 0.219999944374215 z_k1_s1071
                                 - 0.0729999208605475 z_k1_s1072
                                 - 0.162999461195544 z_k1_s1073
                                 - 0.122000017698418 z_k1_s1074
                                 - 0.174999773706799 z_k1_s1075
                                 - 0.10800000910188 z_k1_s1076
                                 - 0.147999404306406 z_k1_s1077
                                 - 0.106999974319855 z_k1_s1100
                                 - 0.168999631094243 z_k1_s1101
                                 - 0.281000198363142 z_k1_s1102
                                 - 0.197999389601683 z_k1_s1103
                                 - 0.25799981299564 z_k1_s1104
                                 - 0.171000537473318 z_k1_s1105
                                 - 0.182000503562301 z_k1_s1106
                                 - 0.307000270300614 z_k1_s1107
                                 - 0.29599949081609 z_k1_s1110
                                 - 0.132999606475298 z_k1_s1111
                                 - 0.0889997106376842 z_k1_s1112
                                 - 0.123999583315466 z_k1_s1113
                                 - 0.156999618068333 z_k1_s1114
                                 - 0.147999594908502 z_k1_s1115
                                 - 0.241998749858931 z_k1_s1116
                                 - 0.315999375060667 z_k1_s1117
                                 - 0.239999691648785 z_k1_s1120
                                 - 0.150999488246182 z_k1_s1121
                                 - 0.190000268387835 z_k1_s1122
                                 - 0.113999792033267 z_k1_s1123
                                 - 0.243999645509661 z_k1_s1124
                                 - 0.125999776615707 z_k1_s1125
                                 - 0.0889998344031699 z_k1_s1126
                                 - 0.0759998613555116 z_k1_s1127
                                 - 0.298000783448984 z_k1_s1130
                                 - 0.232000043047734 z_k1_s1131
                                 - 0.667999391154944 z_k1_s1132
                                 - 0.175000039428245 z_k1_s1133
                                 - 0.168999996573577 z_k1_s1134
                                 - 0.215999928768619 z_k1_s1135
                                 - 0.32500030379523 z_k1_s1136
                                 - 0.607000972965099 z_k1_s1137
                                 - 0.0900000602363676 z_k1_s1140
                                 - 0.113000044874576 z_k1_s1141
                                 - 0.0329999990200209 z_k1_s1142
                                 - 0.133000212038084 z_k1_s1143
                                 - 0.0719997713893271 z_k1_s1144
                                 - 0.044999960918242 z_k1_s1145
                                 - 0.110000227399875 z_k1_s1146
                                 - 0.0959996951857694 z_k1_s1147
                                 - 0.0949996371643697 z_k1_s1150
                                 - 0.120000122575637 z_k1_s1151
                                 - 0.114999516093742 z_k1_s1152
                                 - 0.152999880359132 z_k1_s1153
                                 - 0.0930002435710144 z_k1_s1154
                                 - 0.10999975861125 z_k1_s1155
                                 - 0.0509999884196906 z_k1_s1156
                                 - 0.112000001204008 z_k1_s1157
                                 - 0.0889999222694309 z_k1_s1160
                                 - 0.167000292242158 z_k1_s1161
                                 - 0.0629999266783813 z_k1_s1162
                                 - 0.098999611066329 z_k1_s1163
                                 - 0.144999846450226 z_k1_s1164
                                 - 0.233999864519551 z_k1_s1165
                                 - 0.10499955846432 z_k1_s1166
                                 - 0.0559999241807951 z_k1_s1167
                                 - 0.243999208963942 z_k1_s1170
                                 - 0.342998216193591 z_k1_s1171
                                 - 0.0829996241807508 z_k1_s1172
                                 - 0.201999583715172 z_k1_s1173
                                 - 0.2619986106436 z_k1_s1174
                                 - 0.10999964589685 z_k1_s1175
                                 - 0.09399965005021 z_k1_s1176
                                 - 0.193999509192134 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.799377 w_k1_ADM_s000
                                 + 0.912797 w_k1_AXP_s000
                                 + 0.650961 w_k1_DIS_s000
                                 + 0.763181 w_k1_FDX_s000
                                 + 0.758437 w_k1_IBM_s000
                                 + 0.777123 w_k1_INTC_s000
                                 + 0.684332 w_k1_KO_s000
                                 + 0.663269 w_k1_MAT_s000
                                 + 0.681658 w_k1_MSFT_s000
                                 + 0.823433 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.873312 w_k1_ADM_s000
                                 + 1.29454 w_k1_AXP_s000
                                 + 0.970022 w_k1_DIS_s000
                                 + 1.0873 w_k1_FDX_s000 + 1.06626 w_k1_IBM_s000
                                 + 0.943207 w_k1_INTC_s000
                                 + 1.0273 w_k1_KO_s000 + 1.54726 w_k1_MAT_s000
                                 + 1.0512 w_k1_MSFT_s000 + 1.12784 w_k1_PG_s000
                                 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.22714 w_k1_ADM_s000
                                 + 1.48102 w_k1_AXP_s000
                                 + 1.41773 w_k1_DIS_s000
                                 + 1.11117 w_k1_FDX_s000
                                 + 1.57209 w_k1_IBM_s000
                                 + 0.929744 w_k1_INTC_s000
                                 + 1.37629 w_k1_KO_s000 + 1.77215 w_k1_MAT_s000
                                 + 1.05323 w_k1_MSFT_s000 + 1.0943 w_k1_PG_s000
                                 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.08878 w_k1_ADM_s000
                                 + 1.13908 w_k1_AXP_s000
                                 + 0.954387 w_k1_DIS_s000
                                 + 1.07861 w_k1_FDX_s000
                                 + 0.754909 w_k1_IBM_s000
                                 + 1.00116 w_k1_INTC_s000
                                 + 0.923219 w_k1_KO_s000
                                 + 0.63816 w_k1_MAT_s000
                                 + 1.00474 w_k1_MSFT_s000
                                 + 0.93468 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.954582 w_k1_ADM_s000
                                 + 1.02757 w_k1_AXP_s000
                                 + 0.858015 w_k1_DIS_s000
                                 + 0.921961 w_k1_FDX_s000
                                 + 0.888189 w_k1_IBM_s000
                                 + 0.927623 w_k1_INTC_s000
                                 + 0.856378 w_k1_KO_s000
                                 + 1.05433 w_k1_MAT_s000
                                 + 0.960552 w_k1_MSFT_s000
                                 + 0.970556 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.869367 w_k1_ADM_s000
                                 + 0.939005 w_k1_AXP_s000
                                 + 1.14025 w_k1_DIS_s000
                                 + 0.947891 w_k1_FDX_s000
                                 + 1.17542 w_k1_IBM_s000
                                 + 0.892609 w_k1_INTC_s000
                                 + 1.00342 w_k1_KO_s000 + 0.95756 w_k1_MAT_s000
                                 + 0.860064 w_k1_MSFT_s000
                                 + 0.979707 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.30237 w_k1_ADM_s000
                                 + 1.17727 w_k1_AXP_s000 + 1.3689 w_k1_DIS_s000
                                 + 1.28299 w_k1_FDX_s000
                                 + 1.07714 w_k1_IBM_s000
                                 + 1.11097 w_k1_INTC_s000
                                 + 1.27653 w_k1_KO_s000
                                 + 0.778152 w_k1_MAT_s000
                                 + 1.19276 w_k1_MSFT_s000 + 1.134 w_k1_PG_s000
                                 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.899762 w_k1_ADM_s000
                                 + 0.765328 w_k1_AXP_s000
                                 + 0.739408 w_k1_DIS_s000
                                 + 0.888402 w_k1_FDX_s000
                                 + 0.831963 w_k1_IBM_s000
                                 + 0.774972 w_k1_INTC_s000
                                 + 0.804297 w_k1_KO_s000
                                 + 0.665628 w_k1_MAT_s000
                                 + 0.898135 w_k1_MSFT_s000
                                 + 0.838713 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.1526 w_k1_ADM_s001
                                 + 1.23757 w_k1_AXP_s001
                                 + 1.37033 w_k1_DIS_s001
                                 + 1.08728 w_k1_FDX_s001
                                 + 1.41728 w_k1_IBM_s001
                                 + 1.00229 w_k1_INTC_s001
                                 + 1.39056 w_k1_KO_s001 + 1.22137 w_k1_MAT_s001
                                 + 1.04362 w_k1_MSFT_s001
                                 + 1.11951 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.21338 w_k1_ADM_s001
                                 + 1.19158 w_k1_AXP_s001
                                 + 0.960334 w_k1_DIS_s001
                                 + 1.0361 w_k1_FDX_s001
                                 + 0.934536 w_k1_IBM_s001
                                 + 1.03456 w_k1_INTC_s001
                                 + 1.04033 w_k1_KO_s001 + 1.03059 w_k1_MAT_s001
                                 + 0.957128 w_k1_MSFT_s001
                                 + 1.14531 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.12765 w_k1_ADM_s001
                                 + 1.19657 w_k1_AXP_s001
                                 + 1.45904 w_k1_DIS_s001
                                 + 1.22363 w_k1_FDX_s001
                                 + 0.912289 w_k1_IBM_s001
                                 + 1.13797 w_k1_INTC_s001
                                 + 1.13405 w_k1_KO_s001
                                 + 0.891846 w_k1_MAT_s001
                                 + 1.23046 w_k1_MSFT_s001
                                 + 1.31184 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.795679 w_k1_ADM_s001
                                 + 0.672001 w_k1_AXP_s001
                                 + 0.985811 w_k1_DIS_s001
                                 + 0.871882 w_k1_FDX_s001
                                 + 0.846615 w_k1_IBM_s001
                                 + 0.932529 w_k1_INTC_s001
                                 + 0.969644 w_k1_KO_s001
                                 + 0.646306 w_k1_MAT_s001
                                 + 0.991097 w_k1_MSFT_s001
                                 + 0.973617 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.87042 w_k1_ADM_s001
                                 + 0.804988 w_k1_AXP_s001
                                 + 0.778735 w_k1_DIS_s001
                                 + 0.893918 w_k1_FDX_s001
                                 + 0.626837 w_k1_IBM_s001
                                 + 0.888314 w_k1_INTC_s001
                                 + 0.75807 w_k1_KO_s001
                                 + 0.611957 w_k1_MAT_s001
                                 + 0.899652 w_k1_MSFT_s001
                                 + 0.921082 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.980172 w_k1_ADM_s001
                                 + 1.12899 w_k1_AXP_s001
                                 + 0.81501 w_k1_DIS_s001
                                 + 0.909791 w_k1_FDX_s001
                                 + 1.1594 w_k1_IBM_s001
                                 + 0.989616 w_k1_INTC_s001
                                 + 1.14439 w_k1_KO_s001 + 1.32016 w_k1_MAT_s001
                                 + 0.964702 w_k1_MSFT_s001
                                 + 1.16324 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.38719 w_k1_ADM_s001
                                 + 1.16469 w_k1_AXP_s001
                                 + 1.17616 w_k1_DIS_s001
                                 + 1.30082 w_k1_FDX_s001
                                 + 0.729702 w_k1_IBM_s001
                                 + 1.0131 w_k1_INTC_s001 + 1.00408 w_k1_KO_s001
                                 + 0.874024 w_k1_MAT_s001
                                 + 1.17059 w_k1_MSFT_s001
                                 + 1.07773 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.882078 w_k1_ADM_s001
                                 + 0.844308 w_k1_AXP_s001
                                 + 0.891163 w_k1_DIS_s001
                                 + 0.784593 w_k1_FDX_s001
                                 + 1.34392 w_k1_IBM_s001
                                 + 0.864918 w_k1_INTC_s001
                                 + 1.06783 w_k1_KO_s001 + 1.04082 w_k1_MAT_s001
                                 + 0.892667 w_k1_MSFT_s001
                                 + 0.963121 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.599233 w_k1_ADM_s002
                                 + 0.937237 w_k1_AXP_s002
                                 + 0.756692 w_k1_DIS_s002
                                 + 0.834131 w_k1_FDX_s002
                                 + 1.01109 w_k1_IBM_s002
                                 + 0.848012 w_k1_INTC_s002
                                 + 0.97958 w_k1_KO_s002
                                 + 0.949516 w_k1_MAT_s002
                                 + 0.893226 w_k1_MSFT_s002
                                 + 0.882833 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.909428 w_k1_ADM_s002
                                 + 1.03321 w_k1_AXP_s002
                                 + 1.07356 w_k1_DIS_s002
                                 + 0.961259 w_k1_FDX_s002
                                 + 1.33628 w_k1_IBM_s002
                                 + 0.969043 w_k1_INTC_s002
                                 + 1.30294 w_k1_KO_s002 + 1.57978 w_k1_MAT_s002
                                 + 1.05382 w_k1_MSFT_s002
                                 + 1.11101 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.719601 w_k1_ADM_s002
                                 + 0.770851 w_k1_AXP_s002
                                 + 0.610845 w_k1_DIS_s002
                                 + 0.736665 w_k1_FDX_s002
                                 + 0.674161 w_k1_IBM_s002
                                 + 0.858336 w_k1_INTC_s002
                                 + 0.747156 w_k1_KO_s002
                                 + 0.634653 w_k1_MAT_s002
                                 + 0.768157 w_k1_MSFT_s002
                                 + 0.781381 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.39055 w_k1_ADM_s002
                                 + 1.66406 w_k1_AXP_s002
                                 + 1.78482 w_k1_DIS_s002
                                 + 1.31156 w_k1_FDX_s002
                                 + 1.42495 w_k1_IBM_s002
                                 + 1.18649 w_k1_INTC_s002
                                 + 1.61479 w_k1_KO_s002 + 1.29723 w_k1_MAT_s002
                                 + 1.27907 w_k1_MSFT_s002
                                 + 1.31034 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.88981 w_k1_ADM_s002
                                 + 1.00942 w_k1_AXP_s002
                                 + 0.766429 w_k1_DIS_s002
                                 + 0.953377 w_k1_FDX_s002
                                 + 0.509356 w_k1_IBM_s002
                                 + 0.840797 w_k1_INTC_s002
                                 + 0.71319 w_k1_KO_s002
                                 + 0.705514 w_k1_MAT_s002
                                 + 0.995289 w_k1_MSFT_s002
                                 + 0.870048 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.00804 w_k1_ADM_s002
                                 + 0.795282 w_k1_AXP_s002
                                 + 0.74975 w_k1_DIS_s002
                                 + 0.903225 w_k1_FDX_s002
                                 + 0.852871 w_k1_IBM_s002
                                 + 0.770639 w_k1_INTC_s002
                                 + 0.972805 w_k1_KO_s002
                                 + 0.779721 w_k1_MAT_s002
                                 + 1.0186 w_k1_MSFT_s002
                                 + 0.996716 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.18832 w_k1_ADM_s002
                                 + 1.20259 w_k1_AXP_s002
                                 + 1.13427 w_k1_DIS_s002
                                 + 1.17817 w_k1_FDX_s002
                                 + 0.923696 w_k1_IBM_s002
                                 + 1.14208 w_k1_INTC_s002
                                 + 1.19232 w_k1_KO_s002 + 1.32429 w_k1_MAT_s002
                                 + 1.15456 w_k1_MSFT_s002
                                 + 1.13503 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.896372 w_k1_ADM_s002
                                 + 1.16622 w_k1_AXP_s002
                                 + 1.00527 w_k1_DIS_s002
                                 + 1.03815 w_k1_FDX_s002
                                 + 0.899914 w_k1_IBM_s002
                                 + 0.943757 w_k1_INTC_s002
                                 + 1.17679 w_k1_KO_s002
                                 + 0.946163 w_k1_MAT_s002
                                 + 1.03083 w_k1_MSFT_s002 + 1.0161 w_k1_PG_s002
                                 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.18303 w_k1_ADM_s003
                                 + 1.46031 w_k1_AXP_s003
                                 + 1.39751 w_k1_DIS_s003
                                 + 1.38544 w_k1_FDX_s003
                                 + 0.864628 w_k1_IBM_s003
                                 + 1.54094 w_k1_INTC_s003
                                 + 1.12809 w_k1_KO_s003
                                 + 0.948441 w_k1_MAT_s003
                                 + 1.33626 w_k1_MSFT_s003
                                 + 1.14245 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.835819 w_k1_ADM_s003
                                 + 0.76509 w_k1_AXP_s003
                                 + 0.697486 w_k1_DIS_s003
                                 + 0.715874 w_k1_FDX_s003
                                 + 1.0149 w_k1_IBM_s003
                                 + 0.779285 w_k1_INTC_s003
                                 + 0.814189 w_k1_KO_s003
                                 + 0.965106 w_k1_MAT_s003
                                 + 0.873652 w_k1_MSFT_s003
                                 + 0.860172 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.1146 w_k1_ADM_s003
                                 + 1.43964 w_k1_AXP_s003
                                 + 1.39461 w_k1_DIS_s003
                                 + 1.18694 w_k1_FDX_s003
                                 + 1.46931 w_k1_IBM_s003
                                 + 1.00975 w_k1_INTC_s003
                                 + 1.27542 w_k1_KO_s003 + 1.71552 w_k1_MAT_s003
                                 + 1.11267 w_k1_MSFT_s003
                                 + 1.29455 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.12148 w_k1_ADM_s003
                                 + 1.04019 w_k1_AXP_s003
                                 + 1.15168 w_k1_DIS_s003
                                 + 1.14107 w_k1_FDX_s003
                                 + 0.755151 w_k1_IBM_s003
                                 + 0.988999 w_k1_INTC_s003
                                 + 0.909968 w_k1_KO_s003
                                 + 0.913206 w_k1_MAT_s003
                                 + 1.06766 w_k1_MSFT_s003
                                 + 1.07629 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.970422 w_k1_ADM_s003
                                 + 0.905121 w_k1_AXP_s003
                                 + 0.817971 w_k1_DIS_s003
                                 + 0.978208 w_k1_FDX_s003
                                 + 0.721884 w_k1_IBM_s003
                                 + 1.06969 w_k1_INTC_s003
                                 + 0.834479 w_k1_KO_s003
                                 + 0.692554 w_k1_MAT_s003
                                 + 0.974011 w_k1_MSFT_s003
                                 + 0.865329 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.946855 w_k1_ADM_s003
                                 + 1.12009 w_k1_AXP_s003
                                 + 0.949185 w_k1_DIS_s003
                                 + 0.842157 w_k1_FDX_s003
                                 + 1.53886 w_k1_IBM_s003
                                 + 0.927172 w_k1_INTC_s003
                                 + 1.23656 w_k1_KO_s003 + 1.37278 w_k1_MAT_s003
                                 + 1.03637 w_k1_MSFT_s003
                                 + 0.958478 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.783343 w_k1_ADM_s003
                                 + 0.771255 w_k1_AXP_s003
                                 + 0.690759 w_k1_DIS_s003
                                 + 0.782083 w_k1_FDX_s003
                                 + 0.68911 w_k1_IBM_s003
                                 + 0.825961 w_k1_INTC_s003
                                 + 0.75193 w_k1_KO_s003
                                 + 0.733605 w_k1_MAT_s003
                                 + 0.913317 w_k1_MSFT_s003
                                 + 0.884402 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.937019 w_k1_ADM_s003
                                 + 1.04146 w_k1_AXP_s003
                                 + 1.14564 w_k1_DIS_s003
                                 + 0.986016 w_k1_FDX_s003
                                 + 0.952593 w_k1_IBM_s003
                                 + 1.03948 w_k1_INTC_s003
                                 + 1.00629 w_k1_KO_s003
                                 + 0.692573 w_k1_MAT_s003
                                 + 1.08453 w_k1_MSFT_s003
                                 + 1.03951 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.61018 w_k1_ADM_s004
                                 + 1.13807 w_k1_AXP_s004
                                 + 1.50994 w_k1_DIS_s004
                                 + 1.35061 w_k1_FDX_s004
                                 + 1.21166 w_k1_IBM_s004
                                 + 0.857144 w_k1_INTC_s004
                                 + 1.43391 w_k1_KO_s004 + 1.19464 w_k1_MAT_s004
                                 + 1.36011 w_k1_MSFT_s004
                                 + 1.11988 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.17487 w_k1_ADM_s004
                                 + 0.850153 w_k1_AXP_s004
                                 + 0.875177 w_k1_DIS_s004
                                 + 1.03043 w_k1_FDX_s004
                                 + 0.767335 w_k1_IBM_s004
                                 + 0.729484 w_k1_INTC_s004
                                 + 0.780058 w_k1_KO_s004
                                 + 0.825734 w_k1_MAT_s004
                                 + 0.861411 w_k1_MSFT_s004
                                 + 0.85764 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.816922 w_k1_ADM_s004
                                 + 0.630831 w_k1_AXP_s004
                                 + 0.489349 w_k1_DIS_s004
                                 + 0.748766 w_k1_FDX_s004
                                 + 0.703283 w_k1_IBM_s004
                                 + 0.570067 w_k1_INTC_s004
                                 + 0.598535 w_k1_KO_s004
                                 + 0.674576 w_k1_MAT_s004
                                 + 0.746912 w_k1_MSFT_s004
                                 + 0.676617 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.42942 w_k1_ADM_s004
                                 + 1.17071 w_k1_AXP_s004
                                 + 0.976779 w_k1_DIS_s004
                                 + 1.0683 w_k1_FDX_s004 + 1.12688 w_k1_IBM_s004
                                 + 0.807762 w_k1_INTC_s004
                                 + 0.994531 w_k1_KO_s004
                                 + 1.29645 w_k1_MAT_s004
                                 + 1.01728 w_k1_MSFT_s004
                                 + 1.00423 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.86574 w_k1_ADM_s004
                                 + 1.98174 w_k1_AXP_s004
                                 + 1.75265 w_k1_DIS_s004
                                 + 1.53787 w_k1_FDX_s004
                                 + 1.46222 w_k1_IBM_s004
                                 + 1.23012 w_k1_INTC_s004
                                 + 1.26845 w_k1_KO_s004 + 1.92153 w_k1_MAT_s004
                                 + 1.48393 w_k1_MSFT_s004
                                 + 1.24073 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.58848 w_k1_ADM_s004
                                 + 1.13309 w_k1_AXP_s004
                                 + 0.99918 w_k1_DIS_s004
                                 + 1.24981 w_k1_FDX_s004
                                 + 0.842424 w_k1_IBM_s004
                                 + 0.919703 w_k1_INTC_s004
                                 + 0.920563 w_k1_KO_s004
                                 + 0.949396 w_k1_MAT_s004
                                 + 0.970291 w_k1_MSFT_s004
                                 + 0.944851 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.1915 w_k1_ADM_s004
                                 + 0.663459 w_k1_AXP_s004
                                 + 0.771689 w_k1_DIS_s004
                                 + 0.780054 w_k1_FDX_s004
                                 + 0.959079 w_k1_IBM_s004
                                 + 0.742014 w_k1_INTC_s004
                                 + 0.840538 w_k1_KO_s004
                                 + 0.737237 w_k1_MAT_s004
                                 + 0.748405 w_k1_MSFT_s004
                                 + 0.844529 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.22013 w_k1_ADM_s004
                                 + 0.756106 w_k1_AXP_s004
                                 + 0.93049 w_k1_DIS_s004
                                 + 0.917188 w_k1_FDX_s004
                                 + 1.3794 w_k1_IBM_s004
                                 + 0.704875 w_k1_INTC_s004
                                 + 1.09713 w_k1_KO_s004 + 1.17815 w_k1_MAT_s004
                                 + 0.920985 w_k1_MSFT_s004
                                 + 0.991543 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.11217 w_k1_ADM_s005
                                 + 1.50118 w_k1_AXP_s005
                                 + 1.32722 w_k1_DIS_s005
                                 + 1.30852 w_k1_FDX_s005
                                 + 0.963735 w_k1_IBM_s005
                                 + 1.31218 w_k1_INTC_s005 + 0.9948 w_k1_KO_s005
                                 + 1.21207 w_k1_MAT_s005
                                 + 1.35583 w_k1_MSFT_s005
                                 + 1.16711 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.778817 w_k1_ADM_s005
                                 + 0.911435 w_k1_AXP_s005
                                 + 1.02133 w_k1_DIS_s005
                                 + 0.945176 w_k1_FDX_s005
                                 + 1.00889 w_k1_IBM_s005
                                 + 1.21489 w_k1_INTC_s005
                                 + 0.913261 w_k1_KO_s005 + 1.0079 w_k1_MAT_s005
                                 + 1.19589 w_k1_MSFT_s005
                                 + 0.916087 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.738637 w_k1_ADM_s005
                                 + 0.800857 w_k1_AXP_s005
                                 + 0.705986 w_k1_DIS_s005
                                 + 0.732766 w_k1_FDX_s005
                                 + 0.915321 w_k1_IBM_s005
                                 + 0.732109 w_k1_INTC_s005
                                 + 0.767295 w_k1_KO_s005
                                 + 0.735966 w_k1_MAT_s005
                                 + 0.790712 w_k1_MSFT_s005
                                 + 0.772344 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.714927 w_k1_ADM_s005
                                 + 0.884988 w_k1_AXP_s005
                                 + 0.817179 w_k1_DIS_s005
                                 + 0.729093 w_k1_FDX_s005
                                 + 1.23654 w_k1_IBM_s005
                                 + 0.904372 w_k1_INTC_s005
                                 + 1.04896 w_k1_KO_s005 + 1.62241 w_k1_MAT_s005
                                 + 0.979835 w_k1_MSFT_s005
                                 + 0.932593 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.972719 w_k1_ADM_s005
                                 + 0.933469 w_k1_AXP_s005
                                 + 0.807487 w_k1_DIS_s005
                                 + 0.941511 w_k1_FDX_s005
                                 + 0.600339 w_k1_IBM_s005
                                 + 0.964534 w_k1_INTC_s005
                                 + 0.872282 w_k1_KO_s005
                                 + 0.732909 w_k1_MAT_s005
                                 + 1.11288 w_k1_MSFT_s005
                                 + 0.992601 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.74368 w_k1_ADM_s005
                                 + 0.941799 w_k1_AXP_s005
                                 + 1.32778 w_k1_DIS_s005
                                 + 0.912719 w_k1_FDX_s005
                                 + 1.55418 w_k1_IBM_s005
                                 + 0.941447 w_k1_INTC_s005
                                 + 1.35775 w_k1_KO_s005 + 1.419 w_k1_MAT_s005
                                 + 1.21824 w_k1_MSFT_s005
                                 + 1.09776 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.03449 w_k1_ADM_s005
                                 + 1.48017 w_k1_AXP_s005
                                 + 1.79344 w_k1_DIS_s005
                                 + 1.36892 w_k1_FDX_s005
                                 + 1.52098 w_k1_IBM_s005
                                 + 1.3422 w_k1_INTC_s005 + 1.52996 w_k1_KO_s005
                                 + 1.67415 w_k1_MAT_s005
                                 + 1.57714 w_k1_MSFT_s005 + 1.437 w_k1_PG_s005
                                 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.746458 w_k1_ADM_s005
                                 + 1.09742 w_k1_AXP_s005
                                 + 0.970456 w_k1_DIS_s005
                                 + 0.99725 w_k1_FDX_s005
                                 + 1.16418 w_k1_IBM_s005
                                 + 1.03282 w_k1_INTC_s005
                                 + 0.970222 w_k1_KO_s005
                                 + 1.57738 w_k1_MAT_s005
                                 + 1.0496 w_k1_MSFT_s005 + 1.11799 w_k1_PG_s005
                                 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.23819 w_k1_ADM_s006
                                 + 1.13282 w_k1_AXP_s006
                                 + 1.26765 w_k1_DIS_s006
                                 + 1.17898 w_k1_FDX_s006
                                 + 0.874149 w_k1_IBM_s006
                                 + 1.18242 w_k1_INTC_s006
                                 + 1.07954 w_k1_KO_s006
                                 + 0.653198 w_k1_MAT_s006
                                 + 1.18901 w_k1_MSFT_s006
                                 + 1.17901 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.954945 w_k1_ADM_s006
                                 + 0.913883 w_k1_AXP_s006
                                 + 0.901763 w_k1_DIS_s006
                                 + 0.858009 w_k1_FDX_s006
                                 + 1.51193 w_k1_IBM_s006
                                 + 1.05681 w_k1_INTC_s006
                                 + 0.952168 w_k1_KO_s006
                                 + 1.05858 w_k1_MAT_s006
                                 + 0.903681 w_k1_MSFT_s006
                                 + 1.05156 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.28987 w_k1_ADM_s006
                                 + 1.16059 w_k1_AXP_s006 + 1.6895 w_k1_DIS_s006
                                 + 1.17568 w_k1_FDX_s006
                                 + 1.75654 w_k1_IBM_s006
                                 + 1.18872 w_k1_INTC_s006
                                 + 1.65972 w_k1_KO_s006 + 1.23335 w_k1_MAT_s006
                                 + 1.23782 w_k1_MSFT_s006
                                 + 1.42656 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.90111 w_k1_ADM_s006
                                 + 0.955175 w_k1_AXP_s006
                                 + 0.875881 w_k1_DIS_s006
                                 + 1.10325 w_k1_FDX_s006
                                 + 0.743423 w_k1_IBM_s006
                                 + 1.17546 w_k1_INTC_s006
                                 + 0.762231 w_k1_KO_s006
                                 + 0.53403 w_k1_MAT_s006
                                 + 0.882203 w_k1_MSFT_s006
                                 + 0.978095 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.12406 w_k1_ADM_s006
                                 + 1.26003 w_k1_AXP_s006
                                 + 1.06885 w_k1_DIS_s006
                                 + 1.03862 w_k1_FDX_s006
                                 + 1.28734 w_k1_IBM_s006
                                 + 1.08101 w_k1_INTC_s006
                                 + 1.15169 w_k1_KO_s006 + 1.0758 w_k1_MAT_s006
                                 + 0.979027 w_k1_MSFT_s006
                                 + 1.19602 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.849735 w_k1_ADM_s006
                                 + 0.817326 w_k1_AXP_s006
                                 + 0.825099 w_k1_DIS_s006
                                 + 0.824859 w_k1_FDX_s006
                                 + 0.921891 w_k1_IBM_s006
                                 + 0.848668 w_k1_INTC_s006
                                 + 0.756144 w_k1_KO_s006
                                 + 0.752186 w_k1_MAT_s006
                                 + 0.746939 w_k1_MSFT_s006
                                 + 0.890671 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.40473 w_k1_ADM_s006
                                 + 1.3381 w_k1_AXP_s006 + 1.57824 w_k1_DIS_s006
                                 + 1.36329 w_k1_FDX_s006
                                 + 1.15028 w_k1_IBM_s006
                                 + 1.4568 w_k1_INTC_s006 + 1.22964 w_k1_KO_s006
                                 + 0.858792 w_k1_MAT_s006
                                 + 1.28983 w_k1_MSFT_s006
                                 + 1.34089 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.0188 w_k1_ADM_s006
                                 + 0.754307 w_k1_AXP_s006
                                 + 0.808974 w_k1_DIS_s006
                                 + 0.892744 w_k1_FDX_s006
                                 + 0.624891 w_k1_IBM_s006
                                 + 0.979661 w_k1_INTC_s006
                                 + 0.742668 w_k1_KO_s006
                                 + 0.450919 w_k1_MAT_s006
                                 + 0.90438 w_k1_MSFT_s006
                                 + 0.880163 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.2258 w_k1_ADM_s007
                                 + 0.85687 w_k1_AXP_s007
                                 + 0.838147 w_k1_DIS_s007
                                 + 1.05786 w_k1_FDX_s007
                                 + 0.737851 w_k1_IBM_s007
                                 + 1.01131 w_k1_INTC_s007
                                 + 0.771426 w_k1_KO_s007
                                 + 0.722316 w_k1_MAT_s007
                                 + 0.942434 w_k1_MSFT_s007
                                 + 0.910216 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.934054 w_k1_ADM_s007
                                 + 1.24432 w_k1_AXP_s007
                                 + 0.824488 w_k1_DIS_s007
                                 + 1.0453 w_k1_FDX_s007 + 1.00316 w_k1_IBM_s007
                                 + 0.962431 w_k1_INTC_s007
                                 + 0.927614 w_k1_KO_s007
                                 + 1.33917 w_k1_MAT_s007
                                 + 1.14585 w_k1_MSFT_s007
                                 + 1.00735 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.09169 w_k1_ADM_s007
                                 + 0.932025 w_k1_AXP_s007
                                 + 0.970841 w_k1_DIS_s007
                                 + 0.984268 w_k1_FDX_s007
                                 + 1.04928 w_k1_IBM_s007
                                 + 0.874795 w_k1_INTC_s007
                                 + 0.965492 w_k1_KO_s007
                                 + 0.900548 w_k1_MAT_s007
                                 + 0.922935 w_k1_MSFT_s007
                                 + 0.965053 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.32865 w_k1_ADM_s007
                                 + 1.27213 w_k1_AXP_s007
                                 + 1.48827 w_k1_DIS_s007
                                 + 1.33763 w_k1_FDX_s007
                                 + 1.24443 w_k1_IBM_s007
                                 + 1.1323 w_k1_INTC_s007 + 1.15828 w_k1_KO_s007
                                 + 1.19259 w_k1_MAT_s007
                                 + 1.21024 w_k1_MSFT_s007
                                 + 1.18588 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.16656 w_k1_ADM_s007
                                 + 0.848934 w_k1_AXP_s007
                                 + 1.06393 w_k1_DIS_s007
                                 + 0.970707 w_k1_FDX_s007
                                 + 1.63043 w_k1_IBM_s007
                                 + 1.07391 w_k1_INTC_s007
                                 + 1.28889 w_k1_KO_s007 + 1.19831 w_k1_MAT_s007
                                 + 1.09998 w_k1_MSFT_s007
                                 + 1.16327 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.15657 w_k1_ADM_s007
                                 + 1.10878 w_k1_AXP_s007
                                 + 1.31935 w_k1_DIS_s007
                                 + 1.20186 w_k1_FDX_s007
                                 + 0.990236 w_k1_IBM_s007
                                 + 1.15989 w_k1_INTC_s007
                                 + 1.04563 w_k1_KO_s007 + 1.10435 w_k1_MAT_s007
                                 + 1.16155 w_k1_MSFT_s007 + 1.0304 w_k1_PG_s007
                                 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.03435 w_k1_ADM_s007
                                 + 0.59979 w_k1_AXP_s007
                                 + 0.600012 w_k1_DIS_s007
                                 + 0.741537 w_k1_FDX_s007
                                 + 0.691856 w_k1_IBM_s007
                                 + 0.757499 w_k1_INTC_s007
                                 + 0.661739 w_k1_KO_s007
                                 + 0.509076 w_k1_MAT_s007
                                 + 0.751228 w_k1_MSFT_s007
                                 + 0.850202 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.3442 w_k1_ADM_s007
                                 + 1.35105 w_k1_AXP_s007
                                 + 0.991784 w_k1_DIS_s007
                                 + 1.31408 w_k1_FDX_s007
                                 + 0.727002 w_k1_IBM_s007
                                 + 1.15729 w_k1_INTC_s007
                                 + 0.866551 w_k1_KO_s007
                                 + 0.740463 w_k1_MAT_s007
                                 + 1.01994 w_k1_MSFT_s007
                                 + 1.04798 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.87805 w_k1_ADM_s010
                                 + 1.13005 w_k1_AXP_s010
                                 + 1.11985 w_k1_DIS_s010
                                 + 0.975443 w_k1_FDX_s010
                                 + 1.59573 w_k1_IBM_s010
                                 + 1.0867 w_k1_INTC_s010 + 1.23166 w_k1_KO_s010
                                 + 1.36563 w_k1_MAT_s010
                                 + 1.1859 w_k1_MSFT_s010 + 1.17025 w_k1_PG_s010
                                 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.726796 w_k1_ADM_s010
                                 + 0.489045 w_k1_AXP_s010
                                 + 0.481935 w_k1_DIS_s010
                                 + 0.634163 w_k1_FDX_s010
                                 + 0.713324 w_k1_IBM_s010
                                 + 0.714313 w_k1_INTC_s010
                                 + 0.643431 w_k1_KO_s010
                                 + 0.62098 w_k1_MAT_s010
                                 + 0.693788 w_k1_MSFT_s010
                                 + 0.831091 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.817373 w_k1_ADM_s010
                                 + 0.896822 w_k1_AXP_s010
                                 + 1.43393 w_k1_DIS_s010
                                 + 1.07956 w_k1_FDX_s010
                                 + 1.30196 w_k1_IBM_s010
                                 + 1.08272 w_k1_INTC_s010 + 1.2212 w_k1_KO_s010
                                 + 1.14753 w_k1_MAT_s010
                                 + 1.25487 w_k1_MSFT_s010
                                 + 1.23431 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.797469 w_k1_ADM_s010
                                 + 0.84478 w_k1_AXP_s010
                                 + 0.891443 w_k1_DIS_s010
                                 + 0.846476 w_k1_FDX_s010
                                 + 0.856797 w_k1_IBM_s010
                                 + 0.978153 w_k1_INTC_s010
                                 + 0.845474 w_k1_KO_s010
                                 + 0.652761 w_k1_MAT_s010
                                 + 0.922102 w_k1_MSFT_s010
                                 + 0.869725 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.980445 w_k1_ADM_s010
                                 + 0.930746 w_k1_AXP_s010
                                 + 0.962021 w_k1_DIS_s010
                                 + 0.99091 w_k1_FDX_s010
                                 + 0.895111 w_k1_IBM_s010
                                 + 0.9444 w_k1_INTC_s010
                                 + 0.845826 w_k1_KO_s010
                                 + 0.667679 w_k1_MAT_s010
                                 + 1.04439 w_k1_MSFT_s010
                                 + 0.954352 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.02704 w_k1_ADM_s010
                                 + 1.15616 w_k1_AXP_s010
                                 + 1.60033 w_k1_DIS_s010
                                 + 1.28904 w_k1_FDX_s010
                                 + 1.28722 w_k1_IBM_s010
                                 + 1.15317 w_k1_INTC_s010
                                 + 1.28554 w_k1_KO_s010 + 1.25249 w_k1_MAT_s010
                                 + 1.25186 w_k1_MSFT_s010
                                 + 1.28562 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.06178 w_k1_ADM_s010
                                 + 1.38889 w_k1_AXP_s010 + 1.3506 w_k1_DIS_s010
                                 + 1.36659 w_k1_FDX_s010
                                 + 0.863895 w_k1_IBM_s010
                                 + 1.0574 w_k1_INTC_s010
                                 + 0.953132 w_k1_KO_s010
                                 + 0.885685 w_k1_MAT_s010
                                 + 1.26186 w_k1_MSFT_s010
                                 + 1.18272 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.87626 w_k1_ADM_s010
                                 + 0.884595 w_k1_AXP_s010
                                 + 0.823525 w_k1_DIS_s010
                                 + 0.767198 w_k1_FDX_s010
                                 + 1.27725 w_k1_IBM_s010
                                 + 0.699577 w_k1_INTC_s010
                                 + 0.846778 w_k1_KO_s010
                                 + 1.29663 w_k1_MAT_s010
                                 + 0.770042 w_k1_MSFT_s010
                                 + 0.938567 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.15694 w_k1_ADM_s011
                                 + 1.62888 w_k1_AXP_s011
                                 + 1.41175 w_k1_DIS_s011
                                 + 1.33069 w_k1_FDX_s011
                                 + 1.14647 w_k1_IBM_s011
                                 + 1.29959 w_k1_INTC_s011
                                 + 1.22946 w_k1_KO_s011 + 1.6587 w_k1_MAT_s011
                                 + 1.07477 w_k1_MSFT_s011
                                 + 1.18028 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.998559 w_k1_ADM_s011
                                 + 1.12078 w_k1_AXP_s011
                                 + 1.50993 w_k1_DIS_s011
                                 + 1.30725 w_k1_FDX_s011
                                 + 0.906649 w_k1_IBM_s011
                                 + 1.44079 w_k1_INTC_s011
                                 + 1.20531 w_k1_KO_s011
                                 + 0.754535 w_k1_MAT_s011
                                 + 1.14253 w_k1_MSFT_s011
                                 + 1.10807 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.694127 w_k1_ADM_s011
                                 + 0.756963 w_k1_AXP_s011
                                 + 0.638661 w_k1_DIS_s011
                                 + 0.807553 w_k1_FDX_s011
                                 + 0.662956 w_k1_IBM_s011
                                 + 0.849578 w_k1_INTC_s011
                                 + 0.665903 w_k1_KO_s011
                                 + 0.637191 w_k1_MAT_s011
                                 + 0.709487 w_k1_MSFT_s011
                                 + 0.822729 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.788829 w_k1_ADM_s011
                                 + 0.708277 w_k1_AXP_s011
                                 + 0.849147 w_k1_DIS_s011
                                 + 0.90905 w_k1_FDX_s011
                                 + 0.728553 w_k1_IBM_s011
                                 + 0.970368 w_k1_INTC_s011
                                 + 0.862042 w_k1_KO_s011
                                 + 0.677954 w_k1_MAT_s011
                                 + 0.915456 w_k1_MSFT_s011
                                 + 0.898251 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.0017 w_k1_ADM_s011
                                 + 1.15254 w_k1_AXP_s011
                                 + 1.10382 w_k1_DIS_s011
                                 + 1.12407 w_k1_FDX_s011
                                 + 0.943979 w_k1_IBM_s011
                                 + 1.22273 w_k1_INTC_s011
                                 + 1.16584 w_k1_KO_s011 + 1.12347 w_k1_MAT_s011
                                 + 1.07367 w_k1_MSFT_s011
                                 + 1.06367 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.949307 w_k1_ADM_s011
                                 + 0.941275 w_k1_AXP_s011
                                 + 0.921674 w_k1_DIS_s011
                                 + 0.941667 w_k1_FDX_s011
                                 + 1.03844 w_k1_IBM_s011
                                 + 0.931822 w_k1_INTC_s011
                                 + 0.96285 w_k1_KO_s011 + 1.12198 w_k1_MAT_s011
                                 + 0.853498 w_k1_MSFT_s011
                                 + 0.91585 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.951094 w_k1_ADM_s011
                                 + 1.17749 w_k1_AXP_s011
                                 + 1.33919 w_k1_DIS_s011 + 1.1373 w_k1_FDX_s011
                                 + 1.41823 w_k1_IBM_s011
                                 + 1.20916 w_k1_INTC_s011
                                 + 1.28029 w_k1_KO_s011 + 1.33834 w_k1_MAT_s011
                                 + 1.04035 w_k1_MSFT_s011
                                 + 1.14968 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.870673 w_k1_ADM_s011
                                 + 0.698656 w_k1_AXP_s011
                                 + 0.639372 w_k1_DIS_s011
                                 + 0.735696 w_k1_FDX_s011
                                 + 1.0239 w_k1_IBM_s011
                                 + 0.793516 w_k1_INTC_s011
                                 + 0.828636 w_k1_KO_s011
                                 + 0.843047 w_k1_MAT_s011
                                 + 0.715666 w_k1_MSFT_s011
                                 + 0.807943 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.04506 w_k1_ADM_s012
                                 + 1.03785 w_k1_AXP_s012
                                 + 1.32922 w_k1_DIS_s012
                                 + 1.18208 w_k1_FDX_s012
                                 + 0.901623 w_k1_IBM_s012
                                 + 1.38284 w_k1_INTC_s012
                                 + 1.11913 w_k1_KO_s012
                                 + 0.768864 w_k1_MAT_s012
                                 + 1.15002 w_k1_MSFT_s012 + 1.0952 w_k1_PG_s012
                                 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.01044 w_k1_ADM_s012
                                 + 1.07848 w_k1_AXP_s012
                                 + 1.28013 w_k1_DIS_s012 + 1.0513 w_k1_FDX_s012
                                 + 1.35369 w_k1_IBM_s012
                                 + 1.19999 w_k1_INTC_s012
                                 + 1.29323 w_k1_KO_s012 + 1.66284 w_k1_MAT_s012
                                 + 1.06937 w_k1_MSFT_s012
                                 + 1.24578 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.04097 w_k1_ADM_s012
                                 + 1.00757 w_k1_AXP_s012
                                 + 1.08311 w_k1_DIS_s012
                                 + 1.03547 w_k1_FDX_s012
                                 + 0.883125 w_k1_IBM_s012
                                 + 1.11371 w_k1_INTC_s012
                                 + 0.845266 w_k1_KO_s012 + 0.874 w_k1_MAT_s012
                                 + 0.978586 w_k1_MSFT_s012
                                 + 1.00893 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.708935 w_k1_ADM_s012
                                 + 0.7466 w_k1_AXP_s012
                                 + 0.804088 w_k1_DIS_s012
                                 + 0.867029 w_k1_FDX_s012
                                 + 0.7034 w_k1_IBM_s012
                                 + 0.924003 w_k1_INTC_s012
                                 + 0.805405 w_k1_KO_s012
                                 + 0.810084 w_k1_MAT_s012
                                 + 0.851231 w_k1_MSFT_s012
                                 + 0.86133 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.07605 w_k1_ADM_s012
                                 + 1.38172 w_k1_AXP_s012
                                 + 1.81269 w_k1_DIS_s012
                                 + 1.37883 w_k1_FDX_s012
                                 + 1.12627 w_k1_IBM_s012
                                 + 1.61015 w_k1_INTC_s012
                                 + 1.28214 w_k1_KO_s012 + 1.11888 w_k1_MAT_s012
                                 + 1.39073 w_k1_MSFT_s012
                                 + 1.36994 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.773326 w_k1_ADM_s012
                                 + 1.02112 w_k1_AXP_s012
                                 + 0.912381 w_k1_DIS_s012
                                 + 0.911158 w_k1_FDX_s012
                                 + 1.21759 w_k1_IBM_s012
                                 + 1.04045 w_k1_INTC_s012 + 1.0127 w_k1_KO_s012
                                 + 1.48385 w_k1_MAT_s012
                                 + 0.913941 w_k1_MSFT_s012
                                 + 1.02969 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.735541 w_k1_ADM_s012
                                 + 0.806397 w_k1_AXP_s012
                                 + 0.702305 w_k1_DIS_s012
                                 + 0.836379 w_k1_FDX_s012
                                 + 0.985215 w_k1_IBM_s012
                                 + 0.903803 w_k1_INTC_s012
                                 + 0.859962 w_k1_KO_s012
                                 + 1.11012 w_k1_MAT_s012
                                 + 0.768187 w_k1_MSFT_s012
                                 + 0.8789 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.892013 w_k1_ADM_s012
                                 + 0.786453 w_k1_AXP_s012
                                 + 0.966042 w_k1_DIS_s012
                                 + 0.839973 w_k1_FDX_s012
                                 + 1.20751 w_k1_IBM_s012
                                 + 1.09899 w_k1_INTC_s012
                                 + 1.18126 w_k1_KO_s012 + 1.10654 w_k1_MAT_s012
                                 + 0.891348 w_k1_MSFT_s012
                                 + 0.956151 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.70105 w_k1_ADM_s013
                                 + 0.590721 w_k1_AXP_s013
                                 + 0.520796 w_k1_DIS_s013
                                 + 0.64764 w_k1_FDX_s013
                                 + 0.743862 w_k1_IBM_s013
                                 + 0.721844 w_k1_INTC_s013
                                 + 0.756661 w_k1_KO_s013
                                 + 0.812361 w_k1_MAT_s013
                                 + 0.727878 w_k1_MSFT_s013
                                 + 0.789544 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.16721 w_k1_ADM_s013
                                 + 1.41682 w_k1_AXP_s013
                                 + 1.47604 w_k1_DIS_s013
                                 + 1.29528 w_k1_FDX_s013
                                 + 1.20828 w_k1_IBM_s013
                                 + 1.15538 w_k1_INTC_s013
                                 + 1.32012 w_k1_KO_s013 + 1.23534 w_k1_MAT_s013
                                 + 1.29007 w_k1_MSFT_s013
                                 + 1.34011 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.839985 w_k1_ADM_s013
                                 + 0.660096 w_k1_AXP_s013
                                 + 0.710419 w_k1_DIS_s013
                                 + 0.866179 w_k1_FDX_s013
                                 + 0.72413 w_k1_IBM_s013
                                 + 0.921521 w_k1_INTC_s013
                                 + 0.853157 w_k1_KO_s013
                                 + 0.688605 w_k1_MAT_s013
                                 + 0.902222 w_k1_MSFT_s013
                                 + 0.847722 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.678468 w_k1_ADM_s013
                                 + 0.625672 w_k1_AXP_s013
                                 + 0.597249 w_k1_DIS_s013
                                 + 0.673406 w_k1_FDX_s013
                                 + 1.05378 w_k1_IBM_s013
                                 + 0.852995 w_k1_INTC_s013
                                 + 0.883775 w_k1_KO_s013
                                 + 0.86908 w_k1_MAT_s013
                                 + 0.811803 w_k1_MSFT_s013
                                 + 0.787263 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.917998 w_k1_ADM_s013
                                 + 1.15743 w_k1_AXP_s013
                                 + 1.08861 w_k1_DIS_s013
                                 + 0.992379 w_k1_FDX_s013
                                 + 1.56605 w_k1_IBM_s013
                                 + 1.00533 w_k1_INTC_s013
                                 + 1.20817 w_k1_KO_s013 + 1.55326 w_k1_MAT_s013
                                 + 1.14477 w_k1_MSFT_s013
                                 + 1.04084 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.914237 w_k1_ADM_s013
                                 + 0.982645 w_k1_AXP_s013
                                 + 0.963587 w_k1_DIS_s013
                                 + 0.918678 w_k1_FDX_s013
                                 + 1.03641 w_k1_IBM_s013
                                 + 0.948211 w_k1_INTC_s013
                                 + 1.05001 w_k1_KO_s013 + 1.16109 w_k1_MAT_s013
                                 + 1.03608 w_k1_MSFT_s013
                                 + 1.06226 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.01273 w_k1_ADM_s013
                                 + 1.16015 w_k1_AXP_s013
                                 + 0.795973 w_k1_DIS_s013
                                 + 1.01077 w_k1_FDX_s013
                                 + 0.791051 w_k1_IBM_s013
                                 + 0.964051 w_k1_INTC_s013
                                 + 0.77811 w_k1_KO_s013
                                 + 0.780327 w_k1_MAT_s013
                                 + 0.890426 w_k1_MSFT_s013
                                 + 0.970501 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.15314 w_k1_ADM_s013
                                 + 0.914578 w_k1_AXP_s013
                                 + 1.04199 w_k1_DIS_s013
                                 + 1.04212 w_k1_FDX_s013
                                 + 0.729241 w_k1_IBM_s013
                                 + 0.84717 w_k1_INTC_s013
                                 + 0.906965 w_k1_KO_s013
                                 + 0.596931 w_k1_MAT_s013
                                 + 1.00737 w_k1_MSFT_s013 + 1.0109 w_k1_PG_s013
                                 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.846771 w_k1_ADM_s014
                                 + 0.714994 w_k1_AXP_s014
                                 + 0.944029 w_k1_DIS_s014
                                 + 0.948385 w_k1_FDX_s014
                                 + 0.778309 w_k1_IBM_s014
                                 + 0.835349 w_k1_INTC_s014
                                 + 0.921075 w_k1_KO_s014
                                 + 0.818158 w_k1_MAT_s014
                                 + 1.03414 w_k1_MSFT_s014
                                 + 0.84224 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.766395 w_k1_ADM_s014
                                 + 0.715583 w_k1_AXP_s014
                                 + 0.605534 w_k1_DIS_s014
                                 + 0.770889 w_k1_FDX_s014
                                 + 0.739467 w_k1_IBM_s014
                                 + 0.889951 w_k1_INTC_s014
                                 + 0.739259 w_k1_KO_s014
                                 + 0.749794 w_k1_MAT_s014
                                 + 0.791849 w_k1_MSFT_s014
                                 + 0.784827 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.925724 w_k1_ADM_s014
                                 + 1.03102 w_k1_AXP_s014
                                 + 1.32939 w_k1_DIS_s014
                                 + 1.07845 w_k1_FDX_s014 + 1.6326 w_k1_IBM_s014
                                 + 1.01544 w_k1_INTC_s014
                                 + 1.43351 w_k1_KO_s014 + 1.23856 w_k1_MAT_s014
                                 + 0.925111 w_k1_MSFT_s014
                                 + 0.96635 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.909306 w_k1_ADM_s014
                                 + 1.04936 w_k1_AXP_s014
                                 + 0.921124 w_k1_DIS_s014
                                 + 0.970718 w_k1_FDX_s014
                                 + 1.39681 w_k1_IBM_s014
                                 + 0.936685 w_k1_INTC_s014
                                 + 1.0998 w_k1_KO_s014 + 1.66911 w_k1_MAT_s014
                                 + 0.926242 w_k1_MSFT_s014
                                 + 0.940172 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.912857 w_k1_ADM_s014
                                 + 0.663477 w_k1_AXP_s014
                                 + 0.819404 w_k1_DIS_s014
                                 + 0.870009 w_k1_FDX_s014
                                 + 1.03845 w_k1_IBM_s014
                                 + 0.982459 w_k1_INTC_s014
                                 + 1.04412 w_k1_KO_s014
                                 + 0.700144 w_k1_MAT_s014
                                 + 0.914734 w_k1_MSFT_s014
                                 + 0.987532 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.23258 w_k1_ADM_s014
                                 + 1.52564 w_k1_AXP_s014
                                 + 1.88872 w_k1_DIS_s014
                                 + 1.38984 w_k1_FDX_s014
                                 + 1.78112 w_k1_IBM_s014
                                 + 1.19923 w_k1_INTC_s014
                                 + 1.80992 w_k1_KO_s014 + 1.62593 w_k1_MAT_s014
                                 + 1.32924 w_k1_MSFT_s014 + 1.182 w_k1_PG_s014
                                 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.28366 w_k1_ADM_s014
                                 + 1.03641 w_k1_AXP_s014
                                 + 1.18474 w_k1_DIS_s014
                                 + 1.28329 w_k1_FDX_s014
                                 + 0.953967 w_k1_IBM_s014
                                 + 1.30201 w_k1_INTC_s014
                                 + 1.08529 w_k1_KO_s014
                                 + 0.957352 w_k1_MAT_s014
                                 + 1.23301 w_k1_MSFT_s014
                                 + 1.04952 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.21489 w_k1_ADM_s014
                                 + 1.04407 w_k1_AXP_s014
                                 + 1.00224 w_k1_DIS_s014
                                 + 1.14863 w_k1_FDX_s014
                                 + 0.968289 w_k1_IBM_s014
                                 + 0.908935 w_k1_INTC_s014
                                 + 1.01301 w_k1_KO_s014
                                 + 0.890559 w_k1_MAT_s014
                                 + 1.00396 w_k1_MSFT_s014
                                 + 0.930898 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.820225 w_k1_ADM_s015
                                 + 0.526133 w_k1_AXP_s015
                                 + 0.643005 w_k1_DIS_s015
                                 + 0.752399 w_k1_FDX_s015
                                 + 0.722942 w_k1_IBM_s015
                                 + 0.677267 w_k1_INTC_s015
                                 + 0.774761 w_k1_KO_s015
                                 + 0.443724 w_k1_MAT_s015
                                 + 0.799586 w_k1_MSFT_s015
                                 + 0.907009 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.902156 w_k1_ADM_s015
                                 + 0.973984 w_k1_AXP_s015
                                 + 0.896029 w_k1_DIS_s015
                                 + 0.919051 w_k1_FDX_s015
                                 + 1.17331 w_k1_IBM_s015
                                 + 1.09103 w_k1_INTC_s015 + 1.2118 w_k1_KO_s015
                                 + 1.38622 w_k1_MAT_s015
                                 + 1.16673 w_k1_MSFT_s015
                                 + 1.06848 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.05674 w_k1_ADM_s015
                                 + 0.861516 w_k1_AXP_s015
                                 + 0.723591 w_k1_DIS_s015
                                 + 0.930983 w_k1_FDX_s015
                                 + 0.897562 w_k1_IBM_s015
                                 + 0.864997 w_k1_INTC_s015
                                 + 1.00066 w_k1_KO_s015 + 0.96775 w_k1_MAT_s015
                                 + 0.935561 w_k1_MSFT_s015
                                 + 0.967933 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.52745 w_k1_ADM_s015
                                 + 1.43123 w_k1_AXP_s015
                                 + 1.52803 w_k1_DIS_s015
                                 + 1.29496 w_k1_FDX_s015
                                 + 1.14223 w_k1_IBM_s015
                                 + 1.17226 w_k1_INTC_s015
                                 + 1.43249 w_k1_KO_s015 + 1.47159 w_k1_MAT_s015
                                 + 1.24097 w_k1_MSFT_s015
                                 + 1.37236 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.49962 w_k1_ADM_s015
                                 + 1.18631 w_k1_AXP_s015
                                 + 1.14248 w_k1_DIS_s015
                                 + 1.25871 w_k1_FDX_s015
                                 + 0.889892 w_k1_IBM_s015
                                 + 1.05118 w_k1_INTC_s015
                                 + 1.21065 w_k1_KO_s015
                                 + 0.855846 w_k1_MAT_s015
                                 + 1.24584 w_k1_MSFT_s015
                                 + 1.09859 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.37165 w_k1_ADM_s015
                                 + 1.10245 w_k1_AXP_s015 + 1.3055 w_k1_DIS_s015
                                 + 1.21213 w_k1_FDX_s015
                                 + 0.969445 w_k1_IBM_s015
                                 + 1.17246 w_k1_INTC_s015
                                 + 1.24197 w_k1_KO_s015
                                 + 0.981358 w_k1_MAT_s015
                                 + 1.26609 w_k1_MSFT_s015
                                 + 1.11113 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.04784 w_k1_ADM_s015
                                 + 1.14746 w_k1_AXP_s015
                                 + 1.25131 w_k1_DIS_s015
                                 + 1.11422 w_k1_FDX_s015
                                 + 1.57273 w_k1_IBM_s015
                                 + 0.990767 w_k1_INTC_s015
                                 + 1.4568 w_k1_KO_s015 + 1.51772 w_k1_MAT_s015
                                 + 1.11313 w_k1_MSFT_s015
                                 + 1.18275 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.1946 w_k1_ADM_s015
                                 + 0.772025 w_k1_AXP_s015
                                 + 0.851333 w_k1_DIS_s015
                                 + 1.01826 w_k1_FDX_s015
                                 + 0.624999 w_k1_IBM_s015
                                 + 0.843065 w_k1_INTC_s015
                                 + 0.825719 w_k1_KO_s015
                                 + 0.880778 w_k1_MAT_s015
                                 + 0.96934 w_k1_MSFT_s015
                                 + 0.970209 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.989002 w_k1_ADM_s016
                                 + 1.02886 w_k1_AXP_s016
                                 + 1.19369 w_k1_DIS_s016
                                 + 1.08418 w_k1_FDX_s016
                                 + 1.18428 w_k1_IBM_s016
                                 + 1.25039 w_k1_INTC_s016
                                 + 1.20612 w_k1_KO_s016
                                 + 0.940331 w_k1_MAT_s016
                                 + 1.19579 w_k1_MSFT_s016
                                 + 1.09347 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.908732 w_k1_ADM_s016
                                 + 1.13985 w_k1_AXP_s016
                                 + 1.02311 w_k1_DIS_s016
                                 + 1.17333 w_k1_FDX_s016
                                 + 0.76742 w_k1_IBM_s016
                                 + 1.05156 w_k1_INTC_s016
                                 + 0.945704 w_k1_KO_s016
                                 + 0.812163 w_k1_MAT_s016
                                 + 1.13958 w_k1_MSFT_s016
                                 + 1.00464 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.56821 w_k1_ADM_s016
                                 + 1.43884 w_k1_AXP_s016
                                 + 1.66863 w_k1_DIS_s016
                                 + 1.31393 w_k1_FDX_s016
                                 + 1.30677 w_k1_IBM_s016
                                 + 1.27534 w_k1_INTC_s016
                                 + 1.44351 w_k1_KO_s016 + 1.38603 w_k1_MAT_s016
                                 + 1.39113 w_k1_MSFT_s016
                                 + 1.38231 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.755187 w_k1_ADM_s016
                                 + 0.609126 w_k1_AXP_s016
                                 + 0.672203 w_k1_DIS_s016
                                 + 0.670616 w_k1_FDX_s016
                                 + 0.922816 w_k1_IBM_s016
                                 + 0.943007 w_k1_INTC_s016
                                 + 0.944036 w_k1_KO_s016
                                 + 0.645054 w_k1_MAT_s016
                                 + 0.832153 w_k1_MSFT_s016
                                 + 0.848691 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.716487 w_k1_ADM_s016
                                 + 0.783422 w_k1_AXP_s016
                                 + 0.661953 w_k1_DIS_s016
                                 + 0.726579 w_k1_FDX_s016
                                 + 1.33201 w_k1_IBM_s016
                                 + 0.974424 w_k1_INTC_s016
                                 + 1.03439 w_k1_KO_s016 + 1.21717 w_k1_MAT_s016
                                 + 0.783515 w_k1_MSFT_s016
                                 + 0.835104 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.791939 w_k1_ADM_s016
                                 + 0.809944 w_k1_AXP_s016
                                 + 0.744219 w_k1_DIS_s016
                                 + 0.880844 w_k1_FDX_s016
                                 + 0.71814 w_k1_IBM_s016
                                 + 1.01774 w_k1_INTC_s016
                                 + 0.785884 w_k1_KO_s016
                                 + 0.644212 w_k1_MAT_s016
                                 + 0.852954 w_k1_MSFT_s016
                                 + 0.902339 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.20847 w_k1_ADM_s016
                                 + 1.13133 w_k1_AXP_s016
                                 + 1.40672 w_k1_DIS_s016
                                 + 1.25513 w_k1_FDX_s016
                                 + 0.914727 w_k1_IBM_s016
                                 + 1.21791 w_k1_INTC_s016
                                 + 0.94699 w_k1_KO_s016
                                 + 0.807726 w_k1_MAT_s016
                                 + 1.20702 w_k1_MSFT_s016
                                 + 1.04075 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.01169 w_k1_ADM_s016
                                 + 0.989854 w_k1_AXP_s016
                                 + 0.912993 w_k1_DIS_s016
                                 + 0.913534 w_k1_FDX_s016
                                 + 1.0132 w_k1_IBM_s016
                                 + 0.936771 w_k1_INTC_s016
                                 + 0.930373 w_k1_KO_s016
                                 + 1.03404 w_k1_MAT_s016
                                 + 0.9506 w_k1_MSFT_s016 + 1.01541 w_k1_PG_s016
                                 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.803036 w_k1_ADM_s017
                                 + 0.755716 w_k1_AXP_s017
                                 + 0.666179 w_k1_DIS_s017
                                 + 0.717994 w_k1_FDX_s017
                                 + 0.958327 w_k1_IBM_s017
                                 + 0.695627 w_k1_INTC_s017
                                 + 0.777073 w_k1_KO_s017
                                 + 0.801198 w_k1_MAT_s017
                                 + 0.766478 w_k1_MSFT_s017
                                 + 0.915204 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.925474 w_k1_ADM_s017
                                 + 1.33187 w_k1_AXP_s017
                                 + 1.70972 w_k1_DIS_s017 + 1.2826 w_k1_FDX_s017
                                 + 1.34423 w_k1_IBM_s017
                                 + 1.27509 w_k1_INTC_s017
                                 + 1.36747 w_k1_KO_s017 + 1.23418 w_k1_MAT_s017
                                 + 1.54345 w_k1_MSFT_s017
                                 + 1.18047 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.832336 w_k1_ADM_s017
                                 + 0.772776 w_k1_AXP_s017
                                 + 0.74845 w_k1_DIS_s017
                                 + 0.933508 w_k1_FDX_s017
                                 + 0.662178 w_k1_IBM_s017
                                 + 0.954926 w_k1_INTC_s017
                                 + 0.750306 w_k1_KO_s017
                                 + 0.364758 w_k1_MAT_s017
                                 + 0.946976 w_k1_MSFT_s017
                                 + 0.833553 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.874143 w_k1_ADM_s017
                                 + 1.02444 w_k1_AXP_s017
                                 + 0.931358 w_k1_DIS_s017
                                 + 0.886912 w_k1_FDX_s017
                                 + 0.78721 w_k1_IBM_s017
                                 + 0.99887 w_k1_INTC_s017
                                 + 0.853326 w_k1_KO_s017
                                 + 0.835086 w_k1_MAT_s017
                                 + 1.09272 w_k1_MSFT_s017
                                 + 0.996311 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.14816 w_k1_ADM_s017
                                 + 1.35118 w_k1_AXP_s017
                                 + 0.932885 w_k1_DIS_s017
                                 + 0.937251 w_k1_FDX_s017
                                 + 1.28307 w_k1_IBM_s017
                                 + 1.02027 w_k1_INTC_s017
                                 + 1.05901 w_k1_KO_s017 + 1.70461 w_k1_MAT_s017
                                 + 0.983766 w_k1_MSFT_s017
                                 + 1.06837 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.06674 w_k1_ADM_s017
                                 + 1.22972 w_k1_AXP_s017
                                 + 1.23657 w_k1_DIS_s017
                                 + 1.14825 w_k1_FDX_s017
                                 + 1.04252 w_k1_IBM_s017
                                 + 1.31948 w_k1_INTC_s017
                                 + 1.01407 w_k1_KO_s017 + 1.07151 w_k1_MAT_s017
                                 + 1.17899 w_k1_MSFT_s017
                                 + 1.11261 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.970671 w_k1_ADM_s017
                                 + 0.725735 w_k1_AXP_s017
                                 + 0.987971 w_k1_DIS_s017
                                 + 0.836177 w_k1_FDX_s017
                                 + 1.02101 w_k1_IBM_s017
                                 + 1.02654 w_k1_INTC_s017
                                 + 1.05524 w_k1_KO_s017
                                 + 0.838852 w_k1_MAT_s017
                                 + 0.976529 w_k1_MSFT_s017
                                 + 0.914921 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.873086 w_k1_ADM_s017
                                 + 0.921422 w_k1_AXP_s017
                                 + 0.974119 w_k1_DIS_s017
                                 + 0.9106 w_k1_FDX_s017 + 1.25708 w_k1_IBM_s017
                                 + 1.07398 w_k1_INTC_s017 + 1.0159 w_k1_KO_s017
                                 + 1.14129 w_k1_MAT_s017
                                 + 0.929672 w_k1_MSFT_s017
                                 + 1.02334 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.715871 w_k1_ADM_s100
                                 + 0.823558 w_k1_AXP_s100
                                 + 0.727572 w_k1_DIS_s100
                                 + 0.827746 w_k1_FDX_s100
                                 + 0.73489 w_k1_IBM_s100
                                 + 0.73556 w_k1_INTC_s100
                                 + 0.82262 w_k1_KO_s100
                                 + 0.760174 w_k1_MAT_s100
                                 + 0.745024 w_k1_MSFT_s100
                                 + 0.871529 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.843788 w_k1_ADM_s100
                                 + 1.56745 w_k1_AXP_s100
                                 + 1.57039 w_k1_DIS_s100
                                 + 1.26782 w_k1_FDX_s100
                                 + 1.34727 w_k1_IBM_s100
                                 + 1.35539 w_k1_INTC_s100
                                 + 1.45898 w_k1_KO_s100 + 1.59308 w_k1_MAT_s100
                                 + 1.28269 w_k1_MSFT_s100
                                 + 1.18038 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.944248 w_k1_ADM_s100
                                 + 1.03214 w_k1_AXP_s100
                                 + 1.02567 w_k1_DIS_s100
                                 + 1.00748 w_k1_FDX_s100
                                 + 1.21916 w_k1_IBM_s100
                                 + 1.01588 w_k1_INTC_s100
                                 + 1.20185 w_k1_KO_s100 + 1.12205 w_k1_MAT_s100
                                 + 1.13578 w_k1_MSFT_s100
                                 + 0.992484 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.07134 w_k1_ADM_s100
                                 + 1.23475 w_k1_AXP_s100
                                 + 1.03428 w_k1_DIS_s100
                                 + 1.09002 w_k1_FDX_s100
                                 + 0.906773 w_k1_IBM_s100
                                 + 1.06308 w_k1_INTC_s100
                                 + 1.04717 w_k1_KO_s100 + 1.06215 w_k1_MAT_s100
                                 + 1.10963 w_k1_MSFT_s100
                                 + 1.01076 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.711615 w_k1_ADM_s100
                                 + 0.798424 w_k1_AXP_s100
                                 + 0.663747 w_k1_DIS_s100
                                 + 0.738478 w_k1_FDX_s100
                                 + 0.913691 w_k1_IBM_s100
                                 + 0.795611 w_k1_INTC_s100
                                 + 0.916971 w_k1_KO_s100
                                 + 1.09326 w_k1_MAT_s100
                                 + 0.794783 w_k1_MSFT_s100
                                 + 0.770393 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.787867 w_k1_ADM_s100
                                 + 0.90592 w_k1_AXP_s100
                                 + 0.914969 w_k1_DIS_s100
                                 + 1.02342 w_k1_FDX_s100
                                 + 0.671359 w_k1_IBM_s100
                                 + 0.975562 w_k1_INTC_s100
                                 + 0.924989 w_k1_KO_s100
                                 + 0.938378 w_k1_MAT_s100
                                 + 1.01594 w_k1_MSFT_s100
                                 + 0.904159 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.954597 w_k1_ADM_s100
                                 + 0.866974 w_k1_AXP_s100
                                 + 0.757672 w_k1_DIS_s100
                                 + 0.83226 w_k1_FDX_s100
                                 + 1.34938 w_k1_IBM_s100
                                 + 0.850709 w_k1_INTC_s100
                                 + 0.996312 w_k1_KO_s100
                                 + 1.40575 w_k1_MAT_s100
                                 + 0.729693 w_k1_MSFT_s100
                                 + 0.852647 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.858873 w_k1_ADM_s100
                                 + 1.19975 w_k1_AXP_s100
                                 + 1.71746 w_k1_DIS_s100
                                 + 1.36976 w_k1_FDX_s100
                                 + 1.00737 w_k1_IBM_s100
                                 + 1.13061 w_k1_INTC_s100
                                 + 1.34595 w_k1_KO_s100 + 1.04466 w_k1_MAT_s100
                                 + 1.33834 w_k1_MSFT_s100
                                 + 1.04304 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.632419 w_k1_ADM_s101
                                 + 0.752442 w_k1_AXP_s101
                                 + 0.586069 w_k1_DIS_s101
                                 + 0.64652 w_k1_FDX_s101
                                 + 0.78617 w_k1_IBM_s101
                                 + 0.872048 w_k1_INTC_s101
                                 + 0.759821 w_k1_KO_s101
                                 + 0.649515 w_k1_MAT_s101
                                 + 0.657834 w_k1_MSFT_s101
                                 + 0.740432 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.35872 w_k1_ADM_s101
                                 + 1.46226 w_k1_AXP_s101
                                 + 1.81488 w_k1_DIS_s101
                                 + 1.42278 w_k1_FDX_s101
                                 + 1.39556 w_k1_IBM_s101
                                 + 1.17513 w_k1_INTC_s101
                                 + 1.45637 w_k1_KO_s101 + 1.30757 w_k1_MAT_s101
                                 + 1.38139 w_k1_MSFT_s101
                                 + 1.26698 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.759918 w_k1_ADM_s101
                                 + 1.0259 w_k1_AXP_s101
                                 + 0.851224 w_k1_DIS_s101
                                 + 0.891766 w_k1_FDX_s101
                                 + 1.51755 w_k1_IBM_s101
                                 + 1.14006 w_k1_INTC_s101 + 1.1789 w_k1_KO_s101
                                 + 1.23069 w_k1_MAT_s101
                                 + 0.985626 w_k1_MSFT_s101
                                 + 1.13146 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.931214 w_k1_ADM_s101
                                 + 0.711154 w_k1_AXP_s101
                                 + 0.752458 w_k1_DIS_s101
                                 + 0.954868 w_k1_FDX_s101
                                 + 0.679918 w_k1_IBM_s101
                                 + 0.955114 w_k1_INTC_s101
                                 + 0.775791 w_k1_KO_s101
                                 + 0.514806 w_k1_MAT_s101
                                 + 0.937173 w_k1_MSFT_s101
                                 + 0.907337 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.17286 w_k1_ADM_s101
                                 + 1.57692 w_k1_AXP_s101
                                 + 1.14736 w_k1_DIS_s101
                                 + 1.13933 w_k1_FDX_s101
                                 + 1.08782 w_k1_IBM_s101
                                 + 1.01288 w_k1_INTC_s101
                                 + 1.15849 w_k1_KO_s101 + 1.72138 w_k1_MAT_s101
                                 + 1.06068 w_k1_MSFT_s101
                                 + 1.11424 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.820898 w_k1_ADM_s101
                                 + 0.990486 w_k1_AXP_s101
                                 + 0.711345 w_k1_DIS_s101
                                 + 0.840445 w_k1_FDX_s101
                                 + 1.04023 w_k1_IBM_s101
                                 + 0.868251 w_k1_INTC_s101
                                 + 0.973845 w_k1_KO_s101
                                 + 1.06424 w_k1_MAT_s101
                                 + 0.769233 w_k1_MSFT_s101
                                 + 0.981506 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.87633 w_k1_ADM_s101
                                 + 0.904252 w_k1_AXP_s101
                                 + 1.10527 w_k1_DIS_s101
                                 + 1.00041 w_k1_FDX_s101
                                 + 0.996772 w_k1_IBM_s101
                                 + 0.910909 w_k1_INTC_s101
                                 + 0.903295 w_k1_KO_s101
                                 + 1.04235 w_k1_MAT_s101
                                 + 0.885467 w_k1_MSFT_s101
                                 + 0.995259 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.14808 w_k1_ADM_s101
                                 + 1.46327 w_k1_AXP_s101
                                 + 1.20922 w_k1_DIS_s101
                                 + 1.39371 w_k1_FDX_s101
                                 + 0.858287 w_k1_IBM_s101
                                 + 1.23248 w_k1_INTC_s101
                                 + 1.13394 w_k1_KO_s101 + 1.3268 w_k1_MAT_s101
                                 + 1.26782 w_k1_MSFT_s101
                                 + 1.10913 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.02058 w_k1_ADM_s102
                                 + 0.965046 w_k1_AXP_s102
                                 + 1.25243 w_k1_DIS_s102
                                 + 0.965364 w_k1_FDX_s102
                                 + 1.58553 w_k1_IBM_s102
                                 + 1.20552 w_k1_INTC_s102
                                 + 1.23736 w_k1_KO_s102 + 1.10578 w_k1_MAT_s102
                                 + 0.966523 w_k1_MSFT_s102
                                 + 0.9702 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.22187 w_k1_ADM_s102
                                 + 1.58311 w_k1_AXP_s102
                                 + 1.23198 w_k1_DIS_s102
                                 + 1.30233 w_k1_FDX_s102
                                 + 0.923441 w_k1_IBM_s102
                                 + 1.22918 w_k1_INTC_s102
                                 + 0.92036 w_k1_KO_s102 + 1.21743 w_k1_MAT_s102
                                 + 1.13707 w_k1_MSFT_s102
                                 + 1.09867 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.12294 w_k1_ADM_s102
                                 + 0.77601 w_k1_AXP_s102
                                 + 1.11047 w_k1_DIS_s102
                                 + 0.880119 w_k1_FDX_s102
                                 + 0.889107 w_k1_IBM_s102
                                 + 0.917571 w_k1_INTC_s102
                                 + 0.957975 w_k1_KO_s102
                                 + 0.644065 w_k1_MAT_s102
                                 + 0.943416 w_k1_MSFT_s102
                                 + 0.941502 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.15401 w_k1_ADM_s102
                                 + 0.828005 w_k1_AXP_s102
                                 + 0.729247 w_k1_DIS_s102
                                 + 0.951622 w_k1_FDX_s102
                                 + 0.675542 w_k1_IBM_s102
                                 + 1.05344 w_k1_INTC_s102
                                 + 0.712465 w_k1_KO_s102
                                 + 0.590018 w_k1_MAT_s102
                                 + 0.954451 w_k1_MSFT_s102
                                 + 0.832322 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.74369 w_k1_ADM_s102
                                 + 1.41028 w_k1_AXP_s102 + 1.6298 w_k1_DIS_s102
                                 + 1.35266 w_k1_FDX_s102
                                 + 1.00261 w_k1_IBM_s102
                                 + 1.57332 w_k1_INTC_s102
                                 + 1.15637 w_k1_KO_s102
                                 + 0.634488 w_k1_MAT_s102
                                 + 1.35415 w_k1_MSFT_s102
                                 + 1.16745 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.46214 w_k1_ADM_s102
                                 + 1.16124 w_k1_AXP_s102
                                 + 1.04958 w_k1_DIS_s102
                                 + 0.992479 w_k1_FDX_s102
                                 + 1.11117 w_k1_IBM_s102
                                 + 1.08444 w_k1_INTC_s102
                                 + 1.12013 w_k1_KO_s102 + 1.16623 w_k1_MAT_s102
                                 + 1.00425 w_k1_MSFT_s102
                                 + 1.02958 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.942721 w_k1_ADM_s102
                                 + 0.697877 w_k1_AXP_s102
                                 + 0.633129 w_k1_DIS_s102
                                 + 0.746456 w_k1_FDX_s102
                                 + 0.784591 w_k1_IBM_s102
                                 + 0.713973 w_k1_INTC_s102
                                 + 0.721109 w_k1_KO_s102
                                 + 0.512126 w_k1_MAT_s102
                                 + 0.694241 w_k1_MSFT_s102
                                 + 0.768637 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.01836 w_k1_ADM_s102
                                 + 0.879169 w_k1_AXP_s102
                                 + 0.791359 w_k1_DIS_s102
                                 + 0.819713 w_k1_FDX_s102
                                 + 1.02331 w_k1_IBM_s102
                                 + 0.81982 w_k1_INTC_s102
                                 + 0.913722 w_k1_KO_s102
                                 + 0.753335 w_k1_MAT_s102
                                 + 0.785855 w_k1_MSFT_s102
                                 + 0.908859 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.923877 w_k1_ADM_s103
                                 + 0.821396 w_k1_AXP_s103
                                 + 0.724484 w_k1_DIS_s103
                                 + 0.772142 w_k1_FDX_s103
                                 + 0.839346 w_k1_IBM_s103
                                 + 1.11567 w_k1_INTC_s103
                                 + 0.811507 w_k1_KO_s103
                                 + 0.800528 w_k1_MAT_s103
                                 + 0.992923 w_k1_MSFT_s103
                                 + 0.96841 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.04438 w_k1_ADM_s103
                                 + 0.911908 w_k1_AXP_s103
                                 + 1.01798 w_k1_DIS_s103
                                 + 0.993327 w_k1_FDX_s103
                                 + 0.840475 w_k1_IBM_s103
                                 + 1.33953 w_k1_INTC_s103
                                 + 0.807209 w_k1_KO_s103
                                 + 0.887936 w_k1_MAT_s103
                                 + 1.27239 w_k1_MSFT_s103
                                 + 0.970585 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.05609 w_k1_ADM_s103
                                 + 0.970879 w_k1_AXP_s103
                                 + 0.945698 w_k1_DIS_s103
                                 + 0.951386 w_k1_FDX_s103 + 0.653 w_k1_IBM_s103
                                 + 1.14266 w_k1_INTC_s103
                                 + 0.763146 w_k1_KO_s103
                                 + 0.58625 w_k1_MAT_s103
                                 + 1.16918 w_k1_MSFT_s103
                                 + 0.945056 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.3699 w_k1_ADM_s103
                                 + 1.52467 w_k1_AXP_s103
                                 + 1.49706 w_k1_DIS_s103
                                 + 1.41738 w_k1_FDX_s103
                                 + 0.876222 w_k1_IBM_s103
                                 + 1.47142 w_k1_INTC_s103
                                 + 0.980856 w_k1_KO_s103
                                 + 1.27683 w_k1_MAT_s103
                                 + 1.50725 w_k1_MSFT_s103
                                 + 1.24523 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.932979 w_k1_ADM_s103
                                 + 0.803718 w_k1_AXP_s103
                                 + 0.787219 w_k1_DIS_s103
                                 + 0.72158 w_k1_FDX_s103
                                 + 1.08456 w_k1_IBM_s103
                                 + 0.930804 w_k1_INTC_s103
                                 + 0.995628 w_k1_KO_s103
                                 + 1.14873 w_k1_MAT_s103
                                 + 0.909403 w_k1_MSFT_s103
                                 + 0.957807 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.876845 w_k1_ADM_s103
                                 + 0.659064 w_k1_AXP_s103
                                 + 0.654143 w_k1_DIS_s103
                                 + 0.679356 w_k1_FDX_s103
                                 + 0.644674 w_k1_IBM_s103
                                 + 0.759323 w_k1_INTC_s103
                                 + 0.6578 w_k1_KO_s103 + 0.511091 w_k1_MAT_s103
                                 + 0.757534 w_k1_MSFT_s103
                                 + 0.858565 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.17642 w_k1_ADM_s103
                                 + 0.948088 w_k1_AXP_s103
                                 + 1.27289 w_k1_DIS_s103
                                 + 1.01618 w_k1_FDX_s103 + 1.1165 w_k1_IBM_s103
                                 + 1.12635 w_k1_INTC_s103
                                 + 1.18157 w_k1_KO_s103 + 1.24437 w_k1_MAT_s103
                                 + 1.38816 w_k1_MSFT_s103
                                 + 1.21842 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.00464 w_k1_ADM_s103
                                 + 1.28664 w_k1_AXP_s103
                                 + 1.12505 w_k1_DIS_s103
                                 + 0.990623 w_k1_FDX_s103
                                 + 1.56152 w_k1_IBM_s103
                                 + 0.936896 w_k1_INTC_s103
                                 + 1.04154 w_k1_KO_s103 + 2.10168 w_k1_MAT_s103
                                 + 1.08949 w_k1_MSFT_s103
                                 + 1.20017 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.902574 w_k1_ADM_s104
                                 + 1.09734 w_k1_AXP_s104
                                 + 1.30092 w_k1_DIS_s104
                                 + 0.984936 w_k1_FDX_s104
                                 + 1.22366 w_k1_IBM_s104
                                 + 0.932479 w_k1_INTC_s104
                                 + 1.01654 w_k1_KO_s104 + 1.32436 w_k1_MAT_s104
                                 + 0.958568 w_k1_MSFT_s104
                                 + 0.973076 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.17843 w_k1_ADM_s104
                                 + 0.986063 w_k1_AXP_s104
                                 + 1.09837 w_k1_DIS_s104
                                 + 1.03251 w_k1_FDX_s104
                                 + 0.733381 w_k1_IBM_s104
                                 + 1.00826 w_k1_INTC_s104
                                 + 0.834446 w_k1_KO_s104
                                 + 0.57633 w_k1_MAT_s104
                                 + 1.03289 w_k1_MSFT_s104
                                 + 0.931405 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.943526 w_k1_ADM_s104
                                 + 0.562525 w_k1_AXP_s104
                                 + 0.750483 w_k1_DIS_s104
                                 + 0.790929 w_k1_FDX_s104
                                 + 0.603444 w_k1_IBM_s104
                                 + 0.71582 w_k1_INTC_s104
                                 + 0.618001 w_k1_KO_s104
                                 + 0.443725 w_k1_MAT_s104
                                 + 0.879084 w_k1_MSFT_s104
                                 + 0.818232 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.59868 w_k1_ADM_s104
                                 + 0.600624 w_k1_AXP_s104
                                 + 0.699562 w_k1_DIS_s104
                                 + 0.688622 w_k1_FDX_s104
                                 + 0.87159 w_k1_IBM_s104
                                 + 0.884299 w_k1_INTC_s104
                                 + 0.763556 w_k1_KO_s104
                                 + 0.754605 w_k1_MAT_s104
                                 + 0.753714 w_k1_MSFT_s104
                                 + 0.772307 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.04153 w_k1_ADM_s104
                                 + 1.24794 w_k1_AXP_s104
                                 + 0.89089 w_k1_DIS_s104
                                 + 1.01099 w_k1_FDX_s104
                                 + 1.26435 w_k1_IBM_s104
                                 + 0.973807 w_k1_INTC_s104
                                 + 0.952215 w_k1_KO_s104
                                 + 1.14983 w_k1_MAT_s104
                                 + 0.902946 w_k1_MSFT_s104
                                 + 0.913317 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.15912 w_k1_ADM_s104
                                 + 1.51369 w_k1_AXP_s104
                                 + 1.57989 w_k1_DIS_s104
                                 + 1.32612 w_k1_FDX_s104
                                 + 1.28532 w_k1_IBM_s104
                                 + 1.49286 w_k1_INTC_s104
                                 + 1.28017 w_k1_KO_s104 + 1.16065 w_k1_MAT_s104
                                 + 1.26858 w_k1_MSFT_s104
                                 + 1.35289 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.691116 w_k1_ADM_s104
                                 + 0.772469 w_k1_AXP_s104
                                 + 0.525391 w_k1_DIS_s104
                                 + 0.68486 w_k1_FDX_s104
                                 + 1.30687 w_k1_IBM_s104
                                 + 0.851861 w_k1_INTC_s104
                                 + 0.764823 w_k1_KO_s104
                                 + 0.918671 w_k1_MAT_s104
                                 + 0.676803 w_k1_MSFT_s104
                                 + 0.85277 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.758598 w_k1_ADM_s104
                                 + 1.07798 w_k1_AXP_s104
                                 + 0.869495 w_k1_DIS_s104
                                 + 0.912613 w_k1_FDX_s104
                                 + 1.03078 w_k1_IBM_s104
                                 + 0.894466 w_k1_INTC_s104
                                 + 0.876441 w_k1_KO_s104
                                 + 0.996767 w_k1_MAT_s104
                                 + 0.921154 w_k1_MSFT_s104
                                 + 1.00256 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.18724 w_k1_ADM_s105
                                 + 1.16839 w_k1_AXP_s105
                                 + 1.30575 w_k1_DIS_s105
                                 + 1.25666 w_k1_FDX_s105
                                 + 1.19849 w_k1_IBM_s105
                                 + 0.88389 w_k1_INTC_s105
                                 + 0.982499 w_k1_KO_s105
                                 + 1.30724 w_k1_MAT_s105
                                 + 1.14353 w_k1_MSFT_s105
                                 + 1.07311 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.846146 w_k1_ADM_s105
                                 + 0.754772 w_k1_AXP_s105
                                 + 0.611092 w_k1_DIS_s105
                                 + 0.753355 w_k1_FDX_s105
                                 + 0.915165 w_k1_IBM_s105
                                 + 0.874344 w_k1_INTC_s105
                                 + 0.794259 w_k1_KO_s105 + 1.0001 w_k1_MAT_s105
                                 + 0.774307 w_k1_MSFT_s105
                                 + 0.910435 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.10763 w_k1_ADM_s105
                                 + 0.709354 w_k1_AXP_s105
                                 + 0.711361 w_k1_DIS_s105
                                 + 0.835187 w_k1_FDX_s105
                                 + 0.635801 w_k1_IBM_s105
                                 + 0.794421 w_k1_INTC_s105
                                 + 0.767764 w_k1_KO_s105
                                 + 0.551569 w_k1_MAT_s105
                                 + 0.895695 w_k1_MSFT_s105
                                 + 0.869233 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.0292 w_k1_ADM_s105
                                 + 0.880116 w_k1_AXP_s105
                                 + 1.2133 w_k1_DIS_s105 + 1.22923 w_k1_FDX_s105
                                 + 0.846687 w_k1_IBM_s105
                                 + 1.45734 w_k1_INTC_s105
                                 + 0.909342 w_k1_KO_s105
                                 + 0.685937 w_k1_MAT_s105
                                 + 1.25011 w_k1_MSFT_s105
                                 + 1.14439 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.97283 w_k1_ADM_s105
                                 + 0.847617 w_k1_AXP_s105
                                 + 1.10518 w_k1_DIS_s105
                                 + 1.00082 w_k1_FDX_s105
                                 + 1.03242 w_k1_IBM_s105
                                 + 0.92395 w_k1_INTC_s105
                                 + 1.06285 w_k1_KO_s105 + 1.24077 w_k1_MAT_s105
                                 + 1.22126 w_k1_MSFT_s105
                                 + 0.985417 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.915744 w_k1_ADM_s105
                                 + 0.956184 w_k1_AXP_s105
                                 + 1.05838 w_k1_DIS_s105
                                 + 1.06534 w_k1_FDX_s105
                                 + 1.07781 w_k1_IBM_s105
                                 + 0.926467 w_k1_INTC_s105
                                 + 0.968946 w_k1_KO_s105
                                 + 1.29014 w_k1_MAT_s105
                                 + 1.04676 w_k1_MSFT_s105
                                 + 0.980013 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.33392 w_k1_ADM_s105
                                 + 0.866184 w_k1_AXP_s105
                                 + 0.831127 w_k1_DIS_s105
                                 + 0.945357 w_k1_FDX_s105
                                 + 0.9943 w_k1_IBM_s105
                                 + 0.803516 w_k1_INTC_s105
                                 + 0.968095 w_k1_KO_s105
                                 + 0.938824 w_k1_MAT_s105
                                 + 1.02075 w_k1_MSFT_s105
                                 + 1.01024 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.00215 w_k1_ADM_s105
                                 + 0.902445 w_k1_AXP_s105
                                 + 1.00456 w_k1_DIS_s105
                                 + 0.914885 w_k1_FDX_s105
                                 + 1.57482 w_k1_IBM_s105
                                 + 0.975666 w_k1_INTC_s105
                                 + 1.2088 w_k1_KO_s105 + 1.52271 w_k1_MAT_s105
                                 + 1.05378 w_k1_MSFT_s105
                                 + 1.18674 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.19267 w_k1_ADM_s106
                                 + 0.917693 w_k1_AXP_s106
                                 + 1.50485 w_k1_DIS_s106
                                 + 1.15316 w_k1_FDX_s106
                                 + 1.23228 w_k1_IBM_s106
                                 + 1.13162 w_k1_INTC_s106
                                 + 1.54781 w_k1_KO_s106 + 1.37118 w_k1_MAT_s106
                                 + 1.27888 w_k1_MSFT_s106
                                 + 1.14205 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.895621 w_k1_ADM_s106
                                 + 0.704395 w_k1_AXP_s106
                                 + 0.698175 w_k1_DIS_s106
                                 + 0.880652 w_k1_FDX_s106
                                 + 0.692332 w_k1_IBM_s106
                                 + 0.723204 w_k1_INTC_s106
                                 + 0.776318 w_k1_KO_s106
                                 + 0.68531 w_k1_MAT_s106
                                 + 0.788182 w_k1_MSFT_s106
                                 + 0.828011 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.04727 w_k1_ADM_s106
                                 + 0.901817 w_k1_AXP_s106
                                 + 1.05315 w_k1_DIS_s106
                                 + 1.12025 w_k1_FDX_s106
                                 + 0.76745 w_k1_IBM_s106
                                 + 0.866047 w_k1_INTC_s106
                                 + 0.871168 w_k1_KO_s106
                                 + 0.917301 w_k1_MAT_s106
                                 + 0.992459 w_k1_MSFT_s106
                                 + 0.975122 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.67541 w_k1_ADM_s106
                                 + 1.42993 w_k1_AXP_s106
                                 + 1.76212 w_k1_DIS_s106
                                 + 1.46678 w_k1_FDX_s106
                                 + 1.11383 w_k1_IBM_s106
                                 + 1.13683 w_k1_INTC_s106 + 1.2825 w_k1_KO_s106
                                 + 1.58635 w_k1_MAT_s106
                                 + 1.33725 w_k1_MSFT_s106
                                 + 1.24757 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.23853 w_k1_ADM_s106
                                 + 1.15277 w_k1_AXP_s106
                                 + 1.36507 w_k1_DIS_s106
                                 + 1.13446 w_k1_FDX_s106 + 1.5574 w_k1_IBM_s106
                                 + 0.902952 w_k1_INTC_s106
                                 + 1.36537 w_k1_KO_s106 + 1.73396 w_k1_MAT_s106
                                 + 1.18494 w_k1_MSFT_s106
                                 + 1.04084 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.03871 w_k1_ADM_s106
                                 + 0.780514 w_k1_AXP_s106
                                 + 0.916616 w_k1_DIS_s106
                                 + 0.935899 w_k1_FDX_s106
                                 + 1.12135 w_k1_IBM_s106
                                 + 0.976566 w_k1_INTC_s106
                                 + 1.01975 w_k1_KO_s106
                                 + 0.910004 w_k1_MAT_s106
                                 + 0.965503 w_k1_MSFT_s106
                                 + 0.917762 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.13062 w_k1_ADM_s106
                                 + 1.27604 w_k1_AXP_s106
                                 + 0.930602 w_k1_DIS_s106
                                 + 1.02261 w_k1_FDX_s106
                                 + 1.07759 w_k1_IBM_s106
                                 + 0.689262 w_k1_INTC_s106
                                 + 0.90299 w_k1_KO_s106 + 1.30827 w_k1_MAT_s106
                                 + 0.869097 w_k1_MSFT_s106
                                 + 0.992361 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.43653 w_k1_ADM_s106
                                 + 1.28156 w_k1_AXP_s106
                                 + 1.18533 w_k1_DIS_s106
                                 + 1.20789 w_k1_FDX_s106
                                 + 1.05702 w_k1_IBM_s106
                                 + 0.74866 w_k1_INTC_s106
                                 + 0.995331 w_k1_KO_s106
                                 + 1.21401 w_k1_MAT_s106
                                 + 1.0164 w_k1_MSFT_s106
                                 + 0.966297 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.802935 w_k1_ADM_s107
                                 + 0.696803 w_k1_AXP_s107
                                 + 0.770064 w_k1_DIS_s107
                                 + 0.819663 w_k1_FDX_s107
                                 + 0.78393 w_k1_IBM_s107
                                 + 0.822779 w_k1_INTC_s107
                                 + 0.804832 w_k1_KO_s107
                                 + 0.570698 w_k1_MAT_s107
                                 + 0.767465 w_k1_MSFT_s107
                                 + 0.755443 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.883155 w_k1_ADM_s107
                                 + 1.02892 w_k1_AXP_s107
                                 + 1.12724 w_k1_DIS_s107
                                 + 1.08496 w_k1_FDX_s107
                                 + 1.13371 w_k1_IBM_s107
                                 + 1.06497 w_k1_INTC_s107
                                 + 1.08927 w_k1_KO_s107 + 1.13967 w_k1_MAT_s107
                                 + 0.936571 w_k1_MSFT_s107
                                 + 1.11342 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.970841 w_k1_ADM_s107
                                 + 0.849906 w_k1_AXP_s107
                                 + 0.941385 w_k1_DIS_s107
                                 + 0.885396 w_k1_FDX_s107
                                 + 1.5428 w_k1_IBM_s107
                                 + 1.00698 w_k1_INTC_s107
                                 + 1.25433 w_k1_KO_s107 + 1.36872 w_k1_MAT_s107
                                 + 0.865391 w_k1_MSFT_s107
                                 + 0.994628 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.02789 w_k1_ADM_s107
                                 + 0.808077 w_k1_AXP_s107
                                 + 0.807422 w_k1_DIS_s107
                                 + 0.840602 w_k1_FDX_s107
                                 + 1.18084 w_k1_IBM_s107
                                 + 0.886198 w_k1_INTC_s107
                                 + 1.03329 w_k1_KO_s107 + 1.19234 w_k1_MAT_s107
                                 + 0.752452 w_k1_MSFT_s107
                                 + 0.998116 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.40207 w_k1_ADM_s107
                                 + 1.45344 w_k1_AXP_s107
                                 + 1.35986 w_k1_DIS_s107
                                 + 1.26734 w_k1_FDX_s107
                                 + 1.55012 w_k1_IBM_s107
                                 + 1.23209 w_k1_INTC_s107
                                 + 1.36059 w_k1_KO_s107 + 1.63424 w_k1_MAT_s107
                                 + 1.17601 w_k1_MSFT_s107
                                 + 1.14773 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.14605 w_k1_ADM_s107
                                 + 0.980864 w_k1_AXP_s107
                                 + 1.06631 w_k1_DIS_s107
                                 + 1.15793 w_k1_FDX_s107
                                 + 0.74219 w_k1_IBM_s107
                                 + 1.15667 w_k1_INTC_s107
                                 + 1.02119 w_k1_KO_s107
                                 + 0.711352 w_k1_MAT_s107
                                 + 1.02652 w_k1_MSFT_s107
                                 + 0.917887 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.82499 w_k1_ADM_s107
                                 + 1.12912 w_k1_AXP_s107
                                 + 1.44367 w_k1_DIS_s107
                                 + 1.46478 w_k1_FDX_s107 + 1.076 w_k1_IBM_s107
                                 + 1.22652 w_k1_INTC_s107
                                 + 1.35262 w_k1_KO_s107 + 1.10694 w_k1_MAT_s107
                                 + 1.35454 w_k1_MSFT_s107
                                 + 1.09659 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.3703 w_k1_ADM_s107
                                 + 1.05512 w_k1_AXP_s107
                                 + 0.910763 w_k1_DIS_s107
                                 + 1.06987 w_k1_FDX_s107
                                 + 0.915799 w_k1_IBM_s107
                                 + 1.06504 w_k1_INTC_s107
                                 + 1.09633 w_k1_KO_s107
                                 + 0.962855 w_k1_MAT_s107
                                 + 0.924887 w_k1_MSFT_s107
                                 + 0.958508 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.10186 w_k1_ADM_s110
                                 + 1.53558 w_k1_AXP_s110
                                 + 1.69157 w_k1_DIS_s110
                                 + 1.48254 w_k1_FDX_s110
                                 + 1.19673 w_k1_IBM_s110
                                 + 1.04066 w_k1_INTC_s110
                                 + 1.39709 w_k1_KO_s110 + 1.3953 w_k1_MAT_s110
                                 + 1.19371 w_k1_MSFT_s110
                                 + 1.31373 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.22077 w_k1_ADM_s110
                                 + 1.21879 w_k1_AXP_s110
                                 + 1.16775 w_k1_DIS_s110
                                 + 1.32414 w_k1_FDX_s110
                                 + 0.774533 w_k1_IBM_s110
                                 + 0.986003 w_k1_INTC_s110
                                 + 1.02285 w_k1_KO_s110 + 1.1069 w_k1_MAT_s110
                                 + 1.00464 w_k1_MSFT_s110
                                 + 1.00765 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.894745 w_k1_ADM_s110
                                 + 1.03766 w_k1_AXP_s110
                                 + 0.875926 w_k1_DIS_s110
                                 + 0.951394 w_k1_FDX_s110
                                 + 1.33782 w_k1_IBM_s110
                                 + 0.831649 w_k1_INTC_s110
                                 + 1.03606 w_k1_KO_s110 + 1.06986 w_k1_MAT_s110
                                 + 0.875769 w_k1_MSFT_s110
                                 + 1.00295 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.992407 w_k1_ADM_s110
                                 + 0.832017 w_k1_AXP_s110
                                 + 0.960939 w_k1_DIS_s110
                                 + 0.938461 w_k1_FDX_s110
                                 + 0.814528 w_k1_IBM_s110
                                 + 0.976111 w_k1_INTC_s110
                                 + 1.04545 w_k1_KO_s110
                                 + 0.847617 w_k1_MAT_s110
                                 + 0.95645 w_k1_MSFT_s110
                                 + 0.87061 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.18504 w_k1_ADM_s110
                                 + 1.20214 w_k1_AXP_s110
                                 + 0.967687 w_k1_DIS_s110
                                 + 1.11569 w_k1_FDX_s110
                                 + 1.02099 w_k1_IBM_s110
                                 + 0.975134 w_k1_INTC_s110
                                 + 1.0916 w_k1_KO_s110 + 1.19183 w_k1_MAT_s110
                                 + 0.951264 w_k1_MSFT_s110
                                 + 1.05712 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.0847 w_k1_ADM_s110
                                 + 1.44495 w_k1_AXP_s110
                                 + 1.25972 w_k1_DIS_s110 + 1.2059 w_k1_FDX_s110
                                 + 1.6639 w_k1_IBM_s110
                                 + 1.17206 w_k1_INTC_s110
                                 + 1.48469 w_k1_KO_s110 + 1.89392 w_k1_MAT_s110
                                 + 1.16527 w_k1_MSFT_s110
                                 + 1.22007 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.71964 w_k1_ADM_s110
                                 + 0.650941 w_k1_AXP_s110
                                 + 0.54027 w_k1_DIS_s110
                                 + 0.713112 w_k1_FDX_s110
                                 + 0.675999 w_k1_IBM_s110
                                 + 0.69642 w_k1_INTC_s110
                                 + 0.750109 w_k1_KO_s110
                                 + 0.730836 w_k1_MAT_s110
                                 + 0.782527 w_k1_MSFT_s110
                                 + 0.712151 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.943598 w_k1_ADM_s110
                                 + 0.697706 w_k1_AXP_s110
                                 + 0.657827 w_k1_DIS_s110
                                 + 0.810873 w_k1_FDX_s110
                                 + 0.98526 w_k1_IBM_s110
                                 + 0.913115 w_k1_INTC_s110
                                 + 0.906658 w_k1_KO_s110
                                 + 0.666263 w_k1_MAT_s110
                                 + 0.76699 w_k1_MSFT_s110
                                 + 0.821048 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.920934 w_k1_ADM_s111
                                 + 0.582102 w_k1_AXP_s111
                                 + 0.613576 w_k1_DIS_s111
                                 + 0.726272 w_k1_FDX_s111
                                 + 0.733801 w_k1_IBM_s111
                                 + 0.599694 w_k1_INTC_s111
                                 + 0.5795 w_k1_KO_s111 + 0.516662 w_k1_MAT_s111
                                 + 0.614891 w_k1_MSFT_s111
                                 + 0.784452 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.45506 w_k1_ADM_s111
                                 + 0.91969 w_k1_AXP_s111
                                 + 1.02766 w_k1_DIS_s111
                                 + 0.94501 w_k1_FDX_s111
                                 + 1.18774 w_k1_IBM_s111
                                 + 0.732715 w_k1_INTC_s111
                                 + 1.06855 w_k1_KO_s111 + 1.26205 w_k1_MAT_s111
                                 + 0.915457 w_k1_MSFT_s111
                                 + 1.10522 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.895855 w_k1_ADM_s111
                                 + 1.12389 w_k1_AXP_s111
                                 + 1.60213 w_k1_DIS_s111 + 1.3305 w_k1_FDX_s111
                                 + 0.991885 w_k1_IBM_s111
                                 + 1.25662 w_k1_INTC_s111 + 1.1808 w_k1_KO_s111
                                 + 0.95047 w_k1_MAT_s111
                                 + 1.30378 w_k1_MSFT_s111
                                 + 1.26843 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.55684 w_k1_ADM_s111
                                 + 1.14113 w_k1_AXP_s111
                                 + 1.31225 w_k1_DIS_s111
                                 + 1.09261 w_k1_FDX_s111
                                 + 1.65085 w_k1_IBM_s111
                                 + 0.947811 w_k1_INTC_s111
                                 + 1.3278 w_k1_KO_s111 + 1.53776 w_k1_MAT_s111
                                 + 1.01674 w_k1_MSFT_s111
                                 + 1.19626 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.41678 w_k1_ADM_s111
                                 + 1.11506 w_k1_AXP_s111
                                 + 0.911768 w_k1_DIS_s111
                                 + 1.07557 w_k1_FDX_s111
                                 + 1.08089 w_k1_IBM_s111
                                 + 0.686222 w_k1_INTC_s111
                                 + 0.95868 w_k1_KO_s111 + 1.5349 w_k1_MAT_s111
                                 + 0.923832 w_k1_MSFT_s111
                                 + 1.01314 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.46629 w_k1_ADM_s111
                                 + 1.52078 w_k1_AXP_s111
                                 + 1.50613 w_k1_DIS_s111
                                 + 1.33565 w_k1_FDX_s111
                                 + 1.36238 w_k1_IBM_s111
                                 + 1.37541 w_k1_INTC_s111
                                 + 1.32416 w_k1_KO_s111 + 1.52979 w_k1_MAT_s111
                                 + 1.26834 w_k1_MSFT_s111
                                 + 1.26894 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.04047 w_k1_ADM_s111
                                 + 0.843132 w_k1_AXP_s111
                                 + 0.718837 w_k1_DIS_s111
                                 + 0.946806 w_k1_FDX_s111
                                 + 0.725915 w_k1_IBM_s111
                                 + 0.725778 w_k1_INTC_s111
                                 + 0.74092 w_k1_KO_s111 + 1.00323 w_k1_MAT_s111
                                 + 0.84715 w_k1_MSFT_s111
                                 + 0.904757 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.05207 w_k1_ADM_s111
                                 + 1.00825 w_k1_AXP_s111
                                 + 1.07808 w_k1_DIS_s111
                                 + 1.08251 w_k1_FDX_s111
                                 + 0.89974 w_k1_IBM_s111
                                 + 0.846588 w_k1_INTC_s111
                                 + 0.871943 w_k1_KO_s111
                                 + 0.936766 w_k1_MAT_s111
                                 + 0.917338 w_k1_MSFT_s111
                                 + 1.0703 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.872569 w_k1_ADM_s112
                                 + 0.585013 w_k1_AXP_s112
                                 + 0.620477 w_k1_DIS_s112
                                 + 0.736555 w_k1_FDX_s112
                                 + 0.777715 w_k1_IBM_s112
                                 + 0.792835 w_k1_INTC_s112
                                 + 0.749908 w_k1_KO_s112
                                 + 0.697914 w_k1_MAT_s112
                                 + 0.783545 w_k1_MSFT_s112
                                 + 0.779971 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.942306 w_k1_ADM_s112
                                 + 0.96738 w_k1_AXP_s112
                                 + 1.06856 w_k1_DIS_s112
                                 + 0.990026 w_k1_FDX_s112
                                 + 1.34828 w_k1_IBM_s112
                                 + 0.894504 w_k1_INTC_s112
                                 + 1.14351 w_k1_KO_s112 + 1.12027 w_k1_MAT_s112
                                 + 0.938656 w_k1_MSFT_s112
                                 + 0.836553 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.05384 w_k1_ADM_s112
                                 + 0.916091 w_k1_AXP_s112
                                 + 0.788828 w_k1_DIS_s112
                                 + 0.950439 w_k1_FDX_s112
                                 + 0.963162 w_k1_IBM_s112
                                 + 0.948145 w_k1_INTC_s112
                                 + 0.942428 w_k1_KO_s112 + 1.1411 w_k1_MAT_s112
                                 + 0.947278 w_k1_MSFT_s112
                                 + 0.85178 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.46623 w_k1_ADM_s112
                                 + 0.894671 w_k1_AXP_s112
                                 + 0.933096 w_k1_DIS_s112
                                 + 1.06289 w_k1_FDX_s112
                                 + 0.719752 w_k1_IBM_s112
                                 + 1.25269 w_k1_INTC_s112
                                 + 0.86494 w_k1_KO_s112
                                 + 0.462815 w_k1_MAT_s112
                                 + 1.14054 w_k1_MSFT_s112
                                 + 0.913225 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.14244 w_k1_ADM_s112
                                 + 0.952759 w_k1_AXP_s112
                                 + 1.17563 w_k1_DIS_s112
                                 + 1.03754 w_k1_FDX_s112
                                 + 0.998297 w_k1_IBM_s112
                                 + 0.986101 w_k1_INTC_s112
                                 + 1.18056 w_k1_KO_s112 + 1.10117 w_k1_MAT_s112
                                 + 1.15226 w_k1_MSFT_s112
                                 + 0.963117 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.35239 w_k1_ADM_s112
                                 + 0.871371 w_k1_AXP_s112
                                 + 1.08161 w_k1_DIS_s112
                                 + 1.00956 w_k1_FDX_s112
                                 + 1.21223 w_k1_IBM_s112
                                 + 1.10561 w_k1_INTC_s112
                                 + 1.28093 w_k1_KO_s112 + 1.29801 w_k1_MAT_s112
                                 + 1.27709 w_k1_MSFT_s112
                                 + 1.03199 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.46403 w_k1_ADM_s112
                                 + 1.45078 w_k1_AXP_s112
                                 + 1.72574 w_k1_DIS_s112
                                 + 1.33739 w_k1_FDX_s112
                                 + 1.02228 w_k1_IBM_s112
                                 + 1.47833 w_k1_INTC_s112
                                 + 1.34793 w_k1_KO_s112 + 1.22115 w_k1_MAT_s112
                                 + 1.43738 w_k1_MSFT_s112
                                 + 1.12714 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.31456 w_k1_ADM_s112
                                 + 1.31898 w_k1_AXP_s112
                                 + 1.04188 w_k1_DIS_s112
                                 + 0.847054 w_k1_FDX_s112
                                 + 1.65031 w_k1_IBM_s112
                                 + 1.22795 w_k1_INTC_s112 + 1.1744 w_k1_KO_s112
                                 + 1.42895 w_k1_MAT_s112
                                 + 0.937285 w_k1_MSFT_s112
                                 + 0.947934 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.08092 w_k1_ADM_s113
                                 + 0.954155 w_k1_AXP_s113
                                 + 0.757651 w_k1_DIS_s113 + 1.04 w_k1_FDX_s113
                                 + 0.678264 w_k1_IBM_s113
                                 + 0.921622 w_k1_INTC_s113
                                 + 0.814526 w_k1_KO_s113
                                 + 0.946519 w_k1_MAT_s113
                                 + 0.983602 w_k1_MSFT_s113
                                 + 1.02113 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.843823 w_k1_ADM_s113
                                 + 0.723 w_k1_AXP_s113 + 0.744224 w_k1_DIS_s113
                                 + 0.892724 w_k1_FDX_s113
                                 + 0.705599 w_k1_IBM_s113
                                 + 0.866794 w_k1_INTC_s113
                                 + 0.854065 w_k1_KO_s113
                                 + 0.622186 w_k1_MAT_s113
                                 + 0.801643 w_k1_MSFT_s113
                                 + 0.885863 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.0846 w_k1_ADM_s113
                                 + 1.13856 w_k1_AXP_s113
                                 + 1.03653 w_k1_DIS_s113
                                 + 0.949039 w_k1_FDX_s113
                                 + 1.2199 w_k1_IBM_s113
                                 + 1.14183 w_k1_INTC_s113
                                 + 1.15777 w_k1_KO_s113 + 1.37438 w_k1_MAT_s113
                                 + 0.935514 w_k1_MSFT_s113
                                 + 1.01821 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.856545 w_k1_ADM_s113
                                 + 0.659221 w_k1_AXP_s113
                                 + 0.572235 w_k1_DIS_s113
                                 + 0.709819 w_k1_FDX_s113
                                 + 0.669028 w_k1_IBM_s113
                                 + 0.761875 w_k1_INTC_s113
                                 + 0.693754 w_k1_KO_s113
                                 + 0.440522 w_k1_MAT_s113
                                 + 0.712946 w_k1_MSFT_s113
                                 + 0.831476 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.13609 w_k1_ADM_s113
                                 + 0.839509 w_k1_AXP_s113
                                 + 0.906818 w_k1_DIS_s113
                                 + 0.858424 w_k1_FDX_s113
                                 + 1.38988 w_k1_IBM_s113
                                 + 1.01317 w_k1_INTC_s113
                                 + 1.19186 w_k1_KO_s113 + 1.30668 w_k1_MAT_s113
                                 + 0.905736 w_k1_MSFT_s113
                                 + 1.04681 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.70202 w_k1_ADM_s113
                                 + 1.5983 w_k1_AXP_s113 + 1.99865 w_k1_DIS_s113
                                 + 1.42045 w_k1_FDX_s113
                                 + 1.57842 w_k1_IBM_s113
                                 + 1.31808 w_k1_INTC_s113
                                 + 1.87321 w_k1_KO_s113 + 1.6909 w_k1_MAT_s113
                                 + 1.33551 w_k1_MSFT_s113
                                 + 1.50048 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.04976 w_k1_ADM_s113
                                 + 0.757376 w_k1_AXP_s113
                                 + 0.910271 w_k1_DIS_s113
                                 + 0.934851 w_k1_FDX_s113
                                 + 0.894445 w_k1_IBM_s113
                                 + 1.05026 w_k1_INTC_s113
                                 + 1.06042 w_k1_KO_s113
                                 + 0.885825 w_k1_MAT_s113
                                 + 1.09796 w_k1_MSFT_s113
                                 + 1.02229 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.26263 w_k1_ADM_s113
                                 + 1.34908 w_k1_AXP_s113
                                 + 1.34028 w_k1_DIS_s113
                                 + 1.31948 w_k1_FDX_s113
                                 + 0.915086 w_k1_IBM_s113
                                 + 1.13833 w_k1_INTC_s113
                                 + 1.04501 w_k1_KO_s113 + 1.04239 w_k1_MAT_s113
                                 + 1.20813 w_k1_MSFT_s113 + 1.1459 w_k1_PG_s113
                                 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.04008 w_k1_ADM_s114
                                 + 0.949822 w_k1_AXP_s114
                                 + 0.921334 w_k1_DIS_s114
                                 + 1.03624 w_k1_FDX_s114
                                 + 0.646304 w_k1_IBM_s114
                                 + 0.93485 w_k1_INTC_s114
                                 + 0.741266 w_k1_KO_s114
                                 + 0.719146 w_k1_MAT_s114
                                 + 1.01476 w_k1_MSFT_s114
                                 + 0.83726 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.02018 w_k1_ADM_s114
                                 + 1.1869 w_k1_AXP_s114 + 1.335 w_k1_DIS_s114
                                 + 1.04417 w_k1_FDX_s114
                                 + 1.40653 w_k1_IBM_s114
                                 + 1.37235 w_k1_INTC_s114
                                 + 1.28519 w_k1_KO_s114 + 1.52691 w_k1_MAT_s114
                                 + 1.25432 w_k1_MSFT_s114
                                 + 1.04146 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.11206 w_k1_ADM_s114
                                 + 1.54813 w_k1_AXP_s114
                                 + 1.21029 w_k1_DIS_s114
                                 + 1.23263 w_k1_FDX_s114
                                 + 1.07411 w_k1_IBM_s114
                                 + 1.25614 w_k1_INTC_s114
                                 + 0.892884 w_k1_KO_s114
                                 + 1.00566 w_k1_MAT_s114
                                 + 1.09312 w_k1_MSFT_s114 + 1.1588 w_k1_PG_s114
                                 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.03117 w_k1_ADM_s114
                                 + 0.952471 w_k1_AXP_s114
                                 + 0.884873 w_k1_DIS_s114
                                 + 0.824924 w_k1_FDX_s114
                                 + 1.04351 w_k1_IBM_s114
                                 + 0.980696 w_k1_INTC_s114
                                 + 0.884649 w_k1_KO_s114
                                 + 0.943951 w_k1_MAT_s114
                                 + 0.872354 w_k1_MSFT_s114
                                 + 0.952314 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.887841 w_k1_ADM_s114
                                 + 0.741476 w_k1_AXP_s114
                                 + 0.734559 w_k1_DIS_s114
                                 + 0.828738 w_k1_FDX_s114
                                 + 0.681099 w_k1_IBM_s114
                                 + 0.944884 w_k1_INTC_s114
                                 + 0.743026 w_k1_KO_s114
                                 + 0.709869 w_k1_MAT_s114
                                 + 0.87745 w_k1_MSFT_s114
                                 + 0.852953 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.31019 w_k1_ADM_s114
                                 + 1.02564 w_k1_AXP_s114
                                 + 1.38501 w_k1_DIS_s114
                                 + 1.25853 w_k1_FDX_s114
                                 + 0.919504 w_k1_IBM_s114
                                 + 1.27717 w_k1_INTC_s114
                                 + 0.973219 w_k1_KO_s114
                                 + 0.631808 w_k1_MAT_s114
                                 + 1.19418 w_k1_MSFT_s114
                                 + 0.956784 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.18906 w_k1_ADM_s114
                                 + 1.20699 w_k1_AXP_s114
                                 + 1.50471 w_k1_DIS_s114
                                 + 1.25777 w_k1_FDX_s114
                                 + 1.14662 w_k1_IBM_s114
                                 + 1.05043 w_k1_INTC_s114
                                 + 1.03213 w_k1_KO_s114
                                 + 0.987691 w_k1_MAT_s114
                                 + 1.06892 w_k1_MSFT_s114
                                 + 0.980929 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.875641 w_k1_ADM_s114
                                 + 0.830059 w_k1_AXP_s114
                                 + 1.12293 w_k1_DIS_s114
                                 + 1.05828 w_k1_FDX_s114
                                 + 0.850892 w_k1_IBM_s114
                                 + 0.976421 w_k1_INTC_s114
                                 + 0.908882 w_k1_KO_s114
                                 + 0.84879 w_k1_MAT_s114
                                 + 1.17778 w_k1_MSFT_s114
                                 + 0.922085 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.979207 w_k1_ADM_s115
                                 + 0.854161 w_k1_AXP_s115
                                 + 0.614767 w_k1_DIS_s115
                                 + 0.770749 w_k1_FDX_s115
                                 + 0.774951 w_k1_IBM_s115
                                 + 0.728566 w_k1_INTC_s115
                                 + 0.75931 w_k1_KO_s115 + 0.82399 w_k1_MAT_s115
                                 + 0.762868 w_k1_MSFT_s115
                                 + 0.780628 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.975607 w_k1_ADM_s115
                                 + 1.06739 w_k1_AXP_s115
                                 + 1.42077 w_k1_DIS_s115
                                 + 1.08986 w_k1_FDX_s115
                                 + 0.897431 w_k1_IBM_s115
                                 + 1.10933 w_k1_INTC_s115
                                 + 1.18526 w_k1_KO_s115
                                 + 0.668693 w_k1_MAT_s115
                                 + 1.19042 w_k1_MSFT_s115
                                 + 1.00481 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.12891 w_k1_ADM_s115
                                 + 0.976088 w_k1_AXP_s115
                                 + 1.41174 w_k1_DIS_s115
                                 + 1.06067 w_k1_FDX_s115
                                 + 1.31178 w_k1_IBM_s115
                                 + 1.14512 w_k1_INTC_s115
                                 + 1.35776 w_k1_KO_s115
                                 + 0.874519 w_k1_MAT_s115
                                 + 1.1172 w_k1_MSFT_s115 + 1.0693 w_k1_PG_s115
                                 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.99367 w_k1_ADM_s115
                                 + 0.802176 w_k1_AXP_s115
                                 + 0.911565 w_k1_DIS_s115
                                 + 0.838688 w_k1_FDX_s115
                                 + 1.01059 w_k1_IBM_s115
                                 + 0.940842 w_k1_INTC_s115
                                 + 0.930653 w_k1_KO_s115
                                 + 0.673816 w_k1_MAT_s115
                                 + 0.922953 w_k1_MSFT_s115
                                 + 0.969024 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.712925 w_k1_ADM_s115
                                 + 0.540307 w_k1_AXP_s115
                                 + 0.715687 w_k1_DIS_s115
                                 + 0.784999 w_k1_FDX_s115
                                 + 0.628493 w_k1_IBM_s115
                                 + 0.817265 w_k1_INTC_s115
                                 + 0.710716 w_k1_KO_s115
                                 + 0.445697 w_k1_MAT_s115
                                 + 0.822418 w_k1_MSFT_s115
                                 + 0.732901 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.22013 w_k1_ADM_s115
                                 + 1.47641 w_k1_AXP_s115
                                 + 1.56797 w_k1_DIS_s115
                                 + 1.26125 w_k1_FDX_s115
                                 + 1.21643 w_k1_IBM_s115
                                 + 1.36358 w_k1_INTC_s115
                                 + 1.27903 w_k1_KO_s115 + 1.03458 w_k1_MAT_s115
                                 + 1.4336 w_k1_MSFT_s115 + 1.15655 w_k1_PG_s115
                                 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.86137 w_k1_ADM_s115
                                 + 1.13401 w_k1_AXP_s115
                                 + 0.777463 w_k1_DIS_s115
                                 + 0.870159 w_k1_FDX_s115
                                 + 1.15649 w_k1_IBM_s115
                                 + 1.17203 w_k1_INTC_s115
                                 + 0.984145 w_k1_KO_s115
                                 + 0.992776 w_k1_MAT_s115
                                 + 0.989815 w_k1_MSFT_s115
                                 + 0.959555 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.07339 w_k1_ADM_s115
                                 + 1.05552 w_k1_AXP_s115
                                 + 0.989063 w_k1_DIS_s115
                                 + 1.10113 w_k1_FDX_s115
                                 + 0.718528 w_k1_IBM_s115
                                 + 0.949379 w_k1_INTC_s115
                                 + 0.731842 w_k1_KO_s115
                                 + 0.479333 w_k1_MAT_s115
                                 + 0.97274 w_k1_MSFT_s115
                                 + 0.85994 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.2838 w_k1_ADM_s116
                                 + 1.07585 w_k1_AXP_s116
                                 + 1.62594 w_k1_DIS_s116
                                 + 1.18035 w_k1_FDX_s116
                                 + 1.29179 w_k1_IBM_s116
                                 + 1.10951 w_k1_INTC_s116
                                 + 1.36944 w_k1_KO_s116 + 1.02233 w_k1_MAT_s116
                                 + 1.39631 w_k1_MSFT_s116
                                 + 1.18358 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.802511 w_k1_ADM_s116
                                 + 0.728221 w_k1_AXP_s116
                                 + 0.879786 w_k1_DIS_s116
                                 + 0.879814 w_k1_FDX_s116
                                 + 0.751052 w_k1_IBM_s116
                                 + 0.925091 w_k1_INTC_s116
                                 + 0.726991 w_k1_KO_s116 + 0.5492 w_k1_MAT_s116
                                 + 0.924724 w_k1_MSFT_s116
                                 + 0.8654 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.10093 w_k1_ADM_s116
                                 + 1.08768 w_k1_AXP_s116
                                 + 1.32393 w_k1_DIS_s116
                                 + 1.10604 w_k1_FDX_s116
                                 + 0.774386 w_k1_IBM_s116
                                 + 1.23556 w_k1_INTC_s116
                                 + 0.936015 w_k1_KO_s116
                                 + 0.755609 w_k1_MAT_s116
                                 + 1.07991 w_k1_MSFT_s116
                                 + 1.14077 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.816641 w_k1_ADM_s116
                                 + 0.732542 w_k1_AXP_s116
                                 + 0.670887 w_k1_DIS_s116
                                 + 0.748175 w_k1_FDX_s116
                                 + 0.676014 w_k1_IBM_s116
                                 + 0.811648 w_k1_INTC_s116
                                 + 0.58594 w_k1_KO_s116
                                 + 0.612562 w_k1_MAT_s116
                                 + 0.67224 w_k1_MSFT_s116
                                 + 0.831422 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.804489 w_k1_ADM_s116
                                 + 1.30336 w_k1_AXP_s116
                                 + 1.02466 w_k1_DIS_s116
                                 + 0.976488 w_k1_FDX_s116
                                 + 1.45384 w_k1_IBM_s116
                                 + 1.07053 w_k1_INTC_s116
                                 + 1.07913 w_k1_KO_s116 + 1.61548 w_k1_MAT_s116
                                 + 0.862651 w_k1_MSFT_s116
                                 + 1.02678 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.903954 w_k1_ADM_s116
                                 + 1.12359 w_k1_AXP_s116
                                 + 0.869459 w_k1_DIS_s116
                                 + 0.894882 w_k1_FDX_s116
                                 + 1.03549 w_k1_IBM_s116
                                 + 1.15854 w_k1_INTC_s116
                                 + 0.851971 w_k1_KO_s116
                                 + 1.04005 w_k1_MAT_s116
                                 + 0.90952 w_k1_MSFT_s116
                                 + 1.00274 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.27085 w_k1_ADM_s116
                                 + 1.42367 w_k1_AXP_s116
                                 + 1.12293 w_k1_DIS_s116
                                 + 1.09914 w_k1_FDX_s116
                                 + 1.03313 w_k1_IBM_s116
                                 + 1.11461 w_k1_INTC_s116
                                 + 0.969907 w_k1_KO_s116
                                 + 1.11105 w_k1_MAT_s116
                                 + 1.06976 w_k1_MSFT_s116
                                 + 0.984958 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.39049 w_k1_ADM_s116
                                 + 1.20883 w_k1_AXP_s116
                                 + 1.41103 w_k1_DIS_s116
                                 + 1.13259 w_k1_FDX_s116
                                 + 1.05952 w_k1_IBM_s116
                                 + 0.967869 w_k1_INTC_s116
                                 + 0.941451 w_k1_KO_s116
                                 + 0.637341 w_k1_MAT_s116
                                 + 1.24308 w_k1_MSFT_s116
                                 + 1.14244 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.01414 w_k1_ADM_s117
                                 + 0.816105 w_k1_AXP_s117
                                 + 1.03319 w_k1_DIS_s117
                                 + 0.886844 w_k1_FDX_s117
                                 + 0.885617 w_k1_IBM_s117
                                 + 0.887085 w_k1_INTC_s117
                                 + 0.97391 w_k1_KO_s117
                                 + 0.947212 w_k1_MAT_s117
                                 + 1.03908 w_k1_MSFT_s117
                                 + 0.979081 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.02968 w_k1_ADM_s117
                                 + 1.26539 w_k1_AXP_s117
                                 + 1.19054 w_k1_DIS_s117 + 1.0771 w_k1_FDX_s117
                                 + 0.889516 w_k1_IBM_s117
                                 + 1.13919 w_k1_INTC_s117
                                 + 1.04774 w_k1_KO_s117 + 1.36187 w_k1_MAT_s117
                                 + 1.17803 w_k1_MSFT_s117
                                 + 1.03623 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.33714 w_k1_ADM_s117
                                 + 1.69208 w_k1_AXP_s117
                                 + 1.43056 w_k1_DIS_s117
                                 + 1.38796 w_k1_FDX_s117
                                 + 1.15593 w_k1_IBM_s117
                                 + 1.40752 w_k1_INTC_s117
                                 + 1.10194 w_k1_KO_s117 + 1.62891 w_k1_MAT_s117
                                 + 1.06797 w_k1_MSFT_s117
                                 + 1.21349 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.16013 w_k1_ADM_s117
                                 + 1.18765 w_k1_AXP_s117 + 1.4739 w_k1_DIS_s117
                                 + 1.14946 w_k1_FDX_s117 + 1.4567 w_k1_IBM_s117
                                 + 1.02753 w_k1_INTC_s117
                                 + 1.30333 w_k1_KO_s117 + 1.49681 w_k1_MAT_s117
                                 + 1.13555 w_k1_MSFT_s117
                                 + 1.14284 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.870393 w_k1_ADM_s117
                                 + 0.979674 w_k1_AXP_s117
                                 + 0.94419 w_k1_DIS_s117
                                 + 1.00398 w_k1_FDX_s117
                                 + 1.17899 w_k1_IBM_s117
                                 + 1.04478 w_k1_INTC_s117
                                 + 0.95766 w_k1_KO_s117 + 1.34802 w_k1_MAT_s117
                                 + 0.975882 w_k1_MSFT_s117
                                 + 0.918621 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.11485 w_k1_ADM_s117
                                 + 0.886563 w_k1_AXP_s117
                                 + 1.07016 w_k1_DIS_s117
                                 + 1.17498 w_k1_FDX_s117
                                 + 0.669596 w_k1_IBM_s117
                                 + 1.09337 w_k1_INTC_s117
                                 + 0.789482 w_k1_KO_s117
                                 + 0.68058 w_k1_MAT_s117
                                 + 1.14804 w_k1_MSFT_s117
                                 + 0.915401 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.26889 w_k1_ADM_s117
                                 + 1.43626 w_k1_AXP_s117
                                 + 1.87406 w_k1_DIS_s117
                                 + 1.44058 w_k1_FDX_s117
                                 + 1.24308 w_k1_IBM_s117
                                 + 1.46086 w_k1_INTC_s117
                                 + 1.23281 w_k1_KO_s117 + 1.47795 w_k1_MAT_s117
                                 + 1.42009 w_k1_MSFT_s117
                                 + 1.19147 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.705339 w_k1_ADM_s117
                                 + 0.645055 w_k1_AXP_s117
                                 + 0.648081 w_k1_DIS_s117
                                 + 0.770243 w_k1_FDX_s117
                                 + 0.741688 w_k1_IBM_s117
                                 + 0.954107 w_k1_INTC_s117
                                 + 0.766499 w_k1_KO_s117
                                 + 0.576324 w_k1_MAT_s117
                                 + 0.812651 w_k1_MSFT_s117
                                 + 0.778892 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0237552146168263 w_k1_ADM_s000
                                 + 0.0574595676734151 w_k1_ADM_s001
                                 + 0.0187074183081041 w_k1_ADM_s002
                                 + 0.0565080197093178 w_k1_ADM_s003
                                 + 0.0321531624666278 w_k1_ADM_s004
                                 + 0.0253415597111753 w_k1_ADM_s005
                                 + 0.0825737243442028 w_k1_ADM_s006
                                 + 0.0521304553906775 w_k1_ADM_s007
                                 + 0.0837426971941199 w_k1_ADM_s010
                                 + 0.0848901197071578 w_k1_ADM_s011
                                 + 0.047498830531667 w_k1_ADM_s012
                                 + 0.0630642175484216 w_k1_ADM_s013
                                 + 0.0263099595548481 w_k1_ADM_s014
                                 + 0.0290120244496045 w_k1_ADM_s015
                                 + 0.0526212015550049 w_k1_ADM_s016
                                 + 0.0199347093295146 w_k1_ADM_s017
                                 + 0.023347917925213 w_k1_ADM_s100
                                 + 0.0157904443783678 w_k1_ADM_s101
                                 + 0.0137169952033964 w_k1_ADM_s102
                                 + 0.0173318189706628 w_k1_ADM_s103
                                 + 0.00656625360907426 w_k1_ADM_s104
                                 + 0.0104615650562725 w_k1_ADM_s105
                                 + 0.0328848629783723 w_k1_ADM_s106
                                 + 0.0146598147568137 w_k1_ADM_s107
                                 + 0.0168401223890422 w_k1_ADM_s110
                                 + 0.0176258689622639 w_k1_ADM_s111
                                 + 0.0139844441084252 w_k1_ADM_s112
                                 + 0.0305951488986611 w_k1_ADM_s113
                                 + 0.00720472443298104 w_k1_ADM_s114
                                 + 0.00856618509563955 w_k1_ADM_s115
                                 + 0.00937965240847228 w_k1_ADM_s116
                                 + 0.0155274115326974 w_k1_ADM_s117
                                 + 0.0251569784526105 w_k1_AXP_s000
                                 + 0.0567936659426824 w_k1_AXP_s001
                                 + 0.02147999950023 w_k1_AXP_s002
                                 + 0.0621264099067392 w_k1_AXP_s003
                                 + 0.0241386653337553 w_k1_AXP_s004
                                 + 0.0309787746609063 w_k1_AXP_s005
                                 + 0.0777487115541359 w_k1_AXP_s006
                                 + 0.0422760911638692 w_k1_AXP_s007
                                 + 0.0905320944752844 w_k1_AXP_s010
                                 + 0.0932834367039355 w_k1_AXP_s011
                                 + 0.0503024052213341 w_k1_AXP_s012
                                 + 0.0680008779978172 w_k1_AXP_s013
                                 + 0.0243464033384402 w_k1_AXP_s014
                                 + 0.0251853978304665 w_k1_AXP_s015
                                 + 0.0528394784011491 w_k1_AXP_s016
                                 + 0.0221445155657005 w_k1_AXP_s017
                                 + 0.0284398852371345 w_k1_AXP_s100
                                 + 0.0179790375451525 w_k1_AXP_s101
                                 + 0.0115466599560609 w_k1_AXP_s102
                                 + 0.0162115673304595 w_k1_AXP_s103
                                 + 0.00722184730820095 w_k1_AXP_s104
                                 + 0.00915096112103744 w_k1_AXP_s105
                                 + 0.0278930403749341 w_k1_AXP_s106
                                 + 0.0127435227196925 w_k1_AXP_s107
                                 + 0.0171651024711552 w_k1_AXP_s110
                                 + 0.0145892791091887 w_k1_AXP_s111
                                 + 0.0113415355484767 w_k1_AXP_s112
                                 + 0.0288014606340992 w_k1_AXP_s113
                                 + 0.00709365371780892 w_k1_AXP_s114
                                 + 0.00832920559960867 w_k1_AXP_s115
                                 + 0.01027494836921 w_k1_AXP_s116
                                 + 0.0162784543063577 w_k1_AXP_s117
                                 + 0.0243823199768957 w_k1_DIS_s000
                                 + 0.0593954213562966 w_k1_DIS_s001
                                 + 0.0199604983357884 w_k1_DIS_s002
                                 + 0.0598277049353598 w_k1_DIS_s003
                                 + 0.0242809920412674 w_k1_DIS_s004
                                 + 0.0304126156568788 w_k1_DIS_s005
                                 + 0.0817099120166296 w_k1_DIS_s006
                                 + 0.0444512245334193 w_k1_DIS_s007
                                 + 0.100606687846239 w_k1_DIS_s010
                                 + 0.0957108230872228 w_k1_DIS_s011
                                 + 0.0566225795864653 w_k1_DIS_s012
                                 + 0.0656001264122418 w_k1_DIS_s013
                                 + 0.0269784391723516 w_k1_DIS_s014
                                 + 0.0256583523615676 w_k1_DIS_s015
                                 + 0.0553229865941015 w_k1_DIS_s016
                                 + 0.0223166707634093 w_k1_DIS_s017
                                 + 0.027279412323118 w_k1_DIS_s100
                                 + 0.0165107288313746 w_k1_DIS_s101
                                 + 0.0121918391331303 w_k1_DIS_s102
                                 + 0.0169715816577861 w_k1_DIS_s103
                                 + 0.00718843827826756 w_k1_DIS_s104
                                 + 0.010164489189476 w_k1_DIS_s105
                                 + 0.0312277346884376 w_k1_DIS_s106
                                 + 0.0134257988833362 w_k1_DIS_s107
                                 + 0.0158010632194127 w_k1_DIS_s110
                                 + 0.0150428823804127 w_k1_DIS_s111
                                 + 0.0122242581490226 w_k1_DIS_s112
                                 + 0.0288533326020203 w_k1_DIS_s113
                                 + 0.00779935896675259 w_k1_DIS_s114
                                 + 0.00920175016900036 w_k1_DIS_s115
                                 + 0.00990411655794244 w_k1_DIS_s116
                                 + 0.0174389583714878 w_k1_DIS_s117
                                 + 0.0240248455847466 w_k1_FDX_s000
                                 + 0.0558335062207777 w_k1_FDX_s001
                                 + 0.0197222743065633 w_k1_FDX_s002
                                 + 0.0573629688300591 w_k1_FDX_s003
                                 + 0.0253426614680801 w_k1_FDX_s004
                                 + 0.0290139995459246 w_k1_FDX_s005
                                 + 0.0789127080061245 w_k1_FDX_s006
                                 + 0.0465180236396144 w_k1_FDX_s007
                                 + 0.09354265208874 w_k1_FDX_s010
                                 + 0.0928754416375778 w_k1_FDX_s011
                                 + 0.05213467025634 w_k1_FDX_s012
                                 + 0.0649663998670478 w_k1_FDX_s013
                                 + 0.027068617293631 w_k1_FDX_s014
                                 + 0.0267305676849982 w_k1_FDX_s015
                                 + 0.0532635119378638 w_k1_FDX_s016
                                 + 0.0208869318004802 w_k1_FDX_s017
                                 + 0.0268776680734843 w_k1_FDX_s100
                                 + 0.0172244455211298 w_k1_FDX_s101
                                 + 0.0112079696240709 w_k1_FDX_s102
                                 + 0.015566373043416 w_k1_FDX_s103
                                 + 0.00681746930192822 w_k1_FDX_s104
                                 + 0.0101971884049778 w_k1_FDX_s105
                                 + 0.0310792256861509 w_k1_FDX_s106
                                 + 0.0138134236604789 w_k1_FDX_s107
                                 + 0.0170834985410775 w_k1_FDX_s110
                                 + 0.0153229749004011 w_k1_FDX_s111
                                 + 0.0119178649804863 w_k1_FDX_s112
                                 + 0.0283185190307523 w_k1_FDX_s113
                                 + 0.00717897065777394 w_k1_FDX_s114
                                 + 0.00833746863329374 w_k1_FDX_s115
                                 + 0.00925556771922246 w_k1_FDX_s116
                                 + 0.0161035274188954 w_k1_FDX_s117
                                 + 0.0240631066379608 w_k1_IBM_s000
                                 + 0.0561016535192094 w_k1_IBM_s001
                                 + 0.0193525105288487 w_k1_IBM_s002
                                 + 0.059609826851161 w_k1_IBM_s003
                                 + 0.0246469129915273 w_k1_IBM_s004
                                 + 0.030135147807103 w_k1_IBM_s005
                                 + 0.0802242381942387 w_k1_IBM_s006
                                 + 0.045985887590002 w_k1_IBM_s007
                                 + 0.0972460118596666 w_k1_IBM_s010
                                 + 0.0966865226843608 w_k1_IBM_s011
                                 + 0.052489651843169 w_k1_IBM_s012
                                 + 0.0696241229237035 w_k1_IBM_s013
                                 + 0.0294865327398752 w_k1_IBM_s014
                                 + 0.026920578553194 w_k1_IBM_s015
                                 + 0.0544257973527274 w_k1_IBM_s016
                                 + 0.0224064280751345 w_k1_IBM_s017
                                 + 0.0265921565577828 w_k1_IBM_s100
                                 + 0.01598418802741 w_k1_IBM_s101
                                 + 0.0124092492142476 w_k1_IBM_s102
                                 + 0.0163828918658011 w_k1_IBM_s103
                                 + 0.00725091555403982 w_k1_IBM_s104
                                 + 0.0107345182619248 w_k1_IBM_s105
                                 + 0.0291509574827917 w_k1_IBM_s106
                                 + 0.0138082154611482 w_k1_IBM_s107
                                 + 0.0176959987082204 w_k1_IBM_s110
                                 + 0.0149947467056376 w_k1_IBM_s111
                                 + 0.0126802447224719 w_k1_IBM_s112
                                 + 0.0271977589728771 w_k1_IBM_s113
                                 + 0.00689568481994621 w_k1_IBM_s114
                                 + 0.00818499230059688 w_k1_IBM_s115
                                 + 0.0097702882735247 w_k1_IBM_s116
                                 + 0.0155467205571258 w_k1_IBM_s117
                                 + 0.0218896424445352 w_k1_INTC_s000
                                 + 0.054731455079245 w_k1_INTC_s001
                                 + 0.0187917332942239 w_k1_INTC_s002
                                 + 0.0591534351240327 w_k1_INTC_s003
                                 + 0.0192400463423079 w_k1_INTC_s004
                                 + 0.0308957996601807 w_k1_INTC_s005
                                 + 0.0835765287563051 w_k1_INTC_s006
                                 + 0.0441024282862948 w_k1_INTC_s007
                                 + 0.0912270281370131 w_k1_INTC_s010
                                 + 0.0972394570486154 w_k1_INTC_s011
                                 + 0.0588664317507322 w_k1_INTC_s012
                                 + 0.0632755303330152 w_k1_INTC_s013
                                 + 0.0262370003869244 w_k1_INTC_s014
                                 + 0.0252272319929215 w_k1_INTC_s015
                                 + 0.0579761498009303 w_k1_INTC_s016
                                 + 0.0228855075544587 w_k1_INTC_s017
                                 + 0.0264230409291602 w_k1_INTC_s100
                                 + 0.0167473622019356 w_k1_INTC_s101
                                 + 0.0123291829299864 w_k1_INTC_s102
                                 + 0.0183452364927167 w_k1_INTC_s103
                                 + 0.00716458897440004 w_k1_INTC_s104
                                 + 0.00943397234062748 w_k1_INTC_s105
                                 + 0.0258298138939335 w_k1_INTC_s106
                                 + 0.0136330021014231 w_k1_INTC_s107
                                 + 0.0156403027478858 w_k1_INTC_s110
                                 + 0.0125878227309952 w_k1_INTC_s111
                                 + 0.0125311406812082 w_k1_INTC_s112
                                 + 0.0286004337245831 w_k1_INTC_s113
                                 + 0.00745885251925523 w_k1_INTC_s114
                                 + 0.00870073481413023 w_k1_INTC_s115
                                 + 0.0100898883397495 w_k1_INTC_s116
                                 + 0.0164799027922269 w_k1_INTC_s117
                                 + 0.0236017673272404 w_k1_KO_s000
                                 + 0.0594353157450945 w_k1_KO_s001
                                 + 0.0221641301854803 w_k1_KO_s002
                                 + 0.058316451859446 w_k1_KO_s003
                                 + 0.0232654381838499 w_k1_KO_s004
                                 + 0.028933073315585 w_k1_KO_s005
                                 + 0.0756023940473724 w_k1_KO_s006
                                 + 0.0426665573651884 w_k1_KO_s007
                                 + 0.0900726312418736 w_k1_KO_s010
                                 + 0.0953910813105104 w_k1_KO_s011
                                 + 0.0526350434090232 w_k1_KO_s012
                                 + 0.06785458597063 w_k1_KO_s013
                                 + 0.0290670282925369 w_k1_KO_s014
                                 + 0.0296990338744864 w_k1_KO_s015
                                 + 0.0551909247660345 w_k1_KO_s016
                                 + 0.0209353053503501 w_k1_KO_s017
                                 + 0.0286876892965378 w_k1_KO_s100
                                 + 0.0165315455531811 w_k1_KO_s101
                                 + 0.0115868756664687 w_k1_KO_s102
                                 + 0.0155574019128733 w_k1_KO_s103
                                 + 0.00647001991607977 w_k1_KO_s104
                                 + 0.00975246740680992 w_k1_KO_s105
                                 + 0.0297131848804146 w_k1_KO_s106
                                 + 0.0141504566478318 w_k1_KO_s107
                                 + 0.017708612448066 w_k1_KO_s110
                                 + 0.0138470863375215 w_k1_KO_s111
                                 + 0.0128898456595304 w_k1_KO_s112
                                 + 0.0292066560674974 w_k1_KO_s113
                                 + 0.00657143636795082 w_k1_KO_s114
                                 + 0.00851844501257388 w_k1_KO_s115
                                 + 0.00870660245733945 w_k1_KO_s116
                                 + 0.0155407141223553 w_k1_KO_s117
                                 + 0.0224011577489601 w_k1_MAT_s000
                                 + 0.0523504645517663 w_k1_MAT_s001
                                 + 0.0215140097427188 w_k1_MAT_s002
                                 + 0.0583668790042577 w_k1_MAT_s003
                                 + 0.0254494868264779 w_k1_MAT_s004
                                 + 0.0335212119984783 w_k1_MAT_s005
                                 + 0.061188948107145 w_k1_MAT_s006
                                 + 0.0407935422807893 w_k1_MAT_s007
                                 + 0.0847863788183584 w_k1_MAT_s010
                                 + 0.0988240527550589 w_k1_MAT_s011
                                 + 0.0562168972266931 w_k1_MAT_s012
                                 + 0.0690697447987945 w_k1_MAT_s013
                                 + 0.0265858082805787 w_k1_MAT_s014
                                 + 0.0280665984802532 w_k1_MAT_s015
                                 + 0.0496026644216707 w_k1_MAT_s016
                                 + 0.0215356803770797 w_k1_MAT_s017
                                 + 0.0292298249408183 w_k1_MAT_s100
                                 + 0.0175232478268855 w_k1_MAT_s101
                                 + 0.00995199375607002 w_k1_MAT_s102
                                 + 0.0186052484821253 w_k1_MAT_s103
                                 + 0.00642473155738058 w_k1_MAT_s104
                                 + 0.0113135227788568 w_k1_MAT_s105
                                 + 0.0314101378678853 w_k1_MAT_s106
                                 + 0.0131695750519755 w_k1_MAT_s107
                                 + 0.0177372997283033 w_k1_MAT_s110
                                 + 0.0160224188695238 w_k1_MAT_s111
                                 + 0.0125574567085837 w_k1_MAT_s112
                                 + 0.0292827528631371 w_k1_MAT_s113
                                 + 0.0066566810090796 w_k1_MAT_s114
                                 + 0.00621755363094757 w_k1_MAT_s115
                                 + 0.00920917280530704 w_k1_MAT_s116
                                 + 0.0181456815618531 w_k1_MAT_s117
                                 + 0.0227315706794081 w_k1_MSFT_s000
                                 + 0.056418214447273 w_k1_MSFT_s001
                                 + 0.0202693201315276 w_k1_MSFT_s002
                                 + 0.0606252057206373 w_k1_MSFT_s003
                                 + 0.0235874546911878 w_k1_MSFT_s004
                                 + 0.0332553458242533 w_k1_MSFT_s005
                                 + 0.0753587298499786 w_k1_MSFT_s006
                                 + 0.0446470587392311 w_k1_MSFT_s007
                                 + 0.0984394242390281 w_k1_MSFT_s010
                                 + 0.084763070791414 w_k1_MSFT_s011
                                 + 0.0512019959825422 w_k1_MSFT_s012
                                 + 0.0678831048754091 w_k1_MSFT_s013
                                 + 0.0261556152679921 w_k1_MSFT_s014
                                 + 0.0279822382141811 w_k1_MSFT_s015
                                 + 0.0558176503504278 w_k1_MSFT_s016
                                 + 0.022521682281401 w_k1_MSFT_s017
                                 + 0.0274126246884676 w_k1_MSFT_s100
                                 + 0.016379795142463 w_k1_MSFT_s101
                                 + 0.0110811911959973 w_k1_MSFT_s102
                                 + 0.0191874223022831 w_k1_MSFT_s103
                                 + 0.00679235759273083 w_k1_MSFT_s104
                                 + 0.0106391894659105 w_k1_MSFT_s105
                                 + 0.0293046289189579 w_k1_MSFT_s106
                                 + 0.012483491664285 w_k1_MSFT_s107
                                 + 0.0155555268341268 w_k1_MSFT_s110
                                 + 0.0137351677575577 w_k1_MSFT_s111
                                 + 0.0128101572935382 w_k1_MSFT_s112
                                 + 0.0276049660242772 w_k1_MSFT_s113
                                 + 0.00732725120116653 w_k1_MSFT_s114
                                 + 0.00878620050048926 w_k1_MSFT_s115
                                 + 0.00933134941801257 w_k1_MSFT_s116
                                 + 0.0164872420240869 w_k1_MSFT_s117
                                 + 0.0233254192105683 w_k1_PG_s000
                                 + 0.0602892104243721 w_k1_PG_s001
                                 + 0.020164408332648 w_k1_PG_s002
                                 + 0.0584424617416534 w_k1_PG_s003
                                 + 0.0226498194996676 w_k1_PG_s004
                                 + 0.0300898082212506 w_k1_PG_s005
                                 + 0.0831593020052 w_k1_PG_s006
                                 + 0.0452748388350195 w_k1_PG_s007
                                 + 0.0968527054922749 w_k1_PG_s010
                                 + 0.0906250402674066 w_k1_PG_s011
                                 + 0.0540532907542146 w_k1_PG_s012
                                 + 0.0680122166624396 w_k1_PG_s013
                                 + 0.0251050902157503 w_k1_PG_s014
                                 + 0.0278271688052082 w_k1_PG_s015
                                 + 0.054585811882984 w_k1_PG_s016
                                 + 0.0215599125831563 w_k1_PG_s017
                                 + 0.0254146083846389 w_k1_PG_s100
                                 + 0.0165812193649111 w_k1_PG_s101
                                 + 0.011016897666724 w_k1_PG_s102
                                 + 0.0175081679356242 w_k1_PG_s103
                                 + 0.00697272071648665 w_k1_PG_s104
                                 + 0.0102823699683387 w_k1_PG_s105
                                 + 0.0284516372077971 w_k1_PG_s106
                                 + 0.0128747052661094 w_k1_PG_s107
                                 + 0.0162811413867597 w_k1_PG_s110
                                 + 0.0154444625520591 w_k1_PG_s111
                                 + 0.0111684660505011 w_k1_PG_s112
                                 + 0.0286431122200863 w_k1_PG_s113
                                 + 0.00658826795571356 w_k1_PG_s114
                                 + 0.00806597005808452 w_k1_PG_s115
                                 + 0.00954960136868409 w_k1_PG_s116
                                 + 0.0153036608959872 w_k1_PG_s117 >= 1.19
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

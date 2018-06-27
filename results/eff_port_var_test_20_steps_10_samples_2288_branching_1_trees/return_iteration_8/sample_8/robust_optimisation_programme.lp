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
 w_k1_ADM_s0_return_constraint:  1.18134 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.960862 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.00208 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.18363 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.08789 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.91363 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.10359 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.89947 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.935479 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.44788 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.0879 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.902412 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.36322 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.756716 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.11381 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.970264 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.870555 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.01577 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.90616 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.14422 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.07833 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.11888 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.20208 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.857068 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.00319 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.06243 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.15083 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.771488 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.30224 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.38169 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.50055 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.989569 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.13174 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.898105 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.22644 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.826705 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.756841 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.861061 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.29659 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.93732 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.96434 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.45077 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.41505 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.927453 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.03349 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.17954 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.942146 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.69199 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.47479 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.78643 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.35213 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.723592 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.919083 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.830028 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.76201 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.28494 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.676737 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.974748 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.19025 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.25282 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.01375 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.864843 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.00478 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.817337 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.471 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.698097 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.49256 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.16265 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.47358 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.957974 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.972513 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.919685 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.30957 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.726901 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.732828 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.639543 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.38801 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.93844 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.929379 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.58932 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.40053 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.84906 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.898928 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.18973 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.950506 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.81558 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.38252 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.880839 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.53871 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.671487 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.15402 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.85352 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.532335 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.49285 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.822964 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.14872 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.849115 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.05268 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.213 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.84083 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.828331 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.796662 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.52482 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.7301 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.41965 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.35211 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.80125 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.924247 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.17965 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.748093 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.45122 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.697908 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.668003 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.644266 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.24859 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.946908 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.921268 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.31949 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.21596 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.915519 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.14358 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.12879 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.906746 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.52799 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.18398 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.957799 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.45145 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.771097 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.03281 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.798391 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.725708 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.164 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.782825 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.18089 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.951001 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.13574 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.0246 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.803293 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.924921 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.981694 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.12139 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.741295 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.27778 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.33259 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.58589 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.830757 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.11735 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.877067 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.20274 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.894785 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.700622 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.827134 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.16814 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.97674 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.967633 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.31111 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.32757 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.869382 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.596188 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.10896 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.64339 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.47841 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.70682 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.783719 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.799381 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.837988 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.08797 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.49552 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.886589 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.6291 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.811644 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.586704 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.05185 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.840253 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.22838 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.59467 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.8848 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.606433 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.97888 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.0444 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.17551 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.780303 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.11068 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.50908 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.965326 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.845174 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.49376 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.650861 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.982635 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.511951 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.22133 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.928955 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.885073 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.0554 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.2997 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.976678 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.960779 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.15819 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.03506 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.44312 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.37538 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.975752 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.40025 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.803693 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.11948 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.736664 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.964294 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.45276 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.0564 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.0753 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.09636 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.17779 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.904986 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.820531 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.902195 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.902939 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.19955 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.784219 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.17512 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.07434 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.33063 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.861229 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.04144 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.842407 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.04444 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.860943 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.835182 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.872831 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.18071 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.94296 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.989012 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.37421 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.33547 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.882242 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.831282 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.09523 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.2258 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.53127 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.36512 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.908038 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.08401 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.757671 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.10964 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.992482 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.78731 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.44578 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.849838 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.799035 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.923453 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.941086 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.14501 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.17638 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.860904 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.721965 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.74608 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.868243 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.25094 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.10551 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.31731 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.24558 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.11887 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.835568 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.40194 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.764576 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.852512 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.623301 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.2316 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.963213 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.01096 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.54254 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.60683 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.858503 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.589838 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.36609 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.4473 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.28067 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.73336 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.817598 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.710772 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.83008 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.06175 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.19641 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.81329 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.40901 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.708293 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.575077 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.00964 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.02505 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.98024 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.26623 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.766587 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.54958 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 2.24595 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.851885 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.979887 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.803529 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.42823 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.42641 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.917924 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.854808 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.66602 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.576485 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.959799 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.416918 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.18881 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.930903 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.977842 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.23308 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.22315 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.990461 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.953448 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.12526 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.964298 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.37023 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.20327 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.977346 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.28261 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.772915 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.11657 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.805591 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.787422 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.28623 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.929419 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.11614 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.925182 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.0413 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.07832 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.866125 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.836988 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.954217 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.30746 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.84813 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.19189 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.25296 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.41432 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.803699 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.04381 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.836308 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.02823 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.778979 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.735586 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.728615 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.14129 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.938793 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.00258 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.19284 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.1908 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.952645 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.876828 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.01407 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.06973 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.39293 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.12575 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.837143 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.15286 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.802287 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.10764 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.965974 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.841757 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.26714 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.836422 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.985674 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.00998 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.07624 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.17634 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.01766 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.923186 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.859073 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.33287 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.939972 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.18189 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.15302 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.29787 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.963985 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.03467 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.01434 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.17898 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.851177 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.831342 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.805617 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.348000320204034 z_k1_s0000
                                 - 0.227999984684532 z_k1_s0001
                                 - 0.317000904464087 z_k1_s0002
                                 - 0.364000510007194 z_k1_s0003
                                 - 0.253000145251445 z_k1_s0004
                                 - 0.323002335645526 z_k1_s0005
                                 - 0.120000553120342 z_k1_s0006
                                 - 0.223001693377873 z_k1_s0007
                                 - 0.541997274173676 z_k1_s0010
                                 - 0.349000417852449 z_k1_s0011
                                 - 0.423000391403078 z_k1_s0012
                                 - 0.363000051119578 z_k1_s0013
                                 - 0.624997967965361 z_k1_s0014
                                 - 0.520997824272982 z_k1_s0015
                                 - 0.272999540694329 z_k1_s0016
                                 - 0.24999952178541 z_k1_s0017
                                 - 0.238999834873235 z_k1_s0020
                                 - 0.36199931611377 z_k1_s0021
                                 - 0.356998905966113 z_k1_s0022
                                 - 0.664999096681101 z_k1_s0023
                                 - 0.13899982931378 z_k1_s0024
                                 - 0.215999876992527 z_k1_s0025
                                 - 0.174999889179139 z_k1_s0026
                                 - 0.258000911234703 z_k1_s0027
                                 - 0.22199966138223 z_k1_s0030
                                 - 0.0629999487949289 z_k1_s0031
                                 - 0.132999902755954 z_k1_s0032
                                 - 0.0849999231598296 z_k1_s0033
                                 - 0.101999712391919 z_k1_s0034
                                 - 0.095999959192249 z_k1_s0035
                                 - 0.225000026481756 z_k1_s0036
                                 - 0.0509999538958977 z_k1_s0037
                                 - 0.111000000194194 z_k1_s0040
                                 - 0.21900000038314 z_k1_s0041
                                 - 0.0990000001732004 z_k1_s0042
                                 - 0.0729999601277134 z_k1_s0043
                                 - 0.267999600468865 z_k1_s0044
                                 - 0.08000004013996 z_k1_s0045
                                 - 0.0609999601067194 z_k1_s0046
                                 - 0.288999600505605 z_k1_s0047
                                 - 0.336999844748252 z_k1_s0050
                                 - 0.342999844071365 z_k1_s0051
                                 - 0.306999848132685 z_k1_s0052
                                 - 0.606998759188823 z_k1_s0053
                                 - 0.310999730448146 z_k1_s0054
                                 - 0.227999974278307 z_k1_s0055
                                 - 0.485001117617806 z_k1_s0056
                                 - 0.899001070912627 z_k1_s0057
                                 - 0.217000427834891 z_k1_s0060
                                 - 0.326999252738776 z_k1_s0061
                                 - 0.17399924526503 z_k1_s0062
                                 - 0.082999849372188 z_k1_s0063
                                 - 0.180999510194934 z_k1_s0064
                                 - 0.114999903852918 z_k1_s0065
                                 - 0.196999226035614 z_k1_s0066
                                 - 0.263000389376058 z_k1_s0067
                                 - 0.365999002666187 z_k1_s0070
                                 - 0.504998789484383 z_k1_s0071
                                 - 0.528998313856887 z_k1_s0072
                                 - 0.503997831804461 z_k1_s0073
                                 - 0.344999223340699 z_k1_s0074
                                 - 0.418998538820775 z_k1_s0075
                                 - 0.584997204059397 z_k1_s0076
                                 - 0.656999687167846 z_k1_s0077
                                 - 0.0780002603530255 z_k1_s0100
                                 - 0.129000750085076 z_k1_s0101
                                 - 0.0760002891774431 z_k1_s0102
                                 - 0.112000089839152 z_k1_s0103
                                 - 0.147000384164908 z_k1_s0104
                                 - 0.122000478219107 z_k1_s0105
                                 - 0.296000846005808 z_k1_s0106
                                 - 0.105000350475227 z_k1_s0107
                                 - 0.0940001205370573 z_k1_s0110
                                 - 0.0770000803122408 z_k1_s0111
                                 - 0.0330000653481527 z_k1_s0112
                                 - 0.0940001205370573 z_k1_s0113
                                 - 0.0290000482423013 z_k1_s0114
                                 - 0.0390000693568952 z_k1_s0115
                                 - 0.0240000593350391 z_k1_s0116
                                 - 0.0430000864627466 z_k1_s0117
                                 - 0.0960002059866495 z_k1_s0120
                                 - 0.0230000730489369 z_k1_s0121
                                 - 0.0840000096129433 z_k1_s0122
                                 - 0.0370000898177941 z_k1_s0123
                                 - 0.0240000742467124 z_k1_s0124
                                 - 0.0860001940088944 z_k1_s0125
                                 - 0.0430000970044472 z_k1_s0126
                                 - 0.062000256262482 z_k1_s0127
                                 - 0.173000888884018 z_k1_s0130
                                 - 0.147000715155627 z_k1_s0131
                                 - 0.0760002941590759 z_k1_s0132
                                 - 0.0610002406698025 z_k1_s0133
                                 - 0.090000668136999 z_k1_s0134
                                 - 0.0830003075473582 z_k1_s0135
                                 - 0.19000073539769 z_k1_s0136
                                 - 0.048000153805607 z_k1_s0137
                                 - 0.179001172493824 z_k1_s0140
                                 - 0.107000266595022 z_k1_s0141
                                 - 0.0720003370963857 z_k1_s0142
                                 - 0.0600002177133862 z_k1_s0143
                                 - 0.213001336944917 z_k1_s0144
                                 - 0.0690002835505351 z_k1_s0145
                                 - 0.111000843591637 z_k1_s0146
                                 - 0.137000802053527 z_k1_s0147
                                 - 0.0560001463065148 z_k1_s0150
                                 - 0.0560001463065148 z_k1_s0151
                                 - 0.113000171832515 z_k1_s0152
                                 - 0.0610001643053217 z_k1_s0153
                                 - 0.114000051052 z_k1_s0154
                                 - 0.10899996395304 z_k1_s0155
                                 - 0.125999983688891 z_k1_s0156
                                 - 0.0560001463065148 z_k1_s0157
                                 - 0.0420002362344397 z_k1_s0160
                                 - 0.0210000439170993 z_k1_s0161
                                 - 0.0919999874650545 z_k1_s0162
                                 - 0.0679999584741183 z_k1_s0163
                                 - 0.0470002113575012 z_k1_s0164
                                 - 0.0380001077357494 z_k1_s0165
                                 - 0.020000048892487 z_k1_s0166
                                 - 0.0430000828588109 z_k1_s0167
                                 - 0.151000160931937 z_k1_s0170
                                 - 0.0400000561210662 z_k1_s0171
                                 - 0.0340000622529314 z_k1_s0172
                                 - 0.0470000732172653 z_k1_s0173
                                 - 0.0580002802258895 z_k1_s0174
                                 - 0.0310000410688221 z_k1_s0175
                                 - 0.0390000813930856 z_k1_s0176
                                 - 0.085000276882538 z_k1_s0177
                                 - 0.387999956418992 z_k1_s1000
                                 - 0.613999954018719 z_k1_s1001
                                 - 0.67799999279918 z_k1_s1002
                                 - 0.587000704048978 z_k1_s1003
                                 - 0.354000075160648 z_k1_s1004
                                 - 0.369999861905722 z_k1_s1005
                                 - 0.794000899151419 z_k1_s1006
                                 - 0.674000269062952 z_k1_s1007
                                 - 0.072000037943388 z_k1_s1010
                                 - 0.314000260850375 z_k1_s1011
                                 - 0.172000444892702 z_k1_s1012
                                 - 0.272999346805809 z_k1_s1013
                                 - 0.288999464237724 z_k1_s1014
                                 - 0.198000226969374 z_k1_s1015
                                 - 0.100000079949163 z_k1_s1016
                                 - 0.217000570794867 z_k1_s1017
                                 - 0.264000924192738 z_k1_s1020
                                 - 0.266001037962042 z_k1_s1021
                                 - 0.650000789381576 z_k1_s1022
                                 - 0.717001362990537 z_k1_s1023
                                 - 0.243001062770286 z_k1_s1024
                                 - 0.490003114882179 z_k1_s1025
                                 - 0.598003925803638 z_k1_s1026
                                 - 0.581000482904809 z_k1_s1027
                                 - 0.314001243403324 z_k1_s1030
                                 - 0.363001394660343 z_k1_s1031
                                 - 0.361000017868321 z_k1_s1032
                                 - 0.953004586536589 z_k1_s1033
                                 - 0.293001226550527 z_k1_s1034
                                 - 0.384002418917565 z_k1_s1035
                                 - 0.211000856923984 z_k1_s1036
                                 - 0.47900065624357 z_k1_s1037
                                 - 0.166000154748032 z_k1_s1040
                                 - 0.157000153544833 z_k1_s1041
                                 - 0.113000121151387 z_k1_s1042
                                 - 0.0720001289257157 z_k1_s1043
                                 - 0.130000149935235 z_k1_s1044
                                 - 0.0780000899611412 z_k1_s1045
                                 - 0.233000296260873 z_k1_s1046
                                 - 0.243999767508685 z_k1_s1047
                                 - 0.27800090897537 z_k1_s1050
                                 - 0.293000869288826 z_k1_s1051
                                 - 0.542000040371575 z_k1_s1052
                                 - 0.392001287840113 z_k1_s1053
                                 - 0.341001252653743 z_k1_s1054
                                 - 0.393001597082145 z_k1_s1055
                                 - 0.807001182216006 z_k1_s1056
                                 - 1.20700579459548 z_k1_s1057
                                 - 0.308999452419632 z_k1_s1060
                                 - 0.473000319000331 z_k1_s1061
                                 - 0.1490001471338 z_k1_s1062
                                 - 0.339999569834197 z_k1_s1063
                                 - 0.132000295442308 z_k1_s1064
                                 - 0.29299980085153 z_k1_s1065
                                 - 0.503001250693254 z_k1_s1066
                                 - 0.336999476664904 z_k1_s1067
                                 - 0.354000438374538 z_k1_s1070
                                 - 0.543002094898829 z_k1_s1071
                                 - 0.352000681152478 z_k1_s1072
                                 - 0.175000172564743 z_k1_s1073
                                 - 0.170000200708661 z_k1_s1074
                                 - 0.3830012591014 z_k1_s1075
                                 - 0.626002148630633 z_k1_s1076
                                 - 0.291000850867995 z_k1_s1077
                                 - 0.334999514330274 z_k1_s1100
                                 - 0.826999212817868 z_k1_s1101
                                 - 0.298999614033536 z_k1_s1102
                                 - 0.39899792228158 z_k1_s1103
                                 - 0.373998433644472 z_k1_s1104
                                 - 0.497998443921739 z_k1_s1105
                                 - 0.478000267810504 z_k1_s1106
                                 - 0.326999536486555 z_k1_s1107
                                 - 1.20399478970639 z_k1_s1110
                                 - 0.961995292698055 z_k1_s1111
                                 - 0.891996817639929 z_k1_s1112
                                 - 0.651994097761663 z_k1_s1113
                                 - 0.262999039525773 z_k1_s1114
                                 - 0.278998602653737 z_k1_s1115
                                 - 1.27699152144589 z_k1_s1116
                                 - 0.652995577312968 z_k1_s1117
                                 - 2.78900593218197 z_k1_s1120
                                 - 2.14800323094018 z_k1_s1121
                                 - 2.23599678891017 z_k1_s1122
                                 - 1.22399892467975 z_k1_s1123
                                 - 0.781999870722986 z_k1_s1124
                                 - 2.56600191228767 z_k1_s1125
                                 - 1.96500540939276 z_k1_s1126
                                 - 3.02199895266728 z_k1_s1127
                                 - 0.242999872251618 z_k1_s1130
                                 - 0.42800089540828 z_k1_s1131
                                 - 0.429999925350443 z_k1_s1132
                                 - 0.724997877674315 z_k1_s1133
                                 - 0.819998811878666 z_k1_s1134
                                 - 0.759999297643231 z_k1_s1135
                                 - 0.463999786350383 z_k1_s1136
                                 - 0.217999733987204 z_k1_s1137
                                 - 1.20100025935267 z_k1_s1140
                                 - 0.756003938464037 z_k1_s1141
                                 - 0.487001246546695 z_k1_s1142
                                 - 0.306000333375607 z_k1_s1143
                                 - 1.31800457237288 z_k1_s1144
                                 - 0.495000987040909 z_k1_s1145
                                 - 1.03600147477567 z_k1_s1146
                                 - 1.02000331759007 z_k1_s1147
                                 - 0.303999934450783 z_k1_s1150
                                 - 0.879997473569108 z_k1_s1151
                                 - 0.388000007209511 z_k1_s1152
                                 - 0.950001645032863 z_k1_s1153
                                 - 0.501000187303953 z_k1_s1154
                                 - 0.337999635034226 z_k1_s1155
                                 - 0.541998331834661 z_k1_s1156
                                 - 1.0300000699943 z_k1_s1157
                                 - 0.587998756418206 z_k1_s1160
                                 - 0.674003477430424 z_k1_s1161
                                 - 0.836002208279303 z_k1_s1162
                                 - 1.14000094765129 z_k1_s1163
                                 - 0.660003236801117 z_k1_s1164
                                 - 0.540000687835089 z_k1_s1165
                                 - 0.750003162273421 z_k1_s1166
                                 - 0.487000668667996 z_k1_s1167
                                 - 0.675999838727806 z_k1_s1170
                                 - 0.324999567946349 z_k1_s1171
                                 - 0.286999902078545 z_k1_s1172
                                 - 0.60899972926672 z_k1_s1173
                                 - 0.337999550664203 z_k1_s1174
                                 - 0.281999596748855 z_k1_s1175
                                 - 0.528999268388081 z_k1_s1176
                                 - 0.640999176218777 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.04617 w_k1_ADM_s000
                                 + 1.45082 w_k1_AXP_s000
                                 + 1.64973 w_k1_DIS_s000
                                 + 1.30117 w_k1_FDX_s000
                                 + 1.25719 w_k1_IBM_s000
                                 + 1.25135 w_k1_INTC_s000
                                 + 1.28389 w_k1_KO_s000 + 1.53299 w_k1_MAT_s000
                                 + 1.27326 w_k1_MSFT_s000
                                 + 1.18931 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.661947 w_k1_ADM_s000
                                 + 0.720376 w_k1_AXP_s000
                                 + 0.985793 w_k1_DIS_s000
                                 + 0.842486 w_k1_FDX_s000
                                 + 1.00374 w_k1_IBM_s000
                                 + 1.17344 w_k1_INTC_s000
                                 + 0.930112 w_k1_KO_s000
                                 + 0.903011 w_k1_MAT_s000
                                 + 0.960564 w_k1_MSFT_s000
                                 + 1.05375 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.891784 w_k1_ADM_s000
                                 + 1.06279 w_k1_AXP_s000
                                 + 0.907695 w_k1_DIS_s000
                                 + 1.02533 w_k1_FDX_s000
                                 + 0.915672 w_k1_IBM_s000
                                 + 1.25432 w_k1_INTC_s000
                                 + 0.835815 w_k1_KO_s000
                                 + 1.08078 w_k1_MAT_s000
                                 + 1.04516 w_k1_MSFT_s000
                                 + 0.933508 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.01013 w_k1_ADM_s000
                                 + 1.06903 w_k1_AXP_s000
                                 + 1.19865 w_k1_DIS_s000
                                 + 1.13347 w_k1_FDX_s000
                                 + 0.899487 w_k1_IBM_s000
                                 + 1.05529 w_k1_INTC_s000
                                 + 0.920699 w_k1_KO_s000
                                 + 0.990485 w_k1_MAT_s000
                                 + 1.18639 w_k1_MSFT_s000
                                 + 1.09996 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.793897 w_k1_ADM_s000
                                 + 0.64616 w_k1_AXP_s000
                                 + 0.761731 w_k1_DIS_s000
                                 + 0.765328 w_k1_FDX_s000
                                 + 0.707665 w_k1_IBM_s000
                                 + 0.939544 w_k1_INTC_s000
                                 + 0.750457 w_k1_KO_s000
                                 + 0.807481 w_k1_MAT_s000
                                 + 0.877352 w_k1_MSFT_s000
                                 + 0.832145 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.849032 w_k1_ADM_s000
                                 + 0.77612 w_k1_AXP_s000
                                 + 0.72387 w_k1_DIS_s000
                                 + 0.909355 w_k1_FDX_s000
                                 + 0.913102 w_k1_IBM_s000
                                 + 1.0452 w_k1_INTC_s000
                                 + 0.754064 w_k1_KO_s000 + 0.9018 w_k1_MAT_s000
                                 + 0.99511 w_k1_MSFT_s000
                                 + 0.930146 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.592291 w_k1_ADM_s000
                                 + 0.568633 w_k1_AXP_s000
                                 + 0.556187 w_k1_DIS_s000
                                 + 0.70124 w_k1_FDX_s000
                                 + 0.662302 w_k1_IBM_s000
                                 + 0.846202 w_k1_INTC_s000
                                 + 0.586292 w_k1_KO_s000
                                 + 0.645658 w_k1_MAT_s000
                                 + 0.715679 w_k1_MSFT_s000
                                 + 0.826012 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.904339 w_k1_ADM_s000
                                 + 1.24336 w_k1_AXP_s000
                                 + 1.05232 w_k1_DIS_s000
                                 + 0.896316 w_k1_FDX_s000
                                 + 1.66022 w_k1_IBM_s000
                                 + 0.999116 w_k1_INTC_s000
                                 + 0.930736 w_k1_KO_s000
                                 + 1.97612 w_k1_MAT_s000
                                 + 0.997119 w_k1_MSFT_s000 + 1.007 w_k1_PG_s000
                                 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.09018 w_k1_ADM_s001
                                 + 1.03223 w_k1_AXP_s001
                                 + 0.824554 w_k1_DIS_s001
                                 + 0.928543 w_k1_FDX_s001
                                 + 1.08705 w_k1_IBM_s001
                                 + 0.920841 w_k1_INTC_s001
                                 + 0.95299 w_k1_KO_s001
                                 + 0.983687 w_k1_MAT_s001
                                 + 0.914253 w_k1_MSFT_s001
                                 + 1.03068 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.26059 w_k1_ADM_s001
                                 + 0.955798 w_k1_AXP_s001
                                 + 1.24936 w_k1_DIS_s001
                                 + 1.06149 w_k1_FDX_s001
                                 + 0.863262 w_k1_IBM_s001
                                 + 0.973767 w_k1_INTC_s001
                                 + 1.02996 w_k1_KO_s001
                                 + 0.516179 w_k1_MAT_s001
                                 + 0.877657 w_k1_MSFT_s001
                                 + 1.04323 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.973669 w_k1_ADM_s001
                                 + 0.750743 w_k1_AXP_s001
                                 + 0.715242 w_k1_DIS_s001
                                 + 0.790602 w_k1_FDX_s001
                                 + 0.735887 w_k1_IBM_s001
                                 + 0.804284 w_k1_INTC_s001
                                 + 0.769752 w_k1_KO_s001
                                 + 0.484423 w_k1_MAT_s001
                                 + 0.789929 w_k1_MSFT_s001
                                 + 0.756665 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.13164 w_k1_ADM_s001
                                 + 0.878474 w_k1_AXP_s001
                                 + 1.08261 w_k1_DIS_s001
                                 + 0.891921 w_k1_FDX_s001
                                 + 1.46931 w_k1_IBM_s001
                                 + 0.789605 w_k1_INTC_s001
                                 + 1.18761 w_k1_KO_s001 + 1.10341 w_k1_MAT_s001
                                 + 1.00246 w_k1_MSFT_s001
                                 + 1.10599 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.40451 w_k1_ADM_s001
                                 + 1.34148 w_k1_AXP_s001
                                 + 1.29234 w_k1_DIS_s001
                                 + 1.26691 w_k1_FDX_s001
                                 + 1.14362 w_k1_IBM_s001
                                 + 1.27101 w_k1_INTC_s001 + 1.0636 w_k1_KO_s001
                                 + 0.873849 w_k1_MAT_s001
                                 + 1.27862 w_k1_MSFT_s001 + 1.2409 w_k1_PG_s001
                                 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.15891 w_k1_ADM_s001
                                 + 0.953368 w_k1_AXP_s001
                                 + 0.881428 w_k1_DIS_s001
                                 + 1.06394 w_k1_FDX_s001
                                 + 0.802281 w_k1_IBM_s001
                                 + 1.03265 w_k1_INTC_s001
                                 + 0.94194 w_k1_KO_s001
                                 + 0.663847 w_k1_MAT_s001
                                 + 0.997986 w_k1_MSFT_s001
                                 + 1.0669 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.09492 w_k1_ADM_s001
                                 + 0.65806 w_k1_AXP_s001
                                 + 1.15131 w_k1_DIS_s001 + 1.014 w_k1_FDX_s001
                                 + 0.983286 w_k1_IBM_s001
                                 + 1.04646 w_k1_INTC_s001
                                 + 1.08587 w_k1_KO_s001
                                 + 0.526706 w_k1_MAT_s001
                                 + 1.09551 w_k1_MSFT_s001
                                 + 1.10265 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.63516 w_k1_ADM_s001
                                 + 1.17682 w_k1_AXP_s001
                                 + 1.82535 w_k1_DIS_s001
                                 + 1.34152 w_k1_FDX_s001
                                 + 1.30587 w_k1_IBM_s001
                                 + 1.21272 w_k1_INTC_s001
                                 + 1.50678 w_k1_KO_s001
                                 + 0.930645 w_k1_MAT_s001
                                 + 1.38455 w_k1_MSFT_s001
                                 + 1.23192 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.03868 w_k1_ADM_s002
                                 + 1.54776 w_k1_AXP_s002 + 1.6611 w_k1_DIS_s002
                                 + 1.41389 w_k1_FDX_s002
                                 + 1.15702 w_k1_IBM_s002
                                 + 1.23917 w_k1_INTC_s002
                                 + 1.40249 w_k1_KO_s002 + 1.59341 w_k1_MAT_s002
                                 + 1.30173 w_k1_MSFT_s002
                                 + 1.13807 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.07094 w_k1_ADM_s002
                                 + 1.1441 w_k1_AXP_s002 + 1.0081 w_k1_DIS_s002
                                 + 1.03882 w_k1_FDX_s002
                                 + 0.773071 w_k1_IBM_s002
                                 + 0.974447 w_k1_INTC_s002
                                 + 0.853898 w_k1_KO_s002
                                 + 0.824205 w_k1_MAT_s002
                                 + 0.854855 w_k1_MSFT_s002
                                 + 0.937028 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.08732 w_k1_ADM_s002
                                 + 1.26003 w_k1_AXP_s002
                                 + 1.28821 w_k1_DIS_s002
                                 + 1.18519 w_k1_FDX_s002
                                 + 0.939133 w_k1_IBM_s002
                                 + 1.02213 w_k1_INTC_s002
                                 + 0.953568 w_k1_KO_s002
                                 + 0.818203 w_k1_MAT_s002
                                 + 1.08906 w_k1_MSFT_s002
                                 + 0.986498 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.967452 w_k1_ADM_s002
                                 + 1.2595 w_k1_AXP_s002 + 1.11654 w_k1_DIS_s002
                                 + 0.930013 w_k1_FDX_s002
                                 + 1.37862 w_k1_IBM_s002
                                 + 1.07268 w_k1_INTC_s002 + 1.1512 w_k1_KO_s002
                                 + 1.44134 w_k1_MAT_s002
                                 + 1.00245 w_k1_MSFT_s002 + 1.05 w_k1_PG_s002
                                 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.905544 w_k1_ADM_s002
                                 + 0.977205 w_k1_AXP_s002
                                 + 0.836834 w_k1_DIS_s002
                                 + 1.00841 w_k1_FDX_s002
                                 + 0.587426 w_k1_IBM_s002
                                 + 0.826091 w_k1_INTC_s002
                                 + 0.663828 w_k1_KO_s002
                                 + 0.531096 w_k1_MAT_s002
                                 + 0.961206 w_k1_MSFT_s002
                                 + 0.76543 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.794066 w_k1_ADM_s002
                                 + 0.733879 w_k1_AXP_s002
                                 + 0.888892 w_k1_DIS_s002
                                 + 0.93718 w_k1_FDX_s002
                                 + 0.871103 w_k1_IBM_s002
                                 + 1.03471 w_k1_INTC_s002
                                 + 0.926425 w_k1_KO_s002
                                 + 0.864764 w_k1_MAT_s002
                                 + 0.982658 w_k1_MSFT_s002
                                 + 0.926953 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.661174 w_k1_ADM_s002
                                 + 0.804788 w_k1_AXP_s002
                                 + 0.687589 w_k1_DIS_s002
                                 + 0.716816 w_k1_FDX_s002
                                 + 0.935912 w_k1_IBM_s002
                                 + 0.814957 w_k1_INTC_s002
                                 + 0.781873 w_k1_KO_s002
                                 + 1.04793 w_k1_MAT_s002
                                 + 0.752293 w_k1_MSFT_s002
                                 + 0.849418 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.890612 w_k1_ADM_s002
                                 + 0.764912 w_k1_AXP_s002
                                 + 0.661255 w_k1_DIS_s002
                                 + 0.774027 w_k1_FDX_s002
                                 + 0.691602 w_k1_IBM_s002
                                 + 0.759568 w_k1_INTC_s002
                                 + 0.756068 w_k1_KO_s002
                                 + 0.959556 w_k1_MAT_s002
                                 + 0.801952 w_k1_MSFT_s002
                                 + 0.846805 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.834305 w_k1_ADM_s003
                                 + 0.687894 w_k1_AXP_s003
                                 + 0.831504 w_k1_DIS_s003
                                 + 0.829028 w_k1_FDX_s003
                                 + 0.863276 w_k1_IBM_s003
                                 + 0.967238 w_k1_INTC_s003
                                 + 0.896719 w_k1_KO_s003
                                 + 0.875883 w_k1_MAT_s003
                                 + 1.03391 w_k1_MSFT_s003
                                 + 0.865883 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.37256 w_k1_ADM_s003
                                 + 1.09177 w_k1_AXP_s003
                                 + 2.01936 w_k1_DIS_s003
                                 + 1.31597 w_k1_FDX_s003
                                 + 1.12111 w_k1_IBM_s003
                                 + 1.03039 w_k1_INTC_s003 + 1.4179 w_k1_KO_s003
                                 + 1.34875 w_k1_MAT_s003
                                 + 1.43137 w_k1_MSFT_s003
                                 + 1.06289 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.01446 w_k1_ADM_s003
                                 + 0.890619 w_k1_AXP_s003
                                 + 0.923328 w_k1_DIS_s003
                                 + 1.03307 w_k1_FDX_s003
                                 + 0.675539 w_k1_IBM_s003
                                 + 0.995525 w_k1_INTC_s003
                                 + 0.717591 w_k1_KO_s003
                                 + 0.681291 w_k1_MAT_s003
                                 + 0.898155 w_k1_MSFT_s003
                                 + 0.865208 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.09181 w_k1_ADM_s003
                                 + 1.02424 w_k1_AXP_s003
                                 + 1.35771 w_k1_DIS_s003
                                 + 1.35819 w_k1_FDX_s003
                                 + 0.692219 w_k1_IBM_s003
                                 + 1.2342 w_k1_INTC_s003 + 1.02783 w_k1_KO_s003
                                 + 0.832303 w_k1_MAT_s003
                                 + 1.2793 w_k1_MSFT_s003
                                 + 0.928693 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.997634 w_k1_ADM_s003
                                 + 1.17471 w_k1_AXP_s003
                                 + 1.18363 w_k1_DIS_s003
                                 + 1.08675 w_k1_FDX_s003
                                 + 1.20265 w_k1_IBM_s003
                                 + 0.825071 w_k1_INTC_s003
                                 + 1.08099 w_k1_KO_s003 + 1.23095 w_k1_MAT_s003
                                 + 1.13401 w_k1_MSFT_s003 + 0.9432 w_k1_PG_s003
                                 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.01443 w_k1_ADM_s003
                                 + 1.20652 w_k1_AXP_s003
                                 + 1.43451 w_k1_DIS_s003
                                 + 1.09634 w_k1_FDX_s003
                                 + 1.73456 w_k1_IBM_s003
                                 + 0.996677 w_k1_INTC_s003
                                 + 1.29728 w_k1_KO_s003 + 2.51902 w_k1_MAT_s003
                                 + 1.13987 w_k1_MSFT_s003
                                 + 1.22744 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.32138 w_k1_ADM_s003
                                 + 1.03857 w_k1_AXP_s003
                                 + 1.02428 w_k1_DIS_s003
                                 + 0.978615 w_k1_FDX_s003
                                 + 1.07154 w_k1_IBM_s003
                                 + 0.778448 w_k1_INTC_s003
                                 + 1.01447 w_k1_KO_s003 + 1.55047 w_k1_MAT_s003
                                 + 1.00037 w_k1_MSFT_s003
                                 + 0.915862 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.655876 w_k1_ADM_s003
                                 + 0.710305 w_k1_AXP_s003
                                 + 0.613626 w_k1_DIS_s003
                                 + 0.812739 w_k1_FDX_s003
                                 + 0.530417 w_k1_IBM_s003
                                 + 0.800478 w_k1_INTC_s003
                                 + 0.625281 w_k1_KO_s003
                                 + 0.841847 w_k1_MAT_s003
                                 + 0.879188 w_k1_MSFT_s003
                                 + 0.718337 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.939962 w_k1_ADM_s004
                                 + 1.08941 w_k1_AXP_s004
                                 + 1.16203 w_k1_DIS_s004
                                 + 0.945234 w_k1_FDX_s004
                                 + 1.63608 w_k1_IBM_s004
                                 + 1.02521 w_k1_INTC_s004
                                 + 1.28621 w_k1_KO_s004 + 1.14539 w_k1_MAT_s004
                                 + 0.932818 w_k1_MSFT_s004
                                 + 1.27195 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.869688 w_k1_ADM_s004
                                 + 0.696152 w_k1_AXP_s004
                                 + 0.824701 w_k1_DIS_s004
                                 + 0.861156 w_k1_FDX_s004
                                 + 0.916839 w_k1_IBM_s004
                                 + 0.674053 w_k1_INTC_s004
                                 + 0.933798 w_k1_KO_s004
                                 + 1.09536 w_k1_MAT_s004
                                 + 0.905663 w_k1_MSFT_s004
                                 + 1.04655 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.781829 w_k1_ADM_s004
                                 + 0.898577 w_k1_AXP_s004
                                 + 0.634129 w_k1_DIS_s004
                                 + 0.763801 w_k1_FDX_s004
                                 + 0.917299 w_k1_IBM_s004
                                 + 0.598792 w_k1_INTC_s004
                                 + 0.761272 w_k1_KO_s004
                                 + 1.04551 w_k1_MAT_s004
                                 + 0.689636 w_k1_MSFT_s004
                                 + 0.855728 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.09268 w_k1_ADM_s004
                                 + 0.690857 w_k1_AXP_s004
                                 + 0.678259 w_k1_DIS_s004
                                 + 0.83976 w_k1_FDX_s004
                                 + 0.563817 w_k1_IBM_s004
                                 + 0.760624 w_k1_INTC_s004
                                 + 0.824599 w_k1_KO_s004
                                 + 0.546881 w_k1_MAT_s004
                                 + 0.907925 w_k1_MSFT_s004
                                 + 0.901441 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.860747 w_k1_ADM_s004
                                 + 0.972458 w_k1_AXP_s004
                                 + 0.99951 w_k1_DIS_s004
                                 + 1.10231 w_k1_FDX_s004
                                 + 0.834123 w_k1_IBM_s004
                                 + 1.07195 w_k1_INTC_s004
                                 + 0.935533 w_k1_KO_s004
                                 + 0.856758 w_k1_MAT_s004
                                 + 1.02091 w_k1_MSFT_s004
                                 + 0.990814 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.40976 w_k1_ADM_s004
                                 + 1.22873 w_k1_AXP_s004
                                 + 1.71536 w_k1_DIS_s004
                                 + 1.34282 w_k1_FDX_s004
                                 + 1.06738 w_k1_IBM_s004
                                 + 0.943242 w_k1_INTC_s004
                                 + 1.19623 w_k1_KO_s004 + 1.33962 w_k1_MAT_s004
                                 + 1.17846 w_k1_MSFT_s004 + 1.2454 w_k1_PG_s004
                                 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.15433 w_k1_ADM_s004
                                 + 1.09163 w_k1_AXP_s004
                                 + 1.56594 w_k1_DIS_s004
                                 + 1.18563 w_k1_FDX_s004
                                 + 1.26777 w_k1_IBM_s004
                                 + 1.00486 w_k1_INTC_s004
                                 + 1.43228 w_k1_KO_s004 + 1.33446 w_k1_MAT_s004
                                 + 1.05197 w_k1_MSFT_s004
                                 + 1.25207 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.25648 w_k1_ADM_s004
                                 + 1.15085 w_k1_AXP_s004
                                 + 1.04576 w_k1_DIS_s004
                                 + 1.08891 w_k1_FDX_s004
                                 + 1.18237 w_k1_IBM_s004
                                 + 0.987192 w_k1_INTC_s004
                                 + 1.11704 w_k1_KO_s004 + 1.35005 w_k1_MAT_s004
                                 + 1.08334 w_k1_MSFT_s004
                                 + 1.05301 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.807275 w_k1_ADM_s005
                                 + 0.683057 w_k1_AXP_s005
                                 + 0.774141 w_k1_DIS_s005
                                 + 0.762849 w_k1_FDX_s005
                                 + 0.977876 w_k1_IBM_s005
                                 + 0.961392 w_k1_INTC_s005
                                 + 0.787295 w_k1_KO_s005
                                 + 0.879072 w_k1_MAT_s005
                                 + 0.85666 w_k1_MSFT_s005
                                 + 0.946533 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.00799 w_k1_ADM_s005
                                 + 0.885552 w_k1_AXP_s005
                                 + 1.05103 w_k1_DIS_s005
                                 + 0.898241 w_k1_FDX_s005
                                 + 1.61182 w_k1_IBM_s005
                                 + 0.87439 w_k1_INTC_s005
                                 + 1.23618 w_k1_KO_s005 + 1.27331 w_k1_MAT_s005
                                 + 0.872471 w_k1_MSFT_s005
                                 + 1.15326 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.27333 w_k1_ADM_s005
                                 + 1.51979 w_k1_AXP_s005
                                 + 1.57403 w_k1_DIS_s005 + 1.3422 w_k1_FDX_s005
                                 + 1.16293 w_k1_IBM_s005
                                 + 1.53422 w_k1_INTC_s005
                                 + 1.34795 w_k1_KO_s005 + 1.00743 w_k1_MAT_s005
                                 + 1.29638 w_k1_MSFT_s005 + 1.3631 w_k1_PG_s005
                                 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.948982 w_k1_ADM_s005
                                 + 1.06217 w_k1_AXP_s005
                                 + 1.43392 w_k1_DIS_s005
                                 + 1.16772 w_k1_FDX_s005
                                 + 1.15994 w_k1_IBM_s005
                                 + 1.24587 w_k1_INTC_s005
                                 + 1.14838 w_k1_KO_s005 + 1.04048 w_k1_MAT_s005
                                 + 1.21173 w_k1_MSFT_s005
                                 + 1.22781 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.84247 w_k1_ADM_s005
                                 + 1.52069 w_k1_AXP_s005
                                 + 1.06786 w_k1_DIS_s005
                                 + 1.14116 w_k1_FDX_s005
                                 + 1.29936 w_k1_IBM_s005
                                 + 1.12307 w_k1_INTC_s005
                                 + 1.08029 w_k1_KO_s005 + 1.52878 w_k1_MAT_s005
                                 + 0.925334 w_k1_MSFT_s005
                                 + 1.18051 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.00586 w_k1_ADM_s005
                                 + 0.917041 w_k1_AXP_s005
                                 + 0.732327 w_k1_DIS_s005
                                 + 0.854132 w_k1_FDX_s005
                                 + 0.843251 w_k1_IBM_s005
                                 + 0.76199 w_k1_INTC_s005
                                 + 0.740226 w_k1_KO_s005
                                 + 0.70964 w_k1_MAT_s005
                                 + 0.774367 w_k1_MSFT_s005
                                 + 0.89843 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.968499 w_k1_ADM_s005
                                 + 0.780222 w_k1_AXP_s005
                                 + 0.80888 w_k1_DIS_s005
                                 + 0.904582 w_k1_FDX_s005
                                 + 0.661306 w_k1_IBM_s005
                                 + 0.979976 w_k1_INTC_s005
                                 + 0.831597 w_k1_KO_s005
                                 + 0.755956 w_k1_MAT_s005
                                 + 0.920702 w_k1_MSFT_s005
                                 + 1.00052 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.01949 w_k1_ADM_s005
                                 + 0.86264 w_k1_AXP_s005
                                 + 1.10132 w_k1_DIS_s005
                                 + 0.98157 w_k1_FDX_s005
                                 + 0.925221 w_k1_IBM_s005
                                 + 1.06726 w_k1_INTC_s005
                                 + 0.943036 w_k1_KO_s005
                                 + 0.676889 w_k1_MAT_s005
                                 + 1.06966 w_k1_MSFT_s005
                                 + 1.03033 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.79637 w_k1_ADM_s006
                                 + 0.835964 w_k1_AXP_s006
                                 + 0.787414 w_k1_DIS_s006
                                 + 0.827153 w_k1_FDX_s006
                                 + 1.25023 w_k1_IBM_s006
                                 + 0.838372 w_k1_INTC_s006
                                 + 1.1387 w_k1_KO_s006 + 1.97613 w_k1_MAT_s006
                                 + 0.83569 w_k1_MSFT_s006
                                 + 0.859537 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.981643 w_k1_ADM_s006
                                 + 1.11102 w_k1_AXP_s006
                                 + 0.977148 w_k1_DIS_s006
                                 + 1.04977 w_k1_FDX_s006
                                 + 1.13317 w_k1_IBM_s006
                                 + 0.909527 w_k1_INTC_s006
                                 + 1.00985 w_k1_KO_s006 + 1.82765 w_k1_MAT_s006
                                 + 1.0334 w_k1_MSFT_s006
                                 + 0.896067 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.691306 w_k1_ADM_s006
                                 + 0.713367 w_k1_AXP_s006
                                 + 0.633957 w_k1_DIS_s006
                                 + 0.731941 w_k1_FDX_s006
                                 + 0.738656 w_k1_IBM_s006
                                 + 0.738294 w_k1_INTC_s006
                                 + 0.678389 w_k1_KO_s006
                                 + 0.673535 w_k1_MAT_s006
                                 + 0.725536 w_k1_MSFT_s006
                                 + 0.802819 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.896365 w_k1_ADM_s006
                                 + 1.62142 w_k1_AXP_s006
                                 + 1.64798 w_k1_DIS_s006
                                 + 1.40122 w_k1_FDX_s006
                                 + 1.47623 w_k1_IBM_s006
                                 + 1.07669 w_k1_INTC_s006
                                 + 1.42233 w_k1_KO_s006 + 2.63163 w_k1_MAT_s006
                                 + 1.31167 w_k1_MSFT_s006
                                 + 1.32501 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.822993 w_k1_ADM_s006
                                 + 0.923269 w_k1_AXP_s006
                                 + 1.25391 w_k1_DIS_s006
                                 + 1.05451 w_k1_FDX_s006
                                 + 1.06041 w_k1_IBM_s006
                                 + 1.03814 w_k1_INTC_s006
                                 + 1.21774 w_k1_KO_s006 + 1.23141 w_k1_MAT_s006
                                 + 1.10692 w_k1_MSFT_s006
                                 + 1.00367 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.838068 w_k1_ADM_s006
                                 + 1.21412 w_k1_AXP_s006 + 1.5038 w_k1_DIS_s006
                                 + 1.32427 w_k1_FDX_s006
                                 + 1.17411 w_k1_IBM_s006
                                 + 1.35137 w_k1_INTC_s006
                                 + 1.23316 w_k1_KO_s006 + 1.55814 w_k1_MAT_s006
                                 + 1.24716 w_k1_MSFT_s006
                                 + 0.988762 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.934287 w_k1_ADM_s006
                                 + 1.18894 w_k1_AXP_s006
                                 + 1.07304 w_k1_DIS_s006
                                 + 1.13914 w_k1_FDX_s006
                                 + 0.93669 w_k1_IBM_s006
                                 + 0.984646 w_k1_INTC_s006
                                 + 1.03983 w_k1_KO_s006 + 1.45326 w_k1_MAT_s006
                                 + 1.04982 w_k1_MSFT_s006
                                 + 1.08657 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.708854 w_k1_ADM_s006
                                 + 0.887458 w_k1_AXP_s006
                                 + 0.763325 w_k1_DIS_s006
                                 + 0.90697 w_k1_FDX_s006
                                 + 0.752515 w_k1_IBM_s006
                                 + 0.781165 w_k1_INTC_s006
                                 + 0.754658 w_k1_KO_s006
                                 + 0.902022 w_k1_MAT_s006
                                 + 0.762074 w_k1_MSFT_s006
                                 + 0.762536 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.09021 w_k1_ADM_s007
                                 + 0.945367 w_k1_AXP_s007
                                 + 1.0053 w_k1_DIS_s007
                                 + 0.958493 w_k1_FDX_s007
                                 + 1.11235 w_k1_IBM_s007
                                 + 0.889744 w_k1_INTC_s007
                                 + 1.15925 w_k1_KO_s007
                                 + 0.952423 w_k1_MAT_s007
                                 + 1.01197 w_k1_MSFT_s007
                                 + 1.13174 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.911589 w_k1_ADM_s007
                                 + 1.08837 w_k1_AXP_s007
                                 + 0.863511 w_k1_DIS_s007
                                 + 0.816197 w_k1_FDX_s007
                                 + 1.52094 w_k1_IBM_s007
                                 + 0.979937 w_k1_INTC_s007
                                 + 1.14487 w_k1_KO_s007 + 1.50206 w_k1_MAT_s007
                                 + 0.86394 w_k1_MSFT_s007
                                 + 0.962501 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.11965 w_k1_ADM_s007
                                 + 1.00041 w_k1_AXP_s007
                                 + 1.01358 w_k1_DIS_s007 + 1.1336 w_k1_FDX_s007
                                 + 0.741053 w_k1_IBM_s007
                                 + 1.17857 w_k1_INTC_s007
                                 + 0.959294 w_k1_KO_s007
                                 + 0.617086 w_k1_MAT_s007
                                 + 1.11648 w_k1_MSFT_s007
                                 + 1.05353 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.84896 w_k1_ADM_s007
                                 + 0.658544 w_k1_AXP_s007
                                 + 0.619803 w_k1_DIS_s007
                                 + 0.823797 w_k1_FDX_s007
                                 + 0.522568 w_k1_IBM_s007
                                 + 0.889174 w_k1_INTC_s007
                                 + 0.643612 w_k1_KO_s007
                                 + 0.40286 w_k1_MAT_s007
                                 + 0.825857 w_k1_MSFT_s007
                                 + 0.832523 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.43722 w_k1_ADM_s007
                                 + 1.57035 w_k1_AXP_s007
                                 + 1.66176 w_k1_DIS_s007
                                 + 1.45709 w_k1_FDX_s007
                                 + 1.45994 w_k1_IBM_s007
                                 + 1.37524 w_k1_INTC_s007
                                 + 1.43856 w_k1_KO_s007 + 1.26905 w_k1_MAT_s007
                                 + 1.31807 w_k1_MSFT_s007
                                 + 1.19748 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.954715 w_k1_ADM_s007
                                 + 0.834664 w_k1_AXP_s007
                                 + 0.783252 w_k1_DIS_s007
                                 + 0.758835 w_k1_FDX_s007
                                 + 0.974063 w_k1_IBM_s007
                                 + 0.851996 w_k1_INTC_s007
                                 + 0.921303 w_k1_KO_s007
                                 + 1.02971 w_k1_MAT_s007
                                 + 0.869818 w_k1_MSFT_s007
                                 + 0.912694 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.13253 w_k1_ADM_s007
                                 + 1.26901 w_k1_AXP_s007
                                 + 1.22087 w_k1_DIS_s007 + 1.1478 w_k1_FDX_s007
                                 + 1.31984 w_k1_IBM_s007
                                 + 1.13429 w_k1_INTC_s007
                                 + 1.34179 w_k1_KO_s007 + 1.44369 w_k1_MAT_s007
                                 + 1.08849 w_k1_MSFT_s007
                                 + 1.23035 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.08792 w_k1_ADM_s007
                                 + 0.877586 w_k1_AXP_s007
                                 + 0.764263 w_k1_DIS_s007
                                 + 0.859636 w_k1_FDX_s007
                                 + 0.775371 w_k1_IBM_s007
                                 + 1.02282 w_k1_INTC_s007
                                 + 0.865882 w_k1_KO_s007
                                 + 0.678592 w_k1_MAT_s007
                                 + 0.8726 w_k1_MSFT_s007
                                 + 0.844167 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.01112 w_k1_ADM_s010
                                 + 1.23343 w_k1_AXP_s010
                                 + 1.23352 w_k1_DIS_s010
                                 + 0.985774 w_k1_FDX_s010
                                 + 1.61557 w_k1_IBM_s010
                                 + 1.03181 w_k1_INTC_s010
                                 + 1.37807 w_k1_KO_s010 + 1.54932 w_k1_MAT_s010
                                 + 1.02823 w_k1_MSFT_s010 + 1.1411 w_k1_PG_s010
                                 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.45255 w_k1_ADM_s010
                                 + 1.50341 w_k1_AXP_s010
                                 + 1.65329 w_k1_DIS_s010
                                 + 1.29153 w_k1_FDX_s010
                                 + 1.17857 w_k1_IBM_s010
                                 + 1.07927 w_k1_INTC_s010
                                 + 1.27293 w_k1_KO_s010 + 1.17931 w_k1_MAT_s010
                                 + 1.14091 w_k1_MSFT_s010
                                 + 1.16002 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.3188 w_k1_ADM_s010
                                 + 1.26824 w_k1_AXP_s010
                                 + 1.19001 w_k1_DIS_s010
                                 + 1.18602 w_k1_FDX_s010
                                 + 0.787983 w_k1_IBM_s010
                                 + 1.02056 w_k1_INTC_s010
                                 + 1.00121 w_k1_KO_s010
                                 + 0.773263 w_k1_MAT_s010
                                 + 1.13692 w_k1_MSFT_s010
                                 + 1.00282 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.12691 w_k1_ADM_s010
                                 + 0.909885 w_k1_AXP_s010
                                 + 1.05601 w_k1_DIS_s010
                                 + 1.11584 w_k1_FDX_s010
                                 + 0.692525 w_k1_IBM_s010
                                 + 0.849308 w_k1_INTC_s010
                                 + 0.794041 w_k1_KO_s010
                                 + 0.652395 w_k1_MAT_s010
                                 + 0.95326 w_k1_MSFT_s010
                                 + 0.811326 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.18892 w_k1_ADM_s010
                                 + 1.04469 w_k1_AXP_s010
                                 + 0.955563 w_k1_DIS_s010
                                 + 0.975758 w_k1_FDX_s010
                                 + 0.882906 w_k1_IBM_s010
                                 + 0.930372 w_k1_INTC_s010
                                 + 0.966112 w_k1_KO_s010
                                 + 0.739195 w_k1_MAT_s010
                                 + 1.03171 w_k1_MSFT_s010
                                 + 0.992452 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.15039 w_k1_ADM_s010
                                 + 1.08014 w_k1_AXP_s010
                                 + 0.905584 w_k1_DIS_s010
                                 + 0.92515 w_k1_FDX_s010
                                 + 1.16283 w_k1_IBM_s010
                                 + 0.773866 w_k1_INTC_s010
                                 + 1.02177 w_k1_KO_s010 + 1.08716 w_k1_MAT_s010
                                 + 0.821973 w_k1_MSFT_s010
                                 + 0.955459 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.844267 w_k1_ADM_s010
                                 + 0.795173 w_k1_AXP_s010
                                 + 0.798403 w_k1_DIS_s010
                                 + 0.828376 w_k1_FDX_s010
                                 + 0.806184 w_k1_IBM_s010
                                 + 0.799922 w_k1_INTC_s010
                                 + 0.764404 w_k1_KO_s010
                                 + 0.798537 w_k1_MAT_s010
                                 + 0.837032 w_k1_MSFT_s010
                                 + 0.883699 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.3721 w_k1_ADM_s010
                                 + 1.54094 w_k1_AXP_s010
                                 + 1.21865 w_k1_DIS_s010
                                 + 1.16726 w_k1_FDX_s010
                                 + 1.16682 w_k1_IBM_s010
                                 + 1.09888 w_k1_INTC_s010
                                 + 1.17063 w_k1_KO_s010 + 1.09738 w_k1_MAT_s010
                                 + 1.16062 w_k1_MSFT_s010
                                 + 1.07881 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.668679 w_k1_ADM_s011
                                 + 1.05717 w_k1_AXP_s011
                                 + 1.05194 w_k1_DIS_s011
                                 + 0.879866 w_k1_FDX_s011
                                 + 1.10479 w_k1_IBM_s011
                                 + 1.45628 w_k1_INTC_s011
                                 + 1.09755 w_k1_KO_s011 + 1.26606 w_k1_MAT_s011
                                 + 1.10404 w_k1_MSFT_s011
                                 + 1.01344 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.22943 w_k1_ADM_s011
                                 + 1.10548 w_k1_AXP_s011
                                 + 1.27053 w_k1_DIS_s011
                                 + 1.03568 w_k1_FDX_s011
                                 + 1.27567 w_k1_IBM_s011
                                 + 1.18744 w_k1_INTC_s011 + 1.4877 w_k1_KO_s011
                                 + 1.65714 w_k1_MAT_s011
                                 + 1.15721 w_k1_MSFT_s011
                                 + 1.00495 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.93697 w_k1_ADM_s011
                                 + 1.37464 w_k1_AXP_s011
                                 + 1.21029 w_k1_DIS_s011
                                 + 1.11402 w_k1_FDX_s011 + 1.7686 w_k1_IBM_s011
                                 + 2.12498 w_k1_INTC_s011
                                 + 1.40877 w_k1_KO_s011 + 1.61563 w_k1_MAT_s011
                                 + 1.39669 w_k1_MSFT_s011
                                 + 1.24021 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.738223 w_k1_ADM_s011
                                 + 0.868579 w_k1_AXP_s011
                                 + 0.648556 w_k1_DIS_s011
                                 + 0.827158 w_k1_FDX_s011
                                 + 0.876604 w_k1_IBM_s011
                                 + 1.1363 w_k1_INTC_s011
                                 + 0.789064 w_k1_KO_s011
                                 + 0.988474 w_k1_MAT_s011
                                 + 0.890307 w_k1_MSFT_s011
                                 + 0.725864 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.76152 w_k1_ADM_s011
                                 + 0.851092 w_k1_AXP_s011
                                 + 0.815715 w_k1_DIS_s011
                                 + 0.860196 w_k1_FDX_s011
                                 + 0.636702 w_k1_IBM_s011
                                 + 1.27107 w_k1_INTC_s011
                                 + 0.849415 w_k1_KO_s011
                                 + 0.470341 w_k1_MAT_s011
                                 + 0.876605 w_k1_MSFT_s011
                                 + 0.734873 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.16816 w_k1_ADM_s011
                                 + 1.27852 w_k1_AXP_s011
                                 + 1.47151 w_k1_DIS_s011
                                 + 1.34867 w_k1_FDX_s011
                                 + 0.986631 w_k1_IBM_s011
                                 + 1.6212 w_k1_INTC_s011 + 1.55848 w_k1_KO_s011
                                 + 1.28063 w_k1_MAT_s011
                                 + 1.5648 w_k1_MSFT_s011 + 1.32217 w_k1_PG_s011
                                 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.912355 w_k1_ADM_s011
                                 + 1.12963 w_k1_AXP_s011
                                 + 0.899882 w_k1_DIS_s011
                                 + 1.04509 w_k1_FDX_s011
                                 + 0.850771 w_k1_IBM_s011
                                 + 1.25719 w_k1_INTC_s011
                                 + 0.976711 w_k1_KO_s011
                                 + 0.980444 w_k1_MAT_s011
                                 + 1.03857 w_k1_MSFT_s011
                                 + 0.865711 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.1158 w_k1_ADM_s011
                                 + 0.785894 w_k1_AXP_s011
                                 + 1.13508 w_k1_DIS_s011
                                 + 1.02147 w_k1_FDX_s011
                                 + 0.691822 w_k1_IBM_s011
                                 + 1.33867 w_k1_INTC_s011
                                 + 1.06068 w_k1_KO_s011 + 0.61665 w_k1_MAT_s011
                                 + 1.17202 w_k1_MSFT_s011
                                 + 1.09732 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.18545 w_k1_ADM_s012
                                 + 1.13989 w_k1_AXP_s012
                                 + 1.11803 w_k1_DIS_s012
                                 + 1.14517 w_k1_FDX_s012
                                 + 0.613003 w_k1_IBM_s012
                                 + 0.859336 w_k1_INTC_s012
                                 + 0.762131 w_k1_KO_s012
                                 + 0.689487 w_k1_MAT_s012
                                 + 0.89831 w_k1_MSFT_s012
                                 + 0.983077 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.10821 w_k1_ADM_s012
                                 + 1.55822 w_k1_AXP_s012
                                 + 1.83903 w_k1_DIS_s012
                                 + 1.45793 w_k1_FDX_s012
                                 + 1.58813 w_k1_IBM_s012
                                 + 1.42469 w_k1_INTC_s012
                                 + 1.49529 w_k1_KO_s012 + 1.95063 w_k1_MAT_s012
                                 + 1.47331 w_k1_MSFT_s012
                                 + 1.31829 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.945071 w_k1_ADM_s012
                                 + 1.0309 w_k1_AXP_s012 + 1.31908 w_k1_DIS_s012
                                 + 1.07018 w_k1_FDX_s012
                                 + 0.929692 w_k1_IBM_s012
                                 + 0.948861 w_k1_INTC_s012
                                 + 0.87894 w_k1_KO_s012
                                 + 0.856357 w_k1_MAT_s012
                                 + 1.00246 w_k1_MSFT_s012
                                 + 0.924954 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.984294 w_k1_ADM_s012
                                 + 1.21554 w_k1_AXP_s012
                                 + 1.12401 w_k1_DIS_s012
                                 + 1.01627 w_k1_FDX_s012
                                 + 1.42448 w_k1_IBM_s012
                                 + 0.928774 w_k1_INTC_s012
                                 + 1.06128 w_k1_KO_s012 + 1.51322 w_k1_MAT_s012
                                 + 0.950411 w_k1_MSFT_s012
                                 + 0.949608 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.10892 w_k1_ADM_s012
                                 + 1.44385 w_k1_AXP_s012
                                 + 1.41823 w_k1_DIS_s012
                                 + 0.981764 w_k1_FDX_s012
                                 + 1.91426 w_k1_IBM_s012
                                 + 0.885137 w_k1_INTC_s012
                                 + 1.39733 w_k1_KO_s012 + 3.0325 w_k1_MAT_s012
                                 + 1.11243 w_k1_MSFT_s012
                                 + 1.26875 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.945526 w_k1_ADM_s012
                                 + 0.858849 w_k1_AXP_s012
                                 + 0.965981 w_k1_DIS_s012
                                 + 0.838217 w_k1_FDX_s012
                                 + 1.02405 w_k1_IBM_s012
                                 + 0.788571 w_k1_INTC_s012
                                 + 0.92558 w_k1_KO_s012
                                 + 0.694519 w_k1_MAT_s012
                                 + 0.857749 w_k1_MSFT_s012
                                 + 0.848652 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.18324 w_k1_ADM_s012
                                 + 1.6958 w_k1_AXP_s012 + 1.70494 w_k1_DIS_s012
                                 + 1.36822 w_k1_FDX_s012
                                 + 1.06924 w_k1_IBM_s012
                                 + 1.35797 w_k1_INTC_s012
                                 + 1.21954 w_k1_KO_s012 + 1.16208 w_k1_MAT_s012
                                 + 1.37973 w_k1_MSFT_s012
                                 + 1.20601 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.789281 w_k1_ADM_s012
                                 + 0.697197 w_k1_AXP_s012
                                 + 0.722886 w_k1_DIS_s012
                                 + 0.7605 w_k1_FDX_s012 + 0.64123 w_k1_IBM_s012
                                 + 0.814065 w_k1_INTC_s012
                                 + 0.691188 w_k1_KO_s012
                                 + 0.651994 w_k1_MAT_s012
                                 + 0.856891 w_k1_MSFT_s012
                                 + 0.857773 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.08221 w_k1_ADM_s013
                                 + 1.02545 w_k1_AXP_s013
                                 + 0.895652 w_k1_DIS_s013
                                 + 0.977538 w_k1_FDX_s013
                                 + 1.35157 w_k1_IBM_s013
                                 + 0.920733 w_k1_INTC_s013
                                 + 1.04779 w_k1_KO_s013 + 1.21989 w_k1_MAT_s013
                                 + 0.891466 w_k1_MSFT_s013
                                 + 1.01057 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.09343 w_k1_ADM_s013
                                 + 0.669471 w_k1_AXP_s013
                                 + 0.812841 w_k1_DIS_s013
                                 + 0.875565 w_k1_FDX_s013
                                 + 0.81188 w_k1_IBM_s013
                                 + 0.696432 w_k1_INTC_s013
                                 + 0.761406 w_k1_KO_s013
                                 + 0.816219 w_k1_MAT_s013
                                 + 0.857356 w_k1_MSFT_s013
                                 + 0.817794 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.54115 w_k1_ADM_s013
                                 + 1.30287 w_k1_AXP_s013
                                 + 1.19041 w_k1_DIS_s013
                                 + 1.35113 w_k1_FDX_s013
                                 + 0.678171 w_k1_IBM_s013
                                 + 0.94975 w_k1_INTC_s013
                                 + 0.97403 w_k1_KO_s013
                                 + 0.760524 w_k1_MAT_s013
                                 + 1.08722 w_k1_MSFT_s013
                                 + 1.08885 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.15548 w_k1_ADM_s013
                                 + 0.972022 w_k1_AXP_s013
                                 + 0.536525 w_k1_DIS_s013
                                 + 0.85145 w_k1_FDX_s013
                                 + 0.632422 w_k1_IBM_s013
                                 + 0.784946 w_k1_INTC_s013
                                 + 0.668897 w_k1_KO_s013
                                 + 0.874215 w_k1_MAT_s013
                                 + 0.647801 w_k1_MSFT_s013
                                 + 0.824447 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.902254 w_k1_ADM_s013
                                 + 0.745877 w_k1_AXP_s013
                                 + 0.650062 w_k1_DIS_s013
                                 + 0.699431 w_k1_FDX_s013
                                 + 1.39525 w_k1_IBM_s013
                                 + 0.611875 w_k1_INTC_s013
                                 + 0.94481 w_k1_KO_s013 + 1.29041 w_k1_MAT_s013
                                 + 0.67461 w_k1_MSFT_s013
                                 + 0.785401 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.58355 w_k1_ADM_s013
                                 + 1.3668 w_k1_AXP_s013 + 1.67614 w_k1_DIS_s013
                                 + 1.43767 w_k1_FDX_s013
                                 + 1.30616 w_k1_IBM_s013
                                 + 1.18208 w_k1_INTC_s013
                                 + 1.39636 w_k1_KO_s013 + 1.13851 w_k1_MAT_s013
                                 + 1.24053 w_k1_MSFT_s013
                                 + 1.27731 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.21882 w_k1_ADM_s013
                                 + 1.0734 w_k1_AXP_s013 + 1.08487 w_k1_DIS_s013
                                 + 1.06697 w_k1_FDX_s013
                                 + 1.07252 w_k1_IBM_s013
                                 + 1.00258 w_k1_INTC_s013
                                 + 1.00047 w_k1_KO_s013
                                 + 0.957008 w_k1_MAT_s013
                                 + 1.06149 w_k1_MSFT_s013
                                 + 0.958711 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.61449 w_k1_ADM_s013
                                 + 1.59342 w_k1_AXP_s013
                                 + 1.00789 w_k1_DIS_s013 + 1.2449 w_k1_FDX_s013
                                 + 0.932439 w_k1_IBM_s013
                                 + 1.02401 w_k1_INTC_s013
                                 + 0.932552 w_k1_KO_s013
                                 + 1.43637 w_k1_MAT_s013
                                 + 0.949821 w_k1_MSFT_s013
                                 + 0.98382 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.818567 w_k1_ADM_s014
                                 + 0.715755 w_k1_AXP_s014
                                 + 0.660551 w_k1_DIS_s014
                                 + 0.943378 w_k1_FDX_s014
                                 + 0.65449 w_k1_IBM_s014
                                 + 0.660328 w_k1_INTC_s014
                                 + 0.733104 w_k1_KO_s014
                                 + 0.757147 w_k1_MAT_s014
                                 + 0.766437 w_k1_MSFT_s014
                                 + 0.915749 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.80658 w_k1_ADM_s014
                                 + 0.952842 w_k1_AXP_s014
                                 + 0.712089 w_k1_DIS_s014
                                 + 0.820647 w_k1_FDX_s014
                                 + 1.17393 w_k1_IBM_s014
                                 + 0.857276 w_k1_INTC_s014
                                 + 0.845328 w_k1_KO_s014
                                 + 1.06116 w_k1_MAT_s014
                                 + 0.878884 w_k1_MSFT_s014
                                 + 1.02638 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.02438 w_k1_ADM_s014
                                 + 1.06051 w_k1_AXP_s014
                                 + 0.91498 w_k1_DIS_s014
                                 + 0.999637 w_k1_FDX_s014
                                 + 1.2788 w_k1_IBM_s014
                                 + 0.852139 w_k1_INTC_s014
                                 + 1.06119 w_k1_KO_s014 + 1.10856 w_k1_MAT_s014
                                 + 0.883591 w_k1_MSFT_s014
                                 + 1.11857 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.863714 w_k1_ADM_s014
                                 + 1.1683 w_k1_AXP_s014 + 1.24709 w_k1_DIS_s014
                                 + 1.2368 w_k1_FDX_s014
                                 + 0.686691 w_k1_IBM_s014
                                 + 0.938099 w_k1_INTC_s014
                                 + 0.883578 w_k1_KO_s014
                                 + 0.688121 w_k1_MAT_s014
                                 + 1.10804 w_k1_MSFT_s014 + 1.0365 w_k1_PG_s014
                                 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.1963 w_k1_ADM_s014
                                 + 0.881928 w_k1_AXP_s014
                                 + 0.952269 w_k1_DIS_s014
                                 + 1.02301 w_k1_FDX_s014
                                 + 0.94507 w_k1_IBM_s014
                                 + 0.7903 w_k1_INTC_s014
                                 + 0.977887 w_k1_KO_s014
                                 + 0.730098 w_k1_MAT_s014
                                 + 0.972029 w_k1_MSFT_s014
                                 + 0.921666 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.982212 w_k1_ADM_s014
                                 + 1.80501 w_k1_AXP_s014
                                 + 1.73774 w_k1_DIS_s014
                                 + 1.46544 w_k1_FDX_s014
                                 + 1.53171 w_k1_IBM_s014
                                 + 1.32348 w_k1_INTC_s014
                                 + 1.48978 w_k1_KO_s014 + 1.718 w_k1_MAT_s014
                                 + 1.41203 w_k1_MSFT_s014
                                 + 1.21276 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.903145 w_k1_ADM_s014
                                 + 1.05603 w_k1_AXP_s014
                                 + 1.13455 w_k1_DIS_s014
                                 + 0.977206 w_k1_FDX_s014
                                 + 1.4525 w_k1_IBM_s014
                                 + 0.766306 w_k1_INTC_s014
                                 + 1.29986 w_k1_KO_s014 + 1.25781 w_k1_MAT_s014
                                 + 1.0062 w_k1_MSFT_s014 + 1.00667 w_k1_PG_s014
                                 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.25771 w_k1_ADM_s014
                                 + 1.216 w_k1_AXP_s014 + 1.32514 w_k1_DIS_s014
                                 + 1.31804 w_k1_FDX_s014
                                 + 0.921601 w_k1_IBM_s014
                                 + 1.03771 w_k1_INTC_s014
                                 + 0.928534 w_k1_KO_s014
                                 + 0.724077 w_k1_MAT_s014
                                 + 1.11706 w_k1_MSFT_s014
                                 + 1.08253 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.17652 w_k1_ADM_s015
                                 + 0.898654 w_k1_AXP_s015
                                 + 0.927571 w_k1_DIS_s015
                                 + 1.11116 w_k1_FDX_s015
                                 + 0.993486 w_k1_IBM_s015
                                 + 1.07939 w_k1_INTC_s015
                                 + 1.11503 w_k1_KO_s015
                                 + 0.940983 w_k1_MAT_s015
                                 + 1.20902 w_k1_MSFT_s015
                                 + 1.01217 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.639501 w_k1_ADM_s015
                                 + 0.623381 w_k1_AXP_s015
                                 + 0.471354 w_k1_DIS_s015
                                 + 0.632178 w_k1_FDX_s015
                                 + 0.703248 w_k1_IBM_s015
                                 + 0.991163 w_k1_INTC_s015
                                 + 0.709493 w_k1_KO_s015
                                 + 0.51235 w_k1_MAT_s015
                                 + 0.806244 w_k1_MSFT_s015
                                 + 0.80891 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.61221 w_k1_ADM_s015
                                 + 1.58254 w_k1_AXP_s015
                                 + 1.79182 w_k1_DIS_s015 + 1.4238 w_k1_FDX_s015
                                 + 1.32057 w_k1_IBM_s015
                                 + 1.55831 w_k1_INTC_s015
                                 + 1.61229 w_k1_KO_s015 + 1.51401 w_k1_MAT_s015
                                 + 1.41347 w_k1_MSFT_s015 + 1.1786 w_k1_PG_s015
                                 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.39705 w_k1_ADM_s015
                                 + 1.33275 w_k1_AXP_s015
                                 + 0.978426 w_k1_DIS_s015
                                 + 1.29406 w_k1_FDX_s015
                                 + 0.785854 w_k1_IBM_s015
                                 + 1.07851 w_k1_INTC_s015
                                 + 0.89778 w_k1_KO_s015 + 1.05522 w_k1_MAT_s015
                                 + 0.919266 w_k1_MSFT_s015
                                 + 0.997619 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.13119 w_k1_ADM_s015
                                 + 1.01361 w_k1_AXP_s015
                                 + 1.17763 w_k1_DIS_s015
                                 + 1.01426 w_k1_FDX_s015
                                 + 1.21409 w_k1_IBM_s015
                                 + 1.0705 w_k1_INTC_s015 + 1.21678 w_k1_KO_s015
                                 + 1.20208 w_k1_MAT_s015
                                 + 1.13004 w_k1_MSFT_s015
                                 + 1.01893 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.98132 w_k1_ADM_s015
                                 + 0.956796 w_k1_AXP_s015
                                 + 0.77955 w_k1_DIS_s015
                                 + 0.931747 w_k1_FDX_s015
                                 + 0.964032 w_k1_IBM_s015
                                 + 1.16479 w_k1_INTC_s015
                                 + 1.00512 w_k1_KO_s015
                                 + 0.637644 w_k1_MAT_s015
                                 + 0.988504 w_k1_MSFT_s015
                                 + 0.881317 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.05774 w_k1_ADM_s015
                                 + 0.716095 w_k1_AXP_s015
                                 + 0.60495 w_k1_DIS_s015
                                 + 0.764084 w_k1_FDX_s015
                                 + 0.94379 w_k1_IBM_s015
                                 + 1.12091 w_k1_INTC_s015
                                 + 0.879449 w_k1_KO_s015
                                 + 0.710925 w_k1_MAT_s015
                                 + 0.88362 w_k1_MSFT_s015
                                 + 0.844725 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.15123 w_k1_ADM_s015
                                 + 1.15565 w_k1_AXP_s015
                                 + 1.01125 w_k1_DIS_s015
                                 + 1.00795 w_k1_FDX_s015
                                 + 1.34724 w_k1_IBM_s015
                                 + 1.16605 w_k1_INTC_s015
                                 + 1.16824 w_k1_KO_s015 + 1.33881 w_k1_MAT_s015
                                 + 1.11337 w_k1_MSFT_s015
                                 + 1.14828 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.22152 w_k1_ADM_s016
                                 + 0.791321 w_k1_AXP_s016
                                 + 0.77236 w_k1_DIS_s016
                                 + 1.01598 w_k1_FDX_s016
                                 + 0.69851 w_k1_IBM_s016
                                 + 0.853148 w_k1_INTC_s016
                                 + 0.768827 w_k1_KO_s016
                                 + 0.551133 w_k1_MAT_s016
                                 + 0.986835 w_k1_MSFT_s016
                                 + 0.898443 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.735115 w_k1_ADM_s016
                                 + 0.641449 w_k1_AXP_s016
                                 + 0.631376 w_k1_DIS_s016
                                 + 0.696167 w_k1_FDX_s016
                                 + 1.00138 w_k1_IBM_s016
                                 + 0.975213 w_k1_INTC_s016
                                 + 0.783843 w_k1_KO_s016
                                 + 0.513597 w_k1_MAT_s016
                                 + 0.75938 w_k1_MSFT_s016
                                 + 0.822997 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.892078 w_k1_ADM_s016
                                 + 1.02827 w_k1_AXP_s016
                                 + 0.981253 w_k1_DIS_s016 + 1.048 w_k1_FDX_s016
                                 + 1.1724 w_k1_IBM_s016
                                 + 1.02617 w_k1_INTC_s016
                                 + 0.986671 w_k1_KO_s016
                                 + 1.00202 w_k1_MAT_s016
                                 + 0.991096 w_k1_MSFT_s016
                                 + 0.92826 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.19996 w_k1_ADM_s016
                                 + 1.06148 w_k1_AXP_s016
                                 + 1.13273 w_k1_DIS_s016
                                 + 1.16866 w_k1_FDX_s016
                                 + 0.813117 w_k1_IBM_s016
                                 + 1.03812 w_k1_INTC_s016
                                 + 1.07393 w_k1_KO_s016 + 1.05046 w_k1_MAT_s016
                                 + 1.28505 w_k1_MSFT_s016
                                 + 1.01754 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.10171 w_k1_ADM_s016
                                 + 0.886234 w_k1_AXP_s016
                                 + 0.786546 w_k1_DIS_s016
                                 + 0.907705 w_k1_FDX_s016
                                 + 1.03953 w_k1_IBM_s016
                                 + 0.917736 w_k1_INTC_s016
                                 + 0.970738 w_k1_KO_s016
                                 + 1.02238 w_k1_MAT_s016
                                 + 0.968979 w_k1_MSFT_s016
                                 + 1.02649 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.906475 w_k1_ADM_s016
                                 + 0.583986 w_k1_AXP_s016
                                 + 0.580243 w_k1_DIS_s016
                                 + 0.788196 w_k1_FDX_s016
                                 + 0.674344 w_k1_IBM_s016
                                 + 0.753067 w_k1_INTC_s016
                                 + 0.664395 w_k1_KO_s016
                                 + 0.550719 w_k1_MAT_s016
                                 + 0.683611 w_k1_MSFT_s016
                                 + 0.905075 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.17942 w_k1_ADM_s016
                                 + 1.24241 w_k1_AXP_s016
                                 + 1.38215 w_k1_DIS_s016
                                 + 1.46168 w_k1_FDX_s016
                                 + 1.15016 w_k1_IBM_s016
                                 + 1.37919 w_k1_INTC_s016
                                 + 1.19154 w_k1_KO_s016
                                 + 0.949862 w_k1_MAT_s016
                                 + 1.23236 w_k1_MSFT_s016
                                 + 1.12155 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.31139 w_k1_ADM_s016
                                 + 1.35942 w_k1_AXP_s016
                                 + 1.39647 w_k1_DIS_s016
                                 + 1.15965 w_k1_FDX_s016
                                 + 1.56432 w_k1_IBM_s016
                                 + 1.24355 w_k1_INTC_s016
                                 + 1.28954 w_k1_KO_s016 + 1.32548 w_k1_MAT_s016
                                 + 1.14048 w_k1_MSFT_s016
                                 + 1.09738 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.895929 w_k1_ADM_s017
                                 + 0.777981 w_k1_AXP_s017
                                 + 0.718375 w_k1_DIS_s017
                                 + 0.813661 w_k1_FDX_s017
                                 + 0.795829 w_k1_IBM_s017
                                 + 0.826231 w_k1_INTC_s017
                                 + 0.687925 w_k1_KO_s017
                                 + 0.552475 w_k1_MAT_s017
                                 + 0.808022 w_k1_MSFT_s017
                                 + 0.904837 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.69676 w_k1_ADM_s017
                                 + 1.49863 w_k1_AXP_s017
                                 + 1.48391 w_k1_DIS_s017
                                 + 1.39123 w_k1_FDX_s017
                                 + 1.59872 w_k1_IBM_s017
                                 + 1.31925 w_k1_INTC_s017
                                 + 1.34471 w_k1_KO_s017 + 1.3067 w_k1_MAT_s017
                                 + 1.26722 w_k1_MSFT_s017
                                 + 1.07295 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.08884 w_k1_ADM_s017
                                 + 1.24442 w_k1_AXP_s017
                                 + 1.76147 w_k1_DIS_s017
                                 + 1.23603 w_k1_FDX_s017
                                 + 1.16288 w_k1_IBM_s017
                                 + 1.29523 w_k1_INTC_s017
                                 + 1.45678 w_k1_KO_s017 + 1.15324 w_k1_MAT_s017
                                 + 1.24121 w_k1_MSFT_s017
                                 + 1.18693 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.66372 w_k1_ADM_s017
                                 + 1.35485 w_k1_AXP_s017 + 1.6434 w_k1_DIS_s017
                                 + 1.29884 w_k1_FDX_s017
                                 + 1.11348 w_k1_IBM_s017
                                 + 1.34161 w_k1_INTC_s017
                                 + 1.28252 w_k1_KO_s017
                                 + 0.989356 w_k1_MAT_s017
                                 + 1.29833 w_k1_MSFT_s017
                                 + 1.22474 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.99111 w_k1_ADM_s017
                                 + 1.23655 w_k1_AXP_s017
                                 + 0.934366 w_k1_DIS_s017
                                 + 0.934407 w_k1_FDX_s017
                                 + 1.31664 w_k1_IBM_s017
                                 + 1.12792 w_k1_INTC_s017
                                 + 0.938148 w_k1_KO_s017
                                 + 1.30064 w_k1_MAT_s017
                                 + 0.892604 w_k1_MSFT_s017
                                 + 0.928648 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.998912 w_k1_ADM_s017
                                 + 0.874467 w_k1_AXP_s017
                                 + 1.22886 w_k1_DIS_s017
                                 + 1.04593 w_k1_FDX_s017
                                 + 1.33702 w_k1_IBM_s017
                                 + 1.08303 w_k1_INTC_s017
                                 + 1.19645 w_k1_KO_s017 + 1.07105 w_k1_MAT_s017
                                 + 1.18587 w_k1_MSFT_s017 + 1.0624 w_k1_PG_s017
                                 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.97667 w_k1_ADM_s017
                                 + 1.00761 w_k1_AXP_s017
                                 + 1.40682 w_k1_DIS_s017
                                 + 1.19574 w_k1_FDX_s017
                                 + 0.954698 w_k1_IBM_s017
                                 + 1.16806 w_k1_INTC_s017
                                 + 0.964222 w_k1_KO_s017
                                 + 0.756384 w_k1_MAT_s017
                                 + 1.19179 w_k1_MSFT_s017
                                 + 0.947263 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.21739 w_k1_ADM_s017
                                 + 1.04683 w_k1_AXP_s017
                                 + 0.977104 w_k1_DIS_s017
                                 + 1.13405 w_k1_FDX_s017
                                 + 0.742575 w_k1_IBM_s017
                                 + 1.03091 w_k1_INTC_s017
                                 + 0.795678 w_k1_KO_s017
                                 + 0.682232 w_k1_MAT_s017
                                 + 1.04331 w_k1_MSFT_s017
                                 + 1.00361 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.619664 w_k1_ADM_s100
                                 + 0.643688 w_k1_AXP_s100
                                 + 0.617745 w_k1_DIS_s100
                                 + 0.702313 w_k1_FDX_s100
                                 + 0.809445 w_k1_IBM_s100
                                 + 0.728281 w_k1_INTC_s100
                                 + 0.746646 w_k1_KO_s100
                                 + 0.708878 w_k1_MAT_s100
                                 + 0.681603 w_k1_MSFT_s100
                                 + 0.791713 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.896111 w_k1_ADM_s100
                                 + 0.886082 w_k1_AXP_s100
                                 + 0.795564 w_k1_DIS_s100
                                 + 0.979258 w_k1_FDX_s100
                                 + 0.667637 w_k1_IBM_s100
                                 + 0.965005 w_k1_INTC_s100
                                 + 0.744169 w_k1_KO_s100
                                 + 0.678339 w_k1_MAT_s100
                                 + 0.935792 w_k1_MSFT_s100
                                 + 0.864441 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.06288 w_k1_ADM_s100
                                 + 1.16056 w_k1_AXP_s100
                                 + 1.43718 w_k1_DIS_s100
                                 + 1.16812 w_k1_FDX_s100
                                 + 1.22112 w_k1_IBM_s100
                                 + 1.23312 w_k1_INTC_s100
                                 + 1.30666 w_k1_KO_s100 + 1.26829 w_k1_MAT_s100
                                 + 1.19553 w_k1_MSFT_s100
                                 + 1.06763 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.16124 w_k1_ADM_s100
                                 + 1.26065 w_k1_AXP_s100
                                 + 1.18685 w_k1_DIS_s100
                                 + 1.10583 w_k1_FDX_s100
                                 + 1.50367 w_k1_IBM_s100
                                 + 1.16945 w_k1_INTC_s100
                                 + 1.44279 w_k1_KO_s100 + 2.02456 w_k1_MAT_s100
                                 + 1.19168 w_k1_MSFT_s100
                                 + 1.02767 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.01104 w_k1_ADM_s100
                                 + 0.86998 w_k1_AXP_s100
                                 + 0.960164 w_k1_DIS_s100
                                 + 0.993066 w_k1_FDX_s100
                                 + 1.14492 w_k1_IBM_s100
                                 + 0.932314 w_k1_INTC_s100
                                 + 1.10053 w_k1_KO_s100 + 1.13724 w_k1_MAT_s100
                                 + 0.916791 w_k1_MSFT_s100
                                 + 0.865099 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.805527 w_k1_ADM_s100
                                 + 0.736205 w_k1_AXP_s100
                                 + 0.723037 w_k1_DIS_s100
                                 + 0.802963 w_k1_FDX_s100
                                 + 1.25819 w_k1_IBM_s100
                                 + 0.822396 w_k1_INTC_s100
                                 + 1.06338 w_k1_KO_s100 + 1.31627 w_k1_MAT_s100
                                 + 0.784875 w_k1_MSFT_s100
                                 + 0.891462 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.0812 w_k1_ADM_s100
                                 + 1.00699 w_k1_AXP_s100
                                 + 1.30003 w_k1_DIS_s100 + 1.2339 w_k1_FDX_s100
                                 + 0.899447 w_k1_IBM_s100
                                 + 1.19943 w_k1_INTC_s100
                                 + 1.08465 w_k1_KO_s100
                                 + 0.783231 w_k1_MAT_s100
                                 + 1.16453 w_k1_MSFT_s100 + 1.0565 w_k1_PG_s100
                                 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.88463 w_k1_ADM_s100
                                 + 0.902332 w_k1_AXP_s100
                                 + 0.8417 w_k1_DIS_s100
                                 + 0.902266 w_k1_FDX_s100
                                 + 0.943317 w_k1_IBM_s100
                                 + 0.821193 w_k1_INTC_s100
                                 + 0.875691 w_k1_KO_s100
                                 + 1.00816 w_k1_MAT_s100
                                 + 0.824943 w_k1_MSFT_s100
                                 + 0.827608 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.02126 w_k1_ADM_s101
                                 + 1.34597 w_k1_AXP_s101
                                 + 1.01085 w_k1_DIS_s101
                                 + 0.987649 w_k1_FDX_s101
                                 + 1.15901 w_k1_IBM_s101
                                 + 0.92848 w_k1_INTC_s101
                                 + 0.918057 w_k1_KO_s101
                                 + 1.44556 w_k1_MAT_s101
                                 + 0.912611 w_k1_MSFT_s101
                                 + 0.978347 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.07413 w_k1_ADM_s101
                                 + 0.752048 w_k1_AXP_s101
                                 + 0.643259 w_k1_DIS_s101
                                 + 0.788117 w_k1_FDX_s101
                                 + 0.71208 w_k1_IBM_s101
                                 + 0.989405 w_k1_INTC_s101
                                 + 0.705617 w_k1_KO_s101
                                 + 0.755696 w_k1_MAT_s101
                                 + 0.86673 w_k1_MSFT_s101
                                 + 0.720901 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.0764 w_k1_ADM_s101
                                 + 1.24566 w_k1_AXP_s101
                                 + 0.812334 w_k1_DIS_s101
                                 + 0.94842 w_k1_FDX_s101
                                 + 0.930281 w_k1_IBM_s101
                                 + 1.11936 w_k1_INTC_s101
                                 + 0.817193 w_k1_KO_s101
                                 + 1.16137 w_k1_MAT_s101
                                 + 1.01553 w_k1_MSFT_s101
                                 + 0.993178 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.12099 w_k1_ADM_s101
                                 + 1.05873 w_k1_AXP_s101
                                 + 0.912265 w_k1_DIS_s101
                                 + 0.774775 w_k1_FDX_s101
                                 + 1.37117 w_k1_IBM_s101
                                 + 1.05064 w_k1_INTC_s101
                                 + 1.02937 w_k1_KO_s101 + 1.21213 w_k1_MAT_s101
                                 + 0.837019 w_k1_MSFT_s101
                                 + 0.890434 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.07321 w_k1_ADM_s101
                                 + 1.18452 w_k1_AXP_s101
                                 + 1.26773 w_k1_DIS_s101
                                 + 1.20486 w_k1_FDX_s101
                                 + 1.08507 w_k1_IBM_s101
                                 + 1.12138 w_k1_INTC_s101
                                 + 1.13656 w_k1_KO_s101 + 1.23864 w_k1_MAT_s101
                                 + 1.15306 w_k1_MSFT_s101
                                 + 1.16563 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.27354 w_k1_ADM_s101
                                 + 1.49447 w_k1_AXP_s101
                                 + 1.40197 w_k1_DIS_s101
                                 + 1.15811 w_k1_FDX_s101
                                 + 1.49011 w_k1_IBM_s101
                                 + 1.09118 w_k1_INTC_s101
                                 + 1.20598 w_k1_KO_s101 + 1.94216 w_k1_MAT_s101
                                 + 1.03958 w_k1_MSFT_s101
                                 + 1.19904 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.04373 w_k1_ADM_s101
                                 + 0.816069 w_k1_AXP_s101
                                 + 0.711335 w_k1_DIS_s101
                                 + 0.729588 w_k1_FDX_s101
                                 + 1.08172 w_k1_IBM_s101
                                 + 0.83221 w_k1_INTC_s101
                                 + 0.934933 w_k1_KO_s101
                                 + 0.824822 w_k1_MAT_s101
                                 + 0.790771 w_k1_MSFT_s101
                                 + 0.798597 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.990316 w_k1_ADM_s101
                                 + 0.792176 w_k1_AXP_s101
                                 + 1.0414 w_k1_DIS_s101 + 0.97977 w_k1_FDX_s101
                                 + 0.891712 w_k1_IBM_s101
                                 + 0.942149 w_k1_INTC_s101
                                 + 0.883135 w_k1_KO_s101
                                 + 0.782579 w_k1_MAT_s101
                                 + 0.989397 w_k1_MSFT_s101
                                 + 0.799175 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.23738 w_k1_ADM_s102
                                 + 1.25331 w_k1_AXP_s102 + 1.5191 w_k1_DIS_s102
                                 + 1.21707 w_k1_FDX_s102
                                 + 1.21495 w_k1_IBM_s102
                                 + 1.17909 w_k1_INTC_s102
                                 + 1.21777 w_k1_KO_s102 + 1.36927 w_k1_MAT_s102
                                 + 1.35811 w_k1_MSFT_s102
                                 + 1.17901 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.705526 w_k1_ADM_s102
                                 + 0.597922 w_k1_AXP_s102
                                 + 0.468398 w_k1_DIS_s102
                                 + 0.677218 w_k1_FDX_s102
                                 + 0.694463 w_k1_IBM_s102
                                 + 0.727339 w_k1_INTC_s102
                                 + 0.71085 w_k1_KO_s102
                                 + 0.833518 w_k1_MAT_s102
                                 + 0.705827 w_k1_MSFT_s102
                                 + 0.853766 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.969838 w_k1_ADM_s102
                                 + 0.788507 w_k1_AXP_s102
                                 + 0.900348 w_k1_DIS_s102
                                 + 0.933338 w_k1_FDX_s102
                                 + 0.740836 w_k1_IBM_s102
                                 + 0.941457 w_k1_INTC_s102
                                 + 0.86636 w_k1_KO_s102
                                 + 0.846589 w_k1_MAT_s102
                                 + 1.12176 w_k1_MSFT_s102
                                 + 0.992896 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.967077 w_k1_ADM_s102
                                 + 1.14562 w_k1_AXP_s102
                                 + 0.956501 w_k1_DIS_s102
                                 + 1.0221 w_k1_FDX_s102 + 1.2875 w_k1_IBM_s102
                                 + 0.857906 w_k1_INTC_s102
                                 + 1.1686 w_k1_KO_s102 + 1.92924 w_k1_MAT_s102
                                 + 1.08494 w_k1_MSFT_s102
                                 + 1.13541 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.850485 w_k1_ADM_s102
                                 + 1.00785 w_k1_AXP_s102
                                 + 0.743948 w_k1_DIS_s102
                                 + 0.921174 w_k1_FDX_s102
                                 + 0.902261 w_k1_IBM_s102
                                 + 0.958344 w_k1_INTC_s102
                                 + 0.837152 w_k1_KO_s102
                                 + 1.20975 w_k1_MAT_s102
                                 + 0.989102 w_k1_MSFT_s102
                                 + 0.975813 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.17868 w_k1_ADM_s102
                                 + 0.978185 w_k1_AXP_s102
                                 + 1.47177 w_k1_DIS_s102
                                 + 1.28601 w_k1_FDX_s102
                                 + 0.854333 w_k1_IBM_s102
                                 + 1.20396 w_k1_INTC_s102
                                 + 1.14135 w_k1_KO_s102
                                 + 0.940961 w_k1_MAT_s102
                                 + 1.37161 w_k1_MSFT_s102
                                 + 1.23933 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.817215 w_k1_ADM_s102
                                 + 0.785752 w_k1_AXP_s102
                                 + 0.81552 w_k1_DIS_s102
                                 + 0.83329 w_k1_FDX_s102
                                 + 1.04528 w_k1_IBM_s102
                                 + 0.75549 w_k1_INTC_s102
                                 + 0.879952 w_k1_KO_s102
                                 + 1.37506 w_k1_MAT_s102
                                 + 0.901819 w_k1_MSFT_s102
                                 + 0.997874 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.4276 w_k1_ADM_s102
                                 + 1.0432 w_k1_AXP_s102 + 1.28992 w_k1_DIS_s102
                                 + 1.22526 w_k1_FDX_s102
                                 + 1.23865 w_k1_IBM_s102
                                 + 0.776098 w_k1_INTC_s102
                                 + 1.13291 w_k1_KO_s102 + 1.06053 w_k1_MAT_s102
                                 + 1.22981 w_k1_MSFT_s102
                                 + 1.04813 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.979908 w_k1_ADM_s103
                                 + 1.19228 w_k1_AXP_s103
                                 + 0.876281 w_k1_DIS_s103 + 0.994 w_k1_FDX_s103
                                 + 0.840097 w_k1_IBM_s103
                                 + 1.10419 w_k1_INTC_s103
                                 + 0.866871 w_k1_KO_s103
                                 + 0.858771 w_k1_MAT_s103
                                 + 0.897654 w_k1_MSFT_s103
                                 + 1.01524 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.18014 w_k1_ADM_s103
                                 + 1.14668 w_k1_AXP_s103
                                 + 1.30441 w_k1_DIS_s103
                                 + 1.23455 w_k1_FDX_s103
                                 + 0.890495 w_k1_IBM_s103
                                 + 1.43532 w_k1_INTC_s103
                                 + 1.12788 w_k1_KO_s103 + 0.82967 w_k1_MAT_s103
                                 + 1.16461 w_k1_MSFT_s103
                                 + 1.22836 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.724359 w_k1_ADM_s103
                                 + 0.623775 w_k1_AXP_s103
                                 + 0.630335 w_k1_DIS_s103
                                 + 0.659306 w_k1_FDX_s103
                                 + 0.744339 w_k1_IBM_s103
                                 + 0.732885 w_k1_INTC_s103
                                 + 0.743934 w_k1_KO_s103
                                 + 0.657665 w_k1_MAT_s103
                                 + 0.739575 w_k1_MSFT_s103
                                 + 0.750148 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.811419 w_k1_ADM_s103
                                 + 0.756483 w_k1_AXP_s103
                                 + 0.782358 w_k1_DIS_s103
                                 + 0.816162 w_k1_FDX_s103
                                 + 0.943164 w_k1_IBM_s103
                                 + 0.749529 w_k1_INTC_s103
                                 + 0.830151 w_k1_KO_s103
                                 + 0.836758 w_k1_MAT_s103
                                 + 0.81487 w_k1_MSFT_s103
                                 + 0.924367 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.21734 w_k1_ADM_s103
                                 + 1.06543 w_k1_AXP_s103
                                 + 1.44752 w_k1_DIS_s103
                                 + 1.07813 w_k1_FDX_s103
                                 + 1.41378 w_k1_IBM_s103
                                 + 0.949836 w_k1_INTC_s103
                                 + 1.35671 w_k1_KO_s103 + 1.36438 w_k1_MAT_s103
                                 + 1.10563 w_k1_MSFT_s103
                                 + 1.25431 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.12561 w_k1_ADM_s103
                                 + 1.16975 w_k1_AXP_s103
                                 + 1.04559 w_k1_DIS_s103
                                 + 1.13755 w_k1_FDX_s103
                                 + 1.19012 w_k1_IBM_s103
                                 + 1.06509 w_k1_INTC_s103 + 1.0933 w_k1_KO_s103
                                 + 1.19597 w_k1_MAT_s103
                                 + 1.11968 w_k1_MSFT_s103
                                 + 1.27679 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.21037 w_k1_ADM_s103
                                 + 1.59665 w_k1_AXP_s103
                                 + 1.64597 w_k1_DIS_s103
                                 + 1.51347 w_k1_FDX_s103
                                 + 1.13447 w_k1_IBM_s103
                                 + 1.26672 w_k1_INTC_s103
                                 + 1.32623 w_k1_KO_s103 + 1.25655 w_k1_MAT_s103
                                 + 1.47561 w_k1_MSFT_s103
                                 + 1.30267 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.928604 w_k1_ADM_s103
                                 + 0.861468 w_k1_AXP_s103
                                 + 1.07263 w_k1_DIS_s103
                                 + 0.977733 w_k1_FDX_s103
                                 + 1.15124 w_k1_IBM_s103
                                 + 0.958688 w_k1_INTC_s103
                                 + 1.11818 w_k1_KO_s103
                                 + 0.971084 w_k1_MAT_s103
                                 + 1.04175 w_k1_MSFT_s103
                                 + 1.05905 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.828526 w_k1_ADM_s104
                                 + 0.783418 w_k1_AXP_s104
                                 + 0.761366 w_k1_DIS_s104
                                 + 0.830684 w_k1_FDX_s104
                                 + 1.11273 w_k1_IBM_s104
                                 + 0.894201 w_k1_INTC_s104
                                 + 0.871394 w_k1_KO_s104
                                 + 0.752343 w_k1_MAT_s104
                                 + 0.80276 w_k1_MSFT_s104
                                 + 0.825922 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.21888 w_k1_ADM_s104
                                 + 1.02627 w_k1_AXP_s104
                                 + 0.924956 w_k1_DIS_s104
                                 + 1.17066 w_k1_FDX_s104
                                 + 0.83866 w_k1_IBM_s104
                                 + 0.815754 w_k1_INTC_s104
                                 + 0.898553 w_k1_KO_s104
                                 + 0.806417 w_k1_MAT_s104
                                 + 0.866206 w_k1_MSFT_s104
                                 + 0.990065 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.846809 w_k1_ADM_s104
                                 + 0.626286 w_k1_AXP_s104
                                 + 0.584166 w_k1_DIS_s104
                                 + 0.827099 w_k1_FDX_s104
                                 + 0.749155 w_k1_IBM_s104
                                 + 0.685833 w_k1_INTC_s104
                                 + 0.598362 w_k1_KO_s104
                                 + 0.580262 w_k1_MAT_s104
                                 + 0.660485 w_k1_MSFT_s104
                                 + 0.747109 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.629977 w_k1_ADM_s104
                                 + 0.709451 w_k1_AXP_s104
                                 + 0.543099 w_k1_DIS_s104
                                 + 0.750345 w_k1_FDX_s104
                                 + 0.890538 w_k1_IBM_s104
                                 + 0.780887 w_k1_INTC_s104
                                 + 0.619537 w_k1_KO_s104
                                 + 0.682363 w_k1_MAT_s104
                                 + 0.701509 w_k1_MSFT_s104
                                 + 0.727959 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.11435 w_k1_ADM_s104
                                 + 0.75486 w_k1_AXP_s104
                                 + 0.772648 w_k1_DIS_s104
                                 + 1.02168 w_k1_FDX_s104
                                 + 0.687373 w_k1_IBM_s104
                                 + 0.81234 w_k1_INTC_s104
                                 + 0.66209 w_k1_KO_s104
                                 + 0.582381 w_k1_MAT_s104
                                 + 0.814038 w_k1_MSFT_s104
                                 + 0.80016 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.66486 w_k1_ADM_s104
                                 + 1.67707 w_k1_AXP_s104
                                 + 1.71814 w_k1_DIS_s104
                                 + 1.57756 w_k1_FDX_s104
                                 + 0.860425 w_k1_IBM_s104
                                 + 1.07827 w_k1_INTC_s104
                                 + 0.980687 w_k1_KO_s104
                                 + 0.862505 w_k1_MAT_s104
                                 + 1.19186 w_k1_MSFT_s104
                                 + 1.17856 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.05396 w_k1_ADM_s104
                                 + 0.907094 w_k1_AXP_s104
                                 + 1.23418 w_k1_DIS_s104
                                 + 1.07445 w_k1_FDX_s104 + 1.5385 w_k1_IBM_s104
                                 + 0.996922 w_k1_INTC_s104
                                 + 1.25913 w_k1_KO_s104 + 1.4702 w_k1_MAT_s104
                                 + 1.07536 w_k1_MSFT_s104
                                 + 1.14184 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.18907 w_k1_ADM_s104
                                 + 1.21561 w_k1_AXP_s104 + 1.2703 w_k1_DIS_s104
                                 + 1.22573 w_k1_FDX_s104
                                 + 1.05638 w_k1_IBM_s104
                                 + 1.02604 w_k1_INTC_s104
                                 + 0.964496 w_k1_KO_s104
                                 + 0.794614 w_k1_MAT_s104
                                 + 0.993213 w_k1_MSFT_s104
                                 + 1.03204 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.869175 w_k1_ADM_s105
                                 + 0.64721 w_k1_AXP_s105
                                 + 0.663246 w_k1_DIS_s105
                                 + 0.828645 w_k1_FDX_s105
                                 + 0.524294 w_k1_IBM_s105
                                 + 0.769026 w_k1_INTC_s105
                                 + 0.629676 w_k1_KO_s105
                                 + 0.579552 w_k1_MAT_s105
                                 + 0.85434 w_k1_MSFT_s105
                                 + 0.794843 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.63206 w_k1_ADM_s105
                                 + 1.42123 w_k1_AXP_s105
                                 + 1.72518 w_k1_DIS_s105 + 1.4085 w_k1_FDX_s105
                                 + 1.55368 w_k1_IBM_s105
                                 + 1.17295 w_k1_INTC_s105
                                 + 1.54658 w_k1_KO_s105 + 1.74478 w_k1_MAT_s105
                                 + 1.42534 w_k1_MSFT_s105
                                 + 1.37776 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.15868 w_k1_ADM_s105
                                 + 1.02919 w_k1_AXP_s105
                                 + 0.747614 w_k1_DIS_s105
                                 + 0.819384 w_k1_FDX_s105
                                 + 1.18763 w_k1_IBM_s105
                                 + 0.853474 w_k1_INTC_s105
                                 + 0.98636 w_k1_KO_s105
                                 + 0.905765 w_k1_MAT_s105
                                 + 0.735973 w_k1_MSFT_s105
                                 + 0.907432 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.40245 w_k1_ADM_s105
                                 + 1.45921 w_k1_AXP_s105
                                 + 1.55505 w_k1_DIS_s105
                                 + 1.39521 w_k1_FDX_s105
                                 + 0.96523 w_k1_IBM_s105
                                 + 1.24138 w_k1_INTC_s105
                                 + 1.22812 w_k1_KO_s105 + 1.11379 w_k1_MAT_s105
                                 + 1.2949 w_k1_MSFT_s105 + 1.07971 w_k1_PG_s105
                                 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.04212 w_k1_ADM_s105
                                 + 0.81734 w_k1_AXP_s105 + 1.068 w_k1_DIS_s105
                                 + 1.01277 w_k1_FDX_s105
                                 + 0.965823 w_k1_IBM_s105
                                 + 0.961695 w_k1_INTC_s105
                                 + 1.08417 w_k1_KO_s105
                                 + 0.850822 w_k1_MAT_s105
                                 + 1.13493 w_k1_MSFT_s105
                                 + 0.984923 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.05417 w_k1_ADM_s105
                                 + 1.2984 w_k1_AXP_s105 + 1.09985 w_k1_DIS_s105
                                 + 1.06781 w_k1_FDX_s105
                                 + 1.39761 w_k1_IBM_s105
                                 + 0.997679 w_k1_INTC_s105
                                 + 1.29346 w_k1_KO_s105 + 1.41419 w_k1_MAT_s105
                                 + 1.1585 w_k1_MSFT_s105 + 1.0101 w_k1_PG_s105
                                 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.14158 w_k1_ADM_s105
                                 + 1.12931 w_k1_AXP_s105
                                 + 1.11065 w_k1_DIS_s105
                                 + 1.17289 w_k1_FDX_s105
                                 + 1.03543 w_k1_IBM_s105
                                 + 1.25555 w_k1_INTC_s105
                                 + 1.07744 w_k1_KO_s105
                                 + 0.959371 w_k1_MAT_s105
                                 + 1.15507 w_k1_MSFT_s105
                                 + 0.954867 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.90105 w_k1_ADM_s105
                                 + 0.834943 w_k1_AXP_s105
                                 + 0.767019 w_k1_DIS_s105
                                 + 0.88363 w_k1_FDX_s105
                                 + 0.789588 w_k1_IBM_s105
                                 + 0.89575 w_k1_INTC_s105
                                 + 0.688307 w_k1_KO_s105
                                 + 0.649843 w_k1_MAT_s105
                                 + 0.725902 w_k1_MSFT_s105
                                 + 0.778771 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.22235 w_k1_ADM_s106
                                 + 0.949564 w_k1_AXP_s106
                                 + 1.16946 w_k1_DIS_s106
                                 + 1.19329 w_k1_FDX_s106
                                 + 0.911785 w_k1_IBM_s106
                                 + 1.04998 w_k1_INTC_s106
                                 + 0.996405 w_k1_KO_s106
                                 + 1.00695 w_k1_MAT_s106
                                 + 1.15867 w_k1_MSFT_s106
                                 + 1.13734 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.03599 w_k1_ADM_s106
                                 + 1.0723 w_k1_AXP_s106 + 1.13061 w_k1_DIS_s106
                                 + 1.04817 w_k1_FDX_s106
                                 + 1.29095 w_k1_IBM_s106
                                 + 0.881515 w_k1_INTC_s106
                                 + 1.04368 w_k1_KO_s106 + 1.90962 w_k1_MAT_s106
                                 + 1.06324 w_k1_MSFT_s106
                                 + 1.07288 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.92804 w_k1_ADM_s106
                                 + 0.673492 w_k1_AXP_s106
                                 + 0.545947 w_k1_DIS_s106
                                 + 0.693936 w_k1_FDX_s106
                                 + 1.01061 w_k1_IBM_s106
                                 + 0.566687 w_k1_INTC_s106
                                 + 0.71375 w_k1_KO_s106 + 1.28665 w_k1_MAT_s106
                                 + 0.685566 w_k1_MSFT_s106
                                 + 0.828851 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.54865 w_k1_ADM_s106
                                 + 1.37922 w_k1_AXP_s106
                                 + 1.61514 w_k1_DIS_s106
                                 + 1.37785 w_k1_FDX_s106
                                 + 1.26657 w_k1_IBM_s106
                                 + 1.10449 w_k1_INTC_s106
                                 + 1.12936 w_k1_KO_s106 + 1.72137 w_k1_MAT_s106
                                 + 1.25569 w_k1_MSFT_s106
                                 + 1.14489 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.30123 w_k1_ADM_s106
                                 + 1.01402 w_k1_AXP_s106
                                 + 0.696132 w_k1_DIS_s106
                                 + 0.960507 w_k1_FDX_s106
                                 + 0.804649 w_k1_IBM_s106
                                 + 0.856532 w_k1_INTC_s106
                                 + 0.663085 w_k1_KO_s106
                                 + 0.985046 w_k1_MAT_s106
                                 + 0.87355 w_k1_MSFT_s106
                                 + 0.901222 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.1778 w_k1_ADM_s106
                                 + 0.82637 w_k1_AXP_s106
                                 + 1.04681 w_k1_DIS_s106
                                 + 1.06476 w_k1_FDX_s106
                                 + 0.813615 w_k1_IBM_s106
                                 + 0.967306 w_k1_INTC_s106
                                 + 0.83194 w_k1_KO_s106 + 0.97593 w_k1_MAT_s106
                                 + 1.0198 w_k1_MSFT_s106 + 1.05486 w_k1_PG_s106
                                 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.07723 w_k1_ADM_s106
                                 + 0.87716 w_k1_AXP_s106
                                 + 0.757537 w_k1_DIS_s106
                                 + 0.870734 w_k1_FDX_s106
                                 + 1.01614 w_k1_IBM_s106
                                 + 0.850222 w_k1_INTC_s106
                                 + 0.854498 w_k1_KO_s106
                                 + 1.37429 w_k1_MAT_s106
                                 + 0.872443 w_k1_MSFT_s106
                                 + 0.905742 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.838944 w_k1_ADM_s106
                                 + 0.705678 w_k1_AXP_s106
                                 + 0.568756 w_k1_DIS_s106
                                 + 0.843226 w_k1_FDX_s106
                                 + 0.706124 w_k1_IBM_s106
                                 + 0.759189 w_k1_INTC_s106
                                 + 0.634511 w_k1_KO_s106
                                 + 0.957227 w_k1_MAT_s106
                                 + 0.739791 w_k1_MSFT_s106
                                 + 0.794098 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.692033 w_k1_ADM_s107
                                 + 0.668937 w_k1_AXP_s107
                                 + 0.662431 w_k1_DIS_s107
                                 + 0.721927 w_k1_FDX_s107
                                 + 0.796453 w_k1_IBM_s107
                                 + 0.879044 w_k1_INTC_s107
                                 + 0.692639 w_k1_KO_s107
                                 + 0.544269 w_k1_MAT_s107
                                 + 0.852427 w_k1_MSFT_s107
                                 + 0.781749 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.06031 w_k1_ADM_s107
                                 + 1.1508 w_k1_AXP_s107 + 1.28905 w_k1_DIS_s107
                                 + 1.26659 w_k1_FDX_s107
                                 + 0.947071 w_k1_IBM_s107
                                 + 1.12605 w_k1_INTC_s107
                                 + 1.03978 w_k1_KO_s107
                                 + 0.929385 w_k1_MAT_s107
                                 + 1.10917 w_k1_MSFT_s107
                                 + 1.02509 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.26357 w_k1_ADM_s107
                                 + 1.68919 w_k1_AXP_s107
                                 + 1.63416 w_k1_DIS_s107
                                 + 1.47796 w_k1_FDX_s107
                                 + 1.37771 w_k1_IBM_s107
                                 + 1.40257 w_k1_INTC_s107
                                 + 1.27571 w_k1_KO_s107 + 1.96156 w_k1_MAT_s107
                                 + 1.45932 w_k1_MSFT_s107
                                 + 1.22726 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.09201 w_k1_ADM_s107
                                 + 1.37516 w_k1_AXP_s107
                                 + 0.948508 w_k1_DIS_s107
                                 + 1.04746 w_k1_FDX_s107
                                 + 1.20352 w_k1_IBM_s107
                                 + 1.01498 w_k1_INTC_s107
                                 + 0.870131 w_k1_KO_s107
                                 + 1.10017 w_k1_MAT_s107
                                 + 0.831208 w_k1_MSFT_s107
                                 + 0.961444 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.929307 w_k1_ADM_s107
                                 + 0.677648 w_k1_AXP_s107
                                 + 0.551109 w_k1_DIS_s107
                                 + 0.877693 w_k1_FDX_s107
                                 + 0.608594 w_k1_IBM_s107
                                 + 0.886308 w_k1_INTC_s107
                                 + 0.626749 w_k1_KO_s107
                                 + 0.525557 w_k1_MAT_s107
                                 + 0.85298 w_k1_MSFT_s107
                                 + 0.841576 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.00312 w_k1_ADM_s107
                                 + 1.13691 w_k1_AXP_s107
                                 + 0.872054 w_k1_DIS_s107
                                 + 0.924525 w_k1_FDX_s107
                                 + 1.00111 w_k1_IBM_s107
                                 + 0.893097 w_k1_INTC_s107
                                 + 0.853344 w_k1_KO_s107
                                 + 0.852835 w_k1_MAT_s107
                                 + 0.950352 w_k1_MSFT_s107
                                 + 0.901233 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.888948 w_k1_ADM_s107
                                 + 0.815149 w_k1_AXP_s107
                                 + 0.970804 w_k1_DIS_s107
                                 + 1.00789 w_k1_FDX_s107
                                 + 0.846045 w_k1_IBM_s107
                                 + 1.00357 w_k1_INTC_s107
                                 + 0.786798 w_k1_KO_s107
                                 + 0.585152 w_k1_MAT_s107
                                 + 1.02107 w_k1_MSFT_s107
                                 + 0.914879 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.927521 w_k1_ADM_s107
                                 + 1.02496 w_k1_AXP_s107
                                 + 1.20209 w_k1_DIS_s107
                                 + 1.04393 w_k1_FDX_s107
                                 + 1.40098 w_k1_IBM_s107
                                 + 1.0969 w_k1_INTC_s107 + 1.07828 w_k1_KO_s107
                                 + 1.01863 w_k1_MAT_s107
                                 + 1.02967 w_k1_MSFT_s107
                                 + 1.05061 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.19454 w_k1_ADM_s110
                                 + 1.61151 w_k1_AXP_s110
                                 + 1.73635 w_k1_DIS_s110
                                 + 1.27801 w_k1_FDX_s110
                                 + 1.55309 w_k1_IBM_s110
                                 + 1.0389 w_k1_INTC_s110 + 1.64138 w_k1_KO_s110
                                 + 1.62734 w_k1_MAT_s110
                                 + 1.24232 w_k1_MSFT_s110
                                 + 1.18798 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.975168 w_k1_ADM_s110
                                 + 0.989618 w_k1_AXP_s110
                                 + 1.4129 w_k1_DIS_s110 + 1.11843 w_k1_FDX_s110
                                 + 0.94006 w_k1_IBM_s110
                                 + 1.04776 w_k1_INTC_s110
                                 + 1.24629 w_k1_KO_s110
                                 + 0.910295 w_k1_MAT_s110
                                 + 1.19828 w_k1_MSFT_s110
                                 + 1.05467 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.944297 w_k1_ADM_s110
                                 + 1.19899 w_k1_AXP_s110
                                 + 0.93108 w_k1_DIS_s110
                                 + 0.928643 w_k1_FDX_s110
                                 + 1.2275 w_k1_IBM_s110
                                 + 0.976866 w_k1_INTC_s110
                                 + 0.996137 w_k1_KO_s110 + 1.1899 w_k1_MAT_s110
                                 + 0.860244 w_k1_MSFT_s110
                                 + 0.983091 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.791377 w_k1_ADM_s110
                                 + 0.694321 w_k1_AXP_s110
                                 + 0.757309 w_k1_DIS_s110
                                 + 0.85366 w_k1_FDX_s110
                                 + 0.572374 w_k1_IBM_s110
                                 + 0.949989 w_k1_INTC_s110
                                 + 0.704425 w_k1_KO_s110
                                 + 0.438046 w_k1_MAT_s110
                                 + 0.816297 w_k1_MSFT_s110
                                 + 0.816449 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.29023 w_k1_ADM_s110
                                 + 1.51767 w_k1_AXP_s110
                                 + 1.22611 w_k1_DIS_s110
                                 + 1.24827 w_k1_FDX_s110
                                 + 0.727298 w_k1_IBM_s110
                                 + 1.04851 w_k1_INTC_s110
                                 + 1.09272 w_k1_KO_s110 + 1.01977 w_k1_MAT_s110
                                 + 1.13057 w_k1_MSFT_s110
                                 + 1.06765 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.943935 w_k1_ADM_s110
                                 + 0.977765 w_k1_AXP_s110
                                 + 0.828723 w_k1_DIS_s110
                                 + 0.979433 w_k1_FDX_s110
                                 + 0.777994 w_k1_IBM_s110
                                 + 1.03633 w_k1_INTC_s110
                                 + 0.942216 w_k1_KO_s110
                                 + 0.700741 w_k1_MAT_s110
                                 + 0.968542 w_k1_MSFT_s110
                                 + 0.946908 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.8402 w_k1_ADM_s110
                                 + 0.791202 w_k1_AXP_s110
                                 + 0.846592 w_k1_DIS_s110
                                 + 0.831444 w_k1_FDX_s110
                                 + 1.10344 w_k1_IBM_s110
                                 + 0.90788 w_k1_INTC_s110
                                 + 1.04871 w_k1_KO_s110
                                 + 0.748729 w_k1_MAT_s110
                                 + 1.00301 w_k1_MSFT_s110
                                 + 0.842676 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.7077 w_k1_ADM_s110
                                 + 0.756319 w_k1_AXP_s110
                                 + 0.580808 w_k1_DIS_s110
                                 + 0.728097 w_k1_FDX_s110
                                 + 0.826143 w_k1_IBM_s110
                                 + 0.737634 w_k1_INTC_s110
                                 + 0.710025 w_k1_KO_s110
                                 + 0.571267 w_k1_MAT_s110
                                 + 0.670196 w_k1_MSFT_s110
                                 + 0.839308 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.69838 w_k1_ADM_s111
                                 + 0.696926 w_k1_AXP_s111
                                 + 0.683728 w_k1_DIS_s111
                                 + 0.775668 w_k1_FDX_s111
                                 + 1.00954 w_k1_IBM_s111
                                 + 0.792142 w_k1_INTC_s111
                                 + 0.770166 w_k1_KO_s111
                                 + 1.02975 w_k1_MAT_s111
                                 + 0.872842 w_k1_MSFT_s111
                                 + 0.930238 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.985527 w_k1_ADM_s111
                                 + 0.999704 w_k1_AXP_s111
                                 + 0.96827 w_k1_DIS_s111
                                 + 0.928158 w_k1_FDX_s111
                                 + 1.0035 w_k1_IBM_s111
                                 + 1.00868 w_k1_INTC_s111
                                 + 0.919523 w_k1_KO_s111
                                 + 1.02356 w_k1_MAT_s111
                                 + 0.959188 w_k1_MSFT_s111
                                 + 0.991708 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.935011 w_k1_ADM_s111
                                 + 1.16035 w_k1_AXP_s111 + 1.2286 w_k1_DIS_s111
                                 + 1.11158 w_k1_FDX_s111
                                 + 1.26453 w_k1_IBM_s111
                                 + 1.1925 w_k1_INTC_s111
                                 + 0.980369 w_k1_KO_s111
                                 + 1.40435 w_k1_MAT_s111
                                 + 1.08088 w_k1_MSFT_s111
                                 + 1.12716 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.11168 w_k1_ADM_s111
                                 + 1.17304 w_k1_AXP_s111
                                 + 1.40373 w_k1_DIS_s111
                                 + 1.09146 w_k1_FDX_s111
                                 + 1.67579 w_k1_IBM_s111
                                 + 1.26982 w_k1_INTC_s111
                                 + 1.27091 w_k1_KO_s111 + 1.4527 w_k1_MAT_s111
                                 + 1.13978 w_k1_MSFT_s111
                                 + 1.18398 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.914663 w_k1_ADM_s111
                                 + 1.35214 w_k1_AXP_s111
                                 + 1.23331 w_k1_DIS_s111
                                 + 1.43554 w_k1_FDX_s111
                                 + 0.878087 w_k1_IBM_s111
                                 + 1.35875 w_k1_INTC_s111
                                 + 0.867117 w_k1_KO_s111 + 1.2256 w_k1_MAT_s111
                                 + 1.19191 w_k1_MSFT_s111
                                 + 1.21815 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.35156 w_k1_ADM_s111
                                 + 1.35208 w_k1_AXP_s111 + 1.6986 w_k1_DIS_s111
                                 + 1.41365 w_k1_FDX_s111
                                 + 1.15112 w_k1_IBM_s111
                                 + 1.2326 w_k1_INTC_s111 + 1.24776 w_k1_KO_s111
                                 + 1.12624 w_k1_MAT_s111
                                 + 1.41859 w_k1_MSFT_s111
                                 + 1.26596 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.80016 w_k1_ADM_s111
                                 + 0.890668 w_k1_AXP_s111
                                 + 0.916452 w_k1_DIS_s111
                                 + 1.03279 w_k1_FDX_s111
                                 + 0.74149 w_k1_IBM_s111
                                 + 1.10624 w_k1_INTC_s111
                                 + 0.753697 w_k1_KO_s111
                                 + 0.772951 w_k1_MAT_s111
                                 + 1.06127 w_k1_MSFT_s111
                                 + 1.02329 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.847582 w_k1_ADM_s111
                                 + 0.788744 w_k1_AXP_s111
                                 + 0.65925 w_k1_DIS_s111
                                 + 0.761806 w_k1_FDX_s111
                                 + 0.712462 w_k1_IBM_s111
                                 + 0.745254 w_k1_INTC_s111
                                 + 0.652833 w_k1_KO_s111
                                 + 0.75579 w_k1_MAT_s111
                                 + 0.812475 w_k1_MSFT_s111
                                 + 0.877778 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.732883 w_k1_ADM_s112
                                 + 0.927481 w_k1_AXP_s112
                                 + 0.700929 w_k1_DIS_s112
                                 + 0.797731 w_k1_FDX_s112
                                 + 1.13702 w_k1_IBM_s112
                                 + 0.935345 w_k1_INTC_s112
                                 + 0.865067 w_k1_KO_s112
                                 + 1.15131 w_k1_MAT_s112
                                 + 0.765331 w_k1_MSFT_s112
                                 + 0.885583 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.20452 w_k1_ADM_s112
                                 + 1.45326 w_k1_AXP_s112
                                 + 1.33416 w_k1_DIS_s112
                                 + 1.27945 w_k1_FDX_s112
                                 + 1.30134 w_k1_IBM_s112
                                 + 1.34031 w_k1_INTC_s112
                                 + 1.14713 w_k1_KO_s112 + 1.34464 w_k1_MAT_s112
                                 + 1.11711 w_k1_MSFT_s112
                                 + 1.21483 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.01349 w_k1_ADM_s112
                                 + 1.04634 w_k1_AXP_s112
                                 + 0.915532 w_k1_DIS_s112
                                 + 1.08274 w_k1_FDX_s112
                                 + 0.676116 w_k1_IBM_s112
                                 + 1.03921 w_k1_INTC_s112
                                 + 0.88809 w_k1_KO_s112
                                 + 0.714707 w_k1_MAT_s112
                                 + 1.06008 w_k1_MSFT_s112
                                 + 0.975651 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.53253 w_k1_ADM_s112
                                 + 1.46767 w_k1_AXP_s112
                                 + 1.50776 w_k1_DIS_s112
                                 + 1.41094 w_k1_FDX_s112
                                 + 0.873768 w_k1_IBM_s112
                                 + 1.16374 w_k1_INTC_s112
                                 + 1.04636 w_k1_KO_s112
                                 + 0.984293 w_k1_MAT_s112
                                 + 1.26464 w_k1_MSFT_s112
                                 + 1.18464 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.24733 w_k1_ADM_s112
                                 + 0.958973 w_k1_AXP_s112
                                 + 1.4757 w_k1_DIS_s112 + 1.17977 w_k1_FDX_s112
                                 + 0.900374 w_k1_IBM_s112
                                 + 1.15388 w_k1_INTC_s112
                                 + 1.15973 w_k1_KO_s112
                                 + 0.667615 w_k1_MAT_s112
                                 + 1.2226 w_k1_MSFT_s112 + 1.15649 w_k1_PG_s112
                                 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.905519 w_k1_ADM_s112
                                 + 0.823331 w_k1_AXP_s112
                                 + 0.66971 w_k1_DIS_s112
                                 + 0.809615 w_k1_FDX_s112
                                 + 0.661068 w_k1_IBM_s112
                                 + 0.844099 w_k1_INTC_s112
                                 + 0.668259 w_k1_KO_s112
                                 + 0.658949 w_k1_MAT_s112
                                 + 0.859366 w_k1_MSFT_s112
                                 + 0.870258 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.987067 w_k1_ADM_s112
                                 + 0.874122 w_k1_AXP_s112
                                 + 0.998249 w_k1_DIS_s112
                                 + 0.912668 w_k1_FDX_s112
                                 + 0.94639 w_k1_IBM_s112
                                 + 1.06051 w_k1_INTC_s112
                                 + 0.984253 w_k1_KO_s112
                                 + 0.93653 w_k1_MAT_s112
                                 + 1.16666 w_k1_MSFT_s112
                                 + 0.95753 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.01469 w_k1_ADM_s112
                                 + 1.12152 w_k1_AXP_s112
                                 + 1.04266 w_k1_DIS_s112
                                 + 0.953562 w_k1_FDX_s112
                                 + 1.43695 w_k1_IBM_s112
                                 + 1.12605 w_k1_INTC_s112
                                 + 1.19315 w_k1_KO_s112 + 1.62237 w_k1_MAT_s112
                                 + 1.11224 w_k1_MSFT_s112
                                 + 1.14242 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.927119 w_k1_ADM_s113
                                 + 1.0121 w_k1_AXP_s113 + 1.41446 w_k1_DIS_s113
                                 + 1.11557 w_k1_FDX_s113
                                 + 1.12106 w_k1_IBM_s113
                                 + 1.08323 w_k1_INTC_s113
                                 + 1.21187 w_k1_KO_s113 + 1.299 w_k1_MAT_s113
                                 + 1.12585 w_k1_MSFT_s113
                                 + 0.952582 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.38322 w_k1_ADM_s113
                                 + 1.26379 w_k1_AXP_s113
                                 + 1.54708 w_k1_DIS_s113
                                 + 1.37426 w_k1_FDX_s113
                                 + 0.969962 w_k1_IBM_s113
                                 + 1.57605 w_k1_INTC_s113
                                 + 1.17428 w_k1_KO_s113 + 1.05834 w_k1_MAT_s113
                                 + 1.38816 w_k1_MSFT_s113
                                 + 1.07119 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.865788 w_k1_ADM_s113
                                 + 0.791204 w_k1_AXP_s113
                                 + 0.945418 w_k1_DIS_s113
                                 + 0.857783 w_k1_FDX_s113
                                 + 1.24669 w_k1_IBM_s113
                                 + 0.924339 w_k1_INTC_s113
                                 + 1.02772 w_k1_KO_s113 + 1.07621 w_k1_MAT_s113
                                 + 0.798922 w_k1_MSFT_s113
                                 + 0.875559 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.13698 w_k1_ADM_s113
                                 + 1.07236 w_k1_AXP_s113
                                 + 1.02762 w_k1_DIS_s113
                                 + 1.10839 w_k1_FDX_s113
                                 + 0.860477 w_k1_IBM_s113
                                 + 1.13115 w_k1_INTC_s113
                                 + 0.972317 w_k1_KO_s113 + 1.0688 w_k1_MAT_s113
                                 + 0.953307 w_k1_MSFT_s113
                                 + 0.924678 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.15989 w_k1_ADM_s113
                                 + 1.34366 w_k1_AXP_s113
                                 + 1.14009 w_k1_DIS_s113
                                 + 1.16548 w_k1_FDX_s113
                                 + 1.45301 w_k1_IBM_s113
                                 + 1.36189 w_k1_INTC_s113
                                 + 1.23757 w_k1_KO_s113 + 1.5371 w_k1_MAT_s113
                                 + 1.09667 w_k1_MSFT_s113
                                 + 1.16046 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.875704 w_k1_ADM_s113
                                 + 0.761197 w_k1_AXP_s113
                                 + 0.762257 w_k1_DIS_s113
                                 + 0.775733 w_k1_FDX_s113
                                 + 0.816229 w_k1_IBM_s113
                                 + 0.90011 w_k1_INTC_s113
                                 + 0.924044 w_k1_KO_s113
                                 + 0.971248 w_k1_MAT_s113
                                 + 0.780776 w_k1_MSFT_s113
                                 + 0.80888 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.0326 w_k1_ADM_s113
                                 + 0.898925 w_k1_AXP_s113
                                 + 1.17716 w_k1_DIS_s113
                                 + 0.91781 w_k1_FDX_s113
                                 + 1.04674 w_k1_IBM_s113
                                 + 0.981656 w_k1_INTC_s113
                                 + 1.16589 w_k1_KO_s113 + 1.06322 w_k1_MAT_s113
                                 + 0.980499 w_k1_MSFT_s113
                                 + 0.973793 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.14556 w_k1_ADM_s113
                                 + 0.941768 w_k1_AXP_s113
                                 + 1.05379 w_k1_DIS_s113
                                 + 1.01704 w_k1_FDX_s113 + 0.7103 w_k1_IBM_s113
                                 + 1.09097 w_k1_INTC_s113
                                 + 1.00873 w_k1_KO_s113
                                 + 0.951904 w_k1_MAT_s113
                                 + 1.0488 w_k1_MSFT_s113 + 0.95807 w_k1_PG_s113
                                 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.936485 w_k1_ADM_s114
                                 + 1.10422 w_k1_AXP_s114
                                 + 0.977776 w_k1_DIS_s114
                                 + 0.932566 w_k1_FDX_s114
                                 + 1.21694 w_k1_IBM_s114
                                 + 1.14112 w_k1_INTC_s114
                                 + 0.989184 w_k1_KO_s114
                                 + 1.07388 w_k1_MAT_s114
                                 + 1.10708 w_k1_MSFT_s114
                                 + 1.10912 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.54397 w_k1_ADM_s114
                                 + 1.42771 w_k1_AXP_s114
                                 + 1.63056 w_k1_DIS_s114
                                 + 1.37765 w_k1_FDX_s114
                                 + 1.33737 w_k1_IBM_s114
                                 + 1.44336 w_k1_INTC_s114
                                 + 1.58987 w_k1_KO_s114 + 1.35871 w_k1_MAT_s114
                                 + 1.5498 w_k1_MSFT_s114 + 1.40853 w_k1_PG_s114
                                 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.16787 w_k1_ADM_s114
                                 + 1.18675 w_k1_AXP_s114
                                 + 1.16168 w_k1_DIS_s114
                                 + 1.18463 w_k1_FDX_s114
                                 + 0.93115 w_k1_IBM_s114
                                 + 1.11949 w_k1_INTC_s114
                                 + 1.03908 w_k1_KO_s114
                                 + 0.833031 w_k1_MAT_s114
                                 + 1.28745 w_k1_MSFT_s114
                                 + 1.03967 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.698006 w_k1_ADM_s114
                                 + 0.738387 w_k1_AXP_s114
                                 + 0.716018 w_k1_DIS_s114
                                 + 0.729147 w_k1_FDX_s114
                                 + 0.844965 w_k1_IBM_s114
                                 + 0.982594 w_k1_INTC_s114
                                 + 0.796943 w_k1_KO_s114
                                 + 0.870317 w_k1_MAT_s114
                                 + 0.855245 w_k1_MSFT_s114
                                 + 0.856779 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.827683 w_k1_ADM_s114
                                 + 0.800442 w_k1_AXP_s114
                                 + 0.939307 w_k1_DIS_s114
                                 + 0.885938 w_k1_FDX_s114
                                 + 0.919274 w_k1_IBM_s114
                                 + 0.926967 w_k1_INTC_s114
                                 + 0.83974 w_k1_KO_s114
                                 + 0.738263 w_k1_MAT_s114
                                 + 0.940185 w_k1_MSFT_s114
                                 + 0.945828 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.11713 w_k1_ADM_s114
                                 + 1.2025 w_k1_AXP_s114 + 1.15549 w_k1_DIS_s114
                                 + 1.23791 w_k1_FDX_s114
                                 + 0.716047 w_k1_IBM_s114
                                 + 1.3387 w_k1_INTC_s114
                                 + 0.852185 w_k1_KO_s114
                                 + 0.655128 w_k1_MAT_s114
                                 + 1.22623 w_k1_MSFT_s114
                                 + 0.91536 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.841703 w_k1_ADM_s114
                                 + 0.951254 w_k1_AXP_s114
                                 + 0.79621 w_k1_DIS_s114
                                 + 0.959426 w_k1_FDX_s114
                                 + 0.695274 w_k1_IBM_s114
                                 + 0.981061 w_k1_INTC_s114
                                 + 0.799401 w_k1_KO_s114 + 0.9601 w_k1_MAT_s114
                                 + 0.904698 w_k1_MSFT_s114
                                 + 0.921486 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.861087 w_k1_ADM_s114
                                 + 1.12087 w_k1_AXP_s114
                                 + 1.25108 w_k1_DIS_s114
                                 + 0.939666 w_k1_FDX_s114
                                 + 1.56555 w_k1_IBM_s114
                                 + 1.20293 w_k1_INTC_s114
                                 + 1.21803 w_k1_KO_s114 + 1.43613 w_k1_MAT_s114
                                 + 1.18326 w_k1_MSFT_s114
                                 + 1.12335 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.60149 w_k1_ADM_s115
                                 + 1.7974 w_k1_AXP_s115 + 1.7674 w_k1_DIS_s115
                                 + 1.52041 w_k1_FDX_s115
                                 + 1.31032 w_k1_IBM_s115
                                 + 1.34953 w_k1_INTC_s115
                                 + 1.33046 w_k1_KO_s115 + 1.79018 w_k1_MAT_s115
                                 + 1.41676 w_k1_MSFT_s115
                                 + 1.28331 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.11112 w_k1_ADM_s115
                                 + 1.14073 w_k1_AXP_s115
                                 + 0.996051 w_k1_DIS_s115
                                 + 1.01965 w_k1_FDX_s115
                                 + 1.23181 w_k1_IBM_s115
                                 + 1.05807 w_k1_INTC_s115 + 1.0708 w_k1_KO_s115
                                 + 1.05792 w_k1_MAT_s115
                                 + 1.04193 w_k1_MSFT_s115
                                 + 0.989552 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.704747 w_k1_ADM_s115
                                 + 0.804574 w_k1_AXP_s115
                                 + 0.756397 w_k1_DIS_s115
                                 + 0.816566 w_k1_FDX_s115
                                 + 1.14062 w_k1_IBM_s115
                                 + 1.04342 w_k1_INTC_s115
                                 + 0.924224 w_k1_KO_s115
                                 + 1.04066 w_k1_MAT_s115
                                 + 0.882706 w_k1_MSFT_s115
                                 + 0.930895 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.874778 w_k1_ADM_s115
                                 + 0.725805 w_k1_AXP_s115
                                 + 0.664385 w_k1_DIS_s115
                                 + 0.704477 w_k1_FDX_s115
                                 + 0.882639 w_k1_IBM_s115
                                 + 0.772747 w_k1_INTC_s115
                                 + 0.688874 w_k1_KO_s115
                                 + 0.703604 w_k1_MAT_s115
                                 + 0.781213 w_k1_MSFT_s115
                                 + 0.774103 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.15293 w_k1_ADM_s115
                                 + 1.04375 w_k1_AXP_s115
                                 + 1.00876 w_k1_DIS_s115
                                 + 0.883974 w_k1_FDX_s115
                                 + 1.31731 w_k1_IBM_s115
                                 + 0.929242 w_k1_INTC_s115
                                 + 1.09207 w_k1_KO_s115 + 1.17701 w_k1_MAT_s115
                                 + 0.946053 w_k1_MSFT_s115
                                 + 1.03038 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.51419 w_k1_ADM_s115
                                 + 1.29883 w_k1_AXP_s115
                                 + 1.29858 w_k1_DIS_s115
                                 + 1.34676 w_k1_FDX_s115
                                 + 0.832628 w_k1_IBM_s115
                                 + 1.14513 w_k1_INTC_s115
                                 + 0.921153 w_k1_KO_s115
                                 + 0.917227 w_k1_MAT_s115
                                 + 1.30343 w_k1_MSFT_s115
                                 + 1.03402 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.42103 w_k1_ADM_s115
                                 + 1.44107 w_k1_AXP_s115
                                 + 1.56067 w_k1_DIS_s115
                                 + 1.24402 w_k1_FDX_s115
                                 + 1.21378 w_k1_IBM_s115
                                 + 1.37624 w_k1_INTC_s115
                                 + 1.18971 w_k1_KO_s115 + 1.38714 w_k1_MAT_s115
                                 + 1.24173 w_k1_MSFT_s115
                                 + 1.16229 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.18418 w_k1_ADM_s115
                                 + 0.921527 w_k1_AXP_s115
                                 + 1.02283 w_k1_DIS_s115
                                 + 1.02677 w_k1_FDX_s115
                                 + 0.818504 w_k1_IBM_s115
                                 + 1.13343 w_k1_INTC_s115
                                 + 0.999263 w_k1_KO_s115
                                 + 0.764668 w_k1_MAT_s115
                                 + 1.16505 w_k1_MSFT_s115
                                 + 1.04822 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.93949 w_k1_ADM_s116
                                 + 0.74431 w_k1_AXP_s116
                                 + 0.794288 w_k1_DIS_s116
                                 + 0.990108 w_k1_FDX_s116
                                 + 0.858959 w_k1_IBM_s116
                                 + 0.928503 w_k1_INTC_s116
                                 + 0.92199 w_k1_KO_s116
                                 + 0.824858 w_k1_MAT_s116
                                 + 1.04576 w_k1_MSFT_s116
                                 + 0.998277 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.969591 w_k1_ADM_s116
                                 + 0.789178 w_k1_AXP_s116
                                 + 0.659788 w_k1_DIS_s116
                                 + 0.809612 w_k1_FDX_s116
                                 + 0.88994 w_k1_IBM_s116
                                 + 0.717808 w_k1_INTC_s116
                                 + 0.83799 w_k1_KO_s116
                                 + 0.918719 w_k1_MAT_s116
                                 + 0.847191 w_k1_MSFT_s116
                                 + 0.907113 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.46405 w_k1_ADM_s116
                                 + 1.36773 w_k1_AXP_s116
                                 + 1.20704 w_k1_DIS_s116
                                 + 1.24812 w_k1_FDX_s116
                                 + 1.08838 w_k1_IBM_s116
                                 + 1.14112 w_k1_INTC_s116
                                 + 1.10534 w_k1_KO_s116
                                 + 0.945776 w_k1_MAT_s116
                                 + 1.10664 w_k1_MSFT_s116 + 1.1025 w_k1_PG_s116
                                 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.882916 w_k1_ADM_s116
                                 + 0.61 w_k1_AXP_s116 + 0.760612 w_k1_DIS_s116
                                 + 0.893593 w_k1_FDX_s116
                                 + 0.723003 w_k1_IBM_s116
                                 + 0.902227 w_k1_INTC_s116
                                 + 0.730207 w_k1_KO_s116
                                 + 0.528372 w_k1_MAT_s116
                                 + 0.913363 w_k1_MSFT_s116
                                 + 0.922921 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.35012 w_k1_ADM_s116
                                 + 1.11665 w_k1_AXP_s116
                                 + 1.48486 w_k1_DIS_s116
                                 + 1.23075 w_k1_FDX_s116
                                 + 1.10584 w_k1_IBM_s116
                                 + 1.27567 w_k1_INTC_s116 + 1.2676 w_k1_KO_s116
                                 + 0.790113 w_k1_MAT_s116
                                 + 1.2356 w_k1_MSFT_s116 + 1.22351 w_k1_PG_s116
                                 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.461 w_k1_ADM_s116
                                 + 1.36881 w_k1_AXP_s116
                                 + 1.67559 w_k1_DIS_s116
                                 + 1.32425 w_k1_FDX_s116
                                 + 1.58701 w_k1_IBM_s116
                                 + 1.00278 w_k1_INTC_s116
                                 + 1.50056 w_k1_KO_s116 + 1.80604 w_k1_MAT_s116
                                 + 1.23073 w_k1_MSFT_s116 + 1.3125 w_k1_PG_s116
                                 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.98463 w_k1_ADM_s116
                                 + 0.918691 w_k1_AXP_s116
                                 + 0.750027 w_k1_DIS_s116
                                 + 0.889192 w_k1_FDX_s116
                                 + 1.25458 w_k1_IBM_s116
                                 + 0.919794 w_k1_INTC_s116
                                 + 0.94825 w_k1_KO_s116
                                 + 0.899893 w_k1_MAT_s116
                                 + 0.807089 w_k1_MSFT_s116
                                 + 0.981915 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.07903 w_k1_ADM_s116
                                 + 0.775585 w_k1_AXP_s116
                                 + 1.11358 w_k1_DIS_s116
                                 + 1.04008 w_k1_FDX_s116
                                 + 1.04644 w_k1_IBM_s116
                                 + 0.943959 w_k1_INTC_s116
                                 + 1.06904 w_k1_KO_s116
                                 + 0.728821 w_k1_MAT_s116
                                 + 1.02905 w_k1_MSFT_s116
                                 + 1.08562 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.35594 w_k1_ADM_s117
                                 + 1.41464 w_k1_AXP_s117
                                 + 1.19388 w_k1_DIS_s117
                                 + 1.23572 w_k1_FDX_s117
                                 + 1.07685 w_k1_IBM_s117
                                 + 1.09614 w_k1_INTC_s117
                                 + 1.29127 w_k1_KO_s117 + 1.13238 w_k1_MAT_s117
                                 + 1.30048 w_k1_MSFT_s117
                                 + 1.22567 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.10606 w_k1_ADM_s117
                                 + 0.899432 w_k1_AXP_s117
                                 + 0.769275 w_k1_DIS_s117
                                 + 0.914369 w_k1_FDX_s117
                                 + 0.769748 w_k1_IBM_s117
                                 + 0.981227 w_k1_INTC_s117
                                 + 0.824353 w_k1_KO_s117
                                 + 0.586109 w_k1_MAT_s117
                                 + 0.923156 w_k1_MSFT_s117
                                 + 0.987829 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.08413 w_k1_ADM_s117
                                 + 1.33931 w_k1_AXP_s117
                                 + 1.58938 w_k1_DIS_s117
                                 + 1.26606 w_k1_FDX_s117
                                 + 0.869722 w_k1_IBM_s117
                                 + 0.92938 w_k1_INTC_s117
                                 + 1.25553 w_k1_KO_s117 + 1.63012 w_k1_MAT_s117
                                 + 1.26207 w_k1_MSFT_s117
                                 + 1.20241 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.797111 w_k1_ADM_s117
                                 + 1.10914 w_k1_AXP_s117
                                 + 1.07418 w_k1_DIS_s117
                                 + 1.01931 w_k1_FDX_s117
                                 + 1.21916 w_k1_IBM_s117
                                 + 0.981745 w_k1_INTC_s117
                                 + 1.29538 w_k1_KO_s117 + 1.79605 w_k1_MAT_s117
                                 + 1.13367 w_k1_MSFT_s117
                                 + 1.05838 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.54591 w_k1_ADM_s117
                                 + 1.65687 w_k1_AXP_s117
                                 + 1.77437 w_k1_DIS_s117
                                 + 1.61268 w_k1_FDX_s117
                                 + 1.13412 w_k1_IBM_s117
                                 + 1.43329 w_k1_INTC_s117
                                 + 1.29479 w_k1_KO_s117 + 1.35056 w_k1_MAT_s117
                                 + 1.56714 w_k1_MSFT_s117
                                 + 1.15264 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.773529 w_k1_ADM_s117
                                 + 0.954199 w_k1_AXP_s117
                                 + 0.891397 w_k1_DIS_s117
                                 + 0.791663 w_k1_FDX_s117
                                 + 1.49384 w_k1_IBM_s117
                                 + 0.979335 w_k1_INTC_s117
                                 + 1.09634 w_k1_KO_s117 + 1.45724 w_k1_MAT_s117
                                 + 0.969896 w_k1_MSFT_s117
                                 + 0.950359 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.824957 w_k1_ADM_s117
                                 + 0.728797 w_k1_AXP_s117
                                 + 0.781087 w_k1_DIS_s117
                                 + 0.855287 w_k1_FDX_s117
                                 + 0.790142 w_k1_IBM_s117
                                 + 0.93558 w_k1_INTC_s117
                                 + 0.897206 w_k1_KO_s117
                                 + 0.666283 w_k1_MAT_s117
                                 + 1.03729 w_k1_MSFT_s117
                                 + 0.933966 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.749768 w_k1_ADM_s117
                                 + 0.693646 w_k1_AXP_s117
                                 + 0.504779 w_k1_DIS_s117
                                 + 0.64748 w_k1_FDX_s117
                                 + 0.922699 w_k1_IBM_s117
                                 + 0.669366 w_k1_INTC_s117
                                 + 0.796833 w_k1_KO_s117
                                 + 0.848857 w_k1_MAT_s117
                                 + 0.724686 w_k1_MSFT_s117
                                 + 0.853282 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0191321532195833 w_k1_ADM_s000
                                 + 0.0404277613070018 w_k1_ADM_s001
                                 + 0.0231032335664717 w_k1_ADM_s002
                                 + 0.0102931728509209 w_k1_ADM_s003
                                 + 0.0122896113276927 w_k1_ADM_s004
                                 + 0.0346232467184463 w_k1_ADM_s005
                                 + 0.0130431580362279 w_k1_ADM_s006
                                 + 0.0415268803253959 w_k1_ADM_s007
                                 + 0.0120178541941375 w_k1_ADM_s010
                                 + 0.00395353745918544 w_k1_ADM_s011
                                 + 0.00462841935449234 w_k1_ADM_s012
                                 + 0.0105728175984512 w_k1_ADM_s013
                                 + 0.00953551002682115 w_k1_ADM_s014
                                 + 0.00802761561823998 w_k1_ADM_s015
                                 + 0.00396615057677095 w_k1_ADM_s016
                                 + 0.00548391175331119 w_k1_ADM_s017
                                 + 0.0430359079264174 w_k1_ADM_s100
                                 + 0.017835692203774 w_k1_ADM_s101
                                 + 0.0394049080251591 w_k1_ADM_s102
                                 + 0.0325997454533364 w_k1_ADM_s103
                                 + 0.0128037842882951 w_k1_ADM_s104
                                 + 0.046760769515253 w_k1_ADM_s105
                                 + 0.0287397714763286 w_k1_ADM_s106
                                 + 0.0282518133350669 w_k1_ADM_s107
                                 + 0.0339039126267801 w_k1_ADM_s110
                                 + 0.0554065752622267 w_k1_ADM_s111
                                 + 0.170782628400362 w_k1_ADM_s112
                                 + 0.0435940686094125 w_k1_ADM_s113
                                 + 0.0646849792043797 w_k1_ADM_s114
                                 + 0.0564843321366881 w_k1_ADM_s115
                                 + 0.0638038928725948 w_k1_ADM_s116
                                 + 0.0373032390173093 w_k1_ADM_s117
                                 + 0.0215483920459157 w_k1_AXP_s000
                                 + 0.0333846988022765 w_k1_AXP_s001
                                 + 0.0270400840150253 w_k1_AXP_s002
                                 + 0.00932556003516264 w_k1_AXP_s003
                                 + 0.0117087483657497 w_k1_AXP_s004
                                 + 0.0348118780242187 w_k1_AXP_s005
                                 + 0.0157776717020019 w_k1_AXP_s006
                                 + 0.0396718241914372 w_k1_AXP_s007
                                 + 0.0117096123596143 w_k1_AXP_s010
                                 + 0.00446954671447887 w_k1_AXP_s011
                                 + 0.00501499192706577 w_k1_AXP_s012
                                 + 0.00895133999626007 w_k1_AXP_s013
                                 + 0.00972741905145763 w_k1_AXP_s014
                                 + 0.00720146292249433 w_k1_AXP_s015
                                 + 0.00360635698752464 w_k1_AXP_s016
                                 + 0.00510515418261995 w_k1_AXP_s017
                                 + 0.0430875904234977 w_k1_AXP_s100
                                 + 0.0172808048586324 w_k1_AXP_s101
                                 + 0.0362406894734087 w_k1_AXP_s102
                                 + 0.0324763577836987 w_k1_AXP_s103
                                 + 0.0114992826632168 w_k1_AXP_s104
                                 + 0.0443430259627726 w_k1_AXP_s105
                                 + 0.0242490049066773 w_k1_AXP_s106
                                 + 0.0305612758500918 w_k1_AXP_s107
                                 + 0.0367384093959384 w_k1_AXP_s110
                                 + 0.0598591125872492 w_k1_AXP_s111
                                 + 0.178138689746124 w_k1_AXP_s112
                                 + 0.042072347777134 w_k1_AXP_s113
                                 + 0.0698843485517583 w_k1_AXP_s114
                                 + 0.0524409308400861 w_k1_AXP_s115
                                 + 0.0535127047374888 w_k1_AXP_s116
                                 + 0.0396772386494477 w_k1_AXP_s117
                                 + 0.0225085834840998 w_k1_DIS_s000
                                 + 0.0361604398620905 w_k1_DIS_s001
                                 + 0.0256357452552806 w_k1_DIS_s002
                                 + 0.0107022170879506 w_k1_DIS_s003
                                 + 0.0122473014988111 w_k1_DIS_s004
                                 + 0.0385647342741577 w_k1_DIS_s005
                                 + 0.0154952274496552 w_k1_DIS_s006
                                 + 0.0377038572753278 w_k1_DIS_s007
                                 + 0.0113344129997547 w_k1_DIS_s010
                                 + 0.00449067916050113 w_k1_DIS_s011
                                 + 0.00537264019617924 w_k1_DIS_s012
                                 + 0.00849767587386763 w_k1_DIS_s013
                                 + 0.00965358029074399 w_k1_DIS_s014
                                 + 0.00692574759430394 w_k1_DIS_s015
                                 + 0.00359707600890576 w_k1_DIS_s016
                                 + 0.00535169639219849 w_k1_DIS_s017
                                 + 0.0460620384146204 w_k1_DIS_s100
                                 + 0.0160461591096345 w_k1_DIS_s101
                                 + 0.039357579330564 w_k1_DIS_s102
                                 + 0.0340852346038637 w_k1_DIS_s103
                                 + 0.0120869577535258 w_k1_DIS_s104
                                 + 0.0432317503637013 w_k1_DIS_s105
                                 + 0.0249795176283357 w_k1_DIS_s106
                                 + 0.0306089323494928 w_k1_DIS_s107
                                 + 0.0379656200608912 w_k1_DIS_s110
                                 + 0.0616486223670139 w_k1_DIS_s111
                                 + 0.166982548232623 w_k1_DIS_s112
                                 + 0.0444753237129089 w_k1_DIS_s113
                                 + 0.0710282046246152 w_k1_DIS_s114
                                 + 0.0518216475522686 w_k1_DIS_s115
                                 + 0.0577759525401303 w_k1_DIS_s116
                                 + 0.0375527119561677 w_k1_DIS_s117
                                 + 0.0215388962495656 w_k1_FDX_s000
                                 + 0.0349024804601953 w_k1_FDX_s001
                                 + 0.0242328291883815 w_k1_FDX_s002
                                 + 0.00997529068216184 w_k1_FDX_s003
                                 + 0.0122029511534411 w_k1_FDX_s004
                                 + 0.0355683459332968 w_k1_FDX_s005
                                 + 0.015725246686179 w_k1_FDX_s006
                                 + 0.0383473641791965 w_k1_FDX_s007
                                 + 0.0108267925980598 w_k1_FDX_s010
                                 + 0.00423520107103537 w_k1_FDX_s011
                                 + 0.00472600285082023 w_k1_FDX_s012
                                 + 0.0089720534203581 w_k1_FDX_s013
                                 + 0.0101091918686298 w_k1_FDX_s014
                                 + 0.00707360700537736 w_k1_FDX_s015
                                 + 0.0038488833323638 w_k1_FDX_s016
                                 + 0.00511231154189112 w_k1_FDX_s017
                                 + 0.045513569944962 w_k1_FDX_s100
                                 + 0.0155630088483658 w_k1_FDX_s101
                                 + 0.0390514833538913 w_k1_FDX_s102
                                 + 0.0331646979443873 w_k1_FDX_s103
                                 + 0.0127446807468931 w_k1_FDX_s104
                                 + 0.0441216308641659 w_k1_FDX_s105
                                 + 0.0259728436180235 w_k1_FDX_s106
                                 + 0.0308490015438437 w_k1_FDX_s107
                                 + 0.0356147089468864 w_k1_FDX_s110
                                 + 0.0611820719596402 w_k1_FDX_s111
                                 + 0.167962459791068 w_k1_FDX_s112
                                 + 0.0422452174540545 w_k1_FDX_s113
                                 + 0.0669442254053631 w_k1_FDX_s114
                                 + 0.0497548135172544 w_k1_FDX_s115
                                 + 0.0589080288099628 w_k1_FDX_s116
                                 + 0.0375244876672679 w_k1_FDX_s117
                                 + 0.0220771971030633 w_k1_IBM_s000
                                 + 0.0346274736321076 w_k1_IBM_s001
                                 + 0.0242045796755106 w_k1_IBM_s002
                                 + 0.00968297449637029 w_k1_IBM_s003
                                 + 0.0124233729200799 w_k1_IBM_s004
                                 + 0.0369237017796005 w_k1_IBM_s005
                                 + 0.0160229381303239 w_k1_IBM_s006
                                 + 0.0402391113003127 w_k1_IBM_s007
                                 + 0.0104830208497595 w_k1_IBM_s010
                                 + 0.00449951909301055 w_k1_IBM_s011
                                 + 0.00445920274008917 w_k1_IBM_s012
                                 + 0.00925810759771837 w_k1_IBM_s013
                                 + 0.00970519340243968 w_k1_IBM_s014
                                 + 0.00730028149227125 w_k1_IBM_s015
                                 + 0.00378271736743751 w_k1_IBM_s016
                                 + 0.00494156374893141 w_k1_IBM_s017
                                 + 0.0465535794274382 w_k1_IBM_s100
                                 + 0.0175168019629505 w_k1_IBM_s101
                                 + 0.038927730460837 w_k1_IBM_s102
                                 + 0.0341665553019693 w_k1_IBM_s103
                                 + 0.0123785580820353 w_k1_IBM_s104
                                 + 0.0429029105056015 w_k1_IBM_s105
                                 + 0.0256726116943485 w_k1_IBM_s106
                                 + 0.0291597716681468 w_k1_IBM_s107
                                 + 0.0335015244646458 w_k1_IBM_s110
                                 + 0.0636561298846823 w_k1_IBM_s111
                                 + 0.171502393150966 w_k1_IBM_s112
                                 + 0.0429981493480048 w_k1_IBM_s113
                                 + 0.0706785541547047 w_k1_IBM_s114
                                 + 0.0520572238158173 w_k1_IBM_s115
                                 + 0.0587640208515166 w_k1_IBM_s116
                                 + 0.0378422459935716 w_k1_IBM_s117
                                 + 0.0238441725302772 w_k1_INTC_s000
                                 + 0.0338702709391554 w_k1_INTC_s001
                                 + 0.0240405145730308 w_k1_INTC_s002
                                 + 0.00912765776809933 w_k1_INTC_s003
                                 + 0.010855579303527 w_k1_INTC_s004
                                 + 0.0380891694536805 w_k1_INTC_s005
                                 + 0.0143990088930427 w_k1_INTC_s006
                                 + 0.0405910643586815 w_k1_INTC_s007
                                 + 0.00975730952965874 w_k1_INTC_s010
                                 + 0.00593083850832693 w_k1_INTC_s011
                                 + 0.00427259154677359 w_k1_INTC_s012
                                 + 0.00774552540927896 w_k1_INTC_s013
                                 + 0.00814451852875033 w_k1_INTC_s014
                                 + 0.00813362590890107 w_k1_INTC_s015
                                 + 0.00374118710973362 w_k1_INTC_s016
                                 + 0.00516800477929688 w_k1_INTC_s017
                                 + 0.0453776778028616 w_k1_INTC_s100
                                 + 0.0168467851921914 w_k1_INTC_s101
                                 + 0.0345734340252318 w_k1_INTC_s102
                                 + 0.0326040770357735 w_k1_INTC_s103
                                 + 0.0108258030600108 w_k1_INTC_s104
                                 + 0.0432110521269944 w_k1_INTC_s105
                                 + 0.0228135465999508 w_k1_INTC_s106
                                 + 0.0303412236383285 w_k1_INTC_s107
                                 + 0.0346905856909912 w_k1_INTC_s110
                                 + 0.0641625189204735 w_k1_INTC_s111
                                 + 0.177908831620644 w_k1_INTC_s112
                                 + 0.0464947310692583 w_k1_INTC_s113
                                 + 0.0743531617710736 w_k1_INTC_s114
                                 + 0.0524627054877061 w_k1_INTC_s115
                                 + 0.0554528746403886 w_k1_INTC_s116
                                 + 0.0360911086364402 w_k1_INTC_s117
                                 + 0.0197028990099837 w_k1_KO_s000
                                 + 0.0346131516069273 w_k1_KO_s001
                                 + 0.0237454864934993 w_k1_KO_s002
                                 + 0.00966148521187648 w_k1_KO_s003
                                 + 0.0123944674971325 w_k1_KO_s004
                                 + 0.0355607077646567 w_k1_KO_s005
                                 + 0.015789556103419 w_k1_KO_s006
                                 + 0.0407044056407345 w_k1_KO_s007
                                 + 0.0105257971119711 w_k1_KO_s010
                                 + 0.00492848822679498 w_k1_KO_s011
                                 + 0.00429085190733452 w_k1_KO_s012
                                 + 0.00843809788275044 w_k1_KO_s013
                                 + 0.00933679647104188 w_k1_KO_s014
                                 + 0.00763630525957346 w_k1_KO_s015
                                 + 0.00362705725942726 w_k1_KO_s016
                                 + 0.00464214838587592 w_k1_KO_s017
                                 + 0.0471391976437882 w_k1_KO_s100
                                 + 0.0156162267670994 w_k1_KO_s101
                                 + 0.0385873160844637 w_k1_KO_s102
                                 + 0.0337411064164325 w_k1_KO_s103
                                 + 0.0108922615646967 w_k1_KO_s104
                                 + 0.0422255263130672 w_k1_KO_s105
                                 + 0.0226680944940394 w_k1_KO_s106
                                 + 0.0265081801022897 w_k1_KO_s107
                                 + 0.0377081021870151 w_k1_KO_s110
                                 + 0.0547990047070258 w_k1_KO_s111
                                 + 0.163046489456975 w_k1_KO_s112
                                 + 0.044218787345345 w_k1_KO_s113
                                 + 0.0673904513274367 w_k1_KO_s114
                                 + 0.048923298470402 w_k1_KO_s115
                                 + 0.0574217772892852 w_k1_KO_s116
                                 + 0.0402223222722501 w_k1_KO_s117
                                 + 0.0245624652773809 w_k1_MAT_s000
                                 + 0.0258721935732889 w_k1_MAT_s001
                                 + 0.0262133887431646 w_k1_MAT_s002
                                 + 0.0119994655056168 w_k1_MAT_s003
                                 + 0.0131879628491519 w_k1_MAT_s004
                                 + 0.0328625469536239 w_k1_MAT_s005
                                 + 0.0228767450448112 w_k1_MAT_s006
                                 + 0.0379625996490746 w_k1_MAT_s007
                                 + 0.0099770464921872 w_k1_MAT_s010
                                 + 0.00506473564424746 w_k1_MAT_s011
                                 + 0.00461881140367306 w_k1_MAT_s012
                                 + 0.00903566851196163 w_k1_MAT_s013
                                 + 0.00883050145222017 w_k1_MAT_s014
                                 + 0.00687929292830605 w_k1_MAT_s015
                                 + 0.00342522833259361 w_k1_MAT_s016
                                 + 0.00417530948397099 w_k1_MAT_s017
                                 + 0.0493085249738027 w_k1_MAT_s100
                                 + 0.0186685274373195 w_k1_MAT_s101
                                 + 0.0471026050346665 w_k1_MAT_s102
                                 + 0.0319498120394843 w_k1_MAT_s103
                                 + 0.010456242992517 w_k1_MAT_s104
                                 + 0.0400435791459088 w_k1_MAT_s105
                                 + 0.034212022121526 w_k1_MAT_s106
                                 + 0.0265904014281799 w_k1_MAT_s107
                                 + 0.0310358745968188 w_k1_MAT_s110
                                 + 0.0654143150313121 w_k1_MAT_s111
                                 + 0.178581838095035 w_k1_MAT_s112
                                 + 0.0470569189622186 w_k1_MAT_s113
                                 + 0.0674577009397323 w_k1_MAT_s114
                                 + 0.0498652285609339 w_k1_MAT_s115
                                 + 0.0502384957399631 w_k1_MAT_s116
                                 + 0.0428162173605214 w_k1_MAT_s117
                                 + 0.0227690083438535 w_k1_MSFT_s000
                                 + 0.0346415260223101 w_k1_MSFT_s001
                                 + 0.0236040876888324 w_k1_MSFT_s002
                                 + 0.0104291682725585 w_k1_MSFT_s003
                                 + 0.0118157072310976 w_k1_MSFT_s004
                                 + 0.0359396003871478 w_k1_MSFT_s005
                                 + 0.0150539198828048 w_k1_MSFT_s006
                                 + 0.0384276197076868 w_k1_MSFT_s007
                                 + 0.0104212246022651 w_k1_MSFT_s010
                                 + 0.00484436581380731 w_k1_MSFT_s011
                                 + 0.00452417093098173 w_k1_MSFT_s012
                                 + 0.00813356569233331 w_k1_MSFT_s013
                                 + 0.00930536546734134 w_k1_MSFT_s014
                                 + 0.00738909315305354 w_k1_MSFT_s015
                                 + 0.00381165986890297 w_k1_MSFT_s016
                                 + 0.00499617992194236 w_k1_MSFT_s017
                                 + 0.0444472190522869 w_k1_MSFT_s100
                                 + 0.0157388639993132 w_k1_MSFT_s101
                                 + 0.042195988698739 w_k1_MSFT_s102
                                 + 0.033124456446794 w_k1_MSFT_s103
                                 + 0.0108608957163597 w_k1_MSFT_s104
                                 + 0.0421224869668737 w_k1_MSFT_s105
                                 + 0.02492283904416 w_k1_MSFT_s106
                                 + 0.0301100398654339 w_k1_MSFT_s107
                                 + 0.0359382329228757 w_k1_MSFT_s110
                                 + 0.0627593410062877 w_k1_MSFT_s111
                                 + 0.17267211231831 w_k1_MSFT_s112
                                 + 0.0407864383743955 w_k1_MSFT_s113
                                 + 0.0738030405795388 w_k1_MSFT_s114
                                 + 0.0521978304888133 w_k1_MSFT_s115
                                 + 0.0573886581513914 w_k1_MSFT_s116
                                 + 0.0404821993774967 w_k1_MSFT_s117
                                 + 0.0218510086447233 w_k1_PG_s000
                                 + 0.0358467238574983 w_k1_PG_s001
                                 + 0.0233537068634092 w_k1_PG_s002
                                 + 0.00909943608648362 w_k1_PG_s003
                                 + 0.0126676862536217 w_k1_PG_s004
                                 + 0.0386180151400671 w_k1_PG_s005
                                 + 0.0143917069392241 w_k1_PG_s006
                                 + 0.0394709881896834 w_k1_PG_s007
                                 + 0.0104304110197894 w_k1_PG_s010
                                 + 0.00422641071803673 w_k1_PG_s011
                                 + 0.00446003110633571 w_k1_PG_s012
                                 + 0.00834173240783779 w_k1_PG_s013
                                 + 0.00956516292624495 w_k1_PG_s014
                                 + 0.0067897844934384 w_k1_PG_s015
                                 + 0.00361867057725151 w_k1_PG_s016
                                 + 0.00486513906239725 w_k1_PG_s017
                                 + 0.0419780338213818 w_k1_PG_s100
                                 + 0.0153827691405735 w_k1_PG_s101
                                 + 0.0404793183517247 w_k1_PG_s102
                                 + 0.0355636739575299 w_k1_PG_s103
                                 + 0.0114319537053406 w_k1_PG_s104
                                 + 0.0398311864279452 w_k1_PG_s105
                                 + 0.025229063573349 w_k1_PG_s106
                                 + 0.0280030081434402 w_k1_PG_s107
                                 + 0.0343800008588993 w_k1_PG_s110
                                 + 0.0640488239461361 w_k1_PG_s111
                                 + 0.171823079905503 w_k1_PG_s112
                                 + 0.0396385027783462 w_k1_PG_s113
                                 + 0.0696559419470041 w_k1_PG_s114
                                 + 0.0493286067607134 w_k1_PG_s115
                                 + 0.059536146842636 w_k1_PG_s116
                                 + 0.0383785445288385 w_k1_PG_s117 >= 1.11
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

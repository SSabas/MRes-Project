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
 w_k1_ADM_s0_return_constraint:  1.00841 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  1.05236 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.837951 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.1395 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.914016 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.974353 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.766593 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.892143 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.679364 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.32187 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.02032 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.07283 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.876146 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.3001 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.99143 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.60042 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.978367 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.18903 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.23429 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.785145 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.34217 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.675411 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.27838 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.24956 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.92822 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.986997 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.13149 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.08392 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.810316 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.00342 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.66678 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.912385 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.30465 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.44846 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.1889 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.16332 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.913619 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.00214 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.05278 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  1.01372 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.82037 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.34591 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.989066 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.12987 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.69442 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.840627 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.540463 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.30626 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.8116 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.82149 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.818049 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.23616 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.0626 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.52363 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.836859 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.09729 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.12447 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.664566 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.14229 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.77177 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.52002 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.17313 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.824035 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.928477 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.04319 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.1823 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.580082 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.853596 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.85012 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.962563 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.0524 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.67974 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.13142 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.14386 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.629904 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.725463 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  0.979434 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  1.0642 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.885187 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.36197 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.08247 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.02055 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.607354 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.675756 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.525198 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.21361 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.889612 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.924446 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.630436 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.34701 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.2 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 2.06824 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.86971 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.24013 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.26233 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.731271 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.43763 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.778453 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.55269 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.27144 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.699584 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.897656 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.24251 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.958804 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.509365 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.837119 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 2.11579 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.960944 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.29778 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.55204 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.25727 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.918864 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.519234 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.679152 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  0.956036 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  1.03198 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.90527 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.17223 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.96985 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.09886 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.75453 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.751535 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.659916 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.26901 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.884617 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.02406 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.805577 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.15555 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.07683 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.42863 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.963684 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.965403 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.09997 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.808125 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.36767 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.744933 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.35903 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.21043 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.831916 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.92104 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.28022 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.961596 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.716546 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.964348 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.71615 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.926711 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.2769 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.30047 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.15786 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.971616 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.662006 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.901431 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.25238 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.911202 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.815663 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.31738 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.18741 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.84677 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.909604 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.20773 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.594269 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.99717 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.0295 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.671855 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.745539 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.43717 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.04276 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.35286 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.86275 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.80309 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.3301 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.741925 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.764532 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.2779 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.4975 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.984025 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.982778 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.22723 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.768759 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.80194 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.643199 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.736992 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.38673 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.0416 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.794934 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.78508 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.18144 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.47355 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.791883 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.692182 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 0.992997 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.986729 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.926387 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.05992 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.9865 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.1117 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.845714 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.907038 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.794427 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.08615 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.886174 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.850979 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.703463 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.16394 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.08038 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.22292 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.07468 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.02094 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.22105 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.792161 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.39872 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.06419 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.30183 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.22572 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.928525 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.09986 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.16926 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.987808 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.796926 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.03452 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.29858 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.943484 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.18119 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.2882 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.06231 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.946423 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.67445 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.858453 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.12363 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.01863 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.814917 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.24329 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.03258 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.896207 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.810012 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.06555 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.645081 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.08371 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.00878 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.84386 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.676403 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.3132 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.00942 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.56194 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.902615 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.31442 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.23167 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.712269 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.07393 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.03044 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.43719 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.05689 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.940063 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.999189 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.986038 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.30989 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.627048 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.898293 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.57551 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.04393 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.03247 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.55026 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.08619 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.17595 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.691567 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.748347 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.34226 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.98666 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.617611 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.19515 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.11234 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.861713 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.19606 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.35087 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.625556 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.39591 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.04971 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.872952 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.957051 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.61105 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.16702 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.49831 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.872324 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.93137 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.40683 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.727956 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.891612 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.26089 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.55218 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.86406 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.11341 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.02641 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.706941 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.82868 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.457859 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.780223 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.80511 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.07959 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.07897 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.79146 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.04651 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.53785 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.709182 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.521702 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 0.988307 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 1.01559 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.885925 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.08173 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.05953 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.02881 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.808678 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.793523 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.730631 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.20098 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.917074 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.08366 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.745752 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.12676 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.04319 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.3044 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.05527 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.988032 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.09752 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.900488 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.32398 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.878263 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.19773 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.15922 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.81799 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.983875 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.17372 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.879006 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.714586 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.9465 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.49527 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.967704 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.32425 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.4202 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.17699 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.893321 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.707584 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.949023 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.01427 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.02295 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.893105 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.08246 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.98155 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.00029 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.882408 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.892175 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.768973 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.08287 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.944294 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.920622 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.88592 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.20979 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.13467 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.32397 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.0742 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.26952 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.19849 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.927327 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.26135 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.940395 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.35092 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.12084 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.937986 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.07009 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.12512 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.04873 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.815792 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.970943 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.22313 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.998028 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.035 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.16146 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.05206 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.06171 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.769066 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.802653 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.22199932616676 z_k1_s0000
                                 - 0.261999995331452 z_k1_s0001
                                 - 0.172999338589497 z_k1_s0002
                                 - 0.0709999017181254 z_k1_s0003
                                 - 0.230000065199323 z_k1_s0004
                                 - 0.0839998922467909 z_k1_s0005
                                 - 0.25799962581517 z_k1_s0006
                                 - 0.213000007754188 z_k1_s0007
                                 - 0.115000051226905 z_k1_s0010
                                 - 0.220999316705813 z_k1_s0011
                                 - 0.149999910469915 z_k1_s0012
                                 - 0.554999309138775 z_k1_s0013
                                 - 0.138999929022118 z_k1_s0014
                                 - 0.327000380184271 z_k1_s0015
                                 - 0.177999977664278 z_k1_s0016
                                 - 0.11299995652733 z_k1_s0017
                                 - 0.173999684404191 z_k1_s0020
                                 - 0.180000252245699 z_k1_s0021
                                 - 0.0970000570379623 z_k1_s0022
                                 - 0.124000030324819 z_k1_s0023
                                 - 0.239999475660955 z_k1_s0024
                                 - 0.126000047471993 z_k1_s0025
                                 - 0.195000380849505 z_k1_s0026
                                 - 0.155000037906024 z_k1_s0027
                                 - 0.266999818917784 z_k1_s0030
                                 - 0.977999729539203 z_k1_s0031
                                 - 0.957998201354721 z_k1_s0032
                                 - 0.370000076419696 z_k1_s0033
                                 - 1.06700004511171 z_k1_s0034
                                 - 0.331999992368502 z_k1_s0035
                                 - 1.037998675094 z_k1_s0036
                                 - 0.629002103563008 z_k1_s0037
                                 - 0.0969999076093363 z_k1_s0040
                                 - 0.252999925950076 z_k1_s0041
                                 - 0.0989999710239427 z_k1_s0042
                                 - 0.300000712193532 z_k1_s0043
                                 - 0.18899930468226 z_k1_s0044
                                 - 0.248999447120966 z_k1_s0045
                                 - 0.318999906632704 z_k1_s0046
                                 - 0.253999605657482 z_k1_s0047
                                 - 0.257998457815309 z_k1_s0050
                                 - 0.374998256589782 z_k1_s0051
                                 - 0.446996979379251 z_k1_s0052
                                 - 0.255997838796317 z_k1_s0053
                                 - 0.210999119000735 z_k1_s0054
                                 - 0.235998074577554 z_k1_s0055
                                 - 0.11999944210873 z_k1_s0056
                                 - 0.238999003106042 z_k1_s0057
                                 - 0.320999879473359 z_k1_s0060
                                 - 0.0669999938963221 z_k1_s0061
                                 - 0.0820000506194208 z_k1_s0062
                                 - 0.167000279383581 z_k1_s0063
                                 - 0.310000801610442 z_k1_s0064
                                 - 0.198999492258875 z_k1_s0065
                                 - 0.128000048503464 z_k1_s0066
                                 - 0.116000030925005 z_k1_s0067
                                 - 0.398999237302157 z_k1_s0070
                                 - 0.902996415087475 z_k1_s0071
                                 - 1.39799541431958 z_k1_s0072
                                 - 0.734000568513886 z_k1_s0073
                                 - 0.355999177631777 z_k1_s0074
                                 - 0.409999376329221 z_k1_s0075
                                 - 0.308999551205152 z_k1_s0076
                                 - 0.32899932018258 z_k1_s0077
                                 - 0.31400046921487 z_k1_s0100
                                 - 0.164999911615675 z_k1_s0101
                                 - 0.29900025994995 z_k1_s0102
                                 - 0.325999767427371 z_k1_s0103
                                 - 0.241000754591412 z_k1_s0104
                                 - 0.12199996362248 z_k1_s0105
                                 - 0.45599940805809 z_k1_s0106
                                 - 0.249999141751492 z_k1_s0107
                                 - 0.296999576705252 z_k1_s0110
                                 - 0.213000437354513 z_k1_s0111
                                 - 0.125000014300926 z_k1_s0112
                                 - 0.0800000700486127 z_k1_s0113
                                 - 0.0649999963644781 z_k1_s0114
                                 - 0.252000186053119 z_k1_s0115
                                 - 0.209999924377524 z_k1_s0116
                                 - 0.141999830236192 z_k1_s0117
                                 - 0.242000032292805 z_k1_s0120
                                 - 0.260000050950949 z_k1_s0121
                                 - 0.4279995694269 z_k1_s0122
                                 - 0.2869993764381 z_k1_s0123
                                 - 0.203000038700164 z_k1_s0124
                                 - 0.722001466510065 z_k1_s0125
                                 - 0.323001099754544 z_k1_s0126
                                 - 0.345000591781115 z_k1_s0127
                                 - 0.0770000278352386 z_k1_s0130
                                 - 0.170000163259693 z_k1_s0131
                                 - 0.0510000489779078 z_k1_s0132
                                 - 0.105999907102959 z_k1_s0133
                                 - 0.0879998078401077 z_k1_s0134
                                 - 0.0620000031829052 z_k1_s0135
                                 - 0.0760000319984207 z_k1_s0136
                                 - 0.240999780573703 z_k1_s0137
                                 - 0.0899999573964353 z_k1_s0140
                                 - 0.122999494808619 z_k1_s0141
                                 - 0.114999932017672 z_k1_s0142
                                 - 0.264999535945178 z_k1_s0143
                                 - 0.261999577998616 z_k1_s0144
                                 - 0.165999704709057 z_k1_s0145
                                 - 0.0669999953728923 z_k1_s0146
                                 - 0.130999788999308 z_k1_s0147
                                 - 0.311999438946275 z_k1_s0150
                                 - 0.249999563641544 z_k1_s0151
                                 - 0.491999564191445 z_k1_s0152
                                 - 0.289999438896284 z_k1_s0153
                                 - 0.469997504344933 z_k1_s0154
                                 - 0.32199940775996 z_k1_s0155
                                 - 0.96399850415668 z_k1_s0156
                                 - 0.332999407784956 z_k1_s0157
                                 - 0.36799946897855 z_k1_s0160
                                 - 0.111000088786503 z_k1_s0161
                                 - 0.115000067405833 z_k1_s0162
                                 - 0.158000232463688 z_k1_s0163
                                 - 0.107999997121989 z_k1_s0164
                                 - 0.208000324205364 z_k1_s0165
                                 - 0.150000275225028 z_k1_s0166
                                 - 0.218000629753744 z_k1_s0167
                                 - 0.24999826240576 z_k1_s0170
                                 - 0.141999312565156 z_k1_s0171
                                 - 0.137999334126692 z_k1_s0172
                                 - 0.194999143874301 z_k1_s0173
                                 - 0.111999552276328 z_k1_s0174
                                 - 0.277998579472958 z_k1_s0175
                                 - 0.258999422886997 z_k1_s0176
                                 - 0.185999543386214 z_k1_s0177
                                 - 0.578999465770952 z_k1_s1000
                                 - 0.502999659095836 z_k1_s1001
                                 - 1.73499737671231 z_k1_s1002
                                 - 0.851996478043985 z_k1_s1003
                                 - 1.40999515539536 z_k1_s1004
                                 - 0.709999026100577 z_k1_s1005
                                 - 0.787001916972363 z_k1_s1006
                                 - 0.778000633565295 z_k1_s1007
                                 - 1.22399710420905 z_k1_s1010
                                 - 0.678998126142537 z_k1_s1011
                                 - 0.535999320299766 z_k1_s1012
                                 - 0.435999182453213 z_k1_s1013
                                 - 0.632998662892173 z_k1_s1014
                                 - 0.956997254477938 z_k1_s1015
                                 - 0.548998219741587 z_k1_s1016
                                 - 0.441999136164093 z_k1_s1017
                                 - 0.503999139895702 z_k1_s1020
                                 - 1.41899822408109 z_k1_s1021
                                 - 0.355999059209618 z_k1_s1022
                                 - 0.316999396536128 z_k1_s1023
                                 - 1.02199738110159 z_k1_s1024
                                 - 1.20399619571033 z_k1_s1025
                                 - 0.42199907154806 z_k1_s1026
                                 - 0.587998413346297 z_k1_s1027
                                 - 0.26599991465659 z_k1_s1030
                                 - 0.147999950618704 z_k1_s1031
                                 - 0.384999743687157 z_k1_s1032
                                 - 0.262000959285911 z_k1_s1033
                                 - 0.294000171252045 z_k1_s1034
                                 - 0.336000556145228 z_k1_s1035
                                 - 0.194000047782636 z_k1_s1036
                                 - 0.638001161138865 z_k1_s1037
                                 - 0.162000128416883 z_k1_s1040
                                 - 0.164000036991139 z_k1_s1041
                                 - 0.601999876858862 z_k1_s1042
                                 - 0.194999989692497 z_k1_s1043
                                 - 0.125999947679754 z_k1_s1044
                                 - 0.578000973967793 z_k1_s1045
                                 - 0.18099994477251 z_k1_s1046
                                 - 0.275001126964106 z_k1_s1047
                                 - 0.123999964532098 z_k1_s1050
                                 - 0.262999687251154 z_k1_s1051
                                 - 0.189999698234677 z_k1_s1052
                                 - 0.172999910929461 z_k1_s1053
                                 - 0.236999922313769 z_k1_s1054
                                 - 0.131999927605138 z_k1_s1055
                                 - 0.340000348104122 z_k1_s1056
                                 - 0.0750000181347352 z_k1_s1057
                                 - 1.00599901811322 z_k1_s1060
                                 - 1.38599952312676 z_k1_s1061
                                 - 1.16399845514465 z_k1_s1062
                                 - 1.19500062194981 z_k1_s1063
                                 - 0.858000753793459 z_k1_s1064
                                 - 0.60200010439466 z_k1_s1065
                                 - 0.713000638385715 z_k1_s1066
                                 - 0.418999938485562 z_k1_s1067
                                 - 0.340999820252574 z_k1_s1070
                                 - 0.668000632816914 z_k1_s1071
                                 - 0.146999942421775 z_k1_s1072
                                 - 0.713999414077331 z_k1_s1073
                                 - 0.586000319788596 z_k1_s1074
                                 - 0.466001168892908 z_k1_s1075
                                 - 0.48300149522793 z_k1_s1076
                                 - 0.16799988315348 z_k1_s1077
                                 - 0.127999995872775 z_k1_s1100
                                 - 0.147000051425007 z_k1_s1101
                                 - 0.391000438648274 z_k1_s1102
                                 - 0.352000794325467 z_k1_s1103
                                 - 0.156000064691848 z_k1_s1104
                                 - 0.421001309204757 z_k1_s1105
                                 - 0.376999040788166 z_k1_s1106
                                 - 0.507000334198558 z_k1_s1107
                                 - 0.793001315932216 z_k1_s1110
                                 - 0.463999896822355 z_k1_s1111
                                 - 0.838001119769589 z_k1_s1112
                                 - 0.385999614304494 z_k1_s1113
                                 - 0.637999916493215 z_k1_s1114
                                 - 0.250999735892565 z_k1_s1115
                                 - 0.254000034081598 z_k1_s1116
                                 - 1.04500115122113 z_k1_s1117
                                 - 0.443999417132001 z_k1_s1120
                                 - 0.409000815781558 z_k1_s1121
                                 - 0.370000814276718 z_k1_s1122
                                 - 0.302000411652879 z_k1_s1123
                                 - 0.274999410611028 z_k1_s1124
                                 - 0.20000006771713 z_k1_s1125
                                 - 0.235999929106208 z_k1_s1126
                                 - 0.364000014045173 z_k1_s1127
                                 - 0.238999110333243 z_k1_s1130
                                 - 0.246000139573316 z_k1_s1131
                                 - 0.289999635366823 z_k1_s1132
                                 - 0.507001031803673 z_k1_s1133
                                 - 0.227999236384866 z_k1_s1134
                                 - 0.15200003332334 z_k1_s1135
                                 - 0.225999814030714 z_k1_s1136
                                 - 0.14599993566056 z_k1_s1137
                                 - 0.811002439150429 z_k1_s1140
                                 - 0.403999392176853 z_k1_s1141
                                 - 0.428001253488683 z_k1_s1142
                                 - 0.458999909632981 z_k1_s1143
                                 - 0.242000712403422 z_k1_s1144
                                 - 0.496999271618598 z_k1_s1145
                                 - 0.677003453935194 z_k1_s1146
                                 - 0.393001635290938 z_k1_s1147
                                 - 0.388999899761338 z_k1_s1150
                                 - 0.268000544617168 z_k1_s1151
                                 - 0.124999967310198 z_k1_s1152
                                 - 0.179000118800232 z_k1_s1153
                                 - 0.207999754628138 z_k1_s1154
                                 - 0.325000549106621 z_k1_s1155
                                 - 0.212999820460557 z_k1_s1156
                                 - 0.158000028804102 z_k1_s1157
                                 - 0.698001853798426 z_k1_s1160
                                 - 0.958997586255839 z_k1_s1161
                                 - 0.537999317710684 z_k1_s1162
                                 - 1.36799905118866 z_k1_s1163
                                 - 0.954999753049855 z_k1_s1164
                                 - 0.427999432647203 z_k1_s1165
                                 - 1.14499925416001 z_k1_s1166
                                 - 0.516999129398923 z_k1_s1167
                                 - 0.25100020158954 z_k1_s1170
                                 - 0.263000451702165 z_k1_s1171
                                 - 0.734000559682511 z_k1_s1172
                                 - 0.250000297722004 z_k1_s1173
                                 - 1.23000341125931 z_k1_s1174
                                 - 0.478002710358228 z_k1_s1175
                                 - 0.477001402788492 z_k1_s1176
                                 - 0.996001575417874 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.13889 w_k1_ADM_s000
                                 + 1.69637 w_k1_AXP_s000
                                 + 1.69238 w_k1_DIS_s000
                                 + 1.40041 w_k1_FDX_s000 + 1.1997 w_k1_IBM_s000
                                 + 1.07652 w_k1_INTC_s000
                                 + 1.49035 w_k1_KO_s000 + 1.56482 w_k1_MAT_s000
                                 + 1.26252 w_k1_MSFT_s000
                                 + 1.16197 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.05443 w_k1_ADM_s000
                                 + 1.21219 w_k1_AXP_s000
                                 + 1.13805 w_k1_DIS_s000
                                 + 0.950323 w_k1_FDX_s000
                                 + 1.31985 w_k1_IBM_s000
                                 + 0.98433 w_k1_INTC_s000
                                 + 1.20447 w_k1_KO_s000 + 1.10509 w_k1_MAT_s000
                                 + 0.968175 w_k1_MSFT_s000
                                 + 1.02343 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.990777 w_k1_ADM_s000
                                 + 1.26029 w_k1_AXP_s000
                                 + 1.23174 w_k1_DIS_s000
                                 + 1.22942 w_k1_FDX_s000
                                 + 0.77271 w_k1_IBM_s000
                                 + 1.10129 w_k1_INTC_s000
                                 + 0.926889 w_k1_KO_s000
                                 + 0.709675 w_k1_MAT_s000
                                 + 1.17455 w_k1_MSFT_s000
                                 + 0.988189 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.674207 w_k1_ADM_s000
                                 + 0.692314 w_k1_AXP_s000
                                 + 0.667758 w_k1_DIS_s000
                                 + 0.632666 w_k1_FDX_s000
                                 + 0.737962 w_k1_IBM_s000
                                 + 0.638921 w_k1_INTC_s000
                                 + 0.736895 w_k1_KO_s000
                                 + 0.724002 w_k1_MAT_s000
                                 + 0.640422 w_k1_MSFT_s000
                                 + 0.74602 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.80317 w_k1_ADM_s000
                                 + 0.828398 w_k1_AXP_s000
                                 + 0.742711 w_k1_DIS_s000
                                 + 0.840513 w_k1_FDX_s000
                                 + 0.773792 w_k1_IBM_s000
                                 + 0.834167 w_k1_INTC_s000
                                 + 0.770882 w_k1_KO_s000
                                 + 0.720476 w_k1_MAT_s000
                                 + 0.886578 w_k1_MSFT_s000
                                 + 0.928606 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.42067 w_k1_ADM_s000
                                 + 1.55126 w_k1_AXP_s000
                                 + 1.41404 w_k1_DIS_s000
                                 + 1.15811 w_k1_FDX_s000
                                 + 1.65193 w_k1_IBM_s000
                                 + 0.958409 w_k1_INTC_s000
                                 + 1.39816 w_k1_KO_s000 + 1.69738 w_k1_MAT_s000
                                 + 1.12279 w_k1_MSFT_s000
                                 + 1.22841 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.02133 w_k1_ADM_s000
                                 + 1.25679 w_k1_AXP_s000
                                 + 1.03182 w_k1_DIS_s000
                                 + 1.11183 w_k1_FDX_s000
                                 + 0.947106 w_k1_IBM_s000
                                 + 1.06483 w_k1_INTC_s000
                                 + 1.11973 w_k1_KO_s000 + 1.09272 w_k1_MAT_s000
                                 + 1.01049 w_k1_MSFT_s000
                                 + 1.00068 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.958386 w_k1_ADM_s000
                                 + 0.966786 w_k1_AXP_s000
                                 + 0.811574 w_k1_DIS_s000
                                 + 0.783382 w_k1_FDX_s000
                                 + 1.07591 w_k1_IBM_s000
                                 + 0.868782 w_k1_INTC_s000
                                 + 0.947571 w_k1_KO_s000
                                 + 0.791986 w_k1_MAT_s000
                                 + 0.88145 w_k1_MSFT_s000
                                 + 0.963278 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.19501 w_k1_ADM_s001
                                 + 1.47643 w_k1_AXP_s001
                                 + 1.07882 w_k1_DIS_s001
                                 + 1.10008 w_k1_FDX_s001
                                 + 1.18741 w_k1_IBM_s001
                                 + 1.16381 w_k1_INTC_s001
                                 + 0.946695 w_k1_KO_s001
                                 + 1.21704 w_k1_MAT_s001
                                 + 1.03822 w_k1_MSFT_s001
                                 + 1.09666 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.05119 w_k1_ADM_s001
                                 + 0.927587 w_k1_AXP_s001
                                 + 1.06256 w_k1_DIS_s001
                                 + 0.963514 w_k1_FDX_s001
                                 + 1.23105 w_k1_IBM_s001
                                 + 0.875956 w_k1_INTC_s001
                                 + 1.10687 w_k1_KO_s001 + 1.10456 w_k1_MAT_s001
                                 + 1.22453 w_k1_MSFT_s001
                                 + 1.08112 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.48524 w_k1_ADM_s001
                                 + 1.62943 w_k1_AXP_s001
                                 + 1.61583 w_k1_DIS_s001
                                 + 1.51233 w_k1_FDX_s001
                                 + 1.11934 w_k1_IBM_s001
                                 + 0.887335 w_k1_INTC_s001
                                 + 1.16905 w_k1_KO_s001 + 1.43492 w_k1_MAT_s001
                                 + 1.38914 w_k1_MSFT_s001
                                 + 1.21501 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.849646 w_k1_ADM_s001
                                 + 0.607231 w_k1_AXP_s001
                                 + 0.724473 w_k1_DIS_s001
                                 + 0.79092 w_k1_FDX_s001
                                 + 0.769308 w_k1_IBM_s001
                                 + 0.660577 w_k1_INTC_s001
                                 + 0.646265 w_k1_KO_s001
                                 + 0.689242 w_k1_MAT_s001
                                 + 0.840232 w_k1_MSFT_s001
                                 + 0.938327 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.1397 w_k1_ADM_s001
                                 + 1.60197 w_k1_AXP_s001
                                 + 1.00803 w_k1_DIS_s001
                                 + 1.15014 w_k1_FDX_s001
                                 + 0.852994 w_k1_IBM_s001
                                 + 0.976668 w_k1_INTC_s001
                                 + 0.822799 w_k1_KO_s001
                                 + 1.01002 w_k1_MAT_s001
                                 + 1.0931 w_k1_MSFT_s001 + 1.15647 w_k1_PG_s001
                                 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.10851 w_k1_ADM_s001
                                 + 1.03557 w_k1_AXP_s001
                                 + 1.13299 w_k1_DIS_s001
                                 + 1.12439 w_k1_FDX_s001
                                 + 0.889166 w_k1_IBM_s001
                                 + 0.919542 w_k1_INTC_s001
                                 + 0.853572 w_k1_KO_s001
                                 + 0.769605 w_k1_MAT_s001
                                 + 0.93154 w_k1_MSFT_s001
                                 + 1.05094 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.10397 w_k1_ADM_s001
                                 + 1.15388 w_k1_AXP_s001
                                 + 1.10539 w_k1_DIS_s001 + 1.0615 w_k1_FDX_s001
                                 + 1.12497 w_k1_IBM_s001
                                 + 1.01754 w_k1_INTC_s001
                                 + 1.05444 w_k1_KO_s001 + 1.51514 w_k1_MAT_s001
                                 + 1.07478 w_k1_MSFT_s001
                                 + 1.14161 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.999793 w_k1_ADM_s001
                                 + 1.15135 w_k1_AXP_s001
                                 + 0.873805 w_k1_DIS_s001
                                 + 0.86591 w_k1_FDX_s001
                                 + 1.61244 w_k1_IBM_s001
                                 + 1.02599 w_k1_INTC_s001
                                 + 1.12882 w_k1_KO_s001 + 1.39034 w_k1_MAT_s001
                                 + 0.955435 w_k1_MSFT_s001
                                 + 1.14271 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.68615 w_k1_ADM_s002
                                 + 1.61896 w_k1_AXP_s002
                                 + 1.38256 w_k1_DIS_s002
                                 + 1.34513 w_k1_FDX_s002
                                 + 1.21953 w_k1_IBM_s002
                                 + 0.795863 w_k1_INTC_s002
                                 + 1.11116 w_k1_KO_s002 + 1.16023 w_k1_MAT_s002
                                 + 1.06206 w_k1_MSFT_s002
                                 + 1.22969 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.09587 w_k1_ADM_s002
                                 + 0.945005 w_k1_AXP_s002
                                 + 0.764147 w_k1_DIS_s002
                                 + 0.888528 w_k1_FDX_s002
                                 + 0.902158 w_k1_IBM_s002
                                 + 0.838314 w_k1_INTC_s002
                                 + 0.97235 w_k1_KO_s002
                                 + 0.804334 w_k1_MAT_s002
                                 + 0.847348 w_k1_MSFT_s002
                                 + 0.878818 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.753127 w_k1_ADM_s002
                                 + 0.626012 w_k1_AXP_s002
                                 + 0.41886 w_k1_DIS_s002
                                 + 0.598984 w_k1_FDX_s002
                                 + 0.903303 w_k1_IBM_s002
                                 + 0.594403 w_k1_INTC_s002
                                 + 0.781651 w_k1_KO_s002
                                 + 1.11705 w_k1_MAT_s002
                                 + 0.710636 w_k1_MSFT_s002
                                 + 0.762214 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.909366 w_k1_ADM_s002
                                 + 1.03841 w_k1_AXP_s002
                                 + 1.08832 w_k1_DIS_s002
                                 + 0.922348 w_k1_FDX_s002
                                 + 1.65489 w_k1_IBM_s002
                                 + 0.722005 w_k1_INTC_s002
                                 + 1.17842 w_k1_KO_s002 + 1.83807 w_k1_MAT_s002
                                 + 0.955961 w_k1_MSFT_s002
                                 + 1.08488 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.20291 w_k1_ADM_s002
                                 + 0.957108 w_k1_AXP_s002
                                 + 0.917139 w_k1_DIS_s002
                                 + 0.949738 w_k1_FDX_s002
                                 + 1.23524 w_k1_IBM_s002
                                 + 0.841147 w_k1_INTC_s002
                                 + 0.939436 w_k1_KO_s002
                                 + 1.04587 w_k1_MAT_s002
                                 + 0.920289 w_k1_MSFT_s002
                                 + 0.901489 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.968728 w_k1_ADM_s002
                                 + 0.871176 w_k1_AXP_s002
                                 + 0.748515 w_k1_DIS_s002
                                 + 0.761641 w_k1_FDX_s002
                                 + 0.966312 w_k1_IBM_s002
                                 + 0.697755 w_k1_INTC_s002
                                 + 0.953644 w_k1_KO_s002
                                 + 0.906306 w_k1_MAT_s002
                                 + 0.761762 w_k1_MSFT_s002
                                 + 0.990177 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.47932 w_k1_ADM_s002
                                 + 1.21764 w_k1_AXP_s002
                                 + 1.16465 w_k1_DIS_s002
                                 + 1.19204 w_k1_FDX_s002
                                 + 0.863593 w_k1_IBM_s002
                                 + 0.89792 w_k1_INTC_s002
                                 + 0.978535 w_k1_KO_s002
                                 + 0.992639 w_k1_MAT_s002
                                 + 1.10699 w_k1_MSFT_s002
                                 + 1.04363 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.982411 w_k1_ADM_s002
                                 + 0.695944 w_k1_AXP_s002
                                 + 0.737733 w_k1_DIS_s002
                                 + 0.907423 w_k1_FDX_s002
                                 + 0.629501 w_k1_IBM_s002
                                 + 0.810169 w_k1_INTC_s002
                                 + 0.739149 w_k1_KO_s002
                                 + 0.739294 w_k1_MAT_s002
                                 + 0.95393 w_k1_MSFT_s002
                                 + 0.897933 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.34763 w_k1_ADM_s003
                                 + 1.43239 w_k1_AXP_s003
                                 + 1.29354 w_k1_DIS_s003
                                 + 1.30344 w_k1_FDX_s003 + 1.1554 w_k1_IBM_s003
                                 + 1.49109 w_k1_INTC_s003
                                 + 1.33583 w_k1_KO_s003 + 1.16812 w_k1_MAT_s003
                                 + 1.40357 w_k1_MSFT_s003 + 1.1517 w_k1_PG_s003
                                 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.969857 w_k1_ADM_s003
                                 + 1.04572 w_k1_AXP_s003
                                 + 1.01798 w_k1_DIS_s003
                                 + 0.993042 w_k1_FDX_s003
                                 + 1.02427 w_k1_IBM_s003
                                 + 1.08517 w_k1_INTC_s003
                                 + 1.00576 w_k1_KO_s003 + 1.00408 w_k1_MAT_s003
                                 + 1.10881 w_k1_MSFT_s003
                                 + 1.09073 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.911259 w_k1_ADM_s003
                                 + 1.16097 w_k1_AXP_s003
                                 + 0.954984 w_k1_DIS_s003
                                 + 1.02339 w_k1_FDX_s003
                                 + 0.872066 w_k1_IBM_s003
                                 + 0.922806 w_k1_INTC_s003
                                 + 0.844207 w_k1_KO_s003
                                 + 1.14921 w_k1_MAT_s003
                                 + 0.874124 w_k1_MSFT_s003
                                 + 0.999129 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.20088 w_k1_ADM_s003
                                 + 1.69935 w_k1_AXP_s003
                                 + 1.98677 w_k1_DIS_s003
                                 + 1.44625 w_k1_FDX_s003
                                 + 1.40917 w_k1_IBM_s003
                                 + 1.64945 w_k1_INTC_s003
                                 + 1.84019 w_k1_KO_s003 + 1.62173 w_k1_MAT_s003
                                 + 1.38356 w_k1_MSFT_s003
                                 + 1.41167 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.704576 w_k1_ADM_s003
                                 + 0.833155 w_k1_AXP_s003
                                 + 0.634764 w_k1_DIS_s003
                                 + 0.728247 w_k1_FDX_s003
                                 + 1.02973 w_k1_IBM_s003
                                 + 0.908716 w_k1_INTC_s003
                                 + 0.850519 w_k1_KO_s003
                                 + 0.979826 w_k1_MAT_s003
                                 + 0.779049 w_k1_MSFT_s003
                                 + 0.953007 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.09878 w_k1_ADM_s003
                                 + 1.37592 w_k1_AXP_s003
                                 + 1.59075 w_k1_DIS_s003
                                 + 1.31936 w_k1_FDX_s003
                                 + 0.794616 w_k1_IBM_s003
                                 + 1.29021 w_k1_INTC_s003 + 1.1514 w_k1_KO_s003
                                 + 0.784806 w_k1_MAT_s003
                                 + 1.41539 w_k1_MSFT_s003
                                 + 1.30831 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.859154 w_k1_ADM_s003
                                 + 0.670609 w_k1_AXP_s003
                                 + 0.800304 w_k1_DIS_s003
                                 + 0.886355 w_k1_FDX_s003
                                 + 0.666599 w_k1_IBM_s003
                                 + 0.878465 w_k1_INTC_s003
                                 + 0.790144 w_k1_KO_s003
                                 + 0.436927 w_k1_MAT_s003
                                 + 0.880325 w_k1_MSFT_s003
                                 + 0.97573 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.03103 w_k1_ADM_s003
                                 + 1.15134 w_k1_AXP_s003
                                 + 1.22181 w_k1_DIS_s003
                                 + 1.11721 w_k1_FDX_s003
                                 + 1.46967 w_k1_IBM_s003
                                 + 1.16886 w_k1_INTC_s003
                                 + 1.37576 w_k1_KO_s003 + 1.41779 w_k1_MAT_s003
                                 + 1.11871 w_k1_MSFT_s003
                                 + 1.21242 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.1761 w_k1_ADM_s004
                                 + 1.33877 w_k1_AXP_s004
                                 + 0.967002 w_k1_DIS_s004
                                 + 1.00439 w_k1_FDX_s004
                                 + 1.19959 w_k1_IBM_s004
                                 + 0.858826 w_k1_INTC_s004
                                 + 0.99087 w_k1_KO_s004 + 1.55414 w_k1_MAT_s004
                                 + 0.986936 w_k1_MSFT_s004
                                 + 1.2238 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.09083 w_k1_ADM_s004
                                 + 1.19924 w_k1_AXP_s004 + 1.5082 w_k1_DIS_s004
                                 + 1.1797 w_k1_FDX_s004 + 1.39756 w_k1_IBM_s004
                                 + 1.28297 w_k1_INTC_s004
                                 + 1.37516 w_k1_KO_s004 + 1.59628 w_k1_MAT_s004
                                 + 1.26943 w_k1_MSFT_s004
                                 + 1.11688 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.30018 w_k1_ADM_s004
                                 + 1.25313 w_k1_AXP_s004
                                 + 1.29694 w_k1_DIS_s004
                                 + 1.05785 w_k1_FDX_s004
                                 + 1.04497 w_k1_IBM_s004
                                 + 1.00858 w_k1_INTC_s004 + 1.0962 w_k1_KO_s004
                                 + 1.08166 w_k1_MAT_s004
                                 + 1.11849 w_k1_MSFT_s004 + 1.1746 w_k1_PG_s004
                                 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.89848 w_k1_ADM_s004
                                 + 1.0004 w_k1_AXP_s004
                                 + 0.905905 w_k1_DIS_s004
                                 + 0.942088 w_k1_FDX_s004
                                 + 0.877499 w_k1_IBM_s004
                                 + 1.08868 w_k1_INTC_s004
                                 + 0.917282 w_k1_KO_s004
                                 + 0.745486 w_k1_MAT_s004
                                 + 1.01751 w_k1_MSFT_s004
                                 + 0.99595 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.847631 w_k1_ADM_s004
                                 + 0.719765 w_k1_AXP_s004
                                 + 0.606287 w_k1_DIS_s004
                                 + 0.713968 w_k1_FDX_s004
                                 + 0.864047 w_k1_IBM_s004
                                 + 0.922542 w_k1_INTC_s004
                                 + 0.71145 w_k1_KO_s004
                                 + 0.592239 w_k1_MAT_s004
                                 + 0.786772 w_k1_MSFT_s004
                                 + 0.952045 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.825086 w_k1_ADM_s004
                                 + 0.817772 w_k1_AXP_s004
                                 + 0.892534 w_k1_DIS_s004
                                 + 0.908338 w_k1_FDX_s004
                                 + 1.11088 w_k1_IBM_s004
                                 + 0.996376 w_k1_INTC_s004
                                 + 1.05105 w_k1_KO_s004 + 1.15462 w_k1_MAT_s004
                                 + 1.08095 w_k1_MSFT_s004
                                 + 1.01375 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.00466 w_k1_ADM_s004
                                 + 1.01774 w_k1_AXP_s004
                                 + 0.977266 w_k1_DIS_s004
                                 + 1.04201 w_k1_FDX_s004
                                 + 0.670532 w_k1_IBM_s004
                                 + 0.933184 w_k1_INTC_s004
                                 + 0.874759 w_k1_KO_s004
                                 + 0.926817 w_k1_MAT_s004
                                 + 1.045 w_k1_MSFT_s004 + 0.97823 w_k1_PG_s004
                                 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.23969 w_k1_ADM_s004
                                 + 1.33667 w_k1_AXP_s004
                                 + 1.38219 w_k1_DIS_s004
                                 + 1.25656 w_k1_FDX_s004 + 1.0212 w_k1_IBM_s004
                                 + 1.34336 w_k1_INTC_s004
                                 + 1.19546 w_k1_KO_s004 + 1.40937 w_k1_MAT_s004
                                 + 1.31787 w_k1_MSFT_s004
                                 + 1.19507 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.85928 w_k1_ADM_s005
                                 + 0.755838 w_k1_AXP_s005
                                 + 0.946476 w_k1_DIS_s005
                                 + 0.870145 w_k1_FDX_s005
                                 + 0.924384 w_k1_IBM_s005
                                 + 1.17364 w_k1_INTC_s005
                                 + 1.04878 w_k1_KO_s005
                                 + 0.865952 w_k1_MAT_s005
                                 + 0.963675 w_k1_MSFT_s005
                                 + 1.06955 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.09197 w_k1_ADM_s005
                                 + 1.17556 w_k1_AXP_s005
                                 + 0.824314 w_k1_DIS_s005
                                 + 1.01717 w_k1_FDX_s005
                                 + 0.782594 w_k1_IBM_s005
                                 + 1.12747 w_k1_INTC_s005 + 0.7791 w_k1_KO_s005
                                 + 0.72427 w_k1_MAT_s005
                                 + 0.927564 w_k1_MSFT_s005
                                 + 0.910061 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.911392 w_k1_ADM_s005
                                 + 0.885365 w_k1_AXP_s005
                                 + 0.853957 w_k1_DIS_s005
                                 + 0.822411 w_k1_FDX_s005
                                 + 1.10282 w_k1_IBM_s005
                                 + 0.936834 w_k1_INTC_s005
                                 + 0.970727 w_k1_KO_s005
                                 + 1.03159 w_k1_MAT_s005
                                 + 0.83995 w_k1_MSFT_s005
                                 + 1.02171 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.795862 w_k1_ADM_s005
                                 + 0.628179 w_k1_AXP_s005
                                 + 0.541177 w_k1_DIS_s005
                                 + 0.781642 w_k1_FDX_s005
                                 + 0.716002 w_k1_IBM_s005
                                 + 0.827429 w_k1_INTC_s005
                                 + 0.708986 w_k1_KO_s005
                                 + 0.506605 w_k1_MAT_s005
                                 + 0.732015 w_k1_MSFT_s005
                                 + 0.940877 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.913373 w_k1_ADM_s005
                                 + 1.07752 w_k1_AXP_s005
                                 + 1.29179 w_k1_DIS_s005 + 1.1846 w_k1_FDX_s005
                                 + 0.910792 w_k1_IBM_s005
                                 + 1.13111 w_k1_INTC_s005
                                 + 1.15144 w_k1_KO_s005
                                 + 0.747935 w_k1_MAT_s005
                                 + 1.09205 w_k1_MSFT_s005
                                 + 1.06972 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.953523 w_k1_ADM_s005
                                 + 1.12991 w_k1_AXP_s005 + 1.1412 w_k1_DIS_s005
                                 + 0.982621 w_k1_FDX_s005
                                 + 1.46777 w_k1_IBM_s005
                                 + 1.37595 w_k1_INTC_s005
                                 + 1.28273 w_k1_KO_s005 + 1.22016 w_k1_MAT_s005
                                 + 1.09753 w_k1_MSFT_s005
                                 + 1.14864 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.21769 w_k1_ADM_s005
                                 + 1.40837 w_k1_AXP_s005
                                 + 1.61589 w_k1_DIS_s005
                                 + 1.13611 w_k1_FDX_s005
                                 + 1.64571 w_k1_IBM_s005
                                 + 1.33788 w_k1_INTC_s005
                                 + 1.57916 w_k1_KO_s005 + 1.23771 w_k1_MAT_s005
                                 + 1.09431 w_k1_MSFT_s005
                                 + 1.17134 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.06228 w_k1_ADM_s005
                                 + 1.56885 w_k1_AXP_s005
                                 + 1.53219 w_k1_DIS_s005 + 1.3442 w_k1_FDX_s005
                                 + 1.11201 w_k1_IBM_s005
                                 + 1.2835 w_k1_INTC_s005 + 1.24295 w_k1_KO_s005
                                 + 1.28746 w_k1_MAT_s005
                                 + 1.21272 w_k1_MSFT_s005
                                 + 1.20065 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.04614 w_k1_ADM_s006
                                 + 1.0254 w_k1_AXP_s006
                                 + 0.946779 w_k1_DIS_s006
                                 + 1.08329 w_k1_FDX_s006
                                 + 0.980257 w_k1_IBM_s006
                                 + 1.15814 w_k1_INTC_s006
                                 + 1.17761 w_k1_KO_s006
                                 + 0.829143 w_k1_MAT_s006
                                 + 1.0997 w_k1_MSFT_s006
                                 + 0.941877 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.55925 w_k1_ADM_s006
                                 + 1.72175 w_k1_AXP_s006
                                 + 1.55482 w_k1_DIS_s006
                                 + 1.36473 w_k1_FDX_s006
                                 + 1.34964 w_k1_IBM_s006
                                 + 1.49558 w_k1_INTC_s006
                                 + 1.31497 w_k1_KO_s006 + 1.7687 w_k1_MAT_s006
                                 + 1.28136 w_k1_MSFT_s006
                                 + 1.13735 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.577804 w_k1_ADM_s006
                                 + 0.494832 w_k1_AXP_s006
                                 + 0.594558 w_k1_DIS_s006
                                 + 0.728615 w_k1_FDX_s006
                                 + 0.709961 w_k1_IBM_s006
                                 + 0.844951 w_k1_INTC_s006
                                 + 0.763098 w_k1_KO_s006
                                 + 0.521856 w_k1_MAT_s006
                                 + 0.816219 w_k1_MSFT_s006
                                 + 0.711037 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.08189 w_k1_ADM_s006
                                 + 1.18221 w_k1_AXP_s006
                                 + 1.30703 w_k1_DIS_s006
                                 + 1.21888 w_k1_FDX_s006
                                 + 1.04563 w_k1_IBM_s006
                                 + 1.51869 w_k1_INTC_s006 + 1.2423 w_k1_KO_s006
                                 + 1.01167 w_k1_MAT_s006
                                 + 1.4413 w_k1_MSFT_s006 + 1.12339 w_k1_PG_s006
                                 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.15983 w_k1_ADM_s006
                                 + 1.16551 w_k1_AXP_s006
                                 + 0.961078 w_k1_DIS_s006
                                 + 1.01881 w_k1_FDX_s006
                                 + 1.36703 w_k1_IBM_s006
                                 + 1.06308 w_k1_INTC_s006
                                 + 1.15125 w_k1_KO_s006 + 1.40634 w_k1_MAT_s006
                                 + 0.975017 w_k1_MSFT_s006
                                 + 0.974672 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.937733 w_k1_ADM_s006
                                 + 0.665509 w_k1_AXP_s006
                                 + 0.731873 w_k1_DIS_s006
                                 + 0.806984 w_k1_FDX_s006
                                 + 0.987792 w_k1_IBM_s006
                                 + 1.0684 w_k1_INTC_s006 + 1.11504 w_k1_KO_s006
                                 + 0.720294 w_k1_MAT_s006
                                 + 0.990984 w_k1_MSFT_s006
                                 + 0.9728 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.992245 w_k1_ADM_s006
                                 + 1.01808 w_k1_AXP_s006
                                 + 0.797699 w_k1_DIS_s006
                                 + 1.02295 w_k1_FDX_s006
                                 + 0.643253 w_k1_IBM_s006
                                 + 1.14926 w_k1_INTC_s006
                                 + 0.937766 w_k1_KO_s006
                                 + 0.783774 w_k1_MAT_s006
                                 + 1.04374 w_k1_MSFT_s006
                                 + 0.916975 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.810544 w_k1_ADM_s006
                                 + 1.05427 w_k1_AXP_s006
                                 + 0.649688 w_k1_DIS_s006
                                 + 0.758783 w_k1_FDX_s006
                                 + 0.89582 w_k1_IBM_s006
                                 + 1.03271 w_k1_INTC_s006
                                 + 0.894231 w_k1_KO_s006
                                 + 0.770192 w_k1_MAT_s006
                                 + 0.817538 w_k1_MSFT_s006
                                 + 0.822199 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.765499 w_k1_ADM_s007
                                 + 1.14586 w_k1_AXP_s007
                                 + 0.803334 w_k1_DIS_s007
                                 + 0.938215 w_k1_FDX_s007
                                 + 1.44094 w_k1_IBM_s007
                                 + 0.822709 w_k1_INTC_s007
                                 + 0.971417 w_k1_KO_s007
                                 + 1.69155 w_k1_MAT_s007
                                 + 0.877482 w_k1_MSFT_s007
                                 + 0.946734 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.11514 w_k1_ADM_s007
                                 + 1.14533 w_k1_AXP_s007
                                 + 1.22946 w_k1_DIS_s007
                                 + 1.13847 w_k1_FDX_s007
                                 + 0.839768 w_k1_IBM_s007
                                 + 1.05398 w_k1_INTC_s007
                                 + 1.12035 w_k1_KO_s007
                                 + 0.949322 w_k1_MAT_s007
                                 + 1.17917 w_k1_MSFT_s007
                                 + 0.964122 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.751222 w_k1_ADM_s007
                                 + 0.827505 w_k1_AXP_s007
                                 + 0.622112 w_k1_DIS_s007
                                 + 0.754885 w_k1_FDX_s007
                                 + 0.879191 w_k1_IBM_s007
                                 + 0.694531 w_k1_INTC_s007
                                 + 0.827418 w_k1_KO_s007
                                 + 1.19546 w_k1_MAT_s007
                                 + 0.79526 w_k1_MSFT_s007
                                 + 0.831454 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.994678 w_k1_ADM_s007
                                 + 1.25871 w_k1_AXP_s007
                                 + 1.21652 w_k1_DIS_s007
                                 + 1.08967 w_k1_FDX_s007
                                 + 1.15128 w_k1_IBM_s007
                                 + 1.24932 w_k1_INTC_s007
                                 + 1.15946 w_k1_KO_s007 + 1.52726 w_k1_MAT_s007
                                 + 1.34372 w_k1_MSFT_s007
                                 + 0.97796 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.804559 w_k1_ADM_s007
                                 + 0.847648 w_k1_AXP_s007
                                 + 0.676551 w_k1_DIS_s007
                                 + 0.935011 w_k1_FDX_s007
                                 + 0.654766 w_k1_IBM_s007
                                 + 0.913469 w_k1_INTC_s007
                                 + 0.671606 w_k1_KO_s007
                                 + 0.751101 w_k1_MAT_s007
                                 + 0.830107 w_k1_MSFT_s007
                                 + 0.861426 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.746786 w_k1_ADM_s007
                                 + 0.834863 w_k1_AXP_s007
                                 + 0.885248 w_k1_DIS_s007
                                 + 0.818898 w_k1_FDX_s007
                                 + 1.24699 w_k1_IBM_s007
                                 + 0.770236 w_k1_INTC_s007
                                 + 1.02962 w_k1_KO_s007 + 1.11726 w_k1_MAT_s007
                                 + 0.876291 w_k1_MSFT_s007
                                 + 0.943528 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.829616 w_k1_ADM_s007
                                 + 1.10705 w_k1_AXP_s007
                                 + 1.31249 w_k1_DIS_s007
                                 + 1.04101 w_k1_FDX_s007
                                 + 1.76245 w_k1_IBM_s007
                                 + 0.995361 w_k1_INTC_s007
                                 + 1.57693 w_k1_KO_s007 + 2.12278 w_k1_MAT_s007
                                 + 1.3269 w_k1_MSFT_s007 + 1.15283 w_k1_PG_s007
                                 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.35394 w_k1_ADM_s007
                                 + 1.91982 w_k1_AXP_s007
                                 + 1.95494 w_k1_DIS_s007
                                 + 1.54116 w_k1_FDX_s007
                                 + 1.31572 w_k1_IBM_s007
                                 + 1.36015 w_k1_INTC_s007
                                 + 1.52251 w_k1_KO_s007 + 2.04179 w_k1_MAT_s007
                                 + 1.54819 w_k1_MSFT_s007 + 1.3474 w_k1_PG_s007
                                 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.04306 w_k1_ADM_s010
                                 + 0.948716 w_k1_AXP_s010
                                 + 1.01372 w_k1_DIS_s010
                                 + 0.977198 w_k1_FDX_s010
                                 + 1.25905 w_k1_IBM_s010
                                 + 0.975338 w_k1_INTC_s010
                                 + 1.18103 w_k1_KO_s010 + 1.12622 w_k1_MAT_s010
                                 + 1.04409 w_k1_MSFT_s010
                                 + 0.945988 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.882956 w_k1_ADM_s010
                                 + 0.852844 w_k1_AXP_s010
                                 + 0.58742 w_k1_DIS_s010
                                 + 0.869206 w_k1_FDX_s010
                                 + 0.696753 w_k1_IBM_s010
                                 + 1.10673 w_k1_INTC_s010
                                 + 0.718436 w_k1_KO_s010
                                 + 0.450783 w_k1_MAT_s010
                                 + 0.891191 w_k1_MSFT_s010
                                 + 0.788146 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.20805 w_k1_ADM_s010
                                 + 1.35694 w_k1_AXP_s010
                                 + 0.858626 w_k1_DIS_s010
                                 + 1.07933 w_k1_FDX_s010
                                 + 0.934935 w_k1_IBM_s010
                                 + 1.05726 w_k1_INTC_s010
                                 + 0.969526 w_k1_KO_s010
                                 + 1.23657 w_k1_MAT_s010
                                 + 0.98716 w_k1_MSFT_s010
                                 + 0.958598 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.3329 w_k1_ADM_s010
                                 + 1.40574 w_k1_AXP_s010
                                 + 1.41079 w_k1_DIS_s010 + 1.3662 w_k1_FDX_s010
                                 + 1.42827 w_k1_IBM_s010
                                 + 1.15564 w_k1_INTC_s010
                                 + 1.42548 w_k1_KO_s010 + 1.44115 w_k1_MAT_s010
                                 + 1.31286 w_k1_MSFT_s010
                                 + 1.13791 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.928909 w_k1_ADM_s010
                                 + 0.698406 w_k1_AXP_s010
                                 + 0.693832 w_k1_DIS_s010
                                 + 0.724397 w_k1_FDX_s010
                                 + 0.961752 w_k1_IBM_s010
                                 + 0.83809 w_k1_INTC_s010
                                 + 0.890977 w_k1_KO_s010
                                 + 0.895303 w_k1_MAT_s010
                                 + 0.919644 w_k1_MSFT_s010
                                 + 0.947027 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.07757 w_k1_ADM_s010
                                 + 0.967004 w_k1_AXP_s010
                                 + 1.05667 w_k1_DIS_s010
                                 + 1.10593 w_k1_FDX_s010
                                 + 0.705326 w_k1_IBM_s010
                                 + 1.32167 w_k1_INTC_s010
                                 + 0.874857 w_k1_KO_s010
                                 + 0.662247 w_k1_MAT_s010
                                 + 1.17617 w_k1_MSFT_s010
                                 + 1.06279 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.28029 w_k1_ADM_s010
                                 + 1.18106 w_k1_AXP_s010
                                 + 1.15727 w_k1_DIS_s010
                                 + 1.20498 w_k1_FDX_s010
                                 + 1.01488 w_k1_IBM_s010
                                 + 1.19018 w_k1_INTC_s010
                                 + 1.11572 w_k1_KO_s010 + 1.11509 w_k1_MAT_s010
                                 + 1.25354 w_k1_MSFT_s010 + 1.1621 w_k1_PG_s010
                                 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.02731 w_k1_ADM_s010
                                 + 0.694218 w_k1_AXP_s010
                                 + 1.0203 w_k1_DIS_s010
                                 + 0.955274 w_k1_FDX_s010
                                 + 0.952397 w_k1_IBM_s010
                                 + 0.903491 w_k1_INTC_s010
                                 + 1.18144 w_k1_KO_s010
                                 + 0.741009 w_k1_MAT_s010
                                 + 1.12871 w_k1_MSFT_s010 + 1.0284 w_k1_PG_s010
                                 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.665602 w_k1_ADM_s011
                                 + 0.891807 w_k1_AXP_s011
                                 + 0.919071 w_k1_DIS_s011
                                 + 0.979672 w_k1_FDX_s011
                                 + 1.22673 w_k1_IBM_s011
                                 + 1.22937 w_k1_INTC_s011
                                 + 1.00778 w_k1_KO_s011 + 1.31068 w_k1_MAT_s011
                                 + 1.04397 w_k1_MSFT_s011
                                 + 1.03666 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.92005 w_k1_ADM_s011
                                 + 1.32543 w_k1_AXP_s011
                                 + 1.24499 w_k1_DIS_s011
                                 + 1.50104 w_k1_FDX_s011
                                 + 1.03784 w_k1_IBM_s011
                                 + 1.50333 w_k1_INTC_s011
                                 + 1.07418 w_k1_KO_s011 + 1.17477 w_k1_MAT_s011
                                 + 1.40635 w_k1_MSFT_s011
                                 + 1.28219 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.626797 w_k1_ADM_s011
                                 + 0.634989 w_k1_AXP_s011
                                 + 0.973014 w_k1_DIS_s011
                                 + 0.942206 w_k1_FDX_s011
                                 + 0.885767 w_k1_IBM_s011
                                 + 1.36078 w_k1_INTC_s011
                                 + 0.982848 w_k1_KO_s011
                                 + 0.71391 w_k1_MAT_s011
                                 + 1.20181 w_k1_MSFT_s011
                                 + 0.993123 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.17556 w_k1_ADM_s011
                                 + 1.46018 w_k1_AXP_s011
                                 + 1.10346 w_k1_DIS_s011
                                 + 1.17629 w_k1_FDX_s011
                                 + 1.76648 w_k1_IBM_s011
                                 + 1.28909 w_k1_INTC_s011
                                 + 1.29616 w_k1_KO_s011 + 2.12616 w_k1_MAT_s011
                                 + 1.2025 w_k1_MSFT_s011 + 1.30362 w_k1_PG_s011
                                 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.759234 w_k1_ADM_s011
                                 + 0.616036 w_k1_AXP_s011
                                 + 0.636919 w_k1_DIS_s011
                                 + 0.856948 w_k1_FDX_s011
                                 + 0.54032 w_k1_IBM_s011
                                 + 1.07554 w_k1_INTC_s011
                                 + 0.712141 w_k1_KO_s011
                                 + 0.495123 w_k1_MAT_s011
                                 + 0.88815 w_k1_MSFT_s011
                                 + 0.828748 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.831861 w_k1_ADM_s011
                                 + 0.819153 w_k1_AXP_s011
                                 + 0.692927 w_k1_DIS_s011
                                 + 0.81188 w_k1_FDX_s011
                                 + 0.845232 w_k1_IBM_s011
                                 + 0.891081 w_k1_INTC_s011
                                 + 0.728642 w_k1_KO_s011
                                 + 0.983313 w_k1_MAT_s011
                                 + 0.866325 w_k1_MSFT_s011
                                 + 0.877956 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.878661 w_k1_ADM_s011
                                 + 1.09603 w_k1_AXP_s011
                                 + 0.968158 w_k1_DIS_s011
                                 + 1.23415 w_k1_FDX_s011
                                 + 0.685974 w_k1_IBM_s011
                                 + 1.10864 w_k1_INTC_s011
                                 + 0.851094 w_k1_KO_s011
                                 + 0.850233 w_k1_MAT_s011
                                 + 1.23364 w_k1_MSFT_s011
                                 + 0.989116 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.883841 w_k1_ADM_s011
                                 + 1.16552 w_k1_AXP_s011
                                 + 1.68334 w_k1_DIS_s011
                                 + 1.30269 w_k1_FDX_s011
                                 + 1.18071 w_k1_IBM_s011
                                 + 1.44079 w_k1_INTC_s011
                                 + 1.30292 w_k1_KO_s011 + 1.45168 w_k1_MAT_s011
                                 + 1.74937 w_k1_MSFT_s011
                                 + 1.20245 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.22497 w_k1_ADM_s012
                                 + 1.40105 w_k1_AXP_s012
                                 + 1.24183 w_k1_DIS_s012
                                 + 1.33017 w_k1_FDX_s012
                                 + 0.833694 w_k1_IBM_s012
                                 + 1.1762 w_k1_INTC_s012
                                 + 0.932413 w_k1_KO_s012
                                 + 0.880798 w_k1_MAT_s012
                                 + 1.07323 w_k1_MSFT_s012
                                 + 0.962548 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.776398 w_k1_ADM_s012
                                 + 0.653668 w_k1_AXP_s012
                                 + 0.698001 w_k1_DIS_s012
                                 + 0.821587 w_k1_FDX_s012
                                 + 0.730927 w_k1_IBM_s012
                                 + 0.663271 w_k1_INTC_s012
                                 + 0.59737 w_k1_KO_s012 + 0.66458 w_k1_MAT_s012
                                 + 0.744929 w_k1_MSFT_s012
                                 + 0.738741 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.08906 w_k1_ADM_s012
                                 + 1.01165 w_k1_AXP_s012
                                 + 1.02298 w_k1_DIS_s012
                                 + 0.959793 w_k1_FDX_s012
                                 + 1.34046 w_k1_IBM_s012
                                 + 0.741388 w_k1_INTC_s012
                                 + 1.05952 w_k1_KO_s012 + 1.53828 w_k1_MAT_s012
                                 + 0.90247 w_k1_MSFT_s012
                                 + 1.00968 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.0261 w_k1_ADM_s012
                                 + 1.07355 w_k1_AXP_s012
                                 + 1.17969 w_k1_DIS_s012
                                 + 1.05218 w_k1_FDX_s012
                                 + 1.03105 w_k1_IBM_s012
                                 + 1.0231 w_k1_INTC_s012
                                 + 0.912554 w_k1_KO_s012
                                 + 0.819882 w_k1_MAT_s012
                                 + 0.956503 w_k1_MSFT_s012
                                 + 0.922387 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.13568 w_k1_ADM_s012
                                 + 1.56336 w_k1_AXP_s012
                                 + 1.73389 w_k1_DIS_s012 + 1.2312 w_k1_FDX_s012
                                 + 1.57362 w_k1_IBM_s012
                                 + 1.11207 w_k1_INTC_s012 + 1.4588 w_k1_KO_s012
                                 + 1.78992 w_k1_MAT_s012
                                 + 1.07814 w_k1_MSFT_s012
                                 + 1.14033 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.938602 w_k1_ADM_s012
                                 + 0.842044 w_k1_AXP_s012
                                 + 0.85111 w_k1_DIS_s012
                                 + 0.840246 w_k1_FDX_s012
                                 + 0.985862 w_k1_IBM_s012
                                 + 1.03163 w_k1_INTC_s012
                                 + 0.910935 w_k1_KO_s012
                                 + 0.926415 w_k1_MAT_s012
                                 + 0.909767 w_k1_MSFT_s012
                                 + 0.846008 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.930662 w_k1_ADM_s012
                                 + 0.96326 w_k1_AXP_s012
                                 + 0.957826 w_k1_DIS_s012
                                 + 1.03767 w_k1_FDX_s012
                                 + 0.639176 w_k1_IBM_s012
                                 + 0.919328 w_k1_INTC_s012
                                 + 0.77211 w_k1_KO_s012
                                 + 0.609442 w_k1_MAT_s012
                                 + 0.947681 w_k1_MSFT_s012
                                 + 0.800349 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.26829 w_k1_ADM_s012
                                 + 1.15676 w_k1_AXP_s012
                                 + 1.56945 w_k1_DIS_s012 + 1.3109 w_k1_FDX_s012
                                 + 1.03403 w_k1_IBM_s012
                                 + 1.19768 w_k1_INTC_s012
                                 + 1.26457 w_k1_KO_s012
                                 + 0.939595 w_k1_MAT_s012
                                 + 1.15843 w_k1_MSFT_s012
                                 + 1.20047 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.12366 w_k1_ADM_s013
                                 + 1.5064 w_k1_AXP_s013 + 1.40148 w_k1_DIS_s013
                                 + 1.36166 w_k1_FDX_s013
                                 + 1.45658 w_k1_IBM_s013
                                 + 1.26576 w_k1_INTC_s013
                                 + 1.31647 w_k1_KO_s013 + 1.66732 w_k1_MAT_s013
                                 + 1.28366 w_k1_MSFT_s013
                                 + 1.20101 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.0502 w_k1_ADM_s013
                                 + 0.994237 w_k1_AXP_s013
                                 + 1.11302 w_k1_DIS_s013
                                 + 1.16835 w_k1_FDX_s013
                                 + 0.749785 w_k1_IBM_s013
                                 + 1.03913 w_k1_INTC_s013
                                 + 0.966737 w_k1_KO_s013
                                 + 0.905922 w_k1_MAT_s013
                                 + 1.11883 w_k1_MSFT_s013
                                 + 1.02054 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.13024 w_k1_ADM_s013
                                 + 1.87158 w_k1_AXP_s013
                                 + 2.05818 w_k1_DIS_s013 + 1.4839 w_k1_FDX_s013
                                 + 1.18431 w_k1_IBM_s013
                                 + 1.36074 w_k1_INTC_s013
                                 + 1.68466 w_k1_KO_s013 + 1.19333 w_k1_MAT_s013
                                 + 1.61676 w_k1_MSFT_s013
                                 + 1.35129 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.918621 w_k1_ADM_s013
                                 + 1.32515 w_k1_AXP_s013
                                 + 1.27781 w_k1_DIS_s013
                                 + 1.17373 w_k1_FDX_s013
                                 + 1.01848 w_k1_IBM_s013
                                 + 1.30075 w_k1_INTC_s013 + 1.0705 w_k1_KO_s013
                                 + 1.07829 w_k1_MAT_s013
                                 + 1.25561 w_k1_MSFT_s013
                                 + 1.08091 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.943007 w_k1_ADM_s013
                                 + 1.08267 w_k1_AXP_s013
                                 + 1.11598 w_k1_DIS_s013
                                 + 1.04307 w_k1_FDX_s013
                                 + 1.51672 w_k1_IBM_s013
                                 + 1.1128 w_k1_INTC_s013 + 1.39278 w_k1_KO_s013
                                 + 1.6403 w_k1_MAT_s013
                                 + 1.25956 w_k1_MSFT_s013
                                 + 1.13519 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.11503 w_k1_ADM_s013
                                 + 1.12726 w_k1_AXP_s013
                                 + 1.06707 w_k1_DIS_s013
                                 + 1.04893 w_k1_FDX_s013
                                 + 1.14248 w_k1_IBM_s013
                                 + 1.28826 w_k1_INTC_s013
                                 + 1.05318 w_k1_KO_s013
                                 + 0.882599 w_k1_MAT_s013
                                 + 1.13335 w_k1_MSFT_s013
                                 + 0.864664 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.481709 w_k1_ADM_s013
                                 + 0.74396 w_k1_AXP_s013
                                 + 0.812834 w_k1_DIS_s013
                                 + 0.860178 w_k1_FDX_s013
                                 + 1.14312 w_k1_IBM_s013
                                 + 0.870501 w_k1_INTC_s013
                                 + 0.888125 w_k1_KO_s013
                                 + 1.12378 w_k1_MAT_s013
                                 + 0.933267 w_k1_MSFT_s013
                                 + 0.745816 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.716717 w_k1_ADM_s013
                                 + 0.852673 w_k1_AXP_s013
                                 + 0.702238 w_k1_DIS_s013
                                 + 0.896306 w_k1_FDX_s013
                                 + 0.802704 w_k1_IBM_s013
                                 + 0.946797 w_k1_INTC_s013
                                 + 0.829972 w_k1_KO_s013
                                 + 1.03174 w_k1_MAT_s013
                                 + 0.934374 w_k1_MSFT_s013
                                 + 0.840852 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.878755 w_k1_ADM_s014
                                 + 0.839844 w_k1_AXP_s014
                                 + 0.987649 w_k1_DIS_s014
                                 + 0.907034 w_k1_FDX_s014
                                 + 1.18431 w_k1_IBM_s014
                                 + 0.914348 w_k1_INTC_s014
                                 + 1.11571 w_k1_KO_s014 + 1.16117 w_k1_MAT_s014
                                 + 1.17742 w_k1_MSFT_s014
                                 + 1.12815 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.865428 w_k1_ADM_s014
                                 + 1.35408 w_k1_AXP_s014
                                 + 1.20401 w_k1_DIS_s014
                                 + 1.05055 w_k1_FDX_s014
                                 + 1.58629 w_k1_IBM_s014
                                 + 1.10656 w_k1_INTC_s014
                                 + 1.28913 w_k1_KO_s014 + 1.74257 w_k1_MAT_s014
                                 + 1.1523 w_k1_MSFT_s014 + 1.03887 w_k1_PG_s014
                                 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.13304 w_k1_ADM_s014
                                 + 1.26789 w_k1_AXP_s014
                                 + 1.22992 w_k1_DIS_s014
                                 + 1.25845 w_k1_FDX_s014
                                 + 1.09652 w_k1_IBM_s014
                                 + 0.977376 w_k1_INTC_s014
                                 + 1.1174 w_k1_KO_s014 + 1.12586 w_k1_MAT_s014
                                 + 1.23909 w_k1_MSFT_s014
                                 + 0.999879 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.961166 w_k1_ADM_s014
                                 + 0.974525 w_k1_AXP_s014
                                 + 0.911628 w_k1_DIS_s014
                                 + 1.04905 w_k1_FDX_s014
                                 + 0.856154 w_k1_IBM_s014
                                 + 1.00702 w_k1_INTC_s014
                                 + 0.943276 w_k1_KO_s014
                                 + 0.946547 w_k1_MAT_s014
                                 + 1.26284 w_k1_MSFT_s014
                                 + 0.97764 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.90352 w_k1_ADM_s014
                                 + 0.760327 w_k1_AXP_s014
                                 + 0.611299 w_k1_DIS_s014
                                 + 0.767092 w_k1_FDX_s014
                                 + 0.747574 w_k1_IBM_s014
                                 + 0.777772 w_k1_INTC_s014
                                 + 0.811954 w_k1_KO_s014
                                 + 0.668684 w_k1_MAT_s014
                                 + 0.881463 w_k1_MSFT_s014
                                 + 0.828493 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.747792 w_k1_ADM_s014
                                 + 0.813912 w_k1_AXP_s014
                                 + 0.821015 w_k1_DIS_s014
                                 + 0.811031 w_k1_FDX_s014
                                 + 1.05737 w_k1_IBM_s014
                                 + 0.901622 w_k1_INTC_s014
                                 + 0.968793 w_k1_KO_s014
                                 + 0.972968 w_k1_MAT_s014
                                 + 0.919713 w_k1_MSFT_s014
                                 + 0.94827 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.09365 w_k1_ADM_s014
                                 + 1.55856 w_k1_AXP_s014
                                 + 1.34021 w_k1_DIS_s014
                                 + 1.26893 w_k1_FDX_s014
                                 + 0.911021 w_k1_IBM_s014
                                 + 1.26639 w_k1_INTC_s014
                                 + 1.03585 w_k1_KO_s014 + 1.00511 w_k1_MAT_s014
                                 + 1.3971 w_k1_MSFT_s014 + 1.0943 w_k1_PG_s014
                                 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.987944 w_k1_ADM_s014
                                 + 1.09592 w_k1_AXP_s014
                                 + 1.16224 w_k1_DIS_s014
                                 + 1.06887 w_k1_FDX_s014
                                 + 1.21776 w_k1_IBM_s014
                                 + 0.801976 w_k1_INTC_s014
                                 + 1.07064 w_k1_KO_s014 + 1.51694 w_k1_MAT_s014
                                 + 1.0047 w_k1_MSFT_s014 + 1.01747 w_k1_PG_s014
                                 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.15892 w_k1_ADM_s015
                                 + 0.869087 w_k1_AXP_s015
                                 + 0.887616 w_k1_DIS_s015
                                 + 0.945944 w_k1_FDX_s015
                                 + 0.605713 w_k1_IBM_s015
                                 + 0.972097 w_k1_INTC_s015
                                 + 0.906656 w_k1_KO_s015
                                 + 0.782082 w_k1_MAT_s015
                                 + 0.993715 w_k1_MSFT_s015
                                 + 0.930647 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.47647 w_k1_ADM_s015
                                 + 1.31157 w_k1_AXP_s015 + 1.4792 w_k1_DIS_s015
                                 + 1.56626 w_k1_FDX_s015
                                 + 0.670745 w_k1_IBM_s015
                                 + 1.14956 w_k1_INTC_s015
                                 + 1.02467 w_k1_KO_s015
                                 + 0.753048 w_k1_MAT_s015
                                 + 1.23545 w_k1_MSFT_s015
                                 + 1.04406 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.780306 w_k1_ADM_s015
                                 + 0.915645 w_k1_AXP_s015
                                 + 0.801478 w_k1_DIS_s015
                                 + 0.871088 w_k1_FDX_s015
                                 + 1.41389 w_k1_IBM_s015
                                 + 0.945851 w_k1_INTC_s015
                                 + 1.07678 w_k1_KO_s015 + 1.49669 w_k1_MAT_s015
                                 + 0.861378 w_k1_MSFT_s015
                                 + 0.896673 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.871468 w_k1_ADM_s015
                                 + 0.909674 w_k1_AXP_s015
                                 + 0.85852 w_k1_DIS_s015
                                 + 0.924399 w_k1_FDX_s015
                                 + 0.9629 w_k1_IBM_s015
                                 + 1.26392 w_k1_INTC_s015
                                 + 0.940624 w_k1_KO_s015
                                 + 0.924789 w_k1_MAT_s015
                                 + 0.841016 w_k1_MSFT_s015
                                 + 0.929269 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.773093 w_k1_ADM_s015
                                 + 0.690979 w_k1_AXP_s015
                                 + 0.654418 w_k1_DIS_s015
                                 + 0.803793 w_k1_FDX_s015
                                 + 0.720053 w_k1_IBM_s015
                                 + 0.829938 w_k1_INTC_s015
                                 + 0.695292 w_k1_KO_s015
                                 + 0.786936 w_k1_MAT_s015
                                 + 0.754054 w_k1_MSFT_s015
                                 + 0.802392 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.967798 w_k1_ADM_s015
                                 + 1.19931 w_k1_AXP_s015
                                 + 0.954011 w_k1_DIS_s015
                                 + 1.0427 w_k1_FDX_s015 + 1.16271 w_k1_IBM_s015
                                 + 1.0682 w_k1_INTC_s015 + 1.22294 w_k1_KO_s015
                                 + 1.44342 w_k1_MAT_s015
                                 + 0.925323 w_k1_MSFT_s015
                                 + 1.0423 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.17925 w_k1_ADM_s015
                                 + 1.11665 w_k1_AXP_s015 + 1.3436 w_k1_DIS_s015
                                 + 1.21759 w_k1_FDX_s015
                                 + 1.03867 w_k1_IBM_s015
                                 + 1.4632 w_k1_INTC_s015 + 1.26593 w_k1_KO_s015
                                 + 0.891666 w_k1_MAT_s015
                                 + 1.25658 w_k1_MSFT_s015
                                 + 1.16691 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.838226 w_k1_ADM_s015
                                 + 0.669518 w_k1_AXP_s015
                                 + 0.544242 w_k1_DIS_s015
                                 + 0.633348 w_k1_FDX_s015
                                 + 0.993471 w_k1_IBM_s015
                                 + 0.792423 w_k1_INTC_s015
                                 + 0.908252 w_k1_KO_s015
                                 + 0.894538 w_k1_MAT_s015
                                 + 0.694836 w_k1_MSFT_s015
                                 + 0.854334 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.790172 w_k1_ADM_s016
                                 + 1.25532 w_k1_AXP_s016
                                 + 1.05052 w_k1_DIS_s016
                                 + 1.04228 w_k1_FDX_s016
                                 + 1.30857 w_k1_IBM_s016
                                 + 1.05666 w_k1_INTC_s016
                                 + 1.05796 w_k1_KO_s016 + 1.36127 w_k1_MAT_s016
                                 + 1.00564 w_k1_MSFT_s016
                                 + 1.00881 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.904052 w_k1_ADM_s016
                                 + 1.55847 w_k1_AXP_s016
                                 + 1.43554 w_k1_DIS_s016 + 1.3145 w_k1_FDX_s016
                                 + 1.53668 w_k1_IBM_s016
                                 + 1.52442 w_k1_INTC_s016
                                 + 1.30114 w_k1_KO_s016 + 2.0883 w_k1_MAT_s016
                                 + 1.35101 w_k1_MSFT_s016 + 1.0907 w_k1_PG_s016
                                 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.691096 w_k1_ADM_s016
                                 + 0.612801 w_k1_AXP_s016
                                 + 0.779522 w_k1_DIS_s016
                                 + 0.870301 w_k1_FDX_s016
                                 + 0.766457 w_k1_IBM_s016
                                 + 0.714196 w_k1_INTC_s016
                                 + 0.676513 w_k1_KO_s016
                                 + 0.567376 w_k1_MAT_s016
                                 + 0.860492 w_k1_MSFT_s016
                                 + 0.761971 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.949517 w_k1_ADM_s016
                                 + 0.982396 w_k1_AXP_s016
                                 + 0.834425 w_k1_DIS_s016
                                 + 0.957171 w_k1_FDX_s016
                                 + 0.975254 w_k1_IBM_s016
                                 + 0.801224 w_k1_INTC_s016
                                 + 0.853356 w_k1_KO_s016
                                 + 0.993205 w_k1_MAT_s016
                                 + 0.93859 w_k1_MSFT_s016
                                 + 0.837443 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.648886 w_k1_ADM_s016
                                 + 0.69391 w_k1_AXP_s016
                                 + 0.631878 w_k1_DIS_s016
                                 + 0.722103 w_k1_FDX_s016
                                 + 0.730257 w_k1_IBM_s016
                                 + 0.653861 w_k1_INTC_s016
                                 + 0.634622 w_k1_KO_s016
                                 + 0.645177 w_k1_MAT_s016
                                 + 0.720592 w_k1_MSFT_s016
                                 + 0.773694 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.951276 w_k1_ADM_s016
                                 + 1.0548 w_k1_AXP_s016
                                 + 0.903776 w_k1_DIS_s016
                                 + 1.07028 w_k1_FDX_s016
                                 + 0.799189 w_k1_IBM_s016
                                 + 1.13986 w_k1_INTC_s016
                                 + 0.853188 w_k1_KO_s016
                                 + 1.01689 w_k1_MAT_s016
                                 + 1.12396 w_k1_MSFT_s016
                                 + 0.897136 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.28371 w_k1_ADM_s016
                                 + 1.44808 w_k1_AXP_s016
                                 + 1.40379 w_k1_DIS_s016
                                 + 1.40498 w_k1_FDX_s016
                                 + 1.03793 w_k1_IBM_s016
                                 + 1.06739 w_k1_INTC_s016
                                 + 1.08607 w_k1_KO_s016 + 1.09488 w_k1_MAT_s016
                                 + 1.41264 w_k1_MSFT_s016
                                 + 1.02837 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.659971 w_k1_ADM_s016
                                 + 0.942318 w_k1_AXP_s016
                                 + 0.860429 w_k1_DIS_s016
                                 + 0.831232 w_k1_FDX_s016
                                 + 1.19597 w_k1_IBM_s016
                                 + 0.906978 w_k1_INTC_s016
                                 + 0.926395 w_k1_KO_s016
                                 + 1.30476 w_k1_MAT_s016
                                 + 0.932317 w_k1_MSFT_s016
                                 + 0.870294 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.00333 w_k1_ADM_s017
                                 + 1.17317 w_k1_AXP_s017
                                 + 0.999759 w_k1_DIS_s017
                                 + 0.941998 w_k1_FDX_s017
                                 + 1.28528 w_k1_IBM_s017
                                 + 0.896403 w_k1_INTC_s017
                                 + 0.989647 w_k1_KO_s017
                                 + 1.03515 w_k1_MAT_s017
                                 + 0.960293 w_k1_MSFT_s017
                                 + 1.08976 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.797808 w_k1_ADM_s017
                                 + 0.620756 w_k1_AXP_s017
                                 + 0.565263 w_k1_DIS_s017
                                 + 0.663662 w_k1_FDX_s017
                                 + 0.690421 w_k1_IBM_s017
                                 + 0.604731 w_k1_INTC_s017
                                 + 0.791011 w_k1_KO_s017
                                 + 0.840347 w_k1_MAT_s017
                                 + 0.780989 w_k1_MSFT_s017
                                 + 0.826045 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.910597 w_k1_ADM_s017
                                 + 0.677267 w_k1_AXP_s017
                                 + 0.612405 w_k1_DIS_s017
                                 + 0.775614 w_k1_FDX_s017
                                 + 0.949828 w_k1_IBM_s017
                                 + 0.721186 w_k1_INTC_s017
                                 + 0.730514 w_k1_KO_s017
                                 + 0.829482 w_k1_MAT_s017
                                 + 0.809603 w_k1_MSFT_s017
                                 + 0.892377 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.51175 w_k1_ADM_s017
                                 + 1.28026 w_k1_AXP_s017
                                 + 1.43725 w_k1_DIS_s017
                                 + 1.36049 w_k1_FDX_s017
                                 + 0.917361 w_k1_IBM_s017
                                 + 1.06927 w_k1_INTC_s017
                                 + 0.962307 w_k1_KO_s017
                                 + 0.869402 w_k1_MAT_s017
                                 + 1.06076 w_k1_MSFT_s017
                                 + 1.19396 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.09859 w_k1_ADM_s017
                                 + 0.877147 w_k1_AXP_s017
                                 + 0.863636 w_k1_DIS_s017
                                 + 0.759917 w_k1_FDX_s017
                                 + 1.09736 w_k1_IBM_s017
                                 + 0.721709 w_k1_INTC_s017
                                 + 0.780925 w_k1_KO_s017
                                 + 0.875203 w_k1_MAT_s017
                                 + 0.790641 w_k1_MSFT_s017
                                 + 0.848787 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.929346 w_k1_ADM_s017
                                 + 0.939701 w_k1_AXP_s017
                                 + 1.09739 w_k1_DIS_s017
                                 + 0.984911 w_k1_FDX_s017
                                 + 0.976728 w_k1_IBM_s017
                                 + 0.906906 w_k1_INTC_s017
                                 + 1.08735 w_k1_KO_s017
                                 + 0.904356 w_k1_MAT_s017
                                 + 1.01864 w_k1_MSFT_s017
                                 + 1.04442 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.02682 w_k1_ADM_s017
                                 + 0.835877 w_k1_AXP_s017
                                 + 0.892702 w_k1_DIS_s017
                                 + 0.95264 w_k1_FDX_s017
                                 + 0.716174 w_k1_IBM_s017
                                 + 0.748845 w_k1_INTC_s017
                                 + 0.80203 w_k1_KO_s017
                                 + 0.776139 w_k1_MAT_s017
                                 + 0.816253 w_k1_MSFT_s017
                                 + 0.971111 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.42901 w_k1_ADM_s017
                                 + 1.04841 w_k1_AXP_s017
                                 + 1.38115 w_k1_DIS_s017
                                 + 1.18464 w_k1_FDX_s017
                                 + 1.46256 w_k1_IBM_s017
                                 + 0.778011 w_k1_INTC_s017
                                 + 1.34861 w_k1_KO_s017 + 1.42609 w_k1_MAT_s017
                                 + 1.08039 w_k1_MSFT_s017
                                 + 1.14754 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.26477 w_k1_ADM_s100
                                 + 1.32611 w_k1_AXP_s100
                                 + 1.12118 w_k1_DIS_s100
                                 + 1.24074 w_k1_FDX_s100
                                 + 0.819923 w_k1_IBM_s100
                                 + 1.05564 w_k1_INTC_s100
                                 + 0.977611 w_k1_KO_s100
                                 + 1.45463 w_k1_MAT_s100
                                 + 1.06778 w_k1_MSFT_s100
                                 + 1.04806 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.08414 w_k1_ADM_s100
                                 + 0.858468 w_k1_AXP_s100
                                 + 0.724118 w_k1_DIS_s100
                                 + 0.892967 w_k1_FDX_s100
                                 + 0.6213 w_k1_IBM_s100
                                 + 0.917627 w_k1_INTC_s100
                                 + 0.782719 w_k1_KO_s100
                                 + 0.797406 w_k1_MAT_s100
                                 + 0.971425 w_k1_MSFT_s100
                                 + 0.88126 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.30659 w_k1_ADM_s100
                                 + 0.989549 w_k1_AXP_s100
                                 + 1.30845 w_k1_DIS_s100
                                 + 1.12835 w_k1_FDX_s100
                                 + 1.12184 w_k1_IBM_s100
                                 + 1.21213 w_k1_INTC_s100
                                 + 1.20014 w_k1_KO_s100 + 1.15266 w_k1_MAT_s100
                                 + 1.14278 w_k1_MSFT_s100
                                 + 1.00357 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.742108 w_k1_ADM_s100
                                 + 0.650849 w_k1_AXP_s100
                                 + 0.568772 w_k1_DIS_s100
                                 + 0.686557 w_k1_FDX_s100
                                 + 0.814893 w_k1_IBM_s100
                                 + 0.795137 w_k1_INTC_s100
                                 + 0.751902 w_k1_KO_s100
                                 + 0.815252 w_k1_MAT_s100
                                 + 0.733927 w_k1_MSFT_s100
                                 + 0.791278 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.950315 w_k1_ADM_s100
                                 + 1.05865 w_k1_AXP_s100 + 0.979 w_k1_DIS_s100
                                 + 0.861463 w_k1_FDX_s100
                                 + 1.36916 w_k1_IBM_s100
                                 + 0.960679 w_k1_INTC_s100
                                 + 1.07611 w_k1_KO_s100 + 1.30807 w_k1_MAT_s100
                                 + 0.998897 w_k1_MSFT_s100
                                 + 0.847699 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.65751 w_k1_ADM_s100
                                 + 1.77121 w_k1_AXP_s100
                                 + 2.19524 w_k1_DIS_s100
                                 + 1.61593 w_k1_FDX_s100
                                 + 1.39136 w_k1_IBM_s100
                                 + 1.60305 w_k1_INTC_s100
                                 + 1.80755 w_k1_KO_s100 + 1.67543 w_k1_MAT_s100
                                 + 1.71509 w_k1_MSFT_s100
                                 + 1.27014 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.955734 w_k1_ADM_s100
                                 + 0.758079 w_k1_AXP_s100
                                 + 0.827131 w_k1_DIS_s100
                                 + 0.846284 w_k1_FDX_s100
                                 + 0.774747 w_k1_IBM_s100
                                 + 0.790076 w_k1_INTC_s100
                                 + 0.799071 w_k1_KO_s100
                                 + 0.656449 w_k1_MAT_s100
                                 + 0.919858 w_k1_MSFT_s100
                                 + 0.744732 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.874193 w_k1_ADM_s100
                                 + 0.940484 w_k1_AXP_s100
                                 + 1.01059 w_k1_DIS_s100
                                 + 1.01969 w_k1_FDX_s100
                                 + 0.822107 w_k1_IBM_s100
                                 + 1.04812 w_k1_INTC_s100
                                 + 0.950628 w_k1_KO_s100
                                 + 0.680474 w_k1_MAT_s100
                                 + 1.01282 w_k1_MSFT_s100 + 1.0043 w_k1_PG_s100
                                 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.961641 w_k1_ADM_s101
                                 + 0.819649 w_k1_AXP_s101
                                 + 0.925815 w_k1_DIS_s101
                                 + 0.940918 w_k1_FDX_s101
                                 + 0.840688 w_k1_IBM_s101
                                 + 1.15621 w_k1_INTC_s101 + 1.074 w_k1_KO_s101
                                 + 0.93366 w_k1_MAT_s101
                                 + 1.03951 w_k1_MSFT_s101
                                 + 1.00004 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.12509 w_k1_ADM_s101
                                 + 1.07315 w_k1_AXP_s101
                                 + 1.35093 w_k1_DIS_s101
                                 + 1.16755 w_k1_FDX_s101
                                 + 1.37181 w_k1_IBM_s101
                                 + 1.00049 w_k1_INTC_s101
                                 + 1.38762 w_k1_KO_s101 + 1.7876 w_k1_MAT_s101
                                 + 1.08865 w_k1_MSFT_s101
                                 + 1.14131 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.01174 w_k1_ADM_s101
                                 + 0.905027 w_k1_AXP_s101
                                 + 1.13996 w_k1_DIS_s101
                                 + 1.16478 w_k1_FDX_s101
                                 + 0.895602 w_k1_IBM_s101
                                 + 1.03296 w_k1_INTC_s101
                                 + 1.00334 w_k1_KO_s101
                                 + 0.696437 w_k1_MAT_s101
                                 + 0.948008 w_k1_MSFT_s101
                                 + 0.918323 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.01185 w_k1_ADM_s101
                                 + 1.27065 w_k1_AXP_s101
                                 + 0.854908 w_k1_DIS_s101
                                 + 1.02255 w_k1_FDX_s101
                                 + 1.05802 w_k1_IBM_s101
                                 + 0.89248 w_k1_INTC_s101
                                 + 0.991175 w_k1_KO_s101
                                 + 1.90032 w_k1_MAT_s101
                                 + 0.836713 w_k1_MSFT_s101
                                 + 1.06226 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.01568 w_k1_ADM_s101
                                 + 0.851169 w_k1_AXP_s101
                                 + 0.878025 w_k1_DIS_s101
                                 + 0.864152 w_k1_FDX_s101
                                 + 1.3265 w_k1_IBM_s101
                                 + 0.778364 w_k1_INTC_s101
                                 + 1.17489 w_k1_KO_s101 + 1.14691 w_k1_MAT_s101
                                 + 0.854602 w_k1_MSFT_s101
                                 + 0.922801 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.789824 w_k1_ADM_s101
                                 + 0.580328 w_k1_AXP_s101
                                 + 0.671462 w_k1_DIS_s101
                                 + 0.743015 w_k1_FDX_s101
                                 + 0.807373 w_k1_IBM_s101
                                 + 0.640163 w_k1_INTC_s101
                                 + 0.900902 w_k1_KO_s101
                                 + 0.86458 w_k1_MAT_s101
                                 + 0.80007 w_k1_MSFT_s101
                                 + 0.821738 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.46216 w_k1_ADM_s101
                                 + 1.59365 w_k1_AXP_s101 + 1.8915 w_k1_DIS_s101
                                 + 1.52629 w_k1_FDX_s101
                                 + 1.30268 w_k1_IBM_s101
                                 + 1.40872 w_k1_INTC_s101
                                 + 1.46178 w_k1_KO_s101 + 1.31071 w_k1_MAT_s101
                                 + 1.34958 w_k1_MSFT_s101 + 1.135 w_k1_PG_s101
                                 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.34075 w_k1_ADM_s101
                                 + 1.1273 w_k1_AXP_s101 + 1.39346 w_k1_DIS_s101
                                 + 1.33277 w_k1_FDX_s101
                                 + 0.986205 w_k1_IBM_s101
                                 + 1.09814 w_k1_INTC_s101
                                 + 1.27474 w_k1_KO_s101 + 1.26388 w_k1_MAT_s101
                                 + 1.31946 w_k1_MSFT_s101 + 1.0977 w_k1_PG_s101
                                 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.978756 w_k1_ADM_s102
                                 + 1.0836 w_k1_AXP_s102 + 1.02545 w_k1_DIS_s102
                                 + 1.13466 w_k1_FDX_s102
                                 + 0.767066 w_k1_IBM_s102
                                 + 1.01036 w_k1_INTC_s102
                                 + 0.767082 w_k1_KO_s102
                                 + 0.816953 w_k1_MAT_s102
                                 + 0.969418 w_k1_MSFT_s102
                                 + 0.882058 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.975396 w_k1_ADM_s102
                                 + 1.09234 w_k1_AXP_s102
                                 + 1.18927 w_k1_DIS_s102
                                 + 1.01669 w_k1_FDX_s102
                                 + 1.06069 w_k1_IBM_s102
                                 + 1.1868 w_k1_INTC_s102 + 1.05017 w_k1_KO_s102
                                 + 1.0454 w_k1_MAT_s102 + 1.0469 w_k1_MSFT_s102
                                 + 1.11555 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.907562 w_k1_ADM_s102
                                 + 0.902861 w_k1_AXP_s102
                                 + 0.945645 w_k1_DIS_s102
                                 + 1.06349 w_k1_FDX_s102
                                 + 0.868049 w_k1_IBM_s102
                                 + 0.852425 w_k1_INTC_s102
                                 + 0.951287 w_k1_KO_s102
                                 + 0.623525 w_k1_MAT_s102
                                 + 0.971056 w_k1_MSFT_s102
                                 + 0.952022 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.86735 w_k1_ADM_s102
                                 + 0.656844 w_k1_AXP_s102
                                 + 0.849675 w_k1_DIS_s102
                                 + 0.868107 w_k1_FDX_s102
                                 + 0.677319 w_k1_IBM_s102
                                 + 0.83243 w_k1_INTC_s102
                                 + 0.870726 w_k1_KO_s102
                                 + 0.407806 w_k1_MAT_s102
                                 + 0.885835 w_k1_MSFT_s102
                                 + 0.850937 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.93481 w_k1_ADM_s102
                                 + 1.38983 w_k1_AXP_s102
                                 + 1.24814 w_k1_DIS_s102
                                 + 1.08242 w_k1_FDX_s102
                                 + 1.52461 w_k1_IBM_s102
                                 + 1.13496 w_k1_INTC_s102
                                 + 1.25311 w_k1_KO_s102 + 1.19752 w_k1_MAT_s102
                                 + 1.06424 w_k1_MSFT_s102
                                 + 1.05676 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.778483 w_k1_ADM_s102
                                 + 0.771993 w_k1_AXP_s102
                                 + 0.705173 w_k1_DIS_s102
                                 + 0.765452 w_k1_FDX_s102
                                 + 0.912265 w_k1_IBM_s102
                                 + 0.809346 w_k1_INTC_s102 + 0.741 w_k1_KO_s102
                                 + 0.676664 w_k1_MAT_s102
                                 + 0.730747 w_k1_MSFT_s102
                                 + 0.784846 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.34547 w_k1_ADM_s102
                                 + 1.63483 w_k1_AXP_s102
                                 + 1.38932 w_k1_DIS_s102 + 1.3685 w_k1_FDX_s102
                                 + 0.952809 w_k1_IBM_s102
                                 + 1.08274 w_k1_INTC_s102
                                 + 1.10953 w_k1_KO_s102 + 1.24191 w_k1_MAT_s102
                                 + 1.19069 w_k1_MSFT_s102
                                 + 1.09585 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.935671 w_k1_ADM_s102
                                 + 1.04626 w_k1_AXP_s102
                                 + 0.906425 w_k1_DIS_s102
                                 + 0.956705 w_k1_FDX_s102
                                 + 0.875457 w_k1_IBM_s102
                                 + 0.914073 w_k1_INTC_s102
                                 + 0.908957 w_k1_KO_s102
                                 + 0.836037 w_k1_MAT_s102
                                 + 0.976822 w_k1_MSFT_s102
                                 + 0.95342 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.856508 w_k1_ADM_s103
                                 + 0.752523 w_k1_AXP_s103
                                 + 0.677587 w_k1_DIS_s103
                                 + 0.845044 w_k1_FDX_s103
                                 + 0.747761 w_k1_IBM_s103
                                 + 0.756383 w_k1_INTC_s103
                                 + 0.899581 w_k1_KO_s103
                                 + 0.511653 w_k1_MAT_s103
                                 + 0.701807 w_k1_MSFT_s103
                                 + 0.811802 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.981175 w_k1_ADM_s103
                                 + 0.812799 w_k1_AXP_s103
                                 + 0.769724 w_k1_DIS_s103
                                 + 0.94622 w_k1_FDX_s103
                                 + 0.67341 w_k1_IBM_s103
                                 + 0.740961 w_k1_INTC_s103
                                 + 0.737172 w_k1_KO_s103
                                 + 0.707253 w_k1_MAT_s103
                                 + 0.836724 w_k1_MSFT_s103
                                 + 0.795698 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.04436 w_k1_ADM_s103
                                 + 1.05261 w_k1_AXP_s103
                                 + 0.995986 w_k1_DIS_s103
                                 + 1.04096 w_k1_FDX_s103
                                 + 1.01231 w_k1_IBM_s103
                                 + 1.02592 w_k1_INTC_s103
                                 + 1.15079 w_k1_KO_s103 + 1.20641 w_k1_MAT_s103
                                 + 1.055 w_k1_MSFT_s103 + 1.04187 w_k1_PG_s103
                                 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.11422 w_k1_ADM_s103
                                 + 1.58307 w_k1_AXP_s103
                                 + 1.65489 w_k1_DIS_s103
                                 + 1.28083 w_k1_FDX_s103
                                 + 1.30781 w_k1_IBM_s103
                                 + 0.934978 w_k1_INTC_s103
                                 + 1.25597 w_k1_KO_s103 + 1.84838 w_k1_MAT_s103
                                 + 1.05107 w_k1_MSFT_s103
                                 + 1.13586 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.35882 w_k1_ADM_s103
                                 + 1.34134 w_k1_AXP_s103
                                 + 1.30925 w_k1_DIS_s103
                                 + 1.20488 w_k1_FDX_s103
                                 + 1.53335 w_k1_IBM_s103
                                 + 1.04663 w_k1_INTC_s103
                                 + 1.29923 w_k1_KO_s103 + 1.71525 w_k1_MAT_s103
                                 + 0.973839 w_k1_MSFT_s103
                                 + 1.07973 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.879489 w_k1_ADM_s103
                                 + 1.00607 w_k1_AXP_s103
                                 + 0.904106 w_k1_DIS_s103
                                 + 0.944913 w_k1_FDX_s103
                                 + 1.18039 w_k1_IBM_s103
                                 + 0.880708 w_k1_INTC_s103
                                 + 1.05792 w_k1_KO_s103 + 1.19576 w_k1_MAT_s103
                                 + 0.963081 w_k1_MSFT_s103
                                 + 0.936785 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.759515 w_k1_ADM_s103
                                 + 0.810338 w_k1_AXP_s103
                                 + 0.726396 w_k1_DIS_s103
                                 + 0.77448 w_k1_FDX_s103 + 1.0907 w_k1_IBM_s103
                                 + 0.752855 w_k1_INTC_s103
                                 + 1.01684 w_k1_KO_s103 + 1.24247 w_k1_MAT_s103
                                 + 0.783349 w_k1_MSFT_s103
                                 + 0.935128 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.01275 w_k1_ADM_s103
                                 + 1.31275 w_k1_AXP_s103
                                 + 1.16791 w_k1_DIS_s103
                                 + 1.22441 w_k1_FDX_s103
                                 + 0.849089 w_k1_IBM_s103
                                 + 1.06667 w_k1_INTC_s103
                                 + 1.09205 w_k1_KO_s103 + 1.00187 w_k1_MAT_s103
                                 + 0.902732 w_k1_MSFT_s103
                                 + 1.03539 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.27087 w_k1_ADM_s104
                                 + 1.49081 w_k1_AXP_s104
                                 + 1.08249 w_k1_DIS_s104
                                 + 1.09093 w_k1_FDX_s104
                                 + 1.00343 w_k1_IBM_s104
                                 + 0.97339 w_k1_INTC_s104
                                 + 1.29642 w_k1_KO_s104 + 1.76377 w_k1_MAT_s104
                                 + 1.07751 w_k1_MSFT_s104
                                 + 1.15179 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.19516 w_k1_ADM_s104
                                 + 1.67397 w_k1_AXP_s104 + 1.2672 w_k1_DIS_s104
                                 + 1.26618 w_k1_FDX_s104
                                 + 1.51546 w_k1_IBM_s104
                                 + 1.1148 w_k1_INTC_s104 + 1.41048 w_k1_KO_s104
                                 + 2.02027 w_k1_MAT_s104
                                 + 1.14168 w_k1_MSFT_s104
                                 + 1.22103 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.06889 w_k1_ADM_s104
                                 + 0.66526 w_k1_AXP_s104
                                 + 0.685431 w_k1_DIS_s104
                                 + 0.804577 w_k1_FDX_s104
                                 + 0.758948 w_k1_IBM_s104
                                 + 0.97871 w_k1_INTC_s104
                                 + 0.851141 w_k1_KO_s104
                                 + 0.786004 w_k1_MAT_s104
                                 + 0.898731 w_k1_MSFT_s104
                                 + 0.829281 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.33363 w_k1_ADM_s104
                                 + 1.16473 w_k1_AXP_s104
                                 + 1.26162 w_k1_DIS_s104
                                 + 1.25171 w_k1_FDX_s104
                                 + 0.994056 w_k1_IBM_s104
                                 + 1.20354 w_k1_INTC_s104
                                 + 1.14461 w_k1_KO_s104
                                 + 0.826774 w_k1_MAT_s104
                                 + 1.08905 w_k1_MSFT_s104
                                 + 1.15333 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 2.0396 w_k1_ADM_s104
                                 + 1.53133 w_k1_AXP_s104
                                 + 1.79159 w_k1_DIS_s104
                                 + 1.52633 w_k1_FDX_s104
                                 + 1.17169 w_k1_IBM_s104
                                 + 1.2803 w_k1_INTC_s104 + 1.59779 w_k1_KO_s104
                                 + 1.34622 w_k1_MAT_s104
                                 + 1.49335 w_k1_MSFT_s104
                                 + 1.22345 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.12195 w_k1_ADM_s104
                                 + 0.973282 w_k1_AXP_s104
                                 + 0.791779 w_k1_DIS_s104
                                 + 0.912897 w_k1_FDX_s104
                                 + 1.0408 w_k1_IBM_s104
                                 + 0.914137 w_k1_INTC_s104
                                 + 0.854495 w_k1_KO_s104
                                 + 1.08798 w_k1_MAT_s104
                                 + 0.898907 w_k1_MSFT_s104
                                 + 0.935643 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.04704 w_k1_ADM_s104
                                 + 1.10606 w_k1_AXP_s104
                                 + 1.31313 w_k1_DIS_s104
                                 + 1.03672 w_k1_FDX_s104
                                 + 1.27782 w_k1_IBM_s104
                                 + 1.06299 w_k1_INTC_s104
                                 + 1.27368 w_k1_KO_s104 + 1.59596 w_k1_MAT_s104
                                 + 1.08341 w_k1_MSFT_s104
                                 + 0.928671 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.04533 w_k1_ADM_s104
                                 + 0.994907 w_k1_AXP_s104
                                 + 0.946808 w_k1_DIS_s104
                                 + 1.12764 w_k1_FDX_s104
                                 + 0.762245 w_k1_IBM_s104
                                 + 1.14078 w_k1_INTC_s104
                                 + 0.969938 w_k1_KO_s104
                                 + 0.952905 w_k1_MAT_s104
                                 + 1.15503 w_k1_MSFT_s104
                                 + 1.04416 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.954276 w_k1_ADM_s105
                                 + 1.34875 w_k1_AXP_s105
                                 + 1.00721 w_k1_DIS_s105
                                 + 1.19222 w_k1_FDX_s105
                                 + 0.779232 w_k1_IBM_s105
                                 + 1.07886 w_k1_INTC_s105
                                 + 0.887084 w_k1_KO_s105
                                 + 1.00027 w_k1_MAT_s105
                                 + 1.20394 w_k1_MSFT_s105
                                 + 1.01113 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.677679 w_k1_ADM_s105
                                 + 0.636234 w_k1_AXP_s105
                                 + 0.645558 w_k1_DIS_s105
                                 + 0.832396 w_k1_FDX_s105
                                 + 0.62539 w_k1_IBM_s105
                                 + 0.846494 w_k1_INTC_s105
                                 + 0.681489 w_k1_KO_s105
                                 + 0.541429 w_k1_MAT_s105
                                 + 0.744269 w_k1_MSFT_s105
                                 + 0.835874 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.857961 w_k1_ADM_s105
                                 + 0.830488 w_k1_AXP_s105
                                 + 0.922242 w_k1_DIS_s105
                                 + 0.85344 w_k1_FDX_s105
                                 + 0.998707 w_k1_IBM_s105
                                 + 0.9734 w_k1_INTC_s105 + 0.93232 w_k1_KO_s105
                                 + 0.89541 w_k1_MAT_s105
                                 + 1.00299 w_k1_MSFT_s105
                                 + 0.920193 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.29345 w_k1_ADM_s105
                                 + 1.08184 w_k1_AXP_s105
                                 + 1.48549 w_k1_DIS_s105
                                 + 1.24653 w_k1_FDX_s105
                                 + 0.968586 w_k1_IBM_s105
                                 + 1.28172 w_k1_INTC_s105
                                 + 1.30733 w_k1_KO_s105 + 1.10168 w_k1_MAT_s105
                                 + 1.2981 w_k1_MSFT_s105 + 1.27758 w_k1_PG_s105
                                 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.01277 w_k1_ADM_s105
                                 + 1.20196 w_k1_AXP_s105
                                 + 1.27829 w_k1_DIS_s105
                                 + 1.07618 w_k1_FDX_s105
                                 + 1.29921 w_k1_IBM_s105
                                 + 1.0472 w_k1_INTC_s105 + 1.10338 w_k1_KO_s105
                                 + 1.15185 w_k1_MAT_s105
                                 + 1.03747 w_k1_MSFT_s105
                                 + 1.20327 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.28529 w_k1_ADM_s105
                                 + 1.77743 w_k1_AXP_s105
                                 + 1.94466 w_k1_DIS_s105 + 1.4093 w_k1_FDX_s105
                                 + 1.78721 w_k1_IBM_s105
                                 + 1.60251 w_k1_INTC_s105
                                 + 1.89358 w_k1_KO_s105 + 1.94155 w_k1_MAT_s105
                                 + 1.46909 w_k1_MSFT_s105
                                 + 1.49892 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.754911 w_k1_ADM_s105
                                 + 0.812733 w_k1_AXP_s105
                                 + 0.732712 w_k1_DIS_s105
                                 + 0.794914 w_k1_FDX_s105
                                 + 0.843168 w_k1_IBM_s105
                                 + 0.95725 w_k1_INTC_s105
                                 + 0.792464 w_k1_KO_s105
                                 + 0.724689 w_k1_MAT_s105
                                 + 0.905492 w_k1_MSFT_s105
                                 + 0.920774 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.836697 w_k1_ADM_s105
                                 + 0.795917 w_k1_AXP_s105
                                 + 0.776635 w_k1_DIS_s105
                                 + 0.689518 w_k1_FDX_s105
                                 + 1.33672 w_k1_IBM_s105
                                 + 0.874062 w_k1_INTC_s105
                                 + 0.941586 w_k1_KO_s105
                                 + 1.00637 w_k1_MAT_s105
                                 + 0.732586 w_k1_MSFT_s105
                                 + 0.919844 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.24735 w_k1_ADM_s106
                                 + 1.53411 w_k1_AXP_s106
                                 + 1.66638 w_k1_DIS_s106
                                 + 1.37729 w_k1_FDX_s106
                                 + 1.39092 w_k1_IBM_s106
                                 + 1.45329 w_k1_INTC_s106
                                 + 1.41214 w_k1_KO_s106 + 1.70578 w_k1_MAT_s106
                                 + 1.24124 w_k1_MSFT_s106
                                 + 1.28251 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.901287 w_k1_ADM_s106
                                 + 0.960521 w_k1_AXP_s106
                                 + 0.872053 w_k1_DIS_s106
                                 + 0.855369 w_k1_FDX_s106
                                 + 1.31679 w_k1_IBM_s106
                                 + 0.88956 w_k1_INTC_s106
                                 + 1.06007 w_k1_KO_s106 + 1.59292 w_k1_MAT_s106
                                 + 0.874947 w_k1_MSFT_s106
                                 + 0.879751 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.71724 w_k1_ADM_s106
                                 + 0.582217 w_k1_AXP_s106
                                 + 0.600186 w_k1_DIS_s106
                                 + 0.717553 w_k1_FDX_s106
                                 + 0.755149 w_k1_IBM_s106
                                 + 0.694209 w_k1_INTC_s106
                                 + 0.729949 w_k1_KO_s106
                                 + 0.817974 w_k1_MAT_s106
                                 + 0.730393 w_k1_MSFT_s106
                                 + 0.779154 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.02239 w_k1_ADM_s106
                                 + 1.39246 w_k1_AXP_s106
                                 + 1.26584 w_k1_DIS_s106
                                 + 1.16297 w_k1_FDX_s106 + 1.1102 w_k1_IBM_s106
                                 + 1.1221 w_k1_INTC_s106 + 1.24505 w_k1_KO_s106
                                 + 1.71963 w_k1_MAT_s106
                                 + 1.14023 w_k1_MSFT_s106
                                 + 1.01966 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.992268 w_k1_ADM_s106
                                 + 0.881279 w_k1_AXP_s106
                                 + 0.978276 w_k1_DIS_s106
                                 + 1.00644 w_k1_FDX_s106
                                 + 0.623352 w_k1_IBM_s106
                                 + 0.959118 w_k1_INTC_s106
                                 + 0.882569 w_k1_KO_s106
                                 + 0.862273 w_k1_MAT_s106
                                 + 1.01932 w_k1_MSFT_s106
                                 + 0.95177 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.06671 w_k1_ADM_s106
                                 + 0.982289 w_k1_AXP_s106
                                 + 1.3297 w_k1_DIS_s106 + 1.11141 w_k1_FDX_s106
                                 + 0.929079 w_k1_IBM_s106
                                 + 1.11852 w_k1_INTC_s106
                                 + 1.11049 w_k1_KO_s106 + 1.12626 w_k1_MAT_s106
                                 + 1.07705 w_k1_MSFT_s106
                                 + 0.939888 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.802606 w_k1_ADM_s106
                                 + 0.747188 w_k1_AXP_s106
                                 + 0.998855 w_k1_DIS_s106
                                 + 0.901296 w_k1_FDX_s106
                                 + 0.90714 w_k1_IBM_s106
                                 + 1.08591 w_k1_INTC_s106
                                 + 1.05836 w_k1_KO_s106
                                 + 0.559694 w_k1_MAT_s106
                                 + 0.939043 w_k1_MSFT_s106
                                 + 0.865233 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.891489 w_k1_ADM_s106
                                 + 1.25781 w_k1_AXP_s106
                                 + 0.94652 w_k1_DIS_s106
                                 + 1.02805 w_k1_FDX_s106
                                 + 0.847978 w_k1_IBM_s106
                                 + 0.988214 w_k1_INTC_s106
                                 + 0.939192 w_k1_KO_s106
                                 + 0.971509 w_k1_MAT_s106
                                 + 0.989006 w_k1_MSFT_s106
                                 + 0.918517 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.29435 w_k1_ADM_s107
                                 + 1.12921 w_k1_AXP_s107
                                 + 1.12759 w_k1_DIS_s107
                                 + 1.25463 w_k1_FDX_s107
                                 + 0.852634 w_k1_IBM_s107
                                 + 1.15456 w_k1_INTC_s107
                                 + 0.834505 w_k1_KO_s107
                                 + 0.502342 w_k1_MAT_s107
                                 + 1.06882 w_k1_MSFT_s107
                                 + 0.952429 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.891665 w_k1_ADM_s107
                                 + 0.9 w_k1_AXP_s107 + 1.10841 w_k1_DIS_s107
                                 + 1.08796 w_k1_FDX_s107
                                 + 0.89878 w_k1_IBM_s107
                                 + 1.07399 w_k1_INTC_s107
                                 + 0.978834 w_k1_KO_s107
                                 + 0.689866 w_k1_MAT_s107
                                 + 1.11392 w_k1_MSFT_s107
                                 + 0.957824 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.953248 w_k1_ADM_s107
                                 + 0.866625 w_k1_AXP_s107
                                 + 0.824359 w_k1_DIS_s107
                                 + 0.865253 w_k1_FDX_s107
                                 + 1.41648 w_k1_IBM_s107
                                 + 0.885201 w_k1_INTC_s107
                                 + 1.11223 w_k1_KO_s107 + 1.11061 w_k1_MAT_s107
                                 + 0.993275 w_k1_MSFT_s107
                                 + 0.987251 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.895237 w_k1_ADM_s107
                                 + 1.03151 w_k1_AXP_s107
                                 + 0.857457 w_k1_DIS_s107
                                 + 0.880552 w_k1_FDX_s107
                                 + 1.07781 w_k1_IBM_s107
                                 + 0.792125 w_k1_INTC_s107
                                 + 0.872009 w_k1_KO_s107
                                 + 1.07694 w_k1_MAT_s107
                                 + 0.882896 w_k1_MSFT_s107
                                 + 0.933831 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.34945 w_k1_ADM_s107
                                 + 1.29531 w_k1_AXP_s107
                                 + 1.32576 w_k1_DIS_s107
                                 + 1.29633 w_k1_FDX_s107
                                 + 1.21556 w_k1_IBM_s107
                                 + 1.2502 w_k1_INTC_s107 + 1.28083 w_k1_KO_s107
                                 + 1.15786 w_k1_MAT_s107
                                 + 1.23836 w_k1_MSFT_s107
                                 + 1.19649 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.884688 w_k1_ADM_s107
                                 + 0.692089 w_k1_AXP_s107
                                 + 0.748923 w_k1_DIS_s107
                                 + 0.830111 w_k1_FDX_s107
                                 + 0.611077 w_k1_IBM_s107
                                 + 0.712158 w_k1_INTC_s107
                                 + 0.711341 w_k1_KO_s107
                                 + 0.480952 w_k1_MAT_s107
                                 + 0.785256 w_k1_MSFT_s107
                                 + 0.798167 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.832737 w_k1_ADM_s107
                                 + 0.84057 w_k1_AXP_s107
                                 + 0.821638 w_k1_DIS_s107
                                 + 0.847051 w_k1_FDX_s107
                                 + 0.91156 w_k1_IBM_s107
                                 + 0.891759 w_k1_INTC_s107
                                 + 0.721435 w_k1_KO_s107
                                 + 0.837072 w_k1_MAT_s107
                                 + 0.87122 w_k1_MSFT_s107
                                 + 0.85347 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.85959 w_k1_ADM_s107
                                 + 1.23306 w_k1_AXP_s107
                                 + 1.21347 w_k1_DIS_s107
                                 + 1.14797 w_k1_FDX_s107
                                 + 1.63561 w_k1_IBM_s107
                                 + 1.13668 w_k1_INTC_s107
                                 + 1.37348 w_k1_KO_s107 + 1.81574 w_k1_MAT_s107
                                 + 1.10449 w_k1_MSFT_s107
                                 + 1.15786 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.571492 w_k1_ADM_s110
                                 + 1.18448 w_k1_AXP_s110
                                 + 0.974131 w_k1_DIS_s110
                                 + 0.880905 w_k1_FDX_s110
                                 + 1.83302 w_k1_IBM_s110
                                 + 1.27171 w_k1_INTC_s110
                                 + 1.24245 w_k1_KO_s110 + 1.84216 w_k1_MAT_s110
                                 + 1.02955 w_k1_MSFT_s110
                                 + 1.20463 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.946238 w_k1_ADM_s110
                                 + 0.622798 w_k1_AXP_s110
                                 + 0.78976 w_k1_DIS_s110
                                 + 0.765644 w_k1_FDX_s110
                                 + 0.897704 w_k1_IBM_s110
                                 + 0.874347 w_k1_INTC_s110
                                 + 0.897484 w_k1_KO_s110
                                 + 0.677562 w_k1_MAT_s110
                                 + 0.931831 w_k1_MSFT_s110
                                 + 1.03723 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.10855 w_k1_ADM_s110
                                 + 1.0083 w_k1_AXP_s110 + 1.14831 w_k1_DIS_s110
                                 + 1.00349 w_k1_FDX_s110
                                 + 1.13633 w_k1_IBM_s110
                                 + 1.07366 w_k1_INTC_s110
                                 + 1.21652 w_k1_KO_s110 + 1.21814 w_k1_MAT_s110
                                 + 1.07329 w_k1_MSFT_s110
                                 + 1.30765 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.790102 w_k1_ADM_s110
                                 + 0.7759 w_k1_AXP_s110
                                 + 0.654775 w_k1_DIS_s110
                                 + 0.697341 w_k1_FDX_s110
                                 + 0.834932 w_k1_IBM_s110
                                 + 0.85406 w_k1_INTC_s110
                                 + 0.697298 w_k1_KO_s110
                                 + 0.667002 w_k1_MAT_s110
                                 + 0.69973 w_k1_MSFT_s110
                                 + 0.927823 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.989643 w_k1_ADM_s110
                                 + 1.04922 w_k1_AXP_s110
                                 + 1.49482 w_k1_DIS_s110
                                 + 1.14462 w_k1_FDX_s110
                                 + 1.01425 w_k1_IBM_s110
                                 + 1.19729 w_k1_INTC_s110
                                 + 1.25684 w_k1_KO_s110
                                 + 0.978118 w_k1_MAT_s110
                                 + 1.34355 w_k1_MSFT_s110
                                 + 1.33544 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.05704 w_k1_ADM_s110
                                 + 1.04332 w_k1_AXP_s110
                                 + 0.881526 w_k1_DIS_s110
                                 + 0.985582 w_k1_FDX_s110
                                 + 1.06121 w_k1_IBM_s110
                                 + 1.07791 w_k1_INTC_s110
                                 + 1.09653 w_k1_KO_s110 + 1.05446 w_k1_MAT_s110
                                 + 0.976541 w_k1_MSFT_s110
                                 + 1.17144 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.08461 w_k1_ADM_s110
                                 + 1.60934 w_k1_AXP_s110
                                 + 1.75693 w_k1_DIS_s110
                                 + 1.28547 w_k1_FDX_s110
                                 + 1.48418 w_k1_IBM_s110
                                 + 1.46728 w_k1_INTC_s110
                                 + 1.41073 w_k1_KO_s110 + 1.32609 w_k1_MAT_s110
                                 + 1.57082 w_k1_MSFT_s110
                                 + 1.36056 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.956833 w_k1_ADM_s110
                                 + 0.826788 w_k1_AXP_s110
                                 + 0.914114 w_k1_DIS_s110
                                 + 0.988802 w_k1_FDX_s110
                                 + 0.685391 w_k1_IBM_s110
                                 + 1.00137 w_k1_INTC_s110
                                 + 0.756658 w_k1_KO_s110
                                 + 0.483954 w_k1_MAT_s110
                                 + 0.938402 w_k1_MSFT_s110
                                 + 1.00991 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.984044 w_k1_ADM_s111
                                 + 0.711298 w_k1_AXP_s111
                                 + 0.607915 w_k1_DIS_s111
                                 + 0.747164 w_k1_FDX_s111
                                 + 0.856598 w_k1_IBM_s111
                                 + 0.84636 w_k1_INTC_s111
                                 + 0.681908 w_k1_KO_s111
                                 + 0.519153 w_k1_MAT_s111
                                 + 0.806769 w_k1_MSFT_s111
                                 + 0.772371 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.12431 w_k1_ADM_s111
                                 + 0.850678 w_k1_AXP_s111
                                 + 0.904387 w_k1_DIS_s111
                                 + 0.909242 w_k1_FDX_s111
                                 + 1.0102 w_k1_IBM_s111
                                 + 0.775247 w_k1_INTC_s111
                                 + 1.01778 w_k1_KO_s111 + 1.09139 w_k1_MAT_s111
                                 + 1.05646 w_k1_MSFT_s111
                                 + 0.998633 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.33509 w_k1_ADM_s111
                                 + 1.2073 w_k1_AXP_s111 + 1.17003 w_k1_DIS_s111
                                 + 1.13743 w_k1_FDX_s111
                                 + 1.13254 w_k1_IBM_s111
                                 + 0.908275 w_k1_INTC_s111
                                 + 0.945039 w_k1_KO_s111
                                 + 1.00948 w_k1_MAT_s111
                                 + 0.978629 w_k1_MSFT_s111
                                 + 0.989003 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.33512 w_k1_ADM_s111
                                 + 1.40843 w_k1_AXP_s111
                                 + 1.56184 w_k1_DIS_s111
                                 + 1.32737 w_k1_FDX_s111
                                 + 1.36127 w_k1_IBM_s111
                                 + 1.10855 w_k1_INTC_s111 + 1.134 w_k1_KO_s111
                                 + 1.33665 w_k1_MAT_s111
                                 + 1.28598 w_k1_MSFT_s111 + 1.1428 w_k1_PG_s111
                                 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.02095 w_k1_ADM_s111
                                 + 0.947025 w_k1_AXP_s111
                                 + 0.847642 w_k1_DIS_s111
                                 + 0.89036 w_k1_FDX_s111
                                 + 1.29256 w_k1_IBM_s111
                                 + 0.948516 w_k1_INTC_s111
                                 + 0.981751 w_k1_KO_s111
                                 + 1.04738 w_k1_MAT_s111
                                 + 0.95731 w_k1_MSFT_s111
                                 + 0.864965 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.21507 w_k1_ADM_s111
                                 + 0.940666 w_k1_AXP_s111
                                 + 1.37871 w_k1_DIS_s111
                                 + 1.17011 w_k1_FDX_s111
                                 + 0.812351 w_k1_IBM_s111
                                 + 1.03623 w_k1_INTC_s111
                                 + 0.991924 w_k1_KO_s111
                                 + 0.843869 w_k1_MAT_s111
                                 + 1.24246 w_k1_MSFT_s111
                                 + 1.08323 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.32684 w_k1_ADM_s111
                                 + 1.05716 w_k1_AXP_s111
                                 + 1.29014 w_k1_DIS_s111
                                 + 1.00782 w_k1_FDX_s111
                                 + 1.74477 w_k1_IBM_s111
                                 + 1.02241 w_k1_INTC_s111
                                 + 1.47786 w_k1_KO_s111 + 1.81857 w_k1_MAT_s111
                                 + 1.13353 w_k1_MSFT_s111
                                 + 1.10677 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.16592 w_k1_ADM_s111
                                 + 0.902612 w_k1_AXP_s111
                                 + 0.911115 w_k1_DIS_s111
                                 + 1.05916 w_k1_FDX_s111
                                 + 0.83766 w_k1_IBM_s111
                                 + 1.00606 w_k1_INTC_s111
                                 + 0.896923 w_k1_KO_s111
                                 + 0.840906 w_k1_MAT_s111
                                 + 1.06288 w_k1_MSFT_s111
                                 + 0.969593 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.921106 w_k1_ADM_s112
                                 + 0.677233 w_k1_AXP_s112
                                 + 0.756376 w_k1_DIS_s112
                                 + 0.749336 w_k1_FDX_s112
                                 + 0.895823 w_k1_IBM_s112
                                 + 0.75287 w_k1_INTC_s112
                                 + 0.896939 w_k1_KO_s112
                                 + 0.689076 w_k1_MAT_s112
                                 + 0.809317 w_k1_MSFT_s112
                                 + 0.958192 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.870345 w_k1_ADM_s112
                                 + 0.982924 w_k1_AXP_s112
                                 + 0.927113 w_k1_DIS_s112
                                 + 0.947856 w_k1_FDX_s112
                                 + 1.06287 w_k1_IBM_s112
                                 + 0.959359 w_k1_INTC_s112
                                 + 0.984654 w_k1_KO_s112
                                 + 0.654715 w_k1_MAT_s112
                                 + 0.958359 w_k1_MSFT_s112
                                 + 1.00223 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.29992 w_k1_ADM_s112
                                 + 1.44307 w_k1_AXP_s112
                                 + 1.44769 w_k1_DIS_s112
                                 + 1.18337 w_k1_FDX_s112
                                 + 1.13882 w_k1_IBM_s112
                                 + 1.02163 w_k1_INTC_s112 + 1.1966 w_k1_KO_s112
                                 + 0.967623 w_k1_MAT_s112
                                 + 1.11481 w_k1_MSFT_s112
                                 + 1.17856 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.968049 w_k1_ADM_s112
                                 + 0.992562 w_k1_AXP_s112
                                 + 0.600422 w_k1_DIS_s112
                                 + 0.818164 w_k1_FDX_s112
                                 + 0.753937 w_k1_IBM_s112
                                 + 0.688087 w_k1_INTC_s112
                                 + 0.79859 w_k1_KO_s112
                                 + 0.779549 w_k1_MAT_s112
                                 + 0.76662 w_k1_MSFT_s112
                                 + 0.857083 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.09939 w_k1_ADM_s112
                                 + 0.940079 w_k1_AXP_s112
                                 + 1.06146 w_k1_DIS_s112
                                 + 0.958215 w_k1_FDX_s112
                                 + 1.35207 w_k1_IBM_s112
                                 + 0.961363 w_k1_INTC_s112
                                 + 1.18595 w_k1_KO_s112
                                 + 0.936314 w_k1_MAT_s112
                                 + 0.915035 w_k1_MSFT_s112
                                 + 1.13137 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.0862 w_k1_ADM_s112
                                 + 1.46785 w_k1_AXP_s112
                                 + 1.07634 w_k1_DIS_s112
                                 + 1.11981 w_k1_FDX_s112 + 1.259 w_k1_IBM_s112
                                 + 0.940262 w_k1_INTC_s112
                                 + 1.10309 w_k1_KO_s112 + 1.0189 w_k1_MAT_s112
                                 + 1.00218 w_k1_MSFT_s112
                                 + 1.03173 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.14515 w_k1_ADM_s112
                                 + 1.39811 w_k1_AXP_s112
                                 + 1.50057 w_k1_DIS_s112 + 1.2963 w_k1_FDX_s112
                                 + 1.5398 w_k1_IBM_s112
                                 + 1.20949 w_k1_INTC_s112
                                 + 1.32481 w_k1_KO_s112 + 1.11232 w_k1_MAT_s112
                                 + 1.22302 w_k1_MSFT_s112
                                 + 1.20116 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.16415 w_k1_ADM_s112
                                 + 1.14064 w_k1_AXP_s112
                                 + 1.02412 w_k1_DIS_s112
                                 + 1.14621 w_k1_FDX_s112
                                 + 0.946749 w_k1_IBM_s112
                                 + 1.01114 w_k1_INTC_s112
                                 + 0.995251 w_k1_KO_s112
                                 + 0.603018 w_k1_MAT_s112
                                 + 1.01012 w_k1_MSFT_s112
                                 + 1.06294 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.952564 w_k1_ADM_s113
                                 + 1.44116 w_k1_AXP_s113
                                 + 1.57317 w_k1_DIS_s113 + 1.2821 w_k1_FDX_s113
                                 + 1.52549 w_k1_IBM_s113
                                 + 1.28884 w_k1_INTC_s113
                                 + 1.26941 w_k1_KO_s113 + 1.50477 w_k1_MAT_s113
                                 + 1.12589 w_k1_MSFT_s113
                                 + 1.23254 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.812913 w_k1_ADM_s113
                                 + 0.767363 w_k1_AXP_s113
                                 + 0.892995 w_k1_DIS_s113
                                 + 0.856192 w_k1_FDX_s113
                                 + 1.33394 w_k1_IBM_s113
                                 + 0.72455 w_k1_INTC_s113
                                 + 1.03336 w_k1_KO_s113 + 1.28446 w_k1_MAT_s113
                                 + 0.77256 w_k1_MSFT_s113 + 1.1036 w_k1_PG_s113
                                 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.863817 w_k1_ADM_s113
                                 + 0.806728 w_k1_AXP_s113
                                 + 1.03444 w_k1_DIS_s113
                                 + 0.938573 w_k1_FDX_s113
                                 + 0.903819 w_k1_IBM_s113
                                 + 0.996726 w_k1_INTC_s113
                                 + 0.965091 w_k1_KO_s113
                                 + 0.982637 w_k1_MAT_s113
                                 + 0.985465 w_k1_MSFT_s113
                                 + 0.994506 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.873487 w_k1_ADM_s113
                                 + 0.749052 w_k1_AXP_s113
                                 + 0.819684 w_k1_DIS_s113
                                 + 0.841754 w_k1_FDX_s113
                                 + 0.729776 w_k1_IBM_s113
                                 + 0.825692 w_k1_INTC_s113
                                 + 0.761083 w_k1_KO_s113
                                 + 0.661189 w_k1_MAT_s113
                                 + 0.791469 w_k1_MSFT_s113
                                 + 0.914903 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.844003 w_k1_ADM_s113
                                 + 0.972117 w_k1_AXP_s113
                                 + 1.26882 w_k1_DIS_s113
                                 + 1.17681 w_k1_FDX_s113
                                 + 1.03868 w_k1_IBM_s113
                                 + 0.991621 w_k1_INTC_s113
                                 + 1.04688 w_k1_KO_s113 + 1.34065 w_k1_MAT_s113
                                 + 1.03854 w_k1_MSFT_s113
                                 + 1.11292 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.13252 w_k1_ADM_s113
                                 + 1.29969 w_k1_AXP_s113 + 1.0384 w_k1_DIS_s113
                                 + 1.09829 w_k1_FDX_s113
                                 + 0.794682 w_k1_IBM_s113
                                 + 0.829485 w_k1_INTC_s113
                                 + 0.768646 w_k1_KO_s113
                                 + 1.12429 w_k1_MAT_s113
                                 + 0.821492 w_k1_MSFT_s113
                                 + 0.952474 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.13793 w_k1_ADM_s113
                                 + 1.07846 w_k1_AXP_s113
                                 + 1.00299 w_k1_DIS_s113
                                 + 1.00894 w_k1_FDX_s113
                                 + 1.26495 w_k1_IBM_s113
                                 + 0.900566 w_k1_INTC_s113
                                 + 1.05425 w_k1_KO_s113 + 1.76613 w_k1_MAT_s113
                                 + 0.874016 w_k1_MSFT_s113
                                 + 1.01685 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.00362 w_k1_ADM_s113
                                 + 0.980999 w_k1_AXP_s113
                                 + 1.20253 w_k1_DIS_s113
                                 + 1.21811 w_k1_FDX_s113
                                 + 0.715871 w_k1_IBM_s113
                                 + 0.943798 w_k1_INTC_s113
                                 + 0.957459 w_k1_KO_s113
                                 + 0.976293 w_k1_MAT_s113
                                 + 1.08538 w_k1_MSFT_s113
                                 + 1.17354 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.12351 w_k1_ADM_s114
                                 + 0.893796 w_k1_AXP_s114
                                 + 0.754072 w_k1_DIS_s114
                                 + 0.821129 w_k1_FDX_s114
                                 + 1.06734 w_k1_IBM_s114
                                 + 0.899709 w_k1_INTC_s114
                                 + 1.04909 w_k1_KO_s114 + 1.24722 w_k1_MAT_s114
                                 + 0.880885 w_k1_MSFT_s114
                                 + 0.974405 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.04948 w_k1_ADM_s114
                                 + 1.13943 w_k1_AXP_s114
                                 + 0.900042 w_k1_DIS_s114
                                 + 1.06987 w_k1_FDX_s114
                                 + 0.774181 w_k1_IBM_s114
                                 + 1.0953 w_k1_INTC_s114
                                 + 0.927328 w_k1_KO_s114
                                 + 0.756179 w_k1_MAT_s114
                                 + 0.971744 w_k1_MSFT_s114
                                 + 0.984238 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.04293 w_k1_ADM_s114
                                 + 1.21931 w_k1_AXP_s114
                                 + 1.09472 w_k1_DIS_s114
                                 + 1.03097 w_k1_FDX_s114
                                 + 1.10935 w_k1_IBM_s114
                                 + 1.08779 w_k1_INTC_s114
                                 + 1.08323 w_k1_KO_s114 + 1.31327 w_k1_MAT_s114
                                 + 0.99311 w_k1_MSFT_s114
                                 + 0.995182 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.0736 w_k1_ADM_s114
                                 + 1.12784 w_k1_AXP_s114
                                 + 1.40666 w_k1_DIS_s114
                                 + 1.03805 w_k1_FDX_s114
                                 + 1.29173 w_k1_IBM_s114
                                 + 0.946791 w_k1_INTC_s114
                                 + 1.34975 w_k1_KO_s114 + 1.72213 w_k1_MAT_s114
                                 + 1.06744 w_k1_MSFT_s114
                                 + 1.27914 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.133 w_k1_ADM_s114
                                 + 1.23645 w_k1_AXP_s114
                                 + 1.00895 w_k1_DIS_s114
                                 + 0.973449 w_k1_FDX_s114
                                 + 1.45155 w_k1_IBM_s114
                                 + 1.05266 w_k1_INTC_s114
                                 + 1.33604 w_k1_KO_s114 + 1.34668 w_k1_MAT_s114
                                 + 0.913858 w_k1_MSFT_s114
                                 + 1.08223 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.989899 w_k1_ADM_s114
                                 + 0.852759 w_k1_AXP_s114
                                 + 0.825721 w_k1_DIS_s114
                                 + 0.934419 w_k1_FDX_s114
                                 + 0.868105 w_k1_IBM_s114
                                 + 1.05136 w_k1_INTC_s114
                                 + 0.948934 w_k1_KO_s114 + 1.0262 w_k1_MAT_s114
                                 + 0.939463 w_k1_MSFT_s114
                                 + 1.00171 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.34654 w_k1_ADM_s114
                                 + 1.61333 w_k1_AXP_s114
                                 + 1.44855 w_k1_DIS_s114
                                 + 1.33632 w_k1_FDX_s114
                                 + 1.26557 w_k1_IBM_s114
                                 + 1.09741 w_k1_INTC_s114
                                 + 1.37597 w_k1_KO_s114 + 1.59882 w_k1_MAT_s114
                                 + 1.22243 w_k1_MSFT_s114
                                 + 1.28617 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.666006 w_k1_ADM_s114
                                 + 0.733063 w_k1_AXP_s114
                                 + 0.671507 w_k1_DIS_s114
                                 + 0.770347 w_k1_FDX_s114
                                 + 0.685578 w_k1_IBM_s114
                                 + 0.824959 w_k1_INTC_s114
                                 + 0.809472 w_k1_KO_s114
                                 + 0.963175 w_k1_MAT_s114
                                 + 0.746358 w_k1_MSFT_s114
                                 + 0.834971 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.639541 w_k1_ADM_s115
                                 + 0.945753 w_k1_AXP_s115
                                 + 0.805783 w_k1_DIS_s115
                                 + 0.809213 w_k1_FDX_s115
                                 + 1.09819 w_k1_IBM_s115
                                 + 0.890244 w_k1_INTC_s115
                                 + 0.893635 w_k1_KO_s115
                                 + 1.01455 w_k1_MAT_s115
                                 + 0.906853 w_k1_MSFT_s115
                                 + 0.862816 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.12268 w_k1_ADM_s115
                                 + 0.896068 w_k1_AXP_s115
                                 + 1.02768 w_k1_DIS_s115
                                 + 1.04464 w_k1_FDX_s115
                                 + 0.781752 w_k1_IBM_s115
                                 + 0.963116 w_k1_INTC_s115
                                 + 0.894687 w_k1_KO_s115
                                 + 0.565688 w_k1_MAT_s115
                                 + 1.09002 w_k1_MSFT_s115
                                 + 0.935486 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.05792 w_k1_ADM_s115
                                 + 1.0671 w_k1_AXP_s115 + 1.04077 w_k1_DIS_s115
                                 + 0.966835 w_k1_FDX_s115
                                 + 1.17736 w_k1_IBM_s115
                                 + 0.969098 w_k1_INTC_s115
                                 + 1.07666 w_k1_KO_s115 + 1.10326 w_k1_MAT_s115
                                 + 1.0803 w_k1_MSFT_s115
                                 + 0.926752 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.03611 w_k1_ADM_s115
                                 + 1.15344 w_k1_AXP_s115
                                 + 0.851566 w_k1_DIS_s115
                                 + 0.965449 w_k1_FDX_s115
                                 + 0.851654 w_k1_IBM_s115
                                 + 0.793207 w_k1_INTC_s115
                                 + 0.809048 w_k1_KO_s115
                                 + 0.831884 w_k1_MAT_s115
                                 + 0.845451 w_k1_MSFT_s115
                                 + 0.878123 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.94323 w_k1_ADM_s115
                                 + 1.10349 w_k1_AXP_s115
                                 + 1.26243 w_k1_DIS_s115
                                 + 1.04125 w_k1_FDX_s115 + 1.0452 w_k1_IBM_s115
                                 + 1.11882 w_k1_INTC_s115
                                 + 1.06319 w_k1_KO_s115 + 1.00744 w_k1_MAT_s115
                                 + 1.33925 w_k1_MSFT_s115
                                 + 1.17571 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.38137 w_k1_ADM_s115
                                 + 1.38517 w_k1_AXP_s115
                                 + 1.44405 w_k1_DIS_s115
                                 + 1.30225 w_k1_FDX_s115 + 1.216 w_k1_IBM_s115
                                 + 1.41272 w_k1_INTC_s115
                                 + 1.22765 w_k1_KO_s115 + 1.13553 w_k1_MAT_s115
                                 + 1.29462 w_k1_MSFT_s115
                                 + 1.17262 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.858469 w_k1_ADM_s115
                                 + 0.761468 w_k1_AXP_s115
                                 + 0.654022 w_k1_DIS_s115
                                 + 0.86773 w_k1_FDX_s115
                                 + 0.703767 w_k1_IBM_s115
                                 + 0.909741 w_k1_INTC_s115
                                 + 0.693765 w_k1_KO_s115
                                 + 0.823361 w_k1_MAT_s115
                                 + 1.01709 w_k1_MSFT_s115
                                 + 0.766189 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.961141 w_k1_ADM_s115
                                 + 1.22001 w_k1_AXP_s115
                                 + 1.25794 w_k1_DIS_s115
                                 + 1.03143 w_k1_FDX_s115
                                 + 1.73226 w_k1_IBM_s115
                                 + 0.993548 w_k1_INTC_s115
                                 + 1.32128 w_k1_KO_s115 + 2.09998 w_k1_MAT_s115
                                 + 1.16618 w_k1_MSFT_s115
                                 + 1.14207 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.605216 w_k1_ADM_s116
                                 + 0.54901 w_k1_AXP_s116
                                 + 0.517047 w_k1_DIS_s116
                                 + 0.674328 w_k1_FDX_s116
                                 + 0.628697 w_k1_IBM_s116
                                 + 0.719943 w_k1_INTC_s116
                                 + 0.652592 w_k1_KO_s116
                                 + 0.479072 w_k1_MAT_s116
                                 + 0.61887 w_k1_MSFT_s116
                                 + 0.75981 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.794353 w_k1_ADM_s116
                                 + 0.91424 w_k1_AXP_s116
                                 + 0.812499 w_k1_DIS_s116
                                 + 0.833442 w_k1_FDX_s116
                                 + 1.48484 w_k1_IBM_s116
                                 + 0.921235 w_k1_INTC_s116
                                 + 0.996741 w_k1_KO_s116
                                 + 1.37139 w_k1_MAT_s116
                                 + 0.815562 w_k1_MSFT_s116
                                 + 0.90246 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.734845 w_k1_ADM_s116
                                 + 0.689514 w_k1_AXP_s116
                                 + 1.03105 w_k1_DIS_s116
                                 + 0.887567 w_k1_FDX_s116
                                 + 1.0743 w_k1_IBM_s116
                                 + 0.956618 w_k1_INTC_s116
                                 + 1.06943 w_k1_KO_s116 + 1.02104 w_k1_MAT_s116
                                 + 1.0685 w_k1_MSFT_s116
                                 + 0.973628 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.922581 w_k1_ADM_s116
                                 + 1.471 w_k1_AXP_s116 + 1.65398 w_k1_DIS_s116
                                 + 1.30846 w_k1_FDX_s116
                                 + 1.34572 w_k1_IBM_s116
                                 + 1.30117 w_k1_INTC_s116 + 1.321 w_k1_KO_s116
                                 + 1.54066 w_k1_MAT_s116
                                 + 1.27942 w_k1_MSFT_s116 + 1.1809 w_k1_PG_s116
                                 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.841255 w_k1_ADM_s116
                                 + 0.980895 w_k1_AXP_s116
                                 + 1.39788 w_k1_DIS_s116
                                 + 1.18976 w_k1_FDX_s116
                                 + 0.865423 w_k1_IBM_s116
                                 + 1.21596 w_k1_INTC_s116
                                 + 1.03661 w_k1_KO_s116 + 1.18562 w_k1_MAT_s116
                                 + 1.26595 w_k1_MSFT_s116
                                 + 1.10358 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.73714 w_k1_ADM_s116
                                 + 0.558504 w_k1_AXP_s116
                                 + 0.752431 w_k1_DIS_s116
                                 + 0.857499 w_k1_FDX_s116
                                 + 0.537753 w_k1_IBM_s116
                                 + 0.932494 w_k1_INTC_s116
                                 + 0.844263 w_k1_KO_s116
                                 + 0.669389 w_k1_MAT_s116
                                 + 1.03366 w_k1_MSFT_s116
                                 + 0.81987 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.982499 w_k1_ADM_s116
                                 + 0.936909 w_k1_AXP_s116
                                 + 0.933927 w_k1_DIS_s116
                                 + 1.04929 w_k1_FDX_s116
                                 + 0.929628 w_k1_IBM_s116
                                 + 0.994856 w_k1_INTC_s116
                                 + 0.894854 w_k1_KO_s116
                                 + 0.743778 w_k1_MAT_s116
                                 + 1.00276 w_k1_MSFT_s116
                                 + 0.942428 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.822849 w_k1_ADM_s116
                                 + 0.693725 w_k1_AXP_s116
                                 + 0.783417 w_k1_DIS_s116
                                 + 0.929288 w_k1_FDX_s116
                                 + 0.650207 w_k1_IBM_s116
                                 + 0.899061 w_k1_INTC_s116
                                 + 0.669914 w_k1_KO_s116
                                 + 0.462287 w_k1_MAT_s116
                                 + 0.848551 w_k1_MSFT_s116
                                 + 0.794452 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.830823 w_k1_ADM_s117
                                 + 0.699141 w_k1_AXP_s117
                                 + 1.20698 w_k1_DIS_s117
                                 + 0.883881 w_k1_FDX_s117 + 1.341 w_k1_IBM_s117
                                 + 0.893776 w_k1_INTC_s117
                                 + 1.31101 w_k1_KO_s117 + 1.52576 w_k1_MAT_s117
                                 + 0.950152 w_k1_MSFT_s117
                                 + 1.16949 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.741866 w_k1_ADM_s117
                                 + 0.888994 w_k1_AXP_s117
                                 + 0.679951 w_k1_DIS_s117
                                 + 0.734656 w_k1_FDX_s117
                                 + 1.08073 w_k1_IBM_s117
                                 + 0.834647 w_k1_INTC_s117
                                 + 0.86337 w_k1_KO_s117 + 1.35867 w_k1_MAT_s117
                                 + 0.75001 w_k1_MSFT_s117
                                 + 0.955339 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.947983 w_k1_ADM_s117
                                 + 0.890006 w_k1_AXP_s117
                                 + 0.904741 w_k1_DIS_s117
                                 + 0.927964 w_k1_FDX_s117
                                 + 0.650601 w_k1_IBM_s117
                                 + 0.875615 w_k1_INTC_s117
                                 + 0.78465 w_k1_KO_s117 + 0.98015 w_k1_MAT_s117
                                 + 0.800612 w_k1_MSFT_s117
                                 + 0.9803 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.906097 w_k1_ADM_s117
                                 + 1.4458 w_k1_AXP_s117 + 1.72826 w_k1_DIS_s117
                                 + 1.1412 w_k1_FDX_s117 + 1.80827 w_k1_IBM_s117
                                 + 1.15095 w_k1_INTC_s117
                                 + 1.54536 w_k1_KO_s117 + 2.6871 w_k1_MAT_s117
                                 + 1.09653 w_k1_MSFT_s117
                                 + 1.33382 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.21011 w_k1_ADM_s117
                                 + 1.30759 w_k1_AXP_s117
                                 + 1.53711 w_k1_DIS_s117
                                 + 1.28048 w_k1_FDX_s117
                                 + 1.02406 w_k1_IBM_s117
                                 + 1.04978 w_k1_INTC_s117
                                 + 1.28526 w_k1_KO_s117 + 1.35028 w_k1_MAT_s117
                                 + 1.04317 w_k1_MSFT_s117
                                 + 1.16534 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.925441 w_k1_ADM_s117
                                 + 0.911803 w_k1_AXP_s117
                                 + 0.92111 w_k1_DIS_s117
                                 + 0.947813 w_k1_FDX_s117
                                 + 0.994354 w_k1_IBM_s117
                                 + 1.09722 w_k1_INTC_s117 + 1.0028 w_k1_KO_s117
                                 + 1.14115 w_k1_MAT_s117
                                 + 0.970807 w_k1_MSFT_s117
                                 + 1.0486 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.837474 w_k1_ADM_s117
                                 + 0.722942 w_k1_AXP_s117
                                 + 0.681368 w_k1_DIS_s117
                                 + 0.73789 w_k1_FDX_s117
                                 + 0.722137 w_k1_IBM_s117
                                 + 0.93253 w_k1_INTC_s117
                                 + 0.69886 w_k1_KO_s117
                                 + 0.835023 w_k1_MAT_s117
                                 + 0.815742 w_k1_MSFT_s117
                                 + 0.841203 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.15617 w_k1_ADM_s117
                                 + 1.17033 w_k1_AXP_s117
                                 + 1.14087 w_k1_DIS_s117
                                 + 0.985884 w_k1_FDX_s117
                                 + 1.10102 w_k1_IBM_s117
                                 + 1.11922 w_k1_INTC_s117
                                 + 1.05886 w_k1_KO_s117
                                 + 0.970692 w_k1_MAT_s117
                                 + 1.0001 w_k1_MSFT_s117 + 1.10836 w_k1_PG_s117
                                 >= 1
 k1_return_constraint:           0.0152007012663247 w_k1_ADM_s000
                                 + 0.0189446190440644 w_k1_ADM_s001
                                 + 0.0152796046671216 w_k1_ADM_s002
                                 + 0.0528254535233664 w_k1_ADM_s003
                                 + 0.0178933876757053 w_k1_ADM_s004
                                 + 0.0206006465615575 w_k1_ADM_s005
                                 + 0.0143552360170514 w_k1_ADM_s006
                                 + 0.0438710051594871 w_k1_ADM_s007
                                 + 0.0246491071661912 w_k1_ADM_s010
                                 + 0.011350524304717 w_k1_ADM_s011
                                 + 0.0290529432482925 w_k1_ADM_s012
                                 + 0.00781526896240573 w_k1_ADM_s013
                                 + 0.0113409329667453 w_k1_ADM_s014
                                 + 0.0345824084954002 w_k1_ADM_s015
                                 + 0.0122500907145107 w_k1_ADM_s016
                                 + 0.0169771013046558 w_k1_ADM_s017
                                 + 0.0812588405932762 w_k1_ADM_s100
                                 + 0.0571855458635225 w_k1_ADM_s101
                                 + 0.0548604211516622 w_k1_ADM_s102
                                 + 0.0255553367568299 w_k1_ADM_s103
                                 + 0.0268787566829556 w_k1_ADM_s104
                                 + 0.0141244551233358 w_k1_ADM_s105
                                 + 0.0699995781514381 w_k1_ADM_s106
                                 + 0.0356600000551441 w_k1_ADM_s107
                                 + 0.024172192524684 w_k1_ADM_s110
                                 + 0.0544794404226223 w_k1_ADM_s111
                                 + 0.0275184260787561 w_k1_ADM_s112
                                 + 0.018892796263407 w_k1_ADM_s113
                                 + 0.0421383584989911 w_k1_ADM_s114
                                 + 0.0184721568046405 w_k1_ADM_s115
                                 + 0.055109271987246 w_k1_ADM_s116
                                 + 0.0480781855444764 w_k1_ADM_s117
                                 + 0.0181238437574201 w_k1_AXP_s000
                                 + 0.0185301137260076 w_k1_AXP_s001
                                 + 0.0132607098425793 w_k1_AXP_s002
                                 + 0.0591219693699105 w_k1_AXP_s003
                                 + 0.0186128122195035 w_k1_AXP_s004
                                 + 0.0223037549135742 w_k1_AXP_s005
                                 + 0.0142887079460356 w_k1_AXP_s006
                                 + 0.0518992092934398 w_k1_AXP_s007
                                 + 0.0230102017047694 w_k1_AXP_s010
                                 + 0.0138551051296883 w_k1_AXP_s011
                                 + 0.0288563787080126 w_k1_AXP_s012
                                 + 0.00948129608062267 w_k1_AXP_s013
                                 + 0.0122849632307158 w_k1_AXP_s014
                                 + 0.0332368201705674 w_k1_AXP_s015
                                 + 0.015776176903979 w_k1_AXP_s016
                                 + 0.0149552044203566 w_k1_AXP_s017
                                 + 0.0754956758811836 w_k1_AXP_s100
                                 + 0.0523834884025573 w_k1_AXP_s101
                                 + 0.0628077402367719 w_k1_AXP_s102
                                 + 0.0286762160869272 w_k1_AXP_s103
                                 + 0.0237295411840622 w_k1_AXP_s104
                                 + 0.0153503333850558 w_k1_AXP_s105
                                 + 0.0762352901669885 w_k1_AXP_s106
                                 + 0.0354486919724707 w_k1_AXP_s107
                                 + 0.0253934683415383 w_k1_AXP_s110
                                 + 0.0456620498653751 w_k1_AXP_s111
                                 + 0.028336370899927 w_k1_AXP_s112
                                 + 0.0195308073297546 w_k1_AXP_s113
                                 + 0.0432811175282868 w_k1_AXP_s114
                                 + 0.0198255871862703 w_k1_AXP_s115
                                 + 0.0625045782202 w_k1_AXP_s116
                                 + 0.0497868502580214 w_k1_AXP_s117
                                 + 0.0166305450024939 w_k1_DIS_s000
                                 + 0.0180944934056866 w_k1_DIS_s001
                                 + 0.0120957445447929 w_k1_DIS_s002
                                 + 0.0579559487291236 w_k1_DIS_s003
                                 + 0.0187519453224901 w_k1_DIS_s004
                                 + 0.0217554733888778 w_k1_DIS_s005
                                 + 0.012961636812476 w_k1_DIS_s006
                                 + 0.0484589920007567 w_k1_DIS_s007
                                 + 0.022107957092355 w_k1_DIS_s010
                                 + 0.014064153202959 w_k1_DIS_s011
                                 + 0.0307573092499616 w_k1_DIS_s012
                                 + 0.00932921585692123 w_k1_DIS_s013
                                 + 0.0115849828916691 w_k1_DIS_s014
                                 + 0.0338125865120009 w_k1_DIS_s015
                                 + 0.0142179176953277 w_k1_DIS_s016
                                 + 0.0158488125060772 w_k1_DIS_s017
                                 + 0.0814433964859861 w_k1_DIS_s100
                                 + 0.0588694498221728 w_k1_DIS_s101
                                 + 0.0605428276102343 w_k1_DIS_s102
                                 + 0.0268594295823189 w_k1_DIS_s103
                                 + 0.0222326863440483 w_k1_DIS_s104
                                 + 0.0159391047982002 w_k1_DIS_s105
                                 + 0.0784495400869492 w_k1_DIS_s106
                                 + 0.0358494106799063 w_k1_DIS_s107
                                 + 0.0265038720342985 w_k1_DIS_s110
                                 + 0.0465173215638521 w_k1_DIS_s111
                                 + 0.0266607770171671 w_k1_DIS_s112
                                 + 0.021606026634204 w_k1_DIS_s113
                                 + 0.0398849505397889 w_k1_DIS_s114
                                 + 0.0194135823504894 w_k1_DIS_s115
                                 + 0.0708881833284253 w_k1_DIS_s116
                                 + 0.0537019098753968 w_k1_DIS_s117
                                 + 0.0156179402464505 w_k1_FDX_s000
                                 + 0.0181959471466329 w_k1_FDX_s001
                                 + 0.0126346239930138 w_k1_FDX_s002
                                 + 0.0567256128876601 w_k1_FDX_s003
                                 + 0.0179592589270336 w_k1_FDX_s004
                                 + 0.0211308818680886 w_k1_FDX_s005
                                 + 0.0139785045255371 w_k1_FDX_s006
                                 + 0.0475484859738782 w_k1_FDX_s007
                                 + 0.0231615186677482 w_k1_FDX_s010
                                 + 0.0152701208512677 w_k1_FDX_s011
                                 + 0.0289230215828017 w_k1_FDX_s012
                                 + 0.00941768520042524 w_k1_FDX_s013
                                 + 0.0119421830365146 w_k1_FDX_s014
                                 + 0.0348153735559164 w_k1_FDX_s015
                                 + 0.0147334821247711 w_k1_FDX_s016
                                 + 0.0152805594372478 w_k1_FDX_s017
                                 + 0.0753149230887656 w_k1_FDX_s100
                                 + 0.0569968147887852 w_k1_FDX_s101
                                 + 0.0583621825582752 w_k1_FDX_s102
                                 + 0.0270432054559001 w_k1_FDX_s103
                                 + 0.0233054444778142 w_k1_FDX_s104
                                 + 0.0150762502040249 w_k1_FDX_s105
                                 + 0.0740204659535569 w_k1_FDX_s106
                                 + 0.036589605424683 w_k1_FDX_s107
                                 + 0.0244257116762682 w_k1_FDX_s110
                                 + 0.0470447879139224 w_k1_FDX_s111
                                 + 0.0261593015313774 w_k1_FDX_s112
                                 + 0.0205711621711422 w_k1_FDX_s113
                                 + 0.0392330756024827 w_k1_FDX_s114
                                 + 0.0187602170740267 w_k1_FDX_s115
                                 + 0.0672254048533583 w_k1_FDX_s116
                                 + 0.0474346413739741 w_k1_FDX_s117
                                 + 0.0158846281823375 w_k1_IBM_s000
                                 + 0.0179525373823749 w_k1_IBM_s001
                                 + 0.0135159930834276 w_k1_IBM_s002
                                 + 0.0564594690365488 w_k1_IBM_s003
                                 + 0.0174984185107207 w_k1_IBM_s004
                                 + 0.0221003397225196 w_k1_IBM_s005
                                 + 0.0144452765587096 w_k1_IBM_s006
                                 + 0.0512921513973524 w_k1_IBM_s007
                                 + 0.0227418378597122 w_k1_IBM_s010
                                 + 0.0139727260301923 w_k1_IBM_s011
                                 + 0.0285585628560627 w_k1_IBM_s012
                                 + 0.00892612402972491 w_k1_IBM_s013
                                 + 0.0124663274244512 w_k1_IBM_s014
                                 + 0.033764584223617 w_k1_IBM_s015
                                 + 0.0155586897076309 w_k1_IBM_s016
                                 + 0.0158127428532301 w_k1_IBM_s017
                                 + 0.0759562444318666 w_k1_IBM_s100
                                 + 0.0566519124488492 w_k1_IBM_s101
                                 + 0.0598881502915064 w_k1_IBM_s102
                                 + 0.0263168805310967 w_k1_IBM_s103
                                 + 0.0225193867306652 w_k1_IBM_s104
                                 + 0.0154917713256043 w_k1_IBM_s105
                                 + 0.0752625269603672 w_k1_IBM_s106
                                 + 0.0358106011331314 w_k1_IBM_s107
                                 + 0.0261681310962465 w_k1_IBM_s110
                                 + 0.0496961588245395 w_k1_IBM_s111
                                 + 0.0281314107709385 w_k1_IBM_s112
                                 + 0.0207285006645696 w_k1_IBM_s113
                                 + 0.0415504336375948 w_k1_IBM_s114
                                 + 0.0197252327615673 w_k1_IBM_s115
                                 + 0.0673892254739917 w_k1_IBM_s116
                                 + 0.0472641250195325 w_k1_IBM_s117
                                 + 0.0146490797121947 w_k1_INTC_s000
                                 + 0.0156065012521609 w_k1_INTC_s001
                                 + 0.0102702537884856 w_k1_INTC_s002
                                 + 0.059987704687786 w_k1_INTC_s003
                                 + 0.018957069125668 w_k1_INTC_s004
                                 + 0.0238686343662776 w_k1_INTC_s005
                                 + 0.0160394076773778 w_k1_INTC_s006
                                 + 0.0456399818850597 w_k1_INTC_s007
                                 + 0.0231354369228918 w_k1_INTC_s010
                                 + 0.0169042605230783 w_k1_INTC_s011
                                 + 0.027487666283218 w_k1_INTC_s012
                                 + 0.00953527294058528 w_k1_INTC_s013
                                 + 0.0114100734072527 w_k1_INTC_s014
                                 + 0.0383100514470842 w_k1_INTC_s015
                                 + 0.014323260452474 w_k1_INTC_s016
                                 + 0.0128961027922468 w_k1_INTC_s017
                                 + 0.0778322338330725 w_k1_INTC_s100
                                 + 0.0540141424349796 w_k1_INTC_s101
                                 + 0.0588939437010886 w_k1_INTC_s102
                                 + 0.023810224563773 w_k1_INTC_s103
                                 + 0.0236019685957871 w_k1_INTC_s104
                                 + 0.0161382693280027 w_k1_INTC_s105
                                 + 0.075284957024482 w_k1_INTC_s106
                                 + 0.0349299913181956 w_k1_INTC_s107
                                 + 0.0271317617395926 w_k1_INTC_s110
                                 + 0.0439618707941953 w_k1_INTC_s111
                                 + 0.0241838019445747 w_k1_INTC_s112
                                 + 0.0188744066122577 w_k1_INTC_s113
                                 + 0.0391674941530919 w_k1_INTC_s114
                                 + 0.0191014614833702 w_k1_INTC_s115
                                 + 0.0684491438168707 w_k1_INTC_s116
                                 + 0.047495603532299 w_k1_INTC_s117
                                 + 0.0164456923761839 w_k1_KO_s000
                                 + 0.0159625578952346 w_k1_KO_s001
                                 + 0.01241315199294 w_k1_KO_s002
                                 + 0.058052115659969 w_k1_KO_s003
                                 + 0.0180660753025922 w_k1_KO_s004
                                 + 0.0221039245053152 w_k1_KO_s005
                                 + 0.0153869990558763 w_k1_KO_s006
                                 + 0.0505644770344198 w_k1_KO_s007
                                 + 0.0236956582224688 w_k1_KO_s010
                                 + 0.0134831112448595 w_k1_KO_s011
                                 + 0.0273583937914336 w_k1_KO_s012
                                 + 0.00920486451554915 w_k1_KO_s013
                                 + 0.012206513340939 w_k1_KO_s014
                                 + 0.0358497240413983 w_k1_KO_s015
                                 + 0.0135725290779547 w_k1_KO_s016
                                 + 0.0149650369243648 w_k1_KO_s017
                                 + 0.0785172946580392 w_k1_KO_s100
                                 + 0.0619851845350281 w_k1_KO_s101
                                 + 0.0566699714970767 w_k1_KO_s102
                                 + 0.027519408161582 w_k1_KO_s103
                                 + 0.0236941514544594 w_k1_KO_s104
                                 + 0.0154404867074816 w_k1_KO_s105
                                 + 0.0780125570968439 w_k1_KO_s106
                                 + 0.033857906377522 w_k1_KO_s107
                                 + 0.0258525049510591 w_k1_KO_s110
                                 + 0.0443066283514696 w_k1_KO_s111
                                 + 0.0270656223534084 w_k1_KO_s112
                                 + 0.0195691178380159 w_k1_KO_s113
                                 + 0.0435321582392759 w_k1_KO_s114
                                 + 0.0184346696333835 w_k1_KO_s115
                                 + 0.065161204922241 w_k1_KO_s116
                                 + 0.0496660320431667 w_k1_KO_s117
                                 + 0.015700035762087 w_k1_MAT_s000
                                 + 0.0180069045458578 w_k1_MAT_s001
                                 + 0.0135629305788291 w_k1_MAT_s002
                                 + 0.0564621164555671 w_k1_MAT_s003
                                 + 0.0193840725932724 w_k1_MAT_s004
                                 + 0.0198781686970128 w_k1_MAT_s005
                                 + 0.0136536915677853 w_k1_MAT_s006
                                 + 0.06377571659672 w_k1_MAT_s007
                                 + 0.0225785688232997 w_k1_MAT_s010
                                 + 0.0156349486954011 w_k1_MAT_s011
                                 + 0.0283287086385356 w_k1_MAT_s012
                                 + 0.0099067275776531 w_k1_MAT_s013
                                 + 0.0130191733119004 w_k1_MAT_s014
                                 + 0.0342891359111485 w_k1_MAT_s015
                                 + 0.0168478638851808 w_k1_MAT_s016
                                 + 0.0147781867639468 w_k1_MAT_s017
                                 + 0.0801774070375191 w_k1_MAT_s100
                                 + 0.063900064191861 w_k1_MAT_s101
                                 + 0.0530064940412304 w_k1_MAT_s102
                                 + 0.0297581121875021 w_k1_MAT_s103
                                 + 0.0260084632389032 w_k1_MAT_s104
                                 + 0.0147829237433666 w_k1_MAT_s105
                                 + 0.091548426694584 w_k1_MAT_s106
                                 + 0.0317630525121602 w_k1_MAT_s107
                                 + 0.0238829163890812 w_k1_MAT_s110
                                 + 0.0450068944312133 w_k1_MAT_s111
                                 + 0.0211044559703729 w_k1_MAT_s112
                                 + 0.0231404696791207 w_k1_MAT_s113
                                 + 0.0496638722328589 w_k1_MAT_s114
                                 + 0.019188471311408 w_k1_MAT_s115
                                 + 0.0681588523670227 w_k1_MAT_s116
                                 + 0.0570294304255975 w_k1_MAT_s117
                                 + 0.0152928884249547 w_k1_MSFT_s000
                                 + 0.0182054438764667 w_k1_MSFT_s001
                                 + 0.0120536546745597 w_k1_MSFT_s002
                                 + 0.0572709725571282 w_k1_MSFT_s003
                                 + 0.0191883127511763 w_k1_MSFT_s004
                                 + 0.0206990293274332 w_k1_MSFT_s005
                                 + 0.0147437674995578 w_k1_MSFT_s006
                                 + 0.0508712314347722 w_k1_MSFT_s007
                                 + 0.0241696196184491 w_k1_MSFT_s010
                                 + 0.0163955652226991 w_k1_MSFT_s011
                                 + 0.0269565241925446 w_k1_MSFT_s012
                                 + 0.00981813513341169 w_k1_MSFT_s013
                                 + 0.0133368354942402 w_k1_MSFT_s014
                                 + 0.0338167042506919 w_k1_MSFT_s015
                                 + 0.0149404135236931 w_k1_MSFT_s016
                                 + 0.0145363646289889 w_k1_MSFT_s017
                                 + 0.0785295078614253 w_k1_MSFT_s100
                                 + 0.0551523028170521 w_k1_MSFT_s101
                                 + 0.0564494622337395 w_k1_MSFT_s102
                                 + 0.0232989018266599 w_k1_MSFT_s103
                                 + 0.0233665585592374 w_k1_MSFT_s104
                                 + 0.0156278165477849 w_k1_MSFT_s105
                                 + 0.0728100748124686 w_k1_MSFT_s106
                                 + 0.0358292961080022 w_k1_MSFT_s107
                                 + 0.0262341066455272 w_k1_MSFT_s110
                                 + 0.0476769450565017 w_k1_MSFT_s111
                                 + 0.0250369253346778 w_k1_MSFT_s112
                                 + 0.0186384266842637 w_k1_MSFT_s113
                                 + 0.0383096701780797 w_k1_MSFT_s114
                                 + 0.0203147731712 w_k1_MSFT_s115
                                 + 0.0677743942502353 w_k1_MSFT_s116
                                 + 0.0442988608451852 w_k1_MSFT_s117
                                 + 0.0153013763753876 w_k1_PG_s000
                                 + 0.0190480472771353 w_k1_PG_s001
                                 + 0.0126442022078981 w_k1_PG_s002
                                 + 0.0608035772890739 w_k1_PG_s003
                                 + 0.018643117394448 w_k1_PG_s004
                                 + 0.0223907840688059 w_k1_PG_s005
                                 + 0.0133294012166537 w_k1_PG_s006
                                 + 0.0462156740460614 w_k1_PG_s007
                                 + 0.0222957473519099 w_k1_PG_s010
                                 + 0.0146300036587983 w_k1_PG_s011
                                 + 0.0263685896222962 w_k1_PG_s012
                                 + 0.00862294845787513 w_k1_PG_s013
                                 + 0.0118445790706491 w_k1_PG_s014
                                 + 0.0338416053082914 w_k1_PG_s015
                                 + 0.0132639619790974 w_k1_PG_s016
                                 + 0.0159607468312507 w_k1_PG_s017
                                 + 0.0692995824033674 w_k1_PG_s100
                                 + 0.0543318665099914 w_k1_PG_s101
                                 + 0.0568420007452886 w_k1_PG_s102
                                 + 0.0250661460480421 w_k1_PG_s103
                                 + 0.0226115726878763 w_k1_PG_s104
                                 + 0.0160615632845637 w_k1_PG_s105
                                 + 0.0701916895309542 w_k1_PG_s106
                                 + 0.0345632369136953 w_k1_PG_s107
                                 + 0.0287101267745175 w_k1_PG_s110
                                 + 0.0446384626965083 w_k1_PG_s111
                                 + 0.0271811315841913 w_k1_PG_s112
                                 + 0.0211799052514293 w_k1_PG_s113
                                 + 0.041595767454608 w_k1_PG_s114
                                 + 0.018286689337625 w_k1_PG_s115
                                 + 0.064297199012437 w_k1_PG_s116
                                 + 0.0503758209456859 w_k1_PG_s117 >= 1.18
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

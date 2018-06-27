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
 w_k1_ADM_s0_return_constraint:  1.13897 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.903197 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.03404 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.964657 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.00578 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.10397 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.14568 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.35044 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.843862 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.3093 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.683271 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.03038 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.17994 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.15559 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.902061 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.29273 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.05523 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.83461 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.01574 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.25555 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.13433 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.42284 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.36702 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.934715 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.97028 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.85038 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.10595 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.891866 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.1902 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.675203 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.977783 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.93401 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.30523 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.23301 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.06616 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.768721 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.868515 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.27439 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.27783 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.90805 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.953818 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.02205 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.925155 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.12268 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.07419 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.68738 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.776169 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.57463 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.565318 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.845298 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.976548 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.13939 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.924527 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.29337 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.54164 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.716163 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.09448 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.980469 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.08978 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.76215 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.52643 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.07199 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.10543 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.829386 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.56268 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.789299 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.14293 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.602648 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.963428 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.964251 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.10944 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.32809 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.0103 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.684362 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.913224 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.79387 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.28483 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.84349 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.05832 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.07992 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.943235 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.34794 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.06626 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.88825 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.646853 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.40621 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.502751 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.800296 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.11876 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.67223 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.809011 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.08582 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.54085 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.587626 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.0363 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.18571 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.967819 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.96575 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.79761 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.10957 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.930279 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.804368 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.61037 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.661239 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.26422 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.576024 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.938899 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.742064 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.02945 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.30151 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.908587 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.549883 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.936017 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.70526 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.20755 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.908503 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.08885 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.98709 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.976285 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.08599 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.00252 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.37667 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.801676 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.41122 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.560716 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.80959 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.07125 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.22841 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.82444 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.15233 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.22336 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.756416 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.881225 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.18479 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.979486 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.57069 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.57874 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.928692 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.947899 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.837942 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.26579 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.836055 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.18188 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.691961 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.0087 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.930673 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.23539 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.31824 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.911252 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.663839 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.868839 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.35479 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.14593 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.95741 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.758072 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.18831 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.834782 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.41221 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.16329 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.65088 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.617091 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.918543 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.719034 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.99019 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.795403 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.24435 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.1278 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.834644 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.59703 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.695661 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.74346 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.764517 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.96606 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.39013 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.0288 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.58874 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.22815 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.0245 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.66215 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.567332 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.934095 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.74587 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.885188 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.699023 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.59863 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.941606 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.4525 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.847905 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.09922 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.44031 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.08769 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.909658 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.06993 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.0105 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.04748 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.12687 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.01196 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.15984 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.910529 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.3585 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.637159 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.846934 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.14047 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.33566 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.962109 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.0268 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.22706 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.813307 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.938277 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.07745 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.06319 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.56149 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.34178 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.05008 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.992275 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.786826 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.15226 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.859658 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.04837 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.722265 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.990649 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.86968 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.20037 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.17889 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.03747 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.762648 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.953302 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.31945 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.20154 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.923201 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.903215 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.15189 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.930416 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.23025 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.15228 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.67249 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.731843 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.15762 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.708366 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.87006 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.999882 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.28735 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.917727 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.845461 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.31813 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.699428 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.12056 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.980541 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.901842 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.41237 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.29726 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.15447 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.12916 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.827398 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.44073 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.625298 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.11425 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.617221 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.91689 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.755571 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.844997 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.16842 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.26793 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.726228 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.983444 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.50446 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.23154 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.00285 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.786026 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.19522 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.822985 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.26866 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.36499 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.97039 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.641429 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.3221 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.69782 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.869731 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.775064 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.993563 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.09688 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.01768 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.89534 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.56648 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.56532 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.815444 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.970253 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.42338 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.15609 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.29888 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.44938 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.910519 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.85733 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.565261 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.22146 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.576753 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.976269 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.739722 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.633804 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.00794 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.59392 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.87858 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.935376 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.72173 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.12156 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.915258 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.03304 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.01756 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.943646 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.06799 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.154 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.30893 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.785985 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.36644 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.666823 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.910269 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.13187 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.42583 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.893412 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.0973 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.23785 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.777724 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.880496 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.18401 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.03746 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.51152 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.46966 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.00943 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.994011 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.905462 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.45653 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.861675 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.11613 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.77694 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.01588 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.874209 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.27082 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.23128 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.06167 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.695258 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.9598 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.43467 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.12499 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.953907 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.01442 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.995044 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.913099 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.06376 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.0193 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.32859 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.793686 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.14762 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.740205 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.913351 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.956372 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.04796 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.974009 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.08688 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.29907 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.86031 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.11022 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.0999 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.969277 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.46462 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.34425 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.08277 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.06454 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.92878 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.40388 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.903429 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.158 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.757231 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.984702 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.973506 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.09363 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.13935 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.08365 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.884239 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.03509 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.27634 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.181999233729943 z_k1_s0000
                                 - 0.189000318104867 z_k1_s0001
                                 - 0.304999670591984 z_k1_s0002
                                 - 0.226999481850257 z_k1_s0003
                                 - 0.226999481850257 z_k1_s0004
                                 - 0.298000396218191 z_k1_s0005
                                 - 0.25100009684806 z_k1_s0006
                                 - 0.13100009886097 z_k1_s0007
                                 - 0.373998293434247 z_k1_s0010
                                 - 0.0859995536375495 z_k1_s0011
                                 - 0.11199948750438 z_k1_s0012
                                 - 0.165999091498289 z_k1_s0013
                                 - 0.155999065203606 z_k1_s0014
                                 - 0.150999724552989 z_k1_s0015
                                 - 0.182998463702526 z_k1_s0016
                                 - 0.116999366152377 z_k1_s0017
                                 - 0.176999340485774 z_k1_s0020
                                 - 0.438998078538704 z_k1_s0021
                                 - 0.412999116364264 z_k1_s0022
                                 - 0.136999466307723 z_k1_s0023
                                 - 0.129999457054185 z_k1_s0024
                                 - 0.112999613280615 z_k1_s0025
                                 - 0.22599922656123 z_k1_s0026
                                 - 0.151999486136732 z_k1_s0027
                                 - 0.14399950066693 z_k1_s0030
                                 - 0.178000271767539 z_k1_s0031
                                 - 0.0999998945405692 z_k1_s0032
                                 - 0.0579999639794932 z_k1_s0033
                                 - 0.184999993494776 z_k1_s0034
                                 - 0.170999407041979 z_k1_s0035
                                 - 0.17499990117087 z_k1_s0036
                                 - 0.132000304276901 z_k1_s0037
                                 - 0.0419999490384548 z_k1_s0040
                                 - 0.0549999885788997 z_k1_s0041
                                 - 0.107000001540756 z_k1_s0042
                                 - 0.148000103906786 z_k1_s0043
                                 - 0.0699999392640786 z_k1_s0044
                                 - 0.156999595158147 z_k1_s0045
                                 - 0.0629999961576439 z_k1_s0046
                                 - 0.0839996076770629 z_k1_s0047
                                 - 0.515000281581511 z_k1_s0050
                                 - 0.209999749822915 z_k1_s0051
                                 - 0.238999838075007 z_k1_s0052
                                 - 0.423999447535836 z_k1_s0053
                                 - 0.818998185420187 z_k1_s0054
                                 - 0.311998421018082 z_k1_s0055
                                 - 0.174999739735813 z_k1_s0056
                                 - 0.413000678287255 z_k1_s0057
                                 - 0.183999918376663 z_k1_s0060
                                 - 0.0960001054184268 z_k1_s0061
                                 - 0.311000549202639 z_k1_s0062
                                 - 0.162999839594444 z_k1_s0063
                                 - 0.330000311719535 z_k1_s0064
                                 - 0.261000747578745 z_k1_s0065
                                 - 0.16700047212509 z_k1_s0066
                                 - 0.109000053840639 z_k1_s0067
                                 - 0.158999688882637 z_k1_s0070
                                 - 0.139999794110407 z_k1_s0071
                                 - 0.209999691165611 z_k1_s0072
                                 - 0.229999352610365 z_k1_s0073
                                 - 0.124000281354316 z_k1_s0074
                                 - 0.0689999139829659 z_k1_s0075
                                 - 0.0679999309107282 z_k1_s0076
                                 - 0.0819999103217689 z_k1_s0077
                                 - 0.529002398124239 z_k1_s0100
                                 - 0.619001636034078 z_k1_s0101
                                 - 0.698000770587674 z_k1_s0102
                                 - 0.20800063603223 z_k1_s0103
                                 - 0.221000565252659 z_k1_s0104
                                 - 0.288000744621136 z_k1_s0105
                                 - 0.76400139072513 z_k1_s0106
                                 - 0.210000461896416 z_k1_s0107
                                 - 0.51900070117526 z_k1_s0110
                                 - 0.213999708938767 z_k1_s0111
                                 - 0.207000829301636 z_k1_s0112
                                 - 0.227001024268087 z_k1_s0113
                                 - 0.125000029939189 z_k1_s0114
                                 - 0.16000017303029 z_k1_s0115
                                 - 0.271999481940722 z_k1_s0116
                                 - 0.256999335715884 z_k1_s0117
                                 - 0.288000249844092 z_k1_s0120
                                 - 0.21300005367635 z_k1_s0121
                                 - 0.319999072048908 z_k1_s0122
                                 - 0.362999795011788 z_k1_s0123
                                 - 0.312000451497778 z_k1_s0124
                                 - 0.203999978056218 z_k1_s0125
                                 - 0.270000098603827 z_k1_s0126
                                 - 0.200000016780608 z_k1_s0127
                                 - 1.10600273017109 z_k1_s0130
                                 - 0.372000766834784 z_k1_s0131
                                 - 1.1470027317414 z_k1_s0132
                                 - 0.528001337249926 z_k1_s0133
                                 - 0.320000576696403 z_k1_s0134
                                 - 0.997003666730157 z_k1_s0135
                                 - 0.573002709756992 z_k1_s0136
                                 - 0.467001093010624 z_k1_s0137
                                 - 0.175999450050323 z_k1_s0140
                                 - 0.0880000350252497 z_k1_s0141
                                 - 0.117000078272216 z_k1_s0142
                                 - 0.124000035262848 z_k1_s0143
                                 - 0.160000345500534 z_k1_s0144
                                 - 0.432000622851354 z_k1_s0145
                                 - 0.0960000523002764 z_k1_s0146
                                 - 0.357000751523061 z_k1_s0147
                                 - 0.2310007471762 z_k1_s0150
                                 - 0.197999891721879 z_k1_s0151
                                 - 0.175999903752782 z_k1_s0152
                                 - 0.0740000071779109 z_k1_s0153
                                 - 0.285000558827499 z_k1_s0154
                                 - 0.324999901679914 z_k1_s0155
                                 - 0.229000430633309 z_k1_s0156
                                 - 0.229000430633309 z_k1_s0157
                                 - 0.712002174691988 z_k1_s0160
                                 - 0.532999204322924 z_k1_s0161
                                 - 0.726999584885079 z_k1_s0162
                                 - 0.474000167397263 z_k1_s0163
                                 - 0.258999890864971 z_k1_s0164
                                 - 0.292999890135661 z_k1_s0165
                                 - 1.24299904112264 z_k1_s0166
                                 - 0.789997763495573 z_k1_s0167
                                 - 0.188000002396025 z_k1_s0170
                                 - 0.186999381430554 z_k1_s0171
                                 - 0.0880001667089376 z_k1_s0172
                                 - 0.237000781675528 z_k1_s0173
                                 - 0.324001068619709 z_k1_s0174
                                 - 0.32799984647802 z_k1_s0175
                                 - 0.13600013972279 z_k1_s0176
                                 - 0.365000586179014 z_k1_s0177
                                 - 0.440999129276301 z_k1_s1000
                                 - 0.336999580880138 z_k1_s1001
                                 - 0.277999969110111 z_k1_s1002
                                 - 1.09899842452943 z_k1_s1003
                                 - 0.951000075438065 z_k1_s1004
                                 - 0.250999674254652 z_k1_s1005
                                 - 0.542997654109766 z_k1_s1006
                                 - 0.260999798871478 z_k1_s1007
                                 - 0.58400085093719 z_k1_s1010
                                 - 0.486999401388978 z_k1_s1011
                                 - 0.600999024465582 z_k1_s1012
                                 - 0.349999196831793 z_k1_s1013
                                 - 0.26200020601149 z_k1_s1014
                                 - 0.230000048803937 z_k1_s1015
                                 - 0.146000068584251 z_k1_s1016
                                 - 0.22300012252732 z_k1_s1017
                                 - 0.263999122891687 z_k1_s1020
                                 - 0.334999073179455 z_k1_s1021
                                 - 0.380999085840417 z_k1_s1022
                                 - 0.319999019583629 z_k1_s1023
                                 - 0.737997175730086 z_k1_s1024
                                 - 0.29599938892902 z_k1_s1025
                                 - 1.18399846591351 z_k1_s1026
                                 - 1.03299671278562 z_k1_s1027
                                 - 1.79800550530286 z_k1_s1030
                                 - 1.2240003439872 z_k1_s1031
                                 - 0.444000004779637 z_k1_s1032
                                 - 1.36299578304981 z_k1_s1033
                                 - 1.98700135841735 z_k1_s1034
                                 - 1.07499870211249 z_k1_s1035
                                 - 1.3859967895139 z_k1_s1036
                                 - 0.923000299395587 z_k1_s1037
                                 - 0.38800129703812 z_k1_s1040
                                 - 0.149000057469353 z_k1_s1041
                                 - 0.218999945169073 z_k1_s1042
                                 - 0.482999115888084 z_k1_s1043
                                 - 0.354999898648499 z_k1_s1044
                                 - 0.564000562117492 z_k1_s1045
                                 - 0.2800011893986 z_k1_s1046
                                 - 0.29300074718862 z_k1_s1047
                                 - 1.38800584395526 z_k1_s1050
                                 - 0.324001553573622 z_k1_s1051
                                 - 0.688003898734116 z_k1_s1052
                                 - 0.693005069361895 z_k1_s1053
                                 - 0.502003218557539 z_k1_s1054
                                 - 0.291000971959748 z_k1_s1055
                                 - 0.329001295295508 z_k1_s1056
                                 - 0.548004462293817 z_k1_s1057
                                 - 1.09999975657527 z_k1_s1060
                                 - 0.729999462054846 z_k1_s1061
                                 - 0.792999730412988 z_k1_s1062
                                 - 2.93200085676107 z_k1_s1063
                                 - 1.00299958984108 z_k1_s1064
                                 - 2.16799425063685 z_k1_s1065
                                 - 0.812998879087864 z_k1_s1066
                                 - 0.811998973409072 z_k1_s1067
                                 - 0.467998498252852 z_k1_s1070
                                 - 0.558000902338865 z_k1_s1071
                                 - 0.320999853410258 z_k1_s1072
                                 - 0.574999800555733 z_k1_s1073
                                 - 0.18199986637593 z_k1_s1074
                                 - 0.896000022465843 z_k1_s1075
                                 - 0.463001640246767 z_k1_s1076
                                 - 0.221999788414615 z_k1_s1077
                                 - 0.485999652236897 z_k1_s1100
                                 - 0.60399804399779 z_k1_s1101
                                 - 0.224999346351639 z_k1_s1102
                                 - 0.371998890541449 z_k1_s1103
                                 - 0.479999612147663 z_k1_s1104
                                 - 0.398997273447469 z_k1_s1105
                                 - 0.707997540547493 z_k1_s1106
                                 - 0.320999268781172 z_k1_s1107
                                 - 0.1530001100844 z_k1_s1110
                                 - 0.107999986694861 z_k1_s1111
                                 - 0.230000974479902 z_k1_s1112
                                 - 0.177999963745232 z_k1_s1113
                                 - 0.234000486905586 z_k1_s1114
                                 - 0.321999430271657 z_k1_s1115
                                 - 0.374000827806369 z_k1_s1116
                                 - 0.334999779655335 z_k1_s1117
                                 - 0.0510000575985423 z_k1_s1120
                                 - 0.0780001550449342 z_k1_s1121
                                 - 0.0969999947735186 z_k1_s1122
                                 - 0.0680000767980564 z_k1_s1123
                                 - 0.0530001220279942 z_k1_s1124
                                 - 0.0930000285148692 z_k1_s1125
                                 - 0.200999930499674 z_k1_s1126
                                 - 0.172000175124466 z_k1_s1127
                                 - 0.788997551779335 z_k1_s1130
                                 - 0.171999729214163 z_k1_s1131
                                 - 0.84999970512677 z_k1_s1132
                                 - 0.783998081307312 z_k1_s1133
                                 - 0.267999611463885 z_k1_s1134
                                 - 0.248999829172541 z_k1_s1135
                                 - 0.195999699776594 z_k1_s1136
                                 - 0.280999670289271 z_k1_s1137
                                 - 0.498000260537769 z_k1_s1140
                                 - 0.23500088933021 z_k1_s1141
                                 - 0.70900285598874 z_k1_s1142
                                 - 0.836002448182538 z_k1_s1143
                                 - 0.21400076831801 z_k1_s1144
                                 - 0.247000823522447 z_k1_s1145
                                 - 0.187000522759169 z_k1_s1146
                                 - 0.223000640236926 z_k1_s1147
                                 - 0.299999170871226 z_k1_s1150
                                 - 0.696997758213255 z_k1_s1151
                                 - 0.150999564736895 z_k1_s1152
                                 - 0.565998728999662 z_k1_s1153
                                 - 0.215999327067469 z_k1_s1154
                                 - 0.383999014674984 z_k1_s1155
                                 - 0.486000384897155 z_k1_s1156
                                 - 0.364999566200251 z_k1_s1157
                                 - 0.128999995563336 z_k1_s1160
                                 - 0.194000618327785 z_k1_s1161
                                 - 0.410999985864581 z_k1_s1162
                                 - 0.10100005902633 z_k1_s1163
                                 - 0.22599936722726 z_k1_s1164
                                 - 0.272000615645151 z_k1_s1165
                                 - 0.42999936021114 z_k1_s1166
                                 - 0.111999933648014 z_k1_s1167
                                 - 0.759999132195719 z_k1_s1170
                                 - 0.271999850785635 z_k1_s1171
                                 - 0.198999759324949 z_k1_s1172
                                 - 0.256999464492802 z_k1_s1173
                                 - 0.500000101443714 z_k1_s1174
                                 - 0.212999608865564 z_k1_s1175
                                 - 0.20099981083066 z_k1_s1176
                                 - 0.152999852191999 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.89627 w_k1_ADM_s000
                                 + 0.879446 w_k1_AXP_s000
                                 + 1.05601 w_k1_DIS_s000
                                 + 1.02812 w_k1_FDX_s000
                                 + 0.762541 w_k1_IBM_s000
                                 + 1.07888 w_k1_INTC_s000
                                 + 0.86564 w_k1_KO_s000
                                 + 0.580733 w_k1_MAT_s000
                                 + 0.878446 w_k1_MSFT_s000
                                 + 0.923784 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.666186 w_k1_ADM_s000
                                 + 0.736482 w_k1_AXP_s000
                                 + 0.648756 w_k1_DIS_s000
                                 + 0.761957 w_k1_FDX_s000
                                 + 0.895583 w_k1_IBM_s000
                                 + 0.798715 w_k1_INTC_s000
                                 + 0.740522 w_k1_KO_s000
                                 + 0.618254 w_k1_MAT_s000
                                 + 0.738741 w_k1_MSFT_s000
                                 + 0.864235 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.1041 w_k1_ADM_s000
                                 + 1.14238 w_k1_AXP_s000 + 1.2795 w_k1_DIS_s000
                                 + 1.09081 w_k1_FDX_s000
                                 + 1.59324 w_k1_IBM_s000
                                 + 1.06122 w_k1_INTC_s000
                                 + 1.20456 w_k1_KO_s000
                                 + 0.952632 w_k1_MAT_s000
                                 + 0.943967 w_k1_MSFT_s000
                                 + 1.02989 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.970938 w_k1_ADM_s000
                                 + 1.01936 w_k1_AXP_s000
                                 + 0.939946 w_k1_DIS_s000
                                 + 0.875108 w_k1_FDX_s000
                                 + 1.12695 w_k1_IBM_s000
                                 + 0.817204 w_k1_INTC_s000
                                 + 1.0819 w_k1_KO_s000 + 0.906596 w_k1_MAT_s000
                                 + 0.813169 w_k1_MSFT_s000
                                 + 0.898613 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.731975 w_k1_ADM_s000
                                 + 0.706624 w_k1_AXP_s000
                                 + 0.779558 w_k1_DIS_s000
                                 + 0.843796 w_k1_FDX_s000
                                 + 0.750114 w_k1_IBM_s000
                                 + 0.848909 w_k1_INTC_s000
                                 + 0.748689 w_k1_KO_s000
                                 + 0.564491 w_k1_MAT_s000
                                 + 0.732876 w_k1_MSFT_s000
                                 + 0.803738 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.01411 w_k1_ADM_s000
                                 + 1.11291 w_k1_AXP_s000
                                 + 1.37339 w_k1_DIS_s000
                                 + 1.15044 w_k1_FDX_s000
                                 + 1.19815 w_k1_IBM_s000
                                 + 1.00809 w_k1_INTC_s000
                                 + 1.12651 w_k1_KO_s000
                                 + 0.830076 w_k1_MAT_s000
                                 + 0.974193 w_k1_MSFT_s000
                                 + 1.13256 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.26315 w_k1_ADM_s000
                                 + 1.23478 w_k1_AXP_s000
                                 + 1.28575 w_k1_DIS_s000
                                 + 1.31209 w_k1_FDX_s000
                                 + 0.903756 w_k1_IBM_s000
                                 + 1.22226 w_k1_INTC_s000
                                 + 0.983651 w_k1_KO_s000
                                 + 0.694466 w_k1_MAT_s000
                                 + 1.05098 w_k1_MSFT_s000
                                 + 1.05431 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.04939 w_k1_ADM_s000
                                 + 1.08156 w_k1_AXP_s000
                                 + 0.792776 w_k1_DIS_s000
                                 + 0.985944 w_k1_FDX_s000
                                 + 0.872263 w_k1_IBM_s000
                                 + 1.02012 w_k1_INTC_s000
                                 + 0.740379 w_k1_KO_s000
                                 + 0.852317 w_k1_MAT_s000
                                 + 0.904576 w_k1_MSFT_s000
                                 + 0.839665 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.01202 w_k1_ADM_s001
                                 + 1.07478 w_k1_AXP_s001
                                 + 0.774914 w_k1_DIS_s001
                                 + 0.993749 w_k1_FDX_s001
                                 + 0.898201 w_k1_IBM_s001
                                 + 1.11925 w_k1_INTC_s001
                                 + 0.832023 w_k1_KO_s001
                                 + 0.77006 w_k1_MAT_s001
                                 + 1.03564 w_k1_MSFT_s001
                                 + 0.936239 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.897121 w_k1_ADM_s001
                                 + 1.21868 w_k1_AXP_s001
                                 + 1.56255 w_k1_DIS_s001
                                 + 1.15795 w_k1_FDX_s001
                                 + 1.76672 w_k1_IBM_s001
                                 + 1.3913 w_k1_INTC_s001 + 1.40208 w_k1_KO_s001
                                 + 1.82357 w_k1_MAT_s001
                                 + 1.36793 w_k1_MSFT_s001
                                 + 1.05362 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.08761 w_k1_ADM_s001
                                 + 1.42976 w_k1_AXP_s001
                                 + 1.30399 w_k1_DIS_s001
                                 + 1.33945 w_k1_FDX_s001
                                 + 1.13742 w_k1_IBM_s001
                                 + 1.62476 w_k1_INTC_s001 + 1.0659 w_k1_KO_s001
                                 + 0.911389 w_k1_MAT_s001
                                 + 1.31169 w_k1_MSFT_s001
                                 + 1.06936 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.761543 w_k1_ADM_s001
                                 + 0.672491 w_k1_AXP_s001
                                 + 0.562901 w_k1_DIS_s001
                                 + 0.665814 w_k1_FDX_s001
                                 + 0.955237 w_k1_IBM_s001
                                 + 0.720554 w_k1_INTC_s001
                                 + 0.647971 w_k1_KO_s001
                                 + 0.722815 w_k1_MAT_s001
                                 + 0.649904 w_k1_MSFT_s001
                                 + 0.756651 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.776378 w_k1_ADM_s001
                                 + 0.811567 w_k1_AXP_s001
                                 + 0.766649 w_k1_DIS_s001
                                 + 0.858304 w_k1_FDX_s001
                                 + 1.4132 w_k1_IBM_s001
                                 + 0.967898 w_k1_INTC_s001
                                 + 0.912134 w_k1_KO_s001
                                 + 1.20376 w_k1_MAT_s001
                                 + 1.00579 w_k1_MSFT_s001
                                 + 0.92114 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.972467 w_k1_ADM_s001
                                 + 0.992146 w_k1_AXP_s001
                                 + 1.30605 w_k1_DIS_s001 + 1.0083 w_k1_FDX_s001
                                 + 1.4038 w_k1_IBM_s001
                                 + 1.31637 w_k1_INTC_s001
                                 + 1.18481 w_k1_KO_s001 + 1.23921 w_k1_MAT_s001
                                 + 1.30717 w_k1_MSFT_s001
                                 + 0.942404 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.00281 w_k1_ADM_s001
                                 + 0.776948 w_k1_AXP_s001
                                 + 0.888912 w_k1_DIS_s001
                                 + 1.0383 w_k1_FDX_s001 + 0.71719 w_k1_IBM_s001
                                 + 1.21559 w_k1_INTC_s001
                                 + 0.736116 w_k1_KO_s001
                                 + 0.547585 w_k1_MAT_s001
                                 + 1.11793 w_k1_MSFT_s001
                                 + 0.852011 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.24625 w_k1_ADM_s001
                                 + 1.28791 w_k1_AXP_s001
                                 + 1.90939 w_k1_DIS_s001
                                 + 1.53522 w_k1_FDX_s001
                                 + 1.32952 w_k1_IBM_s001
                                 + 1.38356 w_k1_INTC_s001
                                 + 1.30373 w_k1_KO_s001 + 0.88701 w_k1_MAT_s001
                                 + 1.36193 w_k1_MSFT_s001
                                 + 1.21413 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.03222 w_k1_ADM_s002
                                 + 0.916366 w_k1_AXP_s002
                                 + 1.07024 w_k1_DIS_s002
                                 + 0.951031 w_k1_FDX_s002
                                 + 1.36718 w_k1_IBM_s002
                                 + 0.970153 w_k1_INTC_s002
                                 + 1.07498 w_k1_KO_s002
                                 + 0.847343 w_k1_MAT_s002
                                 + 1.02214 w_k1_MSFT_s002
                                 + 0.987333 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.855228 w_k1_ADM_s002
                                 + 0.905817 w_k1_AXP_s002
                                 + 0.669398 w_k1_DIS_s002
                                 + 0.856859 w_k1_FDX_s002
                                 + 0.69784 w_k1_IBM_s002
                                 + 0.882267 w_k1_INTC_s002
                                 + 0.648627 w_k1_KO_s002
                                 + 0.864022 w_k1_MAT_s002
                                 + 0.935273 w_k1_MSFT_s002
                                 + 0.86137 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.751857 w_k1_ADM_s002
                                 + 0.681935 w_k1_AXP_s002
                                 + 0.850923 w_k1_DIS_s002
                                 + 0.840351 w_k1_FDX_s002
                                 + 1.06428 w_k1_IBM_s002
                                 + 0.879703 w_k1_INTC_s002
                                 + 0.989931 w_k1_KO_s002
                                 + 0.804964 w_k1_MAT_s002
                                 + 0.978183 w_k1_MSFT_s002
                                 + 0.999212 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.984818 w_k1_ADM_s002
                                 + 1.37411 w_k1_AXP_s002 + 1.2752 w_k1_DIS_s002
                                 + 1.11192 w_k1_FDX_s002
                                 + 1.36082 w_k1_IBM_s002
                                 + 1.07731 w_k1_INTC_s002
                                 + 1.16849 w_k1_KO_s002 + 2.12999 w_k1_MAT_s002
                                 + 1.26565 w_k1_MSFT_s002
                                 + 1.13879 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.00097 w_k1_ADM_s002
                                 + 1.11404 w_k1_AXP_s002
                                 + 0.846197 w_k1_DIS_s002
                                 + 0.841722 w_k1_FDX_s002
                                 + 1.67019 w_k1_IBM_s002
                                 + 0.941864 w_k1_INTC_s002
                                 + 1.17783 w_k1_KO_s002 + 1.36938 w_k1_MAT_s002
                                 + 0.953762 w_k1_MSFT_s002
                                 + 0.918301 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.37063 w_k1_ADM_s002
                                 + 1.44607 w_k1_AXP_s002
                                 + 1.83406 w_k1_DIS_s002
                                 + 1.54507 w_k1_FDX_s002
                                 + 1.23658 w_k1_IBM_s002
                                 + 1.31402 w_k1_INTC_s002
                                 + 1.22192 w_k1_KO_s002 + 1.31542 w_k1_MAT_s002
                                 + 1.62705 w_k1_MSFT_s002
                                 + 1.33292 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.940933 w_k1_ADM_s002
                                 + 1.15093 w_k1_AXP_s002
                                 + 1.16162 w_k1_DIS_s002
                                 + 1.21069 w_k1_FDX_s002
                                 + 1.03096 w_k1_IBM_s002
                                 + 1.0926 w_k1_INTC_s002 + 1.01617 w_k1_KO_s002
                                 + 1.30309 w_k1_MAT_s002
                                 + 1.17533 w_k1_MSFT_s002
                                 + 0.996654 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.08505 w_k1_ADM_s002
                                 + 0.81671 w_k1_AXP_s002
                                 + 1.11109 w_k1_DIS_s002
                                 + 1.02743 w_k1_FDX_s002
                                 + 0.927357 w_k1_IBM_s002
                                 + 0.858839 w_k1_INTC_s002
                                 + 0.889863 w_k1_KO_s002
                                 + 0.626763 w_k1_MAT_s002
                                 + 1.14128 w_k1_MSFT_s002
                                 + 1.15875 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.972535 w_k1_ADM_s003
                                 + 0.844657 w_k1_AXP_s003
                                 + 1.06127 w_k1_DIS_s003
                                 + 0.931205 w_k1_FDX_s003
                                 + 1.31379 w_k1_IBM_s003
                                 + 0.977775 w_k1_INTC_s003
                                 + 1.12279 w_k1_KO_s003
                                 + 0.936101 w_k1_MAT_s003
                                 + 1.00721 w_k1_MSFT_s003
                                 + 1.00363 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.32274 w_k1_ADM_s003
                                 + 0.837192 w_k1_AXP_s003
                                 + 0.804734 w_k1_DIS_s003
                                 + 0.870394 w_k1_FDX_s003 + 1.038 w_k1_IBM_s003
                                 + 0.727177 w_k1_INTC_s003
                                 + 0.976666 w_k1_KO_s003
                                 + 0.738379 w_k1_MAT_s003
                                 + 0.810385 w_k1_MSFT_s003
                                 + 0.946542 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.737747 w_k1_ADM_s003
                                 + 1.18849 w_k1_AXP_s003
                                 + 0.623747 w_k1_DIS_s003
                                 + 0.815716 w_k1_FDX_s003
                                 + 1.09111 w_k1_IBM_s003
                                 + 0.834155 w_k1_INTC_s003
                                 + 0.675172 w_k1_KO_s003
                                 + 0.602388 w_k1_MAT_s003
                                 + 0.738304 w_k1_MSFT_s003
                                 + 0.805197 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.09351 w_k1_ADM_s003
                                 + 1.51301 w_k1_AXP_s003
                                 + 1.10815 w_k1_DIS_s003
                                 + 1.01391 w_k1_FDX_s003
                                 + 1.89326 w_k1_IBM_s003
                                 + 0.852971 w_k1_INTC_s003
                                 + 1.04409 w_k1_KO_s003 + 2.08759 w_k1_MAT_s003
                                 + 0.905994 w_k1_MSFT_s003
                                 + 0.928714 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.43508 w_k1_ADM_s003
                                 + 1.24569 w_k1_AXP_s003
                                 + 1.21483 w_k1_DIS_s003
                                 + 1.10289 w_k1_FDX_s003
                                 + 1.11429 w_k1_IBM_s003
                                 + 1.07878 w_k1_INTC_s003
                                 + 1.00663 w_k1_KO_s003 + 1.27991 w_k1_MAT_s003
                                 + 1.06491 w_k1_MSFT_s003
                                 + 1.12506 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.884665 w_k1_ADM_s003
                                 + 0.758585 w_k1_AXP_s003
                                 + 0.703376 w_k1_DIS_s003
                                 + 0.797431 w_k1_FDX_s003
                                 + 0.783133 w_k1_IBM_s003
                                 + 0.680585 w_k1_INTC_s003
                                 + 0.609474 w_k1_KO_s003
                                 + 0.606747 w_k1_MAT_s003
                                 + 0.687497 w_k1_MSFT_s003
                                 + 0.855303 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.1547 w_k1_ADM_s003
                                 + 1.17149 w_k1_AXP_s003
                                 + 1.01685 w_k1_DIS_s003
                                 + 0.966301 w_k1_FDX_s003
                                 + 1.29265 w_k1_IBM_s003
                                 + 0.886387 w_k1_INTC_s003
                                 + 0.997047 w_k1_KO_s003
                                 + 0.924999 w_k1_MAT_s003
                                 + 0.962956 w_k1_MSFT_s003
                                 + 0.972459 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.10555 w_k1_ADM_s003
                                 + 1.29055 w_k1_AXP_s003
                                 + 1.69362 w_k1_DIS_s003
                                 + 1.43999 w_k1_FDX_s003
                                 + 1.24495 w_k1_IBM_s003
                                 + 1.33738 w_k1_INTC_s003
                                 + 1.27999 w_k1_KO_s003 + 1.08351 w_k1_MAT_s003
                                 + 1.32326 w_k1_MSFT_s003
                                 + 1.16414 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.878181 w_k1_ADM_s004
                                 + 1.0232 w_k1_AXP_s004
                                 + 0.871309 w_k1_DIS_s004
                                 + 0.874622 w_k1_FDX_s004
                                 + 1.74318 w_k1_IBM_s004
                                 + 0.849175 w_k1_INTC_s004
                                 + 1.10423 w_k1_KO_s004 + 1.68739 w_k1_MAT_s004
                                 + 1.07346 w_k1_MSFT_s004
                                 + 1.02896 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.04508 w_k1_ADM_s004
                                 + 0.840182 w_k1_AXP_s004
                                 + 0.846919 w_k1_DIS_s004
                                 + 0.906556 w_k1_FDX_s004
                                 + 1.00082 w_k1_IBM_s004
                                 + 0.884721 w_k1_INTC_s004
                                 + 0.800335 w_k1_KO_s004
                                 + 0.693591 w_k1_MAT_s004
                                 + 0.938673 w_k1_MSFT_s004
                                 + 0.83103 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.04673 w_k1_ADM_s004
                                 + 0.700544 w_k1_AXP_s004
                                 + 0.649166 w_k1_DIS_s004
                                 + 0.900931 w_k1_FDX_s004
                                 + 0.558995 w_k1_IBM_s004
                                 + 0.824063 w_k1_INTC_s004
                                 + 0.61557 w_k1_KO_s004
                                 + 0.510956 w_k1_MAT_s004
                                 + 0.830751 w_k1_MSFT_s004
                                 + 0.751997 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.947207 w_k1_ADM_s004
                                 + 1.04573 w_k1_AXP_s004
                                 + 1.07357 w_k1_DIS_s004
                                 + 1.11904 w_k1_FDX_s004
                                 + 1.32433 w_k1_IBM_s004
                                 + 1.03833 w_k1_INTC_s004
                                 + 0.958813 w_k1_KO_s004
                                 + 0.943266 w_k1_MAT_s004
                                 + 1.05256 w_k1_MSFT_s004
                                 + 1.07615 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.736081 w_k1_ADM_s004
                                 + 0.688307 w_k1_AXP_s004
                                 + 0.694628 w_k1_DIS_s004
                                 + 0.752889 w_k1_FDX_s004
                                 + 0.772748 w_k1_IBM_s004
                                 + 0.847518 w_k1_INTC_s004
                                 + 0.687158 w_k1_KO_s004
                                 + 0.848893 w_k1_MAT_s004
                                 + 0.810678 w_k1_MSFT_s004
                                 + 0.701793 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.15358 w_k1_ADM_s004
                                 + 1.13616 w_k1_AXP_s004
                                 + 1.25194 w_k1_DIS_s004
                                 + 1.26158 w_k1_FDX_s004
                                 + 0.943639 w_k1_IBM_s004
                                 + 1.60446 w_k1_INTC_s004
                                 + 1.05593 w_k1_KO_s004
                                 + 0.751521 w_k1_MAT_s004
                                 + 1.33631 w_k1_MSFT_s004
                                 + 1.04667 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.962721 w_k1_ADM_s004
                                 + 0.762317 w_k1_AXP_s004
                                 + 0.747837 w_k1_DIS_s004
                                 + 0.846699 w_k1_FDX_s004
                                 + 1.13628 w_k1_IBM_s004
                                 + 0.898875 w_k1_INTC_s004
                                 + 0.776255 w_k1_KO_s004
                                 + 0.824386 w_k1_MAT_s004
                                 + 0.884586 w_k1_MSFT_s004
                                 + 0.775572 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.916235 w_k1_ADM_s004
                                 + 1.01474 w_k1_AXP_s004
                                 + 0.891859 w_k1_DIS_s004
                                 + 1.11089 w_k1_FDX_s004
                                 + 0.862117 w_k1_IBM_s004
                                 + 1.14503 w_k1_INTC_s004
                                 + 0.852932 w_k1_KO_s004
                                 + 1.05969 w_k1_MAT_s004
                                 + 1.1632 w_k1_MSFT_s004
                                 + 0.976631 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.892598 w_k1_ADM_s005
                                 + 1.16225 w_k1_AXP_s005
                                 + 1.01921 w_k1_DIS_s005
                                 + 1.10493 w_k1_FDX_s005
                                 + 0.960235 w_k1_IBM_s005
                                 + 1.1353 w_k1_INTC_s005 + 1.06417 w_k1_KO_s005
                                 + 0.696574 w_k1_MAT_s005
                                 + 1.04189 w_k1_MSFT_s005
                                 + 0.936263 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.969879 w_k1_ADM_s005
                                 + 0.998938 w_k1_AXP_s005
                                 + 1.51057 w_k1_DIS_s005
                                 + 1.07723 w_k1_FDX_s005
                                 + 1.48061 w_k1_IBM_s005
                                 + 1.27166 w_k1_INTC_s005 + 1.4832 w_k1_KO_s005
                                 + 1.42924 w_k1_MAT_s005
                                 + 1.42007 w_k1_MSFT_s005 + 1.1398 w_k1_PG_s005
                                 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.651695 w_k1_ADM_s005
                                 + 0.625473 w_k1_AXP_s005
                                 + 0.722079 w_k1_DIS_s005
                                 + 0.723548 w_k1_FDX_s005
                                 + 0.717627 w_k1_IBM_s005
                                 + 0.865939 w_k1_INTC_s005
                                 + 0.833746 w_k1_KO_s005
                                 + 0.589252 w_k1_MAT_s005
                                 + 0.819033 w_k1_MSFT_s005
                                 + 0.795697 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.89257 w_k1_ADM_s005
                                 + 0.833671 w_k1_AXP_s005
                                 + 0.810124 w_k1_DIS_s005
                                 + 0.809065 w_k1_FDX_s005
                                 + 1.27568 w_k1_IBM_s005
                                 + 0.786047 w_k1_INTC_s005
                                 + 1.0179 w_k1_KO_s005 + 1.32505 w_k1_MAT_s005
                                 + 0.755521 w_k1_MSFT_s005
                                 + 0.912934 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.832257 w_k1_ADM_s005
                                 + 0.968551 w_k1_AXP_s005
                                 + 0.774061 w_k1_DIS_s005
                                 + 0.900935 w_k1_FDX_s005
                                 + 0.880142 w_k1_IBM_s005
                                 + 0.990911 w_k1_INTC_s005
                                 + 0.867806 w_k1_KO_s005
                                 + 1.10858 w_k1_MAT_s005
                                 + 0.91785 w_k1_MSFT_s005
                                 + 0.943492 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.966575 w_k1_ADM_s005
                                 + 1.26097 w_k1_AXP_s005
                                 + 1.21896 w_k1_DIS_s005 + 1.1754 w_k1_FDX_s005
                                 + 1.47676 w_k1_IBM_s005
                                 + 1.11918 w_k1_INTC_s005
                                 + 1.29857 w_k1_KO_s005 + 2.14794 w_k1_MAT_s005
                                 + 1.09944 w_k1_MSFT_s005
                                 + 1.10685 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.980805 w_k1_ADM_s005
                                 + 1.43297 w_k1_AXP_s005
                                 + 1.22622 w_k1_DIS_s005
                                 + 1.36477 w_k1_FDX_s005
                                 + 0.805177 w_k1_IBM_s005
                                 + 1.22638 w_k1_INTC_s005
                                 + 1.00153 w_k1_KO_s005
                                 + 0.899887 w_k1_MAT_s005
                                 + 1.22977 w_k1_MSFT_s005 + 1.057 w_k1_PG_s005
                                 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.997762 w_k1_ADM_s005
                                 + 1.13986 w_k1_AXP_s005
                                 + 1.48559 w_k1_DIS_s005
                                 + 1.22595 w_k1_FDX_s005 + 1.0609 w_k1_IBM_s005
                                 + 1.29883 w_k1_INTC_s005
                                 + 1.17304 w_k1_KO_s005 + 1.2114 w_k1_MAT_s005
                                 + 1.20759 w_k1_MSFT_s005
                                 + 1.05176 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.838838 w_k1_ADM_s006
                                 + 0.905873 w_k1_AXP_s006
                                 + 0.696541 w_k1_DIS_s006
                                 + 0.857906 w_k1_FDX_s006
                                 + 0.97662 w_k1_IBM_s006
                                 + 0.79646 w_k1_INTC_s006
                                 + 0.844292 w_k1_KO_s006
                                 + 0.973653 w_k1_MAT_s006
                                 + 0.853658 w_k1_MSFT_s006
                                 + 0.935184 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.641654 w_k1_ADM_s006
                                 + 0.745675 w_k1_AXP_s006
                                 + 0.780129 w_k1_DIS_s006
                                 + 0.739966 w_k1_FDX_s006
                                 + 1.0658 w_k1_IBM_s006
                                 + 0.693384 w_k1_INTC_s006
                                 + 0.92864 w_k1_KO_s006
                                 + 0.843239 w_k1_MAT_s006
                                 + 0.864646 w_k1_MSFT_s006
                                 + 0.823734 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.842941 w_k1_ADM_s006
                                 + 1.10652 w_k1_AXP_s006
                                 + 0.935243 w_k1_DIS_s006
                                 + 0.999825 w_k1_FDX_s006
                                 + 1.17312 w_k1_IBM_s006
                                 + 0.904335 w_k1_INTC_s006
                                 + 1.05229 w_k1_KO_s006 + 1.63356 w_k1_MAT_s006
                                 + 1.01471 w_k1_MSFT_s006
                                 + 1.14462 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.0207 w_k1_ADM_s006
                                 + 1.34393 w_k1_AXP_s006
                                 + 1.50922 w_k1_DIS_s006
                                 + 1.37373 w_k1_FDX_s006
                                 + 1.08557 w_k1_IBM_s006
                                 + 1.09673 w_k1_INTC_s006
                                 + 1.03844 w_k1_KO_s006 + 1.10695 w_k1_MAT_s006
                                 + 1.2636 w_k1_MSFT_s006 + 1.28764 w_k1_PG_s006
                                 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.943926 w_k1_ADM_s006
                                 + 1.03497 w_k1_AXP_s006
                                 + 0.953913 w_k1_DIS_s006
                                 + 1.05788 w_k1_FDX_s006
                                 + 0.82705 w_k1_IBM_s006
                                 + 0.901025 w_k1_INTC_s006
                                 + 0.871382 w_k1_KO_s006
                                 + 1.03577 w_k1_MAT_s006
                                 + 1.03385 w_k1_MSFT_s006
                                 + 1.01827 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.881207 w_k1_ADM_s006
                                 + 1.08181 w_k1_AXP_s006
                                 + 1.22033 w_k1_DIS_s006
                                 + 1.10739 w_k1_FDX_s006
                                 + 1.02344 w_k1_IBM_s006
                                 + 0.963807 w_k1_INTC_s006
                                 + 1.16037 w_k1_KO_s006
                                 + 0.835605 w_k1_MAT_s006
                                 + 1.09393 w_k1_MSFT_s006
                                 + 1.08234 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.789196 w_k1_ADM_s006
                                 + 0.751178 w_k1_AXP_s006
                                 + 0.64007 w_k1_DIS_s006
                                 + 0.796834 w_k1_FDX_s006
                                 + 0.759877 w_k1_IBM_s006
                                 + 0.841885 w_k1_INTC_s006
                                 + 0.61855 w_k1_KO_s006
                                 + 0.715122 w_k1_MAT_s006
                                 + 0.821758 w_k1_MSFT_s006
                                 + 0.758488 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.90432 w_k1_ADM_s006
                                 + 1.35709 w_k1_AXP_s006
                                 + 1.24579 w_k1_DIS_s006
                                 + 1.09745 w_k1_FDX_s006
                                 + 1.66551 w_k1_IBM_s006
                                 + 1.00204 w_k1_INTC_s006
                                 + 1.19553 w_k1_KO_s006 + 2.07926 w_k1_MAT_s006
                                 + 1.13996 w_k1_MSFT_s006
                                 + 1.11836 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.53084 w_k1_ADM_s007
                                 + 1.47328 w_k1_AXP_s007
                                 + 1.70337 w_k1_DIS_s007
                                 + 1.34158 w_k1_FDX_s007
                                 + 1.10786 w_k1_IBM_s007
                                 + 0.936616 w_k1_INTC_s007
                                 + 1.39061 w_k1_KO_s007 + 1.19975 w_k1_MAT_s007
                                 + 1.19083 w_k1_MSFT_s007
                                 + 1.27529 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.21871 w_k1_ADM_s007
                                 + 1.27015 w_k1_AXP_s007
                                 + 1.24185 w_k1_DIS_s007 + 1.0096 w_k1_FDX_s007
                                 + 1.44824 w_k1_IBM_s007
                                 + 0.806692 w_k1_INTC_s007
                                 + 1.25768 w_k1_KO_s007 + 1.9271 w_k1_MAT_s007
                                 + 0.936646 w_k1_MSFT_s007
                                 + 1.13591 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.74324 w_k1_ADM_s007
                                 + 0.76412 w_k1_AXP_s007
                                 + 0.806246 w_k1_DIS_s007
                                 + 0.788286 w_k1_FDX_s007
                                 + 0.997263 w_k1_IBM_s007
                                 + 0.812661 w_k1_INTC_s007
                                 + 0.97253 w_k1_KO_s007 + 1.36436 w_k1_MAT_s007
                                 + 0.832464 w_k1_MSFT_s007
                                 + 0.947469 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.00419 w_k1_ADM_s007
                                 + 1.15459 w_k1_AXP_s007
                                 + 1.09011 w_k1_DIS_s007
                                 + 1.13049 w_k1_FDX_s007
                                 + 0.92147 w_k1_IBM_s007
                                 + 0.900299 w_k1_INTC_s007
                                 + 1.13355 w_k1_KO_s007 + 1.26391 w_k1_MAT_s007
                                 + 1.03453 w_k1_MSFT_s007
                                 + 1.05458 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.90967 w_k1_ADM_s007
                                 + 0.899903 w_k1_AXP_s007
                                 + 0.930893 w_k1_DIS_s007
                                 + 0.998445 w_k1_FDX_s007
                                 + 1.05627 w_k1_IBM_s007
                                 + 0.90223 w_k1_INTC_s007
                                 + 0.988997 w_k1_KO_s007
                                 + 1.15566 w_k1_MAT_s007
                                 + 1.03399 w_k1_MSFT_s007
                                 + 1.09061 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.24614 w_k1_ADM_s007
                                 + 1.04972 w_k1_AXP_s007
                                 + 0.832492 w_k1_DIS_s007
                                 + 0.939318 w_k1_FDX_s007
                                 + 0.998226 w_k1_IBM_s007
                                 + 0.900388 w_k1_INTC_s007
                                 + 0.853571 w_k1_KO_s007
                                 + 1.00681 w_k1_MAT_s007
                                 + 0.852452 w_k1_MSFT_s007
                                 + 1.00526 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.818529 w_k1_ADM_s007
                                 + 1.07575 w_k1_AXP_s007
                                 + 0.751925 w_k1_DIS_s007
                                 + 0.933121 w_k1_FDX_s007
                                 + 0.719928 w_k1_IBM_s007
                                 + 0.899155 w_k1_INTC_s007
                                 + 0.834396 w_k1_KO_s007
                                 + 1.00793 w_k1_MAT_s007
                                 + 0.852298 w_k1_MSFT_s007
                                 + 1.08184 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.812573 w_k1_ADM_s007
                                 + 0.78964 w_k1_AXP_s007
                                 + 0.669734 w_k1_DIS_s007
                                 + 0.863381 w_k1_FDX_s007
                                 + 0.665362 w_k1_IBM_s007
                                 + 0.875147 w_k1_INTC_s007
                                 + 0.709246 w_k1_KO_s007
                                 + 0.726767 w_k1_MAT_s007
                                 + 0.807591 w_k1_MSFT_s007
                                 + 0.794054 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.15093 w_k1_ADM_s010
                                 + 0.834156 w_k1_AXP_s010
                                 + 1.12284 w_k1_DIS_s010
                                 + 1.09389 w_k1_FDX_s010
                                 + 0.729199 w_k1_IBM_s010
                                 + 1.11195 w_k1_INTC_s010
                                 + 0.992765 w_k1_KO_s010
                                 + 0.531552 w_k1_MAT_s010
                                 + 1.24427 w_k1_MSFT_s010
                                 + 1.06093 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.15316 w_k1_ADM_s010
                                 + 0.942695 w_k1_AXP_s010
                                 + 1.17716 w_k1_DIS_s010
                                 + 1.04165 w_k1_FDX_s010
                                 + 1.22702 w_k1_IBM_s010
                                 + 1.06863 w_k1_INTC_s010
                                 + 1.11614 w_k1_KO_s010
                                 + 0.959068 w_k1_MAT_s010
                                 + 1.01898 w_k1_MSFT_s010
                                 + 0.978713 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.706433 w_k1_ADM_s010
                                 + 0.738928 w_k1_AXP_s010
                                 + 0.666126 w_k1_DIS_s010
                                 + 0.755152 w_k1_FDX_s010
                                 + 0.862376 w_k1_IBM_s010
                                 + 0.718861 w_k1_INTC_s010
                                 + 0.720577 w_k1_KO_s010
                                 + 0.752612 w_k1_MAT_s010
                                 + 0.776804 w_k1_MSFT_s010
                                 + 0.76204 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.822941 w_k1_ADM_s010
                                 + 0.778168 w_k1_AXP_s010
                                 + 0.6901 w_k1_DIS_s010
                                 + 0.740596 w_k1_FDX_s010
                                 + 1.25012 w_k1_IBM_s010
                                 + 0.853311 w_k1_INTC_s010
                                 + 0.886286 w_k1_KO_s010 + 1.0161 w_k1_MAT_s010
                                 + 0.819835 w_k1_MSFT_s010
                                 + 0.904369 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.15706 w_k1_ADM_s010
                                 + 1.25035 w_k1_AXP_s010
                                 + 1.78913 w_k1_DIS_s010
                                 + 1.43604 w_k1_FDX_s010
                                 + 1.13834 w_k1_IBM_s010
                                 + 1.21732 w_k1_INTC_s010 + 1.3989 w_k1_KO_s010
                                 + 1.29703 w_k1_MAT_s010
                                 + 1.4034 w_k1_MSFT_s010 + 1.32095 w_k1_PG_s010
                                 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.15005 w_k1_ADM_s010
                                 + 1.42738 w_k1_AXP_s010
                                 + 1.75682 w_k1_DIS_s010
                                 + 1.24178 w_k1_FDX_s010
                                 + 1.62404 w_k1_IBM_s010
                                 + 1.19751 w_k1_INTC_s010
                                 + 1.31577 w_k1_KO_s010 + 1.5796 w_k1_MAT_s010
                                 + 1.25405 w_k1_MSFT_s010
                                 + 1.12705 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.07692 w_k1_ADM_s010
                                 + 1.01269 w_k1_AXP_s010
                                 + 1.04188 w_k1_DIS_s010
                                 + 1.03021 w_k1_FDX_s010
                                 + 0.968206 w_k1_IBM_s010
                                 + 0.886592 w_k1_INTC_s010
                                 + 0.906863 w_k1_KO_s010
                                 + 1.19894 w_k1_MAT_s010
                                 + 1.07633 w_k1_MSFT_s010
                                 + 1.08323 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.16914 w_k1_ADM_s010
                                 + 1.08713 w_k1_AXP_s010
                                 + 1.55034 w_k1_DIS_s010
                                 + 1.12762 w_k1_FDX_s010
                                 + 1.19417 w_k1_IBM_s010
                                 + 1.1727 w_k1_INTC_s010 + 1.2384 w_k1_KO_s010
                                 + 0.891228 w_k1_MAT_s010
                                 + 1.20508 w_k1_MSFT_s010
                                 + 1.10885 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.95976 w_k1_ADM_s011
                                 + 1.00361 w_k1_AXP_s011
                                 + 1.01874 w_k1_DIS_s011
                                 + 1.01661 w_k1_FDX_s011
                                 + 0.948523 w_k1_IBM_s011
                                 + 0.936856 w_k1_INTC_s011
                                 + 0.999841 w_k1_KO_s011
                                 + 0.965585 w_k1_MAT_s011
                                 + 0.902147 w_k1_MSFT_s011
                                 + 1.02217 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.22532 w_k1_ADM_s011
                                 + 1.45799 w_k1_AXP_s011 + 1.701 w_k1_DIS_s011
                                 + 1.37944 w_k1_FDX_s011
                                 + 1.43058 w_k1_IBM_s011
                                 + 1.38536 w_k1_INTC_s011
                                 + 1.45874 w_k1_KO_s011 + 1.35955 w_k1_MAT_s011
                                 + 1.32125 w_k1_MSFT_s011
                                 + 1.24864 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.08682 w_k1_ADM_s011
                                 + 1.07484 w_k1_AXP_s011
                                 + 0.737777 w_k1_DIS_s011
                                 + 1.12362 w_k1_FDX_s011
                                 + 0.737902 w_k1_IBM_s011
                                 + 1.07748 w_k1_INTC_s011
                                 + 0.791803 w_k1_KO_s011
                                 + 0.726608 w_k1_MAT_s011
                                 + 0.908238 w_k1_MSFT_s011
                                 + 0.90305 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.940155 w_k1_ADM_s011
                                 + 1.30824 w_k1_AXP_s011 + 1.2939 w_k1_DIS_s011
                                 + 1.13666 w_k1_FDX_s011
                                 + 1.54243 w_k1_IBM_s011
                                 + 1.49284 w_k1_INTC_s011
                                 + 1.34789 w_k1_KO_s011 + 1.4658 w_k1_MAT_s011
                                 + 1.16493 w_k1_MSFT_s011
                                 + 1.18493 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.27996 w_k1_ADM_s011
                                 + 1.17216 w_k1_AXP_s011
                                 + 1.30939 w_k1_DIS_s011
                                 + 1.32519 w_k1_FDX_s011
                                 + 1.07805 w_k1_IBM_s011
                                 + 1.22116 w_k1_INTC_s011
                                 + 1.21699 w_k1_KO_s011 + 1.29984 w_k1_MAT_s011
                                 + 1.26089 w_k1_MSFT_s011
                                 + 1.14214 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.0475 w_k1_ADM_s011
                                 + 1.60418 w_k1_AXP_s011 + 1.354 w_k1_DIS_s011
                                 + 1.32114 w_k1_FDX_s011
                                 + 1.00167 w_k1_IBM_s011
                                 + 1.4858 w_k1_INTC_s011 + 1.24277 w_k1_KO_s011
                                 + 1.21052 w_k1_MAT_s011
                                 + 1.31832 w_k1_MSFT_s011
                                 + 1.33175 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.64999 w_k1_ADM_s011
                                 + 0.773663 w_k1_AXP_s011
                                 + 0.622597 w_k1_DIS_s011
                                 + 0.784518 w_k1_FDX_s011
                                 + 0.831695 w_k1_IBM_s011
                                 + 0.887817 w_k1_INTC_s011
                                 + 0.811705 w_k1_KO_s011
                                 + 0.849481 w_k1_MAT_s011
                                 + 0.76726 w_k1_MSFT_s011
                                 + 0.939568 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.764011 w_k1_ADM_s011
                                 + 0.756084 w_k1_AXP_s011
                                 + 0.855805 w_k1_DIS_s011
                                 + 0.925117 w_k1_FDX_s011
                                 + 1.01179 w_k1_IBM_s011
                                 + 0.992107 w_k1_INTC_s011
                                 + 0.967064 w_k1_KO_s011
                                 + 0.811057 w_k1_MAT_s011
                                 + 0.871208 w_k1_MSFT_s011
                                 + 0.840144 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.28322 w_k1_ADM_s012
                                 + 0.933668 w_k1_AXP_s012
                                 + 0.800379 w_k1_DIS_s012
                                 + 0.889966 w_k1_FDX_s012
                                 + 1.14722 w_k1_IBM_s012
                                 + 0.961236 w_k1_INTC_s012
                                 + 1.05418 w_k1_KO_s012
                                 + 0.941962 w_k1_MAT_s012
                                 + 0.865345 w_k1_MSFT_s012
                                 + 0.982234 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.31592 w_k1_ADM_s012
                                 + 1.022 w_k1_AXP_s012 + 1.26357 w_k1_DIS_s012
                                 + 1.11387 w_k1_FDX_s012
                                 + 1.27896 w_k1_IBM_s012
                                 + 1.32124 w_k1_INTC_s012
                                 + 1.34808 w_k1_KO_s012 + 0.94191 w_k1_MAT_s012
                                 + 1.30116 w_k1_MSFT_s012
                                 + 1.13884 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.04036 w_k1_ADM_s012
                                 + 0.653658 w_k1_AXP_s012
                                 + 0.753147 w_k1_DIS_s012
                                 + 0.844578 w_k1_FDX_s012
                                 + 0.839528 w_k1_IBM_s012
                                 + 0.905649 w_k1_INTC_s012
                                 + 0.980059 w_k1_KO_s012
                                 + 0.733832 w_k1_MAT_s012
                                 + 0.876898 w_k1_MSFT_s012
                                 + 0.870717 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.16712 w_k1_ADM_s012
                                 + 1.11355 w_k1_AXP_s012
                                 + 1.21056 w_k1_DIS_s012
                                 + 1.16602 w_k1_FDX_s012
                                 + 0.96347 w_k1_IBM_s012
                                 + 1.22611 w_k1_INTC_s012
                                 + 1.01228 w_k1_KO_s012
                                 + 0.818876 w_k1_MAT_s012
                                 + 1.05245 w_k1_MSFT_s012
                                 + 0.995458 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.14832 w_k1_ADM_s012
                                 + 1.38255 w_k1_AXP_s012
                                 + 1.29093 w_k1_DIS_s012
                                 + 1.03469 w_k1_FDX_s012
                                 + 1.57183 w_k1_IBM_s012
                                 + 1.09267 w_k1_INTC_s012
                                 + 1.36834 w_k1_KO_s012 + 1.58641 w_k1_MAT_s012
                                 + 1.04293 w_k1_MSFT_s012
                                 + 1.10913 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.28166 w_k1_ADM_s012
                                 + 0.811914 w_k1_AXP_s012
                                 + 1.02719 w_k1_DIS_s012
                                 + 1.14057 w_k1_FDX_s012
                                 + 0.770447 w_k1_IBM_s012
                                 + 1.1175 w_k1_INTC_s012
                                 + 0.931526 w_k1_KO_s012
                                 + 0.570402 w_k1_MAT_s012
                                 + 1.13304 w_k1_MSFT_s012
                                 + 0.948504 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.980901 w_k1_ADM_s012
                                 + 0.76742 w_k1_AXP_s012
                                 + 0.648733 w_k1_DIS_s012
                                 + 0.908751 w_k1_FDX_s012
                                 + 0.630706 w_k1_IBM_s012
                                 + 0.875976 w_k1_INTC_s012
                                 + 0.718157 w_k1_KO_s012
                                 + 0.625268 w_k1_MAT_s012
                                 + 0.898222 w_k1_MSFT_s012
                                 + 0.828736 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.35503 w_k1_ADM_s012
                                 + 0.992215 w_k1_AXP_s012
                                 + 0.825687 w_k1_DIS_s012
                                 + 1.00948 w_k1_FDX_s012
                                 + 0.769365 w_k1_IBM_s012
                                 + 0.820696 w_k1_INTC_s012
                                 + 0.811444 w_k1_KO_s012
                                 + 0.783579 w_k1_MAT_s012
                                 + 0.887969 w_k1_MSFT_s012
                                 + 0.850379 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.932453 w_k1_ADM_s013
                                 + 0.897046 w_k1_AXP_s013
                                 + 0.983652 w_k1_DIS_s013
                                 + 0.998703 w_k1_FDX_s013
                                 + 0.830797 w_k1_IBM_s013
                                 + 1.13101 w_k1_INTC_s013
                                 + 0.856054 w_k1_KO_s013
                                 + 0.907112 w_k1_MAT_s013
                                 + 1.06492 w_k1_MSFT_s013
                                 + 0.929788 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.37762 w_k1_ADM_s013
                                 + 1.42957 w_k1_AXP_s013 + 1.858 w_k1_DIS_s013
                                 + 1.56778 w_k1_FDX_s013
                                 + 0.937037 w_k1_IBM_s013
                                 + 1.5491 w_k1_INTC_s013 + 1.12781 w_k1_KO_s013
                                 + 1.27814 w_k1_MAT_s013
                                 + 1.56041 w_k1_MSFT_s013
                                 + 1.31437 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.959662 w_k1_ADM_s013
                                 + 1.05818 w_k1_AXP_s013
                                 + 1.00762 w_k1_DIS_s013
                                 + 0.892674 w_k1_FDX_s013
                                 + 1.41304 w_k1_IBM_s013
                                 + 1.13583 w_k1_INTC_s013
                                 + 1.04646 w_k1_KO_s013 + 1.31543 w_k1_MAT_s013
                                 + 0.890854 w_k1_MSFT_s013
                                 + 0.996086 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.08336 w_k1_ADM_s013
                                 + 0.985814 w_k1_AXP_s013
                                 + 1.14977 w_k1_DIS_s013
                                 + 1.14619 w_k1_FDX_s013
                                 + 0.813793 w_k1_IBM_s013
                                 + 1.23756 w_k1_INTC_s013
                                 + 0.88241 w_k1_KO_s013
                                 + 0.711253 w_k1_MAT_s013
                                 + 1.23296 w_k1_MSFT_s013
                                 + 1.01718 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.15208 w_k1_ADM_s013
                                 + 1.47919 w_k1_AXP_s013
                                 + 1.07781 w_k1_DIS_s013
                                 + 1.18122 w_k1_FDX_s013
                                 + 1.05057 w_k1_IBM_s013
                                 + 1.08075 w_k1_INTC_s013
                                 + 0.935233 w_k1_KO_s013
                                 + 1.52708 w_k1_MAT_s013
                                 + 1.10209 w_k1_MSFT_s013
                                 + 1.03379 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.734661 w_k1_ADM_s013
                                 + 0.810482 w_k1_AXP_s013
                                 + 0.747989 w_k1_DIS_s013
                                 + 0.873752 w_k1_FDX_s013
                                 + 0.884418 w_k1_IBM_s013
                                 + 0.876332 w_k1_INTC_s013
                                 + 0.789427 w_k1_KO_s013
                                 + 0.955516 w_k1_MAT_s013
                                 + 0.853943 w_k1_MSFT_s013
                                 + 0.817616 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.29893 w_k1_ADM_s013
                                 + 1.23639 w_k1_AXP_s013
                                 + 1.35716 w_k1_DIS_s013
                                 + 1.29188 w_k1_FDX_s013
                                 + 1.28493 w_k1_IBM_s013
                                 + 1.06259 w_k1_INTC_s013
                                 + 1.11788 w_k1_KO_s013 + 1.52744 w_k1_MAT_s013
                                 + 1.25683 w_k1_MSFT_s013
                                 + 1.11282 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.661456 w_k1_ADM_s013
                                 + 0.637413 w_k1_AXP_s013
                                 + 0.585698 w_k1_DIS_s013
                                 + 0.672163 w_k1_FDX_s013
                                 + 0.682907 w_k1_IBM_s013
                                 + 0.790618 w_k1_INTC_s013
                                 + 0.559054 w_k1_KO_s013
                                 + 0.597956 w_k1_MAT_s013
                                 + 0.745651 w_k1_MSFT_s013
                                 + 0.757538 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.78645 w_k1_ADM_s014
                                 + 0.720323 w_k1_AXP_s014
                                 + 0.837744 w_k1_DIS_s014
                                 + 0.868997 w_k1_FDX_s014
                                 + 1.10547 w_k1_IBM_s014
                                 + 0.94703 w_k1_INTC_s014
                                 + 0.967538 w_k1_KO_s014
                                 + 0.892039 w_k1_MAT_s014
                                 + 0.939815 w_k1_MSFT_s014
                                 + 0.947622 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.919688 w_k1_ADM_s014
                                 + 1.08621 w_k1_AXP_s014
                                 + 0.672609 w_k1_DIS_s014
                                 + 0.887568 w_k1_FDX_s014
                                 + 0.834399 w_k1_IBM_s014
                                 + 0.848126 w_k1_INTC_s014
                                 + 0.763941 w_k1_KO_s014
                                 + 1.10978 w_k1_MAT_s014
                                 + 0.780155 w_k1_MSFT_s014
                                 + 0.897506 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.28732 w_k1_ADM_s014
                                 + 1.71107 w_k1_AXP_s014
                                 + 1.43665 w_k1_DIS_s014
                                 + 1.44667 w_k1_FDX_s014
                                 + 0.990928 w_k1_IBM_s014
                                 + 1.03269 w_k1_INTC_s014
                                 + 1.17022 w_k1_KO_s014 + 1.66944 w_k1_MAT_s014
                                 + 1.14581 w_k1_MSFT_s014
                                 + 1.32575 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.2557 w_k1_ADM_s014
                                 + 1.79047 w_k1_AXP_s014
                                 + 1.57193 w_k1_DIS_s014
                                 + 1.42306 w_k1_FDX_s014
                                 + 1.49386 w_k1_IBM_s014
                                 + 1.33739 w_k1_INTC_s014 + 1.4522 w_k1_KO_s014
                                 + 1.74257 w_k1_MAT_s014
                                 + 1.24431 w_k1_MSFT_s014
                                 + 1.31743 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.05127 w_k1_ADM_s014
                                 + 1.0809 w_k1_AXP_s014 + 1.17162 w_k1_DIS_s014
                                 + 1.05381 w_k1_FDX_s014
                                 + 1.22998 w_k1_IBM_s014
                                 + 1.09448 w_k1_INTC_s014
                                 + 1.08882 w_k1_KO_s014 + 1.06049 w_k1_MAT_s014
                                 + 1.09138 w_k1_MSFT_s014
                                 + 0.918254 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.02389 w_k1_ADM_s014
                                 + 0.920511 w_k1_AXP_s014
                                 + 1.05797 w_k1_DIS_s014
                                 + 1.13501 w_k1_FDX_s014
                                 + 0.930384 w_k1_IBM_s014
                                 + 1.23798 w_k1_INTC_s014
                                 + 1.04241 w_k1_KO_s014 + 1.03259 w_k1_MAT_s014
                                 + 1.31192 w_k1_MSFT_s014
                                 + 1.05292 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.870691 w_k1_ADM_s014
                                 + 1.35771 w_k1_AXP_s014
                                 + 1.36046 w_k1_DIS_s014
                                 + 1.02475 w_k1_FDX_s014
                                 + 1.84087 w_k1_IBM_s014
                                 + 1.07999 w_k1_INTC_s014
                                 + 1.30474 w_k1_KO_s014 + 1.60353 w_k1_MAT_s014
                                 + 0.920484 w_k1_MSFT_s014
                                 + 1.0836 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.817892 w_k1_ADM_s014
                                 + 0.774059 w_k1_AXP_s014
                                 + 0.65795 w_k1_DIS_s014
                                 + 0.910295 w_k1_FDX_s014
                                 + 0.752062 w_k1_IBM_s014
                                 + 1.123 w_k1_INTC_s014 + 0.825476 w_k1_KO_s014
                                 + 0.662319 w_k1_MAT_s014
                                 + 0.97325 w_k1_MSFT_s014
                                 + 0.865235 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.812319 w_k1_ADM_s015
                                 + 0.735842 w_k1_AXP_s015
                                 + 0.596266 w_k1_DIS_s015
                                 + 0.815378 w_k1_FDX_s015
                                 + 0.675176 w_k1_IBM_s015
                                 + 0.875205 w_k1_INTC_s015
                                 + 0.579511 w_k1_KO_s015
                                 + 0.606262 w_k1_MAT_s015
                                 + 0.865766 w_k1_MSFT_s015
                                 + 0.718425 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.907683 w_k1_ADM_s015
                                 + 1.21076 w_k1_AXP_s015 + 1.2883 w_k1_DIS_s015
                                 + 1.13319 w_k1_FDX_s015
                                 + 1.12047 w_k1_IBM_s015
                                 + 1.37972 w_k1_INTC_s015
                                 + 1.02577 w_k1_KO_s015
                                 + 0.949614 w_k1_MAT_s015
                                 + 1.22516 w_k1_MSFT_s015
                                 + 1.03699 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.769318 w_k1_ADM_s015
                                 + 1.10557 w_k1_AXP_s015
                                 + 1.05588 w_k1_DIS_s015
                                 + 0.990092 w_k1_FDX_s015
                                 + 1.35201 w_k1_IBM_s015
                                 + 1.0425 w_k1_INTC_s015
                                 + 0.993945 w_k1_KO_s015
                                 + 1.39917 w_k1_MAT_s015
                                 + 0.993759 w_k1_MSFT_s015
                                 + 0.974053 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.02488 w_k1_ADM_s015
                                 + 1.57391 w_k1_AXP_s015
                                 + 1.60153 w_k1_DIS_s015
                                 + 1.23017 w_k1_FDX_s015
                                 + 1.71795 w_k1_IBM_s015
                                 + 1.29794 w_k1_INTC_s015
                                 + 1.48627 w_k1_KO_s015 + 2.09901 w_k1_MAT_s015
                                 + 1.33921 w_k1_MSFT_s015
                                 + 1.24926 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.02065 w_k1_ADM_s015
                                 + 0.982962 w_k1_AXP_s015
                                 + 1.08655 w_k1_DIS_s015
                                 + 0.964423 w_k1_FDX_s015
                                 + 1.05528 w_k1_IBM_s015
                                 + 1.33566 w_k1_INTC_s015
                                 + 1.04559 w_k1_KO_s015
                                 + 0.673273 w_k1_MAT_s015
                                 + 0.990895 w_k1_MSFT_s015
                                 + 1.07174 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.885377 w_k1_ADM_s015
                                 + 0.889221 w_k1_AXP_s015
                                 + 0.881289 w_k1_DIS_s015
                                 + 1.01985 w_k1_FDX_s015
                                 + 0.790015 w_k1_IBM_s015
                                 + 1.05167 w_k1_INTC_s015
                                 + 0.912106 w_k1_KO_s015
                                 + 0.966188 w_k1_MAT_s015
                                 + 0.914657 w_k1_MSFT_s015
                                 + 0.931446 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.62262 w_k1_ADM_s015
                                 + 0.715516 w_k1_AXP_s015
                                 + 0.780732 w_k1_DIS_s015
                                 + 0.780728 w_k1_FDX_s015
                                 + 0.96791 w_k1_IBM_s015
                                 + 0.776149 w_k1_INTC_s015
                                 + 0.864146 w_k1_KO_s015
                                 + 1.00877 w_k1_MAT_s015
                                 + 0.90234 w_k1_MSFT_s015
                                 + 0.782749 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.04904 w_k1_ADM_s015
                                 + 1.43662 w_k1_AXP_s015
                                 + 1.63011 w_k1_DIS_s015
                                 + 1.35677 w_k1_FDX_s015
                                 + 1.12384 w_k1_IBM_s015
                                 + 1.40036 w_k1_INTC_s015
                                 + 1.24329 w_k1_KO_s015
                                 + 0.748406 w_k1_MAT_s015
                                 + 1.42741 w_k1_MSFT_s015
                                 + 1.03602 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.910726 w_k1_ADM_s016
                                 + 0.959501 w_k1_AXP_s016
                                 + 1.05537 w_k1_DIS_s016
                                 + 1.00221 w_k1_FDX_s016
                                 + 1.04022 w_k1_IBM_s016
                                 + 0.997089 w_k1_INTC_s016
                                 + 1.08258 w_k1_KO_s016
                                 + 0.998678 w_k1_MAT_s016
                                 + 1.07617 w_k1_MSFT_s016
                                 + 1.11094 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.930758 w_k1_ADM_s016
                                 + 0.881649 w_k1_AXP_s016
                                 + 0.795848 w_k1_DIS_s016
                                 + 1.01963 w_k1_FDX_s016
                                 + 0.816416 w_k1_IBM_s016
                                 + 1.00308 w_k1_INTC_s016
                                 + 0.920625 w_k1_KO_s016
                                 + 0.96833 w_k1_MAT_s016
                                 + 0.982822 w_k1_MSFT_s016
                                 + 1.04935 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.05095 w_k1_ADM_s016
                                 + 1.07547 w_k1_AXP_s016
                                 + 1.25673 w_k1_DIS_s016
                                 + 1.08768 w_k1_FDX_s016
                                 + 1.08637 w_k1_IBM_s016
                                 + 1.20059 w_k1_INTC_s016
                                 + 1.28662 w_k1_KO_s016 + 1.20049 w_k1_MAT_s016
                                 + 1.21068 w_k1_MSFT_s016
                                 + 1.18308 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.5887 w_k1_ADM_s016
                                 + 1.48935 w_k1_AXP_s016
                                 + 1.70404 w_k1_DIS_s016
                                 + 1.50262 w_k1_FDX_s016
                                 + 1.38016 w_k1_IBM_s016
                                 + 1.1716 w_k1_INTC_s016 + 1.37622 w_k1_KO_s016
                                 + 1.40743 w_k1_MAT_s016
                                 + 1.24048 w_k1_MSFT_s016
                                 + 1.44736 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.21058 w_k1_ADM_s016
                                 + 1.03011 w_k1_AXP_s016
                                 + 1.05269 w_k1_DIS_s016
                                 + 1.15557 w_k1_FDX_s016
                                 + 0.720694 w_k1_IBM_s016
                                 + 0.930289 w_k1_INTC_s016
                                 + 0.984372 w_k1_KO_s016
                                 + 0.978375 w_k1_MAT_s016
                                 + 1.05675 w_k1_MSFT_s016
                                 + 1.10748 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.95904 w_k1_ADM_s016
                                 + 0.814034 w_k1_AXP_s016
                                 + 0.921692 w_k1_DIS_s016
                                 + 0.844878 w_k1_FDX_s016
                                 + 0.731026 w_k1_IBM_s016
                                 + 0.856878 w_k1_INTC_s016
                                 + 0.920678 w_k1_KO_s016
                                 + 0.664832 w_k1_MAT_s016
                                 + 0.810741 w_k1_MSFT_s016
                                 + 0.973094 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.824204 w_k1_ADM_s016
                                 + 0.84391 w_k1_AXP_s016
                                 + 0.666329 w_k1_DIS_s016
                                 + 0.810157 w_k1_FDX_s016
                                 + 0.956945 w_k1_IBM_s016
                                 + 0.796458 w_k1_INTC_s016
                                 + 0.857717 w_k1_KO_s016
                                 + 0.978392 w_k1_MAT_s016
                                 + 0.796982 w_k1_MSFT_s016
                                 + 0.942248 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.0127 w_k1_ADM_s016
                                 + 1.2649 w_k1_AXP_s016 + 1.25466 w_k1_DIS_s016
                                 + 1.14633 w_k1_FDX_s016
                                 + 1.35117 w_k1_IBM_s016
                                 + 1.20949 w_k1_INTC_s016
                                 + 1.21601 w_k1_KO_s016 + 1.67017 w_k1_MAT_s016
                                 + 1.01559 w_k1_MSFT_s016
                                 + 1.25305 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.13063 w_k1_ADM_s017
                                 + 1.02691 w_k1_AXP_s017
                                 + 1.07465 w_k1_DIS_s017
                                 + 1.09944 w_k1_FDX_s017
                                 + 0.775198 w_k1_IBM_s017
                                 + 1.07856 w_k1_INTC_s017
                                 + 0.968748 w_k1_KO_s017
                                 + 0.823865 w_k1_MAT_s017
                                 + 1.09227 w_k1_MSFT_s017 + 1.005 w_k1_PG_s017
                                 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.09933 w_k1_ADM_s017
                                 + 1.02541 w_k1_AXP_s017
                                 + 0.962475 w_k1_DIS_s017
                                 + 0.939191 w_k1_FDX_s017
                                 + 0.827628 w_k1_IBM_s017
                                 + 0.853503 w_k1_INTC_s017
                                 + 0.842688 w_k1_KO_s017
                                 + 1.13367 w_k1_MAT_s017
                                 + 0.924879 w_k1_MSFT_s017
                                 + 0.903976 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.80217 w_k1_ADM_s017
                                 + 1.25573 w_k1_AXP_s017 + 1.5699 w_k1_DIS_s017
                                 + 1.21221 w_k1_FDX_s017
                                 + 2.06088 w_k1_IBM_s017
                                 + 1.14609 w_k1_INTC_s017
                                 + 1.74749 w_k1_KO_s017 + 2.74696 w_k1_MAT_s017
                                 + 1.37012 w_k1_MSFT_s017
                                 + 1.39611 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.36426 w_k1_ADM_s017
                                 + 1.2369 w_k1_AXP_s017 + 1.23978 w_k1_DIS_s017
                                 + 1.09918 w_k1_FDX_s017
                                 + 1.35732 w_k1_IBM_s017
                                 + 1.00746 w_k1_INTC_s017 + 1.1005 w_k1_KO_s017
                                 + 1.18621 w_k1_MAT_s017
                                 + 0.994432 w_k1_MSFT_s017
                                 + 1.03547 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.03824 w_k1_ADM_s017
                                 + 0.993119 w_k1_AXP_s017
                                 + 0.851746 w_k1_DIS_s017
                                 + 0.820224 w_k1_FDX_s017
                                 + 1.12333 w_k1_IBM_s017
                                 + 0.791139 w_k1_INTC_s017
                                 + 1.07111 w_k1_KO_s017 + 1.20799 w_k1_MAT_s017
                                 + 0.856588 w_k1_MSFT_s017
                                 + 0.927528 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.30368 w_k1_ADM_s017
                                 + 1.4522 w_k1_AXP_s017 + 1.47984 w_k1_DIS_s017
                                 + 1.2484 w_k1_FDX_s017 + 1.14229 w_k1_IBM_s017
                                 + 1.07609 w_k1_INTC_s017
                                 + 1.32314 w_k1_KO_s017 + 1.61511 w_k1_MAT_s017
                                 + 1.1935 w_k1_MSFT_s017 + 1.15308 w_k1_PG_s017
                                 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.76809 w_k1_ADM_s017
                                 + 0.66704 w_k1_AXP_s017
                                 + 0.981434 w_k1_DIS_s017
                                 + 0.815457 w_k1_FDX_s017
                                 + 0.832273 w_k1_IBM_s017
                                 + 0.945534 w_k1_INTC_s017
                                 + 0.946548 w_k1_KO_s017
                                 + 0.798519 w_k1_MAT_s017
                                 + 0.940636 w_k1_MSFT_s017
                                 + 0.81536 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.903359 w_k1_ADM_s017
                                 + 0.913445 w_k1_AXP_s017
                                 + 0.661503 w_k1_DIS_s017
                                 + 0.812599 w_k1_FDX_s017
                                 + 0.693158 w_k1_IBM_s017
                                 + 1.00736 w_k1_INTC_s017
                                 + 0.70867 w_k1_KO_s017
                                 + 0.805045 w_k1_MAT_s017
                                 + 0.902179 w_k1_MSFT_s017
                                 + 0.743548 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.09441 w_k1_ADM_s100
                                 + 0.76933 w_k1_AXP_s100
                                 + 0.936502 w_k1_DIS_s100
                                 + 1.00001 w_k1_FDX_s100
                                 + 0.655683 w_k1_IBM_s100
                                 + 1.11992 w_k1_INTC_s100
                                 + 0.743891 w_k1_KO_s100
                                 + 0.475195 w_k1_MAT_s100
                                 + 1.06591 w_k1_MSFT_s100
                                 + 0.81311 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.01878 w_k1_ADM_s100
                                 + 0.956592 w_k1_AXP_s100
                                 + 0.709413 w_k1_DIS_s100
                                 + 0.88437 w_k1_FDX_s100
                                 + 1.09552 w_k1_IBM_s100
                                 + 0.909811 w_k1_INTC_s100
                                 + 0.812197 w_k1_KO_s100
                                 + 0.882836 w_k1_MAT_s100
                                 + 0.82328 w_k1_MSFT_s100
                                 + 0.848824 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.58588 w_k1_ADM_s100
                                 + 1.44268 w_k1_AXP_s100
                                 + 1.84865 w_k1_DIS_s100
                                 + 1.56889 w_k1_FDX_s100
                                 + 1.31388 w_k1_IBM_s100
                                 + 1.27801 w_k1_INTC_s100
                                 + 1.55998 w_k1_KO_s100 + 1.2844 w_k1_MAT_s100
                                 + 1.45066 w_k1_MSFT_s100
                                 + 1.22876 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.04841 w_k1_ADM_s100
                                 + 1.22318 w_k1_AXP_s100
                                 + 1.25844 w_k1_DIS_s100
                                 + 1.10127 w_k1_FDX_s100
                                 + 1.33946 w_k1_IBM_s100
                                 + 0.938083 w_k1_INTC_s100
                                 + 1.26427 w_k1_KO_s100 + 1.58101 w_k1_MAT_s100
                                 + 1.11877 w_k1_MSFT_s100
                                 + 1.03994 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.998607 w_k1_ADM_s100
                                 + 1.1538 w_k1_AXP_s100 + 1.12307 w_k1_DIS_s100
                                 + 1.10077 w_k1_FDX_s100
                                 + 0.88158 w_k1_IBM_s100
                                 + 1.27025 w_k1_INTC_s100
                                 + 0.881285 w_k1_KO_s100
                                 + 0.801895 w_k1_MAT_s100
                                 + 1.10297 w_k1_MSFT_s100
                                 + 0.976142 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.8133 w_k1_ADM_s100
                                 + 0.645214 w_k1_AXP_s100
                                 + 0.946513 w_k1_DIS_s100
                                 + 0.776234 w_k1_FDX_s100
                                 + 1.25826 w_k1_IBM_s100
                                 + 0.96025 w_k1_INTC_s100
                                 + 0.967024 w_k1_KO_s100
                                 + 0.829321 w_k1_MAT_s100
                                 + 1.0585 w_k1_MSFT_s100
                                 + 0.905481 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.835869 w_k1_ADM_s100
                                 + 0.772852 w_k1_AXP_s100
                                 + 0.77224 w_k1_DIS_s100
                                 + 0.839951 w_k1_FDX_s100
                                 + 0.713892 w_k1_IBM_s100
                                 + 0.955529 w_k1_INTC_s100
                                 + 0.747227 w_k1_KO_s100
                                 + 0.623481 w_k1_MAT_s100
                                 + 0.930979 w_k1_MSFT_s100
                                 + 0.873171 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.754412 w_k1_ADM_s100
                                 + 0.571475 w_k1_AXP_s100
                                 + 0.547805 w_k1_DIS_s100
                                 + 0.71653 w_k1_FDX_s100
                                 + 0.664829 w_k1_IBM_s100
                                 + 0.736246 w_k1_INTC_s100
                                 + 0.656936 w_k1_KO_s100
                                 + 0.525293 w_k1_MAT_s100
                                 + 0.758515 w_k1_MSFT_s100
                                 + 0.753766 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.03751 w_k1_ADM_s101
                                 + 0.823834 w_k1_AXP_s101
                                 + 0.866085 w_k1_DIS_s101
                                 + 0.911714 w_k1_FDX_s101
                                 + 1.08227 w_k1_IBM_s101
                                 + 0.813984 w_k1_INTC_s101
                                 + 1.08053 w_k1_KO_s101 + 1.27784 w_k1_MAT_s101
                                 + 0.901088 w_k1_MSFT_s101
                                 + 0.993319 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.929614 w_k1_ADM_s101
                                 + 0.963524 w_k1_AXP_s101
                                 + 1.0151 w_k1_DIS_s101 + 1.04686 w_k1_FDX_s101
                                 + 0.774448 w_k1_IBM_s101
                                 + 0.967654 w_k1_INTC_s101
                                 + 0.940941 w_k1_KO_s101
                                 + 0.885983 w_k1_MAT_s101
                                 + 1.0746 w_k1_MSFT_s101
                                 + 0.927765 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.830985 w_k1_ADM_s101
                                 + 0.678678 w_k1_AXP_s101
                                 + 0.773349 w_k1_DIS_s101
                                 + 0.881226 w_k1_FDX_s101
                                 + 0.780822 w_k1_IBM_s101
                                 + 0.732616 w_k1_INTC_s101
                                 + 0.905457 w_k1_KO_s101
                                 + 0.721682 w_k1_MAT_s101
                                 + 0.871864 w_k1_MSFT_s101
                                 + 0.821187 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.05523 w_k1_ADM_s101
                                 + 1.17797 w_k1_AXP_s101
                                 + 1.48735 w_k1_DIS_s101 + 1.3326 w_k1_FDX_s101
                                 + 1.09562 w_k1_IBM_s101
                                 + 1.11171 w_k1_INTC_s101
                                 + 1.40785 w_k1_KO_s101 + 1.28151 w_k1_MAT_s101
                                 + 1.4377 w_k1_MSFT_s101 + 1.25541 w_k1_PG_s101
                                 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.956528 w_k1_ADM_s101
                                 + 0.853109 w_k1_AXP_s101
                                 + 0.964309 w_k1_DIS_s101
                                 + 0.933436 w_k1_FDX_s101
                                 + 1.27775 w_k1_IBM_s101
                                 + 0.854754 w_k1_INTC_s101
                                 + 1.19009 w_k1_KO_s101 + 1.44799 w_k1_MAT_s101
                                 + 1.06241 w_k1_MSFT_s101
                                 + 1.08961 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.51618 w_k1_ADM_s101
                                 + 1.1302 w_k1_AXP_s101 + 1.35332 w_k1_DIS_s101
                                 + 1.24914 w_k1_FDX_s101
                                 + 1.41789 w_k1_IBM_s101
                                 + 1.10171 w_k1_INTC_s101
                                 + 1.31747 w_k1_KO_s101 + 1.70151 w_k1_MAT_s101
                                 + 1.31367 w_k1_MSFT_s101
                                 + 1.16758 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.743347 w_k1_ADM_s101
                                 + 1.16892 w_k1_AXP_s101
                                 + 1.11178 w_k1_DIS_s101
                                 + 1.05466 w_k1_FDX_s101 + 1.7742 w_k1_IBM_s101
                                 + 0.951347 w_k1_INTC_s101
                                 + 1.43711 w_k1_KO_s101 + 2.35352 w_k1_MAT_s101
                                 + 1.07261 w_k1_MSFT_s101
                                 + 1.27656 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.02135 w_k1_ADM_s101
                                 + 0.961531 w_k1_AXP_s101
                                 + 1.40277 w_k1_DIS_s101 + 1.2508 w_k1_FDX_s101
                                 + 0.817432 w_k1_IBM_s101
                                 + 1.06923 w_k1_INTC_s101
                                 + 1.10475 w_k1_KO_s101
                                 + 0.736594 w_k1_MAT_s101
                                 + 1.32638 w_k1_MSFT_s101
                                 + 1.08261 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.14948 w_k1_ADM_s102
                                 + 1.25468 w_k1_AXP_s102
                                 + 1.16479 w_k1_DIS_s102
                                 + 1.05877 w_k1_FDX_s102
                                 + 1.62668 w_k1_IBM_s102
                                 + 0.882983 w_k1_INTC_s102
                                 + 1.12446 w_k1_KO_s102 + 1.52095 w_k1_MAT_s102
                                 + 1.03598 w_k1_MSFT_s102
                                 + 1.12034 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.975259 w_k1_ADM_s102
                                 + 1.13433 w_k1_AXP_s102
                                 + 1.26777 w_k1_DIS_s102
                                 + 1.10586 w_k1_FDX_s102
                                 + 1.35169 w_k1_IBM_s102
                                 + 1.09419 w_k1_INTC_s102
                                 + 1.11542 w_k1_KO_s102
                                 + 0.918976 w_k1_MAT_s102
                                 + 1.0871 w_k1_MSFT_s102 + 1.01159 w_k1_PG_s102
                                 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.697405 w_k1_ADM_s102
                                 + 0.78504 w_k1_AXP_s102
                                 + 0.67668 w_k1_DIS_s102
                                 + 0.740439 w_k1_FDX_s102
                                 + 1.14953 w_k1_IBM_s102
                                 + 0.764757 w_k1_INTC_s102
                                 + 0.8182 w_k1_KO_s102 + 0.893345 w_k1_MAT_s102
                                 + 0.659884 w_k1_MSFT_s102
                                 + 0.881193 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.15708 w_k1_ADM_s102
                                 + 1.76118 w_k1_AXP_s102
                                 + 1.27174 w_k1_DIS_s102
                                 + 1.22209 w_k1_FDX_s102
                                 + 1.17737 w_k1_IBM_s102
                                 + 0.949524 w_k1_INTC_s102
                                 + 0.929839 w_k1_KO_s102
                                 + 1.63756 w_k1_MAT_s102
                                 + 0.964704 w_k1_MSFT_s102
                                 + 1.07461 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.34778 w_k1_ADM_s102
                                 + 1.42854 w_k1_AXP_s102 + 1.4177 w_k1_DIS_s102
                                 + 1.25077 w_k1_FDX_s102
                                 + 1.03277 w_k1_IBM_s102
                                 + 1.1262 w_k1_INTC_s102
                                 + 0.989812 w_k1_KO_s102
                                 + 1.13871 w_k1_MAT_s102
                                 + 1.12734 w_k1_MSFT_s102 + 1.0086 w_k1_PG_s102
                                 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.16881 w_k1_ADM_s102
                                 + 0.992349 w_k1_AXP_s102
                                 + 1.41185 w_k1_DIS_s102
                                 + 1.17887 w_k1_FDX_s102
                                 + 0.942175 w_k1_IBM_s102
                                 + 1.13952 w_k1_INTC_s102
                                 + 1.10434 w_k1_KO_s102
                                 + 0.974226 w_k1_MAT_s102
                                 + 1.31665 w_k1_MSFT_s102 + 1.0624 w_k1_PG_s102
                                 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.965932 w_k1_ADM_s102
                                 + 0.86517 w_k1_AXP_s102
                                 + 0.822569 w_k1_DIS_s102
                                 + 0.903969 w_k1_FDX_s102
                                 + 0.716614 w_k1_IBM_s102
                                 + 0.97805 w_k1_INTC_s102
                                 + 0.753555 w_k1_KO_s102
                                 + 0.449136 w_k1_MAT_s102
                                 + 0.861168 w_k1_MSFT_s102
                                 + 0.929285 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.840371 w_k1_ADM_s102
                                 + 0.952463 w_k1_AXP_s102
                                 + 0.916376 w_k1_DIS_s102
                                 + 0.940168 w_k1_FDX_s102
                                 + 1.04161 w_k1_IBM_s102
                                 + 0.937837 w_k1_INTC_s102
                                 + 0.957124 w_k1_KO_s102
                                 + 0.890855 w_k1_MAT_s102
                                 + 0.981988 w_k1_MSFT_s102
                                 + 0.995509 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.971955 w_k1_ADM_s103
                                 + 0.959106 w_k1_AXP_s103
                                 + 1.02382 w_k1_DIS_s103
                                 + 0.937606 w_k1_FDX_s103
                                 + 1.07893 w_k1_IBM_s103
                                 + 1.06686 w_k1_INTC_s103
                                 + 1.07462 w_k1_KO_s103 + 1.09987 w_k1_MAT_s103
                                 + 1.08408 w_k1_MSFT_s103
                                 + 0.996706 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.31686 w_k1_ADM_s103
                                 + 1.34072 w_k1_AXP_s103
                                 + 1.65974 w_k1_DIS_s103
                                 + 1.26286 w_k1_FDX_s103
                                 + 1.24689 w_k1_IBM_s103
                                 + 1.28511 w_k1_INTC_s103
                                 + 1.58882 w_k1_KO_s103 + 1.43708 w_k1_MAT_s103
                                 + 1.1652 w_k1_MSFT_s103 + 1.16913 w_k1_PG_s103
                                 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.820602 w_k1_ADM_s103
                                 + 0.657046 w_k1_AXP_s103
                                 + 0.554366 w_k1_DIS_s103
                                 + 0.655431 w_k1_FDX_s103
                                 + 0.956014 w_k1_IBM_s103
                                 + 0.859149 w_k1_INTC_s103
                                 + 0.88498 w_k1_KO_s103
                                 + 0.757749 w_k1_MAT_s103
                                 + 0.796879 w_k1_MSFT_s103
                                 + 0.935134 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.01576 w_k1_ADM_s103
                                 + 1.30226 w_k1_AXP_s103
                                 + 0.969725 w_k1_DIS_s103
                                 + 1.10203 w_k1_FDX_s103
                                 + 1.00283 w_k1_IBM_s103
                                 + 1.12121 w_k1_INTC_s103 + 1.0056 w_k1_KO_s103
                                 + 1.24273 w_k1_MAT_s103
                                 + 0.991505 w_k1_MSFT_s103 + 1.056 w_k1_PG_s103
                                 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.2251 w_k1_ADM_s103
                                 + 1.09163 w_k1_AXP_s103
                                 + 1.08125 w_k1_DIS_s103
                                 + 1.15732 w_k1_FDX_s103
                                 + 0.726935 w_k1_IBM_s103
                                 + 1.2128 w_k1_INTC_s103 + 1.01347 w_k1_KO_s103
                                 + 0.719742 w_k1_MAT_s103
                                 + 1.14404 w_k1_MSFT_s103
                                 + 0.994959 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.850931 w_k1_ADM_s103
                                 + 0.972404 w_k1_AXP_s103
                                 + 0.90701 w_k1_DIS_s103
                                 + 0.850281 w_k1_FDX_s103
                                 + 1.62539 w_k1_IBM_s103
                                 + 0.886327 w_k1_INTC_s103
                                 + 1.21453 w_k1_KO_s103 + 1.65565 w_k1_MAT_s103
                                 + 0.918008 w_k1_MSFT_s103
                                 + 1.01934 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.744007 w_k1_ADM_s103
                                 + 0.797005 w_k1_AXP_s103
                                 + 0.729825 w_k1_DIS_s103
                                 + 0.779454 w_k1_FDX_s103
                                 + 0.771469 w_k1_IBM_s103
                                 + 0.833158 w_k1_INTC_s103
                                 + 0.745952 w_k1_KO_s103
                                 + 0.867089 w_k1_MAT_s103
                                 + 0.843402 w_k1_MSFT_s103
                                 + 0.814235 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.856802 w_k1_ADM_s103
                                 + 0.675505 w_k1_AXP_s103
                                 + 0.799988 w_k1_DIS_s103
                                 + 0.784813 w_k1_FDX_s103
                                 + 1.05907 w_k1_IBM_s103
                                 + 0.996252 w_k1_INTC_s103
                                 + 1.01096 w_k1_KO_s103
                                 + 0.810071 w_k1_MAT_s103
                                 + 0.901767 w_k1_MSFT_s103
                                 + 0.87355 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.932884 w_k1_ADM_s104
                                 + 0.954236 w_k1_AXP_s104
                                 + 0.712722 w_k1_DIS_s104
                                 + 0.870554 w_k1_FDX_s104
                                 + 0.912971 w_k1_IBM_s104
                                 + 0.928574 w_k1_INTC_s104
                                 + 0.80394 w_k1_KO_s104
                                 + 0.904957 w_k1_MAT_s104
                                 + 0.711317 w_k1_MSFT_s104
                                 + 0.824943 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.998693 w_k1_ADM_s104
                                 + 0.953683 w_k1_AXP_s104
                                 + 1.34691 w_k1_DIS_s104
                                 + 0.991132 w_k1_FDX_s104
                                 + 1.37976 w_k1_IBM_s104
                                 + 1.01817 w_k1_INTC_s104 + 1.1841 w_k1_KO_s104
                                 + 1.10638 w_k1_MAT_s104
                                 + 0.869775 w_k1_MSFT_s104
                                 + 0.952325 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.57586 w_k1_ADM_s104
                                 + 1.57889 w_k1_AXP_s104
                                 + 1.80007 w_k1_DIS_s104
                                 + 1.38188 w_k1_FDX_s104
                                 + 1.52323 w_k1_IBM_s104
                                 + 1.52529 w_k1_INTC_s104
                                 + 1.68066 w_k1_KO_s104 + 1.87632 w_k1_MAT_s104
                                 + 1.35335 w_k1_MSFT_s104
                                 + 1.37228 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.25552 w_k1_ADM_s104
                                 + 1.17134 w_k1_AXP_s104
                                 + 1.06445 w_k1_DIS_s104
                                 + 1.09922 w_k1_FDX_s104
                                 + 1.16376 w_k1_IBM_s104
                                 + 1.10106 w_k1_INTC_s104
                                 + 1.00441 w_k1_KO_s104 + 1.50152 w_k1_MAT_s104
                                 + 0.98105 w_k1_MSFT_s104 + 1.0208 w_k1_PG_s104
                                 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.872273 w_k1_ADM_s104
                                 + 0.626279 w_k1_AXP_s104
                                 + 0.746702 w_k1_DIS_s104
                                 + 0.790185 w_k1_FDX_s104
                                 + 0.805514 w_k1_IBM_s104
                                 + 0.769802 w_k1_INTC_s104
                                 + 0.708438 w_k1_KO_s104
                                 + 0.679397 w_k1_MAT_s104
                                 + 0.715321 w_k1_MSFT_s104
                                 + 0.81901 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.12031 w_k1_ADM_s104
                                 + 0.79443 w_k1_AXP_s104
                                 + 0.875422 w_k1_DIS_s104
                                 + 0.999701 w_k1_FDX_s104
                                 + 0.756086 w_k1_IBM_s104
                                 + 0.874365 w_k1_INTC_s104
                                 + 0.844375 w_k1_KO_s104
                                 + 0.77881 w_k1_MAT_s104
                                 + 0.797741 w_k1_MSFT_s104
                                 + 0.862081 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.91558 w_k1_ADM_s104
                                 + 0.863638 w_k1_AXP_s104
                                 + 0.90549 w_k1_DIS_s104
                                 + 0.895175 w_k1_FDX_s104
                                 + 1.28785 w_k1_IBM_s104
                                 + 1.07561 w_k1_INTC_s104
                                 + 1.13807 w_k1_KO_s104 + 1.15612 w_k1_MAT_s104
                                 + 0.898945 w_k1_MSFT_s104
                                 + 0.860195 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.44158 w_k1_ADM_s104
                                 + 1.13106 w_k1_AXP_s104
                                 + 1.37162 w_k1_DIS_s104
                                 + 1.33925 w_k1_FDX_s104
                                 + 0.934772 w_k1_IBM_s104
                                 + 1.06212 w_k1_INTC_s104
                                 + 1.05903 w_k1_KO_s104
                                 + 0.731289 w_k1_MAT_s104
                                 + 0.995999 w_k1_MSFT_s104
                                 + 1.03585 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.938285 w_k1_ADM_s105
                                 + 1.00822 w_k1_AXP_s105
                                 + 1.11365 w_k1_DIS_s105
                                 + 1.02836 w_k1_FDX_s105
                                 + 1.24762 w_k1_IBM_s105
                                 + 0.972704 w_k1_INTC_s105
                                 + 1.15942 w_k1_KO_s105 + 1.13391 w_k1_MAT_s105
                                 + 0.96931 w_k1_MSFT_s105
                                 + 0.918229 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.556079 w_k1_ADM_s105
                                 + 0.547951 w_k1_AXP_s105
                                 + 0.396036 w_k1_DIS_s105
                                 + 0.643387 w_k1_FDX_s105
                                 + 0.750302 w_k1_IBM_s105
                                 + 0.710154 w_k1_INTC_s105
                                 + 0.647306 w_k1_KO_s105
                                 + 0.737259 w_k1_MAT_s105
                                 + 0.662787 w_k1_MSFT_s105
                                 + 0.679305 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.815674 w_k1_ADM_s105
                                 + 0.675297 w_k1_AXP_s105
                                 + 0.821034 w_k1_DIS_s105
                                 + 0.906321 w_k1_FDX_s105
                                 + 0.806216 w_k1_IBM_s105
                                 + 0.818609 w_k1_INTC_s105
                                 + 0.913377 w_k1_KO_s105
                                 + 0.823203 w_k1_MAT_s105
                                 + 0.86968 w_k1_MSFT_s105
                                 + 0.930571 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.920922 w_k1_ADM_s105
                                 + 0.767402 w_k1_AXP_s105
                                 + 0.76743 w_k1_DIS_s105
                                 + 0.805891 w_k1_FDX_s105
                                 + 1.08107 w_k1_IBM_s105
                                 + 0.72978 w_k1_INTC_s105
                                 + 1.07668 w_k1_KO_s105 + 1.23102 w_k1_MAT_s105
                                 + 0.912845 w_k1_MSFT_s105
                                 + 0.859525 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.14937 w_k1_ADM_s105
                                 + 1.21695 w_k1_AXP_s105
                                 + 1.19004 w_k1_DIS_s105
                                 + 1.35241 w_k1_FDX_s105
                                 + 0.99947 w_k1_IBM_s105
                                 + 1.0776 w_k1_INTC_s105 + 1.14183 w_k1_KO_s105
                                 + 1.32403 w_k1_MAT_s105
                                 + 1.17641 w_k1_MSFT_s105
                                 + 1.02551 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.25308 w_k1_ADM_s105
                                 + 1.38764 w_k1_AXP_s105
                                 + 2.01445 w_k1_DIS_s105
                                 + 1.59619 w_k1_FDX_s105
                                 + 1.42628 w_k1_IBM_s105
                                 + 1.22559 w_k1_INTC_s105
                                 + 1.75771 w_k1_KO_s105 + 1.44601 w_k1_MAT_s105
                                 + 1.47355 w_k1_MSFT_s105
                                 + 1.23776 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.893779 w_k1_ADM_s105
                                 + 0.656334 w_k1_AXP_s105
                                 + 0.840647 w_k1_DIS_s105
                                 + 0.927935 w_k1_FDX_s105
                                 + 0.656218 w_k1_IBM_s105
                                 + 0.917019 w_k1_INTC_s105
                                 + 0.783603 w_k1_KO_s105
                                 + 0.429048 w_k1_MAT_s105
                                 + 0.962204 w_k1_MSFT_s105
                                 + 0.821864 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.0039 w_k1_ADM_s105
                                 + 0.94345 w_k1_AXP_s105
                                 + 1.46138 w_k1_DIS_s105
                                 + 1.24461 w_k1_FDX_s105 + 1.1418 w_k1_IBM_s105
                                 + 1.08338 w_k1_INTC_s105
                                 + 1.39527 w_k1_KO_s105 + 1.34747 w_k1_MAT_s105
                                 + 1.26602 w_k1_MSFT_s105
                                 + 1.15332 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.616324 w_k1_ADM_s106
                                 + 0.700852 w_k1_AXP_s106
                                 + 0.630985 w_k1_DIS_s106
                                 + 0.741843 w_k1_FDX_s106
                                 + 0.746144 w_k1_IBM_s106
                                 + 0.887023 w_k1_INTC_s106
                                 + 0.684443 w_k1_KO_s106
                                 + 0.407569 w_k1_MAT_s106
                                 + 0.854589 w_k1_MSFT_s106
                                 + 0.784635 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.12404 w_k1_ADM_s106
                                 + 1.04928 w_k1_AXP_s106
                                 + 1.13578 w_k1_DIS_s106
                                 + 1.17444 w_k1_FDX_s106
                                 + 0.644132 w_k1_IBM_s106
                                 + 1.28532 w_k1_INTC_s106
                                 + 0.799255 w_k1_KO_s106
                                 + 0.422064 w_k1_MAT_s106
                                 + 1.2516 w_k1_MSFT_s106
                                 + 0.856474 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.986209 w_k1_ADM_s106
                                 + 1.16808 w_k1_AXP_s106
                                 + 1.14573 w_k1_DIS_s106
                                 + 0.994344 w_k1_FDX_s106
                                 + 1.77145 w_k1_IBM_s106
                                 + 1.04299 w_k1_INTC_s106
                                 + 1.19713 w_k1_KO_s106 + 1.60653 w_k1_MAT_s106
                                 + 1.1523 w_k1_MSFT_s106
                                 + 0.963254 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.904282 w_k1_ADM_s106
                                 + 0.97381 w_k1_AXP_s106
                                 + 0.796916 w_k1_DIS_s106
                                 + 0.905321 w_k1_FDX_s106
                                 + 0.936237 w_k1_IBM_s106
                                 + 0.789792 w_k1_INTC_s106
                                 + 0.796381 w_k1_KO_s106
                                 + 0.762474 w_k1_MAT_s106
                                 + 0.857247 w_k1_MSFT_s106
                                 + 0.860962 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.24932 w_k1_ADM_s106
                                 + 1.23156 w_k1_AXP_s106
                                 + 1.04392 w_k1_DIS_s106
                                 + 1.14358 w_k1_FDX_s106
                                 + 1.26254 w_k1_IBM_s106
                                 + 1.09044 w_k1_INTC_s106
                                 + 1.10746 w_k1_KO_s106
                                 + 0.879159 w_k1_MAT_s106
                                 + 1.20978 w_k1_MSFT_s106 + 1.1355 w_k1_PG_s106
                                 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.21485 w_k1_ADM_s106
                                 + 1.21563 w_k1_AXP_s106
                                 + 1.18978 w_k1_DIS_s106
                                 + 1.11683 w_k1_FDX_s106
                                 + 1.07751 w_k1_IBM_s106
                                 + 1.07859 w_k1_INTC_s106
                                 + 1.00667 w_k1_KO_s106 + 1.06047 w_k1_MAT_s106
                                 + 1.25183 w_k1_MSFT_s106
                                 + 0.972865 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.936484 w_k1_ADM_s106
                                 + 0.732089 w_k1_AXP_s106
                                 + 1.03354 w_k1_DIS_s106
                                 + 0.84552 w_k1_FDX_s106
                                 + 1.21857 w_k1_IBM_s106
                                 + 1.08278 w_k1_INTC_s106 + 1.1974 w_k1_KO_s106
                                 + 0.741929 w_k1_MAT_s106
                                 + 1.14359 w_k1_MSFT_s106
                                 + 1.05487 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.31018 w_k1_ADM_s106
                                 + 1.64152 w_k1_AXP_s106
                                 + 1.69425 w_k1_DIS_s106
                                 + 1.43598 w_k1_FDX_s106
                                 + 1.47058 w_k1_IBM_s106
                                 + 1.16362 w_k1_INTC_s106
                                 + 1.29968 w_k1_KO_s106 + 1.48614 w_k1_MAT_s106
                                 + 1.34724 w_k1_MSFT_s106
                                 + 1.14601 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.23599 w_k1_ADM_s107
                                 + 0.891345 w_k1_AXP_s107
                                 + 1.15682 w_k1_DIS_s107
                                 + 1.02148 w_k1_FDX_s107
                                 + 1.07485 w_k1_IBM_s107
                                 + 0.935753 w_k1_INTC_s107
                                 + 1.1385 w_k1_KO_s107 + 0.942128 w_k1_MAT_s107
                                 + 1.05152 w_k1_MSFT_s107
                                 + 1.04435 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.87147 w_k1_ADM_s107
                                 + 0.784975 w_k1_AXP_s107
                                 + 0.648243 w_k1_DIS_s107
                                 + 0.669484 w_k1_FDX_s107
                                 + 1.00759 w_k1_IBM_s107
                                 + 0.880123 w_k1_INTC_s107
                                 + 1.04421 w_k1_KO_s107 + 1.01046 w_k1_MAT_s107
                                 + 0.802522 w_k1_MSFT_s107
                                 + 0.908494 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.44272 w_k1_ADM_s107
                                 + 1.4962 w_k1_AXP_s107 + 1.41805 w_k1_DIS_s107
                                 + 1.21557 w_k1_FDX_s107 + 1.3396 w_k1_IBM_s107
                                 + 0.931303 w_k1_INTC_s107
                                 + 1.39231 w_k1_KO_s107 + 2.08915 w_k1_MAT_s107
                                 + 1.11707 w_k1_MSFT_s107
                                 + 1.21509 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.961989 w_k1_ADM_s107
                                 + 0.816211 w_k1_AXP_s107
                                 + 0.806637 w_k1_DIS_s107
                                 + 0.816787 w_k1_FDX_s107
                                 + 0.703954 w_k1_IBM_s107
                                 + 0.99402 w_k1_INTC_s107
                                 + 0.859183 w_k1_KO_s107
                                 + 0.614098 w_k1_MAT_s107
                                 + 0.867434 w_k1_MSFT_s107
                                 + 0.889777 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.43411 w_k1_ADM_s107
                                 + 1.41858 w_k1_AXP_s107
                                 + 2.03353 w_k1_DIS_s107
                                 + 1.44707 w_k1_FDX_s107
                                 + 1.49553 w_k1_IBM_s107
                                 + 1.26815 w_k1_INTC_s107
                                 + 1.61973 w_k1_KO_s107 + 1.43713 w_k1_MAT_s107
                                 + 1.33559 w_k1_MSFT_s107
                                 + 1.30044 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.32059 w_k1_ADM_s107
                                 + 1.00393 w_k1_AXP_s107
                                 + 1.32543 w_k1_DIS_s107
                                 + 1.23662 w_k1_FDX_s107
                                 + 0.980361 w_k1_IBM_s107
                                 + 1.10842 w_k1_INTC_s107
                                 + 1.35013 w_k1_KO_s107
                                 + 0.960711 w_k1_MAT_s107
                                 + 1.18019 w_k1_MSFT_s107
                                 + 1.18555 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.927381 w_k1_ADM_s107
                                 + 1.06003 w_k1_AXP_s107
                                 + 0.774502 w_k1_DIS_s107
                                 + 0.906056 w_k1_FDX_s107
                                 + 1.00731 w_k1_IBM_s107
                                 + 0.798316 w_k1_INTC_s107
                                 + 0.980433 w_k1_KO_s107
                                 + 0.968399 w_k1_MAT_s107
                                 + 0.798405 w_k1_MSFT_s107
                                 + 0.934336 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.16059 w_k1_ADM_s107
                                 + 1.03928 w_k1_AXP_s107
                                 + 1.04778 w_k1_DIS_s107
                                 + 0.883644 w_k1_FDX_s107
                                 + 1.53285 w_k1_IBM_s107
                                 + 1.13088 w_k1_INTC_s107
                                 + 1.47413 w_k1_KO_s107 + 1.68147 w_k1_MAT_s107
                                 + 0.991811 w_k1_MSFT_s107
                                 + 1.20983 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.09378 w_k1_ADM_s110
                                 + 1.24937 w_k1_AXP_s110
                                 + 1.19182 w_k1_DIS_s110
                                 + 1.07309 w_k1_FDX_s110
                                 + 1.19533 w_k1_IBM_s110
                                 + 1.06887 w_k1_INTC_s110
                                 + 1.24706 w_k1_KO_s110 + 1.24672 w_k1_MAT_s110
                                 + 1.01974 w_k1_MSFT_s110
                                 + 1.05865 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.18329 w_k1_ADM_s110
                                 + 1.13056 w_k1_AXP_s110
                                 + 1.16891 w_k1_DIS_s110
                                 + 1.08151 w_k1_FDX_s110
                                 + 1.01126 w_k1_IBM_s110
                                 + 1.08002 w_k1_INTC_s110
                                 + 1.10725 w_k1_KO_s110
                                 + 0.684484 w_k1_MAT_s110
                                 + 1.2341 w_k1_MSFT_s110 + 1.04991 w_k1_PG_s110
                                 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.59358 w_k1_ADM_s110
                                 + 1.5527 w_k1_AXP_s110 + 1.58328 w_k1_DIS_s110
                                 + 1.4997 w_k1_FDX_s110 + 1.05045 w_k1_IBM_s110
                                 + 1.21346 w_k1_INTC_s110
                                 + 1.45078 w_k1_KO_s110 + 1.5799 w_k1_MAT_s110
                                 + 1.43953 w_k1_MSFT_s110
                                 + 1.40908 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.916962 w_k1_ADM_s110
                                 + 0.80705 w_k1_AXP_s110
                                 + 0.620085 w_k1_DIS_s110
                                 + 0.799223 w_k1_FDX_s110
                                 + 0.661203 w_k1_IBM_s110
                                 + 0.838986 w_k1_INTC_s110
                                 + 0.710394 w_k1_KO_s110
                                 + 0.876687 w_k1_MAT_s110
                                 + 0.794835 w_k1_MSFT_s110
                                 + 0.88178 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.1555 w_k1_ADM_s110
                                 + 1.33077 w_k1_AXP_s110
                                 + 1.15808 w_k1_DIS_s110
                                 + 1.01626 w_k1_FDX_s110
                                 + 1.62401 w_k1_IBM_s110
                                 + 1.01104 w_k1_INTC_s110
                                 + 1.36316 w_k1_KO_s110 + 1.71775 w_k1_MAT_s110
                                 + 1.08354 w_k1_MSFT_s110
                                 + 1.09823 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.768149 w_k1_ADM_s110
                                 + 0.732481 w_k1_AXP_s110
                                 + 0.703804 w_k1_DIS_s110
                                 + 0.760399 w_k1_FDX_s110
                                 + 0.956616 w_k1_IBM_s110
                                 + 0.877956 w_k1_INTC_s110
                                 + 0.934411 w_k1_KO_s110
                                 + 0.815936 w_k1_MAT_s110
                                 + 0.78152 w_k1_MSFT_s110
                                 + 0.872356 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.986205 w_k1_ADM_s110
                                 + 0.955198 w_k1_AXP_s110
                                 + 0.823937 w_k1_DIS_s110
                                 + 0.966253 w_k1_FDX_s110
                                 + 0.949198 w_k1_IBM_s110
                                 + 0.85106 w_k1_INTC_s110
                                 + 1.00304 w_k1_KO_s110
                                 + 0.856296 w_k1_MAT_s110
                                 + 0.88242 w_k1_MSFT_s110
                                 + 1.07987 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.10308 w_k1_ADM_s110
                                 + 1.29455 w_k1_AXP_s110
                                 + 1.13391 w_k1_DIS_s110 + 1.1349 w_k1_FDX_s110
                                 + 0.76052 w_k1_IBM_s110
                                 + 1.18328 w_k1_INTC_s110 + 0.9784 w_k1_KO_s110
                                 + 0.882649 w_k1_MAT_s110
                                 + 1.18063 w_k1_MSFT_s110 + 1.1436 w_k1_PG_s110
                                 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.745379 w_k1_ADM_s111
                                 + 0.641457 w_k1_AXP_s111
                                 + 0.595396 w_k1_DIS_s111
                                 + 0.685823 w_k1_FDX_s111
                                 + 0.783913 w_k1_IBM_s111
                                 + 0.627782 w_k1_INTC_s111
                                 + 0.674318 w_k1_KO_s111
                                 + 0.897057 w_k1_MAT_s111
                                 + 0.766637 w_k1_MSFT_s111
                                 + 0.756826 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.922973 w_k1_ADM_s111
                                 + 1.20939 w_k1_AXP_s111
                                 + 0.886349 w_k1_DIS_s111
                                 + 1.00598 w_k1_FDX_s111
                                 + 0.914964 w_k1_IBM_s111
                                 + 1.12719 w_k1_INTC_s111
                                 + 0.944759 w_k1_KO_s111
                                 + 0.858272 w_k1_MAT_s111
                                 + 0.994597 w_k1_MSFT_s111
                                 + 0.938584 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.918264 w_k1_ADM_s111
                                 + 1.04081 w_k1_AXP_s111
                                 + 1.16752 w_k1_DIS_s111
                                 + 1.14943 w_k1_FDX_s111
                                 + 0.770353 w_k1_IBM_s111
                                 + 1.23999 w_k1_INTC_s111
                                 + 1.06519 w_k1_KO_s111
                                 + 0.934902 w_k1_MAT_s111
                                 + 1.24396 w_k1_MSFT_s111
                                 + 0.997905 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.40423 w_k1_ADM_s111
                                 + 1.52489 w_k1_AXP_s111
                                 + 1.33078 w_k1_DIS_s111
                                 + 1.17793 w_k1_FDX_s111
                                 + 1.12621 w_k1_IBM_s111
                                 + 1.16833 w_k1_INTC_s111
                                 + 1.42137 w_k1_KO_s111 + 1.56914 w_k1_MAT_s111
                                 + 1.21568 w_k1_MSFT_s111
                                 + 1.02723 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.845418 w_k1_ADM_s111
                                 + 0.743337 w_k1_AXP_s111
                                 + 0.821435 w_k1_DIS_s111
                                 + 0.860915 w_k1_FDX_s111
                                 + 0.655923 w_k1_IBM_s111
                                 + 0.997683 w_k1_INTC_s111
                                 + 0.91517 w_k1_KO_s111
                                 + 0.803337 w_k1_MAT_s111
                                 + 0.910646 w_k1_MSFT_s111
                                 + 0.826477 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.1108 w_k1_ADM_s111
                                 + 1.05402 w_k1_AXP_s111
                                 + 1.50029 w_k1_DIS_s111
                                 + 1.28129 w_k1_FDX_s111 + 1.2096 w_k1_IBM_s111
                                 + 1.14351 w_k1_INTC_s111
                                 + 1.27295 w_k1_KO_s111 + 1.19537 w_k1_MAT_s111
                                 + 1.19335 w_k1_MSFT_s111
                                 + 1.05565 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.15827 w_k1_ADM_s111
                                 + 1.03919 w_k1_AXP_s111
                                 + 0.944149 w_k1_DIS_s111
                                 + 0.888518 w_k1_FDX_s111
                                 + 1.24406 w_k1_IBM_s111
                                 + 0.917522 w_k1_INTC_s111
                                 + 1.04224 w_k1_KO_s111 + 1.42969 w_k1_MAT_s111
                                 + 1.00828 w_k1_MSFT_s111
                                 + 0.974698 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.938517 w_k1_ADM_s111
                                 + 0.896373 w_k1_AXP_s111
                                 + 0.955831 w_k1_DIS_s111
                                 + 0.898417 w_k1_FDX_s111
                                 + 0.908553 w_k1_IBM_s111
                                 + 0.976962 w_k1_INTC_s111
                                 + 1.00503 w_k1_KO_s111 + 1.15761 w_k1_MAT_s111
                                 + 1.12618 w_k1_MSFT_s111 + 0.9336 w_k1_PG_s111
                                 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.19061 w_k1_ADM_s112
                                 + 0.986018 w_k1_AXP_s112
                                 + 1.15751 w_k1_DIS_s112
                                 + 0.944006 w_k1_FDX_s112
                                 + 1.40287 w_k1_IBM_s112
                                 + 0.830482 w_k1_INTC_s112
                                 + 1.57046 w_k1_KO_s112 + 1.63308 w_k1_MAT_s112
                                 + 0.877096 w_k1_MSFT_s112
                                 + 1.16023 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.28198 w_k1_ADM_s112
                                 + 1.43918 w_k1_AXP_s112
                                 + 1.71444 w_k1_DIS_s112
                                 + 1.34908 w_k1_FDX_s112
                                 + 1.16598 w_k1_IBM_s112
                                 + 1.1444 w_k1_INTC_s112 + 1.35443 w_k1_KO_s112
                                 + 1.08056 w_k1_MAT_s112
                                 + 1.12361 w_k1_MSFT_s112
                                 + 1.28025 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.07293 w_k1_ADM_s112
                                 + 1.16727 w_k1_AXP_s112
                                 + 1.28917 w_k1_DIS_s112
                                 + 1.18543 w_k1_FDX_s112
                                 + 1.27988 w_k1_IBM_s112
                                 + 1.13232 w_k1_INTC_s112
                                 + 1.17015 w_k1_KO_s112 + 1.23544 w_k1_MAT_s112
                                 + 0.984302 w_k1_MSFT_s112
                                 + 1.1384 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.936981 w_k1_ADM_s112
                                 + 1.55991 w_k1_AXP_s112
                                 + 1.08482 w_k1_DIS_s112
                                 + 1.04459 w_k1_FDX_s112
                                 + 1.18643 w_k1_IBM_s112
                                 + 0.726282 w_k1_INTC_s112
                                 + 0.975641 w_k1_KO_s112
                                 + 1.71783 w_k1_MAT_s112
                                 + 0.68348 w_k1_MSFT_s112
                                 + 0.934234 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.866621 w_k1_ADM_s112
                                 + 0.783168 w_k1_AXP_s112
                                 + 1.24668 w_k1_DIS_s112
                                 + 0.93952 w_k1_FDX_s112
                                 + 1.16738 w_k1_IBM_s112
                                 + 0.900573 w_k1_INTC_s112
                                 + 1.21049 w_k1_KO_s112
                                 + 0.819781 w_k1_MAT_s112
                                 + 1.11365 w_k1_MSFT_s112
                                 + 0.930464 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.721263 w_k1_ADM_s112
                                 + 0.753011 w_k1_AXP_s112
                                 + 0.926106 w_k1_DIS_s112
                                 + 0.921271 w_k1_FDX_s112
                                 + 0.714046 w_k1_IBM_s112
                                 + 0.709646 w_k1_INTC_s112
                                 + 0.835752 w_k1_KO_s112
                                 + 0.774277 w_k1_MAT_s112
                                 + 0.85027 w_k1_MSFT_s112
                                 + 0.801306 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.823834 w_k1_ADM_s112
                                 + 0.692442 w_k1_AXP_s112
                                 + 0.720152 w_k1_DIS_s112
                                 + 0.795251 w_k1_FDX_s112
                                 + 0.888892 w_k1_IBM_s112
                                 + 0.759628 w_k1_INTC_s112
                                 + 0.913547 w_k1_KO_s112
                                 + 0.765164 w_k1_MAT_s112
                                 + 0.710053 w_k1_MSFT_s112
                                 + 0.927525 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.944195 w_k1_ADM_s112
                                 + 1.15319 w_k1_AXP_s112
                                 + 1.14069 w_k1_DIS_s112
                                 + 1.10652 w_k1_FDX_s112
                                 + 0.913311 w_k1_IBM_s112
                                 + 0.960696 w_k1_INTC_s112
                                 + 1.01664 w_k1_KO_s112 + 1.2057 w_k1_MAT_s112
                                 + 1.03662 w_k1_MSFT_s112
                                 + 0.931016 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.908043 w_k1_ADM_s113
                                 + 1.11044 w_k1_AXP_s113
                                 + 1.16362 w_k1_DIS_s113
                                 + 0.979031 w_k1_FDX_s113
                                 + 1.20005 w_k1_IBM_s113
                                 + 1.07687 w_k1_INTC_s113
                                 + 1.14946 w_k1_KO_s113 + 1.63966 w_k1_MAT_s113
                                 + 1.15075 w_k1_MSFT_s113
                                 + 1.04838 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.614246 w_k1_ADM_s113
                                 + 0.750231 w_k1_AXP_s113
                                 + 0.691077 w_k1_DIS_s113
                                 + 0.671869 w_k1_FDX_s113
                                 + 0.894672 w_k1_IBM_s113
                                 + 0.840623 w_k1_INTC_s113
                                 + 0.739535 w_k1_KO_s113
                                 + 0.705177 w_k1_MAT_s113
                                 + 0.77235 w_k1_MSFT_s113
                                 + 0.792027 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.01458 w_k1_ADM_s113
                                 + 1.01932 w_k1_AXP_s113
                                 + 1.04423 w_k1_DIS_s113
                                 + 0.990755 w_k1_FDX_s113
                                 + 0.933746 w_k1_IBM_s113
                                 + 1.12129 w_k1_INTC_s113
                                 + 1.08882 w_k1_KO_s113 + 1.03831 w_k1_MAT_s113
                                 + 1.05371 w_k1_MSFT_s113
                                 + 1.09169 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.08763 w_k1_ADM_s113
                                 + 1.35281 w_k1_AXP_s113
                                 + 1.63595 w_k1_DIS_s113
                                 + 1.30301 w_k1_FDX_s113
                                 + 1.25438 w_k1_IBM_s113
                                 + 1.27166 w_k1_INTC_s113 + 1.2847 w_k1_KO_s113
                                 + 1.09447 w_k1_MAT_s113
                                 + 1.34445 w_k1_MSFT_s113
                                 + 1.05652 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.878231 w_k1_ADM_s113
                                 + 0.667964 w_k1_AXP_s113
                                 + 0.720584 w_k1_DIS_s113
                                 + 0.750801 w_k1_FDX_s113
                                 + 1.13935 w_k1_IBM_s113
                                 + 0.739173 w_k1_INTC_s113
                                 + 0.849126 w_k1_KO_s113
                                 + 1.10559 w_k1_MAT_s113
                                 + 0.851736 w_k1_MSFT_s113
                                 + 0.871968 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.781731 w_k1_ADM_s113
                                 + 0.862032 w_k1_AXP_s113
                                 + 0.69569 w_k1_DIS_s113
                                 + 0.778198 w_k1_FDX_s113
                                 + 0.770094 w_k1_IBM_s113
                                 + 0.715097 w_k1_INTC_s113
                                 + 0.713315 w_k1_KO_s113
                                 + 0.755806 w_k1_MAT_s113
                                 + 0.723927 w_k1_MSFT_s113
                                 + 0.817319 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.05336 w_k1_ADM_s113
                                 + 1.09574 w_k1_AXP_s113
                                 + 0.896005 w_k1_DIS_s113
                                 + 0.849521 w_k1_FDX_s113
                                 + 1.70447 w_k1_IBM_s113
                                 + 0.857786 w_k1_INTC_s113
                                 + 1.1234 w_k1_KO_s113 + 1.6335 w_k1_MAT_s113
                                 + 0.860384 w_k1_MSFT_s113
                                 + 1.18309 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.1059 w_k1_ADM_s113
                                 + 0.948069 w_k1_AXP_s113
                                 + 0.924903 w_k1_DIS_s113
                                 + 1.02375 w_k1_FDX_s113
                                 + 0.60284 w_k1_IBM_s113
                                 + 0.940886 w_k1_INTC_s113
                                 + 0.91572 w_k1_KO_s113 + 1.03074 w_k1_MAT_s113
                                 + 1.07934 w_k1_MSFT_s113
                                 + 0.949861 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.929012 w_k1_ADM_s114
                                 + 1.45668 w_k1_AXP_s114
                                 + 1.69214 w_k1_DIS_s114
                                 + 1.32937 w_k1_FDX_s114
                                 + 1.48393 w_k1_IBM_s114
                                 + 1.5125 w_k1_INTC_s114 + 1.31921 w_k1_KO_s114
                                 + 1.29801 w_k1_MAT_s114
                                 + 1.30771 w_k1_MSFT_s114
                                 + 1.24824 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.912212 w_k1_ADM_s114
                                 + 0.901678 w_k1_AXP_s114
                                 + 0.927934 w_k1_DIS_s114
                                 + 0.932339 w_k1_FDX_s114
                                 + 0.681337 w_k1_IBM_s114
                                 + 0.725852 w_k1_INTC_s114
                                 + 0.787549 w_k1_KO_s114
                                 + 0.443631 w_k1_MAT_s114
                                 + 0.810657 w_k1_MSFT_s114
                                 + 0.783077 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.03628 w_k1_ADM_s114
                                 + 1.46886 w_k1_AXP_s114
                                 + 1.23581 w_k1_DIS_s114
                                 + 1.17064 w_k1_FDX_s114
                                 + 1.17269 w_k1_IBM_s114
                                 + 0.949684 w_k1_INTC_s114
                                 + 0.882096 w_k1_KO_s114
                                 + 1.04589 w_k1_MAT_s114
                                 + 0.90048 w_k1_MSFT_s114
                                 + 1.11677 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.703691 w_k1_ADM_s114
                                 + 0.904444 w_k1_AXP_s114
                                 + 0.890504 w_k1_DIS_s114
                                 + 0.916223 w_k1_FDX_s114
                                 + 1.10573 w_k1_IBM_s114
                                 + 0.851652 w_k1_INTC_s114
                                 + 0.947943 w_k1_KO_s114
                                 + 0.924165 w_k1_MAT_s114
                                 + 0.80516 w_k1_MSFT_s114
                                 + 0.953878 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.907341 w_k1_ADM_s114
                                 + 1.18943 w_k1_AXP_s114
                                 + 1.00169 w_k1_DIS_s114
                                 + 1.14796 w_k1_FDX_s114
                                 + 0.770489 w_k1_IBM_s114
                                 + 0.851975 w_k1_INTC_s114
                                 + 0.860429 w_k1_KO_s114
                                 + 0.823618 w_k1_MAT_s114
                                 + 0.911317 w_k1_MSFT_s114
                                 + 1.07331 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.660351 w_k1_ADM_s114
                                 + 0.76327 w_k1_AXP_s114
                                 + 0.620169 w_k1_DIS_s114
                                 + 0.626568 w_k1_FDX_s114
                                 + 0.880364 w_k1_IBM_s114
                                 + 0.865968 w_k1_INTC_s114
                                 + 0.694581 w_k1_KO_s114
                                 + 0.82362 w_k1_MAT_s114
                                 + 0.647754 w_k1_MSFT_s114
                                 + 0.796968 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.896269 w_k1_ADM_s114
                                 + 0.940267 w_k1_AXP_s114
                                 + 1.40024 w_k1_DIS_s114
                                 + 1.05561 w_k1_FDX_s114
                                 + 0.889836 w_k1_IBM_s114
                                 + 1.18211 w_k1_INTC_s114
                                 + 0.969154 w_k1_KO_s114
                                 + 0.726592 w_k1_MAT_s114
                                 + 1.16082 w_k1_MSFT_s114
                                 + 0.987403 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.845262 w_k1_ADM_s114
                                 + 0.661811 w_k1_AXP_s114
                                 + 0.617685 w_k1_DIS_s114
                                 + 0.704665 w_k1_FDX_s114
                                 + 0.636568 w_k1_IBM_s114
                                 + 0.702567 w_k1_INTC_s114
                                 + 0.667062 w_k1_KO_s114
                                 + 0.446499 w_k1_MAT_s114
                                 + 0.676214 w_k1_MSFT_s114
                                 + 0.787575 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.20618 w_k1_ADM_s115
                                 + 1.23242 w_k1_AXP_s115
                                 + 1.58119 w_k1_DIS_s115
                                 + 1.23254 w_k1_FDX_s115
                                 + 1.58921 w_k1_IBM_s115
                                 + 1.06596 w_k1_INTC_s115
                                 + 1.42768 w_k1_KO_s115 + 1.19145 w_k1_MAT_s115
                                 + 1.44431 w_k1_MSFT_s115
                                 + 1.11733 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.808311 w_k1_ADM_s115
                                 + 0.903255 w_k1_AXP_s115
                                 + 0.850795 w_k1_DIS_s115
                                 + 0.960228 w_k1_FDX_s115
                                 + 0.860753 w_k1_IBM_s115
                                 + 0.92673 w_k1_INTC_s115
                                 + 0.89575 w_k1_KO_s115
                                 + 0.749575 w_k1_MAT_s115
                                 + 0.970213 w_k1_MSFT_s115
                                 + 0.823496 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.882405 w_k1_ADM_s115
                                 + 1.4161 w_k1_AXP_s115 + 1.13474 w_k1_DIS_s115
                                 + 1.17532 w_k1_FDX_s115
                                 + 1.34644 w_k1_IBM_s115
                                 + 1.08317 w_k1_INTC_s115
                                 + 1.25851 w_k1_KO_s115 + 1.25493 w_k1_MAT_s115
                                 + 1.21037 w_k1_MSFT_s115
                                 + 1.05782 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.23432 w_k1_ADM_s115
                                 + 0.963532 w_k1_AXP_s115
                                 + 0.947198 w_k1_DIS_s115
                                 + 1.04743 w_k1_FDX_s115
                                 + 1.02026 w_k1_IBM_s115
                                 + 0.903073 w_k1_INTC_s115
                                 + 1.03145 w_k1_KO_s115 + 1.038 w_k1_MAT_s115
                                 + 1.09349 w_k1_MSFT_s115
                                 + 1.00607 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.09753 w_k1_ADM_s115
                                 + 1.07921 w_k1_AXP_s115
                                 + 1.15573 w_k1_DIS_s115
                                 + 1.19455 w_k1_FDX_s115
                                 + 0.691536 w_k1_IBM_s115
                                 + 1.25424 w_k1_INTC_s115
                                 + 1.00022 w_k1_KO_s115
                                 + 0.606709 w_k1_MAT_s115
                                 + 1.19374 w_k1_MSFT_s115
                                 + 0.897054 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.76742 w_k1_ADM_s115
                                 + 0.741017 w_k1_AXP_s115
                                 + 0.737571 w_k1_DIS_s115
                                 + 0.870377 w_k1_FDX_s115
                                 + 0.617529 w_k1_IBM_s115
                                 + 0.937285 w_k1_INTC_s115
                                 + 0.784737 w_k1_KO_s115
                                 + 0.701696 w_k1_MAT_s115
                                 + 0.982566 w_k1_MSFT_s115
                                 + 0.770634 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.27056 w_k1_ADM_s115
                                 + 1.3094 w_k1_AXP_s115 + 1.45896 w_k1_DIS_s115
                                 + 1.33324 w_k1_FDX_s115
                                 + 1.01613 w_k1_IBM_s115
                                 + 1.24047 w_k1_INTC_s115
                                 + 1.33528 w_k1_KO_s115 + 1.00494 w_k1_MAT_s115
                                 + 1.39094 w_k1_MSFT_s115
                                 + 1.12323 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.743069 w_k1_ADM_s115
                                 + 0.847743 w_k1_AXP_s115
                                 + 0.825147 w_k1_DIS_s115
                                 + 0.79794 w_k1_FDX_s115
                                 + 1.37784 w_k1_IBM_s115
                                 + 0.925989 w_k1_INTC_s115
                                 + 1.11719 w_k1_KO_s115 + 1.29903 w_k1_MAT_s115
                                 + 1.01241 w_k1_MSFT_s115
                                 + 0.882087 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.885857 w_k1_ADM_s116
                                 + 0.855289 w_k1_AXP_s116
                                 + 0.957357 w_k1_DIS_s116
                                 + 1.00332 w_k1_FDX_s116
                                 + 0.70091 w_k1_IBM_s116
                                 + 0.948555 w_k1_INTC_s116
                                 + 0.914494 w_k1_KO_s116
                                 + 0.696873 w_k1_MAT_s116
                                 + 1.05851 w_k1_MSFT_s116
                                 + 0.871381 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.2295 w_k1_ADM_s116
                                 + 1.22696 w_k1_AXP_s116
                                 + 1.08051 w_k1_DIS_s116
                                 + 0.976686 w_k1_FDX_s116
                                 + 1.36885 w_k1_IBM_s116
                                 + 0.953659 w_k1_INTC_s116
                                 + 1.18077 w_k1_KO_s116 + 1.59559 w_k1_MAT_s116
                                 + 0.938897 w_k1_MSFT_s116
                                 + 1.04827 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.00696 w_k1_ADM_s116
                                 + 0.827542 w_k1_AXP_s116
                                 + 0.832239 w_k1_DIS_s116
                                 + 0.94075 w_k1_FDX_s116 + 1.068 w_k1_IBM_s116
                                 + 0.888844 w_k1_INTC_s116
                                 + 1.02383 w_k1_KO_s116 + 1.17927 w_k1_MAT_s116
                                 + 0.944675 w_k1_MSFT_s116
                                 + 0.958214 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.869628 w_k1_ADM_s116
                                 + 0.88788 w_k1_AXP_s116
                                 + 0.76524 w_k1_DIS_s116
                                 + 0.828028 w_k1_FDX_s116
                                 + 1.26034 w_k1_IBM_s116
                                 + 0.925918 w_k1_INTC_s116
                                 + 1.08355 w_k1_KO_s116 + 1.29913 w_k1_MAT_s116
                                 + 0.891033 w_k1_MSFT_s116
                                 + 0.815906 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.745141 w_k1_ADM_s116
                                 + 0.596967 w_k1_AXP_s116
                                 + 0.587125 w_k1_DIS_s116
                                 + 0.769555 w_k1_FDX_s116
                                 + 0.773092 w_k1_IBM_s116
                                 + 0.703648 w_k1_INTC_s116
                                 + 0.811355 w_k1_KO_s116
                                 + 0.723044 w_k1_MAT_s116
                                 + 0.774741 w_k1_MSFT_s116
                                 + 0.837125 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.17432 w_k1_ADM_s116
                                 + 1.00073 w_k1_AXP_s116
                                 + 0.892013 w_k1_DIS_s116
                                 + 1.02409 w_k1_FDX_s116
                                 + 0.837531 w_k1_IBM_s116
                                 + 0.916189 w_k1_INTC_s116
                                 + 1.03586 w_k1_KO_s116 + 1.10102 w_k1_MAT_s116
                                 + 0.918635 w_k1_MSFT_s116
                                 + 1.01235 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.37818 w_k1_ADM_s116
                                 + 1.23328 w_k1_AXP_s116 + 1.4161 w_k1_DIS_s116
                                 + 1.17116 w_k1_FDX_s116
                                 + 0.98195 w_k1_IBM_s116
                                 + 1.16193 w_k1_INTC_s116
                                 + 1.33104 w_k1_KO_s116 + 1.18382 w_k1_MAT_s116
                                 + 1.35719 w_k1_MSFT_s116
                                 + 1.06589 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.2218 w_k1_ADM_s116
                                 + 1.31963 w_k1_AXP_s116
                                 + 1.36499 w_k1_DIS_s116
                                 + 1.26927 w_k1_FDX_s116
                                 + 1.54465 w_k1_IBM_s116
                                 + 1.0037 w_k1_INTC_s116 + 1.44641 w_k1_KO_s116
                                 + 2.26228 w_k1_MAT_s116
                                 + 1.19946 w_k1_MSFT_s116
                                 + 1.22444 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.02108 w_k1_ADM_s117
                                 + 0.819315 w_k1_AXP_s117
                                 + 0.655466 w_k1_DIS_s117
                                 + 0.878074 w_k1_FDX_s117
                                 + 0.757638 w_k1_IBM_s117
                                 + 0.796065 w_k1_INTC_s117
                                 + 0.640746 w_k1_KO_s117
                                 + 0.71756 w_k1_MAT_s117
                                 + 0.709873 w_k1_MSFT_s117
                                 + 0.960813 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.12344 w_k1_ADM_s117
                                 + 0.896895 w_k1_AXP_s117
                                 + 1.16542 w_k1_DIS_s117 + 1.1245 w_k1_FDX_s117
                                 + 0.855752 w_k1_IBM_s117
                                 + 0.927594 w_k1_INTC_s117
                                 + 1.0586 w_k1_KO_s117 + 1.04101 w_k1_MAT_s117
                                 + 0.993955 w_k1_MSFT_s117
                                 + 1.07283 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.01203 w_k1_ADM_s117
                                 + 0.881112 w_k1_AXP_s117
                                 + 0.906318 w_k1_DIS_s117
                                 + 0.919869 w_k1_FDX_s117
                                 + 1.55991 w_k1_IBM_s117
                                 + 0.873999 w_k1_INTC_s117
                                 + 1.16659 w_k1_KO_s117 + 1.55823 w_k1_MAT_s117
                                 + 0.840818 w_k1_MSFT_s117
                                 + 1.10639 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.29587 w_k1_ADM_s117
                                 + 1.42835 w_k1_AXP_s117
                                 + 1.59061 w_k1_DIS_s117
                                 + 1.47376 w_k1_FDX_s117
                                 + 1.19632 w_k1_IBM_s117
                                 + 1.13112 w_k1_INTC_s117
                                 + 1.20574 w_k1_KO_s117 + 1.71267 w_k1_MAT_s117
                                 + 1.27131 w_k1_MSFT_s117
                                 + 1.23295 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.940304 w_k1_ADM_s117
                                 + 0.857568 w_k1_AXP_s117
                                 + 0.861178 w_k1_DIS_s117
                                 + 0.903699 w_k1_FDX_s117
                                 + 1.06972 w_k1_IBM_s117
                                 + 0.813166 w_k1_INTC_s117
                                 + 0.97886 w_k1_KO_s117 + 1.094 w_k1_MAT_s117
                                 + 0.912843 w_k1_MSFT_s117
                                 + 0.962076 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.52595 w_k1_ADM_s117
                                 + 1.01621 w_k1_AXP_s117
                                 + 1.08913 w_k1_DIS_s117
                                 + 1.35327 w_k1_FDX_s117
                                 + 0.673229 w_k1_IBM_s117
                                 + 1.01097 w_k1_INTC_s117
                                 + 0.964057 w_k1_KO_s117
                                 + 0.854456 w_k1_MAT_s117
                                 + 1.02774 w_k1_MSFT_s117
                                 + 1.18715 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.51324 w_k1_ADM_s117
                                 + 1.28524 w_k1_AXP_s117 + 1.514 w_k1_DIS_s117
                                 + 1.27449 w_k1_FDX_s117
                                 + 1.68948 w_k1_IBM_s117
                                 + 1.07147 w_k1_INTC_s117
                                 + 1.59442 w_k1_KO_s117 + 1.54565 w_k1_MAT_s117
                                 + 1.19203 w_k1_MSFT_s117
                                 + 1.47588 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.06743 w_k1_ADM_s117
                                 + 1.44443 w_k1_AXP_s117
                                 + 1.41497 w_k1_DIS_s117
                                 + 1.39231 w_k1_FDX_s117
                                 + 0.859224 w_k1_IBM_s117
                                 + 1.31372 w_k1_INTC_s117
                                 + 1.08096 w_k1_KO_s117
                                 + 0.872648 w_k1_MAT_s117
                                 + 1.27871 w_k1_MSFT_s117
                                 + 1.13484 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0176906647195499 w_k1_ADM_s000
                                 + 0.0130115260805617 w_k1_ADM_s001
                                 + 0.0166616491414819 w_k1_ADM_s002
                                 + 0.0127746310003835 w_k1_ADM_s003
                                 + 0.00716801809461421 w_k1_ADM_s004
                                 + 0.0276447060415858 w_k1_ADM_s005
                                 + 0.0141633261869307 w_k1_ADM_s006
                                 + 0.0112213917049775 w_k1_ADM_s007
                                 + 0.0364213017457181 w_k1_ADM_s010
                                 + 0.0189946550790566 w_k1_ADM_s011
                                 + 0.0256202162638796 w_k1_ADM_s012
                                 + 0.0537083878440178 w_k1_ADM_s013
                                 + 0.0151176984989303 w_k1_ADM_s014
                                 + 0.0154005257116442 w_k1_ADM_s015
                                 + 0.0508067066494518 w_k1_ADM_s016
                                 + 0.0209823904073353 w_k1_ADM_s017
                                 + 0.0422362747316719 w_k1_ADM_s100
                                 + 0.0286300090637349 w_k1_ADM_s101
                                 + 0.0461853507190858 w_k1_ADM_s102
                                 + 0.102792863230855 w_k1_ADM_s103
                                 + 0.0308255362241571 w_k1_ADM_s104
                                 + 0.0446773709544966 w_k1_ADM_s105
                                 + 0.106440044354154 w_k1_ADM_s106
                                 + 0.0421226505847201 w_k1_ADM_s107
                                 + 0.0385939234645433 w_k1_ADM_s110
                                 + 0.0197798076250467 w_k1_ADM_s111
                                 + 0.00769505582106232 w_k1_ADM_s112
                                 + 0.0348441861494503 w_k1_ADM_s113
                                 + 0.0271340737830952 w_k1_ADM_s114
                                 + 0.0317757705393111 w_k1_ADM_s115
                                 + 0.0207176752904735 w_k1_ADM_s116
                                 + 0.0287868348325397 w_k1_ADM_s117
                                 + 0.0182273881522338 w_k1_AXP_s000
                                 + 0.0134781951424381 w_k1_AXP_s001
                                 + 0.0172221772527944 w_k1_AXP_s002
                                 + 0.0121278784391424 w_k1_AXP_s003
                                 + 0.00678732544193205 w_k1_AXP_s004
                                 + 0.0321949428714347 w_k1_AXP_s005
                                 + 0.0169871774387248 w_k1_AXP_s006
                                 + 0.011600118664317 w_k1_AXP_s007
                                 + 0.0339184095417547 w_k1_AXP_s010
                                 + 0.0216028645430455 w_k1_AXP_s011
                                 + 0.021026042020237 w_k1_AXP_s012
                                 + 0.0554570548916485 w_k1_AXP_s013
                                 + 0.0162186221149454 w_k1_AXP_s014
                                 + 0.0178274149325352 w_k1_AXP_s015
                                 + 0.0519446114896568 w_k1_AXP_s016
                                 + 0.0201067932233398 w_k1_AXP_s017
                                 + 0.0423500692545022 w_k1_AXP_s100
                                 + 0.0263907345796489 w_k1_AXP_s101
                                 + 0.0493015359767396 w_k1_AXP_s102
                                 + 0.103747639442223 w_k1_AXP_s103
                                 + 0.0266748567573403 w_k1_AXP_s104
                                 + 0.043210396239681 w_k1_AXP_s105
                                 + 0.111172428613529 w_k1_AXP_s106
                                 + 0.0368398426443449 w_k1_AXP_s107
                                 + 0.0396248204767539 w_k1_AXP_s110
                                 + 0.0194185251165416 w_k1_AXP_s111
                                 + 0.00830910141495378 w_k1_AXP_s112
                                 + 0.0380702781470689 w_k1_AXP_s113
                                 + 0.0350134743438078 w_k1_AXP_s114
                                 + 0.0322193322208998 w_k1_AXP_s115
                                 + 0.0186338038233073 w_k1_AXP_s116
                                 + 0.0253362724910578 w_k1_AXP_s117
                                 + 0.0193122956980459 w_k1_DIS_s000
                                 + 0.0136655905010511 w_k1_DIS_s001
                                 + 0.0175809192837283 w_k1_DIS_s002
                                 + 0.0116923994093711 w_k1_DIS_s003
                                 + 0.00678732228678636 w_k1_DIS_s004
                                 + 0.0320058765979793 w_k1_DIS_s005
                                 + 0.0161590145684995 w_k1_DIS_s006
                                 + 0.0114365189593738 w_k1_DIS_s007
                                 + 0.0395408081335901 w_k1_DIS_s010
                                 + 0.0210877875680555 w_k1_DIS_s011
                                 + 0.0213270207089436 w_k1_DIS_s012
                                 + 0.0568374777339531 w_k1_DIS_s013
                                 + 0.0157963581609218 w_k1_DIS_s014
                                 + 0.0184533921362567 w_k1_DIS_s015
                                 + 0.0525909667130407 w_k1_DIS_s016
                                 + 0.0195027477393178 w_k1_DIS_s017
                                 + 0.044169322831272 w_k1_DIS_s100
                                 + 0.0302455143014742 w_k1_DIS_s101
                                 + 0.0478167449100513 w_k1_DIS_s102
                                 + 0.103136299817478 w_k1_DIS_s103
                                 + 0.0279981196429926 w_k1_DIS_s104
                                 + 0.0503180399193527 w_k1_DIS_s105
                                 + 0.106108077648484 w_k1_DIS_s106
                                 + 0.0397101029066082 w_k1_DIS_s107
                                 + 0.0365617483457858 w_k1_DIS_s110
                                 + 0.0204085526039232 w_k1_DIS_s111
                                 + 0.0088472844730995 w_k1_DIS_s112
                                 + 0.0400899231349885 w_k1_DIS_s113
                                 + 0.0344854226547452 w_k1_DIS_s114
                                 + 0.0331791286501501 w_k1_DIS_s115
                                 + 0.0188957700633913 w_k1_DIS_s116
                                 + 0.0258766790519374 w_k1_DIS_s117
                                 + 0.0185533922996344 w_k1_FDX_s000
                                 + 0.0138756056157525 w_k1_FDX_s001
                                 + 0.0175768753250175 w_k1_FDX_s002
                                 + 0.0112897824965624 w_k1_FDX_s003
                                 + 0.00746038395189271 w_k1_FDX_s004
                                 + 0.0316096791008137 w_k1_FDX_s005
                                 + 0.0165457945264933 w_k1_FDX_s006
                                 + 0.0110307602965203 w_k1_FDX_s007
                                 + 0.0360347641437641 w_k1_FDX_s010
                                 + 0.0214160867968974 w_k1_FDX_s011
                                 + 0.0218985287607162 w_k1_FDX_s012
                                 + 0.0562014960153561 w_k1_FDX_s013
                                 + 0.0165905604633471 w_k1_FDX_s014
                                 + 0.0177381331621999 w_k1_FDX_s015
                                 + 0.0521948689367567 w_k1_FDX_s016
                                 + 0.0183223389808488 w_k1_FDX_s017
                                 + 0.0427025366776713 w_k1_FDX_s100
                                 + 0.030030585120329 w_k1_FDX_s101
                                 + 0.0463664850018809 w_k1_FDX_s102
                                 + 0.100429861932376 w_k1_FDX_s103
                                 + 0.0280640757475341 w_k1_FDX_s104
                                 + 0.0494861767210401 w_k1_FDX_s105
                                 + 0.105379965471859 w_k1_FDX_s106
                                 + 0.0369852622561078 w_k1_FDX_s107
                                 + 0.0364910123166146 w_k1_FDX_s110
                                 + 0.0193492348226113 w_k1_FDX_s111
                                 + 0.00825031646994427 w_k1_FDX_s112
                                 + 0.0360087796472224 w_k1_FDX_s113
                                 + 0.0323204637689766 w_k1_FDX_s114
                                 + 0.0334080341214023 w_k1_FDX_s115
                                 + 0.0188741333704717 w_k1_FDX_s116
                                 + 0.0274429959608856 w_k1_FDX_s117
                                 + 0.0191823613763933 w_k1_IBM_s000
                                 + 0.014930506615492 w_k1_IBM_s001
                                 + 0.0190512949580656 w_k1_IBM_s002
                                 + 0.0132347560525624 w_k1_IBM_s003
                                 + 0.00730318382650149 w_k1_IBM_s004
                                 + 0.0327848905117147 w_k1_IBM_s005
                                 + 0.0167228947312787 w_k1_IBM_s006
                                 + 0.0110363508652263 w_k1_IBM_s007
                                 + 0.0371702576964474 w_k1_IBM_s010
                                 + 0.020825809441115 w_k1_IBM_s011
                                 + 0.021929523246068 w_k1_IBM_s012
                                 + 0.0559102444712977 w_k1_IBM_s013
                                 + 0.0161310060743168 w_k1_IBM_s014
                                 + 0.0177942287917951 w_k1_IBM_s015
                                 + 0.0527752779116399 w_k1_IBM_s016
                                 + 0.0190837022011108 w_k1_IBM_s017
                                 + 0.0421103585526567 w_k1_IBM_s100
                                 + 0.0296414798393453 w_k1_IBM_s101
                                 + 0.0466249826538444 w_k1_IBM_s102
                                 + 0.10495927607634 w_k1_IBM_s103
                                 + 0.0280237933262501 w_k1_IBM_s104
                                 + 0.050370636964235 w_k1_IBM_s105
                                 + 0.112279493720016 w_k1_IBM_s106
                                 + 0.0385731742435763 w_k1_IBM_s107
                                 + 0.037514139313255 w_k1_IBM_s110
                                 + 0.0190902349186106 w_k1_IBM_s111
                                 + 0.00831353218987594 w_k1_IBM_s112
                                 + 0.0387840727292543 w_k1_IBM_s113
                                 + 0.0334563661347606 w_k1_IBM_s114
                                 + 0.032407347224842 w_k1_IBM_s115
                                 + 0.01919983352284 w_k1_IBM_s116
                                 + 0.0257574788035395 w_k1_IBM_s117
                                 + 0.0179002567913845 w_k1_INTC_s000
                                 + 0.0157392180174347 w_k1_INTC_s001
                                 + 0.0171833281771623 w_k1_INTC_s002
                                 + 0.010507306434507 w_k1_INTC_s003
                                 + 0.0079020961307678 w_k1_INTC_s004
                                 + 0.033037414316043 w_k1_INTC_s005
                                 + 0.0147183949264154 w_k1_INTC_s006
                                 + 0.00946493034485109 w_k1_INTC_s007
                                 + 0.034664999400444 w_k1_INTC_s010
                                 + 0.0223144091796615 w_k1_INTC_s011
                                 + 0.0226268124630467 w_k1_INTC_s012
                                 + 0.0598103324352759 w_k1_INTC_s013
                                 + 0.0174248934454322 w_k1_INTC_s014
                                 + 0.0197576287979816 w_k1_INTC_s015
                                 + 0.0511023824057078 w_k1_INTC_s016
                                 + 0.0180756571703285 w_k1_INTC_s017
                                 + 0.0434676849652153 w_k1_INTC_s100
                                 + 0.026306876019329 w_k1_INTC_s101
                                 + 0.0449009972106674 w_k1_INTC_s102
                                 + 0.108377927270356 w_k1_INTC_s103
                                 + 0.027566407542244 w_k1_INTC_s104
                                 + 0.0444216132148516 w_k1_INTC_s105
                                 + 0.103140145654092 w_k1_INTC_s106
                                 + 0.0364417406873752 w_k1_INTC_s107
                                 + 0.0357491124346297 w_k1_INTC_s110
                                 + 0.0198305274464895 w_k1_INTC_s111
                                 + 0.00722492905704532 w_k1_INTC_s112
                                 + 0.0375298145329141 w_k1_INTC_s113
                                 + 0.0308305927737186 w_k1_INTC_s114
                                 + 0.0321209821059116 w_k1_INTC_s115
                                 + 0.0178647813536548 w_k1_INTC_s116
                                 + 0.0236022049388618 w_k1_INTC_s117
                                 + 0.0176002451274173 w_k1_KO_s000
                                 + 0.0126713800303908 w_k1_KO_s001
                                 + 0.0170004599312002 w_k1_KO_s002
                                 + 0.0109749075596128 w_k1_KO_s003
                                 + 0.0063259805033438 w_k1_KO_s004
                                 + 0.0326599080910425 w_k1_KO_s005
                                 + 0.0156505226382655 w_k1_KO_s006
                                 + 0.0115855747479982 w_k1_KO_s007
                                 + 0.0354438850048289 w_k1_KO_s010
                                 + 0.0212124975028133 w_k1_KO_s011
                                 + 0.0224496587457919 w_k1_KO_s012
                                 + 0.0502051419587504 w_k1_KO_s013
                                 + 0.0159898567952132 w_k1_KO_s014
                                 + 0.016989198840905 w_k1_KO_s015
                                 + 0.0540068980540185 w_k1_KO_s016
                                 + 0.0192273165975621 w_k1_KO_s017
                                 + 0.0408289794429858 w_k1_KO_s100
                                 + 0.0319719281955579 w_k1_KO_s101
                                 + 0.0421807847572117 w_k1_KO_s102
                                 + 0.109268364865074 w_k1_KO_s103
                                 + 0.0269823524837802 w_k1_KO_s104
                                 + 0.0530067976190183 w_k1_KO_s105
                                 + 0.0994271677664595 w_k1_KO_s106
                                 + 0.0434215356117924 w_k1_KO_s107
                                 + 0.0391689714230322 w_k1_KO_s110
                                 + 0.0205372608582215 w_k1_KO_s111
                                 + 0.00865953726357031 w_k1_KO_s112
                                 + 0.0384950379164072 w_k1_KO_s113
                                 + 0.029456149948869 w_k1_KO_s114
                                 + 0.0340057795043985 w_k1_KO_s115
                                 + 0.0207673700829343 w_k1_KO_s116
                                 + 0.0249756981138138 w_k1_KO_s117
                                 + 0.0138035921803311 w_k1_MAT_s000
                                 + 0.0124578169101744 w_k1_MAT_s001
                                 + 0.0186996565779787 w_k1_MAT_s002
                                 + 0.0109298372268633 w_k1_MAT_s003
                                 + 0.00621654360171693 w_k1_MAT_s004
                                 + 0.0359739535842551 w_k1_MAT_s005
                                 + 0.0185453695114066 w_k1_MAT_s006
                                 + 0.0137867310809838 w_k1_MAT_s007
                                 + 0.0345625060442576 w_k1_MAT_s010
                                 + 0.0207089210680037 w_k1_MAT_s011
                                 + 0.0194085041762836 w_k1_MAT_s012
                                 + 0.0595887454807515 w_k1_MAT_s013
                                 + 0.0167220803857552 w_k1_MAT_s014
                                 + 0.016379392315087 w_k1_MAT_s015
                                 + 0.0575082341828084 w_k1_MAT_s016
                                 + 0.0221333432129331 w_k1_MAT_s017
                                 + 0.0404807819678921 w_k1_MAT_s100
                                 + 0.0333858618171893 w_k1_MAT_s101
                                 + 0.0415452848329529 w_k1_MAT_s102
                                 + 0.109262633171629 w_k1_MAT_s103
                                 + 0.0287054032537773 w_k1_MAT_s104
                                 + 0.0519725129932219 w_k1_MAT_s105
                                 + 0.0925680515098166 w_k1_MAT_s106
                                 + 0.0397248600739988 w_k1_MAT_s107
                                 + 0.0374059626088301 w_k1_MAT_s110
                                 + 0.0221967246503336 w_k1_MAT_s111
                                 + 0.00880856065150443 w_k1_MAT_s112
                                 + 0.0424989991774212 w_k1_MAT_s113
                                 + 0.0287993854705764 w_k1_MAT_s114
                                 + 0.0301993259283096 w_k1_MAT_s115
                                 + 0.0224063687733272 w_k1_MAT_s116
                                 + 0.0275193148718883 w_k1_MAT_s117
                                 + 0.0161096534804275 w_k1_MSFT_s000
                                 + 0.0147797007235132 w_k1_MSFT_s001
                                 + 0.0191582545240765 w_k1_MSFT_s002
                                 + 0.010734222786926 w_k1_MSFT_s003
                                 + 0.00761366415241827 w_k1_MSFT_s004
                                 + 0.0315956317660765 w_k1_MSFT_s005
                                 + 0.0164979811948296 w_k1_MSFT_s006
                                 + 0.0104444302011311 w_k1_MSFT_s007
                                 + 0.0374841259360615 w_k1_MSFT_s010
                                 + 0.020045452790459 w_k1_MSFT_s011
                                 + 0.0216566101607074 w_k1_MSFT_s012
                                 + 0.0570353520980713 w_k1_MSFT_s013
                                 + 0.0169960386180208 w_k1_MSFT_s014
                                 + 0.0182975999526609 w_k1_MSFT_s015
                                 + 0.0506243715843148 w_k1_MSFT_s016
                                 + 0.0186077624865464 w_k1_MSFT_s017
                                 + 0.043984196833049 w_k1_MSFT_s100
                                 + 0.0310962881331649 w_k1_MSFT_s101
                                 + 0.0445350813714552 w_k1_MSFT_s102
                                 + 0.103419685405516 w_k1_MSFT_s103
                                 + 0.024232176773556 w_k1_MSFT_s104
                                 + 0.0482081846575172 w_k1_MSFT_s105
                                 + 0.112320015181349 w_k1_MSFT_s106
                                 + 0.0368764358904939 w_k1_MSFT_s107
                                 + 0.0369621400400899 w_k1_MSFT_s110
                                 + 0.020789322503275 w_k1_MSFT_s111
                                 + 0.00733445827405412 w_k1_MSFT_s112
                                 + 0.0387093413949718 w_k1_MSFT_s113
                                 + 0.028761920709767 w_k1_MSFT_s114
                                 + 0.0359188538054712 w_k1_MSFT_s115
                                 + 0.0193984071450748 w_k1_MSFT_s116
                                 + 0.0241447710061582 w_k1_MSFT_s117
                                 + 0.0174414901458144 w_k1_PG_s000
                                 + 0.0127010265631266 w_k1_PG_s001
                                 + 0.0179295487370427 w_k1_PG_s002
                                 + 0.0112563140736147 w_k1_PG_s003
                                 + 0.00673006787554631 w_k1_PG_s004
                                 + 0.0303619439583444 w_k1_PG_s005
                                 + 0.0168410501345108 w_k1_PG_s006
                                 + 0.0114659499645337 w_k1_PG_s007
                                 + 0.0356404347142101 w_k1_PG_s010
                                 + 0.0208095418965805 w_k1_PG_s011
                                 + 0.0209881477243657 w_k1_PG_s012
                                 + 0.0533427997328728 w_k1_PG_s013
                                 + 0.0157893398550322 w_k1_PG_s014
                                 + 0.0165982434428437 w_k1_PG_s015
                                 + 0.0562951565445319 w_k1_PG_s016
                                 + 0.0178726272330983 w_k1_PG_s017
                                 + 0.0395557092861061 w_k1_PG_s100
                                 + 0.0294666673013604 w_k1_PG_s101
                                 + 0.0450170160484631 w_k1_PG_s102
                                 + 0.100852095137321 w_k1_PG_s103
                                 + 0.0257687325864017 w_k1_PG_s104
                                 + 0.04507912262457 w_k1_PG_s105
                                 + 0.0981276418423267 w_k1_PG_s106
                                 + 0.0389747385026271 w_k1_PG_s107
                                 + 0.0380056808628676 w_k1_PG_s110
                                 + 0.018401359181531 w_k1_PG_s111
                                 + 0.00803387835093544 w_k1_PG_s112
                                 + 0.0365563880303511 w_k1_PG_s113
                                 + 0.0318170062275072 w_k1_PG_s114
                                 + 0.0299587498203894 w_k1_PG_s115
                                 + 0.0185202442827354 w_k1_PG_s116
                                 + 0.0276324801591829 w_k1_PG_s117 >= 1.13
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

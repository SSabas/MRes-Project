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
 w_k1_ADM_s0_return_constraint:  1.14718 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.938148 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.2617 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.976921 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.0263 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.13323 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.726831 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.02184 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.993937 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.911465 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.911467 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.799636 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.14168 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.24522 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.14482 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.29047 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.14026 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.0564 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.02548 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.835846 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.832726 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.806198 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.54541 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.752011 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.10138 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.16983 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.794942 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.05305 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.850672 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.91327 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.41159 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.10542 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.06774 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.74581 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.827701 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.16948 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.942857 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.08681 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.29668 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.929196 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.28726 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.893882 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.943313 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.38155 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.637492 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.16033 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.944427 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.947001 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.903744 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.692261 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.50636 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.44589 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.36181 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.71127 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.27603 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.06836 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.08979 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.919462 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.810163 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.672172 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.69912 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.20524 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.47866 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.24676 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.749318 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.07874 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.684005 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.997786 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.49935 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.32503 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.986205 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.553881 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.961797 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.36396 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.695383 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.01848 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.38716 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.888819 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.37988 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.879029 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.994296 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.41654 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.595301 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.35568 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.961938 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.859607 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.05727 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.671079 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.91497 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.39114 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.35743 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.71443 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.41919 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.79424 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.962288 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.720249 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.04564 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.77062 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.74433 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.991705 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.06403 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.50528 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.726213 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.0215 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.676095 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.963087 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.78002 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.60717 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.31322 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.548614 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.836568 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.30685 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.808404 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.984357 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.23789 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.932476 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.25882 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.899926 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.993593 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.119 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.697573 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.17659 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.945141 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.0427 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.849093 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.745365 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.43762 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.13027 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.21748 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.30369 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.31454 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.994168 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.84395 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.789125 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.903817 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.841618 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.56905 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.878342 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.19539 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.16491 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.771437 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.11002 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.791766 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.853983 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.44076 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.47017 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.14212 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.670356 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.867699 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.2 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.88992 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.12719 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.22917 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.949945 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.09779 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.01509 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.893133 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.57775 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.570626 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.920371 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.874588 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.649356 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.40938 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.822561 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.20808 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.51066 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.21832 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 2.39863 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.82416 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.655476 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.35502 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.890414 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.997349 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.590699 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.02974 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.82954 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.17509 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.35943 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.02641 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.684116 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.674759 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.34961 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.56714 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.868646 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.12046 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.79924 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.44531 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.27435 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.837983 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.83681 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.17908 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 1.00519 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.09833 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.923687 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.925853 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.14827 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.802077 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.11065 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.04067 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.02622 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.972129 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.905883 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.59601 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.12632 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.13213 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.3448 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.21199 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.990702 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.914414 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.858809 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.02419 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.925837 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.49473 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.9603 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.13576 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.33646 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.921991 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.16696 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.79419 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.943311 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.28104 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.27075 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.909195 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.690252 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.899039 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.06416 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.903039 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.00611 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.18956 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.906455 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.18342 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.964114 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.918567 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.28916 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.701455 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.26815 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.977237 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.812845 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.1384 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.838439 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.39737 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.33412 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.30652 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.96705 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.21642 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.75967 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.07016 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.838298 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.05639 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.650053 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.33144 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.20206 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.0508 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.47827 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.904462 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.842467 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.796602 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.18183 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.71049 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.38385 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.29315 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.807938 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.1651 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.26668 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.921308 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.0444 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.28369 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.93377 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.38356 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.994357 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.813889 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.62592 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.450335 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.775727 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.650456 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.664357 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.08486 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.63167 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.742577 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.66021 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.52902 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 2.33227 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.84725 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.715795 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.51859 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.997244 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.829252 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.458329 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.54741 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 2.01012 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.31609 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.41986 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.923794 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.869788 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.743834 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.48399 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.82106 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.11391 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.12469 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.646616 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.44955 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.36555 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.774476 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.02324 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.2109 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.951646 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.21138 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.94218 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.984816 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.15643 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.768818 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.28044 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.01978 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.940501 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.957219 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.774494 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.32439 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.09968 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.14149 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.31084 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.2281 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.952626 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.897471 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.746867 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.996362 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.846715 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.46622 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.919463 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.06402 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.26642 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.868004 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.00822 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.769943 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.927886 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.43209 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.25657 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.11151 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.665015 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.795687 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.03053 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.888308 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.01939 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.16181 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.977329 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.14135 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.939207 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.915528 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.06739 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.777655 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.15167 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.91061 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.910999 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.00565 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.870937 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.15785 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.16334 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.13674 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.27713 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.1807 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.876272 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.996119 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.845078 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.00054 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.834198 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.35562 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.05506 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.10831 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.24945 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.953839 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.02739 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.835995 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.993609 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.52799 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.32201 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.145 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.840176 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.07358 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.28177 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.92248 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.05219 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.0450000602826744 z_k1_s0000
                                 - 0.0290000628822114 z_k1_s0001
                                 - 0.103000186047097 z_k1_s0002
                                 - 0.108999946884341 z_k1_s0003
                                 - 0.0420000768638666 z_k1_s0004
                                 - 0.10200005424058 z_k1_s0005
                                 - 0.0380000646387276 z_k1_s0006
                                 - 0.0470000663952439 z_k1_s0007
                                 - 0.209000248071834 z_k1_s0010
                                 - 0.0920002323458467 z_k1_s0011
                                 - 0.209000248071834 z_k1_s0012
                                 - 0.0730001993483776 z_k1_s0013
                                 - 0.0780002019845364 z_k1_s0014
                                 - 0.211000662767475 z_k1_s0015
                                 - 0.181000187349213 z_k1_s0016
                                 - 0.0960002574348392 z_k1_s0017
                                 - 0.318000699082657 z_k1_s0020
                                 - 0.217001177172148 z_k1_s0021
                                 - 0.173000603762758 z_k1_s0022
                                 - 0.169000982272556 z_k1_s0023
                                 - 0.237000863628882 z_k1_s0024
                                 - 0.109000343896634 z_k1_s0025
                                 - 0.283000458280513 z_k1_s0026
                                 - 0.0730002766722259 z_k1_s0027
                                 - 0.0989998363203667 z_k1_s0030
                                 - 0.0949998230316989 z_k1_s0031
                                 - 0.130000172632145 z_k1_s0032
                                 - 0.0939998456344878 z_k1_s0033
                                 - 0.0679999148078823 z_k1_s0034
                                 - 0.0679999148078823 z_k1_s0035
                                 - 0.180000079490934 z_k1_s0036
                                 - 0.30299947704419 z_k1_s0037
                                 - 0.0850002035673263 z_k1_s0040
                                 - 0.222001137036456 z_k1_s0041
                                 - 0.174000050972033 z_k1_s0042
                                 - 0.102000351481109 z_k1_s0043
                                 - 0.146000400104028 z_k1_s0044
                                 - 0.104000280600116 z_k1_s0045
                                 - 0.11400003339547 z_k1_s0046
                                 - 0.124999857950644 z_k1_s0047
                                 - 0.106000480246086 z_k1_s0050
                                 - 0.12000002540988 z_k1_s0051
                                 - 0.0350000401112643 z_k1_s0052
                                 - 0.088000411034505 z_k1_s0053
                                 - 0.0460000751405527 z_k1_s0054
                                 - 0.129000387016164 z_k1_s0055
                                 - 0.0330000396877663 z_k1_s0056
                                 - 0.097000118639801 z_k1_s0057
                                 - 0.055000229562511 z_k1_s0060
                                 - 0.0860002950612582 z_k1_s0061
                                 - 0.0729999761096112 z_k1_s0062
                                 - 0.0800003795455582 z_k1_s0063
                                 - 0.0870002809805416 z_k1_s0064
                                 - 0.0400000893725864 z_k1_s0065
                                 - 0.055000229562511 z_k1_s0066
                                 - 0.0260000355021378 z_k1_s0067
                                 - 0.196000607165538 z_k1_s0070
                                 - 0.0510000385807161 z_k1_s0071
                                 - 0.19100074244207 z_k1_s0072
                                 - 0.0680000514409548 z_k1_s0073
                                 - 0.195000634220845 z_k1_s0074
                                 - 0.228999714340429 z_k1_s0075
                                 - 0.154000561487289 z_k1_s0076
                                 - 0.0980000671825459 z_k1_s0077
                                 - 0.616000624435867 z_k1_s0100
                                 - 0.300999563600867 z_k1_s0101
                                 - 0.401999332098343 z_k1_s0102
                                 - 0.35099930587714 z_k1_s0103
                                 - 0.495999590976208 z_k1_s0104
                                 - 0.405999228880615 z_k1_s0105
                                 - 0.704997811592468 z_k1_s0106
                                 - 0.853000187523785 z_k1_s0107
                                 - 0.185999778577551 z_k1_s0110
                                 - 0.0440000299118999 z_k1_s0111
                                 - 0.136999751000663 z_k1_s0112
                                 - 0.0990000042267701 z_k1_s0113
                                 - 0.0529999920688762 z_k1_s0114
                                 - 0.132999954708688 z_k1_s0115
                                 - 0.14600013365767 z_k1_s0116
                                 - 0.0429999967388998 z_k1_s0117
                                 - 0.420999963621268 z_k1_s0120
                                 - 0.465000475638599 z_k1_s0121
                                 - 0.168999981339655 z_k1_s0122
                                 - 0.423998723077227 z_k1_s0123
                                 - 0.40900004579832 z_k1_s0124
                                 - 0.177999919706866 z_k1_s0125
                                 - 0.0949999187649075 z_k1_s0126
                                 - 0.278999309383337 z_k1_s0127
                                 - 0.139999937033083 z_k1_s0130
                                 - 0.18999981859499 z_k1_s0131
                                 - 0.240999306312504 z_k1_s0132
                                 - 0.351000120387682 z_k1_s0133
                                 - 0.226000416482917 z_k1_s0134
                                 - 0.218000113041316 z_k1_s0135
                                 - 0.108999864620839 z_k1_s0136
                                 - 0.444000529524233 z_k1_s0137
                                 - 0.358999349150379 z_k1_s0140
                                 - 0.428000231870826 z_k1_s0141
                                 - 0.476997902258024 z_k1_s0142
                                 - 0.294999461962852 z_k1_s0143
                                 - 1.18799913278471 z_k1_s0144
                                 - 0.338999216010869 z_k1_s0145
                                 - 0.461998379752218 z_k1_s0146
                                 - 0.300999270965175 z_k1_s0147
                                 - 0.205999208791007 z_k1_s0150
                                 - 0.211999045086048 z_k1_s0151
                                 - 0.194999207083769 z_k1_s0152
                                 - 0.15299980971909 z_k1_s0153
                                 - 0.126999855074655 z_k1_s0154
                                 - 0.166999729573848 z_k1_s0155
                                 - 0.209000032437516 z_k1_s0156
                                 - 0.541999096303178 z_k1_s0157
                                 - 0.214000670327794 z_k1_s0160
                                 - 0.120999800578426 z_k1_s0161
                                 - 0.136999932851452 z_k1_s0162
                                 - 0.16999981322745 z_k1_s0163
                                 - 0.203999130033567 z_k1_s0164
                                 - 0.311999619226887 z_k1_s0165
                                 - 0.491998640550827 z_k1_s0166
                                 - 0.143999934658456 z_k1_s0167
                                 - 0.494998269426053 z_k1_s0170
                                 - 0.414998016429627 z_k1_s0171
                                 - 0.289998939518444 z_k1_s0172
                                 - 0.354997826682308 z_k1_s0173
                                 - 0.605997302062861 z_k1_s0174
                                 - 0.28299924698019 z_k1_s0175
                                 - 0.330999728376978 z_k1_s0176
                                 - 0.520997033254159 z_k1_s0177
                                 - 0.688001868080066 z_k1_s1000
                                 - 0.733001127642202 z_k1_s1001
                                 - 0.61800196802457 z_k1_s1002
                                 - 1.64200874506785 z_k1_s1003
                                 - 1.06900093918949 z_k1_s1004
                                 - 1.23700448597451 z_k1_s1005
                                 - 0.728002019026714 z_k1_s1006
                                 - 0.567001593517026 z_k1_s1007
                                 - 0.435000568005864 z_k1_s1010
                                 - 0.12799980058249 z_k1_s1011
                                 - 0.408000136474795 z_k1_s1012
                                 - 0.421000450656333 z_k1_s1013
                                 - 0.290000378670576 z_k1_s1014
                                 - 0.253999707529231 z_k1_s1015
                                 - 0.408000136474795 z_k1_s1016
                                 - 0.528999303782998 z_k1_s1017
                                 - 1.23500370655023 z_k1_s1020
                                 - 0.534000397275491 z_k1_s1021
                                 - 0.44100048351072 z_k1_s1022
                                 - 0.40900016612593 z_k1_s1023
                                 - 0.848998766539507 z_k1_s1024
                                 - 0.816999107854978 z_k1_s1025
                                 - 1.79200064734145 z_k1_s1026
                                 - 0.509999994561833 z_k1_s1027
                                 - 0.508999339534129 z_k1_s1030
                                 - 1.32000259467329 z_k1_s1031
                                 - 0.542999301462141 z_k1_s1032
                                 - 0.650999723562554 z_k1_s1033
                                 - 1.39700178440482 z_k1_s1034
                                 - 1.43299730930779 z_k1_s1035
                                 - 0.838998064953631 z_k1_s1036
                                 - 1.0009971595052 z_k1_s1037
                                 - 1.86599475166559 z_k1_s1040
                                 - 1.55100168248452 z_k1_s1041
                                 - 0.801999035794281 z_k1_s1042
                                 - 2.71699925265583 z_k1_s1043
                                 - 1.03699808958165 z_k1_s1044
                                 - 0.815998979424166 z_k1_s1045
                                 - 0.772999152560946 z_k1_s1046
                                 - 1.76299403389031 z_k1_s1047
                                 - 0.791998750300398 z_k1_s1050
                                 - 0.378999788063375 z_k1_s1051
                                 - 0.650001075140081 z_k1_s1052
                                 - 1.29399892341486 z_k1_s1053
                                 - 0.590997449292649 z_k1_s1054
                                 - 0.528000103700089 z_k1_s1055
                                 - 0.603998819835195 z_k1_s1056
                                 - 0.782999531001385 z_k1_s1057
                                 - 2.56700590260056 z_k1_s1060
                                 - 1.45300467028368 z_k1_s1061
                                 - 1.16100033794504 z_k1_s1062
                                 - 0.696000527733806 z_k1_s1063
                                 - 2.20800573951844 z_k1_s1064
                                 - 1.55899638220442 z_k1_s1065
                                 - 1.14900683728064 z_k1_s1066
                                 - 0.543000641375324 z_k1_s1067
                                 - 0.956007010109401 z_k1_s1070
                                 - 0.955004926020269 z_k1_s1071
                                 - 1.04200461403367 z_k1_s1072
                                 - 0.819003326048271 z_k1_s1073
                                 - 0.666999695777933 z_k1_s1074
                                 - 0.331001526994247 z_k1_s1075
                                 - 0.760005625300575 z_k1_s1076
                                 - 0.733005722124002 z_k1_s1077
                                 - 0.136000244487353 z_k1_s1100
                                 - 0.0710000950779254 z_k1_s1101
                                 - 0.158000045271846 z_k1_s1102
                                 - 0.0969998596414174 z_k1_s1103
                                 - 0.0340000611218382 z_k1_s1104
                                 - 0.0849999683044211 z_k1_s1105
                                 - 0.0710000950779254 z_k1_s1106
                                 - 0.0859999592491708 z_k1_s1107
                                 - 0.0650001116574826 z_k1_s1110
                                 - 0.183000170238603 z_k1_s1111
                                 - 0.273000805241789 z_k1_s1112
                                 - 0.0950000430915759 z_k1_s1113
                                 - 0.123999780647655 z_k1_s1114
                                 - 0.0640000378795975 z_k1_s1115
                                 - 0.223000719351418 z_k1_s1116
                                 - 0.173999866538026 z_k1_s1117
                                 - 0.140999749361144 z_k1_s1120
                                 - 0.0969997705748121 z_k1_s1121
                                 - 0.340000079621128 z_k1_s1122
                                 - 0.296998651260368 z_k1_s1123
                                 - 0.299999124726584 z_k1_s1124
                                 - 0.207998893064854 z_k1_s1125
                                 - 0.151999699407825 z_k1_s1126
                                 - 0.251000321425614 z_k1_s1127
                                 - 0.339000053967963 z_k1_s1130
                                 - 0.133000013896574 z_k1_s1131
                                 - 0.127000050364405 z_k1_s1132
                                 - 0.221000653701945 z_k1_s1133
                                 - 0.143000070583545 z_k1_s1134
                                 - 0.21700091294721 z_k1_s1135
                                 - 0.370999742006177 z_k1_s1136
                                 - 0.210999539814774 z_k1_s1137
                                 - 0.484000243329569 z_k1_s1140
                                 - 0.377999893117303 z_k1_s1141
                                 - 0.277000644026765 z_k1_s1142
                                 - 0.113999853710137 z_k1_s1143
                                 - 0.227999707420274 z_k1_s1144
                                 - 0.112999838785509 z_k1_s1145
                                 - 0.111999823860881 z_k1_s1146
                                 - 0.346999635753551 z_k1_s1147
                                 - 0.1229997852138 z_k1_s1150
                                 - 0.0580000357269451 z_k1_s1151
                                 - 0.0510000155667835 z_k1_s1152
                                 - 0.128999868151135 z_k1_s1153
                                 - 0.105999933224998 z_k1_s1154
                                 - 0.0919997397045487 z_k1_s1155
                                 - 0.103000274756645 z_k1_s1156
                                 - 0.103999905579219 z_k1_s1157
                                 - 0.337999705252915 z_k1_s1160
                                 - 0.271000274588317 z_k1_s1161
                                 - 0.473000932026393 z_k1_s1162
                                 - 0.17300022559774 z_k1_s1163
                                 - 0.43499998560088 z_k1_s1164
                                 - 0.862001822060864 z_k1_s1165
                                 - 0.198000284466794 z_k1_s1166
                                 - 0.495998856808555 z_k1_s1167
                                 - 0.239000680036913 z_k1_s1170
                                 - 0.265000498249718 z_k1_s1171
                                 - 0.135000218583762 z_k1_s1172
                                 - 0.126000124771382 z_k1_s1173
                                 - 0.322000752794237 z_k1_s1174
                                 - 0.134000264760256 z_k1_s1175
                                 - 0.252000589143316 z_k1_s1176
                                 - 0.225000647306729 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.10462 w_k1_ADM_s000
                                 + 1.27036 w_k1_AXP_s000
                                 + 1.72036 w_k1_DIS_s000
                                 + 1.48398 w_k1_FDX_s000
                                 + 1.13621 w_k1_IBM_s000
                                 + 1.3605 w_k1_INTC_s000 + 1.45023 w_k1_KO_s000
                                 + 1.32868 w_k1_MAT_s000
                                 + 1.4803 w_k1_MSFT_s000 + 1.25129 w_k1_PG_s000
                                 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.47074 w_k1_ADM_s000
                                 + 1.65411 w_k1_AXP_s000
                                 + 1.52413 w_k1_DIS_s000
                                 + 1.49592 w_k1_FDX_s000
                                 + 0.927226 w_k1_IBM_s000
                                 + 1.44311 w_k1_INTC_s000
                                 + 1.21174 w_k1_KO_s000
                                 + 0.878897 w_k1_MAT_s000
                                 + 1.29389 w_k1_MSFT_s000
                                 + 1.15514 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.958737 w_k1_ADM_s000
                                 + 0.989504 w_k1_AXP_s000
                                 + 0.956454 w_k1_DIS_s000
                                 + 1.14063 w_k1_FDX_s000
                                 + 0.857486 w_k1_IBM_s000
                                 + 1.10927 w_k1_INTC_s000
                                 + 0.887507 w_k1_KO_s000
                                 + 1.03086 w_k1_MAT_s000
                                 + 0.989561 w_k1_MSFT_s000
                                 + 1.03777 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.867989 w_k1_ADM_s000
                                 + 0.821629 w_k1_AXP_s000
                                 + 0.750806 w_k1_DIS_s000
                                 + 0.916661 w_k1_FDX_s000
                                 + 0.667504 w_k1_IBM_s000
                                 + 0.809401 w_k1_INTC_s000
                                 + 0.730171 w_k1_KO_s000
                                 + 0.866501 w_k1_MAT_s000
                                 + 0.865499 w_k1_MSFT_s000
                                 + 0.809815 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.12228 w_k1_ADM_s000
                                 + 0.932693 w_k1_AXP_s000
                                 + 1.18483 w_k1_DIS_s000
                                 + 1.12292 w_k1_FDX_s000
                                 + 1.43818 w_k1_IBM_s000
                                 + 0.963417 w_k1_INTC_s000
                                 + 1.1509 w_k1_KO_s000 + 1.17205 w_k1_MAT_s000
                                 + 1.03199 w_k1_MSFT_s000
                                 + 0.977291 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.21429 w_k1_ADM_s000
                                 + 1.10781 w_k1_AXP_s000
                                 + 1.31779 w_k1_DIS_s000
                                 + 1.16117 w_k1_FDX_s000
                                 + 0.965016 w_k1_IBM_s000
                                 + 1.16219 w_k1_INTC_s000
                                 + 1.10073 w_k1_KO_s000
                                 + 0.873553 w_k1_MAT_s000
                                 + 1.17031 w_k1_MSFT_s000 + 1.1176 w_k1_PG_s000
                                 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.17392 w_k1_ADM_s000
                                 + 0.943095 w_k1_AXP_s000
                                 + 0.995627 w_k1_DIS_s000
                                 + 0.958868 w_k1_FDX_s000
                                 + 1.38863 w_k1_IBM_s000
                                 + 0.981664 w_k1_INTC_s000
                                 + 1.2628 w_k1_KO_s000 + 0.972062 w_k1_MAT_s000
                                 + 1.01869 w_k1_MSFT_s000
                                 + 0.977086 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.641515 w_k1_ADM_s000
                                 + 0.834437 w_k1_AXP_s000
                                 + 0.753874 w_k1_DIS_s000
                                 + 0.776898 w_k1_FDX_s000
                                 + 1.26786 w_k1_IBM_s000
                                 + 0.718764 w_k1_INTC_s000
                                 + 0.806065 w_k1_KO_s000
                                 + 1.29477 w_k1_MAT_s000
                                 + 0.831136 w_k1_MSFT_s000
                                 + 0.889588 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.1077 w_k1_ADM_s001
                                 + 0.913957 w_k1_AXP_s001
                                 + 0.810696 w_k1_DIS_s001
                                 + 0.977371 w_k1_FDX_s001
                                 + 0.778452 w_k1_IBM_s001
                                 + 0.980573 w_k1_INTC_s001
                                 + 0.812996 w_k1_KO_s001
                                 + 0.797059 w_k1_MAT_s001
                                 + 0.865923 w_k1_MSFT_s001
                                 + 0.861482 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.39846 w_k1_ADM_s001
                                 + 1.00722 w_k1_AXP_s001
                                 + 0.963054 w_k1_DIS_s001
                                 + 1.29313 w_k1_FDX_s001
                                 + 0.697692 w_k1_IBM_s001
                                 + 1.21539 w_k1_INTC_s001
                                 + 0.749871 w_k1_KO_s001
                                 + 0.644299 w_k1_MAT_s001
                                 + 0.994178 w_k1_MSFT_s001
                                 + 0.822442 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.873808 w_k1_ADM_s001
                                 + 0.740917 w_k1_AXP_s001
                                 + 0.68791 w_k1_DIS_s001
                                 + 0.741179 w_k1_FDX_s001
                                 + 0.957935 w_k1_IBM_s001
                                 + 0.971377 w_k1_INTC_s001
                                 + 0.758267 w_k1_KO_s001
                                 + 0.810252 w_k1_MAT_s001
                                 + 0.695162 w_k1_MSFT_s001
                                 + 0.738784 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.89214 w_k1_ADM_s001
                                 + 0.902986 w_k1_AXP_s001
                                 + 0.763785 w_k1_DIS_s001
                                 + 0.91185 w_k1_FDX_s001
                                 + 1.13837 w_k1_IBM_s001
                                 + 1.14662 w_k1_INTC_s001
                                 + 0.907385 w_k1_KO_s001
                                 + 1.25727 w_k1_MAT_s001
                                 + 0.830434 w_k1_MSFT_s001
                                 + 0.936076 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.11846 w_k1_ADM_s001
                                 + 1.42267 w_k1_AXP_s001
                                 + 1.16503 w_k1_DIS_s001
                                 + 1.21444 w_k1_FDX_s001
                                 + 0.834629 w_k1_IBM_s001
                                 + 1.5111 w_k1_INTC_s001
                                 + 0.913837 w_k1_KO_s001
                                 + 1.08566 w_k1_MAT_s001
                                 + 1.00396 w_k1_MSFT_s001
                                 + 0.987134 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.1434 w_k1_ADM_s001
                                 + 0.76915 w_k1_AXP_s001
                                 + 1.32676 w_k1_DIS_s001 + 1.1907 w_k1_FDX_s001
                                 + 1.0553 w_k1_IBM_s001
                                 + 1.17912 w_k1_INTC_s001
                                 + 1.17405 w_k1_KO_s001
                                 + 0.811291 w_k1_MAT_s001
                                 + 1.13862 w_k1_MSFT_s001
                                 + 1.02707 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.11576 w_k1_ADM_s001
                                 + 1.26367 w_k1_AXP_s001 + 1.2532 w_k1_DIS_s001
                                 + 1.17383 w_k1_FDX_s001
                                 + 1.30416 w_k1_IBM_s001
                                 + 1.23244 w_k1_INTC_s001 + 1.1667 w_k1_KO_s001
                                 + 1.54978 w_k1_MAT_s001
                                 + 1.10044 w_k1_MSFT_s001
                                 + 1.04993 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.982775 w_k1_ADM_s001
                                 + 0.890824 w_k1_AXP_s001
                                 + 1.10813 w_k1_DIS_s001
                                 + 0.868481 w_k1_FDX_s001
                                 + 1.51282 w_k1_IBM_s001
                                 + 1.00378 w_k1_INTC_s001
                                 + 1.22963 w_k1_KO_s001 + 1.25975 w_k1_MAT_s001
                                 + 0.779256 w_k1_MSFT_s001
                                 + 0.859394 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.914085 w_k1_ADM_s002
                                 + 0.958607 w_k1_AXP_s002
                                 + 1.00211 w_k1_DIS_s002
                                 + 1.03589 w_k1_FDX_s002
                                 + 1.08222 w_k1_IBM_s002
                                 + 1.08722 w_k1_INTC_s002
                                 + 0.975698 w_k1_KO_s002
                                 + 1.41363 w_k1_MAT_s002
                                 + 0.932302 w_k1_MSFT_s002
                                 + 1.00982 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.18428 w_k1_ADM_s002
                                 + 1.33919 w_k1_AXP_s002
                                 + 1.41051 w_k1_DIS_s002
                                 + 1.35528 w_k1_FDX_s002
                                 + 1.08786 w_k1_IBM_s002
                                 + 1.2698 w_k1_INTC_s002 + 1.21753 w_k1_KO_s002
                                 + 1.01863 w_k1_MAT_s002
                                 + 1.27163 w_k1_MSFT_s002 + 1.0263 w_k1_PG_s002
                                 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.779021 w_k1_ADM_s002
                                 + 0.608617 w_k1_AXP_s002
                                 + 0.601153 w_k1_DIS_s002
                                 + 0.777286 w_k1_FDX_s002
                                 + 0.804197 w_k1_IBM_s002
                                 + 0.803171 w_k1_INTC_s002
                                 + 0.86627 w_k1_KO_s002
                                 + 0.951427 w_k1_MAT_s002
                                 + 0.812388 w_k1_MSFT_s002
                                 + 0.969908 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.09922 w_k1_ADM_s002
                                 + 1.18293 w_k1_AXP_s002
                                 + 0.998471 w_k1_DIS_s002
                                 + 0.939904 w_k1_FDX_s002
                                 + 1.56936 w_k1_IBM_s002
                                 + 1.02378 w_k1_INTC_s002
                                 + 1.27049 w_k1_KO_s002 + 1.8035 w_k1_MAT_s002
                                 + 0.886872 w_k1_MSFT_s002
                                 + 1.08324 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.973852 w_k1_ADM_s002
                                 + 0.751864 w_k1_AXP_s002
                                 + 0.911946 w_k1_DIS_s002
                                 + 1.01524 w_k1_FDX_s002
                                 + 0.722058 w_k1_IBM_s002
                                 + 0.900761 w_k1_INTC_s002
                                 + 0.742143 w_k1_KO_s002
                                 + 0.614484 w_k1_MAT_s002
                                 + 0.904168 w_k1_MSFT_s002
                                 + 0.885532 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.15278 w_k1_ADM_s002
                                 + 1.01244 w_k1_AXP_s002
                                 + 1.22378 w_k1_DIS_s002
                                 + 1.18536 w_k1_FDX_s002
                                 + 0.903738 w_k1_IBM_s002
                                 + 1.14635 w_k1_INTC_s002
                                 + 1.24582 w_k1_KO_s002 + 1.07304 w_k1_MAT_s002
                                 + 1.21631 w_k1_MSFT_s002
                                 + 1.14027 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.01076 w_k1_ADM_s002
                                 + 0.918491 w_k1_AXP_s002
                                 + 1.03645 w_k1_DIS_s002
                                 + 0.947915 w_k1_FDX_s002
                                 + 1.21108 w_k1_IBM_s002
                                 + 0.871859 w_k1_INTC_s002
                                 + 0.986274 w_k1_KO_s002 + 1.1032 w_k1_MAT_s002
                                 + 0.876163 w_k1_MSFT_s002
                                 + 1.10856 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.923228 w_k1_ADM_s002
                                 + 0.65451 w_k1_AXP_s002
                                 + 0.695797 w_k1_DIS_s002
                                 + 0.738682 w_k1_FDX_s002
                                 + 0.785437 w_k1_IBM_s002
                                 + 0.741016 w_k1_INTC_s002
                                 + 0.768334 w_k1_KO_s002
                                 + 0.778022 w_k1_MAT_s002
                                 + 0.802618 w_k1_MSFT_s002
                                 + 0.922772 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.39882 w_k1_ADM_s003
                                 + 1.37989 w_k1_AXP_s003
                                 + 1.53393 w_k1_DIS_s003
                                 + 1.35508 w_k1_FDX_s003
                                 + 1.12186 w_k1_IBM_s003
                                 + 1.4269 w_k1_INTC_s003 + 1.22069 w_k1_KO_s003
                                 + 1.33855 w_k1_MAT_s003
                                 + 1.45429 w_k1_MSFT_s003
                                 + 1.17907 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.27659 w_k1_ADM_s003
                                 + 1.51278 w_k1_AXP_s003
                                 + 1.20071 w_k1_DIS_s003
                                 + 1.12159 w_k1_FDX_s003
                                 + 1.50786 w_k1_IBM_s003
                                 + 1.12542 w_k1_INTC_s003
                                 + 1.25531 w_k1_KO_s003 + 1.65441 w_k1_MAT_s003
                                 + 1.20416 w_k1_MSFT_s003
                                 + 1.12342 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.15453 w_k1_ADM_s003
                                 + 1.15655 w_k1_AXP_s003
                                 + 0.847431 w_k1_DIS_s003
                                 + 0.971635 w_k1_FDX_s003
                                 + 0.910572 w_k1_IBM_s003
                                 + 0.988318 w_k1_INTC_s003
                                 + 0.882842 w_k1_KO_s003
                                 + 0.689329 w_k1_MAT_s003
                                 + 0.941133 w_k1_MSFT_s003
                                 + 1.02889 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.10009 w_k1_ADM_s003
                                 + 0.837117 w_k1_AXP_s003
                                 + 0.773029 w_k1_DIS_s003
                                 + 0.821902 w_k1_FDX_s003
                                 + 1.15829 w_k1_IBM_s003
                                 + 0.984329 w_k1_INTC_s003
                                 + 0.989744 w_k1_KO_s003
                                 + 0.751723 w_k1_MAT_s003
                                 + 0.929229 w_k1_MSFT_s003
                                 + 0.91258 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.28565 w_k1_ADM_s003
                                 + 1.75062 w_k1_AXP_s003
                                 + 1.28561 w_k1_DIS_s003
                                 + 1.18023 w_k1_FDX_s003
                                 + 1.24124 w_k1_IBM_s003
                                 + 1.2965 w_k1_INTC_s003 + 1.15945 w_k1_KO_s003
                                 + 1.21382 w_k1_MAT_s003
                                 + 1.04511 w_k1_MSFT_s003
                                 + 1.04393 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.736776 w_k1_ADM_s003
                                 + 0.756703 w_k1_AXP_s003
                                 + 0.730766 w_k1_DIS_s003
                                 + 0.799522 w_k1_FDX_s003
                                 + 0.836482 w_k1_IBM_s003
                                 + 0.983339 w_k1_INTC_s003
                                 + 0.753572 w_k1_KO_s003
                                 + 0.678502 w_k1_MAT_s003
                                 + 0.858272 w_k1_MSFT_s003
                                 + 0.841988 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.904015 w_k1_ADM_s003
                                 + 0.745655 w_k1_AXP_s003
                                 + 0.649508 w_k1_DIS_s003
                                 + 0.781788 w_k1_FDX_s003
                                 + 0.747583 w_k1_IBM_s003
                                 + 0.684344 w_k1_INTC_s003
                                 + 0.673043 w_k1_KO_s003
                                 + 0.711574 w_k1_MAT_s003
                                 + 0.714318 w_k1_MSFT_s003
                                 + 0.729965 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.28779 w_k1_ADM_s003
                                 + 0.89173 w_k1_AXP_s003
                                 + 1.10191 w_k1_DIS_s003
                                 + 1.06987 w_k1_FDX_s003
                                 + 0.854899 w_k1_IBM_s003
                                 + 1.18001 w_k1_INTC_s003
                                 + 1.00513 w_k1_KO_s003
                                 + 0.668559 w_k1_MAT_s003
                                 + 1.04659 w_k1_MSFT_s003
                                 + 1.05417 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.991705 w_k1_ADM_s004
                                 + 1.18561 w_k1_AXP_s004
                                 + 1.28594 w_k1_DIS_s004
                                 + 1.14379 w_k1_FDX_s004
                                 + 1.81726 w_k1_IBM_s004
                                 + 1.28429 w_k1_INTC_s004
                                 + 1.50765 w_k1_KO_s004 + 1.55002 w_k1_MAT_s004
                                 + 1.07992 w_k1_MSFT_s004
                                 + 1.10976 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.13191 w_k1_ADM_s004
                                 + 1.17703 w_k1_AXP_s004
                                 + 1.03144 w_k1_DIS_s004
                                 + 1.05222 w_k1_FDX_s004
                                 + 1.08079 w_k1_IBM_s004
                                 + 0.894925 w_k1_INTC_s004
                                 + 1.09054 w_k1_KO_s004
                                 + 0.972194 w_k1_MAT_s004
                                 + 1.05818 w_k1_MSFT_s004
                                 + 0.890897 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.10669 w_k1_ADM_s004
                                 + 1.04648 w_k1_AXP_s004
                                 + 1.19627 w_k1_DIS_s004
                                 + 1.20152 w_k1_FDX_s004
                                 + 0.820456 w_k1_IBM_s004
                                 + 1.25618 w_k1_INTC_s004
                                 + 1.02419 w_k1_KO_s004
                                 + 0.836934 w_k1_MAT_s004
                                 + 1.3069 w_k1_MSFT_s004 + 1.04834 w_k1_PG_s004
                                 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.851947 w_k1_ADM_s004
                                 + 0.730764 w_k1_AXP_s004
                                 + 0.6246 w_k1_DIS_s004
                                 + 0.941234 w_k1_FDX_s004
                                 + 0.605097 w_k1_IBM_s004
                                 + 1.04093 w_k1_INTC_s004
                                 + 0.862655 w_k1_KO_s004
                                 + 0.451579 w_k1_MAT_s004
                                 + 0.828817 w_k1_MSFT_s004
                                 + 0.859922 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.981649 w_k1_ADM_s004
                                 + 0.848999 w_k1_AXP_s004
                                 + 0.753396 w_k1_DIS_s004
                                 + 0.833742 w_k1_FDX_s004
                                 + 0.763744 w_k1_IBM_s004
                                 + 1.12548 w_k1_INTC_s004
                                 + 0.942511 w_k1_KO_s004
                                 + 0.784652 w_k1_MAT_s004
                                 + 0.97407 w_k1_MSFT_s004
                                 + 0.817072 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.22751 w_k1_ADM_s004
                                 + 1.36841 w_k1_AXP_s004
                                 + 1.55411 w_k1_DIS_s004
                                 + 1.36386 w_k1_FDX_s004
                                 + 1.30756 w_k1_IBM_s004
                                 + 1.26614 w_k1_INTC_s004
                                 + 1.45681 w_k1_KO_s004 + 1.75837 w_k1_MAT_s004
                                 + 1.32764 w_k1_MSFT_s004
                                 + 1.11442 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.08317 w_k1_ADM_s004
                                 + 1.22373 w_k1_AXP_s004
                                 + 0.857388 w_k1_DIS_s004
                                 + 1.13167 w_k1_FDX_s004
                                 + 0.927376 w_k1_IBM_s004
                                 + 1.18114 w_k1_INTC_s004
                                 + 0.930733 w_k1_KO_s004
                                 + 1.37425 w_k1_MAT_s004
                                 + 1.0007 w_k1_MSFT_s004 + 1.01917 w_k1_PG_s004
                                 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.884394 w_k1_ADM_s004
                                 + 0.89777 w_k1_AXP_s004
                                 + 0.731268 w_k1_DIS_s004
                                 + 0.83621 w_k1_FDX_s004
                                 + 1.26128 w_k1_IBM_s004
                                 + 0.863554 w_k1_INTC_s004
                                 + 1.0801 w_k1_KO_s004 + 0.86764 w_k1_MAT_s004
                                 + 0.804519 w_k1_MSFT_s004
                                 + 0.909594 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.23476 w_k1_ADM_s005
                                 + 1.68289 w_k1_AXP_s005 + 1.1686 w_k1_DIS_s005
                                 + 1.2357 w_k1_FDX_s005 + 1.24711 w_k1_IBM_s005
                                 + 0.918364 w_k1_INTC_s005
                                 + 1.0263 w_k1_KO_s005 + 1.44549 w_k1_MAT_s005
                                 + 1.00996 w_k1_MSFT_s005
                                 + 1.03599 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.33119 w_k1_ADM_s005
                                 + 1.01506 w_k1_AXP_s005
                                 + 1.25586 w_k1_DIS_s005
                                 + 1.20753 w_k1_FDX_s005
                                 + 0.885662 w_k1_IBM_s005
                                 + 1.21625 w_k1_INTC_s005
                                 + 1.05336 w_k1_KO_s005
                                 + 0.656287 w_k1_MAT_s005
                                 + 1.22256 w_k1_MSFT_s005
                                 + 0.990319 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.860382 w_k1_ADM_s005
                                 + 0.742581 w_k1_AXP_s005
                                 + 0.854557 w_k1_DIS_s005
                                 + 0.897084 w_k1_FDX_s005
                                 + 0.868653 w_k1_IBM_s005
                                 + 0.958669 w_k1_INTC_s005
                                 + 0.857154 w_k1_KO_s005
                                 + 0.728191 w_k1_MAT_s005
                                 + 0.935052 w_k1_MSFT_s005
                                 + 0.805206 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.747709 w_k1_ADM_s005
                                 + 0.718393 w_k1_AXP_s005
                                 + 0.653027 w_k1_DIS_s005
                                 + 0.801919 w_k1_FDX_s005
                                 + 0.679831 w_k1_IBM_s005
                                 + 0.742557 w_k1_INTC_s005
                                 + 0.645617 w_k1_KO_s005
                                 + 0.814948 w_k1_MAT_s005
                                 + 0.847396 w_k1_MSFT_s005
                                 + 0.885323 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.798852 w_k1_ADM_s005
                                 + 0.784828 w_k1_AXP_s005
                                 + 0.732211 w_k1_DIS_s005
                                 + 0.769564 w_k1_FDX_s005
                                 + 1.29668 w_k1_IBM_s005
                                 + 0.796146 w_k1_INTC_s005
                                 + 0.970005 w_k1_KO_s005
                                 + 0.995288 w_k1_MAT_s005
                                 + 0.86706 w_k1_MSFT_s005
                                 + 0.839824 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.859897 w_k1_ADM_s005
                                 + 0.992419 w_k1_AXP_s005
                                 + 1.05901 w_k1_DIS_s005
                                 + 1.05694 w_k1_FDX_s005
                                 + 1.18732 w_k1_IBM_s005
                                 + 1.01919 w_k1_INTC_s005
                                 + 0.990118 w_k1_KO_s005
                                 + 1.05207 w_k1_MAT_s005
                                 + 1.10233 w_k1_MSFT_s005
                                 + 0.987616 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.41397 w_k1_ADM_s005
                                 + 1.2371 w_k1_AXP_s005 + 1.60415 w_k1_DIS_s005
                                 + 1.20429 w_k1_FDX_s005
                                 + 1.95357 w_k1_IBM_s005
                                 + 0.905951 w_k1_INTC_s005
                                 + 1.44506 w_k1_KO_s005 + 1.80467 w_k1_MAT_s005
                                 + 1.13268 w_k1_MSFT_s005
                                 + 1.00842 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.91484 w_k1_ADM_s005
                                 + 1.03909 w_k1_AXP_s005
                                 + 0.838295 w_k1_DIS_s005
                                 + 1.00201 w_k1_FDX_s005
                                 + 0.822077 w_k1_IBM_s005
                                 + 1.15036 w_k1_INTC_s005
                                 + 0.837149 w_k1_KO_s005
                                 + 0.75278 w_k1_MAT_s005
                                 + 1.00967 w_k1_MSFT_s005
                                 + 0.836473 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.22524 w_k1_ADM_s006
                                 + 1.32552 w_k1_AXP_s006
                                 + 1.74527 w_k1_DIS_s006
                                 + 1.34185 w_k1_FDX_s006
                                 + 1.36942 w_k1_IBM_s006
                                 + 0.998182 w_k1_INTC_s006
                                 + 1.45229 w_k1_KO_s006 + 1.85256 w_k1_MAT_s006
                                 + 1.25528 w_k1_MSFT_s006
                                 + 1.29261 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.06308 w_k1_ADM_s006
                                 + 0.765455 w_k1_AXP_s006
                                 + 1.01722 w_k1_DIS_s006
                                 + 0.867853 w_k1_FDX_s006
                                 + 1.33272 w_k1_IBM_s006
                                 + 0.836941 w_k1_INTC_s006
                                 + 1.11639 w_k1_KO_s006 + 1.48832 w_k1_MAT_s006
                                 + 1.03486 w_k1_MSFT_s006
                                 + 1.03683 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.09 w_k1_ADM_s006
                                 + 0.749306 w_k1_AXP_s006
                                 + 0.680966 w_k1_DIS_s006
                                 + 0.758899 w_k1_FDX_s006
                                 + 0.791243 w_k1_IBM_s006
                                 + 0.684388 w_k1_INTC_s006
                                 + 0.804076 w_k1_KO_s006 + 1.0241 w_k1_MAT_s006
                                 + 0.835715 w_k1_MSFT_s006
                                 + 0.844841 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.16543 w_k1_ADM_s006
                                 + 0.83866 w_k1_AXP_s006
                                 + 0.874887 w_k1_DIS_s006
                                 + 0.966191 w_k1_FDX_s006
                                 + 0.724946 w_k1_IBM_s006
                                 + 0.714897 w_k1_INTC_s006
                                 + 0.963386 w_k1_KO_s006
                                 + 1.32326 w_k1_MAT_s006
                                 + 1.04922 w_k1_MSFT_s006
                                 + 1.07278 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.00304 w_k1_ADM_s006
                                 + 0.970538 w_k1_AXP_s006
                                 + 1.13219 w_k1_DIS_s006
                                 + 0.989695 w_k1_FDX_s006
                                 + 1.00312 w_k1_IBM_s006
                                 + 1.11658 w_k1_INTC_s006
                                 + 1.21411 w_k1_KO_s006 + 1.64388 w_k1_MAT_s006
                                 + 1.23371 w_k1_MSFT_s006
                                 + 1.07033 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.31267 w_k1_ADM_s006
                                 + 0.89607 w_k1_AXP_s006
                                 + 0.935845 w_k1_DIS_s006
                                 + 1.18781 w_k1_FDX_s006
                                 + 0.590414 w_k1_IBM_s006
                                 + 0.763695 w_k1_INTC_s006
                                 + 0.690872 w_k1_KO_s006
                                 + 0.833216 w_k1_MAT_s006
                                 + 0.916435 w_k1_MSFT_s006
                                 + 0.977731 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.56481 w_k1_ADM_s006
                                 + 1.18088 w_k1_AXP_s006
                                 + 1.23594 w_k1_DIS_s006
                                 + 1.25859 w_k1_FDX_s006
                                 + 0.986472 w_k1_IBM_s006
                                 + 0.817358 w_k1_INTC_s006
                                 + 1.01425 w_k1_KO_s006 + 1.52765 w_k1_MAT_s006
                                 + 1.08739 w_k1_MSFT_s006
                                 + 1.18533 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.40173 w_k1_ADM_s006
                                 + 1.71093 w_k1_AXP_s006 + 1.7488 w_k1_DIS_s006
                                 + 1.51982 w_k1_FDX_s006 + 1.5014 w_k1_IBM_s006
                                 + 1.21002 w_k1_INTC_s006
                                 + 1.44813 w_k1_KO_s006 + 2.02434 w_k1_MAT_s006
                                 + 1.51102 w_k1_MSFT_s006
                                 + 1.30599 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.766201 w_k1_ADM_s007
                                 + 0.908016 w_k1_AXP_s007
                                 + 0.985875 w_k1_DIS_s007
                                 + 0.935123 w_k1_FDX_s007 + 1.336 w_k1_IBM_s007
                                 + 1.16984 w_k1_INTC_s007
                                 + 1.26617 w_k1_KO_s007 + 1.28766 w_k1_MAT_s007
                                 + 1.07283 w_k1_MSFT_s007 + 1.0024 w_k1_PG_s007
                                 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.802847 w_k1_ADM_s007
                                 + 0.898445 w_k1_AXP_s007
                                 + 0.639147 w_k1_DIS_s007
                                 + 0.932904 w_k1_FDX_s007
                                 + 0.554847 w_k1_IBM_s007
                                 + 0.943029 w_k1_INTC_s007
                                 + 0.732291 w_k1_KO_s007
                                 + 0.513945 w_k1_MAT_s007
                                 + 0.891083 w_k1_MSFT_s007
                                 + 0.870628 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.93411 w_k1_ADM_s007
                                 + 0.934786 w_k1_AXP_s007
                                 + 0.741918 w_k1_DIS_s007
                                 + 0.934918 w_k1_FDX_s007
                                 + 0.872221 w_k1_IBM_s007
                                 + 0.971599 w_k1_INTC_s007
                                 + 0.937247 w_k1_KO_s007
                                 + 0.75153 w_k1_MAT_s007
                                 + 0.886396 w_k1_MSFT_s007
                                 + 0.905043 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.06656 w_k1_ADM_s007
                                 + 0.977943 w_k1_AXP_s007
                                 + 0.845331 w_k1_DIS_s007
                                 + 1.02547 w_k1_FDX_s007
                                 + 1.05666 w_k1_IBM_s007
                                 + 1.11879 w_k1_INTC_s007
                                 + 1.06653 w_k1_KO_s007
                                 + 0.743106 w_k1_MAT_s007
                                 + 0.928504 w_k1_MSFT_s007
                                 + 1.09175 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.37131 w_k1_ADM_s007
                                 + 1.53858 w_k1_AXP_s007
                                 + 1.68189 w_k1_DIS_s007
                                 + 1.32092 w_k1_FDX_s007
                                 + 1.65427 w_k1_IBM_s007
                                 + 1.03272 w_k1_INTC_s007 + 1.3825 w_k1_KO_s007
                                 + 1.60531 w_k1_MAT_s007
                                 + 1.15544 w_k1_MSFT_s007
                                 + 1.26796 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.965716 w_k1_ADM_s007
                                 + 0.973179 w_k1_AXP_s007
                                 + 1.09536 w_k1_DIS_s007
                                 + 1.10872 w_k1_FDX_s007
                                 + 0.787755 w_k1_IBM_s007
                                 + 1.09908 w_k1_INTC_s007
                                 + 0.989908 w_k1_KO_s007
                                 + 0.636827 w_k1_MAT_s007
                                 + 1.11838 w_k1_MSFT_s007
                                 + 0.926897 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.23905 w_k1_ADM_s007
                                 + 1.28964 w_k1_AXP_s007
                                 + 1.18643 w_k1_DIS_s007
                                 + 1.33425 w_k1_FDX_s007
                                 + 1.06301 w_k1_IBM_s007
                                 + 1.33417 w_k1_INTC_s007
                                 + 1.11754 w_k1_KO_s007
                                 + 0.866514 w_k1_MAT_s007
                                 + 1.14092 w_k1_MSFT_s007
                                 + 0.996547 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.658183 w_k1_ADM_s007
                                 + 0.643933 w_k1_AXP_s007
                                 + 0.587383 w_k1_DIS_s007
                                 + 0.822696 w_k1_FDX_s007
                                 + 0.65255 w_k1_IBM_s007
                                 + 0.780063 w_k1_INTC_s007
                                 + 0.772388 w_k1_KO_s007
                                 + 0.550855 w_k1_MAT_s007
                                 + 0.827451 w_k1_MSFT_s007
                                 + 0.800603 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.87108 w_k1_ADM_s010
                                 + 0.828279 w_k1_AXP_s010
                                 + 0.982875 w_k1_DIS_s010
                                 + 0.963839 w_k1_FDX_s010
                                 + 1.02214 w_k1_IBM_s010
                                 + 0.833225 w_k1_INTC_s010
                                 + 0.91597 w_k1_KO_s010
                                 + 0.865282 w_k1_MAT_s010
                                 + 0.972727 w_k1_MSFT_s010
                                 + 0.921431 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.02064 w_k1_ADM_s010
                                 + 0.865534 w_k1_AXP_s010
                                 + 0.584481 w_k1_DIS_s010
                                 + 0.888084 w_k1_FDX_s010
                                 + 0.702465 w_k1_IBM_s010
                                 + 0.960035 w_k1_INTC_s010
                                 + 0.626411 w_k1_KO_s010
                                 + 0.615161 w_k1_MAT_s010
                                 + 0.769399 w_k1_MSFT_s010
                                 + 0.755168 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.46627 w_k1_ADM_s010
                                 + 1.49603 w_k1_AXP_s010
                                 + 1.49499 w_k1_DIS_s010
                                 + 1.26595 w_k1_FDX_s010
                                 + 1.79022 w_k1_IBM_s010
                                 + 1.14949 w_k1_INTC_s010
                                 + 1.42898 w_k1_KO_s010 + 1.54705 w_k1_MAT_s010
                                 + 1.1539 w_k1_MSFT_s010 + 1.23531 w_k1_PG_s010
                                 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.01408 w_k1_ADM_s010
                                 + 1.12698 w_k1_AXP_s010
                                 + 0.801492 w_k1_DIS_s010
                                 + 0.913467 w_k1_FDX_s010
                                 + 1.01601 w_k1_IBM_s010
                                 + 0.853692 w_k1_INTC_s010
                                 + 0.839729 w_k1_KO_s010
                                 + 0.954227 w_k1_MAT_s010
                                 + 0.814733 w_k1_MSFT_s010
                                 + 0.908887 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.813849 w_k1_ADM_s010
                                 + 0.794777 w_k1_AXP_s010
                                 + 0.829941 w_k1_DIS_s010
                                 + 0.782907 w_k1_FDX_s010
                                 + 1.20791 w_k1_IBM_s010
                                 + 0.954783 w_k1_INTC_s010
                                 + 0.940429 w_k1_KO_s010
                                 + 0.84996 w_k1_MAT_s010
                                 + 0.869661 w_k1_MSFT_s010
                                 + 1.03724 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.772879 w_k1_ADM_s010
                                 + 0.663917 w_k1_AXP_s010
                                 + 0.817579 w_k1_DIS_s010
                                 + 0.844972 w_k1_FDX_s010
                                 + 0.737666 w_k1_IBM_s010
                                 + 0.93582 w_k1_INTC_s010
                                 + 0.880842 w_k1_KO_s010
                                 + 0.593538 w_k1_MAT_s010
                                 + 0.944478 w_k1_MSFT_s010
                                 + 0.999041 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.04465 w_k1_ADM_s010
                                 + 1.38906 w_k1_AXP_s010
                                 + 1.48802 w_k1_DIS_s010
                                 + 1.24926 w_k1_FDX_s010
                                 + 1.20876 w_k1_IBM_s010
                                 + 1.27298 w_k1_INTC_s010
                                 + 1.30777 w_k1_KO_s010
                                 + 0.961891 w_k1_MAT_s010
                                 + 1.21516 w_k1_MSFT_s010
                                 + 1.23989 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.10857 w_k1_ADM_s010
                                 + 1.37481 w_k1_AXP_s010
                                 + 1.06998 w_k1_DIS_s010
                                 + 1.11806 w_k1_FDX_s010
                                 + 0.939725 w_k1_IBM_s010
                                 + 1.21678 w_k1_INTC_s010
                                 + 0.975644 w_k1_KO_s010
                                 + 1.11684 w_k1_MAT_s010
                                 + 1.03194 w_k1_MSFT_s010
                                 + 1.17007 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.704519 w_k1_ADM_s011
                                 + 0.687062 w_k1_AXP_s011
                                 + 0.811956 w_k1_DIS_s011
                                 + 0.801932 w_k1_FDX_s011
                                 + 0.832855 w_k1_IBM_s011
                                 + 0.743197 w_k1_INTC_s011
                                 + 0.708748 w_k1_KO_s011
                                 + 0.83102 w_k1_MAT_s011
                                 + 0.786941 w_k1_MSFT_s011
                                 + 0.898191 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.15622 w_k1_ADM_s011
                                 + 1.33977 w_k1_AXP_s011
                                 + 1.03924 w_k1_DIS_s011
                                 + 0.899784 w_k1_FDX_s011
                                 + 1.74599 w_k1_IBM_s011
                                 + 0.656043 w_k1_INTC_s011
                                 + 1.13428 w_k1_KO_s011 + 2.17634 w_k1_MAT_s011
                                 + 0.873596 w_k1_MSFT_s011
                                 + 1.01683 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.978427 w_k1_ADM_s011
                                 + 0.858407 w_k1_AXP_s011
                                 + 0.925466 w_k1_DIS_s011
                                 + 0.972551 w_k1_FDX_s011
                                 + 0.683438 w_k1_IBM_s011
                                 + 0.874571 w_k1_INTC_s011
                                 + 0.810781 w_k1_KO_s011
                                 + 0.806315 w_k1_MAT_s011
                                 + 0.895072 w_k1_MSFT_s011
                                 + 0.96019 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.96416 w_k1_ADM_s011
                                 + 1.37064 w_k1_AXP_s011
                                 + 1.61014 w_k1_DIS_s011 + 1.3376 w_k1_FDX_s011
                                 + 0.919364 w_k1_IBM_s011
                                 + 1.36414 w_k1_INTC_s011
                                 + 1.05539 w_k1_KO_s011 + 1.06468 w_k1_MAT_s011
                                 + 1.25927 w_k1_MSFT_s011
                                 + 1.27466 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.02421 w_k1_ADM_s011
                                 + 0.97565 w_k1_AXP_s011
                                 + 0.990742 w_k1_DIS_s011 + 1.037 w_k1_FDX_s011
                                 + 0.626136 w_k1_IBM_s011
                                 + 0.84288 w_k1_INTC_s011
                                 + 0.720697 w_k1_KO_s011
                                 + 0.731653 w_k1_MAT_s011
                                 + 0.834838 w_k1_MSFT_s011
                                 + 0.951195 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.672864 w_k1_ADM_s011
                                 + 0.733321 w_k1_AXP_s011
                                 + 0.880642 w_k1_DIS_s011
                                 + 0.724442 w_k1_FDX_s011 + 1.117 w_k1_IBM_s011
                                 + 0.703035 w_k1_INTC_s011
                                 + 0.9708 w_k1_KO_s011 + 1.29977 w_k1_MAT_s011
                                 + 0.824372 w_k1_MSFT_s011
                                 + 0.987117 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.90931 w_k1_ADM_s011
                                 + 1.20753 w_k1_AXP_s011 + 1.4597 w_k1_DIS_s011
                                 + 1.12626 w_k1_FDX_s011
                                 + 1.33103 w_k1_IBM_s011
                                 + 1.02687 w_k1_INTC_s011
                                 + 1.15404 w_k1_KO_s011 + 1.27739 w_k1_MAT_s011
                                 + 1.07058 w_k1_MSFT_s011 + 1.14 w_k1_PG_s011
                                 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.11242 w_k1_ADM_s011
                                 + 1.31018 w_k1_AXP_s011
                                 + 1.87988 w_k1_DIS_s011
                                 + 1.36352 w_k1_FDX_s011 + 0.7251 w_k1_IBM_s011
                                 + 1.02557 w_k1_INTC_s011
                                 + 1.15943 w_k1_KO_s011
                                 + 0.943471 w_k1_MAT_s011
                                 + 1.27229 w_k1_MSFT_s011
                                 + 1.33663 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.986662 w_k1_ADM_s012
                                 + 1.04162 w_k1_AXP_s012
                                 + 1.28241 w_k1_DIS_s012 + 1.1042 w_k1_FDX_s012
                                 + 1.39835 w_k1_IBM_s012
                                 + 1.27238 w_k1_INTC_s012
                                 + 1.00759 w_k1_KO_s012 + 1.26137 w_k1_MAT_s012
                                 + 1.22741 w_k1_MSFT_s012
                                 + 0.995926 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.14707 w_k1_ADM_s012
                                 + 1.37569 w_k1_AXP_s012
                                 + 1.15086 w_k1_DIS_s012
                                 + 1.34289 w_k1_FDX_s012
                                 + 0.901683 w_k1_IBM_s012
                                 + 1.20812 w_k1_INTC_s012
                                 + 0.818895 w_k1_KO_s012
                                 + 1.15024 w_k1_MAT_s012
                                 + 1.07421 w_k1_MSFT_s012
                                 + 0.999669 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.23433 w_k1_ADM_s012
                                 + 1.30828 w_k1_AXP_s012
                                 + 0.785966 w_k1_DIS_s012
                                 + 1.00691 w_k1_FDX_s012
                                 + 1.00928 w_k1_IBM_s012
                                 + 0.9194 w_k1_INTC_s012
                                 + 0.832645 w_k1_KO_s012 + 1.3026 w_k1_MAT_s012
                                 + 0.817252 w_k1_MSFT_s012
                                 + 0.911785 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.982088 w_k1_ADM_s012
                                 + 0.907888 w_k1_AXP_s012
                                 + 0.8666 w_k1_DIS_s012
                                 + 0.935054 w_k1_FDX_s012
                                 + 0.775588 w_k1_IBM_s012
                                 + 0.904832 w_k1_INTC_s012
                                 + 0.68544 w_k1_KO_s012
                                 + 0.883842 w_k1_MAT_s012
                                 + 0.93658 w_k1_MSFT_s012
                                 + 0.857523 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.01067 w_k1_ADM_s012
                                 + 0.6454 w_k1_AXP_s012 + 0.62625 w_k1_DIS_s012
                                 + 0.795223 w_k1_FDX_s012
                                 + 0.728662 w_k1_IBM_s012
                                 + 0.865449 w_k1_INTC_s012
                                 + 0.728378 w_k1_KO_s012
                                 + 0.703716 w_k1_MAT_s012
                                 + 0.82567 w_k1_MSFT_s012
                                 + 0.812131 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.962192 w_k1_ADM_s012
                                 + 0.761864 w_k1_AXP_s012
                                 + 0.777791 w_k1_DIS_s012
                                 + 0.794957 w_k1_FDX_s012
                                 + 1.15985 w_k1_IBM_s012
                                 + 0.777616 w_k1_INTC_s012
                                 + 0.885326 w_k1_KO_s012 + 1.211 w_k1_MAT_s012
                                 + 0.757532 w_k1_MSFT_s012
                                 + 0.922062 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.629696 w_k1_ADM_s012
                                 + 0.626681 w_k1_AXP_s012
                                 + 0.542485 w_k1_DIS_s012
                                 + 0.667183 w_k1_FDX_s012
                                 + 0.602683 w_k1_IBM_s012
                                 + 0.831864 w_k1_INTC_s012
                                 + 0.598971 w_k1_KO_s012
                                 + 0.62217 w_k1_MAT_s012
                                 + 0.719354 w_k1_MSFT_s012
                                 + 0.722989 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.12867 w_k1_ADM_s012
                                 + 0.860049 w_k1_AXP_s012
                                 + 1.29569 w_k1_DIS_s012
                                 + 1.08865 w_k1_FDX_s012
                                 + 1.02369 w_k1_IBM_s012
                                 + 1.04511 w_k1_INTC_s012
                                 + 1.03463 w_k1_KO_s012
                                 + 0.961037 w_k1_MAT_s012
                                 + 1.27862 w_k1_MSFT_s012
                                 + 1.01656 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.27317 w_k1_ADM_s013
                                 + 1.4732 w_k1_AXP_s013 + 1.84453 w_k1_DIS_s013
                                 + 1.35096 w_k1_FDX_s013
                                 + 1.56193 w_k1_IBM_s013
                                 + 1.42652 w_k1_INTC_s013
                                 + 1.60686 w_k1_KO_s013 + 1.68154 w_k1_MAT_s013
                                 + 1.2405 w_k1_MSFT_s013 + 1.2338 w_k1_PG_s013
                                 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.14453 w_k1_ADM_s013
                                 + 1.11583 w_k1_AXP_s013
                                 + 1.02668 w_k1_DIS_s013
                                 + 1.02591 w_k1_FDX_s013
                                 + 1.16327 w_k1_IBM_s013
                                 + 0.891269 w_k1_INTC_s013
                                 + 1.17279 w_k1_KO_s013 + 1.28567 w_k1_MAT_s013
                                 + 0.946486 w_k1_MSFT_s013
                                 + 1.07056 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.83818 w_k1_ADM_s013
                                 + 1.0087 w_k1_AXP_s013 + 1.16816 w_k1_DIS_s013
                                 + 0.984826 w_k1_FDX_s013
                                 + 1.04545 w_k1_IBM_s013
                                 + 0.995666 w_k1_INTC_s013
                                 + 1.15413 w_k1_KO_s013 + 1.27644 w_k1_MAT_s013
                                 + 0.956256 w_k1_MSFT_s013
                                 + 1.19952 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.951464 w_k1_ADM_s013
                                 + 0.952774 w_k1_AXP_s013
                                 + 0.902797 w_k1_DIS_s013
                                 + 0.88025 w_k1_FDX_s013
                                 + 1.36971 w_k1_IBM_s013
                                 + 1.03746 w_k1_INTC_s013
                                 + 1.27853 w_k1_KO_s013 + 1.31299 w_k1_MAT_s013
                                 + 0.831996 w_k1_MSFT_s013
                                 + 1.11973 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.2861 w_k1_ADM_s013
                                 + 1.25745 w_k1_AXP_s013
                                 + 1.27536 w_k1_DIS_s013
                                 + 1.29271 w_k1_FDX_s013
                                 + 1.02048 w_k1_IBM_s013
                                 + 1.01247 w_k1_INTC_s013
                                 + 1.06869 w_k1_KO_s013
                                 + 0.966739 w_k1_MAT_s013
                                 + 0.970722 w_k1_MSFT_s013
                                 + 1.1496 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.970269 w_k1_ADM_s013
                                 + 1.02633 w_k1_AXP_s013
                                 + 0.950737 w_k1_DIS_s013
                                 + 1.04109 w_k1_FDX_s013
                                 + 0.758896 w_k1_IBM_s013
                                 + 0.962519 w_k1_INTC_s013
                                 + 1.02488 w_k1_KO_s013 + 1.23976 w_k1_MAT_s013
                                 + 0.867161 w_k1_MSFT_s013
                                 + 0.942723 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.36722 w_k1_ADM_s013
                                 + 1.18856 w_k1_AXP_s013
                                 + 1.44452 w_k1_DIS_s013
                                 + 1.19275 w_k1_FDX_s013
                                 + 1.15481 w_k1_IBM_s013
                                 + 1.21198 w_k1_INTC_s013
                                 + 1.39593 w_k1_KO_s013 + 1.03327 w_k1_MAT_s013
                                 + 1.1989 w_k1_MSFT_s013 + 1.13931 w_k1_PG_s013
                                 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.70932 w_k1_ADM_s013
                                 + 0.727664 w_k1_AXP_s013
                                 + 0.62846 w_k1_DIS_s013
                                 + 0.801151 w_k1_FDX_s013
                                 + 0.783204 w_k1_IBM_s013
                                 + 0.832291 w_k1_INTC_s013
                                 + 0.785468 w_k1_KO_s013
                                 + 0.833168 w_k1_MAT_s013
                                 + 0.817099 w_k1_MSFT_s013
                                 + 0.951061 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.883797 w_k1_ADM_s014
                                 + 0.789248 w_k1_AXP_s014
                                 + 0.972358 w_k1_DIS_s014
                                 + 0.993404 w_k1_FDX_s014
                                 + 0.719054 w_k1_IBM_s014
                                 + 0.751333 w_k1_INTC_s014
                                 + 0.928571 w_k1_KO_s014
                                 + 0.710472 w_k1_MAT_s014
                                 + 0.936945 w_k1_MSFT_s014
                                 + 0.956595 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.11446 w_k1_ADM_s014
                                 + 1.40721 w_k1_AXP_s014 + 1.5494 w_k1_DIS_s014
                                 + 1.30502 w_k1_FDX_s014
                                 + 1.40888 w_k1_IBM_s014
                                 + 1.2275 w_k1_INTC_s014 + 1.35108 w_k1_KO_s014
                                 + 1.36614 w_k1_MAT_s014
                                 + 1.19866 w_k1_MSFT_s014
                                 + 1.18378 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.780522 w_k1_ADM_s014
                                 + 0.909867 w_k1_AXP_s014
                                 + 0.946059 w_k1_DIS_s014
                                 + 1.05109 w_k1_FDX_s014
                                 + 0.852931 w_k1_IBM_s014
                                 + 1.01378 w_k1_INTC_s014
                                 + 1.00663 w_k1_KO_s014
                                 + 0.755494 w_k1_MAT_s014
                                 + 0.968078 w_k1_MSFT_s014
                                 + 0.936124 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.878755 w_k1_ADM_s014
                                 + 0.881176 w_k1_AXP_s014
                                 + 0.77209 w_k1_DIS_s014
                                 + 1.03911 w_k1_FDX_s014
                                 + 0.580645 w_k1_IBM_s014
                                 + 1.06787 w_k1_INTC_s014
                                 + 0.736618 w_k1_KO_s014
                                 + 0.523898 w_k1_MAT_s014
                                 + 0.996403 w_k1_MSFT_s014
                                 + 0.874209 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.774634 w_k1_ADM_s014
                                 + 1.02946 w_k1_AXP_s014
                                 + 1.00713 w_k1_DIS_s014
                                 + 1.03302 w_k1_FDX_s014
                                 + 1.12308 w_k1_IBM_s014 + 1.046 w_k1_INTC_s014
                                 + 1.08759 w_k1_KO_s014
                                 + 0.965917 w_k1_MAT_s014
                                 + 0.89746 w_k1_MSFT_s014 + 1.0519 w_k1_PG_s014
                                 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.854886 w_k1_ADM_s014
                                 + 1.27336 w_k1_AXP_s014
                                 + 1.18546 w_k1_DIS_s014
                                 + 1.00275 w_k1_FDX_s014
                                 + 1.78185 w_k1_IBM_s014
                                 + 0.908187 w_k1_INTC_s014
                                 + 1.40787 w_k1_KO_s014 + 1.91613 w_k1_MAT_s014
                                 + 0.968449 w_k1_MSFT_s014
                                 + 1.15954 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.580794 w_k1_ADM_s014
                                 + 0.56139 w_k1_AXP_s014
                                 + 0.50651 w_k1_DIS_s014
                                 + 0.709439 w_k1_FDX_s014
                                 + 0.883976 w_k1_IBM_s014
                                 + 0.853132 w_k1_INTC_s014
                                 + 0.766089 w_k1_KO_s014
                                 + 0.662808 w_k1_MAT_s014
                                 + 0.743927 w_k1_MSFT_s014
                                 + 0.787337 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.12879 w_k1_ADM_s014
                                 + 1.25111 w_k1_AXP_s014
                                 + 1.37396 w_k1_DIS_s014
                                 + 1.48509 w_k1_FDX_s014
                                 + 0.964514 w_k1_IBM_s014
                                 + 1.03925 w_k1_INTC_s014 + 1.0904 w_k1_KO_s014
                                 + 1.04971 w_k1_MAT_s014
                                 + 1.1484 w_k1_MSFT_s014 + 1.26585 w_k1_PG_s014
                                 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.664106 w_k1_ADM_s015
                                 + 0.57343 w_k1_AXP_s015
                                 + 0.685385 w_k1_DIS_s015
                                 + 0.699678 w_k1_FDX_s015
                                 + 1.11723 w_k1_IBM_s015
                                 + 0.798375 w_k1_INTC_s015
                                 + 0.936729 w_k1_KO_s015
                                 + 0.773405 w_k1_MAT_s015
                                 + 0.806997 w_k1_MSFT_s015
                                 + 0.927257 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.20733 w_k1_ADM_s015
                                 + 1.38992 w_k1_AXP_s015
                                 + 1.40243 w_k1_DIS_s015
                                 + 1.34601 w_k1_FDX_s015
                                 + 1.12997 w_k1_IBM_s015
                                 + 1.31966 w_k1_INTC_s015
                                 + 1.19444 w_k1_KO_s015
                                 + 0.797833 w_k1_MAT_s015
                                 + 1.22216 w_k1_MSFT_s015
                                 + 1.12376 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.01022 w_k1_ADM_s015
                                 + 1.05755 w_k1_AXP_s015
                                 + 1.25175 w_k1_DIS_s015
                                 + 1.08548 w_k1_FDX_s015
                                 + 1.14028 w_k1_IBM_s015
                                 + 1.1701 w_k1_INTC_s015 + 1.21836 w_k1_KO_s015
                                 + 1.0602 w_k1_MAT_s015
                                 + 1.28219 w_k1_MSFT_s015
                                 + 1.16439 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.779604 w_k1_ADM_s015
                                 + 0.856635 w_k1_AXP_s015
                                 + 1.03234 w_k1_DIS_s015
                                 + 0.86455 w_k1_FDX_s015
                                 + 1.42995 w_k1_IBM_s015
                                 + 0.867433 w_k1_INTC_s015 + 1.17 w_k1_KO_s015
                                 + 1.16824 w_k1_MAT_s015
                                 + 0.914919 w_k1_MSFT_s015
                                 + 1.06313 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.34939 w_k1_ADM_s015
                                 + 1.68738 w_k1_AXP_s015
                                 + 1.79304 w_k1_DIS_s015
                                 + 1.36378 w_k1_FDX_s015
                                 + 1.56938 w_k1_IBM_s015
                                 + 1.30627 w_k1_INTC_s015
                                 + 1.50623 w_k1_KO_s015 + 1.50536 w_k1_MAT_s015
                                 + 1.35105 w_k1_MSFT_s015
                                 + 1.23169 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.12528 w_k1_ADM_s015
                                 + 1.16999 w_k1_AXP_s015
                                 + 1.14938 w_k1_DIS_s015
                                 + 1.31622 w_k1_FDX_s015
                                 + 0.770298 w_k1_IBM_s015
                                 + 0.937918 w_k1_INTC_s015
                                 + 0.984861 w_k1_KO_s015
                                 + 0.79639 w_k1_MAT_s015
                                 + 1.10701 w_k1_MSFT_s015
                                 + 1.06402 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.12957 w_k1_ADM_s015
                                 + 1.00281 w_k1_AXP_s015
                                 + 0.923373 w_k1_DIS_s015
                                 + 1.03978 w_k1_FDX_s015
                                 + 0.935836 w_k1_IBM_s015
                                 + 0.970214 w_k1_INTC_s015
                                 + 0.927616 w_k1_KO_s015
                                 + 0.903831 w_k1_MAT_s015
                                 + 1.10146 w_k1_MSFT_s015
                                 + 1.02567 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.671535 w_k1_ADM_s015
                                 + 0.78167 w_k1_AXP_s015
                                 + 0.762913 w_k1_DIS_s015
                                 + 0.876657 w_k1_FDX_s015
                                 + 0.792153 w_k1_IBM_s015
                                 + 0.840381 w_k1_INTC_s015
                                 + 0.765865 w_k1_KO_s015
                                 + 0.814387 w_k1_MAT_s015
                                 + 0.835113 w_k1_MSFT_s015
                                 + 0.908927 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.08944 w_k1_ADM_s016
                                 + 1.09435 w_k1_AXP_s016 + 1.2178 w_k1_DIS_s016
                                 + 1.04795 w_k1_FDX_s016
                                 + 1.12314 w_k1_IBM_s016
                                 + 1.02505 w_k1_INTC_s016
                                 + 1.29401 w_k1_KO_s016 + 1.21462 w_k1_MAT_s016
                                 + 1.07225 w_k1_MSFT_s016
                                 + 1.14483 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.50388 w_k1_ADM_s016
                                 + 1.53533 w_k1_AXP_s016
                                 + 1.89621 w_k1_DIS_s016
                                 + 1.47341 w_k1_FDX_s016 + 1.2869 w_k1_IBM_s016
                                 + 1.3609 w_k1_INTC_s016 + 1.34713 w_k1_KO_s016
                                 + 1.35122 w_k1_MAT_s016
                                 + 1.37204 w_k1_MSFT_s016
                                 + 1.25922 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.16025 w_k1_ADM_s016
                                 + 1.08133 w_k1_AXP_s016
                                 + 0.970765 w_k1_DIS_s016
                                 + 1.00768 w_k1_FDX_s016
                                 + 1.30454 w_k1_IBM_s016
                                 + 0.809058 w_k1_INTC_s016
                                 + 0.952701 w_k1_KO_s016
                                 + 1.25096 w_k1_MAT_s016
                                 + 0.933596 w_k1_MSFT_s016
                                 + 0.96458 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.11231 w_k1_ADM_s016
                                 + 1.41256 w_k1_AXP_s016
                                 + 1.26772 w_k1_DIS_s016
                                 + 1.14426 w_k1_FDX_s016
                                 + 1.39582 w_k1_IBM_s016
                                 + 1.3362 w_k1_INTC_s016 + 1.38986 w_k1_KO_s016
                                 + 1.80839 w_k1_MAT_s016
                                 + 1.14582 w_k1_MSFT_s016
                                 + 1.14403 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.40641 w_k1_ADM_s016
                                 + 1.11226 w_k1_AXP_s016
                                 + 1.24407 w_k1_DIS_s016
                                 + 1.26645 w_k1_FDX_s016
                                 + 0.974677 w_k1_IBM_s016
                                 + 1.10933 w_k1_INTC_s016
                                 + 1.16512 w_k1_KO_s016
                                 + 0.965094 w_k1_MAT_s016
                                 + 1.15869 w_k1_MSFT_s016
                                 + 1.03468 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.35595 w_k1_ADM_s016
                                 + 0.948693 w_k1_AXP_s016
                                 + 0.957606 w_k1_DIS_s016
                                 + 1.17751 w_k1_FDX_s016
                                 + 0.74773 w_k1_IBM_s016
                                 + 1.11677 w_k1_INTC_s016
                                 + 0.965227 w_k1_KO_s016
                                 + 0.800514 w_k1_MAT_s016
                                 + 1.06671 w_k1_MSFT_s016 + 1.0617 w_k1_PG_s016
                                 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.10518 w_k1_ADM_s016
                                 + 0.929318 w_k1_AXP_s016
                                 + 0.849709 w_k1_DIS_s016
                                 + 0.907737 w_k1_FDX_s016
                                 + 0.931465 w_k1_IBM_s016
                                 + 0.740987 w_k1_INTC_s016
                                 + 0.843228 w_k1_KO_s016
                                 + 0.965337 w_k1_MAT_s016
                                 + 0.81736 w_k1_MSFT_s016
                                 + 0.872847 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.744154 w_k1_ADM_s016
                                 + 0.699626 w_k1_AXP_s016
                                 + 0.540703 w_k1_DIS_s016
                                 + 0.707521 w_k1_FDX_s016
                                 + 0.711299 w_k1_IBM_s016
                                 + 0.803923 w_k1_INTC_s016
                                 + 0.664573 w_k1_KO_s016
                                 + 0.672652 w_k1_MAT_s016
                                 + 0.708482 w_k1_MSFT_s016
                                 + 0.806343 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.12094 w_k1_ADM_s017
                                 + 1.37355 w_k1_AXP_s017
                                 + 1.79815 w_k1_DIS_s017
                                 + 1.38142 w_k1_FDX_s017
                                 + 1.25454 w_k1_IBM_s017
                                 + 1.06777 w_k1_INTC_s017
                                 + 1.69932 w_k1_KO_s017 + 1.75305 w_k1_MAT_s017
                                 + 1.36685 w_k1_MSFT_s017
                                 + 1.35255 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.792177 w_k1_ADM_s017
                                 + 0.859444 w_k1_AXP_s017
                                 + 0.911542 w_k1_DIS_s017
                                 + 0.862916 w_k1_FDX_s017
                                 + 1.51281 w_k1_IBM_s017
                                 + 0.812731 w_k1_INTC_s017
                                 + 1.29866 w_k1_KO_s017 + 1.60815 w_k1_MAT_s017
                                 + 1.04172 w_k1_MSFT_s017
                                 + 1.01119 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.30357 w_k1_ADM_s017
                                 + 1.16717 w_k1_AXP_s017
                                 + 1.28208 w_k1_DIS_s017
                                 + 1.24366 w_k1_FDX_s017
                                 + 1.13047 w_k1_IBM_s017
                                 + 1.10124 w_k1_INTC_s017
                                 + 1.31155 w_k1_KO_s017 + 1.73732 w_k1_MAT_s017
                                 + 1.31221 w_k1_MSFT_s017
                                 + 1.16953 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.28046 w_k1_ADM_s017
                                 + 1.53375 w_k1_AXP_s017
                                 + 1.17702 w_k1_DIS_s017
                                 + 1.39418 w_k1_FDX_s017
                                 + 0.911342 w_k1_IBM_s017
                                 + 1.11572 w_k1_INTC_s017
                                 + 1.00514 w_k1_KO_s017
                                 + 0.815735 w_k1_MAT_s017
                                 + 1.14426 w_k1_MSFT_s017
                                 + 1.06278 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.88791 w_k1_ADM_s017
                                 + 1.00772 w_k1_AXP_s017
                                 + 0.756514 w_k1_DIS_s017
                                 + 0.920885 w_k1_FDX_s017
                                 + 1.0645 w_k1_IBM_s017
                                 + 0.981219 w_k1_INTC_s017
                                 + 1.12006 w_k1_KO_s017 + 1.19362 w_k1_MAT_s017
                                 + 0.839883 w_k1_MSFT_s017
                                 + 0.908864 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.99114 w_k1_ADM_s017
                                 + 0.924684 w_k1_AXP_s017
                                 + 1.09815 w_k1_DIS_s017
                                 + 0.959433 w_k1_FDX_s017
                                 + 1.04339 w_k1_IBM_s017
                                 + 0.951832 w_k1_INTC_s017
                                 + 1.24262 w_k1_KO_s017 + 1.22516 w_k1_MAT_s017
                                 + 1.12763 w_k1_MSFT_s017
                                 + 1.02317 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.665484 w_k1_ADM_s017
                                 + 0.62224 w_k1_AXP_s017
                                 + 0.539132 w_k1_DIS_s017
                                 + 0.655263 w_k1_FDX_s017
                                 + 0.707986 w_k1_IBM_s017
                                 + 0.772331 w_k1_INTC_s017
                                 + 0.763911 w_k1_KO_s017
                                 + 0.854737 w_k1_MAT_s017
                                 + 0.759718 w_k1_MSFT_s017
                                 + 0.777318 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.07509 w_k1_ADM_s017
                                 + 0.850551 w_k1_AXP_s017
                                 + 0.852306 w_k1_DIS_s017
                                 + 1.01619 w_k1_FDX_s017
                                 + 0.668664 w_k1_IBM_s017
                                 + 0.805384 w_k1_INTC_s017
                                 + 0.915324 w_k1_KO_s017
                                 + 0.919309 w_k1_MAT_s017
                                 + 0.92075 w_k1_MSFT_s017
                                 + 0.919056 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.694762 w_k1_ADM_s100
                                 + 0.948012 w_k1_AXP_s100
                                 + 0.903027 w_k1_DIS_s100
                                 + 0.865961 w_k1_FDX_s100
                                 + 1.49407 w_k1_IBM_s100
                                 + 0.765125 w_k1_INTC_s100
                                 + 1.24274 w_k1_KO_s100 + 1.42736 w_k1_MAT_s100
                                 + 0.794504 w_k1_MSFT_s100
                                 + 0.958941 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.89675 w_k1_ADM_s100
                                 + 0.690886 w_k1_AXP_s100
                                 + 0.812725 w_k1_DIS_s100
                                 + 0.979364 w_k1_FDX_s100
                                 + 0.564475 w_k1_IBM_s100
                                 + 0.698162 w_k1_INTC_s100
                                 + 0.692224 w_k1_KO_s100
                                 + 0.467398 w_k1_MAT_s100
                                 + 0.825694 w_k1_MSFT_s100
                                 + 0.773911 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.899287 w_k1_ADM_s100
                                 + 1.07865 w_k1_AXP_s100
                                 + 0.836265 w_k1_DIS_s100
                                 + 0.997914 w_k1_FDX_s100
                                 + 0.892926 w_k1_IBM_s100
                                 + 0.917848 w_k1_INTC_s100
                                 + 0.958362 w_k1_KO_s100
                                 + 0.755421 w_k1_MAT_s100
                                 + 0.859611 w_k1_MSFT_s100
                                 + 0.851655 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.825737 w_k1_ADM_s100
                                 + 0.92401 w_k1_AXP_s100
                                 + 1.11231 w_k1_DIS_s100
                                 + 1.00144 w_k1_FDX_s100 + 1.0098 w_k1_IBM_s100
                                 + 0.902569 w_k1_INTC_s100
                                 + 1.10576 w_k1_KO_s100
                                 + 0.714188 w_k1_MAT_s100
                                 + 1.04948 w_k1_MSFT_s100
                                 + 1.03514 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.848147 w_k1_ADM_s100
                                 + 0.735111 w_k1_AXP_s100
                                 + 0.699914 w_k1_DIS_s100
                                 + 0.836036 w_k1_FDX_s100
                                 + 0.776175 w_k1_IBM_s100
                                 + 0.915118 w_k1_INTC_s100
                                 + 0.785011 w_k1_KO_s100
                                 + 0.400749 w_k1_MAT_s100
                                 + 0.777499 w_k1_MSFT_s100
                                 + 0.815767 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.13682 w_k1_ADM_s100
                                 + 1.44468 w_k1_AXP_s100
                                 + 1.48344 w_k1_DIS_s100
                                 + 1.33486 w_k1_FDX_s100
                                 + 1.22234 w_k1_IBM_s100
                                 + 1.08208 w_k1_INTC_s100
                                 + 1.31247 w_k1_KO_s100 + 1.44907 w_k1_MAT_s100
                                 + 1.18771 w_k1_MSFT_s100
                                 + 1.10904 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.959568 w_k1_ADM_s100
                                 + 0.974583 w_k1_AXP_s100
                                 + 1.55584 w_k1_DIS_s100
                                 + 1.15046 w_k1_FDX_s100
                                 + 1.77366 w_k1_IBM_s100
                                 + 1.09666 w_k1_INTC_s100
                                 + 1.57877 w_k1_KO_s100 + 1.42501 w_k1_MAT_s100
                                 + 1.22283 w_k1_MSFT_s100
                                 + 1.11034 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.824869 w_k1_ADM_s100
                                 + 0.597143 w_k1_AXP_s100
                                 + 0.686935 w_k1_DIS_s100
                                 + 0.730545 w_k1_FDX_s100
                                 + 0.860674 w_k1_IBM_s100
                                 + 0.674431 w_k1_INTC_s100
                                 + 0.933574 w_k1_KO_s100
                                 + 0.831404 w_k1_MAT_s100
                                 + 0.778261 w_k1_MSFT_s100
                                 + 0.858385 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.30533 w_k1_ADM_s101
                                 + 0.818065 w_k1_AXP_s101
                                 + 0.850536 w_k1_DIS_s101
                                 + 1.04988 w_k1_FDX_s101
                                 + 0.800385 w_k1_IBM_s101
                                 + 0.993336 w_k1_INTC_s101
                                 + 0.945441 w_k1_KO_s101
                                 + 0.783037 w_k1_MAT_s101
                                 + 1.03707 w_k1_MSFT_s101
                                 + 0.894679 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.28678 w_k1_ADM_s101
                                 + 0.560401 w_k1_AXP_s101
                                 + 0.795771 w_k1_DIS_s101
                                 + 0.772473 w_k1_FDX_s101
                                 + 0.80879 w_k1_IBM_s101
                                 + 0.800641 w_k1_INTC_s101
                                 + 0.986598 w_k1_KO_s101
                                 + 0.533695 w_k1_MAT_s101
                                 + 1.00453 w_k1_MSFT_s101
                                 + 0.954036 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.68561 w_k1_ADM_s101
                                 + 1.43558 w_k1_AXP_s101
                                 + 1.61434 w_k1_DIS_s101
                                 + 1.39557 w_k1_FDX_s101
                                 + 1.34024 w_k1_IBM_s101
                                 + 1.31438 w_k1_INTC_s101 + 1.5658 w_k1_KO_s101
                                 + 1.52157 w_k1_MAT_s101
                                 + 1.33183 w_k1_MSFT_s101
                                 + 1.44577 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.09836 w_k1_ADM_s101
                                 + 1.10138 w_k1_AXP_s101
                                 + 1.01034 w_k1_DIS_s101
                                 + 1.10926 w_k1_FDX_s101
                                 + 1.01975 w_k1_IBM_s101
                                 + 1.03133 w_k1_INTC_s101
                                 + 1.04915 w_k1_KO_s101 + 1.21915 w_k1_MAT_s101
                                 + 1.06258 w_k1_MSFT_s101
                                 + 1.15799 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.888513 w_k1_ADM_s101
                                 + 0.621324 w_k1_AXP_s101
                                 + 0.55472 w_k1_DIS_s101
                                 + 0.724155 w_k1_FDX_s101
                                 + 0.64423 w_k1_IBM_s101
                                 + 0.655467 w_k1_INTC_s101
                                 + 0.787302 w_k1_KO_s101
                                 + 0.629679 w_k1_MAT_s101
                                 + 0.656443 w_k1_MSFT_s101
                                 + 0.820205 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.52253 w_k1_ADM_s101
                                 + 1.35818 w_k1_AXP_s101
                                 + 1.19902 w_k1_DIS_s101
                                 + 1.32467 w_k1_FDX_s101
                                 + 0.915338 w_k1_IBM_s101
                                 + 1.26146 w_k1_INTC_s101
                                 + 1.15424 w_k1_KO_s101 + 1.01892 w_k1_MAT_s101
                                 + 1.27092 w_k1_MSFT_s101 + 1.1537 w_k1_PG_s101
                                 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.41716 w_k1_ADM_s101
                                 + 0.999375 w_k1_AXP_s101
                                 + 1.13745 w_k1_DIS_s101
                                 + 1.02277 w_k1_FDX_s101
                                 + 1.40825 w_k1_IBM_s101
                                 + 1.35371 w_k1_INTC_s101
                                 + 1.57118 w_k1_KO_s101 + 1.2421 w_k1_MAT_s101
                                 + 1.26743 w_k1_MSFT_s101
                                 + 1.17809 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.988971 w_k1_ADM_s101
                                 + 0.832402 w_k1_AXP_s101
                                 + 0.717582 w_k1_DIS_s101
                                 + 0.750278 w_k1_FDX_s101
                                 + 1.00439 w_k1_IBM_s101
                                 + 1.0082 w_k1_INTC_s101 + 1.10121 w_k1_KO_s101
                                 + 0.954762 w_k1_MAT_s101
                                 + 0.795245 w_k1_MSFT_s101
                                 + 0.926848 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.954956 w_k1_ADM_s102
                                 + 1.05986 w_k1_AXP_s102
                                 + 1.11827 w_k1_DIS_s102
                                 + 1.05795 w_k1_FDX_s102
                                 + 1.04226 w_k1_IBM_s102
                                 + 1.00138 w_k1_INTC_s102
                                 + 1.16116 w_k1_KO_s102 + 1.15924 w_k1_MAT_s102
                                 + 1.05308 w_k1_MSFT_s102 + 1.0949 w_k1_PG_s102
                                 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.746138 w_k1_ADM_s102
                                 + 0.655552 w_k1_AXP_s102
                                 + 0.634304 w_k1_DIS_s102
                                 + 0.824602 w_k1_FDX_s102
                                 + 0.552504 w_k1_IBM_s102
                                 + 0.898317 w_k1_INTC_s102
                                 + 0.721736 w_k1_KO_s102
                                 + 0.782172 w_k1_MAT_s102
                                 + 0.847788 w_k1_MSFT_s102
                                 + 0.834875 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.977193 w_k1_ADM_s102
                                 + 0.970958 w_k1_AXP_s102
                                 + 1.03055 w_k1_DIS_s102
                                 + 0.85785 w_k1_FDX_s102
                                 + 1.31026 w_k1_IBM_s102
                                 + 0.876328 w_k1_INTC_s102
                                 + 1.3102 w_k1_KO_s102 + 1.49361 w_k1_MAT_s102
                                 + 0.925832 w_k1_MSFT_s102
                                 + 1.11754 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.900186 w_k1_ADM_s102
                                 + 0.854185 w_k1_AXP_s102
                                 + 0.786847 w_k1_DIS_s102
                                 + 0.741567 w_k1_FDX_s102
                                 + 1.1633 w_k1_IBM_s102
                                 + 0.867664 w_k1_INTC_s102
                                 + 1.03849 w_k1_KO_s102 + 1.16329 w_k1_MAT_s102
                                 + 0.86576 w_k1_MSFT_s102
                                 + 0.860229 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.40063 w_k1_ADM_s102
                                 + 1.57389 w_k1_AXP_s102
                                 + 1.71248 w_k1_DIS_s102
                                 + 1.34483 w_k1_FDX_s102
                                 + 1.27638 w_k1_IBM_s102
                                 + 1.26201 w_k1_INTC_s102
                                 + 1.53002 w_k1_KO_s102 + 2.04946 w_k1_MAT_s102
                                 + 1.24423 w_k1_MSFT_s102
                                 + 1.41392 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.07628 w_k1_ADM_s102
                                 + 1.07808 w_k1_AXP_s102
                                 + 1.19257 w_k1_DIS_s102
                                 + 1.12818 w_k1_FDX_s102
                                 + 0.744867 w_k1_IBM_s102
                                 + 1.09007 w_k1_INTC_s102 + 1.0348 w_k1_KO_s102
                                 + 0.80554 w_k1_MAT_s102
                                 + 1.13385 w_k1_MSFT_s102
                                 + 0.986646 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.00031 w_k1_ADM_s102
                                 + 0.8011 w_k1_AXP_s102
                                 + 0.857359 w_k1_DIS_s102
                                 + 0.84187 w_k1_FDX_s102
                                 + 0.814878 w_k1_IBM_s102
                                 + 0.823696 w_k1_INTC_s102
                                 + 0.996187 w_k1_KO_s102
                                 + 0.963391 w_k1_MAT_s102
                                 + 0.90001 w_k1_MSFT_s102
                                 + 0.966734 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.23891 w_k1_ADM_s102
                                 + 1.08916 w_k1_AXP_s102
                                 + 1.30919 w_k1_DIS_s102
                                 + 1.03808 w_k1_FDX_s102
                                 + 1.61828 w_k1_IBM_s102
                                 + 1.17102 w_k1_INTC_s102
                                 + 1.52066 w_k1_KO_s102 + 1.71765 w_k1_MAT_s102
                                 + 1.05268 w_k1_MSFT_s102
                                 + 1.27335 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.715615 w_k1_ADM_s103
                                 + 0.655661 w_k1_AXP_s103
                                 + 0.68296 w_k1_DIS_s103
                                 + 0.898837 w_k1_FDX_s103
                                 + 0.628075 w_k1_IBM_s103
                                 + 0.864961 w_k1_INTC_s103
                                 + 0.784673 w_k1_KO_s103
                                 + 0.533672 w_k1_MAT_s103
                                 + 0.847665 w_k1_MSFT_s103
                                 + 0.88445 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.10768 w_k1_ADM_s103
                                 + 1.24317 w_k1_AXP_s103
                                 + 1.29825 w_k1_DIS_s103
                                 + 1.29683 w_k1_FDX_s103
                                 + 0.937676 w_k1_IBM_s103
                                 + 1.0536 w_k1_INTC_s103 + 1.1583 w_k1_KO_s103
                                 + 1.1094 w_k1_MAT_s103
                                 + 1.11886 w_k1_MSFT_s103
                                 + 1.04469 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.12069 w_k1_ADM_s103
                                 + 1.48741 w_k1_AXP_s103 + 1.3937 w_k1_DIS_s103
                                 + 1.23374 w_k1_FDX_s103
                                 + 1.92408 w_k1_IBM_s103
                                 + 0.99085 w_k1_INTC_s103
                                 + 1.43128 w_k1_KO_s103 + 2.21237 w_k1_MAT_s103
                                 + 1.03307 w_k1_MSFT_s103
                                 + 1.19126 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.871819 w_k1_ADM_s103
                                 + 0.617097 w_k1_AXP_s103
                                 + 0.718715 w_k1_DIS_s103
                                 + 0.864045 w_k1_FDX_s103
                                 + 0.568094 w_k1_IBM_s103
                                 + 0.772849 w_k1_INTC_s103
                                 + 0.747742 w_k1_KO_s103
                                 + 0.564892 w_k1_MAT_s103
                                 + 0.865237 w_k1_MSFT_s103
                                 + 0.821317 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.819123 w_k1_ADM_s103
                                 + 0.802367 w_k1_AXP_s103
                                 + 0.73765 w_k1_DIS_s103
                                 + 0.88718 w_k1_FDX_s103
                                 + 0.874388 w_k1_IBM_s103
                                 + 0.71783 w_k1_INTC_s103
                                 + 0.80563 w_k1_KO_s103 + 0.83896 w_k1_MAT_s103
                                 + 0.840142 w_k1_MSFT_s103
                                 + 0.8205 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.791353 w_k1_ADM_s103
                                 + 0.904897 w_k1_AXP_s103
                                 + 1.00342 w_k1_DIS_s103
                                 + 0.924572 w_k1_FDX_s103
                                 + 1.07983 w_k1_IBM_s103
                                 + 0.952927 w_k1_INTC_s103
                                 + 1.0977 w_k1_KO_s103 + 1.27767 w_k1_MAT_s103
                                 + 0.919996 w_k1_MSFT_s103
                                 + 0.965338 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.927071 w_k1_ADM_s103
                                 + 0.878872 w_k1_AXP_s103
                                 + 1.05349 w_k1_DIS_s103
                                 + 1.11255 w_k1_FDX_s103
                                 + 0.809828 w_k1_IBM_s103
                                 + 0.84934 w_k1_INTC_s103
                                 + 0.87558 w_k1_KO_s103
                                 + 0.930902 w_k1_MAT_s103
                                 + 0.947334 w_k1_MSFT_s103
                                 + 0.981135 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.12855 w_k1_ADM_s103
                                 + 1.42034 w_k1_AXP_s103
                                 + 1.24403 w_k1_DIS_s103
                                 + 1.22397 w_k1_FDX_s103
                                 + 1.38683 w_k1_IBM_s103
                                 + 1.00993 w_k1_INTC_s103
                                 + 1.26598 w_k1_KO_s103 + 1.60049 w_k1_MAT_s103
                                 + 1.07675 w_k1_MSFT_s103
                                 + 1.21781 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.899318 w_k1_ADM_s104
                                 + 0.722873 w_k1_AXP_s104
                                 + 0.769915 w_k1_DIS_s104
                                 + 0.881087 w_k1_FDX_s104
                                 + 0.75794 w_k1_IBM_s104
                                 + 0.928962 w_k1_INTC_s104
                                 + 0.891883 w_k1_KO_s104
                                 + 0.831874 w_k1_MAT_s104
                                 + 0.882071 w_k1_MSFT_s104
                                 + 0.81929 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.10822 w_k1_ADM_s104
                                 + 1.23483 w_k1_AXP_s104
                                 + 1.18407 w_k1_DIS_s104
                                 + 1.13734 w_k1_FDX_s104
                                 + 1.11677 w_k1_IBM_s104
                                 + 0.859979 w_k1_INTC_s104
                                 + 1.13469 w_k1_KO_s104 + 1.71519 w_k1_MAT_s104
                                 + 1.00755 w_k1_MSFT_s104
                                 + 1.08396 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.00124 w_k1_ADM_s104
                                 + 1.14679 w_k1_AXP_s104 + 1.5888 w_k1_DIS_s104
                                 + 1.14035 w_k1_FDX_s104
                                 + 1.17783 w_k1_IBM_s104
                                 + 1.00819 w_k1_INTC_s104
                                 + 1.39959 w_k1_KO_s104 + 1.32142 w_k1_MAT_s104
                                 + 1.18557 w_k1_MSFT_s104
                                 + 1.06727 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.993627 w_k1_ADM_s104
                                 + 0.942998 w_k1_AXP_s104
                                 + 0.770878 w_k1_DIS_s104
                                 + 0.834506 w_k1_FDX_s104
                                 + 1.10868 w_k1_IBM_s104
                                 + 0.839642 w_k1_INTC_s104
                                 + 1.02653 w_k1_KO_s104 + 1.19811 w_k1_MAT_s104
                                 + 0.77867 w_k1_MSFT_s104
                                 + 0.872189 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.15214 w_k1_ADM_s104
                                 + 1.49628 w_k1_AXP_s104
                                 + 1.47996 w_k1_DIS_s104
                                 + 1.20575 w_k1_FDX_s104
                                 + 1.48961 w_k1_IBM_s104
                                 + 0.937482 w_k1_INTC_s104
                                 + 1.49465 w_k1_KO_s104 + 1.80769 w_k1_MAT_s104
                                 + 1.09622 w_k1_MSFT_s104
                                 + 1.11582 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.627333 w_k1_ADM_s104
                                 + 0.544072 w_k1_AXP_s104
                                 + 0.477174 w_k1_DIS_s104
                                 + 0.610641 w_k1_FDX_s104
                                 + 0.72447 w_k1_IBM_s104
                                 + 0.62412 w_k1_INTC_s104
                                 + 0.739973 w_k1_KO_s104
                                 + 0.773648 w_k1_MAT_s104
                                 + 0.663385 w_k1_MSFT_s104
                                 + 0.733703 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.46028 w_k1_ADM_s104
                                 + 1.53882 w_k1_AXP_s104
                                 + 1.52924 w_k1_DIS_s104
                                 + 1.45066 w_k1_FDX_s104
                                 + 0.879649 w_k1_IBM_s104
                                 + 1.1973 w_k1_INTC_s104 + 1.20093 w_k1_KO_s104
                                 + 1.04974 w_k1_MAT_s104
                                 + 1.2967 w_k1_MSFT_s104 + 1.24003 w_k1_PG_s104
                                 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.00226 w_k1_ADM_s104
                                 + 1.16162 w_k1_AXP_s104
                                 + 1.15541 w_k1_DIS_s104
                                 + 1.13507 w_k1_FDX_s104
                                 + 0.815431 w_k1_IBM_s104
                                 + 1.05795 w_k1_INTC_s104
                                 + 0.920794 w_k1_KO_s104
                                 + 0.876996 w_k1_MAT_s104
                                 + 0.926133 w_k1_MSFT_s104
                                 + 0.975483 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.808142 w_k1_ADM_s105
                                 + 1.03684 w_k1_AXP_s105 + 1.0978 w_k1_DIS_s105
                                 + 0.977725 w_k1_FDX_s105
                                 + 1.45277 w_k1_IBM_s105
                                 + 0.946575 w_k1_INTC_s105
                                 + 1.18412 w_k1_KO_s105 + 1.38457 w_k1_MAT_s105
                                 + 1.04266 w_k1_MSFT_s105
                                 + 1.05405 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.11117 w_k1_ADM_s105
                                 + 1.2074 w_k1_AXP_s105
                                 + 0.972063 w_k1_DIS_s105
                                 + 1.16868 w_k1_FDX_s105
                                 + 1.44869 w_k1_IBM_s105
                                 + 1.09874 w_k1_INTC_s105
                                 + 1.20431 w_k1_KO_s105 + 1.40149 w_k1_MAT_s105
                                 + 1.1307 w_k1_MSFT_s105 + 1.08866 w_k1_PG_s105
                                 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.44092 w_k1_ADM_s105
                                 + 1.23264 w_k1_AXP_s105
                                 + 1.52693 w_k1_DIS_s105
                                 + 1.45142 w_k1_FDX_s105
                                 + 1.32998 w_k1_IBM_s105
                                 + 1.14936 w_k1_INTC_s105
                                 + 1.39598 w_k1_KO_s105 + 1.17349 w_k1_MAT_s105
                                 + 1.31629 w_k1_MSFT_s105
                                 + 1.23275 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.02718 w_k1_ADM_s105
                                 + 0.799503 w_k1_AXP_s105
                                 + 0.770426 w_k1_DIS_s105
                                 + 0.894497 w_k1_FDX_s105
                                 + 0.743555 w_k1_IBM_s105
                                 + 0.771055 w_k1_INTC_s105
                                 + 0.896948 w_k1_KO_s105
                                 + 0.831943 w_k1_MAT_s105
                                 + 0.951536 w_k1_MSFT_s105
                                 + 0.859234 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.886286 w_k1_ADM_s105
                                 + 0.811582 w_k1_AXP_s105
                                 + 0.689678 w_k1_DIS_s105
                                 + 0.759271 w_k1_FDX_s105
                                 + 1.1359 w_k1_IBM_s105
                                 + 0.708286 w_k1_INTC_s105
                                 + 1.12195 w_k1_KO_s105 + 1.12013 w_k1_MAT_s105
                                 + 0.912003 w_k1_MSFT_s105
                                 + 0.910521 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.14055 w_k1_ADM_s105
                                 + 1.11545 w_k1_AXP_s105 + 1.2985 w_k1_DIS_s105
                                 + 1.14837 w_k1_FDX_s105
                                 + 1.00381 w_k1_IBM_s105
                                 + 0.957526 w_k1_INTC_s105
                                 + 1.04983 w_k1_KO_s105
                                 + 0.920123 w_k1_MAT_s105
                                 + 1.09886 w_k1_MSFT_s105
                                 + 1.12071 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.12055 w_k1_ADM_s105
                                 + 0.93066 w_k1_AXP_s105
                                 + 1.13581 w_k1_DIS_s105
                                 + 1.19251 w_k1_FDX_s105
                                 + 0.743863 w_k1_IBM_s105
                                 + 1.00965 w_k1_INTC_s105
                                 + 0.961459 w_k1_KO_s105
                                 + 0.530804 w_k1_MAT_s105
                                 + 1.23199 w_k1_MSFT_s105
                                 + 0.971478 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.921567 w_k1_ADM_s105
                                 + 1.09252 w_k1_AXP_s105
                                 + 0.808266 w_k1_DIS_s105
                                 + 0.938446 w_k1_FDX_s105
                                 + 1.07198 w_k1_IBM_s105
                                 + 0.750962 w_k1_INTC_s105
                                 + 0.953047 w_k1_KO_s105
                                 + 1.20937 w_k1_MAT_s105
                                 + 0.980234 w_k1_MSFT_s105
                                 + 1.02413 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.07213 w_k1_ADM_s106
                                 + 0.983335 w_k1_AXP_s106
                                 + 1.06168 w_k1_DIS_s106
                                 + 1.10418 w_k1_FDX_s106
                                 + 0.732924 w_k1_IBM_s106
                                 + 1.23819 w_k1_INTC_s106
                                 + 0.93322 w_k1_KO_s106
                                 + 0.774328 w_k1_MAT_s106
                                 + 1.13154 w_k1_MSFT_s106
                                 + 1.08545 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.02703 w_k1_ADM_s106
                                 + 1.20056 w_k1_AXP_s106
                                 + 0.801644 w_k1_DIS_s106
                                 + 0.962725 w_k1_FDX_s106
                                 + 0.93823 w_k1_IBM_s106
                                 + 0.949508 w_k1_INTC_s106
                                 + 0.877074 w_k1_KO_s106
                                 + 0.992785 w_k1_MAT_s106
                                 + 0.942054 w_k1_MSFT_s106
                                 + 1.04011 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.22591 w_k1_ADM_s106
                                 + 1.76217 w_k1_AXP_s106
                                 + 1.67215 w_k1_DIS_s106
                                 + 1.41604 w_k1_FDX_s106
                                 + 1.40757 w_k1_IBM_s106
                                 + 1.40274 w_k1_INTC_s106
                                 + 1.24872 w_k1_KO_s106 + 1.4396 w_k1_MAT_s106
                                 + 1.38526 w_k1_MSFT_s106
                                 + 1.31421 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.730398 w_k1_ADM_s106
                                 + 0.887487 w_k1_AXP_s106
                                 + 0.708539 w_k1_DIS_s106
                                 + 0.780522 w_k1_FDX_s106
                                 + 0.947959 w_k1_IBM_s106
                                 + 0.936961 w_k1_INTC_s106
                                 + 0.759058 w_k1_KO_s106
                                 + 1.00695 w_k1_MAT_s106
                                 + 0.884689 w_k1_MSFT_s106
                                 + 0.86866 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.18949 w_k1_ADM_s106
                                 + 1.31299 w_k1_AXP_s106
                                 + 1.33426 w_k1_DIS_s106
                                 + 1.17117 w_k1_FDX_s106
                                 + 1.29202 w_k1_IBM_s106
                                 + 1.07058 w_k1_INTC_s106
                                 + 1.20562 w_k1_KO_s106 + 1.11975 w_k1_MAT_s106
                                 + 1.24704 w_k1_MSFT_s106
                                 + 1.13371 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.923234 w_k1_ADM_s106
                                 + 0.79518 w_k1_AXP_s106
                                 + 0.801348 w_k1_DIS_s106
                                 + 0.851033 w_k1_FDX_s106
                                 + 1.17288 w_k1_IBM_s106
                                 + 0.9948 w_k1_INTC_s106 + 1.04336 w_k1_KO_s106
                                 + 1.05491 w_k1_MAT_s106
                                 + 1.05155 w_k1_MSFT_s106
                                 + 1.00032 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.746493 w_k1_ADM_s106
                                 + 0.598163 w_k1_AXP_s106
                                 + 0.503976 w_k1_DIS_s106
                                 + 0.62846 w_k1_FDX_s106
                                 + 0.692586 w_k1_IBM_s106
                                 + 0.680334 w_k1_INTC_s106
                                 + 0.597639 w_k1_KO_s106
                                 + 0.408712 w_k1_MAT_s106
                                 + 0.674296 w_k1_MSFT_s106
                                 + 0.70881 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.08066 w_k1_ADM_s106
                                 + 0.905074 w_k1_AXP_s106
                                 + 0.781632 w_k1_DIS_s106
                                 + 0.913063 w_k1_FDX_s106
                                 + 0.844924 w_k1_IBM_s106
                                 + 0.883804 w_k1_INTC_s106
                                 + 0.749805 w_k1_KO_s106
                                 + 0.429049 w_k1_MAT_s106
                                 + 0.855344 w_k1_MSFT_s106
                                 + 0.8678 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.926238 w_k1_ADM_s107
                                 + 0.763015 w_k1_AXP_s107
                                 + 0.674862 w_k1_DIS_s107
                                 + 0.847296 w_k1_FDX_s107
                                 + 0.709204 w_k1_IBM_s107
                                 + 0.852628 w_k1_INTC_s107
                                 + 0.667051 w_k1_KO_s107
                                 + 0.542066 w_k1_MAT_s107
                                 + 0.811974 w_k1_MSFT_s107
                                 + 0.833985 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.07115 w_k1_ADM_s107
                                 + 1.27958 w_k1_AXP_s107
                                 + 1.51679 w_k1_DIS_s107
                                 + 1.29481 w_k1_FDX_s107 + 1.2565 w_k1_IBM_s107
                                 + 1.29944 w_k1_INTC_s107
                                 + 1.43095 w_k1_KO_s107 + 1.35516 w_k1_MAT_s107
                                 + 1.3132 w_k1_MSFT_s107 + 1.32832 w_k1_PG_s107
                                 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.885203 w_k1_ADM_s107
                                 + 1.02453 w_k1_AXP_s107
                                 + 0.941135 w_k1_DIS_s107
                                 + 1.05779 w_k1_FDX_s107
                                 + 1.15946 w_k1_IBM_s107
                                 + 1.04221 w_k1_INTC_s107
                                 + 1.04352 w_k1_KO_s107 + 1.13714 w_k1_MAT_s107
                                 + 0.946841 w_k1_MSFT_s107
                                 + 1.04775 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.659845 w_k1_ADM_s107
                                 + 0.782959 w_k1_AXP_s107
                                 + 0.698287 w_k1_DIS_s107
                                 + 0.769143 w_k1_FDX_s107
                                 + 1.01123 w_k1_IBM_s107
                                 + 0.810445 w_k1_INTC_s107
                                 + 0.842315 w_k1_KO_s107
                                 + 0.916866 w_k1_MAT_s107
                                 + 0.762255 w_k1_MSFT_s107
                                 + 0.833641 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.00347 w_k1_ADM_s107
                                 + 1.13341 w_k1_AXP_s107
                                 + 0.907421 w_k1_DIS_s107
                                 + 1.07801 w_k1_FDX_s107
                                 + 0.777557 w_k1_IBM_s107
                                 + 1.09348 w_k1_INTC_s107
                                 + 0.889205 w_k1_KO_s107
                                 + 0.806102 w_k1_MAT_s107
                                 + 0.988545 w_k1_MSFT_s107
                                 + 0.947761 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.77904 w_k1_ADM_s107
                                 + 0.993583 w_k1_AXP_s107
                                 + 1.04122 w_k1_DIS_s107
                                 + 1.03531 w_k1_FDX_s107
                                 + 1.65367 w_k1_IBM_s107
                                 + 1.20545 w_k1_INTC_s107
                                 + 1.15579 w_k1_KO_s107 + 1.17276 w_k1_MAT_s107
                                 + 0.935343 w_k1_MSFT_s107
                                 + 1.28026 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.18958 w_k1_ADM_s107
                                 + 1.16959 w_k1_AXP_s107
                                 + 1.21427 w_k1_DIS_s107
                                 + 1.26345 w_k1_FDX_s107
                                 + 0.884476 w_k1_IBM_s107
                                 + 1.02535 w_k1_INTC_s107
                                 + 1.09689 w_k1_KO_s107 + 0.97063 w_k1_MAT_s107
                                 + 1.27748 w_k1_MSFT_s107
                                 + 0.972125 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.01276 w_k1_ADM_s107
                                 + 0.807947 w_k1_AXP_s107
                                 + 0.837623 w_k1_DIS_s107
                                 + 0.902919 w_k1_FDX_s107
                                 + 1.01177 w_k1_IBM_s107
                                 + 0.882767 w_k1_INTC_s107
                                 + 0.998722 w_k1_KO_s107
                                 + 0.923831 w_k1_MAT_s107
                                 + 0.953705 w_k1_MSFT_s107
                                 + 1.0005 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.648918 w_k1_ADM_s110
                                 + 1.10609 w_k1_AXP_s110
                                 + 0.742153 w_k1_DIS_s110
                                 + 0.917254 w_k1_FDX_s110
                                 + 1.03583 w_k1_IBM_s110
                                 + 0.985216 w_k1_INTC_s110
                                 + 0.867908 w_k1_KO_s110
                                 + 1.16285 w_k1_MAT_s110
                                 + 0.815102 w_k1_MSFT_s110
                                 + 0.849825 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.596903 w_k1_ADM_s110
                                 + 0.939463 w_k1_AXP_s110
                                 + 0.703343 w_k1_DIS_s110
                                 + 0.957839 w_k1_FDX_s110
                                 + 0.78646 w_k1_IBM_s110
                                 + 1.2154 w_k1_INTC_s110
                                 + 0.792966 w_k1_KO_s110
                                 + 0.800507 w_k1_MAT_s110
                                 + 0.861796 w_k1_MSFT_s110
                                 + 0.873053 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.09395 w_k1_ADM_s110
                                 + 1.19065 w_k1_AXP_s110
                                 + 1.15337 w_k1_DIS_s110
                                 + 0.999327 w_k1_FDX_s110
                                 + 1.47958 w_k1_IBM_s110
                                 + 0.956027 w_k1_INTC_s110
                                 + 1.35103 w_k1_KO_s110 + 1.39734 w_k1_MAT_s110
                                 + 0.857322 w_k1_MSFT_s110
                                 + 0.99873 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.12495 w_k1_ADM_s110
                                 + 0.983156 w_k1_AXP_s110
                                 + 0.986687 w_k1_DIS_s110
                                 + 1.03178 w_k1_FDX_s110
                                 + 0.897299 w_k1_IBM_s110
                                 + 0.772081 w_k1_INTC_s110
                                 + 0.993741 w_k1_KO_s110
                                 + 0.920122 w_k1_MAT_s110
                                 + 0.853108 w_k1_MSFT_s110
                                 + 0.893706 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.49315 w_k1_ADM_s110
                                 + 1.35616 w_k1_AXP_s110
                                 + 1.82215 w_k1_DIS_s110
                                 + 1.42986 w_k1_FDX_s110
                                 + 0.864943 w_k1_IBM_s110
                                 + 1.09663 w_k1_INTC_s110
                                 + 1.29328 w_k1_KO_s110 + 0.93198 w_k1_MAT_s110
                                 + 1.18019 w_k1_MSFT_s110
                                 + 1.17241 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.878513 w_k1_ADM_s110
                                 + 0.788079 w_k1_AXP_s110
                                 + 0.704141 w_k1_DIS_s110
                                 + 0.813886 w_k1_FDX_s110
                                 + 0.748743 w_k1_IBM_s110
                                 + 0.796258 w_k1_INTC_s110
                                 + 0.732605 w_k1_KO_s110
                                 + 0.645667 w_k1_MAT_s110
                                 + 0.812416 w_k1_MSFT_s110
                                 + 0.771195 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.39945 w_k1_ADM_s110
                                 + 1.85207 w_k1_AXP_s110
                                 + 1.63888 w_k1_DIS_s110
                                 + 1.53662 w_k1_FDX_s110
                                 + 1.16995 w_k1_IBM_s110
                                 + 1.26959 w_k1_INTC_s110
                                 + 1.15847 w_k1_KO_s110 + 1.01989 w_k1_MAT_s110
                                 + 1.05908 w_k1_MSFT_s110
                                 + 1.06766 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.05994 w_k1_ADM_s110
                                 + 1.295 w_k1_AXP_s110 + 1.14245 w_k1_DIS_s110
                                 + 1.14392 w_k1_FDX_s110
                                 + 0.719588 w_k1_IBM_s110
                                 + 1.05327 w_k1_INTC_s110
                                 + 0.90815 w_k1_KO_s110
                                 + 0.671165 w_k1_MAT_s110
                                 + 1.03165 w_k1_MSFT_s110
                                 + 0.93385 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.995289 w_k1_ADM_s111
                                 + 0.613085 w_k1_AXP_s111
                                 + 0.59227 w_k1_DIS_s111
                                 + 0.736902 w_k1_FDX_s111
                                 + 1.07678 w_k1_IBM_s111
                                 + 0.751059 w_k1_INTC_s111
                                 + 0.770283 w_k1_KO_s111
                                 + 0.826904 w_k1_MAT_s111
                                 + 0.793857 w_k1_MSFT_s111
                                 + 0.773001 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.23115 w_k1_ADM_s111
                                 + 1.05141 w_k1_AXP_s111
                                 + 0.852564 w_k1_DIS_s111
                                 + 1.03762 w_k1_FDX_s111
                                 + 0.741915 w_k1_IBM_s111
                                 + 0.881124 w_k1_INTC_s111
                                 + 0.802192 w_k1_KO_s111
                                 + 0.797364 w_k1_MAT_s111
                                 + 0.961855 w_k1_MSFT_s111
                                 + 0.87985 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.898002 w_k1_ADM_s111
                                 + 0.727049 w_k1_AXP_s111
                                 + 0.804328 w_k1_DIS_s111
                                 + 0.828652 w_k1_FDX_s111
                                 + 0.878871 w_k1_IBM_s111
                                 + 0.928422 w_k1_INTC_s111
                                 + 0.866406 w_k1_KO_s111
                                 + 0.836854 w_k1_MAT_s111
                                 + 0.96718 w_k1_MSFT_s111
                                 + 0.902069 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 2.03542 w_k1_ADM_s111
                                 + 1.04664 w_k1_AXP_s111
                                 + 1.28585 w_k1_DIS_s111
                                 + 1.18432 w_k1_FDX_s111
                                 + 0.888096 w_k1_IBM_s111
                                 + 1.08553 w_k1_INTC_s111
                                 + 1.04723 w_k1_KO_s111
                                 + 0.844379 w_k1_MAT_s111
                                 + 1.23874 w_k1_MSFT_s111
                                 + 1.16041 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.28933 w_k1_ADM_s111
                                 + 1.17473 w_k1_AXP_s111
                                 + 1.28765 w_k1_DIS_s111
                                 + 1.12518 w_k1_FDX_s111
                                 + 1.03774 w_k1_IBM_s111
                                 + 1.28764 w_k1_INTC_s111
                                 + 1.11774 w_k1_KO_s111
                                 + 0.696841 w_k1_MAT_s111
                                 + 1.23764 w_k1_MSFT_s111
                                 + 1.05684 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.78909 w_k1_ADM_s111
                                 + 1.62664 w_k1_AXP_s111
                                 + 1.77742 w_k1_DIS_s111
                                 + 1.49973 w_k1_FDX_s111
                                 + 1.19031 w_k1_IBM_s111
                                 + 1.21741 w_k1_INTC_s111
                                 + 1.28708 w_k1_KO_s111 + 1.42388 w_k1_MAT_s111
                                 + 1.43249 w_k1_MSFT_s111
                                 + 1.23072 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.20401 w_k1_ADM_s111
                                 + 0.869706 w_k1_AXP_s111
                                 + 0.940728 w_k1_DIS_s111
                                 + 0.867161 w_k1_FDX_s111
                                 + 1.08266 w_k1_IBM_s111
                                 + 0.883013 w_k1_INTC_s111
                                 + 0.913597 w_k1_KO_s111
                                 + 0.925003 w_k1_MAT_s111
                                 + 0.987399 w_k1_MSFT_s111
                                 + 0.911591 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.40445 w_k1_ADM_s111
                                 + 1.26608 w_k1_AXP_s111
                                 + 1.15738 w_k1_DIS_s111 + 1.0919 w_k1_FDX_s111
                                 + 1.41492 w_k1_IBM_s111
                                 + 1.04423 w_k1_INTC_s111
                                 + 1.13658 w_k1_KO_s111 + 1.11485 w_k1_MAT_s111
                                 + 1.04734 w_k1_MSFT_s111
                                 + 1.09729 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.844797 w_k1_ADM_s112
                                 + 0.854495 w_k1_AXP_s112
                                 + 0.592918 w_k1_DIS_s112
                                 + 0.797833 w_k1_FDX_s112
                                 + 1.0244 w_k1_IBM_s112
                                 + 0.924984 w_k1_INTC_s112
                                 + 0.838422 w_k1_KO_s112
                                 + 1.10905 w_k1_MAT_s112
                                 + 0.797237 w_k1_MSFT_s112
                                 + 0.944497 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.31449 w_k1_ADM_s112
                                 + 1.39577 w_k1_AXP_s112
                                 + 1.47114 w_k1_DIS_s112
                                 + 1.18513 w_k1_FDX_s112
                                 + 1.55104 w_k1_IBM_s112
                                 + 1.2072 w_k1_INTC_s112 + 1.35659 w_k1_KO_s112
                                 + 1.11716 w_k1_MAT_s112
                                 + 1.26669 w_k1_MSFT_s112
                                 + 1.32885 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.22321 w_k1_ADM_s112
                                 + 1.19401 w_k1_AXP_s112 + 1.3434 w_k1_DIS_s112
                                 + 1.27488 w_k1_FDX_s112
                                 + 1.15797 w_k1_IBM_s112
                                 + 1.40866 w_k1_INTC_s112
                                 + 1.16914 w_k1_KO_s112 + 1.52978 w_k1_MAT_s112
                                 + 1.25328 w_k1_MSFT_s112
                                 + 1.12862 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.02436 w_k1_ADM_s112
                                 + 0.997575 w_k1_AXP_s112
                                 + 1.08016 w_k1_DIS_s112
                                 + 1.15253 w_k1_FDX_s112
                                 + 0.910335 w_k1_IBM_s112
                                 + 1.12257 w_k1_INTC_s112
                                 + 0.954286 w_k1_KO_s112
                                 + 0.790164 w_k1_MAT_s112
                                 + 1.14146 w_k1_MSFT_s112 + 1.0562 w_k1_PG_s112
                                 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.10804 w_k1_ADM_s112
                                 + 0.869123 w_k1_AXP_s112
                                 + 0.886252 w_k1_DIS_s112
                                 + 0.886656 w_k1_FDX_s112
                                 + 1.03463 w_k1_IBM_s112
                                 + 0.887771 w_k1_INTC_s112
                                 + 1.00136 w_k1_KO_s112 + 1.13667 w_k1_MAT_s112
                                 + 0.951903 w_k1_MSFT_s112
                                 + 1.03318 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.05644 w_k1_ADM_s112
                                 + 0.785495 w_k1_AXP_s112
                                 + 0.767863 w_k1_DIS_s112
                                 + 0.857427 w_k1_FDX_s112
                                 + 0.696017 w_k1_IBM_s112
                                 + 0.957738 w_k1_INTC_s112
                                 + 0.775822 w_k1_KO_s112
                                 + 0.58482 w_k1_MAT_s112
                                 + 0.817265 w_k1_MSFT_s112
                                 + 0.820441 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.18921 w_k1_ADM_s112
                                 + 1.12421 w_k1_AXP_s112
                                 + 0.913608 w_k1_DIS_s112
                                 + 0.969137 w_k1_FDX_s112
                                 + 1.21478 w_k1_IBM_s112
                                 + 1.00753 w_k1_INTC_s112 + 1.1033 w_k1_KO_s112
                                 + 1.22198 w_k1_MAT_s112
                                 + 0.913383 w_k1_MSFT_s112
                                 + 1.00516 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.863457 w_k1_ADM_s112
                                 + 0.784901 w_k1_AXP_s112
                                 + 0.706286 w_k1_DIS_s112
                                 + 0.77262 w_k1_FDX_s112
                                 + 0.871729 w_k1_IBM_s112
                                 + 0.859564 w_k1_INTC_s112
                                 + 0.753308 w_k1_KO_s112
                                 + 0.753788 w_k1_MAT_s112
                                 + 0.822749 w_k1_MSFT_s112
                                 + 0.806908 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.960051 w_k1_ADM_s113
                                 + 0.794641 w_k1_AXP_s113
                                 + 0.77656 w_k1_DIS_s113
                                 + 0.817334 w_k1_FDX_s113
                                 + 0.678428 w_k1_IBM_s113
                                 + 0.637517 w_k1_INTC_s113
                                 + 0.775939 w_k1_KO_s113
                                 + 0.82973 w_k1_MAT_s113
                                 + 0.831638 w_k1_MSFT_s113
                                 + 0.865067 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.69106 w_k1_ADM_s113
                                 + 0.72467 w_k1_AXP_s113
                                 + 0.585223 w_k1_DIS_s113
                                 + 0.688367 w_k1_FDX_s113
                                 + 0.979178 w_k1_IBM_s113
                                 + 0.834666 w_k1_INTC_s113
                                 + 0.799171 w_k1_KO_s113
                                 + 1.07639 w_k1_MAT_s113
                                 + 0.74732 w_k1_MSFT_s113
                                 + 0.841473 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.853164 w_k1_ADM_s113
                                 + 1.12323 w_k1_AXP_s113
                                 + 1.07786 w_k1_DIS_s113
                                 + 0.871072 w_k1_FDX_s113
                                 + 1.65949 w_k1_IBM_s113
                                 + 0.964745 w_k1_INTC_s113
                                 + 1.38429 w_k1_KO_s113 + 1.99709 w_k1_MAT_s113
                                 + 1.00783 w_k1_MSFT_s113
                                 + 1.04773 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.742771 w_k1_ADM_s113
                                 + 0.891242 w_k1_AXP_s113
                                 + 0.822184 w_k1_DIS_s113
                                 + 0.843381 w_k1_FDX_s113
                                 + 0.944591 w_k1_IBM_s113
                                 + 1.09647 w_k1_INTC_s113
                                 + 0.926727 w_k1_KO_s113
                                 + 1.13847 w_k1_MAT_s113
                                 + 1.06017 w_k1_MSFT_s113
                                 + 0.938319 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.32024 w_k1_ADM_s113
                                 + 1.48917 w_k1_AXP_s113
                                 + 1.59526 w_k1_DIS_s113
                                 + 1.29119 w_k1_FDX_s113
                                 + 1.29583 w_k1_IBM_s113
                                 + 1.37422 w_k1_INTC_s113
                                 + 1.24451 w_k1_KO_s113 + 1.76895 w_k1_MAT_s113
                                 + 1.3605 w_k1_MSFT_s113 + 1.31878 w_k1_PG_s113
                                 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.08049 w_k1_ADM_s113
                                 + 1.15177 w_k1_AXP_s113
                                 + 1.00508 w_k1_DIS_s113 + 1.08 w_k1_FDX_s113
                                 + 0.806258 w_k1_IBM_s113
                                 + 0.9431 w_k1_INTC_s113 + 0.90006 w_k1_KO_s113
                                 + 1.14995 w_k1_MAT_s113
                                 + 1.06325 w_k1_MSFT_s113
                                 + 0.998364 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.813215 w_k1_ADM_s113
                                 + 1.08438 w_k1_AXP_s113
                                 + 0.97731 w_k1_DIS_s113
                                 + 0.989696 w_k1_FDX_s113
                                 + 1.24094 w_k1_IBM_s113
                                 + 0.99553 w_k1_INTC_s113
                                 + 1.08723 w_k1_KO_s113 + 1.43383 w_k1_MAT_s113
                                 + 1.06521 w_k1_MSFT_s113
                                 + 0.98814 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.15671 w_k1_ADM_s113
                                 + 1.23703 w_k1_AXP_s113
                                 + 1.35123 w_k1_DIS_s113
                                 + 1.31831 w_k1_FDX_s113
                                 + 0.717953 w_k1_IBM_s113
                                 + 1.37108 w_k1_INTC_s113
                                 + 1.04199 w_k1_KO_s113
                                 + 0.891993 w_k1_MAT_s113
                                 + 1.5385 w_k1_MSFT_s113 + 1.0744 w_k1_PG_s113
                                 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.975 w_k1_ADM_s114
                                 + 0.69251 w_k1_AXP_s114
                                 + 0.652327 w_k1_DIS_s114
                                 + 0.78101 w_k1_FDX_s114
                                 + 0.717824 w_k1_IBM_s114
                                 + 0.775486 w_k1_INTC_s114
                                 + 0.73607 w_k1_KO_s114
                                 + 0.580257 w_k1_MAT_s114
                                 + 0.870917 w_k1_MSFT_s114
                                 + 0.83903 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.01706 w_k1_ADM_s114
                                 + 0.962872 w_k1_AXP_s114
                                 + 1.22034 w_k1_DIS_s114
                                 + 1.13621 w_k1_FDX_s114
                                 + 0.864848 w_k1_IBM_s114
                                 + 1.02238 w_k1_INTC_s114
                                 + 1.17864 w_k1_KO_s114
                                 + 0.711413 w_k1_MAT_s114
                                 + 1.27225 w_k1_MSFT_s114
                                 + 0.949545 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.31337 w_k1_ADM_s114
                                 + 1.10113 w_k1_AXP_s114 + 1.4321 w_k1_DIS_s114
                                 + 1.08693 w_k1_FDX_s114
                                 + 1.58609 w_k1_IBM_s114
                                 + 1.02275 w_k1_INTC_s114
                                 + 1.38125 w_k1_KO_s114 + 1.33377 w_k1_MAT_s114
                                 + 1.1561 w_k1_MSFT_s114 + 1.09105 w_k1_PG_s114
                                 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.57235 w_k1_ADM_s114
                                 + 1.0616 w_k1_AXP_s114 + 1.35318 w_k1_DIS_s114
                                 + 1.17272 w_k1_FDX_s114
                                 + 1.12733 w_k1_IBM_s114
                                 + 1.08106 w_k1_INTC_s114
                                 + 1.23271 w_k1_KO_s114 + 1.06246 w_k1_MAT_s114
                                 + 1.26957 w_k1_MSFT_s114
                                 + 1.05841 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.30173 w_k1_ADM_s114
                                 + 1.0124 w_k1_AXP_s114 + 0.77668 w_k1_DIS_s114
                                 + 0.895007 w_k1_FDX_s114
                                 + 0.875731 w_k1_IBM_s114
                                 + 1.06457 w_k1_INTC_s114
                                 + 0.883866 w_k1_KO_s114
                                 + 0.958579 w_k1_MAT_s114
                                 + 1.05264 w_k1_MSFT_s114 + 1.0019 w_k1_PG_s114
                                 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.50644 w_k1_ADM_s114
                                 + 1.28765 w_k1_AXP_s114
                                 + 1.03976 w_k1_DIS_s114 + 1.2392 w_k1_FDX_s114
                                 + 1.06876 w_k1_IBM_s114
                                 + 1.11134 w_k1_INTC_s114
                                 + 1.15637 w_k1_KO_s114 + 1.19129 w_k1_MAT_s114
                                 + 1.14451 w_k1_MSFT_s114
                                 + 1.17354 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.83234 w_k1_ADM_s114
                                 + 1.59676 w_k1_AXP_s114
                                 + 1.74203 w_k1_DIS_s114
                                 + 1.49494 w_k1_FDX_s114
                                 + 1.28385 w_k1_IBM_s114
                                 + 1.19789 w_k1_INTC_s114
                                 + 1.40837 w_k1_KO_s114 + 1.14432 w_k1_MAT_s114
                                 + 1.34692 w_k1_MSFT_s114
                                 + 1.24113 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.938929 w_k1_ADM_s114
                                 + 0.764335 w_k1_AXP_s114
                                 + 0.725017 w_k1_DIS_s114
                                 + 0.747936 w_k1_FDX_s114
                                 + 1.12716 w_k1_IBM_s114
                                 + 0.806297 w_k1_INTC_s114
                                 + 0.968424 w_k1_KO_s114
                                 + 0.906943 w_k1_MAT_s114
                                 + 0.773148 w_k1_MSFT_s114
                                 + 0.978524 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.02975 w_k1_ADM_s115
                                 + 0.796214 w_k1_AXP_s115
                                 + 0.660651 w_k1_DIS_s115
                                 + 0.895445 w_k1_FDX_s115
                                 + 0.90624 w_k1_IBM_s115
                                 + 0.853067 w_k1_INTC_s115
                                 + 0.834464 w_k1_KO_s115
                                 + 0.95599 w_k1_MAT_s115
                                 + 0.969698 w_k1_MSFT_s115
                                 + 0.848867 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.81006 w_k1_ADM_s115
                                 + 1.53548 w_k1_AXP_s115 + 1.7626 w_k1_DIS_s115
                                 + 1.5578 w_k1_FDX_s115 + 1.31132 w_k1_IBM_s115
                                 + 1.57162 w_k1_INTC_s115
                                 + 1.56706 w_k1_KO_s115 + 1.03378 w_k1_MAT_s115
                                 + 1.84724 w_k1_MSFT_s115
                                 + 1.20505 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.910754 w_k1_ADM_s115
                                 + 1.03721 w_k1_AXP_s115
                                 + 0.881254 w_k1_DIS_s115
                                 + 0.982882 w_k1_FDX_s115
                                 + 1.0626 w_k1_IBM_s115
                                 + 0.942713 w_k1_INTC_s115
                                 + 1.20766 w_k1_KO_s115 + 1.28892 w_k1_MAT_s115
                                 + 1.11694 w_k1_MSFT_s115
                                 + 0.977392 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.805932 w_k1_ADM_s115
                                 + 0.771111 w_k1_AXP_s115
                                 + 0.755597 w_k1_DIS_s115
                                 + 0.866075 w_k1_FDX_s115
                                 + 0.729165 w_k1_IBM_s115
                                 + 0.85916 w_k1_INTC_s115
                                 + 0.83372 w_k1_KO_s115
                                 + 0.674144 w_k1_MAT_s115
                                 + 0.891128 w_k1_MSFT_s115
                                 + 0.892963 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.25527 w_k1_ADM_s115
                                 + 1.29314 w_k1_AXP_s115
                                 + 1.40287 w_k1_DIS_s115
                                 + 1.25576 w_k1_FDX_s115
                                 + 0.968383 w_k1_IBM_s115
                                 + 1.02308 w_k1_INTC_s115
                                 + 1.10198 w_k1_KO_s115 + 1.24262 w_k1_MAT_s115
                                 + 1.22678 w_k1_MSFT_s115
                                 + 1.17858 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.44498 w_k1_ADM_s115
                                 + 1.21601 w_k1_AXP_s115
                                 + 1.22835 w_k1_DIS_s115 + 1.1222 w_k1_FDX_s115
                                 + 1.50348 w_k1_IBM_s115
                                 + 1.0289 w_k1_INTC_s115 + 1.42112 w_k1_KO_s115
                                 + 1.76864 w_k1_MAT_s115
                                 + 1.19765 w_k1_MSFT_s115
                                 + 1.13198 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.992052 w_k1_ADM_s115
                                 + 0.813856 w_k1_AXP_s115
                                 + 1.21282 w_k1_DIS_s115
                                 + 1.04548 w_k1_FDX_s115
                                 + 0.81997 w_k1_IBM_s115
                                 + 0.953507 w_k1_INTC_s115
                                 + 1.07508 w_k1_KO_s115
                                 + 0.804722 w_k1_MAT_s115
                                 + 1.20026 w_k1_MSFT_s115 + 1.1284 w_k1_PG_s115
                                 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.978732 w_k1_ADM_s115
                                 + 0.897099 w_k1_AXP_s115
                                 + 0.77701 w_k1_DIS_s115
                                 + 0.849344 w_k1_FDX_s115
                                 + 1.26436 w_k1_IBM_s115
                                 + 1.16547 w_k1_INTC_s115
                                 + 1.22702 w_k1_KO_s115 + 1.26913 w_k1_MAT_s115
                                 + 1.03957 w_k1_MSFT_s115
                                 + 1.03335 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.06154 w_k1_ADM_s116
                                 + 1.17353 w_k1_AXP_s116
                                 + 0.943969 w_k1_DIS_s116
                                 + 1.08874 w_k1_FDX_s116
                                 + 0.649067 w_k1_IBM_s116
                                 + 0.905745 w_k1_INTC_s116
                                 + 0.738171 w_k1_KO_s116
                                 + 0.676581 w_k1_MAT_s116
                                 + 1.08216 w_k1_MSFT_s116
                                 + 0.889787 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.833476 w_k1_ADM_s116
                                 + 0.800659 w_k1_AXP_s116
                                 + 0.5956 w_k1_DIS_s116
                                 + 0.693677 w_k1_FDX_s116
                                 + 0.952189 w_k1_IBM_s116
                                 + 0.685232 w_k1_INTC_s116
                                 + 0.937939 w_k1_KO_s116
                                 + 1.11126 w_k1_MAT_s116
                                 + 0.711868 w_k1_MSFT_s116
                                 + 0.938566 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.816619 w_k1_ADM_s116
                                 + 0.958023 w_k1_AXP_s116
                                 + 0.708764 w_k1_DIS_s116
                                 + 0.871136 w_k1_FDX_s116
                                 + 0.723835 w_k1_IBM_s116
                                 + 0.885183 w_k1_INTC_s116
                                 + 0.726179 w_k1_KO_s116
                                 + 0.804084 w_k1_MAT_s116
                                 + 0.842964 w_k1_MSFT_s116
                                 + 0.946031 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.00673 w_k1_ADM_s116
                                 + 1.54178 w_k1_AXP_s116 + 1.009 w_k1_DIS_s116
                                 + 1.12421 w_k1_FDX_s116
                                 + 0.999542 w_k1_IBM_s116
                                 + 1.14417 w_k1_INTC_s116
                                 + 1.06825 w_k1_KO_s116 + 1.0199 w_k1_MAT_s116
                                 + 1.08451 w_k1_MSFT_s116
                                 + 1.05606 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.51958 w_k1_ADM_s116
                                 + 1.58051 w_k1_AXP_s116
                                 + 1.66781 w_k1_DIS_s116
                                 + 1.38814 w_k1_FDX_s116
                                 + 1.07242 w_k1_IBM_s116
                                 + 1.10255 w_k1_INTC_s116
                                 + 1.29446 w_k1_KO_s116 + 1.27781 w_k1_MAT_s116
                                 + 1.3754 w_k1_MSFT_s116 + 1.2819 w_k1_PG_s116
                                 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.871156 w_k1_ADM_s116
                                 + 1.27581 w_k1_AXP_s116
                                 + 0.981526 w_k1_DIS_s116
                                 + 0.969365 w_k1_FDX_s116
                                 + 1.31186 w_k1_IBM_s116
                                 + 0.849076 w_k1_INTC_s116
                                 + 1.1014 w_k1_KO_s116 + 1.41992 w_k1_MAT_s116
                                 + 0.899707 w_k1_MSFT_s116
                                 + 1.22876 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.22423 w_k1_ADM_s116
                                 + 1.21377 w_k1_AXP_s116
                                 + 1.23693 w_k1_DIS_s116
                                 + 1.16627 w_k1_FDX_s116
                                 + 0.862487 w_k1_IBM_s116
                                 + 1.10771 w_k1_INTC_s116
                                 + 0.978578 w_k1_KO_s116
                                 + 1.02646 w_k1_MAT_s116
                                 + 1.20226 w_k1_MSFT_s116
                                 + 1.13648 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.870017 w_k1_ADM_s116
                                 + 0.93142 w_k1_AXP_s116 + 1.0584 w_k1_DIS_s116
                                 + 0.965528 w_k1_FDX_s116
                                 + 1.07556 w_k1_IBM_s116
                                 + 1.03031 w_k1_INTC_s116
                                 + 0.991114 w_k1_KO_s116
                                 + 0.751721 w_k1_MAT_s116
                                 + 0.959321 w_k1_MSFT_s116
                                 + 0.941018 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.927721 w_k1_ADM_s117
                                 + 0.94547 w_k1_AXP_s117
                                 + 1.00041 w_k1_DIS_s117
                                 + 0.907572 w_k1_FDX_s117
                                 + 0.897081 w_k1_IBM_s117
                                 + 0.892841 w_k1_INTC_s117
                                 + 0.835752 w_k1_KO_s117
                                 + 0.890534 w_k1_MAT_s117
                                 + 0.955892 w_k1_MSFT_s117
                                 + 0.929616 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.14667 w_k1_ADM_s117
                                 + 1.19437 w_k1_AXP_s117
                                 + 1.23822 w_k1_DIS_s117 + 1.2657 w_k1_FDX_s117
                                 + 0.945878 w_k1_IBM_s117
                                 + 1.02386 w_k1_INTC_s117
                                 + 1.03077 w_k1_KO_s117 + 1.0998 w_k1_MAT_s117
                                 + 1.17895 w_k1_MSFT_s117
                                 + 1.18031 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.98169 w_k1_ADM_s117
                                 + 0.805539 w_k1_AXP_s117
                                 + 0.939413 w_k1_DIS_s117
                                 + 0.957752 w_k1_FDX_s117
                                 + 0.65163 w_k1_IBM_s117
                                 + 0.9165 w_k1_INTC_s117
                                 + 0.746099 w_k1_KO_s117
                                 + 0.447172 w_k1_MAT_s117
                                 + 0.889844 w_k1_MSFT_s117
                                 + 0.785887 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.05733 w_k1_ADM_s117
                                 + 1.54294 w_k1_AXP_s117
                                 + 1.36432 w_k1_DIS_s117
                                 + 1.02247 w_k1_FDX_s117
                                 + 1.71378 w_k1_IBM_s117
                                 + 1.25226 w_k1_INTC_s117
                                 + 1.44634 w_k1_KO_s117 + 1.72773 w_k1_MAT_s117
                                 + 1.11938 w_k1_MSFT_s117
                                 + 1.19463 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.19257 w_k1_ADM_s117
                                 + 1.52495 w_k1_AXP_s117 + 1.1834 w_k1_DIS_s117
                                 + 1.10962 w_k1_FDX_s117
                                 + 1.10938 w_k1_IBM_s117
                                 + 1.02143 w_k1_INTC_s117
                                 + 1.04409 w_k1_KO_s117 + 1.32763 w_k1_MAT_s117
                                 + 1.04012 w_k1_MSFT_s117
                                 + 1.13521 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.91686 w_k1_ADM_s117
                                 + 0.927636 w_k1_AXP_s117
                                 + 0.875161 w_k1_DIS_s117
                                 + 0.814269 w_k1_FDX_s117
                                 + 1.46556 w_k1_IBM_s117
                                 + 0.834359 w_k1_INTC_s117
                                 + 1.05492 w_k1_KO_s117 + 1.32979 w_k1_MAT_s117
                                 + 0.79768 w_k1_MSFT_s117
                                 + 1.06046 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.13507 w_k1_ADM_s117
                                 + 0.850711 w_k1_AXP_s117
                                 + 1.02294 w_k1_DIS_s117
                                 + 0.903959 w_k1_FDX_s117
                                 + 1.23806 w_k1_IBM_s117
                                 + 1.03435 w_k1_INTC_s117 + 1.0635 w_k1_KO_s117
                                 + 1.03149 w_k1_MAT_s117
                                 + 0.988659 w_k1_MSFT_s117
                                 + 1.08104 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.987302 w_k1_ADM_s117
                                 + 0.70806 w_k1_AXP_s117
                                 + 0.565535 w_k1_DIS_s117
                                 + 0.704896 w_k1_FDX_s117
                                 + 0.828459 w_k1_IBM_s117
                                 + 0.687437 w_k1_INTC_s117
                                 + 0.767592 w_k1_KO_s117
                                 + 0.649425 w_k1_MAT_s117
                                 + 0.830753 w_k1_MSFT_s117
                                 + 0.901433 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.00531474138940073 w_k1_ADM_s000
                                 + 0.012327183920866 w_k1_ADM_s001
                                 + 0.0157810881032694 w_k1_ADM_s002
                                 + 0.0120370342872771 w_k1_ADM_s003
                                 + 0.0112005651550173 w_k1_ADM_s004
                                 + 0.0066961504313321 w_k1_ADM_s005
                                 + 0.0059390010937371 w_k1_ADM_s006
                                 + 0.0118593430934929 w_k1_ADM_s007
                                 + 0.0418872238620464 w_k1_ADM_s010
                                 + 0.00735777633346537 w_k1_ADM_s011
                                 + 0.0253313196101156 w_k1_ADM_s012
                                 + 0.0189781212714669 w_k1_ADM_s013
                                 + 0.0324397235310203 w_k1_ADM_s014
                                 + 0.0166837480020396 w_k1_ADM_s015
                                 + 0.0212402421981872 w_k1_ADM_s016
                                 + 0.0331516634038032 w_k1_ADM_s017
                                 + 0.0652613568418171 w_k1_ADM_s100
                                 + 0.0362842376488121 w_k1_ADM_s101
                                 + 0.0686978420002736 w_k1_ADM_s102
                                 + 0.0718828498337451 w_k1_ADM_s103
                                 + 0.11502094753257 w_k1_ADM_s104
                                 + 0.0585135070188338 w_k1_ADM_s105
                                 + 0.116863265897412 w_k1_ADM_s106
                                 + 0.0594486838876434 w_k1_ADM_s107
                                 + 0.00728553966315755 w_k1_ADM_s110
                                 + 0.01515762486157 w_k1_ADM_s111
                                 + 0.0191638335730382 w_k1_ADM_s112
                                 + 0.0165890264044756 w_k1_ADM_s113
                                 + 0.0239745212889871 w_k1_ADM_s114
                                 + 0.00852022147625978 w_k1_ADM_s115
                                 + 0.0323097916027886 w_k1_ADM_s116
                                 + 0.0180639596603259 w_k1_ADM_s117
                                 + 0.0052383830718953 w_k1_AXP_s000
                                 + 0.0109195525576146 w_k1_AXP_s001
                                 + 0.0149691245354009 w_k1_AXP_s002
                                 + 0.0108427271819499 w_k1_AXP_s003
                                 + 0.0113670044887174 w_k1_AXP_s004
                                 + 0.00695144402493834 w_k1_AXP_s005
                                 + 0.00490238569168119 w_k1_AXP_s006
                                 + 0.0125342996148859 w_k1_AXP_s007
                                 + 0.0458347430096357 w_k1_AXP_s010
                                 + 0.0082191659016667 w_k1_AXP_s011
                                 + 0.0238332921623576 w_k1_AXP_s012
                                 + 0.0195633555393327 w_k1_AXP_s013
                                 + 0.0387018691808005 w_k1_AXP_s014
                                 + 0.0179301102768913 w_k1_AXP_s015
                                 + 0.0188910454901724 w_k1_AXP_s016
                                 + 0.034409847026603 w_k1_AXP_s017
                                 + 0.0696348398184722 w_k1_AXP_s100
                                 + 0.0285023500108255 w_k1_AXP_s101
                                 + 0.0664461519075731 w_k1_AXP_s102
                                 + 0.0776086251879914 w_k1_AXP_s103
                                 + 0.119789873510216 w_k1_AXP_s104
                                 + 0.0560071447173836 w_k1_AXP_s105
                                 + 0.122497387604951 w_k1_AXP_s106
                                 + 0.062262502732864 w_k1_AXP_s107
                                 + 0.00856582526886772 w_k1_AXP_s110
                                 + 0.011941887085851 w_k1_AXP_s111
                                 + 0.0175012309035132 w_k1_AXP_s112
                                 + 0.0183158381072286 w_k1_AXP_s113
                                 + 0.0194556843699592 w_k1_AXP_s114
                                 + 0.00765433949354403 w_k1_AXP_s115
                                 + 0.0382308809501327 w_k1_AXP_s116
                                 + 0.0183466734073894 w_k1_AXP_s117
                                 + 0.00559412544572521 w_k1_DIS_s000
                                 + 0.0116159668949156 w_k1_DIS_s001
                                 + 0.0159113005162453 w_k1_DIS_s002
                                 + 0.0103665988100453 w_k1_DIS_s003
                                 + 0.0107092112534861 w_k1_DIS_s004
                                 + 0.00666497755813829 w_k1_DIS_s005
                                 + 0.00552553842155585 w_k1_DIS_s006
                                 + 0.0124409926398072 w_k1_DIS_s007
                                 + 0.0436901908117108 w_k1_DIS_s010
                                 + 0.00946528598892124 w_k1_DIS_s011
                                 + 0.0238292620382932 w_k1_DIS_s012
                                 + 0.019836924248235 w_k1_DIS_s013
                                 + 0.0393716107627836 w_k1_DIS_s014
                                 + 0.0186668578238029 w_k1_DIS_s015
                                 + 0.0188703672582292 w_k1_DIS_s016
                                 + 0.0344973315142831 w_k1_DIS_s017
                                 + 0.0766562732635729 w_k1_DIS_s100
                                 + 0.0286494658106991 w_k1_DIS_s101
                                 + 0.0712837659457763 w_k1_DIS_s102
                                 + 0.0788352404438997 w_k1_DIS_s103
                                 + 0.117850142797914 w_k1_DIS_s104
                                 + 0.0563941032726769 w_k1_DIS_s105
                                 + 0.115234924536341 w_k1_DIS_s106
                                 + 0.0613301053605545 w_k1_DIS_s107
                                 + 0.00765227885365142 w_k1_DIS_s110
                                 + 0.0122084522148521 w_k1_DIS_s111
                                 + 0.017455993346912 w_k1_DIS_s112
                                 + 0.0175359626101908 w_k1_DIS_s113
                                 + 0.0206923297386174 w_k1_DIS_s114
                                 + 0.00793348441621179 w_k1_DIS_s115
                                 + 0.0333172447582437 w_k1_DIS_s116
                                 + 0.0174930775812444 w_k1_DIS_s117
                                 + 0.00566115498903735 w_k1_FDX_s000
                                 + 0.0118651377250178 w_k1_FDX_s001
                                 + 0.0160882852032451 w_k1_FDX_s002
                                 + 0.010437898485683 w_k1_FDX_s003
                                 + 0.0113299219661303 w_k1_FDX_s004
                                 + 0.0068653797927501 w_k1_FDX_s005
                                 + 0.00523487289815993 w_k1_FDX_s006
                                 + 0.0127674117411392 w_k1_FDX_s007
                                 + 0.042563861476791 w_k1_FDX_s010
                                 + 0.00828788558888885 w_k1_FDX_s011
                                 + 0.0248980557633167 w_k1_FDX_s012
                                 + 0.0193519906047197 w_k1_FDX_s013
                                 + 0.0406501387189892 w_k1_FDX_s014
                                 + 0.0185889927579182 w_k1_FDX_s015
                                 + 0.019093461148737 w_k1_FDX_s016
                                 + 0.0347339638109751 w_k1_FDX_s017
                                 + 0.0737145221492622 w_k1_FDX_s100
                                 + 0.029526243526635 w_k1_FDX_s101
                                 + 0.0653005670194941 w_k1_FDX_s102
                                 + 0.0812465844036384 w_k1_FDX_s103
                                 + 0.114611552181321 w_k1_FDX_s104
                                 + 0.0582833327984875 w_k1_FDX_s105
                                 + 0.115511532228457 w_k1_FDX_s106
                                 + 0.0646251901364186 w_k1_FDX_s107
                                 + 0.00776002505830584 w_k1_FDX_s110
                                 + 0.011953892942877 w_k1_FDX_s111
                                 + 0.0178878704002023 w_k1_FDX_s112
                                 + 0.0172998404198413 w_k1_FDX_s113
                                 + 0.0201332377766826 w_k1_FDX_s114
                                 + 0.00794711563535434 w_k1_FDX_s115
                                 + 0.0331177503234941 w_k1_FDX_s116
                                 + 0.016632605264881 w_k1_FDX_s117
                                 + 0.00510291137703919 w_k1_IBM_s000
                                 + 0.0117924892057335 w_k1_IBM_s001
                                 + 0.0165427300094875 w_k1_IBM_s002
                                 + 0.0101644785627105 w_k1_IBM_s003
                                 + 0.0110975782361755 w_k1_IBM_s004
                                 + 0.00685723589710196 w_k1_IBM_s005
                                 + 0.00509870377232392 w_k1_IBM_s006
                                 + 0.0125923527235805 w_k1_IBM_s007
                                 + 0.0446974003484791 w_k1_IBM_s010
                                 + 0.00823638341504804 w_k1_IBM_s011
                                 + 0.0235474458311267 w_k1_IBM_s012
                                 + 0.0204209773349508 w_k1_IBM_s013
                                 + 0.0407625445471678 w_k1_IBM_s014
                                 + 0.0186372338551314 w_k1_IBM_s015
                                 + 0.0180490964652906 w_k1_IBM_s016
                                 + 0.0342324565497353 w_k1_IBM_s017
                                 + 0.0777261767656513 w_k1_IBM_s100
                                 + 0.0295303653196084 w_k1_IBM_s101
                                 + 0.0661363292027453 w_k1_IBM_s102
                                 + 0.0780859974248945 w_k1_IBM_s103
                                 + 0.113567828912351 w_k1_IBM_s104
                                 + 0.0601627102496663 w_k1_IBM_s105
                                 + 0.1147452559156 w_k1_IBM_s106
                                 + 0.0639416860084071 w_k1_IBM_s107
                                 + 0.00755525798109697 w_k1_IBM_s110
                                 + 0.0122255230793898 w_k1_IBM_s111
                                 + 0.0181757814080475 w_k1_IBM_s112
                                 + 0.0175186703426985 w_k1_IBM_s113
                                 + 0.0209755405157701 w_k1_IBM_s114
                                 + 0.00792697495500865 w_k1_IBM_s115
                                 + 0.0329430115290189 w_k1_IBM_s116
                                 + 0.0182097008041685 w_k1_IBM_s117
                                 + 0.00535644827410961 w_k1_INTC_s000
                                 + 0.012895739011973 w_k1_INTC_s001
                                 + 0.0157251825710924 w_k1_INTC_s002
                                 + 0.0110493909703262 w_k1_INTC_s003
                                 + 0.0117118346126466 w_k1_INTC_s004
                                 + 0.00651775986314966 w_k1_INTC_s005
                                 + 0.00438135833663419 w_k1_INTC_s006
                                 + 0.0127401699011281 w_k1_INTC_s007
                                 + 0.0435285260624199 w_k1_INTC_s010
                                 + 0.00754165184439503 w_k1_INTC_s011
                                 + 0.0249947075063598 w_k1_INTC_s012
                                 + 0.0191344825183441 w_k1_INTC_s013
                                 + 0.0385117121936345 w_k1_INTC_s014
                                 + 0.0178590558738206 w_k1_INTC_s015
                                 + 0.0177708828365455 w_k1_INTC_s016
                                 + 0.0312049049420916 w_k1_INTC_s017
                                 + 0.0658499298552189 w_k1_INTC_s100
                                 + 0.0310118833880118 w_k1_INTC_s101
                                 + 0.0649306054810443 w_k1_INTC_s102
                                 + 0.0696405795033278 w_k1_INTC_s103
                                 + 0.104292609363649 w_k1_INTC_s104
                                 + 0.050329373687307 w_k1_INTC_s105
                                 + 0.120151385036429 w_k1_INTC_s106
                                 + 0.0636053637371322 w_k1_INTC_s107
                                 + 0.00731916769922016 w_k1_INTC_s110
                                 + 0.0118283988974387 w_k1_INTC_s111
                                 + 0.0189430098753948 w_k1_INTC_s112
                                 + 0.0175177787756435 w_k1_INTC_s113
                                 + 0.0195058915763238 w_k1_INTC_s114
                                 + 0.0077751617512154 w_k1_INTC_s115
                                 + 0.0305034742578463 w_k1_INTC_s116
                                 + 0.0162226176807573 w_k1_INTC_s117
                                 + 0.00517886958266746 w_k1_KO_s000
                                 + 0.0111184446273426 w_k1_KO_s001
                                 + 0.0158594530926125 w_k1_KO_s002
                                 + 0.0100369472456025 w_k1_KO_s003
                                 + 0.0116668347384226 w_k1_KO_s004
                                 + 0.00623243003143481 w_k1_KO_s005
                                 + 0.00538353194365757 w_k1_KO_s006
                                 + 0.0128112881515115 w_k1_KO_s007
                                 + 0.0420025390643666 w_k1_KO_s010
                                 + 0.0078295450635325 w_k1_KO_s011
                                 + 0.0203738243613454 w_k1_KO_s012
                                 + 0.0214055162927413 w_k1_KO_s013
                                 + 0.0406054553481507 w_k1_KO_s014
                                 + 0.0182750700793481 w_k1_KO_s015
                                 + 0.0185611680330629 w_k1_KO_s016
                                 + 0.0387742169225602 w_k1_KO_s017
                                 + 0.0791174007337093 w_k1_KO_s100
                                 + 0.0336316643406143 w_k1_KO_s101
                                 + 0.075271057408344 w_k1_KO_s102
                                 + 0.0789264092335289 w_k1_KO_s103
                                 + 0.120411447776559 w_k1_KO_s104
                                 + 0.0600662843464504 w_k1_KO_s105
                                 + 0.110304892153356 w_k1_KO_s106
                                 + 0.0632286040292193 w_k1_KO_s107
                                 + 0.00750787244234006 w_k1_KO_s110
                                 + 0.0115535642444729 w_k1_KO_s111
                                 + 0.0174929461786857 w_k1_KO_s112
                                 + 0.0174644560904594 w_k1_KO_s113
                                 + 0.0215089053252144 w_k1_KO_s114
                                 + 0.00848564853662661 w_k1_KO_s115
                                 + 0.0322982316330477 w_k1_KO_s116
                                 + 0.0167413105717725 w_k1_KO_s117
                                 + 0.00522027421396847 w_k1_MAT_s000
                                 + 0.0114429644870094 w_k1_MAT_s001
                                 + 0.0177156684884494 w_k1_MAT_s002
                                 + 0.00909293761814688 w_k1_MAT_s003
                                 + 0.0110181786632244 w_k1_MAT_s004
                                 + 0.00643254254935075 w_k1_MAT_s005
                                 + 0.00723508511119735 w_k1_MAT_s006
                                 + 0.0111896452959449 w_k1_MAT_s007
                                 + 0.0396837508831139 w_k1_MAT_s010
                                 + 0.00904912154805338 w_k1_MAT_s011
                                 + 0.0249139866405038 w_k1_MAT_s012
                                 + 0.0221947792358279 w_k1_MAT_s013
                                 + 0.0377393919657809 w_k1_MAT_s014
                                 + 0.0166841445209918 w_k1_MAT_s015
                                 + 0.0192067897828107 w_k1_MAT_s016
                                 + 0.0416046571968765 w_k1_MAT_s017
                                 + 0.0669390891165011 w_k1_MAT_s100
                                 + 0.029962557936455 w_k1_MAT_s101
                                 + 0.0798432765066375 w_k1_MAT_s102
                                 + 0.086911466841824 w_k1_MAT_s103
                                 + 0.133910305354508 w_k1_MAT_s104
                                 + 0.0588240537534899 w_k1_MAT_s105
                                 + 0.106220416710808 w_k1_MAT_s106
                                 + 0.0608893761075521 w_k1_MAT_s107
                                 + 0.00741716946643841 w_k1_MAT_s110
                                 + 0.0108614104802202 w_k1_MAT_s111
                                 + 0.0185712654051594 w_k1_MAT_s112
                                 + 0.021523323918404 w_k1_MAT_s113
                                 + 0.0183637783928812 w_k1_MAT_s114
                                 + 0.00839553312705132 w_k1_MAT_s115
                                 + 0.0344252444264748 w_k1_MAT_s116
                                 + 0.017940954974672 w_k1_MAT_s117
                                 + 0.00540889868563847 w_k1_MSFT_s000
                                 + 0.0107090088699893 w_k1_MSFT_s001
                                 + 0.0151625642006428 w_k1_MSFT_s002
                                 + 0.0104318762720144 w_k1_MSFT_s003
                                 + 0.0113358501135724 w_k1_MSFT_s004
                                 + 0.00678463946421036 w_k1_MSFT_s005
                                 + 0.00546067998594647 w_k1_MSFT_s006
                                 + 0.0122637402015536 w_k1_MSFT_s007
                                 + 0.0413236594365471 w_k1_MSFT_s010
                                 + 0.00797002613318894 w_k1_MSFT_s011
                                 + 0.024490843946102 w_k1_MSFT_s012
                                 + 0.0177788691769327 w_k1_MSFT_s013
                                 + 0.0368894471213011 w_k1_MSFT_s014
                                 + 0.0185463515455202 w_k1_MSFT_s015
                                 + 0.0179151679679869 w_k1_MSFT_s016
                                 + 0.0345490918275767 w_k1_MSFT_s017
                                 + 0.0703819919214814 w_k1_MSFT_s100
                                 + 0.0302141685053847 w_k1_MSFT_s101
                                 + 0.0664805464619614 w_k1_MSFT_s102
                                 + 0.0739725111448932 w_k1_MSFT_s103
                                 + 0.105883378901571 w_k1_MSFT_s104
                                 + 0.059720298611576 w_k1_MSFT_s105
                                 + 0.121295672377738 w_k1_MSFT_s106
                                 + 0.06280189759883 w_k1_MSFT_s107
                                 + 0.00663348445331349 w_k1_MSFT_s110
                                 + 0.0125691590486766 w_k1_MSFT_s111
                                 + 0.0180131072388815 w_k1_MSFT_s112
                                 + 0.0188870486496841 w_k1_MSFT_s113
                                 + 0.0215587357741804 w_k1_MSFT_s114
                                 + 0.00870296361572758 w_k1_MSFT_s115
                                 + 0.0323274741166669 w_k1_MSFT_s116
                                 + 0.0167992369702954 w_k1_MSFT_s117
                                 + 0.00518949137088749 w_k1_PG_s000
                                 + 0.0104470325471305 w_k1_PG_s001
                                 + 0.0160994233795353 w_k1_PG_s002
                                 + 0.0102203967502506 w_k1_PG_s003
                                 + 0.010273108990242 w_k1_PG_s004
                                 + 0.00615195466479285 w_k1_PG_s005
                                 + 0.00539135259245486 w_k1_PG_s006
                                 + 0.0117941610903953 w_k1_PG_s007
                                 + 0.0440278996827069 w_k1_PG_s010
                                 + 0.00875156095595679 w_k1_PG_s011
                                 + 0.022504036876194 w_k1_PG_s012
                                 + 0.0207002700354257 w_k1_PG_s013
                                 + 0.0394200324996075 w_k1_PG_s014
                                 + 0.0186008224932856 w_k1_PG_s015
                                 + 0.0181186873016968 w_k1_PG_s016
                                 + 0.0338204177125387 w_k1_PG_s017
                                 + 0.0699204280246298 w_k1_PG_s100
                                 + 0.0309055301075436 w_k1_PG_s101
                                 + 0.0703100030886266 w_k1_PG_s102
                                 + 0.0758246990224925 w_k1_PG_s103
                                 + 0.108698179517738 w_k1_PG_s104
                                 + 0.0567906034829302 w_k1_PG_s105
                                 + 0.117764042580531 w_k1_PG_s106
                                 + 0.0636847974284155 w_k1_PG_s107
                                 + 0.00683580581226695 w_k1_PG_s110
                                 + 0.0117179023158997 w_k1_PG_s111
                                 + 0.0179541495145159 w_k1_PG_s112
                                 + 0.0174413367870952 w_k1_PG_s113
                                 + 0.0202749532113891 w_k1_PG_s114
                                 + 0.00792107597323288 w_k1_PG_s115
                                 + 0.0349385860637212 w_k1_PG_s116
                                 + 0.0177446616083844 w_k1_PG_s117 >= 1.219
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

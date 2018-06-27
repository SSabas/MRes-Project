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
 w_k1_ADM_s0_return_constraint:  1.20938 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.914873 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.01527 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.875447 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.994751 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.13837 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.82341 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.760063 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.38584 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.903605 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.95543 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.20673 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.20797 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.15651 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.03918 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.942215 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.15056 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.37482 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.766453 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.23362 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.1986 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.16519 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.15467 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.02404 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.40422 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.13402 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.925286 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.76151 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.808119 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.19864 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.916709 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.10427 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.953694 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.771875 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.871042 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.688518 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.976256 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.0081 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.29845 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.898789 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.14639 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.891783 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.858907 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.17767 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.01752 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.604963 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.76693 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.95876 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.08413 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.08871 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.01905 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.917979 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.976766 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.837269 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.09975 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.43512 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.635569 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.01391 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.36332 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.36112 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.43802 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.1721 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.72311 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.20743 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.837031 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.671103 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.933328 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.20276 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.39212 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.42379 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.826155 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.0239 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.739746 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.654615 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.876593 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.07545 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.42875 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.923541 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.16482 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.752814 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.858351 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.21496 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.949472 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.630122 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.55954 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.857809 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.30448 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.29418 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.90552 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.15181 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.92074 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.02569 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.00811 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.77592 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.587937 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.43143 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.963871 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.47655 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.15091 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.993636 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.77093 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.37616 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.80952 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.57982 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.944441 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.18227 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.35443 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.42161 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.0437 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.850218 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.671698 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.546475 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.771023 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.962652 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.22956 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.95153 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.10602 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.929062 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.921104 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.06237 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.798965 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.717581 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.35588 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.950591 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.16211 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.34024 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.09074 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.05378 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.07424 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.883933 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.925565 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.55182 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.699955 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.24505 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.06252 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.12145 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.10153 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.10574 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.48153 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.31836 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.856943 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.710414 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.833458 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.14991 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.20311 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.3831 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.911886 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.882194 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.911044 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.649499 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.910706 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.12448 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.20505 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.982112 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.20445 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.768547 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.771542 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.28939 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.63486 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.794595 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.29753 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.883256 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.14899 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.601193 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.619328 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.851126 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.624955 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.01704 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.56401 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.06446 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.776681 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.03077 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.03417 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.18663 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.5115 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.775432 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.1321 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.784033 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.914871 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.704085 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.37926 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.21319 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.51015 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.971244 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.20526 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.03476 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.563231 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.931114 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.833854 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.734964 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.20593 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.95595 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.962258 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.844887 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.918204 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.997097 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.971016 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.900963 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.48797 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.14572 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.41002 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.2361 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.17065 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.26925 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.00025 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.990006 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.893784 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.36226 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.804534 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.19036 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.13476 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.22052 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.0758 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.09254 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.44591 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.1612 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.851144 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.78391 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.01744 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.07777 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.28671 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.43374 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.978922 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.08781 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.943161 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.883491 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.945529 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.25792 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.30038 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.931658 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.08309 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.735056 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.869658 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.17224 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.26556 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.779633 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.46782 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.806967 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.23913 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.11295 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.74494 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.02209 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.854516 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.04657 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.26173 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.32884 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.769713 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.3669 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.01404 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.38018 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.37051 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.945692 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.4363 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.12153 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.926537 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.652938 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.07854 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.21011 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.4035 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.19121 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.15021 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.922584 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.780123 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.820739 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.905607 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.885188 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.2405 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.959533 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.46634 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.898982 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.724705 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.30553 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.31064 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.525359 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.40476 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.840211 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.11842 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.696856 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.588755 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.801247 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.634212 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.897036 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.34954 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.01341 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.647365 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.776079 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.05882 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.06794 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.74058 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.968474 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.50058 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.06426 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.976309 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.553556 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.41018 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.15968 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.67104 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.18842 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.909246 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.17582 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.667932 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.810788 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.65483 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.710448 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.22077 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.966217 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.05049 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.927176 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.926621 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.0726 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.907646 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.691944 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.31174 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.944072 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.24364 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.31573 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.0132 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.16989 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.926636 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.965862 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.935442 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.28241 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.703019 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.16503 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.00354 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.04464 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.03846 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.07889 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.42987 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.28244 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.875155 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.680657 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.906761 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.08303 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.10007 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.28702 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.02579 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.877765 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.854739 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.7322 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.915188 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.08775 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.17912 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.93 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.08449 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.88926 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.953565 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.18644 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.883288 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.769407 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.14881 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.880568 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.07175 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.07756 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.854531 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.06919 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.897053 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.898633 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.07195 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.26675 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.830385 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.12256 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.04583 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.11903 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.1561 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.00177 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.14588 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.14239 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.916295 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.748101 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.963032 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.14075 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.11195 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.11369 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.928769 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.883617 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.812738 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.761142 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.873472 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.927809 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.249999541112282 z_k1_s0000
                                 - 0.12200009552701 z_k1_s0001
                                 - 0.189000667849572 z_k1_s0002
                                 - 0.159999894231988 z_k1_s0003
                                 - 0.312000655052959 z_k1_s0004
                                 - 0.223000116848512 z_k1_s0005
                                 - 0.0760000672102737 z_k1_s0006
                                 - 0.23399970828919 z_k1_s0007
                                 - 0.111000466658505 z_k1_s0010
                                 - 0.375001738630662 z_k1_s0011
                                 - 0.119000493087922 z_k1_s0012
                                 - 0.25100132897471 z_k1_s0013
                                 - 0.421000891096253 z_k1_s0014
                                 - 0.35400216900522 z_k1_s0015
                                 - 0.229000256790264 z_k1_s0016
                                 - 0.139000559161462 z_k1_s0017
                                 - 0.416001668975791 z_k1_s0020
                                 - 0.115000210721024 z_k1_s0021
                                 - 0.162000377798522 z_k1_s0022
                                 - 0.182000543976426 z_k1_s0023
                                 - 0.0920001685768193 z_k1_s0024
                                 - 0.221001352144594 z_k1_s0025
                                 - 0.231000318030654 z_k1_s0026
                                 - 0.443000738872974 z_k1_s0027
                                 - 0.291999670642238 z_k1_s0030
                                 - 0.410999899488739 z_k1_s0031
                                 - 0.492999660065874 z_k1_s0032
                                 - 0.448998044465943 z_k1_s0033
                                 - 0.482000033414807 z_k1_s0034
                                 - 0.188999887968636 z_k1_s0035
                                 - 0.448998044465943 z_k1_s0036
                                 - 0.34400104297927 z_k1_s0037
                                 - 0.167999997365165 z_k1_s0040
                                 - 0.36099903378309 z_k1_s0041
                                 - 0.255000000656053 z_k1_s0042
                                 - 0.235999928020128 z_k1_s0043
                                 - 0.537998653923256 z_k1_s0044
                                 - 0.281000648894571 z_k1_s0045
                                 - 0.264000405530886 z_k1_s0046
                                 - 0.504000774195328 z_k1_s0047
                                 - 0.0450000674731105 z_k1_s0050
                                 - 0.1659998423939 z_k1_s0051
                                 - 0.08299992119695 z_k1_s0052
                                 - 0.0720000090769182 z_k1_s0053
                                 - 0.0550000550004521 z_k1_s0054
                                 - 0.0670000565132718 z_k1_s0055
                                 - 0.175000399728844 z_k1_s0056
                                 - 0.160999807430205 z_k1_s0057
                                 - 0.0650000923456778 z_k1_s0060
                                 - 0.169000426818933 z_k1_s0061
                                 - 0.0690000274099625 z_k1_s0062
                                 - 0.0770000531386745 z_k1_s0063
                                 - 0.130000029091213 z_k1_s0064
                                 - 0.0410000151595419 z_k1_s0065
                                 - 0.0460000701400225 z_k1_s0066
                                 - 0.181000309811859 z_k1_s0067
                                 - 0.315000369679181 z_k1_s0070
                                 - 0.193999610009482 z_k1_s0071
                                 - 0.443998668073756 z_k1_s0072
                                 - 0.172999770663988 z_k1_s0073
                                 - 0.107999941447363 z_k1_s0074
                                 - 0.201999001732209 z_k1_s0075
                                 - 0.24900029222259 z_k1_s0076
                                 - 0.16799982656266 z_k1_s0077
                                 - 0.154000213337293 z_k1_s0100
                                 - 0.243999810379324 z_k1_s0101
                                 - 0.133999803095335 z_k1_s0102
                                 - 0.206999363396256 z_k1_s0103
                                 - 0.174999055248625 z_k1_s0104
                                 - 0.119999849655487 z_k1_s0105
                                 - 0.0929998254430857 z_k1_s0106
                                 - 0.32399912975048 z_k1_s0107
                                 - 0.146000155439737 z_k1_s0110
                                 - 0.603000960999984 z_k1_s0111
                                 - 0.313000028482211 z_k1_s0112
                                 - 0.31900024516894 z_k1_s0113
                                 - 0.344000280696299 z_k1_s0114
                                 - 0.372999593347467 z_k1_s0115
                                 - 0.214000095954032 z_k1_s0116
                                 - 0.290000932517772 z_k1_s0117
                                 - 0.231000950592454 z_k1_s0120
                                 - 0.153000663639274 z_k1_s0121
                                 - 0.122000455542176 z_k1_s0122
                                 - 0.17400003550887 z_k1_s0123
                                 - 0.145000466258886 z_k1_s0124
                                 - 0.311001614391852 z_k1_s0125
                                 - 0.0750002456856463 z_k1_s0126
                                 - 0.0990003138250172 z_k1_s0127
                                 - 0.0919996987839768 z_k1_s0130
                                 - 0.25699929015913 z_k1_s0131
                                 - 0.160999256672711 z_k1_s0132
                                 - 0.163999021251233 z_k1_s0133
                                 - 0.151999962937144 z_k1_s0134
                                 - 0.0679997444621841 z_k1_s0135
                                 - 0.0709997252399552 z_k1_s0136
                                 - 0.115999545006145 z_k1_s0137
                                 - 0.216999032914076 z_k1_s0140
                                 - 0.244999284095815 z_k1_s0141
                                 - 0.23299965215987 z_k1_s0142
                                 - 0.191999105965524 z_k1_s0143
                                 - 0.151999722346879 z_k1_s0144
                                 - 0.141999751567458 z_k1_s0145
                                 - 0.242999622939291 z_k1_s0146
                                 - 0.240999961782766 z_k1_s0147
                                 - 0.0580000135828337 z_k1_s0150
                                 - 0.18299986732516 z_k1_s0151
                                 - 0.136999622511205 z_k1_s0152
                                 - 0.137999926441922 z_k1_s0153
                                 - 0.0729999669439149 z_k1_s0154
                                 - 0.0509999868676659 z_k1_s0155
                                 - 0.099000193143101 z_k1_s0156
                                 - 0.068999963220961 z_k1_s0157
                                 - 0.178999757356995 z_k1_s0160
                                 - 0.0659999444137477 z_k1_s0161
                                 - 0.149000079487069 z_k1_s0162
                                 - 0.0760000236370308 z_k1_s0163
                                 - 0.0690000055007275 z_k1_s0164
                                 - 0.0960000887836097 z_k1_s0165
                                 - 0.119000401617143 z_k1_s0166
                                 - 0.178999757356995 z_k1_s0167
                                 - 0.0849999214914966 z_k1_s0170
                                 - 0.13099948669667 z_k1_s0171
                                 - 0.14599989880654 z_k1_s0172
                                 - 0.0799999629213745 z_k1_s0173
                                 - 0.0969998041798063 z_k1_s0174
                                 - 0.126000243325885 z_k1_s0175
                                 - 0.04099996423472 z_k1_s0176
                                 - 0.187999979915168 z_k1_s0177
                                 - 0.36300135903799 z_k1_s1000
                                 - 0.339000348604943 z_k1_s1001
                                 - 0.115000081904301 z_k1_s1002
                                 - 0.397999178833074 z_k1_s1003
                                 - 0.35299995052338 z_k1_s1004
                                 - 0.267999828159279 z_k1_s1005
                                 - 0.274000673268045 z_k1_s1006
                                 - 0.260001071349608 z_k1_s1007
                                 - 1.05100035109248 z_k1_s1010
                                 - 0.562999014856892 z_k1_s1011
                                 - 0.293000091817405 z_k1_s1012
                                 - 0.398000026888452 z_k1_s1013
                                 - 0.401999933414933 z_k1_s1014
                                 - 0.698002116378024 z_k1_s1015
                                 - 0.878000185070553 z_k1_s1016
                                 - 0.267000016895016 z_k1_s1017
                                 - 0.786003552478971 z_k1_s1020
                                 - 1.06700473346679 z_k1_s1021
                                 - 0.979003765839301 z_k1_s1022
                                 - 0.305000808611997 z_k1_s1023
                                 - 0.361000872937279 z_k1_s1024
                                 - 0.267000814955673 z_k1_s1025
                                 - 0.481998937217371 z_k1_s1026
                                 - 0.419000789580968 z_k1_s1027
                                 - 0.554001065415242 z_k1_s1030
                                 - 0.301001291574159 z_k1_s1031
                                 - 0.165999943975594 z_k1_s1032
                                 - 0.369999276439432 z_k1_s1033
                                 - 0.201000144130741 z_k1_s1034
                                 - 0.260000028752164 z_k1_s1035
                                 - 0.598999634414304 z_k1_s1036
                                 - 0.235000077641392 z_k1_s1037
                                 - 0.751001265250991 z_k1_s1040
                                 - 2.57200890898353 z_k1_s1041
                                 - 1.01800595734763 z_k1_s1042
                                 - 0.630001220960547 z_k1_s1043
                                 - 0.649001548105987 z_k1_s1044
                                 - 0.771001704456913 z_k1_s1045
                                 - 1.60699988082286 z_k1_s1046
                                 - 1.01200528498474 z_k1_s1047
                                 - 1.04000187596165 z_k1_s1050
                                 - 0.717997945918302 z_k1_s1051
                                 - 0.523998570823373 z_k1_s1052
                                 - 0.293999922248873 z_k1_s1053
                                 - 0.56700135575115 z_k1_s1054
                                 - 0.544998063369804 z_k1_s1055
                                 - 1.00300181404666 z_k1_s1056
                                 - 0.32800012835721 z_k1_s1057
                                 - 0.169999940112165 z_k1_s1060
                                 - 0.22800020508234 z_k1_s1061
                                 - 0.355001649317466 z_k1_s1062
                                 - 0.631001259174717 z_k1_s1063
                                 - 0.419001281284226 z_k1_s1064
                                 - 0.391001043298569 z_k1_s1065
                                 - 0.564001145709266 z_k1_s1066
                                 - 0.434000496776087 z_k1_s1067
                                 - 0.262000060576493 z_k1_s1070
                                 - 0.947001553864473 z_k1_s1071
                                 - 0.274000115854844 z_k1_s1072
                                 - 0.409000565786164 z_k1_s1073
                                 - 0.43399867486461 z_k1_s1074
                                 - 0.340000075985524 z_k1_s1075
                                 - 0.255000314914329 z_k1_s1076
                                 - 0.517999749613562 z_k1_s1077
                                 - 0.547999189558295 z_k1_s1100
                                 - 1.9939976953641 z_k1_s1101
                                 - 0.398999308182921 z_k1_s1102
                                 - 0.531999077577227 z_k1_s1103
                                 - 1.26099992107078 z_k1_s1104
                                 - 1.92699761359282 z_k1_s1105
                                 - 0.611998863183687 z_k1_s1106
                                 - 0.469999417949468 z_k1_s1107
                                 - 1.03499687976635 z_k1_s1110
                                 - 1.18899876525282 z_k1_s1111
                                 - 1.67900234161936 z_k1_s1112
                                 - 0.39100007981146 z_k1_s1113
                                 - 0.463999891925408 z_k1_s1114
                                 - 0.754997691699135 z_k1_s1115
                                 - 0.774000563563774 z_k1_s1116
                                 - 0.375999671175814 z_k1_s1117
                                 - 0.762999279523062 z_k1_s1120
                                 - 0.304999666104676 z_k1_s1121
                                 - 0.310999650442482 z_k1_s1122
                                 - 0.774998554799351 z_k1_s1123
                                 - 0.431998178922695 z_k1_s1124
                                 - 0.359000565243912 z_k1_s1125
                                 - 0.271999752246745 z_k1_s1126
                                 - 0.249999925241344 z_k1_s1127
                                 - 0.54499986940343 z_k1_s1130
                                 - 0.225999773850965 z_k1_s1131
                                 - 0.700000405572596 z_k1_s1132
                                 - 0.578998225810175 z_k1_s1133
                                 - 0.485998561907968 z_k1_s1134
                                 - 0.276999610758605 z_k1_s1135
                                 - 0.221999696356346 z_k1_s1136
                                 - 0.253999658514011 z_k1_s1137
                                 - 0.271999957604584 z_k1_s1140
                                 - 0.314999876784749 z_k1_s1141
                                 - 0.893000290694292 z_k1_s1142
                                 - 0.300999770261026 z_k1_s1143
                                 - 0.345999224659054 z_k1_s1144
                                 - 0.395001445491361 z_k1_s1145
                                 - 0.411999678842337 z_k1_s1146
                                 - 0.426000793365666 z_k1_s1147
                                 - 0.361000442814621 z_k1_s1150
                                 - 0.553000678328214 z_k1_s1151
                                 - 0.321000393749289 z_k1_s1152
                                 - 0.908003527787203 z_k1_s1153
                                 - 0.466000571611117 z_k1_s1154
                                 - 1.08100313649372 z_k1_s1155
                                 - 1.55300130146047 z_k1_s1156
                                 - 0.791998557489404 z_k1_s1157
                                 - 0.356000277014836 z_k1_s1160
                                 - 0.309000127302109 z_k1_s1161
                                 - 0.775999472424655 z_k1_s1162
                                 - 0.962002472608946 z_k1_s1163
                                 - 1.08100008688993 z_k1_s1164
                                 - 0.348000292340364 z_k1_s1165
                                 - 0.661000891480109 z_k1_s1166
                                 - 0.302000380462146 z_k1_s1167
                                 - 0.266999173058519 z_k1_s1170
                                 - 0.291999139860883 z_k1_s1171
                                 - 0.99099700443423 z_k1_s1172
                                 - 0.423998667146166 z_k1_s1173
                                 - 0.655000354938612 z_k1_s1174
                                 - 0.312999094478915 z_k1_s1175
                                 - 0.811995912436903 z_k1_s1176
                                 - 0.619999701577195 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.952876 w_k1_ADM_s000
                                 + 0.872024 w_k1_AXP_s000
                                 + 0.804323 w_k1_DIS_s000
                                 + 0.988753 w_k1_FDX_s000
                                 + 0.855084 w_k1_IBM_s000
                                 + 1.02796 w_k1_INTC_s000
                                 + 0.777618 w_k1_KO_s000
                                 + 0.877807 w_k1_MAT_s000
                                 + 1.00546 w_k1_MSFT_s000
                                 + 1.09096 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.58716 w_k1_ADM_s000
                                 + 1.38046 w_k1_AXP_s000
                                 + 1.87367 w_k1_DIS_s000
                                 + 1.59721 w_k1_FDX_s000
                                 + 1.82128 w_k1_IBM_s000
                                 + 1.63752 w_k1_INTC_s000
                                 + 1.62134 w_k1_KO_s000 + 1.98157 w_k1_MAT_s000
                                 + 1.39479 w_k1_MSFT_s000 + 1.4773 w_k1_PG_s000
                                 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.26721 w_k1_ADM_s000
                                 + 0.989194 w_k1_AXP_s000
                                 + 1.18895 w_k1_DIS_s000
                                 + 1.12654 w_k1_FDX_s000
                                 + 0.777302 w_k1_IBM_s000
                                 + 1.08658 w_k1_INTC_s000
                                 + 0.923338 w_k1_KO_s000
                                 + 0.988663 w_k1_MAT_s000
                                 + 1.05275 w_k1_MSFT_s000
                                 + 1.19779 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.307 w_k1_ADM_s000
                                 + 1.29241 w_k1_AXP_s000
                                 + 1.14709 w_k1_DIS_s000
                                 + 1.19818 w_k1_FDX_s000
                                 + 1.43026 w_k1_IBM_s000
                                 + 1.00927 w_k1_INTC_s000
                                 + 1.18244 w_k1_KO_s000 + 1.51929 w_k1_MAT_s000
                                 + 0.920813 w_k1_MSFT_s000
                                 + 1.1407 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.923793 w_k1_ADM_s000
                                 + 0.972507 w_k1_AXP_s000
                                 + 0.790806 w_k1_DIS_s000
                                 + 0.882944 w_k1_FDX_s000
                                 + 1.06444 w_k1_IBM_s000
                                 + 1.11906 w_k1_INTC_s000
                                 + 0.989292 w_k1_KO_s000 + 1.0968 w_k1_MAT_s000
                                 + 0.945694 w_k1_MSFT_s000
                                 + 1.08715 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.967352 w_k1_ADM_s000
                                 + 0.811956 w_k1_AXP_s000
                                 + 0.631885 w_k1_DIS_s000
                                 + 0.830664 w_k1_FDX_s000
                                 + 0.701745 w_k1_IBM_s000
                                 + 1.07005 w_k1_INTC_s000
                                 + 0.694705 w_k1_KO_s000
                                 + 0.695187 w_k1_MAT_s000
                                 + 0.880292 w_k1_MSFT_s000
                                 + 0.910383 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.796687 w_k1_ADM_s000
                                 + 0.781903 w_k1_AXP_s000
                                 + 0.600238 w_k1_DIS_s000
                                 + 0.863301 w_k1_FDX_s000
                                 + 0.567174 w_k1_IBM_s000
                                 + 0.962007 w_k1_INTC_s000
                                 + 0.562496 w_k1_KO_s000
                                 + 0.55281 w_k1_MAT_s000
                                 + 0.783193 w_k1_MSFT_s000
                                 + 0.88924 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.39589 w_k1_ADM_s000
                                 + 1.36401 w_k1_AXP_s000
                                 + 1.61551 w_k1_DIS_s000
                                 + 1.30065 w_k1_FDX_s000
                                 + 1.15534 w_k1_IBM_s000
                                 + 1.37885 w_k1_INTC_s000
                                 + 1.30721 w_k1_KO_s000 + 1.31706 w_k1_MAT_s000
                                 + 1.29264 w_k1_MSFT_s000
                                 + 1.37682 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.923998 w_k1_ADM_s001
                                 + 1.22549 w_k1_AXP_s001
                                 + 0.641729 w_k1_DIS_s001
                                 + 0.877083 w_k1_FDX_s001
                                 + 1.26428 w_k1_IBM_s001
                                 + 0.879346 w_k1_INTC_s001
                                 + 0.931377 w_k1_KO_s001
                                 + 1.51247 w_k1_MAT_s001
                                 + 0.818088 w_k1_MSFT_s001
                                 + 0.90872 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.09281 w_k1_ADM_s001
                                 + 1.14115 w_k1_AXP_s001
                                 + 1.02148 w_k1_DIS_s001
                                 + 1.04246 w_k1_FDX_s001
                                 + 1.01367 w_k1_IBM_s001
                                 + 0.928551 w_k1_INTC_s001
                                 + 0.960928 w_k1_KO_s001
                                 + 1.18772 w_k1_MAT_s001
                                 + 1.0054 w_k1_MSFT_s001 + 1.01022 w_k1_PG_s001
                                 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.14557 w_k1_ADM_s001
                                 + 1.45109 w_k1_AXP_s001
                                 + 1.77892 w_k1_DIS_s001
                                 + 1.34441 w_k1_FDX_s001 + 1.0853 w_k1_IBM_s001
                                 + 1.43289 w_k1_INTC_s001
                                 + 1.19679 w_k1_KO_s001
                                 + 0.914533 w_k1_MAT_s001
                                 + 1.45329 w_k1_MSFT_s001
                                 + 1.11281 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.48621 w_k1_ADM_s001
                                 + 1.136 w_k1_AXP_s001 + 1.2508 w_k1_DIS_s001
                                 + 1.24515 w_k1_FDX_s001
                                 + 0.732037 w_k1_IBM_s001
                                 + 1.10599 w_k1_INTC_s001
                                 + 0.979436 w_k1_KO_s001
                                 + 0.938645 w_k1_MAT_s001
                                 + 1.43286 w_k1_MSFT_s001
                                 + 1.14871 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.770868 w_k1_ADM_s001
                                 + 0.85208 w_k1_AXP_s001
                                 + 0.651948 w_k1_DIS_s001
                                 + 0.797222 w_k1_FDX_s001
                                 + 0.720914 w_k1_IBM_s001
                                 + 0.881558 w_k1_INTC_s001
                                 + 0.666031 w_k1_KO_s001
                                 + 0.597769 w_k1_MAT_s001
                                 + 0.870058 w_k1_MSFT_s001
                                 + 0.791247 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.999342 w_k1_ADM_s001
                                 + 1.24679 w_k1_AXP_s001
                                 + 0.837555 w_k1_DIS_s001
                                 + 0.980328 w_k1_FDX_s001
                                 + 0.816844 w_k1_IBM_s001
                                 + 1.02835 w_k1_INTC_s001
                                 + 0.796419 w_k1_KO_s001
                                 + 1.12311 w_k1_MAT_s001
                                 + 1.03567 w_k1_MSFT_s001
                                 + 0.983616 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.05734 w_k1_ADM_s001
                                 + 0.963048 w_k1_AXP_s001
                                 + 0.964639 w_k1_DIS_s001
                                 + 0.856279 w_k1_FDX_s001
                                 + 1.51936 w_k1_IBM_s001
                                 + 0.835909 w_k1_INTC_s001
                                 + 1.25462 w_k1_KO_s001 + 1.58377 w_k1_MAT_s001
                                 + 1.18622 w_k1_MSFT_s001
                                 + 1.07883 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.21896 w_k1_ADM_s001
                                 + 1.56564 w_k1_AXP_s001
                                 + 1.53693 w_k1_DIS_s001
                                 + 1.20069 w_k1_FDX_s001 + 1.4803 w_k1_IBM_s001
                                 + 1.07406 w_k1_INTC_s001
                                 + 1.45342 w_k1_KO_s001 + 1.75144 w_k1_MAT_s001
                                 + 1.22238 w_k1_MSFT_s001
                                 + 1.20085 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.08679 w_k1_ADM_s002
                                 + 0.853194 w_k1_AXP_s002
                                 + 0.683833 w_k1_DIS_s002
                                 + 0.848405 w_k1_FDX_s002
                                 + 0.8994 w_k1_IBM_s002
                                 + 0.76506 w_k1_INTC_s002
                                 + 0.774996 w_k1_KO_s002
                                 + 0.77213 w_k1_MAT_s002
                                 + 0.912683 w_k1_MSFT_s002
                                 + 0.977584 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.12758 w_k1_ADM_s002
                                 + 1.44506 w_k1_AXP_s002 + 1.9309 w_k1_DIS_s002
                                 + 1.55743 w_k1_FDX_s002
                                 + 0.821912 w_k1_IBM_s002
                                 + 1.26734 w_k1_INTC_s002
                                 + 1.05902 w_k1_KO_s002
                                 + 0.937672 w_k1_MAT_s002
                                 + 1.39982 w_k1_MSFT_s002
                                 + 1.26516 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.849493 w_k1_ADM_s002
                                 + 0.581856 w_k1_AXP_s002
                                 + 0.546043 w_k1_DIS_s002
                                 + 0.676477 w_k1_FDX_s002
                                 + 0.733798 w_k1_IBM_s002
                                 + 0.59739 w_k1_INTC_s002
                                 + 0.704125 w_k1_KO_s002
                                 + 0.610285 w_k1_MAT_s002
                                 + 0.672617 w_k1_MSFT_s002
                                 + 0.818188 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.05642 w_k1_ADM_s002
                                 + 0.940334 w_k1_AXP_s002
                                 + 1.21791 w_k1_DIS_s002
                                 + 1.17425 w_k1_FDX_s002
                                 + 0.989353 w_k1_IBM_s002
                                 + 1.11801 w_k1_INTC_s002
                                 + 1.10554 w_k1_KO_s002
                                 + 0.808903 w_k1_MAT_s002
                                 + 1.22484 w_k1_MSFT_s002
                                 + 1.18938 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.925782 w_k1_ADM_s002
                                 + 1.05224 w_k1_AXP_s002
                                 + 0.966778 w_k1_DIS_s002
                                 + 0.964225 w_k1_FDX_s002
                                 + 1.24196 w_k1_IBM_s002
                                 + 0.807575 w_k1_INTC_s002
                                 + 0.886143 w_k1_KO_s002 + 1.6444 w_k1_MAT_s002
                                 + 0.899436 w_k1_MSFT_s002
                                 + 1.12207 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.06586 w_k1_ADM_s002
                                 + 0.910294 w_k1_AXP_s002
                                 + 1.1618 w_k1_DIS_s002 + 1.08514 w_k1_FDX_s002
                                 + 0.784139 w_k1_IBM_s002
                                 + 0.969273 w_k1_INTC_s002
                                 + 0.887792 w_k1_KO_s002
                                 + 0.541324 w_k1_MAT_s002
                                 + 0.963196 w_k1_MSFT_s002
                                 + 1.01363 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.890224 w_k1_ADM_s002
                                 + 1.34729 w_k1_AXP_s002
                                 + 1.87782 w_k1_DIS_s002
                                 + 1.25896 w_k1_FDX_s002
                                 + 1.73247 w_k1_IBM_s002
                                 + 1.3546 w_k1_INTC_s002 + 1.37719 w_k1_KO_s002
                                 + 1.48969 w_k1_MAT_s002
                                 + 1.18938 w_k1_MSFT_s002
                                 + 1.28701 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.984646 w_k1_ADM_s002
                                 + 1.00637 w_k1_AXP_s002 + 1.0036 w_k1_DIS_s002
                                 + 0.899374 w_k1_FDX_s002
                                 + 1.33242 w_k1_IBM_s002
                                 + 0.802822 w_k1_INTC_s002
                                 + 0.99467 w_k1_KO_s002 + 1.25913 w_k1_MAT_s002
                                 + 0.916686 w_k1_MSFT_s002
                                 + 1.04767 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.36313 w_k1_ADM_s003
                                 + 1.34776 w_k1_AXP_s003
                                 + 1.46474 w_k1_DIS_s003 + 1.3778 w_k1_FDX_s003
                                 + 0.79296 w_k1_IBM_s003
                                 + 0.920017 w_k1_INTC_s003
                                 + 1.12095 w_k1_KO_s003 + 0.91842 w_k1_MAT_s003
                                 + 1.24583 w_k1_MSFT_s003
                                 + 1.14647 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.17152 w_k1_ADM_s003
                                 + 1.26238 w_k1_AXP_s003
                                 + 1.00914 w_k1_DIS_s003
                                 + 1.06834 w_k1_FDX_s003 + 1.1373 w_k1_IBM_s003
                                 + 1.07992 w_k1_INTC_s003
                                 + 1.14103 w_k1_KO_s003 + 1.3623 w_k1_MAT_s003
                                 + 1.17589 w_k1_MSFT_s003
                                 + 1.11974 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.862256 w_k1_ADM_s003
                                 + 0.840843 w_k1_AXP_s003
                                 + 0.839739 w_k1_DIS_s003
                                 + 0.80226 w_k1_FDX_s003
                                 + 0.989471 w_k1_IBM_s003
                                 + 0.734134 w_k1_INTC_s003
                                 + 1.01487 w_k1_KO_s003
                                 + 0.894497 w_k1_MAT_s003
                                 + 0.844396 w_k1_MSFT_s003
                                 + 1.00328 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.965935 w_k1_ADM_s003
                                 + 0.71471 w_k1_AXP_s003
                                 + 0.716938 w_k1_DIS_s003
                                 + 0.763341 w_k1_FDX_s003
                                 + 0.712685 w_k1_IBM_s003
                                 + 0.758185 w_k1_INTC_s003
                                 + 0.881371 w_k1_KO_s003
                                 + 0.730279 w_k1_MAT_s003
                                 + 0.743538 w_k1_MSFT_s003
                                 + 0.847442 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.20814 w_k1_ADM_s003
                                 + 1.22047 w_k1_AXP_s003
                                 + 1.07209 w_k1_DIS_s003 + 1.1388 w_k1_FDX_s003
                                 + 0.870478 w_k1_IBM_s003
                                 + 0.86115 w_k1_INTC_s003
                                 + 1.09399 w_k1_KO_s003
                                 + 0.960354 w_k1_MAT_s003
                                 + 1.00245 w_k1_MSFT_s003
                                 + 1.05148 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.9229 w_k1_ADM_s003
                                 + 0.762347 w_k1_AXP_s003
                                 + 1.11324 w_k1_DIS_s003
                                 + 0.965406 w_k1_FDX_s003
                                 + 0.856623 w_k1_IBM_s003
                                 + 0.893533 w_k1_INTC_s003
                                 + 1.04046 w_k1_KO_s003
                                 + 0.690683 w_k1_MAT_s003
                                 + 1.01063 w_k1_MSFT_s003
                                 + 0.955457 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.02701 w_k1_ADM_s003
                                 + 1.37242 w_k1_AXP_s003
                                 + 1.42064 w_k1_DIS_s003
                                 + 1.13731 w_k1_FDX_s003
                                 + 1.61364 w_k1_IBM_s003
                                 + 1.19782 w_k1_INTC_s003
                                 + 1.51386 w_k1_KO_s003 + 1.70079 w_k1_MAT_s003
                                 + 1.16424 w_k1_MSFT_s003
                                 + 1.12801 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.01436 w_k1_ADM_s003
                                 + 0.955166 w_k1_AXP_s003
                                 + 0.856685 w_k1_DIS_s003
                                 + 0.867668 w_k1_FDX_s003
                                 + 1.2914 w_k1_IBM_s003
                                 + 0.789326 w_k1_INTC_s003
                                 + 1.07534 w_k1_KO_s003 + 1.09961 w_k1_MAT_s003
                                 + 0.81248 w_k1_MSFT_s003
                                 + 0.942289 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.13785 w_k1_ADM_s004
                                 + 0.69929 w_k1_AXP_s004
                                 + 0.864858 w_k1_DIS_s004
                                 + 0.783954 w_k1_FDX_s004
                                 + 0.977118 w_k1_IBM_s004
                                 + 0.77456 w_k1_INTC_s004
                                 + 0.883374 w_k1_KO_s004
                                 + 0.664579 w_k1_MAT_s004
                                 + 0.843663 w_k1_MSFT_s004
                                 + 0.912193 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.02308 w_k1_ADM_s004
                                 + 1.02988 w_k1_AXP_s004
                                 + 0.935444 w_k1_DIS_s004
                                 + 0.831141 w_k1_FDX_s004
                                 + 1.38305 w_k1_IBM_s004
                                 + 0.883232 w_k1_INTC_s004
                                 + 1.09522 w_k1_KO_s004 + 1.17345 w_k1_MAT_s004
                                 + 0.862916 w_k1_MSFT_s004
                                 + 0.985452 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.06628 w_k1_ADM_s004
                                 + 0.693205 w_k1_AXP_s004
                                 + 0.564694 w_k1_DIS_s004
                                 + 0.737345 w_k1_FDX_s004
                                 + 0.722075 w_k1_IBM_s004
                                 + 0.6264 w_k1_INTC_s004
                                 + 0.687603 w_k1_KO_s004
                                 + 0.760344 w_k1_MAT_s004
                                 + 0.782376 w_k1_MSFT_s004
                                 + 0.878238 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.41904 w_k1_ADM_s004
                                 + 1.55439 w_k1_AXP_s004
                                 + 1.62893 w_k1_DIS_s004
                                 + 1.12368 w_k1_FDX_s004
                                 + 1.60809 w_k1_IBM_s004
                                 + 1.2155 w_k1_INTC_s004 + 1.39059 w_k1_KO_s004
                                 + 1.19243 w_k1_MAT_s004
                                 + 1.19883 w_k1_MSFT_s004
                                 + 1.17222 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.24252 w_k1_ADM_s004
                                 + 1.11168 w_k1_AXP_s004
                                 + 1.06575 w_k1_DIS_s004
                                 + 0.941417 w_k1_FDX_s004
                                 + 1.10992 w_k1_IBM_s004
                                 + 0.884589 w_k1_INTC_s004
                                 + 1.13881 w_k1_KO_s004 + 1.12715 w_k1_MAT_s004
                                 + 1.01385 w_k1_MSFT_s004
                                 + 1.17674 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.71433 w_k1_ADM_s004
                                 + 1.42923 w_k1_AXP_s004
                                 + 1.95266 w_k1_DIS_s004
                                 + 1.41271 w_k1_FDX_s004
                                 + 0.977102 w_k1_IBM_s004
                                 + 0.957868 w_k1_INTC_s004
                                 + 1.23703 w_k1_KO_s004
                                 + 0.950466 w_k1_MAT_s004
                                 + 1.21789 w_k1_MSFT_s004
                                 + 1.21048 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.995403 w_k1_ADM_s004
                                 + 0.901328 w_k1_AXP_s004
                                 + 0.726242 w_k1_DIS_s004
                                 + 0.764701 w_k1_FDX_s004
                                 + 0.920711 w_k1_IBM_s004
                                 + 0.608933 w_k1_INTC_s004
                                 + 0.819436 w_k1_KO_s004
                                 + 0.894371 w_k1_MAT_s004
                                 + 0.703894 w_k1_MSFT_s004
                                 + 0.903604 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.24991 w_k1_ADM_s004
                                 + 1.04924 w_k1_AXP_s004
                                 + 1.16938 w_k1_DIS_s004
                                 + 1.08468 w_k1_FDX_s004
                                 + 0.819115 w_k1_IBM_s004
                                 + 1.12034 w_k1_INTC_s004
                                 + 1.00262 w_k1_KO_s004
                                 + 0.935679 w_k1_MAT_s004
                                 + 1.04263 w_k1_MSFT_s004
                                 + 0.933461 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.26138 w_k1_ADM_s005
                                 + 1.31565 w_k1_AXP_s005
                                 + 1.45194 w_k1_DIS_s005
                                 + 1.30439 w_k1_FDX_s005 + 1.4401 w_k1_IBM_s005
                                 + 1.28483 w_k1_INTC_s005
                                 + 1.02771 w_k1_KO_s005 + 1.21302 w_k1_MAT_s005
                                 + 1.12479 w_k1_MSFT_s005
                                 + 0.977396 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.905871 w_k1_ADM_s005
                                 + 0.729014 w_k1_AXP_s005
                                 + 0.738156 w_k1_DIS_s005
                                 + 0.807161 w_k1_FDX_s005
                                 + 1.06782 w_k1_IBM_s005
                                 + 0.945389 w_k1_INTC_s005
                                 + 0.785329 w_k1_KO_s005
                                 + 0.847608 w_k1_MAT_s005
                                 + 0.782704 w_k1_MSFT_s005
                                 + 0.891754 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.07442 w_k1_ADM_s005
                                 + 0.803696 w_k1_AXP_s005
                                 + 0.66827 w_k1_DIS_s005
                                 + 0.809116 w_k1_FDX_s005
                                 + 0.591974 w_k1_IBM_s005
                                 + 0.856522 w_k1_INTC_s005
                                 + 0.597629 w_k1_KO_s005
                                 + 0.61116 w_k1_MAT_s005
                                 + 0.779563 w_k1_MSFT_s005
                                 + 0.792722 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.26743 w_k1_ADM_s005
                                 + 1.56582 w_k1_AXP_s005
                                 + 1.83236 w_k1_DIS_s005 + 1.5439 w_k1_FDX_s005
                                 + 1.20082 w_k1_IBM_s005
                                 + 1.74658 w_k1_INTC_s005
                                 + 1.13832 w_k1_KO_s005
                                 + 0.908891 w_k1_MAT_s005
                                 + 1.41238 w_k1_MSFT_s005
                                 + 1.12452 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.917984 w_k1_ADM_s005
                                 + 0.835283 w_k1_AXP_s005
                                 + 1.37892 w_k1_DIS_s005
                                 + 1.08238 w_k1_FDX_s005
                                 + 1.31825 w_k1_IBM_s005
                                 + 1.24838 w_k1_INTC_s005
                                 + 1.10644 w_k1_KO_s005 + 0.9161 w_k1_MAT_s005
                                 + 1.27791 w_k1_MSFT_s005
                                 + 1.06786 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.65045 w_k1_ADM_s005
                                 + 0.499527 w_k1_AXP_s005
                                 + 0.615833 w_k1_DIS_s005
                                 + 0.634697 w_k1_FDX_s005
                                 + 0.777539 w_k1_IBM_s005
                                 + 0.892832 w_k1_INTC_s005
                                 + 0.658761 w_k1_KO_s005
                                 + 0.499394 w_k1_MAT_s005
                                 + 0.757725 w_k1_MSFT_s005
                                 + 0.794921 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.03234 w_k1_ADM_s005
                                 + 0.865786 w_k1_AXP_s005
                                 + 1.01573 w_k1_DIS_s005
                                 + 1.02048 w_k1_FDX_s005
                                 + 0.819929 w_k1_IBM_s005
                                 + 1.03549 w_k1_INTC_s005
                                 + 0.833765 w_k1_KO_s005
                                 + 0.499748 w_k1_MAT_s005
                                 + 1.00572 w_k1_MSFT_s005
                                 + 0.993236 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.11024 w_k1_ADM_s005
                                 + 1.16616 w_k1_AXP_s005
                                 + 0.957503 w_k1_DIS_s005
                                 + 1.12072 w_k1_FDX_s005
                                 + 1.07839 w_k1_IBM_s005
                                 + 1.11693 w_k1_INTC_s005
                                 + 0.860811 w_k1_KO_s005
                                 + 0.717745 w_k1_MAT_s005
                                 + 1.00465 w_k1_MSFT_s005
                                 + 1.06076 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.78591 w_k1_ADM_s006
                                 + 0.686357 w_k1_AXP_s006
                                 + 0.784861 w_k1_DIS_s006
                                 + 0.850258 w_k1_FDX_s006
                                 + 0.757443 w_k1_IBM_s006
                                 + 0.899927 w_k1_INTC_s006
                                 + 0.888311 w_k1_KO_s006
                                 + 0.730341 w_k1_MAT_s006
                                 + 0.888107 w_k1_MSFT_s006
                                 + 1.06184 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.33242 w_k1_ADM_s006
                                 + 1.4557 w_k1_AXP_s006 + 1.45476 w_k1_DIS_s006
                                 + 1.2966 w_k1_FDX_s006 + 1.1046 w_k1_IBM_s006
                                 + 1.28614 w_k1_INTC_s006
                                 + 1.57506 w_k1_KO_s006 + 1.05046 w_k1_MAT_s006
                                 + 1.30457 w_k1_MSFT_s006
                                 + 1.26478 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.680127 w_k1_ADM_s006
                                 + 0.64016 w_k1_AXP_s006
                                 + 0.764063 w_k1_DIS_s006
                                 + 0.711062 w_k1_FDX_s006
                                 + 0.90161 w_k1_IBM_s006
                                 + 0.800098 w_k1_INTC_s006
                                 + 0.896428 w_k1_KO_s006
                                 + 0.733127 w_k1_MAT_s006
                                 + 0.91074 w_k1_MSFT_s006
                                 + 0.832764 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.84063 w_k1_ADM_s006
                                 + 1.04619 w_k1_AXP_s006
                                 + 0.739234 w_k1_DIS_s006
                                 + 0.912731 w_k1_FDX_s006
                                 + 1.08832 w_k1_IBM_s006
                                 + 1.01001 w_k1_INTC_s006
                                 + 0.952019 w_k1_KO_s006
                                 + 1.14078 w_k1_MAT_s006
                                 + 0.830198 w_k1_MSFT_s006
                                 + 0.947506 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.961647 w_k1_ADM_s006
                                 + 1.06256 w_k1_AXP_s006
                                 + 1.07759 w_k1_DIS_s006
                                 + 0.922163 w_k1_FDX_s006
                                 + 1.38644 w_k1_IBM_s006
                                 + 1.12412 w_k1_INTC_s006
                                 + 1.10464 w_k1_KO_s006 + 1.07352 w_k1_MAT_s006
                                 + 0.982805 w_k1_MSFT_s006
                                 + 0.970904 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.74439 w_k1_ADM_s006
                                 + 0.664456 w_k1_AXP_s006
                                 + 0.608759 w_k1_DIS_s006
                                 + 0.747518 w_k1_FDX_s006
                                 + 0.686136 w_k1_IBM_s006
                                 + 0.764052 w_k1_INTC_s006
                                 + 0.830926 w_k1_KO_s006
                                 + 0.74439 w_k1_MAT_s006
                                 + 0.776012 w_k1_MSFT_s006
                                 + 0.84438 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.994366 w_k1_ADM_s006
                                 + 0.940205 w_k1_AXP_s006
                                 + 0.784033 w_k1_DIS_s006
                                 + 0.94136 w_k1_FDX_s006
                                 + 0.656219 w_k1_IBM_s006
                                 + 0.886561 w_k1_INTC_s006
                                 + 0.814172 w_k1_KO_s006
                                 + 0.693241 w_k1_MAT_s006
                                 + 1.04293 w_k1_MSFT_s006
                                 + 0.790471 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.895811 w_k1_ADM_s006
                                 + 0.95616 w_k1_AXP_s006
                                 + 1.10864 w_k1_DIS_s006
                                 + 1.03757 w_k1_FDX_s006
                                 + 0.956291 w_k1_IBM_s006
                                 + 0.814911 w_k1_INTC_s006
                                 + 1.17723 w_k1_KO_s006
                                 + 0.912233 w_k1_MAT_s006
                                 + 1.07646 w_k1_MSFT_s006
                                 + 1.03752 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.958708 w_k1_ADM_s007
                                 + 0.77263 w_k1_AXP_s007
                                 + 0.817301 w_k1_DIS_s007
                                 + 0.92061 w_k1_FDX_s007
                                 + 0.65235 w_k1_IBM_s007
                                 + 0.827839 w_k1_INTC_s007
                                 + 0.902804 w_k1_KO_s007
                                 + 0.568991 w_k1_MAT_s007
                                 + 0.889696 w_k1_MSFT_s007
                                 + 0.799803 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.967042 w_k1_ADM_s007
                                 + 1.02968 w_k1_AXP_s007
                                 + 1.37277 w_k1_DIS_s007
                                 + 1.29279 w_k1_FDX_s007
                                 + 0.930817 w_k1_IBM_s007
                                 + 1.31983 w_k1_INTC_s007
                                 + 1.00571 w_k1_KO_s007
                                 + 0.554788 w_k1_MAT_s007
                                 + 1.10983 w_k1_MSFT_s007
                                 + 0.987876 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.961081 w_k1_ADM_s007
                                 + 0.832109 w_k1_AXP_s007
                                 + 0.883343 w_k1_DIS_s007
                                 + 0.881096 w_k1_FDX_s007
                                 + 1.07858 w_k1_IBM_s007
                                 + 0.793381 w_k1_INTC_s007
                                 + 1.0663 w_k1_KO_s007 + 0.687909 w_k1_MAT_s007
                                 + 0.82464 w_k1_MSFT_s007
                                 + 0.90106 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.746927 w_k1_ADM_s007
                                 + 0.744768 w_k1_AXP_s007
                                 + 0.666266 w_k1_DIS_s007
                                 + 0.718933 w_k1_FDX_s007
                                 + 0.886253 w_k1_IBM_s007
                                 + 0.756216 w_k1_INTC_s007
                                 + 0.908446 w_k1_KO_s007
                                 + 0.746982 w_k1_MAT_s007
                                 + 0.796272 w_k1_MSFT_s007
                                 + 0.79011 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.26292 w_k1_ADM_s007
                                 + 1.56343 w_k1_AXP_s007
                                 + 1.73263 w_k1_DIS_s007
                                 + 1.50192 w_k1_FDX_s007
                                 + 1.31879 w_k1_IBM_s007
                                 + 1.17434 w_k1_INTC_s007
                                 + 1.51414 w_k1_KO_s007 + 1.47413 w_k1_MAT_s007
                                 + 1.38748 w_k1_MSFT_s007
                                 + 1.17853 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.1736 w_k1_ADM_s007
                                 + 1.15268 w_k1_AXP_s007
                                 + 1.32096 w_k1_DIS_s007
                                 + 1.12365 w_k1_FDX_s007
                                 + 1.42016 w_k1_IBM_s007
                                 + 1.18648 w_k1_INTC_s007
                                 + 1.24997 w_k1_KO_s007
                                 + 0.890828 w_k1_MAT_s007
                                 + 1.07298 w_k1_MSFT_s007
                                 + 1.04927 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.23451 w_k1_ADM_s007
                                 + 1.38091 w_k1_AXP_s007
                                 + 1.19699 w_k1_DIS_s007
                                 + 1.26821 w_k1_FDX_s007
                                 + 1.00601 w_k1_IBM_s007
                                 + 0.898997 w_k1_INTC_s007
                                 + 1.01938 w_k1_KO_s007
                                 + 0.881896 w_k1_MAT_s007
                                 + 0.992232 w_k1_MSFT_s007
                                 + 1.04279 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.921943 w_k1_ADM_s007
                                 + 0.953494 w_k1_AXP_s007
                                 + 0.97906 w_k1_DIS_s007
                                 + 0.945019 w_k1_FDX_s007
                                 + 1.46054 w_k1_IBM_s007
                                 + 0.756616 w_k1_INTC_s007
                                 + 1.0468 w_k1_KO_s007 + 1.2226 w_k1_MAT_s007
                                 + 0.859039 w_k1_MSFT_s007
                                 + 0.912885 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.40173 w_k1_ADM_s010
                                 + 1.32595 w_k1_AXP_s010
                                 + 1.34145 w_k1_DIS_s010
                                 + 1.29234 w_k1_FDX_s010
                                 + 1.02883 w_k1_IBM_s010
                                 + 1.15204 w_k1_INTC_s010
                                 + 1.06621 w_k1_KO_s010
                                 + 0.911832 w_k1_MAT_s010
                                 + 1.13264 w_k1_MSFT_s010
                                 + 1.08291 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.848804 w_k1_ADM_s010
                                 + 0.667621 w_k1_AXP_s010
                                 + 1.03544 w_k1_DIS_s010
                                 + 0.905426 w_k1_FDX_s010
                                 + 0.862042 w_k1_IBM_s010
                                 + 1.03418 w_k1_INTC_s010
                                 + 0.95393 w_k1_KO_s010
                                 + 0.492835 w_k1_MAT_s010
                                 + 0.974364 w_k1_MSFT_s010
                                 + 1.01525 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.804094 w_k1_ADM_s010
                                 + 0.739648 w_k1_AXP_s010
                                 + 0.666944 w_k1_DIS_s010
                                 + 0.739408 w_k1_FDX_s010
                                 + 0.682703 w_k1_IBM_s010
                                 + 0.799058 w_k1_INTC_s010
                                 + 0.750287 w_k1_KO_s010
                                 + 0.794999 w_k1_MAT_s010
                                 + 0.795362 w_k1_MSFT_s010
                                 + 0.901344 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.20705 w_k1_ADM_s010
                                 + 1.3697 w_k1_AXP_s010 + 1.64598 w_k1_DIS_s010
                                 + 1.40552 w_k1_FDX_s010
                                 + 1.38533 w_k1_IBM_s010
                                 + 1.44677 w_k1_INTC_s010
                                 + 1.43059 w_k1_KO_s010 + 1.3841 w_k1_MAT_s010
                                 + 1.34049 w_k1_MSFT_s010 + 1.2958 w_k1_PG_s010
                                 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.07134 w_k1_ADM_s010
                                 + 0.883428 w_k1_AXP_s010
                                 + 0.766611 w_k1_DIS_s010
                                 + 0.78464 w_k1_FDX_s010
                                 + 1.16571 w_k1_IBM_s010
                                 + 0.921393 w_k1_INTC_s010
                                 + 0.87157 w_k1_KO_s010
                                 + 0.782044 w_k1_MAT_s010
                                 + 0.831113 w_k1_MSFT_s010
                                 + 0.982715 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.03336 w_k1_ADM_s010
                                 + 0.812816 w_k1_AXP_s010
                                 + 1.19169 w_k1_DIS_s010
                                 + 1.17056 w_k1_FDX_s010
                                 + 0.832406 w_k1_IBM_s010
                                 + 1.03378 w_k1_INTC_s010
                                 + 0.952504 w_k1_KO_s010
                                 + 0.536912 w_k1_MAT_s010
                                 + 1.17713 w_k1_MSFT_s010
                                 + 1.14644 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.06773 w_k1_ADM_s010
                                 + 1.09658 w_k1_AXP_s010
                                 + 1.06813 w_k1_DIS_s010
                                 + 0.968479 w_k1_FDX_s010
                                 + 1.55177 w_k1_IBM_s010
                                 + 1.01122 w_k1_INTC_s010
                                 + 1.32292 w_k1_KO_s010 + 1.60058 w_k1_MAT_s010
                                 + 1.06999 w_k1_MSFT_s010
                                 + 1.21007 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.18299 w_k1_ADM_s010
                                 + 1.06068 w_k1_AXP_s010
                                 + 1.05818 w_k1_DIS_s010
                                 + 1.03208 w_k1_FDX_s010
                                 + 1.05868 w_k1_IBM_s010 + 1.033 w_k1_INTC_s010
                                 + 1.01093 w_k1_KO_s010
                                 + 0.824211 w_k1_MAT_s010
                                 + 1.00545 w_k1_MSFT_s010
                                 + 1.02238 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.894794 w_k1_ADM_s011
                                 + 0.925759 w_k1_AXP_s011
                                 + 0.884495 w_k1_DIS_s011
                                 + 0.846003 w_k1_FDX_s011
                                 + 1.48422 w_k1_IBM_s011
                                 + 0.848876 w_k1_INTC_s011
                                 + 1.07328 w_k1_KO_s011 + 1.26158 w_k1_MAT_s011
                                 + 0.834326 w_k1_MSFT_s011
                                 + 1.03573 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.14531 w_k1_ADM_s011
                                 + 1.06934 w_k1_AXP_s011
                                 + 1.00458 w_k1_DIS_s011
                                 + 1.10059 w_k1_FDX_s011
                                 + 0.820255 w_k1_IBM_s011
                                 + 1.15098 w_k1_INTC_s011
                                 + 0.825186 w_k1_KO_s011
                                 + 0.619972 w_k1_MAT_s011
                                 + 1.0529 w_k1_MSFT_s011 + 1.0547 w_k1_PG_s011
                                 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.28088 w_k1_ADM_s011
                                 + 1.19695 w_k1_AXP_s011
                                 + 1.42947 w_k1_DIS_s011
                                 + 1.36904 w_k1_FDX_s011
                                 + 0.924096 w_k1_IBM_s011
                                 + 1.0989 w_k1_INTC_s011 + 1.23598 w_k1_KO_s011
                                 + 1.23011 w_k1_MAT_s011
                                 + 1.25257 w_k1_MSFT_s011
                                 + 1.29689 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.936918 w_k1_ADM_s011
                                 + 0.705315 w_k1_AXP_s011
                                 + 0.670355 w_k1_DIS_s011
                                 + 0.804416 w_k1_FDX_s011
                                 + 0.901814 w_k1_IBM_s011
                                 + 0.709764 w_k1_INTC_s011
                                 + 0.82262 w_k1_KO_s011
                                 + 0.890875 w_k1_MAT_s011
                                 + 0.799656 w_k1_MSFT_s011
                                 + 1.02234 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.43897 w_k1_ADM_s011
                                 + 1.34245 w_k1_AXP_s011
                                 + 1.30543 w_k1_DIS_s011
                                 + 1.27892 w_k1_FDX_s011
                                 + 1.43702 w_k1_IBM_s011
                                 + 1.20042 w_k1_INTC_s011
                                 + 1.15538 w_k1_KO_s011
                                 + 0.898847 w_k1_MAT_s011
                                 + 1.04795 w_k1_MSFT_s011 + 1.3615 w_k1_PG_s011
                                 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.770555 w_k1_ADM_s011
                                 + 0.886337 w_k1_AXP_s011
                                 + 0.851952 w_k1_DIS_s011
                                 + 0.937198 w_k1_FDX_s011
                                 + 1.19183 w_k1_IBM_s011
                                 + 1.01916 w_k1_INTC_s011
                                 + 0.940436 w_k1_KO_s011
                                 + 1.07308 w_k1_MAT_s011
                                 + 0.878461 w_k1_MSFT_s011
                                 + 1.04668 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.982865 w_k1_ADM_s011
                                 + 0.689772 w_k1_AXP_s011
                                 + 0.588943 w_k1_DIS_s011
                                 + 0.839487 w_k1_FDX_s011
                                 + 0.614937 w_k1_IBM_s011
                                 + 0.699687 w_k1_INTC_s011
                                 + 0.641428 w_k1_KO_s011 + 0.8719 w_k1_MAT_s011
                                 + 0.708623 w_k1_MSFT_s011
                                 + 0.863957 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.955391 w_k1_ADM_s011
                                 + 0.860741 w_k1_AXP_s011
                                 + 0.891062 w_k1_DIS_s011
                                 + 0.953471 w_k1_FDX_s011
                                 + 0.94465 w_k1_IBM_s011
                                 + 0.839486 w_k1_INTC_s011
                                 + 0.989382 w_k1_KO_s011
                                 + 0.721014 w_k1_MAT_s011
                                 + 0.929065 w_k1_MSFT_s011
                                 + 0.991621 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.734384 w_k1_ADM_s012
                                 + 0.820662 w_k1_AXP_s012
                                 + 0.923532 w_k1_DIS_s012
                                 + 0.909295 w_k1_FDX_s012
                                 + 0.791634 w_k1_IBM_s012
                                 + 1.17241 w_k1_INTC_s012
                                 + 0.799818 w_k1_KO_s012
                                 + 1.04278 w_k1_MAT_s012
                                 + 0.95676 w_k1_MSFT_s012
                                 + 0.825428 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.767805 w_k1_ADM_s012
                                 + 0.928437 w_k1_AXP_s012
                                 + 0.987997 w_k1_DIS_s012
                                 + 0.842497 w_k1_FDX_s012
                                 + 1.35875 w_k1_IBM_s012
                                 + 1.06046 w_k1_INTC_s012
                                 + 1.06136 w_k1_KO_s012 + 1.18534 w_k1_MAT_s012
                                 + 0.907101 w_k1_MSFT_s012
                                 + 0.854861 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.654439 w_k1_ADM_s012
                                 + 0.638223 w_k1_AXP_s012
                                 + 0.615788 w_k1_DIS_s012
                                 + 0.743864 w_k1_FDX_s012
                                 + 0.652343 w_k1_IBM_s012
                                 + 0.966213 w_k1_INTC_s012
                                 + 0.669187 w_k1_KO_s012
                                 + 0.922241 w_k1_MAT_s012
                                 + 0.839829 w_k1_MSFT_s012
                                 + 0.733257 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.09424 w_k1_ADM_s012
                                 + 1.39111 w_k1_AXP_s012
                                 + 1.66053 w_k1_DIS_s012
                                 + 1.24737 w_k1_FDX_s012
                                 + 1.58498 w_k1_IBM_s012
                                 + 1.30922 w_k1_INTC_s012
                                 + 1.23746 w_k1_KO_s012 + 1.56263 w_k1_MAT_s012
                                 + 1.13042 w_k1_MSFT_s012 + 1.0614 w_k1_PG_s012
                                 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.740116 w_k1_ADM_s012
                                 + 0.824339 w_k1_AXP_s012
                                 + 0.811772 w_k1_DIS_s012
                                 + 0.81458 w_k1_FDX_s012
                                 + 1.06267 w_k1_IBM_s012
                                 + 0.852967 w_k1_INTC_s012
                                 + 0.892863 w_k1_KO_s012
                                 + 1.33023 w_k1_MAT_s012
                                 + 0.708133 w_k1_MSFT_s012
                                 + 0.870383 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.829479 w_k1_ADM_s012
                                 + 1.09181 w_k1_AXP_s012
                                 + 1.30821 w_k1_DIS_s012
                                 + 1.06797 w_k1_FDX_s012
                                 + 1.06991 w_k1_IBM_s012
                                 + 1.27719 w_k1_INTC_s012
                                 + 1.15623 w_k1_KO_s012
                                 + 0.945607 w_k1_MAT_s012
                                 + 1.04031 w_k1_MSFT_s012
                                 + 0.98012 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.824468 w_k1_ADM_s012
                                 + 1.31258 w_k1_AXP_s012 + 1.4541 w_k1_DIS_s012
                                 + 1.44826 w_k1_FDX_s012
                                 + 0.928974 w_k1_IBM_s012
                                 + 1.23439 w_k1_INTC_s012
                                 + 1.10376 w_k1_KO_s012 + 1.04729 w_k1_MAT_s012
                                 + 1.09124 w_k1_MSFT_s012
                                 + 1.14897 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.06268 w_k1_ADM_s012
                                 + 1.13278 w_k1_AXP_s012
                                 + 0.826824 w_k1_DIS_s012
                                 + 0.999357 w_k1_FDX_s012
                                 + 0.91821 w_k1_IBM_s012
                                 + 0.986134 w_k1_INTC_s012
                                 + 0.743557 w_k1_KO_s012
                                 + 0.765371 w_k1_MAT_s012
                                 + 0.878101 w_k1_MSFT_s012
                                 + 0.881441 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.17913 w_k1_ADM_s013
                                 + 1.5317 w_k1_AXP_s013 + 1.39696 w_k1_DIS_s013
                                 + 1.21857 w_k1_FDX_s013
                                 + 1.29985 w_k1_IBM_s013
                                 + 1.13069 w_k1_INTC_s013
                                 + 1.23865 w_k1_KO_s013 + 1.71693 w_k1_MAT_s013
                                 + 1.32424 w_k1_MSFT_s013
                                 + 1.03164 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.938101 w_k1_ADM_s013
                                 + 0.867726 w_k1_AXP_s013
                                 + 0.74063 w_k1_DIS_s013
                                 + 0.740386 w_k1_FDX_s013
                                 + 0.925518 w_k1_IBM_s013
                                 + 0.962061 w_k1_INTC_s013
                                 + 0.982447 w_k1_KO_s013
                                 + 1.26466 w_k1_MAT_s013
                                 + 0.92315 w_k1_MSFT_s013
                                 + 0.933811 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.22168 w_k1_ADM_s013
                                 + 1.09878 w_k1_AXP_s013
                                 + 1.23057 w_k1_DIS_s013
                                 + 0.957956 w_k1_FDX_s013
                                 + 1.0421 w_k1_IBM_s013 + 1.0997 w_k1_INTC_s013
                                 + 1.12125 w_k1_KO_s013 + 1.31074 w_k1_MAT_s013
                                 + 1.19128 w_k1_MSFT_s013
                                 + 1.02099 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.88726 w_k1_ADM_s013
                                 + 0.880558 w_k1_AXP_s013
                                 + 1.05593 w_k1_DIS_s013
                                 + 0.973693 w_k1_FDX_s013
                                 + 0.810707 w_k1_IBM_s013
                                 + 0.995237 w_k1_INTC_s013
                                 + 1.00535 w_k1_KO_s013
                                 + 0.855914 w_k1_MAT_s013
                                 + 1.00345 w_k1_MSFT_s013
                                 + 0.966634 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.989501 w_k1_ADM_s013
                                 + 0.900311 w_k1_AXP_s013
                                 + 0.670623 w_k1_DIS_s013
                                 + 0.892985 w_k1_FDX_s013
                                 + 0.601674 w_k1_IBM_s013
                                 + 0.812091 w_k1_INTC_s013
                                 + 0.678704 w_k1_KO_s013
                                 + 0.633029 w_k1_MAT_s013
                                 + 0.92067 w_k1_MSFT_s013
                                 + 0.729362 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.45749 w_k1_ADM_s013
                                 + 1.69005 w_k1_AXP_s013
                                 + 2.05526 w_k1_DIS_s013
                                 + 1.41294 w_k1_FDX_s013
                                 + 1.63305 w_k1_IBM_s013
                                 + 1.39232 w_k1_INTC_s013
                                 + 1.75293 w_k1_KO_s013 + 1.66637 w_k1_MAT_s013
                                 + 1.52183 w_k1_MSFT_s013
                                 + 1.23156 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.53788 w_k1_ADM_s013
                                 + 1.1323 w_k1_AXP_s013 + 0.87393 w_k1_DIS_s013
                                 + 1.1143 w_k1_FDX_s013
                                 + 0.698626 w_k1_IBM_s013
                                 + 0.870855 w_k1_INTC_s013
                                 + 0.83467 w_k1_KO_s013
                                 + 0.590186 w_k1_MAT_s013
                                 + 0.894044 w_k1_MSFT_s013
                                 + 0.902141 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.44245 w_k1_ADM_s013
                                 + 1.14026 w_k1_AXP_s013
                                 + 1.39328 w_k1_DIS_s013 + 1.1938 w_k1_FDX_s013
                                 + 0.96401 w_k1_IBM_s013
                                 + 0.976937 w_k1_INTC_s013
                                 + 1.03787 w_k1_KO_s013
                                 + 0.938079 w_k1_MAT_s013
                                 + 1.26059 w_k1_MSFT_s013
                                 + 0.919563 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.995788 w_k1_ADM_s014
                                 + 1.14511 w_k1_AXP_s014
                                 + 1.22444 w_k1_DIS_s014
                                 + 1.01569 w_k1_FDX_s014
                                 + 1.73152 w_k1_IBM_s014
                                 + 1.00811 w_k1_INTC_s014
                                 + 1.36125 w_k1_KO_s014 + 1.6572 w_k1_MAT_s014
                                 + 1.23283 w_k1_MSFT_s014
                                 + 1.04959 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.6409 w_k1_ADM_s014
                                 + 0.906884 w_k1_AXP_s014
                                 + 0.914362 w_k1_DIS_s014
                                 + 0.818915 w_k1_FDX_s014
                                 + 1.24505 w_k1_IBM_s014
                                 + 1.13122 w_k1_INTC_s014 + 1.052 w_k1_KO_s014
                                 + 1.06747 w_k1_MAT_s014
                                 + 0.879499 w_k1_MSFT_s014
                                 + 0.808985 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.09298 w_k1_ADM_s014
                                 + 1.13958 w_k1_AXP_s014
                                 + 1.49562 w_k1_DIS_s014
                                 + 1.11062 w_k1_FDX_s014
                                 + 1.12223 w_k1_IBM_s014
                                 + 1.2542 w_k1_INTC_s014 + 1.30479 w_k1_KO_s014
                                 + 0.896389 w_k1_MAT_s014
                                 + 1.23713 w_k1_MSFT_s014
                                 + 1.08582 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.39134 w_k1_ADM_s014
                                 + 1.13421 w_k1_AXP_s014
                                 + 1.04152 w_k1_DIS_s014
                                 + 1.12958 w_k1_FDX_s014
                                 + 0.657354 w_k1_IBM_s014
                                 + 1.1065 w_k1_INTC_s014
                                 + 0.787686 w_k1_KO_s014
                                 + 0.65945 w_k1_MAT_s014
                                 + 1.05353 w_k1_MSFT_s014
                                 + 0.915098 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.882457 w_k1_ADM_s014
                                 + 0.566487 w_k1_AXP_s014
                                 + 1.12014 w_k1_DIS_s014
                                 + 0.931024 w_k1_FDX_s014
                                 + 0.818252 w_k1_IBM_s014
                                 + 1.05449 w_k1_INTC_s014
                                 + 1.05278 w_k1_KO_s014
                                 + 0.765793 w_k1_MAT_s014
                                 + 1.13028 w_k1_MSFT_s014
                                 + 0.922292 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.14076 w_k1_ADM_s014
                                 + 1.64414 w_k1_AXP_s014
                                 + 1.49524 w_k1_DIS_s014
                                 + 1.40035 w_k1_FDX_s014
                                 + 0.859862 w_k1_IBM_s014
                                 + 1.43361 w_k1_INTC_s014 + 1.1125 w_k1_KO_s014
                                 + 0.972857 w_k1_MAT_s014
                                 + 1.26685 w_k1_MSFT_s014
                                 + 0.988768 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.859657 w_k1_ADM_s014
                                 + 0.743382 w_k1_AXP_s014
                                 + 0.80095 w_k1_DIS_s014
                                 + 0.846891 w_k1_FDX_s014
                                 + 0.603464 w_k1_IBM_s014
                                 + 0.748441 w_k1_INTC_s014
                                 + 0.731397 w_k1_KO_s014
                                 + 0.564664 w_k1_MAT_s014
                                 + 0.847238 w_k1_MSFT_s014
                                 + 0.71631 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.935898 w_k1_ADM_s014
                                 + 1.18892 w_k1_AXP_s014
                                 + 0.930148 w_k1_DIS_s014
                                 + 0.954791 w_k1_FDX_s014
                                 + 0.990679 w_k1_IBM_s014
                                 + 1.10644 w_k1_INTC_s014
                                 + 0.955338 w_k1_KO_s014
                                 + 0.989876 w_k1_MAT_s014
                                 + 0.967555 w_k1_MSFT_s014
                                 + 0.91293 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.02144 w_k1_ADM_s015
                                 + 1.29087 w_k1_AXP_s015
                                 + 1.37077 w_k1_DIS_s015
                                 + 1.14761 w_k1_FDX_s015
                                 + 1.65512 w_k1_IBM_s015
                                 + 1.22728 w_k1_INTC_s015
                                 + 1.13058 w_k1_KO_s015 + 1.24325 w_k1_MAT_s015
                                 + 1.03284 w_k1_MSFT_s015
                                 + 0.993524 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.13528 w_k1_ADM_s015
                                 + 1.13088 w_k1_AXP_s015
                                 + 0.961248 w_k1_DIS_s015
                                 + 1.00103 w_k1_FDX_s015
                                 + 1.26347 w_k1_IBM_s015
                                 + 1.06445 w_k1_INTC_s015
                                 + 0.922794 w_k1_KO_s015
                                 + 1.25275 w_k1_MAT_s015
                                 + 0.960411 w_k1_MSFT_s015
                                 + 0.962127 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.34416 w_k1_ADM_s015
                                 + 1.38939 w_k1_AXP_s015
                                 + 1.40055 w_k1_DIS_s015
                                 + 1.22581 w_k1_FDX_s015
                                 + 1.07187 w_k1_IBM_s015
                                 + 1.03621 w_k1_INTC_s015
                                 + 0.998011 w_k1_KO_s015
                                 + 1.01586 w_k1_MAT_s015
                                 + 1.14252 w_k1_MSFT_s015
                                 + 1.32191 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.38544 w_k1_ADM_s015
                                 + 1.10743 w_k1_AXP_s015
                                 + 0.93227 w_k1_DIS_s015
                                 + 0.997477 w_k1_FDX_s015
                                 + 0.843887 w_k1_IBM_s015
                                 + 0.835753 w_k1_INTC_s015
                                 + 0.914438 w_k1_KO_s015
                                 + 1.01827 w_k1_MAT_s015
                                 + 0.917893 w_k1_MSFT_s015
                                 + 0.975462 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.727439 w_k1_ADM_s015
                                 + 0.64885 w_k1_AXP_s015
                                 + 0.570323 w_k1_DIS_s015
                                 + 0.754544 w_k1_FDX_s015
                                 + 0.667842 w_k1_IBM_s015
                                 + 0.701384 w_k1_INTC_s015
                                 + 0.621384 w_k1_KO_s015
                                 + 0.708395 w_k1_MAT_s015
                                 + 0.745969 w_k1_MSFT_s015
                                 + 0.825583 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.60826 w_k1_ADM_s015
                                 + 0.763107 w_k1_AXP_s015
                                 + 0.745827 w_k1_DIS_s015
                                 + 0.749962 w_k1_FDX_s015
                                 + 1.15668 w_k1_IBM_s015
                                 + 0.907534 w_k1_INTC_s015
                                 + 0.786193 w_k1_KO_s015 + 1.0734 w_k1_MAT_s015
                                 + 0.812962 w_k1_MSFT_s015
                                 + 0.763025 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.952266 w_k1_ADM_s015
                                 + 1.00626 w_k1_AXP_s015
                                 + 0.759303 w_k1_DIS_s015
                                 + 0.873355 w_k1_FDX_s015
                                 + 0.817495 w_k1_IBM_s015
                                 + 0.827494 w_k1_INTC_s015
                                 + 0.714137 w_k1_KO_s015
                                 + 1.08804 w_k1_MAT_s015
                                 + 0.892817 w_k1_MSFT_s015
                                 + 0.926684 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.983771 w_k1_ADM_s015
                                 + 1.00063 w_k1_AXP_s015
                                 + 1.18592 w_k1_DIS_s015
                                 + 1.06109 w_k1_FDX_s015
                                 + 0.85193 w_k1_IBM_s015
                                 + 1.22811 w_k1_INTC_s015
                                 + 0.985161 w_k1_KO_s015
                                 + 0.858123 w_k1_MAT_s015
                                 + 1.09404 w_k1_MSFT_s015
                                 + 0.930477 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.815124 w_k1_ADM_s016
                                 + 0.696642 w_k1_AXP_s016
                                 + 0.646306 w_k1_DIS_s016
                                 + 0.756985 w_k1_FDX_s016
                                 + 1.00893 w_k1_IBM_s016
                                 + 0.927483 w_k1_INTC_s016
                                 + 0.793467 w_k1_KO_s016
                                 + 0.808484 w_k1_MAT_s016
                                 + 0.673232 w_k1_MSFT_s016
                                 + 0.815001 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.741128 w_k1_ADM_s016
                                 + 0.844316 w_k1_AXP_s016
                                 + 0.948346 w_k1_DIS_s016
                                 + 0.833219 w_k1_FDX_s016
                                 + 1.46183 w_k1_IBM_s016
                                 + 0.928435 w_k1_INTC_s016
                                 + 1.05317 w_k1_KO_s016 + 1.43363 w_k1_MAT_s016
                                 + 0.789456 w_k1_MSFT_s016
                                 + 0.8498 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.736609 w_k1_ADM_s016
                                 + 0.738235 w_k1_AXP_s016
                                 + 0.848993 w_k1_DIS_s016
                                 + 0.933948 w_k1_FDX_s016
                                 + 0.709459 w_k1_IBM_s016
                                 + 0.857463 w_k1_INTC_s016
                                 + 0.870337 w_k1_KO_s016
                                 + 0.799988 w_k1_MAT_s016
                                 + 0.850447 w_k1_MSFT_s016
                                 + 0.800765 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.17181 w_k1_ADM_s016
                                 + 1.155 w_k1_AXP_s016 + 1.21287 w_k1_DIS_s016
                                 + 1.2146 w_k1_FDX_s016
                                 + 0.901139 w_k1_IBM_s016
                                 + 1.20612 w_k1_INTC_s016
                                 + 0.977145 w_k1_KO_s016
                                 + 0.924434 w_k1_MAT_s016
                                 + 0.970091 w_k1_MSFT_s016
                                 + 0.962595 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.972953 w_k1_ADM_s016
                                 + 1.37168 w_k1_AXP_s016
                                 + 1.85322 w_k1_DIS_s016
                                 + 1.40032 w_k1_FDX_s016
                                 + 1.33545 w_k1_IBM_s016
                                 + 1.39644 w_k1_INTC_s016
                                 + 1.43876 w_k1_KO_s016 + 1.44454 w_k1_MAT_s016
                                 + 1.38937 w_k1_MSFT_s016
                                 + 1.29932 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.08124 w_k1_ADM_s016
                                 + 1.4259 w_k1_AXP_s016 + 1.5922 w_k1_DIS_s016
                                 + 1.29728 w_k1_FDX_s016
                                 + 1.19738 w_k1_IBM_s016
                                 + 1.02331 w_k1_INTC_s016
                                 + 1.32577 w_k1_KO_s016 + 1.86862 w_k1_MAT_s016
                                 + 1.16883 w_k1_MSFT_s016
                                 + 1.29261 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.776368 w_k1_ADM_s016
                                 + 0.994972 w_k1_AXP_s016
                                 + 1.08722 w_k1_DIS_s016
                                 + 0.993909 w_k1_FDX_s016
                                 + 0.92758 w_k1_IBM_s016
                                 + 0.97359 w_k1_INTC_s016
                                 + 1.04261 w_k1_KO_s016 + 1.49946 w_k1_MAT_s016
                                 + 1.03679 w_k1_MSFT_s016
                                 + 1.11113 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.955385 w_k1_ADM_s016
                                 + 1.12827 w_k1_AXP_s016
                                 + 1.06805 w_k1_DIS_s016
                                 + 1.04974 w_k1_FDX_s016
                                 + 1.10853 w_k1_IBM_s016
                                 + 1.05335 w_k1_INTC_s016
                                 + 1.22335 w_k1_KO_s016 + 1.38759 w_k1_MAT_s016
                                 + 0.955747 w_k1_MSFT_s016
                                 + 0.994726 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.11011 w_k1_ADM_s017
                                 + 0.911549 w_k1_AXP_s017
                                 + 0.916048 w_k1_DIS_s017
                                 + 1.05817 w_k1_FDX_s017
                                 + 0.887772 w_k1_IBM_s017
                                 + 1.02111 w_k1_INTC_s017
                                 + 0.819922 w_k1_KO_s017
                                 + 0.969365 w_k1_MAT_s017
                                 + 1.08057 w_k1_MSFT_s017
                                 + 1.06675 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.9697 w_k1_ADM_s017
                                 + 1.19661 w_k1_AXP_s017
                                 + 1.33735 w_k1_DIS_s017
                                 + 1.13687 w_k1_FDX_s017
                                 + 1.62101 w_k1_IBM_s017
                                 + 1.11412 w_k1_INTC_s017
                                 + 1.32689 w_k1_KO_s017 + 1.49561 w_k1_MAT_s017
                                 + 1.14704 w_k1_MSFT_s017
                                 + 1.06276 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.04227 w_k1_ADM_s017
                                 + 0.706917 w_k1_AXP_s017
                                 + 0.842657 w_k1_DIS_s017
                                 + 0.851957 w_k1_FDX_s017
                                 + 1.03998 w_k1_IBM_s017
                                 + 0.836616 w_k1_INTC_s017
                                 + 0.884382 w_k1_KO_s017
                                 + 0.915373 w_k1_MAT_s017
                                 + 0.986822 w_k1_MSFT_s017
                                 + 0.875002 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.97429 w_k1_ADM_s017
                                 + 0.626172 w_k1_AXP_s017
                                 + 0.562041 w_k1_DIS_s017
                                 + 0.738455 w_k1_FDX_s017
                                 + 0.778947 w_k1_IBM_s017
                                 + 0.82691 w_k1_INTC_s017
                                 + 0.680779 w_k1_KO_s017 + 0.7448 w_k1_MAT_s017
                                 + 0.81066 w_k1_MSFT_s017
                                 + 0.880796 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.801838 w_k1_ADM_s017
                                 + 0.832311 w_k1_AXP_s017
                                 + 0.850386 w_k1_DIS_s017
                                 + 0.943771 w_k1_FDX_s017
                                 + 0.806585 w_k1_IBM_s017
                                 + 1.11539 w_k1_INTC_s017
                                 + 0.86955 w_k1_KO_s017
                                 + 0.948118 w_k1_MAT_s017
                                 + 1.08182 w_k1_MSFT_s017
                                 + 0.948549 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.19684 w_k1_ADM_s017
                                 + 1.05467 w_k1_AXP_s017
                                 + 1.36502 w_k1_DIS_s017
                                 + 1.36253 w_k1_FDX_s017
                                 + 1.16895 w_k1_IBM_s017
                                 + 1.20158 w_k1_INTC_s017
                                 + 1.17024 w_k1_KO_s017 + 1.30076 w_k1_MAT_s017
                                 + 1.2609 w_k1_MSFT_s017 + 1.11665 w_k1_PG_s017
                                 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.06509 w_k1_ADM_s017
                                 + 0.777709 w_k1_AXP_s017
                                 + 1.18282 w_k1_DIS_s017
                                 + 1.14805 w_k1_FDX_s017
                                 + 0.858203 w_k1_IBM_s017
                                 + 1.10873 w_k1_INTC_s017
                                 + 0.903754 w_k1_KO_s017
                                 + 0.506377 w_k1_MAT_s017
                                 + 1.28704 w_k1_MSFT_s017
                                 + 0.847683 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.0062 w_k1_ADM_s017
                                 + 1.02669 w_k1_AXP_s017
                                 + 1.06584 w_k1_DIS_s017
                                 + 1.09491 w_k1_FDX_s017
                                 + 1.17078 w_k1_IBM_s017
                                 + 1.01071 w_k1_INTC_s017
                                 + 1.15976 w_k1_KO_s017 + 1.57931 w_k1_MAT_s017
                                 + 1.0571 w_k1_MSFT_s017 + 1.10467 w_k1_PG_s017
                                 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.753936 w_k1_ADM_s100
                                 + 1.03167 w_k1_AXP_s100
                                 + 0.942218 w_k1_DIS_s100
                                 + 0.986423 w_k1_FDX_s100
                                 + 0.756003 w_k1_IBM_s100
                                 + 1.08227 w_k1_INTC_s100
                                 + 0.822114 w_k1_KO_s100
                                 + 0.832156 w_k1_MAT_s100
                                 + 0.968408 w_k1_MSFT_s100
                                 + 0.963858 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.863265 w_k1_ADM_s100
                                 + 0.943894 w_k1_AXP_s100
                                 + 1.07146 w_k1_DIS_s100
                                 + 0.905448 w_k1_FDX_s100
                                 + 1.03766 w_k1_IBM_s100
                                 + 0.996726 w_k1_INTC_s100
                                 + 0.957885 w_k1_KO_s100
                                 + 1.20402 w_k1_MAT_s100
                                 + 1.03504 w_k1_MSFT_s100
                                 + 1.00668 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.821165 w_k1_ADM_s100
                                 + 0.875541 w_k1_AXP_s100
                                 + 0.715976 w_k1_DIS_s100
                                 + 0.892567 w_k1_FDX_s100
                                 + 0.77324 w_k1_IBM_s100
                                 + 0.920061 w_k1_INTC_s100
                                 + 0.690743 w_k1_KO_s100
                                 + 0.870598 w_k1_MAT_s100
                                 + 0.936524 w_k1_MSFT_s100
                                 + 0.842555 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.940915 w_k1_ADM_s100
                                 + 0.777248 w_k1_AXP_s100
                                 + 0.93245 w_k1_DIS_s100
                                 + 0.889632 w_k1_FDX_s100
                                 + 1.04206 w_k1_IBM_s100
                                 + 0.800759 w_k1_INTC_s100
                                 + 0.853108 w_k1_KO_s100
                                 + 0.916614 w_k1_MAT_s100
                                 + 0.892525 w_k1_MSFT_s100
                                 + 0.987312 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.03947 w_k1_ADM_s100
                                 + 1.14399 w_k1_AXP_s100
                                 + 1.08157 w_k1_DIS_s100
                                 + 1.02395 w_k1_FDX_s100
                                 + 1.22593 w_k1_IBM_s100
                                 + 1.13954 w_k1_INTC_s100
                                 + 0.967123 w_k1_KO_s100
                                 + 1.34316 w_k1_MAT_s100
                                 + 1.04078 w_k1_MSFT_s100
                                 + 1.14015 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.785284 w_k1_ADM_s100
                                 + 0.831541 w_k1_AXP_s100
                                 + 0.626693 w_k1_DIS_s100
                                 + 0.786956 w_k1_FDX_s100
                                 + 0.67014 w_k1_IBM_s100
                                 + 0.716256 w_k1_INTC_s100
                                 + 0.610944 w_k1_KO_s100
                                 + 0.688565 w_k1_MAT_s100
                                 + 0.769561 w_k1_MSFT_s100
                                 + 0.773678 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.57982 w_k1_ADM_s100
                                 + 1.44431 w_k1_AXP_s100
                                 + 1.49933 w_k1_DIS_s100
                                 + 1.16041 w_k1_FDX_s100
                                 + 1.59361 w_k1_IBM_s100
                                 + 1.1234 w_k1_INTC_s100 + 1.40227 w_k1_KO_s100
                                 + 1.68186 w_k1_MAT_s100
                                 + 1.1612 w_k1_MSFT_s100 + 1.24288 w_k1_PG_s100
                                 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.03041 w_k1_ADM_s100
                                 + 1.19535 w_k1_AXP_s100
                                 + 1.69057 w_k1_DIS_s100
                                 + 1.29395 w_k1_FDX_s100
                                 + 1.12789 w_k1_IBM_s100
                                 + 1.42377 w_k1_INTC_s100 + 1.1268 w_k1_KO_s100
                                 + 1.15556 w_k1_MAT_s100
                                 + 1.29061 w_k1_MSFT_s100
                                 + 1.03034 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.880738 w_k1_ADM_s101
                                 + 0.833123 w_k1_AXP_s101
                                 + 0.692944 w_k1_DIS_s101
                                 + 0.802019 w_k1_FDX_s101
                                 + 0.797277 w_k1_IBM_s101
                                 + 0.885196 w_k1_INTC_s101
                                 + 0.779236 w_k1_KO_s101
                                 + 0.566581 w_k1_MAT_s101
                                 + 0.822157 w_k1_MSFT_s101
                                 + 0.894433 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.17873 w_k1_ADM_s101
                                 + 1.05306 w_k1_AXP_s101
                                 + 1.01239 w_k1_DIS_s101
                                 + 1.08546 w_k1_FDX_s101
                                 + 0.94116 w_k1_IBM_s101
                                 + 0.884698 w_k1_INTC_s101
                                 + 0.888576 w_k1_KO_s101
                                 + 0.781696 w_k1_MAT_s101
                                 + 0.861487 w_k1_MSFT_s101
                                 + 1.02402 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.75295 w_k1_ADM_s101
                                 + 0.672361 w_k1_AXP_s101
                                 + 1.01243 w_k1_DIS_s101
                                 + 0.865321 w_k1_FDX_s101
                                 + 1.04189 w_k1_IBM_s101
                                 + 1.01071 w_k1_INTC_s101
                                 + 1.06331 w_k1_KO_s101
                                 + 0.751749 w_k1_MAT_s101
                                 + 0.922505 w_k1_MSFT_s101
                                 + 0.920163 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.99572 w_k1_ADM_s101
                                 + 1.15668 w_k1_AXP_s101
                                 + 1.04416 w_k1_DIS_s101
                                 + 0.96563 w_k1_FDX_s101
                                 + 1.63685 w_k1_IBM_s101
                                 + 1.21873 w_k1_INTC_s101
                                 + 1.01684 w_k1_KO_s101 + 1.23188 w_k1_MAT_s101
                                 + 0.939953 w_k1_MSFT_s101
                                 + 1.09486 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.14597 w_k1_ADM_s101
                                 + 0.82356 w_k1_AXP_s101
                                 + 0.966955 w_k1_DIS_s101
                                 + 1.05545 w_k1_FDX_s101
                                 + 0.677892 w_k1_IBM_s101
                                 + 0.972144 w_k1_INTC_s101
                                 + 0.732432 w_k1_KO_s101
                                 + 0.452607 w_k1_MAT_s101
                                 + 1.01618 w_k1_MSFT_s101
                                 + 0.876334 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.21052 w_k1_ADM_s101
                                 + 1.3088 w_k1_AXP_s101 + 1.45111 w_k1_DIS_s101
                                 + 1.19985 w_k1_FDX_s101
                                 + 1.25516 w_k1_IBM_s101
                                 + 1.02558 w_k1_INTC_s101
                                 + 1.23561 w_k1_KO_s101 + 1.27268 w_k1_MAT_s101
                                 + 1.16385 w_k1_MSFT_s101
                                 + 1.24366 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.01646 w_k1_ADM_s101
                                 + 0.919037 w_k1_AXP_s101
                                 + 1.07615 w_k1_DIS_s101
                                 + 1.02775 w_k1_FDX_s101
                                 + 1.08169 w_k1_IBM_s101
                                 + 1.14569 w_k1_INTC_s101
                                 + 1.01792 w_k1_KO_s101
                                 + 0.966715 w_k1_MAT_s101
                                 + 1.03647 w_k1_MSFT_s101
                                 + 1.01643 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.42529 w_k1_ADM_s101
                                 + 1.32373 w_k1_AXP_s101
                                 + 1.14358 w_k1_DIS_s101
                                 + 1.32796 w_k1_FDX_s101
                                 + 0.848807 w_k1_IBM_s101
                                 + 1.16767 w_k1_INTC_s101
                                 + 0.88953 w_k1_KO_s101
                                 + 0.703883 w_k1_MAT_s101
                                 + 1.17571 w_k1_MSFT_s101
                                 + 1.08207 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.714045 w_k1_ADM_s102
                                 + 0.615587 w_k1_AXP_s102
                                 + 0.64926 w_k1_DIS_s102
                                 + 0.778186 w_k1_FDX_s102
                                 + 0.799288 w_k1_IBM_s102
                                 + 0.789914 w_k1_INTC_s102
                                 + 0.68915 w_k1_KO_s102 + 0.736 w_k1_MAT_s102
                                 + 0.744363 w_k1_MSFT_s102
                                 + 0.915138 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.09944 w_k1_ADM_s102
                                 + 1.13244 w_k1_AXP_s102
                                 + 0.93869 w_k1_DIS_s102
                                 + 0.986913 w_k1_FDX_s102
                                 + 0.99872 w_k1_IBM_s102
                                 + 0.995098 w_k1_INTC_s102
                                 + 0.934319 w_k1_KO_s102
                                 + 0.960439 w_k1_MAT_s102
                                 + 0.892514 w_k1_MSFT_s102
                                 + 1.1502 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.01 w_k1_ADM_s102
                                 + 0.786257 w_k1_AXP_s102
                                 + 0.995239 w_k1_DIS_s102
                                 + 1.09011 w_k1_FDX_s102
                                 + 0.758208 w_k1_IBM_s102
                                 + 0.956625 w_k1_INTC_s102
                                 + 0.95443 w_k1_KO_s102
                                 + 0.826681 w_k1_MAT_s102
                                 + 1.09434 w_k1_MSFT_s102
                                 + 1.15038 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.836208 w_k1_ADM_s102
                                 + 0.801288 w_k1_AXP_s102
                                 + 0.713776 w_k1_DIS_s102
                                 + 0.773862 w_k1_FDX_s102 + 1.444 w_k1_IBM_s102
                                 + 0.785438 w_k1_INTC_s102
                                 + 0.962355 w_k1_KO_s102
                                 + 1.06591 w_k1_MAT_s102
                                 + 0.850119 w_k1_MSFT_s102
                                 + 1.07187 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.16163 w_k1_ADM_s102
                                 + 1.51132 w_k1_AXP_s102
                                 + 1.74345 w_k1_DIS_s102 + 1.5321 w_k1_FDX_s102
                                 + 0.989717 w_k1_IBM_s102
                                 + 1.34559 w_k1_INTC_s102
                                 + 1.20342 w_k1_KO_s102
                                 + 0.808173 w_k1_MAT_s102
                                 + 1.43613 w_k1_MSFT_s102
                                 + 1.36633 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.921102 w_k1_ADM_s102
                                 + 1.14609 w_k1_AXP_s102
                                 + 1.40568 w_k1_DIS_s102
                                 + 0.994135 w_k1_FDX_s102
                                 + 1.69246 w_k1_IBM_s102
                                 + 0.963262 w_k1_INTC_s102
                                 + 1.38102 w_k1_KO_s102 + 1.72294 w_k1_MAT_s102
                                 + 1.06188 w_k1_MSFT_s102
                                 + 1.29271 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.13726 w_k1_ADM_s102
                                 + 1.39931 w_k1_AXP_s102
                                 + 1.45301 w_k1_DIS_s102
                                 + 1.32876 w_k1_FDX_s102
                                 + 1.25292 w_k1_IBM_s102
                                 + 0.949743 w_k1_INTC_s102
                                 + 1.22907 w_k1_KO_s102 + 1.49082 w_k1_MAT_s102
                                 + 1.2321 w_k1_MSFT_s102 + 1.35799 w_k1_PG_s102
                                 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.17287 w_k1_ADM_s102
                                 + 1.06675 w_k1_AXP_s102
                                 + 1.18877 w_k1_DIS_s102
                                 + 1.15993 w_k1_FDX_s102
                                 + 1.24129 w_k1_IBM_s102
                                 + 1.04047 w_k1_INTC_s102
                                 + 1.04866 w_k1_KO_s102
                                 + 0.975303 w_k1_MAT_s102
                                 + 1.11035 w_k1_MSFT_s102 + 1.1049 w_k1_PG_s102
                                 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.14854 w_k1_ADM_s103
                                 + 0.82773 w_k1_AXP_s103
                                 + 1.01383 w_k1_DIS_s103
                                 + 0.963427 w_k1_FDX_s103
                                 + 1.01264 w_k1_IBM_s103
                                 + 1.04977 w_k1_INTC_s103
                                 + 0.887657 w_k1_KO_s103
                                 + 0.613572 w_k1_MAT_s103
                                 + 0.980162 w_k1_MSFT_s103
                                 + 0.967939 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.09132 w_k1_ADM_s103
                                 + 1.0739 w_k1_AXP_s103
                                 + 0.878828 w_k1_DIS_s103
                                 + 0.924522 w_k1_FDX_s103
                                 + 1.30971 w_k1_IBM_s103
                                 + 0.975164 w_k1_INTC_s103
                                 + 1.13176 w_k1_KO_s103 + 1.2138 w_k1_MAT_s103
                                 + 1.01239 w_k1_MSFT_s103
                                 + 0.994982 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.05873 w_k1_ADM_s103
                                 + 0.898057 w_k1_AXP_s103
                                 + 1.29671 w_k1_DIS_s103
                                 + 1.01119 w_k1_FDX_s103
                                 + 1.57453 w_k1_IBM_s103
                                 + 1.14745 w_k1_INTC_s103
                                 + 1.45431 w_k1_KO_s103
                                 + 0.792963 w_k1_MAT_s103
                                 + 1.02278 w_k1_MSFT_s103
                                 + 1.06399 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.923479 w_k1_ADM_s103
                                 + 0.69959 w_k1_AXP_s103
                                 + 0.77088 w_k1_DIS_s103
                                 + 0.817536 w_k1_FDX_s103
                                 + 0.983529 w_k1_IBM_s103
                                 + 0.888739 w_k1_INTC_s103
                                 + 0.862222 w_k1_KO_s103
                                 + 0.581729 w_k1_MAT_s103
                                 + 0.82768 w_k1_MSFT_s103
                                 + 0.865409 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.17118 w_k1_ADM_s103
                                 + 0.950545 w_k1_AXP_s103
                                 + 0.910165 w_k1_DIS_s103
                                 + 1.06784 w_k1_FDX_s103
                                 + 0.646832 w_k1_IBM_s103
                                 + 0.924642 w_k1_INTC_s103
                                 + 0.718512 w_k1_KO_s103
                                 + 0.62921 w_k1_MAT_s103
                                 + 0.897139 w_k1_MSFT_s103
                                 + 0.841791 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.437 w_k1_ADM_s103
                                 + 1.37388 w_k1_AXP_s103
                                 + 1.44812 w_k1_DIS_s103
                                 + 1.28116 w_k1_FDX_s103
                                 + 1.46473 w_k1_IBM_s103
                                 + 1.38516 w_k1_INTC_s103
                                 + 1.15859 w_k1_KO_s103
                                 + 0.950958 w_k1_MAT_s103
                                 + 1.18504 w_k1_MSFT_s103
                                 + 1.14709 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.14442 w_k1_ADM_s103
                                 + 1.16531 w_k1_AXP_s103
                                 + 1.21729 w_k1_DIS_s103
                                 + 1.17634 w_k1_FDX_s103
                                 + 0.999716 w_k1_IBM_s103
                                 + 1.04826 w_k1_INTC_s103
                                 + 0.978999 w_k1_KO_s103
                                 + 0.803108 w_k1_MAT_s103
                                 + 0.992038 w_k1_MSFT_s103
                                 + 0.985969 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.86136 w_k1_ADM_s103
                                 + 0.589572 w_k1_AXP_s103
                                 + 0.655176 w_k1_DIS_s103
                                 + 0.797248 w_k1_FDX_s103
                                 + 0.72802 w_k1_IBM_s103
                                 + 0.985371 w_k1_INTC_s103
                                 + 0.634548 w_k1_KO_s103
                                 + 0.35607 w_k1_MAT_s103
                                 + 0.828726 w_k1_MSFT_s103
                                 + 0.721777 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.86921 w_k1_ADM_s104
                                 + 1.24934 w_k1_AXP_s104
                                 + 1.41028 w_k1_DIS_s104
                                 + 1.10947 w_k1_FDX_s104
                                 + 1.40336 w_k1_IBM_s104
                                 + 1.23645 w_k1_INTC_s104
                                 + 1.29301 w_k1_KO_s104 + 1.27466 w_k1_MAT_s104
                                 + 1.21188 w_k1_MSFT_s104
                                 + 1.10362 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.85951 w_k1_ADM_s104
                                 + 0.928379 w_k1_AXP_s104
                                 + 0.697165 w_k1_DIS_s104
                                 + 0.831195 w_k1_FDX_s104
                                 + 0.874069 w_k1_IBM_s104
                                 + 0.96083 w_k1_INTC_s104
                                 + 0.861435 w_k1_KO_s104
                                 + 0.974183 w_k1_MAT_s104
                                 + 0.804601 w_k1_MSFT_s104
                                 + 0.860872 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.959591 w_k1_ADM_s104
                                 + 1.30176 w_k1_AXP_s104
                                 + 1.01783 w_k1_DIS_s104
                                 + 1.07293 w_k1_FDX_s104
                                 + 1.11087 w_k1_IBM_s104
                                 + 1.1471 w_k1_INTC_s104 + 1.05346 w_k1_KO_s104
                                 + 1.25662 w_k1_MAT_s104
                                 + 1.01132 w_k1_MSFT_s104 + 1.0581 w_k1_PG_s104
                                 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.944443 w_k1_ADM_s104
                                 + 0.991975 w_k1_AXP_s104
                                 + 1.04143 w_k1_DIS_s104
                                 + 1.17066 w_k1_FDX_s104
                                 + 0.575718 w_k1_IBM_s104
                                 + 1.14208 w_k1_INTC_s104
                                 + 0.795231 w_k1_KO_s104
                                 + 0.56386 w_k1_MAT_s104
                                 + 1.14796 w_k1_MSFT_s104
                                 + 0.933613 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.07036 w_k1_ADM_s104
                                 + 1.39696 w_k1_AXP_s104
                                 + 1.30433 w_k1_DIS_s104
                                 + 1.35855 w_k1_FDX_s104
                                 + 0.881057 w_k1_IBM_s104
                                 + 1.41004 w_k1_INTC_s104
                                 + 0.946404 w_k1_KO_s104
                                 + 0.854043 w_k1_MAT_s104
                                 + 1.26063 w_k1_MSFT_s104
                                 + 1.07206 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.27653 w_k1_ADM_s104
                                 + 1.72347 w_k1_AXP_s104
                                 + 1.55413 w_k1_DIS_s104
                                 + 1.38015 w_k1_FDX_s104 + 1.4247 w_k1_IBM_s104
                                 + 1.30181 w_k1_INTC_s104
                                 + 1.18888 w_k1_KO_s104 + 1.23268 w_k1_MAT_s104
                                 + 1.33173 w_k1_MSFT_s104
                                 + 1.25528 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.942482 w_k1_ADM_s104
                                 + 0.89557 w_k1_AXP_s104
                                 + 1.06991 w_k1_DIS_s104
                                 + 1.02441 w_k1_FDX_s104
                                 + 0.915753 w_k1_IBM_s104
                                 + 1.04361 w_k1_INTC_s104
                                 + 1.09121 w_k1_KO_s104
                                 + 0.714668 w_k1_MAT_s104
                                 + 0.972069 w_k1_MSFT_s104
                                 + 1.0427 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.958348 w_k1_ADM_s104
                                 + 0.98172 w_k1_AXP_s104
                                 + 0.805208 w_k1_DIS_s104
                                 + 0.884024 w_k1_FDX_s104
                                 + 1.32263 w_k1_IBM_s104
                                 + 0.989103 w_k1_INTC_s104
                                 + 0.978916 w_k1_KO_s104
                                 + 1.30962 w_k1_MAT_s104
                                 + 0.883756 w_k1_MSFT_s104
                                 + 0.970237 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.821951 w_k1_ADM_s105
                                 + 0.730878 w_k1_AXP_s105
                                 + 0.811294 w_k1_DIS_s105
                                 + 0.763667 w_k1_FDX_s105
                                 + 1.11362 w_k1_IBM_s105
                                 + 0.819941 w_k1_INTC_s105
                                 + 1.05662 w_k1_KO_s105 + 1.168 w_k1_MAT_s105
                                 + 0.937582 w_k1_MSFT_s105
                                 + 0.946481 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.16962 w_k1_ADM_s105
                                 + 0.9771 w_k1_AXP_s105 + 1.15785 w_k1_DIS_s105
                                 + 1.00377 w_k1_FDX_s105
                                 + 1.35411 w_k1_IBM_s105
                                 + 1.13549 w_k1_INTC_s105
                                 + 1.33257 w_k1_KO_s105 + 1.25648 w_k1_MAT_s105
                                 + 1.30153 w_k1_MSFT_s105
                                 + 1.19299 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.957405 w_k1_ADM_s105
                                 + 0.812619 w_k1_AXP_s105
                                 + 0.83552 w_k1_DIS_s105
                                 + 0.978783 w_k1_FDX_s105
                                 + 0.601695 w_k1_IBM_s105
                                 + 0.954072 w_k1_INTC_s105
                                 + 0.874637 w_k1_KO_s105
                                 + 0.600403 w_k1_MAT_s105
                                 + 1.06227 w_k1_MSFT_s105
                                 + 0.940545 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.627238 w_k1_ADM_s105
                                 + 0.541216 w_k1_AXP_s105
                                 + 0.537918 w_k1_DIS_s105
                                 + 0.658052 w_k1_FDX_s105
                                 + 0.74256 w_k1_IBM_s105
                                 + 0.670428 w_k1_INTC_s105
                                 + 0.652957 w_k1_KO_s105
                                 + 0.798989 w_k1_MAT_s105
                                 + 0.747905 w_k1_MSFT_s105
                                 + 0.726112 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.29259 w_k1_ADM_s105
                                 + 1.37564 w_k1_AXP_s105
                                 + 1.44232 w_k1_DIS_s105
                                 + 1.33028 w_k1_FDX_s105
                                 + 1.08532 w_k1_IBM_s105
                                 + 1.24344 w_k1_INTC_s105
                                 + 1.18082 w_k1_KO_s105 + 1.43966 w_k1_MAT_s105
                                 + 1.37236 w_k1_MSFT_s105
                                 + 1.30253 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.08781 w_k1_ADM_s105
                                 + 1.25388 w_k1_AXP_s105
                                 + 0.809862 w_k1_DIS_s105
                                 + 0.847629 w_k1_FDX_s105
                                 + 1.58111 w_k1_IBM_s105
                                 + 0.966317 w_k1_INTC_s105 + 1.18 w_k1_KO_s105
                                 + 1.83764 w_k1_MAT_s105
                                 + 0.941324 w_k1_MSFT_s105
                                 + 1.03657 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.01377 w_k1_ADM_s105
                                 + 1.20159 w_k1_AXP_s105
                                 + 1.02474 w_k1_DIS_s105
                                 + 1.13255 w_k1_FDX_s105
                                 + 0.801804 w_k1_IBM_s105
                                 + 1.13106 w_k1_INTC_s105
                                 + 0.860638 w_k1_KO_s105
                                 + 1.21039 w_k1_MAT_s105
                                 + 1.10849 w_k1_MSFT_s105
                                 + 1.04975 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.88439 w_k1_ADM_s105
                                 + 0.684628 w_k1_AXP_s105
                                 + 0.635577 w_k1_DIS_s105
                                 + 0.803588 w_k1_FDX_s105
                                 + 0.643629 w_k1_IBM_s105
                                 + 0.824574 w_k1_INTC_s105
                                 + 0.771338 w_k1_KO_s105
                                 + 0.661945 w_k1_MAT_s105
                                 + 0.950645 w_k1_MSFT_s105
                                 + 0.867081 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.952911 w_k1_ADM_s106
                                 + 0.92966 w_k1_AXP_s106
                                 + 1.04037 w_k1_DIS_s106
                                 + 1.13017 w_k1_FDX_s106
                                 + 0.668445 w_k1_IBM_s106
                                 + 1.17256 w_k1_INTC_s106
                                 + 0.945561 w_k1_KO_s106
                                 + 0.538957 w_k1_MAT_s106
                                 + 1.11632 w_k1_MSFT_s106
                                 + 0.896373 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.05781 w_k1_ADM_s106
                                 + 0.834974 w_k1_AXP_s106
                                 + 1.18063 w_k1_DIS_s106
                                 + 1.02544 w_k1_FDX_s106
                                 + 1.00002 w_k1_IBM_s106
                                 + 0.877762 w_k1_INTC_s106
                                 + 1.19039 w_k1_KO_s106 + 1.00674 w_k1_MAT_s106
                                 + 1.169 w_k1_MSFT_s106 + 1.058 w_k1_PG_s106
                                 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.35314 w_k1_ADM_s106
                                 + 1.5047 w_k1_AXP_s106 + 1.39778 w_k1_DIS_s106
                                 + 1.1287 w_k1_FDX_s106 + 1.56835 w_k1_IBM_s106
                                 + 1.04186 w_k1_INTC_s106
                                 + 1.45342 w_k1_KO_s106 + 1.57804 w_k1_MAT_s106
                                 + 1.15074 w_k1_MSFT_s106
                                 + 1.27525 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.859695 w_k1_ADM_s106
                                 + 0.896078 w_k1_AXP_s106
                                 + 0.809158 w_k1_DIS_s106
                                 + 0.913674 w_k1_FDX_s106
                                 + 0.870635 w_k1_IBM_s106
                                 + 1.03179 w_k1_INTC_s106
                                 + 0.935937 w_k1_KO_s106
                                 + 0.805696 w_k1_MAT_s106
                                 + 0.937756 w_k1_MSFT_s106
                                 + 0.934897 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.856426 w_k1_ADM_s106
                                 + 0.851791 w_k1_AXP_s106
                                 + 0.586214 w_k1_DIS_s106
                                 + 0.800902 w_k1_FDX_s106
                                 + 0.729004 w_k1_IBM_s106
                                 + 0.819465 w_k1_INTC_s106
                                 + 0.740549 w_k1_KO_s106
                                 + 0.710009 w_k1_MAT_s106
                                 + 0.810542 w_k1_MSFT_s106
                                 + 0.81151 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.46758 w_k1_ADM_s106
                                 + 0.985097 w_k1_AXP_s106
                                 + 1.10138 w_k1_DIS_s106
                                 + 1.18994 w_k1_FDX_s106
                                 + 0.831072 w_k1_IBM_s106
                                 + 0.903727 w_k1_INTC_s106
                                 + 0.977383 w_k1_KO_s106
                                 + 0.689918 w_k1_MAT_s106
                                 + 1.05508 w_k1_MSFT_s106
                                 + 0.985916 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.957298 w_k1_ADM_s106
                                 + 0.971506 w_k1_AXP_s106
                                 + 0.817126 w_k1_DIS_s106
                                 + 0.926839 w_k1_FDX_s106
                                 + 1.26232 w_k1_IBM_s106
                                 + 0.950249 w_k1_INTC_s106
                                 + 1.05263 w_k1_KO_s106 + 1.16225 w_k1_MAT_s106
                                 + 0.898454 w_k1_MSFT_s106
                                 + 0.931762 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.4626 w_k1_ADM_s106
                                 + 1.23619 w_k1_AXP_s106
                                 + 1.38704 w_k1_DIS_s106
                                 + 1.31941 w_k1_FDX_s106
                                 + 1.02783 w_k1_IBM_s106
                                 + 0.978563 w_k1_INTC_s106
                                 + 0.949817 w_k1_KO_s106
                                 + 1.11471 w_k1_MAT_s106
                                 + 1.18681 w_k1_MSFT_s106
                                 + 1.06824 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.926506 w_k1_ADM_s107
                                 + 0.740601 w_k1_AXP_s107
                                 + 0.763426 w_k1_DIS_s107
                                 + 0.879271 w_k1_FDX_s107
                                 + 0.651216 w_k1_IBM_s107
                                 + 0.922282 w_k1_INTC_s107
                                 + 0.733229 w_k1_KO_s107
                                 + 0.807224 w_k1_MAT_s107
                                 + 0.91261 w_k1_MSFT_s107
                                 + 0.840936 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.919636 w_k1_ADM_s107
                                 + 0.972493 w_k1_AXP_s107
                                 + 1.13568 w_k1_DIS_s107
                                 + 0.964868 w_k1_FDX_s107
                                 + 0.940461 w_k1_IBM_s107
                                 + 0.875397 w_k1_INTC_s107
                                 + 0.950848 w_k1_KO_s107
                                 + 0.851185 w_k1_MAT_s107
                                 + 1.0412 w_k1_MSFT_s107
                                 + 0.967245 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.16283 w_k1_ADM_s107
                                 + 1.22459 w_k1_AXP_s107
                                 + 0.928711 w_k1_DIS_s107
                                 + 1.02825 w_k1_FDX_s107
                                 + 0.995342 w_k1_IBM_s107
                                 + 0.997878 w_k1_INTC_s107
                                 + 0.95413 w_k1_KO_s107 + 1.27055 w_k1_MAT_s107
                                 + 0.967542 w_k1_MSFT_s107
                                 + 0.944325 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.716638 w_k1_ADM_s107
                                 + 0.728629 w_k1_AXP_s107
                                 + 0.621674 w_k1_DIS_s107
                                 + 0.75692 w_k1_FDX_s107
                                 + 0.794645 w_k1_IBM_s107
                                 + 0.804604 w_k1_INTC_s107
                                 + 0.810474 w_k1_KO_s107
                                 + 0.905584 w_k1_MAT_s107
                                 + 0.739476 w_k1_MSFT_s107
                                 + 0.861297 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.16571 w_k1_ADM_s107
                                 + 1.26264 w_k1_AXP_s107
                                 + 1.45284 w_k1_DIS_s107
                                 + 1.25747 w_k1_FDX_s107
                                 + 0.88504 w_k1_IBM_s107
                                 + 0.964973 w_k1_INTC_s107
                                 + 1.20276 w_k1_KO_s107 + 1.28119 w_k1_MAT_s107
                                 + 1.13457 w_k1_MSFT_s107
                                 + 1.12403 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.859399 w_k1_ADM_s107
                                 + 1.11265 w_k1_AXP_s107
                                 + 0.852245 w_k1_DIS_s107
                                 + 0.86182 w_k1_FDX_s107
                                 + 1.24786 w_k1_IBM_s107
                                 + 1.04358 w_k1_INTC_s107
                                 + 0.983847 w_k1_KO_s107
                                 + 1.18963 w_k1_MAT_s107
                                 + 0.868722 w_k1_MSFT_s107
                                 + 0.913253 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.903159 w_k1_ADM_s107
                                 + 0.918005 w_k1_AXP_s107
                                 + 0.764711 w_k1_DIS_s107
                                 + 0.933783 w_k1_FDX_s107
                                 + 0.764053 w_k1_IBM_s107
                                 + 1.01283 w_k1_INTC_s107
                                 + 0.770132 w_k1_KO_s107
                                 + 0.786469 w_k1_MAT_s107
                                 + 0.861538 w_k1_MSFT_s107
                                 + 0.885369 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.1703 w_k1_ADM_s107
                                 + 1.26958 w_k1_AXP_s107 + 1.5032 w_k1_DIS_s107
                                 + 1.20816 w_k1_FDX_s107
                                 + 1.31672 w_k1_IBM_s107
                                 + 1.20282 w_k1_INTC_s107
                                 + 1.30193 w_k1_KO_s107 + 1.32271 w_k1_MAT_s107
                                 + 1.1003 w_k1_MSFT_s107 + 1.25913 w_k1_PG_s107
                                 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.850709 w_k1_ADM_s110
                                 + 0.672325 w_k1_AXP_s110
                                 + 0.551355 w_k1_DIS_s110
                                 + 0.757353 w_k1_FDX_s110
                                 + 0.600074 w_k1_IBM_s110
                                 + 0.752739 w_k1_INTC_s110
                                 + 0.553131 w_k1_KO_s110
                                 + 0.51864 w_k1_MAT_s110
                                 + 0.840622 w_k1_MSFT_s110
                                 + 0.752619 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.23819 w_k1_ADM_s110
                                 + 0.777103 w_k1_AXP_s110
                                 + 0.83886 w_k1_DIS_s110
                                 + 0.902099 w_k1_FDX_s110
                                 + 0.768475 w_k1_IBM_s110
                                 + 0.739709 w_k1_INTC_s110
                                 + 0.800665 w_k1_KO_s110
                                 + 0.782897 w_k1_MAT_s110
                                 + 0.983731 w_k1_MSFT_s110
                                 + 0.920133 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 2.17402 w_k1_ADM_s110
                                 + 1.56064 w_k1_AXP_s110 + 1.611 w_k1_DIS_s110
                                 + 1.46175 w_k1_FDX_s110
                                 + 1.05908 w_k1_IBM_s110
                                 + 1.27632 w_k1_INTC_s110
                                 + 1.37148 w_k1_KO_s110 + 1.35116 w_k1_MAT_s110
                                 + 1.40878 w_k1_MSFT_s110
                                 + 1.14817 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 2.11188 w_k1_ADM_s110
                                 + 1.84402 w_k1_AXP_s110
                                 + 1.87939 w_k1_DIS_s110
                                 + 1.35519 w_k1_FDX_s110
                                 + 1.75968 w_k1_IBM_s110
                                 + 1.12534 w_k1_INTC_s110
                                 + 1.44607 w_k1_KO_s110 + 1.5873 w_k1_MAT_s110
                                 + 1.20479 w_k1_MSFT_s110
                                 + 1.30355 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.02519 w_k1_ADM_s110
                                 + 0.834494 w_k1_AXP_s110
                                 + 0.775343 w_k1_DIS_s110
                                 + 0.713997 w_k1_FDX_s110
                                 + 1.22901 w_k1_IBM_s110
                                 + 0.736657 w_k1_INTC_s110
                                 + 1.06799 w_k1_KO_s110 + 1.10634 w_k1_MAT_s110
                                 + 0.803009 w_k1_MSFT_s110
                                 + 0.891361 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.967494 w_k1_ADM_s110
                                 + 0.994883 w_k1_AXP_s110
                                 + 1.10801 w_k1_DIS_s110
                                 + 1.03479 w_k1_FDX_s110
                                 + 1.14616 w_k1_IBM_s110
                                 + 0.854597 w_k1_INTC_s110
                                 + 0.999426 w_k1_KO_s110 + 1.1467 w_k1_MAT_s110
                                 + 1.09233 w_k1_MSFT_s110
                                 + 1.05694 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.37095 w_k1_ADM_s110
                                 + 1.4175 w_k1_AXP_s110 + 0.97955 w_k1_DIS_s110
                                 + 1.17809 w_k1_FDX_s110
                                 + 1.17225 w_k1_IBM_s110
                                 + 1.03033 w_k1_INTC_s110
                                 + 1.03019 w_k1_KO_s110 + 1.39808 w_k1_MAT_s110
                                 + 1.05364 w_k1_MSFT_s110
                                 + 1.21119 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.36561 w_k1_ADM_s110
                                 + 1.06806 w_k1_AXP_s110
                                 + 1.75489 w_k1_DIS_s110
                                 + 1.48692 w_k1_FDX_s110
                                 + 0.869666 w_k1_IBM_s110
                                 + 1.3314 w_k1_INTC_s110 + 1.23122 w_k1_KO_s110
                                 + 0.627139 w_k1_MAT_s110
                                 + 1.58581 w_k1_MSFT_s110
                                 + 1.23906 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.47405 w_k1_ADM_s111
                                 + 1.52015 w_k1_AXP_s111 + 1.26 w_k1_DIS_s111
                                 + 1.19873 w_k1_FDX_s111
                                 + 1.03549 w_k1_IBM_s111
                                 + 0.975209 w_k1_INTC_s111
                                 + 1.16245 w_k1_KO_s111 + 1.2659 w_k1_MAT_s111
                                 + 1.1932 w_k1_MSFT_s111 + 1.10732 w_k1_PG_s111
                                 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.06182 w_k1_ADM_s111
                                 + 0.845181 w_k1_AXP_s111
                                 + 0.851501 w_k1_DIS_s111
                                 + 0.906035 w_k1_FDX_s111
                                 + 0.726906 w_k1_IBM_s111
                                 + 0.873827 w_k1_INTC_s111
                                 + 0.775885 w_k1_KO_s111
                                 + 0.696489 w_k1_MAT_s111
                                 + 0.888446 w_k1_MSFT_s111
                                 + 0.89842 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.12005 w_k1_ADM_s111
                                 + 0.899672 w_k1_AXP_s111
                                 + 0.972788 w_k1_DIS_s111
                                 + 0.918071 w_k1_FDX_s111
                                 + 1.20761 w_k1_IBM_s111
                                 + 0.982773 w_k1_INTC_s111
                                 + 1.11722 w_k1_KO_s111 + 1.11744 w_k1_MAT_s111
                                 + 1.07048 w_k1_MSFT_s111
                                 + 1.02813 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.1031 w_k1_ADM_s111
                                 + 1.16604 w_k1_AXP_s111
                                 + 0.912075 w_k1_DIS_s111
                                 + 1.03148 w_k1_FDX_s111
                                 + 1.01933 w_k1_IBM_s111
                                 + 0.94592 w_k1_INTC_s111
                                 + 1.00017 w_k1_KO_s111 + 1.13469 w_k1_MAT_s111
                                 + 0.912236 w_k1_MSFT_s111
                                 + 0.922854 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 2.12171 w_k1_ADM_s111
                                 + 1.69685 w_k1_AXP_s111
                                 + 1.89916 w_k1_DIS_s111 + 1.6054 w_k1_FDX_s111
                                 + 1.07482 w_k1_IBM_s111
                                 + 1.11965 w_k1_INTC_s111
                                 + 1.40066 w_k1_KO_s111 + 1.4911 w_k1_MAT_s111
                                 + 1.48417 w_k1_MSFT_s111
                                 + 1.46545 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.912706 w_k1_ADM_s111
                                 + 0.749793 w_k1_AXP_s111
                                 + 0.608025 w_k1_DIS_s111
                                 + 0.684463 w_k1_FDX_s111
                                 + 0.800782 w_k1_IBM_s111
                                 + 0.801415 w_k1_INTC_s111
                                 + 0.738305 w_k1_KO_s111
                                 + 0.662446 w_k1_MAT_s111
                                 + 0.678133 w_k1_MSFT_s111
                                 + 0.739657 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.28703 w_k1_ADM_s111
                                 + 0.898123 w_k1_AXP_s111
                                 + 1.19107 w_k1_DIS_s111
                                 + 1.08116 w_k1_FDX_s111
                                 + 0.701913 w_k1_IBM_s111
                                 + 1.01203 w_k1_INTC_s111
                                 + 1.02015 w_k1_KO_s111
                                 + 0.458044 w_k1_MAT_s111
                                 + 1.24838 w_k1_MSFT_s111
                                 + 0.999881 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.08884 w_k1_ADM_s111
                                 + 1.35443 w_k1_AXP_s111
                                 + 0.818875 w_k1_DIS_s111
                                 + 0.927101 w_k1_FDX_s111
                                 + 1.20432 w_k1_IBM_s111
                                 + 0.750474 w_k1_INTC_s111
                                 + 0.976471 w_k1_KO_s111
                                 + 1.61216 w_k1_MAT_s111
                                 + 0.829844 w_k1_MSFT_s111
                                 + 0.903373 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.08945 w_k1_ADM_s112
                                 + 1.36074 w_k1_AXP_s112
                                 + 1.17002 w_k1_DIS_s112
                                 + 1.13994 w_k1_FDX_s112
                                 + 0.895686 w_k1_IBM_s112
                                 + 1.27725 w_k1_INTC_s112
                                 + 0.981023 w_k1_KO_s112
                                 + 1.47543 w_k1_MAT_s112
                                 + 1.05489 w_k1_MSFT_s112
                                 + 1.07324 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.54039 w_k1_ADM_s112
                                 + 1.58359 w_k1_AXP_s112
                                 + 1.49482 w_k1_DIS_s112
                                 + 1.37732 w_k1_FDX_s112
                                 + 1.40587 w_k1_IBM_s112
                                 + 1.11801 w_k1_INTC_s112
                                 + 1.46566 w_k1_KO_s112 + 1.93768 w_k1_MAT_s112
                                 + 1.23877 w_k1_MSFT_s112
                                 + 1.36049 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.861312 w_k1_ADM_s112
                                 + 1.15004 w_k1_AXP_s112
                                 + 0.841662 w_k1_DIS_s112
                                 + 0.976001 w_k1_FDX_s112
                                 + 0.890794 w_k1_IBM_s112
                                 + 0.955811 w_k1_INTC_s112
                                 + 0.91905 w_k1_KO_s112 + 1.12033 w_k1_MAT_s112
                                 + 0.88933 w_k1_MSFT_s112
                                 + 1.03268 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.730217 w_k1_ADM_s112
                                 + 0.878996 w_k1_AXP_s112
                                 + 0.66763 w_k1_DIS_s112
                                 + 0.869055 w_k1_FDX_s112
                                 + 0.769721 w_k1_IBM_s112
                                 + 0.739956 w_k1_INTC_s112
                                 + 0.776797 w_k1_KO_s112
                                 + 1.00591 w_k1_MAT_s112
                                 + 0.762594 w_k1_MSFT_s112
                                 + 0.892198 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.11582 w_k1_ADM_s112
                                 + 1.02762 w_k1_AXP_s112 + 0.9716 w_k1_DIS_s112
                                 + 0.934863 w_k1_FDX_s112
                                 + 1.14895 w_k1_IBM_s112
                                 + 0.946454 w_k1_INTC_s112
                                 + 1.13783 w_k1_KO_s112 + 1.38356 w_k1_MAT_s112
                                 + 0.979742 w_k1_MSFT_s112
                                 + 1.05553 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.813293 w_k1_ADM_s112
                                 + 0.868441 w_k1_AXP_s112
                                 + 0.700456 w_k1_DIS_s112
                                 + 0.764763 w_k1_FDX_s112
                                 + 1.17997 w_k1_IBM_s112
                                 + 0.869575 w_k1_INTC_s112
                                 + 0.803011 w_k1_KO_s112
                                 + 1.16041 w_k1_MAT_s112
                                 + 0.764925 w_k1_MSFT_s112
                                 + 0.931463 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.11185 w_k1_ADM_s112
                                 + 1.42328 w_k1_AXP_s112
                                 + 1.61028 w_k1_DIS_s112
                                 + 1.29385 w_k1_FDX_s112
                                 + 1.03429 w_k1_IBM_s112
                                 + 1.53868 w_k1_INTC_s112
                                 + 1.21256 w_k1_KO_s112 + 1.26983 w_k1_MAT_s112
                                 + 1.27603 w_k1_MSFT_s112
                                 + 1.10051 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.867076 w_k1_ADM_s112
                                 + 1.26799 w_k1_AXP_s112
                                 + 0.87914 w_k1_DIS_s112
                                 + 0.902352 w_k1_FDX_s112
                                 + 1.62819 w_k1_IBM_s112
                                 + 1.14238 w_k1_INTC_s112
                                 + 1.04789 w_k1_KO_s112 + 1.90865 w_k1_MAT_s112
                                 + 0.842967 w_k1_MSFT_s112
                                 + 1.04604 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.33731 w_k1_ADM_s113
                                 + 1.38702 w_k1_AXP_s113 + 1.4036 w_k1_DIS_s113
                                 + 1.35973 w_k1_FDX_s113
                                 + 1.17044 w_k1_IBM_s113
                                 + 1.13379 w_k1_INTC_s113
                                 + 1.25356 w_k1_KO_s113 + 1.27218 w_k1_MAT_s113
                                 + 1.15612 w_k1_MSFT_s113
                                 + 1.24783 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.13317 w_k1_ADM_s113
                                 + 1.06177 w_k1_AXP_s113
                                 + 0.869616 w_k1_DIS_s113
                                 + 0.808199 w_k1_FDX_s113
                                 + 1.46006 w_k1_IBM_s113
                                 + 0.839745 w_k1_INTC_s113
                                 + 1.03801 w_k1_KO_s113 + 1.43897 w_k1_MAT_s113
                                 + 0.812188 w_k1_MSFT_s113
                                 + 0.969608 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.907705 w_k1_ADM_s113
                                 + 0.86522 w_k1_AXP_s113
                                 + 0.984717 w_k1_DIS_s113
                                 + 1.02549 w_k1_FDX_s113
                                 + 0.809292 w_k1_IBM_s113
                                 + 0.978047 w_k1_INTC_s113
                                 + 0.944226 w_k1_KO_s113
                                 + 0.744006 w_k1_MAT_s113
                                 + 1.04608 w_k1_MSFT_s113
                                 + 0.93509 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.08612 w_k1_ADM_s113
                                 + 1.1317 w_k1_AXP_s113 + 1.19371 w_k1_DIS_s113
                                 + 1.08763 w_k1_FDX_s113 + 1.177 w_k1_IBM_s113
                                 + 0.919105 w_k1_INTC_s113
                                 + 1.18275 w_k1_KO_s113 + 1.24568 w_k1_MAT_s113
                                 + 1.20443 w_k1_MSFT_s113 + 1.076 w_k1_PG_s113
                                 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.19002 w_k1_ADM_s113
                                 + 0.984582 w_k1_AXP_s113
                                 + 0.914944 w_k1_DIS_s113
                                 + 0.941699 w_k1_FDX_s113
                                 + 1.21583 w_k1_IBM_s113
                                 + 0.825356 w_k1_INTC_s113
                                 + 1.26377 w_k1_KO_s113 + 1.09514 w_k1_MAT_s113
                                 + 1.00466 w_k1_MSFT_s113
                                 + 1.09108 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.87554 w_k1_ADM_s113
                                 + 1.03836 w_k1_AXP_s113
                                 + 0.687878 w_k1_DIS_s113
                                 + 0.880821 w_k1_FDX_s113
                                 + 1.00668 w_k1_IBM_s113
                                 + 0.905653 w_k1_INTC_s113
                                 + 0.904134 w_k1_KO_s113
                                 + 1.11977 w_k1_MAT_s113
                                 + 0.878467 w_k1_MSFT_s113
                                 + 1.09439 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.29056 w_k1_ADM_s113
                                 + 0.769282 w_k1_AXP_s113
                                 + 0.635245 w_k1_DIS_s113
                                 + 0.776309 w_k1_FDX_s113
                                 + 0.886439 w_k1_IBM_s113
                                 + 0.647161 w_k1_INTC_s113
                                 + 0.882034 w_k1_KO_s113
                                 + 0.708256 w_k1_MAT_s113
                                 + 0.769684 w_k1_MSFT_s113
                                 + 0.812194 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.899599 w_k1_ADM_s113
                                 + 0.699717 w_k1_AXP_s113
                                 + 0.75171 w_k1_DIS_s113
                                 + 0.787433 w_k1_FDX_s113
                                 + 0.730605 w_k1_IBM_s113
                                 + 0.833043 w_k1_INTC_s113
                                 + 0.901875 w_k1_KO_s113
                                 + 0.713952 w_k1_MAT_s113
                                 + 0.863384 w_k1_MSFT_s113
                                 + 0.907228 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.02455 w_k1_ADM_s114
                                 + 1.10147 w_k1_AXP_s114
                                 + 1.36131 w_k1_DIS_s114
                                 + 0.972978 w_k1_FDX_s114
                                 + 1.63872 w_k1_IBM_s114
                                 + 1.26356 w_k1_INTC_s114
                                 + 1.19459 w_k1_KO_s114 + 1.06173 w_k1_MAT_s114
                                 + 1.11158 w_k1_MSFT_s114
                                 + 1.07523 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.10918 w_k1_ADM_s114
                                 + 1.76425 w_k1_AXP_s114
                                 + 1.89152 w_k1_DIS_s114
                                 + 1.43444 w_k1_FDX_s114 + 1.2415 w_k1_IBM_s114
                                 + 1.44255 w_k1_INTC_s114
                                 + 1.22339 w_k1_KO_s114 + 1.12385 w_k1_MAT_s114
                                 + 1.42391 w_k1_MSFT_s114
                                 + 1.12981 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.09936 w_k1_ADM_s114
                                 + 1.17732 w_k1_AXP_s114
                                 + 1.09411 w_k1_DIS_s114
                                 + 0.995558 w_k1_FDX_s114
                                 + 1.14718 w_k1_IBM_s114
                                 + 0.985218 w_k1_INTC_s114
                                 + 1.00022 w_k1_KO_s114
                                 + 0.891051 w_k1_MAT_s114
                                 + 1.0274 w_k1_MSFT_s114 + 1.04711 w_k1_PG_s114
                                 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.13102 w_k1_ADM_s114
                                 + 1.11266 w_k1_AXP_s114 + 1.3855 w_k1_DIS_s114
                                 + 1.1842 w_k1_FDX_s114 + 1.01377 w_k1_IBM_s114
                                 + 1.10028 w_k1_INTC_s114
                                 + 1.04803 w_k1_KO_s114
                                 + 0.687482 w_k1_MAT_s114
                                 + 1.21121 w_k1_MSFT_s114
                                 + 1.17056 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.871496 w_k1_ADM_s114
                                 + 0.591937 w_k1_AXP_s114
                                 + 0.620996 w_k1_DIS_s114
                                 + 0.618291 w_k1_FDX_s114
                                 + 0.935285 w_k1_IBM_s114
                                 + 0.697003 w_k1_INTC_s114
                                 + 0.712291 w_k1_KO_s114
                                 + 0.525301 w_k1_MAT_s114
                                 + 0.776221 w_k1_MSFT_s114
                                 + 0.815754 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.880762 w_k1_ADM_s114
                                 + 0.821356 w_k1_AXP_s114
                                 + 0.971011 w_k1_DIS_s114
                                 + 0.932424 w_k1_FDX_s114
                                 + 0.822512 w_k1_IBM_s114
                                 + 1.00137 w_k1_INTC_s114
                                 + 0.824918 w_k1_KO_s114
                                 + 0.525175 w_k1_MAT_s114
                                 + 1.02156 w_k1_MSFT_s114
                                 + 0.926021 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.806197 w_k1_ADM_s114
                                 + 0.947632 w_k1_AXP_s114
                                 + 0.910196 w_k1_DIS_s114
                                 + 0.784433 w_k1_FDX_s114
                                 + 1.18622 w_k1_IBM_s114
                                 + 0.753208 w_k1_INTC_s114
                                 + 0.910453 w_k1_KO_s114
                                 + 0.75134 w_k1_MAT_s114
                                 + 0.847909 w_k1_MSFT_s114
                                 + 0.854174 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.880888 w_k1_ADM_s114
                                 + 0.767652 w_k1_AXP_s114
                                 + 0.636846 w_k1_DIS_s114
                                 + 0.838533 w_k1_FDX_s114
                                 + 0.765799 w_k1_IBM_s114
                                 + 0.894476 w_k1_INTC_s114
                                 + 0.694322 w_k1_KO_s114 + 0.5882 w_k1_MAT_s114
                                 + 0.926582 w_k1_MSFT_s114
                                 + 0.897322 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.19417 w_k1_ADM_s115
                                 + 1.3132 w_k1_AXP_s115
                                 + 0.817671 w_k1_DIS_s115
                                 + 1.15187 w_k1_FDX_s115
                                 + 0.857622 w_k1_IBM_s115
                                 + 1.00662 w_k1_INTC_s115
                                 + 0.774507 w_k1_KO_s115
                                 + 1.09673 w_k1_MAT_s115
                                 + 1.00321 w_k1_MSFT_s115
                                 + 0.956478 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.19158 w_k1_ADM_s115
                                 + 0.967943 w_k1_AXP_s115
                                 + 1.11673 w_k1_DIS_s115
                                 + 1.09788 w_k1_FDX_s115
                                 + 0.781518 w_k1_IBM_s115
                                 + 0.853141 w_k1_INTC_s115
                                 + 0.84021 w_k1_KO_s115
                                 + 0.706099 w_k1_MAT_s115
                                 + 0.946537 w_k1_MSFT_s115
                                 + 0.943668 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.997477 w_k1_ADM_s115
                                 + 1.19167 w_k1_AXP_s115
                                 + 0.948952 w_k1_DIS_s115
                                 + 1.00889 w_k1_FDX_s115
                                 + 0.813966 w_k1_IBM_s115
                                 + 0.856058 w_k1_INTC_s115
                                 + 0.855823 w_k1_KO_s115
                                 + 1.14748 w_k1_MAT_s115
                                 + 0.920394 w_k1_MSFT_s115
                                 + 0.939789 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.18804 w_k1_ADM_s115
                                 + 1.17724 w_k1_AXP_s115
                                 + 1.30287 w_k1_DIS_s115
                                 + 1.31532 w_k1_FDX_s115
                                 + 0.961159 w_k1_IBM_s115
                                 + 1.12916 w_k1_INTC_s115 + 1.1479 w_k1_KO_s115
                                 + 1.11989 w_k1_MAT_s115
                                 + 1.29312 w_k1_MSFT_s115 + 1.2177 w_k1_PG_s115
                                 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.08384 w_k1_ADM_s115
                                 + 1.68743 w_k1_AXP_s115
                                 + 1.13116 w_k1_DIS_s115
                                 + 1.06198 w_k1_FDX_s115
                                 + 1.67172 w_k1_IBM_s115
                                 + 0.898766 w_k1_INTC_s115
                                 + 1.26463 w_k1_KO_s115 + 2.10229 w_k1_MAT_s115
                                 + 0.877973 w_k1_MSFT_s115
                                 + 1.13846 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.919484 w_k1_ADM_s115
                                 + 0.712633 w_k1_AXP_s115
                                 + 0.674222 w_k1_DIS_s115
                                 + 0.866151 w_k1_FDX_s115
                                 + 0.668399 w_k1_IBM_s115
                                 + 0.891065 w_k1_INTC_s115
                                 + 0.730508 w_k1_KO_s115
                                 + 0.593103 w_k1_MAT_s115
                                 + 0.851232 w_k1_MSFT_s115
                                 + 0.962256 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.11587 w_k1_ADM_s115
                                 + 1.19611 w_k1_AXP_s115 + 1.3254 w_k1_DIS_s115
                                 + 1.04793 w_k1_FDX_s115
                                 + 1.29164 w_k1_IBM_s115
                                 + 1.01178 w_k1_INTC_s115
                                 + 1.16281 w_k1_KO_s115 + 1.44876 w_k1_MAT_s115
                                 + 1.04052 w_k1_MSFT_s115
                                 + 1.07754 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.847647 w_k1_ADM_s115
                                 + 0.683653 w_k1_AXP_s115
                                 + 0.72858 w_k1_DIS_s115 + 0.7803 w_k1_FDX_s115
                                 + 0.99329 w_k1_IBM_s115
                                 + 0.65332 w_k1_INTC_s115
                                 + 0.852306 w_k1_KO_s115
                                 + 0.830151 w_k1_MAT_s115
                                 + 0.744588 w_k1_MSFT_s115
                                 + 0.85418 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.743975 w_k1_ADM_s116
                                 + 0.624023 w_k1_AXP_s116
                                 + 0.699941 w_k1_DIS_s116
                                 + 0.741661 w_k1_FDX_s116
                                 + 0.648955 w_k1_IBM_s116
                                 + 0.716238 w_k1_INTC_s116
                                 + 0.640004 w_k1_KO_s116
                                 + 0.693608 w_k1_MAT_s116
                                 + 0.741504 w_k1_MSFT_s116
                                 + 0.783942 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.881881 w_k1_ADM_s116
                                 + 0.67347 w_k1_AXP_s116
                                 + 0.696391 w_k1_DIS_s116
                                 + 0.714935 w_k1_FDX_s116
                                 + 0.823261 w_k1_IBM_s116
                                 + 0.998437 w_k1_INTC_s116
                                 + 0.836371 w_k1_KO_s116
                                 + 0.700833 w_k1_MAT_s116
                                 + 0.805363 w_k1_MSFT_s116
                                 + 0.885655 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.899436 w_k1_ADM_s116
                                 + 0.88233 w_k1_AXP_s116
                                 + 0.813614 w_k1_DIS_s116
                                 + 0.882896 w_k1_FDX_s116
                                 + 0.882422 w_k1_IBM_s116
                                 + 0.857279 w_k1_INTC_s116
                                 + 0.842154 w_k1_KO_s116
                                 + 1.17025 w_k1_MAT_s116
                                 + 0.821308 w_k1_MSFT_s116
                                 + 0.871434 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.06115 w_k1_ADM_s116
                                 + 1.22702 w_k1_AXP_s116
                                 + 1.18871 w_k1_DIS_s116
                                 + 1.09188 w_k1_FDX_s116
                                 + 1.19504 w_k1_IBM_s116
                                 + 1.15828 w_k1_INTC_s116
                                 + 1.02762 w_k1_KO_s116 + 1.62548 w_k1_MAT_s116
                                 + 1.07153 w_k1_MSFT_s116
                                 + 1.12561 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.856397 w_k1_ADM_s116
                                 + 0.836815 w_k1_AXP_s116
                                 + 1.02546 w_k1_DIS_s116
                                 + 1.00152 w_k1_FDX_s116
                                 + 0.918904 w_k1_IBM_s116
                                 + 1.04245 w_k1_INTC_s116
                                 + 0.986852 w_k1_KO_s116
                                 + 1.09075 w_k1_MAT_s116
                                 + 1.03025 w_k1_MSFT_s116
                                 + 1.01394 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.946359 w_k1_ADM_s116
                                 + 0.833289 w_k1_AXP_s116
                                 + 1.01024 w_k1_DIS_s116
                                 + 0.929346 w_k1_FDX_s116
                                 + 1.36147 w_k1_IBM_s116
                                 + 0.957989 w_k1_INTC_s116
                                 + 1.09785 w_k1_KO_s116 + 1.33372 w_k1_MAT_s116
                                 + 0.858651 w_k1_MSFT_s116
                                 + 0.949296 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.14667 w_k1_ADM_s116
                                 + 1.36844 w_k1_AXP_s116
                                 + 1.67917 w_k1_DIS_s116
                                 + 1.32458 w_k1_FDX_s116
                                 + 1.22754 w_k1_IBM_s116
                                 + 1.30587 w_k1_INTC_s116
                                 + 1.29999 w_k1_KO_s116 + 1.73402 w_k1_MAT_s116
                                 + 1.22768 w_k1_MSFT_s116
                                 + 1.21448 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.995234 w_k1_ADM_s116
                                 + 0.863515 w_k1_AXP_s116
                                 + 1.10558 w_k1_DIS_s116 + 1.0978 w_k1_FDX_s116
                                 + 0.801756 w_k1_IBM_s116
                                 + 1.04098 w_k1_INTC_s116
                                 + 1.04165 w_k1_KO_s116
                                 + 0.883831 w_k1_MAT_s116
                                 + 1.11944 w_k1_MSFT_s116
                                 + 1.00534 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.03995 w_k1_ADM_s117
                                 + 1.03485 w_k1_AXP_s117 + 1.3664 w_k1_DIS_s117
                                 + 1.06339 w_k1_FDX_s117
                                 + 1.37082 w_k1_IBM_s117
                                 + 0.856731 w_k1_INTC_s117
                                 + 1.4705 w_k1_KO_s117 + 1.6969 w_k1_MAT_s117
                                 + 1.05914 w_k1_MSFT_s117
                                 + 1.22482 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.978591 w_k1_ADM_s117
                                 + 1.20426 w_k1_AXP_s117
                                 + 1.35122 w_k1_DIS_s117
                                 + 1.24896 w_k1_FDX_s117
                                 + 0.826511 w_k1_IBM_s117
                                 + 1.06812 w_k1_INTC_s117
                                 + 1.20278 w_k1_KO_s117 + 1.26938 w_k1_MAT_s117
                                 + 1.12605 w_k1_MSFT_s117
                                 + 1.05641 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.16938 w_k1_ADM_s117
                                 + 0.967648 w_k1_AXP_s117
                                 + 0.891919 w_k1_DIS_s117
                                 + 0.966927 w_k1_FDX_s117
                                 + 0.795141 w_k1_IBM_s117
                                 + 0.881028 w_k1_INTC_s117
                                 + 0.845751 w_k1_KO_s117
                                 + 0.671134 w_k1_MAT_s117
                                 + 0.911868 w_k1_MSFT_s117
                                 + 0.937503 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.33697 w_k1_ADM_s117
                                 + 1.32874 w_k1_AXP_s117
                                 + 1.59942 w_k1_DIS_s117
                                 + 1.24494 w_k1_FDX_s117
                                 + 1.39839 w_k1_IBM_s117
                                 + 1.1097 w_k1_INTC_s117 + 1.3761 w_k1_KO_s117
                                 + 1.32925 w_k1_MAT_s117
                                 + 1.25454 w_k1_MSFT_s117 + 1.0723 w_k1_PG_s117
                                 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.965741 w_k1_ADM_s117
                                 + 1.10038 w_k1_AXP_s117
                                 + 1.13117 w_k1_DIS_s117
                                 + 0.961671 w_k1_FDX_s117
                                 + 1.01101 w_k1_IBM_s117
                                 + 0.938985 w_k1_INTC_s117
                                 + 1.2446 w_k1_KO_s117 + 1.46913 w_k1_MAT_s117
                                 + 0.949981 w_k1_MSFT_s117
                                 + 1.00239 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.49535 w_k1_ADM_s117
                                 + 1.14444 w_k1_AXP_s117
                                 + 1.69409 w_k1_DIS_s117
                                 + 1.29246 w_k1_FDX_s117
                                 + 0.93395 w_k1_IBM_s117
                                 + 0.984944 w_k1_INTC_s117
                                 + 1.2795 w_k1_KO_s117 + 1.00033 w_k1_MAT_s117
                                 + 1.3616 w_k1_MSFT_s117 + 1.20758 w_k1_PG_s117
                                 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.12233 w_k1_ADM_s117
                                 + 1.00234 w_k1_AXP_s117
                                 + 1.01186 w_k1_DIS_s117
                                 + 0.90552 w_k1_FDX_s117
                                 + 1.22499 w_k1_IBM_s117
                                 + 0.895243 w_k1_INTC_s117
                                 + 1.07061 w_k1_KO_s117 + 1.10142 w_k1_MAT_s117
                                 + 0.871295 w_k1_MSFT_s117
                                 + 0.948334 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.717257 w_k1_ADM_s117
                                 + 0.697065 w_k1_AXP_s117
                                 + 0.697925 w_k1_DIS_s117
                                 + 0.697241 w_k1_FDX_s117
                                 + 0.892723 w_k1_IBM_s117
                                 + 0.770231 w_k1_INTC_s117
                                 + 0.868046 w_k1_KO_s117
                                 + 0.983306 w_k1_MAT_s117
                                 + 0.769313 w_k1_MSFT_s117
                                 + 0.811475 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0177160538093675 w_k1_ADM_s000
                                 + 0.0211160654279241 w_k1_ADM_s001
                                 + 0.0187423512694485 w_k1_ADM_s002
                                 + 0.0330514085084201 w_k1_ADM_s003
                                 + 0.0321022876027528 w_k1_ADM_s004
                                 + 0.00841046071281201 w_k1_ADM_s005
                                 + 0.00751338221127509 w_k1_ADM_s006
                                 + 0.018812769582289 w_k1_ADM_s007
                                 + 0.0157465471996201 w_k1_ADM_s010
                                 + 0.0279087591270928 w_k1_ADM_s011
                                 + 0.0108968516087056 w_k1_ADM_s012
                                 + 0.012177959034406 w_k1_ADM_s013
                                 + 0.0162547363474706 w_k1_ADM_s014
                                 + 0.00888618609642613 w_k1_ADM_s015
                                 + 0.00827968477601886 w_k1_ADM_s016
                                 + 0.00912918435403925 w_k1_ADM_s017
                                 + 0.0231341265137322 w_k1_ADM_s100
                                 + 0.0478482068972059 w_k1_ADM_s101
                                 + 0.0468306350656118 w_k1_ADM_s102
                                 + 0.0297917101429854 w_k1_ADM_s103
                                 + 0.0859860859022743 w_k1_ADM_s104
                                 + 0.0501334992267281 w_k1_ADM_s105
                                 + 0.0353336646695315 w_k1_ADM_s106
                                 + 0.0335999550826632 w_k1_ADM_s107
                                 + 0.0956406542177804 w_k1_ADM_s110
                                 + 0.081791481414947 w_k1_ADM_s111
                                 + 0.0342804956578776 w_k1_ADM_s112
                                 + 0.0358505937501832 w_k1_ADM_s113
                                 + 0.0330708577501147 w_k1_ADM_s114
                                 + 0.0639229337899814 w_k1_ADM_s115
                                 + 0.0456975681159972 w_k1_ADM_s116
                                 + 0.047457694874248 w_k1_ADM_s117
                                 + 0.0164325736822216 w_k1_AXP_s000
                                 + 0.0226003700948015 w_k1_AXP_s001
                                 + 0.0184154405810002 w_k1_AXP_s002
                                 + 0.0332496430952968 w_k1_AXP_s003
                                 + 0.0279933523748604 w_k1_AXP_s004
                                 + 0.00778339686436362 w_k1_AXP_s005
                                 + 0.0079704522200205 w_k1_AXP_s006
                                 + 0.0184716061288522 w_k1_AXP_s007
                                 + 0.0144751370338942 w_k1_AXP_s010
                                 + 0.0256924853178082 w_k1_AXP_s011
                                 + 0.0132121547533792 w_k1_AXP_s012
                                 + 0.0114966715491995 w_k1_AXP_s013
                                 + 0.0174070730144155 w_k1_AXP_s014
                                 + 0.00879940374634921 w_k1_AXP_s015
                                 + 0.00930095236986013 w_k1_AXP_s016
                                 + 0.00826066857668115 w_k1_AXP_s017
                                 + 0.0243772481113722 w_k1_AXP_s100
                                 + 0.0453081163745408 w_k1_AXP_s101
                                 + 0.0467934428949886 w_k1_AXP_s102
                                 + 0.0257457204564811 w_k1_AXP_s103
                                 + 0.0994431409770734 w_k1_AXP_s104
                                 + 0.0493970587894055 w_k1_AXP_s105
                                 + 0.0327452676191957 w_k1_AXP_s106
                                 + 0.0356655622169189 w_k1_AXP_s107
                                 + 0.0786061636737296 w_k1_AXP_s110
                                 + 0.0710258594359074 w_k1_AXP_s111
                                 + 0.0401995071962425 w_k1_AXP_s112
                                 + 0.0337143114317721 w_k1_AXP_s113
                                 + 0.0348828682353852 w_k1_AXP_s114
                                 + 0.0641651795859449 w_k1_AXP_s115
                                 + 0.0465524238682947 w_k1_AXP_s116
                                 + 0.0447530128584376 w_k1_AXP_s117
                                 + 0.0164920433854153 w_k1_DIS_s000
                                 + 0.019854374976018 w_k1_DIS_s001
                                 + 0.0204072469962571 w_k1_DIS_s002
                                 + 0.0323806960419464 w_k1_DIS_s003
                                 + 0.0291458225533101 w_k1_DIS_s004
                                 + 0.00824279947873872 w_k1_DIS_s005
                                 + 0.00808288139661305 w_k1_DIS_s006
                                 + 0.0194772325228545 w_k1_DIS_s007
                                 + 0.016086620509281 w_k1_DIS_s010
                                 + 0.0254745537837526 w_k1_DIS_s011
                                 + 0.0144403609524614 w_k1_DIS_s012
                                 + 0.0115551408717063 w_k1_DIS_s013
                                 + 0.0183955103784319 w_k1_DIS_s014
                                 + 0.00812611255207639 w_k1_DIS_s015
                                 + 0.00998241412662632 w_k1_DIS_s016
                                 + 0.00924400871415992 w_k1_DIS_s017
                                 + 0.025588185744094 w_k1_DIS_s100
                                 + 0.0466226622616997 w_k1_DIS_s101
                                 + 0.0490710184770419 w_k1_DIS_s102
                                 + 0.0276924730646 w_k1_DIS_s103
                                 + 0.0912346489269365 w_k1_DIS_s104
                                 + 0.0476649637323941 w_k1_DIS_s105
                                 + 0.0319194183748719 w_k1_DIS_s106
                                 + 0.0367819309125483 w_k1_DIS_s107
                                 + 0.0815456948441929 w_k1_DIS_s110
                                 + 0.0687651752203283 w_k1_DIS_s111
                                 + 0.0345678711673992 w_k1_DIS_s112
                                 + 0.0330911479442712 w_k1_DIS_s113
                                 + 0.0360489280301049 w_k1_DIS_s114
                                 + 0.0629169609973147 w_k1_DIS_s115
                                 + 0.0514317470353245 w_k1_DIS_s116
                                 + 0.0484692861637824 w_k1_DIS_s117
                                 + 0.0167735298452619 w_k1_FDX_s000
                                 + 0.0200645479543813 w_k1_FDX_s001
                                 + 0.0187311611921016 w_k1_FDX_s002
                                 + 0.0312014922918413 w_k1_FDX_s003
                                 + 0.025369711818177 w_k1_FDX_s004
                                 + 0.00832079421549463 w_k1_FDX_s005
                                 + 0.00775369018559952 w_k1_FDX_s006
                                 + 0.0192010507977545 w_k1_FDX_s007
                                 + 0.0151220636216583 w_k1_FDX_s010
                                 + 0.0271797294693555 w_k1_FDX_s011
                                 + 0.0130455616700434 w_k1_FDX_s012
                                 + 0.0106571014882233 w_k1_FDX_s013
                                 + 0.0167295329006113 w_k1_FDX_s014
                                 + 0.0080834423653987 w_k1_FDX_s015
                                 + 0.00949300352292495 w_k1_FDX_s016
                                 + 0.00938473542450113 w_k1_FDX_s017
                                 + 0.0234847804044581 w_k1_FDX_s100
                                 + 0.0461061376136636 w_k1_FDX_s101
                                 + 0.0491294492022126 w_k1_FDX_s102
                                 + 0.0272208531494147 w_k1_FDX_s103
                                 + 0.0928748344355996 w_k1_FDX_s104
                                 + 0.0483702143939594 w_k1_FDX_s105
                                 + 0.0329935876766119 w_k1_FDX_s106
                                 + 0.0343812259364248 w_k1_FDX_s107
                                 + 0.0783223852235806 w_k1_FDX_s110
                                 + 0.0670978995905481 w_k1_FDX_s111
                                 + 0.0352283215249714 w_k1_FDX_s112
                                 + 0.0334528422917861 w_k1_FDX_s113
                                 + 0.0322461481111258 w_k1_FDX_s114
                                 + 0.0621775161517184 w_k1_FDX_s115
                                 + 0.0483360508267735 w_k1_FDX_s116
                                 + 0.0433669556408451 w_k1_FDX_s117
                                 + 0.0161376866679612 w_k1_IBM_s000
                                 + 0.0197972351941797 w_k1_IBM_s001
                                 + 0.0204563076578615 w_k1_IBM_s002
                                 + 0.0325701350997993 w_k1_IBM_s003
                                 + 0.027546782183532 w_k1_IBM_s004
                                 + 0.00819362859653019 w_k1_IBM_s005
                                 + 0.00793567462987997 w_k1_IBM_s006
                                 + 0.0194344567105973 w_k1_IBM_s007
                                 + 0.0153823510105141 w_k1_IBM_s010
                                 + 0.0263266521200299 w_k1_IBM_s011
                                 + 0.0139353831681793 w_k1_IBM_s012
                                 + 0.0102210414121508 w_k1_IBM_s013
                                 + 0.0170033449686961 w_k1_IBM_s014
                                 + 0.00837972364451122 w_k1_IBM_s015
                                 + 0.00967178593834254 w_k1_IBM_s016
                                 + 0.00982784214392432 w_k1_IBM_s017
                                 + 0.0247211236127566 w_k1_IBM_s100
                                 + 0.046495231482018 w_k1_IBM_s101
                                 + 0.0480977632854507 w_k1_IBM_s102
                                 + 0.0286126239299993 w_k1_IBM_s103
                                 + 0.0927599070868862 w_k1_IBM_s104
                                 + 0.051564160505153 w_k1_IBM_s105
                                 + 0.0323621127901399 w_k1_IBM_s106
                                 + 0.0334424380565554 w_k1_IBM_s107
                                 + 0.0810448504772967 w_k1_IBM_s110
                                 + 0.0646156729155799 w_k1_IBM_s111
                                 + 0.0359409514515453 w_k1_IBM_s112
                                 + 0.0346795272598872 w_k1_IBM_s113
                                 + 0.0362983528249447 w_k1_IBM_s114
                                 + 0.0616996781434628 w_k1_IBM_s115
                                 + 0.0484046926811697 w_k1_IBM_s116
                                 + 0.0449095843225326 w_k1_IBM_s117
                                 + 0.018071463861413 w_k1_INTC_s000
                                 + 0.0196982878241083 w_k1_INTC_s001
                                 + 0.0172133788750942 w_k1_INTC_s002
                                 + 0.0280814283777164 w_k1_INTC_s003
                                 + 0.0236604189661868 w_k1_INTC_s004
                                 + 0.00901114337995983 w_k1_INTC_s005
                                 + 0.00774573942591714 w_k1_INTC_s006
                                 + 0.0171735969745741 w_k1_INTC_s007
                                 + 0.0155033894692009 w_k1_INTC_s010
                                 + 0.0257462458583487 w_k1_INTC_s011
                                 + 0.014898575542207 w_k1_INTC_s012
                                 + 0.0108481082931852 w_k1_INTC_s013
                                 + 0.0181295948928505 w_k1_INTC_s014
                                 + 0.00787417538363161 w_k1_INTC_s015
                                 + 0.00945722307967849 w_k1_INTC_s016
                                 + 0.00916105372637575 w_k1_INTC_s017
                                 + 0.0242747258735202 w_k1_INTC_s100
                                 + 0.0463396088034804 w_k1_INTC_s101
                                 + 0.0449543196082328 w_k1_INTC_s102
                                 + 0.0279987317796828 w_k1_INTC_s103
                                 + 0.0988397974333656 w_k1_INTC_s104
                                 + 0.0500164647915688 w_k1_INTC_s105
                                 + 0.0307774129101363 w_k1_INTC_s106
                                 + 0.0332808914832169 w_k1_INTC_s107
                                 + 0.0682745785437159 w_k1_INTC_s110
                                 + 0.0625832418159698 w_k1_INTC_s111
                                 + 0.0361141503831701 w_k1_INTC_s112
                                 + 0.0303174058359135 w_k1_INTC_s113
                                 + 0.0333714936111149 w_k1_INTC_s114
                                 + 0.0560604771989916 w_k1_INTC_s115
                                 + 0.0498084094505379 w_k1_INTC_s116
                                 + 0.0401204275473432 w_k1_INTC_s117
                                 + 0.0156812509232109 w_k1_KO_s000
                                 + 0.019036594829076 w_k1_KO_s001
                                 + 0.0179596396074216 w_k1_KO_s002
                                 + 0.0346593296772395 w_k1_KO_s003
                                 + 0.0272923526911571 w_k1_KO_s004
                                 + 0.00697664618404761 w_k1_KO_s005
                                 + 0.00887287354014829 w_k1_KO_s006
                                 + 0.0195579553082812 w_k1_KO_s007
                                 + 0.0151102086881666 w_k1_KO_s010
                                 + 0.0247598540076252 w_k1_KO_s011
                                 + 0.0128955630088322 w_k1_KO_s012
                                 + 0.0111385600442389 w_k1_KO_s013
                                 + 0.0173434241236602 w_k1_KO_s014
                                 + 0.00721496959754951 w_k1_KO_s015
                                 + 0.00985081734719956 w_k1_KO_s016
                                 + 0.00913982634653707 w_k1_KO_s017
                                 + 0.0222444302270364 w_k1_KO_s100
                                 + 0.0432363148084546 w_k1_KO_s101
                                 + 0.0460147569890864 w_k1_KO_s102
                                 + 0.0257405436135937 w_k1_KO_s103
                                 + 0.090351843703028 w_k1_KO_s104
                                 + 0.0513479184411078 w_k1_KO_s105
                                 + 0.0323705272856992 w_k1_KO_s106
                                 + 0.034127642484792 w_k1_KO_s107
                                 + 0.0769794159667504 w_k1_KO_s110
                                 + 0.0675661408501807 w_k1_KO_s111
                                 + 0.0345499676599632 w_k1_KO_s112
                                 + 0.0355307118692793 w_k1_KO_s113
                                 + 0.0316213367119521 w_k1_KO_s114
                                 + 0.0592112114742894 w_k1_KO_s115
                                 + 0.0475107685128211 w_k1_KO_s116
                                 + 0.0478864927069324 w_k1_KO_s117
                                 + 0.0173858163671201 w_k1_MAT_s000
                                 + 0.0221309249492968 w_k1_MAT_s001
                                 + 0.0184795983010267 w_k1_MAT_s002
                                 + 0.033323121010588 w_k1_MAT_s003
                                 + 0.0259174869372478 w_k1_MAT_s004
                                 + 0.00598313051502685 w_k1_MAT_s005
                                 + 0.00730507547273724 w_k1_MAT_s006
                                 + 0.0148566069549449 w_k1_MAT_s007
                                 + 0.0127089512519336 w_k1_MAT_s010
                                 + 0.0233239183629689 w_k1_MAT_s011
                                 + 0.0144794072581236 w_k1_MAT_s012
                                 + 0.0119462388883011 w_k1_MAT_s013
                                 + 0.0158693131171078 w_k1_MAT_s014
                                 + 0.00854438045374169 w_k1_MAT_s015
                                 + 0.0113466879094515 w_k1_MAT_s016
                                 + 0.0104508345296613 w_k1_MAT_s017
                                 + 0.0259511574484439 w_k1_MAT_s100
                                 + 0.038531157461246 w_k1_MAT_s101
                                 + 0.046167227231532 w_k1_MAT_s102
                                 + 0.0199060431615181 w_k1_MAT_s103
                                 + 0.0907584512466679 w_k1_MAT_s104
                                 + 0.0591532609166735 w_k1_MAT_s105
                                 + 0.0309630994126426 w_k1_MAT_s106
                                 + 0.0358230381421736 w_k1_MAT_s107
                                 + 0.0798402430626036 w_k1_MAT_s110
                                 + 0.0661088875203392 w_k1_MAT_s111
                                 + 0.0468158386369521 w_k1_MAT_s112
                                 + 0.0344158434047666 w_k1_MAT_s113
                                 + 0.0259476798080172 w_k1_MAT_s114
                                 + 0.0669981794912643 w_k1_MAT_s115
                                 + 0.0599165314692253 w_k1_MAT_s116
                                 + 0.0483180179686399 w_k1_MAT_s117
                                 + 0.0162119195743497 w_k1_MSFT_s000
                                 + 0.0217490769641804 w_k1_MSFT_s001
                                 + 0.0184899861619327 w_k1_MSFT_s002
                                 + 0.0307363201139638 w_k1_MSFT_s003
                                 + 0.0253466934244704 w_k1_MSFT_s004
                                 + 0.00805742003579467 w_k1_MSFT_s005
                                 + 0.00807361927913173 w_k1_MSFT_s006
                                 + 0.0175742819970659 w_k1_MSFT_s007
                                 + 0.0150820358196663 w_k1_MSFT_s010
                                 + 0.0251309213954323 w_k1_MSFT_s011
                                 + 0.0125394541250751 w_k1_MSFT_s012
                                 + 0.0116856918997566 w_k1_MSFT_s013
                                 + 0.0176428108490487 w_k1_MSFT_s014
                                 + 0.00778648333668376 w_k1_MSFT_s015
                                 + 0.00875587255194481 w_k1_MSFT_s016
                                 + 0.00966352106172909 w_k1_MSFT_s017
                                 + 0.0239270396347383 w_k1_MSFT_s100
                                 + 0.0443830849410892 w_k1_MSFT_s101
                                 + 0.0472911401483758 w_k1_MSFT_s102
                                 + 0.0260118048972536 w_k1_MSFT_s103
                                 + 0.0903370777116194 w_k1_MSFT_s104
                                 + 0.054008737795539 w_k1_MSFT_s105
                                 + 0.0323050711986553 w_k1_MSFT_s106
                                 + 0.0337008962400802 w_k1_MSFT_s107
                                 + 0.0813293478369758 w_k1_MSFT_s110
                                 + 0.0692425371332752 w_k1_MSFT_s111
                                 + 0.0330597211140686 w_k1_MSFT_s112
                                 + 0.0336502599844063 w_k1_MSFT_s113
                                 + 0.0344907606144995 w_k1_MSFT_s114
                                 + 0.05890161133418 w_k1_MSFT_s115
                                 + 0.0474306200199837 w_k1_MSFT_s116
                                 + 0.0428005614808813 w_k1_MSFT_s117
                                 + 0.0179383003543839 w_k1_PG_s000
                                 + 0.0199574433382865 w_k1_PG_s001
                                 + 0.0198984734739774 w_k1_PG_s002
                                 + 0.031881157336237 w_k1_PG_s003
                                 + 0.0269183725824809 w_k1_PG_s004
                                 + 0.00795366186551455 w_k1_PG_s005
                                 + 0.00798177112988471 w_k1_PG_s006
                                 + 0.0173259628800207 w_k1_PG_s007
                                 + 0.0155683222538863 w_k1_PG_s010
                                 + 0.0285048066428189 w_k1_PG_s011
                                 + 0.0120007108970879 w_k1_PG_s012
                                 + 0.0102313440936452 w_k1_PG_s013
                                 + 0.0152932653427819 w_k1_PG_s014
                                 + 0.00804534944535233 w_k1_PG_s015
                                 + 0.009184674162072 w_k1_PG_s016
                                 + 0.00903248680989692 w_k1_PG_s017
                                 + 0.0239924781939977 w_k1_PG_s100
                                 + 0.0462363575924531 w_k1_PG_s101
                                 + 0.0535562278769673 w_k1_PG_s102
                                 + 0.0256020935545766 w_k1_PG_s103
                                 + 0.0902941526987986 w_k1_PG_s104
                                 + 0.0518799624444754 w_k1_PG_s105
                                 + 0.031508940513702 w_k1_PG_s106
                                 + 0.0342365593907153 w_k1_PG_s107
                                 + 0.078831155069871 w_k1_PG_s110
                                 + 0.0665334605844178 w_k1_PG_s111
                                 + 0.0359767677255448 w_k1_PG_s112
                                 + 0.0342091518450185 w_k1_PG_s113
                                 + 0.0331796953331882 w_k1_PG_s114
                                 + 0.0619514167556511 w_k1_PG_s115
                                 + 0.0484463294315287 w_k1_PG_s116
                                 + 0.0432690801567497 w_k1_PG_s117 >= 1.1
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

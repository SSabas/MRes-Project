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
 w_k1_ADM_s0_return_constraint:  1.16652 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.9777 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.989985 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.09742 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.980273 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.891781 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.912298 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.21355 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.900108 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.887789 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.1137 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.27057 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.984825 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.721929 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.821291 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.0522 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.971815 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.37852 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.847613 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.3994 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.98853 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.56413 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.2276 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.18716 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.48494 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.22822 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.0664 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.45442 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.32907 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.923572 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.03302 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.664833 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.13633 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.11913 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.949973 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.27378 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.72726 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.881114 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.28276 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.905967 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.949162 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.27123 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.25825 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.953352 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.916976 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.03394 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.651586 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.824064 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.34016 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.17791 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.09011 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.634656 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.747877 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.0118 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.842896 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.2951 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.90564 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.67729 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.878677 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.37878 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.07684 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.00595 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.35678 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.13236 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.05921 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.5226 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.2813 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.708647 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.11841 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.584769 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.09275 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.950876 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.792286 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.52827 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.564848 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.765241 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.3216 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.870396 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.951066 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.3539 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.42233 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.888117 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.923042 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.26474 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.77156 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.832924 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.2094 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.57572 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.10891 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.582272 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.859288 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.08373 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.76924 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.33614 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.950675 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.83374 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.651838 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.96475 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.03285 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.93636 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.44665 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.18531 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.866452 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.91419 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.28747 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.688087 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.18922 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.534273 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.10038 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.997463 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.22232 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.67427 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.58279 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.765283 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.18114 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.914907 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.98902 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.16621 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.24129 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.906715 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.889263 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.28895 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.822093 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.994667 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.28343 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.24928 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.997188 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.680849 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.80333 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.01117 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.958131 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.27823 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.894776 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.35881 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.779478 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.6176 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.11266 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.981883 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.34163 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.04701 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.912926 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.43917 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.0391 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.748619 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.02878 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.616842 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.11085 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.10057 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.991013 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.49076 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.773718 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.839739 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.14689 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.979636 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.896853 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.54099 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.23844 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.884361 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.14823 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.587386 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.751086 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.773162 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.909002 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.23519 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.72101 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.744133 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.737841 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.01521 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.608753 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.882498 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.45452 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.55429 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.956647 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.868169 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.689535 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.860029 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.937663 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.33361 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.37789 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.38591 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.88531 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.871405 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.73569 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.88654 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.09503 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.685101 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.02595 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.19958 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.555476 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.999783 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.18076 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.966201 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.982538 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.10861 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.16741 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.84212 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.888174 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.995915 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.955526 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.940391 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.24416 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.10634 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.977243 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.742048 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.780057 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.957679 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.911656 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.05065 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.14717 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.27145 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.831803 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.12857 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.931398 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.811717 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.05973 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.993289 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.912927 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.21759 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.06024 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.811877 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.09533 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.703677 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.1222 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.09052 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.16364 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.55557 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.942248 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.8669 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.18551 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.933511 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.910082 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.38823 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.29038 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.863337 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.01313 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.880655 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.833475 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.832309 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.03773 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.37966 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.41107 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.676751 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.774641 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.04926 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.780177 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.960628 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.06753 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.49307 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.811182 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.1378 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.889843 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.878785 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.16754 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.15446 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.12139 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.41557 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.46022 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.877929 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.3958 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.671432 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.0492 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.895118 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.24225 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.32166 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.688731 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.898629 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.31735 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.888252 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.803598 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.67856 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.3101 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.907581 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.09659 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.675297 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.559158 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.71212 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.14517 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.10215 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.74576 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.597704 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.592274 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.780375 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.655756 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.895584 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.30709 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.61905 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.770523 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.635106 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.909956 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.03615 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.19915 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.34802 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.54655 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.42792 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 2.22359 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.81713 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.97706 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.86521 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.1221 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.730102 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.07797 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.2837 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.447553 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.13596 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.18184 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.924697 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.01913 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.18325 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.15206 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.879803 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.871801 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.09337 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.867872 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.960479 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.15609 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.16293 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.01 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.658237 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.78186 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.976076 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.890616 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.19467 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.947335 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.31591 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.767588 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.24464 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.16156 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.960831 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.35073 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.24918 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.948907 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.29851 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.17845 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.882012 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.03742 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.687415 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.00514 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.17706 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.16543 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.39949 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.822969 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.855808 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.15796 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.934914 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.870984 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.13477 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.16807 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.928138 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.967792 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.07898 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.95968 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.01187 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.1494 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.31309 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.25849 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.86236 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.854089 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.03135 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.962036 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.17282 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.978761 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.30247 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.842899 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.20367 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.999813 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.95594 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.21249 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.07555 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.00409 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.33307 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.18427 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.837227 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.07585 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.77836 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.09183 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.992779 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.0642 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.27118 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.830878 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.890287 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.433001138793313 z_k1_s0000
                                 - 0.447001607448423 z_k1_s0001
                                 - 0.422001327065953 z_k1_s0002
                                 - 0.681002206162311 z_k1_s0003
                                 - 0.809003151999099 z_k1_s0004
                                 - 0.638001323223272 z_k1_s0005
                                 - 0.798001114265116 z_k1_s0006
                                 - 1.33700635794878 z_k1_s0007
                                 - 0.114000079215356 z_k1_s0010
                                 - 0.204999608720417 z_k1_s0011
                                 - 0.232000647771764 z_k1_s0012
                                 - 0.274000691851111 z_k1_s0013
                                 - 0.0910000955052508 z_k1_s0014
                                 - 0.16299956464107 z_k1_s0015
                                 - 0.152999756288913 z_k1_s0016
                                 - 0.18300059634586 z_k1_s0017
                                 - 0.412999721017818 z_k1_s0020
                                 - 0.398999613525626 z_k1_s0021
                                 - 0.120000034218314 z_k1_s0022
                                 - 0.350999737838374 z_k1_s0023
                                 - 0.0880000595934486 z_k1_s0024
                                 - 0.0860000935231618 z_k1_s0025
                                 - 0.166000116335371 z_k1_s0026
                                 - 0.101999994585548 z_k1_s0027
                                 - 0.468000412185516 z_k1_s0030
                                 - 0.628003498740691 z_k1_s0031
                                 - 0.393002834192507 z_k1_s0032
                                 - 0.337000582792981 z_k1_s0033
                                 - 0.195000729413115 z_k1_s0034
                                 - 0.511003395694312 z_k1_s0035
                                 - 0.668001760868309 z_k1_s0036
                                 - 0.146000683397098 z_k1_s0037
                                 - 0.364000023805637 z_k1_s0040
                                 - 0.504001494499807 z_k1_s0041
                                 - 0.217999874367141 z_k1_s0042
                                 - 0.243999848924693 z_k1_s0043
                                 - 0.505000894290607 z_k1_s0044
                                 - 0.752000842587303 z_k1_s0045
                                 - 0.252999767040783 z_k1_s0046
                                 - 0.960001399047554 z_k1_s0047
                                 - 0.360000431308345 z_k1_s0050
                                 - 0.335000302592327 z_k1_s0051
                                 - 0.673002546073711 z_k1_s0052
                                 - 1.03199923074732 z_k1_s0053
                                 - 1.10299994638138 z_k1_s0054
                                 - 0.343000737622094 z_k1_s0055
                                 - 0.436000735084897 z_k1_s0056
                                 - 1.18800388482155 z_k1_s0057
                                 - 0.175000194945789 z_k1_s0060
                                 - 0.252000313697961 z_k1_s0061
                                 - 0.532999744223944 z_k1_s0062
                                 - 0.178000188865995 z_k1_s0063
                                 - 0.420999421602487 z_k1_s0064
                                 - 0.557000794786445 z_k1_s0065
                                 - 0.162000221291561 z_k1_s0066
                                 - 0.469999047498976 z_k1_s0067
                                 - 0.516002452636128 z_k1_s0070
                                 - 0.283000731170971 z_k1_s0071
                                 - 0.250000774043586 z_k1_s0072
                                 - 0.760000978969018 z_k1_s0073
                                 - 0.498001435018552 z_k1_s0074
                                 - 0.4340016338324 z_k1_s0075
                                 - 0.786002448861072 z_k1_s0076
                                 - 0.290000745210172 z_k1_s0077
                                 - 0.0989998172216961 z_k1_s0100
                                 - 0.182999714682364 z_k1_s0101
                                 - 0.0969996545207717 z_k1_s0102
                                 - 0.0729997829031261 z_k1_s0103
                                 - 0.113999736982725 z_k1_s0104
                                 - 0.0889998707143438 z_k1_s0105
                                 - 0.0389998780784011 z_k1_s0106
                                 - 0.172999768175011 z_k1_s0107
                                 - 0.330999190444599 z_k1_s0110
                                 - 0.1939996876015 z_k1_s0111
                                 - 0.176999531884374 z_k1_s0112
                                 - 0.236000115843687 z_k1_s0113
                                 - 0.423998420842991 z_k1_s0114
                                 - 0.221999356431412 z_k1_s0115
                                 - 0.117999687922916 z_k1_s0116
                                 - 0.147999570954275 z_k1_s0117
                                 - 0.0699999677200989 z_k1_s0120
                                 - 0.0409999029505326 z_k1_s0121
                                 - 0.0849997654463067 z_k1_s0122
                                 - 0.0539998735128933 z_k1_s0123
                                 - 0.076999909792241 z_k1_s0124
                                 - 0.0909997939365253 z_k1_s0125
                                 - 0.10499967808081 z_k1_s0126
                                 - 0.0239999498612712 z_k1_s0127
                                 - 0.146999179509702 z_k1_s0130
                                 - 0.10899982300306 z_k1_s0131
                                 - 0.0909997940531122 z_k1_s0132
                                 - 0.0989998709752026 z_k1_s0133
                                 - 0.106999832597489 z_k1_s0134
                                 - 0.0979998757724168 z_k1_s0135
                                 - 0.180999938802852 z_k1_s0136
                                 - 0.320999267192857 z_k1_s0137
                                 - 0.185999644292724 z_k1_s0140
                                 - 0.0559999494213617 z_k1_s0141
                                 - 0.0759998687862484 z_k1_s0142
                                 - 0.0719998411133513 z_k1_s0143
                                 - 0.248999258892358 z_k1_s0144
                                 - 0.0679999229402536 z_k1_s0145
                                 - 0.164000039590787 z_k1_s0146
                                 - 0.223999359686249 z_k1_s0147
                                 - 0.0909994730600568 z_k1_s0150
                                 - 0.116999993504276 z_k1_s0151
                                 - 0.136999774462763 z_k1_s0152
                                 - 0.109999466270166 z_k1_s0153
                                 - 0.0589998906263269 z_k1_s0154
                                 - 0.0349999521765962 z_k1_s0155
                                 - 0.0539998782868564 z_k1_s0156
                                 - 0.0679996578579482 z_k1_s0157
                                 - 0.210999650612075 z_k1_s0160
                                 - 0.0799997108412308 z_k1_s0161
                                 - 0.100999576307291 z_k1_s0162
                                 - 0.0939995830187585 z_k1_s0163
                                 - 0.216998762348672 z_k1_s0164
                                 - 0.148999677370919 z_k1_s0165
                                 - 0.149999529327019 z_k1_s0166
                                 - 0.141999569682849 z_k1_s0167
                                 - 0.0309999683677031 z_k1_s0170
                                 - 0.0309999683677031 z_k1_s0171
                                 - 0.107000050959334 z_k1_s0172
                                 - 0.0329999498043453 z_k1_s0173
                                 - 0.036999952077591 z_k1_s0174
                                 - 0.0629997501539013 z_k1_s0175
                                 - 0.0489997618975217 z_k1_s0176
                                 - 0.042999935787479 z_k1_s0177
                                 - 1.0449988665762 z_k1_s1000
                                 - 0.257999661030413 z_k1_s1001
                                 - 0.45299722986 z_k1_s1002
                                 - 0.557999343530743 z_k1_s1003
                                 - 0.829997089054907 z_k1_s1004
                                 - 0.442000971467658 z_k1_s1005
                                 - 0.313999414990842 z_k1_s1006
                                 - 0.219999739700296 z_k1_s1007
                                 - 0.602001132035187 z_k1_s1010
                                 - 0.690002526631103 z_k1_s1011
                                 - 0.357001756488395 z_k1_s1012
                                 - 0.291000543891088 z_k1_s1013
                                 - 0.336000476809993 z_k1_s1014
                                 - 0.24200066878173 z_k1_s1015
                                 - 0.634001336160301 z_k1_s1016
                                 - 0.181000300499528 z_k1_s1017
                                 - 0.460998818151905 z_k1_s1020
                                 - 0.351999627568838 z_k1_s1021
                                 - 0.141999868266957 z_k1_s1022
                                 - 0.330000668844801 z_k1_s1023
                                 - 0.106999840516709 z_k1_s1024
                                 - 0.1329999425711 z_k1_s1025
                                 - 0.196999708274686 z_k1_s1026
                                 - 0.31399976031984 z_k1_s1027
                                 - 0.152000492059097 z_k1_s1030
                                 - 0.0620000451819795 z_k1_s1031
                                 - 0.145999477267159 z_k1_s1032
                                 - 0.170999962566406 z_k1_s1033
                                 - 0.270999539763081 z_k1_s1034
                                 - 0.163000185510697 z_k1_s1035
                                 - 0.143999533003232 z_k1_s1036
                                 - 0.0729999750336107 z_k1_s1037
                                 - 0.180000586493791 z_k1_s1040
                                 - 0.422001219626011 z_k1_s1041
                                 - 0.176000723478899 z_k1_s1042
                                 - 0.522002617015205 z_k1_s1043
                                 - 0.186000622116974 z_k1_s1044
                                 - 0.402001904551551 z_k1_s1045
                                 - 0.120000471362809 z_k1_s1046
                                 - 0.403002473057386 z_k1_s1047
                                 - 0.13100011795694 z_k1_s1050
                                 - 0.226000211452424 z_k1_s1051
                                 - 0.092999872158941 z_k1_s1052
                                 - 0.0839999181435283 z_k1_s1053
                                 - 0.210999593427383 z_k1_s1054
                                 - 0.0479999978819746 z_k1_s1055
                                 - 0.0889999504853703 z_k1_s1056
                                 - 0.159999992939915 z_k1_s1057
                                 - 0.118999911778978 z_k1_s1060
                                 - 0.129999949926564 z_k1_s1061
                                 - 0.347999825034232 z_k1_s1062
                                 - 0.255000625239789 z_k1_s1063
                                 - 0.179999726052387 z_k1_s1064
                                 - 0.160999746524644 z_k1_s1065
                                 - 0.157999839757008 z_k1_s1066
                                 - 0.54899870846027 z_k1_s1067
                                 - 0.478999521060509 z_k1_s1070
                                 - 1.2589992446626 z_k1_s1071
                                 - 0.488999592932269 z_k1_s1072
                                 - 0.42599963033978 z_k1_s1073
                                 - 0.868001336472783 z_k1_s1074
                                 - 0.443999759708949 z_k1_s1075
                                 - 0.265999950990419 z_k1_s1076
                                 - 0.670999920599104 z_k1_s1077
                                 - 0.614998406426317 z_k1_s1100
                                 - 1.09699728247585 z_k1_s1101
                                 - 1.2989942871404 z_k1_s1102
                                 - 0.910997809692151 z_k1_s1103
                                 - 0.569998555356005 z_k1_s1104
                                 - 1.13399571681338 z_k1_s1105
                                 - 0.313998946075105 z_k1_s1106
                                 - 0.686999228455758 z_k1_s1107
                                 - 0.234999916188253 z_k1_s1110
                                 - 0.38000003581075 z_k1_s1111
                                 - 0.470999529733975 z_k1_s1112
                                 - 0.398998887201567 z_k1_s1113
                                 - 0.426999616408498 z_k1_s1114
                                 - 0.479000559792899 z_k1_s1115
                                 - 0.388999037627563 z_k1_s1116
                                 - 0.0959999939100879 z_k1_s1117
                                 - 1.68400212921401 z_k1_s1120
                                 - 1.1239981183395 z_k1_s1121
                                 - 1.26000398646689 z_k1_s1122
                                 - 2.30799882555874 z_k1_s1123
                                 - 0.548000470298112 z_k1_s1124
                                 - 1.0509974875362 z_k1_s1125
                                 - 1.63100236137062 z_k1_s1126
                                 - 0.470000127554909 z_k1_s1127
                                 - 1.60400253068166 z_k1_s1130
                                 - 0.593000345636057 z_k1_s1131
                                 - 1.09399627452005 z_k1_s1132
                                 - 1.09499732692864 z_k1_s1133
                                 - 0.590999952019587 z_k1_s1134
                                 - 0.965998462277104 z_k1_s1135
                                 - 0.909999419420815 z_k1_s1136
                                 - 1.70300404708962 z_k1_s1137
                                 - 0.217000227564103 z_k1_s1140
                                 - 0.121000023935672 z_k1_s1141
                                 - 0.388000555714722 z_k1_s1142
                                 - 0.220000144064929 z_k1_s1143
                                 - 0.328999838396834 z_k1_s1144
                                 - 0.415000246622462 z_k1_s1145
                                 - 0.265999822278258 z_k1_s1146
                                 - 0.25599936327499 z_k1_s1147
                                 - 0.375000131872701 z_k1_s1150
                                 - 0.322000201816068 z_k1_s1151
                                 - 1.40099911705977 z_k1_s1152
                                 - 0.432998787073705 z_k1_s1153
                                 - 0.3909994913763 z_k1_s1154
                                 - 0.315000188933106 z_k1_s1155
                                 - 0.36500033678286 z_k1_s1156
                                 - 0.306000060712102 z_k1_s1157
                                 - 0.288000115419149 z_k1_s1160
                                 - 0.885002466175189 z_k1_s1161
                                 - 0.1789999829028 z_k1_s1162
                                 - 0.821998861624264 z_k1_s1163
                                 - 0.771000067086479 z_k1_s1164
                                 - 0.548001865085032 z_k1_s1165
                                 - 0.210000338360008 z_k1_s1166
                                 - 0.233000373544198 z_k1_s1167
                                 - 0.742995531977383 z_k1_s1170
                                 - 0.555998879087474 z_k1_s1171
                                 - 0.963995893014895 z_k1_s1172
                                 - 0.815999303837487 z_k1_s1173
                                 - 0.788996824659919 z_k1_s1174
                                 - 0.391998651752716 z_k1_s1175
                                 - 0.589000150532509 z_k1_s1176
                                 - 0.426998465423711 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.861771 w_k1_ADM_s000
                                 + 0.967761 w_k1_AXP_s000
                                 + 0.925776 w_k1_DIS_s000
                                 + 0.890974 w_k1_FDX_s000
                                 + 1.41194 w_k1_IBM_s000
                                 + 1.11917 w_k1_INTC_s000
                                 + 1.22492 w_k1_KO_s000 + 1.64422 w_k1_MAT_s000
                                 + 0.948557 w_k1_MSFT_s000
                                 + 1.09565 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.51384 w_k1_ADM_s000
                                 + 1.57722 w_k1_AXP_s000
                                 + 1.41763 w_k1_DIS_s000
                                 + 1.41749 w_k1_FDX_s000
                                 + 1.06044 w_k1_IBM_s000
                                 + 1.17548 w_k1_INTC_s000
                                 + 1.28839 w_k1_KO_s000 + 1.25146 w_k1_MAT_s000
                                 + 1.21999 w_k1_MSFT_s000
                                 + 1.29767 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.78227 w_k1_ADM_s000
                                 + 0.730856 w_k1_AXP_s000
                                 + 0.866757 w_k1_DIS_s000
                                 + 0.952351 w_k1_FDX_s000
                                 + 0.647444 w_k1_IBM_s000
                                 + 1.01434 w_k1_INTC_s000
                                 + 0.868037 w_k1_KO_s000
                                 + 0.800071 w_k1_MAT_s000
                                 + 1.01575 w_k1_MSFT_s000
                                 + 0.951493 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.996545 w_k1_ADM_s000
                                 + 0.947051 w_k1_AXP_s000
                                 + 0.98333 w_k1_DIS_s000
                                 + 0.961956 w_k1_FDX_s000
                                 + 1.11804 w_k1_IBM_s000
                                 + 0.947729 w_k1_INTC_s000
                                 + 1.08423 w_k1_KO_s000
                                 + 0.985542 w_k1_MAT_s000
                                 + 1.02343 w_k1_MSFT_s000
                                 + 1.02128 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.729111 w_k1_ADM_s000
                                 + 0.722382 w_k1_AXP_s000
                                 + 0.650328 w_k1_DIS_s000
                                 + 0.754851 w_k1_FDX_s000
                                 + 0.824852 w_k1_IBM_s000
                                 + 0.739988 w_k1_INTC_s000
                                 + 0.795737 w_k1_KO_s000
                                 + 0.74802 w_k1_MAT_s000
                                 + 0.787872 w_k1_MSFT_s000
                                 + 0.949813 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.855867 w_k1_ADM_s000
                                 + 0.976885 w_k1_AXP_s000
                                 + 0.689684 w_k1_DIS_s000
                                 + 0.788079 w_k1_FDX_s000
                                 + 1.10825 w_k1_IBM_s000
                                 + 0.730832 w_k1_INTC_s000
                                 + 0.850871 w_k1_KO_s000
                                 + 1.19317 w_k1_MAT_s000
                                 + 0.678385 w_k1_MSFT_s000
                                 + 1.00166 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.11655 w_k1_ADM_s000
                                 + 1.26896 w_k1_AXP_s000
                                 + 1.34067 w_k1_DIS_s000
                                 + 1.16612 w_k1_FDX_s000
                                 + 1.32322 w_k1_IBM_s000
                                 + 1.10588 w_k1_INTC_s000
                                 + 1.35553 w_k1_KO_s000 + 1.32327 w_k1_MAT_s000
                                 + 1.14775 w_k1_MSFT_s000 + 1.3739 w_k1_PG_s000
                                 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.13516 w_k1_ADM_s000
                                 + 1.07756 w_k1_AXP_s000
                                 + 1.13584 w_k1_DIS_s000
                                 + 1.13914 w_k1_FDX_s000
                                 + 0.798678 w_k1_IBM_s000
                                 + 0.951557 w_k1_INTC_s000
                                 + 0.957223 w_k1_KO_s000
                                 + 0.845415 w_k1_MAT_s000
                                 + 1.12744 w_k1_MSFT_s000
                                 + 1.06962 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.9966 w_k1_ADM_s001
                                 + 1.06449 w_k1_AXP_s001
                                 + 0.951661 w_k1_DIS_s001
                                 + 0.956468 w_k1_FDX_s001
                                 + 1.30634 w_k1_IBM_s001
                                 + 1.12928 w_k1_INTC_s001
                                 + 1.12986 w_k1_KO_s001 + 1.44933 w_k1_MAT_s001
                                 + 1.10984 w_k1_MSFT_s001
                                 + 0.953879 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.896563 w_k1_ADM_s001
                                 + 1.11143 w_k1_AXP_s001
                                 + 0.996401 w_k1_DIS_s001
                                 + 1.0565 w_k1_FDX_s001 + 0.86704 w_k1_IBM_s001
                                 + 1.01558 w_k1_INTC_s001
                                 + 1.02326 w_k1_KO_s001 + 0.78122 w_k1_MAT_s001
                                 + 0.955758 w_k1_MSFT_s001
                                 + 1.00909 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.686585 w_k1_ADM_s001
                                 + 0.76993 w_k1_AXP_s001
                                 + 0.694988 w_k1_DIS_s001
                                 + 0.695934 w_k1_FDX_s001
                                 + 1.09226 w_k1_IBM_s001
                                 + 0.823603 w_k1_INTC_s001
                                 + 0.889289 w_k1_KO_s001 + 1.2404 w_k1_MAT_s001
                                 + 0.76272 w_k1_MSFT_s001 + 0.8973 w_k1_PG_s001
                                 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.29354 w_k1_ADM_s001
                                 + 1.25622 w_k1_AXP_s001
                                 + 1.58506 w_k1_DIS_s001
                                 + 1.31351 w_k1_FDX_s001
                                 + 1.20273 w_k1_IBM_s001
                                 + 1.30991 w_k1_INTC_s001
                                 + 1.31792 w_k1_KO_s001 + 1.61595 w_k1_MAT_s001
                                 + 1.29953 w_k1_MSFT_s001
                                 + 1.33821 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.05856 w_k1_ADM_s001
                                 + 1.17079 w_k1_AXP_s001
                                 + 1.04967 w_k1_DIS_s001
                                 + 1.27573 w_k1_FDX_s001
                                 + 0.89983 w_k1_IBM_s001
                                 + 1.14627 w_k1_INTC_s001
                                 + 0.969224 w_k1_KO_s001 + 1.1823 w_k1_MAT_s001
                                 + 1.16945 w_k1_MSFT_s001
                                 + 1.01743 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.825775 w_k1_ADM_s001
                                 + 0.899091 w_k1_AXP_s001
                                 + 1.13288 w_k1_DIS_s001
                                 + 0.971996 w_k1_FDX_s001
                                 + 1.32507 w_k1_IBM_s001
                                 + 1.08302 w_k1_INTC_s001
                                 + 1.18126 w_k1_KO_s001 + 1.48699 w_k1_MAT_s001
                                 + 1.16461 w_k1_MSFT_s001
                                 + 1.21596 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.817005 w_k1_ADM_s001
                                 + 0.670591 w_k1_AXP_s001
                                 + 0.92552 w_k1_DIS_s001
                                 + 0.843051 w_k1_FDX_s001
                                 + 0.878934 w_k1_IBM_s001
                                 + 1.14278 w_k1_INTC_s001
                                 + 1.06287 w_k1_KO_s001
                                 + 0.516406 w_k1_MAT_s001
                                 + 0.968329 w_k1_MSFT_s001
                                 + 0.992534 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.898771 w_k1_ADM_s001
                                 + 0.852706 w_k1_AXP_s001
                                 + 0.632648 w_k1_DIS_s001
                                 + 0.875432 w_k1_FDX_s001
                                 + 0.807765 w_k1_IBM_s001
                                 + 0.917863 w_k1_INTC_s001
                                 + 0.721136 w_k1_KO_s001
                                 + 0.713137 w_k1_MAT_s001
                                 + 0.857206 w_k1_MSFT_s001
                                 + 0.815023 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.833817 w_k1_ADM_s002
                                 + 0.827697 w_k1_AXP_s002
                                 + 0.865944 w_k1_DIS_s002
                                 + 0.812846 w_k1_FDX_s002
                                 + 0.965893 w_k1_IBM_s002
                                 + 0.849735 w_k1_INTC_s002
                                 + 0.967529 w_k1_KO_s002
                                 + 0.814574 w_k1_MAT_s002
                                 + 0.903767 w_k1_MSFT_s002
                                 + 0.906135 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.969175 w_k1_ADM_s002
                                 + 1.18189 w_k1_AXP_s002 + 1.1166 w_k1_DIS_s002
                                 + 1.08145 w_k1_FDX_s002
                                 + 1.07405 w_k1_IBM_s002
                                 + 0.876412 w_k1_INTC_s002
                                 + 1.06692 w_k1_KO_s002 + 1.34372 w_k1_MAT_s002
                                 + 0.963611 w_k1_MSFT_s002
                                 + 1.08156 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.07587 w_k1_ADM_s002
                                 + 1.18512 w_k1_AXP_s002
                                 + 1.59065 w_k1_DIS_s002
                                 + 1.32935 w_k1_FDX_s002
                                 + 1.16314 w_k1_IBM_s002
                                 + 1.26412 w_k1_INTC_s002
                                 + 1.35015 w_k1_KO_s002 + 1.09698 w_k1_MAT_s002
                                 + 1.43879 w_k1_MSFT_s002
                                 + 1.13673 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.897299 w_k1_ADM_s002
                                 + 1.02679 w_k1_AXP_s002
                                 + 1.11812 w_k1_DIS_s002
                                 + 1.05006 w_k1_FDX_s002
                                 + 1.54354 w_k1_IBM_s002
                                 + 0.872598 w_k1_INTC_s002
                                 + 1.11469 w_k1_KO_s002 + 1.959 w_k1_MAT_s002
                                 + 0.963159 w_k1_MSFT_s002
                                 + 1.04811 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.689896 w_k1_ADM_s002
                                 + 0.875846 w_k1_AXP_s002
                                 + 0.760855 w_k1_DIS_s002
                                 + 0.929955 w_k1_FDX_s002
                                 + 0.558166 w_k1_IBM_s002
                                 + 1.01233 w_k1_INTC_s002
                                 + 0.705374 w_k1_KO_s002
                                 + 0.695332 w_k1_MAT_s002
                                 + 0.841794 w_k1_MSFT_s002
                                 + 0.757856 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.0951 w_k1_ADM_s002
                                 + 0.787785 w_k1_AXP_s002
                                 + 0.885735 w_k1_DIS_s002
                                 + 1.03612 w_k1_FDX_s002
                                 + 0.587288 w_k1_IBM_s002
                                 + 0.694574 w_k1_INTC_s002
                                 + 0.878005 w_k1_KO_s002
                                 + 0.715823 w_k1_MAT_s002
                                 + 0.966727 w_k1_MSFT_s002
                                 + 0.954047 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.13137 w_k1_ADM_s002
                                 + 1.02714 w_k1_AXP_s002
                                 + 1.00424 w_k1_DIS_s002
                                 + 1.06546 w_k1_FDX_s002
                                 + 0.74504 w_k1_IBM_s002
                                 + 0.974943 w_k1_INTC_s002
                                 + 1.02853 w_k1_KO_s002
                                 + 0.692696 w_k1_MAT_s002
                                 + 1.02973 w_k1_MSFT_s002 + 1.0269 w_k1_PG_s002
                                 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.708963 w_k1_ADM_s002
                                 + 0.711286 w_k1_AXP_s002
                                 + 0.596996 w_k1_DIS_s002
                                 + 0.727627 w_k1_FDX_s002
                                 + 0.663251 w_k1_IBM_s002
                                 + 0.789906 w_k1_INTC_s002
                                 + 0.667208 w_k1_KO_s002
                                 + 0.527092 w_k1_MAT_s002
                                 + 0.732941 w_k1_MSFT_s002
                                 + 0.652041 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.09506 w_k1_ADM_s003
                                 + 0.984068 w_k1_AXP_s003
                                 + 0.876835 w_k1_DIS_s003
                                 + 0.864498 w_k1_FDX_s003
                                 + 1.32654 w_k1_IBM_s003
                                 + 0.888606 w_k1_INTC_s003
                                 + 1.17735 w_k1_KO_s003 + 1.74047 w_k1_MAT_s003
                                 + 0.712653 w_k1_MSFT_s003
                                 + 0.948095 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.15308 w_k1_ADM_s003
                                 + 1.17057 w_k1_AXP_s003
                                 + 1.01355 w_k1_DIS_s003
                                 + 0.990941 w_k1_FDX_s003
                                 + 0.894427 w_k1_IBM_s003
                                 + 0.964972 w_k1_INTC_s003
                                 + 0.89481 w_k1_KO_s003
                                 + 0.972642 w_k1_MAT_s003
                                 + 0.956455 w_k1_MSFT_s003
                                 + 0.987047 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.14524 w_k1_ADM_s003
                                 + 0.99325 w_k1_AXP_s003 + 1.269 w_k1_DIS_s003
                                 + 1.03618 w_k1_FDX_s003 + 1.0344 w_k1_IBM_s003
                                 + 1.02697 w_k1_INTC_s003
                                 + 1.07649 w_k1_KO_s003 + 1.30515 w_k1_MAT_s003
                                 + 1.03846 w_k1_MSFT_s003
                                 + 1.04704 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.57869 w_k1_ADM_s003
                                 + 1.53993 w_k1_AXP_s003
                                 + 1.59637 w_k1_DIS_s003
                                 + 1.36968 w_k1_FDX_s003
                                 + 1.35971 w_k1_IBM_s003
                                 + 1.15696 w_k1_INTC_s003
                                 + 1.42197 w_k1_KO_s003 + 1.43687 w_k1_MAT_s003
                                 + 1.12696 w_k1_MSFT_s003
                                 + 1.35668 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.11345 w_k1_ADM_s003
                                 + 1.05016 w_k1_AXP_s003
                                 + 0.769022 w_k1_DIS_s003
                                 + 0.930403 w_k1_FDX_s003
                                 + 0.571768 w_k1_IBM_s003
                                 + 0.945882 w_k1_INTC_s003
                                 + 0.804475 w_k1_KO_s003
                                 + 1.13339 w_k1_MAT_s003
                                 + 0.87039 w_k1_MSFT_s003
                                 + 0.864588 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.36091 w_k1_ADM_s003
                                 + 1.28658 w_k1_AXP_s003
                                 + 1.16127 w_k1_DIS_s003
                                 + 1.16511 w_k1_FDX_s003
                                 + 1.00156 w_k1_IBM_s003
                                 + 1.04879 w_k1_INTC_s003
                                 + 1.04198 w_k1_KO_s003 + 1.51017 w_k1_MAT_s003
                                 + 1.08153 w_k1_MSFT_s003
                                 + 1.04162 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.770382 w_k1_ADM_s003
                                 + 0.776175 w_k1_AXP_s003
                                 + 0.803202 w_k1_DIS_s003
                                 + 0.80293 w_k1_FDX_s003
                                 + 0.716374 w_k1_IBM_s003
                                 + 0.904044 w_k1_INTC_s003
                                 + 0.834792 w_k1_KO_s003
                                 + 0.717562 w_k1_MAT_s003
                                 + 0.791143 w_k1_MSFT_s003
                                 + 0.81117 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.989871 w_k1_ADM_s003
                                 + 1.0233 w_k1_AXP_s003 + 1.15953 w_k1_DIS_s003
                                 + 1.15782 w_k1_FDX_s003
                                 + 0.596075 w_k1_IBM_s003
                                 + 1.06584 w_k1_INTC_s003
                                 + 0.87021 w_k1_KO_s003
                                 + 0.666091 w_k1_MAT_s003
                                 + 1.03283 w_k1_MSFT_s003
                                 + 1.02007 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.21658 w_k1_ADM_s004
                                 + 1.25737 w_k1_AXP_s004
                                 + 0.792944 w_k1_DIS_s004
                                 + 0.96491 w_k1_FDX_s004
                                 + 0.944574 w_k1_IBM_s004
                                 + 0.94615 w_k1_INTC_s004
                                 + 0.92795 w_k1_KO_s004 + 1.0655 w_k1_MAT_s004
                                 + 0.95211 w_k1_MSFT_s004
                                 + 0.982631 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.08744 w_k1_ADM_s004
                                 + 1.13288 w_k1_AXP_s004
                                 + 1.08866 w_k1_DIS_s004
                                 + 0.95617 w_k1_FDX_s004
                                 + 1.18345 w_k1_IBM_s004
                                 + 0.981192 w_k1_INTC_s004
                                 + 1.07257 w_k1_KO_s004
                                 + 0.940617 w_k1_MAT_s004
                                 + 0.927201 w_k1_MSFT_s004
                                 + 1.02139 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.913473 w_k1_ADM_s004
                                 + 0.895859 w_k1_AXP_s004
                                 + 1.20217 w_k1_DIS_s004
                                 + 0.908507 w_k1_FDX_s004
                                 + 1.2512 w_k1_IBM_s004
                                 + 0.968046 w_k1_INTC_s004
                                 + 1.15976 w_k1_KO_s004
                                 + 0.975688 w_k1_MAT_s004
                                 + 1.0099 w_k1_MSFT_s004 + 1.17889 w_k1_PG_s004
                                 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.16782 w_k1_ADM_s004
                                 + 0.876523 w_k1_AXP_s004
                                 + 1.3593 w_k1_DIS_s004 + 1.13443 w_k1_FDX_s004
                                 + 0.749834 w_k1_IBM_s004
                                 + 0.889233 w_k1_INTC_s004
                                 + 1.11206 w_k1_KO_s004
                                 + 0.723602 w_k1_MAT_s004
                                 + 1.15451 w_k1_MSFT_s004
                                 + 1.11979 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.974645 w_k1_ADM_s004
                                 + 1.04562 w_k1_AXP_s004
                                 + 1.05026 w_k1_DIS_s004
                                 + 1.10281 w_k1_FDX_s004
                                 + 0.908625 w_k1_IBM_s004
                                 + 0.978147 w_k1_INTC_s004
                                 + 0.97606 w_k1_KO_s004
                                 + 0.993887 w_k1_MAT_s004
                                 + 1.06376 w_k1_MSFT_s004
                                 + 0.89937 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.4815 w_k1_ADM_s004
                                 + 1.36143 w_k1_AXP_s004
                                 + 1.65733 w_k1_DIS_s004
                                 + 1.37529 w_k1_FDX_s004
                                 + 1.23601 w_k1_IBM_s004
                                 + 0.979645 w_k1_INTC_s004
                                 + 1.23431 w_k1_KO_s004 + 1.05007 w_k1_MAT_s004
                                 + 1.10974 w_k1_MSFT_s004
                                 + 1.25562 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.865045 w_k1_ADM_s004
                                 + 0.843098 w_k1_AXP_s004
                                 + 0.823484 w_k1_DIS_s004
                                 + 1.02597 w_k1_FDX_s004
                                 + 0.61875 w_k1_IBM_s004
                                 + 0.962965 w_k1_INTC_s004
                                 + 0.760902 w_k1_KO_s004
                                 + 0.576357 w_k1_MAT_s004
                                 + 0.867999 w_k1_MSFT_s004
                                 + 0.989252 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.898898 w_k1_ADM_s004
                                 + 0.683135 w_k1_AXP_s004
                                 + 0.774466 w_k1_DIS_s004
                                 + 0.768417 w_k1_FDX_s004
                                 + 0.8377 w_k1_IBM_s004
                                 + 0.829852 w_k1_INTC_s004
                                 + 0.847146 w_k1_KO_s004
                                 + 0.809599 w_k1_MAT_s004
                                 + 0.802587 w_k1_MSFT_s004
                                 + 0.854326 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.778298 w_k1_ADM_s005
                                 + 0.79924 w_k1_AXP_s005
                                 + 0.645399 w_k1_DIS_s005
                                 + 0.737571 w_k1_FDX_s005
                                 + 1.31328 w_k1_IBM_s005
                                 + 0.74285 w_k1_INTC_s005
                                 + 0.948772 w_k1_KO_s005
                                 + 1.26951 w_k1_MAT_s005
                                 + 0.725641 w_k1_MSFT_s005
                                 + 0.893928 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.765712 w_k1_ADM_s005
                                 + 0.786328 w_k1_AXP_s005
                                 + 0.602402 w_k1_DIS_s005
                                 + 0.797446 w_k1_FDX_s005
                                 + 0.769727 w_k1_IBM_s005
                                 + 0.847971 w_k1_INTC_s005
                                 + 0.711921 w_k1_KO_s005
                                 + 0.731929 w_k1_MAT_s005
                                 + 0.751584 w_k1_MSFT_s005
                                 + 0.890296 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.2605 w_k1_ADM_s005
                                 + 1.14166 w_k1_AXP_s005
                                 + 1.15834 w_k1_DIS_s005
                                 + 1.27091 w_k1_FDX_s005
                                 + 0.812945 w_k1_IBM_s005
                                 + 1.19541 w_k1_INTC_s005
                                 + 1.08974 w_k1_KO_s005
                                 + 0.928452 w_k1_MAT_s005
                                 + 1.26007 w_k1_MSFT_s005
                                 + 1.06726 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.984624 w_k1_ADM_s005
                                 + 1.46593 w_k1_AXP_s005
                                 + 1.09763 w_k1_DIS_s005
                                 + 1.16918 w_k1_FDX_s005
                                 + 1.02855 w_k1_IBM_s005
                                 + 1.1071 w_k1_INTC_s005 + 1.13194 w_k1_KO_s005
                                 + 1.49079 w_k1_MAT_s005
                                 + 1.09654 w_k1_MSFT_s005
                                 + 1.03444 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.846382 w_k1_ADM_s005
                                 + 1.15363 w_k1_AXP_s005
                                 + 0.82919 w_k1_DIS_s005
                                 + 0.961788 w_k1_FDX_s005
                                 + 1.06319 w_k1_IBM_s005
                                 + 0.919148 w_k1_INTC_s005
                                 + 0.807048 w_k1_KO_s005 + 1.3638 w_k1_MAT_s005
                                 + 0.954999 w_k1_MSFT_s005
                                 + 0.96217 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.738376 w_k1_ADM_s005
                                 + 0.668637 w_k1_AXP_s005
                                 + 0.533205 w_k1_DIS_s005
                                 + 0.671635 w_k1_FDX_s005
                                 + 0.692304 w_k1_IBM_s005
                                 + 0.720229 w_k1_INTC_s005
                                 + 0.654494 w_k1_KO_s005
                                 + 0.719393 w_k1_MAT_s005
                                 + 0.724921 w_k1_MSFT_s005
                                 + 0.714299 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.743879 w_k1_ADM_s005
                                 + 0.760877 w_k1_AXP_s005
                                 + 0.822304 w_k1_DIS_s005
                                 + 0.82637 w_k1_FDX_s005
                                 + 0.900688 w_k1_IBM_s005
                                 + 0.836096 w_k1_INTC_s005
                                 + 0.833258 w_k1_KO_s005 + 0.9475 w_k1_MAT_s005
                                 + 0.903641 w_k1_MSFT_s005
                                 + 0.792981 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.987076 w_k1_ADM_s005
                                 + 1.39917 w_k1_AXP_s005
                                 + 1.53701 w_k1_DIS_s005
                                 + 1.18587 w_k1_FDX_s005
                                 + 1.47142 w_k1_IBM_s005
                                 + 1.39265 w_k1_INTC_s005
                                 + 1.41462 w_k1_KO_s005 + 1.63739 w_k1_MAT_s005
                                 + 1.29761 w_k1_MSFT_s005
                                 + 1.14242 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.26316 w_k1_ADM_s006
                                 + 1.75104 w_k1_AXP_s006
                                 + 1.61689 w_k1_DIS_s006
                                 + 1.32664 w_k1_FDX_s006
                                 + 1.20398 w_k1_IBM_s006
                                 + 1.28827 w_k1_INTC_s006
                                 + 1.21386 w_k1_KO_s006 + 1.39315 w_k1_MAT_s006
                                 + 1.19746 w_k1_MSFT_s006
                                 + 1.15779 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.2062 w_k1_ADM_s006
                                 + 1.3408 w_k1_AXP_s006 + 1.45196 w_k1_DIS_s006
                                 + 1.2754 w_k1_FDX_s006
                                 + 0.969418 w_k1_IBM_s006
                                 + 1.33103 w_k1_INTC_s006
                                 + 1.00764 w_k1_KO_s006
                                 + 0.879108 w_k1_MAT_s006
                                 + 1.18992 w_k1_MSFT_s006
                                 + 0.996497 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.932547 w_k1_ADM_s006
                                 + 1.16959 w_k1_AXP_s006
                                 + 0.903502 w_k1_DIS_s006
                                 + 0.942326 w_k1_FDX_s006
                                 + 0.993562 w_k1_IBM_s006
                                 + 0.928786 w_k1_INTC_s006
                                 + 0.8734 w_k1_KO_s006 + 1.10841 w_k1_MAT_s006
                                 + 0.841236 w_k1_MSFT_s006
                                 + 0.90173 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.08317 w_k1_ADM_s006
                                 + 1.28824 w_k1_AXP_s006
                                 + 1.29113 w_k1_DIS_s006 + 1.0355 w_k1_FDX_s006
                                 + 1.67318 w_k1_IBM_s006
                                 + 1.03402 w_k1_INTC_s006
                                 + 1.20483 w_k1_KO_s006 + 1.67692 w_k1_MAT_s006
                                 + 0.82645 w_k1_MSFT_s006
                                 + 0.992531 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.920936 w_k1_ADM_s006
                                 + 0.848019 w_k1_AXP_s006
                                 + 0.849276 w_k1_DIS_s006
                                 + 0.819935 w_k1_FDX_s006
                                 + 1.23938 w_k1_IBM_s006
                                 + 0.882543 w_k1_INTC_s006
                                 + 1.10698 w_k1_KO_s006 + 1.11018 w_k1_MAT_s006
                                 + 0.916827 w_k1_MSFT_s006
                                 + 0.970529 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.770148 w_k1_ADM_s006
                                 + 0.736269 w_k1_AXP_s006
                                 + 0.692852 w_k1_DIS_s006
                                 + 0.728898 w_k1_FDX_s006
                                 + 0.77789 w_k1_IBM_s006
                                 + 0.826545 w_k1_INTC_s006
                                 + 0.664836 w_k1_KO_s006
                                 + 0.673692 w_k1_MAT_s006
                                 + 0.808296 w_k1_MSFT_s006
                                 + 0.757198 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.72513 w_k1_ADM_s006
                                 + 1.9165 w_k1_AXP_s006 + 2.10821 w_k1_DIS_s006
                                 + 1.49404 w_k1_FDX_s006 + 1.4206 w_k1_IBM_s006
                                 + 1.46786 w_k1_INTC_s006
                                 + 1.54556 w_k1_KO_s006 + 1.64715 w_k1_MAT_s006
                                 + 1.35951 w_k1_MSFT_s006
                                 + 1.31488 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.06334 w_k1_ADM_s006
                                 + 1.13637 w_k1_AXP_s006
                                 + 1.03664 w_k1_DIS_s006
                                 + 1.04912 w_k1_FDX_s006
                                 + 0.84793 w_k1_IBM_s006
                                 + 1.12401 w_k1_INTC_s006
                                 + 0.80062 w_k1_KO_s006
                                 + 0.793472 w_k1_MAT_s006
                                 + 1.0087 w_k1_MSFT_s006
                                 + 0.923957 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.15932 w_k1_ADM_s007
                                 + 1.03654 w_k1_AXP_s007
                                 + 1.05101 w_k1_DIS_s007
                                 + 1.22319 w_k1_FDX_s007
                                 + 0.690029 w_k1_IBM_s007
                                 + 1.0015 w_k1_INTC_s007
                                 + 0.900847 w_k1_KO_s007
                                 + 1.06554 w_k1_MAT_s007
                                 + 1.16442 w_k1_MSFT_s007
                                 + 0.963977 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.34764 w_k1_ADM_s007
                                 + 1.46244 w_k1_AXP_s007
                                 + 1.34253 w_k1_DIS_s007
                                 + 1.43536 w_k1_FDX_s007
                                 + 0.881827 w_k1_IBM_s007
                                 + 1.34213 w_k1_INTC_s007
                                 + 1.10606 w_k1_KO_s007 + 1.14926 w_k1_MAT_s007
                                 + 1.38936 w_k1_MSFT_s007
                                 + 1.13856 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.43794 w_k1_ADM_s007
                                 + 1.54376 w_k1_AXP_s007
                                 + 1.50335 w_k1_DIS_s007
                                 + 1.42369 w_k1_FDX_s007
                                 + 1.57896 w_k1_IBM_s007
                                 + 1.04786 w_k1_INTC_s007
                                 + 1.51336 w_k1_KO_s007 + 1.67637 w_k1_MAT_s007
                                 + 1.22079 w_k1_MSFT_s007
                                 + 1.15855 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.935231 w_k1_ADM_s007
                                 + 0.978647 w_k1_AXP_s007
                                 + 0.711653 w_k1_DIS_s007
                                 + 0.868836 w_k1_FDX_s007
                                 + 1.23168 w_k1_IBM_s007
                                 + 0.943795 w_k1_INTC_s007
                                 + 1.00314 w_k1_KO_s007 + 1.33054 w_k1_MAT_s007
                                 + 0.935384 w_k1_MSFT_s007
                                 + 0.956774 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.968837 w_k1_ADM_s007
                                 + 1.22731 w_k1_AXP_s007
                                 + 0.985344 w_k1_DIS_s007
                                 + 1.00873 w_k1_FDX_s007
                                 + 1.65329 w_k1_IBM_s007
                                 + 1.02351 w_k1_INTC_s007
                                 + 1.30965 w_k1_KO_s007 + 1.7167 w_k1_MAT_s007
                                 + 1.12605 w_k1_MSFT_s007 + 1.1136 w_k1_PG_s007
                                 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.80828 w_k1_ADM_s007
                                 + 0.664386 w_k1_AXP_s007
                                 + 0.679891 w_k1_DIS_s007
                                 + 0.739056 w_k1_FDX_s007
                                 + 0.612683 w_k1_IBM_s007
                                 + 0.676353 w_k1_INTC_s007
                                 + 0.710615 w_k1_KO_s007
                                 + 0.785671 w_k1_MAT_s007
                                 + 0.751799 w_k1_MSFT_s007
                                 + 0.76634 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.895838 w_k1_ADM_s007
                                 + 0.814378 w_k1_AXP_s007
                                 + 0.773411 w_k1_DIS_s007
                                 + 0.970266 w_k1_FDX_s007
                                 + 0.851675 w_k1_IBM_s007
                                 + 0.954885 w_k1_INTC_s007
                                 + 0.89305 w_k1_KO_s007
                                 + 0.891318 w_k1_MAT_s007
                                 + 0.973613 w_k1_MSFT_s007
                                 + 0.940523 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.986185 w_k1_ADM_s007
                                 + 1.3527 w_k1_AXP_s007 + 1.20508 w_k1_DIS_s007
                                 + 1.18556 w_k1_FDX_s007
                                 + 1.15902 w_k1_IBM_s007
                                 + 0.962076 w_k1_INTC_s007 + 1.193 w_k1_KO_s007
                                 + 1.98038 w_k1_MAT_s007
                                 + 1.22775 w_k1_MSFT_s007 + 1.0564 w_k1_PG_s007
                                 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.37577 w_k1_ADM_s010
                                 + 1.31319 w_k1_AXP_s010
                                 + 1.34358 w_k1_DIS_s010
                                 + 1.27184 w_k1_FDX_s010
                                 + 0.981736 w_k1_IBM_s010
                                 + 1.32408 w_k1_INTC_s010
                                 + 1.19795 w_k1_KO_s010 + 1.37602 w_k1_MAT_s010
                                 + 1.3961 w_k1_MSFT_s010 + 1.17233 w_k1_PG_s010
                                 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.84987 w_k1_ADM_s010
                                 + 0.732507 w_k1_AXP_s010
                                 + 0.690286 w_k1_DIS_s010
                                 + 0.93253 w_k1_FDX_s010
                                 + 0.694779 w_k1_IBM_s010
                                 + 1.02826 w_k1_INTC_s010
                                 + 0.905127 w_k1_KO_s010
                                 + 0.795616 w_k1_MAT_s010
                                 + 1.09635 w_k1_MSFT_s010
                                 + 0.930601 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.37135 w_k1_ADM_s010
                                 + 0.976038 w_k1_AXP_s010
                                 + 1.23901 w_k1_DIS_s010
                                 + 1.20118 w_k1_FDX_s010
                                 + 0.865703 w_k1_IBM_s010
                                 + 1.33174 w_k1_INTC_s010
                                 + 1.15205 w_k1_KO_s010
                                 + 0.931545 w_k1_MAT_s010
                                 + 1.2696 w_k1_MSFT_s010 + 1.16281 w_k1_PG_s010
                                 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.05514 w_k1_ADM_s010
                                 + 1.40914 w_k1_AXP_s010
                                 + 1.21454 w_k1_DIS_s010
                                 + 1.33785 w_k1_FDX_s010
                                 + 1.35239 w_k1_IBM_s010
                                 + 1.24462 w_k1_INTC_s010
                                 + 1.20289 w_k1_KO_s010 + 1.68153 w_k1_MAT_s010
                                 + 1.23933 w_k1_MSFT_s010
                                 + 1.13196 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.949867 w_k1_ADM_s010
                                 + 0.963924 w_k1_AXP_s010
                                 + 0.846639 w_k1_DIS_s010
                                 + 0.936538 w_k1_FDX_s010
                                 + 1.1442 w_k1_IBM_s010
                                 + 1.09405 w_k1_INTC_s010
                                 + 1.05419 w_k1_KO_s010 + 1.26127 w_k1_MAT_s010
                                 + 0.933518 w_k1_MSFT_s010
                                 + 0.999078 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.22346 w_k1_ADM_s010
                                 + 0.935508 w_k1_AXP_s010
                                 + 1.03543 w_k1_DIS_s010
                                 + 0.997372 w_k1_FDX_s010
                                 + 1.45579 w_k1_IBM_s010
                                 + 0.997763 w_k1_INTC_s010
                                 + 1.11774 w_k1_KO_s010 + 1.26471 w_k1_MAT_s010
                                 + 0.965384 w_k1_MSFT_s010
                                 + 1.03432 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.61659 w_k1_ADM_s010
                                 + 1.08085 w_k1_AXP_s010
                                 + 1.96248 w_k1_DIS_s010
                                 + 1.50091 w_k1_FDX_s010
                                 + 1.36264 w_k1_IBM_s010
                                 + 1.38816 w_k1_INTC_s010
                                 + 1.72282 w_k1_KO_s010 + 1.1143 w_k1_MAT_s010
                                 + 1.91255 w_k1_MSFT_s010
                                 + 1.25916 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.816394 w_k1_ADM_s010
                                 + 0.67535 w_k1_AXP_s010
                                 + 0.604392 w_k1_DIS_s010
                                 + 0.722058 w_k1_FDX_s010
                                 + 0.946283 w_k1_IBM_s010
                                 + 0.813961 w_k1_INTC_s010
                                 + 0.733861 w_k1_KO_s010
                                 + 0.816646 w_k1_MAT_s010
                                 + 0.845375 w_k1_MSFT_s010
                                 + 0.768134 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.16092 w_k1_ADM_s011
                                 + 0.813861 w_k1_AXP_s011
                                 + 0.870597 w_k1_DIS_s011
                                 + 0.968082 w_k1_FDX_s011
                                 + 0.72139 w_k1_IBM_s011
                                 + 0.867606 w_k1_INTC_s011
                                 + 0.861495 w_k1_KO_s011
                                 + 0.651359 w_k1_MAT_s011
                                 + 0.895576 w_k1_MSFT_s011
                                 + 0.953731 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.742191 w_k1_ADM_s011
                                 + 0.953005 w_k1_AXP_s011
                                 + 0.725543 w_k1_DIS_s011
                                 + 0.873911 w_k1_FDX_s011
                                 + 0.867125 w_k1_IBM_s011
                                 + 0.990337 w_k1_INTC_s011
                                 + 0.821386 w_k1_KO_s011
                                 + 0.889834 w_k1_MAT_s011
                                 + 0.869275 w_k1_MSFT_s011
                                 + 0.847945 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.35555 w_k1_ADM_s011
                                 + 1.16209 w_k1_AXP_s011
                                 + 1.38578 w_k1_DIS_s011
                                 + 1.36891 w_k1_FDX_s011
                                 + 0.917256 w_k1_IBM_s011
                                 + 1.2672 w_k1_INTC_s011 + 1.04188 w_k1_KO_s011
                                 + 0.839055 w_k1_MAT_s011
                                 + 1.30135 w_k1_MSFT_s011
                                 + 1.08225 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.751641 w_k1_ADM_s011
                                 + 0.839668 w_k1_AXP_s011
                                 + 0.885302 w_k1_DIS_s011
                                 + 0.922659 w_k1_FDX_s011
                                 + 1.13629 w_k1_IBM_s011
                                 + 1.07405 w_k1_INTC_s011
                                 + 0.95699 w_k1_KO_s011 + 1.11908 w_k1_MAT_s011
                                 + 0.993646 w_k1_MSFT_s011
                                 + 0.980351 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.18642 w_k1_ADM_s011
                                 + 1.30708 w_k1_AXP_s011
                                 + 1.07767 w_k1_DIS_s011 + 1.1462 w_k1_FDX_s011
                                 + 1.12016 w_k1_IBM_s011
                                 + 1.1854 w_k1_INTC_s011
                                 + 0.974394 w_k1_KO_s011
                                 + 1.11498 w_k1_MAT_s011
                                 + 1.1076 w_k1_MSFT_s011 + 1.14154 w_k1_PG_s011
                                 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.0938 w_k1_ADM_s011
                                 + 1.34111 w_k1_AXP_s011
                                 + 1.55872 w_k1_DIS_s011
                                 + 1.36183 w_k1_FDX_s011
                                 + 1.39224 w_k1_IBM_s011
                                 + 1.3294 w_k1_INTC_s011 + 1.2001 w_k1_KO_s011
                                 + 1.1559 w_k1_MAT_s011
                                 + 1.15815 w_k1_MSFT_s011
                                 + 1.20372 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.795614 w_k1_ADM_s011
                                 + 0.677226 w_k1_AXP_s011
                                 + 0.699458 w_k1_DIS_s011
                                 + 0.764938 w_k1_FDX_s011
                                 + 0.901071 w_k1_IBM_s011
                                 + 0.859971 w_k1_INTC_s011
                                 + 0.782542 w_k1_KO_s011
                                 + 0.938984 w_k1_MAT_s011
                                 + 0.804455 w_k1_MSFT_s011
                                 + 0.893107 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.890313 w_k1_ADM_s011
                                 + 0.980811 w_k1_AXP_s011
                                 + 1.04082 w_k1_DIS_s011
                                 + 0.913459 w_k1_FDX_s011
                                 + 1.59954 w_k1_IBM_s011
                                 + 1.11161 w_k1_INTC_s011
                                 + 1.30505 w_k1_KO_s011 + 1.63337 w_k1_MAT_s011
                                 + 0.993126 w_k1_MSFT_s011
                                 + 0.997602 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.28535 w_k1_ADM_s012
                                 + 1.06841 w_k1_AXP_s012
                                 + 1.06336 w_k1_DIS_s012
                                 + 1.08816 w_k1_FDX_s012
                                 + 0.745177 w_k1_IBM_s012
                                 + 1.10566 w_k1_INTC_s012
                                 + 0.750703 w_k1_KO_s012
                                 + 0.755181 w_k1_MAT_s012
                                 + 1.01729 w_k1_MSFT_s012
                                 + 0.814373 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.29014 w_k1_ADM_s012
                                 + 1.60764 w_k1_AXP_s012
                                 + 1.45176 w_k1_DIS_s012
                                 + 1.34056 w_k1_FDX_s012 + 1.0837 w_k1_IBM_s012
                                 + 1.34986 w_k1_INTC_s012
                                 + 1.17846 w_k1_KO_s012 + 0.93809 w_k1_MAT_s012
                                 + 1.29415 w_k1_MSFT_s012
                                 + 1.14872 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.984065 w_k1_ADM_s012
                                 + 0.94159 w_k1_AXP_s012
                                 + 0.995072 w_k1_DIS_s012
                                 + 1.04042 w_k1_FDX_s012 + 1.1233 w_k1_IBM_s012
                                 + 0.967564 w_k1_INTC_s012
                                 + 0.903671 w_k1_KO_s012
                                 + 0.975758 w_k1_MAT_s012
                                 + 1.16963 w_k1_MSFT_s012
                                 + 0.88387 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.06532 w_k1_ADM_s012
                                 + 1.24479 w_k1_AXP_s012
                                 + 1.25481 w_k1_DIS_s012
                                 + 1.14506 w_k1_FDX_s012
                                 + 0.947625 w_k1_IBM_s012
                                 + 1.17599 w_k1_INTC_s012
                                 + 1.06736 w_k1_KO_s012
                                 + 0.896124 w_k1_MAT_s012
                                 + 1.06814 w_k1_MSFT_s012
                                 + 1.04178 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.35583 w_k1_ADM_s012
                                 + 1.42195 w_k1_AXP_s012
                                 + 1.43106 w_k1_DIS_s012
                                 + 1.27527 w_k1_FDX_s012
                                 + 1.30206 w_k1_IBM_s012
                                 + 1.1876 w_k1_INTC_s012 + 1.17111 w_k1_KO_s012
                                 + 1.21982 w_k1_MAT_s012
                                 + 1.13922 w_k1_MSFT_s012 + 1.1039 w_k1_PG_s012
                                 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.11305 w_k1_ADM_s012
                                 + 0.883693 w_k1_AXP_s012
                                 + 0.844758 w_k1_DIS_s012
                                 + 0.876887 w_k1_FDX_s012
                                 + 1.14073 w_k1_IBM_s012
                                 + 0.816588 w_k1_INTC_s012
                                 + 0.818721 w_k1_KO_s012
                                 + 0.609602 w_k1_MAT_s012
                                 + 0.780667 w_k1_MSFT_s012
                                 + 0.837167 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.823561 w_k1_ADM_s012
                                 + 0.669911 w_k1_AXP_s012
                                 + 0.532933 w_k1_DIS_s012
                                 + 0.6773 w_k1_FDX_s012
                                 + 0.768654 w_k1_IBM_s012
                                 + 0.632516 w_k1_INTC_s012
                                 + 0.613317 w_k1_KO_s012
                                 + 0.548343 w_k1_MAT_s012
                                 + 0.639971 w_k1_MSFT_s012
                                 + 0.671685 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.20451 w_k1_ADM_s012
                                 + 1.43791 w_k1_AXP_s012
                                 + 0.90754 w_k1_DIS_s012
                                 + 1.01715 w_k1_FDX_s012
                                 + 1.06969 w_k1_IBM_s012
                                 + 0.99283 w_k1_INTC_s012
                                 + 0.779968 w_k1_KO_s012
                                 + 0.915565 w_k1_MAT_s012
                                 + 0.859317 w_k1_MSFT_s012
                                 + 0.912775 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.07187 w_k1_ADM_s013
                                 + 1.38134 w_k1_AXP_s013
                                 + 0.999715 w_k1_DIS_s013
                                 + 0.999618 w_k1_FDX_s013
                                 + 1.27691 w_k1_IBM_s013
                                 + 1.03868 w_k1_INTC_s013
                                 + 1.22356 w_k1_KO_s013 + 1.7782 w_k1_MAT_s013
                                 + 0.951898 w_k1_MSFT_s013
                                 + 1.12129 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.01677 w_k1_ADM_s013
                                 + 0.884698 w_k1_AXP_s013
                                 + 1.08693 w_k1_DIS_s013
                                 + 1.01006 w_k1_FDX_s013
                                 + 1.21008 w_k1_IBM_s013
                                 + 1.25101 w_k1_INTC_s013
                                 + 1.32856 w_k1_KO_s013 + 1.027 w_k1_MAT_s013
                                 + 1.13352 w_k1_MSFT_s013
                                 + 0.990317 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.01223 w_k1_ADM_s013
                                 + 0.95958 w_k1_AXP_s013
                                 + 1.22706 w_k1_DIS_s013
                                 + 1.28815 w_k1_FDX_s013
                                 + 0.771479 w_k1_IBM_s013
                                 + 1.16521 w_k1_INTC_s013
                                 + 1.08774 w_k1_KO_s013
                                 + 0.938232 w_k1_MAT_s013
                                 + 1.12818 w_k1_MSFT_s013 + 1.0253 w_k1_PG_s013
                                 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.992327 w_k1_ADM_s013
                                 + 0.790893 w_k1_AXP_s013
                                 + 0.934765 w_k1_DIS_s013
                                 + 1.05295 w_k1_FDX_s013
                                 + 0.613435 w_k1_IBM_s013
                                 + 1.25694 w_k1_INTC_s013
                                 + 0.790038 w_k1_KO_s013
                                 + 0.34893 w_k1_MAT_s013
                                 + 0.897602 w_k1_MSFT_s013
                                 + 0.841662 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.07241 w_k1_ADM_s013
                                 + 1.50967 w_k1_AXP_s013
                                 + 1.43706 w_k1_DIS_s013
                                 + 1.31633 w_k1_FDX_s013
                                 + 1.35785 w_k1_IBM_s013
                                 + 1.44813 w_k1_INTC_s013
                                 + 1.41404 w_k1_KO_s013 + 1.49591 w_k1_MAT_s013
                                 + 1.26166 w_k1_MSFT_s013
                                 + 1.10329 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.08573 w_k1_ADM_s013
                                 + 1.1573 w_k1_AXP_s013 + 1.61084 w_k1_DIS_s013
                                 + 1.2891 w_k1_FDX_s013 + 1.17612 w_k1_IBM_s013
                                 + 1.45055 w_k1_INTC_s013
                                 + 1.37675 w_k1_KO_s013
                                 + 0.860127 w_k1_MAT_s013
                                 + 1.30299 w_k1_MSFT_s013
                                 + 0.988215 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.04804 w_k1_ADM_s013
                                 + 1.20119 w_k1_AXP_s013
                                 + 1.03414 w_k1_DIS_s013
                                 + 1.04742 w_k1_FDX_s013
                                 + 0.94481 w_k1_IBM_s013
                                 + 0.973306 w_k1_INTC_s013
                                 + 0.965845 w_k1_KO_s013
                                 + 0.83518 w_k1_MAT_s013
                                 + 0.880161 w_k1_MSFT_s013
                                 + 0.824107 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.736791 w_k1_ADM_s013
                                 + 0.773713 w_k1_AXP_s013
                                 + 0.709338 w_k1_DIS_s013
                                 + 0.841678 w_k1_FDX_s013
                                 + 0.75844 w_k1_IBM_s013
                                 + 0.804407 w_k1_INTC_s013
                                 + 0.756074 w_k1_KO_s013
                                 + 0.853764 w_k1_MAT_s013
                                 + 0.743414 w_k1_MSFT_s013
                                 + 0.775847 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.0234 w_k1_ADM_s014
                                 + 0.760782 w_k1_AXP_s014 + 0.67 w_k1_DIS_s014
                                 + 0.768259 w_k1_FDX_s014
                                 + 0.747878 w_k1_IBM_s014
                                 + 0.715575 w_k1_INTC_s014
                                 + 0.677116 w_k1_KO_s014
                                 + 0.475601 w_k1_MAT_s014
                                 + 0.797594 w_k1_MSFT_s014
                                 + 0.743705 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.19222 w_k1_ADM_s014
                                 + 1.61202 w_k1_AXP_s014 + 1.2862 w_k1_DIS_s014
                                 + 1.15824 w_k1_FDX_s014
                                 + 0.897485 w_k1_IBM_s014
                                 + 0.860973 w_k1_INTC_s014
                                 + 1.15275 w_k1_KO_s014 + 1.33654 w_k1_MAT_s014
                                 + 1.00361 w_k1_MSFT_s014
                                 + 1.11648 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.44332 w_k1_ADM_s014
                                 + 1.63295 w_k1_AXP_s014
                                 + 1.33813 w_k1_DIS_s014
                                 + 1.03701 w_k1_FDX_s014 + 1.8067 w_k1_IBM_s014
                                 + 0.99374 w_k1_INTC_s014
                                 + 1.59279 w_k1_KO_s014 + 1.94328 w_k1_MAT_s014
                                 + 1.0513 w_k1_MSFT_s014 + 1.37959 w_k1_PG_s014
                                 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.52157 w_k1_ADM_s014
                                 + 0.843675 w_k1_AXP_s014
                                 + 1.06535 w_k1_DIS_s014
                                 + 1.10662 w_k1_FDX_s014
                                 + 0.682443 w_k1_IBM_s014
                                 + 0.829008 w_k1_INTC_s014
                                 + 0.840338 w_k1_KO_s014
                                 + 0.423523 w_k1_MAT_s014
                                 + 0.93091 w_k1_MSFT_s014
                                 + 1.02192 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.848568 w_k1_ADM_s014
                                 + 0.896229 w_k1_AXP_s014
                                 + 0.941829 w_k1_DIS_s014
                                 + 0.770972 w_k1_FDX_s014
                                 + 1.17586 w_k1_IBM_s014
                                 + 0.708681 w_k1_INTC_s014
                                 + 1.01828 w_k1_KO_s014
                                 + 0.965048 w_k1_MAT_s014
                                 + 0.76608 w_k1_MSFT_s014
                                 + 0.964568 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.19064 w_k1_ADM_s014
                                 + 1.73724 w_k1_AXP_s014
                                 + 0.969827 w_k1_DIS_s014
                                 + 1.01144 w_k1_FDX_s014
                                 + 1.32594 w_k1_IBM_s014
                                 + 0.805142 w_k1_INTC_s014
                                 + 0.980376 w_k1_KO_s014
                                 + 1.55334 w_k1_MAT_s014
                                 + 0.830542 w_k1_MSFT_s014
                                 + 1.08287 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.20565 w_k1_ADM_s014
                                 + 1.00978 w_k1_AXP_s014
                                 + 0.83572 w_k1_DIS_s014
                                 + 0.912855 w_k1_FDX_s014
                                 + 0.893157 w_k1_IBM_s014
                                 + 0.70746 w_k1_INTC_s014
                                 + 0.940536 w_k1_KO_s014
                                 + 1.06076 w_k1_MAT_s014
                                 + 0.759242 w_k1_MSFT_s014
                                 + 0.929893 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.45284 w_k1_ADM_s014
                                 + 1.26809 w_k1_AXP_s014 + 1.3638 w_k1_DIS_s014
                                 + 1.09826 w_k1_FDX_s014
                                 + 1.10951 w_k1_IBM_s014
                                 + 0.797009 w_k1_INTC_s014
                                 + 1.19364 w_k1_KO_s014 + 1.27599 w_k1_MAT_s014
                                 + 1.02986 w_k1_MSFT_s014
                                 + 1.08289 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.02681 w_k1_ADM_s015
                                 + 0.978625 w_k1_AXP_s015
                                 + 0.966323 w_k1_DIS_s015
                                 + 1.04728 w_k1_FDX_s015
                                 + 0.73256 w_k1_IBM_s015
                                 + 0.986431 w_k1_INTC_s015
                                 + 0.898066 w_k1_KO_s015
                                 + 0.90038 w_k1_MAT_s015
                                 + 0.98959 w_k1_MSFT_s015
                                 + 1.09063 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.766102 w_k1_ADM_s015
                                 + 0.637068 w_k1_AXP_s015
                                 + 0.613914 w_k1_DIS_s015
                                 + 0.776731 w_k1_FDX_s015
                                 + 0.848831 w_k1_IBM_s015
                                 + 0.857437 w_k1_INTC_s015
                                 + 0.69607 w_k1_KO_s015
                                 + 0.574437 w_k1_MAT_s015
                                 + 0.710479 w_k1_MSFT_s015
                                 + 0.864217 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.955968 w_k1_ADM_s015
                                 + 0.894385 w_k1_AXP_s015
                                 + 0.899977 w_k1_DIS_s015
                                 + 0.958051 w_k1_FDX_s015
                                 + 0.909654 w_k1_IBM_s015
                                 + 1.08134 w_k1_INTC_s015
                                 + 0.930775 w_k1_KO_s015
                                 + 0.760003 w_k1_MAT_s015
                                 + 0.985957 w_k1_MSFT_s015
                                 + 1.00295 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.05095 w_k1_ADM_s015
                                 + 1.05597 w_k1_AXP_s015
                                 + 1.29627 w_k1_DIS_s015
                                 + 1.32689 w_k1_FDX_s015
                                 + 0.718415 w_k1_IBM_s015
                                 + 1.3186 w_k1_INTC_s015
                                 + 0.862249 w_k1_KO_s015
                                 + 0.485032 w_k1_MAT_s015
                                 + 1.18321 w_k1_MSFT_s015
                                 + 0.930524 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.635048 w_k1_ADM_s015
                                 + 0.978343 w_k1_AXP_s015
                                 + 1.20176 w_k1_DIS_s015
                                 + 0.986259 w_k1_FDX_s015
                                 + 1.67381 w_k1_IBM_s015
                                 + 0.96394 w_k1_INTC_s015
                                 + 1.18859 w_k1_KO_s015 + 1.6716 w_k1_MAT_s015
                                 + 0.904554 w_k1_MSFT_s015
                                 + 1.01827 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.981952 w_k1_ADM_s015
                                 + 1.52694 w_k1_AXP_s015 + 0.8991 w_k1_DIS_s015
                                 + 1.31078 w_k1_FDX_s015
                                 + 1.04395 w_k1_IBM_s015
                                 + 1.3121 w_k1_INTC_s015
                                 + 0.975584 w_k1_KO_s015
                                 + 1.04709 w_k1_MAT_s015
                                 + 1.0269 w_k1_MSFT_s015
                                 + 0.866777 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.67048 w_k1_ADM_s015
                                 + 1.42347 w_k1_AXP_s015 + 1.6368 w_k1_DIS_s015
                                 + 1.35122 w_k1_FDX_s015 + 1.3866 w_k1_IBM_s015
                                 + 1.6623 w_k1_INTC_s015 + 1.53806 w_k1_KO_s015
                                 + 0.991732 w_k1_MAT_s015
                                 + 1.24749 w_k1_MSFT_s015
                                 + 1.43034 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.24255 w_k1_ADM_s015
                                 + 0.800167 w_k1_AXP_s015
                                 + 0.756329 w_k1_DIS_s015
                                 + 0.864352 w_k1_FDX_s015
                                 + 1.16973 w_k1_IBM_s015
                                 + 0.851793 w_k1_INTC_s015
                                 + 0.932463 w_k1_KO_s015
                                 + 0.965656 w_k1_MAT_s015
                                 + 0.890634 w_k1_MSFT_s015
                                 + 0.84602 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.95838 w_k1_ADM_s016
                                 + 1.33875 w_k1_AXP_s016
                                 + 1.06177 w_k1_DIS_s016
                                 + 1.07128 w_k1_FDX_s016
                                 + 0.985287 w_k1_IBM_s016
                                 + 0.901143 w_k1_INTC_s016
                                 + 1.06172 w_k1_KO_s016 + 1.42948 w_k1_MAT_s016
                                 + 0.885232 w_k1_MSFT_s016
                                 + 1.07347 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.639579 w_k1_ADM_s016
                                 + 0.66227 w_k1_AXP_s016
                                 + 0.651608 w_k1_DIS_s016
                                 + 0.738012 w_k1_FDX_s016
                                 + 1.10317 w_k1_IBM_s016
                                 + 0.868229 w_k1_INTC_s016
                                 + 0.89195 w_k1_KO_s016
                                 + 0.887789 w_k1_MAT_s016
                                 + 0.834098 w_k1_MSFT_s016
                                 + 0.878106 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.26337 w_k1_ADM_s016
                                 + 1.34115 w_k1_AXP_s016
                                 + 1.43291 w_k1_DIS_s016
                                 + 1.15975 w_k1_FDX_s016
                                 + 1.47649 w_k1_IBM_s016
                                 + 1.10805 w_k1_INTC_s016
                                 + 1.34246 w_k1_KO_s016 + 1.41529 w_k1_MAT_s016
                                 + 1.17287 w_k1_MSFT_s016
                                 + 1.18461 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.24846 w_k1_ADM_s016
                                 + 1.56974 w_k1_AXP_s016
                                 + 1.57639 w_k1_DIS_s016
                                 + 1.57752 w_k1_FDX_s016
                                 + 0.965744 w_k1_IBM_s016
                                 + 1.36685 w_k1_INTC_s016
                                 + 1.26566 w_k1_KO_s016 + 1.29662 w_k1_MAT_s016
                                 + 1.27283 w_k1_MSFT_s016 + 1.2908 w_k1_PG_s016
                                 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.863453 w_k1_ADM_s016
                                 + 0.684442 w_k1_AXP_s016
                                 + 0.667081 w_k1_DIS_s016
                                 + 0.781642 w_k1_FDX_s016
                                 + 0.777492 w_k1_IBM_s016
                                 + 0.779702 w_k1_INTC_s016
                                 + 0.722332 w_k1_KO_s016
                                 + 0.590715 w_k1_MAT_s016
                                 + 0.749751 w_k1_MSFT_s016
                                 + 0.81455 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.934287 w_k1_ADM_s016
                                 + 1.02471 w_k1_AXP_s016
                                 + 1.08081 w_k1_DIS_s016
                                 + 0.950689 w_k1_FDX_s016 + 1.4 w_k1_IBM_s016
                                 + 0.875409 w_k1_INTC_s016
                                 + 1.25279 w_k1_KO_s016 + 1.50116 w_k1_MAT_s016
                                 + 0.940225 w_k1_MSFT_s016
                                 + 1.21258 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.768546 w_k1_ADM_s016
                                 + 0.714413 w_k1_AXP_s016
                                 + 1.11894 w_k1_DIS_s016
                                 + 0.927373 w_k1_FDX_s016
                                 + 0.977627 w_k1_IBM_s016
                                 + 1.04679 w_k1_INTC_s016
                                 + 1.02892 w_k1_KO_s016 + 1.04028 w_k1_MAT_s016
                                 + 1.04663 w_k1_MSFT_s016
                                 + 1.05879 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.754879 w_k1_ADM_s016
                                 + 0.911971 w_k1_AXP_s016
                                 + 1.01046 w_k1_DIS_s016 + 1.1595 w_k1_FDX_s016
                                 + 0.738599 w_k1_IBM_s016
                                 + 1.11905 w_k1_INTC_s016
                                 + 0.826877 w_k1_KO_s016
                                 + 0.858474 w_k1_MAT_s016
                                 + 1.13426 w_k1_MSFT_s016
                                 + 0.998855 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.702456 w_k1_ADM_s017
                                 + 0.69412 w_k1_AXP_s017
                                 + 0.627552 w_k1_DIS_s017
                                 + 0.72133 w_k1_FDX_s017
                                 + 0.649235 w_k1_IBM_s017
                                 + 1.08273 w_k1_INTC_s017
                                 + 0.829957 w_k1_KO_s017
                                 + 0.943707 w_k1_MAT_s017
                                 + 0.988395 w_k1_MSFT_s017
                                 + 0.769165 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.701071 w_k1_ADM_s017
                                 + 0.924002 w_k1_AXP_s017
                                 + 0.66962 w_k1_DIS_s017
                                 + 0.816387 w_k1_FDX_s017
                                 + 1.0069 w_k1_IBM_s017 + 1.0344 w_k1_INTC_s017
                                 + 0.978866 w_k1_KO_s017
                                 + 1.44462 w_k1_MAT_s017
                                 + 1.01337 w_k1_MSFT_s017
                                 + 0.915731 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.04517 w_k1_ADM_s017
                                 + 1.42368 w_k1_AXP_s017
                                 + 1.32354 w_k1_DIS_s017 + 1.1931 w_k1_FDX_s017
                                 + 1.26352 w_k1_IBM_s017
                                 + 1.58862 w_k1_INTC_s017
                                 + 1.50098 w_k1_KO_s017 + 2.39718 w_k1_MAT_s017
                                 + 1.57354 w_k1_MSFT_s017
                                 + 1.30592 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.634188 w_k1_ADM_s017
                                 + 0.934391 w_k1_AXP_s017
                                 + 0.935728 w_k1_DIS_s017
                                 + 0.862355 w_k1_FDX_s017
                                 + 1.50216 w_k1_IBM_s017
                                 + 1.29355 w_k1_INTC_s017
                                 + 1.18071 w_k1_KO_s017 + 1.84386 w_k1_MAT_s017
                                 + 1.07838 w_k1_MSFT_s017
                                 + 0.95253 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.01281 w_k1_ADM_s017
                                 + 0.803138 w_k1_AXP_s017
                                 + 0.762019 w_k1_DIS_s017
                                 + 0.948204 w_k1_FDX_s017
                                 + 0.668176 w_k1_IBM_s017
                                 + 1.21869 w_k1_INTC_s017
                                 + 0.877593 w_k1_KO_s017
                                 + 0.917661 w_k1_MAT_s017
                                 + 1.01084 w_k1_MSFT_s017
                                 + 0.884643 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.09711 w_k1_ADM_s017
                                 + 0.815721 w_k1_AXP_s017
                                 + 0.91457 w_k1_DIS_s017
                                 + 0.844876 w_k1_FDX_s017
                                 + 0.91859 w_k1_IBM_s017
                                 + 1.06892 w_k1_INTC_s017
                                 + 1.10808 w_k1_KO_s017
                                 + 0.953231 w_k1_MAT_s017
                                 + 1.05713 w_k1_MSFT_s017
                                 + 0.907838 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.840853 w_k1_ADM_s017
                                 + 0.870575 w_k1_AXP_s017
                                 + 0.90743 w_k1_DIS_s017
                                 + 1.02791 w_k1_FDX_s017
                                 + 0.671461 w_k1_IBM_s017
                                 + 1.38531 w_k1_INTC_s017
                                 + 1.03189 w_k1_KO_s017 + 1.0761 w_k1_MAT_s017
                                 + 1.27798 w_k1_MSFT_s017 + 1.112 w_k1_PG_s017
                                 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.816171 w_k1_ADM_s017
                                 + 1.44463 w_k1_AXP_s017
                                 + 0.923889 w_k1_DIS_s017
                                 + 1.0989 w_k1_FDX_s017 + 1.19381 w_k1_IBM_s017
                                 + 1.39712 w_k1_INTC_s017
                                 + 0.921703 w_k1_KO_s017
                                 + 1.93842 w_k1_MAT_s017
                                 + 1.21587 w_k1_MSFT_s017
                                 + 0.941326 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.23703 w_k1_ADM_s100
                                 + 1.04538 w_k1_AXP_s100
                                 + 1.13977 w_k1_DIS_s100
                                 + 1.09712 w_k1_FDX_s100
                                 + 0.954508 w_k1_IBM_s100
                                 + 1.42308 w_k1_INTC_s100
                                 + 1.11818 w_k1_KO_s100
                                 + 0.873882 w_k1_MAT_s100
                                 + 1.1519 w_k1_MSFT_s100 + 1.00491 w_k1_PG_s100
                                 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.957822 w_k1_ADM_s100
                                 + 0.689077 w_k1_AXP_s100
                                 + 0.682868 w_k1_DIS_s100
                                 + 0.83728 w_k1_FDX_s100
                                 + 0.578792 w_k1_IBM_s100
                                 + 0.959443 w_k1_INTC_s100
                                 + 0.668647 w_k1_KO_s100
                                 + 0.550197 w_k1_MAT_s100
                                 + 0.817279 w_k1_MSFT_s100
                                 + 0.851187 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.02636 w_k1_ADM_s100
                                 + 1.0282 w_k1_AXP_s100 + 1.01062 w_k1_DIS_s100
                                 + 0.873676 w_k1_FDX_s100
                                 + 1.48696 w_k1_IBM_s100
                                 + 0.955809 w_k1_INTC_s100
                                 + 1.22627 w_k1_KO_s100 + 1.54531 w_k1_MAT_s100
                                 + 0.939965 w_k1_MSFT_s100
                                 + 1.07198 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.967291 w_k1_ADM_s100
                                 + 0.828038 w_k1_AXP_s100
                                 + 0.982005 w_k1_DIS_s100
                                 + 0.987839 w_k1_FDX_s100
                                 + 0.726885 w_k1_IBM_s100
                                 + 0.8632 w_k1_INTC_s100
                                 + 0.840542 w_k1_KO_s100
                                 + 0.737242 w_k1_MAT_s100
                                 + 0.937421 w_k1_MSFT_s100
                                 + 0.895008 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.856527 w_k1_ADM_s100
                                 + 0.8217 w_k1_AXP_s100 + 0.77707 w_k1_DIS_s100
                                 + 0.820346 w_k1_FDX_s100
                                 + 0.955073 w_k1_IBM_s100
                                 + 1.04169 w_k1_INTC_s100
                                 + 0.822128 w_k1_KO_s100
                                 + 0.855594 w_k1_MAT_s100
                                 + 0.798552 w_k1_MSFT_s100
                                 + 0.950115 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.14057 w_k1_ADM_s100
                                 + 1.35818 w_k1_AXP_s100 + 1.595 w_k1_DIS_s100
                                 + 1.25162 w_k1_FDX_s100
                                 + 1.73449 w_k1_IBM_s100
                                 + 1.16536 w_k1_INTC_s100
                                 + 1.39777 w_k1_KO_s100 + 1.38146 w_k1_MAT_s100
                                 + 1.1737 w_k1_MSFT_s100 + 1.23541 w_k1_PG_s100
                                 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.40956 w_k1_ADM_s100
                                 + 1.58088 w_k1_AXP_s100
                                 + 1.33037 w_k1_DIS_s100 + 1.3833 w_k1_FDX_s100
                                 + 1.05823 w_k1_IBM_s100
                                 + 1.13346 w_k1_INTC_s100
                                 + 0.961097 w_k1_KO_s100
                                 + 1.10131 w_k1_MAT_s100
                                 + 1.05258 w_k1_MSFT_s100
                                 + 1.05849 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.722367 w_k1_ADM_s100
                                 + 0.60463 w_k1_AXP_s100
                                 + 0.498514 w_k1_DIS_s100
                                 + 0.672897 w_k1_FDX_s100
                                 + 0.61561 w_k1_IBM_s100
                                 + 0.837621 w_k1_INTC_s100
                                 + 0.598618 w_k1_KO_s100
                                 + 0.494735 w_k1_MAT_s100
                                 + 0.68591 w_k1_MSFT_s100
                                 + 0.718978 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.932694 w_k1_ADM_s101
                                 + 0.955119 w_k1_AXP_s101
                                 + 1.01887 w_k1_DIS_s101
                                 + 1.05368 w_k1_FDX_s101
                                 + 0.905453 w_k1_IBM_s101
                                 + 1.16571 w_k1_INTC_s101
                                 + 0.876076 w_k1_KO_s101
                                 + 0.684197 w_k1_MAT_s101
                                 + 1.01664 w_k1_MSFT_s101
                                 + 1.04864 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.858905 w_k1_ADM_s101
                                 + 0.784645 w_k1_AXP_s101
                                 + 0.668094 w_k1_DIS_s101
                                 + 0.908213 w_k1_FDX_s101
                                 + 0.69281 w_k1_IBM_s101
                                 + 1.15262 w_k1_INTC_s101
                                 + 0.768933 w_k1_KO_s101
                                 + 0.57512 w_k1_MAT_s101
                                 + 0.906403 w_k1_MSFT_s101
                                 + 0.912606 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.18193 w_k1_ADM_s101
                                 + 1.28423 w_k1_AXP_s101
                                 + 1.24859 w_k1_DIS_s101
                                 + 1.18091 w_k1_FDX_s101
                                 + 0.931316 w_k1_IBM_s101
                                 + 1.36589 w_k1_INTC_s101
                                 + 1.13389 w_k1_KO_s101
                                 + 0.830271 w_k1_MAT_s101
                                 + 1.26218 w_k1_MSFT_s101
                                 + 1.07424 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.11518 w_k1_ADM_s101
                                 + 1.45728 w_k1_AXP_s101
                                 + 1.22024 w_k1_DIS_s101
                                 + 1.04457 w_k1_FDX_s101
                                 + 1.65843 w_k1_IBM_s101
                                 + 1.21357 w_k1_INTC_s101
                                 + 1.39439 w_k1_KO_s101 + 1.44294 w_k1_MAT_s101
                                 + 1.02649 w_k1_MSFT_s101
                                 + 1.17455 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.33065 w_k1_ADM_s101
                                 + 1.33365 w_k1_AXP_s101
                                 + 1.52507 w_k1_DIS_s101
                                 + 1.34878 w_k1_FDX_s101
                                 + 1.35236 w_k1_IBM_s101
                                 + 1.38175 w_k1_INTC_s101
                                 + 1.44964 w_k1_KO_s101
                                 + 0.965461 w_k1_MAT_s101
                                 + 1.3132 w_k1_MSFT_s101 + 1.18198 w_k1_PG_s101
                                 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.771587 w_k1_ADM_s101
                                 + 1.05487 w_k1_AXP_s101
                                 + 0.962569 w_k1_DIS_s101
                                 + 0.968407 w_k1_FDX_s101
                                 + 1.3048 w_k1_IBM_s101 + 1.1747 w_k1_INTC_s101
                                 + 0.984373 w_k1_KO_s101
                                 + 1.26418 w_k1_MAT_s101
                                 + 1.08945 w_k1_MSFT_s101
                                 + 1.04116 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.951903 w_k1_ADM_s101
                                 + 0.81067 w_k1_AXP_s101
                                 + 0.772524 w_k1_DIS_s101
                                 + 0.870681 w_k1_FDX_s101
                                 + 1.06792 w_k1_IBM_s101
                                 + 1.0516 w_k1_INTC_s101
                                 + 0.917071 w_k1_KO_s101 + 1.0561 w_k1_MAT_s101
                                 + 0.908848 w_k1_MSFT_s101
                                 + 0.983559 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.04048 w_k1_ADM_s101
                                 + 1.03027 w_k1_AXP_s101
                                 + 1.50482 w_k1_DIS_s101
                                 + 1.09076 w_k1_FDX_s101
                                 + 1.22529 w_k1_IBM_s101
                                 + 1.23328 w_k1_INTC_s101
                                 + 1.36561 w_k1_KO_s101
                                 + 0.863951 w_k1_MAT_s101
                                 + 1.28936 w_k1_MSFT_s101
                                 + 1.08273 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.779844 w_k1_ADM_s102
                                 + 0.672431 w_k1_AXP_s102
                                 + 0.778937 w_k1_DIS_s102
                                 + 0.814523 w_k1_FDX_s102
                                 + 0.71976 w_k1_IBM_s102
                                 + 0.854375 w_k1_INTC_s102
                                 + 0.743104 w_k1_KO_s102
                                 + 0.567503 w_k1_MAT_s102
                                 + 0.810134 w_k1_MSFT_s102
                                 + 0.84312 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.07727 w_k1_ADM_s102
                                 + 0.890172 w_k1_AXP_s102
                                 + 1.14336 w_k1_DIS_s102
                                 + 1.03051 w_k1_FDX_s102
                                 + 0.876595 w_k1_IBM_s102
                                 + 1.22997 w_k1_INTC_s102
                                 + 0.98964 w_k1_KO_s102
                                 + 0.504566 w_k1_MAT_s102
                                 + 1.05898 w_k1_MSFT_s102
                                 + 0.941982 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.923189 w_k1_ADM_s102
                                 + 1.17911 w_k1_AXP_s102
                                 + 1.13132 w_k1_DIS_s102
                                 + 0.959193 w_k1_FDX_s102
                                 + 1.56617 w_k1_IBM_s102
                                 + 1.17066 w_k1_INTC_s102
                                 + 1.14634 w_k1_KO_s102 + 1.47854 w_k1_MAT_s102
                                 + 1.05333 w_k1_MSFT_s102
                                 + 0.984854 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.28989 w_k1_ADM_s102
                                 + 1.28415 w_k1_AXP_s102 + 1.3972 w_k1_DIS_s102
                                 + 1.39141 w_k1_FDX_s102
                                 + 0.944069 w_k1_IBM_s102
                                 + 1.38011 w_k1_INTC_s102
                                 + 1.05991 w_k1_KO_s102 + 1.1278 w_k1_MAT_s102
                                 + 1.29538 w_k1_MSFT_s102
                                 + 1.01692 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.66094 w_k1_ADM_s102
                                 + 0.813206 w_k1_AXP_s102
                                 + 1.07401 w_k1_DIS_s102
                                 + 0.837119 w_k1_FDX_s102
                                 + 1.0189 w_k1_IBM_s102
                                 + 1.10798 w_k1_INTC_s102
                                 + 0.970092 w_k1_KO_s102
                                 + 0.865625 w_k1_MAT_s102
                                 + 0.861514 w_k1_MSFT_s102
                                 + 0.893773 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.77311 w_k1_ADM_s102
                                 + 0.629192 w_k1_AXP_s102
                                 + 0.707495 w_k1_DIS_s102
                                 + 0.655819 w_k1_FDX_s102
                                 + 1.11575 w_k1_IBM_s102
                                 + 0.800224 w_k1_INTC_s102
                                 + 0.856451 w_k1_KO_s102
                                 + 1.23759 w_k1_MAT_s102
                                 + 0.777503 w_k1_MSFT_s102
                                 + 0.830085 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.06559 w_k1_ADM_s102
                                 + 1.1531 w_k1_AXP_s102 + 1.67181 w_k1_DIS_s102
                                 + 1.24459 w_k1_FDX_s102
                                 + 1.29034 w_k1_IBM_s102
                                 + 1.59899 w_k1_INTC_s102
                                 + 1.23628 w_k1_KO_s102 + 0.85629 w_k1_MAT_s102
                                 + 1.35369 w_k1_MSFT_s102
                                 + 1.02713 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.14515 w_k1_ADM_s102
                                 + 0.988987 w_k1_AXP_s102
                                 + 0.97067 w_k1_DIS_s102
                                 + 1.00609 w_k1_FDX_s102
                                 + 1.19203 w_k1_IBM_s102
                                 + 1.0247 w_k1_INTC_s102 + 1.09587 w_k1_KO_s102
                                 + 1.1641 w_k1_MAT_s102
                                 + 1.02547 w_k1_MSFT_s102
                                 + 1.00424 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.17443 w_k1_ADM_s103
                                 + 1.0192 w_k1_AXP_s103 + 1.4591 w_k1_DIS_s103
                                 + 1.1584 w_k1_FDX_s103 + 1.11083 w_k1_IBM_s103
                                 + 1.12917 w_k1_INTC_s103
                                 + 1.26099 w_k1_KO_s103 + 1.20529 w_k1_MAT_s103
                                 + 1.22963 w_k1_MSFT_s103 + 1.1827 w_k1_PG_s103
                                 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.41247 w_k1_ADM_s103
                                 + 1.5925 w_k1_AXP_s103 + 1.36364 w_k1_DIS_s103
                                 + 1.47076 w_k1_FDX_s103
                                 + 0.945086 w_k1_IBM_s103
                                 + 1.23499 w_k1_INTC_s103
                                 + 1.10629 w_k1_KO_s103 + 1.08227 w_k1_MAT_s103
                                 + 1.04684 w_k1_MSFT_s103
                                 + 0.981086 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.27473 w_k1_ADM_s103
                                 + 1.46249 w_k1_AXP_s103
                                 + 1.51603 w_k1_DIS_s103
                                 + 1.21285 w_k1_FDX_s103
                                 + 1.48339 w_k1_IBM_s103
                                 + 1.29614 w_k1_INTC_s103
                                 + 1.17934 w_k1_KO_s103 + 1.16967 w_k1_MAT_s103
                                 + 0.915357 w_k1_MSFT_s103
                                 + 1.17516 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.895594 w_k1_ADM_s103
                                 + 0.970443 w_k1_AXP_s103
                                 + 1.00745 w_k1_DIS_s103
                                 + 0.860452 w_k1_FDX_s103
                                 + 1.30887 w_k1_IBM_s103
                                 + 1.14242 w_k1_INTC_s103
                                 + 1.23271 w_k1_KO_s103 + 1.17881 w_k1_MAT_s103
                                 + 0.942851 w_k1_MSFT_s103
                                 + 0.926298 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.770172 w_k1_ADM_s103
                                 + 0.797122 w_k1_AXP_s103
                                 + 0.823497 w_k1_DIS_s103
                                 + 0.842001 w_k1_FDX_s103
                                 + 0.786732 w_k1_IBM_s103
                                 + 0.78103 w_k1_INTC_s103
                                 + 0.889735 w_k1_KO_s103
                                 + 0.756471 w_k1_MAT_s103
                                 + 0.813366 w_k1_MSFT_s103
                                 + 0.952507 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.752422 w_k1_ADM_s103
                                 + 0.787541 w_k1_AXP_s103
                                 + 0.953633 w_k1_DIS_s103
                                 + 0.957996 w_k1_FDX_s103
                                 + 0.875842 w_k1_IBM_s103
                                 + 0.991862 w_k1_INTC_s103
                                 + 0.875472 w_k1_KO_s103
                                 + 0.892303 w_k1_MAT_s103
                                 + 0.929374 w_k1_MSFT_s103
                                 + 0.928712 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.996821 w_k1_ADM_s103
                                 + 1.10116 w_k1_AXP_s103
                                 + 1.01875 w_k1_DIS_s103
                                 + 1.04138 w_k1_FDX_s103
                                 + 1.06695 w_k1_IBM_s103
                                 + 1.08295 w_k1_INTC_s103
                                 + 1.08103 w_k1_KO_s103 + 1.42949 w_k1_MAT_s103
                                 + 0.803079 w_k1_MSFT_s103
                                 + 1.12635 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.22163 w_k1_ADM_s103
                                 + 1.17985 w_k1_AXP_s103
                                 + 1.29774 w_k1_DIS_s103
                                 + 1.27426 w_k1_FDX_s103
                                 + 0.863184 w_k1_IBM_s103
                                 + 1.14789 w_k1_INTC_s103
                                 + 1.04526 w_k1_KO_s103
                                 + 0.864875 w_k1_MAT_s103
                                 + 1.02342 w_k1_MSFT_s103
                                 + 1.23062 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.855056 w_k1_ADM_s104
                                 + 0.930593 w_k1_AXP_s104
                                 + 0.793041 w_k1_DIS_s104
                                 + 0.850563 w_k1_FDX_s104
                                 + 0.781455 w_k1_IBM_s104
                                 + 0.699011 w_k1_INTC_s104
                                 + 0.690092 w_k1_KO_s104
                                 + 0.82878 w_k1_MAT_s104
                                 + 0.68823 w_k1_MSFT_s104
                                 + 0.810809 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.20278 w_k1_ADM_s104
                                 + 0.999242 w_k1_AXP_s104
                                 + 0.977318 w_k1_DIS_s104
                                 + 1.07418 w_k1_FDX_s104
                                 + 0.715211 w_k1_IBM_s104
                                 + 0.860035 w_k1_INTC_s104
                                 + 0.79782 w_k1_KO_s104
                                 + 0.689436 w_k1_MAT_s104
                                 + 0.93489 w_k1_MSFT_s104
                                 + 0.952199 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.957431 w_k1_ADM_s104
                                 + 0.757923 w_k1_AXP_s104
                                 + 0.6311 w_k1_DIS_s104
                                 + 0.721299 w_k1_FDX_s104
                                 + 0.954653 w_k1_IBM_s104
                                 + 0.853118 w_k1_INTC_s104
                                 + 0.841901 w_k1_KO_s104
                                 + 0.931384 w_k1_MAT_s104
                                 + 0.74206 w_k1_MSFT_s104
                                 + 0.862431 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.0404 w_k1_ADM_s104
                                 + 0.989855 w_k1_AXP_s104
                                 + 1.02697 w_k1_DIS_s104
                                 + 0.961599 w_k1_FDX_s104
                                 + 1.01883 w_k1_IBM_s104
                                 + 0.945998 w_k1_INTC_s104
                                 + 0.965775 w_k1_KO_s104
                                 + 0.935461 w_k1_MAT_s104
                                 + 1.02856 w_k1_MSFT_s104
                                 + 0.979774 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.891514 w_k1_ADM_s104
                                 + 0.63139 w_k1_AXP_s104
                                 + 0.735878 w_k1_DIS_s104
                                 + 0.683173 w_k1_FDX_s104
                                 + 1.13672 w_k1_IBM_s104
                                 + 0.829146 w_k1_INTC_s104
                                 + 0.929955 w_k1_KO_s104
                                 + 0.995427 w_k1_MAT_s104
                                 + 0.870261 w_k1_MSFT_s104
                                 + 0.937417 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.34093 w_k1_ADM_s104
                                 + 1.34769 w_k1_AXP_s104
                                 + 1.46794 w_k1_DIS_s104
                                 + 1.24424 w_k1_FDX_s104 + 1.1262 w_k1_IBM_s104
                                 + 1.34912 w_k1_INTC_s104
                                 + 1.06981 w_k1_KO_s104 + 1.11453 w_k1_MAT_s104
                                 + 1.28793 w_k1_MSFT_s104
                                 + 1.14056 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.700819 w_k1_ADM_s104
                                 + 0.520159 w_k1_AXP_s104
                                 + 0.783187 w_k1_DIS_s104
                                 + 0.712834 w_k1_FDX_s104
                                 + 0.755926 w_k1_IBM_s104
                                 + 0.937511 w_k1_INTC_s104
                                 + 0.804369 w_k1_KO_s104
                                 + 0.561845 w_k1_MAT_s104
                                 + 0.873338 w_k1_MSFT_s104
                                 + 0.829006 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.757062 w_k1_ADM_s104
                                 + 1.22854 w_k1_AXP_s104
                                 + 1.12258 w_k1_DIS_s104
                                 + 0.932067 w_k1_FDX_s104
                                 + 1.50066 w_k1_IBM_s104
                                 + 1.0679 w_k1_INTC_s104 + 1.18866 w_k1_KO_s104
                                 + 1.30264 w_k1_MAT_s104
                                 + 0.868967 w_k1_MSFT_s104
                                 + 1.04173 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.1442 w_k1_ADM_s105
                                 + 0.989552 w_k1_AXP_s105
                                 + 0.758457 w_k1_DIS_s105
                                 + 0.978275 w_k1_FDX_s105
                                 + 0.905054 w_k1_IBM_s105
                                 + 0.770828 w_k1_INTC_s105
                                 + 0.988795 w_k1_KO_s105
                                 + 1.10838 w_k1_MAT_s105
                                 + 0.885579 w_k1_MSFT_s105
                                 + 0.885851 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.06296 w_k1_ADM_s105
                                 + 0.689874 w_k1_AXP_s105
                                 + 0.855856 w_k1_DIS_s105
                                 + 0.851821 w_k1_FDX_s105
                                 + 0.831561 w_k1_IBM_s105
                                 + 0.978025 w_k1_INTC_s105
                                 + 0.877898 w_k1_KO_s105
                                 + 0.679546 w_k1_MAT_s105
                                 + 1.02696 w_k1_MSFT_s105
                                 + 0.84526 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.15516 w_k1_ADM_s105
                                 + 0.945991 w_k1_AXP_s105
                                 + 1.32481 w_k1_DIS_s105
                                 + 1.19887 w_k1_FDX_s105
                                 + 0.883291 w_k1_IBM_s105
                                 + 1.08208 w_k1_INTC_s105
                                 + 1.14553 w_k1_KO_s105 + 1.02524 w_k1_MAT_s105
                                 + 1.50593 w_k1_MSFT_s105
                                 + 1.22564 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.37169 w_k1_ADM_s105
                                 + 1.1743 w_k1_AXP_s105 + 1.12195 w_k1_DIS_s105
                                 + 1.01284 w_k1_FDX_s105
                                 + 1.13545 w_k1_IBM_s105
                                 + 1.0928 w_k1_INTC_s105 + 1.2058 w_k1_KO_s105
                                 + 1.26279 w_k1_MAT_s105
                                 + 1.13274 w_k1_MSFT_s105
                                 + 1.05421 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.1192 w_k1_ADM_s105
                                 + 0.891931 w_k1_AXP_s105
                                 + 0.852484 w_k1_DIS_s105
                                 + 0.84185 w_k1_FDX_s105
                                 + 1.32058 w_k1_IBM_s105
                                 + 0.93289 w_k1_INTC_s105
                                 + 1.07452 w_k1_KO_s105 + 1.10364 w_k1_MAT_s105
                                 + 1.01248 w_k1_MSFT_s105
                                 + 0.948668 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.1532 w_k1_ADM_s105
                                 + 1.4738 w_k1_AXP_s105 + 1.03534 w_k1_DIS_s105
                                 + 1.21805 w_k1_FDX_s105
                                 + 0.807771 w_k1_IBM_s105
                                 + 1.0452 w_k1_INTC_s105
                                 + 0.825748 w_k1_KO_s105
                                 + 1.15487 w_k1_MAT_s105
                                 + 1.15166 w_k1_MSFT_s105
                                 + 0.836103 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.796499 w_k1_ADM_s105
                                 + 0.901829 w_k1_AXP_s105
                                 + 0.819217 w_k1_DIS_s105
                                 + 0.982249 w_k1_FDX_s105
                                 + 0.600785 w_k1_IBM_s105
                                 + 1.07336 w_k1_INTC_s105
                                 + 0.753899 w_k1_KO_s105
                                 + 0.860176 w_k1_MAT_s105
                                 + 1.09308 w_k1_MSFT_s105
                                 + 0.80682 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.958867 w_k1_ADM_s105
                                 + 1.79085 w_k1_AXP_s105
                                 + 1.50321 w_k1_DIS_s105
                                 + 1.34666 w_k1_FDX_s105
                                 + 1.39411 w_k1_IBM_s105
                                 + 1.74593 w_k1_INTC_s105
                                 + 1.51788 w_k1_KO_s105 + 1.89551 w_k1_MAT_s105
                                 + 1.54432 w_k1_MSFT_s105
                                 + 1.30047 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.920637 w_k1_ADM_s106
                                 + 1.26195 w_k1_AXP_s106
                                 + 0.80242 w_k1_DIS_s106
                                 + 0.948732 w_k1_FDX_s106
                                 + 0.798246 w_k1_IBM_s106
                                 + 1.19118 w_k1_INTC_s106
                                 + 0.821838 w_k1_KO_s106
                                 + 0.742919 w_k1_MAT_s106
                                 + 0.843449 w_k1_MSFT_s106
                                 + 0.936075 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.03273 w_k1_ADM_s106
                                 + 1.68255 w_k1_AXP_s106
                                 + 1.85881 w_k1_DIS_s106 + 1.5081 w_k1_FDX_s106
                                 + 1.23873 w_k1_IBM_s106
                                 + 1.66842 w_k1_INTC_s106
                                 + 1.21959 w_k1_KO_s106 + 1.122 w_k1_MAT_s106
                                 + 1.34778 w_k1_MSFT_s106
                                 + 1.12416 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.901196 w_k1_ADM_s106
                                 + 0.734764 w_k1_AXP_s106
                                 + 0.911525 w_k1_DIS_s106
                                 + 0.822465 w_k1_FDX_s106
                                 + 0.965734 w_k1_IBM_s106
                                 + 0.770764 w_k1_INTC_s106
                                 + 0.913106 w_k1_KO_s106
                                 + 0.818294 w_k1_MAT_s106
                                 + 0.777699 w_k1_MSFT_s106
                                 + 0.947365 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.930109 w_k1_ADM_s106
                                 + 0.764071 w_k1_AXP_s106
                                 + 0.595445 w_k1_DIS_s106
                                 + 0.757231 w_k1_FDX_s106
                                 + 0.708184 w_k1_IBM_s106
                                 + 0.736083 w_k1_INTC_s106
                                 + 0.652296 w_k1_KO_s106
                                 + 0.574272 w_k1_MAT_s106
                                 + 0.74845 w_k1_MSFT_s106
                                 + 0.846471 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.946696 w_k1_ADM_s106
                                 + 1.12612 w_k1_AXP_s106
                                 + 1.07286 w_k1_DIS_s106
                                 + 1.11466 w_k1_FDX_s106
                                 + 0.686137 w_k1_IBM_s106
                                 + 1.16782 w_k1_INTC_s106
                                 + 0.781291 w_k1_KO_s106
                                 + 0.690593 w_k1_MAT_s106
                                 + 0.974541 w_k1_MSFT_s106
                                 + 0.807546 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.15018 w_k1_ADM_s106
                                 + 1.59451 w_k1_AXP_s106
                                 + 1.16223 w_k1_DIS_s106
                                 + 1.11243 w_k1_FDX_s106
                                 + 1.02527 w_k1_IBM_s106
                                 + 1.22446 w_k1_INTC_s106
                                 + 0.940739 w_k1_KO_s106
                                 + 1.13453 w_k1_MAT_s106
                                 + 1.17823 w_k1_MSFT_s106
                                 + 1.19951 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.09141 w_k1_ADM_s106
                                 + 1.17215 w_k1_AXP_s106 + 1.1877 w_k1_DIS_s106
                                 + 1.24609 w_k1_FDX_s106
                                 + 1.01239 w_k1_IBM_s106
                                 + 1.11332 w_k1_INTC_s106
                                 + 1.02443 w_k1_KO_s106
                                 + 0.759868 w_k1_MAT_s106
                                 + 1.13319 w_k1_MSFT_s106
                                 + 0.97736 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.96836 w_k1_ADM_s106
                                 + 1.04927 w_k1_AXP_s106
                                 + 1.06296 w_k1_DIS_s106
                                 + 0.943473 w_k1_FDX_s106
                                 + 1.46522 w_k1_IBM_s106
                                 + 1.10861 w_k1_INTC_s106
                                 + 1.21689 w_k1_KO_s106 + 1.25711 w_k1_MAT_s106
                                 + 1.03666 w_k1_MSFT_s106
                                 + 1.02003 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.08674 w_k1_ADM_s107
                                 + 0.938661 w_k1_AXP_s107
                                 + 1.11881 w_k1_DIS_s107
                                 + 0.980823 w_k1_FDX_s107
                                 + 1.04069 w_k1_IBM_s107
                                 + 1.1332 w_k1_INTC_s107 + 1.05276 w_k1_KO_s107
                                 + 0.989695 w_k1_MAT_s107
                                 + 0.974094 w_k1_MSFT_s107
                                 + 1.00075 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.908908 w_k1_ADM_s107
                                 + 0.799512 w_k1_AXP_s107
                                 + 0.91093 w_k1_DIS_s107
                                 + 0.931887 w_k1_FDX_s107
                                 + 0.763762 w_k1_IBM_s107
                                 + 0.974979 w_k1_INTC_s107
                                 + 0.769723 w_k1_KO_s107
                                 + 0.836317 w_k1_MAT_s107
                                 + 0.939411 w_k1_MSFT_s107
                                 + 0.92921 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.94974 w_k1_ADM_s107
                                 + 1.20161 w_k1_AXP_s107
                                 + 1.26666 w_k1_DIS_s107
                                 + 1.13054 w_k1_FDX_s107
                                 + 1.63834 w_k1_IBM_s107
                                 + 0.986662 w_k1_INTC_s107
                                 + 1.11183 w_k1_KO_s107 + 1.63576 w_k1_MAT_s107
                                 + 0.92839 w_k1_MSFT_s107
                                 + 1.09619 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.990374 w_k1_ADM_s107
                                 + 1.43795 w_k1_AXP_s107
                                 + 1.04986 w_k1_DIS_s107
                                 + 1.00657 w_k1_FDX_s107
                                 + 1.10067 w_k1_IBM_s107
                                 + 0.947818 w_k1_INTC_s107
                                 + 1.01686 w_k1_KO_s107 + 1.71853 w_k1_MAT_s107
                                 + 0.924835 w_k1_MSFT_s107
                                 + 1.10021 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.18944 w_k1_ADM_s107
                                 + 1.02458 w_k1_AXP_s107
                                 + 0.918177 w_k1_DIS_s107
                                 + 1.09719 w_k1_FDX_s107
                                 + 0.610938 w_k1_IBM_s107
                                 + 1.03285 w_k1_INTC_s107
                                 + 0.773063 w_k1_KO_s107
                                 + 0.765628 w_k1_MAT_s107
                                 + 1.05918 w_k1_MSFT_s107
                                 + 0.966165 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.13515 w_k1_ADM_s107
                                 + 1.2339 w_k1_AXP_s107 + 1.62613 w_k1_DIS_s107
                                 + 1.27992 w_k1_FDX_s107
                                 + 1.04716 w_k1_IBM_s107
                                 + 1.31217 w_k1_INTC_s107
                                 + 1.29191 w_k1_KO_s107
                                 + 0.903301 w_k1_MAT_s107
                                 + 1.28312 w_k1_MSFT_s107
                                 + 1.22349 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.20278 w_k1_ADM_s107
                                 + 1.41419 w_k1_AXP_s107
                                 + 1.99615 w_k1_DIS_s107
                                 + 1.34471 w_k1_FDX_s107
                                 + 1.56277 w_k1_IBM_s107
                                 + 1.03552 w_k1_INTC_s107
                                 + 1.36484 w_k1_KO_s107 + 1.94972 w_k1_MAT_s107
                                 + 1.19342 w_k1_MSFT_s107
                                 + 1.31167 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.901449 w_k1_ADM_s107
                                 + 0.859782 w_k1_AXP_s107
                                 + 0.874938 w_k1_DIS_s107
                                 + 0.875296 w_k1_FDX_s107
                                 + 1.32364 w_k1_IBM_s107
                                 + 0.851846 w_k1_INTC_s107
                                 + 0.926183 w_k1_KO_s107 + 1.0818 w_k1_MAT_s107
                                 + 0.849994 w_k1_MSFT_s107
                                 + 1.00397 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.05486 w_k1_ADM_s110
                                 + 1.23232 w_k1_AXP_s110
                                 + 1.05577 w_k1_DIS_s110
                                 + 1.23627 w_k1_FDX_s110
                                 + 0.704859 w_k1_IBM_s110
                                 + 1.31273 w_k1_INTC_s110
                                 + 0.914458 w_k1_KO_s110 + 0.8454 w_k1_MAT_s110
                                 + 1.21167 w_k1_MSFT_s110
                                 + 1.13637 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.02772 w_k1_ADM_s110
                                 + 1.07188 w_k1_AXP_s110
                                 + 0.991984 w_k1_DIS_s110
                                 + 0.888802 w_k1_FDX_s110
                                 + 1.37105 w_k1_IBM_s110
                                 + 0.95881 w_k1_INTC_s110
                                 + 1.19922 w_k1_KO_s110 + 1.19664 w_k1_MAT_s110
                                 + 0.985467 w_k1_MSFT_s110
                                 + 1.19225 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.15551 w_k1_ADM_s110
                                 + 1.21492 w_k1_AXP_s110
                                 + 1.43753 w_k1_DIS_s110
                                 + 1.20077 w_k1_FDX_s110
                                 + 1.05714 w_k1_IBM_s110
                                 + 0.950691 w_k1_INTC_s110
                                 + 1.12043 w_k1_KO_s110 + 1.10383 w_k1_MAT_s110
                                 + 1.24143 w_k1_MSFT_s110 + 1.0781 w_k1_PG_s110
                                 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.04322 w_k1_ADM_s110
                                 + 0.879578 w_k1_AXP_s110
                                 + 0.893622 w_k1_DIS_s110
                                 + 0.925351 w_k1_FDX_s110
                                 + 1.06532 w_k1_IBM_s110
                                 + 0.962252 w_k1_INTC_s110
                                 + 0.961313 w_k1_KO_s110
                                 + 1.04617 w_k1_MAT_s110
                                 + 0.946038 w_k1_MSFT_s110
                                 + 1.03773 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.08923 w_k1_ADM_s110
                                 + 1.64845 w_k1_AXP_s110
                                 + 1.08035 w_k1_DIS_s110
                                 + 1.30193 w_k1_FDX_s110
                                 + 1.03751 w_k1_IBM_s110
                                 + 1.23232 w_k1_INTC_s110
                                 + 1.01137 w_k1_KO_s110 + 1.36496 w_k1_MAT_s110
                                 + 1.0711 w_k1_MSFT_s110 + 1.24076 w_k1_PG_s110
                                 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.900118 w_k1_ADM_s110
                                 + 0.736679 w_k1_AXP_s110
                                 + 0.773885 w_k1_DIS_s110
                                 + 0.901601 w_k1_FDX_s110
                                 + 0.706611 w_k1_IBM_s110
                                 + 0.830444 w_k1_INTC_s110
                                 + 0.84342 w_k1_KO_s110
                                 + 0.691004 w_k1_MAT_s110
                                 + 1.06042 w_k1_MSFT_s110
                                 + 0.975421 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.62545 w_k1_ADM_s110
                                 + 1.57603 w_k1_AXP_s110
                                 + 1.37257 w_k1_DIS_s110
                                 + 1.09245 w_k1_FDX_s110
                                 + 1.86044 w_k1_IBM_s110
                                 + 1.2142 w_k1_INTC_s110 + 1.54286 w_k1_KO_s110
                                 + 1.82218 w_k1_MAT_s110
                                 + 1.23635 w_k1_MSFT_s110
                                 + 1.36269 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.879128 w_k1_ADM_s110
                                 + 0.711212 w_k1_AXP_s110
                                 + 0.57021 w_k1_DIS_s110
                                 + 0.686072 w_k1_FDX_s110
                                 + 0.865015 w_k1_IBM_s110
                                 + 0.667049 w_k1_INTC_s110
                                 + 0.66192 w_k1_KO_s110
                                 + 0.690617 w_k1_MAT_s110
                                 + 0.694182 w_k1_MSFT_s110
                                 + 0.819892 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.886193 w_k1_ADM_s111
                                 + 1.12079 w_k1_AXP_s111
                                 + 0.976673 w_k1_DIS_s111
                                 + 0.877818 w_k1_FDX_s111
                                 + 1.72786 w_k1_IBM_s111
                                 + 1.10801 w_k1_INTC_s111
                                 + 1.49521 w_k1_KO_s111 + 1.42329 w_k1_MAT_s111
                                 + 1.04948 w_k1_MSFT_s111
                                 + 1.10372 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.964022 w_k1_ADM_s111
                                 + 1.0471 w_k1_AXP_s111 + 1.05431 w_k1_DIS_s111
                                 + 1.0178 w_k1_FDX_s111 + 1.2561 w_k1_IBM_s111
                                 + 1.11928 w_k1_INTC_s111
                                 + 1.19541 w_k1_KO_s111 + 1.14084 w_k1_MAT_s111
                                 + 1.08107 w_k1_MSFT_s111 + 0.9326 w_k1_PG_s111
                                 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.80221 w_k1_ADM_s111
                                 + 0.696875 w_k1_AXP_s111
                                 + 0.878785 w_k1_DIS_s111
                                 + 0.886022 w_k1_FDX_s111
                                 + 0.927556 w_k1_IBM_s111
                                 + 0.988144 w_k1_INTC_s111
                                 + 1.03578 w_k1_KO_s111 + 0.8031 w_k1_MAT_s111
                                 + 1.04691 w_k1_MSFT_s111
                                 + 0.981842 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.37074 w_k1_ADM_s111
                                 + 1.29461 w_k1_AXP_s111
                                 + 1.48619 w_k1_DIS_s111
                                 + 1.42617 w_k1_FDX_s111
                                 + 1.28931 w_k1_IBM_s111
                                 + 1.5627 w_k1_INTC_s111 + 1.45056 w_k1_KO_s111
                                 + 1.27265 w_k1_MAT_s111
                                 + 1.3834 w_k1_MSFT_s111 + 1.19274 w_k1_PG_s111
                                 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.18578 w_k1_ADM_s111
                                 + 0.990516 w_k1_AXP_s111
                                 + 1.04458 w_k1_DIS_s111
                                 + 1.16195 w_k1_FDX_s111
                                 + 0.77449 w_k1_IBM_s111
                                 + 0.942884 w_k1_INTC_s111
                                 + 1.12615 w_k1_KO_s111 + 0.88817 w_k1_MAT_s111
                                 + 1.08539 w_k1_MSFT_s111
                                 + 1.05342 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.994144 w_k1_ADM_s111
                                 + 0.708899 w_k1_AXP_s111
                                 + 0.672586 w_k1_DIS_s111
                                 + 0.846213 w_k1_FDX_s111
                                 + 0.692931 w_k1_IBM_s111
                                 + 0.718923 w_k1_INTC_s111
                                 + 0.788583 w_k1_KO_s111
                                 + 0.719556 w_k1_MAT_s111
                                 + 0.77769 w_k1_MSFT_s111
                                 + 0.804149 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.18883 w_k1_ADM_s111
                                 + 1.14458 w_k1_AXP_s111
                                 + 1.19432 w_k1_DIS_s111
                                 + 1.16086 w_k1_FDX_s111
                                 + 0.966524 w_k1_IBM_s111
                                 + 1.14097 w_k1_INTC_s111
                                 + 1.31978 w_k1_KO_s111
                                 + 0.828745 w_k1_MAT_s111
                                 + 1.08506 w_k1_MSFT_s111
                                 + 1.07403 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.10235 w_k1_ADM_s111
                                 + 0.88717 w_k1_AXP_s111
                                 + 1.15632 w_k1_DIS_s111
                                 + 0.936879 w_k1_FDX_s111
                                 + 1.16046 w_k1_IBM_s111
                                 + 0.886833 w_k1_INTC_s111
                                 + 1.38665 w_k1_KO_s111 + 1.20344 w_k1_MAT_s111
                                 + 1.03891 w_k1_MSFT_s111
                                 + 1.23117 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.768564 w_k1_ADM_s112
                                 + 0.809263 w_k1_AXP_s112
                                 + 0.665117 w_k1_DIS_s112
                                 + 0.858003 w_k1_FDX_s112
                                 + 0.839985 w_k1_IBM_s112
                                 + 0.921767 w_k1_INTC_s112
                                 + 0.952151 w_k1_KO_s112 + 0.8605 w_k1_MAT_s112
                                 + 0.958141 w_k1_MSFT_s112
                                 + 0.932175 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.862248 w_k1_ADM_s112
                                 + 0.956357 w_k1_AXP_s112
                                 + 1.04416 w_k1_DIS_s112
                                 + 0.911187 w_k1_FDX_s112
                                 + 1.46977 w_k1_IBM_s112
                                 + 0.973407 w_k1_INTC_s112
                                 + 1.35889 w_k1_KO_s112 + 1.33395 w_k1_MAT_s112
                                 + 1.03824 w_k1_MSFT_s112
                                 + 1.06921 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.879909 w_k1_ADM_s112
                                 + 0.892768 w_k1_AXP_s112
                                 + 0.885637 w_k1_DIS_s112
                                 + 0.8623 w_k1_FDX_s112 + 1.06308 w_k1_IBM_s112
                                 + 0.891963 w_k1_INTC_s112
                                 + 1.04879 w_k1_KO_s112 + 1.19247 w_k1_MAT_s112
                                 + 1.0603 w_k1_MSFT_s112
                                 + 0.894843 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.988221 w_k1_ADM_s112
                                 + 1.11775 w_k1_AXP_s112
                                 + 1.21596 w_k1_DIS_s112
                                 + 1.16719 w_k1_FDX_s112
                                 + 0.835452 w_k1_IBM_s112
                                 + 1.17564 w_k1_INTC_s112
                                 + 1.11324 w_k1_KO_s112 + 1.21381 w_k1_MAT_s112
                                 + 1.27521 w_k1_MSFT_s112
                                 + 1.01627 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.06867 w_k1_ADM_s112
                                 + 1.0703 w_k1_AXP_s112
                                 + 0.849014 w_k1_DIS_s112
                                 + 1.03785 w_k1_FDX_s112
                                 + 0.621082 w_k1_IBM_s112
                                 + 0.98235 w_k1_INTC_s112
                                 + 0.748717 w_k1_KO_s112
                                 + 0.47803 w_k1_MAT_s112
                                 + 1.0288 w_k1_MSFT_s112
                                 + 0.847771 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.824498 w_k1_ADM_s112
                                 + 0.750546 w_k1_AXP_s112
                                 + 0.758454 w_k1_DIS_s112
                                 + 0.75889 w_k1_FDX_s112
                                 + 1.26826 w_k1_IBM_s112
                                 + 0.82456 w_k1_INTC_s112
                                 + 1.09501 w_k1_KO_s112 + 1.40743 w_k1_MAT_s112
                                 + 0.993716 w_k1_MSFT_s112
                                 + 0.95421 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.2 w_k1_ADM_s112
                                 + 1.53028 w_k1_AXP_s112
                                 + 1.46425 w_k1_DIS_s112
                                 + 1.28063 w_k1_FDX_s112
                                 + 1.25813 w_k1_IBM_s112
                                 + 1.11708 w_k1_INTC_s112 + 1.4237 w_k1_KO_s112
                                 + 1.38249 w_k1_MAT_s112
                                 + 1.23126 w_k1_MSFT_s112
                                 + 1.19985 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.595582 w_k1_ADM_s112
                                 + 0.585885 w_k1_AXP_s112
                                 + 0.582733 w_k1_DIS_s112
                                 + 0.658626 w_k1_FDX_s112
                                 + 0.856828 w_k1_IBM_s112
                                 + 0.718756 w_k1_INTC_s112
                                 + 0.813358 w_k1_KO_s112
                                 + 0.711856 w_k1_MAT_s112
                                 + 0.799109 w_k1_MSFT_s112
                                 + 0.777132 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.16873 w_k1_ADM_s113
                                 + 1.01 w_k1_AXP_s113 + 1.10735 w_k1_DIS_s113
                                 + 1.1547 w_k1_FDX_s113
                                 + 0.823984 w_k1_IBM_s113
                                 + 1.09579 w_k1_INTC_s113
                                 + 0.958912 w_k1_KO_s113
                                 + 0.925889 w_k1_MAT_s113
                                 + 1.02751 w_k1_MSFT_s113
                                 + 0.971661 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.581749 w_k1_ADM_s113
                                 + 0.570213 w_k1_AXP_s113
                                 + 0.523441 w_k1_DIS_s113
                                 + 0.666962 w_k1_FDX_s113
                                 + 0.804696 w_k1_IBM_s113
                                 + 0.81569 w_k1_INTC_s113
                                 + 0.72076 w_k1_KO_s113
                                 + 0.685167 w_k1_MAT_s113
                                 + 0.668477 w_k1_MSFT_s113
                                 + 0.780275 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.882986 w_k1_ADM_s113
                                 + 0.888439 w_k1_AXP_s113
                                 + 0.688023 w_k1_DIS_s113
                                 + 0.856108 w_k1_FDX_s113
                                 + 0.796837 w_k1_IBM_s113
                                 + 0.943945 w_k1_INTC_s113
                                 + 0.790484 w_k1_KO_s113
                                 + 0.89096 w_k1_MAT_s113
                                 + 0.764614 w_k1_MSFT_s113
                                 + 0.886974 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.670123 w_k1_ADM_s113
                                 + 0.5791 w_k1_AXP_s113
                                 + 0.793364 w_k1_DIS_s113
                                 + 0.799091 w_k1_FDX_s113
                                 + 1.00293 w_k1_IBM_s113
                                 + 0.755899 w_k1_INTC_s113
                                 + 0.934525 w_k1_KO_s113
                                 + 0.854743 w_k1_MAT_s113
                                 + 0.812923 w_k1_MSFT_s113
                                 + 0.972869 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.02976 w_k1_ADM_s113
                                 + 1.24717 w_k1_AXP_s113
                                 + 1.42588 w_k1_DIS_s113
                                 + 1.09664 w_k1_FDX_s113
                                 + 1.68972 w_k1_IBM_s113
                                 + 1.25423 w_k1_INTC_s113
                                 + 1.32159 w_k1_KO_s113 + 1.40321 w_k1_MAT_s113
                                 + 1.1426 w_k1_MSFT_s113 + 1.03981 w_k1_PG_s113
                                 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.99758 w_k1_ADM_s113
                                 + 0.759895 w_k1_AXP_s113
                                 + 1.11453 w_k1_DIS_s113
                                 + 0.957541 w_k1_FDX_s113
                                 + 0.936336 w_k1_IBM_s113
                                 + 1.12511 w_k1_INTC_s113
                                 + 1.01885 w_k1_KO_s113
                                 + 0.872347 w_k1_MAT_s113
                                 + 1.01958 w_k1_MSFT_s113
                                 + 1.00117 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.21046 w_k1_ADM_s113
                                 + 1.31382 w_k1_AXP_s113
                                 + 1.39869 w_k1_DIS_s113
                                 + 1.45061 w_k1_FDX_s113
                                 + 1.01029 w_k1_IBM_s113
                                 + 1.04703 w_k1_INTC_s113
                                 + 1.14162 w_k1_KO_s113 + 1.3746 w_k1_MAT_s113
                                 + 1.24357 w_k1_MSFT_s113
                                 + 1.26863 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.950346 w_k1_ADM_s113
                                 + 1.0475 w_k1_AXP_s113
                                 + 0.913906 w_k1_DIS_s113
                                 + 0.961884 w_k1_FDX_s113
                                 + 1.20216 w_k1_IBM_s113
                                 + 0.946155 w_k1_INTC_s113
                                 + 0.993013 w_k1_KO_s113 + 1.1818 w_k1_MAT_s113
                                 + 0.96883 w_k1_MSFT_s113
                                 + 1.05145 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.989404 w_k1_ADM_s114
                                 + 0.879111 w_k1_AXP_s114
                                 + 0.726014 w_k1_DIS_s114
                                 + 0.819092 w_k1_FDX_s114
                                 + 0.645288 w_k1_IBM_s114
                                 + 0.798833 w_k1_INTC_s114
                                 + 0.708434 w_k1_KO_s114
                                 + 0.450534 w_k1_MAT_s114
                                 + 0.769894 w_k1_MSFT_s114
                                 + 0.746126 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.683661 w_k1_ADM_s114
                                 + 0.75352 w_k1_AXP_s114
                                 + 0.842482 w_k1_DIS_s114
                                 + 0.923449 w_k1_FDX_s114
                                 + 0.750446 w_k1_IBM_s114
                                 + 1.02646 w_k1_INTC_s114
                                 + 0.822586 w_k1_KO_s114
                                 + 0.692111 w_k1_MAT_s114
                                 + 0.993142 w_k1_MSFT_s114
                                 + 0.949254 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.22236 w_k1_ADM_s114
                                 + 1.00028 w_k1_AXP_s114
                                 + 1.07845 w_k1_DIS_s114
                                 + 0.98942 w_k1_FDX_s114
                                 + 1.05821 w_k1_IBM_s114
                                 + 0.95607 w_k1_INTC_s114
                                 + 1.12143 w_k1_KO_s114
                                 + 0.749479 w_k1_MAT_s114
                                 + 1.05899 w_k1_MSFT_s114
                                 + 1.06133 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.679086 w_k1_ADM_s114
                                 + 0.774855 w_k1_AXP_s114
                                 + 0.675481 w_k1_DIS_s114
                                 + 0.727401 w_k1_FDX_s114
                                 + 1.05991 w_k1_IBM_s114
                                 + 0.757129 w_k1_INTC_s114
                                 + 0.793268 w_k1_KO_s114
                                 + 0.920063 w_k1_MAT_s114
                                 + 0.765877 w_k1_MSFT_s114
                                 + 0.910325 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.00319 w_k1_ADM_s114
                                 + 0.919378 w_k1_AXP_s114
                                 + 1.21343 w_k1_DIS_s114
                                 + 1.12915 w_k1_FDX_s114
                                 + 0.78255 w_k1_IBM_s114
                                 + 1.11102 w_k1_INTC_s114
                                 + 0.946661 w_k1_KO_s114
                                 + 0.682923 w_k1_MAT_s114
                                 + 1.08767 w_k1_MSFT_s114
                                 + 1.00476 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.26263 w_k1_ADM_s114
                                 + 1.05027 w_k1_AXP_s114
                                 + 1.12844 w_k1_DIS_s114
                                 + 1.04461 w_k1_FDX_s114
                                 + 1.42728 w_k1_IBM_s114
                                 + 1.25706 w_k1_INTC_s114
                                 + 1.26408 w_k1_KO_s114
                                 + 0.955266 w_k1_MAT_s114
                                 + 1.12553 w_k1_MSFT_s114
                                 + 1.02759 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.0299 w_k1_ADM_s114
                                 + 1.01236 w_k1_AXP_s114
                                 + 0.783386 w_k1_DIS_s114
                                 + 0.865384 w_k1_FDX_s114
                                 + 0.998307 w_k1_IBM_s114
                                 + 0.854479 w_k1_INTC_s114
                                 + 0.911783 w_k1_KO_s114
                                 + 0.940058 w_k1_MAT_s114
                                 + 0.884092 w_k1_MSFT_s114
                                 + 0.907034 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.08707 w_k1_ADM_s114
                                 + 1.45819 w_k1_AXP_s114
                                 + 1.69363 w_k1_DIS_s114
                                 + 1.40143 w_k1_FDX_s114
                                 + 1.00213 w_k1_IBM_s114
                                 + 1.12601 w_k1_INTC_s114
                                 + 1.05607 w_k1_KO_s114 + 1.28762 w_k1_MAT_s114
                                 + 1.33363 w_k1_MSFT_s114
                                 + 1.18134 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.07932 w_k1_ADM_s115
                                 + 0.988223 w_k1_AXP_s115
                                 + 0.812323 w_k1_DIS_s115
                                 + 1.09235 w_k1_FDX_s115
                                 + 0.826064 w_k1_IBM_s115
                                 + 0.952138 w_k1_INTC_s115
                                 + 0.799478 w_k1_KO_s115
                                 + 0.727695 w_k1_MAT_s115
                                 + 0.928365 w_k1_MSFT_s115
                                 + 0.919337 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.38209 w_k1_ADM_s115
                                 + 1.51207 w_k1_AXP_s115
                                 + 1.49657 w_k1_DIS_s115
                                 + 1.08464 w_k1_FDX_s115
                                 + 1.94235 w_k1_IBM_s115
                                 + 1.22434 w_k1_INTC_s115
                                 + 1.77645 w_k1_KO_s115 + 2.42755 w_k1_MAT_s115
                                 + 1.14674 w_k1_MSFT_s115
                                 + 1.18837 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.05227 w_k1_ADM_s115
                                 + 0.852991 w_k1_AXP_s115
                                 + 0.767959 w_k1_DIS_s115
                                 + 0.837505 w_k1_FDX_s115
                                 + 0.906362 w_k1_IBM_s115
                                 + 0.766266 w_k1_INTC_s115
                                 + 0.905533 w_k1_KO_s115
                                 + 1.17597 w_k1_MAT_s115
                                 + 0.904068 w_k1_MSFT_s115
                                 + 0.965035 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.982966 w_k1_ADM_s115
                                 + 0.756759 w_k1_AXP_s115
                                 + 1.0987 w_k1_DIS_s115 + 1.0415 w_k1_FDX_s115
                                 + 0.72318 w_k1_IBM_s115
                                 + 0.786974 w_k1_INTC_s115
                                 + 1.08792 w_k1_KO_s115
                                 + 0.738774 w_k1_MAT_s115
                                 + 1.10795 w_k1_MSFT_s115
                                 + 1.03389 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.947985 w_k1_ADM_s115
                                 + 0.994196 w_k1_AXP_s115
                                 + 0.972074 w_k1_DIS_s115
                                 + 0.940814 w_k1_FDX_s115
                                 + 1.43419 w_k1_IBM_s115
                                 + 0.804957 w_k1_INTC_s115
                                 + 1.22657 w_k1_KO_s115 + 1.73465 w_k1_MAT_s115
                                 + 1.01891 w_k1_MSFT_s115
                                 + 0.901818 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.55042 w_k1_ADM_s115
                                 + 1.39546 w_k1_AXP_s115
                                 + 1.62679 w_k1_DIS_s115
                                 + 1.30453 w_k1_FDX_s115
                                 + 1.34423 w_k1_IBM_s115
                                 + 0.961871 w_k1_INTC_s115
                                 + 1.31438 w_k1_KO_s115 + 1.3066 w_k1_MAT_s115
                                 + 1.23509 w_k1_MSFT_s115
                                 + 1.07944 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.00615 w_k1_ADM_s115
                                 + 1.45843 w_k1_AXP_s115
                                 + 1.29237 w_k1_DIS_s115
                                 + 1.38707 w_k1_FDX_s115
                                 + 0.877145 w_k1_IBM_s115
                                 + 1.19608 w_k1_INTC_s115
                                 + 1.00195 w_k1_KO_s115 + 1.05463 w_k1_MAT_s115
                                 + 1.23034 w_k1_MSFT_s115
                                 + 1.14511 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.09151 w_k1_ADM_s115
                                 + 0.744402 w_k1_AXP_s115
                                 + 1.37563 w_k1_DIS_s115
                                 + 1.14131 w_k1_FDX_s115
                                 + 1.19813 w_k1_IBM_s115
                                 + 1.03181 w_k1_INTC_s115
                                 + 1.43155 w_k1_KO_s115 + 1.41081 w_k1_MAT_s115
                                 + 1.34174 w_k1_MSFT_s115
                                 + 1.18092 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.91988 w_k1_ADM_s116
                                 + 0.73288 w_k1_AXP_s116
                                 + 0.516371 w_k1_DIS_s116
                                 + 0.75385 w_k1_FDX_s116
                                 + 0.692656 w_k1_IBM_s116
                                 + 0.726095 w_k1_INTC_s116
                                 + 0.676079 w_k1_KO_s116
                                 + 0.604134 w_k1_MAT_s116
                                 + 0.768301 w_k1_MSFT_s116
                                 + 0.76135 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.963845 w_k1_ADM_s116
                                 + 1.03991 w_k1_AXP_s116
                                 + 0.923318 w_k1_DIS_s116
                                 + 0.920006 w_k1_FDX_s116
                                 + 1.30934 w_k1_IBM_s116
                                 + 0.799811 w_k1_INTC_s116
                                 + 1.05448 w_k1_KO_s116 + 1.15031 w_k1_MAT_s116
                                 + 0.886056 w_k1_MSFT_s116
                                 + 0.923896 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.20683 w_k1_ADM_s116
                                 + 0.97384 w_k1_AXP_s116
                                 + 1.08702 w_k1_DIS_s116
                                 + 1.07177 w_k1_FDX_s116
                                 + 1.02908 w_k1_IBM_s116
                                 + 1.00814 w_k1_INTC_s116
                                 + 1.02884 w_k1_KO_s116 + 1.51541 w_k1_MAT_s116
                                 + 1.19602 w_k1_MSFT_s116
                                 + 1.04476 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.23086 w_k1_ADM_s116
                                 + 1.20771 w_k1_AXP_s116
                                 + 1.16792 w_k1_DIS_s116
                                 + 1.17212 w_k1_FDX_s116
                                 + 0.79508 w_k1_IBM_s116
                                 + 1.00221 w_k1_INTC_s116
                                 + 0.868873 w_k1_KO_s116
                                 + 1.03884 w_k1_MAT_s116
                                 + 1.14443 w_k1_MSFT_s116
                                 + 0.911709 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.38413 w_k1_ADM_s116
                                 + 1.34131 w_k1_AXP_s116 + 1.4747 w_k1_DIS_s116
                                 + 1.23425 w_k1_FDX_s116 + 1.4583 w_k1_IBM_s116
                                 + 1.18969 w_k1_INTC_s116 + 1.3755 w_k1_KO_s116
                                 + 1.01341 w_k1_MAT_s116
                                 + 1.25068 w_k1_MSFT_s116
                                 + 1.19809 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.01147 w_k1_ADM_s116
                                 + 0.798516 w_k1_AXP_s116
                                 + 0.786691 w_k1_DIS_s116
                                 + 0.894565 w_k1_FDX_s116
                                 + 0.843009 w_k1_IBM_s116
                                 + 0.797392 w_k1_INTC_s116
                                 + 0.889038 w_k1_KO_s116
                                 + 1.12379 w_k1_MAT_s116
                                 + 0.91416 w_k1_MSFT_s116
                                 + 0.843466 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.805767 w_k1_ADM_s116
                                 + 0.5404 w_k1_AXP_s116
                                 + 0.598907 w_k1_DIS_s116
                                 + 0.663703 w_k1_FDX_s116
                                 + 0.871361 w_k1_IBM_s116
                                 + 0.67082 w_k1_INTC_s116
                                 + 0.777043 w_k1_KO_s116
                                 + 0.906487 w_k1_MAT_s116
                                 + 0.763261 w_k1_MSFT_s116
                                 + 0.744715 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.965682 w_k1_ADM_s116
                                 + 0.8516 w_k1_AXP_s116
                                 + 0.821037 w_k1_DIS_s116
                                 + 1.01612 w_k1_FDX_s116 + 0.6067 w_k1_IBM_s116
                                 + 0.879138 w_k1_INTC_s116
                                 + 0.87227 w_k1_KO_s116
                                 + 0.937291 w_k1_MAT_s116
                                 + 1.03923 w_k1_MSFT_s116
                                 + 0.901736 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.06419 w_k1_ADM_s117
                                 + 1.04222 w_k1_AXP_s117
                                 + 1.21823 w_k1_DIS_s117
                                 + 1.08567 w_k1_FDX_s117
                                 + 0.979267 w_k1_IBM_s117
                                 + 1.09809 w_k1_INTC_s117
                                 + 1.10351 w_k1_KO_s117 + 0.82553 w_k1_MAT_s117
                                 + 1.13986 w_k1_MSFT_s117
                                 + 1.07491 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.971914 w_k1_ADM_s117
                                 + 0.735401 w_k1_AXP_s117
                                 + 0.836613 w_k1_DIS_s117
                                 + 0.926161 w_k1_FDX_s117
                                 + 0.685352 w_k1_IBM_s117
                                 + 0.984975 w_k1_INTC_s117
                                 + 0.812928 w_k1_KO_s117
                                 + 0.525498 w_k1_MAT_s117
                                 + 0.912061 w_k1_MSFT_s117
                                 + 0.986594 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.23719 w_k1_ADM_s117
                                 + 1.38089 w_k1_AXP_s117 + 1.4213 w_k1_DIS_s117
                                 + 1.25709 w_k1_FDX_s117
                                 + 1.23186 w_k1_IBM_s117
                                 + 1.08679 w_k1_INTC_s117
                                 + 1.24981 w_k1_KO_s117 + 1.25977 w_k1_MAT_s117
                                 + 1.24538 w_k1_MSFT_s117
                                 + 1.19833 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.958826 w_k1_ADM_s117
                                 + 1.03645 w_k1_AXP_s117
                                 + 0.831847 w_k1_DIS_s117
                                 + 0.896736 w_k1_FDX_s117
                                 + 0.967952 w_k1_IBM_s117
                                 + 1.06189 w_k1_INTC_s117
                                 + 0.956909 w_k1_KO_s117
                                 + 0.882638 w_k1_MAT_s117
                                 + 0.981113 w_k1_MSFT_s117
                                 + 0.984145 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.06337 w_k1_ADM_s117
                                 + 1.01839 w_k1_AXP_s117
                                 + 1.06314 w_k1_DIS_s117
                                 + 0.877217 w_k1_FDX_s117
                                 + 1.42029 w_k1_IBM_s117
                                 + 1.08707 w_k1_INTC_s117
                                 + 1.39185 w_k1_KO_s117 + 1.36925 w_k1_MAT_s117
                                 + 1.17565 w_k1_MSFT_s117
                                 + 1.18168 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.782696 w_k1_ADM_s117
                                 + 0.755266 w_k1_AXP_s117
                                 + 0.643477 w_k1_DIS_s117
                                 + 0.751235 w_k1_FDX_s117
                                 + 0.771659 w_k1_IBM_s117
                                 + 0.776757 w_k1_INTC_s117
                                 + 0.717203 w_k1_KO_s117
                                 + 0.721836 w_k1_MAT_s117
                                 + 0.735181 w_k1_MSFT_s117
                                 + 0.857804 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.891175 w_k1_ADM_s117
                                 + 0.965885 w_k1_AXP_s117
                                 + 0.83163 w_k1_DIS_s117 + 0.8571 w_k1_FDX_s117
                                 + 1.1962 w_k1_IBM_s117
                                 + 1.06471 w_k1_INTC_s117
                                 + 0.961092 w_k1_KO_s117
                                 + 0.96616 w_k1_MAT_s117
                                 + 0.90454 w_k1_MSFT_s117
                                 + 1.00206 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.911884 w_k1_ADM_s117
                                 + 1.20169 w_k1_AXP_s117
                                 + 0.882667 w_k1_DIS_s117
                                 + 1.04864 w_k1_FDX_s117 + 0.7001 w_k1_IBM_s117
                                 + 1.10336 w_k1_INTC_s117
                                 + 0.805143 w_k1_KO_s117
                                 + 0.653577 w_k1_MAT_s117
                                 + 0.96157 w_k1_MSFT_s117
                                 + 0.979536 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0560322671654488 w_k1_ADM_s000
                                 + 0.013315337842814 w_k1_ADM_s001
                                 + 0.015901341547059 w_k1_ADM_s002
                                 + 0.0379042038999147 w_k1_ADM_s003
                                 + 0.0416307680293915 w_k1_ADM_s004
                                 + 0.0508495701397522 w_k1_ADM_s005
                                 + 0.0281079460204136 w_k1_ADM_s006
                                 + 0.0387325996092679 w_k1_ADM_s007
                                 + 0.00923227568850632 w_k1_ADM_s010
                                 + 0.0191707872907366 w_k1_ADM_s011
                                 + 0.00605110440558278 w_k1_ADM_s012
                                 + 0.0110609836961097 w_k1_ADM_s013
                                 + 0.0129177933335628 w_k1_ADM_s014
                                 + 0.00676179164560862 w_k1_ADM_s015
                                 + 0.0104738925969554 w_k1_ADM_s016
                                 + 0.00359159182768374 w_k1_ADM_s017
                                 + 0.0436106821117692 w_k1_ADM_s100
                                 + 0.0332625710761056 w_k1_ADM_s101
                                 + 0.020385048851186 w_k1_ADM_s102
                                 + 0.0116942549694749 w_k1_ADM_s103
                                 + 0.0246716032094464 w_k1_ADM_s104
                                 + 0.011285825270866 w_k1_ADM_s105
                                 + 0.0185425970047722 w_k1_ADM_s106
                                 + 0.0501243732163892 w_k1_ADM_s107
                                 + 0.0698345461759828 w_k1_ADM_s110
                                 + 0.0305015039948872 w_k1_ADM_s111
                                 + 0.0934223373578607 w_k1_ADM_s112
                                 + 0.082115993618814 w_k1_ADM_s113
                                 + 0.0232738282170661 w_k1_ADM_s114
                                 + 0.0430992228878396 w_k1_ADM_s115
                                 + 0.0436138619506603 w_k1_ADM_s116
                                 + 0.0536620554509093 w_k1_ADM_s117
                                 + 0.0573842681523542 w_k1_AXP_s000
                                 + 0.0138376347676673 w_k1_AXP_s001
                                 + 0.0170391035619728 w_k1_AXP_s002
                                 + 0.0363509113855665 w_k1_AXP_s003
                                 + 0.0385877424759381 w_k1_AXP_s004
                                 + 0.0632808472091597 w_k1_AXP_s005
                                 + 0.0310869666451194 w_k1_AXP_s006
                                 + 0.0401037650560412 w_k1_AXP_s007
                                 + 0.00813731990041054 w_k1_AXP_s010
                                 + 0.0193511861235456 w_k1_AXP_s011
                                 + 0.00582711291067969 w_k1_AXP_s012
                                 + 0.0120583397538482 w_k1_AXP_s013
                                 + 0.0120757452246543 w_k1_AXP_s014
                                 + 0.00644721236811143 w_k1_AXP_s015
                                 + 0.0115633330757467 w_k1_AXP_s016
                                 + 0.00419213671120085 w_k1_AXP_s017
                                 + 0.0410975828501592 w_k1_AXP_s100
                                 + 0.0340276271422963 w_k1_AXP_s101
                                 + 0.0192293138907462 w_k1_AXP_s102
                                 + 0.0122220703617776 w_k1_AXP_s103
                                 + 0.0245602738827248 w_k1_AXP_s104
                                 + 0.0109789944348596 w_k1_AXP_s105
                                 + 0.0204010406491788 w_k1_AXP_s106
                                 + 0.0504663238009913 w_k1_AXP_s107
                                 + 0.0707167085762771 w_k1_AXP_s110
                                 + 0.0279898114317353 w_k1_AXP_s111
                                 + 0.102890041917444 w_k1_AXP_s112
                                 + 0.0801484795843951 w_k1_AXP_s113
                                 + 0.0222144144665122 w_k1_AXP_s114
                                 + 0.0396860075979208 w_k1_AXP_s115
                                 + 0.0408209168595492 w_k1_AXP_s116
                                 + 0.0554175762847718 w_k1_AXP_s117
                                 + 0.0562458561657746 w_k1_DIS_s000
                                 + 0.0144585476009102 w_k1_DIS_s001
                                 + 0.0175722160395591 w_k1_DIS_s002
                                 + 0.0353277559151589 w_k1_DIS_s003
                                 + 0.0415958530484943 w_k1_DIS_s004
                                 + 0.0562845308882899 w_k1_DIS_s005
                                 + 0.0293245227801784 w_k1_DIS_s006
                                 + 0.0358210991853756 w_k1_DIS_s007
                                 + 0.00837946929818104 w_k1_DIS_s010
                                 + 0.0192267775759878 w_k1_DIS_s011
                                 + 0.00551100655436465 w_k1_DIS_s012
                                 + 0.0119613983827062 w_k1_DIS_s013
                                 + 0.0111806095586656 w_k1_DIS_s014
                                 + 0.00667837751550743 w_k1_DIS_s015
                                 + 0.0118618630738612 w_k1_DIS_s016
                                 + 0.00382713485038601 w_k1_DIS_s017
                                 + 0.0425037001591223 w_k1_DIS_s100
                                 + 0.0338270747756606 w_k1_DIS_s101
                                 + 0.0222642747879268 w_k1_DIS_s102
                                 + 0.013199873351368 w_k1_DIS_s103
                                 + 0.024757060841993 w_k1_DIS_s104
                                 + 0.0105322641926857 w_k1_DIS_s105
                                 + 0.0195763582226315 w_k1_DIS_s106
                                 + 0.0538644515546307 w_k1_DIS_s107
                                 + 0.0673503114015495 w_k1_DIS_s110
                                 + 0.0298085176220065 w_k1_DIS_s111
                                 + 0.101405058335796 w_k1_DIS_s112
                                 + 0.0845654093669252 w_k1_DIS_s113
                                 + 0.0233600055260698 w_k1_DIS_s114
                                 + 0.0412334066815115 w_k1_DIS_s115
                                 + 0.0400563362106409 w_k1_DIS_s116
                                 + 0.0537699312306252 w_k1_DIS_s117
                                 + 0.0564347537429571 w_k1_FDX_s000
                                 + 0.0141069695421097 w_k1_FDX_s001
                                 + 0.0171732307981184 w_k1_FDX_s002
                                 + 0.0337791006546301 w_k1_FDX_s003
                                 + 0.0389638437352076 w_k1_FDX_s004
                                 + 0.0565492856304577 w_k1_FDX_s005
                                 + 0.0272645130866003 w_k1_FDX_s006
                                 + 0.039831620166731 w_k1_FDX_s007
                                 + 0.00889723694386194 w_k1_FDX_s010
                                 + 0.0196378323825549 w_k1_FDX_s011
                                 + 0.00554922588135883 w_k1_FDX_s012
                                 + 0.0120544239251296 w_k1_FDX_s013
                                 + 0.0102271346184048 w_k1_FDX_s014
                                 + 0.0069919717907482 w_k1_FDX_s015
                                 + 0.0116552264856887 w_k1_FDX_s016
                                 + 0.00389719065031337 w_k1_FDX_s017
                                 + 0.0412598803752299 w_k1_FDX_s100
                                 + 0.0342352986981602 w_k1_FDX_s101
                                 + 0.0207149616145576 w_k1_FDX_s102
                                 + 0.0121879187331056 w_k1_FDX_s103
                                 + 0.0232373662410295 w_k1_FDX_s104
                                 + 0.0105622120834538 w_k1_FDX_s105
                                 + 0.0188285092479179 w_k1_FDX_s106
                                 + 0.0509035372346908 w_k1_FDX_s107
                                 + 0.0671697253769194 w_k1_FDX_s110
                                 + 0.0302240988727564 w_k1_FDX_s111
                                 + 0.1001402456853 w_k1_FDX_s112
                                 + 0.085904758138506 w_k1_FDX_s113
                                 + 0.0222736508915454 w_k1_FDX_s114
                                 + 0.0401750463360067 w_k1_FDX_s115
                                 + 0.0400460889864503 w_k1_FDX_s116
                                 + 0.0520436222042302 w_k1_FDX_s117
                                 + 0.0561814137905569 w_k1_IBM_s000
                                 + 0.014897882174312 w_k1_IBM_s001
                                 + 0.0179977183885543 w_k1_IBM_s002
                                 + 0.0323613193510976 w_k1_IBM_s003
                                 + 0.0374508012953851 w_k1_IBM_s004
                                 + 0.0589012960988338 w_k1_IBM_s005
                                 + 0.0286611244080744 w_k1_IBM_s006
                                 + 0.0403121508498719 w_k1_IBM_s007
                                 + 0.00883886265980787 w_k1_IBM_s010
                                 + 0.0196459972496658 w_k1_IBM_s011
                                 + 0.00553691453840672 w_k1_IBM_s012
                                 + 0.0112555621707128 w_k1_IBM_s013
                                 + 0.0115376830639667 w_k1_IBM_s014
                                 + 0.00659333732893478 w_k1_IBM_s015
                                 + 0.0116489157156701 w_k1_IBM_s016
                                 + 0.0040293664453807 w_k1_IBM_s017
                                 + 0.0425304974719854 w_k1_IBM_s100
                                 + 0.0350720650388753 w_k1_IBM_s101
                                 + 0.0206021951929642 w_k1_IBM_s102
                                 + 0.0124045209746464 w_k1_IBM_s103
                                 + 0.0250198015718581 w_k1_IBM_s104
                                 + 0.0107796107451548 w_k1_IBM_s105
                                 + 0.0202562269490036 w_k1_IBM_s106
                                 + 0.0502919126607513 w_k1_IBM_s107
                                 + 0.0685236217868187 w_k1_IBM_s110
                                 + 0.0298467902381361 w_k1_IBM_s111
                                 + 0.104622760874838 w_k1_IBM_s112
                                 + 0.0863857003966657 w_k1_IBM_s113
                                 + 0.0224647254960021 w_k1_IBM_s114
                                 + 0.0418914596316014 w_k1_IBM_s115
                                 + 0.0410668162789209 w_k1_IBM_s116
                                 + 0.055125992817299 w_k1_IBM_s117
                                 + 0.0530315733865646 w_k1_INTC_s000
                                 + 0.0151058119367807 w_k1_INTC_s001
                                 + 0.0154983406815308 w_k1_INTC_s002
                                 + 0.0329527136495954 w_k1_INTC_s003
                                 + 0.0353787545781038 w_k1_INTC_s004
                                 + 0.0577840853772091 w_k1_INTC_s005
                                 + 0.0283797952308465 w_k1_INTC_s006
                                 + 0.0370864362380977 w_k1_INTC_s007
                                 + 0.00947765599620892 w_k1_INTC_s010
                                 + 0.0202079946866932 w_k1_INTC_s011
                                 + 0.00534482596814961 w_k1_INTC_s012
                                 + 0.0125100151554273 w_k1_INTC_s013
                                 + 0.00842287305227401 w_k1_INTC_s014
                                 + 0.00733754879045127 w_k1_INTC_s015
                                 + 0.0111554749099818 w_k1_INTC_s016
                                 + 0.0051868956402835 w_k1_INTC_s017
                                 + 0.0456916891712644 w_k1_INTC_s100
                                 + 0.0397633081124646 w_k1_INTC_s101
                                 + 0.0231022518331777 w_k1_INTC_s102
                                 + 0.0124586606726718 w_k1_INTC_s103
                                 + 0.0237215960997088 w_k1_INTC_s104
                                 + 0.0113632761096854 w_k1_INTC_s105
                                 + 0.020064470632329 w_k1_INTC_s106
                                 + 0.0498270272850427 w_k1_INTC_s107
                                 + 0.0645435069214264 w_k1_INTC_s110
                                 + 0.0305058743788423 w_k1_INTC_s111
                                 + 0.100483321819195 w_k1_INTC_s112
                                 + 0.0849394096881177 w_k1_INTC_s113
                                 + 0.0223782752338929 w_k1_INTC_s114
                                 + 0.0353561722931562 w_k1_INTC_s115
                                 + 0.0362115795554295 w_k1_INTC_s116
                                 + 0.055381134089658 w_k1_INTC_s117
                                 + 0.057591181963301 w_k1_KO_s000
                                 + 0.0148133015628939 w_k1_KO_s001
                                 + 0.0175493659690165 w_k1_KO_s002
                                 + 0.0338923474638675 w_k1_KO_s003
                                 + 0.0382940341817096 w_k1_KO_s004
                                 + 0.0564015206500367 w_k1_KO_s005
                                 + 0.026093570964443 w_k1_KO_s006
                                 + 0.0392710963000742 w_k1_KO_s007
                                 + 0.00897597625684814 w_k1_KO_s010
                                 + 0.0181981429490593 w_k1_KO_s011
                                 + 0.00483111146534999 w_k1_KO_s012
                                 + 0.0120561335507112 w_k1_KO_s013
                                 + 0.0111389230505579 w_k1_KO_s014
                                 + 0.00636260904226114 w_k1_KO_s015
                                 + 0.0116510184429985 w_k1_KO_s016
                                 + 0.00448116959218928 w_k1_KO_s017
                                 + 0.0409918557389587 w_k1_KO_s100
                                 + 0.0342243214187236 w_k1_KO_s101
                                 + 0.0200883145488464 w_k1_KO_s102
                                 + 0.012590291608281 w_k1_KO_s103
                                 + 0.0229180568514528 w_k1_KO_s104
                                 + 0.0111207558845584 w_k1_KO_s105
                                 + 0.0186246362520021 w_k1_KO_s106
                                 + 0.0467936215222154 w_k1_KO_s107
                                 + 0.0668122655004982 w_k1_KO_s110
                                 + 0.0337736264411645 w_k1_KO_s111
                                 + 0.112871367604736 w_k1_KO_s112
                                 + 0.0834884298672974 w_k1_KO_s113
                                 + 0.0221182999367782 w_k1_KO_s114
                                 + 0.0430892525666651 w_k1_KO_s115
                                 + 0.0394042679303125 w_k1_KO_s116
                                 + 0.0554672483871411 w_k1_KO_s117
                                 + 0.0583283180526429 w_k1_MAT_s000
                                 + 0.0161540117459057 w_k1_MAT_s001
                                 + 0.019833098195116 w_k1_MAT_s002
                                 + 0.0399181322838482 w_k1_MAT_s003
                                 + 0.0346577474721925 w_k1_MAT_s004
                                 + 0.0697492531111538 w_k1_MAT_s005
                                 + 0.0283701320260496 w_k1_MAT_s006
                                 + 0.0477615954149269 w_k1_MAT_s007
                                 + 0.0093601470518492 w_k1_MAT_s010
                                 + 0.0188273886511438 w_k1_MAT_s011
                                 + 0.00451603067284588 w_k1_MAT_s012
                                 + 0.0116283930895613 w_k1_MAT_s013
                                 + 0.0114719926500563 w_k1_MAT_s014
                                 + 0.0056110673353462 w_k1_MAT_s015
                                 + 0.0126726930206585 w_k1_MAT_s016
                                 + 0.00621471038226184 w_k1_MAT_s017
                                 + 0.0394195749872029 w_k1_MAT_s100
                                 + 0.0298129811535245 w_k1_MAT_s101
                                 + 0.0181278963299285 w_k1_MAT_s102
                                 + 0.0124208372963685 w_k1_MAT_s103
                                 + 0.0231794309716892 w_k1_MAT_s104
                                 + 0.0116833545286786 w_k1_MAT_s105
                                 + 0.0178264950902807 w_k1_MAT_s106
                                 + 0.0536910971389217 w_k1_MAT_s107
                                 + 0.0682779467357175 w_k1_MAT_s110
                                 + 0.0281586514391003 w_k1_MAT_s111
                                 + 0.115830130544129 w_k1_MAT_s112
                                 + 0.087375590759379 w_k1_MAT_s113
                                 + 0.0187552603243577 w_k1_MAT_s114
                                 + 0.0492845325390251 w_k1_MAT_s115
                                 + 0.0412313143388368 w_k1_MAT_s116
                                 + 0.0505162390258678 w_k1_MAT_s117
                                 + 0.0557517137423152 w_k1_MSFT_s000
                                 + 0.0145674981196152 w_k1_MSFT_s001
                                 + 0.0167137115358611 w_k1_MSFT_s002
                                 + 0.0312375387446402 w_k1_MSFT_s003
                                 + 0.0367755044508515 w_k1_MSFT_s004
                                 + 0.057302368398615 w_k1_MSFT_s005
                                 + 0.0263543688682809 w_k1_MSFT_s006
                                 + 0.0401849179980586 w_k1_MSFT_s007
                                 + 0.00965645717838255 w_k1_MSFT_s010
                                 + 0.0189854978777506 w_k1_MSFT_s011
                                 + 0.00527948679380991 w_k1_MSFT_s012
                                 + 0.011156430374745 w_k1_MSFT_s013
                                 + 0.00953912186550968 w_k1_MSFT_s014
                                 + 0.00655643715442412 w_k1_MSFT_s015
                                 + 0.0111246280385493 w_k1_MSFT_s016
                                 + 0.00484913002810159 w_k1_MSFT_s017
                                 + 0.0402645564451315 w_k1_MSFT_s100
                                 + 0.0350121726489138 w_k1_MSFT_s101
                                 + 0.0210754395405865 w_k1_MSFT_s102
                                 + 0.0110894020304882 w_k1_MSFT_s103
                                 + 0.0232053685447822 w_k1_MSFT_s104
                                 + 0.0119659329625922 w_k1_MSFT_s105
                                 + 0.0185035670689333 w_k1_MSFT_s106
                                 + 0.0487414019436703 w_k1_MSFT_s107
                                 + 0.0697883104990735 w_k1_MSFT_s110
                                 + 0.0306030123742387 w_k1_MSFT_s111
                                 + 0.110516019022223 w_k1_MSFT_s112
                                 + 0.082129277580922 w_k1_MSFT_s113
                                 + 0.0226813436533623 w_k1_MSFT_s114
                                 + 0.0411082089568176 w_k1_MSFT_s115
                                 + 0.0402790203395122 w_k1_MSFT_s116
                                 + 0.0551428169923375 w_k1_MSFT_s117
                                 + 0.0608542810316746 w_k1_PG_s000
                                 + 0.0148224431674814 w_k1_PG_s001
                                 + 0.0169578270065511 w_k1_PG_s002
                                 + 0.033239300115304 w_k1_PG_s003
                                 + 0.0387153088963432 w_k1_PG_s004
                                 + 0.0541509211583092 w_k1_PG_s005
                                 + 0.0258864539348078 w_k1_PG_s006
                                 + 0.0376919276074511 w_k1_PG_s007
                                 + 0.00869728241645986 w_k1_PG_s010
                                 + 0.0190737318487214 w_k1_PG_s011
                                 + 0.0048910371105919 w_k1_PG_s012
                                 + 0.0106250603352044 w_k1_PG_s013
                                 + 0.0108815937472949 w_k1_PG_s014
                                 + 0.0066530343446443 w_k1_PG_s015
                                 + 0.0119581514336974 w_k1_PG_s016
                                 + 0.00408288680009939 w_k1_PG_s017
                                 + 0.0407993893345684 w_k1_PG_s100
                                 + 0.0345494260407446 w_k1_PG_s101
                                 + 0.0191939831061167 w_k1_PG_s102
                                 + 0.0123210631749749 w_k1_PG_s103
                                 + 0.0236317662135421 w_k1_PG_s104
                                 + 0.0102979707663938 w_k1_PG_s105
                                 + 0.0185596381532297 w_k1_PG_s106
                                 + 0.0505838869916581 w_k1_PG_s107
                                 + 0.0715707928603789 w_k1_PG_s110
                                 + 0.0292309789984562 w_k1_PG_s111
                                 + 0.10034291968327 w_k1_PG_s112
                                 + 0.085836177630634 w_k1_PG_s113
                                 + 0.0218954668112113 w_k1_PG_s114
                                 + 0.0399905448557252 w_k1_PG_s115
                                 + 0.0372579854696537 w_k1_PG_s116
                                 + 0.0558251895764603 w_k1_PG_s117 >= 1.08
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

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
 w_k1_ADM_s0_return_constraint:  1.27283 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.952664 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.823296 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.11897 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.956383 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.12254 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.928815 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.58441 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.69219 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.955992 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.29998 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.14836 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.939072 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.984157 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.1032 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.856814 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.3094 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.807027 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.637986 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.848263 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.727963 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.971699 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.788032 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.04591 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.842322 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.960337 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.90778 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.26732 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.684653 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.10368 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.725818 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.19339 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.640299 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.981604 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.882413 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.04621 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.42311 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.05665 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.32437 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.911791 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.898921 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.30581 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.857743 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.27261 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.989969 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.77385 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.704312 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.621405 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.17963 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.02394 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.875077 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.872177 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.42812 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.850387 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.45552 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.722572 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.650794 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.899572 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.878218 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.03153 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.63012 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.907337 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.972624 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.20183 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.01065 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.58588 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.678584 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.30809 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.626752 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.02453 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.65891 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.11972 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.879729 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.11202 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.3509 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.50636 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.3436 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.925156 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.05917 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.06304 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.872831 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.21942 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.794209 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.99269 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.501017 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.555203 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.19355 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.950386 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.779242 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.737936 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.52595 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.865533 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.56157 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.687234 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.561955 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.996576 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.835922 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.32761 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.598476 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.12111 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.831055 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.06646 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.802167 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.67963 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.594102 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.14959 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.647302 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.16983 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.602559 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.24259 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.79181 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.10495 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.89304 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.57111 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.2237 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.923984 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.998885 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.1104 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.948708 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.087 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.744705 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.50563 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.638168 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.764087 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.10184 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.13604 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.853252 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.875511 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.42413 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.891598 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.24917 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.940888 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.709305 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.959866 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.757815 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.13736 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.738497 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.0855 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.854032 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.889249 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.977361 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.47585 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.716382 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.16869 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.800801 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.20117 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.6719 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.00427 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.851735 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.07899 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.57664 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.26267 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.14439 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.996194 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.949785 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.17428 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.777277 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.31551 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.44496 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.49491 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.792982 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.494893 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.31528 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.67554 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.93311 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.710017 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.909536 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.861413 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.64165 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.475616 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.683122 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.09508 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.57911 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.14294 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.627835 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.910685 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.23237 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.72445 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.712823 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.29217 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.943159 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.04279 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.708966 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.823577 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.02909 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.90047 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.0739 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.19523 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.14439 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.57266 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.14004 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.969096 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.00172 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.19436 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.977672 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.10233 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.875301 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.45136 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.681258 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.773438 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.06405 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.992946 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.907892 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.83986 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.25847 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.816994 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.986746 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.828399 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.696903 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.00911 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.851649 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.00774 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.705036 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.950335 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.806866 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.865114 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.96521 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.13425 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.695072 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.14159 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.883313 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.07134 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.789995 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.10433 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.909206 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.92307 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.21919 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.29689 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.19806 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.931266 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.00731 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.12278 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.916635 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.24738 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.06227 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.54068 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.691683 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.573214 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.13933 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.806799 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.835163 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.772584 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.01546 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.723655 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.42678 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.582762 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.645013 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.023 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.0435 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.10884 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.670995 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.979265 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.03716 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.16634 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.723457 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.21305 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.755122 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.11056 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.710976 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.935694 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.817787 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.40316 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.943503 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.20319 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.4274 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.48375 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.30362 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.02466 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.04191 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.18097 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.635361 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.06693 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.17236 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.30973 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.636998 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.250685 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.05052 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.601405 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.666905 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.600466 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.954276 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.927809 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.86509 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.505993 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.687723 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.05267 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.2678 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.03214 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.69952 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.13719 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.09248 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 2.10719 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.831414 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.44219 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.876833 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.18702 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.779293 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.910295 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.12204 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.804 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.22738 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.572 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.68414 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 2.28317 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.14897 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.93094 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.07917 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.02462 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.0053 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.09367 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.78807 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.43788 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.679657 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.690166 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.00456 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.01543 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.798434 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.827564 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.4902 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.887551 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.07936 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.920824 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.780907 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.05491 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.830682 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.33298 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.698516 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.04215 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.832568 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.964917 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.870421 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.29459 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.6957 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.05089 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.857065 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.14816 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.744138 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.13707 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.837343 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.13026 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.44126 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.3202 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.1173 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.935392 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.99944 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.10644 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.974263 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.10716 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.955248 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.33327 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.768794 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.743187 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.10684 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.04788 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.883418 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.884823 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.13124 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.871624 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.19364 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.795341 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.72404 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.941207 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.963804 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.06494 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.749552 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.01037 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.948224 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.999458 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.907634 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.2244 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.783405 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.10271 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.803841 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.08974 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.893034 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.05809 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.909235 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.1143 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.36727 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.2437 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.157999551233047 z_k1_s0000
                                 - 0.348998547709719 z_k1_s0001
                                 - 0.276998135473585 z_k1_s0002
                                 - 0.3119985703655 z_k1_s0003
                                 - 0.628997202858415 z_k1_s0004
                                 - 0.182999379471467 z_k1_s0005
                                 - 0.269998530893864 z_k1_s0006
                                 - 0.233999289573121 z_k1_s0007
                                 - 0.446000334899039 z_k1_s0010
                                 - 0.119999842487855 z_k1_s0011
                                 - 0.137999775721072 z_k1_s0012
                                 - 0.296999221897787 z_k1_s0013
                                 - 0.322999676689719 z_k1_s0014
                                 - 0.247000006527206 z_k1_s0015
                                 - 0.391999888099966 z_k1_s0016
                                 - 0.193999759735352 z_k1_s0017
                                 - 0.177999987613848 z_k1_s0020
                                 - 0.218999327295836 z_k1_s0021
                                 - 0.341000852163413 z_k1_s0022
                                 - 0.195000652633 z_k1_s0023
                                 - 0.140000044864812 z_k1_s0024
                                 - 0.139000051487205 z_k1_s0025
                                 - 0.518000068799827 z_k1_s0026
                                 - 0.213999360407799 z_k1_s0027
                                 - 0.455999190177237 z_k1_s0030
                                 - 0.301999588485779 z_k1_s0031
                                 - 0.10499995669868 z_k1_s0032
                                 - 0.202999246684237 z_k1_s0033
                                 - 0.125999948038416 z_k1_s0034
                                 - 0.210999291213651 z_k1_s0035
                                 - 0.110999906395756 z_k1_s0036
                                 - 0.160000053588442 z_k1_s0037
                                 - 1.84099854068156 z_k1_s0040
                                 - 1.35800110179588 z_k1_s0041
                                 - 0.905998707972473 z_k1_s0042
                                 - 0.930997112055344 z_k1_s0043
                                 - 0.834998776376391 z_k1_s0044
                                 - 0.41500029178003 z_k1_s0045
                                 - 0.452000161823043 z_k1_s0046
                                 - 0.701999826654303 z_k1_s0047
                                 - 0.69099795081161 z_k1_s0050
                                 - 0.20299958501509 z_k1_s0051
                                 - 0.320000051470262 z_k1_s0052
                                 - 0.264999478212751 z_k1_s0053
                                 - 0.239999415804286 z_k1_s0054
                                 - 0.394997436102809 z_k1_s0055
                                 - 0.576997815700627 z_k1_s0056
                                 - 0.422998452653875 z_k1_s0057
                                 - 0.464001918294442 z_k1_s0060
                                 - 0.322002013214802 z_k1_s0061
                                 - 0.391999903037198 z_k1_s0062
                                 - 0.612002479646842 z_k1_s0063
                                 - 0.3200008177879 z_k1_s0064
                                 - 0.141000479369366 z_k1_s0065
                                 - 0.454001801175821 z_k1_s0066
                                 - 0.225000584163392 z_k1_s0067
                                 - 0.441999588549357 z_k1_s0070
                                 - 0.146999927971834 z_k1_s0071
                                 - 0.112999959621884 z_k1_s0072
                                 - 0.25200050612304 z_k1_s0073
                                 - 0.132000025264489 z_k1_s0074
                                 - 0.132000025264489 z_k1_s0075
                                 - 0.141999960402719 z_k1_s0076
                                 - 0.213999430437985 z_k1_s0077
                                 - 0.179000827859136 z_k1_s0100
                                 - 0.164000383049015 z_k1_s0101
                                 - 0.124000375890879 z_k1_s0102
                                 - 0.0650001590072451 z_k1_s0103
                                 - 0.275001080839101 z_k1_s0104
                                 - 0.143999789968854 z_k1_s0105
                                 - 0.064000129353237 z_k1_s0106
                                 - 0.164000383049015 z_k1_s0107
                                 - 0.429002103412085 z_k1_s0110
                                 - 0.155000437393139 z_k1_s0111
                                 - 0.152000431162311 z_k1_s0112
                                 - 0.409000907203088 z_k1_s0113
                                 - 0.18300020688 z_k1_s0114
                                 - 0.0910002467352965 z_k1_s0115
                                 - 0.140000406238998 z_k1_s0116
                                 - 0.173000474778108 z_k1_s0117
                                 - 0.081000219464581 z_k1_s0120
                                 - 0.0620001611037326 z_k1_s0121
                                 - 0.101000256449611 z_k1_s0122
                                 - 0.0780001906917563 z_k1_s0123
                                 - 0.0890002342585929 z_k1_s0124
                                 - 0.162000438929162 z_k1_s0125
                                 - 0.0790002622162185 z_k1_s0126
                                 - 0.277000767718434 z_k1_s0127
                                 - 0.0350000464814035 z_k1_s0130
                                 - 0.0610000525761244 z_k1_s0131
                                 - 0.12300003361734 z_k1_s0132
                                 - 0.048000049528764 z_k1_s0133
                                 - 0.12000012382191 z_k1_s0134
                                 - 0.036000037146573 z_k1_s0135
                                 - 0.124000210882745 z_k1_s0136
                                 - 0.0750002328888904 z_k1_s0137
                                 - 0.0570000206514294 z_k1_s0140
                                 - 0.0650000222393484 z_k1_s0141
                                 - 0.17600003493422 z_k1_s0142
                                 - 0.0819997361509924 z_k1_s0143
                                 - 0.15800031148658 z_k1_s0144
                                 - 0.103000160507047 z_k1_s0145
                                 - 0.0650000222393484 z_k1_s0146
                                 - 0.0410000174755912 z_k1_s0147
                                 - 0.0970004887831992 z_k1_s0150
                                 - 0.0990001400564922 z_k1_s0151
                                 - 0.0520001087206393 z_k1_s0152
                                 - 0.0550000602313558 z_k1_s0153
                                 - 0.108999978425547 z_k1_s0154
                                 - 0.081999940228322 z_k1_s0155
                                 - 0.214000654304508 z_k1_s0156
                                 - 0.0829997658649685 z_k1_s0157
                                 - 0.0979999445525486 z_k1_s0160
                                 - 0.0590000369245614 z_k1_s0161
                                 - 0.0750003223619963 z_k1_s0162
                                 - 0.0630000757838888 z_k1_s0163
                                 - 0.0919997887634635 z_k1_s0164
                                 - 0.0659997636780551 z_k1_s0165
                                 - 0.0560000240300815 z_k1_s0166
                                 - 0.141000086040051 z_k1_s0167
                                 - 0.0629999759970998 z_k1_s0170
                                 - 0.0500000091500629 z_k1_s0171
                                 - 0.200000205800153 z_k1_s0172
                                 - 0.11000024009001 z_k1_s0173
                                 - 0.121999897118226 z_k1_s0174
                                 - 0.141000200079076 z_k1_s0175
                                 - 0.0470000102930581 z_k1_s0176
                                 - 0.113000154347064 z_k1_s0177
                                 - 0.199000809126746 z_k1_s1000
                                 - 0.650004520382435 z_k1_s1001
                                 - 0.80500378994223 z_k1_s1002
                                 - 0.716001582208705 z_k1_s1003
                                 - 0.592004685372465 z_k1_s1004
                                 - 0.35400175789427 z_k1_s1005
                                 - 0.174000764435708 z_k1_s1006
                                 - 0.708004355392405 z_k1_s1007
                                 - 0.435999251384899 z_k1_s1010
                                 - 0.208999317389701 z_k1_s1011
                                 - 0.536000271423375 z_k1_s1012
                                 - 0.941997051803874 z_k1_s1013
                                 - 0.24799906570685 z_k1_s1014
                                 - 0.64099809497449 z_k1_s1015
                                 - 0.632996281377126 z_k1_s1016
                                 - 0.684999409788561 z_k1_s1017
                                 - 0.726998443036035 z_k1_s1020
                                 - 0.966999972450982 z_k1_s1021
                                 - 0.603998488773943 z_k1_s1022
                                 - 0.422000519594795 z_k1_s1023
                                 - 0.612998609939548 z_k1_s1024
                                 - 0.27200020183589 z_k1_s1025
                                 - 0.983999634096735 z_k1_s1026
                                 - 0.515998438488813 z_k1_s1027
                                 - 0.131000052594073 z_k1_s1030
                                 - 0.363000226101154 z_k1_s1031
                                 - 0.117000039196232 z_k1_s1032
                                 - 0.139999964178377 z_k1_s1033
                                 - 0.382000850712658 z_k1_s1034
                                 - 0.160999984275139 z_k1_s1035
                                 - 0.100999951112966 z_k1_s1036
                                 - 0.302999683538857 z_k1_s1037
                                 - 0.390001093720194 z_k1_s1040
                                 - 0.608001086989904 z_k1_s1041
                                 - 0.445999914891295 z_k1_s1042
                                 - 0.421999309247387 z_k1_s1043
                                 - 0.713000334684603 z_k1_s1044
                                 - 0.596998055405219 z_k1_s1045
                                 - 0.238999842808649 z_k1_s1046
                                 - 0.47299816624255 z_k1_s1047
                                 - 0.608000107264767 z_k1_s1050
                                 - 0.392999748993179 z_k1_s1051
                                 - 1.12899825457684 z_k1_s1052
                                 - 0.755998491064737 z_k1_s1053
                                 - 0.403999981037772 z_k1_s1054
                                 - 0.555000111449911 z_k1_s1055
                                 - 1.11699675452818 z_k1_s1056
                                 - 1.89599697768617 z_k1_s1057
                                 - 0.441998826965331 z_k1_s1060
                                 - 0.579000791581175 z_k1_s1061
                                 - 0.614996670973567 z_k1_s1062
                                 - 0.466998796170532 z_k1_s1063
                                 - 0.86199630957708 z_k1_s1064
                                 - 0.433999160634959 z_k1_s1065
                                 - 0.62799690258373 z_k1_s1066
                                 - 0.734998808913541 z_k1_s1067
                                 - 0.664999283981201 z_k1_s1070
                                 - 1.69099862718617 z_k1_s1071
                                 - 1.17299579445369 z_k1_s1072
                                 - 0.896001240566825 z_k1_s1073
                                 - 0.521999803360328 z_k1_s1074
                                 - 0.4019994717881 z_k1_s1075
                                 - 0.875000933131858 z_k1_s1076
                                 - 0.902001328405387 z_k1_s1077
                                 - 0.439999619719877 z_k1_s1100
                                 - 0.299999377191361 z_k1_s1101
                                 - 1.00199937168901 z_k1_s1102
                                 - 0.950999366830639 z_k1_s1103
                                 - 1.19899842543746 z_k1_s1104
                                 - 0.846999960059744 z_k1_s1105
                                 - 0.795999955201376 z_k1_s1106
                                 - 0.617001626931282 z_k1_s1107
                                 - 0.52100131721971 z_k1_s1110
                                 - 0.603001370837869 z_k1_s1111
                                 - 0.439000638000337 z_k1_s1112
                                 - 1.85200481200736 z_k1_s1113
                                 - 0.386000954291721 z_k1_s1114
                                 - 0.415000698600048 z_k1_s1115
                                 - 0.965002385729301 z_k1_s1116
                                 - 1.07500184731345 z_k1_s1117
                                 - 0.289999148937262 z_k1_s1120
                                 - 0.154999784948998 z_k1_s1121
                                 - 0.367999399685448 z_k1_s1122
                                 - 0.14999994720856 z_k1_s1123
                                 - 0.219999815572656 z_k1_s1124
                                 - 0.57599917668146 z_k1_s1125
                                 - 0.616998862652084 z_k1_s1126
                                 - 0.299000515368471 z_k1_s1127
                                 - 0.192999742851242 z_k1_s1130
                                 - 0.215999974123457 z_k1_s1131
                                 - 0.349999827014967 z_k1_s1132
                                 - 0.171999874550245 z_k1_s1133
                                 - 0.214000217751745 z_k1_s1134
                                 - 0.341998779529767 z_k1_s1135
                                 - 0.330000241299497 z_k1_s1136
                                 - 0.205000303079866 z_k1_s1137
                                 - 0.291999756830865 z_k1_s1140
                                 - 0.349999633352131 z_k1_s1141
                                 - 0.37299982093794 z_k1_s1142
                                 - 1.32900034042413 z_k1_s1143
                                 - 0.854996713133757 z_k1_s1144
                                 - 0.616998828372652 z_k1_s1145
                                 - 1.1290013339365 z_k1_s1146
                                 - 0.542999250852181 z_k1_s1147
                                 - 0.953000812295275 z_k1_s1150
                                 - 0.242000158203694 z_k1_s1151
                                 - 0.603999805964123 z_k1_s1152
                                 - 0.255000187734074 z_k1_s1153
                                 - 0.664002789645064 z_k1_s1154
                                 - 0.627000641241572 z_k1_s1155
                                 - 0.428000153607259 z_k1_s1156
                                 - 0.854001441626134 z_k1_s1157
                                 - 0.239999668137583 z_k1_s1160
                                 - 0.237999725156387 z_k1_s1161
                                 - 0.247999690462138 z_k1_s1162
                                 - 0.273000855725379 z_k1_s1163
                                 - 0.268999467364353 z_k1_s1164
                                 - 0.298000768989758 z_k1_s1165
                                 - 0.64099837705813 z_k1_s1166
                                 - 0.29699916990064 z_k1_s1167
                                 - 0.135000106515078 z_k1_s1170
                                 - 0.093000080354834 z_k1_s1171
                                 - 0.0840000523204884 z_k1_s1172
                                 - 0.244000062264246 z_k1_s1173
                                 - 0.135000106515078 z_k1_s1174
                                 - 0.210000130801221 z_k1_s1175
                                 - 0.369000364407905 z_k1_s1176
                                 - 0.300000411144677 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.09936 w_k1_ADM_s000
                                 + 1.04174 w_k1_AXP_s000
                                 + 1.25301 w_k1_DIS_s000
                                 + 1.19054 w_k1_FDX_s000
                                 + 1.11094 w_k1_IBM_s000
                                 + 1.11599 w_k1_INTC_s000
                                 + 1.16893 w_k1_KO_s000 + 1.02119 w_k1_MAT_s000
                                 + 1.26378 w_k1_MSFT_s000
                                 + 1.08749 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.11163 w_k1_ADM_s000
                                 + 1.25706 w_k1_AXP_s000
                                 + 1.26415 w_k1_DIS_s000
                                 + 1.13062 w_k1_FDX_s000
                                 + 1.66145 w_k1_IBM_s000
                                 + 1.11795 w_k1_INTC_s000
                                 + 1.21903 w_k1_KO_s000 + 1.51544 w_k1_MAT_s000
                                 + 1.07482 w_k1_MSFT_s000
                                 + 1.05858 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.09153 w_k1_ADM_s000
                                 + 1.20599 w_k1_AXP_s000
                                 + 1.39866 w_k1_DIS_s000 + 1.2561 w_k1_FDX_s000
                                 + 1.10813 w_k1_IBM_s000
                                 + 1.2356 w_k1_INTC_s000 + 1.18904 w_k1_KO_s000
                                 + 1.17718 w_k1_MAT_s000
                                 + 1.18472 w_k1_MSFT_s000
                                 + 1.21618 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.820677 w_k1_ADM_s000
                                 + 0.833037 w_k1_AXP_s000
                                 + 0.913214 w_k1_DIS_s000
                                 + 1.00037 w_k1_FDX_s000
                                 + 1.17482 w_k1_IBM_s000
                                 + 0.855576 w_k1_INTC_s000
                                 + 0.946558 w_k1_KO_s000
                                 + 1.13409 w_k1_MAT_s000
                                 + 0.798805 w_k1_MSFT_s000
                                 + 1.0095 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.953835 w_k1_ADM_s000
                                 + 0.94423 w_k1_AXP_s000
                                 + 0.758904 w_k1_DIS_s000
                                 + 1.02319 w_k1_FDX_s000
                                 + 0.775536 w_k1_IBM_s000
                                 + 1.03813 w_k1_INTC_s000
                                 + 0.815551 w_k1_KO_s000
                                 + 0.862271 w_k1_MAT_s000
                                 + 0.985995 w_k1_MSFT_s000
                                 + 0.965836 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.605733 w_k1_ADM_s000
                                 + 0.679701 w_k1_AXP_s000
                                 + 0.656286 w_k1_DIS_s000
                                 + 0.785037 w_k1_FDX_s000
                                 + 0.988456 w_k1_IBM_s000
                                 + 0.768034 w_k1_INTC_s000
                                 + 0.802126 w_k1_KO_s000
                                 + 0.881258 w_k1_MAT_s000
                                 + 0.857327 w_k1_MSFT_s000
                                 + 0.840518 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.977358 w_k1_ADM_s000
                                 + 0.601178 w_k1_AXP_s000
                                 + 0.758295 w_k1_DIS_s000
                                 + 0.923784 w_k1_FDX_s000
                                 + 0.680762 w_k1_IBM_s000
                                 + 0.845309 w_k1_INTC_s000
                                 + 0.634126 w_k1_KO_s000
                                 + 0.528464 w_k1_MAT_s000
                                 + 0.802531 w_k1_MSFT_s000
                                 + 0.807283 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.46508 w_k1_ADM_s000
                                 + 1.38675 w_k1_AXP_s000
                                 + 1.23368 w_k1_DIS_s000 + 1.4011 w_k1_FDX_s000
                                 + 0.904119 w_k1_IBM_s000
                                 + 1.302 w_k1_INTC_s000 + 1.0552 w_k1_KO_s000
                                 + 0.898995 w_k1_MAT_s000
                                 + 1.15296 w_k1_MSFT_s000
                                 + 1.11027 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.818739 w_k1_ADM_s001
                                 + 0.922287 w_k1_AXP_s001
                                 + 0.895017 w_k1_DIS_s001
                                 + 0.903174 w_k1_FDX_s001
                                 + 1.36812 w_k1_IBM_s001
                                 + 0.892082 w_k1_INTC_s001
                                 + 1.08208 w_k1_KO_s001 + 1.5648 w_k1_MAT_s001
                                 + 0.94809 w_k1_MSFT_s001
                                 + 1.04827 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.688525 w_k1_ADM_s001
                                 + 0.664584 w_k1_AXP_s001
                                 + 0.604789 w_k1_DIS_s001
                                 + 0.710701 w_k1_FDX_s001
                                 + 0.760219 w_k1_IBM_s001
                                 + 0.742629 w_k1_INTC_s001
                                 + 0.683039 w_k1_KO_s001
                                 + 0.753168 w_k1_MAT_s001
                                 + 0.635606 w_k1_MSFT_s001
                                 + 0.769897 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.42211 w_k1_ADM_s001
                                 + 0.91069 w_k1_AXP_s001
                                 + 1.05315 w_k1_DIS_s001
                                 + 1.22181 w_k1_FDX_s001
                                 + 0.562598 w_k1_IBM_s001
                                 + 0.918577 w_k1_INTC_s001
                                 + 0.75953 w_k1_KO_s001
                                 + 0.692335 w_k1_MAT_s001
                                 + 1.07336 w_k1_MSFT_s001
                                 + 1.08451 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.07767 w_k1_ADM_s001
                                 + 1.13925 w_k1_AXP_s001
                                 + 1.61482 w_k1_DIS_s001
                                 + 1.34322 w_k1_FDX_s001
                                 + 1.46311 w_k1_IBM_s001
                                 + 1.13425 w_k1_INTC_s001 + 1.3435 w_k1_KO_s001
                                 + 1.6903 w_k1_MAT_s001
                                 + 1.24422 w_k1_MSFT_s001
                                 + 1.23503 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.857469 w_k1_ADM_s001
                                 + 0.739022 w_k1_AXP_s001
                                 + 0.817023 w_k1_DIS_s001
                                 + 0.861533 w_k1_FDX_s001
                                 + 0.869708 w_k1_IBM_s001
                                 + 0.831104 w_k1_INTC_s001
                                 + 0.902206 w_k1_KO_s001
                                 + 1.06599 w_k1_MAT_s001
                                 + 0.926977 w_k1_MSFT_s001
                                 + 0.86934 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.919243 w_k1_ADM_s001
                                 + 0.734652 w_k1_AXP_s001
                                 + 0.76146 w_k1_DIS_s001
                                 + 0.999052 w_k1_FDX_s001
                                 + 0.563064 w_k1_IBM_s001
                                 + 0.885165 w_k1_INTC_s001
                                 + 0.722244 w_k1_KO_s001
                                 + 0.439221 w_k1_MAT_s001
                                 + 0.895987 w_k1_MSFT_s001
                                 + 0.890848 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.20078 w_k1_ADM_s001
                                 + 1.06327 w_k1_AXP_s001
                                 + 1.29919 w_k1_DIS_s001 + 1.2136 w_k1_FDX_s001
                                 + 1.10218 w_k1_IBM_s001
                                 + 1.16498 w_k1_INTC_s001
                                 + 1.10495 w_k1_KO_s001
                                 + 0.997228 w_k1_MAT_s001
                                 + 1.25938 w_k1_MSFT_s001
                                 + 1.10169 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.971145 w_k1_ADM_s001
                                 + 1.20178 w_k1_AXP_s001
                                 + 0.982387 w_k1_DIS_s001
                                 + 1.10233 w_k1_FDX_s001
                                 + 0.968423 w_k1_IBM_s001
                                 + 0.905118 w_k1_INTC_s001
                                 + 0.887518 w_k1_KO_s001
                                 + 1.15626 w_k1_MAT_s001
                                 + 0.961303 w_k1_MSFT_s001
                                 + 0.933964 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.16895 w_k1_ADM_s002
                                 + 0.784723 w_k1_AXP_s002
                                 + 0.775894 w_k1_DIS_s002
                                 + 0.863842 w_k1_FDX_s002
                                 + 0.669861 w_k1_IBM_s002
                                 + 0.917151 w_k1_INTC_s002
                                 + 0.765061 w_k1_KO_s002
                                 + 0.75173 w_k1_MAT_s002
                                 + 0.817534 w_k1_MSFT_s002
                                 + 0.853657 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.694187 w_k1_ADM_s002
                                 + 0.612673 w_k1_AXP_s002
                                 + 0.758231 w_k1_DIS_s002
                                 + 0.779655 w_k1_FDX_s002
                                 + 1.28362 w_k1_IBM_s002
                                 + 0.920236 w_k1_INTC_s002
                                 + 0.975495 w_k1_KO_s002
                                 + 1.00731 w_k1_MAT_s002
                                 + 0.921601 w_k1_MSFT_s002
                                 + 0.902151 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.29153 w_k1_ADM_s002
                                 + 1.14354 w_k1_AXP_s002
                                 + 1.36992 w_k1_DIS_s002
                                 + 1.27118 w_k1_FDX_s002
                                 + 1.43289 w_k1_IBM_s002
                                 + 1.42832 w_k1_INTC_s002
                                 + 1.29653 w_k1_KO_s002 + 1.05329 w_k1_MAT_s002
                                 + 1.13166 w_k1_MSFT_s002 + 1.1241 w_k1_PG_s002
                                 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.044 w_k1_ADM_s002
                                 + 0.640643 w_k1_AXP_s002
                                 + 0.609797 w_k1_DIS_s002
                                 + 0.750324 w_k1_FDX_s002
                                 + 0.771723 w_k1_IBM_s002
                                 + 0.922323 w_k1_INTC_s002
                                 + 0.742541 w_k1_KO_s002
                                 + 0.615919 w_k1_MAT_s002
                                 + 0.767183 w_k1_MSFT_s002
                                 + 0.81554 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.19447 w_k1_ADM_s002
                                 + 1.41995 w_k1_AXP_s002
                                 + 1.38713 w_k1_DIS_s002
                                 + 1.26951 w_k1_FDX_s002
                                 + 1.11542 w_k1_IBM_s002
                                 + 1.20999 w_k1_INTC_s002 + 1.0696 w_k1_KO_s002
                                 + 1.34437 w_k1_MAT_s002
                                 + 1.15638 w_k1_MSFT_s002
                                 + 1.13285 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.975242 w_k1_ADM_s002
                                 + 0.97302 w_k1_AXP_s002
                                 + 0.868837 w_k1_DIS_s002
                                 + 1.02407 w_k1_FDX_s002
                                 + 0.808045 w_k1_IBM_s002
                                 + 1.06239 w_k1_INTC_s002
                                 + 0.748882 w_k1_KO_s002
                                 + 0.82018 w_k1_MAT_s002
                                 + 0.931624 w_k1_MSFT_s002
                                 + 0.82674 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.20202 w_k1_ADM_s002
                                 + 0.820738 w_k1_AXP_s002
                                 + 1.00853 w_k1_DIS_s002
                                 + 1.00121 w_k1_FDX_s002 + 1.1002 w_k1_IBM_s002
                                 + 1.08197 w_k1_INTC_s002
                                 + 1.00831 w_k1_KO_s002
                                 + 0.983057 w_k1_MAT_s002
                                 + 0.968786 w_k1_MSFT_s002
                                 + 0.960831 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.18645 w_k1_ADM_s002
                                 + 0.784824 w_k1_AXP_s002
                                 + 1.21992 w_k1_DIS_s002
                                 + 1.17225 w_k1_FDX_s002
                                 + 0.833574 w_k1_IBM_s002
                                 + 1.10053 w_k1_INTC_s002
                                 + 1.01749 w_k1_KO_s002
                                 + 0.813083 w_k1_MAT_s002
                                 + 1.01411 w_k1_MSFT_s002
                                 + 1.13848 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.752726 w_k1_ADM_s003
                                 + 0.874508 w_k1_AXP_s003
                                 + 0.796078 w_k1_DIS_s003
                                 + 0.839751 w_k1_FDX_s003
                                 + 0.955989 w_k1_IBM_s003
                                 + 0.911124 w_k1_INTC_s003
                                 + 0.921119 w_k1_KO_s003
                                 + 1.52995 w_k1_MAT_s003
                                 + 0.890479 w_k1_MSFT_s003
                                 + 0.864199 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.896468 w_k1_ADM_s003
                                 + 1.01773 w_k1_AXP_s003
                                 + 1.06245 w_k1_DIS_s003
                                 + 1.05314 w_k1_FDX_s003
                                 + 0.745239 w_k1_IBM_s003
                                 + 0.950805 w_k1_INTC_s003
                                 + 0.99008 w_k1_KO_s003 + 1.35332 w_k1_MAT_s003
                                 + 1.08286 w_k1_MSFT_s003
                                 + 1.06065 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.970297 w_k1_ADM_s003
                                 + 1.15927 w_k1_AXP_s003
                                 + 1.49749 w_k1_DIS_s003
                                 + 1.08221 w_k1_FDX_s003
                                 + 1.73887 w_k1_IBM_s003
                                 + 1.32669 w_k1_INTC_s003
                                 + 1.72461 w_k1_KO_s003 + 2.57165 w_k1_MAT_s003
                                 + 1.45885 w_k1_MSFT_s003
                                 + 1.27208 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.33718 w_k1_ADM_s003
                                 + 1.36491 w_k1_AXP_s003
                                 + 1.36749 w_k1_DIS_s003
                                 + 1.28701 w_k1_FDX_s003
                                 + 0.976924 w_k1_IBM_s003
                                 + 1.21224 w_k1_INTC_s003
                                 + 1.13784 w_k1_KO_s003 + 1.33259 w_k1_MAT_s003
                                 + 1.28809 w_k1_MSFT_s003
                                 + 1.03098 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.866579 w_k1_ADM_s003
                                 + 1.14013 w_k1_AXP_s003
                                 + 0.903452 w_k1_DIS_s003
                                 + 0.926737 w_k1_FDX_s003
                                 + 1.27949 w_k1_IBM_s003
                                 + 0.993257 w_k1_INTC_s003
                                 + 1.10507 w_k1_KO_s003 + 1.9559 w_k1_MAT_s003
                                 + 1.00773 w_k1_MSFT_s003
                                 + 1.10954 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.06785 w_k1_ADM_s003
                                 + 1.31499 w_k1_AXP_s003
                                 + 1.30417 w_k1_DIS_s003
                                 + 1.22145 w_k1_FDX_s003
                                 + 1.26919 w_k1_IBM_s003
                                 + 1.25528 w_k1_INTC_s003
                                 + 1.22213 w_k1_KO_s003 + 1.5589 w_k1_MAT_s003
                                 + 1.36488 w_k1_MSFT_s003
                                 + 1.34967 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.13076 w_k1_ADM_s003
                                 + 1.14221 w_k1_AXP_s003
                                 + 1.07075 w_k1_DIS_s003
                                 + 0.983889 w_k1_FDX_s003
                                 + 1.52986 w_k1_IBM_s003
                                 + 1.08195 w_k1_INTC_s003
                                 + 1.34511 w_k1_KO_s003 + 1.90425 w_k1_MAT_s003
                                 + 1.0571 w_k1_MSFT_s003 + 1.13767 w_k1_PG_s003
                                 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.614766 w_k1_ADM_s003
                                 + 0.687941 w_k1_AXP_s003
                                 + 0.577064 w_k1_DIS_s003
                                 + 0.713992 w_k1_FDX_s003
                                 + 0.737489 w_k1_IBM_s003
                                 + 0.885585 w_k1_INTC_s003
                                 + 0.765678 w_k1_KO_s003
                                 + 1.06822 w_k1_MAT_s003
                                 + 0.736442 w_k1_MSFT_s003
                                 + 0.796697 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.842036 w_k1_ADM_s004
                                 + 0.826842 w_k1_AXP_s004
                                 + 0.724924 w_k1_DIS_s004
                                 + 0.797891 w_k1_FDX_s004
                                 + 0.786319 w_k1_IBM_s004
                                 + 0.822069 w_k1_INTC_s004
                                 + 0.862035 w_k1_KO_s004
                                 + 0.88716 w_k1_MAT_s004
                                 + 0.87776 w_k1_MSFT_s004
                                 + 0.917944 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.07254 w_k1_ADM_s004
                                 + 0.985813 w_k1_AXP_s004
                                 + 1.02885 w_k1_DIS_s004
                                 + 1.08128 w_k1_FDX_s004
                                 + 0.711109 w_k1_IBM_s004
                                 + 1.21515 w_k1_INTC_s004
                                 + 0.911632 w_k1_KO_s004
                                 + 0.728296 w_k1_MAT_s004
                                 + 1.23738 w_k1_MSFT_s004
                                 + 0.979961 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.02503 w_k1_ADM_s004
                                 + 1.17578 w_k1_AXP_s004 + 1.1915 w_k1_DIS_s004
                                 + 1.08808 w_k1_FDX_s004
                                 + 0.971344 w_k1_IBM_s004
                                 + 1.11235 w_k1_INTC_s004
                                 + 1.12971 w_k1_KO_s004 + 1.35891 w_k1_MAT_s004
                                 + 1.18302 w_k1_MSFT_s004
                                 + 1.02023 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.11003 w_k1_ADM_s004
                                 + 1.26878 w_k1_AXP_s004
                                 + 1.38374 w_k1_DIS_s004
                                 + 1.11894 w_k1_FDX_s004
                                 + 1.68504 w_k1_IBM_s004
                                 + 1.15878 w_k1_INTC_s004
                                 + 1.65229 w_k1_KO_s004 + 2.12696 w_k1_MAT_s004
                                 + 1.24091 w_k1_MSFT_s004
                                 + 1.20584 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.21183 w_k1_ADM_s004
                                 + 1.35013 w_k1_AXP_s004
                                 + 1.65073 w_k1_DIS_s004
                                 + 1.35382 w_k1_FDX_s004
                                 + 1.01779 w_k1_IBM_s004
                                 + 1.47108 w_k1_INTC_s004 + 1.3892 w_k1_KO_s004
                                 + 1.20422 w_k1_MAT_s004
                                 + 1.47845 w_k1_MSFT_s004
                                 + 1.16577 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.00503 w_k1_ADM_s004
                                 + 0.925664 w_k1_AXP_s004
                                 + 0.820502 w_k1_DIS_s004
                                 + 0.878573 w_k1_FDX_s004
                                 + 1.18472 w_k1_IBM_s004
                                 + 0.851092 w_k1_INTC_s004
                                 + 1.04401 w_k1_KO_s004
                                 + 0.842731 w_k1_MAT_s004
                                 + 0.878822 w_k1_MSFT_s004
                                 + 0.998922 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.734179 w_k1_ADM_s004
                                 + 0.8911 w_k1_AXP_s004
                                 + 0.929748 w_k1_DIS_s004
                                 + 0.860879 w_k1_FDX_s004
                                 + 1.55417 w_k1_IBM_s004
                                 + 0.967855 w_k1_INTC_s004
                                 + 1.0537 w_k1_KO_s004 + 1.73536 w_k1_MAT_s004
                                 + 0.966221 w_k1_MSFT_s004
                                 + 0.993641 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.93479 w_k1_ADM_s004
                                 + 0.922185 w_k1_AXP_s004
                                 + 0.94687 w_k1_DIS_s004
                                 + 0.93159 w_k1_FDX_s004
                                 + 1.03816 w_k1_IBM_s004
                                 + 0.900479 w_k1_INTC_s004
                                 + 1.0304 w_k1_KO_s004 + 1.22972 w_k1_MAT_s004
                                 + 1.08673 w_k1_MSFT_s004
                                 + 0.926976 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.84129 w_k1_ADM_s005
                                 + 0.843685 w_k1_AXP_s005
                                 + 0.980073 w_k1_DIS_s005
                                 + 0.929538 w_k1_FDX_s005
                                 + 1.11186 w_k1_IBM_s005
                                 + 1.14507 w_k1_INTC_s005
                                 + 1.12294 w_k1_KO_s005
                                 + 0.928719 w_k1_MAT_s005
                                 + 1.00302 w_k1_MSFT_s005
                                 + 0.927333 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.41829 w_k1_ADM_s005
                                 + 1.15231 w_k1_AXP_s005
                                 + 1.71482 w_k1_DIS_s005
                                 + 1.29234 w_k1_FDX_s005
                                 + 1.29833 w_k1_IBM_s005
                                 + 1.4181 w_k1_INTC_s005 + 1.34197 w_k1_KO_s005
                                 + 0.86796 w_k1_MAT_s005
                                 + 1.36926 w_k1_MSFT_s005
                                 + 1.06106 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.40392 w_k1_ADM_s005
                                 + 1.154 w_k1_AXP_s005 + 1.2203 w_k1_DIS_s005
                                 + 1.17878 w_k1_FDX_s005
                                 + 1.11175 w_k1_IBM_s005
                                 + 1.21555 w_k1_INTC_s005 + 1.17 w_k1_KO_s005
                                 + 0.952056 w_k1_MAT_s005
                                 + 1.15595 w_k1_MSFT_s005
                                 + 1.04686 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.945976 w_k1_ADM_s005
                                 + 1.33786 w_k1_AXP_s005
                                 + 1.03858 w_k1_DIS_s005
                                 + 1.03574 w_k1_FDX_s005
                                 + 1.31724 w_k1_IBM_s005
                                 + 1.06197 w_k1_INTC_s005
                                 + 1.01144 w_k1_KO_s005 + 1.30345 w_k1_MAT_s005
                                 + 0.882179 w_k1_MSFT_s005
                                 + 0.952415 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.10249 w_k1_ADM_s005
                                 + 1.08262 w_k1_AXP_s005
                                 + 1.18505 w_k1_DIS_s005
                                 + 0.994106 w_k1_FDX_s005
                                 + 0.99906 w_k1_IBM_s005
                                 + 1.04449 w_k1_INTC_s005
                                 + 1.05592 w_k1_KO_s005
                                 + 0.870339 w_k1_MAT_s005
                                 + 1.03199 w_k1_MSFT_s005
                                 + 0.998057 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.876158 w_k1_ADM_s005
                                 + 0.6937 w_k1_AXP_s005 + 0.63962 w_k1_DIS_s005
                                 + 0.726685 w_k1_FDX_s005
                                 + 0.872092 w_k1_IBM_s005
                                 + 0.69387 w_k1_INTC_s005
                                 + 0.763761 w_k1_KO_s005
                                 + 0.735514 w_k1_MAT_s005
                                 + 0.710146 w_k1_MSFT_s005
                                 + 0.816977 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.823791 w_k1_ADM_s005
                                 + 0.850268 w_k1_AXP_s005
                                 + 0.880893 w_k1_DIS_s005
                                 + 0.988288 w_k1_FDX_s005
                                 + 0.645614 w_k1_IBM_s005
                                 + 0.865006 w_k1_INTC_s005
                                 + 0.834201 w_k1_KO_s005
                                 + 0.748382 w_k1_MAT_s005
                                 + 0.898803 w_k1_MSFT_s005
                                 + 0.792749 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.30831 w_k1_ADM_s005
                                 + 1.45897 w_k1_AXP_s005
                                 + 1.45694 w_k1_DIS_s005
                                 + 1.35525 w_k1_FDX_s005
                                 + 1.28946 w_k1_IBM_s005
                                 + 1.24306 w_k1_INTC_s005
                                 + 1.32592 w_k1_KO_s005 + 1.63501 w_k1_MAT_s005
                                 + 1.16373 w_k1_MSFT_s005
                                 + 1.12721 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.09519 w_k1_ADM_s006
                                 + 1.04658 w_k1_AXP_s006 + 1.0402 w_k1_DIS_s006
                                 + 1.02542 w_k1_FDX_s006
                                 + 1.32427 w_k1_IBM_s006
                                 + 0.934663 w_k1_INTC_s006
                                 + 1.36653 w_k1_KO_s006 + 1.8674 w_k1_MAT_s006
                                 + 1.13591 w_k1_MSFT_s006
                                 + 1.14952 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.13514 w_k1_ADM_s006
                                 + 1.29025 w_k1_AXP_s006
                                 + 1.13889 w_k1_DIS_s006
                                 + 1.18908 w_k1_FDX_s006
                                 + 0.937234 w_k1_IBM_s006
                                 + 1.2162 w_k1_INTC_s006 + 1.11837 w_k1_KO_s006
                                 + 1.43114 w_k1_MAT_s006
                                 + 1.32776 w_k1_MSFT_s006
                                 + 1.23528 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.49771 w_k1_ADM_s006
                                 + 1.10036 w_k1_AXP_s006
                                 + 1.50051 w_k1_DIS_s006
                                 + 1.24544 w_k1_FDX_s006
                                 + 1.34643 w_k1_IBM_s006
                                 + 1.25977 w_k1_INTC_s006
                                 + 1.55237 w_k1_KO_s006 + 1.42349 w_k1_MAT_s006
                                 + 1.4711 w_k1_MSFT_s006 + 1.2265 w_k1_PG_s006
                                 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.765743 w_k1_ADM_s006
                                 + 0.843442 w_k1_AXP_s006
                                 + 0.867351 w_k1_DIS_s006
                                 + 0.900649 w_k1_FDX_s006
                                 + 0.903835 w_k1_IBM_s006
                                 + 0.981535 w_k1_INTC_s006
                                 + 0.901264 w_k1_KO_s006
                                 + 0.955287 w_k1_MAT_s006
                                 + 0.961613 w_k1_MSFT_s006
                                 + 0.980634 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.17242 w_k1_ADM_s006
                                 + 0.903917 w_k1_AXP_s006
                                 + 1.14746 w_k1_DIS_s006
                                 + 1.12475 w_k1_FDX_s006
                                 + 0.961474 w_k1_IBM_s006
                                 + 1.05597 w_k1_INTC_s006
                                 + 1.16318 w_k1_KO_s006
                                 + 0.840839 w_k1_MAT_s006
                                 + 1.10777 w_k1_MSFT_s006
                                 + 1.17006 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.12158 w_k1_ADM_s006
                                 + 0.840338 w_k1_AXP_s006
                                 + 0.967656 w_k1_DIS_s006
                                 + 1.11664 w_k1_FDX_s006
                                 + 0.717715 w_k1_IBM_s006
                                 + 1.01818 w_k1_INTC_s006
                                 + 0.948665 w_k1_KO_s006 + 0.7012 w_k1_MAT_s006
                                 + 1.0909 w_k1_MSFT_s006 + 1.08567 w_k1_PG_s006
                                 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.874648 w_k1_ADM_s006
                                 + 0.627352 w_k1_AXP_s006
                                 + 0.753655 w_k1_DIS_s006
                                 + 0.815295 w_k1_FDX_s006
                                 + 0.873841 w_k1_IBM_s006
                                 + 0.8787 w_k1_INTC_s006
                                 + 0.825221 w_k1_KO_s006
                                 + 0.712121 w_k1_MAT_s006
                                 + 1.03442 w_k1_MSFT_s006
                                 + 0.935778 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.875602 w_k1_ADM_s006
                                 + 0.696527 w_k1_AXP_s006
                                 + 0.515442 w_k1_DIS_s006
                                 + 0.721998 w_k1_FDX_s006
                                 + 0.697711 w_k1_IBM_s006
                                 + 0.856685 w_k1_INTC_s006
                                 + 0.645505 w_k1_KO_s006
                                 + 0.663408 w_k1_MAT_s006
                                 + 0.777972 w_k1_MSFT_s006
                                 + 0.742576 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.40577 w_k1_ADM_s007
                                 + 1.31126 w_k1_AXP_s007
                                 + 1.30954 w_k1_DIS_s007
                                 + 1.19431 w_k1_FDX_s007
                                 + 1.24536 w_k1_IBM_s007
                                 + 1.01756 w_k1_INTC_s007
                                 + 1.12795 w_k1_KO_s007 + 1.63739 w_k1_MAT_s007
                                 + 1.09256 w_k1_MSFT_s007
                                 + 1.15077 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.96808 w_k1_ADM_s007
                                 + 1.07346 w_k1_AXP_s007
                                 + 0.937466 w_k1_DIS_s007
                                 + 0.969732 w_k1_FDX_s007
                                 + 0.805104 w_k1_IBM_s007
                                 + 1.05165 w_k1_INTC_s007
                                 + 0.882358 w_k1_KO_s007
                                 + 0.916359 w_k1_MAT_s007
                                 + 1.00518 w_k1_MSFT_s007
                                 + 0.966464 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.20931 w_k1_ADM_s007
                                 + 1.09552 w_k1_AXP_s007
                                 + 0.872047 w_k1_DIS_s007
                                 + 1.02101 w_k1_FDX_s007
                                 + 0.738076 w_k1_IBM_s007
                                 + 0.864583 w_k1_INTC_s007
                                 + 0.83085 w_k1_KO_s007
                                 + 0.825115 w_k1_MAT_s007
                                 + 0.949319 w_k1_MSFT_s007
                                 + 0.971238 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.914931 w_k1_ADM_s007
                                 + 0.709381 w_k1_AXP_s007
                                 + 0.827792 w_k1_DIS_s007
                                 + 0.892766 w_k1_FDX_s007
                                 + 0.676624 w_k1_IBM_s007
                                 + 1.02945 w_k1_INTC_s007
                                 + 0.743921 w_k1_KO_s007
                                 + 0.517652 w_k1_MAT_s007
                                 + 0.982225 w_k1_MSFT_s007
                                 + 0.846968 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.17717 w_k1_ADM_s007
                                 + 0.692199 w_k1_AXP_s007
                                 + 1.32435 w_k1_DIS_s007 + 1.0204 w_k1_FDX_s007
                                 + 1.09762 w_k1_IBM_s007
                                 + 1.03751 w_k1_INTC_s007
                                 + 1.16746 w_k1_KO_s007
                                 + 0.926471 w_k1_MAT_s007
                                 + 1.12197 w_k1_MSFT_s007
                                 + 1.02161 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.953387 w_k1_ADM_s007
                                 + 0.640829 w_k1_AXP_s007
                                 + 1.06595 w_k1_DIS_s007
                                 + 0.916489 w_k1_FDX_s007
                                 + 0.933139 w_k1_IBM_s007
                                 + 0.969486 w_k1_INTC_s007
                                 + 0.977908 w_k1_KO_s007
                                 + 0.736192 w_k1_MAT_s007
                                 + 1.06869 w_k1_MSFT_s007
                                 + 0.849717 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.21597 w_k1_ADM_s007
                                 + 1.12037 w_k1_AXP_s007
                                 + 1.40555 w_k1_DIS_s007
                                 + 1.32048 w_k1_FDX_s007
                                 + 0.98251 w_k1_IBM_s007
                                 + 1.32845 w_k1_INTC_s007
                                 + 0.973875 w_k1_KO_s007
                                 + 1.04762 w_k1_MAT_s007
                                 + 1.43323 w_k1_MSFT_s007
                                 + 1.08272 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.02545 w_k1_ADM_s007
                                 + 0.946966 w_k1_AXP_s007
                                 + 0.744798 w_k1_DIS_s007
                                 + 0.842715 w_k1_FDX_s007
                                 + 1.10036 w_k1_IBM_s007
                                 + 1.02582 w_k1_INTC_s007
                                 + 0.909732 w_k1_KO_s007
                                 + 0.750561 w_k1_MAT_s007
                                 + 0.852996 w_k1_MSFT_s007
                                 + 0.852326 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.18664 w_k1_ADM_s010
                                 + 1.16844 w_k1_AXP_s010
                                 + 1.07677 w_k1_DIS_s010 + 1.0403 w_k1_FDX_s010
                                 + 1.25698 w_k1_IBM_s010
                                 + 0.964304 w_k1_INTC_s010
                                 + 1.22284 w_k1_KO_s010 + 1.49527 w_k1_MAT_s010
                                 + 0.921488 w_k1_MSFT_s010
                                 + 1.08721 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.805649 w_k1_ADM_s010
                                 + 0.632222 w_k1_AXP_s010
                                 + 0.712252 w_k1_DIS_s010
                                 + 0.848337 w_k1_FDX_s010
                                 + 0.607058 w_k1_IBM_s010
                                 + 0.906861 w_k1_INTC_s010
                                 + 0.836283 w_k1_KO_s010
                                 + 0.676493 w_k1_MAT_s010
                                 + 0.797922 w_k1_MSFT_s010
                                 + 0.934357 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.19022 w_k1_ADM_s010
                                 + 1.0116 w_k1_AXP_s010
                                 + 0.945167 w_k1_DIS_s010
                                 + 0.92968 w_k1_FDX_s010
                                 + 1.21111 w_k1_IBM_s010
                                 + 0.852128 w_k1_INTC_s010
                                 + 1.13908 w_k1_KO_s010 + 1.48322 w_k1_MAT_s010
                                 + 0.837789 w_k1_MSFT_s010
                                 + 1.02414 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.752579 w_k1_ADM_s010
                                 + 0.672949 w_k1_AXP_s010
                                 + 0.65496 w_k1_DIS_s010
                                 + 0.654124 w_k1_FDX_s010
                                 + 0.987866 w_k1_IBM_s010
                                 + 0.790127 w_k1_INTC_s010
                                 + 0.808408 w_k1_KO_s010
                                 + 0.693615 w_k1_MAT_s010
                                 + 0.613877 w_k1_MSFT_s010
                                 + 0.753246 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.986238 w_k1_ADM_s010
                                 + 1.33968 w_k1_AXP_s010
                                 + 1.63384 w_k1_DIS_s010
                                 + 1.25173 w_k1_FDX_s010
                                 + 1.36956 w_k1_IBM_s010
                                 + 1.2419 w_k1_INTC_s010 + 1.43037 w_k1_KO_s010
                                 + 1.94324 w_k1_MAT_s010
                                 + 1.06011 w_k1_MSFT_s010
                                 + 1.21985 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.04054 w_k1_ADM_s010
                                 + 0.974847 w_k1_AXP_s010
                                 + 1.19208 w_k1_DIS_s010
                                 + 1.09638 w_k1_FDX_s010
                                 + 0.732738 w_k1_IBM_s010
                                 + 1.00214 w_k1_INTC_s010
                                 + 1.09385 w_k1_KO_s010 + 1.24329 w_k1_MAT_s010
                                 + 1.07214 w_k1_MSFT_s010
                                 + 0.992427 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.887109 w_k1_ADM_s010
                                 + 0.751834 w_k1_AXP_s010
                                 + 0.786901 w_k1_DIS_s010
                                 + 0.85057 w_k1_FDX_s010
                                 + 1.02485 w_k1_IBM_s010
                                 + 0.753628 w_k1_INTC_s010
                                 + 0.931959 w_k1_KO_s010
                                 + 1.22508 w_k1_MAT_s010
                                 + 0.799815 w_k1_MSFT_s010
                                 + 0.874284 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.1389 w_k1_ADM_s010
                                 + 0.97095 w_k1_AXP_s010
                                 + 0.867973 w_k1_DIS_s010
                                 + 0.997323 w_k1_FDX_s010
                                 + 0.794103 w_k1_IBM_s010
                                 + 0.942476 w_k1_INTC_s010
                                 + 1.0019 w_k1_KO_s010 + 1.1876 w_k1_MAT_s010
                                 + 0.893412 w_k1_MSFT_s010
                                 + 0.937673 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.796798 w_k1_ADM_s011
                                 + 0.970961 w_k1_AXP_s011
                                 + 0.786833 w_k1_DIS_s011
                                 + 0.939812 w_k1_FDX_s011
                                 + 0.829124 w_k1_IBM_s011
                                 + 0.922444 w_k1_INTC_s011
                                 + 0.965657 w_k1_KO_s011
                                 + 0.837664 w_k1_MAT_s011
                                 + 0.975129 w_k1_MSFT_s011
                                 + 0.905741 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.908776 w_k1_ADM_s011
                                 + 1.03811 w_k1_AXP_s011
                                 + 0.911571 w_k1_DIS_s011
                                 + 1.00734 w_k1_FDX_s011
                                 + 1.44463 w_k1_IBM_s011
                                 + 1.02357 w_k1_INTC_s011
                                 + 1.41804 w_k1_KO_s011 + 1.44186 w_k1_MAT_s011
                                 + 1.0882 w_k1_MSFT_s011 + 1.17142 w_k1_PG_s011
                                 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.17058 w_k1_ADM_s011
                                 + 1.33012 w_k1_AXP_s011
                                 + 1.51958 w_k1_DIS_s011
                                 + 1.45996 w_k1_FDX_s011
                                 + 0.823807 w_k1_IBM_s011
                                 + 1.62175 w_k1_INTC_s011
                                 + 1.28126 w_k1_KO_s011
                                 + 0.719555 w_k1_MAT_s011
                                 + 1.22983 w_k1_MSFT_s011
                                 + 1.16084 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.772779 w_k1_ADM_s011
                                 + 0.735827 w_k1_AXP_s011
                                 + 0.795305 w_k1_DIS_s011
                                 + 0.803485 w_k1_FDX_s011
                                 + 1.03511 w_k1_IBM_s011
                                 + 0.892654 w_k1_INTC_s011
                                 + 0.945917 w_k1_KO_s011
                                 + 0.762091 w_k1_MAT_s011
                                 + 0.823789 w_k1_MSFT_s011
                                 + 0.892864 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.28716 w_k1_ADM_s011
                                 + 1.48841 w_k1_AXP_s011
                                 + 1.46093 w_k1_DIS_s011
                                 + 1.25735 w_k1_FDX_s011
                                 + 1.33231 w_k1_IBM_s011
                                 + 1.40688 w_k1_INTC_s011
                                 + 1.51302 w_k1_KO_s011 + 1.11228 w_k1_MAT_s011
                                 + 1.22489 w_k1_MSFT_s011
                                 + 1.18718 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.144 w_k1_ADM_s011
                                 + 1.43168 w_k1_AXP_s011
                                 + 1.01107 w_k1_DIS_s011
                                 + 1.17108 w_k1_FDX_s011
                                 + 1.16067 w_k1_IBM_s011
                                 + 1.22369 w_k1_INTC_s011
                                 + 1.31638 w_k1_KO_s011 + 1.26091 w_k1_MAT_s011
                                 + 1.06842 w_k1_MSFT_s011
                                 + 1.08322 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.14015 w_k1_ADM_s011
                                 + 0.953308 w_k1_AXP_s011
                                 + 1.21623 w_k1_DIS_s011
                                 + 1.11613 w_k1_FDX_s011
                                 + 0.895147 w_k1_IBM_s011
                                 + 1.08324 w_k1_INTC_s011
                                 + 1.25496 w_k1_KO_s011 + 0.57065 w_k1_MAT_s011
                                 + 1.10777 w_k1_MSFT_s011
                                 + 1.02799 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.31995 w_k1_ADM_s011
                                 + 1.00094 w_k1_AXP_s011
                                 + 0.916021 w_k1_DIS_s011
                                 + 1.09854 w_k1_FDX_s011
                                 + 0.624306 w_k1_IBM_s011
                                 + 0.995113 w_k1_INTC_s011
                                 + 0.985807 w_k1_KO_s011
                                 + 0.60672 w_k1_MAT_s011
                                 + 1.13294 w_k1_MSFT_s011
                                 + 1.00111 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.1982 w_k1_ADM_s012
                                 + 1.325 w_k1_AXP_s012 + 1.53061 w_k1_DIS_s012
                                 + 1.24841 w_k1_FDX_s012
                                 + 1.16026 w_k1_IBM_s012
                                 + 1.2943 w_k1_INTC_s012 + 1.35285 w_k1_KO_s012
                                 + 1.59912 w_k1_MAT_s012
                                 + 1.40702 w_k1_MSFT_s012
                                 + 1.23108 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.838393 w_k1_ADM_s012
                                 + 0.808647 w_k1_AXP_s012
                                 + 1.16093 w_k1_DIS_s012
                                 + 0.924166 w_k1_FDX_s012
                                 + 1.51365 w_k1_IBM_s012
                                 + 1.15424 w_k1_INTC_s012
                                 + 1.30646 w_k1_KO_s012 + 1.06736 w_k1_MAT_s012
                                 + 1.4244 w_k1_MSFT_s012 + 1.05326 w_k1_PG_s012
                                 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.736666 w_k1_ADM_s012
                                 + 1.05297 w_k1_AXP_s012
                                 + 0.986894 w_k1_DIS_s012
                                 + 0.932811 w_k1_FDX_s012
                                 + 1.18651 w_k1_IBM_s012
                                 + 1.16233 w_k1_INTC_s012
                                 + 0.997157 w_k1_KO_s012
                                 + 1.24717 w_k1_MAT_s012
                                 + 1.16629 w_k1_MSFT_s012
                                 + 0.991232 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.00781 w_k1_ADM_s012
                                 + 0.973647 w_k1_AXP_s012
                                 + 1.0956 w_k1_DIS_s012 + 1.04129 w_k1_FDX_s012
                                 + 1.10684 w_k1_IBM_s012
                                 + 1.08841 w_k1_INTC_s012
                                 + 1.10329 w_k1_KO_s012
                                 + 0.960527 w_k1_MAT_s012
                                 + 1.22296 w_k1_MSFT_s012
                                 + 1.11496 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.17789 w_k1_ADM_s012
                                 + 1.35773 w_k1_AXP_s012
                                 + 1.60929 w_k1_DIS_s012
                                 + 1.10279 w_k1_FDX_s012
                                 + 1.96007 w_k1_IBM_s012
                                 + 0.977051 w_k1_INTC_s012
                                 + 1.47644 w_k1_KO_s012 + 1.86517 w_k1_MAT_s012
                                 + 1.17444 w_k1_MSFT_s012
                                 + 1.12996 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.571329 w_k1_ADM_s012
                                 + 0.696832 w_k1_AXP_s012
                                 + 0.561938 w_k1_DIS_s012
                                 + 0.76288 w_k1_FDX_s012
                                 + 0.909072 w_k1_IBM_s012
                                 + 0.844453 w_k1_INTC_s012
                                 + 0.653449 w_k1_KO_s012
                                 + 0.860668 w_k1_MAT_s012
                                 + 0.838738 w_k1_MSFT_s012
                                 + 0.92901 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.683834 w_k1_ADM_s012
                                 + 0.836935 w_k1_AXP_s012
                                 + 0.728135 w_k1_DIS_s012
                                 + 0.788463 w_k1_FDX_s012
                                 + 1.1087 w_k1_IBM_s012
                                 + 1.14554 w_k1_INTC_s012 + 1.0132 w_k1_KO_s012
                                 + 1.02164 w_k1_MAT_s012
                                 + 0.970451 w_k1_MSFT_s012
                                 + 0.879723 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.930355 w_k1_ADM_s012
                                 + 0.993252 w_k1_AXP_s012
                                 + 0.925023 w_k1_DIS_s012
                                 + 0.967061 w_k1_FDX_s012
                                 + 0.727318 w_k1_IBM_s012
                                 + 0.894889 w_k1_INTC_s012
                                 + 0.84123 w_k1_KO_s012 + 1.02647 w_k1_MAT_s012
                                 + 1.0871 w_k1_MSFT_s012 + 1.02932 w_k1_PG_s012
                                 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.943094 w_k1_ADM_s013
                                 + 0.814805 w_k1_AXP_s013
                                 + 0.550618 w_k1_DIS_s013
                                 + 0.679591 w_k1_FDX_s013
                                 + 0.952401 w_k1_IBM_s013
                                 + 0.652326 w_k1_INTC_s013
                                 + 0.763443 w_k1_KO_s013
                                 + 0.805022 w_k1_MAT_s013
                                 + 0.622861 w_k1_MSFT_s013
                                 + 0.848126 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.40755 w_k1_ADM_s013
                                 + 1.1159 w_k1_AXP_s013 + 1.20381 w_k1_DIS_s013
                                 + 1.0352 w_k1_FDX_s013 + 1.49826 w_k1_IBM_s013
                                 + 0.892289 w_k1_INTC_s013
                                 + 1.39709 w_k1_KO_s013 + 1.27563 w_k1_MAT_s013
                                 + 1.02804 w_k1_MSFT_s013
                                 + 1.19706 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.49644 w_k1_ADM_s013
                                 + 1.75671 w_k1_AXP_s013
                                 + 1.56004 w_k1_DIS_s013
                                 + 1.36105 w_k1_FDX_s013 + 1.1354 w_k1_IBM_s013
                                 + 1.06843 w_k1_INTC_s013 + 1.1346 w_k1_KO_s013
                                 + 1.46472 w_k1_MAT_s013
                                 + 1.19431 w_k1_MSFT_s013
                                 + 1.15171 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.755823 w_k1_ADM_s013
                                 + 0.858418 w_k1_AXP_s013
                                 + 0.788993 w_k1_DIS_s013
                                 + 0.835706 w_k1_FDX_s013
                                 + 1.10878 w_k1_IBM_s013
                                 + 0.868655 w_k1_INTC_s013
                                 + 0.950782 w_k1_KO_s013
                                 + 0.605328 w_k1_MAT_s013
                                 + 0.725468 w_k1_MSFT_s013
                                 + 0.912378 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.16611 w_k1_ADM_s013
                                 + 1.15891 w_k1_AXP_s013
                                 + 1.13526 w_k1_DIS_s013
                                 + 1.09918 w_k1_FDX_s013
                                 + 0.828054 w_k1_IBM_s013
                                 + 0.995082 w_k1_INTC_s013
                                 + 1.09401 w_k1_KO_s013 + 1.01349 w_k1_MAT_s013
                                 + 1.11465 w_k1_MSFT_s013
                                 + 1.02569 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.987669 w_k1_ADM_s013
                                 + 1.38531 w_k1_AXP_s013
                                 + 1.00553 w_k1_DIS_s013
                                 + 1.14693 w_k1_FDX_s013
                                 + 0.79352 w_k1_IBM_s013
                                 + 1.14915 w_k1_INTC_s013
                                 + 0.882062 w_k1_KO_s013
                                 + 0.998817 w_k1_MAT_s013
                                 + 1.01857 w_k1_MSFT_s013
                                 + 0.985207 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.05826 w_k1_ADM_s013
                                 + 0.83452 w_k1_AXP_s013
                                 + 0.86087 w_k1_DIS_s013
                                 + 0.829709 w_k1_FDX_s013
                                 + 0.927828 w_k1_IBM_s013
                                 + 0.704535 w_k1_INTC_s013
                                 + 0.990398 w_k1_KO_s013
                                 + 1.00311 w_k1_MAT_s013
                                 + 0.789606 w_k1_MSFT_s013
                                 + 0.948973 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.860249 w_k1_ADM_s013
                                 + 0.820337 w_k1_AXP_s013
                                 + 0.648399 w_k1_DIS_s013
                                 + 0.760277 w_k1_FDX_s013
                                 + 0.708541 w_k1_IBM_s013
                                 + 0.655535 w_k1_INTC_s013
                                 + 0.708725 w_k1_KO_s013
                                 + 1.08454 w_k1_MAT_s013
                                 + 0.741862 w_k1_MSFT_s013
                                 + 0.777424 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.45591 w_k1_ADM_s014
                                 + 1.51073 w_k1_AXP_s014
                                 + 2.02164 w_k1_DIS_s014
                                 + 1.52795 w_k1_FDX_s014
                                 + 1.25837 w_k1_IBM_s014
                                 + 1.32269 w_k1_INTC_s014
                                 + 1.50357 w_k1_KO_s014 + 1.43149 w_k1_MAT_s014
                                 + 1.52538 w_k1_MSFT_s014
                                 + 1.29828 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.06828 w_k1_ADM_s014
                                 + 0.964065 w_k1_AXP_s014
                                 + 1.15456 w_k1_DIS_s014
                                 + 1.05411 w_k1_FDX_s014
                                 + 1.19479 w_k1_IBM_s014
                                 + 1.01513 w_k1_INTC_s014
                                 + 1.05051 w_k1_KO_s014 + 1.16366 w_k1_MAT_s014
                                 + 0.977602 w_k1_MSFT_s014
                                 + 1.09765 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.99043 w_k1_ADM_s014
                                 + 0.98086 w_k1_AXP_s014
                                 + 1.08118 w_k1_DIS_s014
                                 + 1.09687 w_k1_FDX_s014
                                 + 0.737338 w_k1_IBM_s014
                                 + 1.20143 w_k1_INTC_s014
                                 + 0.941693 w_k1_KO_s014
                                 + 0.806628 w_k1_MAT_s014
                                 + 1.05783 w_k1_MSFT_s014
                                 + 1.01486 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.839026 w_k1_ADM_s014
                                 + 0.619009 w_k1_AXP_s014
                                 + 0.632147 w_k1_DIS_s014
                                 + 0.659642 w_k1_FDX_s014
                                 + 1.15159 w_k1_IBM_s014
                                 + 0.77514 w_k1_INTC_s014
                                 + 0.912011 w_k1_KO_s014
                                 + 0.804752 w_k1_MAT_s014
                                 + 0.70399 w_k1_MSFT_s014
                                 + 0.99437 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.14272 w_k1_ADM_s014
                                 + 0.866117 w_k1_AXP_s014
                                 + 0.771346 w_k1_DIS_s014
                                 + 0.797738 w_k1_FDX_s014
                                 + 0.931964 w_k1_IBM_s014
                                 + 0.813561 w_k1_INTC_s014
                                 + 0.934864 w_k1_KO_s014
                                 + 0.855854 w_k1_MAT_s014
                                 + 0.809637 w_k1_MSFT_s014
                                 + 0.969602 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.25156 w_k1_ADM_s014
                                 + 1.20589 w_k1_AXP_s014
                                 + 1.41178 w_k1_DIS_s014
                                 + 1.05149 w_k1_FDX_s014
                                 + 1.42736 w_k1_IBM_s014
                                 + 1.17217 w_k1_INTC_s014
                                 + 1.29291 w_k1_KO_s014 + 1.3624 w_k1_MAT_s014
                                 + 1.10065 w_k1_MSFT_s014
                                 + 1.21382 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.24151 w_k1_ADM_s014
                                 + 0.985343 w_k1_AXP_s014
                                 + 1.45643 w_k1_DIS_s014
                                 + 1.23954 w_k1_FDX_s014
                                 + 0.978711 w_k1_IBM_s014
                                 + 1.33372 w_k1_INTC_s014
                                 + 0.991164 w_k1_KO_s014
                                 + 0.581918 w_k1_MAT_s014
                                 + 1.27918 w_k1_MSFT_s014
                                 + 1.08694 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.1801 w_k1_ADM_s014
                                 + 1.3346 w_k1_AXP_s014 + 1.4763 w_k1_DIS_s014
                                 + 1.27347 w_k1_FDX_s014
                                 + 1.16758 w_k1_IBM_s014
                                 + 1.28254 w_k1_INTC_s014 + 1.2135 w_k1_KO_s014
                                 + 1.38351 w_k1_MAT_s014
                                 + 1.31042 w_k1_MSFT_s014
                                 + 1.05624 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.940058 w_k1_ADM_s015
                                 + 0.916893 w_k1_AXP_s015
                                 + 0.863347 w_k1_DIS_s015
                                 + 0.928059 w_k1_FDX_s015
                                 + 0.845651 w_k1_IBM_s015
                                 + 0.868043 w_k1_INTC_s015
                                 + 0.926713 w_k1_KO_s015
                                 + 0.873833 w_k1_MAT_s015
                                 + 0.94479 w_k1_MSFT_s015
                                 + 0.962908 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.02703 w_k1_ADM_s015
                                 + 1.1551 w_k1_AXP_s015
                                 + 0.987819 w_k1_DIS_s015
                                 + 1.00645 w_k1_FDX_s015
                                 + 1.41175 w_k1_IBM_s015
                                 + 0.758873 w_k1_INTC_s015
                                 + 1.02623 w_k1_KO_s015 + 1.33151 w_k1_MAT_s015
                                 + 0.830467 w_k1_MSFT_s015
                                 + 0.952845 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.04619 w_k1_ADM_s015
                                 + 1.38252 w_k1_AXP_s015
                                 + 1.27306 w_k1_DIS_s015 + 1.1316 w_k1_FDX_s015
                                 + 1.64096 w_k1_IBM_s015
                                 + 1.01572 w_k1_INTC_s015
                                 + 1.29774 w_k1_KO_s015 + 1.67908 w_k1_MAT_s015
                                 + 1.03874 w_k1_MSFT_s015
                                 + 1.25502 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.795572 w_k1_ADM_s015
                                 + 0.706012 w_k1_AXP_s015
                                 + 0.584798 w_k1_DIS_s015
                                 + 0.701563 w_k1_FDX_s015
                                 + 0.974114 w_k1_IBM_s015
                                 + 0.749359 w_k1_INTC_s015
                                 + 0.786478 w_k1_KO_s015
                                 + 0.871093 w_k1_MAT_s015
                                 + 0.769514 w_k1_MSFT_s015
                                 + 0.915663 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.90903 w_k1_ADM_s015
                                 + 1.12277 w_k1_AXP_s015
                                 + 0.776933 w_k1_DIS_s015
                                 + 0.938418 w_k1_FDX_s015
                                 + 0.688908 w_k1_IBM_s015
                                 + 0.793736 w_k1_INTC_s015
                                 + 0.701553 w_k1_KO_s015
                                 + 0.923282 w_k1_MAT_s015
                                 + 0.801613 w_k1_MSFT_s015
                                 + 0.890371 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.48581 w_k1_ADM_s015
                                 + 1.67412 w_k1_AXP_s015
                                 + 1.44612 w_k1_DIS_s015
                                 + 1.31771 w_k1_FDX_s015
                                 + 1.09513 w_k1_IBM_s015
                                 + 0.818186 w_k1_INTC_s015
                                 + 1.11949 w_k1_KO_s015 + 1.69864 w_k1_MAT_s015
                                 + 1.1397 w_k1_MSFT_s015 + 1.10127 w_k1_PG_s015
                                 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.24166 w_k1_ADM_s015
                                 + 1.19657 w_k1_AXP_s015
                                 + 1.05559 w_k1_DIS_s015
                                 + 1.08008 w_k1_FDX_s015
                                 + 0.957249 w_k1_IBM_s015
                                 + 0.996756 w_k1_INTC_s015
                                 + 1.03464 w_k1_KO_s015 + 1.00467 w_k1_MAT_s015
                                 + 1.00404 w_k1_MSFT_s015
                                 + 1.13954 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.938024 w_k1_ADM_s015
                                 + 1.07169 w_k1_AXP_s015
                                 + 0.846945 w_k1_DIS_s015
                                 + 0.822148 w_k1_FDX_s015
                                 + 0.987908 w_k1_IBM_s015
                                 + 0.80325 w_k1_INTC_s015
                                 + 0.831804 w_k1_KO_s015
                                 + 1.21294 w_k1_MAT_s015
                                 + 0.710629 w_k1_MSFT_s015
                                 + 1.00988 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.07732 w_k1_ADM_s016
                                 + 0.74162 w_k1_AXP_s016
                                 + 0.814378 w_k1_DIS_s016
                                 + 0.838406 w_k1_FDX_s016
                                 + 1.02516 w_k1_IBM_s016
                                 + 0.796675 w_k1_INTC_s016
                                 + 1.06475 w_k1_KO_s016 + 1.22001 w_k1_MAT_s016
                                 + 0.878665 w_k1_MSFT_s016
                                 + 0.982166 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.56615 w_k1_ADM_s016
                                 + 1.33854 w_k1_AXP_s016
                                 + 1.75824 w_k1_DIS_s016
                                 + 1.31812 w_k1_FDX_s016
                                 + 1.84879 w_k1_IBM_s016
                                 + 1.26367 w_k1_INTC_s016
                                 + 1.68803 w_k1_KO_s016 + 2.00057 w_k1_MAT_s016
                                 + 1.38792 w_k1_MSFT_s016
                                 + 1.50369 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.19426 w_k1_ADM_s016
                                 + 1.2966 w_k1_AXP_s016 + 1.07158 w_k1_DIS_s016
                                 + 1.05849 w_k1_FDX_s016
                                 + 0.938795 w_k1_IBM_s016
                                 + 0.797749 w_k1_INTC_s016
                                 + 0.83678 w_k1_KO_s016 + 1.68344 w_k1_MAT_s016
                                 + 0.948822 w_k1_MSFT_s016
                                 + 0.904478 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.832299 w_k1_ADM_s016
                                 + 0.783481 w_k1_AXP_s016
                                 + 0.616254 w_k1_DIS_s016
                                 + 0.856076 w_k1_FDX_s016
                                 + 0.821748 w_k1_IBM_s016
                                 + 0.902588 w_k1_INTC_s016
                                 + 0.73057 w_k1_KO_s016 + 1.07754 w_k1_MAT_s016
                                 + 0.901839 w_k1_MSFT_s016
                                 + 0.831921 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.870343 w_k1_ADM_s016
                                 + 0.703094 w_k1_AXP_s016
                                 + 0.627008 w_k1_DIS_s016
                                 + 0.771625 w_k1_FDX_s016
                                 + 0.636328 w_k1_IBM_s016
                                 + 0.749747 w_k1_INTC_s016
                                 + 0.658265 w_k1_KO_s016
                                 + 0.602456 w_k1_MAT_s016
                                 + 0.762941 w_k1_MSFT_s016
                                 + 0.731827 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.02203 w_k1_ADM_s016
                                 + 1.2105 w_k1_AXP_s016 + 1.48995 w_k1_DIS_s016
                                 + 1.42185 w_k1_FDX_s016
                                 + 0.918287 w_k1_IBM_s016
                                 + 0.996929 w_k1_INTC_s016
                                 + 1.20449 w_k1_KO_s016 + 1.2815 w_k1_MAT_s016
                                 + 1.21876 w_k1_MSFT_s016
                                 + 1.26819 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.902076 w_k1_ADM_s016
                                 + 0.982137 w_k1_AXP_s016 + 1.057 w_k1_DIS_s016
                                 + 1.11003 w_k1_FDX_s016
                                 + 0.856679 w_k1_IBM_s016
                                 + 0.893074 w_k1_INTC_s016
                                 + 0.888342 w_k1_KO_s016
                                 + 1.03343 w_k1_MAT_s016
                                 + 0.991817 w_k1_MSFT_s016
                                 + 0.94506 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.44575 w_k1_ADM_s016
                                 + 1.14673 w_k1_AXP_s016
                                 + 0.979585 w_k1_DIS_s016
                                 + 1.16562 w_k1_FDX_s016
                                 + 1.26924 w_k1_IBM_s016
                                 + 0.940851 w_k1_INTC_s016
                                 + 1.07928 w_k1_KO_s016 + 1.57601 w_k1_MAT_s016
                                 + 1.1207 w_k1_MSFT_s016 + 1.05486 w_k1_PG_s016
                                 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.896127 w_k1_ADM_s017
                                 + 0.921959 w_k1_AXP_s017
                                 + 1.06188 w_k1_DIS_s017
                                 + 0.92176 w_k1_FDX_s017
                                 + 1.21834 w_k1_IBM_s017
                                 + 1.06491 w_k1_INTC_s017
                                 + 1.01053 w_k1_KO_s017 + 1.19716 w_k1_MAT_s017
                                 + 0.9154 w_k1_MSFT_s017
                                 + 0.883996 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.00729 w_k1_ADM_s017
                                 + 0.801479 w_k1_AXP_s017
                                 + 0.581013 w_k1_DIS_s017
                                 + 0.728015 w_k1_FDX_s017
                                 + 0.669326 w_k1_IBM_s017
                                 + 0.643863 w_k1_INTC_s017
                                 + 0.814984 w_k1_KO_s017
                                 + 0.744749 w_k1_MAT_s017
                                 + 0.66842 w_k1_MSFT_s017
                                 + 0.855402 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.854365 w_k1_ADM_s017
                                 + 0.960212 w_k1_AXP_s017
                                 + 0.977025 w_k1_DIS_s017
                                 + 1.08074 w_k1_FDX_s017
                                 + 0.721916 w_k1_IBM_s017
                                 + 0.971971 w_k1_INTC_s017
                                 + 0.852704 w_k1_KO_s017
                                 + 0.855428 w_k1_MAT_s017
                                 + 0.93014 w_k1_MSFT_s017
                                 + 0.922668 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.860932 w_k1_ADM_s017
                                 + 0.848566 w_k1_AXP_s017
                                 + 0.891349 w_k1_DIS_s017
                                 + 0.93737 w_k1_FDX_s017
                                 + 0.758111 w_k1_IBM_s017
                                 + 0.957574 w_k1_INTC_s017
                                 + 0.84309 w_k1_KO_s017
                                 + 0.823759 w_k1_MAT_s017
                                 + 0.851916 w_k1_MSFT_s017
                                 + 0.912571 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.0206 w_k1_ADM_s017
                                 + 1.30759 w_k1_AXP_s017
                                 + 1.11438 w_k1_DIS_s017
                                 + 1.06988 w_k1_FDX_s017
                                 + 1.15091 w_k1_IBM_s017
                                 + 1.21894 w_k1_INTC_s017
                                 + 1.14849 w_k1_KO_s017 + 1.46059 w_k1_MAT_s017
                                 + 0.953256 w_k1_MSFT_s017
                                 + 1.02261 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.33429 w_k1_ADM_s017
                                 + 1.27623 w_k1_AXP_s017
                                 + 1.61282 w_k1_DIS_s017
                                 + 1.23437 w_k1_FDX_s017
                                 + 1.41734 w_k1_IBM_s017
                                 + 1.16054 w_k1_INTC_s017
                                 + 1.33538 w_k1_KO_s017 + 1.70433 w_k1_MAT_s017
                                 + 1.11622 w_k1_MSFT_s017
                                 + 1.30176 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.34168 w_k1_ADM_s017
                                 + 1.04462 w_k1_AXP_s017
                                 + 0.867285 w_k1_DIS_s017
                                 + 0.878958 w_k1_FDX_s017
                                 + 1.37916 w_k1_IBM_s017
                                 + 0.986402 w_k1_INTC_s017
                                 + 0.986345 w_k1_KO_s017
                                 + 1.25713 w_k1_MAT_s017
                                 + 0.850849 w_k1_MSFT_s017
                                 + 1.14728 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.04556 w_k1_ADM_s017
                                 + 0.976652 w_k1_AXP_s017
                                 + 0.822876 w_k1_DIS_s017
                                 + 0.868686 w_k1_FDX_s017
                                 + 0.981026 w_k1_IBM_s017
                                 + 0.938747 w_k1_INTC_s017
                                 + 0.905159 w_k1_KO_s017
                                 + 1.00322 w_k1_MAT_s017
                                 + 0.817925 w_k1_MSFT_s017
                                 + 0.878167 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.981585 w_k1_ADM_s100
                                 + 0.793008 w_k1_AXP_s100
                                 + 1.19802 w_k1_DIS_s100
                                 + 0.969559 w_k1_FDX_s100
                                 + 0.935144 w_k1_IBM_s100
                                 + 1.11462 w_k1_INTC_s100
                                 + 1.08361 w_k1_KO_s100
                                 + 0.621935 w_k1_MAT_s100
                                 + 1.21193 w_k1_MSFT_s100
                                 + 1.05718 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.04212 w_k1_ADM_s100
                                 + 1.14042 w_k1_AXP_s100
                                 + 1.47059 w_k1_DIS_s100
                                 + 1.07817 w_k1_FDX_s100
                                 + 1.39924 w_k1_IBM_s100
                                 + 1.06017 w_k1_INTC_s100
                                 + 1.33856 w_k1_KO_s100 + 1.37451 w_k1_MAT_s100
                                 + 1.20651 w_k1_MSFT_s100
                                 + 1.17166 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.0517 w_k1_ADM_s100
                                 + 0.923554 w_k1_AXP_s100
                                 + 0.979911 w_k1_DIS_s100
                                 + 1.08982 w_k1_FDX_s100
                                 + 0.80854 w_k1_IBM_s100
                                 + 0.988729 w_k1_INTC_s100
                                 + 0.908856 w_k1_KO_s100
                                 + 0.88112 w_k1_MAT_s100
                                 + 1.05442 w_k1_MSFT_s100
                                 + 1.05982 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.744713 w_k1_ADM_s100
                                 + 0.769518 w_k1_AXP_s100
                                 + 0.597123 w_k1_DIS_s100
                                 + 0.736016 w_k1_FDX_s100
                                 + 0.719046 w_k1_IBM_s100
                                 + 0.823393 w_k1_INTC_s100
                                 + 0.676223 w_k1_KO_s100
                                 + 0.726962 w_k1_MAT_s100
                                 + 0.815094 w_k1_MSFT_s100
                                 + 0.888373 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.794878 w_k1_ADM_s100
                                 + 0.901185 w_k1_AXP_s100
                                 + 0.771761 w_k1_DIS_s100
                                 + 0.817959 w_k1_FDX_s100
                                 + 1.04056 w_k1_IBM_s100
                                 + 0.983717 w_k1_INTC_s100
                                 + 0.935505 w_k1_KO_s100
                                 + 0.718827 w_k1_MAT_s100
                                 + 0.854567 w_k1_MSFT_s100
                                 + 1.00023 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.859894 w_k1_ADM_s100
                                 + 1.01982 w_k1_AXP_s100
                                 + 1.01244 w_k1_DIS_s100
                                 + 0.980995 w_k1_FDX_s100
                                 + 1.4274 w_k1_IBM_s100 + 1.2697 w_k1_INTC_s100
                                 + 1.11871 w_k1_KO_s100 + 1.13639 w_k1_MAT_s100
                                 + 1.0545 w_k1_MSFT_s100 + 1.10778 w_k1_PG_s100
                                 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.0503 w_k1_ADM_s100
                                 + 1.25864 w_k1_AXP_s100
                                 + 0.913149 w_k1_DIS_s100
                                 + 0.981017 w_k1_FDX_s100
                                 + 1.23343 w_k1_IBM_s100
                                 + 1.0601 w_k1_INTC_s100 + 1.10176 w_k1_KO_s100
                                 + 1.03011 w_k1_MAT_s100
                                 + 0.918479 w_k1_MSFT_s100
                                 + 0.984868 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.39509 w_k1_ADM_s100
                                 + 1.36197 w_k1_AXP_s100
                                 + 1.56312 w_k1_DIS_s100
                                 + 1.43982 w_k1_FDX_s100
                                 + 1.00185 w_k1_IBM_s100
                                 + 1.46959 w_k1_INTC_s100
                                 + 1.24594 w_k1_KO_s100 + 1.0759 w_k1_MAT_s100
                                 + 1.39717 w_k1_MSFT_s100
                                 + 1.19291 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.912674 w_k1_ADM_s101
                                 + 0.986125 w_k1_AXP_s101
                                 + 0.715493 w_k1_DIS_s101
                                 + 0.83142 w_k1_FDX_s101
                                 + 1.02787 w_k1_IBM_s101
                                 + 0.924228 w_k1_INTC_s101
                                 + 0.874542 w_k1_KO_s101
                                 + 0.965522 w_k1_MAT_s101
                                 + 0.860284 w_k1_MSFT_s101
                                 + 1.01912 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.754599 w_k1_ADM_s101
                                 + 0.878289 w_k1_AXP_s101
                                 + 0.710803 w_k1_DIS_s101
                                 + 0.735015 w_k1_FDX_s101
                                 + 1.13787 w_k1_IBM_s101
                                 + 0.943301 w_k1_INTC_s101
                                 + 0.889068 w_k1_KO_s101
                                 + 1.06267 w_k1_MAT_s101
                                 + 0.8242 w_k1_MSFT_s101 + 0.84341 w_k1_PG_s101
                                 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.05099 w_k1_ADM_s101
                                 + 1.27478 w_k1_AXP_s101
                                 + 1.28773 w_k1_DIS_s101
                                 + 0.970309 w_k1_FDX_s101
                                 + 1.4749 w_k1_IBM_s101
                                 + 1.04809 w_k1_INTC_s101
                                 + 1.37242 w_k1_KO_s101 + 1.68121 w_k1_MAT_s101
                                 + 1.13136 w_k1_MSFT_s101
                                 + 1.01869 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.29408 w_k1_ADM_s101
                                 + 1.37267 w_k1_AXP_s101
                                 + 1.38488 w_k1_DIS_s101
                                 + 1.30483 w_k1_FDX_s101 + 0.9094 w_k1_IBM_s101
                                 + 1.05267 w_k1_INTC_s101
                                 + 1.01962 w_k1_KO_s101
                                 + 0.817234 w_k1_MAT_s101
                                 + 1.22956 w_k1_MSFT_s101
                                 + 1.15357 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.815944 w_k1_ADM_s101
                                 + 0.783186 w_k1_AXP_s101
                                 + 0.626477 w_k1_DIS_s101
                                 + 0.904287 w_k1_FDX_s101
                                 + 0.503941 w_k1_IBM_s101
                                 + 0.886954 w_k1_INTC_s101
                                 + 0.622962 w_k1_KO_s101
                                 + 0.463517 w_k1_MAT_s101
                                 + 0.917562 w_k1_MSFT_s101
                                 + 0.811287 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.982067 w_k1_ADM_s101
                                 + 1.01624 w_k1_AXP_s101
                                 + 1.04315 w_k1_DIS_s101
                                 + 1.09588 w_k1_FDX_s101
                                 + 0.743029 w_k1_IBM_s101
                                 + 1.22064 w_k1_INTC_s101
                                 + 0.96981 w_k1_KO_s101
                                 + 0.786595 w_k1_MAT_s101
                                 + 1.05089 w_k1_MSFT_s101
                                 + 1.01135 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.877775 w_k1_ADM_s101
                                 + 0.82266 w_k1_AXP_s101
                                 + 0.782514 w_k1_DIS_s101
                                 + 0.830532 w_k1_FDX_s101
                                 + 0.741681 w_k1_IBM_s101
                                 + 0.784082 w_k1_INTC_s101
                                 + 0.785933 w_k1_KO_s101
                                 + 0.607436 w_k1_MAT_s101
                                 + 0.94261 w_k1_MSFT_s101
                                 + 0.861841 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.805259 w_k1_ADM_s101
                                 + 0.911531 w_k1_AXP_s101
                                 + 1.11831 w_k1_DIS_s101
                                 + 0.93147 w_k1_FDX_s101
                                 + 1.09572 w_k1_IBM_s101
                                 + 1.01016 w_k1_INTC_s101
                                 + 1.07127 w_k1_KO_s101
                                 + 0.786141 w_k1_MAT_s101
                                 + 1.03945 w_k1_MSFT_s101 + 1.0245 w_k1_PG_s101
                                 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.968203 w_k1_ADM_s102
                                 + 0.722243 w_k1_AXP_s102
                                 + 0.707108 w_k1_DIS_s102
                                 + 0.786192 w_k1_FDX_s102
                                 + 0.754463 w_k1_IBM_s102
                                 + 0.8626 w_k1_INTC_s102
                                 + 0.731736 w_k1_KO_s102
                                 + 0.881268 w_k1_MAT_s102
                                 + 0.807531 w_k1_MSFT_s102
                                 + 0.805545 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.20324 w_k1_ADM_s102
                                 + 1.01807 w_k1_AXP_s102
                                 + 0.935627 w_k1_DIS_s102
                                 + 1.10673 w_k1_FDX_s102
                                 + 0.733966 w_k1_IBM_s102
                                 + 1.08496 w_k1_INTC_s102
                                 + 0.80261 w_k1_KO_s102 + 1.01799 w_k1_MAT_s102
                                 + 0.986508 w_k1_MSFT_s102
                                 + 0.992739 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.15621 w_k1_ADM_s102
                                 + 1.27322 w_k1_AXP_s102 + 1.0875 w_k1_DIS_s102
                                 + 1.10467 w_k1_FDX_s102
                                 + 0.99733 w_k1_IBM_s102
                                 + 1.17482 w_k1_INTC_s102
                                 + 0.859693 w_k1_KO_s102
                                 + 0.987422 w_k1_MAT_s102
                                 + 1.04743 w_k1_MSFT_s102
                                 + 1.08616 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.02318 w_k1_ADM_s102
                                 + 1.16454 w_k1_AXP_s102
                                 + 1.13745 w_k1_DIS_s102
                                 + 1.01299 w_k1_FDX_s102
                                 + 1.46489 w_k1_IBM_s102
                                 + 1.04745 w_k1_INTC_s102
                                 + 1.05209 w_k1_KO_s102 + 1.65913 w_k1_MAT_s102
                                 + 1.07305 w_k1_MSFT_s102
                                 + 0.970329 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.20907 w_k1_ADM_s102
                                 + 1.52101 w_k1_AXP_s102
                                 + 1.53793 w_k1_DIS_s102
                                 + 1.30261 w_k1_FDX_s102
                                 + 1.41781 w_k1_IBM_s102
                                 + 1.2617 w_k1_INTC_s102 + 1.22556 w_k1_KO_s102
                                 + 1.81118 w_k1_MAT_s102
                                 + 1.36563 w_k1_MSFT_s102
                                 + 1.19618 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.12397 w_k1_ADM_s102
                                 + 1.1541 w_k1_AXP_s102
                                 + 0.819842 w_k1_DIS_s102
                                 + 0.966657 w_k1_FDX_s102
                                 + 1.20182 w_k1_IBM_s102
                                 + 0.859292 w_k1_INTC_s102
                                 + 0.980616 w_k1_KO_s102
                                 + 1.88137 w_k1_MAT_s102
                                 + 0.963881 w_k1_MSFT_s102
                                 + 1.02387 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.951336 w_k1_ADM_s102
                                 + 0.895834 w_k1_AXP_s102
                                 + 1.11621 w_k1_DIS_s102
                                 + 0.97446 w_k1_FDX_s102 + 1.0354 w_k1_IBM_s102
                                 + 1.04883 w_k1_INTC_s102 + 1.0072 w_k1_KO_s102
                                 + 1.05901 w_k1_MAT_s102
                                 + 1.07085 w_k1_MSFT_s102 + 1.0474 w_k1_PG_s102
                                 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.97086 w_k1_ADM_s102
                                 + 0.81718 w_k1_AXP_s102
                                 + 0.74781 w_k1_DIS_s102
                                 + 0.791629 w_k1_FDX_s102
                                 + 1.06346 w_k1_IBM_s102
                                 + 0.973869 w_k1_INTC_s102
                                 + 0.816992 w_k1_KO_s102
                                 + 1.09485 w_k1_MAT_s102
                                 + 0.874891 w_k1_MSFT_s102
                                 + 0.893251 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.04273 w_k1_ADM_s103
                                 + 1.33517 w_k1_AXP_s103
                                 + 1.30297 w_k1_DIS_s103
                                 + 1.27699 w_k1_FDX_s103 + 1.1504 w_k1_IBM_s103
                                 + 1.33972 w_k1_INTC_s103
                                 + 1.14795 w_k1_KO_s103 + 1.11749 w_k1_MAT_s103
                                 + 0.931898 w_k1_MSFT_s103
                                 + 1.17797 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.907264 w_k1_ADM_s103
                                 + 0.888906 w_k1_AXP_s103
                                 + 0.944978 w_k1_DIS_s103
                                 + 1.02682 w_k1_FDX_s103
                                 + 0.734261 w_k1_IBM_s103
                                 + 1.12918 w_k1_INTC_s103
                                 + 1.05993 w_k1_KO_s103
                                 + 0.861178 w_k1_MAT_s103
                                 + 1.04165 w_k1_MSFT_s103
                                 + 1.05423 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.10684 w_k1_ADM_s103
                                 + 1.46719 w_k1_AXP_s103
                                 + 1.54968 w_k1_DIS_s103
                                 + 1.48125 w_k1_FDX_s103
                                 + 1.28755 w_k1_IBM_s103
                                 + 1.61056 w_k1_INTC_s103
                                 + 1.50789 w_k1_KO_s103 + 1.48082 w_k1_MAT_s103
                                 + 1.40099 w_k1_MSFT_s103
                                 + 1.39717 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.939464 w_k1_ADM_s103
                                 + 0.98057 w_k1_AXP_s103
                                 + 1.01979 w_k1_DIS_s103
                                 + 1.04119 w_k1_FDX_s103
                                 + 1.52978 w_k1_IBM_s103
                                 + 1.22449 w_k1_INTC_s103 + 1.2487 w_k1_KO_s103
                                 + 1.28488 w_k1_MAT_s103
                                 + 0.863222 w_k1_MSFT_s103
                                 + 1.14445 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.06172 w_k1_ADM_s103
                                 + 0.943431 w_k1_AXP_s103
                                 + 0.913895 w_k1_DIS_s103
                                 + 1.02369 w_k1_FDX_s103
                                 + 1.06249 w_k1_IBM_s103
                                 + 1.1004 w_k1_INTC_s103 + 1.04342 w_k1_KO_s103
                                 + 0.924875 w_k1_MAT_s103
                                 + 0.917327 w_k1_MSFT_s103
                                 + 1.19077 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.02579 w_k1_ADM_s103
                                 + 1.04529 w_k1_AXP_s103
                                 + 1.13731 w_k1_DIS_s103
                                 + 1.17849 w_k1_FDX_s103
                                 + 0.983456 w_k1_IBM_s103
                                 + 1.335 w_k1_INTC_s103 + 1.11924 w_k1_KO_s103
                                 + 0.705852 w_k1_MAT_s103
                                 + 1.16071 w_k1_MSFT_s103
                                 + 1.15372 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.47693 w_k1_ADM_s103
                                 + 1.83115 w_k1_AXP_s103
                                 + 1.59594 w_k1_DIS_s103
                                 + 1.52177 w_k1_FDX_s103
                                 + 1.73611 w_k1_IBM_s103
                                 + 1.30904 w_k1_INTC_s103
                                 + 1.48796 w_k1_KO_s103 + 1.75017 w_k1_MAT_s103
                                 + 1.12365 w_k1_MSFT_s103
                                 + 1.47564 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.670233 w_k1_ADM_s103
                                 + 0.616132 w_k1_AXP_s103
                                 + 0.4964 w_k1_DIS_s103
                                 + 0.719038 w_k1_FDX_s103
                                 + 0.940229 w_k1_IBM_s103
                                 + 0.967604 w_k1_INTC_s103
                                 + 0.741248 w_k1_KO_s103
                                 + 0.670623 w_k1_MAT_s103
                                 + 0.732191 w_k1_MSFT_s103
                                 + 0.756007 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.912349 w_k1_ADM_s104
                                 + 0.693583 w_k1_AXP_s104
                                 + 0.615647 w_k1_DIS_s104
                                 + 0.769254 w_k1_FDX_s104
                                 + 0.605964 w_k1_IBM_s104
                                 + 0.750801 w_k1_INTC_s104
                                 + 0.737581 w_k1_KO_s104
                                 + 0.539158 w_k1_MAT_s104
                                 + 0.777264 w_k1_MSFT_s104
                                 + 0.784565 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.18527 w_k1_ADM_s104
                                 + 1.22023 w_k1_AXP_s104
                                 + 1.23643 w_k1_DIS_s104
                                 + 1.15634 w_k1_FDX_s104
                                 + 1.10381 w_k1_IBM_s104
                                 + 1.18503 w_k1_INTC_s104 + 1.1405 w_k1_KO_s104
                                 + 1.54533 w_k1_MAT_s104
                                 + 1.06532 w_k1_MSFT_s104
                                 + 1.06453 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.17659 w_k1_ADM_s104
                                 + 1.63209 w_k1_AXP_s104
                                 + 1.30926 w_k1_DIS_s104
                                 + 1.11698 w_k1_FDX_s104
                                 + 1.64046 w_k1_IBM_s104
                                 + 0.88159 w_k1_INTC_s104
                                 + 1.58112 w_k1_KO_s104 + 2.50556 w_k1_MAT_s104
                                 + 0.986402 w_k1_MSFT_s104
                                 + 1.22235 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.822566 w_k1_ADM_s104
                                 + 0.917577 w_k1_AXP_s104
                                 + 0.925269 w_k1_DIS_s104
                                 + 0.807026 w_k1_FDX_s104
                                 + 1.07094 w_k1_IBM_s104
                                 + 0.861308 w_k1_INTC_s104
                                 + 0.977867 w_k1_KO_s104
                                 + 0.900653 w_k1_MAT_s104
                                 + 0.957762 w_k1_MSFT_s104
                                 + 0.905708 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.00654 w_k1_ADM_s104
                                 + 0.82377 w_k1_AXP_s104
                                 + 0.723729 w_k1_DIS_s104
                                 + 0.948805 w_k1_FDX_s104
                                 + 0.832155 w_k1_IBM_s104
                                 + 0.830071 w_k1_INTC_s104
                                 + 0.94727 w_k1_KO_s104 + 1.05256 w_k1_MAT_s104
                                 + 0.946527 w_k1_MSFT_s104
                                 + 0.995889 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.47037 w_k1_ADM_s104
                                 + 1.30028 w_k1_AXP_s104
                                 + 1.54605 w_k1_DIS_s104
                                 + 1.41267 w_k1_FDX_s104
                                 + 0.975374 w_k1_IBM_s104
                                 + 1.10656 w_k1_INTC_s104
                                 + 1.20146 w_k1_KO_s104 + 1.17918 w_k1_MAT_s104
                                 + 1.26462 w_k1_MSFT_s104
                                 + 1.10698 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.956963 w_k1_ADM_s104
                                 + 0.846138 w_k1_AXP_s104
                                 + 1.05429 w_k1_DIS_s104
                                 + 0.924965 w_k1_FDX_s104
                                 + 1.36851 w_k1_IBM_s104
                                 + 1.03964 w_k1_INTC_s104 + 1.3241 w_k1_KO_s104
                                 + 1.38785 w_k1_MAT_s104
                                 + 1.04213 w_k1_MSFT_s104
                                 + 1.02123 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.937598 w_k1_ADM_s104
                                 + 0.945106 w_k1_AXP_s104
                                 + 0.918795 w_k1_DIS_s104
                                 + 0.986318 w_k1_FDX_s104
                                 + 1.11724 w_k1_IBM_s104
                                 + 0.84399 w_k1_INTC_s104
                                 + 1.06984 w_k1_KO_s104 + 1.46283 w_k1_MAT_s104
                                 + 0.954409 w_k1_MSFT_s104
                                 + 0.942017 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.00041 w_k1_ADM_s105
                                 + 1.03831 w_k1_AXP_s105
                                 + 1.21398 w_k1_DIS_s105 + 1.1031 w_k1_FDX_s105
                                 + 0.784851 w_k1_IBM_s105
                                 + 1.2156 w_k1_INTC_s105 + 1.12193 w_k1_KO_s105
                                 + 0.855904 w_k1_MAT_s105
                                 + 1.28523 w_k1_MSFT_s105 + 1.0553 w_k1_PG_s105
                                 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.799346 w_k1_ADM_s105
                                 + 0.727724 w_k1_AXP_s105
                                 + 0.608733 w_k1_DIS_s105
                                 + 0.804204 w_k1_FDX_s105
                                 + 0.570776 w_k1_IBM_s105
                                 + 0.856785 w_k1_INTC_s105
                                 + 0.684526 w_k1_KO_s105
                                 + 0.493054 w_k1_MAT_s105
                                 + 0.723998 w_k1_MSFT_s105
                                 + 0.757964 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.03843 w_k1_ADM_s105
                                 + 1.1569 w_k1_AXP_s105 + 1.09237 w_k1_DIS_s105
                                 + 0.966658 w_k1_FDX_s105
                                 + 1.28231 w_k1_IBM_s105
                                 + 1.04765 w_k1_INTC_s105
                                 + 1.33176 w_k1_KO_s105 + 1.33431 w_k1_MAT_s105
                                 + 0.942507 w_k1_MSFT_s105
                                 + 1.0822 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.10448 w_k1_ADM_s105
                                 + 1.27612 w_k1_AXP_s105
                                 + 0.984427 w_k1_DIS_s105
                                 + 1.22961 w_k1_FDX_s105
                                 + 0.721872 w_k1_IBM_s105
                                 + 1.1801 w_k1_INTC_s105
                                 + 0.919355 w_k1_KO_s105
                                 + 1.04626 w_k1_MAT_s105
                                 + 1.09895 w_k1_MSFT_s105
                                 + 1.01448 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.996194 w_k1_ADM_s105
                                 + 1.27374 w_k1_AXP_s105
                                 + 0.962351 w_k1_DIS_s105
                                 + 0.873936 w_k1_FDX_s105
                                 + 1.84246 w_k1_IBM_s105
                                 + 1.10218 w_k1_INTC_s105
                                 + 1.36566 w_k1_KO_s105 + 2.35077 w_k1_MAT_s105
                                 + 0.965157 w_k1_MSFT_s105
                                 + 1.18898 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.864016 w_k1_ADM_s105
                                 + 0.757069 w_k1_AXP_s105
                                 + 0.615405 w_k1_DIS_s105
                                 + 0.683581 w_k1_FDX_s105
                                 + 1.01347 w_k1_IBM_s105
                                 + 0.842627 w_k1_INTC_s105
                                 + 0.886669 w_k1_KO_s105
                                 + 0.880713 w_k1_MAT_s105
                                 + 0.707121 w_k1_MSFT_s105
                                 + 0.848678 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.3521 w_k1_ADM_s105
                                 + 1.35099 w_k1_AXP_s105
                                 + 1.53711 w_k1_DIS_s105
                                 + 1.26702 w_k1_FDX_s105
                                 + 1.15989 w_k1_IBM_s105
                                 + 1.34792 w_k1_INTC_s105
                                 + 1.30892 w_k1_KO_s105 + 1.06251 w_k1_MAT_s105
                                 + 1.38285 w_k1_MSFT_s105
                                 + 1.22623 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.01866 w_k1_ADM_s105
                                 + 0.907503 w_k1_AXP_s105
                                 + 0.812454 w_k1_DIS_s105
                                 + 0.861413 w_k1_FDX_s105
                                 + 0.841459 w_k1_IBM_s105
                                 + 1.01048 w_k1_INTC_s105
                                 + 1.01585 w_k1_KO_s105 + 1.0373 w_k1_MAT_s105
                                 + 0.87308 w_k1_MSFT_s105
                                 + 0.860265 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.336 w_k1_ADM_s106
                                 + 1.47675 w_k1_AXP_s106
                                 + 1.52704 w_k1_DIS_s106
                                 + 1.25708 w_k1_FDX_s106
                                 + 0.810093 w_k1_IBM_s106
                                 + 1.14714 w_k1_INTC_s106 + 1.0463 w_k1_KO_s106
                                 + 0.966563 w_k1_MAT_s106
                                 + 1.2349 w_k1_MSFT_s106 + 1.23648 w_k1_PG_s106
                                 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.792904 w_k1_ADM_s106
                                 + 0.807067 w_k1_AXP_s106
                                 + 0.813775 w_k1_DIS_s106
                                 + 0.886376 w_k1_FDX_s106
                                 + 0.72049 w_k1_IBM_s106
                                 + 0.938755 w_k1_INTC_s106
                                 + 0.758837 w_k1_KO_s106
                                 + 0.760037 w_k1_MAT_s106
                                 + 0.918415 w_k1_MSFT_s106
                                 + 0.910603 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.01383 w_k1_ADM_s106
                                 + 0.901827 w_k1_AXP_s106
                                 + 0.975384 w_k1_DIS_s106
                                 + 1.00971 w_k1_FDX_s106
                                 + 0.896294 w_k1_IBM_s106
                                 + 1.13343 w_k1_INTC_s106
                                 + 0.96269 w_k1_KO_s106
                                 + 0.937675 w_k1_MAT_s106
                                 + 1.03257 w_k1_MSFT_s106
                                 + 1.02875 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.14209 w_k1_ADM_s106
                                 + 0.791356 w_k1_AXP_s106
                                 + 1.17299 w_k1_DIS_s106
                                 + 0.902287 w_k1_FDX_s106
                                 + 1.46809 w_k1_IBM_s106
                                 + 1.11812 w_k1_INTC_s106
                                 + 1.29923 w_k1_KO_s106 + 1.0894 w_k1_MAT_s106
                                 + 1.09324 w_k1_MSFT_s106
                                 + 1.11383 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.14379 w_k1_ADM_s106
                                 + 0.946458 w_k1_AXP_s106
                                 + 0.649625 w_k1_DIS_s106
                                 + 0.779203 w_k1_FDX_s106
                                 + 0.820605 w_k1_IBM_s106
                                 + 0.860693 w_k1_INTC_s106
                                 + 0.819894 w_k1_KO_s106
                                 + 0.841579 w_k1_MAT_s106
                                 + 0.714969 w_k1_MSFT_s106
                                 + 0.844129 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.25396 w_k1_ADM_s106
                                 + 1.22213 w_k1_AXP_s106
                                 + 1.12993 w_k1_DIS_s106
                                 + 1.12328 w_k1_FDX_s106
                                 + 0.939429 w_k1_IBM_s106
                                 + 1.23497 w_k1_INTC_s106 + 1.0512 w_k1_KO_s106
                                 + 1.06316 w_k1_MAT_s106
                                 + 1.11081 w_k1_MSFT_s106
                                 + 1.13863 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.816797 w_k1_ADM_s106
                                 + 1.1761 w_k1_AXP_s106 + 0.94884 w_k1_DIS_s106
                                 + 0.848224 w_k1_FDX_s106
                                 + 1.32673 w_k1_IBM_s106
                                 + 1.02646 w_k1_INTC_s106
                                 + 1.18131 w_k1_KO_s106 + 1.71384 w_k1_MAT_s106
                                 + 0.89682 w_k1_MSFT_s106
                                 + 1.01551 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.29791 w_k1_ADM_s106
                                 + 1.58559 w_k1_AXP_s106 + 1.4195 w_k1_DIS_s106
                                 + 1.22395 w_k1_FDX_s106 + 1.2865 w_k1_IBM_s106
                                 + 1.28155 w_k1_INTC_s106
                                 + 1.20336 w_k1_KO_s106 + 1.32929 w_k1_MAT_s106
                                 + 1.18925 w_k1_MSFT_s106
                                 + 1.18014 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.11137 w_k1_ADM_s107
                                 + 0.920645 w_k1_AXP_s107
                                 + 1.04175 w_k1_DIS_s107
                                 + 1.18163 w_k1_FDX_s107
                                 + 0.696401 w_k1_IBM_s107
                                 + 1.15983 w_k1_INTC_s107
                                 + 0.945544 w_k1_KO_s107
                                 + 0.609636 w_k1_MAT_s107
                                 + 1.14766 w_k1_MSFT_s107
                                 + 0.895667 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.798721 w_k1_ADM_s107
                                 + 0.856837 w_k1_AXP_s107
                                 + 0.775342 w_k1_DIS_s107
                                 + 0.889523 w_k1_FDX_s107
                                 + 0.702992 w_k1_IBM_s107
                                 + 0.951582 w_k1_INTC_s107
                                 + 0.682182 w_k1_KO_s107
                                 + 0.753077 w_k1_MAT_s107
                                 + 0.91678 w_k1_MSFT_s107
                                 + 0.818201 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.812615 w_k1_ADM_s107
                                 + 0.800965 w_k1_AXP_s107
                                 + 0.784515 w_k1_DIS_s107
                                 + 0.769123 w_k1_FDX_s107
                                 + 1.04526 w_k1_IBM_s107
                                 + 0.746198 w_k1_INTC_s107
                                 + 0.902038 w_k1_KO_s107
                                 + 1.06704 w_k1_MAT_s107
                                 + 0.860179 w_k1_MSFT_s107
                                 + 0.834494 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.15258 w_k1_ADM_s107
                                 + 1.16762 w_k1_AXP_s107
                                 + 1.17352 w_k1_DIS_s107
                                 + 1.15135 w_k1_FDX_s107
                                 + 1.02599 w_k1_IBM_s107
                                 + 0.97946 w_k1_INTC_s107
                                 + 1.01455 w_k1_KO_s107
                                 + 0.858694 w_k1_MAT_s107
                                 + 1.04608 w_k1_MSFT_s107
                                 + 0.96577 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.28405 w_k1_ADM_s107
                                 + 1.39266 w_k1_AXP_s107
                                 + 1.38098 w_k1_DIS_s107
                                 + 1.28268 w_k1_FDX_s107
                                 + 1.64736 w_k1_IBM_s107
                                 + 0.95227 w_k1_INTC_s107
                                 + 1.25474 w_k1_KO_s107 + 1.88946 w_k1_MAT_s107
                                 + 1.13317 w_k1_MSFT_s107
                                 + 1.15444 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.34359 w_k1_ADM_s107
                                 + 1.30895 w_k1_AXP_s107
                                 + 1.56259 w_k1_DIS_s107 + 1.3685 w_k1_FDX_s107
                                 + 0.970266 w_k1_IBM_s107
                                 + 1.27815 w_k1_INTC_s107
                                 + 1.13738 w_k1_KO_s107 + 1.12881 w_k1_MAT_s107
                                 + 1.45171 w_k1_MSFT_s107
                                 + 1.21426 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.948254 w_k1_ADM_s107
                                 + 1.11393 w_k1_AXP_s107
                                 + 0.945663 w_k1_DIS_s107
                                 + 0.89014 w_k1_FDX_s107
                                 + 1.34268 w_k1_IBM_s107
                                 + 0.886824 w_k1_INTC_s107
                                 + 0.876688 w_k1_KO_s107
                                 + 1.50405 w_k1_MAT_s107
                                 + 0.875122 w_k1_MSFT_s107
                                 + 0.958498 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.0761 w_k1_ADM_s107
                                 + 1.01614 w_k1_AXP_s107
                                 + 1.29308 w_k1_DIS_s107
                                 + 1.05658 w_k1_FDX_s107
                                 + 1.19223 w_k1_IBM_s107
                                 + 1.02452 w_k1_INTC_s107
                                 + 1.09421 w_k1_KO_s107 + 1.40706 w_k1_MAT_s107
                                 + 1.25181 w_k1_MSFT_s107
                                 + 0.979011 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.74373 w_k1_ADM_s110
                                 + 0.685707 w_k1_AXP_s110
                                 + 0.60755 w_k1_DIS_s110
                                 + 0.60879 w_k1_FDX_s110
                                 + 0.99216 w_k1_IBM_s110
                                 + 0.520803 w_k1_INTC_s110
                                 + 0.726019 w_k1_KO_s110
                                 + 0.821752 w_k1_MAT_s110
                                 + 0.621652 w_k1_MSFT_s110
                                 + 0.756893 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.712086 w_k1_ADM_s110
                                 + 0.656941 w_k1_AXP_s110
                                 + 0.652571 w_k1_DIS_s110
                                 + 0.690902 w_k1_FDX_s110
                                 + 0.575421 w_k1_IBM_s110
                                 + 0.734703 w_k1_INTC_s110
                                 + 0.642542 w_k1_KO_s110
                                 + 0.692756 w_k1_MAT_s110
                                 + 0.751555 w_k1_MSFT_s110
                                 + 0.826548 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.1516 w_k1_ADM_s110
                                 + 1.07574 w_k1_AXP_s110
                                 + 0.966924 w_k1_DIS_s110
                                 + 1.00583 w_k1_FDX_s110
                                 + 1.03572 w_k1_IBM_s110
                                 + 0.888609 w_k1_INTC_s110
                                 + 1.09182 w_k1_KO_s110
                                 + 0.973995 w_k1_MAT_s110
                                 + 0.936181 w_k1_MSFT_s110
                                 + 0.989589 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.52637 w_k1_ADM_s110
                                 + 1.77371 w_k1_AXP_s110
                                 + 1.56972 w_k1_DIS_s110
                                 + 1.32242 w_k1_FDX_s110
                                 + 1.06265 w_k1_IBM_s110
                                 + 1.03379 w_k1_INTC_s110
                                 + 1.21572 w_k1_KO_s110 + 1.59615 w_k1_MAT_s110
                                 + 1.138 w_k1_MSFT_s110 + 1.24176 w_k1_PG_s110
                                 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.06513 w_k1_ADM_s110
                                 + 1.04724 w_k1_AXP_s110 + 1.3143 w_k1_DIS_s110
                                 + 1.05242 w_k1_FDX_s110
                                 + 1.41734 w_k1_IBM_s110
                                 + 0.998367 w_k1_INTC_s110
                                 + 1.32151 w_k1_KO_s110 + 1.23841 w_k1_MAT_s110
                                 + 1.07439 w_k1_MSFT_s110
                                 + 1.19295 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.06551 w_k1_ADM_s110
                                 + 0.909262 w_k1_AXP_s110
                                 + 0.717301 w_k1_DIS_s110
                                 + 0.824212 w_k1_FDX_s110
                                 + 0.864079 w_k1_IBM_s110
                                 + 0.692219 w_k1_INTC_s110
                                 + 0.899262 w_k1_KO_s110
                                 + 0.783087 w_k1_MAT_s110
                                 + 0.777925 w_k1_MSFT_s110
                                 + 0.960868 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.23858 w_k1_ADM_s110
                                 + 1.16866 w_k1_AXP_s110
                                 + 1.06764 w_k1_DIS_s110
                                 + 1.17497 w_k1_FDX_s110
                                 + 0.715442 w_k1_IBM_s110
                                 + 0.775213 w_k1_INTC_s110
                                 + 0.992513 w_k1_KO_s110
                                 + 0.984788 w_k1_MAT_s110
                                 + 0.964913 w_k1_MSFT_s110
                                 + 1.09657 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.03757 w_k1_ADM_s110
                                 + 0.988479 w_k1_AXP_s110
                                 + 1.04075 w_k1_DIS_s110
                                 + 0.905528 w_k1_FDX_s110
                                 + 1.08585 w_k1_IBM_s110
                                 + 0.926421 w_k1_INTC_s110
                                 + 1.28498 w_k1_KO_s110
                                 + 0.926781 w_k1_MAT_s110
                                 + 1.03397 w_k1_MSFT_s110
                                 + 1.08885 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.23724 w_k1_ADM_s111
                                 + 1.47767 w_k1_AXP_s111
                                 + 1.66581 w_k1_DIS_s111
                                 + 1.36726 w_k1_FDX_s111
                                 + 1.13201 w_k1_IBM_s111
                                 + 1.29094 w_k1_INTC_s111
                                 + 1.49712 w_k1_KO_s111 + 1.52682 w_k1_MAT_s111
                                 + 1.44119 w_k1_MSFT_s111
                                 + 1.23575 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.819163 w_k1_ADM_s111
                                 + 0.92003 w_k1_AXP_s111
                                 + 0.801542 w_k1_DIS_s111
                                 + 0.914611 w_k1_FDX_s111
                                 + 0.648517 w_k1_IBM_s111
                                 + 0.885042 w_k1_INTC_s111
                                 + 0.844541 w_k1_KO_s111
                                 + 0.837226 w_k1_MAT_s111
                                 + 0.979293 w_k1_MSFT_s111
                                 + 0.933976 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.733703 w_k1_ADM_s111
                                 + 0.689516 w_k1_AXP_s111
                                 + 1.01028 w_k1_DIS_s111
                                 + 0.955075 w_k1_FDX_s111
                                 + 0.835033 w_k1_IBM_s111
                                 + 0.938077 w_k1_INTC_s111
                                 + 0.96432 w_k1_KO_s111 + 0.6933 w_k1_MAT_s111
                                 + 0.955957 w_k1_MSFT_s111
                                 + 0.888609 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.95537 w_k1_ADM_s111
                                 + 1.11614 w_k1_AXP_s111
                                 + 1.24271 w_k1_DIS_s111
                                 + 1.01646 w_k1_FDX_s111
                                 + 1.24194 w_k1_IBM_s111
                                 + 1.22325 w_k1_INTC_s111
                                 + 1.25794 w_k1_KO_s111 + 1.21346 w_k1_MAT_s111
                                 + 1.07944 w_k1_MSFT_s111
                                 + 1.01096 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.60428 w_k1_ADM_s111
                                 + 0.570209 w_k1_AXP_s111
                                 + 0.625891 w_k1_DIS_s111
                                 + 0.688521 w_k1_FDX_s111
                                 + 0.734539 w_k1_IBM_s111
                                 + 0.778142 w_k1_INTC_s111
                                 + 0.711278 w_k1_KO_s111
                                 + 0.863436 w_k1_MAT_s111
                                 + 0.766407 w_k1_MSFT_s111
                                 + 0.772431 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.689041 w_k1_ADM_s111
                                 + 0.840624 w_k1_AXP_s111
                                 + 0.651245 w_k1_DIS_s111
                                 + 0.729996 w_k1_FDX_s111
                                 + 0.895901 w_k1_IBM_s111
                                 + 0.689787 w_k1_INTC_s111
                                 + 0.797353 w_k1_KO_s111
                                 + 1.09837 w_k1_MAT_s111
                                 + 0.793222 w_k1_MSFT_s111
                                 + 0.767056 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.756595 w_k1_ADM_s111
                                 + 0.758741 w_k1_AXP_s111
                                 + 0.844708 w_k1_DIS_s111
                                 + 0.859639 w_k1_FDX_s111
                                 + 1.0733 w_k1_IBM_s111
                                 + 0.867988 w_k1_INTC_s111
                                 + 1.05898 w_k1_KO_s111 + 1.22385 w_k1_MAT_s111
                                 + 0.914556 w_k1_MSFT_s111
                                 + 0.987138 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.24189 w_k1_ADM_s111
                                 + 1.28408 w_k1_AXP_s111
                                 + 1.01444 w_k1_DIS_s111
                                 + 1.20244 w_k1_FDX_s111
                                 + 0.819151 w_k1_IBM_s111
                                 + 0.97192 w_k1_INTC_s111
                                 + 0.962338 w_k1_KO_s111
                                 + 0.915186 w_k1_MAT_s111
                                 + 1.01806 w_k1_MSFT_s111
                                 + 1.08725 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.34935 w_k1_ADM_s112
                                 + 0.915449 w_k1_AXP_s112
                                 + 1.08682 w_k1_DIS_s112 + 1.1677 w_k1_FDX_s112
                                 + 0.843975 w_k1_IBM_s112
                                 + 0.938323 w_k1_INTC_s112
                                 + 1.0324 w_k1_KO_s112 + 0.860158 w_k1_MAT_s112
                                 + 1.17236 w_k1_MSFT_s112
                                 + 1.03768 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.67469 w_k1_ADM_s112
                                 + 1.16896 w_k1_AXP_s112
                                 + 1.45208 w_k1_DIS_s112
                                 + 1.25314 w_k1_FDX_s112
                                 + 0.926807 w_k1_IBM_s112
                                 + 0.849113 w_k1_INTC_s112
                                 + 1.21224 w_k1_KO_s112 + 1.05577 w_k1_MAT_s112
                                 + 1.28894 w_k1_MSFT_s112
                                 + 1.10211 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.15775 w_k1_ADM_s112
                                 + 1.32586 w_k1_AXP_s112
                                 + 1.43105 w_k1_DIS_s112
                                 + 1.19956 w_k1_FDX_s112
                                 + 1.35235 w_k1_IBM_s112
                                 + 0.879531 w_k1_INTC_s112
                                 + 1.29274 w_k1_KO_s112 + 1.72999 w_k1_MAT_s112
                                 + 1.11896 w_k1_MSFT_s112 + 1.2751 w_k1_PG_s112
                                 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.45998 w_k1_ADM_s112
                                 + 1.17722 w_k1_AXP_s112
                                 + 1.21336 w_k1_DIS_s112
                                 + 1.30898 w_k1_FDX_s112
                                 + 0.646149 w_k1_IBM_s112
                                 + 1.11706 w_k1_INTC_s112
                                 + 1.06117 w_k1_KO_s112
                                 + 0.858016 w_k1_MAT_s112
                                 + 1.30865 w_k1_MSFT_s112
                                 + 1.03555 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.25036 w_k1_ADM_s112
                                 + 1.61621 w_k1_AXP_s112
                                 + 1.28148 w_k1_DIS_s112 + 1.3171 w_k1_FDX_s112
                                 + 1.09814 w_k1_IBM_s112
                                 + 0.987204 w_k1_INTC_s112
                                 + 0.983597 w_k1_KO_s112
                                 + 1.53824 w_k1_MAT_s112
                                 + 1.02578 w_k1_MSFT_s112
                                 + 1.03234 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.08201 w_k1_ADM_s112
                                 + 0.746032 w_k1_AXP_s112
                                 + 0.724478 w_k1_DIS_s112
                                 + 0.801655 w_k1_FDX_s112
                                 + 0.987782 w_k1_IBM_s112
                                 + 0.734232 w_k1_INTC_s112
                                 + 0.83791 w_k1_KO_s112
                                 + 0.942353 w_k1_MAT_s112
                                 + 0.911732 w_k1_MSFT_s112
                                 + 0.794415 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.980798 w_k1_ADM_s112
                                 + 0.999791 w_k1_AXP_s112
                                 + 0.850318 w_k1_DIS_s112
                                 + 0.929484 w_k1_FDX_s112
                                 + 0.803068 w_k1_IBM_s112
                                 + 0.826669 w_k1_INTC_s112
                                 + 0.837616 w_k1_KO_s112
                                 + 0.915999 w_k1_MAT_s112
                                 + 0.947443 w_k1_MSFT_s112
                                 + 0.919102 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.996554 w_k1_ADM_s112
                                 + 0.837487 w_k1_AXP_s112
                                 + 0.884076 w_k1_DIS_s112
                                 + 0.829157 w_k1_FDX_s112
                                 + 1.40641 w_k1_IBM_s112
                                 + 0.736546 w_k1_INTC_s112
                                 + 1.04758 w_k1_KO_s112 + 1.35997 w_k1_MAT_s112
                                 + 0.82877 w_k1_MSFT_s112
                                 + 0.790918 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.890304 w_k1_ADM_s113
                                 + 0.731109 w_k1_AXP_s113
                                 + 1.14446 w_k1_DIS_s113
                                 + 1.00413 w_k1_FDX_s113
                                 + 0.786121 w_k1_IBM_s113
                                 + 0.773017 w_k1_INTC_s113
                                 + 0.958509 w_k1_KO_s113
                                 + 0.935623 w_k1_MAT_s113
                                 + 1.0215 w_k1_MSFT_s113 + 1.19508 w_k1_PG_s113
                                 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.782005 w_k1_ADM_s113
                                 + 0.761258 w_k1_AXP_s113
                                 + 0.556654 w_k1_DIS_s113
                                 + 0.682451 w_k1_FDX_s113
                                 + 0.935471 w_k1_IBM_s113
                                 + 0.655552 w_k1_INTC_s113
                                 + 0.764722 w_k1_KO_s113
                                 + 1.03381 w_k1_MAT_s113
                                 + 0.608282 w_k1_MSFT_s113
                                 + 0.878079 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.22428 w_k1_ADM_s113
                                 + 1.14731 w_k1_AXP_s113
                                 + 1.09117 w_k1_DIS_s113
                                 + 1.07315 w_k1_FDX_s113
                                 + 1.33176 w_k1_IBM_s113
                                 + 0.989572 w_k1_INTC_s113
                                 + 1.33511 w_k1_KO_s113 + 1.57287 w_k1_MAT_s113
                                 + 0.975654 w_k1_MSFT_s113
                                 + 1.21203 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.42972 w_k1_ADM_s113
                                 + 1.26625 w_k1_AXP_s113
                                 + 1.55654 w_k1_DIS_s113
                                 + 1.17752 w_k1_FDX_s113
                                 + 1.74413 w_k1_IBM_s113
                                 + 0.943784 w_k1_INTC_s113
                                 + 1.60255 w_k1_KO_s113 + 2.09664 w_k1_MAT_s113
                                 + 1.037 w_k1_MSFT_s113 + 1.52733 w_k1_PG_s113
                                 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.781161 w_k1_ADM_s113
                                 + 1.20288 w_k1_AXP_s113
                                 + 0.881747 w_k1_DIS_s113
                                 + 0.97777 w_k1_FDX_s113
                                 + 0.82382 w_k1_IBM_s113
                                 + 1.11787 w_k1_INTC_s113
                                 + 0.915875 w_k1_KO_s113
                                 + 1.20851 w_k1_MAT_s113
                                 + 0.964708 w_k1_MSFT_s113
                                 + 1.21018 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.994709 w_k1_ADM_s113
                                 + 0.761113 w_k1_AXP_s113
                                 + 0.680011 w_k1_DIS_s113
                                 + 0.807958 w_k1_FDX_s113
                                 + 0.680558 w_k1_IBM_s113
                                 + 0.67324 w_k1_INTC_s113
                                 + 0.808128 w_k1_KO_s113
                                 + 0.959219 w_k1_MAT_s113
                                 + 0.814186 w_k1_MSFT_s113
                                 + 0.964566 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.925508 w_k1_ADM_s113
                                 + 1.029 w_k1_AXP_s113 + 0.789973 w_k1_DIS_s113
                                 + 0.956991 w_k1_FDX_s113
                                 + 1.10105 w_k1_IBM_s113
                                 + 0.792496 w_k1_INTC_s113
                                 + 0.978857 w_k1_KO_s113
                                 + 1.48306 w_k1_MAT_s113
                                 + 0.949396 w_k1_MSFT_s113
                                 + 1.09308 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.30804 w_k1_ADM_s113
                                 + 1.52614 w_k1_AXP_s113
                                 + 1.50622 w_k1_DIS_s113 + 1.3923 w_k1_FDX_s113
                                 + 1.04756 w_k1_IBM_s113
                                 + 0.988824 w_k1_INTC_s113
                                 + 1.21158 w_k1_KO_s113 + 1.48899 w_k1_MAT_s113
                                 + 1.04197 w_k1_MSFT_s113
                                 + 1.16218 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.29752 w_k1_ADM_s114
                                 + 1.40223 w_k1_AXP_s114
                                 + 1.51979 w_k1_DIS_s114
                                 + 1.12525 w_k1_FDX_s114
                                 + 1.84115 w_k1_IBM_s114
                                 + 1.05219 w_k1_INTC_s114
                                 + 1.53663 w_k1_KO_s114 + 1.77227 w_k1_MAT_s114
                                 + 1.2591 w_k1_MSFT_s114 + 1.21552 w_k1_PG_s114
                                 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.727715 w_k1_ADM_s114
                                 + 0.777251 w_k1_AXP_s114
                                 + 0.952557 w_k1_DIS_s114
                                 + 0.768919 w_k1_FDX_s114
                                 + 1.11699 w_k1_IBM_s114
                                 + 0.813376 w_k1_INTC_s114
                                 + 0.863658 w_k1_KO_s114
                                 + 1.03385 w_k1_MAT_s114
                                 + 0.90052 w_k1_MSFT_s114
                                 + 0.944567 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.912301 w_k1_ADM_s114
                                 + 0.866721 w_k1_AXP_s114
                                 + 1.14404 w_k1_DIS_s114
                                 + 0.961092 w_k1_FDX_s114
                                 + 1.11504 w_k1_IBM_s114
                                 + 0.747389 w_k1_INTC_s114
                                 + 1.13984 w_k1_KO_s114
                                 + 0.954782 w_k1_MAT_s114
                                 + 1.00287 w_k1_MSFT_s114
                                 + 1.05988 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.06565 w_k1_ADM_s114
                                 + 0.898646 w_k1_AXP_s114
                                 + 0.988902 w_k1_DIS_s114
                                 + 1.01996 w_k1_FDX_s114
                                 + 0.768408 w_k1_IBM_s114
                                 + 0.859414 w_k1_INTC_s114
                                 + 0.837496 w_k1_KO_s114
                                 + 0.754875 w_k1_MAT_s114
                                 + 1.10915 w_k1_MSFT_s114
                                 + 1.03048 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.06498 w_k1_ADM_s114
                                 + 1.11857 w_k1_AXP_s114
                                 + 1.25189 w_k1_DIS_s114
                                 + 1.11708 w_k1_FDX_s114
                                 + 1.38011 w_k1_IBM_s114
                                 + 0.981125 w_k1_INTC_s114
                                 + 1.22635 w_k1_KO_s114 + 1.48254 w_k1_MAT_s114
                                 + 1.08168 w_k1_MSFT_s114
                                 + 1.15457 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.857351 w_k1_ADM_s114
                                 + 0.599758 w_k1_AXP_s114
                                 + 0.708771 w_k1_DIS_s114
                                 + 0.722287 w_k1_FDX_s114
                                 + 0.733734 w_k1_IBM_s114
                                 + 0.745288 w_k1_INTC_s114
                                 + 0.715117 w_k1_KO_s114
                                 + 0.695412 w_k1_MAT_s114
                                 + 0.936017 w_k1_MSFT_s114
                                 + 0.919848 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.07251 w_k1_ADM_s114
                                 + 1.3338 w_k1_AXP_s114 + 1.32897 w_k1_DIS_s114
                                 + 1.25777 w_k1_FDX_s114
                                 + 1.02022 w_k1_IBM_s114
                                 + 1.15826 w_k1_INTC_s114
                                 + 1.01225 w_k1_KO_s114 + 1.20942 w_k1_MAT_s114
                                 + 1.26586 w_k1_MSFT_s114
                                 + 1.13933 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.938211 w_k1_ADM_s114
                                 + 1.03781 w_k1_AXP_s114
                                 + 0.907103 w_k1_DIS_s114
                                 + 0.92591 w_k1_FDX_s114
                                 + 1.14062 w_k1_IBM_s114
                                 + 1.0046 w_k1_INTC_s114 + 0.98671 w_k1_KO_s114
                                 + 1.28885 w_k1_MAT_s114
                                 + 1.01774 w_k1_MSFT_s114
                                 + 0.903459 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.906856 w_k1_ADM_s115
                                 + 0.871354 w_k1_AXP_s115
                                 + 0.771833 w_k1_DIS_s115
                                 + 0.889167 w_k1_FDX_s115
                                 + 1.05735 w_k1_IBM_s115
                                 + 1.09173 w_k1_INTC_s115
                                 + 0.821383 w_k1_KO_s115
                                 + 0.890393 w_k1_MAT_s115
                                 + 0.888429 w_k1_MSFT_s115
                                 + 0.953423 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.980575 w_k1_ADM_s115
                                 + 1.59295 w_k1_AXP_s115
                                 + 1.64598 w_k1_DIS_s115
                                 + 1.48558 w_k1_FDX_s115
                                 + 1.23675 w_k1_IBM_s115
                                 + 1.37683 w_k1_INTC_s115
                                 + 1.26783 w_k1_KO_s115 + 1.7402 w_k1_MAT_s115
                                 + 1.22538 w_k1_MSFT_s115
                                 + 1.36676 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.944171 w_k1_ADM_s115
                                 + 0.870369 w_k1_AXP_s115
                                 + 0.664058 w_k1_DIS_s115
                                 + 0.84492 w_k1_FDX_s115
                                 + 0.751713 w_k1_IBM_s115
                                 + 0.94143 w_k1_INTC_s115 + 0.7501 w_k1_KO_s115
                                 + 0.830546 w_k1_MAT_s115
                                 + 0.771589 w_k1_MSFT_s115
                                 + 0.90065 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.29877 w_k1_ADM_s115
                                 + 1.32262 w_k1_AXP_s115
                                 + 1.29823 w_k1_DIS_s115
                                 + 1.36562 w_k1_FDX_s115
                                 + 0.78457 w_k1_IBM_s115
                                 + 1.13141 w_k1_INTC_s115
                                 + 0.87582 w_k1_KO_s115
                                 + 0.923847 w_k1_MAT_s115
                                 + 1.07751 w_k1_MSFT_s115
                                 + 1.10339 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.950948 w_k1_ADM_s115
                                 + 0.878587 w_k1_AXP_s115
                                 + 1.05223 w_k1_DIS_s115
                                 + 1.05543 w_k1_FDX_s115
                                 + 0.775883 w_k1_IBM_s115
                                 + 1.15784 w_k1_INTC_s115
                                 + 0.809376 w_k1_KO_s115
                                 + 0.849026 w_k1_MAT_s115
                                 + 1.04526 w_k1_MSFT_s115
                                 + 1.04982 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.19022 w_k1_ADM_s115
                                 + 1.14989 w_k1_AXP_s115
                                 + 1.35372 w_k1_DIS_s115 + 1.2531 w_k1_FDX_s115
                                 + 1.01961 w_k1_IBM_s115
                                 + 1.41926 w_k1_INTC_s115
                                 + 1.06621 w_k1_KO_s115 + 1.2696 w_k1_MAT_s115
                                 + 1.17416 w_k1_MSFT_s115
                                 + 1.13407 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.793562 w_k1_ADM_s115
                                 + 1.3355 w_k1_AXP_s115 + 1.29034 w_k1_DIS_s115
                                 + 1.05602 w_k1_FDX_s115
                                 + 1.59281 w_k1_IBM_s115
                                 + 1.14071 w_k1_INTC_s115
                                 + 1.19252 w_k1_KO_s115 + 1.67392 w_k1_MAT_s115
                                 + 0.938144 w_k1_MSFT_s115
                                 + 1.05694 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.794554 w_k1_ADM_s115
                                 + 0.909553 w_k1_AXP_s115
                                 + 1.25007 w_k1_DIS_s115
                                 + 1.06041 w_k1_FDX_s115
                                 + 1.23471 w_k1_IBM_s115
                                 + 1.23831 w_k1_INTC_s115 + 1.0264 w_k1_KO_s115
                                 + 0.993834 w_k1_MAT_s115
                                 + 1.03405 w_k1_MSFT_s115
                                 + 1.07951 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.37913 w_k1_ADM_s116
                                 + 1.58696 w_k1_AXP_s116 + 1.343 w_k1_DIS_s116
                                 + 1.28948 w_k1_FDX_s116
                                 + 0.991305 w_k1_IBM_s116
                                 + 1.39086 w_k1_INTC_s116
                                 + 1.14911 w_k1_KO_s116
                                 + 0.944956 w_k1_MAT_s116
                                 + 1.19602 w_k1_MSFT_s116
                                 + 1.14841 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.00683 w_k1_ADM_s116
                                 + 1.05047 w_k1_AXP_s116
                                 + 0.699796 w_k1_DIS_s116
                                 + 0.882643 w_k1_FDX_s116
                                 + 0.727623 w_k1_IBM_s116
                                 + 0.880883 w_k1_INTC_s116
                                 + 0.731864 w_k1_KO_s116
                                 + 0.674726 w_k1_MAT_s116
                                 + 0.922801 w_k1_MSFT_s116
                                 + 0.964653 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.15268 w_k1_ADM_s116
                                 + 1.50364 w_k1_AXP_s116
                                 + 1.78729 w_k1_DIS_s116
                                 + 1.30673 w_k1_FDX_s116
                                 + 1.35401 w_k1_IBM_s116
                                 + 1.46417 w_k1_INTC_s116
                                 + 1.56773 w_k1_KO_s116 + 1.42293 w_k1_MAT_s116
                                 + 1.33771 w_k1_MSFT_s116
                                 + 1.33534 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.898716 w_k1_ADM_s116
                                 + 0.728419 w_k1_AXP_s116
                                 + 0.79359 w_k1_DIS_s116
                                 + 0.804901 w_k1_FDX_s116
                                 + 0.900543 w_k1_IBM_s116
                                 + 1.02987 w_k1_INTC_s116
                                 + 0.894569 w_k1_KO_s116
                                 + 0.729933 w_k1_MAT_s116
                                 + 0.882435 w_k1_MSFT_s116
                                 + 0.911526 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.732136 w_k1_ADM_s116
                                 + 0.596367 w_k1_AXP_s116
                                 + 0.618902 w_k1_DIS_s116
                                 + 0.688834 w_k1_FDX_s116
                                 + 0.801071 w_k1_IBM_s116
                                 + 0.715881 w_k1_INTC_s116
                                 + 0.700754 w_k1_KO_s116
                                 + 0.508605 w_k1_MAT_s116
                                 + 0.699353 w_k1_MSFT_s116
                                 + 0.694069 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.937839 w_k1_ADM_s116
                                 + 1.18089 w_k1_AXP_s116
                                 + 0.918534 w_k1_DIS_s116
                                 + 1.08336 w_k1_FDX_s116
                                 + 0.760848 w_k1_IBM_s116
                                 + 1.07011 w_k1_INTC_s116
                                 + 0.869395 w_k1_KO_s116
                                 + 0.94806 w_k1_MAT_s116
                                 + 1.13965 w_k1_MSFT_s116
                                 + 0.976825 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.906891 w_k1_ADM_s116
                                 + 1.24984 w_k1_AXP_s116
                                 + 0.909965 w_k1_DIS_s116
                                 + 0.930091 w_k1_FDX_s116
                                 + 1.37555 w_k1_IBM_s116
                                 + 1.00053 w_k1_INTC_s116
                                 + 0.988743 w_k1_KO_s116
                                 + 1.16275 w_k1_MAT_s116
                                 + 0.968746 w_k1_MSFT_s116
                                 + 0.987129 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.983124 w_k1_ADM_s116
                                 + 1.08784 w_k1_AXP_s116
                                 + 1.24056 w_k1_DIS_s116
                                 + 1.05253 w_k1_FDX_s116
                                 + 1.03789 w_k1_IBM_s116
                                 + 1.17696 w_k1_INTC_s116
                                 + 1.06169 w_k1_KO_s116
                                 + 0.880816 w_k1_MAT_s116
                                 + 1.08374 w_k1_MSFT_s116
                                 + 0.967089 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.42296 w_k1_ADM_s117
                                 + 1.02115 w_k1_AXP_s117
                                 + 1.21187 w_k1_DIS_s117
                                 + 1.15196 w_k1_FDX_s117
                                 + 0.796235 w_k1_IBM_s117
                                 + 0.914109 w_k1_INTC_s117
                                 + 1.08571 w_k1_KO_s117
                                 + 0.736381 w_k1_MAT_s117
                                 + 1.12769 w_k1_MSFT_s117
                                 + 1.27529 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.53203 w_k1_ADM_s117
                                 + 1.18713 w_k1_AXP_s117
                                 + 1.40753 w_k1_DIS_s117
                                 + 1.20286 w_k1_FDX_s117 + 1.7439 w_k1_IBM_s117
                                 + 0.95117 w_k1_INTC_s117
                                 + 1.48262 w_k1_KO_s117 + 1.69897 w_k1_MAT_s117
                                 + 1.13816 w_k1_MSFT_s117
                                 + 1.09478 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.01046 w_k1_ADM_s117
                                 + 0.688185 w_k1_AXP_s117
                                 + 1.01687 w_k1_DIS_s117
                                 + 0.910172 w_k1_FDX_s117
                                 + 0.980745 w_k1_IBM_s117
                                 + 0.762678 w_k1_INTC_s117
                                 + 0.959728 w_k1_KO_s117
                                 + 0.640574 w_k1_MAT_s117
                                 + 0.832608 w_k1_MSFT_s117
                                 + 1.01214 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.759092 w_k1_ADM_s117
                                 + 0.696853 w_k1_AXP_s117
                                 + 0.764666 w_k1_DIS_s117
                                 + 0.763281 w_k1_FDX_s117
                                 + 0.942608 w_k1_IBM_s117
                                 + 0.909555 w_k1_INTC_s117
                                 + 0.831856 w_k1_KO_s117
                                 + 0.801296 w_k1_MAT_s117
                                 + 0.717309 w_k1_MSFT_s117
                                 + 0.881667 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.852963 w_k1_ADM_s117
                                 + 0.781117 w_k1_AXP_s117
                                 + 0.830607 w_k1_DIS_s117
                                 + 0.806574 w_k1_FDX_s117
                                 + 1.2776 w_k1_IBM_s117
                                 + 0.812046 w_k1_INTC_s117
                                 + 1.02333 w_k1_KO_s117 + 1.31542 w_k1_MAT_s117
                                 + 0.902201 w_k1_MSFT_s117
                                 + 1.02742 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.64403 w_k1_ADM_s117
                                 + 1.45513 w_k1_AXP_s117
                                 + 1.34547 w_k1_DIS_s117
                                 + 1.36193 w_k1_FDX_s117
                                 + 1.23087 w_k1_IBM_s117
                                 + 1.05797 w_k1_INTC_s117
                                 + 1.16581 w_k1_KO_s117 + 1.20468 w_k1_MAT_s117
                                 + 1.00759 w_k1_MSFT_s117
                                 + 1.27218 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.939244 w_k1_ADM_s117
                                 + 1.10667 w_k1_AXP_s117
                                 + 1.02357 w_k1_DIS_s117
                                 + 1.07465 w_k1_FDX_s117 + 1.0514 w_k1_IBM_s117
                                 + 1.03734 w_k1_INTC_s117
                                 + 1.11825 w_k1_KO_s117 + 1.00893 w_k1_MAT_s117
                                 + 0.927788 w_k1_MSFT_s117
                                 + 1.10179 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.972786 w_k1_ADM_s117
                                 + 0.726608 w_k1_AXP_s117
                                 + 0.698376 w_k1_DIS_s117
                                 + 0.862621 w_k1_FDX_s117
                                 + 0.738679 w_k1_IBM_s117
                                 + 0.757658 w_k1_INTC_s117
                                 + 0.727382 w_k1_KO_s117
                                 + 0.674231 w_k1_MAT_s117
                                 + 0.822496 w_k1_MSFT_s117
                                 + 0.925367 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0243757868121845 w_k1_ADM_s000
                                 + 0.0212722153132955 w_k1_ADM_s001
                                 + 0.0218342360913895 w_k1_ADM_s002
                                 + 0.0154568443837621 w_k1_ADM_s003
                                 + 0.0738584388403846 w_k1_ADM_s004
                                 + 0.0320859539886773 w_k1_ADM_s005
                                 + 0.0305684782105904 w_k1_ADM_s006
                                 + 0.0180421949372496 w_k1_ADM_s007
                                 + 0.0120565095345591 w_k1_ADM_s010
                                 + 0.01704312940674 w_k1_ADM_s011
                                 + 0.00811167940396334 w_k1_ADM_s012
                                 + 0.00710443473170863 w_k1_ADM_s013
                                 + 0.00834084044107659 w_k1_ADM_s014
                                 + 0.00855513107365878 w_k1_ADM_s015
                                 + 0.00741877255846653 w_k1_ADM_s016
                                 + 0.0086625205838822 w_k1_ADM_s017
                                 + 0.041980153791347 w_k1_ADM_s100
                                 + 0.0427707300314218 w_k1_ADM_s101
                                 + 0.0548150152081046 w_k1_ADM_s102
                                 + 0.016499403334885 w_k1_ADM_s103
                                 + 0.0421601140874938 w_k1_ADM_s104
                                 + 0.0725342196207066 w_k1_ADM_s105
                                 + 0.0520352784304274 w_k1_ADM_s106
                                 + 0.0708636653825312 w_k1_ADM_s107
                                 + 0.0695201377656517 w_k1_ADM_s110
                                 + 0.0581436118122125 w_k1_ADM_s111
                                 + 0.030973724180306 w_k1_ADM_s112
                                 + 0.0209607529953536 w_k1_ADM_s113
                                 + 0.0554996488294133 w_k1_ADM_s114
                                 + 0.043988952517584 w_k1_ADM_s115
                                 + 0.0245155413415865 w_k1_ADM_s116
                                 + 0.0170349004192304 w_k1_ADM_s117
                                 + 0.0240238848304922 w_k1_AXP_s000
                                 + 0.0202523133362807 w_k1_AXP_s001
                                 + 0.0171586664502353 w_k1_AXP_s002
                                 + 0.0176290206124707 w_k1_AXP_s003
                                 + 0.0766909451461542 w_k1_AXP_s004
                                 + 0.0318229803944875 w_k1_AXP_s005
                                 + 0.0269788799384481 w_k1_AXP_s006
                                 + 0.0157763541991971 w_k1_AXP_s007
                                 + 0.0119816180613462 w_k1_AXP_s010
                                 + 0.0178987371309897 w_k1_AXP_s011
                                 + 0.0091473145568243 w_k1_AXP_s012
                                 + 0.00707814344106488 w_k1_AXP_s013
                                 + 0.00751985416153556 w_k1_AXP_s014
                                 + 0.0091869245148653 w_k1_AXP_s015
                                 + 0.0065952334633255 w_k1_AXP_s016
                                 + 0.00882475921863335 w_k1_AXP_s017
                                 + 0.0427133679371918 w_k1_AXP_s100
                                 + 0.0458062130050078 w_k1_AXP_s101
                                 + 0.0531945902023119 w_k1_AXP_s102
                                 + 0.0170683845123774 w_k1_AXP_s103
                                 + 0.0414040132732434 w_k1_AXP_s104
                                 + 0.0735260952171136 w_k1_AXP_s105
                                 + 0.0529443771762696 w_k1_AXP_s106
                                 + 0.0719127087557269 w_k1_AXP_s107
                                 + 0.0682941104870465 w_k1_AXP_s110
                                 + 0.0637597799506794 w_k1_AXP_s111
                                 + 0.0276372527563815 w_k1_AXP_s112
                                 + 0.0209503440983224 w_k1_AXP_s113
                                 + 0.0559489169021601 w_k1_AXP_s114
                                 + 0.0473158604190637 w_k1_AXP_s115
                                 + 0.028392042315388 w_k1_AXP_s116
                                 + 0.0151347099610854 w_k1_AXP_s117
                                 + 0.0240237640991561 w_k1_DIS_s000
                                 + 0.0224853118019035 w_k1_DIS_s001
                                 + 0.0198866326079655 w_k1_DIS_s002
                                 + 0.0171890348248409 w_k1_DIS_s003
                                 + 0.0790333378592511 w_k1_DIS_s004
                                 + 0.0335267167416718 w_k1_DIS_s005
                                 + 0.0293016398228166 w_k1_DIS_s006
                                 + 0.0169826290694306 w_k1_DIS_s007
                                 + 0.0128300259547297 w_k1_DIS_s010
                                 + 0.0172320707112835 w_k1_DIS_s011
                                 + 0.00929107475682312 w_k1_DIS_s012
                                 + 0.00650269399877012 w_k1_DIS_s013
                                 + 0.00854886335459076 w_k1_DIS_s014
                                 + 0.00779363105149925 w_k1_DIS_s015
                                 + 0.00656072664034309 w_k1_DIS_s016
                                 + 0.00886512581881646 w_k1_DIS_s017
                                 + 0.0449154623835342 w_k1_DIS_s100
                                 + 0.0454068127633752 w_k1_DIS_s101
                                 + 0.0520563571135679 w_k1_DIS_s102
                                 + 0.0168161401486612 w_k1_DIS_s103
                                 + 0.0409181787197883 w_k1_DIS_s104
                                 + 0.0694253787553701 w_k1_DIS_s105
                                 + 0.0498333150322466 w_k1_DIS_s106
                                 + 0.0731842096143165 w_k1_DIS_s107
                                 + 0.0660013601340957 w_k1_DIS_s110
                                 + 0.0651377035598622 w_k1_DIS_s111
                                 + 0.0273708695334294 w_k1_DIS_s112
                                 + 0.0198147555046942 w_k1_DIS_s113
                                 + 0.0601878858496981 w_k1_DIS_s114
                                 + 0.0503331895697629 w_k1_DIS_s115
                                 + 0.025407084509211 w_k1_DIS_s116
                                 + 0.0154839036659062 w_k1_DIS_s117
                                 + 0.0260726255088046 w_k1_FDX_s000
                                 + 0.0227026826566511 w_k1_FDX_s001
                                 + 0.019938597575664 w_k1_FDX_s002
                                 + 0.0167381192944612 w_k1_FDX_s003
                                 + 0.0750296414131358 w_k1_FDX_s004
                                 + 0.0322546579848033 w_k1_FDX_s005
                                 + 0.0294805784680543 w_k1_FDX_s006
                                 + 0.0163430467004399 w_k1_FDX_s007
                                 + 0.0120324452433756 w_k1_FDX_s010
                                 + 0.0179283058286494 w_k1_FDX_s011
                                 + 0.00885755780019463 w_k1_FDX_s012
                                 + 0.00627552397899978 w_k1_FDX_s013
                                 + 0.00769878894346922 w_k1_FDX_s014
                                 + 0.00796805799063843 w_k1_FDX_s015
                                 + 0.00684598142231612 w_k1_FDX_s016
                                 + 0.00857775116032858 w_k1_FDX_s017
                                 + 0.043196623039562 w_k1_FDX_s100
                                 + 0.0435584742257937 w_k1_FDX_s101
                                 + 0.0516524618608011 w_k1_FDX_s102
                                 + 0.0181144882448164 w_k1_FDX_s103
                                 + 0.0404925589817374 w_k1_FDX_s104
                                 + 0.0678862671890907 w_k1_FDX_s105
                                 + 0.0470263204692552 w_k1_FDX_s106
                                 + 0.0717535922762626 w_k1_FDX_s107
                                 + 0.061945449194741 w_k1_FDX_s110
                                 + 0.0625652081349106 w_k1_FDX_s111
                                 + 0.0274357604575677 w_k1_FDX_s112
                                 + 0.0200613466249324 w_k1_FDX_s113
                                 + 0.0563525083077341 w_k1_FDX_s114
                                 + 0.0490952369946102 w_k1_FDX_s115
                                 + 0.0248027631078016 w_k1_FDX_s116
                                 + 0.0158030185682987 w_k1_FDX_s117
                                 + 0.0249292851576708 w_k1_IBM_s000
                                 + 0.0225350954828825 w_k1_IBM_s001
                                 + 0.0205621549110804 w_k1_IBM_s002
                                 + 0.0175871465216491 w_k1_IBM_s003
                                 + 0.0763488850152742 w_k1_IBM_s004
                                 + 0.0323888677564908 w_k1_IBM_s005
                                 + 0.0295978614168202 w_k1_IBM_s006
                                 + 0.0156576401993665 w_k1_IBM_s007
                                 + 0.0121691607458264 w_k1_IBM_s010
                                 + 0.017109546101672 w_k1_IBM_s011
                                 + 0.0100477139111194 w_k1_IBM_s012
                                 + 0.00613728690809289 w_k1_IBM_s013
                                 + 0.00779346031036604 w_k1_IBM_s014
                                 + 0.00812438031851077 w_k1_IBM_s015
                                 + 0.00677810175694789 w_k1_IBM_s016
                                 + 0.00853930186252656 w_k1_IBM_s017
                                 + 0.0430657300670794 w_k1_IBM_s100
                                 + 0.0415446946197576 w_k1_IBM_s101
                                 + 0.0524239644433427 w_k1_IBM_s102
                                 + 0.0180649651417916 w_k1_IBM_s103
                                 + 0.0412217458415053 w_k1_IBM_s104
                                 + 0.0689279106595702 w_k1_IBM_s105
                                 + 0.0490586670227155 w_k1_IBM_s106
                                 + 0.0729744588242635 w_k1_IBM_s107
                                 + 0.0632827255408293 w_k1_IBM_s110
                                 + 0.062191524117662 w_k1_IBM_s111
                                 + 0.0270955479513241 w_k1_IBM_s112
                                 + 0.0210703332143259 w_k1_IBM_s113
                                 + 0.0576957373518633 w_k1_IBM_s114
                                 + 0.0485170075288194 w_k1_IBM_s115
                                 + 0.0262492876295657 w_k1_IBM_s116
                                 + 0.0162258367587104 w_k1_IBM_s117
                                 + 0.0250211610541481 w_k1_INTC_s000
                                 + 0.0206996580944845 w_k1_INTC_s001
                                 + 0.0214474002522833 w_k1_INTC_s002
                                 + 0.0173980424107981 w_k1_INTC_s003
                                 + 0.07901371511318 w_k1_INTC_s004
                                 + 0.0329918518341396 w_k1_INTC_s005
                                 + 0.0299299698492102 w_k1_INTC_s006
                                 + 0.0163456160652861 w_k1_INTC_s007
                                 + 0.0116698984493534 w_k1_INTC_s010
                                 + 0.0185861169353335 w_k1_INTC_s011
                                 + 0.0094083589057486 w_k1_INTC_s012
                                 + 0.0054768069731634 w_k1_INTC_s013
                                 + 0.00804942408971457 w_k1_INTC_s014
                                 + 0.00686945877218772 w_k1_INTC_s015
                                 + 0.00586771108493801 w_k1_INTC_s016
                                 + 0.0086379661710519 w_k1_INTC_s017
                                 + 0.0455318229334639 w_k1_INTC_s100
                                 + 0.0434417221426186 w_k1_INTC_s101
                                 + 0.0536959025902331 w_k1_INTC_s102
                                 + 0.0200594552196227 w_k1_INTC_s103
                                 + 0.0367010777320595 w_k1_INTC_s104
                                 + 0.0748517813043196 w_k1_INTC_s105
                                 + 0.0513423235049899 w_k1_INTC_s106
                                 + 0.0684428507222385 w_k1_INTC_s107
                                 + 0.0529510556333879 w_k1_INTC_s110
                                 + 0.0635259619516715 w_k1_INTC_s111
                                 + 0.0226533421666729 w_k1_INTC_s112
                                 + 0.017331768283479 w_k1_INTC_s113
                                 + 0.0516473259307848 w_k1_INTC_s114
                                 + 0.0543517445481228 w_k1_INTC_s115
                                 + 0.0269008440819474 w_k1_INTC_s116
                                 + 0.014397368517348 w_k1_INTC_s117
                                 + 0.0231271371761444 w_k1_KO_s000
                                 + 0.0214353111321234 w_k1_KO_s001
                                 + 0.0193061295490461 w_k1_KO_s002
                                 + 0.0180002084026741 w_k1_KO_s003
                                 + 0.0817965340277121 w_k1_KO_s004
                                 + 0.0328809780758029 w_k1_KO_s005
                                 + 0.031801662186642 w_k1_KO_s006
                                 + 0.0152577545480705 w_k1_KO_s007
                                 + 0.0132465804797514 w_k1_KO_s010
                                 + 0.0195861255464968 w_k1_KO_s011
                                 + 0.00927678772678904 w_k1_KO_s012
                                 + 0.00636136226312787 w_k1_KO_s013
                                 + 0.00789567300645869 w_k1_KO_s014
                                 + 0.00760948035319421 w_k1_KO_s015
                                 + 0.006547060509148 w_k1_KO_s016
                                 + 0.00844739491989655 w_k1_KO_s017
                                 + 0.0432520525856417 w_k1_KO_s100
                                 + 0.0427066312947866 w_k1_KO_s101
                                 + 0.0470197593411343 w_k1_KO_s102
                                 + 0.0184004423113248 w_k1_KO_s103
                                 + 0.0431408638110766 w_k1_KO_s104
                                 + 0.0758167374443369 w_k1_KO_s105
                                 + 0.0488991329505234 w_k1_KO_s106
                                 + 0.0661576163162899 w_k1_KO_s107
                                 + 0.0669140052936358 w_k1_KO_s110
                                 + 0.0670419829419489 w_k1_KO_s111
                                 + 0.0265125675655795 w_k1_KO_s112
                                 + 0.021368721432791 w_k1_KO_s113
                                 + 0.0545753411606892 w_k1_KO_s114
                                 + 0.043588763440906 w_k1_KO_s115
                                 + 0.0247967064506772 w_k1_KO_s116
                                 + 0.0158186433083632 w_k1_KO_s117
                                 + 0.0242682945386778 w_k1_MAT_s000
                                 + 0.024538706667377 w_k1_MAT_s001
                                 + 0.0181912529574789 w_k1_MAT_s002
                                 + 0.0260455287132401 w_k1_MAT_s003
                                 + 0.0883655342252732 w_k1_MAT_s004
                                 + 0.0309083845972118 w_k1_MAT_s005
                                 + 0.0331046320611964 w_k1_MAT_s006
                                 + 0.0161097012400258 w_k1_MAT_s007
                                 + 0.0159420333439882 w_k1_MAT_s010
                                 + 0.0150706180032571 w_k1_MAT_s011
                                 + 0.0106706330854163 w_k1_MAT_s012
                                 + 0.00678508666418175 w_k1_MAT_s013
                                 + 0.00735290127668227 w_k1_MAT_s014
                                 + 0.00907404284537771 w_k1_MAT_s015
                                 + 0.00851832432887073 w_k1_MAT_s016
                                 + 0.00965309810419165 w_k1_MAT_s017
                                 + 0.0401582898617987 w_k1_MAT_s100
                                 + 0.0385619373840063 w_k1_MAT_s101
                                 + 0.0615062206867288 w_k1_MAT_s102
                                 + 0.0165904904841947 w_k1_MAT_s103
                                 + 0.0512544465868765 w_k1_MAT_s104
                                 + 0.076036091995614 w_k1_MAT_s105
                                 + 0.0519285854688541 w_k1_MAT_s106
                                 + 0.077251779132652 w_k1_MAT_s107
                                 + 0.0656711983990575 w_k1_MAT_s110
                                 + 0.0680597280606397 w_k1_MAT_s111
                                 + 0.03031435987599 w_k1_MAT_s112
                                 + 0.0269633066614978 w_k1_MAT_s113
                                 + 0.0600062877106809 w_k1_MAT_s114
                                 + 0.0493187404131476 w_k1_MAT_s115
                                 + 0.0236579182903879 w_k1_MAT_s116
                                 + 0.015118702959176 w_k1_MAT_s117
                                 + 0.0241573095890527 w_k1_MSFT_s000
                                 + 0.0221766832479171 w_k1_MSFT_s001
                                 + 0.0189308839284021 w_k1_MSFT_s002
                                 + 0.0179787286449309 w_k1_MSFT_s003
                                 + 0.0832224814756827 w_k1_MSFT_s004
                                 + 0.0311377121401472 w_k1_MSFT_s005
                                 + 0.0327276383639459 w_k1_MSFT_s006
                                 + 0.0166069310061898 w_k1_MSFT_s007
                                 + 0.0108322238633842 w_k1_MSFT_s010
                                 + 0.0178333802160626 w_k1_MSFT_s011
                                 + 0.0103366343789415 w_k1_MSFT_s012
                                 + 0.00590211157223047 w_k1_MSFT_s013
                                 + 0.00772559950651548 w_k1_MSFT_s014
                                 + 0.00724877739539734 w_k1_MSFT_s015
                                 + 0.00660163188815404 w_k1_MSFT_s016
                                 + 0.00776930280365066 w_k1_MSFT_s017
                                 + 0.0448605370933725 w_k1_MSFT_s100
                                 + 0.0452185584751399 w_k1_MSFT_s101
                                 + 0.0523096429943163 w_k1_MSFT_s102
                                 + 0.0165760102900987 w_k1_MSFT_s103
                                 + 0.0392531200021955 w_k1_MSFT_s104
                                 + 0.0694321595659809 w_k1_MSFT_s105
                                 + 0.0475884500265467 w_k1_MSFT_s106
                                 + 0.0732971014677383 w_k1_MSFT_s107
                                 + 0.058724089679702 w_k1_MSFT_s110
                                 + 0.0636215733706968 w_k1_MSFT_s111
                                 + 0.0273104506696863 w_k1_MSFT_s112
                                 + 0.0186018414998335 w_k1_MSFT_s113
                                 + 0.0601511354479602 w_k1_MSFT_s114
                                 + 0.0459888116673955 w_k1_MSFT_s115
                                 + 0.025499046185586 w_k1_MSFT_s116
                                 + 0.0142554430894795 w_k1_MSFT_s117
                                 + 0.0243219799389372 w_k1_PG_s000
                                 + 0.0219026852465468 w_k1_PG_s001
                                 + 0.0190673175704466 w_k1_PG_s002
                                 + 0.0173558063786508 w_k1_PG_s003
                                 + 0.0755551463068847 w_k1_PG_s004
                                 + 0.0294002411882943 w_k1_PG_s005
                                 + 0.0313150553128721 w_k1_PG_s006
                                 + 0.0155705491130841 w_k1_PG_s007
                                 + 0.012119036189849 w_k1_PG_s010
                                 + 0.0174470476882422 w_k1_PG_s011
                                 + 0.00957789329055313 w_k1_PG_s012
                                 + 0.00622689979040646 w_k1_PG_s013
                                 + 0.00797680789788095 w_k1_PG_s014
                                 + 0.00818393634827433 w_k1_PG_s015
                                 + 0.00657904109860384 w_k1_PG_s016
                                 + 0.00844840567191437 w_k1_PG_s017
                                 + 0.0446144973626161 w_k1_PG_s100
                                 + 0.0435008121173599 w_k1_PG_s101
                                 + 0.0511443374901378 w_k1_PG_s102
                                 + 0.0187942526855913 w_k1_PG_s103
                                 + 0.0394117297229713 w_k1_PG_s104
                                 + 0.0688036848761445 w_k1_PG_s105
                                 + 0.0495353683775977 w_k1_PG_s106
                                 + 0.0663588895058023 w_k1_PG_s107
                                 + 0.0654236926898098 w_k1_PG_s110
                                 + 0.062072930512892 w_k1_PG_s111
                                 + 0.0258458838355723 w_k1_PG_s112
                                 + 0.0229504912698652 w_k1_PG_s113
                                 + 0.0578195431244884 w_k1_PG_s114
                                 + 0.0484714571651446 w_k1_PG_s115
                                 + 0.0248298796208439 w_k1_PG_s116
                                 + 0.0166415669745959 w_k1_PG_s117 >= 1.08
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

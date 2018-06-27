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
 w_k1_ADM_s0_return_constraint:  1.32034 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.966122 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.078 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.795455 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.00378 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.1176 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.752814 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.24429 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.06113 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.996803 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.917416 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.04131 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.18693 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.786908 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.865769 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.31741 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.976028 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.00747 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.07774 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.869366 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.658581 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.23974 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.889601 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.922593 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.35795 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.035 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.880737 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.08689 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.692245 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.820085 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.874117 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.07103 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.02972 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.23774 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.62925 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.957621 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.20075 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.805309 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.34698 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.943777 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.08756 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.746556 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.921282 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.26163 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.7293 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.58314 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.21629 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.02692 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.21382 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.21569 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.59185 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.849213 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.925556 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.44763 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.855745 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.855338 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.27606 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.797128 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.707467 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.06413 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.647558 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.864563 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.56001 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.22075 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.972668 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.20961 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.675842 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.69649 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.848204 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.0777 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.42531 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.36493 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.64141 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.978748 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.51453 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.779363 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.4243 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.955148 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.1863 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.695487 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.937663 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.2235 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.658476 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.75124 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.19907 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.991157 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.829157 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.11633 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.39656 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.789355 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.943724 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.5328 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.954062 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.903103 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.34691 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.771457 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.548609 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.956668 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.812292 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.68946 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.68804 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.986645 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.05775 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.993918 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.582271 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.546573 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.962482 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.12698 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.29578 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.63314 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.620118 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.00877 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.63622 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.680786 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.297 w_0_FDX - w_k1_FDX_s0 + 0.99 b_k1_FDX_s0
                                 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.955621 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.1078 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.792086 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.960777 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.12352 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.788922 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.39147 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.953312 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.0858 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.883056 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.03337 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.27577 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.772144 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.885901 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.41913 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.87189 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.02249 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.13777 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.04839 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.696256 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.04308 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.924662 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.844801 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.5546 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.09459 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.991402 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.16837 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.676476 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.814966 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.835396 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.13118 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.19541 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.33883 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.666619 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.966021 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.18674 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.75762 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.08518 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.00998 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.14673 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.807696 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.820269 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.42977 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.653912 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.22099 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.7674 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.736272 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.988892 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.12844 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.54993 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.14769 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.3663 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.906514 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.04791 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.724846 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.70909 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.508722 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.779429 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.995713 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.653416 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.995131 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.29827 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.12954 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.27209 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.701653 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.896931 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.53406 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.15431 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.663588 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.24673 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.953774 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.953056 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.858281 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.73936 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.672051 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.22274 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.944169 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.07563 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.758399 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.02274 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.0536 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.651997 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.35874 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.922009 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.01474 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.877047 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.931873 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.3419 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.835888 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.02938 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.33568 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.02398 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.00734 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.06641 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.07827 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.794727 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.06057 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.745071 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.81186 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.38234 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.971729 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.907439 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.0745 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.663679 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.826104 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.992698 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.14101 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.27639 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.37363 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.781974 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.03673 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.38774 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.91187 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.24487 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.984979 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.08834 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.746646 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.899972 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.23095 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.743544 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.3515 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.3759 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.856237 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.824986 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.05874 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.37826 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.982719 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.14676 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.16056 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.960079 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.916623 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.55671 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.684341 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.68021 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.00385 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.79355 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.746007 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.30897 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.94617 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.15897 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.875546 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.772247 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.674594 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.03388 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.913789 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.25649 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.17127 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.795084 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.938589 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.57199 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.801622 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.14486 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.04936 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.05736 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.877415 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.811295 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.59197 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.91999 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.67716 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 2.16597 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.779582 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.34224 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.64092 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 2.07175 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.31692 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.42246 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.883511 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.686236 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.679994 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.58289 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.420171 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.578063 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.00482 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.565349 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.921393 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.2022 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.11573 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.02718 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.01371 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.71631 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.594578 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.919784 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.7246 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.25117 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.900228 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.7691 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.725293 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.73241 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.658373 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.27589 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 1.00724 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.00976 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.722617 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.992552 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.04784 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.792897 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.43662 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.07701 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.19182 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.889623 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.10764 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.27398 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.867851 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.956699 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.42922 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.937153 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.1674 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.24093 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.02963 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.835845 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.02724 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.891272 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.761253 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.32313 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.888412 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.908382 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.952793 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.660213 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.754442 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.924287 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.15868 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.22016 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.32371 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.721304 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.999634 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.36315 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.771409 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.16958 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.973604 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.10563 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.855322 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.964904 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.0868 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.838741 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.20531 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.11473 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.89838 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.855658 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.00029 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.15202 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.967153 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.912062 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.21209 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.881442 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.87091 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.23944 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.873004 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.759464 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.960114 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.846788 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.854734 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.26491 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.04346 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.02726 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.964952 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.757997 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.80049 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.98676 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.0666 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.14168 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.23162 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.824335 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.957183 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.19065 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.841611 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.290000858744869 z_k1_s0000
                                 - 0.181999760378892 z_k1_s0001
                                 - 0.365000390972945 z_k1_s0002
                                 - 0.171999632215412 z_k1_s0003
                                 - 0.49599949771809 z_k1_s0004
                                 - 0.370999324672615 z_k1_s0005
                                 - 0.487999966786515 z_k1_s0006
                                 - 0.493998900486185 z_k1_s0007
                                 - 0.207999542581787 z_k1_s0010
                                 - 0.136999836209969 z_k1_s0011
                                 - 0.384000130149926 z_k1_s0012
                                 - 0.222998996181996 z_k1_s0013
                                 - 0.217000662740329 z_k1_s0014
                                 - 0.142999798649855 z_k1_s0015
                                 - 0.353999412951486 z_k1_s0016
                                 - 0.143999822556469 z_k1_s0017
                                 - 0.0740000217225117 z_k1_s0020
                                 - 0.106000064710627 z_k1_s0021
                                 - 0.150000279194298 z_k1_s0022
                                 - 0.146999676970368 z_k1_s0023
                                 - 0.168999535612184 z_k1_s0024
                                 - 0.304000532687108 z_k1_s0025
                                 - 0.221000444492935 z_k1_s0026
                                 - 0.0720000345732556 z_k1_s0027
                                 - 0.654998662784404 z_k1_s0030
                                 - 0.199000133765889 z_k1_s0031
                                 - 0.190000058716164 z_k1_s0032
                                 - 0.383999421120968 z_k1_s0033
                                 - 0.494999650234129 z_k1_s0034
                                 - 0.700000530533288 z_k1_s0035
                                 - 0.14699993231196 z_k1_s0036
                                 - 0.215000035020916 z_k1_s0037
                                 - 0.0859999938175654 z_k1_s0040
                                 - 0.128000110156403 z_k1_s0041
                                 - 0.094000025377346 z_k1_s0042
                                 - 0.157999981755526 z_k1_s0043
                                 - 0.0619999978382453 z_k1_s0044
                                 - 0.279000286372169 z_k1_s0045
                                 - 0.121000304616644 z_k1_s0046
                                 - 0.059000010678333 z_k1_s0047
                                 - 0.191000318845995 z_k1_s0050
                                 - 0.145000272652701 z_k1_s0051
                                 - 0.128000086190058 z_k1_s0052
                                 - 0.109999602288595 z_k1_s0053
                                 - 0.0679999666510125 z_k1_s0054
                                 - 0.181999589895537 z_k1_s0055
                                 - 0.0549999959441884 z_k1_s0056
                                 - 0.0949999133036429 z_k1_s0057
                                 - 0.0669999868803611 z_k1_s0060
                                 - 0.0990002694516205 z_k1_s0061
                                 - 0.0879996398457973 z_k1_s0062
                                 - 0.0510000383447595 z_k1_s0063
                                 - 0.119000012371094 z_k1_s0064
                                 - 0.0760000253941468 z_k1_s0065
                                 - 0.0520000254907332 z_k1_s0066
                                 - 0.219000268968688 z_k1_s0067
                                 - 0.100000139476763 z_k1_s0070
                                 - 0.210000399641265 z_k1_s0071
                                 - 0.258999425490265 z_k1_s0072
                                 - 0.394000698974534 z_k1_s0073
                                 - 0.25600002972432 z_k1_s0074
                                 - 0.119000048563279 z_k1_s0075
                                 - 0.311999424694905 z_k1_s0076
                                 - 0.128999991350913 z_k1_s0077
                                 - 0.0580001047894941 z_k1_s0100
                                 - 0.0980001922167568 z_k1_s0101
                                 - 0.121000102757199 z_k1_s0102
                                 - 0.0340000397731154 z_k1_s0103
                                 - 0.11000003632463 z_k1_s0104
                                 - 0.0720003206493464 z_k1_s0105
                                 - 0.0830003870819147 z_k1_s0106
                                 - 0.0520000571353468 z_k1_s0107
                                 - 0.0480000005424354 z_k1_s0110
                                 - 0.125999916798851 z_k1_s0111
                                 - 0.0809997047277144 z_k1_s0112
                                 - 0.0760000572755503 z_k1_s0113
                                 - 0.0699998033326214 z_k1_s0114
                                 - 0.183999889245947 z_k1_s0115
                                 - 0.0380000286377752 z_k1_s0116
                                 - 0.0540000513852647 z_k1_s0117
                                 - 0.0590000541774439 z_k1_s0120
                                 - 0.0550000932162715 z_k1_s0121
                                 - 0.0180000043253189 z_k1_s0122
                                 - 0.0209999930462026 z_k1_s0123
                                 - 0.0619999348983016 z_k1_s0124
                                 - 0.0609998546579869 z_k1_s0125
                                 - 0.0180000043253189 z_k1_s0126
                                 - 0.065999895859474 z_k1_s0127
                                 - 0.254999991317173 z_k1_s0130
                                 - 0.0740001195803955 z_k1_s0131
                                 - 0.148999994926505 z_k1_s0132
                                 - 0.16399999441575 z_k1_s0133
                                 - 0.0680001197846973 z_k1_s0134
                                 - 0.115000118184333 z_k1_s0135
                                 - 0.204999993019688 z_k1_s0136
                                 - 0.190999993496392 z_k1_s0137
                                 - 0.0930002162164605 z_k1_s0140
                                 - 0.0530000474566419 z_k1_s0141
                                 - 0.0560000194636011 z_k1_s0142
                                 - 0.0570000101325875 z_k1_s0143
                                 - 0.0510000119186328 z_k1_s0144
                                 - 0.104000113575311 z_k1_s0145
                                 - 0.0749998421743427 z_k1_s0146
                                 - 0.0530000474566419 z_k1_s0147
                                 - 0.0600000564094339 z_k1_s0150
                                 - 0.0710001307512237 z_k1_s0151
                                 - 0.0489999820676442 z_k1_s0152
                                 - 0.0629998672297251 z_k1_s0153
                                 - 0.030000028204717 z_k1_s0154
                                 - 0.0540000507684905 z_k1_s0155
                                 - 0.0420000394866038 z_k1_s0156
                                 - 0.0150000141023585 z_k1_s0157
                                 - 0.0789999975609685 z_k1_s0160
                                 - 0.0520000417996405 z_k1_s0161
                                 - 0.0470001769920331 z_k1_s0162
                                 - 0.0530001671612171 z_k1_s0163
                                 - 0.033999995092061 z_k1_s0164
                                 - 0.0430000565458645 z_k1_s0165
                                 - 0.016000024584509 z_k1_s0166
                                 - 0.0569999066072479 z_k1_s0167
                                 - 0.204999810003045 z_k1_s0170
                                 - 0.045999897527621 z_k1_s0171
                                 - 0.0449999273421879 z_k1_s0172
                                 - 0.192999321779382 z_k1_s0173
                                 - 0.0559999377813382 z_k1_s0174
                                 - 0.0559999377813382 z_k1_s0175
                                 - 0.193999291964815 z_k1_s0176
                                 - 0.0509999176544796 z_k1_s0177
                                 - 0.447999803669151 z_k1_s1000
                                 - 0.551000150718036 z_k1_s1001
                                 - 0.219999966337521 z_k1_s1002
                                 - 0.160999913756103 z_k1_s1003
                                 - 0.442000904423444 z_k1_s1004
                                 - 0.273000115673359 z_k1_s1005
                                 - 0.292999793158629 z_k1_s1006
                                 - 0.122000040659891 z_k1_s1007
                                 - 0.692000485826589 z_k1_s1010
                                 - 0.455999875122771 z_k1_s1011
                                 - 0.2519999309889 z_k1_s1012
                                 - 0.985998379313856 z_k1_s1013
                                 - 0.4199998849815 z_k1_s1014
                                 - 0.273999756130806 z_k1_s1015
                                 - 1.10999969602254 z_k1_s1016
                                 - 0.874998916211548 z_k1_s1017
                                 - 1.90500076489428 z_k1_s1020
                                 - 1.71900245823469 z_k1_s1021
                                 - 1.60300376755487 z_k1_s1022
                                 - 0.526000479081428 z_k1_s1023
                                 - 0.697000999019827 z_k1_s1024
                                 - 0.541000621105986 z_k1_s1025
                                 - 0.402000822873911 z_k1_s1026
                                 - 0.458000515657734 z_k1_s1027
                                 - 0.492000775098113 z_k1_s1030
                                 - 1.46299751870523 z_k1_s1031
                                 - 2.14800031973077 z_k1_s1032
                                 - 0.752001240927287 z_k1_s1033
                                 - 1.55800648858381 z_k1_s1034
                                 - 2.15100861053973 z_k1_s1035
                                 - 0.752001240927287 z_k1_s1036
                                 - 2.20999877283969 z_k1_s1037
                                 - 1.91799927294901 z_k1_s1040
                                 - 1.14600154051755 z_k1_s1041
                                 - 1.50600059213824 z_k1_s1042
                                 - 0.733998650914453 z_k1_s1043
                                 - 2.02099894315171 z_k1_s1044
                                 - 1.34299406534489 z_k1_s1045
                                 - 1.00999862528904 z_k1_s1046
                                 - 0.843998594953682 z_k1_s1047
                                 - 0.705998176146046 z_k1_s1050
                                 - 0.323999775296333 z_k1_s1051
                                 - 0.560999716263069 z_k1_s1052
                                 - 0.34299996576366 z_k1_s1053
                                 - 0.230999860861269 z_k1_s1054
                                 - 0.503998386461277 z_k1_s1055
                                 - 0.762999505947837 z_k1_s1056
                                 - 0.359999876729228 z_k1_s1057
                                 - 1.49999591709504 z_k1_s1060
                                 - 1.12699311815357 z_k1_s1061
                                 - 0.460998986195857 z_k1_s1062
                                 - 0.925999763829516 z_k1_s1063
                                 - 1.12099671817901 z_k1_s1064
                                 - 1.3209998413146 z_k1_s1065
                                 - 0.52299846116213 z_k1_s1066
                                 - 0.880001167432306 z_k1_s1067
                                 - 0.621001850234398 z_k1_s1070
                                 - 0.259000068968775 z_k1_s1071
                                 - 0.205000208105121 z_k1_s1072
                                 - 0.740001135911431 z_k1_s1073
                                 - 0.295000195278029 z_k1_s1074
                                 - 0.367999392582177 z_k1_s1075
                                 - 0.367999392582177 z_k1_s1076
                                 - 0.429999683137053 z_k1_s1077
                                 - 0.57200065003371 z_k1_s1100
                                 - 0.303000308356973 z_k1_s1101
                                 - 0.84000021445417 z_k1_s1102
                                 - 0.493000398864516 z_k1_s1103
                                 - 1.23600107155494 z_k1_s1104
                                 - 0.869000410858182 z_k1_s1105
                                 - 0.854000512028775 z_k1_s1106
                                 - 0.713003815035303 z_k1_s1107
                                 - 0.132999812914364 z_k1_s1110
                                 - 0.107999856086092 z_k1_s1111
                                 - 0.122000290913424 z_k1_s1112
                                 - 0.114999896049951 z_k1_s1113
                                 - 0.198999310917427 z_k1_s1114
                                 - 0.0989998385039495 z_k1_s1115
                                 - 0.159999392461307 z_k1_s1116
                                 - 0.246999483555432 z_k1_s1117
                                 - 0.249000156871228 z_k1_s1120
                                 - 0.299000242816501 z_k1_s1121
                                 - 0.302000328581286 z_k1_s1122
                                 - 0.260000227075232 z_k1_s1123
                                 - 0.296000157051715 z_k1_s1124
                                 - 0.853001913234739 z_k1_s1125
                                 - 0.868002342058666 z_k1_s1126
                                 - 0.536999963083411 z_k1_s1127
                                 - 0.503000419856732 z_k1_s1130
                                 - 0.613998928176182 z_k1_s1131
                                 - 0.489998954424129 z_k1_s1132
                                 - 0.200999907590058 z_k1_s1133
                                 - 0.224999891429174 z_k1_s1134
                                 - 0.502000835592296 z_k1_s1135
                                 - 0.520000222347075 z_k1_s1136
                                 - 0.380000030369114 z_k1_s1137
                                 - 0.294999848907325 z_k1_s1140
                                 - 0.164000034172209 z_k1_s1141
                                 - 0.324000464291443 z_k1_s1142
                                 - 0.666999628834527 z_k1_s1143
                                 - 0.15099986850002 z_k1_s1144
                                 - 0.316999414198713 z_k1_s1145
                                 - 0.41300095047027 z_k1_s1146
                                 - 0.573998957602716 z_k1_s1147
                                 - 0.178000097715153 z_k1_s1150
                                 - 0.37599987067928 z_k1_s1151
                                 - 0.308999962595456 z_k1_s1152
                                 - 0.512999638134274 z_k1_s1153
                                 - 0.658999923451149 z_k1_s1154
                                 - 0.545001110532158 z_k1_s1155
                                 - 0.482000805498208 z_k1_s1156
                                 - 0.25799979471081 z_k1_s1157
                                 - 0.384000757910219 z_k1_s1160
                                 - 0.263999818963338 z_k1_s1161
                                 - 0.190999927843925 z_k1_s1162
                                 - 0.239999895493938 z_k1_s1163
                                 - 0.666000436875613 z_k1_s1164
                                 - 0.844000117073639 z_k1_s1165
                                 - 0.361999423863394 z_k1_s1166
                                 - 0.35299932866238 z_k1_s1167
                                 - 1.01400360628877 z_k1_s1170
                                 - 0.365000844324924 z_k1_s1171
                                 - 0.477001009302204 z_k1_s1172
                                 - 0.687002749788109 z_k1_s1173
                                 - 0.711002512539716 z_k1_s1174
                                 - 0.57200343226888 z_k1_s1175
                                 - 1.10600087949881 z_k1_s1176
                                 - 0.520001038566613 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.20698 w_k1_ADM_s000
                                 + 1.34232 w_k1_AXP_s000
                                 + 0.946046 w_k1_DIS_s000
                                 + 1.11623 w_k1_FDX_s000
                                 + 0.944668 w_k1_IBM_s000
                                 + 1.10969 w_k1_INTC_s000
                                 + 0.951594 w_k1_KO_s000
                                 + 0.733217 w_k1_MAT_s000
                                 + 1.0656 w_k1_MSFT_s000
                                 + 0.979313 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.27747 w_k1_ADM_s000
                                 + 1.40566 w_k1_AXP_s000
                                 + 1.52374 w_k1_DIS_s000
                                 + 1.37124 w_k1_FDX_s000
                                 + 0.700597 w_k1_IBM_s000
                                 + 1.69907 w_k1_INTC_s000
                                 + 1.01502 w_k1_KO_s000 + 0.538 w_k1_MAT_s000
                                 + 1.34173 w_k1_MSFT_s000
                                 + 1.07627 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.936381 w_k1_ADM_s000
                                 + 0.929082 w_k1_AXP_s000
                                 + 1.14188 w_k1_DIS_s000
                                 + 1.08141 w_k1_FDX_s000
                                 + 0.910881 w_k1_IBM_s000
                                 + 1.46413 w_k1_INTC_s000
                                 + 1.02563 w_k1_KO_s000
                                 + 0.700865 w_k1_MAT_s000
                                 + 1.09578 w_k1_MSFT_s000
                                 + 0.979897 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.1287 w_k1_ADM_s000
                                 + 0.935998 w_k1_AXP_s000
                                 + 1.02691 w_k1_DIS_s000
                                 + 1.19788 w_k1_FDX_s000
                                 + 0.616833 w_k1_IBM_s000
                                 + 1.5493 w_k1_INTC_s000
                                 + 0.765404 w_k1_KO_s000
                                 + 0.46262 w_k1_MAT_s000
                                 + 1.22764 w_k1_MSFT_s000
                                 + 0.886698 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.486 w_k1_ADM_s000
                                 + 1.52866 w_k1_AXP_s000
                                 + 1.55672 w_k1_DIS_s000 + 1.3522 w_k1_FDX_s000
                                 + 1.30091 w_k1_IBM_s000
                                 + 1.67628 w_k1_INTC_s000 + 1.2817 w_k1_KO_s000
                                 + 1.03672 w_k1_MAT_s000
                                 + 1.30701 w_k1_MSFT_s000
                                 + 1.05468 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.809007 w_k1_ADM_s000
                                 + 1.0496 w_k1_AXP_s000 + 1.03807 w_k1_DIS_s000
                                 + 0.947497 w_k1_FDX_s000
                                 + 1.45753 w_k1_IBM_s000
                                 + 1.11548 w_k1_INTC_s000
                                 + 1.00855 w_k1_KO_s000 + 1.37798 w_k1_MAT_s000
                                 + 1.00121 w_k1_MSFT_s000
                                 + 0.921237 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.901739 w_k1_ADM_s000
                                 + 0.806578 w_k1_AXP_s000
                                 + 0.615172 w_k1_DIS_s000
                                 + 0.738754 w_k1_FDX_s000
                                 + 0.767383 w_k1_IBM_s000
                                 + 0.958422 w_k1_INTC_s000
                                 + 0.762828 w_k1_KO_s000
                                 + 0.639349 w_k1_MAT_s000
                                 + 0.791148 w_k1_MSFT_s000
                                 + 0.774546 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.904663 w_k1_ADM_s000
                                 + 0.875907 w_k1_AXP_s000
                                 + 0.879104 w_k1_DIS_s000
                                 + 0.889171 w_k1_FDX_s000
                                 + 1.04268 w_k1_IBM_s000
                                 + 1.16214 w_k1_INTC_s000
                                 + 0.847795 w_k1_KO_s000
                                 + 0.830395 w_k1_MAT_s000
                                 + 1.04489 w_k1_MSFT_s000
                                 + 0.757608 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.98082 w_k1_ADM_s001
                                 + 0.591952 w_k1_AXP_s001
                                 + 0.963034 w_k1_DIS_s001
                                 + 0.88366 w_k1_FDX_s001
                                 + 0.860888 w_k1_IBM_s001
                                 + 0.799125 w_k1_INTC_s001
                                 + 0.823708 w_k1_KO_s001
                                 + 0.536202 w_k1_MAT_s001
                                 + 0.902686 w_k1_MSFT_s001
                                 + 0.87612 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.2504 w_k1_ADM_s001
                                 + 1.17473 w_k1_AXP_s001
                                 + 1.35043 w_k1_DIS_s001
                                 + 1.37008 w_k1_FDX_s001
                                 + 0.813314 w_k1_IBM_s001
                                 + 1.17704 w_k1_INTC_s001
                                 + 0.877592 w_k1_KO_s001
                                 + 0.475949 w_k1_MAT_s001
                                 + 1.20165 w_k1_MSFT_s001
                                 + 0.924365 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.699877 w_k1_ADM_s001
                                 + 0.659209 w_k1_AXP_s001
                                 + 0.677902 w_k1_DIS_s001
                                 + 0.742423 w_k1_FDX_s001
                                 + 0.672645 w_k1_IBM_s001
                                 + 0.770437 w_k1_INTC_s001
                                 + 0.817263 w_k1_KO_s001
                                 + 0.705376 w_k1_MAT_s001
                                 + 0.811563 w_k1_MSFT_s001
                                 + 0.867661 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.0142 w_k1_ADM_s001
                                 + 0.890366 w_k1_AXP_s001
                                 + 0.827055 w_k1_DIS_s001
                                 + 1.00786 w_k1_FDX_s001
                                 + 0.836956 w_k1_IBM_s001
                                 + 0.988474 w_k1_INTC_s001
                                 + 1.02322 w_k1_KO_s001
                                 + 0.893846 w_k1_MAT_s001
                                 + 1.0191 w_k1_MSFT_s001
                                 + 0.973321 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.735822 w_k1_ADM_s001
                                 + 1.25557 w_k1_AXP_s001
                                 + 1.31673 w_k1_DIS_s001
                                 + 1.11404 w_k1_FDX_s001
                                 + 1.52665 w_k1_IBM_s001
                                 + 1.01468 w_k1_INTC_s001 + 1.2874 w_k1_KO_s001
                                 + 1.97127 w_k1_MAT_s001
                                 + 1.01444 w_k1_MSFT_s001 + 1.1007 w_k1_PG_s001
                                 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.63824 w_k1_ADM_s001
                                 + 1.85662 w_k1_AXP_s001
                                 + 1.77425 w_k1_DIS_s001
                                 + 1.61058 w_k1_FDX_s001
                                 + 1.53933 w_k1_IBM_s001
                                 + 1.39858 w_k1_INTC_s001
                                 + 1.43053 w_k1_KO_s001 + 1.72389 w_k1_MAT_s001
                                 + 1.35779 w_k1_MSFT_s001 + 1.3397 w_k1_PG_s001
                                 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.780725 w_k1_ADM_s001
                                 + 0.786512 w_k1_AXP_s001
                                 + 0.99331 w_k1_DIS_s001
                                 + 0.900502 w_k1_FDX_s001
                                 + 1.26202 w_k1_IBM_s001
                                 + 0.974675 w_k1_INTC_s001
                                 + 1.11017 w_k1_KO_s001 + 1.09325 w_k1_MAT_s001
                                 + 0.934417 w_k1_MSFT_s001
                                 + 1.06007 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.858364 w_k1_ADM_s001
                                 + 1.03396 w_k1_AXP_s001
                                 + 1.13093 w_k1_DIS_s001
                                 + 1.15408 w_k1_FDX_s001
                                 + 0.788685 w_k1_IBM_s001
                                 + 0.95122 w_k1_INTC_s001
                                 + 0.838427 w_k1_KO_s001
                                 + 0.712341 w_k1_MAT_s001
                                 + 0.999654 w_k1_MSFT_s001
                                 + 0.986316 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.696268 w_k1_ADM_s002
                                 + 0.890608 w_k1_AXP_s002
                                 + 1.23014 w_k1_DIS_s002
                                 + 0.972722 w_k1_FDX_s002
                                 + 1.4377 w_k1_IBM_s002
                                 + 0.946547 w_k1_INTC_s002
                                 + 1.20399 w_k1_KO_s002 + 1.10576 w_k1_MAT_s002
                                 + 1.05182 w_k1_MSFT_s002
                                 + 0.940843 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.10236 w_k1_ADM_s002
                                 + 1.62897 w_k1_AXP_s002
                                 + 1.57645 w_k1_DIS_s002
                                 + 1.23225 w_k1_FDX_s002 + 1.7566 w_k1_IBM_s002
                                 + 0.990167 w_k1_INTC_s002
                                 + 1.25448 w_k1_KO_s002 + 1.7049 w_k1_MAT_s002
                                 + 1.01555 w_k1_MSFT_s002
                                 + 1.08504 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.920803 w_k1_ADM_s002
                                 + 1.59348 w_k1_AXP_s002
                                 + 1.66615 w_k1_DIS_s002
                                 + 1.35263 w_k1_FDX_s002
                                 + 1.06881 w_k1_IBM_s002
                                 + 1.68571 w_k1_INTC_s002 + 1.1761 w_k1_KO_s002
                                 + 0.810226 w_k1_MAT_s002
                                 + 1.37717 w_k1_MSFT_s002
                                 + 1.22259 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.57559 w_k1_ADM_s002
                                 + 0.664185 w_k1_AXP_s002
                                 + 0.54907 w_k1_DIS_s002
                                 + 0.714287 w_k1_FDX_s002
                                 + 0.710301 w_k1_IBM_s002
                                 + 0.895983 w_k1_INTC_s002
                                 + 0.681671 w_k1_KO_s002
                                 + 0.542924 w_k1_MAT_s002
                                 + 0.674442 w_k1_MSFT_s002
                                 + 0.710566 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.991162 w_k1_ADM_s002
                                 + 0.913435 w_k1_AXP_s002
                                 + 0.847277 w_k1_DIS_s002
                                 + 0.998255 w_k1_FDX_s002
                                 + 0.725356 w_k1_IBM_s002
                                 + 1.10979 w_k1_INTC_s002
                                 + 0.710066 w_k1_KO_s002
                                 + 0.581301 w_k1_MAT_s002
                                 + 0.968585 w_k1_MSFT_s002
                                 + 0.884841 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.809763 w_k1_ADM_s002
                                 + 0.893456 w_k1_AXP_s002
                                 + 0.986805 w_k1_DIS_s002
                                 + 0.968119 w_k1_FDX_s002
                                 + 0.890669 w_k1_IBM_s002
                                 + 1.2094 w_k1_INTC_s002 + 1.02371 w_k1_KO_s002
                                 + 0.819125 w_k1_MAT_s002
                                 + 1.1379 w_k1_MSFT_s002
                                 + 0.877064 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.0336 w_k1_ADM_s002
                                 + 0.996872 w_k1_AXP_s002
                                 + 0.915775 w_k1_DIS_s002
                                 + 0.936007 w_k1_FDX_s002
                                 + 1.08037 w_k1_IBM_s002
                                 + 1.04353 w_k1_INTC_s002
                                 + 0.943497 w_k1_KO_s002
                                 + 1.05885 w_k1_MAT_s002
                                 + 0.954878 w_k1_MSFT_s002
                                 + 1.0418 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.864701 w_k1_ADM_s002
                                 + 1.59273 w_k1_AXP_s002
                                 + 1.05832 w_k1_DIS_s002
                                 + 1.11835 w_k1_FDX_s002
                                 + 1.40099 w_k1_IBM_s002
                                 + 1.59862 w_k1_INTC_s002
                                 + 1.13745 w_k1_KO_s002 + 1.27676 w_k1_MAT_s002
                                 + 1.10524 w_k1_MSFT_s002
                                 + 1.11896 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.848812 w_k1_ADM_s003
                                 + 0.744246 w_k1_AXP_s003
                                 + 0.766536 w_k1_DIS_s003
                                 + 0.838747 w_k1_FDX_s003
                                 + 0.751174 w_k1_IBM_s003
                                 + 0.876871 w_k1_INTC_s003
                                 + 0.739361 w_k1_KO_s003
                                 + 0.64535 w_k1_MAT_s003
                                 + 0.777892 w_k1_MSFT_s003
                                 + 0.947927 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.20838 w_k1_ADM_s003
                                 + 1.55115 w_k1_AXP_s003
                                 + 1.77751 w_k1_DIS_s003
                                 + 1.48936 w_k1_FDX_s003
                                 + 1.20902 w_k1_IBM_s003
                                 + 1.38868 w_k1_INTC_s003
                                 + 1.52388 w_k1_KO_s003 + 1.86365 w_k1_MAT_s003
                                 + 1.21787 w_k1_MSFT_s003 + 1.4704 w_k1_PG_s003
                                 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.733061 w_k1_ADM_s003
                                 + 0.785622 w_k1_AXP_s003
                                 + 0.833096 w_k1_DIS_s003
                                 + 0.747167 w_k1_FDX_s003
                                 + 1.23308 w_k1_IBM_s003
                                 + 0.815028 w_k1_INTC_s003
                                 + 1.07835 w_k1_KO_s003 + 1.25918 w_k1_MAT_s003
                                 + 0.777497 w_k1_MSFT_s003
                                 + 0.973528 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.1072 w_k1_ADM_s003
                                 + 1.31265 w_k1_AXP_s003
                                 + 1.33656 w_k1_DIS_s003
                                 + 1.17315 w_k1_FDX_s003
                                 + 1.42899 w_k1_IBM_s003
                                 + 1.10441 w_k1_INTC_s003
                                 + 1.27089 w_k1_KO_s003 + 1.97602 w_k1_MAT_s003
                                 + 1.10289 w_k1_MSFT_s003
                                 + 1.20204 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.04402 w_k1_ADM_s003
                                 + 1.02704 w_k1_AXP_s003
                                 + 1.00649 w_k1_DIS_s003
                                 + 0.886612 w_k1_FDX_s003
                                 + 0.96805 w_k1_IBM_s003
                                 + 0.998234 w_k1_INTC_s003
                                 + 1.14835 w_k1_KO_s003 + 1.4589 w_k1_MAT_s003
                                 + 0.977904 w_k1_MSFT_s003
                                 + 1.01384 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.11424 w_k1_ADM_s003
                                 + 1.1192 w_k1_AXP_s003 + 1.10661 w_k1_DIS_s003
                                 + 1.15413 w_k1_FDX_s003
                                 + 0.840534 w_k1_IBM_s003
                                 + 1.01071 w_k1_INTC_s003
                                 + 0.964039 w_k1_KO_s003
                                 + 1.32966 w_k1_MAT_s003
                                 + 1.03772 w_k1_MSFT_s003
                                 + 1.19173 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.83942 w_k1_ADM_s003
                                 + 0.915062 w_k1_AXP_s003
                                 + 0.888883 w_k1_DIS_s003
                                 + 0.854135 w_k1_FDX_s003
                                 + 1.34129 w_k1_IBM_s003
                                 + 0.899572 w_k1_INTC_s003
                                 + 1.00317 w_k1_KO_s003 + 1.11033 w_k1_MAT_s003
                                 + 0.865152 w_k1_MSFT_s003
                                 + 0.979861 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.905911 w_k1_ADM_s003
                                 + 1.00873 w_k1_AXP_s003
                                 + 1.21322 w_k1_DIS_s003
                                 + 0.975671 w_k1_FDX_s003
                                 + 1.57121 w_k1_IBM_s003
                                 + 1.13612 w_k1_INTC_s003
                                 + 1.25351 w_k1_KO_s003 + 1.33709 w_k1_MAT_s003
                                 + 1.00039 w_k1_MSFT_s003
                                 + 1.15899 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.699208 w_k1_ADM_s004
                                 + 0.797589 w_k1_AXP_s004
                                 + 1.19126 w_k1_DIS_s004
                                 + 0.955281 w_k1_FDX_s004
                                 + 1.35206 w_k1_IBM_s004
                                 + 1.17755 w_k1_INTC_s004
                                 + 1.29317 w_k1_KO_s004 + 1.18652 w_k1_MAT_s004
                                 + 1.13785 w_k1_MSFT_s004
                                 + 1.15871 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.986908 w_k1_ADM_s004
                                 + 1.09939 w_k1_AXP_s004 + 1.0254 w_k1_DIS_s004
                                 + 0.967743 w_k1_FDX_s004
                                 + 1.39962 w_k1_IBM_s004
                                 + 0.979805 w_k1_INTC_s004
                                 + 1.4571 w_k1_KO_s004 + 1.67145 w_k1_MAT_s004
                                 + 1.04568 w_k1_MSFT_s004
                                 + 1.16776 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.2544 w_k1_ADM_s004
                                 + 0.965626 w_k1_AXP_s004
                                 + 1.13941 w_k1_DIS_s004 + 1.1932 w_k1_FDX_s004
                                 + 0.783665 w_k1_IBM_s004
                                 + 1.22385 w_k1_INTC_s004
                                 + 1.00535 w_k1_KO_s004
                                 + 0.703746 w_k1_MAT_s004
                                 + 1.03616 w_k1_MSFT_s004
                                 + 1.08965 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.655313 w_k1_ADM_s004
                                 + 0.640046 w_k1_AXP_s004
                                 + 0.706966 w_k1_DIS_s004
                                 + 0.775324 w_k1_FDX_s004
                                 + 0.860942 w_k1_IBM_s004
                                 + 0.887263 w_k1_INTC_s004
                                 + 0.918473 w_k1_KO_s004
                                 + 0.857846 w_k1_MAT_s004
                                 + 0.887577 w_k1_MSFT_s004
                                 + 0.984912 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.03868 w_k1_ADM_s004
                                 + 0.943584 w_k1_AXP_s004
                                 + 1.18931 w_k1_DIS_s004
                                 + 1.13747 w_k1_FDX_s004
                                 + 1.03389 w_k1_IBM_s004
                                 + 0.98353 w_k1_INTC_s004
                                 + 1.18634 w_k1_KO_s004 + 1.0467 w_k1_MAT_s004
                                 + 1.09647 w_k1_MSFT_s004
                                 + 1.29804 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.949921 w_k1_ADM_s004
                                 + 0.847526 w_k1_AXP_s004
                                 + 0.904468 w_k1_DIS_s004
                                 + 0.947487 w_k1_FDX_s004
                                 + 0.889136 w_k1_IBM_s004
                                 + 0.992286 w_k1_INTC_s004
                                 + 0.910015 w_k1_KO_s004
                                 + 0.698914 w_k1_MAT_s004
                                 + 0.833066 w_k1_MSFT_s004
                                 + 1.03047 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.20263 w_k1_ADM_s004
                                 + 1.05666 w_k1_AXP_s004
                                 + 1.25501 w_k1_DIS_s004
                                 + 1.20216 w_k1_FDX_s004
                                 + 1.21051 w_k1_IBM_s004
                                 + 1.30372 w_k1_INTC_s004
                                 + 1.24765 w_k1_KO_s004
                                 + 0.906554 w_k1_MAT_s004
                                 + 1.17294 w_k1_MSFT_s004
                                 + 1.17515 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.22236 w_k1_ADM_s004
                                 + 1.72488 w_k1_AXP_s004
                                 + 1.54078 w_k1_DIS_s004
                                 + 1.47324 w_k1_FDX_s004
                                 + 1.54877 w_k1_IBM_s004
                                 + 1.33126 w_k1_INTC_s004
                                 + 1.45844 w_k1_KO_s004 + 1.9959 w_k1_MAT_s004
                                 + 1.41217 w_k1_MSFT_s004
                                 + 1.23577 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.846434 w_k1_ADM_s005
                                 + 1.00055 w_k1_AXP_s005
                                 + 0.970803 w_k1_DIS_s005
                                 + 0.982928 w_k1_FDX_s005
                                 + 1.17694 w_k1_IBM_s005
                                 + 0.9996 w_k1_INTC_s005 + 1.07965 w_k1_KO_s005
                                 + 1.21954 w_k1_MAT_s005
                                 + 1.06896 w_k1_MSFT_s005
                                 + 1.06041 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.884022 w_k1_ADM_s005
                                 + 1.10684 w_k1_AXP_s005
                                 + 1.37413 w_k1_DIS_s005
                                 + 1.19613 w_k1_FDX_s005
                                 + 1.08283 w_k1_IBM_s005
                                 + 1.32125 w_k1_INTC_s005 + 1.3015 w_k1_KO_s005
                                 + 0.893983 w_k1_MAT_s005
                                 + 1.25368 w_k1_MSFT_s005
                                 + 1.25384 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.76331 w_k1_ADM_s005
                                 + 0.765786 w_k1_AXP_s005
                                 + 0.632525 w_k1_DIS_s005
                                 + 0.777715 w_k1_FDX_s005
                                 + 0.897804 w_k1_IBM_s005
                                 + 0.883994 w_k1_INTC_s005
                                 + 0.853838 w_k1_KO_s005 + 0.6698 w_k1_MAT_s005
                                 + 0.822082 w_k1_MSFT_s005
                                 + 0.969632 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.1956 w_k1_ADM_s005
                                 + 0.801654 w_k1_AXP_s005
                                 + 1.06358 w_k1_DIS_s005
                                 + 1.08787 w_k1_FDX_s005
                                 + 0.911334 w_k1_IBM_s005
                                 + 1.38761 w_k1_INTC_s005 + 1.0959 w_k1_KO_s005
                                 + 0.670658 w_k1_MAT_s005
                                 + 1.16305 w_k1_MSFT_s005
                                 + 1.00782 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.844356 w_k1_ADM_s005
                                 + 1.71089 w_k1_AXP_s005
                                 + 1.24646 w_k1_DIS_s005
                                 + 1.30065 w_k1_FDX_s005
                                 + 1.26254 w_k1_IBM_s005
                                 + 1.52443 w_k1_INTC_s005
                                 + 1.22323 w_k1_KO_s005 + 1.24637 w_k1_MAT_s005
                                 + 1.02304 w_k1_MSFT_s005
                                 + 1.18185 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.09621 w_k1_ADM_s005
                                 + 0.966381 w_k1_AXP_s005
                                 + 0.741731 w_k1_DIS_s005
                                 + 0.931715 w_k1_FDX_s005
                                 + 0.88931 w_k1_IBM_s005
                                 + 0.989401 w_k1_INTC_s005
                                 + 0.942175 w_k1_KO_s005
                                 + 0.839203 w_k1_MAT_s005
                                 + 0.8189 w_k1_MSFT_s005
                                 + 0.884848 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.11497 w_k1_ADM_s005
                                 + 1.82259 w_k1_AXP_s005
                                 + 1.54423 w_k1_DIS_s005
                                 + 1.37099 w_k1_FDX_s005
                                 + 1.13862 w_k1_IBM_s005
                                 + 1.46937 w_k1_INTC_s005
                                 + 1.32569 w_k1_KO_s005 + 1.16124 w_k1_MAT_s005
                                 + 1.37626 w_k1_MSFT_s005 + 1.2302 w_k1_PG_s005
                                 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.788978 w_k1_ADM_s005
                                 + 0.693445 w_k1_AXP_s005
                                 + 0.892034 w_k1_DIS_s005
                                 + 1.04378 w_k1_FDX_s005
                                 + 0.634141 w_k1_IBM_s005
                                 + 1.12218 w_k1_INTC_s005
                                 + 0.815688 w_k1_KO_s005
                                 + 0.446601 w_k1_MAT_s005
                                 + 1.04492 w_k1_MSFT_s005
                                 + 0.995863 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.48228 w_k1_ADM_s006
                                 + 1.8238 w_k1_AXP_s006 + 1.3487 w_k1_DIS_s006
                                 + 1.17089 w_k1_FDX_s006
                                 + 1.40712 w_k1_IBM_s006
                                 + 1.0675 w_k1_INTC_s006 + 1.37024 w_k1_KO_s006
                                 + 1.32326 w_k1_MAT_s006
                                 + 0.985525 w_k1_MSFT_s006
                                 + 1.27317 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.11877 w_k1_ADM_s006
                                 + 1.1508 w_k1_AXP_s006
                                 + 0.901509 w_k1_DIS_s006
                                 + 0.901014 w_k1_FDX_s006
                                 + 1.09917 w_k1_IBM_s006
                                 + 0.743175 w_k1_INTC_s006
                                 + 1.03781 w_k1_KO_s006 + 1.09801 w_k1_MAT_s006
                                 + 0.808691 w_k1_MSFT_s006
                                 + 1.11341 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.51221 w_k1_ADM_s006
                                 + 1.06698 w_k1_AXP_s006 + 1.2881 w_k1_DIS_s006
                                 + 1.1075 w_k1_FDX_s006 + 1.16332 w_k1_IBM_s006
                                 + 1.06744 w_k1_INTC_s006
                                 + 1.24925 w_k1_KO_s006
                                 + 0.891498 w_k1_MAT_s006
                                 + 1.05295 w_k1_MSFT_s006
                                 + 1.22134 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.962537 w_k1_ADM_s006
                                 + 0.989794 w_k1_AXP_s006
                                 + 1.09565 w_k1_DIS_s006
                                 + 0.969846 w_k1_FDX_s006
                                 + 1.63489 w_k1_IBM_s006
                                 + 0.873747 w_k1_INTC_s006
                                 + 1.2908 w_k1_KO_s006 + 1.32099 w_k1_MAT_s006
                                 + 0.947677 w_k1_MSFT_s006
                                 + 1.22932 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.729721 w_k1_ADM_s006
                                 + 0.639572 w_k1_AXP_s006
                                 + 0.658966 w_k1_DIS_s006
                                 + 0.727753 w_k1_FDX_s006
                                 + 0.713956 w_k1_IBM_s006
                                 + 0.659064 w_k1_INTC_s006
                                 + 0.730662 w_k1_KO_s006
                                 + 0.76422 w_k1_MAT_s006
                                 + 0.801896 w_k1_MSFT_s006
                                 + 0.855573 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.03685 w_k1_ADM_s006
                                 + 0.87958 w_k1_AXP_s006
                                 + 0.888288 w_k1_DIS_s006
                                 + 0.929421 w_k1_FDX_s006
                                 + 0.617501 w_k1_IBM_s006
                                 + 0.797537 w_k1_INTC_s006
                                 + 0.667863 w_k1_KO_s006
                                 + 0.727436 w_k1_MAT_s006
                                 + 0.899429 w_k1_MSFT_s006
                                 + 0.928385 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.687673 w_k1_ADM_s006
                                 + 0.754384 w_k1_AXP_s006
                                 + 0.707541 w_k1_DIS_s006
                                 + 0.730686 w_k1_FDX_s006
                                 + 1.0752 w_k1_IBM_s006
                                 + 0.85763 w_k1_INTC_s006
                                 + 0.966396 w_k1_KO_s006
                                 + 0.945238 w_k1_MAT_s006
                                 + 0.840133 w_k1_MSFT_s006
                                 + 0.96231 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.999749 w_k1_ADM_s006
                                 + 0.919172 w_k1_AXP_s006
                                 + 1.04149 w_k1_DIS_s006
                                 + 1.03349 w_k1_FDX_s006
                                 + 0.84789 w_k1_IBM_s006
                                 + 0.821414 w_k1_INTC_s006
                                 + 1.01411 w_k1_KO_s006
                                 + 0.824389 w_k1_MAT_s006
                                 + 0.949391 w_k1_MSFT_s006
                                 + 1.08755 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.10944 w_k1_ADM_s007
                                 + 1.57133 w_k1_AXP_s007
                                 + 1.90892 w_k1_DIS_s007
                                 + 1.28551 w_k1_FDX_s007
                                 + 1.70595 w_k1_IBM_s007
                                 + 1.62867 w_k1_INTC_s007
                                 + 1.53434 w_k1_KO_s007 + 1.34337 w_k1_MAT_s007
                                 + 1.347 w_k1_MSFT_s007 + 1.27748 w_k1_PG_s007
                                 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.09218 w_k1_ADM_s007
                                 + 1.26932 w_k1_AXP_s007
                                 + 1.13936 w_k1_DIS_s007
                                 + 1.13149 w_k1_FDX_s007
                                 + 0.896839 w_k1_IBM_s007
                                 + 1.11504 w_k1_INTC_s007
                                 + 0.92979 w_k1_KO_s007 + 1.14035 w_k1_MAT_s007
                                 + 1.1473 w_k1_MSFT_s007
                                 + 0.941124 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.938486 w_k1_ADM_s007
                                 + 0.984435 w_k1_AXP_s007
                                 + 1.29923 w_k1_DIS_s007
                                 + 1.16833 w_k1_FDX_s007
                                 + 1.07579 w_k1_IBM_s007
                                 + 1.17296 w_k1_INTC_s007
                                 + 1.21685 w_k1_KO_s007 + 1.09817 w_k1_MAT_s007
                                 + 1.16829 w_k1_MSFT_s007
                                 + 1.02865 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.740787 w_k1_ADM_s007
                                 + 0.859506 w_k1_AXP_s007
                                 + 0.833708 w_k1_DIS_s007
                                 + 0.833601 w_k1_FDX_s007
                                 + 0.913325 w_k1_IBM_s007
                                 + 0.855283 w_k1_INTC_s007
                                 + 0.768393 w_k1_KO_s007
                                 + 0.844654 w_k1_MAT_s007
                                 + 0.834364 w_k1_MSFT_s007
                                 + 0.890232 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.784748 w_k1_ADM_s007
                                 + 0.609686 w_k1_AXP_s007
                                 + 0.57623 w_k1_DIS_s007
                                 + 0.680521 w_k1_FDX_s007
                                 + 0.763762 w_k1_IBM_s007
                                 + 0.894474 w_k1_INTC_s007
                                 + 0.689929 w_k1_KO_s007
                                 + 0.536731 w_k1_MAT_s007
                                 + 0.821244 w_k1_MSFT_s007
                                 + 0.698428 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.927155 w_k1_ADM_s007
                                 + 0.779083 w_k1_AXP_s007
                                 + 1.01196 w_k1_DIS_s007
                                 + 0.992231 w_k1_FDX_s007
                                 + 0.897436 w_k1_IBM_s007
                                 + 1.12445 w_k1_INTC_s007
                                 + 0.939316 w_k1_KO_s007
                                 + 0.636294 w_k1_MAT_s007
                                 + 1.08448 w_k1_MSFT_s007
                                 + 0.920945 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.03072 w_k1_ADM_s007
                                 + 1.19275 w_k1_AXP_s007
                                 + 1.08176 w_k1_DIS_s007
                                 + 0.961317 w_k1_FDX_s007
                                 + 1.40393 w_k1_IBM_s007
                                 + 1.20781 w_k1_INTC_s007
                                 + 1.28168 w_k1_KO_s007 + 1.30583 w_k1_MAT_s007
                                 + 1.07974 w_k1_MSFT_s007
                                 + 0.938228 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.02027 w_k1_ADM_s007
                                 + 0.708567 w_k1_AXP_s007
                                 + 0.862405 w_k1_DIS_s007
                                 + 1.01509 w_k1_FDX_s007
                                 + 0.630024 w_k1_IBM_s007
                                 + 0.930388 w_k1_INTC_s007
                                 + 0.814794 w_k1_KO_s007
                                 + 0.469737 w_k1_MAT_s007
                                 + 0.947249 w_k1_MSFT_s007
                                 + 0.910088 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.816635 w_k1_ADM_s010
                                 + 1.03577 w_k1_AXP_s010
                                 + 0.780887 w_k1_DIS_s010
                                 + 0.837462 w_k1_FDX_s010
                                 + 1.30252 w_k1_IBM_s010
                                 + 1.08679 w_k1_INTC_s010
                                 + 0.906184 w_k1_KO_s010
                                 + 0.858009 w_k1_MAT_s010
                                 + 0.975753 w_k1_MSFT_s010
                                 + 0.973813 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.895606 w_k1_ADM_s010
                                 + 0.95741 w_k1_AXP_s010
                                 + 0.819514 w_k1_DIS_s010
                                 + 0.838004 w_k1_FDX_s010
                                 + 1.10514 w_k1_IBM_s010
                                 + 0.726644 w_k1_INTC_s010
                                 + 0.86373 w_k1_KO_s010
                                 + 0.792847 w_k1_MAT_s010
                                 + 0.781788 w_k1_MSFT_s010
                                 + 1.03667 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.06794 w_k1_ADM_s010
                                 + 1.09684 w_k1_AXP_s010
                                 + 1.15107 w_k1_DIS_s010
                                 + 1.16844 w_k1_FDX_s010
                                 + 0.871063 w_k1_IBM_s010
                                 + 1.14391 w_k1_INTC_s010
                                 + 1.02685 w_k1_KO_s010
                                 + 0.666406 w_k1_MAT_s010
                                 + 1.32051 w_k1_MSFT_s010
                                 + 1.21323 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.0503 w_k1_ADM_s010
                                 + 1.30154 w_k1_AXP_s010
                                 + 1.16028 w_k1_DIS_s010
                                 + 1.08119 w_k1_FDX_s010
                                 + 1.04211 w_k1_IBM_s010
                                 + 1.0987 w_k1_INTC_s010 + 1.10985 w_k1_KO_s010
                                 + 1.29237 w_k1_MAT_s010
                                 + 1.17834 w_k1_MSFT_s010
                                 + 1.21144 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.792158 w_k1_ADM_s010
                                 + 0.878583 w_k1_AXP_s010
                                 + 0.671913 w_k1_DIS_s010
                                 + 0.863534 w_k1_FDX_s010
                                 + 0.783018 w_k1_IBM_s010
                                 + 0.837751 w_k1_INTC_s010
                                 + 0.680793 w_k1_KO_s010
                                 + 0.624882 w_k1_MAT_s010
                                 + 0.797292 w_k1_MSFT_s010
                                 + 0.883559 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.11328 w_k1_ADM_s010
                                 + 1.38851 w_k1_AXP_s010
                                 + 0.991043 w_k1_DIS_s010
                                 + 0.990251 w_k1_FDX_s010
                                 + 1.45696 w_k1_IBM_s010
                                 + 1.14425 w_k1_INTC_s010
                                 + 1.08706 w_k1_KO_s010 + 1.01582 w_k1_MAT_s010
                                 + 1.03762 w_k1_MSFT_s010
                                 + 1.22233 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.06786 w_k1_ADM_s010
                                 + 1.59256 w_k1_AXP_s010
                                 + 1.53664 w_k1_DIS_s010
                                 + 1.25372 w_k1_FDX_s010
                                 + 1.33596 w_k1_IBM_s010
                                 + 1.21439 w_k1_INTC_s010
                                 + 1.19538 w_k1_KO_s010 + 1.87411 w_k1_MAT_s010
                                 + 1.12589 w_k1_MSFT_s010
                                 + 1.51837 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.775925 w_k1_ADM_s010
                                 + 0.639433 w_k1_AXP_s010
                                 + 0.69367 w_k1_DIS_s010
                                 + 0.737321 w_k1_FDX_s010
                                 + 0.781989 w_k1_IBM_s010
                                 + 0.59192 w_k1_INTC_s010
                                 + 0.595903 w_k1_KO_s010
                                 + 0.604336 w_k1_MAT_s010
                                 + 0.782188 w_k1_MSFT_s010
                                 + 0.795791 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.38954 w_k1_ADM_s011
                                 + 1.21929 w_k1_AXP_s011
                                 + 1.40203 w_k1_DIS_s011
                                 + 1.19814 w_k1_FDX_s011
                                 + 1.11031 w_k1_IBM_s011
                                 + 1.0945 w_k1_INTC_s011 + 1.10357 w_k1_KO_s011
                                 + 0.828302 w_k1_MAT_s011
                                 + 1.14306 w_k1_MSFT_s011
                                 + 1.18377 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.806143 w_k1_ADM_s011
                                 + 0.771287 w_k1_AXP_s011
                                 + 0.792035 w_k1_DIS_s011
                                 + 0.955366 w_k1_FDX_s011
                                 + 0.637032 w_k1_IBM_s011
                                 + 0.868551 w_k1_INTC_s011
                                 + 0.801869 w_k1_KO_s011
                                 + 0.835469 w_k1_MAT_s011
                                 + 0.856857 w_k1_MSFT_s011
                                 + 0.850738 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.862585 w_k1_ADM_s011
                                 + 0.62669 w_k1_AXP_s011
                                 + 0.686685 w_k1_DIS_s011
                                 + 0.702483 w_k1_FDX_s011
                                 + 1.05768 w_k1_IBM_s011
                                 + 0.828121 w_k1_INTC_s011
                                 + 0.896994 w_k1_KO_s011
                                 + 0.772274 w_k1_MAT_s011
                                 + 0.739902 w_k1_MSFT_s011
                                 + 0.861881 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.20764 w_k1_ADM_s011
                                 + 0.892135 w_k1_AXP_s011
                                 + 1.25434 w_k1_DIS_s011
                                 + 1.06675 w_k1_FDX_s011
                                 + 0.916408 w_k1_IBM_s011
                                 + 1.14014 w_k1_INTC_s011 + 1.1595 w_k1_KO_s011
                                 + 0.624715 w_k1_MAT_s011
                                 + 1.15497 w_k1_MSFT_s011
                                 + 0.937338 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.04888 w_k1_ADM_s011
                                 + 1.09791 w_k1_AXP_s011
                                 + 1.14557 w_k1_DIS_s011
                                 + 1.29023 w_k1_FDX_s011
                                 + 0.885385 w_k1_IBM_s011
                                 + 1.17116 w_k1_INTC_s011
                                 + 0.945412 w_k1_KO_s011
                                 + 0.890414 w_k1_MAT_s011
                                 + 1.07082 w_k1_MSFT_s011
                                 + 0.923029 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.876662 w_k1_ADM_s011
                                 + 1.0976 w_k1_AXP_s011
                                 + 0.957847 w_k1_DIS_s011
                                 + 0.888685 w_k1_FDX_s011
                                 + 1.23326 w_k1_IBM_s011
                                 + 1.04341 w_k1_INTC_s011
                                 + 1.16554 w_k1_KO_s011 + 1.15432 w_k1_MAT_s011
                                 + 1.07042 w_k1_MSFT_s011
                                 + 0.924894 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.947086 w_k1_ADM_s011
                                 + 0.961189 w_k1_AXP_s011
                                 + 1.33698 w_k1_DIS_s011
                                 + 1.01741 w_k1_FDX_s011
                                 + 1.77366 w_k1_IBM_s011
                                 + 1.14807 w_k1_INTC_s011
                                 + 1.66058 w_k1_KO_s011 + 1.80135 w_k1_MAT_s011
                                 + 1.2216 w_k1_MSFT_s011 + 1.27563 w_k1_PG_s011
                                 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.07672 w_k1_ADM_s011
                                 + 1.44344 w_k1_AXP_s011
                                 + 1.88754 w_k1_DIS_s011
                                 + 1.46314 w_k1_FDX_s011
                                 + 1.46793 w_k1_IBM_s011
                                 + 1.47329 w_k1_INTC_s011
                                 + 1.40133 w_k1_KO_s011 + 1.4465 w_k1_MAT_s011
                                 + 1.36466 w_k1_MSFT_s011
                                 + 1.24764 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.844185 w_k1_ADM_s012
                                 + 0.695337 w_k1_AXP_s012
                                 + 0.728001 w_k1_DIS_s012
                                 + 0.81355 w_k1_FDX_s012
                                 + 0.816267 w_k1_IBM_s012
                                 + 0.710528 w_k1_INTC_s012
                                 + 0.808483 w_k1_KO_s012
                                 + 0.825782 w_k1_MAT_s012
                                 + 0.928193 w_k1_MSFT_s012
                                 + 0.991602 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.85032 w_k1_ADM_s012
                                 + 1.41432 w_k1_AXP_s012
                                 + 1.17359 w_k1_DIS_s012
                                 + 1.14183 w_k1_FDX_s012
                                 + 1.14282 w_k1_IBM_s012
                                 + 1.00268 w_k1_INTC_s012 + 1.0845 w_k1_KO_s012
                                 + 1.64171 w_k1_MAT_s012
                                 + 1.15612 w_k1_MSFT_s012
                                 + 1.11533 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.712905 w_k1_ADM_s012
                                 + 0.796039 w_k1_AXP_s012
                                 + 0.949219 w_k1_DIS_s012
                                 + 0.918092 w_k1_FDX_s012
                                 + 0.836159 w_k1_IBM_s012
                                 + 0.970807 w_k1_INTC_s012
                                 + 0.937677 w_k1_KO_s012
                                 + 0.808163 w_k1_MAT_s012
                                 + 1.11662 w_k1_MSFT_s012 + 1.0856 w_k1_PG_s012
                                 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.865833 w_k1_ADM_s012
                                 + 0.987929 w_k1_AXP_s012
                                 + 0.921471 w_k1_DIS_s012
                                 + 1.05764 w_k1_FDX_s012
                                 + 0.716476 w_k1_IBM_s012
                                 + 0.90414 w_k1_INTC_s012
                                 + 0.870374 w_k1_KO_s012 + 0.609 w_k1_MAT_s012
                                 + 1.05898 w_k1_MSFT_s012
                                 + 0.952656 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.569507 w_k1_ADM_s012
                                 + 0.861663 w_k1_AXP_s012
                                 + 0.735427 w_k1_DIS_s012
                                 + 0.760347 w_k1_FDX_s012
                                 + 1.12351 w_k1_IBM_s012
                                 + 0.847692 w_k1_INTC_s012
                                 + 0.952461 w_k1_KO_s012
                                 + 1.17366 w_k1_MAT_s012
                                 + 0.979833 w_k1_MSFT_s012
                                 + 0.917219 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.05092 w_k1_ADM_s012
                                 + 1.30519 w_k1_AXP_s012
                                 + 1.49674 w_k1_DIS_s012
                                 + 1.26174 w_k1_FDX_s012
                                 + 1.45984 w_k1_IBM_s012
                                 + 1.16688 w_k1_INTC_s012
                                 + 1.46516 w_k1_KO_s012 + 1.9059 w_k1_MAT_s012
                                 + 1.37639 w_k1_MSFT_s012
                                 + 1.19127 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.02625 w_k1_ADM_s012
                                 + 0.906667 w_k1_AXP_s012
                                 + 1.31807 w_k1_DIS_s012
                                 + 1.18946 w_k1_FDX_s012
                                 + 0.895975 w_k1_IBM_s012
                                 + 1.01936 w_k1_INTC_s012
                                 + 1.14672 w_k1_KO_s012 + 1.2806 w_k1_MAT_s012
                                 + 1.39519 w_k1_MSFT_s012
                                 + 1.40342 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.930404 w_k1_ADM_s012
                                 + 0.899218 w_k1_AXP_s012
                                 + 1.12185 w_k1_DIS_s012
                                 + 1.03275 w_k1_FDX_s012
                                 + 0.984322 w_k1_IBM_s012
                                 + 1.07721 w_k1_INTC_s012
                                 + 1.16404 w_k1_KO_s012
                                 + 0.754137 w_k1_MAT_s012
                                 + 1.28906 w_k1_MSFT_s012
                                 + 1.01333 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.816457 w_k1_ADM_s013
                                 + 0.81895 w_k1_AXP_s013
                                 + 0.906752 w_k1_DIS_s013
                                 + 0.952843 w_k1_FDX_s013
                                 + 0.914304 w_k1_IBM_s013
                                 + 1.06367 w_k1_INTC_s013
                                 + 0.964144 w_k1_KO_s013
                                 + 0.801925 w_k1_MAT_s013
                                 + 0.90503 w_k1_MSFT_s013
                                 + 1.06216 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.00628 w_k1_ADM_s013
                                 + 1.56644 w_k1_AXP_s013
                                 + 1.18694 w_k1_DIS_s013
                                 + 1.25465 w_k1_FDX_s013 + 0.8954 w_k1_IBM_s013
                                 + 1.40628 w_k1_INTC_s013
                                 + 0.924422 w_k1_KO_s013
                                 + 0.976294 w_k1_MAT_s013
                                 + 1.0364 w_k1_MSFT_s013 + 1.15525 w_k1_PG_s013
                                 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.889069 w_k1_ADM_s013
                                 + 1.09262 w_k1_AXP_s013
                                 + 1.52908 w_k1_DIS_s013
                                 + 1.06971 w_k1_FDX_s013
                                 + 1.39093 w_k1_IBM_s013
                                 + 1.1483 w_k1_INTC_s013 + 1.31625 w_k1_KO_s013
                                 + 1.27734 w_k1_MAT_s013
                                 + 1.18068 w_k1_MSFT_s013
                                 + 1.14497 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.00045 w_k1_ADM_s013
                                 + 1.47272 w_k1_AXP_s013
                                 + 1.73041 w_k1_DIS_s013
                                 + 1.31222 w_k1_FDX_s013
                                 + 1.32169 w_k1_IBM_s013
                                 + 1.32885 w_k1_INTC_s013
                                 + 1.21035 w_k1_KO_s013 + 1.40362 w_k1_MAT_s013
                                 + 1.19022 w_k1_MSFT_s013
                                 + 1.14191 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.769943 w_k1_ADM_s013
                                 + 0.63953 w_k1_AXP_s013
                                 + 0.699257 w_k1_DIS_s013
                                 + 0.797763 w_k1_FDX_s013
                                 + 0.623743 w_k1_IBM_s013
                                 + 0.828614 w_k1_INTC_s013
                                 + 0.637017 w_k1_KO_s013
                                 + 0.435568 w_k1_MAT_s013
                                 + 0.772079 w_k1_MSFT_s013
                                 + 0.781271 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.653982 w_k1_ADM_s013
                                 + 0.743555 w_k1_AXP_s013
                                 + 0.610753 w_k1_DIS_s013
                                 + 0.672214 w_k1_FDX_s013
                                 + 0.914053 w_k1_IBM_s013
                                 + 0.863188 w_k1_INTC_s013
                                 + 0.643409 w_k1_KO_s013
                                 + 0.514578 w_k1_MAT_s013
                                 + 0.630608 w_k1_MSFT_s013
                                 + 0.779656 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.997973 w_k1_ADM_s013
                                 + 1.28202 w_k1_AXP_s013 + 1.0551 w_k1_DIS_s013
                                 + 1.04589 w_k1_FDX_s013
                                 + 1.04244 w_k1_IBM_s013
                                 + 1.17939 w_k1_INTC_s013
                                 + 0.877599 w_k1_KO_s013
                                 + 0.883593 w_k1_MAT_s013
                                 + 0.957013 w_k1_MSFT_s013
                                 + 1.09403 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.91082 w_k1_ADM_s013
                                 + 0.949087 w_k1_AXP_s013
                                 + 0.969036 w_k1_DIS_s013
                                 + 0.887553 w_k1_FDX_s013
                                 + 1.2689 w_k1_IBM_s013
                                 + 0.979544 w_k1_INTC_s013
                                 + 1.09018 w_k1_KO_s013 + 1.16606 w_k1_MAT_s013
                                 + 0.97184 w_k1_MSFT_s013 + 1.0319 w_k1_PG_s013
                                 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.06093 w_k1_ADM_s014
                                 + 1.18975 w_k1_AXP_s014
                                 + 1.45575 w_k1_DIS_s014
                                 + 1.18305 w_k1_FDX_s014 + 1.1411 w_k1_IBM_s014
                                 + 1.30084 w_k1_INTC_s014
                                 + 1.20937 w_k1_KO_s014 + 1.53327 w_k1_MAT_s014
                                 + 1.29426 w_k1_MSFT_s014
                                 + 1.03716 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.834988 w_k1_ADM_s014
                                 + 1.09392 w_k1_AXP_s014
                                 + 0.978965 w_k1_DIS_s014
                                 + 0.814479 w_k1_FDX_s014
                                 + 1.34085 w_k1_IBM_s014
                                 + 1.27109 w_k1_INTC_s014
                                 + 0.96006 w_k1_KO_s014 + 1.16345 w_k1_MAT_s014
                                 + 0.863206 w_k1_MSFT_s014
                                 + 0.906802 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.994974 w_k1_ADM_s014
                                 + 1.15519 w_k1_AXP_s014 + 1.443 w_k1_DIS_s014
                                 + 1.33536 w_k1_FDX_s014
                                 + 0.896712 w_k1_IBM_s014
                                 + 1.25389 w_k1_INTC_s014
                                 + 0.999057 w_k1_KO_s014
                                 + 0.895604 w_k1_MAT_s014
                                 + 1.16909 w_k1_MSFT_s014
                                 + 1.05615 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.15867 w_k1_ADM_s014
                                 + 1.4446 w_k1_AXP_s014 + 1.43762 w_k1_DIS_s014
                                 + 1.28477 w_k1_FDX_s014 + 1.6007 w_k1_IBM_s014
                                 + 1.31057 w_k1_INTC_s014
                                 + 1.24832 w_k1_KO_s014 + 2.01861 w_k1_MAT_s014
                                 + 1.3804 w_k1_MSFT_s014 + 1.21849 w_k1_PG_s014
                                 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.955835 w_k1_ADM_s014
                                 + 0.808197 w_k1_AXP_s014
                                 + 0.946741 w_k1_DIS_s014
                                 + 0.865901 w_k1_FDX_s014
                                 + 1.04026 w_k1_IBM_s014
                                 + 1.07734 w_k1_INTC_s014
                                 + 0.882974 w_k1_KO_s014
                                 + 0.943068 w_k1_MAT_s014
                                 + 0.965499 w_k1_MSFT_s014
                                 + 0.766444 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.830308 w_k1_ADM_s014
                                 + 0.84763 w_k1_AXP_s014
                                 + 0.827765 w_k1_DIS_s014
                                 + 0.940746 w_k1_FDX_s014
                                 + 0.637277 w_k1_IBM_s014
                                 + 0.906955 w_k1_INTC_s014
                                 + 0.62898 w_k1_KO_s014
                                 + 0.832351 w_k1_MAT_s014
                                 + 0.876417 w_k1_MSFT_s014
                                 + 0.780328 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.03029 w_k1_ADM_s014
                                 + 1.25823 w_k1_AXP_s014
                                 + 1.03688 w_k1_DIS_s014
                                 + 1.15191 w_k1_FDX_s014
                                 + 0.812321 w_k1_IBM_s014
                                 + 1.32703 w_k1_INTC_s014
                                 + 0.863433 w_k1_KO_s014
                                 + 1.00572 w_k1_MAT_s014
                                 + 1.00311 w_k1_MSFT_s014
                                 + 0.926072 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.783845 w_k1_ADM_s014
                                 + 0.835217 w_k1_AXP_s014
                                 + 0.75267 w_k1_DIS_s014
                                 + 0.761502 w_k1_FDX_s014
                                 + 0.930636 w_k1_IBM_s014
                                 + 0.893585 w_k1_INTC_s014
                                 + 0.709552 w_k1_KO_s014
                                 + 0.901166 w_k1_MAT_s014
                                 + 0.744533 w_k1_MSFT_s014
                                 + 0.816971 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.41387 w_k1_ADM_s015
                                 + 1.33152 w_k1_AXP_s015
                                 + 0.997989 w_k1_DIS_s015
                                 + 1.07675 w_k1_FDX_s015
                                 + 1.03736 w_k1_IBM_s015
                                 + 1.00094 w_k1_INTC_s015
                                 + 0.946712 w_k1_KO_s015 + 1.368 w_k1_MAT_s015
                                 + 0.895697 w_k1_MSFT_s015
                                 + 1.05414 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.794993 w_k1_ADM_s015
                                 + 0.823529 w_k1_AXP_s015
                                 + 0.906645 w_k1_DIS_s015
                                 + 0.797768 w_k1_FDX_s015
                                 + 1.01333 w_k1_IBM_s015
                                 + 0.746767 w_k1_INTC_s015
                                 + 1.00628 w_k1_KO_s015 + 1.415 w_k1_MAT_s015
                                 + 0.969194 w_k1_MSFT_s015
                                 + 0.974839 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.14814 w_k1_ADM_s015
                                 + 0.574726 w_k1_AXP_s015
                                 + 0.759382 w_k1_DIS_s015
                                 + 0.87804 w_k1_FDX_s015
                                 + 0.621982 w_k1_IBM_s015
                                 + 0.824369 w_k1_INTC_s015
                                 + 0.748228 w_k1_KO_s015
                                 + 0.549753 w_k1_MAT_s015
                                 + 0.89248 w_k1_MSFT_s015
                                 + 0.736839 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.25077 w_k1_ADM_s015
                                 + 1.41215 w_k1_AXP_s015
                                 + 1.47317 w_k1_DIS_s015
                                 + 1.26761 w_k1_FDX_s015
                                 + 1.14365 w_k1_IBM_s015
                                 + 1.12437 w_k1_INTC_s015
                                 + 1.16597 w_k1_KO_s015 + 1.61014 w_k1_MAT_s015
                                 + 1.19067 w_k1_MSFT_s015
                                 + 0.956676 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.13933 w_k1_ADM_s015
                                 + 0.887853 w_k1_AXP_s015
                                 + 0.732667 w_k1_DIS_s015
                                 + 0.770285 w_k1_FDX_s015
                                 + 0.789797 w_k1_IBM_s015
                                 + 0.749368 w_k1_INTC_s015
                                 + 0.874057 w_k1_KO_s015
                                 + 1.21721 w_k1_MAT_s015
                                 + 0.847299 w_k1_MSFT_s015
                                 + 1.00985 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.06788 w_k1_ADM_s015
                                 + 0.806805 w_k1_AXP_s015
                                 + 1.35251 w_k1_DIS_s015
                                 + 0.963173 w_k1_FDX_s015
                                 + 1.32432 w_k1_IBM_s015
                                 + 0.949177 w_k1_INTC_s015
                                 + 1.2866 w_k1_KO_s015 + 0.946367 w_k1_MAT_s015
                                 + 1.13561 w_k1_MSFT_s015
                                 + 1.03715 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.44019 w_k1_ADM_s015
                                 + 1.46734 w_k1_AXP_s015
                                 + 2.21336 w_k1_DIS_s015
                                 + 1.41521 w_k1_FDX_s015
                                 + 1.35225 w_k1_IBM_s015
                                 + 1.02736 w_k1_INTC_s015
                                 + 1.50957 w_k1_KO_s015 + 2.13823 w_k1_MAT_s015
                                 + 1.50784 w_k1_MSFT_s015
                                 + 1.24693 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.813234 w_k1_ADM_s015
                                 + 0.64901 w_k1_AXP_s015
                                 + 0.670183 w_k1_DIS_s015
                                 + 0.674114 w_k1_FDX_s015
                                 + 0.918821 w_k1_IBM_s015
                                 + 0.706783 w_k1_INTC_s015
                                 + 0.743756 w_k1_KO_s015
                                 + 0.676219 w_k1_MAT_s015
                                 + 0.789322 w_k1_MSFT_s015
                                 + 0.714423 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.982236 w_k1_ADM_s016
                                 + 1.03879 w_k1_AXP_s016
                                 + 0.883186 w_k1_DIS_s016
                                 + 0.892855 w_k1_FDX_s016
                                 + 1.1158 w_k1_IBM_s016
                                 + 1.10984 w_k1_INTC_s016
                                 + 1.05648 w_k1_KO_s016 + 1.24786 w_k1_MAT_s016
                                 + 0.807906 w_k1_MSFT_s016
                                 + 0.999161 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.12468 w_k1_ADM_s016
                                 + 0.698757 w_k1_AXP_s016
                                 + 0.884393 w_k1_DIS_s016
                                 + 0.849845 w_k1_FDX_s016
                                 + 0.822011 w_k1_IBM_s016
                                 + 1.07326 w_k1_INTC_s016
                                 + 0.992703 w_k1_KO_s016
                                 + 0.849473 w_k1_MAT_s016
                                 + 0.971207 w_k1_MSFT_s016
                                 + 0.985824 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.968383 w_k1_ADM_s016
                                 + 0.703619 w_k1_AXP_s016
                                 + 0.935605 w_k1_DIS_s016
                                 + 0.987932 w_k1_FDX_s016
                                 + 0.606228 w_k1_IBM_s016
                                 + 1.09088 w_k1_INTC_s016
                                 + 0.930143 w_k1_KO_s016
                                 + 0.608696 w_k1_MAT_s016
                                 + 1.0612 w_k1_MSFT_s016 + 1.07618 w_k1_PG_s016
                                 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.28424 w_k1_ADM_s016
                                 + 1.21779 w_k1_AXP_s016
                                 + 1.15747 w_k1_DIS_s016 + 1.0923 w_k1_FDX_s016
                                 + 1.40884 w_k1_IBM_s016
                                 + 1.02696 w_k1_INTC_s016
                                 + 1.27574 w_k1_KO_s016 + 1.70825 w_k1_MAT_s016
                                 + 1.07165 w_k1_MSFT_s016
                                 + 1.11508 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.915231 w_k1_ADM_s016
                                 + 0.623829 w_k1_AXP_s016
                                 + 0.673124 w_k1_DIS_s016
                                 + 0.840113 w_k1_FDX_s016
                                 + 0.5801 w_k1_IBM_s016
                                 + 1.04328 w_k1_INTC_s016
                                 + 0.745102 w_k1_KO_s016
                                 + 0.565023 w_k1_MAT_s016
                                 + 0.814571 w_k1_MSFT_s016
                                 + 0.767013 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.19749 w_k1_ADM_s016
                                 + 1.18542 w_k1_AXP_s016
                                 + 1.18892 w_k1_DIS_s016 + 1.2004 w_k1_FDX_s016
                                 + 0.784698 w_k1_IBM_s016
                                 + 1.20478 w_k1_INTC_s016
                                 + 1.00176 w_k1_KO_s016
                                 + 0.962286 w_k1_MAT_s016
                                 + 0.902053 w_k1_MSFT_s016
                                 + 0.960231 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.618345 w_k1_ADM_s016
                                 + 0.684912 w_k1_AXP_s016
                                 + 0.712057 w_k1_DIS_s016
                                 + 0.758573 w_k1_FDX_s016
                                 + 0.95287 w_k1_IBM_s016
                                 + 0.897602 w_k1_INTC_s016
                                 + 0.900219 w_k1_KO_s016
                                 + 1.23393 w_k1_MAT_s016
                                 + 0.701963 w_k1_MSFT_s016
                                 + 0.841751 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.69145 w_k1_ADM_s016
                                 + 1.14385 w_k1_AXP_s016
                                 + 1.42743 w_k1_DIS_s016
                                 + 1.40506 w_k1_FDX_s016
                                 + 0.974953 w_k1_IBM_s016
                                 + 1.17912 w_k1_INTC_s016
                                 + 1.27768 w_k1_KO_s016 + 1.67567 w_k1_MAT_s016
                                 + 1.32769 w_k1_MSFT_s016
                                 + 1.19911 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.781483 w_k1_ADM_s017
                                 + 0.893455 w_k1_AXP_s017
                                 + 0.637312 w_k1_DIS_s017
                                 + 0.765919 w_k1_FDX_s017
                                 + 0.820918 w_k1_IBM_s017
                                 + 0.854978 w_k1_INTC_s017
                                 + 0.831266 w_k1_KO_s017
                                 + 0.858282 w_k1_MAT_s017
                                 + 0.756879 w_k1_MSFT_s017
                                 + 0.895894 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.26045 w_k1_ADM_s017
                                 + 1.6702 w_k1_AXP_s017 + 1.41791 w_k1_DIS_s017
                                 + 1.45624 w_k1_FDX_s017
                                 + 0.901717 w_k1_IBM_s017
                                 + 1.37046 w_k1_INTC_s017 + 1.0878 w_k1_KO_s017
                                 + 0.923183 w_k1_MAT_s017
                                 + 1.27104 w_k1_MSFT_s017
                                 + 1.18296 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.0344 w_k1_ADM_s017
                                 + 1.25346 w_k1_AXP_s017
                                 + 0.950865 w_k1_DIS_s017
                                 + 1.11844 w_k1_FDX_s017
                                 + 0.72245 w_k1_IBM_s017
                                 + 1.25958 w_k1_INTC_s017
                                 + 0.791401 w_k1_KO_s017
                                 + 0.559763 w_k1_MAT_s017
                                 + 1.18405 w_k1_MSFT_s017
                                 + 0.974946 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.954961 w_k1_ADM_s017
                                 + 0.978931 w_k1_AXP_s017
                                 + 0.94897 w_k1_DIS_s017
                                 + 0.952634 w_k1_FDX_s017
                                 + 0.831984 w_k1_IBM_s017
                                 + 0.891964 w_k1_INTC_s017
                                 + 0.917294 w_k1_KO_s017
                                 + 0.878058 w_k1_MAT_s017
                                 + 1.02086 w_k1_MSFT_s017
                                 + 1.04042 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.1886 w_k1_ADM_s017
                                 + 1.56601 w_k1_AXP_s017
                                 + 1.81653 w_k1_DIS_s017
                                 + 1.33571 w_k1_FDX_s017
                                 + 1.48332 w_k1_IBM_s017
                                 + 1.35609 w_k1_INTC_s017
                                 + 1.39996 w_k1_KO_s017 + 1.2617 w_k1_MAT_s017
                                 + 1.39242 w_k1_MSFT_s017
                                 + 1.22233 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.917371 w_k1_ADM_s017
                                 + 1.38236 w_k1_AXP_s017
                                 + 1.05626 w_k1_DIS_s017
                                 + 1.09594 w_k1_FDX_s017
                                 + 0.992766 w_k1_IBM_s017
                                 + 1.11646 w_k1_INTC_s017
                                 + 0.922703 w_k1_KO_s017
                                 + 0.987739 w_k1_MAT_s017
                                 + 1.00028 w_k1_MSFT_s017
                                 + 1.05477 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.981363 w_k1_ADM_s017
                                 + 1.13569 w_k1_AXP_s017
                                 + 1.20436 w_k1_DIS_s017
                                 + 1.06983 w_k1_FDX_s017
                                 + 1.28143 w_k1_IBM_s017
                                 + 1.07173 w_k1_INTC_s017
                                 + 1.21576 w_k1_KO_s017 + 1.14991 w_k1_MAT_s017
                                 + 1.15111 w_k1_MSFT_s017
                                 + 1.05963 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.85331 w_k1_ADM_s017
                                 + 1.20516 w_k1_AXP_s017
                                 + 1.22108 w_k1_DIS_s017
                                 + 1.11861 w_k1_FDX_s017
                                 + 0.830948 w_k1_IBM_s017
                                 + 1.3574 w_k1_INTC_s017 + 1.02102 w_k1_KO_s017
                                 + 0.51455 w_k1_MAT_s017
                                 + 1.20371 w_k1_MSFT_s017
                                 + 1.00143 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.01426 w_k1_ADM_s100
                                 + 1.18829 w_k1_AXP_s100
                                 + 1.29542 w_k1_DIS_s100
                                 + 1.06765 w_k1_FDX_s100 + 1.3203 w_k1_IBM_s100
                                 + 1.35992 w_k1_INTC_s100 + 1.2322 w_k1_KO_s100
                                 + 1.30378 w_k1_MAT_s100
                                 + 1.05672 w_k1_MSFT_s100
                                 + 1.18216 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.923697 w_k1_ADM_s100
                                 + 0.909586 w_k1_AXP_s100
                                 + 0.971929 w_k1_DIS_s100
                                 + 0.985678 w_k1_FDX_s100
                                 + 0.872216 w_k1_IBM_s100
                                 + 1.2062 w_k1_INTC_s100
                                 + 0.891654 w_k1_KO_s100
                                 + 1.01041 w_k1_MAT_s100
                                 + 1.05177 w_k1_MSFT_s100 + 1.0061 w_k1_PG_s100
                                 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.26579 w_k1_ADM_s100
                                 + 1.56175 w_k1_AXP_s100
                                 + 1.33825 w_k1_DIS_s100
                                 + 1.38843 w_k1_FDX_s100
                                 + 1.08299 w_k1_IBM_s100
                                 + 1.33638 w_k1_INTC_s100
                                 + 1.16357 w_k1_KO_s100 + 1.43399 w_k1_MAT_s100
                                 + 1.19878 w_k1_MSFT_s100
                                 + 1.27806 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.18957 w_k1_ADM_s100
                                 + 1.3276 w_k1_AXP_s100 + 1.18881 w_k1_DIS_s100
                                 + 1.26027 w_k1_FDX_s100
                                 + 0.731546 w_k1_IBM_s100
                                 + 1.20147 w_k1_INTC_s100
                                 + 1.11241 w_k1_KO_s100
                                 + 0.831631 w_k1_MAT_s100
                                 + 1.1878 w_k1_MSFT_s100 + 1.18202 w_k1_PG_s100
                                 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.871558 w_k1_ADM_s100
                                 + 0.802292 w_k1_AXP_s100
                                 + 0.678078 w_k1_DIS_s100
                                 + 0.78795 w_k1_FDX_s100
                                 + 0.797153 w_k1_IBM_s100
                                 + 0.81403 w_k1_INTC_s100
                                 + 0.744049 w_k1_KO_s100
                                 + 0.777008 w_k1_MAT_s100
                                 + 0.759125 w_k1_MSFT_s100
                                 + 0.880234 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.00111 w_k1_ADM_s100
                                 + 1.25988 w_k1_AXP_s100
                                 + 0.943497 w_k1_DIS_s100
                                 + 1.05315 w_k1_FDX_s100
                                 + 0.689685 w_k1_IBM_s100
                                 + 1.07534 w_k1_INTC_s100
                                 + 0.791728 w_k1_KO_s100
                                 + 0.933899 w_k1_MAT_s100
                                 + 0.924704 w_k1_MSFT_s100
                                 + 1.0023 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.902631 w_k1_ADM_s100
                                 + 1.28271 w_k1_AXP_s100
                                 + 0.869575 w_k1_DIS_s100
                                 + 0.900289 w_k1_FDX_s100
                                 + 1.21675 w_k1_IBM_s100
                                 + 1.09391 w_k1_INTC_s100
                                 + 1.07171 w_k1_KO_s100 + 1.38837 w_k1_MAT_s100
                                 + 0.875454 w_k1_MSFT_s100
                                 + 0.975251 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.28983 w_k1_ADM_s100
                                 + 1.34056 w_k1_AXP_s100
                                 + 1.91654 w_k1_DIS_s100
                                 + 1.35886 w_k1_FDX_s100
                                 + 1.29478 w_k1_IBM_s100
                                 + 1.76343 w_k1_INTC_s100
                                 + 1.63875 w_k1_KO_s100 + 1.5038 w_k1_MAT_s100
                                 + 1.54357 w_k1_MSFT_s100 + 1.3314 w_k1_PG_s100
                                 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.965812 w_k1_ADM_s101
                                 + 0.983604 w_k1_AXP_s101
                                 + 1.13439 w_k1_DIS_s101
                                 + 1.14482 w_k1_FDX_s101
                                 + 0.780457 w_k1_IBM_s101
                                 + 0.940895 w_k1_INTC_s101
                                 + 1.09565 w_k1_KO_s101
                                 + 0.748121 w_k1_MAT_s101
                                 + 1.05687 w_k1_MSFT_s101
                                 + 1.06408 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.96135 w_k1_ADM_s101
                                 + 1.27516 w_k1_AXP_s101
                                 + 1.05917 w_k1_DIS_s101
                                 + 1.02073 w_k1_FDX_s101
                                 + 1.66857 w_k1_IBM_s101
                                 + 1.14876 w_k1_INTC_s101
                                 + 1.35007 w_k1_KO_s101 + 1.80939 w_k1_MAT_s101
                                 + 1.02302 w_k1_MSFT_s101
                                 + 1.17422 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.19198 w_k1_ADM_s101
                                 + 1.43906 w_k1_AXP_s101
                                 + 1.36092 w_k1_DIS_s101
                                 + 1.35209 w_k1_FDX_s101
                                 + 0.906602 w_k1_IBM_s101
                                 + 1.20453 w_k1_INTC_s101
                                 + 1.36743 w_k1_KO_s101 + 1.18845 w_k1_MAT_s101
                                 + 1.22053 w_k1_MSFT_s101
                                 + 1.20795 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.872996 w_k1_ADM_s101
                                 + 1.10153 w_k1_AXP_s101
                                 + 0.909919 w_k1_DIS_s101
                                 + 0.943744 w_k1_FDX_s101
                                 + 1.08042 w_k1_IBM_s101
                                 + 0.922956 w_k1_INTC_s101
                                 + 1.21106 w_k1_KO_s101 + 1.18488 w_k1_MAT_s101
                                 + 0.955385 w_k1_MSFT_s101
                                 + 1.07638 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.920635 w_k1_ADM_s101
                                 + 1.04553 w_k1_AXP_s101
                                 + 0.64171 w_k1_DIS_s101
                                 + 0.882127 w_k1_FDX_s101
                                 + 0.722169 w_k1_IBM_s101
                                 + 0.708791 w_k1_INTC_s101
                                 + 0.82457 w_k1_KO_s101 + 1.01209 w_k1_MAT_s101
                                 + 0.806605 w_k1_MSFT_s101
                                 + 0.914222 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.26163 w_k1_ADM_s101
                                 + 1.62537 w_k1_AXP_s101
                                 + 1.65192 w_k1_DIS_s101
                                 + 1.40868 w_k1_FDX_s101
                                 + 1.33624 w_k1_IBM_s101
                                 + 1.33779 w_k1_INTC_s101
                                 + 1.77068 w_k1_KO_s101 + 1.80223 w_k1_MAT_s101
                                 + 1.36371 w_k1_MSFT_s101
                                 + 1.44473 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.75953 w_k1_ADM_s101
                                 + 0.834015 w_k1_AXP_s101
                                 + 0.829905 w_k1_DIS_s101
                                 + 0.809956 w_k1_FDX_s101
                                 + 0.874515 w_k1_IBM_s101
                                 + 0.966026 w_k1_INTC_s101
                                 + 1.0271 w_k1_KO_s101 + 0.716416 w_k1_MAT_s101
                                 + 0.915658 w_k1_MSFT_s101
                                 + 1.00857 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.02254 w_k1_ADM_s101
                                 + 1.13371 w_k1_AXP_s101
                                 + 1.36491 w_k1_DIS_s101
                                 + 1.16595 w_k1_FDX_s101
                                 + 1.09962 w_k1_IBM_s101
                                 + 1.17115 w_k1_INTC_s101
                                 + 1.30866 w_k1_KO_s101 + 1.24068 w_k1_MAT_s101
                                 + 1.05544 w_k1_MSFT_s101
                                 + 1.06829 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.00783 w_k1_ADM_s102
                                 + 1.00696 w_k1_AXP_s102
                                 + 1.00374 w_k1_DIS_s102
                                 + 0.851052 w_k1_FDX_s102
                                 + 1.27453 w_k1_IBM_s102
                                 + 0.910004 w_k1_INTC_s102
                                 + 1.24934 w_k1_KO_s102 + 1.38054 w_k1_MAT_s102
                                 + 0.925388 w_k1_MSFT_s102
                                 + 1.05243 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.868117 w_k1_ADM_s102
                                 + 0.788745 w_k1_AXP_s102
                                 + 0.71767 w_k1_DIS_s102
                                 + 0.776603 w_k1_FDX_s102
                                 + 0.683314 w_k1_IBM_s102
                                 + 0.667519 w_k1_INTC_s102
                                 + 0.912732 w_k1_KO_s102
                                 + 0.773109 w_k1_MAT_s102
                                 + 0.796749 w_k1_MSFT_s102
                                 + 0.859906 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.04675 w_k1_ADM_s102
                                 + 1.02693 w_k1_AXP_s102
                                 + 1.15344 w_k1_DIS_s102
                                 + 1.09319 w_k1_FDX_s102
                                 + 0.895475 w_k1_IBM_s102
                                 + 0.984684 w_k1_INTC_s102
                                 + 0.9591 w_k1_KO_s102 + 0.841695 w_k1_MAT_s102
                                 + 0.977709 w_k1_MSFT_s102
                                 + 1.02169 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.4226 w_k1_ADM_s102
                                 + 1.3285 w_k1_AXP_s102 + 1.55057 w_k1_DIS_s102
                                 + 1.37168 w_k1_FDX_s102
                                 + 0.806829 w_k1_IBM_s102
                                 + 1.25321 w_k1_INTC_s102
                                 + 1.22534 w_k1_KO_s102
                                 + 0.898853 w_k1_MAT_s102
                                 + 1.36779 w_k1_MSFT_s102
                                 + 1.22714 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.29948 w_k1_ADM_s102
                                 + 1.47017 w_k1_AXP_s102
                                 + 1.90184 w_k1_DIS_s102
                                 + 1.30539 w_k1_FDX_s102
                                 + 1.43402 w_k1_IBM_s102
                                 + 0.965028 w_k1_INTC_s102
                                 + 1.49488 w_k1_KO_s102 + 2.19542 w_k1_MAT_s102
                                 + 1.12027 w_k1_MSFT_s102
                                 + 1.09397 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.00347 w_k1_ADM_s102
                                 + 1.07306 w_k1_AXP_s102
                                 + 0.683944 w_k1_DIS_s102
                                 + 0.863206 w_k1_FDX_s102
                                 + 0.909311 w_k1_IBM_s102
                                 + 0.855014 w_k1_INTC_s102
                                 + 0.913871 w_k1_KO_s102
                                 + 1.25631 w_k1_MAT_s102
                                 + 0.881221 w_k1_MSFT_s102
                                 + 0.941184 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.16062 w_k1_ADM_s102
                                 + 1.5138 w_k1_AXP_s102
                                 + 0.950298 w_k1_DIS_s102
                                 + 1.05674 w_k1_FDX_s102
                                 + 1.03425 w_k1_IBM_s102
                                 + 0.896949 w_k1_INTC_s102
                                 + 0.942609 w_k1_KO_s102
                                 + 2.21416 w_k1_MAT_s102
                                 + 0.900681 w_k1_MSFT_s102
                                 + 1.02929 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.02567 w_k1_ADM_s102
                                 + 0.905573 w_k1_AXP_s102
                                 + 0.930234 w_k1_DIS_s102
                                 + 1.05081 w_k1_FDX_s102
                                 + 0.510688 w_k1_IBM_s102
                                 + 1.02481 w_k1_INTC_s102
                                 + 0.845086 w_k1_KO_s102
                                 + 0.624085 w_k1_MAT_s102
                                 + 1.0455 w_k1_MSFT_s102
                                 + 0.924588 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.15059 w_k1_ADM_s103
                                 + 1.65078 w_k1_AXP_s103
                                 + 1.77716 w_k1_DIS_s103
                                 + 1.34743 w_k1_FDX_s103
                                 + 1.72548 w_k1_IBM_s103
                                 + 1.44267 w_k1_INTC_s103
                                 + 1.67132 w_k1_KO_s103 + 2.05602 w_k1_MAT_s103
                                 + 1.52734 w_k1_MSFT_s103
                                 + 1.23265 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.89994 w_k1_ADM_s103
                                 + 0.74264 w_k1_AXP_s103
                                 + 0.736836 w_k1_DIS_s103
                                 + 0.781934 w_k1_FDX_s103
                                 + 0.911609 w_k1_IBM_s103
                                 + 0.952417 w_k1_INTC_s103
                                 + 0.984077 w_k1_KO_s103
                                 + 1.04539 w_k1_MAT_s103
                                 + 0.875738 w_k1_MSFT_s103
                                 + 0.899044 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.10138 w_k1_ADM_s103
                                 + 1.0193 w_k1_AXP_s103
                                 + 0.937755 w_k1_DIS_s103
                                 + 1.03045 w_k1_FDX_s103
                                 + 0.859421 w_k1_IBM_s103
                                 + 1.08076 w_k1_INTC_s103
                                 + 0.946181 w_k1_KO_s103
                                 + 0.653781 w_k1_MAT_s103
                                 + 1.09956 w_k1_MSFT_s103
                                 + 0.97997 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.85893 w_k1_ADM_s103
                                 + 0.693777 w_k1_AXP_s103
                                 + 0.639308 w_k1_DIS_s103
                                 + 0.829997 w_k1_FDX_s103
                                 + 0.677412 w_k1_IBM_s103
                                 + 0.840725 w_k1_INTC_s103
                                 + 0.823122 w_k1_KO_s103
                                 + 0.681927 w_k1_MAT_s103
                                 + 0.869478 w_k1_MSFT_s103
                                 + 0.862847 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.06516 w_k1_ADM_s103
                                 + 0.92487 w_k1_AXP_s103
                                 + 0.803082 w_k1_DIS_s103
                                 + 0.835695 w_k1_FDX_s103
                                 + 1.24196 w_k1_IBM_s103
                                 + 1.02366 w_k1_INTC_s103
                                 + 1.19445 w_k1_KO_s103 + 1.00735 w_k1_MAT_s103
                                 + 0.89674 w_k1_MSFT_s103
                                 + 0.95515 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.12713 w_k1_ADM_s103
                                 + 1.16193 w_k1_AXP_s103
                                 + 1.47349 w_k1_DIS_s103 + 1.2959 w_k1_FDX_s103
                                 + 0.853515 w_k1_IBM_s103
                                 + 1.28334 w_k1_INTC_s103
                                 + 1.25416 w_k1_KO_s103 + 1.03601 w_k1_MAT_s103
                                 + 1.41761 w_k1_MSFT_s103 + 1.1344 w_k1_PG_s103
                                 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.02125 w_k1_ADM_s103
                                 + 0.69397 w_k1_AXP_s103
                                 + 0.897589 w_k1_DIS_s103
                                 + 1.01803 w_k1_FDX_s103
                                 + 0.605471 w_k1_IBM_s103
                                 + 1.09933 w_k1_INTC_s103
                                 + 0.944443 w_k1_KO_s103
                                 + 0.478756 w_k1_MAT_s103
                                 + 1.12198 w_k1_MSFT_s103
                                 + 0.946376 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.0826 w_k1_ADM_s103
                                 + 1.24636 w_k1_AXP_s103
                                 + 1.19354 w_k1_DIS_s103
                                 + 1.03418 w_k1_FDX_s103
                                 + 1.43465 w_k1_IBM_s103
                                 + 1.16964 w_k1_INTC_s103
                                 + 1.35443 w_k1_KO_s103 + 1.2173 w_k1_MAT_s103
                                 + 1.22039 w_k1_MSFT_s103 + 1.0515 w_k1_PG_s103
                                 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.18589 w_k1_ADM_s104
                                 + 0.92164 w_k1_AXP_s104
                                 + 1.20033 w_k1_DIS_s104 + 1.1102 w_k1_FDX_s104
                                 + 0.87333 w_k1_IBM_s104
                                 + 1.14513 w_k1_INTC_s104
                                 + 1.02395 w_k1_KO_s104
                                 + 0.821071 w_k1_MAT_s104
                                 + 1.2529 w_k1_MSFT_s104 + 1.04946 w_k1_PG_s104
                                 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.96926 w_k1_ADM_s104
                                 + 0.699899 w_k1_AXP_s104
                                 + 0.732341 w_k1_DIS_s104
                                 + 0.807762 w_k1_FDX_s104
                                 + 0.569177 w_k1_IBM_s104
                                 + 0.791929 w_k1_INTC_s104
                                 + 0.745549 w_k1_KO_s104
                                 + 0.77184 w_k1_MAT_s104
                                 + 0.922296 w_k1_MSFT_s104
                                 + 0.83084 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.53205 w_k1_ADM_s104
                                 + 1.55766 w_k1_AXP_s104
                                 + 1.72682 w_k1_DIS_s104
                                 + 1.33139 w_k1_FDX_s104
                                 + 1.21238 w_k1_IBM_s104
                                 + 1.2339 w_k1_INTC_s104 + 1.43108 w_k1_KO_s104
                                 + 1.72061 w_k1_MAT_s104
                                 + 1.37899 w_k1_MSFT_s104
                                 + 1.19187 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.938255 w_k1_ADM_s104
                                 + 0.963332 w_k1_AXP_s104
                                 + 0.856255 w_k1_DIS_s104
                                 + 0.858479 w_k1_FDX_s104
                                 + 0.868685 w_k1_IBM_s104
                                 + 0.87711 w_k1_INTC_s104
                                 + 0.812794 w_k1_KO_s104 + 1.2532 w_k1_MAT_s104
                                 + 1.03832 w_k1_MSFT_s104
                                 + 0.926884 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.0552 w_k1_ADM_s104
                                 + 1.0359 w_k1_AXP_s104 + 1.11198 w_k1_DIS_s104
                                 + 0.981376 w_k1_FDX_s104
                                 + 1.33434 w_k1_IBM_s104
                                 + 1.23563 w_k1_INTC_s104
                                 + 1.18244 w_k1_KO_s104 + 1.13053 w_k1_MAT_s104
                                 + 1.08661 w_k1_MSFT_s104
                                 + 1.07476 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.903805 w_k1_ADM_s104
                                 + 0.723512 w_k1_AXP_s104
                                 + 0.740111 w_k1_DIS_s104
                                 + 0.725915 w_k1_FDX_s104
                                 + 0.914611 w_k1_IBM_s104
                                 + 1.0663 w_k1_INTC_s104
                                 + 0.883917 w_k1_KO_s104
                                 + 0.885637 w_k1_MAT_s104
                                 + 0.849087 w_k1_MSFT_s104
                                 + 0.89186 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.20217 w_k1_ADM_s104
                                 + 0.936659 w_k1_AXP_s104
                                 + 1.2114 w_k1_DIS_s104 + 1.12475 w_k1_FDX_s104
                                 + 0.595693 w_k1_IBM_s104
                                 + 0.995555 w_k1_INTC_s104
                                 + 0.93547 w_k1_KO_s104
                                 + 0.821653 w_k1_MAT_s104
                                 + 1.20715 w_k1_MSFT_s104 + 0.9733 w_k1_PG_s104
                                 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.938951 w_k1_ADM_s104
                                 + 0.881688 w_k1_AXP_s104
                                 + 0.792083 w_k1_DIS_s104
                                 + 0.779057 w_k1_FDX_s104
                                 + 1.64307 w_k1_IBM_s104
                                 + 1.13917 w_k1_INTC_s104
                                 + 1.14303 w_k1_KO_s104 + 1.31498 w_k1_MAT_s104
                                 + 0.851783 w_k1_MSFT_s104
                                 + 1.0517 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.03921 w_k1_ADM_s105
                                 + 1.40464 w_k1_AXP_s105
                                 + 1.10963 w_k1_DIS_s105
                                 + 1.14238 w_k1_FDX_s105
                                 + 1.06748 w_k1_IBM_s105
                                 + 0.984285 w_k1_INTC_s105
                                 + 0.996371 w_k1_KO_s105
                                 + 0.830881 w_k1_MAT_s105
                                 + 1.16543 w_k1_MSFT_s105
                                 + 1.01169 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.04719 w_k1_ADM_s105
                                 + 1.0933 w_k1_AXP_s105
                                 + 0.814246 w_k1_DIS_s105
                                 + 0.889597 w_k1_FDX_s105
                                 + 0.895084 w_k1_IBM_s105
                                 + 0.764115 w_k1_INTC_s105
                                 + 0.865651 w_k1_KO_s105
                                 + 0.895868 w_k1_MAT_s105
                                 + 0.861504 w_k1_MSFT_s105
                                 + 0.91822 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.824861 w_k1_ADM_s105
                                 + 0.80983 w_k1_AXP_s105
                                 + 0.73838 w_k1_DIS_s105
                                 + 0.804266 w_k1_FDX_s105
                                 + 0.700921 w_k1_IBM_s105
                                 + 0.747078 w_k1_INTC_s105
                                 + 0.718275 w_k1_KO_s105
                                 + 0.519214 w_k1_MAT_s105
                                 + 0.867152 w_k1_MSFT_s105
                                 + 0.836288 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.767649 w_k1_ADM_s105
                                 + 0.752527 w_k1_AXP_s105
                                 + 0.765378 w_k1_DIS_s105
                                 + 0.743496 w_k1_FDX_s105
                                 + 1.08394 w_k1_IBM_s105
                                 + 0.884592 w_k1_INTC_s105
                                 + 0.84653 w_k1_KO_s105
                                 + 0.736421 w_k1_MAT_s105
                                 + 0.793726 w_k1_MSFT_s105
                                 + 0.845702 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.963158 w_k1_ADM_s105
                                 + 1.19729 w_k1_AXP_s105
                                 + 0.893163 w_k1_DIS_s105
                                 + 0.911679 w_k1_FDX_s105
                                 + 1.31617 w_k1_IBM_s105
                                 + 0.911063 w_k1_INTC_s105
                                 + 1.05943 w_k1_KO_s105 + 1.03096 w_k1_MAT_s105
                                 + 0.950464 w_k1_MSFT_s105
                                 + 0.953521 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.08388 w_k1_ADM_s105
                                 + 0.882852 w_k1_AXP_s105
                                 + 1.21902 w_k1_DIS_s105
                                 + 1.19096 w_k1_FDX_s105
                                 + 0.854501 w_k1_IBM_s105
                                 + 1.20783 w_k1_INTC_s105
                                 + 0.894058 w_k1_KO_s105
                                 + 0.440121 w_k1_MAT_s105
                                 + 1.16291 w_k1_MSFT_s105
                                 + 1.07091 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.30015 w_k1_ADM_s105
                                 + 1.3719 w_k1_AXP_s105 + 1.52628 w_k1_DIS_s105
                                 + 1.10266 w_k1_FDX_s105
                                 + 1.46089 w_k1_IBM_s105
                                 + 1.02064 w_k1_INTC_s105
                                 + 1.48153 w_k1_KO_s105
                                 + 0.995534 w_k1_MAT_s105
                                 + 1.12095 w_k1_MSFT_s105
                                 + 1.21994 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.921762 w_k1_ADM_s105
                                 + 0.861877 w_k1_AXP_s105
                                 + 0.890099 w_k1_DIS_s105
                                 + 0.899015 w_k1_FDX_s105
                                 + 1.09653 w_k1_IBM_s105
                                 + 0.899852 w_k1_INTC_s105
                                 + 0.939352 w_k1_KO_s105
                                 + 0.832447 w_k1_MAT_s105
                                 + 0.912809 w_k1_MSFT_s105
                                 + 0.935239 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.07737 w_k1_ADM_s106
                                 + 1.24073 w_k1_AXP_s106
                                 + 1.23813 w_k1_DIS_s106 + 1.2511 w_k1_FDX_s106
                                 + 0.979419 w_k1_IBM_s106
                                 + 1.20689 w_k1_INTC_s106
                                 + 1.04394 w_k1_KO_s106 + 1.06513 w_k1_MAT_s106
                                 + 1.24422 w_k1_MSFT_s106
                                 + 1.12191 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.882599 w_k1_ADM_s106
                                 + 0.824952 w_k1_AXP_s106
                                 + 0.785088 w_k1_DIS_s106
                                 + 0.801313 w_k1_FDX_s106
                                 + 1.01218 w_k1_IBM_s106
                                 + 0.8893 w_k1_INTC_s106 + 0.88465 w_k1_KO_s106
                                 + 0.886431 w_k1_MAT_s106
                                 + 0.933838 w_k1_MSFT_s106
                                 + 0.893311 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.11775 w_k1_ADM_s106
                                 + 0.79735 w_k1_AXP_s106
                                 + 1.10992 w_k1_DIS_s106
                                 + 1.14116 w_k1_FDX_s106
                                 + 0.721858 w_k1_IBM_s106
                                 + 1.14724 w_k1_INTC_s106
                                 + 0.773753 w_k1_KO_s106
                                 + 0.557656 w_k1_MAT_s106
                                 + 1.16559 w_k1_MSFT_s106
                                 + 1.00457 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.867722 w_k1_ADM_s106
                                 + 0.92022 w_k1_AXP_s106
                                 + 0.77106 w_k1_DIS_s106
                                 + 0.962886 w_k1_FDX_s106
                                 + 0.791894 w_k1_IBM_s106
                                 + 1.02062 w_k1_INTC_s106
                                 + 0.770498 w_k1_KO_s106
                                 + 0.803757 w_k1_MAT_s106
                                 + 0.960256 w_k1_MSFT_s106
                                 + 0.934628 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.01242 w_k1_ADM_s106
                                 + 0.986032 w_k1_AXP_s106
                                 + 0.972416 w_k1_DIS_s106
                                 + 1.00544 w_k1_FDX_s106
                                 + 1.14053 w_k1_IBM_s106
                                 + 1.06754 w_k1_INTC_s106
                                 + 1.00813 w_k1_KO_s106 + 1.3011 w_k1_MAT_s106
                                 + 0.991714 w_k1_MSFT_s106
                                 + 1.00288 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.732203 w_k1_ADM_s106
                                 + 0.679692 w_k1_AXP_s106
                                 + 0.640849 w_k1_DIS_s106
                                 + 0.783468 w_k1_FDX_s106
                                 + 0.614674 w_k1_IBM_s106
                                 + 0.835818 w_k1_INTC_s106
                                 + 0.641065 w_k1_KO_s106
                                 + 0.772279 w_k1_MAT_s106
                                 + 0.827314 w_k1_MSFT_s106
                                 + 0.810984 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.771845 w_k1_ADM_s106
                                 + 0.796917 w_k1_AXP_s106
                                 + 0.844867 w_k1_DIS_s106
                                 + 0.791059 w_k1_FDX_s106
                                 + 1.47117 w_k1_IBM_s106
                                 + 0.906501 w_k1_INTC_s106
                                 + 1.10302 w_k1_KO_s106 + 1.36284 w_k1_MAT_s106
                                 + 0.880675 w_k1_MSFT_s106
                                 + 1.03012 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.04129 w_k1_ADM_s106
                                 + 1.26222 w_k1_AXP_s106
                                 + 1.44866 w_k1_DIS_s106
                                 + 1.17024 w_k1_FDX_s106
                                 + 1.66578 w_k1_IBM_s106
                                 + 1.19116 w_k1_INTC_s106
                                 + 1.28539 w_k1_KO_s106 + 2.11115 w_k1_MAT_s106
                                 + 1.213 w_k1_MSFT_s106 + 1.12104 w_k1_PG_s106
                                 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.865944 w_k1_ADM_s107
                                 + 1.07655 w_k1_AXP_s107
                                 + 0.885125 w_k1_DIS_s107
                                 + 0.843676 w_k1_FDX_s107
                                 + 1.15079 w_k1_IBM_s107
                                 + 0.748965 w_k1_INTC_s107
                                 + 0.869506 w_k1_KO_s107
                                 + 1.16959 w_k1_MAT_s107
                                 + 0.810005 w_k1_MSFT_s107
                                 + 0.915789 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.17927 w_k1_ADM_s107
                                 + 1.11849 w_k1_AXP_s107
                                 + 0.809885 w_k1_DIS_s107
                                 + 1.04056 w_k1_FDX_s107
                                 + 0.679863 w_k1_IBM_s107
                                 + 0.882979 w_k1_INTC_s107
                                 + 0.678367 w_k1_KO_s107
                                 + 0.762947 w_k1_MAT_s107
                                 + 0.915261 w_k1_MSFT_s107
                                 + 0.917721 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.28767 w_k1_ADM_s107
                                 + 1.75551 w_k1_AXP_s107
                                 + 1.67286 w_k1_DIS_s107
                                 + 1.50178 w_k1_FDX_s107
                                 + 1.16956 w_k1_IBM_s107
                                 + 1.22913 w_k1_INTC_s107
                                 + 1.13192 w_k1_KO_s107
                                 + 0.809899 w_k1_MAT_s107
                                 + 1.16813 w_k1_MSFT_s107
                                 + 1.12277 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.989001 w_k1_ADM_s107
                                 + 1.19939 w_k1_AXP_s107
                                 + 1.17608 w_k1_DIS_s107
                                 + 1.04269 w_k1_FDX_s107
                                 + 1.06068 w_k1_IBM_s107
                                 + 1.04335 w_k1_INTC_s107
                                 + 0.895506 w_k1_KO_s107
                                 + 0.849937 w_k1_MAT_s107
                                 + 0.981032 w_k1_MSFT_s107
                                 + 1.05788 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.31894 w_k1_ADM_s107
                                 + 1.30718 w_k1_AXP_s107
                                 + 1.35347 w_k1_DIS_s107
                                 + 1.27776 w_k1_FDX_s107
                                 + 1.07868 w_k1_IBM_s107
                                 + 1.26016 w_k1_INTC_s107
                                 + 1.07108 w_k1_KO_s107
                                 + 0.794907 w_k1_MAT_s107
                                 + 1.1559 w_k1_MSFT_s107 + 1.09433 w_k1_PG_s107
                                 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.30606 w_k1_ADM_s107
                                 + 1.39618 w_k1_AXP_s107
                                 + 1.27613 w_k1_DIS_s107 + 1.1532 w_k1_FDX_s107
                                 + 1.52601 w_k1_IBM_s107
                                 + 1.0412 w_k1_INTC_s107 + 1.2074 w_k1_KO_s107
                                 + 1.42443 w_k1_MAT_s107
                                 + 1.09971 w_k1_MSFT_s107 + 1.1498 w_k1_PG_s107
                                 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.822939 w_k1_ADM_s107
                                 + 0.528911 w_k1_AXP_s107
                                 + 0.605854 w_k1_DIS_s107
                                 + 0.698578 w_k1_FDX_s107
                                 + 0.6116 w_k1_IBM_s107
                                 + 0.797075 w_k1_INTC_s107
                                 + 0.58295 w_k1_KO_s107
                                 + 0.527914 w_k1_MAT_s107
                                 + 0.74628 w_k1_MSFT_s107
                                 + 0.767608 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.733678 w_k1_ADM_s107
                                 + 0.644775 w_k1_AXP_s107
                                 + 0.844384 w_k1_DIS_s107
                                 + 0.819198 w_k1_FDX_s107
                                 + 0.915863 w_k1_IBM_s107
                                 + 0.861033 w_k1_INTC_s107
                                 + 0.908543 w_k1_KO_s107
                                 + 0.69894 w_k1_MAT_s107
                                 + 0.912193 w_k1_MSFT_s107
                                 + 0.877933 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.04713 w_k1_ADM_s110
                                 + 0.824332 w_k1_AXP_s110
                                 + 0.843421 w_k1_DIS_s110
                                 + 0.956058 w_k1_FDX_s110
                                 + 0.638176 w_k1_IBM_s110
                                 + 0.886783 w_k1_INTC_s110
                                 + 0.813232 w_k1_KO_s110
                                 + 0.542866 w_k1_MAT_s110
                                 + 0.939223 w_k1_MSFT_s110
                                 + 0.915718 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.19595 w_k1_ADM_s110
                                 + 0.863742 w_k1_AXP_s110
                                 + 1.26572 w_k1_DIS_s110
                                 + 1.14345 w_k1_FDX_s110
                                 + 0.855983 w_k1_IBM_s110
                                 + 1.05213 w_k1_INTC_s110
                                 + 1.08697 w_k1_KO_s110 + 0.79035 w_k1_MAT_s110
                                 + 1.2165 w_k1_MSFT_s110 + 1.15138 w_k1_PG_s110
                                 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.2062 w_k1_ADM_s110
                                 + 1.24487 w_k1_AXP_s110
                                 + 0.976424 w_k1_DIS_s110
                                 + 1.14229 w_k1_FDX_s110
                                 + 0.774637 w_k1_IBM_s110
                                 + 1.0246 w_k1_INTC_s110
                                 + 0.890297 w_k1_KO_s110
                                 + 0.83562 w_k1_MAT_s110
                                 + 0.971043 w_k1_MSFT_s110
                                 + 0.981418 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.985092 w_k1_ADM_s110
                                 + 0.699241 w_k1_AXP_s110
                                 + 0.583255 w_k1_DIS_s110
                                 + 0.767336 w_k1_FDX_s110
                                 + 0.884708 w_k1_IBM_s110
                                 + 0.786532 w_k1_INTC_s110
                                 + 0.721521 w_k1_KO_s110
                                 + 0.715998 w_k1_MAT_s110
                                 + 0.755259 w_k1_MSFT_s110
                                 + 0.788285 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.10224 w_k1_ADM_s110
                                 + 1.34323 w_k1_AXP_s110 + 1.3018 w_k1_DIS_s110
                                 + 1.28613 w_k1_FDX_s110
                                 + 1.39153 w_k1_IBM_s110
                                 + 1.10854 w_k1_INTC_s110
                                 + 1.22432 w_k1_KO_s110 + 1.61542 w_k1_MAT_s110
                                 + 1.19709 w_k1_MSFT_s110
                                 + 1.23125 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.13197 w_k1_ADM_s110
                                 + 0.876752 w_k1_AXP_s110
                                 + 0.837052 w_k1_DIS_s110
                                 + 0.926413 w_k1_FDX_s110
                                 + 1.01684 w_k1_IBM_s110
                                 + 0.935723 w_k1_INTC_s110
                                 + 1.01779 w_k1_KO_s110 + 1.00074 w_k1_MAT_s110
                                 + 0.96654 w_k1_MSFT_s110
                                 + 0.923907 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.18686 w_k1_ADM_s110
                                 + 1.37007 w_k1_AXP_s110
                                 + 1.35243 w_k1_DIS_s110
                                 + 1.30625 w_k1_FDX_s110
                                 + 0.94074 w_k1_IBM_s110
                                 + 1.15079 w_k1_INTC_s110
                                 + 1.08347 w_k1_KO_s110 + 1.27995 w_k1_MAT_s110
                                 + 1.22244 w_k1_MSFT_s110
                                 + 1.11968 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.00341 w_k1_ADM_s110
                                 + 0.995545 w_k1_AXP_s110
                                 + 0.844169 w_k1_DIS_s110
                                 + 0.897536 w_k1_FDX_s110
                                 + 1.28278 w_k1_IBM_s110
                                 + 0.917971 w_k1_INTC_s110
                                 + 1.05852 w_k1_KO_s110 + 1.38954 w_k1_MAT_s110
                                 + 0.959301 w_k1_MSFT_s110
                                 + 0.941173 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.900251 w_k1_ADM_s111
                                 + 0.877809 w_k1_AXP_s111
                                 + 0.85893 w_k1_DIS_s111
                                 + 1.01619 w_k1_FDX_s111
                                 + 1.03995 w_k1_IBM_s111
                                 + 0.942435 w_k1_INTC_s111
                                 + 1.00456 w_k1_KO_s111
                                 + 0.760967 w_k1_MAT_s111
                                 + 0.961823 w_k1_MSFT_s111
                                 + 0.989484 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.698575 w_k1_ADM_s111
                                 + 0.939727 w_k1_AXP_s111
                                 + 0.634713 w_k1_DIS_s111
                                 + 0.936618 w_k1_FDX_s111
                                 + 0.814345 w_k1_IBM_s111
                                 + 0.965623 w_k1_INTC_s111
                                 + 0.750543 w_k1_KO_s111
                                 + 0.816947 w_k1_MAT_s111
                                 + 0.787222 w_k1_MSFT_s111
                                 + 0.822284 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.64362 w_k1_ADM_s111
                                 + 0.558499 w_k1_AXP_s111
                                 + 0.547242 w_k1_DIS_s111
                                 + 0.738508 w_k1_FDX_s111
                                 + 0.737635 w_k1_IBM_s111
                                 + 0.856194 w_k1_INTC_s111
                                 + 0.718351 w_k1_KO_s111
                                 + 0.50075 w_k1_MAT_s111
                                 + 0.780801 w_k1_MSFT_s111
                                 + 0.743319 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.980479 w_k1_ADM_s111
                                 + 0.822004 w_k1_AXP_s111
                                 + 0.916332 w_k1_DIS_s111
                                 + 1.0531 w_k1_FDX_s111
                                 + 0.707645 w_k1_IBM_s111
                                 + 1.00517 w_k1_INTC_s111
                                 + 0.820384 w_k1_KO_s111
                                 + 0.59888 w_k1_MAT_s111
                                 + 0.984024 w_k1_MSFT_s111
                                 + 0.926541 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.19069 w_k1_ADM_s111
                                 + 1.30073 w_k1_AXP_s111
                                 + 1.46989 w_k1_DIS_s111
                                 + 1.39607 w_k1_FDX_s111
                                 + 1.37486 w_k1_IBM_s111
                                 + 1.1584 w_k1_INTC_s111 + 1.36514 w_k1_KO_s111
                                 + 1.22375 w_k1_MAT_s111
                                 + 1.30805 w_k1_MSFT_s111
                                 + 1.18931 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.06075 w_k1_ADM_s111
                                 + 1.13681 w_k1_AXP_s111
                                 + 1.01955 w_k1_DIS_s111
                                 + 1.03259 w_k1_FDX_s111
                                 + 1.71996 w_k1_IBM_s111
                                 + 0.936823 w_k1_INTC_s111
                                 + 1.21931 w_k1_KO_s111 + 1.7278 w_k1_MAT_s111
                                 + 1.02922 w_k1_MSFT_s111
                                 + 1.22123 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.810334 w_k1_ADM_s111
                                 + 0.914363 w_k1_AXP_s111
                                 + 0.845635 w_k1_DIS_s111
                                 + 0.853826 w_k1_FDX_s111
                                 + 1.09813 w_k1_IBM_s111
                                 + 1.03447 w_k1_INTC_s111
                                 + 0.96743 w_k1_KO_s111
                                 + 0.913188 w_k1_MAT_s111
                                 + 0.926536 w_k1_MSFT_s111
                                 + 0.851926 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.17901 w_k1_ADM_s111
                                 + 1.06882 w_k1_AXP_s111
                                 + 1.14741 w_k1_DIS_s111
                                 + 1.14848 w_k1_FDX_s111
                                 + 1.02975 w_k1_IBM_s111
                                 + 1.0487 w_k1_INTC_s111 + 1.12446 w_k1_KO_s111
                                 + 0.984797 w_k1_MAT_s111
                                 + 1.05621 w_k1_MSFT_s111
                                 + 1.12705 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.33887 w_k1_ADM_s112
                                 + 1.35937 w_k1_AXP_s112
                                 + 1.28527 w_k1_DIS_s112
                                 + 1.07427 w_k1_FDX_s112
                                 + 1.59814 w_k1_IBM_s112
                                 + 0.948472 w_k1_INTC_s112
                                 + 1.23024 w_k1_KO_s112 + 1.96914 w_k1_MAT_s112
                                 + 0.885393 w_k1_MSFT_s112
                                 + 1.06415 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.68463 w_k1_ADM_s112
                                 + 1.49015 w_k1_AXP_s112
                                 + 1.76695 w_k1_DIS_s112
                                 + 1.46454 w_k1_FDX_s112
                                 + 1.26071 w_k1_IBM_s112
                                 + 1.39749 w_k1_INTC_s112
                                 + 1.28816 w_k1_KO_s112
                                 + 0.914007 w_k1_MAT_s112
                                 + 1.2836 w_k1_MSFT_s112 + 1.18773 w_k1_PG_s112
                                 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.33285 w_k1_ADM_s112
                                 + 1.07368 w_k1_AXP_s112
                                 + 1.17537 w_k1_DIS_s112
                                 + 1.07747 w_k1_FDX_s112
                                 + 1.11431 w_k1_IBM_s112
                                 + 0.916816 w_k1_INTC_s112
                                 + 1.00549 w_k1_KO_s112
                                 + 0.911099 w_k1_MAT_s112
                                 + 0.936314 w_k1_MSFT_s112
                                 + 1.13208 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.54693 w_k1_ADM_s112
                                 + 1.46218 w_k1_AXP_s112
                                 + 1.17129 w_k1_DIS_s112
                                 + 1.18316 w_k1_FDX_s112
                                 + 0.873274 w_k1_IBM_s112
                                 + 1.00805 w_k1_INTC_s112
                                 + 1.05193 w_k1_KO_s112 + 1.55358 w_k1_MAT_s112
                                 + 0.999968 w_k1_MSFT_s112
                                 + 1.17654 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.21762 w_k1_ADM_s112
                                 + 1.20573 w_k1_AXP_s112
                                 + 1.44479 w_k1_DIS_s112
                                 + 1.19031 w_k1_FDX_s112
                                 + 0.905363 w_k1_IBM_s112
                                 + 1.12337 w_k1_INTC_s112
                                 + 1.17847 w_k1_KO_s112 + 0.92235 w_k1_MAT_s112
                                 + 1.18134 w_k1_MSFT_s112
                                 + 1.13213 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.12011 w_k1_ADM_s112
                                 + 0.845718 w_k1_AXP_s112
                                 + 0.900218 w_k1_DIS_s112
                                 + 0.88632 w_k1_FDX_s112
                                 + 0.885448 w_k1_IBM_s112
                                 + 0.851739 w_k1_INTC_s112
                                 + 0.878456 w_k1_KO_s112
                                 + 1.07808 w_k1_MAT_s112
                                 + 0.815848 w_k1_MSFT_s112
                                 + 0.93613 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.063 w_k1_ADM_s112
                                 + 0.775631 w_k1_AXP_s112
                                 + 0.789628 w_k1_DIS_s112
                                 + 0.921525 w_k1_FDX_s112
                                 + 0.68801 w_k1_IBM_s112
                                 + 0.769937 w_k1_INTC_s112
                                 + 0.673761 w_k1_KO_s112
                                 + 0.56478 w_k1_MAT_s112
                                 + 0.838183 w_k1_MSFT_s112
                                 + 0.914526 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.12839 w_k1_ADM_s112
                                 + 0.972847 w_k1_AXP_s112
                                 + 0.888918 w_k1_DIS_s112
                                 + 0.851338 w_k1_FDX_s112
                                 + 1.21942 w_k1_IBM_s112
                                 + 0.802972 w_k1_INTC_s112
                                 + 0.902833 w_k1_KO_s112
                                 + 1.25263 w_k1_MAT_s112
                                 + 0.788343 w_k1_MSFT_s112
                                 + 0.930642 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.934364 w_k1_ADM_s113
                                 + 0.746397 w_k1_AXP_s113
                                 + 0.990282 w_k1_DIS_s113
                                 + 0.979276 w_k1_FDX_s113
                                 + 0.783534 w_k1_IBM_s113
                                 + 0.828319 w_k1_INTC_s113
                                 + 1.00586 w_k1_KO_s113
                                 + 0.679011 w_k1_MAT_s113
                                 + 0.942285 w_k1_MSFT_s113
                                 + 0.918908 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.73197 w_k1_ADM_s113
                                 + 1.56848 w_k1_AXP_s113
                                 + 1.52596 w_k1_DIS_s113
                                 + 1.38791 w_k1_FDX_s113
                                 + 1.20078 w_k1_IBM_s113
                                 + 1.32205 w_k1_INTC_s113 + 1.3712 w_k1_KO_s113
                                 + 1.08367 w_k1_MAT_s113
                                 + 1.34203 w_k1_MSFT_s113
                                 + 1.09141 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.962767 w_k1_ADM_s113
                                 + 0.955322 w_k1_AXP_s113
                                 + 1.19675 w_k1_DIS_s113
                                 + 1.01184 w_k1_FDX_s113
                                 + 1.38334 w_k1_IBM_s113
                                 + 1.05378 w_k1_INTC_s113
                                 + 1.33048 w_k1_KO_s113 + 1.15392 w_k1_MAT_s113
                                 + 1.12426 w_k1_MSFT_s113
                                 + 0.985083 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.949728 w_k1_ADM_s113
                                 + 1.11344 w_k1_AXP_s113
                                 + 0.748769 w_k1_DIS_s113
                                 + 0.994861 w_k1_FDX_s113
                                 + 0.883989 w_k1_IBM_s113
                                 + 0.959846 w_k1_INTC_s113
                                 + 0.779735 w_k1_KO_s113
                                 + 0.863424 w_k1_MAT_s113
                                 + 0.901103 w_k1_MSFT_s113
                                 + 0.889447 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.891297 w_k1_ADM_s113
                                 + 0.875661 w_k1_AXP_s113
                                 + 0.901118 w_k1_DIS_s113
                                 + 0.927896 w_k1_FDX_s113
                                 + 1.01044 w_k1_IBM_s113
                                 + 1.12601 w_k1_INTC_s113
                                 + 0.97133 w_k1_KO_s113 + 1.30931 w_k1_MAT_s113
                                 + 0.986686 w_k1_MSFT_s113
                                 + 0.972268 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.897669 w_k1_ADM_s113
                                 + 0.70924 w_k1_AXP_s113
                                 + 0.637367 w_k1_DIS_s113
                                 + 0.777175 w_k1_FDX_s113
                                 + 0.736337 w_k1_IBM_s113
                                 + 0.653847 w_k1_INTC_s113
                                 + 0.678963 w_k1_KO_s113
                                 + 0.63189 w_k1_MAT_s113
                                 + 0.801779 w_k1_MSFT_s113
                                 + 0.766156 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.05049 w_k1_ADM_s113
                                 + 1.2147 w_k1_AXP_s113 + 1.07417 w_k1_DIS_s113
                                 + 1.15466 w_k1_FDX_s113
                                 + 0.747744 w_k1_IBM_s113
                                 + 1.06916 w_k1_INTC_s113
                                 + 0.876573 w_k1_KO_s113
                                 + 0.824591 w_k1_MAT_s113
                                 + 1.07883 w_k1_MSFT_s113
                                 + 0.991707 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.923476 w_k1_ADM_s113
                                 + 0.799171 w_k1_AXP_s113
                                 + 0.869954 w_k1_DIS_s113
                                 + 0.829886 w_k1_FDX_s113
                                 + 1.12679 w_k1_IBM_s113
                                 + 0.82314 w_k1_INTC_s113
                                 + 1.01146 w_k1_KO_s113 + 1.27327 w_k1_MAT_s113
                                 + 0.906973 w_k1_MSFT_s113
                                 + 0.885386 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.2732 w_k1_ADM_s114
                                 + 1.27457 w_k1_AXP_s114 + 1.2289 w_k1_DIS_s114
                                 + 1.11263 w_k1_FDX_s114
                                 + 1.26589 w_k1_IBM_s114
                                 + 1.16581 w_k1_INTC_s114
                                 + 1.40527 w_k1_KO_s114 + 1.2974 w_k1_MAT_s114
                                 + 1.14025 w_k1_MSFT_s114
                                 + 1.15379 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.838837 w_k1_ADM_s114
                                 + 0.84315 w_k1_AXP_s114
                                 + 0.581151 w_k1_DIS_s114
                                 + 0.696358 w_k1_FDX_s114
                                 + 1.16093 w_k1_IBM_s114
                                 + 0.893976 w_k1_INTC_s114
                                 + 0.996272 w_k1_KO_s114
                                 + 1.20471 w_k1_MAT_s114
                                 + 0.725014 w_k1_MSFT_s114
                                 + 0.897492 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.22931 w_k1_ADM_s114
                                 + 1.06052 w_k1_AXP_s114
                                 + 0.93129 w_k1_DIS_s114
                                 + 0.94924 w_k1_FDX_s114
                                 + 1.17739 w_k1_IBM_s114
                                 + 1.06116 w_k1_INTC_s114 + 1.0301 w_k1_KO_s114
                                 + 0.935353 w_k1_MAT_s114
                                 + 1.02422 w_k1_MSFT_s114
                                 + 0.988017 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.0126 w_k1_ADM_s114
                                 + 0.790377 w_k1_AXP_s114
                                 + 0.751745 w_k1_DIS_s114
                                 + 0.804299 w_k1_FDX_s114
                                 + 0.750325 w_k1_IBM_s114
                                 + 1.08152 w_k1_INTC_s114
                                 + 0.781257 w_k1_KO_s114
                                 + 0.528225 w_k1_MAT_s114
                                 + 0.890135 w_k1_MSFT_s114
                                 + 0.817954 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.47299 w_k1_ADM_s114
                                 + 1.48252 w_k1_AXP_s114 + 1.7238 w_k1_DIS_s114
                                 + 1.42363 w_k1_FDX_s114
                                 + 1.07468 w_k1_IBM_s114
                                 + 1.68494 w_k1_INTC_s114
                                 + 1.44231 w_k1_KO_s114
                                 + 0.959603 w_k1_MAT_s114
                                 + 1.54536 w_k1_MSFT_s114
                                 + 1.29479 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.00797 w_k1_ADM_s114
                                 + 1.16135 w_k1_AXP_s114
                                 + 1.48815 w_k1_DIS_s114
                                 + 1.23415 w_k1_FDX_s114
                                 + 1.58908 w_k1_IBM_s114
                                 + 1.24478 w_k1_INTC_s114 + 1.3368 w_k1_KO_s114
                                 + 1.47627 w_k1_MAT_s114
                                 + 1.40588 w_k1_MSFT_s114
                                 + 1.34955 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.42979 w_k1_ADM_s114
                                 + 1.20503 w_k1_AXP_s114
                                 + 1.20795 w_k1_DIS_s114
                                 + 1.13313 w_k1_FDX_s114
                                 + 0.878217 w_k1_IBM_s114
                                 + 1.07699 w_k1_INTC_s114
                                 + 1.22956 w_k1_KO_s114 + 1.06517 w_k1_MAT_s114
                                 + 1.2694 w_k1_MSFT_s114 + 1.17113 w_k1_PG_s114
                                 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.17431 w_k1_ADM_s114
                                 + 0.876328 w_k1_AXP_s114
                                 + 1.04003 w_k1_DIS_s114
                                 + 0.953037 w_k1_FDX_s114
                                 + 0.959889 w_k1_IBM_s114
                                 + 0.903484 w_k1_INTC_s114
                                 + 0.958304 w_k1_KO_s114
                                 + 0.916942 w_k1_MAT_s114
                                 + 1.05124 w_k1_MSFT_s114
                                 + 1.06772 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.07873 w_k1_ADM_s115
                                 + 0.903764 w_k1_AXP_s115
                                 + 1.05693 w_k1_DIS_s115
                                 + 1.03788 w_k1_FDX_s115
                                 + 1.05741 w_k1_IBM_s115
                                 + 1.17335 w_k1_INTC_s115
                                 + 1.07415 w_k1_KO_s115
                                 + 0.876781 w_k1_MAT_s115
                                 + 1.16525 w_k1_MSFT_s115
                                 + 1.12581 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.11514 w_k1_ADM_s115
                                 + 1.12247 w_k1_AXP_s115
                                 + 1.12077 w_k1_DIS_s115
                                 + 1.10191 w_k1_FDX_s115
                                 + 1.55331 w_k1_IBM_s115
                                 + 1.12215 w_k1_INTC_s115
                                 + 1.18054 w_k1_KO_s115 + 1.40899 w_k1_MAT_s115
                                 + 1.1039 w_k1_MSFT_s115 + 1.15039 w_k1_PG_s115
                                 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.725474 w_k1_ADM_s115
                                 + 0.932551 w_k1_AXP_s115
                                 + 0.640014 w_k1_DIS_s115
                                 + 0.853738 w_k1_FDX_s115
                                 + 1.14038 w_k1_IBM_s115
                                 + 0.919947 w_k1_INTC_s115
                                 + 0.842988 w_k1_KO_s115
                                 + 0.942344 w_k1_MAT_s115
                                 + 0.809125 w_k1_MSFT_s115
                                 + 0.87459 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.26036 w_k1_ADM_s115
                                 + 1.22971 w_k1_AXP_s115
                                 + 1.25976 w_k1_DIS_s115
                                 + 1.26172 w_k1_FDX_s115
                                 + 0.990571 w_k1_IBM_s115
                                 + 1.20553 w_k1_INTC_s115
                                 + 1.15433 w_k1_KO_s115 + 1.00493 w_k1_MAT_s115
                                 + 1.13386 w_k1_MSFT_s115 + 1.0704 w_k1_PG_s115
                                 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.91095 w_k1_ADM_s115
                                 + 0.764569 w_k1_AXP_s115
                                 + 0.839368 w_k1_DIS_s115
                                 + 0.925738 w_k1_FDX_s115
                                 + 0.867877 w_k1_IBM_s115
                                 + 0.989195 w_k1_INTC_s115
                                 + 0.830457 w_k1_KO_s115
                                 + 0.75728 w_k1_MAT_s115
                                 + 0.839133 w_k1_MSFT_s115
                                 + 0.997105 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.04967 w_k1_ADM_s115
                                 + 0.811784 w_k1_AXP_s115
                                 + 0.935107 w_k1_DIS_s115
                                 + 0.894171 w_k1_FDX_s115
                                 + 1.07273 w_k1_IBM_s115
                                 + 0.824778 w_k1_INTC_s115
                                 + 1.03107 w_k1_KO_s115 + 1.00504 w_k1_MAT_s115
                                 + 0.878918 w_k1_MSFT_s115
                                 + 0.946111 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.930712 w_k1_ADM_s115
                                 + 0.747476 w_k1_AXP_s115
                                 + 0.79321 w_k1_DIS_s115 + 0.9215 w_k1_FDX_s115
                                 + 0.70763 w_k1_IBM_s115
                                 + 0.79603 w_k1_INTC_s115
                                 + 0.900999 w_k1_KO_s115
                                 + 0.757762 w_k1_MAT_s115
                                 + 0.936051 w_k1_MSFT_s115
                                 + 0.884233 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.21801 w_k1_ADM_s115
                                 + 1.11636 w_k1_AXP_s115
                                 + 1.64238 w_k1_DIS_s115
                                 + 1.39233 w_k1_FDX_s115 + 1.1438 w_k1_IBM_s115
                                 + 1.14826 w_k1_INTC_s115
                                 + 1.34533 w_k1_KO_s115 + 1.17989 w_k1_MAT_s115
                                 + 1.30552 w_k1_MSFT_s115
                                 + 1.24315 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.15764 w_k1_ADM_s116
                                 + 1.10263 w_k1_AXP_s116
                                 + 1.24844 w_k1_DIS_s116
                                 + 1.13417 w_k1_FDX_s116
                                 + 1.14933 w_k1_IBM_s116
                                 + 0.973308 w_k1_INTC_s116
                                 + 1.11529 w_k1_KO_s116 + 0.97235 w_k1_MAT_s116
                                 + 0.89876 w_k1_MSFT_s116
                                 + 1.11893 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.73077 w_k1_ADM_s116
                                 + 1.6173 w_k1_AXP_s116 + 1.7324 w_k1_DIS_s116
                                 + 1.53036 w_k1_FDX_s116
                                 + 1.32611 w_k1_IBM_s116
                                 + 1.2222 w_k1_INTC_s116 + 1.27642 w_k1_KO_s116
                                 + 1.35178 w_k1_MAT_s116
                                 + 1.11001 w_k1_MSFT_s116
                                 + 1.33228 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.30236 w_k1_ADM_s116
                                 + 1.23503 w_k1_AXP_s116
                                 + 1.67304 w_k1_DIS_s116
                                 + 1.46519 w_k1_FDX_s116
                                 + 1.00622 w_k1_IBM_s116
                                 + 1.2959 w_k1_INTC_s116 + 1.32727 w_k1_KO_s116
                                 + 0.947007 w_k1_MAT_s116
                                 + 1.31262 w_k1_MSFT_s116
                                 + 1.48277 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.25652 w_k1_ADM_s116
                                 + 1.42972 w_k1_AXP_s116
                                 + 1.45622 w_k1_DIS_s116
                                 + 1.35614 w_k1_FDX_s116
                                 + 0.750459 w_k1_IBM_s116
                                 + 1.15822 w_k1_INTC_s116
                                 + 1.08612 w_k1_KO_s116
                                 + 0.792772 w_k1_MAT_s116
                                 + 1.18698 w_k1_MSFT_s116
                                 + 1.07651 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.00293 w_k1_ADM_s116
                                 + 0.77152 w_k1_AXP_s116
                                 + 0.706569 w_k1_DIS_s116
                                 + 0.88795 w_k1_FDX_s116
                                 + 0.701633 w_k1_IBM_s116
                                 + 0.843156 w_k1_INTC_s116
                                 + 0.805455 w_k1_KO_s116
                                 + 0.736183 w_k1_MAT_s116
                                 + 0.77866 w_k1_MSFT_s116
                                 + 1.04277 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.00571 w_k1_ADM_s116
                                 + 0.845395 w_k1_AXP_s116
                                 + 0.869884 w_k1_DIS_s116
                                 + 0.936208 w_k1_FDX_s116
                                 + 0.857225 w_k1_IBM_s116
                                 + 0.898602 w_k1_INTC_s116
                                 + 0.936693 w_k1_KO_s116
                                 + 0.687658 w_k1_MAT_s116
                                 + 0.853056 w_k1_MSFT_s116
                                 + 0.975338 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.886873 w_k1_ADM_s116
                                 + 0.782978 w_k1_AXP_s116
                                 + 0.79592 w_k1_DIS_s116
                                 + 0.79546 w_k1_FDX_s116
                                 + 1.37907 w_k1_IBM_s116
                                 + 0.736712 w_k1_INTC_s116
                                 + 0.963683 w_k1_KO_s116
                                 + 1.07871 w_k1_MAT_s116
                                 + 0.725171 w_k1_MSFT_s116
                                 + 0.995539 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.861805 w_k1_ADM_s116
                                 + 1.08848 w_k1_AXP_s116
                                 + 0.741145 w_k1_DIS_s116
                                 + 0.957175 w_k1_FDX_s116
                                 + 1.31714 w_k1_IBM_s116
                                 + 0.836761 w_k1_INTC_s116
                                 + 0.967461 w_k1_KO_s116
                                 + 1.34535 w_k1_MAT_s116
                                 + 0.78199 w_k1_MSFT_s116
                                 + 1.07118 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.974387 w_k1_ADM_s117
                                 + 1.05999 w_k1_AXP_s117
                                 + 1.26098 w_k1_DIS_s117
                                 + 1.20041 w_k1_FDX_s117
                                 + 0.98693 w_k1_IBM_s117
                                 + 1.0746 w_k1_INTC_s117 + 1.03943 w_k1_KO_s117
                                 + 0.937926 w_k1_MAT_s117
                                 + 1.26731 w_k1_MSFT_s117
                                 + 1.15621 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.839486 w_k1_ADM_s117
                                 + 0.931949 w_k1_AXP_s117
                                 + 0.771318 w_k1_DIS_s117
                                 + 0.832892 w_k1_FDX_s117
                                 + 1.00536 w_k1_IBM_s117
                                 + 0.784006 w_k1_INTC_s117
                                 + 0.882264 w_k1_KO_s117
                                 + 0.956877 w_k1_MAT_s117
                                 + 0.763964 w_k1_MSFT_s117
                                 + 0.983587 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.03453 w_k1_ADM_s117
                                 + 0.708272 w_k1_AXP_s117
                                 + 0.856252 w_k1_DIS_s117
                                 + 0.963812 w_k1_FDX_s117
                                 + 0.747309 w_k1_IBM_s117
                                 + 0.937593 w_k1_INTC_s117
                                 + 0.70209 w_k1_KO_s117
                                 + 0.554528 w_k1_MAT_s117
                                 + 1.03843 w_k1_MSFT_s117
                                 + 1.05567 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.932622 w_k1_ADM_s117
                                 + 0.904731 w_k1_AXP_s117
                                 + 1.03387 w_k1_DIS_s117 + 1.0132 w_k1_FDX_s117
                                 + 1.19113 w_k1_IBM_s117
                                 + 1.10061 w_k1_INTC_s117
                                 + 0.955576 w_k1_KO_s117
                                 + 0.876742 w_k1_MAT_s117
                                 + 1.13115 w_k1_MSFT_s117
                                 + 1.01761 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.1331 w_k1_ADM_s117
                                 + 1.00568 w_k1_AXP_s117
                                 + 1.08327 w_k1_DIS_s117
                                 + 1.20987 w_k1_FDX_s117
                                 + 0.762067 w_k1_IBM_s117
                                 + 1.02234 w_k1_INTC_s117
                                 + 0.837738 w_k1_KO_s117
                                 + 0.623623 w_k1_MAT_s117
                                 + 1.15981 w_k1_MSFT_s117
                                 + 1.09446 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.774317 w_k1_ADM_s117
                                 + 0.622255 w_k1_AXP_s117
                                 + 0.64967 w_k1_DIS_s117
                                 + 0.717877 w_k1_FDX_s117
                                 + 1.1667 w_k1_IBM_s117
                                 + 0.916556 w_k1_INTC_s117
                                 + 0.852268 w_k1_KO_s117
                                 + 0.710072 w_k1_MAT_s117
                                 + 0.895276 w_k1_MSFT_s117
                                 + 0.90442 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.07951 w_k1_ADM_s117
                                 + 1.42191 w_k1_AXP_s117
                                 + 1.36373 w_k1_DIS_s117
                                 + 1.20081 w_k1_FDX_s117
                                 + 1.46509 w_k1_IBM_s117
                                 + 1.1125 w_k1_INTC_s117 + 1.21713 w_k1_KO_s117
                                 + 1.49046 w_k1_MAT_s117
                                 + 1.15783 w_k1_MSFT_s117
                                 + 1.13456 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.735868 w_k1_ADM_s117
                                 + 0.545306 w_k1_AXP_s117
                                 + 0.515939 w_k1_DIS_s117
                                 + 0.714659 w_k1_FDX_s117
                                 + 0.693149 w_k1_IBM_s117
                                 + 0.695013 w_k1_INTC_s117
                                 + 0.589846 w_k1_KO_s117
                                 + 0.502256 w_k1_MAT_s117
                                 + 0.773755 w_k1_MSFT_s117
                                 + 0.75812 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0304258731998542 w_k1_ADM_s000
                                 + 0.016641554547332 w_k1_ADM_s001
                                 + 0.010954651919996 w_k1_ADM_s002
                                 + 0.0297580800458903 w_k1_ADM_s003
                                 + 0.00954973528448377 w_k1_ADM_s004
                                 + 0.00912274002556397 w_k1_ADM_s005
                                 + 0.00812576379074709 w_k1_ADM_s006
                                 + 0.0163966619139212 w_k1_ADM_s007
                                 + 0.00596340430666864 w_k1_ADM_s010
                                 + 0.00658781046228142 w_k1_ADM_s011
                                 + 0.00306883855789555 w_k1_ADM_s012
                                 + 0.0108532173477279 w_k1_ADM_s013
                                 + 0.00518599020409549 w_k1_ADM_s014
                                 + 0.00440866134733848 w_k1_ADM_s015
                                 + 0.0043857517014046 w_k1_ADM_s016
                                 + 0.00800875899391333 w_k1_ADM_s017
                                 + 0.0251370247068665 w_k1_ADM_s100
                                 + 0.0473802173939426 w_k1_ADM_s101
                                 + 0.0822338898855413 w_k1_ADM_s102
                                 + 0.121389006421693 w_k1_ADM_s103
                                 + 0.117342819304955 w_k1_ADM_s104
                                 + 0.0389163120613712 w_k1_ADM_s105
                                 + 0.0735169529130014 w_k1_ADM_s106
                                 + 0.0332705752281314 w_k1_ADM_s107
                                 + 0.0653525768898309 w_k1_ADM_s110
                                 + 0.0114928440136494 w_k1_ADM_s111
                                 + 0.0448631047051993 w_k1_ADM_s112
                                 + 0.0374441152945704 w_k1_ADM_s113
                                 + 0.0339336847991673 w_k1_ADM_s114
                                 + 0.0341727976358313 w_k1_ADM_s115
                                 + 0.0359380821614289 w_k1_ADM_s116
                                 + 0.0525377082568473 w_k1_ADM_s117
                                 + 0.0311913295638189 w_k1_AXP_s000
                                 + 0.0170102127994785 w_k1_AXP_s001
                                 + 0.0133620032300984 w_k1_AXP_s002
                                 + 0.0309270033047396 w_k1_AXP_s003
                                 + 0.00925797137964656 w_k1_AXP_s004
                                 + 0.0099614083209563 w_k1_AXP_s005
                                 + 0.00763981554540087 w_k1_AXP_s006
                                 + 0.0172963799027571 w_k1_AXP_s007
                                 + 0.00692922273166993 w_k1_AXP_s010
                                 + 0.00667554840753887 w_k1_AXP_s011
                                 + 0.00362595614564405 w_k1_AXP_s012
                                 + 0.0130216216779835 w_k1_AXP_s013
                                 + 0.00583662961237059 w_k1_AXP_s014
                                 + 0.00397055391084739 w_k1_AXP_s015
                                 + 0.00364354353756176 w_k1_AXP_s016
                                 + 0.00952220415300413 w_k1_AXP_s017
                                 + 0.0282880745725903 w_k1_AXP_s100
                                 + 0.0551310194322048 w_k1_AXP_s101
                                 + 0.0824761629145829 w_k1_AXP_s102
                                 + 0.118264386752598 w_k1_AXP_s103
                                 + 0.10378104315829 w_k1_AXP_s104
                                 + 0.0413689781079111 w_k1_AXP_s105
                                 + 0.0754125930373145 w_k1_AXP_s106
                                 + 0.035769621263658 w_k1_AXP_s107
                                 + 0.0642564912317451 w_k1_AXP_s110
                                 + 0.0116259012116438 w_k1_AXP_s111
                                 + 0.037624210689732 w_k1_AXP_s112
                                 + 0.0351878313841038 w_k1_AXP_s113
                                 + 0.029777622128554 w_k1_AXP_s114
                                 + 0.0309649644318056 w_k1_AXP_s115
                                 + 0.0332441738456651 w_k1_AXP_s116
                                 + 0.0530156139763901 w_k1_AXP_s117
                                 + 0.0303682518808601 w_k1_DIS_s000
                                 + 0.0188399976292714 w_k1_DIS_s001
                                 + 0.0131053469038677 w_k1_DIS_s002
                                 + 0.0319167920633487 w_k1_DIS_s003
                                 + 0.0102135152679983 w_k1_DIS_s004
                                 + 0.00972059311361799 w_k1_DIS_s005
                                 + 0.00759648657189797 w_k1_DIS_s006
                                 + 0.0181183660137672 w_k1_DIS_s007
                                 + 0.00613211674404667 w_k1_DIS_s010
                                 + 0.00727210930844443 w_k1_DIS_s011
                                 + 0.00378601085991143 w_k1_DIS_s012
                                 + 0.0134984318591533 w_k1_DIS_s013
                                 + 0.0060205147723874 w_k1_DIS_s014
                                 + 0.00452300129809031 w_k1_DIS_s015
                                 + 0.0038784601501073 w_k1_DIS_s016
                                 + 0.00878608003896388 w_k1_DIS_s017
                                 + 0.0264758332604923 w_k1_DIS_s100
                                 + 0.0534574309603188 w_k1_DIS_s101
                                 + 0.0831403792888943 w_k1_DIS_s102
                                 + 0.121808184079167 w_k1_DIS_s103
                                 + 0.115038822981021 w_k1_DIS_s104
                                 + 0.040296589335288 w_k1_DIS_s105
                                 + 0.0762097529474526 w_k1_DIS_s106
                                 + 0.0342758945319728 w_k1_DIS_s107
                                 + 0.060669885844611 w_k1_DIS_s110
                                 + 0.0116708135605127 w_k1_DIS_s111
                                 + 0.0386614233261641 w_k1_DIS_s112
                                 + 0.0358382021873179 w_k1_DIS_s113
                                 + 0.0308888301210588 w_k1_DIS_s114
                                 + 0.0332240310751601 w_k1_DIS_s115
                                 + 0.0336030208907979 w_k1_DIS_s116
                                 + 0.0559727114667844 w_k1_DIS_s117
                                 + 0.0299599569397033 w_k1_FDX_s000
                                 + 0.0183836843691699 w_k1_FDX_s001
                                 + 0.012608873896209 w_k1_FDX_s002
                                 + 0.0302030302595078 w_k1_FDX_s003
                                 + 0.0100794256263666 w_k1_FDX_s004
                                 + 0.0101297095037549 w_k1_FDX_s005
                                 + 0.00736140941942887 w_k1_FDX_s006
                                 + 0.0172036659252163 w_k1_FDX_s007
                                 + 0.00617526440213865 w_k1_FDX_s010
                                 + 0.00687365765985875 w_k1_FDX_s011
                                 + 0.00363215441839574 w_k1_FDX_s012
                                 + 0.0122589279293527 w_k1_FDX_s013
                                 + 0.00569954782959531 w_k1_FDX_s014
                                 + 0.0038880045701469 w_k1_FDX_s015
                                 + 0.00391459160972962 w_k1_FDX_s016
                                 + 0.00858955104010109 w_k1_FDX_s017
                                 + 0.0259512378201417 w_k1_FDX_s100
                                 + 0.0520465250504993 w_k1_FDX_s101
                                 + 0.0771306419051195 w_k1_FDX_s102
                                 + 0.117850764546863 w_k1_FDX_s103
                                 + 0.104420342083668 w_k1_FDX_s104
                                 + 0.037767734898117 w_k1_FDX_s105
                                 + 0.0780301618991202 w_k1_FDX_s106
                                 + 0.0328353397483811 w_k1_FDX_s107
                                 + 0.063813483958451 w_k1_FDX_s110
                                 + 0.0124784108983988 w_k1_FDX_s111
                                 + 0.037038283123549 w_k1_FDX_s112
                                 + 0.0355521816326728 w_k1_FDX_s113
                                 + 0.0290766853427106 w_k1_FDX_s114
                                 + 0.0341089772798768 w_k1_FDX_s115
                                 + 0.0345223448874833 w_k1_FDX_s116
                                 + 0.0564760589124472 w_k1_FDX_s117
                                 + 0.0291558899195712 w_k1_IBM_s000
                                 + 0.0184715709140881 w_k1_IBM_s001
                                 + 0.0129030761711355 w_k1_IBM_s002
                                 + 0.0311816811400977 w_k1_IBM_s003
                                 + 0.0105514191766655 w_k1_IBM_s004
                                 + 0.0096754609602281 w_k1_IBM_s005
                                 + 0.00762335771403934 w_k1_IBM_s006
                                 + 0.0181902793184087 w_k1_IBM_s007
                                 + 0.00667262988294073 w_k1_IBM_s010
                                 + 0.00724443593293417 w_k1_IBM_s011
                                 + 0.00380912138436178 w_k1_IBM_s012
                                 + 0.0132700382090865 w_k1_IBM_s013
                                 + 0.00548221317766493 w_k1_IBM_s014
                                 + 0.00402499286877732 w_k1_IBM_s015
                                 + 0.00358338011335857 w_k1_IBM_s016
                                 + 0.00832484923085525 w_k1_IBM_s017
                                 + 0.0248321903740897 w_k1_IBM_s100
                                 + 0.0519701836869417 w_k1_IBM_s101
                                 + 0.0760355721670535 w_k1_IBM_s102
                                 + 0.119348599854343 w_k1_IBM_s103
                                 + 0.10704197397247 w_k1_IBM_s104
                                 + 0.0405276503812978 w_k1_IBM_s105
                                 + 0.0800173649285156 w_k1_IBM_s106
                                 + 0.0341406252336757 w_k1_IBM_s107
                                 + 0.0603284326720183 w_k1_IBM_s110
                                 + 0.0127155289014884 w_k1_IBM_s111
                                 + 0.0361376234586533 w_k1_IBM_s112
                                 + 0.0340091038492265 w_k1_IBM_s113
                                 + 0.0302546503049016 w_k1_IBM_s114
                                 + 0.0342555169739182 w_k1_IBM_s115
                                 + 0.0331869238608848 w_k1_IBM_s116
                                 + 0.0573250154519884 w_k1_IBM_s117
                                 + 0.0371901058953328 w_k1_INTC_s000
                                 + 0.017459413195983 w_k1_INTC_s001
                                 + 0.0146050197481731 w_k1_INTC_s002
                                 + 0.0300777088669278 w_k1_INTC_s003
                                 + 0.0105603571690948 w_k1_INTC_s004
                                 + 0.0111944749743964 w_k1_INTC_s005
                                 + 0.00647120602507489 w_k1_INTC_s006
                                 + 0.0189745518301007 w_k1_INTC_s007
                                 + 0.00616127978023008 w_k1_INTC_s010
                                 + 0.00712854338802934 w_k1_INTC_s011
                                 + 0.0034671073931749 w_k1_INTC_s012
                                 + 0.0134880922834883 w_k1_INTC_s013
                                 + 0.00629421463084641 w_k1_INTC_s014
                                 + 0.00351793820895626 w_k1_INTC_s015
                                 + 0.00418036033850715 w_k1_INTC_s016
                                 + 0.00882745598218675 w_k1_INTC_s017
                                 + 0.0295032427224756 w_k1_INTC_s100
                                 + 0.0514979863001108 w_k1_INTC_s101
                                 + 0.0708379266987941 w_k1_INTC_s102
                                 + 0.128238223360209 w_k1_INTC_s103
                                 + 0.115021706055724 w_k1_INTC_s104
                                 + 0.0358689629703019 w_k1_INTC_s105
                                 + 0.0810968050877313 w_k1_INTC_s106
                                 + 0.0313632781025134 w_k1_INTC_s107
                                 + 0.0589505334996025 w_k1_INTC_s110
                                 + 0.0119749024370579 w_k1_INTC_s111
                                 + 0.0335154723487183 w_k1_INTC_s112
                                 + 0.0338800267408469 w_k1_INTC_s113
                                 + 0.0316813265676281 w_k1_INTC_s114
                                 + 0.0331480701258277 w_k1_INTC_s115
                                 + 0.0310392892223002 w_k1_INTC_s116
                                 + 0.054221589778289 w_k1_INTC_s117
                                 + 0.0276766479307139 w_k1_KO_s000
                                 + 0.0183123200139758 w_k1_KO_s001
                                 + 0.0122030989776658 w_k1_KO_s002
                                 + 0.0314067165358071 w_k1_KO_s003
                                 + 0.0110180474393707 w_k1_KO_s004
                                 + 0.0102982954756241 w_k1_KO_s005
                                 + 0.0078036330263219 w_k1_KO_s006
                                 + 0.0175999365130187 w_k1_KO_s007
                                 + 0.00582548288692101 w_k1_KO_s010
                                 + 0.00734196967978777 w_k1_KO_s011
                                 + 0.00388398730677996 w_k1_KO_s012
                                 + 0.012143241918641 w_k1_KO_s013
                                 + 0.005032656655024 w_k1_KO_s014
                                 + 0.00408624448819958 w_k1_KO_s015
                                 + 0.00402054201593853 w_k1_KO_s016
                                 + 0.00851095666573106 w_k1_KO_s017
                                 + 0.0253736066283687 w_k1_KO_s100
                                 + 0.0602915950940594 w_k1_KO_s101
                                 + 0.08433269597271 w_k1_KO_s102
                                 + 0.131755595557754 w_k1_KO_s103
                                 + 0.110564770885228 w_k1_KO_s104
                                 + 0.0384112390471393 w_k1_KO_s105
                                 + 0.0731806016227089 w_k1_KO_s106
                                 + 0.0297587095113418 w_k1_KO_s107
                                 + 0.0597581478267251 w_k1_KO_s110
                                 + 0.0122155059766632 w_k1_KO_s111
                                 + 0.0343644982245976 w_k1_KO_s112
                                 + 0.0355608592795796 w_k1_KO_s113
                                 + 0.0313222246997365 w_k1_KO_s114
                                 + 0.0337831832708564 w_k1_KO_s115
                                 + 0.0329679217277384 w_k1_KO_s116
                                 + 0.0510339527877641 w_k1_KO_s117
                                 + 0.0239359465602581 w_k1_MAT_s000
                                 + 0.018107955974652 w_k1_MAT_s001
                                 + 0.0113707667335629 w_k1_MAT_s002
                                 + 0.0389521532320391 w_k1_MAT_s003
                                 + 0.0100502232634473 w_k1_MAT_s004
                                 + 0.00865849850439567 w_k1_MAT_s005
                                 + 0.00719104888832666 w_k1_MAT_s006
                                 + 0.0167216725512313 w_k1_MAT_s007
                                 + 0.00580893366572064 w_k1_MAT_s010
                                 + 0.00676345845641543 w_k1_MAT_s011
                                 + 0.00428201554621721 w_k1_MAT_s012
                                 + 0.0118990328470199 w_k1_MAT_s013
                                 + 0.00627323738429561 w_k1_MAT_s014
                                 + 0.00498490994888402 w_k1_MAT_s015
                                 + 0.00437744999231914 w_k1_MAT_s016
                                 + 0.0078836004063537 w_k1_MAT_s017
                                 + 0.027788480750635 w_k1_MAT_s100
                                 + 0.056102642488442 w_k1_MAT_s101
                                 + 0.0916674320956492 w_k1_MAT_s102
                                 + 0.113062790361115 w_k1_MAT_s103
                                 + 0.113843450413455 w_k1_MAT_s104
                                 + 0.0293997744369618 w_k1_MAT_s105
                                 + 0.08647336884624 w_k1_MAT_s106
                                 + 0.029724077935036 w_k1_MAT_s107
                                 + 0.0655503387606842 w_k1_MAT_s110
                                 + 0.0112333261379748 w_k1_MAT_s111
                                 + 0.0379820050796884 w_k1_MAT_s112
                                 + 0.0327031683214718 w_k1_MAT_s113
                                 + 0.028148025969634 w_k1_MAT_s114
                                 + 0.0320900879888312 w_k1_MAT_s115
                                 + 0.0303747795550911 w_k1_MAT_s116
                                 + 0.0492634069340785 w_k1_MAT_s117
                                 + 0.0308631245670518 w_k1_MSFT_s000
                                 + 0.0178031361989947 w_k1_MSFT_s001
                                 + 0.0129142011119217 w_k1_MSFT_s002
                                 + 0.0287583596889359 w_k1_MSFT_s003
                                 + 0.00994989396828566 w_k1_MSFT_s004
                                 + 0.010126850008254 w_k1_MSFT_s005
                                 + 0.00702467655620069 w_k1_MSFT_s006
                                 + 0.018053251170929 w_k1_MSFT_s007
                                 + 0.0062958885125533 w_k1_MSFT_s010
                                 + 0.00702567316966144 w_k1_MSFT_s011
                                 + 0.00415588306477295 w_k1_MSFT_s012
                                 + 0.0118542430444876 w_k1_MSFT_s013
                                 + 0.00565349599610202 w_k1_MSFT_s014
                                 + 0.00403209499381027 w_k1_MSFT_s015
                                 + 0.00374394964858154 w_k1_MSFT_s016
                                 + 0.0088262989953292 w_k1_MSFT_s017
                                 + 0.0254070471747491 w_k1_MSFT_s100
                                 + 0.0509975225553986 w_k1_MSFT_s101
                                 + 0.0751769220304575 w_k1_MSFT_s102
                                 + 0.130355675610377 w_k1_MSFT_s103
                                 + 0.115733791209639 w_k1_MSFT_s104
                                 + 0.0385019581264835 w_k1_MSFT_s105
                                 + 0.0807790944494306 w_k1_MSFT_s106
                                 + 0.0311805557867733 w_k1_MSFT_s107
                                 + 0.0614133319668332 w_k1_MSFT_s110
                                 + 0.0119268493267088 w_k1_MSFT_s111
                                 + 0.0334350044970621 w_k1_MSFT_s112
                                 + 0.0356012215468868 w_k1_MSFT_s113
                                 + 0.0318752973257723 w_k1_MSFT_s114
                                 + 0.0327417133234571 w_k1_MSFT_s115
                                 + 0.0295087305473784 w_k1_MSFT_s116
                                 + 0.0585598290463503 w_k1_MSFT_s117
                                 + 0.0260719200747327 w_k1_PG_s000
                                 + 0.0180682490213544 w_k1_PG_s001
                                 + 0.0119944758950751 w_k1_PG_s002
                                 + 0.032893382243425 w_k1_PG_s003
                                 + 0.0109024952061391 w_k1_PG_s004
                                 + 0.0102299419610071 w_k1_PG_s005
                                 + 0.00826287478780648 w_k1_PG_s006
                                 + 0.0164107441241645 w_k1_PG_s007
                                 + 0.00698671227397208 w_k1_PG_s010
                                 + 0.00655702568732781 w_k1_PG_s011
                                 + 0.00381070455447781 w_k1_PG_s012
                                 + 0.0127836927611668 w_k1_PG_s013
                                 + 0.00506112625561519 w_k1_PG_s014
                                 + 0.00378226860142592 w_k1_PG_s015
                                 + 0.00389062302882011 w_k1_PG_s016
                                 + 0.00866904767693692 w_k1_PG_s017
                                 + 0.0266631841003426 w_k1_PG_s100
                                 + 0.0547159409062841 w_k1_PG_s101
                                 + 0.0787522614924115 w_k1_PG_s102
                                 + 0.116393254860803 w_k1_PG_s103
                                 + 0.106808124637855 w_k1_PG_s104
                                 + 0.0379848712863018 w_k1_PG_s105
                                 + 0.077389894351369 w_k1_PG_s106
                                 + 0.0322534040587695 w_k1_PG_s107
                                 + 0.060376450510351 w_k1_PG_s110
                                 + 0.0118990684707767 w_k1_PG_s111
                                 + 0.0369509104283517 w_k1_PG_s112
                                 + 0.0324930979886337 w_k1_PG_s113
                                 + 0.0307311722477446 w_k1_PG_s114
                                 + 0.0337196099858701 w_k1_PG_s115
                                 + 0.0357914392442153 w_k1_PG_s116
                                 + 0.0567861041808522 w_k1_PG_s117 >= 1.14
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

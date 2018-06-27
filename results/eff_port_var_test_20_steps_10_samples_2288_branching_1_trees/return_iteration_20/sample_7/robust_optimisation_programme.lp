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
 w_k1_ADM_s0_return_constraint:  0.965383 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  1.01858 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.31065 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.963663 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.09343 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.971221 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.868339 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.928411 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.689819 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.701857 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.866295 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.29144 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.09753 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.24708 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.35199 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.00981 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.994268 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.897389 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.790369 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.08558 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.10734 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.61226 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.40908 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.741751 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.02149 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.21809 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.905047 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.761061 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.10093 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.18458 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.25417 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.961299 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.3988 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.26715 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.20155 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.762955 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.994813 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.719918 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  0.984878 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  1.10545 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.42942 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.898738 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.10753 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.946111 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.1302 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.862305 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.632769 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.73017 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.89747 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.24474 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.04581 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.34036 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.43591 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.945204 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.15654 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.801263 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.734425 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.944342 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.05633 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.574186 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.39308 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.592527 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.844426 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.8968 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.927856 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.654943 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.2214 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.2831 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.36504 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.95598 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.29742 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.61451 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.45663 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.735357 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.824712 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.594696 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.01933 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  1.07 w_0_DIS - w_k1_DIS_s1 + 0.99 b_k1_DIS_s1
                                 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.46399 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.823949 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.03479 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.05148 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.04948 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.749186 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.519538 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.760389 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.909154 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.29104 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.07647 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.55051 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.58783 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.06435 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.32441 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.779598 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.746908 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.979513 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.06614 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.507292 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.41553 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.496582 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.894248 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 2.10452 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.01613 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.664775 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.54809 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.13072 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.508 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.960798 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.51474 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.7187 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.22044 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.580991 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.76699 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.4746 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.02754 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.992951 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.25686 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.886506 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.0906 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.956184 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.08096 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.938189 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.677051 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.73084 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.863481 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.13684 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.22548 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.39701 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.41862 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.918789 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.17391 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.953862 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.79055 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.07162 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.02333 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.607855 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.39019 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.624616 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.93112 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.78585 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.10926 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.813034 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.26456 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.10976 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.23286 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.997392 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.43093 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.4041 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.29067 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.769638 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.985966 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.645183 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.897366 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.26807 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.28248 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.904023 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.777923 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.09081 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.20869 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.789403 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.767413 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.02841 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.41727 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.49644 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.616547 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.0848 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.08345 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.5987 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.32917 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.577327 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.89138 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.7564 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.948979 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.05288 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.03005 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.912114 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.22013 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.48537 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.866209 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.69965 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.33486 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.182 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.36078 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.993953 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.968531 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 2.00717 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.898662 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.883247 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.655998 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.558962 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.01847 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 1.01241 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.20269 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.856945 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.01326 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.01286 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.19265 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.925313 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.701065 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.796039 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.881343 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.10566 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.14606 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.21492 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.28745 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.01433 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.22437 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.991002 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.778527 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.02774 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.999327 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.771634 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.9626 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.732744 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.8657 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.54804 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.08482 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.877255 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.09295 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.01474 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.16144 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.01078 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.0811 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.42847 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.0218 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.794007 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.965929 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.696364 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   0.974967 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.12457 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.28232 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.892907 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.01497 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.0235 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.09318 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.770106 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.712707 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.871722 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.09486 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.25472 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.846824 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.16971 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.21858 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.15455 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.20373 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.700123 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.83951 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.78888 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.05105 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.768882 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.0758 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.627892 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.97427 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.29693 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.912956 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.704419 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.30756 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.02222 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.41944 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.947786 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.29431 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.68759 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.02603 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.752066 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.777388 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.588263 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  0.897173 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.31333 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.43806 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.972893 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.859757 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.07576 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.21684 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.725039 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.752199 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.972727 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.38558 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.37596 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.648054 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.983701 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.33285 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.46312 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.22824 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.526712 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.717965 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.781475 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.937993 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.774134 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.26831 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.888486 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.984569 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.86687 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.831822 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.614229 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.32054 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.21317 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.2863 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.04537 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.14704 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.88824 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.15813 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.831377 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.564139 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.425957 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.03344 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 1.00125 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.20223 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.841266 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.05885 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.00228 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.14073 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.871023 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.674226 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.829881 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.974551 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.13175 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.19531 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.29819 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.32675 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.01149 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.21261 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.897031 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.874455 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.13134 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.0686 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.725635 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.1936 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.745297 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.897414 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.54759 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.11201 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.785602 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.27251 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.976842 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.27521 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.00414 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.3598 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.32256 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.08293 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.703833 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.913038 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.71882 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   0.98653 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.04401 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.22689 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.920675 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.987726 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.02509 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.01409 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.857805 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.727796 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.847459 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.947505 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.07218 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.914924 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.05939 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.2587 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.0347 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.09929 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.89431 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.90276 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.964626 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.00398 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.7819 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.21308 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.811133 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.988473 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.3444 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.03764 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.829469 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.29013 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.12094 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.20299 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.94347 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.17314 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.36741 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.07901 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.801283 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.869171 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.68819 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.151999948396248 z_k1_s0000
                                 - 0.210999865629033 z_k1_s0001
                                 - 0.23999954209953 z_k1_s0002
                                 - 0.300998919116751 z_k1_s0003
                                 - 0.250000417019889 z_k1_s0004
                                 - 0.546000090675855 z_k1_s0005
                                 - 0.190999784587467 z_k1_s0006
                                 - 0.492999029596132 z_k1_s0007
                                 - 0.649996639415097 z_k1_s0010
                                 - 1.07499613064353 z_k1_s0011
                                 - 0.286999289145829 z_k1_s0012
                                 - 0.543997982936045 z_k1_s0013
                                 - 0.239999397059252 z_k1_s0014
                                 - 0.729995625437022 z_k1_s0015
                                 - 0.248999282986719 z_k1_s0016
                                 - 1.1029957757512 z_k1_s0017
                                 - 0.0960000102449604 z_k1_s0020
                                 - 0.313000173924659 z_k1_s0021
                                 - 0.0739999718659213 z_k1_s0022
                                 - 0.264999649952384 z_k1_s0023
                                 - 0.0729999753623265 z_k1_s0024
                                 - 0.0859999299090585 z_k1_s0025
                                 - 0.168000104403646 z_k1_s0026
                                 - 0.0779999578803003 z_k1_s0027
                                 - 0.244000230077003 z_k1_s0030
                                 - 0.315999446231675 z_k1_s0031
                                 - 0.146999810281127 z_k1_s0032
                                 - 0.113999904177152 z_k1_s0033
                                 - 0.357999966654313 z_k1_s0034
                                 - 0.161999828546512 z_k1_s0035
                                 - 0.114999972434781 z_k1_s0036
                                 - 0.219999932692639 z_k1_s0037
                                 - 0.177999904015655 z_k1_s0040
                                 - 0.118999766322444 z_k1_s0041
                                 - 0.177999904015655 z_k1_s0042
                                 - 0.254000774611256 z_k1_s0043
                                 - 0.250999172806519 z_k1_s0044
                                 - 0.224000401547234 z_k1_s0045
                                 - 0.542999376966647 z_k1_s0046
                                 - 0.488000555513957 z_k1_s0047
                                 - 0.29999767366112 z_k1_s0050
                                 - 0.122999404759254 z_k1_s0051
                                 - 0.167999055808422 z_k1_s0052
                                 - 0.357998445656119 z_k1_s0053
                                 - 0.0979995322211616 z_k1_s0054
                                 - 0.478997892484165 z_k1_s0055
                                 - 0.362998340484138 z_k1_s0056
                                 - 0.102999427049181 z_k1_s0057
                                 - 0.170000298939394 z_k1_s0060
                                 - 0.0799998903248643 z_k1_s0061
                                 - 0.340999426109922 z_k1_s0062
                                 - 0.154999255505367 z_k1_s0063
                                 - 0.120999727716546 z_k1_s0064
                                 - 0.126999746090863 z_k1_s0065
                                 - 0.160000046649258 z_k1_s0066
                                 - 0.175999918314419 z_k1_s0067
                                 - 0.390998865502826 z_k1_s0070
                                 - 0.857998605508404 z_k1_s0071
                                 - 0.501998003543298 z_k1_s0072
                                 - 0.611998261273881 z_k1_s0073
                                 - 0.263999341757095 z_k1_s0074
                                 - 0.229999532930495 z_k1_s0075
                                 - 0.454999131527557 z_k1_s0076
                                 - 0.94399934867802 z_k1_s0077
                                 - 1.2330048427125 z_k1_s0100
                                 - 1.53000649679736 z_k1_s0101
                                 - 1.2030032417924 z_k1_s0102
                                 - 0.784001200642878 z_k1_s0103
                                 - 0.865003018122886 z_k1_s0104
                                 - 0.730000823990957 z_k1_s0105
                                 - 0.662000813578213 z_k1_s0106
                                 - 1.34299957940062 z_k1_s0107
                                 - 0.353999779164849 z_k1_s0110
                                 - 0.93799766922273 z_k1_s0111
                                 - 0.583000114409739 z_k1_s0112
                                 - 0.362999931991018 z_k1_s0113
                                 - 0.408999947769486 z_k1_s0114
                                 - 1.02099803995332 z_k1_s0115
                                 - 0.403000009885315 z_k1_s0116
                                 - 0.8489998206071 z_k1_s0117
                                 - 0.669000519369335 z_k1_s0120
                                 - 0.850002043443269 z_k1_s0121
                                 - 0.304999632059517 z_k1_s0122
                                 - 0.286999839610025 z_k1_s0123
                                 - 0.902002021630448 z_k1_s0124
                                 - 0.790997668194276 z_k1_s0125
                                 - 0.974001191428416 z_k1_s0126
                                 - 0.42199958298067 z_k1_s0127
                                 - 0.463001983839831 z_k1_s0130
                                 - 0.359001338606724 z_k1_s0131
                                 - 1.26500765507497 z_k1_s0132
                                 - 0.291001187193899 z_k1_s0133
                                 - 0.805004414773074 z_k1_s0134
                                 - 0.579003630849867 z_k1_s0135
                                 - 0.705001089653008 z_k1_s0136
                                 - 0.556003217683722 z_k1_s0137
                                 - 1.47299725041724 z_k1_s0140
                                 - 0.988004017243998 z_k1_s0141
                                 - 0.714000284303605 z_k1_s0142
                                 - 0.859000388699189 z_k1_s0143
                                 - 1.76000101186345 z_k1_s0144
                                 - 0.801000902181353 z_k1_s0145
                                 - 1.36600242582701 z_k1_s0146
                                 - 1.29299756756787 z_k1_s0147
                                 - 0.456000180685334 z_k1_s0150
                                 - 0.856001960761365 z_k1_s0151
                                 - 0.676001515227409 z_k1_s0152
                                 - 0.564001159006224 z_k1_s0153
                                 - 0.294000135205032 z_k1_s0154
                                 - 0.179000067808495 z_k1_s0155
                                 - 0.18400017893451 z_k1_s0156
                                 - 0.346000224414813 z_k1_s0157
                                 - 0.906002730225417 z_k1_s0160
                                 - 0.477998579343528 z_k1_s0161
                                 - 0.275000233768948 z_k1_s0162
                                 - 0.391999775981379 z_k1_s0163
                                 - 0.415001406277798 z_k1_s0164
                                 - 0.206000298484495 z_k1_s0165
                                 - 0.477998579343528 z_k1_s0166
                                 - 0.662000185703794 z_k1_s0167
                                 - 0.697001359851526 z_k1_s0170
                                 - 0.29600109522833 z_k1_s0171
                                 - 0.439000992537797 z_k1_s0172
                                 - 0.2529999804091 z_k1_s0173
                                 - 0.143000160709536 z_k1_s0174
                                 - 0.361000330186991 z_k1_s0175
                                 - 0.212000179312024 z_k1_s0176
                                 - 0.233000013190997 z_k1_s0177
                                 - 0.0670000864428497 z_k1_s1000
                                 - 0.057000032503596 z_k1_s1001
                                 - 0.102000070480126 z_k1_s1002
                                 - 0.0410000398008412 z_k1_s1003
                                 - 0.0320000439055416 z_k1_s1004
                                 - 0.094999956672607 z_k1_s1005
                                 - 0.0390000407129968 z_k1_s1006
                                 - 0.152000047676235 z_k1_s1007
                                 - 0.0390000612865966 z_k1_s1010
                                 - 0.125999910925638 z_k1_s1011
                                 - 0.0830003856101935 z_k1_s1012
                                 - 0.0720003045292943 z_k1_s1013
                                 - 0.0310000136277722 z_k1_s1014
                                 - 0.149000250094962 z_k1_s1015
                                 - 0.0750002135512438 z_k1_s1016
                                 - 0.0470000467453584 z_k1_s1017
                                 - 0.15299992545259 z_k1_s1020
                                 - 0.0570001772594292 z_k1_s1021
                                 - 0.171000423977961 z_k1_s1022
                                 - 0.0490000994263664 z_k1_s1023
                                 - 0.174000507065523 z_k1_s1024
                                 - 0.180000673240646 z_k1_s1025
                                 - 0.113999923317552 z_k1_s1026
                                 - 0.180000673240646 z_k1_s1027
                                 - 0.047000024440939 z_k1_s1030
                                 - 0.0560000356743138 z_k1_s1031
                                 - 0.0350000299964503 z_k1_s1032
                                 - 0.117999791370723 z_k1_s1033
                                 - 0.127999940741213 z_k1_s1034
                                 - 0.0560000356743138 z_k1_s1035
                                 - 0.0980001086300739 z_k1_s1036
                                 - 0.077999809889094 z_k1_s1037
                                 - 0.229999396896528 z_k1_s1040
                                 - 0.156999441673018 z_k1_s1041
                                 - 0.224999226703338 z_k1_s1042
                                 - 0.143999420770234 z_k1_s1043
                                 - 0.0749999179009087 z_k1_s1044
                                 - 0.113000262770254 z_k1_s1045
                                 - 0.060999968359365 z_k1_s1046
                                 - 0.0489999814952196 z_k1_s1047
                                 - 0.304999465991003 z_k1_s1050
                                 - 0.329001229235794 z_k1_s1051
                                 - 0.232000242172878 z_k1_s1052
                                 - 0.182000317002202 z_k1_s1053
                                 - 0.380000775499283 z_k1_s1054
                                 - 0.118000268023505 z_k1_s1055
                                 - 0.500001836710884 z_k1_s1056
                                 - 0.539001793854036 z_k1_s1057
                                 - 0.0520000225997229 z_k1_s1060
                                 - 0.0640000637227763 z_k1_s1061
                                 - 0.0540000294535652 z_k1_s1062
                                 - 0.146000422530396 z_k1_s1063
                                 - 0.0690000419573381 z_k1_s1064
                                 - 0.150999856164344 z_k1_s1065
                                 - 0.201000416510838 z_k1_s1066
                                 - 0.0410000238036344 z_k1_s1067
                                 - 0.143000677022499 z_k1_s1070
                                 - 0.0640002068768223 z_k1_s1071
                                 - 0.11600058234006 z_k1_s1072
                                 - 0.0750002735401491 z_k1_s1073
                                 - 0.113000150484791 z_k1_s1074
                                 - 0.143000677022499 z_k1_s1075
                                 - 0.169000627753181 z_k1_s1076
                                 - 0.125000929902121 z_k1_s1077
                                 - 0.941998544922654 z_k1_s1100
                                 - 0.45999933625514 z_k1_s1101
                                 - 1.07599910379562 z_k1_s1102
                                 - 0.820997242658199 z_k1_s1103
                                 - 0.749000332212891 z_k1_s1104
                                 - 1.45299758540648 z_k1_s1105
                                 - 0.790997416641324 z_k1_s1106
                                 - 0.387999085815842 z_k1_s1107
                                 - 0.587999169701126 z_k1_s1110
                                 - 1.06099806401998 z_k1_s1111
                                 - 0.581998661231613 z_k1_s1112
                                 - 1.08400058365211 z_k1_s1113
                                 - 0.516999428300882 z_k1_s1114
                                 - 0.680998493493586 z_k1_s1115
                                 - 0.525999164106953 z_k1_s1116
                                 - 1.80699852883949 z_k1_s1117
                                 - 0.109999942923756 z_k1_s1120
                                 - 0.344000278597399 z_k1_s1121
                                 - 0.189999520505109 z_k1_s1122
                                 - 0.107999806834391 z_k1_s1123
                                 - 0.17400044298787 z_k1_s1124
                                 - 0.181999981746489 z_k1_s1125
                                 - 0.285000118756695 z_k1_s1126
                                 - 0.282998977068492 z_k1_s1127
                                 - 0.351999316273309 z_k1_s1130
                                 - 0.266000565404289 z_k1_s1131
                                 - 0.14999978638116 z_k1_s1132
                                 - 0.106999856247219 z_k1_s1133
                                 - 0.35399993222309 z_k1_s1134
                                 - 0.266999024381075 z_k1_s1135
                                 - 0.157999846482748 z_k1_s1136
                                 - 0.194999038166209 z_k1_s1137
                                 - 0.156999755564789 z_k1_s1140
                                 - 0.20700051154127 z_k1_s1141
                                 - 0.178000486475126 z_k1_s1142
                                 - 0.0800001071617965 z_k1_s1143
                                 - 0.283000007024662 z_k1_s1144
                                 - 0.110000009547399 z_k1_s1145
                                 - 0.174000150396864 z_k1_s1146
                                 - 0.188999688189451 z_k1_s1147
                                 - 0.984000227824625 z_k1_s1150
                                 - 0.337999916305607 z_k1_s1151
                                 - 0.920001249494606 z_k1_s1152
                                 - 0.354000192088131 z_k1_s1153
                                 - 0.41199979739973 z_k1_s1154
                                 - 0.660998744065068 z_k1_s1155
                                 - 0.732997594686339 z_k1_s1156
                                 - 0.909998753130443 z_k1_s1157
                                 - 0.210000031272531 z_k1_s1160
                                 - 0.449999552941216 z_k1_s1161
                                 - 0.183000010116065 z_k1_s1162
                                 - 0.310999115272796 z_k1_s1163
                                 - 0.25599964971324 z_k1_s1164
                                 - 0.358000316059798 z_k1_s1165
                                 - 0.134999865882365 z_k1_s1166
                                 - 0.496000370881741 z_k1_s1167
                                 - 0.749999180190392 z_k1_s1170
                                 - 0.234999292020822 z_k1_s1171
                                 - 0.231999342371938 z_k1_s1172
                                 - 0.354998454773108 z_k1_s1173
                                 - 0.558999149686523 z_k1_s1174
                                 - 0.399999170502556 z_k1_s1175
                                 - 0.238999519085239 z_k1_s1176
                                 - 0.171999466796555 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.19845 w_k1_ADM_s000
                                 + 0.967307 w_k1_AXP_s000
                                 + 1.29541 w_k1_DIS_s000
                                 + 1.05865 w_k1_FDX_s000
                                 + 1.10012 w_k1_IBM_s000
                                 + 0.932619 w_k1_INTC_s000
                                 + 1.13361 w_k1_KO_s000 + 1.0478 w_k1_MAT_s000
                                 + 1.02957 w_k1_MSFT_s000 + 1.0357 w_k1_PG_s000
                                 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.36893 w_k1_ADM_s000
                                 + 1.25536 w_k1_AXP_s000
                                 + 1.26203 w_k1_DIS_s000
                                 + 1.10973 w_k1_FDX_s000
                                 + 1.28212 w_k1_IBM_s000
                                 + 0.754709 w_k1_INTC_s000
                                 + 1.30893 w_k1_KO_s000 + 2.13943 w_k1_MAT_s000
                                 + 0.980734 w_k1_MSFT_s000
                                 + 1.25689 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.53537 w_k1_ADM_s000
                                 + 1.47685 w_k1_AXP_s000
                                 + 1.44201 w_k1_DIS_s000 + 1.347 w_k1_FDX_s000
                                 + 1.04467 w_k1_IBM_s000
                                 + 1.17091 w_k1_INTC_s000
                                 + 1.42616 w_k1_KO_s000 + 1.63875 w_k1_MAT_s000
                                 + 1.30429 w_k1_MSFT_s000
                                 + 1.28944 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.03473 w_k1_ADM_s000
                                 + 1.15359 w_k1_AXP_s000
                                 + 0.854134 w_k1_DIS_s000
                                 + 0.805754 w_k1_FDX_s000
                                 + 1.23993 w_k1_IBM_s000
                                 + 0.957006 w_k1_INTC_s000
                                 + 1.04209 w_k1_KO_s000 + 1.28204 w_k1_MAT_s000
                                 + 0.8051 w_k1_MSFT_s000
                                 + 0.922144 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.16756 w_k1_ADM_s000
                                 + 1.03311 w_k1_AXP_s000
                                 + 0.797169 w_k1_DIS_s000
                                 + 0.898978 w_k1_FDX_s000
                                 + 0.807123 w_k1_IBM_s000
                                 + 0.816657 w_k1_INTC_s000
                                 + 0.794411 w_k1_KO_s000
                                 + 0.976599 w_k1_MAT_s000
                                 + 0.800983 w_k1_MSFT_s000
                                 + 0.912224 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.996915 w_k1_ADM_s000
                                 + 0.842787 w_k1_AXP_s000
                                 + 1.0409 w_k1_DIS_s000 + 1.0143 w_k1_FDX_s000
                                 + 0.80903 w_k1_IBM_s000
                                 + 0.802414 w_k1_INTC_s000
                                 + 0.86918 w_k1_KO_s000
                                 + 0.779843 w_k1_MAT_s000
                                 + 0.82987 w_k1_MSFT_s000
                                 + 1.01887 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.42434 w_k1_ADM_s000
                                 + 1.2923 w_k1_AXP_s000 + 1.39117 w_k1_DIS_s000
                                 + 1.03399 w_k1_FDX_s000
                                 + 1.85471 w_k1_IBM_s000
                                 + 1.01166 w_k1_INTC_s000
                                 + 1.39245 w_k1_KO_s000 + 2.2417 w_k1_MAT_s000
                                 + 1.03085 w_k1_MSFT_s000
                                 + 1.30119 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.848095 w_k1_ADM_s000
                                 + 0.608626 w_k1_AXP_s000
                                 + 0.844267 w_k1_DIS_s000
                                 + 0.778068 w_k1_FDX_s000
                                 + 0.858412 w_k1_IBM_s000
                                 + 0.803202 w_k1_INTC_s000
                                 + 0.883693 w_k1_KO_s000
                                 + 0.837697 w_k1_MAT_s000
                                 + 0.780013 w_k1_MSFT_s000
                                 + 0.939809 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.918346 w_k1_ADM_s001
                                 + 0.907397 w_k1_AXP_s001
                                 + 0.76727 w_k1_DIS_s001
                                 + 0.855505 w_k1_FDX_s001
                                 + 0.809054 w_k1_IBM_s001
                                 + 0.791746 w_k1_INTC_s001
                                 + 0.75378 w_k1_KO_s001
                                 + 0.865371 w_k1_MAT_s001
                                 + 0.830921 w_k1_MSFT_s001
                                 + 0.960608 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.2726 w_k1_ADM_s001
                                 + 1.24719 w_k1_AXP_s001
                                 + 1.60545 w_k1_DIS_s001
                                 + 1.26621 w_k1_FDX_s001
                                 + 1.04825 w_k1_IBM_s001
                                 + 1.28515 w_k1_INTC_s001
                                 + 1.32545 w_k1_KO_s001 + 1.11182 w_k1_MAT_s001
                                 + 1.35305 w_k1_MSFT_s001
                                 + 1.14004 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.16047 w_k1_ADM_s001
                                 + 0.955306 w_k1_AXP_s001
                                 + 1.53798 w_k1_DIS_s001
                                 + 1.10405 w_k1_FDX_s001
                                 + 0.838121 w_k1_IBM_s001
                                 + 1.17836 w_k1_INTC_s001
                                 + 1.04381 w_k1_KO_s001
                                 + 0.764292 w_k1_MAT_s001
                                 + 1.18336 w_k1_MSFT_s001
                                 + 0.969357 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.19704 w_k1_ADM_s001
                                 + 1.33196 w_k1_AXP_s001
                                 + 1.26079 w_k1_DIS_s001
                                 + 1.05402 w_k1_FDX_s001
                                 + 1.37273 w_k1_IBM_s001
                                 + 1.02852 w_k1_INTC_s001
                                 + 1.37924 w_k1_KO_s001 + 1.81606 w_k1_MAT_s001
                                 + 1.20844 w_k1_MSFT_s001
                                 + 1.27586 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.705879 w_k1_ADM_s001
                                 + 0.807784 w_k1_AXP_s001
                                 + 0.648449 w_k1_DIS_s001
                                 + 0.765447 w_k1_FDX_s001
                                 + 1.16523 w_k1_IBM_s001
                                 + 0.791017 w_k1_INTC_s001
                                 + 0.956633 w_k1_KO_s001
                                 + 1.37249 w_k1_MAT_s001
                                 + 0.821788 w_k1_MSFT_s001
                                 + 0.901466 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.961928 w_k1_ADM_s001
                                 + 0.666039 w_k1_AXP_s001
                                 + 0.839239 w_k1_DIS_s001
                                 + 0.852763 w_k1_FDX_s001
                                 + 0.633805 w_k1_IBM_s001
                                 + 0.803264 w_k1_INTC_s001
                                 + 0.742238 w_k1_KO_s001
                                 + 0.434527 w_k1_MAT_s001
                                 + 0.963117 w_k1_MSFT_s001
                                 + 0.889549 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.0849 w_k1_ADM_s001
                                 + 1.42844 w_k1_AXP_s001
                                 + 0.945754 w_k1_DIS_s001
                                 + 1.02998 w_k1_FDX_s001
                                 + 0.96981 w_k1_IBM_s001
                                 + 0.838117 w_k1_INTC_s001
                                 + 1.00269 w_k1_KO_s001 + 1.50313 w_k1_MAT_s001
                                 + 0.987317 w_k1_MSFT_s001
                                 + 1.07145 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.948105 w_k1_ADM_s001
                                 + 0.875089 w_k1_AXP_s001
                                 + 1.10385 w_k1_DIS_s001
                                 + 0.979812 w_k1_FDX_s001
                                 + 1.06018 w_k1_IBM_s001
                                 + 0.950751 w_k1_INTC_s001
                                 + 1.01821 w_k1_KO_s001 + 1.0753 w_k1_MAT_s001
                                 + 1.00503 w_k1_MSFT_s001
                                 + 0.938652 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.59167 w_k1_ADM_s002
                                 + 1.50722 w_k1_AXP_s002
                                 + 1.21535 w_k1_DIS_s002
                                 + 1.26592 w_k1_FDX_s002
                                 + 1.14444 w_k1_IBM_s002
                                 + 1.06855 w_k1_INTC_s002
                                 + 1.14564 w_k1_KO_s002 + 1.16922 w_k1_MAT_s002
                                 + 1.20661 w_k1_MSFT_s002
                                 + 1.21157 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.820577 w_k1_ADM_s002
                                 + 0.762391 w_k1_AXP_s002
                                 + 0.874123 w_k1_DIS_s002
                                 + 0.85727 w_k1_FDX_s002
                                 + 0.999074 w_k1_IBM_s002
                                 + 0.854226 w_k1_INTC_s002
                                 + 0.908001 w_k1_KO_s002
                                 + 0.962818 w_k1_MAT_s002
                                 + 0.8528 w_k1_MSFT_s002
                                 + 0.924978 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.60679 w_k1_ADM_s002
                                 + 1.22015 w_k1_AXP_s002 + 1.2427 w_k1_DIS_s002
                                 + 1.34211 w_k1_FDX_s002
                                 + 0.773171 w_k1_IBM_s002
                                 + 1.09082 w_k1_INTC_s002
                                 + 0.825695 w_k1_KO_s002
                                 + 0.688636 w_k1_MAT_s002
                                 + 1.20994 w_k1_MSFT_s002
                                 + 1.04228 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.05516 w_k1_ADM_s002
                                 + 1.03391 w_k1_AXP_s002
                                 + 1.30025 w_k1_DIS_s002
                                 + 1.16357 w_k1_FDX_s002
                                 + 1.31059 w_k1_IBM_s002
                                 + 1.42569 w_k1_INTC_s002
                                 + 1.16922 w_k1_KO_s002
                                 + 0.960522 w_k1_MAT_s002
                                 + 1.27839 w_k1_MSFT_s002
                                 + 1.11693 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.693025 w_k1_ADM_s002
                                 + 0.539656 w_k1_AXP_s002
                                 + 0.707267 w_k1_DIS_s002
                                 + 0.775641 w_k1_FDX_s002
                                 + 0.579335 w_k1_IBM_s002
                                 + 0.905509 w_k1_INTC_s002
                                 + 0.690931 w_k1_KO_s002
                                 + 0.356123 w_k1_MAT_s002
                                 + 0.905891 w_k1_MSFT_s002
                                 + 0.843493 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.0135 w_k1_ADM_s002
                                 + 1.16977 w_k1_AXP_s002
                                 + 1.04886 w_k1_DIS_s002
                                 + 0.970836 w_k1_FDX_s002
                                 + 1.87189 w_k1_IBM_s002
                                 + 1.01089 w_k1_INTC_s002
                                 + 1.22416 w_k1_KO_s002 + 1.93401 w_k1_MAT_s002
                                 + 1.10251 w_k1_MSFT_s002
                                 + 1.11764 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.05635 w_k1_ADM_s002
                                 + 1.04771 w_k1_AXP_s002
                                 + 0.910321 w_k1_DIS_s002
                                 + 1.01746 w_k1_FDX_s002
                                 + 0.766423 w_k1_IBM_s002
                                 + 1.01501 w_k1_INTC_s002
                                 + 0.808178 w_k1_KO_s002
                                 + 0.515283 w_k1_MAT_s002
                                 + 0.842518 w_k1_MSFT_s002
                                 + 0.972635 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.24287 w_k1_ADM_s002
                                 + 1.14573 w_k1_AXP_s002
                                 + 0.727382 w_k1_DIS_s002
                                 + 0.94018 w_k1_FDX_s002 + 1.2936 w_k1_IBM_s002
                                 + 0.948693 w_k1_INTC_s002
                                 + 1.04662 w_k1_KO_s002 + 1.56803 w_k1_MAT_s002
                                 + 0.842855 w_k1_MSFT_s002
                                 + 1.10943 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.14309 w_k1_ADM_s003
                                 + 0.894286 w_k1_AXP_s003
                                 + 0.874956 w_k1_DIS_s003
                                 + 0.99203 w_k1_FDX_s003
                                 + 1.00189 w_k1_IBM_s003
                                 + 0.978373 w_k1_INTC_s003
                                 + 0.988877 w_k1_KO_s003
                                 + 0.780294 w_k1_MAT_s003
                                 + 0.894602 w_k1_MSFT_s003
                                 + 1.02622 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.846747 w_k1_ADM_s003
                                 + 0.775444 w_k1_AXP_s003
                                 + 0.71374 w_k1_DIS_s003
                                 + 0.815716 w_k1_FDX_s003
                                 + 0.844782 w_k1_IBM_s003
                                 + 0.725108 w_k1_INTC_s003
                                 + 0.890698 w_k1_KO_s003
                                 + 0.802678 w_k1_MAT_s003
                                 + 0.704995 w_k1_MSFT_s003
                                 + 0.793621 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.41244 w_k1_ADM_s003
                                 + 1.45406 w_k1_AXP_s003
                                 + 1.83608 w_k1_DIS_s003
                                 + 1.39356 w_k1_FDX_s003
                                 + 1.39871 w_k1_IBM_s003
                                 + 1.48367 w_k1_INTC_s003
                                 + 1.47991 w_k1_KO_s003 + 1.43662 w_k1_MAT_s003
                                 + 1.41933 w_k1_MSFT_s003
                                 + 1.28766 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.04854 w_k1_ADM_s003
                                 + 1.1768 w_k1_AXP_s003 + 1.17058 w_k1_DIS_s003
                                 + 1.10142 w_k1_FDX_s003
                                 + 1.55884 w_k1_IBM_s003
                                 + 0.949087 w_k1_INTC_s003
                                 + 1.30429 w_k1_KO_s003 + 1.97873 w_k1_MAT_s003
                                 + 1.02442 w_k1_MSFT_s003
                                 + 1.06098 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.13069 w_k1_ADM_s003
                                 + 1.08419 w_k1_AXP_s003
                                 + 1.16365 w_k1_DIS_s003
                                 + 1.13945 w_k1_FDX_s003
                                 + 0.993079 w_k1_IBM_s003
                                 + 0.865279 w_k1_INTC_s003
                                 + 1.11497 w_k1_KO_s003 + 0.93437 w_k1_MAT_s003
                                 + 1.01383 w_k1_MSFT_s003
                                 + 1.12198 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.777523 w_k1_ADM_s003
                                 + 0.76338 w_k1_AXP_s003
                                 + 0.71512 w_k1_DIS_s003
                                 + 0.962449 w_k1_FDX_s003
                                 + 0.972156 w_k1_IBM_s003
                                 + 0.871397 w_k1_INTC_s003
                                 + 0.960251 w_k1_KO_s003
                                 + 1.22359 w_k1_MAT_s003
                                 + 0.872198 w_k1_MSFT_s003
                                 + 0.882596 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.74009 w_k1_ADM_s003
                                 + 1.01238 w_k1_AXP_s003
                                 + 1.15039 w_k1_DIS_s003
                                 + 1.30949 w_k1_FDX_s003
                                 + 0.843741 w_k1_IBM_s003
                                 + 0.990596 w_k1_INTC_s003
                                 + 1.10142 w_k1_KO_s003
                                 + 0.799023 w_k1_MAT_s003
                                 + 1.06771 w_k1_MSFT_s003
                                 + 1.06918 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.887114 w_k1_ADM_s003
                                 + 0.87081 w_k1_AXP_s003
                                 + 0.944723 w_k1_DIS_s003
                                 + 0.936303 w_k1_FDX_s003
                                 + 1.44578 w_k1_IBM_s003
                                 + 1.12722 w_k1_INTC_s003
                                 + 1.11411 w_k1_KO_s003 + 1.37766 w_k1_MAT_s003
                                 + 0.949084 w_k1_MSFT_s003
                                 + 0.90946 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.4797 w_k1_ADM_s004
                                 + 1.47634 w_k1_AXP_s004
                                 + 1.60305 w_k1_DIS_s004
                                 + 1.30488 w_k1_FDX_s004
                                 + 1.30711 w_k1_IBM_s004
                                 + 1.26424 w_k1_INTC_s004
                                 + 1.14844 w_k1_KO_s004 + 1.0698 w_k1_MAT_s004
                                 + 1.16187 w_k1_MSFT_s004
                                 + 1.30964 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.32856 w_k1_ADM_s004
                                 + 1.31414 w_k1_AXP_s004
                                 + 1.28231 w_k1_DIS_s004
                                 + 1.35371 w_k1_FDX_s004
                                 + 0.904495 w_k1_IBM_s004
                                 + 1.28026 w_k1_INTC_s004
                                 + 1.00615 w_k1_KO_s004
                                 + 0.986234 w_k1_MAT_s004
                                 + 1.12742 w_k1_MSFT_s004
                                 + 1.11395 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.08108 w_k1_ADM_s004
                                 + 0.864423 w_k1_AXP_s004
                                 + 0.781065 w_k1_DIS_s004
                                 + 0.911695 w_k1_FDX_s004
                                 + 0.752417 w_k1_IBM_s004
                                 + 0.801282 w_k1_INTC_s004
                                 + 0.863001 w_k1_KO_s004
                                 + 0.912215 w_k1_MAT_s004
                                 + 0.78501 w_k1_MSFT_s004
                                 + 1.04919 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.971675 w_k1_ADM_s004
                                 + 1.32048 w_k1_AXP_s004
                                 + 1.02523 w_k1_DIS_s004 + 1.1448 w_k1_FDX_s004
                                 + 1.08742 w_k1_IBM_s004
                                 + 1.06338 w_k1_INTC_s004
                                 + 0.969723 w_k1_KO_s004
                                 + 1.05208 w_k1_MAT_s004
                                 + 0.811901 w_k1_MSFT_s004
                                 + 1.01753 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.1343 w_k1_ADM_s004
                                 + 1.29056 w_k1_AXP_s004
                                 + 1.09295 w_k1_DIS_s004
                                 + 1.13205 w_k1_FDX_s004
                                 + 1.65312 w_k1_IBM_s004
                                 + 1.12954 w_k1_INTC_s004
                                 + 1.33505 w_k1_KO_s004 + 1.39147 w_k1_MAT_s004
                                 + 0.912464 w_k1_MSFT_s004
                                 + 1.06502 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.935763 w_k1_ADM_s004
                                 + 1.0817 w_k1_AXP_s004 + 1.12066 w_k1_DIS_s004
                                 + 1.06203 w_k1_FDX_s004
                                 + 0.871263 w_k1_IBM_s004
                                 + 1.09713 w_k1_INTC_s004
                                 + 0.91125 w_k1_KO_s004
                                 + 0.640873 w_k1_MAT_s004
                                 + 1.02153 w_k1_MSFT_s004
                                 + 1.03368 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.837427 w_k1_ADM_s004
                                 + 0.75256 w_k1_AXP_s004
                                 + 0.805914 w_k1_DIS_s004
                                 + 0.831867 w_k1_FDX_s004
                                 + 1.07355 w_k1_IBM_s004
                                 + 0.850907 w_k1_INTC_s004
                                 + 1.0509 w_k1_KO_s004 + 0.977093 w_k1_MAT_s004
                                 + 0.926442 w_k1_MSFT_s004
                                 + 0.994584 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.895221 w_k1_ADM_s004
                                 + 0.801184 w_k1_AXP_s004
                                 + 0.851883 w_k1_DIS_s004
                                 + 0.939452 w_k1_FDX_s004
                                 + 0.708906 w_k1_IBM_s004
                                 + 0.890927 w_k1_INTC_s004
                                 + 0.822242 w_k1_KO_s004
                                 + 0.666068 w_k1_MAT_s004
                                 + 0.823342 w_k1_MSFT_s004
                                 + 0.914865 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.972243 w_k1_ADM_s005
                                 + 1.49018 w_k1_AXP_s005
                                 + 1.71113 w_k1_DIS_s005
                                 + 1.23906 w_k1_FDX_s005 + 1.3562 w_k1_IBM_s005
                                 + 1.12239 w_k1_INTC_s005
                                 + 1.60116 w_k1_KO_s005 + 1.63083 w_k1_MAT_s005
                                 + 1.14328 w_k1_MSFT_s005
                                 + 1.31234 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.996477 w_k1_ADM_s005
                                 + 1.17563 w_k1_AXP_s005
                                 + 0.944085 w_k1_DIS_s005
                                 + 0.940628 w_k1_FDX_s005
                                 + 1.18027 w_k1_IBM_s005
                                 + 1.08656 w_k1_INTC_s005
                                 + 1.17218 w_k1_KO_s005 + 1.56919 w_k1_MAT_s005
                                 + 0.935221 w_k1_MSFT_s005
                                 + 0.99364 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.032 w_k1_ADM_s005
                                 + 0.986083 w_k1_AXP_s005
                                 + 1.12878 w_k1_DIS_s005
                                 + 0.888621 w_k1_FDX_s005
                                 + 1.68783 w_k1_IBM_s005
                                 + 1.0342 w_k1_INTC_s005 + 1.38012 w_k1_KO_s005
                                 + 1.32598 w_k1_MAT_s005
                                 + 0.98019 w_k1_MSFT_s005
                                 + 1.14661 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.804813 w_k1_ADM_s005
                                 + 0.956378 w_k1_AXP_s005
                                 + 0.885805 w_k1_DIS_s005
                                 + 0.951618 w_k1_FDX_s005
                                 + 0.762464 w_k1_IBM_s005
                                 + 1.08191 w_k1_INTC_s005
                                 + 0.880229 w_k1_KO_s005
                                 + 0.817095 w_k1_MAT_s005
                                 + 0.981395 w_k1_MSFT_s005
                                 + 1.01293 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.95354 w_k1_ADM_s005
                                 + 1.00793 w_k1_AXP_s005
                                 + 1.19985 w_k1_DIS_s005
                                 + 0.989476 w_k1_FDX_s005
                                 + 1.22267 w_k1_IBM_s005
                                 + 1.05317 w_k1_INTC_s005
                                 + 1.22121 w_k1_KO_s005 + 1.05845 w_k1_MAT_s005
                                 + 1.13415 w_k1_MSFT_s005
                                 + 1.06277 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.736111 w_k1_ADM_s005
                                 + 0.64301 w_k1_AXP_s005
                                 + 0.737076 w_k1_DIS_s005
                                 + 0.776399 w_k1_FDX_s005
                                 + 0.833247 w_k1_IBM_s005
                                 + 0.833829 w_k1_INTC_s005
                                 + 0.855143 w_k1_KO_s005 + 0.8939 w_k1_MAT_s005
                                 + 0.856754 w_k1_MSFT_s005
                                 + 0.946246 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.881928 w_k1_ADM_s005
                                 + 0.749674 w_k1_AXP_s005
                                 + 1.13358 w_k1_DIS_s005
                                 + 1.00418 w_k1_FDX_s005
                                 + 0.752571 w_k1_IBM_s005
                                 + 0.859697 w_k1_INTC_s005
                                 + 1.05902 w_k1_KO_s005
                                 + 0.732106 w_k1_MAT_s005
                                 + 0.967717 w_k1_MSFT_s005
                                 + 1.07658 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.14661 w_k1_ADM_s005
                                 + 1.89654 w_k1_AXP_s005
                                 + 1.39132 w_k1_DIS_s005
                                 + 1.26012 w_k1_FDX_s005
                                 + 1.00998 w_k1_IBM_s005
                                 + 0.940718 w_k1_INTC_s005
                                 + 1.19748 w_k1_KO_s005 + 1.51668 w_k1_MAT_s005
                                 + 0.944048 w_k1_MSFT_s005
                                 + 1.13517 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.23757 w_k1_ADM_s006
                                 + 1.01715 w_k1_AXP_s006 + 1.1327 w_k1_DIS_s006
                                 + 1.16973 w_k1_FDX_s006
                                 + 0.888327 w_k1_IBM_s006
                                 + 1.04092 w_k1_INTC_s006
                                 + 0.920602 w_k1_KO_s006
                                 + 0.505041 w_k1_MAT_s006
                                 + 1.00851 w_k1_MSFT_s006
                                 + 1.00477 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.61727 w_k1_ADM_s006
                                 + 1.83148 w_k1_AXP_s006
                                 + 1.97007 w_k1_DIS_s006
                                 + 1.65936 w_k1_FDX_s006
                                 + 1.33806 w_k1_IBM_s006
                                 + 1.16635 w_k1_INTC_s006
                                 + 1.46036 w_k1_KO_s006 + 1.46778 w_k1_MAT_s006
                                 + 1.15384 w_k1_MSFT_s006
                                 + 1.38337 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.03762 w_k1_ADM_s006
                                 + 0.758153 w_k1_AXP_s006
                                 + 1.10682 w_k1_DIS_s006
                                 + 0.975672 w_k1_FDX_s006
                                 + 0.982944 w_k1_IBM_s006
                                 + 1.03478 w_k1_INTC_s006
                                 + 0.995278 w_k1_KO_s006
                                 + 0.576742 w_k1_MAT_s006
                                 + 0.965448 w_k1_MSFT_s006
                                 + 1.01853 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.843081 w_k1_ADM_s006
                                 + 0.736064 w_k1_AXP_s006
                                 + 0.782812 w_k1_DIS_s006
                                 + 0.826874 w_k1_FDX_s006
                                 + 1.04662 w_k1_IBM_s006
                                 + 0.843477 w_k1_INTC_s006
                                 + 0.928984 w_k1_KO_s006 + 1.0208 w_k1_MAT_s006
                                 + 0.816223 w_k1_MSFT_s006
                                 + 0.973502 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.1908 w_k1_ADM_s006
                                 + 1.3901 w_k1_AXP_s006 + 1.24036 w_k1_DIS_s006
                                 + 1.24953 w_k1_FDX_s006
                                 + 0.939106 w_k1_IBM_s006
                                 + 1.24109 w_k1_INTC_s006
                                 + 1.06898 w_k1_KO_s006 + 1.15544 w_k1_MAT_s006
                                 + 1.01192 w_k1_MSFT_s006 + 1.2274 w_k1_PG_s006
                                 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.29257 w_k1_ADM_s006
                                 + 1.23557 w_k1_AXP_s006
                                 + 1.45615 w_k1_DIS_s006
                                 + 1.20605 w_k1_FDX_s006
                                 + 1.37359 w_k1_IBM_s006
                                 + 1.13216 w_k1_INTC_s006
                                 + 1.39044 w_k1_KO_s006 + 1.2633 w_k1_MAT_s006
                                 + 1.27397 w_k1_MSFT_s006
                                 + 1.26209 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.33643 w_k1_ADM_s006
                                 + 1.12874 w_k1_AXP_s006
                                 + 1.00709 w_k1_DIS_s006 + 1.0549 w_k1_FDX_s006
                                 + 1.21758 w_k1_IBM_s006
                                 + 0.953847 w_k1_INTC_s006
                                 + 1.16701 w_k1_KO_s006 + 1.407 w_k1_MAT_s006
                                 + 1.0795 w_k1_MSFT_s006 + 1.10838 w_k1_PG_s006
                                 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.726617 w_k1_ADM_s006
                                 + 0.698547 w_k1_AXP_s006
                                 + 0.582822 w_k1_DIS_s006
                                 + 0.805222 w_k1_FDX_s006
                                 + 0.677673 w_k1_IBM_s006
                                 + 0.783162 w_k1_INTC_s006
                                 + 0.690423 w_k1_KO_s006
                                 + 1.03562 w_k1_MAT_s006
                                 + 0.757035 w_k1_MSFT_s006
                                 + 0.8835 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.767601 w_k1_ADM_s007
                                 + 0.710416 w_k1_AXP_s007
                                 + 0.654112 w_k1_DIS_s007
                                 + 0.753979 w_k1_FDX_s007
                                 + 0.675288 w_k1_IBM_s007
                                 + 0.884709 w_k1_INTC_s007
                                 + 0.763884 w_k1_KO_s007
                                 + 0.653222 w_k1_MAT_s007
                                 + 0.782149 w_k1_MSFT_s007
                                 + 0.82285 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.854118 w_k1_ADM_s007
                                 + 0.863616 w_k1_AXP_s007
                                 + 0.996095 w_k1_DIS_s007
                                 + 0.959511 w_k1_FDX_s007
                                 + 0.729964 w_k1_IBM_s007
                                 + 1.06419 w_k1_INTC_s007
                                 + 0.955885 w_k1_KO_s007
                                 + 0.735622 w_k1_MAT_s007
                                 + 1.06026 w_k1_MSFT_s007
                                 + 0.97168 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.07806 w_k1_ADM_s007
                                 + 0.906142 w_k1_AXP_s007
                                 + 0.882276 w_k1_DIS_s007
                                 + 0.836013 w_k1_FDX_s007
                                 + 1.28184 w_k1_IBM_s007
                                 + 0.817116 w_k1_INTC_s007
                                 + 1.15415 w_k1_KO_s007 + 1.49703 w_k1_MAT_s007
                                 + 0.86802 w_k1_MSFT_s007
                                 + 1.05986 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.26745 w_k1_ADM_s007
                                 + 1.42259 w_k1_AXP_s007
                                 + 1.00714 w_k1_DIS_s007
                                 + 1.05946 w_k1_FDX_s007
                                 + 1.17273 w_k1_IBM_s007
                                 + 1.20814 w_k1_INTC_s007
                                 + 1.08812 w_k1_KO_s007 + 1.21585 w_k1_MAT_s007
                                 + 1.08798 w_k1_MSFT_s007
                                 + 1.21262 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.911947 w_k1_ADM_s007
                                 + 1.18631 w_k1_AXP_s007 + 1.7023 w_k1_DIS_s007
                                 + 1.17937 w_k1_FDX_s007
                                 + 1.96013 w_k1_IBM_s007
                                 + 1.14299 w_k1_INTC_s007
                                 + 1.94204 w_k1_KO_s007 + 2.13239 w_k1_MAT_s007
                                 + 1.4106 w_k1_MSFT_s007 + 1.47298 w_k1_PG_s007
                                 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.661663 w_k1_ADM_s007
                                 + 0.641069 w_k1_AXP_s007
                                 + 0.537479 w_k1_DIS_s007
                                 + 0.646881 w_k1_FDX_s007
                                 + 0.788606 w_k1_IBM_s007
                                 + 0.736356 w_k1_INTC_s007
                                 + 0.729043 w_k1_KO_s007
                                 + 0.776814 w_k1_MAT_s007
                                 + 0.744337 w_k1_MSFT_s007
                                 + 0.907223 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.741917 w_k1_ADM_s007
                                 + 0.854691 w_k1_AXP_s007
                                 + 0.768436 w_k1_DIS_s007
                                 + 0.798454 w_k1_FDX_s007
                                 + 0.947262 w_k1_IBM_s007
                                 + 0.929109 w_k1_INTC_s007
                                 + 1.01932 w_k1_KO_s007 + 0.85793 w_k1_MAT_s007
                                 + 0.812455 w_k1_MSFT_s007
                                 + 0.97834 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.44214 w_k1_ADM_s007
                                 + 1.43064 w_k1_AXP_s007
                                 + 1.57891 w_k1_DIS_s007 + 1.3203 w_k1_FDX_s007
                                 + 1.13645 w_k1_IBM_s007
                                 + 1.27507 w_k1_INTC_s007
                                 + 1.35004 w_k1_KO_s007 + 1.61488 w_k1_MAT_s007
                                 + 1.28442 w_k1_MSFT_s007
                                 + 1.35614 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.07287 w_k1_ADM_s010
                                 + 0.806882 w_k1_AXP_s010
                                 + 1.17891 w_k1_DIS_s010
                                 + 1.10471 w_k1_FDX_s010
                                 + 0.782614 w_k1_IBM_s010
                                 + 1.01303 w_k1_INTC_s010 + 1.0512 w_k1_KO_s010
                                 + 0.635318 w_k1_MAT_s010
                                 + 1.24451 w_k1_MSFT_s010
                                 + 1.07239 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.27589 w_k1_ADM_s010
                                 + 1.39512 w_k1_AXP_s010
                                 + 1.23968 w_k1_DIS_s010
                                 + 1.18291 w_k1_FDX_s010
                                 + 1.02716 w_k1_IBM_s010
                                 + 1.0461 w_k1_INTC_s010 + 1.16927 w_k1_KO_s010
                                 + 1.38804 w_k1_MAT_s010
                                 + 1.16416 w_k1_MSFT_s010
                                 + 1.09988 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.757095 w_k1_ADM_s010
                                 + 0.96344 w_k1_AXP_s010
                                 + 0.78678 w_k1_DIS_s010
                                 + 0.877976 w_k1_FDX_s010
                                 + 0.802799 w_k1_IBM_s010
                                 + 1.02445 w_k1_INTC_s010
                                 + 0.822318 w_k1_KO_s010
                                 + 0.756989 w_k1_MAT_s010
                                 + 0.887788 w_k1_MSFT_s010
                                 + 0.793319 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.00051 w_k1_ADM_s010
                                 + 1.0722 w_k1_AXP_s010 + 1.14329 w_k1_DIS_s010
                                 + 0.982435 w_k1_FDX_s010
                                 + 1.63172 w_k1_IBM_s010
                                 + 1.23464 w_k1_INTC_s010
                                 + 1.40258 w_k1_KO_s010 + 1.08991 w_k1_MAT_s010
                                 + 0.990117 w_k1_MSFT_s010
                                 + 1.06442 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.35595 w_k1_ADM_s010
                                 + 1.38405 w_k1_AXP_s010
                                 + 1.81317 w_k1_DIS_s010
                                 + 1.30511 w_k1_FDX_s010
                                 + 1.51545 w_k1_IBM_s010
                                 + 1.40487 w_k1_INTC_s010
                                 + 1.43496 w_k1_KO_s010
                                 + 0.943914 w_k1_MAT_s010
                                 + 1.39919 w_k1_MSFT_s010
                                 + 1.09495 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.787553 w_k1_ADM_s010
                                 + 0.793586 w_k1_AXP_s010
                                 + 1.01474 w_k1_DIS_s010
                                 + 0.895293 w_k1_FDX_s010
                                 + 1.11295 w_k1_IBM_s010
                                 + 0.814445 w_k1_INTC_s010
                                 + 1.14635 w_k1_KO_s010
                                 + 0.863285 w_k1_MAT_s010
                                 + 0.969623 w_k1_MSFT_s010
                                 + 0.955886 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.805667 w_k1_ADM_s010
                                 + 0.720107 w_k1_AXP_s010
                                 + 0.67783 w_k1_DIS_s010
                                 + 0.712978 w_k1_FDX_s010
                                 + 1.03904 w_k1_IBM_s010
                                 + 0.900075 w_k1_INTC_s010
                                 + 0.885021 w_k1_KO_s010
                                 + 0.684906 w_k1_MAT_s010
                                 + 0.794086 w_k1_MSFT_s010
                                 + 0.823183 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.924911 w_k1_ADM_s010
                                 + 0.715817 w_k1_AXP_s010
                                 + 0.788685 w_k1_DIS_s010
                                 + 0.861484 w_k1_FDX_s010
                                 + 0.625386 w_k1_IBM_s010
                                 + 0.992775 w_k1_INTC_s010
                                 + 0.840734 w_k1_KO_s010
                                 + 0.616667 w_k1_MAT_s010
                                 + 1.05053 w_k1_MSFT_s010
                                 + 0.864727 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.03786 w_k1_ADM_s011
                                 + 1.10894 w_k1_AXP_s011
                                 + 1.17155 w_k1_DIS_s011
                                 + 0.927631 w_k1_FDX_s011
                                 + 1.59545 w_k1_IBM_s011
                                 + 1.00294 w_k1_INTC_s011
                                 + 1.30335 w_k1_KO_s011 + 1.37744 w_k1_MAT_s011
                                 + 1.01059 w_k1_MSFT_s011
                                 + 1.19812 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.02357 w_k1_ADM_s011
                                 + 1.05527 w_k1_AXP_s011 + 1.0439 w_k1_DIS_s011
                                 + 0.981904 w_k1_FDX_s011
                                 + 1.1695 w_k1_IBM_s011 + 0.9804 w_k1_INTC_s011
                                 + 1.18097 w_k1_KO_s011 + 1.16989 w_k1_MAT_s011
                                 + 1.05791 w_k1_MSFT_s011
                                 + 1.07147 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.05788 w_k1_ADM_s011
                                 + 1.51165 w_k1_AXP_s011
                                 + 1.16031 w_k1_DIS_s011
                                 + 1.21446 w_k1_FDX_s011
                                 + 1.05477 w_k1_IBM_s011
                                 + 1.15731 w_k1_INTC_s011
                                 + 1.25694 w_k1_KO_s011 + 1.30165 w_k1_MAT_s011
                                 + 1.0714 w_k1_MSFT_s011 + 1.11004 w_k1_PG_s011
                                 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.745097 w_k1_ADM_s011
                                 + 0.768532 w_k1_AXP_s011
                                 + 0.670111 w_k1_DIS_s011
                                 + 0.854262 w_k1_FDX_s011
                                 + 0.580774 w_k1_IBM_s011
                                 + 0.820312 w_k1_INTC_s011
                                 + 0.669095 w_k1_KO_s011
                                 + 0.466828 w_k1_MAT_s011
                                 + 0.708799 w_k1_MSFT_s011
                                 + 0.833838 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.35759 w_k1_ADM_s011
                                 + 1.60806 w_k1_AXP_s011
                                 + 1.82788 w_k1_DIS_s011
                                 + 1.28843 w_k1_FDX_s011
                                 + 1.58043 w_k1_IBM_s011
                                 + 1.40633 w_k1_INTC_s011
                                 + 1.60762 w_k1_KO_s011 + 1.32 w_k1_MAT_s011
                                 + 1.40813 w_k1_MSFT_s011 + 1.2791 w_k1_PG_s011
                                 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.694603 w_k1_ADM_s011
                                 + 0.695067 w_k1_AXP_s011
                                 + 0.771627 w_k1_DIS_s011
                                 + 0.809648 w_k1_FDX_s011
                                 + 0.879039 w_k1_IBM_s011
                                 + 0.848748 w_k1_INTC_s011
                                 + 0.980837 w_k1_KO_s011
                                 + 0.799966 w_k1_MAT_s011
                                 + 0.895105 w_k1_MSFT_s011
                                 + 0.964558 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.926441 w_k1_ADM_s011
                                 + 1.13072 w_k1_AXP_s011
                                 + 1.19904 w_k1_DIS_s011
                                 + 1.28119 w_k1_FDX_s011
                                 + 0.822169 w_k1_IBM_s011
                                 + 1.42682 w_k1_INTC_s011
                                 + 1.06809 w_k1_KO_s011
                                 + 0.819292 w_k1_MAT_s011
                                 + 1.18433 w_k1_MSFT_s011
                                 + 0.998637 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.99183 w_k1_ADM_s011
                                 + 1.05882 w_k1_AXP_s011
                                 + 0.840633 w_k1_DIS_s011
                                 + 0.995446 w_k1_FDX_s011
                                 + 0.783147 w_k1_IBM_s011
                                 + 0.982854 w_k1_INTC_s011
                                 + 0.884016 w_k1_KO_s011
                                 + 1.06737 w_k1_MAT_s011
                                 + 0.911712 w_k1_MSFT_s011
                                 + 0.946065 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.07732 w_k1_ADM_s012
                                 + 0.903439 w_k1_AXP_s012
                                 + 0.882677 w_k1_DIS_s012
                                 + 1.0142 w_k1_FDX_s012
                                 + 0.768901 w_k1_IBM_s012
                                 + 0.989659 w_k1_INTC_s012
                                 + 0.881779 w_k1_KO_s012
                                 + 0.884442 w_k1_MAT_s012
                                 + 1.11782 w_k1_MSFT_s012
                                 + 0.971482 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.922852 w_k1_ADM_s012
                                 + 0.957632 w_k1_AXP_s012
                                 + 0.929115 w_k1_DIS_s012
                                 + 0.899096 w_k1_FDX_s012
                                 + 1.23635 w_k1_IBM_s012
                                 + 0.88683 w_k1_INTC_s012
                                 + 1.16564 w_k1_KO_s012 + 1.25469 w_k1_MAT_s012
                                 + 0.971541 w_k1_MSFT_s012
                                 + 1.10031 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.715429 w_k1_ADM_s012
                                 + 0.600362 w_k1_AXP_s012
                                 + 0.592074 w_k1_DIS_s012
                                 + 0.705473 w_k1_FDX_s012
                                 + 0.657363 w_k1_IBM_s012
                                 + 0.719213 w_k1_INTC_s012
                                 + 0.709569 w_k1_KO_s012
                                 + 0.695148 w_k1_MAT_s012
                                 + 0.798444 w_k1_MSFT_s012
                                 + 0.790122 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.84766 w_k1_ADM_s012
                                 + 0.775532 w_k1_AXP_s012
                                 + 0.766404 w_k1_DIS_s012
                                 + 0.747424 w_k1_FDX_s012
                                 + 0.994619 w_k1_IBM_s012
                                 + 0.824689 w_k1_INTC_s012
                                 + 0.917897 w_k1_KO_s012
                                 + 0.746736 w_k1_MAT_s012
                                 + 0.820919 w_k1_MSFT_s012
                                 + 0.966323 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.01042 w_k1_ADM_s012
                                 + 1.04934 w_k1_AXP_s012
                                 + 0.793597 w_k1_DIS_s012
                                 + 0.882586 w_k1_FDX_s012
                                 + 0.925454 w_k1_IBM_s012
                                 + 0.868095 w_k1_INTC_s012
                                 + 0.979005 w_k1_KO_s012 + 1.0342 w_k1_MAT_s012
                                 + 0.942024 w_k1_MSFT_s012
                                 + 0.994055 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.20688 w_k1_ADM_s012
                                 + 1.63198 w_k1_AXP_s012
                                 + 1.38305 w_k1_DIS_s012
                                 + 1.12892 w_k1_FDX_s012
                                 + 1.58488 w_k1_IBM_s012
                                 + 1.10229 w_k1_INTC_s012
                                 + 1.57039 w_k1_KO_s012 + 1.7962 w_k1_MAT_s012
                                 + 1.05033 w_k1_MSFT_s012
                                 + 1.21196 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.19953 w_k1_ADM_s012
                                 + 1.51953 w_k1_AXP_s012
                                 + 1.23727 w_k1_DIS_s012
                                 + 1.24113 w_k1_FDX_s012
                                 + 1.02592 w_k1_IBM_s012
                                 + 1.08344 w_k1_INTC_s012 + 1.1468 w_k1_KO_s012
                                 + 1.48214 w_k1_MAT_s012
                                 + 1.35562 w_k1_MSFT_s012
                                 + 1.18584 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.11414 w_k1_ADM_s012
                                 + 0.942276 w_k1_AXP_s012
                                 + 0.683731 w_k1_DIS_s012
                                 + 0.883074 w_k1_FDX_s012
                                 + 0.659067 w_k1_IBM_s012
                                 + 0.789038 w_k1_INTC_s012
                                 + 0.746301 w_k1_KO_s012
                                 + 0.903582 w_k1_MAT_s012
                                 + 0.821577 w_k1_MSFT_s012
                                 + 0.868163 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.14976 w_k1_ADM_s013
                                 + 0.971101 w_k1_AXP_s013
                                 + 1.12029 w_k1_DIS_s013
                                 + 0.990145 w_k1_FDX_s013
                                 + 1.34076 w_k1_IBM_s013
                                 + 0.919081 w_k1_INTC_s013
                                 + 1.34273 w_k1_KO_s013 + 1.31917 w_k1_MAT_s013
                                 + 1.18583 w_k1_MSFT_s013
                                 + 1.14529 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.34329 w_k1_ADM_s013
                                 + 1.57406 w_k1_AXP_s013
                                 + 1.64702 w_k1_DIS_s013
                                 + 1.35639 w_k1_FDX_s013
                                 + 1.15476 w_k1_IBM_s013
                                 + 1.28508 w_k1_INTC_s013
                                 + 1.31076 w_k1_KO_s013 + 1.12799 w_k1_MAT_s013
                                 + 1.38708 w_k1_MSFT_s013
                                 + 1.12022 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.859071 w_k1_ADM_s013
                                 + 0.692293 w_k1_AXP_s013
                                 + 0.652492 w_k1_DIS_s013
                                 + 0.775925 w_k1_FDX_s013
                                 + 0.815611 w_k1_IBM_s013
                                 + 0.734424 w_k1_INTC_s013
                                 + 0.678876 w_k1_KO_s013
                                 + 0.736645 w_k1_MAT_s013
                                 + 0.843064 w_k1_MSFT_s013
                                 + 0.810166 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.50267 w_k1_ADM_s013
                                 + 1.88433 w_k1_AXP_s013
                                 + 1.85229 w_k1_DIS_s013
                                 + 1.62003 w_k1_FDX_s013
                                 + 1.20891 w_k1_IBM_s013
                                 + 1.67093 w_k1_INTC_s013
                                 + 1.40566 w_k1_KO_s013 + 1.25233 w_k1_MAT_s013
                                 + 1.54775 w_k1_MSFT_s013
                                 + 1.28555 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.921811 w_k1_ADM_s013
                                 + 0.858151 w_k1_AXP_s013
                                 + 0.964999 w_k1_DIS_s013
                                 + 1.02032 w_k1_FDX_s013
                                 + 0.866282 w_k1_IBM_s013
                                 + 1.16309 w_k1_INTC_s013
                                 + 1.07546 w_k1_KO_s013 + 1.09531 w_k1_MAT_s013
                                 + 1.10599 w_k1_MSFT_s013
                                 + 0.937448 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.11831 w_k1_ADM_s013
                                 + 1.40462 w_k1_AXP_s013
                                 + 1.05076 w_k1_DIS_s013
                                 + 1.14208 w_k1_FDX_s013
                                 + 1.29439 w_k1_IBM_s013
                                 + 1.18967 w_k1_INTC_s013
                                 + 1.09011 w_k1_KO_s013 + 1.05467 w_k1_MAT_s013
                                 + 1.02446 w_k1_MSFT_s013
                                 + 1.03558 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.18911 w_k1_ADM_s013
                                 + 1.24195 w_k1_AXP_s013
                                 + 1.19652 w_k1_DIS_s013
                                 + 1.22644 w_k1_FDX_s013
                                 + 0.898797 w_k1_IBM_s013
                                 + 1.05119 w_k1_INTC_s013
                                 + 0.973646 w_k1_KO_s013 + 1.0176 w_k1_MAT_s013
                                 + 1.16182 w_k1_MSFT_s013
                                 + 1.10797 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.957221 w_k1_ADM_s013
                                 + 0.90374 w_k1_AXP_s013
                                 + 0.798587 w_k1_DIS_s013
                                 + 0.841768 w_k1_FDX_s013
                                 + 0.955781 w_k1_IBM_s013
                                 + 0.868388 w_k1_INTC_s013
                                 + 0.989846 w_k1_KO_s013
                                 + 1.15626 w_k1_MAT_s013
                                 + 0.985108 w_k1_MSFT_s013
                                 + 0.993923 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.19213 w_k1_ADM_s014
                                 + 1.23108 w_k1_AXP_s014
                                 + 1.25525 w_k1_DIS_s014
                                 + 1.20035 w_k1_FDX_s014
                                 + 1.02695 w_k1_IBM_s014
                                 + 1.37727 w_k1_INTC_s014
                                 + 1.01169 w_k1_KO_s014 + 1.18731 w_k1_MAT_s014
                                 + 1.2018 w_k1_MSFT_s014 + 1.01401 w_k1_PG_s014
                                 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.765126 w_k1_ADM_s014
                                 + 0.683076 w_k1_AXP_s014
                                 + 0.530078 w_k1_DIS_s014
                                 + 0.733388 w_k1_FDX_s014
                                 + 0.57698 w_k1_IBM_s014
                                 + 0.943948 w_k1_INTC_s014
                                 + 0.59431 w_k1_KO_s014
                                 + 0.674834 w_k1_MAT_s014
                                 + 0.663598 w_k1_MSFT_s014
                                 + 0.758939 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.02969 w_k1_ADM_s014
                                 + 0.906081 w_k1_AXP_s014
                                 + 0.915857 w_k1_DIS_s014
                                 + 0.944475 w_k1_FDX_s014
                                 + 1.01429 w_k1_IBM_s014
                                 + 1.20578 w_k1_INTC_s014
                                 + 1.02091 w_k1_KO_s014
                                 + 0.938971 w_k1_MAT_s014
                                 + 1.00186 w_k1_MSFT_s014
                                 + 1.02518 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.27217 w_k1_ADM_s014
                                 + 1.50776 w_k1_AXP_s014
                                 + 1.50586 w_k1_DIS_s014 + 1.4348 w_k1_FDX_s014
                                 + 1.16981 w_k1_IBM_s014
                                 + 1.47242 w_k1_INTC_s014
                                 + 1.22271 w_k1_KO_s014 + 1.67827 w_k1_MAT_s014
                                 + 1.33912 w_k1_MSFT_s014
                                 + 1.21653 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.05042 w_k1_ADM_s014
                                 + 0.882707 w_k1_AXP_s014
                                 + 0.828376 w_k1_DIS_s014
                                 + 0.857894 w_k1_FDX_s014
                                 + 1.06451 w_k1_IBM_s014
                                 + 1.00092 w_k1_INTC_s014
                                 + 0.951025 w_k1_KO_s014
                                 + 0.93316 w_k1_MAT_s014
                                 + 0.886935 w_k1_MSFT_s014
                                 + 0.896079 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.11857 w_k1_ADM_s014
                                 + 1.0163 w_k1_AXP_s014 + 1.06451 w_k1_DIS_s014
                                 + 1.05257 w_k1_FDX_s014
                                 + 0.844012 w_k1_IBM_s014
                                 + 1.11141 w_k1_INTC_s014
                                 + 0.920891 w_k1_KO_s014
                                 + 0.881344 w_k1_MAT_s014
                                 + 1.04221 w_k1_MSFT_s014
                                 + 0.996855 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.19174 w_k1_ADM_s014
                                 + 1.2624 w_k1_AXP_s014 + 1.38759 w_k1_DIS_s014
                                 + 1.13813 w_k1_FDX_s014
                                 + 1.62415 w_k1_IBM_s014
                                 + 1.43371 w_k1_INTC_s014
                                 + 1.34229 w_k1_KO_s014 + 1.52523 w_k1_MAT_s014
                                 + 1.13429 w_k1_MSFT_s014
                                 + 1.13897 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.13245 w_k1_ADM_s014
                                 + 1.03026 w_k1_AXP_s014
                                 + 0.772233 w_k1_DIS_s014
                                 + 1.03181 w_k1_FDX_s014
                                 + 0.682698 w_k1_IBM_s014
                                 + 1.13931 w_k1_INTC_s014
                                 + 0.708945 w_k1_KO_s014
                                 + 0.92641 w_k1_MAT_s014
                                 + 0.974599 w_k1_MSFT_s014
                                 + 0.886155 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.00549 w_k1_ADM_s015
                                 + 1.46585 w_k1_AXP_s015
                                 + 1.17627 w_k1_DIS_s015 + 1.1189 w_k1_FDX_s015
                                 + 1.55924 w_k1_IBM_s015
                                 + 1.24713 w_k1_INTC_s015
                                 + 1.60129 w_k1_KO_s015 + 1.63913 w_k1_MAT_s015
                                 + 1.10085 w_k1_MSFT_s015
                                 + 1.28907 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.700378 w_k1_ADM_s015
                                 + 0.909832 w_k1_AXP_s015
                                 + 0.761946 w_k1_DIS_s015
                                 + 0.796078 w_k1_FDX_s015
                                 + 0.95126 w_k1_IBM_s015
                                 + 0.880631 w_k1_INTC_s015
                                 + 1.07771 w_k1_KO_s015 + 0.80191 w_k1_MAT_s015
                                 + 0.804347 w_k1_MSFT_s015
                                 + 0.982211 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.00616 w_k1_ADM_s015
                                 + 1.33201 w_k1_AXP_s015
                                 + 0.990339 w_k1_DIS_s015
                                 + 1.0171 w_k1_FDX_s015 + 1.03042 w_k1_IBM_s015
                                 + 1.13094 w_k1_INTC_s015
                                 + 1.10485 w_k1_KO_s015
                                 + 0.995527 w_k1_MAT_s015
                                 + 0.973039 w_k1_MSFT_s015
                                 + 1.10737 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.884824 w_k1_ADM_s015
                                 + 0.922075 w_k1_AXP_s015
                                 + 0.889025 w_k1_DIS_s015
                                 + 0.931882 w_k1_FDX_s015
                                 + 0.820947 w_k1_IBM_s015
                                 + 1.076 w_k1_INTC_s015 + 1.00256 w_k1_KO_s015
                                 + 0.630556 w_k1_MAT_s015
                                 + 0.948185 w_k1_MSFT_s015
                                 + 0.953943 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.22031 w_k1_ADM_s015
                                 + 1.04127 w_k1_AXP_s015
                                 + 1.42262 w_k1_DIS_s015
                                 + 1.14471 w_k1_FDX_s015
                                 + 1.04277 w_k1_IBM_s015
                                 + 1.12729 w_k1_INTC_s015
                                 + 1.28716 w_k1_KO_s015
                                 + 0.883915 w_k1_MAT_s015
                                 + 1.14407 w_k1_MSFT_s015
                                 + 1.06347 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.835455 w_k1_ADM_s015
                                 + 0.851383 w_k1_AXP_s015
                                 + 1.16517 w_k1_DIS_s015
                                 + 0.921835 w_k1_FDX_s015
                                 + 1.03798 w_k1_IBM_s015
                                 + 1.18704 w_k1_INTC_s015
                                 + 1.13214 w_k1_KO_s015 + 0.87154 w_k1_MAT_s015
                                 + 1.07174 w_k1_MSFT_s015
                                 + 1.06215 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.07448 w_k1_ADM_s015
                                 + 1.5046 w_k1_AXP_s015 + 1.4189 w_k1_DIS_s015
                                 + 1.32859 w_k1_FDX_s015
                                 + 0.801668 w_k1_IBM_s015
                                 + 1.38984 w_k1_INTC_s015
                                 + 1.23263 w_k1_KO_s015
                                 + 0.839328 w_k1_MAT_s015
                                 + 1.19629 w_k1_MSFT_s015
                                 + 1.14953 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.15252 w_k1_ADM_s015
                                 + 1.4918 w_k1_AXP_s015 + 1.58194 w_k1_DIS_s015
                                 + 1.39355 w_k1_FDX_s015
                                 + 1.07565 w_k1_IBM_s015
                                 + 1.19088 w_k1_INTC_s015
                                 + 1.36588 w_k1_KO_s015 + 1.39576 w_k1_MAT_s015
                                 + 1.20677 w_k1_MSFT_s015
                                 + 1.14544 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.04638 w_k1_ADM_s016
                                 + 0.808617 w_k1_AXP_s016
                                 + 1.06138 w_k1_DIS_s016
                                 + 0.950119 w_k1_FDX_s016
                                 + 1.16334 w_k1_IBM_s016
                                 + 1.03429 w_k1_INTC_s016 + 1.0779 w_k1_KO_s016
                                 + 1.00056 w_k1_MAT_s016
                                 + 0.975162 w_k1_MSFT_s016
                                 + 0.993637 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.02109 w_k1_ADM_s016
                                 + 0.840483 w_k1_AXP_s016
                                 + 0.789068 w_k1_DIS_s016
                                 + 1.00452 w_k1_FDX_s016
                                 + 0.81786 w_k1_IBM_s016
                                 + 0.992498 w_k1_INTC_s016
                                 + 0.758513 w_k1_KO_s016
                                 + 0.690644 w_k1_MAT_s016
                                 + 0.911842 w_k1_MSFT_s016
                                 + 0.965901 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.47805 w_k1_ADM_s016
                                 + 1.32882 w_k1_AXP_s016
                                 + 1.39305 w_k1_DIS_s016
                                 + 1.35927 w_k1_FDX_s016
                                 + 1.01011 w_k1_IBM_s016
                                 + 1.16074 w_k1_INTC_s016
                                 + 1.15738 w_k1_KO_s016 + 1.05284 w_k1_MAT_s016
                                 + 1.20971 w_k1_MSFT_s016
                                 + 1.25215 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.787833 w_k1_ADM_s016
                                 + 0.720256 w_k1_AXP_s016
                                 + 0.676201 w_k1_DIS_s016
                                 + 0.785904 w_k1_FDX_s016
                                 + 0.984992 w_k1_IBM_s016
                                 + 0.762269 w_k1_INTC_s016
                                 + 0.844332 w_k1_KO_s016
                                 + 1.22682 w_k1_MAT_s016
                                 + 0.850415 w_k1_MSFT_s016
                                 + 0.870443 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.824179 w_k1_ADM_s016
                                 + 0.686257 w_k1_AXP_s016
                                 + 0.71537 w_k1_DIS_s016
                                 + 0.824557 w_k1_FDX_s016
                                 + 0.752562 w_k1_IBM_s016
                                 + 0.963863 w_k1_INTC_s016
                                 + 0.705231 w_k1_KO_s016
                                 + 0.495437 w_k1_MAT_s016
                                 + 0.855638 w_k1_MSFT_s016
                                 + 0.957594 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.31887 w_k1_ADM_s016
                                 + 1.34932 w_k1_AXP_s016
                                 + 1.73979 w_k1_DIS_s016
                                 + 1.41682 w_k1_FDX_s016
                                 + 1.32924 w_k1_IBM_s016
                                 + 1.25864 w_k1_INTC_s016
                                 + 1.33878 w_k1_KO_s016 + 1.19583 w_k1_MAT_s016
                                 + 1.43175 w_k1_MSFT_s016
                                 + 1.26793 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.27702 w_k1_ADM_s016
                                 + 1.25996 w_k1_AXP_s016
                                 + 1.22042 w_k1_DIS_s016
                                 + 1.13046 w_k1_FDX_s016
                                 + 1.52436 w_k1_IBM_s016 + 1.035 w_k1_INTC_s016
                                 + 1.24953 w_k1_KO_s016 + 1.15353 w_k1_MAT_s016
                                 + 0.981318 w_k1_MSFT_s016
                                 + 1.12599 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.08522 w_k1_ADM_s016
                                 + 1.09079 w_k1_AXP_s016
                                 + 1.10643 w_k1_DIS_s016
                                 + 1.15584 w_k1_FDX_s016
                                 + 0.886739 w_k1_IBM_s016
                                 + 1.11769 w_k1_INTC_s016
                                 + 0.983905 w_k1_KO_s016
                                 + 0.707823 w_k1_MAT_s016
                                 + 0.9945 w_k1_MSFT_s016 + 1.0438 w_k1_PG_s016
                                 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.1558 w_k1_ADM_s017
                                 + 0.899884 w_k1_AXP_s017
                                 + 0.96491 w_k1_DIS_s017
                                 + 0.978176 w_k1_FDX_s017
                                 + 0.736604 w_k1_IBM_s017
                                 + 1.03924 w_k1_INTC_s017
                                 + 0.88676 w_k1_KO_s017
                                 + 0.756349 w_k1_MAT_s017
                                 + 1.04313 w_k1_MSFT_s017
                                 + 1.02675 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.93537 w_k1_ADM_s017
                                 + 0.642776 w_k1_AXP_s017
                                 + 0.646743 w_k1_DIS_s017
                                 + 0.743585 w_k1_FDX_s017
                                 + 0.500313 w_k1_IBM_s017
                                 + 0.651702 w_k1_INTC_s017
                                 + 0.595988 w_k1_KO_s017
                                 + 0.490211 w_k1_MAT_s017
                                 + 0.737503 w_k1_MSFT_s017
                                 + 0.815282 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.1169 w_k1_ADM_s017
                                 + 1.04895 w_k1_AXP_s017
                                 + 0.968307 w_k1_DIS_s017
                                 + 0.918279 w_k1_FDX_s017
                                 + 0.95449 w_k1_IBM_s017
                                 + 0.863534 w_k1_INTC_s017
                                 + 1.02167 w_k1_KO_s017 + 1.22535 w_k1_MAT_s017
                                 + 0.881262 w_k1_MSFT_s017
                                 + 0.983965 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.62568 w_k1_ADM_s017
                                 + 1.52019 w_k1_AXP_s017
                                 + 1.41478 w_k1_DIS_s017
                                 + 1.21242 w_k1_FDX_s017
                                 + 1.67959 w_k1_IBM_s017
                                 + 1.17609 w_k1_INTC_s017
                                 + 1.62026 w_k1_KO_s017 + 1.75138 w_k1_MAT_s017
                                 + 1.11413 w_k1_MSFT_s017
                                 + 1.18185 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.49903 w_k1_ADM_s017
                                 + 1.31896 w_k1_AXP_s017
                                 + 1.26956 w_k1_DIS_s017
                                 + 1.34376 w_k1_FDX_s017
                                 + 0.844731 w_k1_IBM_s017 + 1.23 w_k1_INTC_s017
                                 + 1.17083 w_k1_KO_s017
                                 + 0.726033 w_k1_MAT_s017
                                 + 1.18038 w_k1_MSFT_s017
                                 + 1.23225 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.61632 w_k1_ADM_s017
                                 + 1.56513 w_k1_AXP_s017
                                 + 1.51912 w_k1_DIS_s017 + 1.2996 w_k1_FDX_s017
                                 + 1.12426 w_k1_IBM_s017
                                 + 1.03259 w_k1_INTC_s017
                                 + 1.09901 w_k1_KO_s017 + 1.27488 w_k1_MAT_s017
                                 + 1.06793 w_k1_MSFT_s017 + 1.2031 w_k1_PG_s017
                                 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.976596 w_k1_ADM_s017
                                 + 0.925861 w_k1_AXP_s017
                                 + 1.02746 w_k1_DIS_s017
                                 + 0.820139 w_k1_FDX_s017
                                 + 1.31537 w_k1_IBM_s017
                                 + 0.918753 w_k1_INTC_s017
                                 + 1.17204 w_k1_KO_s017 + 1.19609 w_k1_MAT_s017
                                 + 1.00944 w_k1_MSFT_s017
                                 + 1.05527 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.1046 w_k1_ADM_s017
                                 + 1.09995 w_k1_AXP_s017
                                 + 1.30438 w_k1_DIS_s017
                                 + 1.07737 w_k1_FDX_s017
                                 + 1.27278 w_k1_IBM_s017
                                 + 0.9921 w_k1_INTC_s017 + 1.23244 w_k1_KO_s017
                                 + 1.39011 w_k1_MAT_s017
                                 + 1.13596 w_k1_MSFT_s017
                                 + 1.12697 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.953924 w_k1_ADM_s100
                                 + 0.764848 w_k1_AXP_s100
                                 + 0.957762 w_k1_DIS_s100
                                 + 0.883746 w_k1_FDX_s100
                                 + 1.01286 w_k1_IBM_s100
                                 + 1.33953 w_k1_INTC_s100
                                 + 1.08666 w_k1_KO_s100 + 0.83674 w_k1_MAT_s100
                                 + 1.02579 w_k1_MSFT_s100
                                 + 0.969104 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.906687 w_k1_ADM_s100
                                 + 1.06558 w_k1_AXP_s100
                                 + 0.797034 w_k1_DIS_s100
                                 + 0.991787 w_k1_FDX_s100
                                 + 0.932321 w_k1_IBM_s100
                                 + 1.31783 w_k1_INTC_s100
                                 + 0.870163 w_k1_KO_s100
                                 + 0.697103 w_k1_MAT_s100
                                 + 0.919863 w_k1_MSFT_s100
                                 + 0.900177 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.04659 w_k1_ADM_s100
                                 + 0.995568 w_k1_AXP_s100
                                 + 1.15143 w_k1_DIS_s100 + 1.1441 w_k1_FDX_s100
                                 + 0.764675 w_k1_IBM_s100
                                 + 1.25527 w_k1_INTC_s100
                                 + 0.925223 w_k1_KO_s100
                                 + 0.619582 w_k1_MAT_s100
                                 + 1.12327 w_k1_MSFT_s100
                                 + 0.913311 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.782529 w_k1_ADM_s100
                                 + 0.987897 w_k1_AXP_s100
                                 + 0.780102 w_k1_DIS_s100
                                 + 0.887006 w_k1_FDX_s100
                                 + 1.49191 w_k1_IBM_s100
                                 + 1.36803 w_k1_INTC_s100
                                 + 1.23219 w_k1_KO_s100 + 1.28085 w_k1_MAT_s100
                                 + 0.917008 w_k1_MSFT_s100
                                 + 1.04253 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.946949 w_k1_ADM_s100
                                 + 1.53421 w_k1_AXP_s100
                                 + 1.59636 w_k1_DIS_s100
                                 + 1.41237 w_k1_FDX_s100
                                 + 1.75708 w_k1_IBM_s100
                                 + 1.75335 w_k1_INTC_s100
                                 + 1.43047 w_k1_KO_s100 + 1.38216 w_k1_MAT_s100
                                 + 1.4275 w_k1_MSFT_s100 + 1.38523 w_k1_PG_s100
                                 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.07011 w_k1_ADM_s100
                                 + 1.25489 w_k1_AXP_s100
                                 + 1.02136 w_k1_DIS_s100
                                 + 1.13211 w_k1_FDX_s100
                                 + 1.21825 w_k1_IBM_s100
                                 + 1.36949 w_k1_INTC_s100
                                 + 1.05481 w_k1_KO_s100 + 1.48559 w_k1_MAT_s100
                                 + 1.08946 w_k1_MSFT_s100
                                 + 1.10021 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.08952 w_k1_ADM_s100
                                 + 1.17098 w_k1_AXP_s100
                                 + 1.55763 w_k1_DIS_s100
                                 + 1.38204 w_k1_FDX_s100
                                 + 1.17551 w_k1_IBM_s100
                                 + 1.59611 w_k1_INTC_s100
                                 + 1.19906 w_k1_KO_s100
                                 + 0.939856 w_k1_MAT_s100
                                 + 1.33619 w_k1_MSFT_s100 + 1.0854 w_k1_PG_s100
                                 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.74703 w_k1_ADM_s100
                                 + 0.77481 w_k1_AXP_s100
                                 + 0.683173 w_k1_DIS_s100
                                 + 0.837051 w_k1_FDX_s100
                                 + 0.778397 w_k1_IBM_s100
                                 + 1.19698 w_k1_INTC_s100
                                 + 0.82882 w_k1_KO_s100
                                 + 0.565432 w_k1_MAT_s100
                                 + 0.819183 w_k1_MSFT_s100
                                 + 0.82094 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.2334 w_k1_ADM_s101
                                 + 1.50187 w_k1_AXP_s101
                                 + 1.72038 w_k1_DIS_s101 + 1.5072 w_k1_FDX_s101
                                 + 1.67699 w_k1_IBM_s101
                                 + 1.15099 w_k1_INTC_s101
                                 + 1.55887 w_k1_KO_s101 + 1.81905 w_k1_MAT_s101
                                 + 1.21163 w_k1_MSFT_s101
                                 + 1.15012 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.856097 w_k1_ADM_s101
                                 + 1.01071 w_k1_AXP_s101
                                 + 1.02667 w_k1_DIS_s101
                                 + 0.995815 w_k1_FDX_s101
                                 + 1.11977 w_k1_IBM_s101
                                 + 1.13719 w_k1_INTC_s101
                                 + 1.07019 w_k1_KO_s101
                                 + 0.817778 w_k1_MAT_s101
                                 + 0.891123 w_k1_MSFT_s101
                                 + 0.9583 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.752709 w_k1_ADM_s101
                                 + 0.699372 w_k1_AXP_s101
                                 + 0.69981 w_k1_DIS_s101
                                 + 0.78021 w_k1_FDX_s101
                                 + 0.84771 w_k1_IBM_s101
                                 + 0.774032 w_k1_INTC_s101
                                 + 0.783922 w_k1_KO_s101
                                 + 0.506035 w_k1_MAT_s101
                                 + 0.630546 w_k1_MSFT_s101
                                 + 0.747445 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.760263 w_k1_ADM_s101
                                 + 0.6395 w_k1_AXP_s101
                                 + 0.702743 w_k1_DIS_s101
                                 + 0.739596 w_k1_FDX_s101
                                 + 0.560669 w_k1_IBM_s101
                                 + 0.780516 w_k1_INTC_s101
                                 + 0.6753 w_k1_KO_s101 + 0.372207 w_k1_MAT_s101
                                 + 0.727099 w_k1_MSFT_s101
                                 + 0.765817 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.725977 w_k1_ADM_s101
                                 + 0.973174 w_k1_AXP_s101
                                 + 0.972772 w_k1_DIS_s101
                                 + 0.994756 w_k1_FDX_s101
                                 + 1.64913 w_k1_IBM_s101
                                 + 1.20286 w_k1_INTC_s101 + 1.4046 w_k1_KO_s101
                                 + 1.563 w_k1_MAT_s101 + 0.91126 w_k1_MSFT_s101
                                 + 0.977057 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.893912 w_k1_ADM_s101
                                 + 0.951677 w_k1_AXP_s101
                                 + 1.10937 w_k1_DIS_s101
                                 + 1.04576 w_k1_FDX_s101
                                 + 0.813011 w_k1_IBM_s101
                                 + 1.04061 w_k1_INTC_s101
                                 + 0.942346 w_k1_KO_s101
                                 + 0.632287 w_k1_MAT_s101
                                 + 0.903009 w_k1_MSFT_s101
                                 + 0.896883 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.828137 w_k1_ADM_s101
                                 + 1.10483 w_k1_AXP_s101
                                 + 1.40673 w_k1_DIS_s101 + 1.0948 w_k1_FDX_s101
                                 + 1.31199 w_k1_IBM_s101
                                 + 1.1702 w_k1_INTC_s101 + 1.45156 w_k1_KO_s101
                                 + 1.00002 w_k1_MAT_s101
                                 + 1.06013 w_k1_MSFT_s101
                                 + 1.04625 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.02245 w_k1_ADM_s101
                                 + 1.14688 w_k1_AXP_s101
                                 + 1.73977 w_k1_DIS_s101
                                 + 1.30374 w_k1_FDX_s101
                                 + 1.29989 w_k1_IBM_s101
                                 + 1.20341 w_k1_INTC_s101
                                 + 1.35551 w_k1_KO_s101 + 1.32239 w_k1_MAT_s101
                                 + 1.13751 w_k1_MSFT_s101
                                 + 1.22501 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.23835 w_k1_ADM_s102
                                 + 1.52628 w_k1_AXP_s102
                                 + 1.53069 w_k1_DIS_s102
                                 + 1.32491 w_k1_FDX_s102
                                 + 1.24108 w_k1_IBM_s102
                                 + 1.30274 w_k1_INTC_s102
                                 + 1.12493 w_k1_KO_s102 + 1.52162 w_k1_MAT_s102
                                 + 1.21197 w_k1_MSFT_s102
                                 + 1.11493 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.31426 w_k1_ADM_s102
                                 + 1.24218 w_k1_AXP_s102 + 1.2492 w_k1_DIS_s102
                                 + 1.24286 w_k1_FDX_s102
                                 + 0.710846 w_k1_IBM_s102
                                 + 1.31517 w_k1_INTC_s102
                                 + 0.953803 w_k1_KO_s102
                                 + 0.656982 w_k1_MAT_s102
                                 + 1.09589 w_k1_MSFT_s102
                                 + 0.958854 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.826904 w_k1_ADM_s102
                                 + 1.05259 w_k1_AXP_s102
                                 + 0.864831 w_k1_DIS_s102
                                 + 0.914697 w_k1_FDX_s102
                                 + 1.10076 w_k1_IBM_s102 + 1.344 w_k1_INTC_s102
                                 + 1.07767 w_k1_KO_s102 + 1.21761 w_k1_MAT_s102
                                 + 1.10255 w_k1_MSFT_s102
                                 + 1.02703 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.780579 w_k1_ADM_s102
                                 + 0.954946 w_k1_AXP_s102
                                 + 0.518039 w_k1_DIS_s102
                                 + 0.719098 w_k1_FDX_s102
                                 + 1.11758 w_k1_IBM_s102
                                 + 0.953059 w_k1_INTC_s102
                                 + 0.830278 w_k1_KO_s102
                                 + 1.36012 w_k1_MAT_s102
                                 + 0.7052 w_k1_MSFT_s102
                                 + 0.941462 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.966128 w_k1_ADM_s102
                                 + 1.03253 w_k1_AXP_s102
                                 + 0.974896 w_k1_DIS_s102
                                 + 1.05378 w_k1_FDX_s102
                                 + 0.749238 w_k1_IBM_s102
                                 + 0.908636 w_k1_INTC_s102
                                 + 0.872717 w_k1_KO_s102
                                 + 1.32008 w_k1_MAT_s102
                                 + 1.12364 w_k1_MSFT_s102
                                 + 1.07214 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.640232 w_k1_ADM_s102
                                 + 0.978873 w_k1_AXP_s102
                                 + 0.769686 w_k1_DIS_s102
                                 + 0.850426 w_k1_FDX_s102
                                 + 0.874311 w_k1_IBM_s102
                                 + 1.01345 w_k1_INTC_s102
                                 + 0.839206 w_k1_KO_s102
                                 + 0.998645 w_k1_MAT_s102
                                 + 0.837103 w_k1_MSFT_s102
                                 + 0.91621 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.861079 w_k1_ADM_s102
                                 + 0.663607 w_k1_AXP_s102
                                 + 0.703502 w_k1_DIS_s102
                                 + 0.819968 w_k1_FDX_s102
                                 + 0.677563 w_k1_IBM_s102
                                 + 0.938542 w_k1_INTC_s102
                                 + 0.684094 w_k1_KO_s102
                                 + 0.548121 w_k1_MAT_s102
                                 + 0.878467 w_k1_MSFT_s102
                                 + 0.79362 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.1103 w_k1_ADM_s102
                                 + 1.34786 w_k1_AXP_s102
                                 + 1.35411 w_k1_DIS_s102
                                 + 1.05441 w_k1_FDX_s102
                                 + 1.50683 w_k1_IBM_s102
                                 + 0.935107 w_k1_INTC_s102
                                 + 1.24247 w_k1_KO_s102 + 2.14116 w_k1_MAT_s102
                                 + 1.12665 w_k1_MSFT_s102
                                 + 1.18147 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.96338 w_k1_ADM_s103
                                 + 0.935505 w_k1_AXP_s103
                                 + 1.09741 w_k1_DIS_s103
                                 + 0.986939 w_k1_FDX_s103
                                 + 0.927212 w_k1_IBM_s103
                                 + 0.923258 w_k1_INTC_s103
                                 + 1.2358 w_k1_KO_s103 + 0.939569 w_k1_MAT_s103
                                 + 0.985045 w_k1_MSFT_s103
                                 + 1.18703 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.6347 w_k1_ADM_s103
                                 + 1.17408 w_k1_AXP_s103
                                 + 1.46968 w_k1_DIS_s103 + 1.0435 w_k1_FDX_s103
                                 + 1.65186 w_k1_IBM_s103
                                 + 0.872143 w_k1_INTC_s103
                                 + 1.46639 w_k1_KO_s103 + 1.88667 w_k1_MAT_s103
                                 + 1.01531 w_k1_MSFT_s103
                                 + 1.32449 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.65104 w_k1_ADM_s103
                                 + 1.34445 w_k1_AXP_s103
                                 + 1.20141 w_k1_DIS_s103
                                 + 1.32096 w_k1_FDX_s103
                                 + 0.882955 w_k1_IBM_s103
                                 + 1.26157 w_k1_INTC_s103
                                 + 1.21395 w_k1_KO_s103 + 1.16581 w_k1_MAT_s103
                                 + 1.15099 w_k1_MSFT_s103
                                 + 1.05608 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.27277 w_k1_ADM_s103
                                 + 0.717885 w_k1_AXP_s103
                                 + 0.852885 w_k1_DIS_s103
                                 + 0.868407 w_k1_FDX_s103
                                 + 0.892418 w_k1_IBM_s103
                                 + 0.850949 w_k1_INTC_s103
                                 + 1.02588 w_k1_KO_s103
                                 + 0.518276 w_k1_MAT_s103
                                 + 0.926812 w_k1_MSFT_s103
                                 + 0.832625 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.03485 w_k1_ADM_s103
                                 + 0.927384 w_k1_AXP_s103
                                 + 0.663356 w_k1_DIS_s103
                                 + 0.729659 w_k1_FDX_s103
                                 + 1.22299 w_k1_IBM_s103
                                 + 0.752184 w_k1_INTC_s103
                                 + 0.99339 w_k1_KO_s103
                                 + 0.919222 w_k1_MAT_s103
                                 + 0.66559 w_k1_MSFT_s103
                                 + 0.903144 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.74087 w_k1_ADM_s103
                                 + 1.34027 w_k1_AXP_s103
                                 + 1.72644 w_k1_DIS_s103 + 1.3586 w_k1_FDX_s103
                                 + 1.2042 w_k1_IBM_s103
                                 + 1.23161 w_k1_INTC_s103
                                 + 1.35607 w_k1_KO_s103 + 1.03105 w_k1_MAT_s103
                                 + 1.20173 w_k1_MSFT_s103
                                 + 1.19301 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.40417 w_k1_ADM_s103
                                 + 0.920691 w_k1_AXP_s103
                                 + 0.980234 w_k1_DIS_s103
                                 + 1.03553 w_k1_FDX_s103
                                 + 0.571413 w_k1_IBM_s103
                                 + 0.759052 w_k1_INTC_s103
                                 + 0.877566 w_k1_KO_s103
                                 + 0.546679 w_k1_MAT_s103
                                 + 0.88842 w_k1_MSFT_s103
                                 + 1.00237 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.19764 w_k1_ADM_s103
                                 + 0.889697 w_k1_AXP_s103
                                 + 1.20578 w_k1_DIS_s103
                                 + 0.973526 w_k1_FDX_s103
                                 + 1.20348 w_k1_IBM_s103
                                 + 1.13437 w_k1_INTC_s103
                                 + 1.32994 w_k1_KO_s103
                                 + 0.749199 w_k1_MAT_s103
                                 + 1.15932 w_k1_MSFT_s103 + 1.0876 w_k1_PG_s103
                                 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.02704 w_k1_ADM_s104
                                 + 0.736619 w_k1_AXP_s104
                                 + 0.770967 w_k1_DIS_s104
                                 + 0.813928 w_k1_FDX_s104
                                 + 0.784336 w_k1_IBM_s104
                                 + 0.786338 w_k1_INTC_s104
                                 + 0.785472 w_k1_KO_s104
                                 + 0.849349 w_k1_MAT_s104
                                 + 0.831771 w_k1_MSFT_s104
                                 + 0.795508 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.47262 w_k1_ADM_s104
                                 + 0.997198 w_k1_AXP_s104
                                 + 1.22996 w_k1_DIS_s104
                                 + 1.04347 w_k1_FDX_s104
                                 + 1.22847 w_k1_IBM_s104
                                 + 1.15122 w_k1_INTC_s104
                                 + 1.13288 w_k1_KO_s104
                                 + 0.803906 w_k1_MAT_s104
                                 + 1.0288 w_k1_MSFT_s104 + 1.06063 w_k1_PG_s104
                                 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.0756 w_k1_ADM_s104
                                 + 0.822873 w_k1_AXP_s104
                                 + 0.951956 w_k1_DIS_s104
                                 + 0.880086 w_k1_FDX_s104
                                 + 0.924745 w_k1_IBM_s104
                                 + 1.1133 w_k1_INTC_s104
                                 + 0.900484 w_k1_KO_s104
                                 + 0.696131 w_k1_MAT_s104
                                 + 0.878805 w_k1_MSFT_s104
                                 + 0.866134 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.26219 w_k1_ADM_s104
                                 + 0.961753 w_k1_AXP_s104
                                 + 1.11655 w_k1_DIS_s104
                                 + 1.20633 w_k1_FDX_s104
                                 + 0.701405 w_k1_IBM_s104
                                 + 0.954623 w_k1_INTC_s104
                                 + 0.852534 w_k1_KO_s104
                                 + 0.918706 w_k1_MAT_s104
                                 + 1.07989 w_k1_MSFT_s104 + 1.0113 w_k1_PG_s104
                                 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.05408 w_k1_ADM_s104
                                 + 1.38029 w_k1_AXP_s104
                                 + 1.97839 w_k1_DIS_s104
                                 + 1.30834 w_k1_FDX_s104
                                 + 1.40201 w_k1_IBM_s104
                                 + 1.38484 w_k1_INTC_s104
                                 + 1.37717 w_k1_KO_s104 + 1.46283 w_k1_MAT_s104
                                 + 1.32633 w_k1_MSFT_s104
                                 + 1.18797 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.1225 w_k1_ADM_s104
                                 + 1.26473 w_k1_AXP_s104
                                 + 1.24731 w_k1_DIS_s104
                                 + 1.17553 w_k1_FDX_s104
                                 + 0.870289 w_k1_IBM_s104
                                 + 1.02009 w_k1_INTC_s104
                                 + 0.932144 w_k1_KO_s104
                                 + 1.04664 w_k1_MAT_s104
                                 + 0.910995 w_k1_MSFT_s104
                                 + 1.01999 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.967985 w_k1_ADM_s104
                                 + 0.728869 w_k1_AXP_s104
                                 + 0.889344 w_k1_DIS_s104
                                 + 0.837803 w_k1_FDX_s104
                                 + 1.35918 w_k1_IBM_s104
                                 + 0.874563 w_k1_INTC_s104
                                 + 0.936175 w_k1_KO_s104
                                 + 1.02251 w_k1_MAT_s104
                                 + 0.748867 w_k1_MSFT_s104
                                 + 0.89699 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.2961 w_k1_ADM_s104
                                 + 1.21238 w_k1_AXP_s104
                                 + 1.30273 w_k1_DIS_s104
                                 + 1.24509 w_k1_FDX_s104
                                 + 1.16409 w_k1_IBM_s104
                                 + 1.09383 w_k1_INTC_s104
                                 + 1.14737 w_k1_KO_s104 + 1.43496 w_k1_MAT_s104
                                 + 1.14839 w_k1_MSFT_s104
                                 + 1.11378 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.47529 w_k1_ADM_s105
                                 + 1.47539 w_k1_AXP_s105
                                 + 1.43644 w_k1_DIS_s105
                                 + 1.40471 w_k1_FDX_s105
                                 + 1.21229 w_k1_IBM_s105
                                 + 1.26516 w_k1_INTC_s105
                                 + 1.10985 w_k1_KO_s105
                                 + 0.840445 w_k1_MAT_s105
                                 + 1.36691 w_k1_MSFT_s105
                                 + 1.10083 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.11959 w_k1_ADM_s105
                                 + 1.19739 w_k1_AXP_s105
                                 + 1.17873 w_k1_DIS_s105
                                 + 1.07073 w_k1_FDX_s105
                                 + 1.17173 w_k1_IBM_s105
                                 + 1.15805 w_k1_INTC_s105
                                 + 1.09414 w_k1_KO_s105 + 1.02262 w_k1_MAT_s105
                                 + 1.07881 w_k1_MSFT_s105
                                 + 1.09408 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.995882 w_k1_ADM_s105
                                 + 1.21733 w_k1_AXP_s105
                                 + 0.884728 w_k1_DIS_s105
                                 + 0.898291 w_k1_FDX_s105
                                 + 1.35684 w_k1_IBM_s105
                                 + 0.916162 w_k1_INTC_s105
                                 + 0.941377 w_k1_KO_s105 + 1.6908 w_k1_MAT_s105
                                 + 0.843873 w_k1_MSFT_s105
                                 + 1.0393 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.01714 w_k1_ADM_s105
                                 + 1.60025 w_k1_AXP_s105
                                 + 1.28018 w_k1_DIS_s105
                                 + 1.18779 w_k1_FDX_s105 + 1.8355 w_k1_IBM_s105
                                 + 1.22199 w_k1_INTC_s105
                                 + 1.24521 w_k1_KO_s105 + 1.89748 w_k1_MAT_s105
                                 + 1.11186 w_k1_MSFT_s105
                                 + 1.26656 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.16537 w_k1_ADM_s105
                                 + 1.2351 w_k1_AXP_s105 + 1.00853 w_k1_DIS_s105
                                 + 1.11253 w_k1_FDX_s105
                                 + 0.787672 w_k1_IBM_s105
                                 + 1.08509 w_k1_INTC_s105
                                 + 0.796321 w_k1_KO_s105
                                 + 1.01367 w_k1_MAT_s105
                                 + 1.15185 w_k1_MSFT_s105
                                 + 1.10166 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.971236 w_k1_ADM_s105
                                 + 0.938984 w_k1_AXP_s105
                                 + 0.858093 w_k1_DIS_s105
                                 + 0.842974 w_k1_FDX_s105
                                 + 1.46211 w_k1_IBM_s105
                                 + 0.867614 w_k1_INTC_s105
                                 + 1.02608 w_k1_KO_s105 + 1.00641 w_k1_MAT_s105
                                 + 0.834796 w_k1_MSFT_s105
                                 + 0.990643 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.763961 w_k1_ADM_s105
                                 + 0.742112 w_k1_AXP_s105
                                 + 0.643271 w_k1_DIS_s105
                                 + 0.770111 w_k1_FDX_s105
                                 + 0.66209 w_k1_IBM_s105
                                 + 0.836173 w_k1_INTC_s105
                                 + 0.646313 w_k1_KO_s105
                                 + 0.651872 w_k1_MAT_s105
                                 + 0.842201 w_k1_MSFT_s105
                                 + 0.805036 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.883787 w_k1_ADM_s105
                                 + 0.779057 w_k1_AXP_s105
                                 + 0.811115 w_k1_DIS_s105
                                 + 0.826992 w_k1_FDX_s105
                                 + 0.980709 w_k1_IBM_s105
                                 + 0.857962 w_k1_INTC_s105
                                 + 0.865927 w_k1_KO_s105 + 1.0276 w_k1_MAT_s105
                                 + 0.965164 w_k1_MSFT_s105
                                 + 0.979741 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.59865 w_k1_ADM_s106
                                 + 1.76464 w_k1_AXP_s106 + 1.4888 w_k1_DIS_s106
                                 + 1.26421 w_k1_FDX_s106
                                 + 1.56446 w_k1_IBM_s106
                                 + 1.38729 w_k1_INTC_s106
                                 + 1.26501 w_k1_KO_s106 + 1.3659 w_k1_MAT_s106
                                 + 1.06174 w_k1_MSFT_s106
                                 + 1.29896 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.935127 w_k1_ADM_s106
                                 + 0.806687 w_k1_AXP_s106
                                 + 0.752597 w_k1_DIS_s106
                                 + 0.883556 w_k1_FDX_s106
                                 + 0.633555 w_k1_IBM_s106
                                 + 0.970778 w_k1_INTC_s106
                                 + 0.629561 w_k1_KO_s106
                                 + 0.472609 w_k1_MAT_s106
                                 + 0.90593 w_k1_MSFT_s106
                                 + 0.78167 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.29102 w_k1_ADM_s106
                                 + 1.16955 w_k1_AXP_s106
                                 + 1.54561 w_k1_DIS_s106
                                 + 1.25095 w_k1_FDX_s106
                                 + 0.947567 w_k1_IBM_s106
                                 + 1.44384 w_k1_INTC_s106
                                 + 1.09484 w_k1_KO_s106
                                 + 0.765624 w_k1_MAT_s106
                                 + 1.24391 w_k1_MSFT_s106
                                 + 1.16996 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.27938 w_k1_ADM_s106
                                 + 1.25773 w_k1_AXP_s106
                                 + 1.12932 w_k1_DIS_s106
                                 + 1.05252 w_k1_FDX_s106
                                 + 1.39732 w_k1_IBM_s106
                                 + 1.03682 w_k1_INTC_s106
                                 + 1.08624 w_k1_KO_s106 + 1.14579 w_k1_MAT_s106
                                 + 0.893741 w_k1_MSFT_s106
                                 + 0.900668 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.20234 w_k1_ADM_s106
                                 + 1.08721 w_k1_AXP_s106
                                 + 0.802567 w_k1_DIS_s106
                                 + 0.870425 w_k1_FDX_s106
                                 + 1.34644 w_k1_IBM_s106
                                 + 0.982633 w_k1_INTC_s106
                                 + 0.888467 w_k1_KO_s106
                                 + 1.05058 w_k1_MAT_s106
                                 + 0.758535 w_k1_MSFT_s106
                                 + 0.973845 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.831276 w_k1_ADM_s106
                                 + 0.842971 w_k1_AXP_s106
                                 + 0.859019 w_k1_DIS_s106
                                 + 0.907578 w_k1_FDX_s106
                                 + 0.944133 w_k1_IBM_s106
                                 + 1.06419 w_k1_INTC_s106
                                 + 0.866897 w_k1_KO_s106
                                 + 0.766334 w_k1_MAT_s106
                                 + 0.876589 w_k1_MSFT_s106
                                 + 0.965117 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.15483 w_k1_ADM_s106
                                 + 0.971968 w_k1_AXP_s106
                                 + 0.903066 w_k1_DIS_s106
                                 + 1.0429 w_k1_FDX_s106
                                 + 0.862449 w_k1_IBM_s106
                                 + 1.10857 w_k1_INTC_s106
                                 + 0.752989 w_k1_KO_s106
                                 + 0.714841 w_k1_MAT_s106
                                 + 0.97602 w_k1_MSFT_s106
                                 + 0.905329 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.51143 w_k1_ADM_s106
                                 + 1.45889 w_k1_AXP_s106
                                 + 1.64697 w_k1_DIS_s106
                                 + 1.48265 w_k1_FDX_s106
                                 + 1.23142 w_k1_IBM_s106
                                 + 1.34227 w_k1_INTC_s106
                                 + 1.11536 w_k1_KO_s106 + 1.05467 w_k1_MAT_s106
                                 + 1.17389 w_k1_MSFT_s106
                                 + 1.02445 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.87727 w_k1_ADM_s107
                                 + 1.16239 w_k1_AXP_s107
                                 + 1.09691 w_k1_DIS_s107
                                 + 1.07257 w_k1_FDX_s107
                                 + 0.928576 w_k1_IBM_s107
                                 + 1.28452 w_k1_INTC_s107
                                 + 0.981801 w_k1_KO_s107
                                 + 1.06877 w_k1_MAT_s107
                                 + 1.11182 w_k1_MSFT_s107
                                 + 1.02307 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.32624 w_k1_ADM_s107
                                 + 0.966923 w_k1_AXP_s107
                                 + 1.19281 w_k1_DIS_s107
                                 + 1.29381 w_k1_FDX_s107
                                 + 0.886209 w_k1_IBM_s107
                                 + 0.913443 w_k1_INTC_s107
                                 + 0.962469 w_k1_KO_s107
                                 + 0.892496 w_k1_MAT_s107
                                 + 1.13565 w_k1_MSFT_s107
                                 + 1.17064 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.771935 w_k1_ADM_s107
                                 + 0.75318 w_k1_AXP_s107
                                 + 0.612302 w_k1_DIS_s107
                                 + 0.862075 w_k1_FDX_s107
                                 + 0.621107 w_k1_IBM_s107
                                 + 0.982572 w_k1_INTC_s107
                                 + 0.570413 w_k1_KO_s107
                                 + 0.533181 w_k1_MAT_s107
                                 + 0.786315 w_k1_MSFT_s107
                                 + 0.722417 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.721581 w_k1_ADM_s107
                                 + 1.21923 w_k1_AXP_s107
                                 + 0.895523 w_k1_DIS_s107
                                 + 0.992089 w_k1_FDX_s107
                                 + 1.57905 w_k1_IBM_s107
                                 + 1.09287 w_k1_INTC_s107 + 1.0686 w_k1_KO_s107
                                 + 2.44246 w_k1_MAT_s107 + 0.795 w_k1_MSFT_s107
                                 + 0.955023 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.01224 w_k1_ADM_s107
                                 + 1.06824 w_k1_AXP_s107
                                 + 1.33044 w_k1_DIS_s107
                                 + 1.10935 w_k1_FDX_s107
                                 + 1.36892 w_k1_IBM_s107
                                 + 1.23472 w_k1_INTC_s107
                                 + 1.35103 w_k1_KO_s107 + 1.47506 w_k1_MAT_s107
                                 + 1.31976 w_k1_MSFT_s107
                                 + 1.07961 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.22823 w_k1_ADM_s107
                                 + 1.45752 w_k1_AXP_s107
                                 + 1.31128 w_k1_DIS_s107
                                 + 1.33954 w_k1_FDX_s107
                                 + 1.22917 w_k1_IBM_s107
                                 + 1.13963 w_k1_INTC_s107
                                 + 1.23537 w_k1_KO_s107 + 1.46524 w_k1_MAT_s107
                                 + 1.25789 w_k1_MSFT_s107 + 1.1006 w_k1_PG_s107
                                 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.927043 w_k1_ADM_s107
                                 + 0.737611 w_k1_AXP_s107
                                 + 0.968678 w_k1_DIS_s107
                                 + 0.947095 w_k1_FDX_s107
                                 + 0.773504 w_k1_IBM_s107
                                 + 1.03185 w_k1_INTC_s107
                                 + 1.00256 w_k1_KO_s107
                                 + 0.764072 w_k1_MAT_s107
                                 + 1.08912 w_k1_MSFT_s107
                                 + 0.872323 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.01651 w_k1_ADM_s107
                                 + 0.960488 w_k1_AXP_s107
                                 + 0.768789 w_k1_DIS_s107
                                 + 0.852042 w_k1_FDX_s107
                                 + 1.0364 w_k1_IBM_s107
                                 + 0.968313 w_k1_INTC_s107
                                 + 0.844962 w_k1_KO_s107
                                 + 1.22545 w_k1_MAT_s107
                                 + 0.832874 w_k1_MSFT_s107
                                 + 0.921804 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.874907 w_k1_ADM_s110
                                 + 1.13815 w_k1_AXP_s110
                                 + 0.85772 w_k1_DIS_s110
                                 + 1.08948 w_k1_FDX_s110
                                 + 0.942208 w_k1_IBM_s110
                                 + 0.999564 w_k1_INTC_s110
                                 + 0.856653 w_k1_KO_s110
                                 + 1.32624 w_k1_MAT_s110
                                 + 0.923412 w_k1_MSFT_s110
                                 + 0.993739 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.48115 w_k1_ADM_s110
                                 + 1.70651 w_k1_AXP_s110
                                 + 1.53761 w_k1_DIS_s110
                                 + 1.41291 w_k1_FDX_s110
                                 + 1.62036 w_k1_IBM_s110
                                 + 1.93659 w_k1_INTC_s110
                                 + 1.55042 w_k1_KO_s110 + 1.89588 w_k1_MAT_s110
                                 + 1.22131 w_k1_MSFT_s110
                                 + 1.26893 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.892797 w_k1_ADM_s110
                                 + 1.17074 w_k1_AXP_s110
                                 + 1.35092 w_k1_DIS_s110
                                 + 1.34109 w_k1_FDX_s110
                                 + 0.809143 w_k1_IBM_s110
                                 + 1.17946 w_k1_INTC_s110
                                 + 1.14189 w_k1_KO_s110 + 1.17685 w_k1_MAT_s110
                                 + 1.23951 w_k1_MSFT_s110 + 1.0679 w_k1_PG_s110
                                 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.28894 w_k1_ADM_s110
                                 + 1.71653 w_k1_AXP_s110
                                 + 1.42928 w_k1_DIS_s110
                                 + 1.38212 w_k1_FDX_s110
                                 + 1.19438 w_k1_IBM_s110
                                 + 1.13919 w_k1_INTC_s110
                                 + 1.18577 w_k1_KO_s110 + 1.41195 w_k1_MAT_s110
                                 + 1.20538 w_k1_MSFT_s110
                                 + 1.08067 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.09764 w_k1_ADM_s110
                                 + 1.19237 w_k1_AXP_s110
                                 + 1.11591 w_k1_DIS_s110
                                 + 1.03899 w_k1_FDX_s110
                                 + 1.17513 w_k1_IBM_s110
                                 + 0.924254 w_k1_INTC_s110
                                 + 1.1123 w_k1_KO_s110 + 1.11791 w_k1_MAT_s110
                                 + 0.940378 w_k1_MSFT_s110
                                 + 1.05833 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.832419 w_k1_ADM_s110
                                 + 0.847379 w_k1_AXP_s110
                                 + 0.62831 w_k1_DIS_s110
                                 + 0.818331 w_k1_FDX_s110
                                 + 0.784426 w_k1_IBM_s110
                                 + 0.931038 w_k1_INTC_s110
                                 + 0.823424 w_k1_KO_s110
                                 + 0.923041 w_k1_MAT_s110
                                 + 0.845454 w_k1_MSFT_s110
                                 + 0.847065 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.918097 w_k1_ADM_s110
                                 + 0.777572 w_k1_AXP_s110
                                 + 0.693434 w_k1_DIS_s110
                                 + 0.800575 w_k1_FDX_s110
                                 + 1.25884 w_k1_IBM_s110
                                 + 0.681044 w_k1_INTC_s110
                                 + 0.827492 w_k1_KO_s110
                                 + 1.17848 w_k1_MAT_s110
                                 + 0.64884 w_k1_MSFT_s110
                                 + 0.796826 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.847533 w_k1_ADM_s110
                                 + 0.884135 w_k1_AXP_s110
                                 + 0.87393 w_k1_DIS_s110
                                 + 1.07631 w_k1_FDX_s110
                                 + 0.714328 w_k1_IBM_s110
                                 + 0.967354 w_k1_INTC_s110
                                 + 0.796557 w_k1_KO_s110
                                 + 0.822609 w_k1_MAT_s110
                                 + 0.972088 w_k1_MSFT_s110
                                 + 0.863473 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.13008 w_k1_ADM_s111
                                 + 0.841612 w_k1_AXP_s111
                                 + 1.05874 w_k1_DIS_s111
                                 + 1.09796 w_k1_FDX_s111
                                 + 0.705294 w_k1_IBM_s111
                                 + 1.12614 w_k1_INTC_s111
                                 + 0.908207 w_k1_KO_s111
                                 + 0.543654 w_k1_MAT_s111
                                 + 1.17976 w_k1_MSFT_s111
                                 + 1.02659 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.31379 w_k1_ADM_s111
                                 + 1.38272 w_k1_AXP_s111
                                 + 1.39419 w_k1_DIS_s111
                                 + 1.35323 w_k1_FDX_s111
                                 + 1.03134 w_k1_IBM_s111
                                 + 1.07952 w_k1_INTC_s111
                                 + 1.22976 w_k1_KO_s111 + 1.08289 w_k1_MAT_s111
                                 + 1.19492 w_k1_MSFT_s111
                                 + 1.18887 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.07139 w_k1_ADM_s111
                                 + 1.03743 w_k1_AXP_s111
                                 + 1.14209 w_k1_DIS_s111
                                 + 1.00961 w_k1_FDX_s111
                                 + 0.969074 w_k1_IBM_s111
                                 + 1.09975 w_k1_INTC_s111
                                 + 1.13268 w_k1_KO_s111
                                 + 0.774451 w_k1_MAT_s111
                                 + 1.06402 w_k1_MSFT_s111
                                 + 0.933446 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.971072 w_k1_ADM_s111
                                 + 1.02933 w_k1_AXP_s111
                                 + 1.02322 w_k1_DIS_s111
                                 + 1.01587 w_k1_FDX_s111
                                 + 1.03048 w_k1_IBM_s111
                                 + 1.02858 w_k1_INTC_s111
                                 + 1.24251 w_k1_KO_s111 + 1.19588 w_k1_MAT_s111
                                 + 1.02249 w_k1_MSFT_s111
                                 + 1.04047 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.92615 w_k1_ADM_s111
                                 + 0.692116 w_k1_AXP_s111
                                 + 0.588651 w_k1_DIS_s111
                                 + 0.756953 w_k1_FDX_s111
                                 + 0.645417 w_k1_IBM_s111
                                 + 0.701198 w_k1_INTC_s111
                                 + 0.728624 w_k1_KO_s111
                                 + 0.550964 w_k1_MAT_s111
                                 + 0.722178 w_k1_MSFT_s111
                                 + 0.819958 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.957981 w_k1_ADM_s111
                                 + 0.758966 w_k1_AXP_s111
                                 + 0.728573 w_k1_DIS_s111
                                 + 0.764358 w_k1_FDX_s111
                                 + 0.980645 w_k1_IBM_s111
                                 + 0.811139 w_k1_INTC_s111
                                 + 0.849504 w_k1_KO_s111
                                 + 0.731536 w_k1_MAT_s111
                                 + 0.823712 w_k1_MSFT_s111
                                 + 0.886162 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.81074 w_k1_ADM_s111
                                 + 0.958695 w_k1_AXP_s111
                                 + 0.869388 w_k1_DIS_s111
                                 + 0.947181 w_k1_FDX_s111
                                 + 0.706866 w_k1_IBM_s111
                                 + 1.01418 w_k1_INTC_s111
                                 + 0.956752 w_k1_KO_s111
                                 + 0.735133 w_k1_MAT_s111
                                 + 0.986949 w_k1_MSFT_s111
                                 + 0.974438 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.881513 w_k1_ADM_s111
                                 + 1.11384 w_k1_AXP_s111
                                 + 0.907725 w_k1_DIS_s111
                                 + 0.952767 w_k1_FDX_s111
                                 + 1.35962 w_k1_IBM_s111
                                 + 1.04607 w_k1_INTC_s111
                                 + 1.15526 w_k1_KO_s111 + 1.1815 w_k1_MAT_s111
                                 + 0.927756 w_k1_MSFT_s111
                                 + 1.00463 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.05424 w_k1_ADM_s112
                                 + 1.12233 w_k1_AXP_s112
                                 + 0.760532 w_k1_DIS_s112
                                 + 0.997001 w_k1_FDX_s112
                                 + 1.08026 w_k1_IBM_s112
                                 + 0.969981 w_k1_INTC_s112
                                 + 0.872591 w_k1_KO_s112
                                 + 0.756477 w_k1_MAT_s112
                                 + 0.891767 w_k1_MSFT_s112
                                 + 1.00265 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.715849 w_k1_ADM_s112
                                 + 0.766175 w_k1_AXP_s112
                                 + 0.725109 w_k1_DIS_s112
                                 + 0.7443 w_k1_FDX_s112
                                 + 0.940205 w_k1_IBM_s112
                                 + 0.868956 w_k1_INTC_s112
                                 + 0.917836 w_k1_KO_s112
                                 + 0.924093 w_k1_MAT_s112
                                 + 0.741099 w_k1_MSFT_s112
                                 + 0.86286 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.925895 w_k1_ADM_s112
                                 + 1.36788 w_k1_AXP_s112
                                 + 1.28881 w_k1_DIS_s112
                                 + 1.27393 w_k1_FDX_s112
                                 + 0.878784 w_k1_IBM_s112
                                 + 1.31986 w_k1_INTC_s112
                                 + 0.946194 w_k1_KO_s112
                                 + 1.03511 w_k1_MAT_s112
                                 + 1.18399 w_k1_MSFT_s112
                                 + 1.01792 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.12808 w_k1_ADM_s112
                                 + 1.51743 w_k1_AXP_s112
                                 + 1.87935 w_k1_DIS_s112
                                 + 1.49317 w_k1_FDX_s112
                                 + 1.19009 w_k1_IBM_s112
                                 + 1.50072 w_k1_INTC_s112
                                 + 1.13041 w_k1_KO_s112
                                 + 0.934893 w_k1_MAT_s112
                                 + 1.33192 w_k1_MSFT_s112
                                 + 1.20718 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.959729 w_k1_ADM_s112
                                 + 1.21832 w_k1_AXP_s112
                                 + 1.12246 w_k1_DIS_s112
                                 + 0.936548 w_k1_FDX_s112
                                 + 1.47795 w_k1_IBM_s112
                                 + 1.33605 w_k1_INTC_s112
                                 + 1.23771 w_k1_KO_s112
                                 + 0.995133 w_k1_MAT_s112
                                 + 0.944028 w_k1_MSFT_s112
                                 + 1.13428 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.72929 w_k1_ADM_s112
                                 + 0.714034 w_k1_AXP_s112
                                 + 0.890148 w_k1_DIS_s112
                                 + 0.837135 w_k1_FDX_s112
                                 + 0.877762 w_k1_IBM_s112
                                 + 1.06447 w_k1_INTC_s112
                                 + 0.77735 w_k1_KO_s112 + 0.50536 w_k1_MAT_s112
                                 + 0.94292 w_k1_MSFT_s112
                                 + 0.937227 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.09273 w_k1_ADM_s112
                                 + 0.99144 w_k1_AXP_s112 + 1.2092 w_k1_DIS_s112
                                 + 1.05447 w_k1_FDX_s112
                                 + 1.00445 w_k1_IBM_s112
                                 + 1.28441 w_k1_INTC_s112
                                 + 1.03047 w_k1_KO_s112
                                 + 0.929809 w_k1_MAT_s112
                                 + 1.15516 w_k1_MSFT_s112 + 1.0587 w_k1_PG_s112
                                 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.878148 w_k1_ADM_s112
                                 + 0.889615 w_k1_AXP_s112
                                 + 0.91323 w_k1_DIS_s112
                                 + 0.976478 w_k1_FDX_s112
                                 + 0.690067 w_k1_IBM_s112
                                 + 1.14897 w_k1_INTC_s112
                                 + 0.813086 w_k1_KO_s112
                                 + 0.694933 w_k1_MAT_s112
                                 + 1.00558 w_k1_MSFT_s112
                                 + 1.03196 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.16648 w_k1_ADM_s113
                                 + 1.097 w_k1_AXP_s113 + 0.962876 w_k1_DIS_s113
                                 + 0.968857 w_k1_FDX_s113
                                 + 1.0964 w_k1_IBM_s113
                                 + 0.748373 w_k1_INTC_s113
                                 + 1.09781 w_k1_KO_s113 + 1.308 w_k1_MAT_s113
                                 + 1.07643 w_k1_MSFT_s113
                                 + 0.991644 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.789905 w_k1_ADM_s113
                                 + 0.570669 w_k1_AXP_s113
                                 + 0.64539 w_k1_DIS_s113
                                 + 0.692828 w_k1_FDX_s113
                                 + 0.768161 w_k1_IBM_s113
                                 + 0.682344 w_k1_INTC_s113
                                 + 0.810076 w_k1_KO_s113
                                 + 0.669153 w_k1_MAT_s113
                                 + 0.842288 w_k1_MSFT_s113
                                 + 0.828441 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.19737 w_k1_ADM_s113
                                 + 1.31868 w_k1_AXP_s113 + 1.1644 w_k1_DIS_s113
                                 + 0.895923 w_k1_FDX_s113
                                 + 1.66816 w_k1_IBM_s113
                                 + 0.829743 w_k1_INTC_s113
                                 + 1.24982 w_k1_KO_s113 + 1.74574 w_k1_MAT_s113
                                 + 1.00013 w_k1_MSFT_s113 + 1.0999 w_k1_PG_s113
                                 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.64419 w_k1_ADM_s113
                                 + 2.27419 w_k1_AXP_s113
                                 + 2.22656 w_k1_DIS_s113
                                 + 1.56282 w_k1_FDX_s113
                                 + 1.36695 w_k1_IBM_s113
                                 + 1.02686 w_k1_INTC_s113
                                 + 1.47353 w_k1_KO_s113 + 1.802 w_k1_MAT_s113
                                 + 1.27348 w_k1_MSFT_s113
                                 + 1.30206 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.03931 w_k1_ADM_s113
                                 + 1.07432 w_k1_AXP_s113
                                 + 1.03643 w_k1_DIS_s113
                                 + 1.07461 w_k1_FDX_s113
                                 + 0.711788 w_k1_IBM_s113
                                 + 1.00801 w_k1_INTC_s113
                                 + 0.970566 w_k1_KO_s113
                                 + 0.744322 w_k1_MAT_s113
                                 + 1.2192 w_k1_MSFT_s113 + 1.01922 w_k1_PG_s113
                                 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.49712 w_k1_ADM_s113
                                 + 1.57801 w_k1_AXP_s113
                                 + 1.82179 w_k1_DIS_s113
                                 + 1.41375 w_k1_FDX_s113
                                 + 1.02687 w_k1_IBM_s113
                                 + 1.09518 w_k1_INTC_s113
                                 + 1.38288 w_k1_KO_s113 + 1.32691 w_k1_MAT_s113
                                 + 1.43562 w_k1_MSFT_s113
                                 + 1.22756 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.694049 w_k1_ADM_s113
                                 + 0.74342 w_k1_AXP_s113
                                 + 0.621906 w_k1_DIS_s113
                                 + 0.583484 w_k1_FDX_s113
                                 + 0.943968 w_k1_IBM_s113
                                 + 0.543751 w_k1_INTC_s113
                                 + 0.703786 w_k1_KO_s113
                                 + 1.11431 w_k1_MAT_s113
                                 + 0.565178 w_k1_MSFT_s113
                                 + 0.707198 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.927997 w_k1_ADM_s113
                                 + 0.888374 w_k1_AXP_s113
                                 + 0.694147 w_k1_DIS_s113
                                 + 0.784172 w_k1_FDX_s113
                                 + 1.23343 w_k1_IBM_s113
                                 + 0.770796 w_k1_INTC_s113
                                 + 0.957032 w_k1_KO_s113
                                 + 1.31199 w_k1_MAT_s113
                                 + 0.863277 w_k1_MSFT_s113
                                 + 0.895795 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.08521 w_k1_ADM_s114
                                 + 0.911086 w_k1_AXP_s114
                                 + 0.886132 w_k1_DIS_s114
                                 + 1.08472 w_k1_FDX_s114
                                 + 0.743547 w_k1_IBM_s114
                                 + 0.925816 w_k1_INTC_s114
                                 + 0.786271 w_k1_KO_s114
                                 + 0.716849 w_k1_MAT_s114
                                 + 0.936979 w_k1_MSFT_s114
                                 + 0.988546 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.03697 w_k1_ADM_s114
                                 + 0.999871 w_k1_AXP_s114
                                 + 1.26986 w_k1_DIS_s114
                                 + 1.12792 w_k1_FDX_s114
                                 + 0.816104 w_k1_IBM_s114
                                 + 1.42154 w_k1_INTC_s114
                                 + 1.06385 w_k1_KO_s114
                                 + 0.842194 w_k1_MAT_s114
                                 + 1.3359 w_k1_MSFT_s114 + 1.06413 w_k1_PG_s114
                                 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.809061 w_k1_ADM_s114
                                 + 0.771227 w_k1_AXP_s114
                                 + 0.648081 w_k1_DIS_s114
                                 + 0.832876 w_k1_FDX_s114
                                 + 0.7075 w_k1_IBM_s114
                                 + 0.977038 w_k1_INTC_s114
                                 + 0.683893 w_k1_KO_s114
                                 + 0.591597 w_k1_MAT_s114
                                 + 0.890155 w_k1_MSFT_s114
                                 + 0.785785 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.761455 w_k1_ADM_s114
                                 + 1.05528 w_k1_AXP_s114
                                 + 0.946167 w_k1_DIS_s114
                                 + 0.900229 w_k1_FDX_s114
                                 + 1.60371 w_k1_IBM_s114
                                 + 1.03817 w_k1_INTC_s114
                                 + 1.29339 w_k1_KO_s114 + 1.7297 w_k1_MAT_s114
                                 + 0.861374 w_k1_MSFT_s114
                                 + 1.07193 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.91233 w_k1_ADM_s114
                                 + 0.821579 w_k1_AXP_s114
                                 + 0.721818 w_k1_DIS_s114
                                 + 0.82214 w_k1_FDX_s114
                                 + 1.01294 w_k1_IBM_s114
                                 + 1.06516 w_k1_INTC_s114
                                 + 1.02599 w_k1_KO_s114
                                 + 0.966706 w_k1_MAT_s114
                                 + 0.935402 w_k1_MSFT_s114
                                 + 0.962087 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.40456 w_k1_ADM_s114
                                 + 1.64689 w_k1_AXP_s114
                                 + 1.34144 w_k1_DIS_s114
                                 + 1.30713 w_k1_FDX_s114
                                 + 1.10867 w_k1_IBM_s114
                                 + 1.54922 w_k1_INTC_s114 + 1.2159 w_k1_KO_s114
                                 + 0.975891 w_k1_MAT_s114
                                 + 1.31667 w_k1_MSFT_s114
                                 + 1.11536 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.20986 w_k1_ADM_s114
                                 + 1.13157 w_k1_AXP_s114
                                 + 1.27583 w_k1_DIS_s114
                                 + 1.16427 w_k1_FDX_s114
                                 + 1.36957 w_k1_IBM_s114
                                 + 1.15181 w_k1_INTC_s114 + 1.3902 w_k1_KO_s114
                                 + 1.14682 w_k1_MAT_s114
                                 + 1.28725 w_k1_MSFT_s114
                                 + 1.07186 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.928078 w_k1_ADM_s114
                                 + 1.07336 w_k1_AXP_s114
                                 + 0.986301 w_k1_DIS_s114
                                 + 1.03959 w_k1_FDX_s114
                                 + 1.15721 w_k1_IBM_s114
                                 + 1.22609 w_k1_INTC_s114
                                 + 1.15004 w_k1_KO_s114
                                 + 0.786486 w_k1_MAT_s114
                                 + 0.967018 w_k1_MSFT_s114
                                 + 1.05036 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.704297 w_k1_ADM_s115
                                 + 0.812226 w_k1_AXP_s115
                                 + 0.876604 w_k1_DIS_s115
                                 + 0.83978 w_k1_FDX_s115
                                 + 0.817353 w_k1_IBM_s115
                                 + 0.982021 w_k1_INTC_s115
                                 + 0.828976 w_k1_KO_s115
                                 + 0.762287 w_k1_MAT_s115
                                 + 1.08077 w_k1_MSFT_s115
                                 + 0.926205 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.881551 w_k1_ADM_s115
                                 + 0.856906 w_k1_AXP_s115
                                 + 0.782336 w_k1_DIS_s115
                                 + 0.903014 w_k1_FDX_s115
                                 + 0.549528 w_k1_IBM_s115
                                 + 0.905704 w_k1_INTC_s115
                                 + 0.62299 w_k1_KO_s115
                                 + 0.436033 w_k1_MAT_s115
                                 + 0.910891 w_k1_MSFT_s115
                                 + 0.84272 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.08431 w_k1_ADM_s115
                                 + 1.01507 w_k1_AXP_s115
                                 + 1.27532 w_k1_DIS_s115 + 1.0235 w_k1_FDX_s115
                                 + 1.09446 w_k1_IBM_s115
                                 + 0.974946 w_k1_INTC_s115
                                 + 1.14493 w_k1_KO_s115
                                 + 0.962409 w_k1_MAT_s115
                                 + 1.17374 w_k1_MSFT_s115
                                 + 1.02602 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.48958 w_k1_ADM_s115
                                 + 1.33904 w_k1_AXP_s115
                                 + 1.71552 w_k1_DIS_s115
                                 + 1.17396 w_k1_FDX_s115
                                 + 1.56873 w_k1_IBM_s115
                                 + 1.1716 w_k1_INTC_s115 + 1.48536 w_k1_KO_s115
                                 + 1.38332 w_k1_MAT_s115
                                 + 1.31541 w_k1_MSFT_s115
                                 + 1.14604 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.06087 w_k1_ADM_s115
                                 + 0.841807 w_k1_AXP_s115
                                 + 0.686729 w_k1_DIS_s115
                                 + 0.874365 w_k1_FDX_s115
                                 + 0.782396 w_k1_IBM_s115
                                 + 0.911818 w_k1_INTC_s115
                                 + 0.793147 w_k1_KO_s115
                                 + 0.553783 w_k1_MAT_s115
                                 + 0.966487 w_k1_MSFT_s115
                                 + 0.988331 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.01565 w_k1_ADM_s115
                                 + 1.06466 w_k1_AXP_s115
                                 + 1.16149 w_k1_DIS_s115
                                 + 0.85978 w_k1_FDX_s115
                                 + 1.64161 w_k1_IBM_s115
                                 + 0.869708 w_k1_INTC_s115
                                 + 1.31808 w_k1_KO_s115 + 1.71789 w_k1_MAT_s115
                                 + 1.02367 w_k1_MSFT_s115
                                 + 1.15646 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.866076 w_k1_ADM_s115
                                 + 1.20577 w_k1_AXP_s115
                                 + 0.990512 w_k1_DIS_s115
                                 + 1.00679 w_k1_FDX_s115
                                 + 1.21595 w_k1_IBM_s115
                                 + 0.994634 w_k1_INTC_s115
                                 + 1.18007 w_k1_KO_s115 + 1.23519 w_k1_MAT_s115
                                 + 1.09923 w_k1_MSFT_s115
                                 + 1.10053 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.17915 w_k1_ADM_s115
                                 + 1.22658 w_k1_AXP_s115
                                 + 1.05838 w_k1_DIS_s115
                                 + 1.15463 w_k1_FDX_s115
                                 + 0.877857 w_k1_IBM_s115
                                 + 0.919309 w_k1_INTC_s115
                                 + 0.829441 w_k1_KO_s115
                                 + 0.985408 w_k1_MAT_s115
                                 + 1.0583 w_k1_MSFT_s115
                                 + 0.988918 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.924076 w_k1_ADM_s116
                                 + 0.960454 w_k1_AXP_s116
                                 + 0.853164 w_k1_DIS_s116
                                 + 0.976107 w_k1_FDX_s116
                                 + 0.867872 w_k1_IBM_s116
                                 + 0.836575 w_k1_INTC_s116
                                 + 0.807973 w_k1_KO_s116
                                 + 0.897984 w_k1_MAT_s116
                                 + 0.979368 w_k1_MSFT_s116
                                 + 0.984624 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.86175 w_k1_ADM_s116
                                 + 1.09455 w_k1_AXP_s116
                                 + 1.01843 w_k1_DIS_s116
                                 + 0.936457 w_k1_FDX_s116
                                 + 0.99324 w_k1_IBM_s116
                                 + 0.887739 w_k1_INTC_s116
                                 + 1.13337 w_k1_KO_s116 + 1.13086 w_k1_MAT_s116
                                 + 0.983145 w_k1_MSFT_s116
                                 + 1.02785 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.836869 w_k1_ADM_s116
                                 + 0.858945 w_k1_AXP_s116
                                 + 0.905465 w_k1_DIS_s116
                                 + 0.95256 w_k1_FDX_s116
                                 + 0.641494 w_k1_IBM_s116
                                 + 0.866284 w_k1_INTC_s116
                                 + 0.856276 w_k1_KO_s116
                                 + 0.758975 w_k1_MAT_s116
                                 + 0.959671 w_k1_MSFT_s116
                                 + 0.887772 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.811551 w_k1_ADM_s116
                                 + 0.797066 w_k1_AXP_s116
                                 + 0.635177 w_k1_DIS_s116
                                 + 0.73516 w_k1_FDX_s116
                                 + 0.77753 w_k1_IBM_s116
                                 + 0.747291 w_k1_INTC_s116
                                 + 0.751314 w_k1_KO_s116
                                 + 0.822094 w_k1_MAT_s116
                                 + 0.739603 w_k1_MSFT_s116
                                 + 0.779102 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.1541 w_k1_ADM_s116
                                 + 1.19387 w_k1_AXP_s116
                                 + 1.31847 w_k1_DIS_s116
                                 + 1.12389 w_k1_FDX_s116
                                 + 1.03969 w_k1_IBM_s116
                                 + 0.95829 w_k1_INTC_s116
                                 + 1.27827 w_k1_KO_s116 + 1.51552 w_k1_MAT_s116
                                 + 1.30338 w_k1_MSFT_s116
                                 + 1.11967 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.39446 w_k1_ADM_s116
                                 + 1.49771 w_k1_AXP_s116
                                 + 1.62692 w_k1_DIS_s116
                                 + 1.39588 w_k1_FDX_s116
                                 + 1.24593 w_k1_IBM_s116
                                 + 0.984077 w_k1_INTC_s116
                                 + 1.21671 w_k1_KO_s116 + 1.46868 w_k1_MAT_s116
                                 + 1.10517 w_k1_MSFT_s116
                                 + 1.19579 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.04569 w_k1_ADM_s116
                                 + 1.02661 w_k1_AXP_s116
                                 + 1.24605 w_k1_DIS_s116
                                 + 1.16928 w_k1_FDX_s116
                                 + 0.828734 w_k1_IBM_s116
                                 + 0.814577 w_k1_INTC_s116
                                 + 1.0104 w_k1_KO_s116 + 0.945841 w_k1_MAT_s116
                                 + 1.10922 w_k1_MSFT_s116
                                 + 1.08164 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.07205 w_k1_ADM_s116
                                 + 1.04192 w_k1_AXP_s116
                                 + 1.03103 w_k1_DIS_s116
                                 + 0.959332 w_k1_FDX_s116
                                 + 1.37285 w_k1_IBM_s116
                                 + 1.00685 w_k1_INTC_s116
                                 + 1.14395 w_k1_KO_s116 + 1.27079 w_k1_MAT_s116
                                 + 1.12221 w_k1_MSFT_s116
                                 + 1.19679 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.30465 w_k1_ADM_s117
                                 + 1.16154 w_k1_AXP_s117
                                 + 1.32358 w_k1_DIS_s117
                                 + 1.26755 w_k1_FDX_s117
                                 + 1.08748 w_k1_IBM_s117
                                 + 1.12901 w_k1_INTC_s117
                                 + 1.22017 w_k1_KO_s117 + 1.49653 w_k1_MAT_s117
                                 + 1.22213 w_k1_MSFT_s117
                                 + 1.16592 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.715412 w_k1_ADM_s117
                                 + 0.662765 w_k1_AXP_s117
                                 + 0.531077 w_k1_DIS_s117
                                 + 0.690785 w_k1_FDX_s117
                                 + 0.83146 w_k1_IBM_s117
                                 + 0.78761 w_k1_INTC_s117
                                 + 0.66662 w_k1_KO_s117
                                 + 0.826414 w_k1_MAT_s117
                                 + 0.663466 w_k1_MSFT_s117
                                 + 0.772194 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.567298 w_k1_ADM_s117
                                 + 0.724805 w_k1_AXP_s117
                                 + 0.818835 w_k1_DIS_s117
                                 + 0.72927 w_k1_FDX_s117
                                 + 1.42249 w_k1_IBM_s117
                                 + 0.847045 w_k1_INTC_s117
                                 + 1.07651 w_k1_KO_s117 + 1.41479 w_k1_MAT_s117
                                 + 0.734924 w_k1_MSFT_s117
                                 + 0.91884 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.02247 w_k1_ADM_s117
                                 + 0.891984 w_k1_AXP_s117
                                 + 0.862861 w_k1_DIS_s117
                                 + 0.955851 w_k1_FDX_s117
                                 + 1.14182 w_k1_IBM_s117
                                 + 0.936533 w_k1_INTC_s117
                                 + 0.918743 w_k1_KO_s117
                                 + 1.20384 w_k1_MAT_s117
                                 + 0.859536 w_k1_MSFT_s117
                                 + 1.00338 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.896475 w_k1_ADM_s117
                                 + 0.839805 w_k1_AXP_s117
                                 + 0.860466 w_k1_DIS_s117
                                 + 1.07129 w_k1_FDX_s117
                                 + 0.715811 w_k1_IBM_s117
                                 + 1.08917 w_k1_INTC_s117
                                 + 0.811747 w_k1_KO_s117
                                 + 0.835119 w_k1_MAT_s117
                                 + 1.02772 w_k1_MSFT_s117
                                 + 0.956849 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.971308 w_k1_ADM_s117
                                 + 0.750824 w_k1_AXP_s117
                                 + 0.674148 w_k1_DIS_s117
                                 + 0.813207 w_k1_FDX_s117
                                 + 0.815496 w_k1_IBM_s117
                                 + 0.702649 w_k1_INTC_s117
                                 + 0.751014 w_k1_KO_s117
                                 + 0.892079 w_k1_MAT_s117
                                 + 0.769767 w_k1_MSFT_s117
                                 + 0.980945 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.24124 w_k1_ADM_s117
                                 + 0.952622 w_k1_AXP_s117
                                 + 0.94217 w_k1_DIS_s117
                                 + 0.995297 w_k1_FDX_s117
                                 + 1.40076 w_k1_IBM_s117
                                 + 0.879848 w_k1_INTC_s117
                                 + 1.28644 w_k1_KO_s117 + 1.71236 w_k1_MAT_s117
                                 + 0.940396 w_k1_MSFT_s117
                                 + 1.0914 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.06693 w_k1_ADM_s117
                                 + 1.52061 w_k1_AXP_s117
                                 + 1.60559 w_k1_DIS_s117
                                 + 1.26926 w_k1_FDX_s117
                                 + 1.75203 w_k1_IBM_s117
                                 + 1.10844 w_k1_INTC_s117
                                 + 1.58399 w_k1_KO_s117 + 2.46071 w_k1_MAT_s117
                                 + 1.13182 w_k1_MSFT_s117
                                 + 1.19385 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0267731390521857 w_k1_ADM_s000
                                 + 0.0513671219870913 w_k1_ADM_s001
                                 + 0.0122032298028916 w_k1_ADM_s002
                                 + 0.0179966857496851 w_k1_ADM_s003
                                 + 0.0224663346045124 w_k1_ADM_s004
                                 + 0.0176001420979166 w_k1_ADM_s005
                                 + 0.0147422937699388 w_k1_ADM_s006
                                 + 0.0444171073709273 w_k1_ADM_s007
                                 + 0.0849348515021618 w_k1_ADM_s010
                                 + 0.0469458457131781 w_k1_ADM_s011
                                 + 0.0547118800415358 w_k1_ADM_s012
                                 + 0.052987034910441 w_k1_ADM_s013
                                 + 0.101768359194557 w_k1_ADM_s014
                                 + 0.0334203050657627 w_k1_ADM_s015
                                 + 0.0409394980627771 w_k1_ADM_s016
                                 + 0.0324634585915948 w_k1_ADM_s017
                                 + 0.00542432886024039 w_k1_ADM_s100
                                 + 0.00539049184210028 w_k1_ADM_s101
                                 + 0.0102539653423825 w_k1_ADM_s102
                                 + 0.0080576921148202 w_k1_ADM_s103
                                 + 0.0121963753433812 w_k1_ADM_s104
                                 + 0.0265026610965124 w_k1_ADM_s105
                                 + 0.00902057206176612 w_k1_ADM_s106
                                 + 0.00927750058284959 w_k1_ADM_s107
                                 + 0.0661104400178177 w_k1_ADM_s110
                                 + 0.0688514880342711 w_k1_ADM_s111
                                 + 0.0151963782671338 w_k1_ADM_s112
                                 + 0.020345120837373 w_k1_ADM_s113
                                 + 0.013885740427697 w_k1_ADM_s114
                                 + 0.0533214898769193 w_k1_ADM_s115
                                 + 0.0245495330593496 w_k1_ADM_s116
                                 + 0.0301101408811571 w_k1_ADM_s117
                                 + 0.0237890483051018 w_k1_AXP_s000
                                 + 0.0493025835658134 w_k1_AXP_s001
                                 + 0.0115297579863527 w_k1_AXP_s002
                                 + 0.0163095672215574 w_k1_AXP_s003
                                 + 0.0227428946010678 w_k1_AXP_s004
                                 + 0.0197394522859977 w_k1_AXP_s005
                                 + 0.0132071457847613 w_k1_AXP_s006
                                 + 0.0454429477577537 w_k1_AXP_s007
                                 + 0.0834363881884311 w_k1_AXP_s010
                                 + 0.0526465116406732 w_k1_AXP_s011
                                 + 0.0593914135118144 w_k1_AXP_s012
                                 + 0.0532096342292636 w_k1_AXP_s013
                                 + 0.0985455537118615 w_k1_AXP_s014
                                 + 0.0411927892733258 w_k1_AXP_s015
                                 + 0.0366924510636777 w_k1_AXP_s016
                                 + 0.0286877590431821 w_k1_AXP_s017
                                 + 0.0058578350684105 w_k1_AXP_s100
                                 + 0.00598749150268564 w_k1_AXP_s101
                                 + 0.0120523638875426 w_k1_AXP_s102
                                 + 0.00594867630037209 w_k1_AXP_s103
                                 + 0.0099992316109523 w_k1_AXP_s104
                                 + 0.0278871261113888 w_k1_AXP_s105
                                 + 0.00847660718090525 w_k1_AXP_s106
                                 + 0.00980773973838568 w_k1_AXP_s107
                                 + 0.0760853688096951 w_k1_AXP_s110
                                 + 0.0707316588111118 w_k1_AXP_s111
                                 + 0.016870623877815 w_k1_AXP_s112
                                 + 0.0207140990704805 w_k1_AXP_s113
                                 + 0.0138513821603668 w_k1_AXP_s114
                                 + 0.0554732682753072 w_k1_AXP_s115
                                 + 0.0259651201912247 w_k1_AXP_s116
                                 + 0.0277071014573619 w_k1_AXP_s117
                                 + 0.0251592562292001 w_k1_DIS_s000
                                 + 0.0557314396575672 w_k1_DIS_s001
                                 + 0.0117830197183385 w_k1_DIS_s002
                                 + 0.0171495012284322 w_k1_DIS_s003
                                 + 0.0221606405870832 w_k1_DIS_s004
                                 + 0.0216164311072804 w_k1_DIS_s005
                                 + 0.0144764995846116 w_k1_DIS_s006
                                 + 0.04582827612988 w_k1_DIS_s007
                                 + 0.0901024600958613 w_k1_DIS_s010
                                 + 0.0504595819551264 w_k1_DIS_s011
                                 + 0.050842532716032 w_k1_DIS_s012
                                 + 0.0514719199688596 w_k1_DIS_s013
                                 + 0.0952471824098032 w_k1_DIS_s014
                                 + 0.0379473382775808 w_k1_DIS_s015
                                 + 0.0395803898396228 w_k1_DIS_s016
                                 + 0.0289869963189614 w_k1_DIS_s017
                                 + 0.00571733962722881 w_k1_DIS_s100
                                 + 0.00687864138285637 w_k1_DIS_s101
                                 + 0.0111078678971521 w_k1_DIS_s102
                                 + 0.00648273946883665 w_k1_DIS_s103
                                 + 0.0115280604250968 w_k1_DIS_s104
                                 + 0.0250749383557038 w_k1_DIS_s105
                                 + 0.00808059425756452 w_k1_DIS_s106
                                 + 0.00969051195893235 w_k1_DIS_s107
                                 + 0.0677863263694116 w_k1_DIS_s110
                                 + 0.0677368150676019 w_k1_DIS_s111
                                 + 0.0174131965001827 w_k1_DIS_s112
                                 + 0.0201043850200412 w_k1_DIS_s113
                                 + 0.0135327447687174 w_k1_DIS_s114
                                 + 0.0564744147523014 w_k1_DIS_s115
                                 + 0.0260027067821824 w_k1_DIS_s116
                                 + 0.0286576734104059 w_k1_DIS_s117
                                 + 0.0232050949562588 w_k1_FDX_s000
                                 + 0.0495090540127702 w_k1_FDX_s001
                                 + 0.011818967254754 w_k1_FDX_s002
                                 + 0.0175065372847233 w_k1_FDX_s003
                                 + 0.0227861651831546 w_k1_FDX_s004
                                 + 0.0194054603688781 w_k1_FDX_s005
                                 + 0.0140733540313793 w_k1_FDX_s006
                                 + 0.0425592201658237 w_k1_FDX_s007
                                 + 0.0840986434110844 w_k1_FDX_s010
                                 + 0.049826020812413 w_k1_FDX_s011
                                 + 0.051429997398439 w_k1_FDX_s012
                                 + 0.0521366446715516 w_k1_FDX_s013
                                 + 0.0964137170473943 w_k1_FDX_s014
                                 + 0.036329894330253 w_k1_FDX_s015
                                 + 0.0396242552676276 w_k1_FDX_s016
                                 + 0.0269797036958213 w_k1_FDX_s017
                                 + 0.00602686222703989 w_k1_FDX_s100
                                 + 0.00632304387590462 w_k1_FDX_s101
                                 + 0.0108491012572009 w_k1_FDX_s102
                                 + 0.00600422531437055 w_k1_FDX_s103
                                 + 0.0106583258214199 w_k1_FDX_s104
                                 + 0.0255832967880288 w_k1_FDX_s105
                                 + 0.00811021819090318 w_k1_FDX_s106
                                 + 0.00994068027906886 w_k1_FDX_s107
                                 + 0.0727205145915383 w_k1_FDX_s110
                                 + 0.0690190521614473 w_k1_FDX_s111
                                 + 0.0166120229313696 w_k1_FDX_s112
                                 + 0.0182992432131579 w_k1_FDX_s113
                                 + 0.0139956682067986 w_k1_FDX_s114
                                 + 0.0520600392381381 w_k1_FDX_s115
                                 + 0.0245046292956954 w_k1_FDX_s116
                                 + 0.0300183110728157 w_k1_FDX_s117
                                 + 0.0248264027578725 w_k1_IBM_s000
                                 + 0.0479323498837114 w_k1_IBM_s001
                                 + 0.012600308500223 w_k1_IBM_s002
                                 + 0.0182284279860018 w_k1_IBM_s003
                                 + 0.0228941434163812 w_k1_IBM_s004
                                 + 0.0200469815611551 w_k1_IBM_s005
                                 + 0.0135762957047226 w_k1_IBM_s006
                                 + 0.0445419826632339 w_k1_IBM_s007
                                 + 0.0843263026502409 w_k1_IBM_s010
                                 + 0.050276478361349 w_k1_IBM_s011
                                 + 0.0541701722611761 w_k1_IBM_s012
                                 + 0.0503077084934856 w_k1_IBM_s013
                                 + 0.0946273642113697 w_k1_IBM_s014
                                 + 0.0369692989593845 w_k1_IBM_s015
                                 + 0.040106236859058 w_k1_IBM_s016
                                 + 0.0260753681271502 w_k1_IBM_s017
                                 + 0.00596290961463479 w_k1_IBM_s100
                                 + 0.00648978544404808 w_k1_IBM_s101
                                 + 0.0110961235833623 w_k1_IBM_s102
                                 + 0.00646139432655966 w_k1_IBM_s103
                                 + 0.0102577858073518 w_k1_IBM_s104
                                 + 0.0273559267244171 w_k1_IBM_s105
                                 + 0.00836386693272322 w_k1_IBM_s106
                                 + 0.00970716445129546 w_k1_IBM_s107
                                 + 0.0677698256229829 w_k1_IBM_s110
                                 + 0.070201429744018 w_k1_IBM_s111
                                 + 0.0163623059611016 w_k1_IBM_s112
                                 + 0.0190256106248369 w_k1_IBM_s113
                                 + 0.0140553641160863 w_k1_IBM_s114
                                 + 0.0564983758749371 w_k1_IBM_s115
                                 + 0.0249343195130724 w_k1_IBM_s116
                                 + 0.0310882735602659 w_k1_IBM_s117
                                 + 0.021015645288576 w_k1_INTC_s000
                                 + 0.0482745248231901 w_k1_INTC_s001
                                 + 0.0122604010161783 w_k1_INTC_s002
                                 + 0.0160699485370129 w_k1_INTC_s003
                                 + 0.0221619871026065 w_k1_INTC_s004
                                 + 0.0194300184440819 w_k1_INTC_s005
                                 + 0.013382698735675 w_k1_INTC_s006
                                 + 0.0450608311265367 w_k1_INTC_s007
                                 + 0.087888957064152 w_k1_INTC_s010
                                 + 0.0509834880078902 w_k1_INTC_s011
                                 + 0.0491511283072417 w_k1_INTC_s012
                                 + 0.0515120881423677 w_k1_INTC_s013
                                 + 0.111705129764136 w_k1_INTC_s014
                                 + 0.0390557497761546 w_k1_INTC_s015
                                 + 0.0392341947975215 w_k1_INTC_s016
                                 + 0.0256848987147483 w_k1_INTC_s017
                                 + 0.00779390031242608 w_k1_INTC_s100
                                 + 0.00645282266205521 w_k1_INTC_s101
                                 + 0.0116664701421203 w_k1_INTC_s102
                                 + 0.0056491751415124 w_k1_INTC_s103
                                 + 0.0107563425058479 w_k1_INTC_s104
                                 + 0.0259707019004691 w_k1_INTC_s105
                                 + 0.00869962914669763 w_k1_INTC_s106
                                 + 0.0103600697429915 w_k1_INTC_s107
                                 + 0.0699588673123214 w_k1_INTC_s110
                                 + 0.0690118054851715 w_k1_INTC_s111
                                 + 0.0193589034761069 w_k1_INTC_s112
                                 + 0.0156473033560398 w_k1_INTC_s113
                                 + 0.0160057958662542 w_k1_INTC_s114
                                 + 0.0510031402747006 w_k1_INTC_s115
                                 + 0.0217308729042903 w_k1_INTC_s116
                                 + 0.0285166447648134 w_k1_INTC_s117
                                 + 0.0247923130781086 w_k1_KO_s000
                                 + 0.0510885685033349 w_k1_KO_s001
                                 + 0.0113825614461591 w_k1_KO_s002
                                 + 0.0181546882936337 w_k1_KO_s003
                                 + 0.0223518772256325 w_k1_KO_s004
                                 + 0.0220855314811246 w_k1_KO_s005
                                 + 0.0137088017216681 w_k1_KO_s006
                                 + 0.0478273774483914 w_k1_KO_s007
                                 + 0.0896707221684083 w_k1_KO_s010
                                 + 0.0538472917941728 w_k1_KO_s011
                                 + 0.0561772416103764 w_k1_KO_s012
                                 + 0.0509379006187632 w_k1_KO_s013
                                 + 0.0901831069209855 w_k1_KO_s014
                                 + 0.04245512956417 w_k1_KO_s015
                                 + 0.0380548599865477 w_k1_KO_s016
                                 + 0.0275272899636895 w_k1_KO_s017
                                 + 0.00586024358905182 w_k1_KO_s100
                                 + 0.00665856256312093 w_k1_KO_s101
                                 + 0.0105598982320541 w_k1_KO_s102
                                 + 0.006965728949059 w_k1_KO_s103
                                 + 0.0100584012762961 w_k1_KO_s104
                                 + 0.0235707957499315 w_k1_KO_s105
                                 + 0.00713071855687322 w_k1_KO_s106
                                 + 0.00952689600068175 w_k1_KO_s107
                                 + 0.0671549814285804 w_k1_KO_s110
                                 + 0.073909107582712 w_k1_KO_s111
                                 + 0.0159420762059606 w_k1_KO_s112
                                 + 0.0195767558311444 w_k1_KO_s113
                                 + 0.0145222287361282 w_k1_KO_s114
                                 + 0.0542324227186653 w_k1_KO_s115
                                 + 0.0253666958008833 w_k1_KO_s116
                                 + 0.0298175852166974 w_k1_KO_s117
                                 + 0.0290096942351181 w_k1_MAT_s000
                                 + 0.0517190390891527 w_k1_MAT_s001
                                 + 0.0112029993825568 w_k1_MAT_s002
                                 + 0.018084941461707 w_k1_MAT_s003
                                 + 0.0208580530282426 w_k1_MAT_s004
                                 + 0.0215141102511169 w_k1_MAT_s005
                                 + 0.0126580709670073 w_k1_MAT_s006
                                 + 0.0503859535496495 w_k1_MAT_s007
                                 + 0.0740049042911323 w_k1_MAT_s010
                                 + 0.0510630345193814 w_k1_MAT_s011
                                 + 0.0626307207313116 w_k1_MAT_s012
                                 + 0.0516469986199164 w_k1_MAT_s013
                                 + 0.101573368515381 w_k1_MAT_s014
                                 + 0.0351573880781056 w_k1_MAT_s015
                                 + 0.0347899440121158 w_k1_MAT_s016
                                 + 0.0279483012574987 w_k1_MAT_s017
                                 + 0.00519469134175434 w_k1_MAT_s100
                                 + 0.00522601181923631 w_k1_MAT_s101
                                 + 0.0140246103367516 w_k1_MAT_s102
                                 + 0.00539184377377121 w_k1_MAT_s103
                                 + 0.00971153435476979 w_k1_MAT_s104
                                 + 0.0271415396071386 w_k1_MAT_s105
                                 + 0.00685034601363768 w_k1_MAT_s106
                                 + 0.0111351258799333 w_k1_MAT_s107
                                 + 0.0797675377507339 w_k1_MAT_s110
                                 + 0.0652034744960064 w_k1_MAT_s111
                                 + 0.0142552944051335 w_k1_MAT_s112
                                 + 0.0214275576007352 w_k1_MAT_s113
                                 + 0.0125967873110034 w_k1_MAT_s114
                                 + 0.0543837569396918 w_k1_MAT_s115
                                 + 0.0276378731465322 w_k1_MAT_s116
                                 + 0.0372835036278644 w_k1_MAT_s117
                                 + 0.0215358552092341 w_k1_MSFT_s000
                                 + 0.0524631666436789 w_k1_MSFT_s001
                                 + 0.0117930111901284 w_k1_MSFT_s002
                                 + 0.0160231821532667 w_k1_MSFT_s003
                                 + 0.0204963027416688 w_k1_MSFT_s004
                                 + 0.0194406744376041 w_k1_MSFT_s005
                                 + 0.0130967930571847 w_k1_MSFT_s006
                                 + 0.0444285974512661 w_k1_MSFT_s007
                                 + 0.0901460006700383 w_k1_MSFT_s010
                                 + 0.049731399356054 w_k1_MSFT_s011
                                 + 0.054003595231729 w_k1_MSFT_s012
                                 + 0.0541418519655085 w_k1_MSFT_s013
                                 + 0.0949693318639873 w_k1_MSFT_s014
                                 + 0.0354892221759938 w_k1_MSFT_s015
                                 + 0.0376285133246824 w_k1_MSFT_s016
                                 + 0.0264711225339923 w_k1_MSFT_s017
                                 + 0.00599137258164527 w_k1_MSFT_s100
                                 + 0.00559992612270216 w_k1_MSFT_s101
                                 + 0.0112012459395443 w_k1_MSFT_s102
                                 + 0.00582787242000002 w_k1_MSFT_s103
                                 + 0.0101043110141439 w_k1_MSFT_s104
                                 + 0.0264751146824173 w_k1_MSFT_s105
                                 + 0.00739861526200841 w_k1_MSFT_s106
                                 + 0.00999695251447443 w_k1_MSFT_s107
                                 + 0.0657816592935099 w_k1_MSFT_s110
                                 + 0.0681904399974972 w_k1_MSFT_s111
                                 + 0.0167150877809612 w_k1_MSFT_s112
                                 + 0.0196177525019121 w_k1_MSFT_s113
                                 + 0.0146729577189558 w_k1_MSFT_s114
                                 + 0.0576047832621758 w_k1_MSFT_s115
                                 + 0.024893949626492 w_k1_MSFT_s116
                                 + 0.0284997377073662 w_k1_MSFT_s117
                                 + 0.0250587075201482 w_k1_PG_s000
                                 + 0.0499003824474926 w_k1_PG_s001
                                 + 0.0118657396168634 w_k1_PG_s002
                                 + 0.0167910511924503 w_k1_PG_s003
                                 + 0.0229626170612934 w_k1_PG_s004
                                 + 0.0213629225416846 w_k1_PG_s005
                                 + 0.0142132783423177 w_k1_PG_s006
                                 + 0.0475246818084079 w_k1_PG_s007
                                 + 0.0814516593109294 w_k1_PG_s010
                                 + 0.0509263055911373 w_k1_PG_s011
                                 + 0.0548018032663195 w_k1_PG_s012
                                 + 0.0501939359177293 w_k1_PG_s013
                                 + 0.090976610205512 w_k1_PG_s014
                                 + 0.0382582091889426 w_k1_PG_s015
                                 + 0.039353063736511 w_k1_PG_s016
                                 + 0.027847730489872 w_k1_PG_s017
                                 + 0.00568102614374204 w_k1_PG_s100
                                 + 0.00582746817875652 w_k1_PG_s101
                                 + 0.0110160268692695 w_k1_PG_s102
                                 + 0.00630650277394601 w_k1_PG_s103
                                 + 0.0100363912399461 w_k1_PG_s104
                                 + 0.0263346782920464 w_k1_PG_s105
                                 + 0.007491505166921 w_k1_PG_s106
                                 + 0.00918680813136065 w_k1_PG_s107
                                 + 0.0654488114636643 w_k1_PG_s110
                                 + 0.0689146855532349 w_k1_PG_s111
                                 + 0.0169260881135787 w_k1_PG_s112
                                 + 0.0184870625437317 w_k1_PG_s113
                                 + 0.0138108331394394 w_k1_PG_s114
                                 + 0.0542407405178228 w_k1_PG_s115
                                 + 0.0252842329350647 w_k1_PG_s116
                                 + 0.0301871331501731 w_k1_PG_s117 >= 1.229
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

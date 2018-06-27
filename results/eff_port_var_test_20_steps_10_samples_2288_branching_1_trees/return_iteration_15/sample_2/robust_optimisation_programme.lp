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
 w_k1_ADM_s0_return_constraint:  1.08525 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.879336 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.878311 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.0583 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.07315 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.03192 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.43427 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.1075 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.36958 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.782085 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.9224 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.848258 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.02512 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.42052 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.53977 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.59831 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.14172 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.836047 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.855746 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.928746 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.00359 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.13444 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.02813 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.0232 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.70692 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.40639 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.955664 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.761343 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.23003 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.02724 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.25507 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.737012 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.948779 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.668354 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.980319 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.3312 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.04028 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.855666 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.09652 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.82442 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.967836 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.29591 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.0935 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.923275 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.82521 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.17315 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.15211 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.775262 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.770521 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.692768 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.02515 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.22659 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.68669 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.82181 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.35272 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.902897 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.558611 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.897701 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.01823 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.16471 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.977481 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.10323 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.639702 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.6828 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.882645 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.676561 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.33173 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.20315 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.46241 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.658397 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.920548 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.658312 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.0509 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.42181 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.02367 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.784602 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.12746 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.758769 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.966696 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.5339 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.1566 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.952053 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.70945 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.14788 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.10956 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.578144 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.708029 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.831338 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.859189 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.735 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.42526 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.93884 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.19842 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.662985 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.513275 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.788371 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.932825 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.14838 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.93897 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.05088 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.529659 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.89823 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.793858 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.589294 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.32971 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.2247 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.44011 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.64402 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.03457 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.533492 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.0911 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 2.00496 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.09672 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.747271 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.08523 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.871878 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.973077 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.19218 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.03452 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.00206 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.496 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.02118 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.25037 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.696639 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.817407 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.806028 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.9931 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.23493 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.39876 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.46134 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.08553 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.691795 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.658227 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.855631 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.964461 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.18587 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.989756 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.991737 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.699272 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.61518 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.791844 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.665131 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.2981 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.05509 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.40189 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.68269 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.95553 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.710437 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.15036 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.54078 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.02298 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.844135 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.1206 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.811854 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.925913 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.59194 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.18277 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.775117 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.16046 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.34108 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.798958 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.17583 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.691196 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.21992 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.966542 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.43116 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.995196 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.33374 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.20221 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.22473 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.712487 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.824567 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.1059 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.667137 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.708681 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.06558 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.746118 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.24294 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.30966 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.08063 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.993821 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.74569 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.885701 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.01607 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.41731 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.602614 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.817772 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.27783 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.09539 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.785708 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.10825 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.923495 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.96573 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.06461 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.983776 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.963987 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.31411 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.08242 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.05367 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.7682 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.742773 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.849553 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.955678 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.09381 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.29286 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.55622 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.03465 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.05373 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.713318 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.870289 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.955626 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.993582 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.942341 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.905501 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.724589 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.54347 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.02807 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.701741 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.11144 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.00273 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.09712 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.706971 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.965366 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.803678 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.992509 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.38178 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.976521 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.804812 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.12197 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.81804 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.982602 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.3313 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.06761 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.933262 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.29496 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.18631 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.92457 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.816009 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.689653 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.04191 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.901334 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.40768 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.24575 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.66709 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.21603 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.03057 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.629238 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.817731 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.02004 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.899727 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.924382 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.11668 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.67505 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.45817 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.19764 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.802499 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.19528 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.62083 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.922052 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.829987 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.11908 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.553676 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.835567 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.36579 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.0747 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.719223 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.0276 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.739793 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.06446 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.83928 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.29503 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.838815 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.73259 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.58402 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.967001 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.962223 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.616289 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.969023 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.016 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.37697 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.17952 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.32717 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.39538 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.14849 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.488756 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.707469 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.108 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.778022 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.685748 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.1426 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.817079 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.13531 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.09333 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.978576 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.00507 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.95883 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.894527 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.923832 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.32762 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.566475 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.791014 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.58412 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.957977 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.702057 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.09769 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.819022 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.995833 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.19946 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.17169 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.02609 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.27491 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.01213 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.07304 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.761245 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.734556 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.86925 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.945613 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.2669 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.30885 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.27807 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.04095 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.780504 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.649813 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.800899 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.890706 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.981905 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.890729 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.915671 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.706978 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.39968 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.866438 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.709126 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.1261 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.917549 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.2582 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.7601 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.10377 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.770036 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.02421 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.47887 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.09169 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.829544 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.04989 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.885701 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.973464 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.23709 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.0109 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.941325 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.25328 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.0762 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.08566 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.879954 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.836359 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.864581 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.02993 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.15608 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.23665 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.30789 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.16276 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.906026 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.752003 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.886538 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.896479 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.985114 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.04991 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.15721 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.802549 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.29786 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.01158 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.833404 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.1576 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.2333 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.25067 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.811627 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.14582 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.709885 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.01358 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.25662 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.11365 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.885611 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.722000323174993 z_k1_s0000
                                 - 0.493000562482899 z_k1_s0001
                                 - 1.50200269210237 z_k1_s0002
                                 - 0.436000389368975 z_k1_s0003
                                 - 1.18800080365656 z_k1_s0004
                                 - 0.564000310715688 z_k1_s0005
                                 - 1.0380003480936 z_k1_s0006
                                 - 1.06800043920619 z_k1_s0007
                                 - 0.989001502241238 z_k1_s0010
                                 - 0.762001216492754 z_k1_s0011
                                 - 1.9670036767268 z_k1_s0012
                                 - 0.736001035630534 z_k1_s0013
                                 - 1.015999852187 z_k1_s0014
                                 - 0.980002052259318 z_k1_s0015
                                 - 1.77900631697665 z_k1_s0016
                                 - 0.620001045539092 z_k1_s0017
                                 - 1.69000058478222 z_k1_s0020
                                 - 0.52000017993299 z_k1_s0021
                                 - 1.02300323952479 z_k1_s0022
                                 - 1.26399723121865 z_k1_s0023
                                 - 0.633000603772644 z_k1_s0024
                                 - 0.81500066674919 z_k1_s0025
                                 - 0.449000732473463 z_k1_s0026
                                 - 1.94199810705497 z_k1_s0027
                                 - 0.492000631768743 z_k1_s0030
                                 - 1.07300139530483 z_k1_s0031
                                 - 0.759004335576663 z_k1_s0032
                                 - 1.46000394593939 z_k1_s0033
                                 - 0.961001854012205 z_k1_s0034
                                 - 0.67400220232338 z_k1_s0035
                                 - 0.53800077422166 z_k1_s0036
                                 - 0.660001267110036 z_k1_s0037
                                 - 1.18899901629391 z_k1_s0040
                                 - 2.18399276841342 z_k1_s0041
                                 - 1.51599401198698 z_k1_s0042
                                 - 0.679999878634866 z_k1_s0043
                                 - 1.34599912582632 z_k1_s0044
                                 - 3.13500002856199 z_k1_s0045
                                 - 1.76000191863056 z_k1_s0046
                                 - 1.74599857598819 z_k1_s0047
                                 - 0.194000119277897 z_k1_s0050
                                 - 0.607999393051433 z_k1_s0051
                                 - 0.372001573960972 z_k1_s0052
                                 - 0.415001721338148 z_k1_s0053
                                 - 0.671999836636282 z_k1_s0054
                                 - 0.237000266655072 z_k1_s0055
                                 - 0.372001573960972 z_k1_s0056
                                 - 0.344001179017407 z_k1_s0057
                                 - 0.688998576198375 z_k1_s0060
                                 - 0.814001223112346 z_k1_s0061
                                 - 1.3329991905521 z_k1_s0062
                                 - 0.816002922041393 z_k1_s0063
                                 - 0.554000664258525 z_k1_s0064
                                 - 0.399000718250083 z_k1_s0065
                                 - 0.555000938522207 z_k1_s0066
                                 - 0.59200245826582 z_k1_s0067
                                 - 0.771001636225109 z_k1_s0070
                                 - 0.325999980338785 z_k1_s0071
                                 - 0.181000140151339 z_k1_s0072
                                 - 0.528998935800937 z_k1_s0073
                                 - 0.191000223474642 z_k1_s0074
                                 - 0.880999132780311 z_k1_s0075
                                 - 0.311000197353943 z_k1_s0076
                                 - 0.230000206435412 z_k1_s0077
                                 - 0.193999905245345 z_k1_s0100
                                 - 0.102999337322544 z_k1_s0101
                                 - 0.0839996357771591 z_k1_s0102
                                 - 0.0599998694833545 z_k1_s0103
                                 - 0.0719997938801516 z_k1_s0104
                                 - 0.123999658765782 z_k1_s0105
                                 - 0.105999854670375 z_k1_s0106
                                 - 0.0819997583774124 z_k1_s0107
                                 - 0.0769998326561277 z_k1_s0110
                                 - 0.0939998409424747 z_k1_s0111
                                 - 0.284999255831317 z_k1_s0112
                                 - 0.335999640989794 z_k1_s0113
                                 - 0.14500010600114 z_k1_s0114
                                 - 0.0719998372836616 z_k1_s0115
                                 - 0.107999876025304 z_k1_s0116
                                 - 0.0839998501975426 z_k1_s0117
                                 - 0.418000422155871 z_k1_s0120
                                 - 0.389998541287039 z_k1_s0121
                                 - 0.316000307266179 z_k1_s0122
                                 - 0.428998147216125 z_k1_s0123
                                 - 0.428000384752974 z_k1_s0124
                                 - 0.238999602470011 z_k1_s0125
                                 - 0.112999825546888 z_k1_s0126
                                 - 0.148999690896461 z_k1_s0127
                                 - 0.377000320207159 z_k1_s0130
                                 - 0.147999661035429 z_k1_s0131
                                 - 0.239999212706505 z_k1_s0132
                                 - 0.18899976465572 z_k1_s0133
                                 - 0.489999308440893 z_k1_s0134
                                 - 0.237999036100928 z_k1_s0135
                                 - 0.133999743594154 z_k1_s0136
                                 - 0.381998563724831 z_k1_s0137
                                 - 0.0919999505032487 z_k1_s0140
                                 - 0.10699997145479 z_k1_s0141
                                 - 0.0659998251874758 z_k1_s0142
                                 - 0.0579998140133206 z_k1_s0143
                                 - 0.227999606465278 z_k1_s0144
                                 - 0.144999312533881 z_k1_s0145
                                 - 0.0699998307745535 z_k1_s0146
                                 - 0.123999817200333 z_k1_s0147
                                 - 0.620997278332632 z_k1_s0150
                                 - 0.551996380745377 z_k1_s0151
                                 - 0.183999063580609 z_k1_s0152
                                 - 0.240999053764678 z_k1_s0153
                                 - 0.378997338954572 z_k1_s0154
                                 - 0.162999365616749 z_k1_s0155
                                 - 0.162999365616749 z_k1_s0156
                                 - 0.396998160061718 z_k1_s0157
                                 - 0.0659998531132047 z_k1_s0160
                                 - 0.25199931545063 z_k1_s0161
                                 - 0.0569999261613131 z_k1_s0162
                                 - 0.0779998529155601 z_k1_s0163
                                 - 0.0919998364850025 z_k1_s0164
                                 - 0.201999721273455 z_k1_s0165
                                 - 0.132999503710287 z_k1_s0166
                                 - 0.0919998364850025 z_k1_s0167
                                 - 0.235998915149848 z_k1_s0170
                                 - 0.145999242932099 z_k1_s0171
                                 - 0.165999282757723 z_k1_s0172
                                 - 0.198998233075368 z_k1_s0173
                                 - 0.154999294653467 z_k1_s0174
                                 - 0.141999370166324 z_k1_s0175
                                 - 0.487996983364421 z_k1_s0176
                                 - 0.157999199227799 z_k1_s0177
                                 - 0.451000352725291 z_k1_s1000
                                 - 0.225000298401003 z_k1_s1001
                                 - 0.170000212405188 z_k1_s1002
                                 - 0.331000271898177 z_k1_s1003
                                 - 0.320999775662035 z_k1_s1004
                                 - 0.330001690776213 z_k1_s1005
                                 - 0.683000117390605 z_k1_s1006
                                 - 0.426000580636584 z_k1_s1007
                                 - 0.62700200093069 z_k1_s1010
                                 - 0.645000648478349 z_k1_s1011
                                 - 0.26600010074604 z_k1_s1012
                                 - 0.452000991544742 z_k1_s1013
                                 - 0.31699998533263 z_k1_s1014
                                 - 0.461000315318571 z_k1_s1015
                                 - 0.830999946476383 z_k1_s1016
                                 - 0.383000070209492 z_k1_s1017
                                 - 0.334000685509718 z_k1_s1020
                                 - 0.540000820577971 z_k1_s1021
                                 - 0.14000017527569 z_k1_s1022
                                 - 0.293000449922872 z_k1_s1023
                                 - 0.455000948996804 z_k1_s1024
                                 - 0.250000529764196 z_k1_s1025
                                 - 0.331001285102242 z_k1_s1026
                                 - 0.186000507444004 z_k1_s1027
                                 - 0.235000464721605 z_k1_s1030
                                 - 0.221999712800728 z_k1_s1031
                                 - 0.229999803982752 z_k1_s1032
                                 - 0.106999971809376 z_k1_s1033
                                 - 0.227000373539582 z_k1_s1034
                                 - 0.293000723291214 z_k1_s1035
                                 - 0.169999925117696 z_k1_s1036
                                 - 0.126000067616664 z_k1_s1037
                                 - 0.0829999664151886 z_k1_s1040
                                 - 0.146999874359459 z_k1_s1041
                                 - 0.291999579995735 z_k1_s1042
                                 - 0.350999066132048 z_k1_s1043
                                 - 0.239999654791015 z_k1_s1044
                                 - 0.360999909747964 z_k1_s1045
                                 - 0.168999842715302 z_k1_s1046
                                 - 0.0729999807988962 z_k1_s1047
                                 - 0.0789999332404974 z_k1_s1050
                                 - 0.195000438713678 z_k1_s1051
                                 - 0.127000496177553 z_k1_s1052
                                 - 0.103999912114073 z_k1_s1053
                                 - 0.123999895212933 z_k1_s1054
                                 - 0.255999783665409 z_k1_s1055
                                 - 0.225999809017119 z_k1_s1056
                                 - 0.0959999188745285 z_k1_s1057
                                 - 0.397000705714928 z_k1_s1060
                                 - 0.153000242459366 z_k1_s1061
                                 - 0.247000653304035 z_k1_s1062
                                 - 0.342999552845476 z_k1_s1063
                                 - 0.220000076869309 z_k1_s1064
                                 - 0.11000022743491 z_k1_s1065
                                 - 0.186000253989762 z_k1_s1066
                                 - 0.234001215762523 z_k1_s1067
                                 - 0.105000040165149 z_k1_s1070
                                 - 0.279999696173736 z_k1_s1071
                                 - 0.0719999879161026 z_k1_s1072
                                 - 0.0920000102427974 z_k1_s1073
                                 - 0.168999926690575 z_k1_s1074
                                 - 0.345000616285482 z_k1_s1075
                                 - 0.162000704786221 z_k1_s1076
                                 - 0.316000537681775 z_k1_s1077
                                 - 0.093999765411717 z_k1_s1100
                                 - 0.168000330863793 z_k1_s1101
                                 - 0.0689999970618596 z_k1_s1102
                                 - 0.0830000698338603 z_k1_s1103
                                 - 0.27200014825551 z_k1_s1104
                                 - 0.0810000594378602 z_k1_s1105
                                 - 0.0790000490418601 z_k1_s1106
                                 - 0.058000030283895 z_k1_s1107
                                 - 0.254000190493958 z_k1_s1110
                                 - 0.319000090423286 z_k1_s1111
                                 - 0.286000639690701 z_k1_s1112
                                 - 0.101000135275248 z_k1_s1113
                                 - 0.111000230649161 z_k1_s1114
                                 - 0.157000201368441 z_k1_s1115
                                 - 0.128000176784481 z_k1_s1116
                                 - 0.444001282597198 z_k1_s1117
                                 - 0.314001347422254 z_k1_s1120
                                 - 0.153000401641285 z_k1_s1121
                                 - 0.143000417468779 z_k1_s1122
                                 - 0.126000392175361 z_k1_s1123
                                 - 0.182999832157228 z_k1_s1124
                                 - 0.151000439606888 z_k1_s1125
                                 - 0.168000464900306 z_k1_s1126
                                 - 0.502001954668097 z_k1_s1127
                                 - 0.112999892057371 z_k1_s1130
                                 - 0.0900000644555374 z_k1_s1131
                                 - 0.231000398746049 z_k1_s1132
                                 - 0.0700000501320847 z_k1_s1133
                                 - 0.248999722816656 z_k1_s1134
                                 - 0.0630000784489004 z_k1_s1135
                                 - 0.137999632211485 z_k1_s1136
                                 - 0.156999968008999 z_k1_s1137
                                 - 0.140999823588426 z_k1_s1140
                                 - 0.123000015691917 z_k1_s1141
                                 - 0.449999625409743 z_k1_s1142
                                 - 0.381998875135036 z_k1_s1143
                                 - 0.433000544840322 z_k1_s1144
                                 - 0.229000508014711 z_k1_s1145
                                 - 0.364999794565614 z_k1_s1146
                                 - 0.0909999882094828 z_k1_s1147
                                 - 0.146999285089442 z_k1_s1150
                                 - 0.187999770834103 z_k1_s1151
                                 - 0.276999836281813 z_k1_s1152
                                 - 0.197000100777605 z_k1_s1153
                                 - 0.212999524679496 z_k1_s1154
                                 - 0.0739998207412617 z_k1_s1155
                                 - 0.0649998831969727 z_k1_s1156
                                 - 0.146999285089442 z_k1_s1157
                                 - 0.395000340237128 z_k1_s1160
                                 - 0.345000075599124 z_k1_s1161
                                 - 0.553000476331979 z_k1_s1162
                                 - 0.151000274085841 z_k1_s1163
                                 - 0.178000241950053 z_k1_s1164
                                 - 0.174000352059245 z_k1_s1165
                                 - 0.280001044372046 z_k1_s1166
                                 - 0.112000198948431 z_k1_s1167
                                 - 0.124000434119341 z_k1_s1170
                                 - 0.16499988951093 z_k1_s1171
                                 - 0.37200162435929 z_k1_s1172
                                 - 0.0780002990429134 z_k1_s1173
                                 - 0.132000555611279 z_k1_s1174
                                 - 0.258000295600753 z_k1_s1175
                                 - 0.292000409439906 z_k1_s1176
                                 - 0.189000253986745 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.34422 w_k1_ADM_s000
                                 + 1.0976 w_k1_AXP_s000
                                 + 0.834409 w_k1_DIS_s000
                                 + 0.906504 w_k1_FDX_s000
                                 + 1.32718 w_k1_IBM_s000
                                 + 0.880635 w_k1_INTC_s000
                                 + 1.01613 w_k1_KO_s000 + 1.14942 w_k1_MAT_s000
                                 + 0.825411 w_k1_MSFT_s000
                                 + 0.989228 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.18663 w_k1_ADM_s000
                                 + 1.05354 w_k1_AXP_s000
                                 + 1.33075 w_k1_DIS_s000
                                 + 1.34906 w_k1_FDX_s000
                                 + 0.707326 w_k1_IBM_s000
                                 + 1.05672 w_k1_INTC_s000
                                 + 0.909329 w_k1_KO_s000
                                 + 0.804404 w_k1_MAT_s000
                                 + 1.3034 w_k1_MSFT_s000 + 1.19243 w_k1_PG_s000
                                 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.09448 w_k1_ADM_s000
                                 + 0.9391 w_k1_AXP_s000 + 1.12578 w_k1_DIS_s000
                                 + 0.995536 w_k1_FDX_s000
                                 + 1.12453 w_k1_IBM_s000
                                 + 1.06266 w_k1_INTC_s000
                                 + 1.06978 w_k1_KO_s000
                                 + 0.813829 w_k1_MAT_s000
                                 + 0.960594 w_k1_MSFT_s000
                                 + 1.07353 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.98061 w_k1_ADM_s000
                                 + 1.33006 w_k1_AXP_s000
                                 + 1.92725 w_k1_DIS_s000
                                 + 1.58352 w_k1_FDX_s000
                                 + 1.11068 w_k1_IBM_s000
                                 + 1.23982 w_k1_INTC_s000
                                 + 1.38596 w_k1_KO_s000 + 1.01217 w_k1_MAT_s000
                                 + 1.42559 w_k1_MSFT_s000
                                 + 1.33922 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.47554 w_k1_ADM_s000
                                 + 1.21662 w_k1_AXP_s000
                                 + 1.32844 w_k1_DIS_s000
                                 + 1.26952 w_k1_FDX_s000
                                 + 1.30515 w_k1_IBM_s000
                                 + 1.06402 w_k1_INTC_s000
                                 + 1.19463 w_k1_KO_s000 + 1.10577 w_k1_MAT_s000
                                 + 1.05033 w_k1_MSFT_s000
                                 + 1.21198 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.942906 w_k1_ADM_s000
                                 + 0.627969 w_k1_AXP_s000
                                 + 0.708275 w_k1_DIS_s000
                                 + 0.780635 w_k1_FDX_s000
                                 + 1.21085 w_k1_IBM_s000
                                 + 0.81125 w_k1_INTC_s000
                                 + 0.983073 w_k1_KO_s000
                                 + 0.859532 w_k1_MAT_s000
                                 + 0.852266 w_k1_MSFT_s000
                                 + 0.913468 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.998525 w_k1_ADM_s000
                                 + 0.737708 w_k1_AXP_s000
                                 + 0.897566 w_k1_DIS_s000
                                 + 0.972508 w_k1_FDX_s000
                                 + 0.685548 w_k1_IBM_s000
                                 + 0.925192 w_k1_INTC_s000
                                 + 0.788974 w_k1_KO_s000
                                 + 0.35915 w_k1_MAT_s000
                                 + 0.891877 w_k1_MSFT_s000
                                 + 0.87699 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.808886 w_k1_ADM_s000
                                 + 0.70206 w_k1_AXP_s000
                                 + 0.666096 w_k1_DIS_s000
                                 + 0.770623 w_k1_FDX_s000
                                 + 0.811206 w_k1_IBM_s000
                                 + 0.687188 w_k1_INTC_s000
                                 + 0.699376 w_k1_KO_s000
                                 + 0.544627 w_k1_MAT_s000
                                 + 0.739533 w_k1_MSFT_s000
                                 + 0.868415 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.823509 w_k1_ADM_s001
                                 + 0.61117 w_k1_AXP_s001
                                 + 0.724073 w_k1_DIS_s001
                                 + 0.773748 w_k1_FDX_s001
                                 + 0.69977 w_k1_IBM_s001
                                 + 0.722647 w_k1_INTC_s001
                                 + 0.70467 w_k1_KO_s001
                                 + 0.474708 w_k1_MAT_s001
                                 + 0.773568 w_k1_MSFT_s001
                                 + 0.898624 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.745754 w_k1_ADM_s001
                                 + 0.778949 w_k1_AXP_s001
                                 + 0.824846 w_k1_DIS_s001
                                 + 0.831098 w_k1_FDX_s001
                                 + 1.24416 w_k1_IBM_s001
                                 + 0.905777 w_k1_INTC_s001
                                 + 0.971344 w_k1_KO_s001
                                 + 0.986849 w_k1_MAT_s001
                                 + 0.959539 w_k1_MSFT_s001
                                 + 0.922772 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.00131 w_k1_ADM_s001
                                 + 1.09684 w_k1_AXP_s001
                                 + 0.988542 w_k1_DIS_s001
                                 + 1.02915 w_k1_FDX_s001
                                 + 1.12753 w_k1_IBM_s001
                                 + 0.885319 w_k1_INTC_s001
                                 + 0.925789 w_k1_KO_s001 + 1.0345 w_k1_MAT_s001
                                 + 1.0577 w_k1_MSFT_s001 + 1.14641 w_k1_PG_s001
                                 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.947174 w_k1_ADM_s001
                                 + 0.935419 w_k1_AXP_s001
                                 + 0.883363 w_k1_DIS_s001
                                 + 0.969199 w_k1_FDX_s001
                                 + 0.754847 w_k1_IBM_s001
                                 + 0.868657 w_k1_INTC_s001
                                 + 0.753639 w_k1_KO_s001
                                 + 0.744677 w_k1_MAT_s001
                                 + 0.852308 w_k1_MSFT_s001
                                 + 0.95175 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.03417 w_k1_ADM_s001
                                 + 0.864828 w_k1_AXP_s001
                                 + 0.963601 w_k1_DIS_s001
                                 + 1.07983 w_k1_FDX_s001
                                 + 0.682292 w_k1_IBM_s001
                                 + 1.07556 w_k1_INTC_s001
                                 + 0.841849 w_k1_KO_s001
                                 + 0.42003 w_k1_MAT_s001
                                 + 1.05981 w_k1_MSFT_s001
                                 + 0.994206 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.727899 w_k1_ADM_s001
                                 + 0.730125 w_k1_AXP_s001
                                 + 0.854369 w_k1_DIS_s001
                                 + 0.839192 w_k1_FDX_s001
                                 + 0.988777 w_k1_IBM_s001
                                 + 0.886561 w_k1_INTC_s001
                                 + 0.904501 w_k1_KO_s001
                                 + 0.682672 w_k1_MAT_s001
                                 + 0.829379 w_k1_MSFT_s001
                                 + 0.933937 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.20134 w_k1_ADM_s001
                                 + 1.31183 w_k1_AXP_s001
                                 + 1.45373 w_k1_DIS_s001
                                 + 1.22091 w_k1_FDX_s001
                                 + 1.24993 w_k1_IBM_s001
                                 + 1.22489 w_k1_INTC_s001
                                 + 1.15048 w_k1_KO_s001 + 0.96436 w_k1_MAT_s001
                                 + 1.18728 w_k1_MSFT_s001
                                 + 1.07119 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.11437 w_k1_ADM_s001
                                 + 1.1277 w_k1_AXP_s001 + 0.98333 w_k1_DIS_s001
                                 + 0.939503 w_k1_FDX_s001
                                 + 1.84231 w_k1_IBM_s001
                                 + 0.995699 w_k1_INTC_s001
                                 + 1.11811 w_k1_KO_s001 + 1.22779 w_k1_MAT_s001
                                 + 0.916158 w_k1_MSFT_s001
                                 + 1.10403 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.841537 w_k1_ADM_s002
                                 + 1.00929 w_k1_AXP_s002
                                 + 1.29066 w_k1_DIS_s002
                                 + 1.08832 w_k1_FDX_s002
                                 + 1.13687 w_k1_IBM_s002
                                 + 1.07453 w_k1_INTC_s002
                                 + 1.00952 w_k1_KO_s002
                                 + 0.991023 w_k1_MAT_s002
                                 + 1.10561 w_k1_MSFT_s002
                                 + 1.00795 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.95931 w_k1_ADM_s002
                                 + 1.19305 w_k1_AXP_s002
                                 + 1.81204 w_k1_DIS_s002
                                 + 1.33784 w_k1_FDX_s002
                                 + 1.03161 w_k1_IBM_s002 + 1.499 w_k1_INTC_s002
                                 + 1.24627 w_k1_KO_s002
                                 + 0.561028 w_k1_MAT_s002
                                 + 1.47234 w_k1_MSFT_s002
                                 + 1.12009 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.06955 w_k1_ADM_s002
                                 + 1.15755 w_k1_AXP_s002
                                 + 1.12732 w_k1_DIS_s002
                                 + 1.15785 w_k1_FDX_s002
                                 + 0.726691 w_k1_IBM_s002
                                 + 1.12183 w_k1_INTC_s002
                                 + 0.925135 w_k1_KO_s002
                                 + 0.70719 w_k1_MAT_s002
                                 + 1.1677 w_k1_MSFT_s002
                                 + 0.897008 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.847851 w_k1_ADM_s002
                                 + 1.05917 w_k1_AXP_s002
                                 + 0.88134 w_k1_DIS_s002
                                 + 0.892205 w_k1_FDX_s002
                                 + 0.986505 w_k1_IBM_s002
                                 + 0.898403 w_k1_INTC_s002
                                 + 0.795149 w_k1_KO_s002
                                 + 1.04923 w_k1_MAT_s002
                                 + 0.874946 w_k1_MSFT_s002
                                 + 0.877102 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.95173 w_k1_ADM_s002
                                 + 1.15609 w_k1_AXP_s002
                                 + 1.13519 w_k1_DIS_s002
                                 + 0.948138 w_k1_FDX_s002
                                 + 1.27442 w_k1_IBM_s002
                                 + 0.979244 w_k1_INTC_s002
                                 + 1.03431 w_k1_KO_s002 + 1.1913 w_k1_MAT_s002
                                 + 0.937495 w_k1_MSFT_s002
                                 + 1.02863 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.852256 w_k1_ADM_s002
                                 + 0.771098 w_k1_AXP_s002
                                 + 0.912018 w_k1_DIS_s002
                                 + 0.807519 w_k1_FDX_s002
                                 + 1.34098 w_k1_IBM_s002
                                 + 1.02129 w_k1_INTC_s002
                                 + 1.22069 w_k1_KO_s002
                                 + 0.904036 w_k1_MAT_s002
                                 + 0.912566 w_k1_MSFT_s002
                                 + 1.00926 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.03331 w_k1_ADM_s002
                                 + 1.72524 w_k1_AXP_s002
                                 + 1.49595 w_k1_DIS_s002
                                 + 1.28215 w_k1_FDX_s002
                                 + 1.49752 w_k1_IBM_s002
                                 + 1.21114 w_k1_INTC_s002
                                 + 1.16834 w_k1_KO_s002 + 1.20306 w_k1_MAT_s002
                                 + 1.04892 w_k1_MSFT_s002
                                 + 1.10191 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.8428 w_k1_ADM_s002
                                 + 0.766871 w_k1_AXP_s002
                                 + 0.851647 w_k1_DIS_s002
                                 + 0.80659 w_k1_FDX_s002
                                 + 0.726708 w_k1_IBM_s002
                                 + 0.939518 w_k1_INTC_s002
                                 + 0.710015 w_k1_KO_s002
                                 + 0.509244 w_k1_MAT_s002
                                 + 0.899831 w_k1_MSFT_s002
                                 + 0.861551 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.979585 w_k1_ADM_s003
                                 + 0.916516 w_k1_AXP_s003
                                 + 1.04974 w_k1_DIS_s003
                                 + 0.941489 w_k1_FDX_s003
                                 + 0.893031 w_k1_IBM_s003
                                 + 0.846193 w_k1_INTC_s003
                                 + 1.00742 w_k1_KO_s003 + 1.18894 w_k1_MAT_s003
                                 + 0.874308 w_k1_MSFT_s003
                                 + 1.0025 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.963734 w_k1_ADM_s003
                                 + 0.802616 w_k1_AXP_s003
                                 + 0.816869 w_k1_DIS_s003
                                 + 0.784288 w_k1_FDX_s003
                                 + 1.01187 w_k1_IBM_s003
                                 + 0.86853 w_k1_INTC_s003
                                 + 0.907388 w_k1_KO_s003
                                 + 0.949824 w_k1_MAT_s003
                                 + 0.781291 w_k1_MSFT_s003
                                 + 0.830422 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.3816 w_k1_ADM_s003
                                 + 0.952353 w_k1_AXP_s003
                                 + 1.19729 w_k1_DIS_s003
                                 + 1.13355 w_k1_FDX_s003
                                 + 0.846156 w_k1_IBM_s003
                                 + 1.07892 w_k1_INTC_s003
                                 + 1.01832 w_k1_KO_s003
                                 + 0.668998 w_k1_MAT_s003
                                 + 0.964605 w_k1_MSFT_s003
                                 + 1.02837 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.13046 w_k1_ADM_s003
                                 + 0.943663 w_k1_AXP_s003
                                 + 1.00975 w_k1_DIS_s003
                                 + 0.979515 w_k1_FDX_s003
                                 + 1.32147 w_k1_IBM_s003
                                 + 1.06878 w_k1_INTC_s003
                                 + 0.983023 w_k1_KO_s003
                                 + 1.47131 w_k1_MAT_s003
                                 + 0.93451 w_k1_MSFT_s003
                                 + 1.01346 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.01877 w_k1_ADM_s003
                                 + 0.714135 w_k1_AXP_s003
                                 + 0.710685 w_k1_DIS_s003
                                 + 0.809299 w_k1_FDX_s003
                                 + 0.635405 w_k1_IBM_s003
                                 + 0.7919 w_k1_INTC_s003
                                 + 0.652211 w_k1_KO_s003
                                 + 0.54805 w_k1_MAT_s003
                                 + 0.761173 w_k1_MSFT_s003
                                 + 0.821888 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.063 w_k1_ADM_s003
                                 + 1.52562 w_k1_AXP_s003
                                 + 1.56991 w_k1_DIS_s003
                                 + 1.28397 w_k1_FDX_s003
                                 + 1.07748 w_k1_IBM_s003
                                 + 1.23811 w_k1_INTC_s003
                                 + 1.31719 w_k1_KO_s003 + 1.5287 w_k1_MAT_s003
                                 + 1.07607 w_k1_MSFT_s003
                                 + 1.20017 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.30589 w_k1_ADM_s003
                                 + 1.5644 w_k1_AXP_s003 + 1.66423 w_k1_DIS_s003
                                 + 1.29162 w_k1_FDX_s003
                                 + 1.75212 w_k1_IBM_s003
                                 + 1.18788 w_k1_INTC_s003
                                 + 1.57334 w_k1_KO_s003 + 1.88772 w_k1_MAT_s003
                                 + 1.10611 w_k1_MSFT_s003
                                 + 1.19801 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.17059 w_k1_ADM_s003
                                 + 1.04656 w_k1_AXP_s003
                                 + 0.887846 w_k1_DIS_s003
                                 + 0.983962 w_k1_FDX_s003
                                 + 0.758347 w_k1_IBM_s003
                                 + 0.821257 w_k1_INTC_s003
                                 + 0.849532 w_k1_KO_s003
                                 + 0.938241 w_k1_MAT_s003
                                 + 0.893286 w_k1_MSFT_s003
                                 + 0.946802 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.2385 w_k1_ADM_s004
                                 + 1.22072 w_k1_AXP_s004
                                 + 1.42121 w_k1_DIS_s004
                                 + 1.28134 w_k1_FDX_s004
                                 + 0.96246 w_k1_IBM_s004
                                 + 1.15283 w_k1_INTC_s004 + 1.1159 w_k1_KO_s004
                                 + 1.15523 w_k1_MAT_s004
                                 + 1.40558 w_k1_MSFT_s004 + 1.0907 w_k1_PG_s004
                                 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.15547 w_k1_ADM_s004
                                 + 1.00197 w_k1_AXP_s004
                                 + 1.22783 w_k1_DIS_s004
                                 + 1.06662 w_k1_FDX_s004
                                 + 1.28606 w_k1_IBM_s004
                                 + 1.15382 w_k1_INTC_s004
                                 + 1.29714 w_k1_KO_s004 + 1.29795 w_k1_MAT_s004
                                 + 1.24295 w_k1_MSFT_s004
                                 + 1.08631 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.46596 w_k1_ADM_s004
                                 + 1.32661 w_k1_AXP_s004
                                 + 1.78468 w_k1_DIS_s004
                                 + 1.38445 w_k1_FDX_s004
                                 + 1.46814 w_k1_IBM_s004
                                 + 1.31975 w_k1_INTC_s004
                                 + 1.39904 w_k1_KO_s004 + 1.44864 w_k1_MAT_s004
                                 + 1.54217 w_k1_MSFT_s004
                                 + 1.26685 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.17265 w_k1_ADM_s004
                                 + 1.39581 w_k1_AXP_s004
                                 + 0.934694 w_k1_DIS_s004
                                 + 1.13194 w_k1_FDX_s004
                                 + 0.96231 w_k1_IBM_s004
                                 + 1.15895 w_k1_INTC_s004
                                 + 0.83459 w_k1_KO_s004 + 1.08476 w_k1_MAT_s004
                                 + 1.07586 w_k1_MSFT_s004
                                 + 1.06917 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.747683 w_k1_ADM_s004
                                 + 0.592061 w_k1_AXP_s004
                                 + 0.547238 w_k1_DIS_s004
                                 + 0.672638 w_k1_FDX_s004
                                 + 0.736455 w_k1_IBM_s004
                                 + 0.755723 w_k1_INTC_s004
                                 + 0.698603 w_k1_KO_s004
                                 + 0.77627 w_k1_MAT_s004
                                 + 0.761033 w_k1_MSFT_s004
                                 + 0.76967 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.95273 w_k1_ADM_s004
                                 + 0.817195 w_k1_AXP_s004
                                 + 0.816955 w_k1_DIS_s004
                                 + 0.884288 w_k1_FDX_s004
                                 + 1.10216 w_k1_IBM_s004
                                 + 1.18381 w_k1_INTC_s004
                                 + 1.01053 w_k1_KO_s004 + 1.12596 w_k1_MAT_s004
                                 + 1.0278 w_k1_MSFT_s004 + 1.00137 w_k1_PG_s004
                                 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.03045 w_k1_ADM_s004
                                 + 0.866845 w_k1_AXP_s004
                                 + 1.02318 w_k1_DIS_s004
                                 + 1.07171 w_k1_FDX_s004
                                 + 0.723811 w_k1_IBM_s004
                                 + 1.14558 w_k1_INTC_s004
                                 + 0.850012 w_k1_KO_s004
                                 + 0.722829 w_k1_MAT_s004
                                 + 1.14733 w_k1_MSFT_s004
                                 + 0.925906 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.925807 w_k1_ADM_s004
                                 + 0.740627 w_k1_AXP_s004
                                 + 0.707319 w_k1_DIS_s004
                                 + 0.848349 w_k1_FDX_s004
                                 + 0.775987 w_k1_IBM_s004
                                 + 0.928483 w_k1_INTC_s004
                                 + 0.810669 w_k1_KO_s004
                                 + 0.708993 w_k1_MAT_s004
                                 + 0.971535 w_k1_MSFT_s004
                                 + 0.794351 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.97067 w_k1_ADM_s005
                                 + 0.907504 w_k1_AXP_s005
                                 + 0.901128 w_k1_DIS_s005
                                 + 0.818252 w_k1_FDX_s005
                                 + 1.35872 w_k1_IBM_s005
                                 + 1.07405 w_k1_INTC_s005
                                 + 1.06388 w_k1_KO_s005 + 1.23516 w_k1_MAT_s005
                                 + 0.986477 w_k1_MSFT_s005
                                 + 1.01562 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.04838 w_k1_ADM_s005
                                 + 0.828151 w_k1_AXP_s005
                                 + 0.750141 w_k1_DIS_s005
                                 + 0.904134 w_k1_FDX_s005
                                 + 0.659302 w_k1_IBM_s005
                                 + 0.88468 w_k1_INTC_s005
                                 + 0.668117 w_k1_KO_s005
                                 + 0.62005 w_k1_MAT_s005
                                 + 0.897024 w_k1_MSFT_s005
                                 + 0.990108 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.04226 w_k1_ADM_s005
                                 + 1.32235 w_k1_AXP_s005
                                 + 1.40462 w_k1_DIS_s005
                                 + 1.19957 w_k1_FDX_s005
                                 + 1.23592 w_k1_IBM_s005
                                 + 1.29189 w_k1_INTC_s005
                                 + 1.08226 w_k1_KO_s005 + 1.1275 w_k1_MAT_s005
                                 + 1.18405 w_k1_MSFT_s005
                                 + 1.09865 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.06991 w_k1_ADM_s005
                                 + 1.05595 w_k1_AXP_s005
                                 + 1.03582 w_k1_DIS_s005
                                 + 1.00666 w_k1_FDX_s005
                                 + 1.11382 w_k1_IBM_s005
                                 + 0.980656 w_k1_INTC_s005
                                 + 0.987303 w_k1_KO_s005
                                 + 1.35299 w_k1_MAT_s005
                                 + 0.882513 w_k1_MSFT_s005
                                 + 1.19666 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.972028 w_k1_ADM_s005
                                 + 0.820509 w_k1_AXP_s005
                                 + 0.805465 w_k1_DIS_s005
                                 + 0.828271 w_k1_FDX_s005
                                 + 0.948326 w_k1_IBM_s005
                                 + 0.801911 w_k1_INTC_s005
                                 + 0.841857 w_k1_KO_s005
                                 + 0.90331 w_k1_MAT_s005
                                 + 0.904554 w_k1_MSFT_s005
                                 + 1.00002 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.24621 w_k1_ADM_s005
                                 + 1.64289 w_k1_AXP_s005
                                 + 1.51589 w_k1_DIS_s005
                                 + 1.15515 w_k1_FDX_s005
                                 + 1.71026 w_k1_IBM_s005
                                 + 1.40936 w_k1_INTC_s005
                                 + 1.40571 w_k1_KO_s005 + 1.56508 w_k1_MAT_s005
                                 + 1.3105 w_k1_MSFT_s005 + 1.23245 w_k1_PG_s005
                                 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.13154 w_k1_ADM_s005
                                 + 1.33165 w_k1_AXP_s005 + 1.3228 w_k1_DIS_s005
                                 + 1.28087 w_k1_FDX_s005
                                 + 0.871578 w_k1_IBM_s005
                                 + 1.33943 w_k1_INTC_s005
                                 + 0.944776 w_k1_KO_s005
                                 + 0.950062 w_k1_MAT_s005
                                 + 1.14052 w_k1_MSFT_s005
                                 + 1.23133 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.05228 w_k1_ADM_s005
                                 + 1.05732 w_k1_AXP_s005
                                 + 1.14469 w_k1_DIS_s005
                                 + 1.06506 w_k1_FDX_s005
                                 + 0.868892 w_k1_IBM_s005
                                 + 1.11005 w_k1_INTC_s005
                                 + 0.870631 w_k1_KO_s005
                                 + 0.605339 w_k1_MAT_s005
                                 + 1.06575 w_k1_MSFT_s005
                                 + 0.983421 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.13326 w_k1_ADM_s006
                                 + 0.768354 w_k1_AXP_s006
                                 + 0.691977 w_k1_DIS_s006
                                 + 0.805246 w_k1_FDX_s006
                                 + 0.678079 w_k1_IBM_s006
                                 + 0.654497 w_k1_INTC_s006
                                 + 0.727368 w_k1_KO_s006
                                 + 0.731983 w_k1_MAT_s006
                                 + 0.793792 w_k1_MSFT_s006
                                 + 0.8152 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.09848 w_k1_ADM_s006
                                 + 1.39932 w_k1_AXP_s006
                                 + 1.48356 w_k1_DIS_s006
                                 + 1.33939 w_k1_FDX_s006
                                 + 1.19779 w_k1_IBM_s006
                                 + 1.33337 w_k1_INTC_s006 + 1.1248 w_k1_KO_s006
                                 + 1.54882 w_k1_MAT_s006
                                 + 1.22875 w_k1_MSFT_s006
                                 + 1.11713 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.981453 w_k1_ADM_s006
                                 + 0.885916 w_k1_AXP_s006
                                 + 1.13177 w_k1_DIS_s006
                                 + 1.04794 w_k1_FDX_s006
                                 + 0.936632 w_k1_IBM_s006
                                 + 1.0698 w_k1_INTC_s006 + 1.03086 w_k1_KO_s006
                                 + 0.686485 w_k1_MAT_s006
                                 + 1.06763 w_k1_MSFT_s006
                                 + 0.983632 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.19908 w_k1_ADM_s006
                                 + 1.21786 w_k1_AXP_s006
                                 + 0.911044 w_k1_DIS_s006
                                 + 0.955123 w_k1_FDX_s006
                                 + 0.853527 w_k1_IBM_s006
                                 + 1.1121 w_k1_INTC_s006
                                 + 0.936438 w_k1_KO_s006
                                 + 0.973646 w_k1_MAT_s006
                                 + 0.964637 w_k1_MSFT_s006
                                 + 0.987506 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.891728 w_k1_ADM_s006
                                 + 0.75504 w_k1_AXP_s006
                                 + 0.778242 w_k1_DIS_s006
                                 + 0.824767 w_k1_FDX_s006
                                 + 0.949489 w_k1_IBM_s006
                                 + 0.883663 w_k1_INTC_s006
                                 + 0.90837 w_k1_KO_s006
                                 + 0.831883 w_k1_MAT_s006
                                 + 0.844996 w_k1_MSFT_s006
                                 + 0.887993 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.890584 w_k1_ADM_s006
                                 + 0.81704 w_k1_AXP_s006
                                 + 0.916223 w_k1_DIS_s006
                                 + 0.761402 w_k1_FDX_s006
                                 + 1.28529 w_k1_IBM_s006
                                 + 0.939034 w_k1_INTC_s006
                                 + 1.07658 w_k1_KO_s006 + 1.05761 w_k1_MAT_s006
                                 + 0.84455 w_k1_MSFT_s006
                                 + 0.885126 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.08066 w_k1_ADM_s006
                                 + 1.22145 w_k1_AXP_s006
                                 + 0.904833 w_k1_DIS_s006
                                 + 0.981773 w_k1_FDX_s006
                                 + 1.15851 w_k1_IBM_s006
                                 + 1.09127 w_k1_INTC_s006
                                 + 1.05461 w_k1_KO_s006 + 1.28665 w_k1_MAT_s006
                                 + 1.03888 w_k1_MSFT_s006
                                 + 1.10341 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.09913 w_k1_ADM_s006
                                 + 1.47489 w_k1_AXP_s006 + 1.0981 w_k1_DIS_s006
                                 + 1.0401 w_k1_FDX_s006 + 1.56311 w_k1_IBM_s006
                                 + 1.17718 w_k1_INTC_s006 + 1.1838 w_k1_KO_s006
                                 + 1.76019 w_k1_MAT_s006
                                 + 0.930943 w_k1_MSFT_s006
                                 + 1.07431 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.794547 w_k1_ADM_s007
                                 + 0.711705 w_k1_AXP_s007
                                 + 0.745235 w_k1_DIS_s007
                                 + 0.934172 w_k1_FDX_s007
                                 + 0.664603 w_k1_IBM_s007
                                 + 0.972371 w_k1_INTC_s007
                                 + 0.709809 w_k1_KO_s007
                                 + 0.780313 w_k1_MAT_s007
                                 + 0.851439 w_k1_MSFT_s007
                                 + 0.845146 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.04139 w_k1_ADM_s007
                                 + 1.37121 w_k1_AXP_s007
                                 + 1.04977 w_k1_DIS_s007
                                 + 1.07826 w_k1_FDX_s007
                                 + 1.86583 w_k1_IBM_s007
                                 + 1.24686 w_k1_INTC_s007
                                 + 1.31198 w_k1_KO_s007 + 2.17546 w_k1_MAT_s007
                                 + 1.08799 w_k1_MSFT_s007
                                 + 1.22106 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.3088 w_k1_ADM_s007
                                 + 0.97451 w_k1_AXP_s007
                                 + 1.27443 w_k1_DIS_s007
                                 + 1.15454 w_k1_FDX_s007
                                 + 0.774367 w_k1_IBM_s007
                                 + 1.1628 w_k1_INTC_s007 + 1.04528 w_k1_KO_s007
                                 + 0.562802 w_k1_MAT_s007
                                 + 1.1263 w_k1_MSFT_s007 + 1.18846 w_k1_PG_s007
                                 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.04291 w_k1_ADM_s007
                                 + 1.21519 w_k1_AXP_s007
                                 + 1.25542 w_k1_DIS_s007
                                 + 1.19963 w_k1_FDX_s007
                                 + 1.25113 w_k1_IBM_s007
                                 + 1.21929 w_k1_INTC_s007
                                 + 1.24523 w_k1_KO_s007 + 1.26048 w_k1_MAT_s007
                                 + 1.16217 w_k1_MSFT_s007 + 1.2204 w_k1_PG_s007
                                 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.12659 w_k1_ADM_s007
                                 + 1.26671 w_k1_AXP_s007 + 1.2475 w_k1_DIS_s007
                                 + 1.21356 w_k1_FDX_s007
                                 + 0.967863 w_k1_IBM_s007
                                 + 1.30037 w_k1_INTC_s007
                                 + 1.18426 w_k1_KO_s007 + 1.03126 w_k1_MAT_s007
                                 + 1.21084 w_k1_MSFT_s007
                                 + 1.24194 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.926363 w_k1_ADM_s007
                                 + 0.916579 w_k1_AXP_s007
                                 + 0.791611 w_k1_DIS_s007
                                 + 0.932216 w_k1_FDX_s007
                                 + 0.984151 w_k1_IBM_s007
                                 + 0.979178 w_k1_INTC_s007
                                 + 0.866457 w_k1_KO_s007
                                 + 0.86744 w_k1_MAT_s007
                                 + 0.884923 w_k1_MSFT_s007
                                 + 1.02192 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.863429 w_k1_ADM_s007
                                 + 1.00375 w_k1_AXP_s007
                                 + 0.797979 w_k1_DIS_s007
                                 + 0.879883 w_k1_FDX_s007
                                 + 1.30323 w_k1_IBM_s007
                                 + 0.900986 w_k1_INTC_s007
                                 + 0.912081 w_k1_KO_s007
                                 + 1.16268 w_k1_MAT_s007
                                 + 0.839206 w_k1_MSFT_s007
                                 + 0.944086 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.04645 w_k1_ADM_s007
                                 + 1.48707 w_k1_AXP_s007
                                 + 1.66197 w_k1_DIS_s007 + 1.4347 w_k1_FDX_s007
                                 + 0.873999 w_k1_IBM_s007
                                 + 1.60393 w_k1_INTC_s007
                                 + 1.24625 w_k1_KO_s007 + 1.03086 w_k1_MAT_s007
                                 + 1.47636 w_k1_MSFT_s007
                                 + 1.14587 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.805532 w_k1_ADM_s010
                                 + 1.08604 w_k1_AXP_s010
                                 + 1.06064 w_k1_DIS_s010
                                 + 0.970921 w_k1_FDX_s010
                                 + 1.36028 w_k1_IBM_s010
                                 + 1.14948 w_k1_INTC_s010 + 1.2525 w_k1_KO_s010
                                 + 1.09461 w_k1_MAT_s010
                                 + 0.93131 w_k1_MSFT_s010
                                 + 0.961207 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.08427 w_k1_ADM_s010
                                 + 0.912376 w_k1_AXP_s010
                                 + 0.91185 w_k1_DIS_s010
                                 + 1.08364 w_k1_FDX_s010
                                 + 0.86763 w_k1_IBM_s010
                                 + 1.02353 w_k1_INTC_s010
                                 + 0.930181 w_k1_KO_s010
                                 + 1.03051 w_k1_MAT_s010
                                 + 1.00686 w_k1_MSFT_s010
                                 + 0.918141 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.839209 w_k1_ADM_s010
                                 + 0.63775 w_k1_AXP_s010
                                 + 0.719113 w_k1_DIS_s010
                                 + 0.77181 w_k1_FDX_s010
                                 + 0.691888 w_k1_IBM_s010
                                 + 1.04828 w_k1_INTC_s010
                                 + 0.743783 w_k1_KO_s010
                                 + 0.576167 w_k1_MAT_s010
                                 + 0.924815 w_k1_MSFT_s010
                                 + 0.8459 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.895534 w_k1_ADM_s010
                                 + 1.19266 w_k1_AXP_s010 + 1.1936 w_k1_DIS_s010
                                 + 1.02506 w_k1_FDX_s010
                                 + 0.974318 w_k1_IBM_s010
                                 + 1.16308 w_k1_INTC_s010
                                 + 0.98017 w_k1_KO_s010 + 1.00383 w_k1_MAT_s010
                                 + 0.958855 w_k1_MSFT_s010
                                 + 0.928167 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.971819 w_k1_ADM_s010
                                 + 0.713907 w_k1_AXP_s010
                                 + 1.17654 w_k1_DIS_s010
                                 + 0.963154 w_k1_FDX_s010
                                 + 1.04056 w_k1_IBM_s010
                                 + 1.12287 w_k1_INTC_s010
                                 + 0.964962 w_k1_KO_s010
                                 + 0.640149 w_k1_MAT_s010
                                 + 1.04238 w_k1_MSFT_s010
                                 + 0.84164 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.07637 w_k1_ADM_s010
                                 + 1.39028 w_k1_AXP_s010
                                 + 1.70437 w_k1_DIS_s010
                                 + 1.38081 w_k1_FDX_s010
                                 + 1.11381 w_k1_IBM_s010
                                 + 1.13731 w_k1_INTC_s010
                                 + 1.16577 w_k1_KO_s010 + 1.02647 w_k1_MAT_s010
                                 + 1.10161 w_k1_MSFT_s010
                                 + 1.11026 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.902439 w_k1_ADM_s010
                                 + 0.952424 w_k1_AXP_s010
                                 + 0.801572 w_k1_DIS_s010
                                 + 0.90142 w_k1_FDX_s010
                                 + 0.896943 w_k1_IBM_s010
                                 + 0.737769 w_k1_INTC_s010
                                 + 0.8917 w_k1_KO_s010 + 0.952654 w_k1_MAT_s010
                                 + 0.84236 w_k1_MSFT_s010
                                 + 0.861204 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.23466 w_k1_ADM_s010
                                 + 0.938664 w_k1_AXP_s010 + 1.072 w_k1_DIS_s010
                                 + 1.23179 w_k1_FDX_s010
                                 + 0.655664 w_k1_IBM_s010
                                 + 1.05343 w_k1_INTC_s010
                                 + 0.769672 w_k1_KO_s010
                                 + 0.487247 w_k1_MAT_s010
                                 + 0.926212 w_k1_MSFT_s010
                                 + 0.77918 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.14337 w_k1_ADM_s011
                                 + 1.28963 w_k1_AXP_s011
                                 + 1.22424 w_k1_DIS_s011
                                 + 1.31519 w_k1_FDX_s011
                                 + 0.918328 w_k1_IBM_s011
                                 + 1.00171 w_k1_INTC_s011
                                 + 0.964081 w_k1_KO_s011
                                 + 1.23258 w_k1_MAT_s011
                                 + 1.11217 w_k1_MSFT_s011
                                 + 1.11551 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.64314 w_k1_ADM_s011
                                 + 0.655654 w_k1_AXP_s011
                                 + 0.616699 w_k1_DIS_s011
                                 + 0.746962 w_k1_FDX_s011
                                 + 0.682233 w_k1_IBM_s011
                                 + 0.713791 w_k1_INTC_s011
                                 + 0.632749 w_k1_KO_s011
                                 + 0.616174 w_k1_MAT_s011
                                 + 0.751772 w_k1_MSFT_s011
                                 + 0.738272 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.3314 w_k1_ADM_s011
                                 + 0.892344 w_k1_AXP_s011
                                 + 1.14999 w_k1_DIS_s011
                                 + 1.17219 w_k1_FDX_s011
                                 + 0.683237 w_k1_IBM_s011
                                 + 0.864158 w_k1_INTC_s011
                                 + 1.0037 w_k1_KO_s011 + 0.957621 w_k1_MAT_s011
                                 + 1.35568 w_k1_MSFT_s011
                                 + 1.04446 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.814026 w_k1_ADM_s011
                                 + 0.764947 w_k1_AXP_s011
                                 + 0.776124 w_k1_DIS_s011
                                 + 0.835233 w_k1_FDX_s011
                                 + 0.949976 w_k1_IBM_s011
                                 + 0.772875 w_k1_INTC_s011
                                 + 0.828503 w_k1_KO_s011 + 1.0374 w_k1_MAT_s011
                                 + 0.897478 w_k1_MSFT_s011
                                 + 0.840181 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.962826 w_k1_ADM_s011
                                 + 1.1513 w_k1_AXP_s011 + 1.03389 w_k1_DIS_s011
                                 + 1.07313 w_k1_FDX_s011
                                 + 1.25848 w_k1_IBM_s011
                                 + 0.962451 w_k1_INTC_s011
                                 + 1.07486 w_k1_KO_s011 + 1.60298 w_k1_MAT_s011
                                 + 1.1068 w_k1_MSFT_s011 + 1.06654 w_k1_PG_s011
                                 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.20514 w_k1_ADM_s011
                                 + 1.69915 w_k1_AXP_s011
                                 + 1.67951 w_k1_DIS_s011
                                 + 1.74452 w_k1_FDX_s011
                                 + 1.20302 w_k1_IBM_s011
                                 + 1.29609 w_k1_INTC_s011
                                 + 1.20554 w_k1_KO_s011 + 1.32505 w_k1_MAT_s011
                                 + 1.21801 w_k1_MSFT_s011
                                 + 1.22788 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.916647 w_k1_ADM_s011
                                 + 1.1793 w_k1_AXP_s011 + 1.54497 w_k1_DIS_s011
                                 + 1.23714 w_k1_FDX_s011
                                 + 1.82547 w_k1_IBM_s011
                                 + 1.23821 w_k1_INTC_s011
                                 + 1.27122 w_k1_KO_s011 + 1.81442 w_k1_MAT_s011
                                 + 1.57692 w_k1_MSFT_s011
                                 + 1.23325 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.82929 w_k1_ADM_s011
                                 + 0.798554 w_k1_AXP_s011
                                 + 0.935258 w_k1_DIS_s011
                                 + 0.783502 w_k1_FDX_s011
                                 + 1.54588 w_k1_IBM_s011
                                 + 0.785595 w_k1_INTC_s011
                                 + 1.23558 w_k1_KO_s011 + 1.17589 w_k1_MAT_s011
                                 + 1.06898 w_k1_MSFT_s011
                                 + 0.930161 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.14281 w_k1_ADM_s012
                                 + 1.13643 w_k1_AXP_s012
                                 + 1.30804 w_k1_DIS_s012
                                 + 1.09997 w_k1_FDX_s012 + 1.6241 w_k1_IBM_s012
                                 + 1.03936 w_k1_INTC_s012
                                 + 1.21327 w_k1_KO_s012 + 1.58762 w_k1_MAT_s012
                                 + 1.02568 w_k1_MSFT_s012
                                 + 0.938852 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.845267 w_k1_ADM_s012
                                 + 0.874088 w_k1_AXP_s012
                                 + 0.782654 w_k1_DIS_s012
                                 + 0.961296 w_k1_FDX_s012
                                 + 0.815205 w_k1_IBM_s012
                                 + 1.18934 w_k1_INTC_s012
                                 + 0.811913 w_k1_KO_s012
                                 + 0.918447 w_k1_MAT_s012
                                 + 0.985746 w_k1_MSFT_s012
                                 + 0.871074 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.939298 w_k1_ADM_s012
                                 + 0.705511 w_k1_AXP_s012
                                 + 0.845174 w_k1_DIS_s012
                                 + 0.853713 w_k1_FDX_s012
                                 + 0.805715 w_k1_IBM_s012
                                 + 0.977064 w_k1_INTC_s012
                                 + 0.86047 w_k1_KO_s012
                                 + 0.626157 w_k1_MAT_s012
                                 + 0.796236 w_k1_MSFT_s012
                                 + 0.777388 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.02109 w_k1_ADM_s012
                                 + 0.918795 w_k1_AXP_s012
                                 + 0.824376 w_k1_DIS_s012
                                 + 0.847878 w_k1_FDX_s012
                                 + 1.18045 w_k1_IBM_s012
                                 + 0.849035 w_k1_INTC_s012
                                 + 1.09965 w_k1_KO_s012 + 1.20601 w_k1_MAT_s012
                                 + 0.848282 w_k1_MSFT_s012
                                 + 0.8551 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.32518 w_k1_ADM_s012
                                 + 1.18679 w_k1_AXP_s012
                                 + 1.29786 w_k1_DIS_s012
                                 + 1.24836 w_k1_FDX_s012
                                 + 0.978917 w_k1_IBM_s012
                                 + 1.15527 w_k1_INTC_s012
                                 + 1.23445 w_k1_KO_s012 + 1.01493 w_k1_MAT_s012
                                 + 1.17442 w_k1_MSFT_s012
                                 + 1.00054 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.1619 w_k1_ADM_s012
                                 + 0.851627 w_k1_AXP_s012
                                 + 1.21709 w_k1_DIS_s012
                                 + 1.12341 w_k1_FDX_s012
                                 + 0.852742 w_k1_IBM_s012
                                 + 1.0889 w_k1_INTC_s012
                                 + 0.986738 w_k1_KO_s012
                                 + 0.950149 w_k1_MAT_s012
                                 + 1.25383 w_k1_MSFT_s012
                                 + 0.93806 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.18979 w_k1_ADM_s012
                                 + 0.995152 w_k1_AXP_s012
                                 + 0.742916 w_k1_DIS_s012
                                 + 1.05007 w_k1_FDX_s012
                                 + 0.66251 w_k1_IBM_s012
                                 + 1.18825 w_k1_INTC_s012
                                 + 0.820253 w_k1_KO_s012
                                 + 0.631978 w_k1_MAT_s012
                                 + 1.00509 w_k1_MSFT_s012
                                 + 1.01881 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.73211 w_k1_ADM_s012
                                 + 0.640134 w_k1_AXP_s012
                                 + 0.757684 w_k1_DIS_s012
                                 + 0.82768 w_k1_FDX_s012
                                 + 0.517317 w_k1_IBM_s012
                                 + 1.03928 w_k1_INTC_s012
                                 + 0.741927 w_k1_KO_s012
                                 + 0.521056 w_k1_MAT_s012
                                 + 0.865177 w_k1_MSFT_s012
                                 + 0.665889 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.23209 w_k1_ADM_s013
                                 + 1.11568 w_k1_AXP_s013
                                 + 1.50645 w_k1_DIS_s013
                                 + 1.20783 w_k1_FDX_s013
                                 + 1.49652 w_k1_IBM_s013
                                 + 0.90783 w_k1_INTC_s013
                                 + 1.24855 w_k1_KO_s013 + 1.24465 w_k1_MAT_s013
                                 + 1.03141 w_k1_MSFT_s013
                                 + 1.15116 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.689643 w_k1_ADM_s013
                                 + 0.612906 w_k1_AXP_s013
                                 + 0.597523 w_k1_DIS_s013
                                 + 0.742152 w_k1_FDX_s013
                                 + 0.790168 w_k1_IBM_s013
                                 + 0.712639 w_k1_INTC_s013
                                 + 0.685285 w_k1_KO_s013
                                 + 0.870879 w_k1_MAT_s013
                                 + 0.766797 w_k1_MSFT_s013
                                 + 0.796056 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.60279 w_k1_ADM_s013
                                 + 1.4586 w_k1_AXP_s013 + 1.14249 w_k1_DIS_s013
                                 + 1.29978 w_k1_FDX_s013
                                 + 0.923096 w_k1_IBM_s013
                                 + 0.80972 w_k1_INTC_s013
                                 + 0.838472 w_k1_KO_s013 + 1.2772 w_k1_MAT_s013
                                 + 1.05486 w_k1_MSFT_s013
                                 + 1.11244 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.03759 w_k1_ADM_s013
                                 + 1.49121 w_k1_AXP_s013
                                 + 1.43963 w_k1_DIS_s013
                                 + 1.34199 w_k1_FDX_s013
                                 + 1.25467 w_k1_IBM_s013
                                 + 0.901185 w_k1_INTC_s013
                                 + 1.00417 w_k1_KO_s013 + 1.67606 w_k1_MAT_s013
                                 + 1.06134 w_k1_MSFT_s013
                                 + 1.07132 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.17822 w_k1_ADM_s013
                                 + 0.856636 w_k1_AXP_s013
                                 + 1.27376 w_k1_DIS_s013
                                 + 1.16911 w_k1_FDX_s013
                                 + 0.88324 w_k1_IBM_s013
                                 + 0.953365 w_k1_INTC_s013
                                 + 1.08865 w_k1_KO_s013
                                 + 0.970186 w_k1_MAT_s013
                                 + 1.21139 w_k1_MSFT_s013
                                 + 1.02194 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.952617 w_k1_ADM_s013
                                 + 0.775193 w_k1_AXP_s013
                                 + 0.732814 w_k1_DIS_s013
                                 + 0.893519 w_k1_FDX_s013
                                 + 0.671268 w_k1_IBM_s013
                                 + 0.552122 w_k1_INTC_s013
                                 + 0.605958 w_k1_KO_s013
                                 + 0.864632 w_k1_MAT_s013
                                 + 0.744863 w_k1_MSFT_s013
                                 + 0.757057 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.765426 w_k1_ADM_s013
                                 + 1.11032 w_k1_AXP_s013
                                 + 0.909296 w_k1_DIS_s013
                                 + 1.13723 w_k1_FDX_s013
                                 + 0.620156 w_k1_IBM_s013
                                 + 0.870846 w_k1_INTC_s013
                                 + 0.667541 w_k1_KO_s013
                                 + 0.769501 w_k1_MAT_s013
                                 + 0.906444 w_k1_MSFT_s013
                                 + 0.909394 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.742792 w_k1_ADM_s013
                                 + 0.740608 w_k1_AXP_s013
                                 + 0.721532 w_k1_DIS_s013
                                 + 0.904803 w_k1_FDX_s013
                                 + 1.0742 w_k1_IBM_s013
                                 + 0.667918 w_k1_INTC_s013
                                 + 0.835865 w_k1_KO_s013
                                 + 1.05064 w_k1_MAT_s013
                                 + 0.781342 w_k1_MSFT_s013
                                 + 0.962524 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.958573 w_k1_ADM_s014
                                 + 1.05652 w_k1_AXP_s014
                                 + 0.668473 w_k1_DIS_s014
                                 + 0.770443 w_k1_FDX_s014
                                 + 1.34797 w_k1_IBM_s014
                                 + 0.975234 w_k1_INTC_s014
                                 + 1.13761 w_k1_KO_s014 + 1.5569 w_k1_MAT_s014
                                 + 0.929624 w_k1_MSFT_s014
                                 + 1.03595 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.993103 w_k1_ADM_s014
                                 + 1.32946 w_k1_AXP_s014 + 1.1212 w_k1_DIS_s014
                                 + 1.23435 w_k1_FDX_s014
                                 + 0.684436 w_k1_IBM_s014
                                 + 1.25874 w_k1_INTC_s014
                                 + 1.04479 w_k1_KO_s014 + 1.04441 w_k1_MAT_s014
                                 + 1.31194 w_k1_MSFT_s014
                                 + 1.09318 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.96613 w_k1_ADM_s014
                                 + 1.028 w_k1_AXP_s014 + 0.996047 w_k1_DIS_s014
                                 + 0.955423 w_k1_FDX_s014
                                 + 0.987953 w_k1_IBM_s014
                                 + 1.16794 w_k1_INTC_s014
                                 + 1.13865 w_k1_KO_s014 + 1.10413 w_k1_MAT_s014
                                 + 1.1185 w_k1_MSFT_s014 + 1.06674 w_k1_PG_s014
                                 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.26864 w_k1_ADM_s014
                                 + 1.60288 w_k1_AXP_s014
                                 + 1.69301 w_k1_DIS_s014
                                 + 1.59837 w_k1_FDX_s014
                                 + 0.838572 w_k1_IBM_s014
                                 + 1.47124 w_k1_INTC_s014
                                 + 1.26827 w_k1_KO_s014 + 1.27442 w_k1_MAT_s014
                                 + 1.61156 w_k1_MSFT_s014
                                 + 1.07707 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.15508 w_k1_ADM_s014
                                 + 1.4509 w_k1_AXP_s014 + 1.25211 w_k1_DIS_s014
                                 + 1.10489 w_k1_FDX_s014
                                 + 1.34364 w_k1_IBM_s014
                                 + 1.13888 w_k1_INTC_s014
                                 + 1.25155 w_k1_KO_s014 + 1.60577 w_k1_MAT_s014
                                 + 1.07868 w_k1_MSFT_s014
                                 + 1.08396 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.08138 w_k1_ADM_s014
                                 + 0.951636 w_k1_AXP_s014
                                 + 1.1548 w_k1_DIS_s014 + 1.11488 w_k1_FDX_s014
                                 + 0.915295 w_k1_IBM_s014
                                 + 1.25752 w_k1_INTC_s014
                                 + 1.17078 w_k1_KO_s014
                                 + 0.870257 w_k1_MAT_s014
                                 + 1.17705 w_k1_MSFT_s014
                                 + 1.00046 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.704816 w_k1_ADM_s014
                                 + 0.725387 w_k1_AXP_s014
                                 + 0.566512 w_k1_DIS_s014
                                 + 0.713822 w_k1_FDX_s014
                                 + 0.798857 w_k1_IBM_s014
                                 + 0.861053 w_k1_INTC_s014
                                 + 0.790077 w_k1_KO_s014
                                 + 0.884203 w_k1_MAT_s014
                                 + 0.81605 w_k1_MSFT_s014
                                 + 0.792592 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.786165 w_k1_ADM_s014
                                 + 0.815594 w_k1_AXP_s014
                                 + 0.678174 w_k1_DIS_s014
                                 + 0.87557 w_k1_FDX_s014 + 0.6426 w_k1_IBM_s014
                                 + 1.07083 w_k1_INTC_s014
                                 + 0.839083 w_k1_KO_s014
                                 + 0.809604 w_k1_MAT_s014
                                 + 0.949269 w_k1_MSFT_s014
                                 + 0.905521 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.912662 w_k1_ADM_s015
                                 + 0.851641 w_k1_AXP_s015
                                 + 0.690809 w_k1_DIS_s015
                                 + 0.817086 w_k1_FDX_s015
                                 + 0.941015 w_k1_IBM_s015
                                 + 0.875783 w_k1_INTC_s015
                                 + 0.872717 w_k1_KO_s015
                                 + 0.980817 w_k1_MAT_s015
                                 + 0.864045 w_k1_MSFT_s015
                                 + 0.782573 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.919847 w_k1_ADM_s015
                                 + 0.969536 w_k1_AXP_s015
                                 + 0.902742 w_k1_DIS_s015
                                 + 1.05784 w_k1_FDX_s015
                                 + 0.685019 w_k1_IBM_s015
                                 + 1.03036 w_k1_INTC_s015
                                 + 0.774675 w_k1_KO_s015
                                 + 0.706618 w_k1_MAT_s015
                                 + 0.989444 w_k1_MSFT_s015
                                 + 0.91417 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.33879 w_k1_ADM_s015
                                 + 1.25169 w_k1_AXP_s015
                                 + 1.40289 w_k1_DIS_s015
                                 + 1.30688 w_k1_FDX_s015
                                 + 1.64984 w_k1_IBM_s015
                                 + 1.05893 w_k1_INTC_s015
                                 + 1.50999 w_k1_KO_s015 + 2.01924 w_k1_MAT_s015
                                 + 1.16359 w_k1_MSFT_s015
                                 + 1.22974 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.913738 w_k1_ADM_s015
                                 + 0.962582 w_k1_AXP_s015
                                 + 1.42099 w_k1_DIS_s015
                                 + 1.26159 w_k1_FDX_s015
                                 + 0.921397 w_k1_IBM_s015
                                 + 1.29478 w_k1_INTC_s015
                                 + 1.43577 w_k1_KO_s015 + 1.13875 w_k1_MAT_s015
                                 + 1.51499 w_k1_MSFT_s015 + 1.1613 w_k1_PG_s015
                                 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.37661 w_k1_ADM_s015
                                 + 1.41969 w_k1_AXP_s015
                                 + 1.46764 w_k1_DIS_s015
                                 + 1.28682 w_k1_FDX_s015
                                 + 1.18107 w_k1_IBM_s015
                                 + 1.17828 w_k1_INTC_s015
                                 + 1.38491 w_k1_KO_s015 + 1.69183 w_k1_MAT_s015
                                 + 1.44959 w_k1_MSFT_s015
                                 + 1.20747 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.02262 w_k1_ADM_s015
                                 + 0.681825 w_k1_AXP_s015
                                 + 0.856929 w_k1_DIS_s015
                                 + 0.857347 w_k1_FDX_s015
                                 + 1.25801 w_k1_IBM_s015
                                 + 1.00583 w_k1_INTC_s015
                                 + 1.36172 w_k1_KO_s015 + 1.15914 w_k1_MAT_s015
                                 + 1.08406 w_k1_MSFT_s015 + 1.0233 w_k1_PG_s015
                                 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.843271 w_k1_ADM_s015
                                 + 1.03542 w_k1_AXP_s015
                                 + 0.964286 w_k1_DIS_s015
                                 + 0.966537 w_k1_FDX_s015
                                 + 1.37088 w_k1_IBM_s015
                                 + 1.1956 w_k1_INTC_s015 + 1.23181 w_k1_KO_s015
                                 + 1.4051 w_k1_MAT_s015
                                 + 1.09334 w_k1_MSFT_s015
                                 + 0.89262 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.00985 w_k1_ADM_s015
                                 + 1.02137 w_k1_AXP_s015
                                 + 0.992821 w_k1_DIS_s015
                                 + 1.04766 w_k1_FDX_s015
                                 + 0.99855 w_k1_IBM_s015
                                 + 1.10714 w_k1_INTC_s015
                                 + 1.12266 w_k1_KO_s015
                                 + 0.994045 w_k1_MAT_s015
                                 + 1.04373 w_k1_MSFT_s015
                                 + 1.01827 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.03502 w_k1_ADM_s016
                                 + 1.10149 w_k1_AXP_s016 + 1.3363 w_k1_DIS_s016
                                 + 1.00116 w_k1_FDX_s016
                                 + 1.67261 w_k1_IBM_s016
                                 + 1.30179 w_k1_INTC_s016
                                 + 1.63597 w_k1_KO_s016 + 1.4463 w_k1_MAT_s016
                                 + 1.16688 w_k1_MSFT_s016 + 1.3407 w_k1_PG_s016
                                 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.969114 w_k1_ADM_s016
                                 + 1.0527 w_k1_AXP_s016 + 1.16845 w_k1_DIS_s016
                                 + 1.0013 w_k1_FDX_s016 + 1.04171 w_k1_IBM_s016
                                 + 1.18437 w_k1_INTC_s016
                                 + 1.10348 w_k1_KO_s016
                                 + 0.967801 w_k1_MAT_s016
                                 + 1.07599 w_k1_MSFT_s016
                                 + 1.23547 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.985914 w_k1_ADM_s016
                                 + 1.0569 w_k1_AXP_s016 + 0.71687 w_k1_DIS_s016
                                 + 1.03298 w_k1_FDX_s016
                                 + 0.755932 w_k1_IBM_s016
                                 + 1.07314 w_k1_INTC_s016
                                 + 0.844675 w_k1_KO_s016
                                 + 0.882206 w_k1_MAT_s016
                                 + 0.990782 w_k1_MSFT_s016
                                 + 0.993928 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.913427 w_k1_ADM_s016
                                 + 0.663414 w_k1_AXP_s016
                                 + 0.872896 w_k1_DIS_s016
                                 + 0.883773 w_k1_FDX_s016
                                 + 0.682211 w_k1_IBM_s016
                                 + 0.973939 w_k1_INTC_s016
                                 + 0.98233 w_k1_KO_s016
                                 + 0.542501 w_k1_MAT_s016
                                 + 1.05414 w_k1_MSFT_s016
                                 + 0.843898 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.10571 w_k1_ADM_s016
                                 + 1.15034 w_k1_AXP_s016
                                 + 1.04392 w_k1_DIS_s016
                                 + 1.10465 w_k1_FDX_s016
                                 + 0.677147 w_k1_IBM_s016
                                 + 1.38429 w_k1_INTC_s016
                                 + 0.961581 w_k1_KO_s016
                                 + 0.604521 w_k1_MAT_s016
                                 + 1.21331 w_k1_MSFT_s016
                                 + 0.987359 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.21148 w_k1_ADM_s016
                                 + 1.29033 w_k1_AXP_s016
                                 + 1.02975 w_k1_DIS_s016
                                 + 1.05604 w_k1_FDX_s016
                                 + 0.998211 w_k1_IBM_s016
                                 + 0.990685 w_k1_INTC_s016
                                 + 0.967445 w_k1_KO_s016
                                 + 1.16129 w_k1_MAT_s016
                                 + 0.998034 w_k1_MSFT_s016
                                 + 0.947534 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.99059 w_k1_ADM_s016
                                 + 0.803268 w_k1_AXP_s016
                                 + 0.771387 w_k1_DIS_s016
                                 + 0.79896 w_k1_FDX_s016
                                 + 0.840885 w_k1_IBM_s016
                                 + 0.985151 w_k1_INTC_s016
                                 + 0.807723 w_k1_KO_s016
                                 + 0.635989 w_k1_MAT_s016
                                 + 0.816219 w_k1_MSFT_s016
                                 + 0.897091 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.760282 w_k1_ADM_s016
                                 + 1.11722 w_k1_AXP_s016
                                 + 1.28621 w_k1_DIS_s016 + 1.0551 w_k1_FDX_s016
                                 + 1.37643 w_k1_IBM_s016
                                 + 1.62849 w_k1_INTC_s016
                                 + 1.58384 w_k1_KO_s016 + 1.03686 w_k1_MAT_s016
                                 + 1.36471 w_k1_MSFT_s016
                                 + 1.11883 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.961464 w_k1_ADM_s017
                                 + 0.951416 w_k1_AXP_s017
                                 + 1.03366 w_k1_DIS_s017
                                 + 0.932544 w_k1_FDX_s017
                                 + 1.17413 w_k1_IBM_s017
                                 + 1.12691 w_k1_INTC_s017
                                 + 1.14537 w_k1_KO_s017
                                 + 0.984482 w_k1_MAT_s017
                                 + 1.12392 w_k1_MSFT_s017
                                 + 1.04471 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.18986 w_k1_ADM_s017
                                 + 1.18645 w_k1_AXP_s017
                                 + 1.32799 w_k1_DIS_s017
                                 + 1.43531 w_k1_FDX_s017
                                 + 0.762485 w_k1_IBM_s017
                                 + 1.4192 w_k1_INTC_s017 + 1.18537 w_k1_KO_s017
                                 + 0.623513 w_k1_MAT_s017
                                 + 1.58362 w_k1_MSFT_s017
                                 + 1.31795 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.50596 w_k1_ADM_s017
                                 + 1.26482 w_k1_AXP_s017
                                 + 1.18496 w_k1_DIS_s017
                                 + 1.22383 w_k1_FDX_s017 + 1.2268 w_k1_IBM_s017
                                 + 1.27808 w_k1_INTC_s017
                                 + 1.28197 w_k1_KO_s017 + 1.12509 w_k1_MAT_s017
                                 + 1.13673 w_k1_MSFT_s017
                                 + 1.15994 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.39667 w_k1_ADM_s017
                                 + 1.45028 w_k1_AXP_s017
                                 + 0.941647 w_k1_DIS_s017
                                 + 1.17197 w_k1_FDX_s017 + 1.2186 w_k1_IBM_s017
                                 + 1.08521 w_k1_INTC_s017
                                 + 1.10414 w_k1_KO_s017 + 1.17241 w_k1_MAT_s017
                                 + 0.9916 w_k1_MSFT_s017 + 1.10308 w_k1_PG_s017
                                 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.983827 w_k1_ADM_s017
                                 + 0.761866 w_k1_AXP_s017
                                 + 0.712046 w_k1_DIS_s017
                                 + 0.743631 w_k1_FDX_s017
                                 + 1.20012 w_k1_IBM_s017
                                 + 0.946305 w_k1_INTC_s017
                                 + 0.901869 w_k1_KO_s017
                                 + 1.09241 w_k1_MAT_s017
                                 + 0.957688 w_k1_MSFT_s017
                                 + 0.937637 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.03379 w_k1_ADM_s017
                                 + 1.30387 w_k1_AXP_s017
                                 + 1.35473 w_k1_DIS_s017
                                 + 1.15087 w_k1_FDX_s017
                                 + 1.66718 w_k1_IBM_s017
                                 + 1.31661 w_k1_INTC_s017
                                 + 1.40744 w_k1_KO_s017 + 1.44732 w_k1_MAT_s017
                                 + 1.24922 w_k1_MSFT_s017
                                 + 1.18329 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.02329 w_k1_ADM_s017
                                 + 0.708631 w_k1_AXP_s017
                                 + 0.770521 w_k1_DIS_s017
                                 + 0.875901 w_k1_FDX_s017
                                 + 0.74187 w_k1_IBM_s017
                                 + 1.00722 w_k1_INTC_s017
                                 + 0.790324 w_k1_KO_s017
                                 + 0.471519 w_k1_MAT_s017
                                 + 0.985479 w_k1_MSFT_s017
                                 + 0.909128 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.3597 w_k1_ADM_s017
                                 + 0.911749 w_k1_AXP_s017
                                 + 1.50067 w_k1_DIS_s017
                                 + 1.27527 w_k1_FDX_s017
                                 + 0.929347 w_k1_IBM_s017
                                 + 1.29886 w_k1_INTC_s017
                                 + 1.17471 w_k1_KO_s017
                                 + 0.850699 w_k1_MAT_s017
                                 + 1.39371 w_k1_MSFT_s017
                                 + 1.17895 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.21182 w_k1_ADM_s100
                                 + 1.16398 w_k1_AXP_s100
                                 + 1.53502 w_k1_DIS_s100
                                 + 1.38391 w_k1_FDX_s100
                                 + 0.992484 w_k1_IBM_s100
                                 + 1.33815 w_k1_INTC_s100
                                 + 1.16675 w_k1_KO_s100 + 1.05322 w_k1_MAT_s100
                                 + 1.50923 w_k1_MSFT_s100
                                 + 1.06762 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.993723 w_k1_ADM_s100
                                 + 0.791695 w_k1_AXP_s100
                                 + 1.01981 w_k1_DIS_s100
                                 + 0.988331 w_k1_FDX_s100
                                 + 0.820107 w_k1_IBM_s100
                                 + 1.20087 w_k1_INTC_s100
                                 + 0.913213 w_k1_KO_s100
                                 + 0.619963 w_k1_MAT_s100
                                 + 1.14789 w_k1_MSFT_s100
                                 + 0.892888 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.01518 w_k1_ADM_s100
                                 + 1.0174 w_k1_AXP_s100
                                 + 0.673919 w_k1_DIS_s100
                                 + 0.983238 w_k1_FDX_s100
                                 + 0.771895 w_k1_IBM_s100
                                 + 1.03057 w_k1_INTC_s100
                                 + 0.684463 w_k1_KO_s100
                                 + 0.810033 w_k1_MAT_s100
                                 + 0.881527 w_k1_MSFT_s100
                                 + 0.864701 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.756931 w_k1_ADM_s100
                                 + 0.554001 w_k1_AXP_s100
                                 + 0.712709 w_k1_DIS_s100
                                 + 0.821681 w_k1_FDX_s100
                                 + 0.785979 w_k1_IBM_s100
                                 + 0.854305 w_k1_INTC_s100
                                 + 0.808524 w_k1_KO_s100
                                 + 0.712983 w_k1_MAT_s100
                                 + 0.906666 w_k1_MSFT_s100
                                 + 0.804509 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.871215 w_k1_ADM_s100
                                 + 1.40665 w_k1_AXP_s100
                                 + 1.56386 w_k1_DIS_s100
                                 + 1.22966 w_k1_FDX_s100
                                 + 1.47524 w_k1_IBM_s100
                                 + 1.25494 w_k1_INTC_s100
                                 + 1.21149 w_k1_KO_s100 + 1.67228 w_k1_MAT_s100
                                 + 1.16152 w_k1_MSFT_s100
                                 + 1.15757 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.691532 w_k1_ADM_s100
                                 + 0.788089 w_k1_AXP_s100
                                 + 0.589267 w_k1_DIS_s100
                                 + 0.755666 w_k1_FDX_s100
                                 + 1.01897 w_k1_IBM_s100
                                 + 0.813844 w_k1_INTC_s100
                                 + 0.735861 w_k1_KO_s100
                                 + 0.792991 w_k1_MAT_s100
                                 + 0.733205 w_k1_MSFT_s100
                                 + 0.766538 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.055 w_k1_ADM_s100
                                 + 0.987102 w_k1_AXP_s100
                                 + 0.955631 w_k1_DIS_s100
                                 + 1.04028 w_k1_FDX_s100
                                 + 1.12349 w_k1_IBM_s100
                                 + 1.07537 w_k1_INTC_s100
                                 + 1.00894 w_k1_KO_s100
                                 + 0.865349 w_k1_MAT_s100
                                 + 0.963381 w_k1_MSFT_s100
                                 + 0.955751 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.787527 w_k1_ADM_s100
                                 + 0.951096 w_k1_AXP_s100
                                 + 0.995592 w_k1_DIS_s100
                                 + 0.956871 w_k1_FDX_s100
                                 + 1.40734 w_k1_IBM_s100
                                 + 1.12324 w_k1_INTC_s100
                                 + 1.15011 w_k1_KO_s100
                                 + 0.941307 w_k1_MAT_s100
                                 + 1.03488 w_k1_MSFT_s100
                                 + 0.973761 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.02044 w_k1_ADM_s101
                                 + 1.28297 w_k1_AXP_s101 + 1.1344 w_k1_DIS_s101
                                 + 1.08327 w_k1_FDX_s101
                                 + 1.36822 w_k1_IBM_s101
                                 + 1.18881 w_k1_INTC_s101
                                 + 1.06706 w_k1_KO_s101 + 1.3574 w_k1_MAT_s101
                                 + 0.907833 w_k1_MSFT_s101
                                 + 1.02287 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.858931 w_k1_ADM_s101
                                 + 0.883903 w_k1_AXP_s101
                                 + 0.763542 w_k1_DIS_s101
                                 + 0.873823 w_k1_FDX_s101
                                 + 0.684053 w_k1_IBM_s101
                                 + 1.12889 w_k1_INTC_s101
                                 + 0.814449 w_k1_KO_s101
                                 + 0.760134 w_k1_MAT_s101
                                 + 0.85539 w_k1_MSFT_s101
                                 + 0.816227 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.03722 w_k1_ADM_s101
                                 + 1.25671 w_k1_AXP_s101
                                 + 1.01448 w_k1_DIS_s101
                                 + 1.18917 w_k1_FDX_s101
                                 + 0.84167 w_k1_IBM_s101
                                 + 1.20015 w_k1_INTC_s101
                                 + 0.851905 w_k1_KO_s101
                                 + 0.719555 w_k1_MAT_s101
                                 + 1.01641 w_k1_MSFT_s101
                                 + 0.83025 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.822944 w_k1_ADM_s101
                                 + 0.756846 w_k1_AXP_s101
                                 + 0.634524 w_k1_DIS_s101
                                 + 0.716601 w_k1_FDX_s101
                                 + 0.875259 w_k1_IBM_s101
                                 + 0.844921 w_k1_INTC_s101
                                 + 0.721869 w_k1_KO_s101
                                 + 0.68435 w_k1_MAT_s101
                                 + 0.724665 w_k1_MSFT_s101
                                 + 0.726967 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.44252 w_k1_ADM_s101
                                 + 1.44608 w_k1_AXP_s101
                                 + 1.67146 w_k1_DIS_s101
                                 + 1.38353 w_k1_FDX_s101
                                 + 1.52722 w_k1_IBM_s101
                                 + 1.11277 w_k1_INTC_s101
                                 + 1.41708 w_k1_KO_s101 + 1.95097 w_k1_MAT_s101
                                 + 1.28112 w_k1_MSFT_s101 + 1.1532 w_k1_PG_s101
                                 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.22064 w_k1_ADM_s101
                                 + 1.25964 w_k1_AXP_s101
                                 + 1.43157 w_k1_DIS_s101
                                 + 1.22278 w_k1_FDX_s101
                                 + 1.09516 w_k1_IBM_s101
                                 + 1.42251 w_k1_INTC_s101
                                 + 1.25358 w_k1_KO_s101
                                 + 0.939092 w_k1_MAT_s101
                                 + 1.24927 w_k1_MSFT_s101
                                 + 1.06504 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.830388 w_k1_ADM_s101
                                 + 0.779493 w_k1_AXP_s101
                                 + 0.905827 w_k1_DIS_s101
                                 + 0.914377 w_k1_FDX_s101
                                 + 1.07017 w_k1_IBM_s101
                                 + 1.07446 w_k1_INTC_s101
                                 + 0.909206 w_k1_KO_s101
                                 + 0.835417 w_k1_MAT_s101
                                 + 1.0163 w_k1_MSFT_s101
                                 + 0.908371 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.12144 w_k1_ADM_s101
                                 + 0.943953 w_k1_AXP_s101
                                 + 1.03408 w_k1_DIS_s101
                                 + 1.04334 w_k1_FDX_s101
                                 + 0.878734 w_k1_IBM_s101
                                 + 1.0561 w_k1_INTC_s101 + 1.02065 w_k1_KO_s101
                                 + 0.772183 w_k1_MAT_s101
                                 + 1.0277 w_k1_MSFT_s101 + 0.88966 w_k1_PG_s101
                                 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.861742 w_k1_ADM_s102
                                 + 0.679004 w_k1_AXP_s102
                                 + 0.696965 w_k1_DIS_s102
                                 + 0.684346 w_k1_FDX_s102
                                 + 1.01784 w_k1_IBM_s102
                                 + 0.945057 w_k1_INTC_s102
                                 + 0.91475 w_k1_KO_s102 + 0.55265 w_k1_MAT_s102
                                 + 0.810814 w_k1_MSFT_s102
                                 + 0.83911 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.889335 w_k1_ADM_s102
                                 + 0.80845 w_k1_AXP_s102
                                 + 0.869254 w_k1_DIS_s102
                                 + 0.864596 w_k1_FDX_s102
                                 + 0.790321 w_k1_IBM_s102
                                 + 1.1288 w_k1_INTC_s102
                                 + 0.947736 w_k1_KO_s102
                                 + 0.569606 w_k1_MAT_s102
                                 + 0.962802 w_k1_MSFT_s102
                                 + 0.915138 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.01315 w_k1_ADM_s102
                                 + 1.21954 w_k1_AXP_s102
                                 + 1.15523 w_k1_DIS_s102 + 0.9401 w_k1_FDX_s102
                                 + 1.6747 w_k1_IBM_s102
                                 + 1.06958 w_k1_INTC_s102
                                 + 1.39249 w_k1_KO_s102 + 1.89805 w_k1_MAT_s102
                                 + 0.969359 w_k1_MSFT_s102
                                 + 1.10946 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.40374 w_k1_ADM_s102
                                 + 1.5721 w_k1_AXP_s102 + 1.63762 w_k1_DIS_s102
                                 + 1.24888 w_k1_FDX_s102
                                 + 1.40305 w_k1_IBM_s102
                                 + 1.20461 w_k1_INTC_s102
                                 + 1.43046 w_k1_KO_s102 + 1.59327 w_k1_MAT_s102
                                 + 1.16973 w_k1_MSFT_s102
                                 + 1.21138 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.01756 w_k1_ADM_s102
                                 + 1.16666 w_k1_AXP_s102
                                 + 1.06867 w_k1_DIS_s102
                                 + 0.902594 w_k1_FDX_s102
                                 + 1.18637 w_k1_IBM_s102
                                 + 0.935528 w_k1_INTC_s102
                                 + 1.06649 w_k1_KO_s102 + 1.07987 w_k1_MAT_s102
                                 + 0.870842 w_k1_MSFT_s102
                                 + 1.04215 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.06061 w_k1_ADM_s102
                                 + 1.12588 w_k1_AXP_s102
                                 + 1.18446 w_k1_DIS_s102
                                 + 1.14379 w_k1_FDX_s102
                                 + 0.918383 w_k1_IBM_s102
                                 + 1.14113 w_k1_INTC_s102 + 1.0412 w_k1_KO_s102
                                 + 0.95683 w_k1_MAT_s102
                                 + 1.1721 w_k1_MSFT_s102 + 1.00027 w_k1_PG_s102
                                 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.10319 w_k1_ADM_s102
                                 + 1.43519 w_k1_AXP_s102
                                 + 1.44604 w_k1_DIS_s102
                                 + 1.29594 w_k1_FDX_s102
                                 + 1.01992 w_k1_IBM_s102
                                 + 1.13706 w_k1_INTC_s102
                                 + 1.16966 w_k1_KO_s102
                                 + 0.778028 w_k1_MAT_s102
                                 + 1.16045 w_k1_MSFT_s102
                                 + 1.20956 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.885498 w_k1_ADM_s102
                                 + 0.900696 w_k1_AXP_s102
                                 + 0.832533 w_k1_DIS_s102
                                 + 0.975131 w_k1_FDX_s102
                                 + 0.54952 w_k1_IBM_s102
                                 + 1.05253 w_k1_INTC_s102
                                 + 0.826444 w_k1_KO_s102
                                 + 0.529472 w_k1_MAT_s102
                                 + 0.967337 w_k1_MSFT_s102
                                 + 0.867723 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.2838 w_k1_ADM_s103
                                 + 1.27819 w_k1_AXP_s103
                                 + 1.16285 w_k1_DIS_s103
                                 + 1.10832 w_k1_FDX_s103
                                 + 0.868915 w_k1_IBM_s103
                                 + 0.84017 w_k1_INTC_s103
                                 + 0.882232 w_k1_KO_s103
                                 + 1.00659 w_k1_MAT_s103
                                 + 0.925891 w_k1_MSFT_s103
                                 + 1.06462 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.841084 w_k1_ADM_s103
                                 + 0.955019 w_k1_AXP_s103
                                 + 0.900847 w_k1_DIS_s103
                                 + 0.955699 w_k1_FDX_s103
                                 + 0.704725 w_k1_IBM_s103
                                 + 0.921013 w_k1_INTC_s103
                                 + 0.944111 w_k1_KO_s103
                                 + 0.996313 w_k1_MAT_s103
                                 + 0.958372 w_k1_MSFT_s103
                                 + 1.02973 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.919495 w_k1_ADM_s103
                                 + 0.845198 w_k1_AXP_s103
                                 + 0.871432 w_k1_DIS_s103
                                 + 0.82215 w_k1_FDX_s103
                                 + 0.932136 w_k1_IBM_s103
                                 + 0.919555 w_k1_INTC_s103
                                 + 0.890054 w_k1_KO_s103
                                 + 0.873822 w_k1_MAT_s103
                                 + 0.94737 w_k1_MSFT_s103
                                 + 0.849477 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.857868 w_k1_ADM_s103
                                 + 0.792431 w_k1_AXP_s103
                                 + 0.763018 w_k1_DIS_s103
                                 + 0.659115 w_k1_FDX_s103
                                 + 1.24758 w_k1_IBM_s103
                                 + 0.780306 w_k1_INTC_s103
                                 + 1.09807 w_k1_KO_s103 + 1.08 w_k1_MAT_s103
                                 + 0.757692 w_k1_MSFT_s103
                                 + 0.884513 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.15858 w_k1_ADM_s103
                                 + 1.31532 w_k1_AXP_s103
                                 + 1.65253 w_k1_DIS_s103
                                 + 1.24149 w_k1_FDX_s103
                                 + 1.20241 w_k1_IBM_s103
                                 + 1.16868 w_k1_INTC_s103
                                 + 1.56483 w_k1_KO_s103 + 1.52168 w_k1_MAT_s103
                                 + 1.31225 w_k1_MSFT_s103
                                 + 1.42664 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.753213 w_k1_ADM_s103
                                 + 1.11751 w_k1_AXP_s103
                                 + 1.10795 w_k1_DIS_s103
                                 + 0.930376 w_k1_FDX_s103
                                 + 1.44421 w_k1_IBM_s103
                                 + 0.927804 w_k1_INTC_s103
                                 + 1.12514 w_k1_KO_s103 + 1.9996 w_k1_MAT_s103
                                 + 0.974268 w_k1_MSFT_s103
                                 + 1.17374 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.685814 w_k1_ADM_s103
                                 + 0.582098 w_k1_AXP_s103
                                 + 0.605806 w_k1_DIS_s103
                                 + 0.72541 w_k1_FDX_s103
                                 + 0.627707 w_k1_IBM_s103
                                 + 0.62546 w_k1_INTC_s103
                                 + 0.717243 w_k1_KO_s103
                                 + 0.853938 w_k1_MAT_s103
                                 + 0.765605 w_k1_MSFT_s103
                                 + 0.780914 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.1115 w_k1_ADM_s103
                                 + 0.866242 w_k1_AXP_s103
                                 + 1.01919 w_k1_DIS_s103
                                 + 0.982465 w_k1_FDX_s103
                                 + 0.845608 w_k1_IBM_s103
                                 + 0.975498 w_k1_INTC_s103
                                 + 1.12099 w_k1_KO_s103 + 1.01326 w_k1_MAT_s103
                                 + 1.06559 w_k1_MSFT_s103
                                 + 1.13435 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.62118 w_k1_ADM_s104
                                 + 0.672109 w_k1_AXP_s104
                                 + 0.717266 w_k1_DIS_s104
                                 + 0.773043 w_k1_FDX_s104
                                 + 0.631561 w_k1_IBM_s104
                                 + 0.686744 w_k1_INTC_s104
                                 + 0.674101 w_k1_KO_s104
                                 + 0.628666 w_k1_MAT_s104
                                 + 0.754807 w_k1_MSFT_s104
                                 + 0.852423 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.11834 w_k1_ADM_s104
                                 + 1.37118 w_k1_AXP_s104
                                 + 1.19038 w_k1_DIS_s104
                                 + 1.03765 w_k1_FDX_s104
                                 + 1.31691 w_k1_IBM_s104
                                 + 0.899642 w_k1_INTC_s104
                                 + 0.990367 w_k1_KO_s104
                                 + 1.60573 w_k1_MAT_s104
                                 + 0.908722 w_k1_MSFT_s104
                                 + 1.05766 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.873901 w_k1_ADM_s104
                                 + 0.816476 w_k1_AXP_s104
                                 + 0.826376 w_k1_DIS_s104
                                 + 0.859879 w_k1_FDX_s104
                                 + 0.799373 w_k1_IBM_s104
                                 + 0.870684 w_k1_INTC_s104
                                 + 0.707455 w_k1_KO_s104
                                 + 0.675434 w_k1_MAT_s104
                                 + 0.810043 w_k1_MSFT_s104
                                 + 0.817259 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.92265 w_k1_ADM_s104
                                 + 1.22564 w_k1_AXP_s104
                                 + 1.44096 w_k1_DIS_s104
                                 + 1.24258 w_k1_FDX_s104
                                 + 1.04361 w_k1_IBM_s104
                                 + 0.954972 w_k1_INTC_s104
                                 + 1.17547 w_k1_KO_s104 + 1.30951 w_k1_MAT_s104
                                 + 1.14161 w_k1_MSFT_s104
                                 + 1.06698 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.891411 w_k1_ADM_s104
                                 + 1.12276 w_k1_AXP_s104
                                 + 1.15478 w_k1_DIS_s104
                                 + 1.03064 w_k1_FDX_s104
                                 + 1.38574 w_k1_IBM_s104
                                 + 0.938359 w_k1_INTC_s104
                                 + 1.22873 w_k1_KO_s104 + 1.31462 w_k1_MAT_s104
                                 + 1.12943 w_k1_MSFT_s104 + 1.0832 w_k1_PG_s104
                                 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.0554 w_k1_ADM_s104
                                 + 0.853454 w_k1_AXP_s104
                                 + 0.913832 w_k1_DIS_s104
                                 + 0.893399 w_k1_FDX_s104
                                 + 1.06635 w_k1_IBM_s104
                                 + 0.702587 w_k1_INTC_s104
                                 + 0.905074 w_k1_KO_s104 + 1.0929 w_k1_MAT_s104
                                 + 0.797669 w_k1_MSFT_s104
                                 + 0.998948 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.966705 w_k1_ADM_s104
                                 + 0.893897 w_k1_AXP_s104
                                 + 1.10282 w_k1_DIS_s104
                                 + 1.09465 w_k1_FDX_s104
                                 + 0.747582 w_k1_IBM_s104
                                 + 0.905867 w_k1_INTC_s104
                                 + 0.928302 w_k1_KO_s104
                                 + 0.750202 w_k1_MAT_s104
                                 + 0.960367 w_k1_MSFT_s104
                                 + 1.00374 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.953036 w_k1_ADM_s104
                                 + 1.5056 w_k1_AXP_s104 + 1.16545 w_k1_DIS_s104
                                 + 1.27524 w_k1_FDX_s104
                                 + 1.10367 w_k1_IBM_s104
                                 + 1.05555 w_k1_INTC_s104
                                 + 0.986746 w_k1_KO_s104
                                 + 1.11536 w_k1_MAT_s104
                                 + 0.969623 w_k1_MSFT_s104
                                 + 1.09594 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.812498 w_k1_ADM_s105
                                 + 0.805048 w_k1_AXP_s105
                                 + 0.71201 w_k1_DIS_s105
                                 + 0.705401 w_k1_FDX_s105
                                 + 0.980428 w_k1_IBM_s105
                                 + 0.932024 w_k1_INTC_s105
                                 + 0.904996 w_k1_KO_s105
                                 + 0.769192 w_k1_MAT_s105
                                 + 0.747602 w_k1_MSFT_s105
                                 + 0.917315 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.04171 w_k1_ADM_s105
                                 + 1.11134 w_k1_AXP_s105
                                 + 1.09592 w_k1_DIS_s105
                                 + 0.960926 w_k1_FDX_s105
                                 + 1.31169 w_k1_IBM_s105
                                 + 1.25721 w_k1_INTC_s105
                                 + 1.30183 w_k1_KO_s105 + 1.08333 w_k1_MAT_s105
                                 + 1.05589 w_k1_MSFT_s105
                                 + 1.07011 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.900912 w_k1_ADM_s105
                                 + 0.704727 w_k1_AXP_s105
                                 + 0.674753 w_k1_DIS_s105
                                 + 0.821771 w_k1_FDX_s105
                                 + 0.843257 w_k1_IBM_s105
                                 + 0.920021 w_k1_INTC_s105
                                 + 0.887565 w_k1_KO_s105
                                 + 0.663523 w_k1_MAT_s105
                                 + 0.809554 w_k1_MSFT_s105
                                 + 0.950497 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.864625 w_k1_ADM_s105
                                 + 0.740739 w_k1_AXP_s105
                                 + 0.652821 w_k1_DIS_s105
                                 + 0.754492 w_k1_FDX_s105
                                 + 0.637353 w_k1_IBM_s105
                                 + 0.908211 w_k1_INTC_s105
                                 + 0.810656 w_k1_KO_s105
                                 + 0.598741 w_k1_MAT_s105
                                 + 0.692747 w_k1_MSFT_s105
                                 + 0.849517 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.972107 w_k1_ADM_s105
                                 + 0.84132 w_k1_AXP_s105
                                 + 0.917938 w_k1_DIS_s105
                                 + 1.06899 w_k1_FDX_s105
                                 + 0.721295 w_k1_IBM_s105
                                 + 0.941035 w_k1_INTC_s105
                                 + 0.959157 w_k1_KO_s105
                                 + 0.767184 w_k1_MAT_s105
                                 + 0.925658 w_k1_MSFT_s105
                                 + 0.987841 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.12966 w_k1_ADM_s105
                                 + 1.23385 w_k1_AXP_s105
                                 + 1.21851 w_k1_DIS_s105
                                 + 1.23538 w_k1_FDX_s105
                                 + 0.899078 w_k1_IBM_s105
                                 + 1.1209 w_k1_INTC_s105 + 1.11087 w_k1_KO_s105
                                 + 1.19886 w_k1_MAT_s105
                                 + 1.08843 w_k1_MSFT_s105
                                 + 1.27388 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.893685 w_k1_ADM_s105
                                 + 1.0723 w_k1_AXP_s105
                                 + 0.886507 w_k1_DIS_s105
                                 + 0.935203 w_k1_FDX_s105
                                 + 1.11456 w_k1_IBM_s105
                                 + 0.951637 w_k1_INTC_s105
                                 + 1.06286 w_k1_KO_s105 + 1.20296 w_k1_MAT_s105
                                 + 0.858389 w_k1_MSFT_s105
                                 + 1.04652 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.01411 w_k1_ADM_s105
                                 + 1.45274 w_k1_AXP_s105
                                 + 1.64929 w_k1_DIS_s105
                                 + 1.41655 w_k1_FDX_s105
                                 + 1.35555 w_k1_IBM_s105
                                 + 1.61295 w_k1_INTC_s105 + 1.5857 w_k1_KO_s105
                                 + 1.71348 w_k1_MAT_s105
                                 + 1.35741 w_k1_MSFT_s105
                                 + 1.34402 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.03246 w_k1_ADM_s106
                                 + 0.920381 w_k1_AXP_s106
                                 + 0.807502 w_k1_DIS_s106
                                 + 1.00292 w_k1_FDX_s106
                                 + 0.677603 w_k1_IBM_s106
                                 + 1.095 w_k1_INTC_s106 + 0.717832 w_k1_KO_s106
                                 + 0.73932 w_k1_MAT_s106
                                 + 1.01214 w_k1_MSFT_s106
                                 + 0.983042 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.863311 w_k1_ADM_s106
                                 + 0.98748 w_k1_AXP_s106
                                 + 0.734519 w_k1_DIS_s106
                                 + 0.854146 w_k1_FDX_s106
                                 + 1.07413 w_k1_IBM_s106
                                 + 1.20152 w_k1_INTC_s106
                                 + 1.01185 w_k1_KO_s106 + 1.02574 w_k1_MAT_s106
                                 + 0.799748 w_k1_MSFT_s106
                                 + 0.975593 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.38092 w_k1_ADM_s106
                                 + 1.31748 w_k1_AXP_s106
                                 + 1.41321 w_k1_DIS_s106
                                 + 1.29336 w_k1_FDX_s106
                                 + 1.34815 w_k1_IBM_s106
                                 + 1.24528 w_k1_INTC_s106
                                 + 1.25202 w_k1_KO_s106 + 1.56933 w_k1_MAT_s106
                                 + 1.37591 w_k1_MSFT_s106
                                 + 1.30247 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.950036 w_k1_ADM_s106
                                 + 0.663042 w_k1_AXP_s106
                                 + 0.67707 w_k1_DIS_s106
                                 + 0.806673 w_k1_FDX_s106
                                 + 0.938024 w_k1_IBM_s106
                                 + 0.99821 w_k1_INTC_s106
                                 + 0.793636 w_k1_KO_s106
                                 + 0.970532 w_k1_MAT_s106
                                 + 0.917503 w_k1_MSFT_s106
                                 + 0.867059 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.878065 w_k1_ADM_s106
                                 + 0.874474 w_k1_AXP_s106
                                 + 0.998556 w_k1_DIS_s106
                                 + 0.918064 w_k1_FDX_s106
                                 + 1.18089 w_k1_IBM_s106
                                 + 1.0325 w_k1_INTC_s106 + 1.07915 w_k1_KO_s106
                                 + 1.35813 w_k1_MAT_s106
                                 + 1.19465 w_k1_MSFT_s106
                                 + 1.03616 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.12253 w_k1_ADM_s106
                                 + 1.22729 w_k1_AXP_s106
                                 + 0.866805 w_k1_DIS_s106
                                 + 1.07423 w_k1_FDX_s106 + 1.2829 w_k1_IBM_s106
                                 + 1.01978 w_k1_INTC_s106
                                 + 1.09887 w_k1_KO_s106 + 1.42699 w_k1_MAT_s106
                                 + 0.875258 w_k1_MSFT_s106
                                 + 1.12441 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.961852 w_k1_ADM_s106
                                 + 0.997305 w_k1_AXP_s106
                                 + 1.20403 w_k1_DIS_s106
                                 + 1.08755 w_k1_FDX_s106
                                 + 1.01746 w_k1_IBM_s106
                                 + 1.20897 w_k1_INTC_s106
                                 + 1.09748 w_k1_KO_s106
                                 + 0.984039 w_k1_MAT_s106
                                 + 1.32634 w_k1_MSFT_s106
                                 + 1.14542 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.10096 w_k1_ADM_s106
                                 + 1.45464 w_k1_AXP_s106 + 1.046 w_k1_DIS_s106
                                 + 1.1399 w_k1_FDX_s106 + 1.0165 w_k1_IBM_s106
                                 + 1.09567 w_k1_INTC_s106
                                 + 0.980228 w_k1_KO_s106
                                 + 1.52532 w_k1_MAT_s106
                                 + 1.08323 w_k1_MSFT_s106
                                 + 1.09265 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.13247 w_k1_ADM_s107
                                 + 0.617661 w_k1_AXP_s107
                                 + 0.827216 w_k1_DIS_s107
                                 + 0.902514 w_k1_FDX_s107
                                 + 0.561066 w_k1_IBM_s107
                                 + 0.866367 w_k1_INTC_s107
                                 + 0.598634 w_k1_KO_s107
                                 + 0.387885 w_k1_MAT_s107
                                 + 0.968256 w_k1_MSFT_s107
                                 + 0.781994 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.21652 w_k1_ADM_s107
                                 + 1.11442 w_k1_AXP_s107
                                 + 1.26757 w_k1_DIS_s107
                                 + 1.22203 w_k1_FDX_s107
                                 + 0.902596 w_k1_IBM_s107
                                 + 0.973516 w_k1_INTC_s107
                                 + 0.906464 w_k1_KO_s107
                                 + 1.10316 w_k1_MAT_s107 + 1.18 w_k1_MSFT_s107
                                 + 1.04021 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.05167 w_k1_ADM_s107
                                 + 0.846333 w_k1_AXP_s107
                                 + 1.2401 w_k1_DIS_s107
                                 + 0.949192 w_k1_FDX_s107
                                 + 0.798808 w_k1_IBM_s107
                                 + 0.949696 w_k1_INTC_s107
                                 + 0.822361 w_k1_KO_s107
                                 + 0.563701 w_k1_MAT_s107
                                 + 0.987568 w_k1_MSFT_s107
                                 + 0.950972 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.774983 w_k1_ADM_s107
                                 + 0.676557 w_k1_AXP_s107
                                 + 0.611772 w_k1_DIS_s107
                                 + 0.723766 w_k1_FDX_s107
                                 + 0.773907 w_k1_IBM_s107
                                 + 0.701067 w_k1_INTC_s107
                                 + 0.5829 w_k1_KO_s107 + 0.695546 w_k1_MAT_s107
                                 + 0.819883 w_k1_MSFT_s107
                                 + 0.810779 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.58693 w_k1_ADM_s107
                                 + 1.7097 w_k1_AXP_s107 + 1.77454 w_k1_DIS_s107
                                 + 1.52886 w_k1_FDX_s107 + 1.3017 w_k1_IBM_s107
                                 + 1.6978 w_k1_INTC_s107 + 1.2097 w_k1_KO_s107
                                 + 1.13216 w_k1_MAT_s107
                                 + 1.68947 w_k1_MSFT_s107
                                 + 1.14754 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.992571 w_k1_ADM_s107
                                 + 0.821445 w_k1_AXP_s107
                                 + 0.837726 w_k1_DIS_s107
                                 + 0.849486 w_k1_FDX_s107
                                 + 0.985942 w_k1_IBM_s107
                                 + 1.04918 w_k1_INTC_s107
                                 + 0.767808 w_k1_KO_s107
                                 + 0.67866 w_k1_MAT_s107
                                 + 0.971763 w_k1_MSFT_s107
                                 + 0.83894 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.970315 w_k1_ADM_s107
                                 + 1.02488 w_k1_AXP_s107
                                 + 0.840611 w_k1_DIS_s107
                                 + 0.982144 w_k1_FDX_s107
                                 + 0.768816 w_k1_IBM_s107
                                 + 1.06782 w_k1_INTC_s107
                                 + 0.731215 w_k1_KO_s107
                                 + 0.651127 w_k1_MAT_s107
                                 + 1.01932 w_k1_MSFT_s107
                                 + 0.825479 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.923586 w_k1_ADM_s107
                                 + 1.07004 w_k1_AXP_s107
                                 + 1.06444 w_k1_DIS_s107
                                 + 0.885268 w_k1_FDX_s107
                                 + 1.42235 w_k1_IBM_s107
                                 + 1.06181 w_k1_INTC_s107
                                 + 1.10245 w_k1_KO_s107 + 1.26384 w_k1_MAT_s107
                                 + 1.00355 w_k1_MSFT_s107
                                 + 0.953397 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.02638 w_k1_ADM_s110
                                 + 1.15267 w_k1_AXP_s110
                                 + 0.880989 w_k1_DIS_s110
                                 + 1.03111 w_k1_FDX_s110
                                 + 0.914776 w_k1_IBM_s110
                                 + 1.01927 w_k1_INTC_s110
                                 + 0.98815 w_k1_KO_s110 + 1.37274 w_k1_MAT_s110
                                 + 0.959305 w_k1_MSFT_s110
                                 + 0.792572 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.23526 w_k1_ADM_s110
                                 + 1.21765 w_k1_AXP_s110
                                 + 1.43893 w_k1_DIS_s110
                                 + 1.30269 w_k1_FDX_s110
                                 + 1.31581 w_k1_IBM_s110
                                 + 1.2958 w_k1_INTC_s110 + 1.59963 w_k1_KO_s110
                                 + 1.86493 w_k1_MAT_s110
                                 + 1.29197 w_k1_MSFT_s110
                                 + 1.22899 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.880589 w_k1_ADM_s110
                                 + 0.719707 w_k1_AXP_s110
                                 + 0.929693 w_k1_DIS_s110
                                 + 0.922596 w_k1_FDX_s110
                                 + 0.676585 w_k1_IBM_s110
                                 + 1.14786 w_k1_INTC_s110
                                 + 0.981189 w_k1_KO_s110
                                 + 0.648417 w_k1_MAT_s110
                                 + 1.09909 w_k1_MSFT_s110
                                 + 0.847666 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.5243 w_k1_ADM_s110
                                 + 1.47052 w_k1_AXP_s110
                                 + 1.69567 w_k1_DIS_s110
                                 + 1.24842 w_k1_FDX_s110
                                 + 1.54389 w_k1_IBM_s110
                                 + 1.26587 w_k1_INTC_s110
                                 + 1.79628 w_k1_KO_s110 + 2.57268 w_k1_MAT_s110
                                 + 1.1644 w_k1_MSFT_s110 + 1.1922 w_k1_PG_s110
                                 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.969164 w_k1_ADM_s110
                                 + 0.736774 w_k1_AXP_s110
                                 + 0.837006 w_k1_DIS_s110
                                 + 0.832847 w_k1_FDX_s110
                                 + 0.987218 w_k1_IBM_s110
                                 + 0.8833 w_k1_INTC_s110 + 1.01926 w_k1_KO_s110
                                 + 0.852982 w_k1_MAT_s110
                                 + 0.928777 w_k1_MSFT_s110
                                 + 0.87686 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.06453 w_k1_ADM_s110
                                 + 0.924302 w_k1_AXP_s110
                                 + 1.20662 w_k1_DIS_s110
                                 + 1.16507 w_k1_FDX_s110
                                 + 0.910908 w_k1_IBM_s110
                                 + 1.05496 w_k1_INTC_s110
                                 + 1.20475 w_k1_KO_s110 + 1.00268 w_k1_MAT_s110
                                 + 1.1262 w_k1_MSFT_s110
                                 + 0.954336 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.16355 w_k1_ADM_s110
                                 + 0.757423 w_k1_AXP_s110
                                 + 1.04425 w_k1_DIS_s110
                                 + 0.933701 w_k1_FDX_s110
                                 + 1.47423 w_k1_IBM_s110
                                 + 1.13205 w_k1_INTC_s110
                                 + 1.50224 w_k1_KO_s110 + 1.39847 w_k1_MAT_s110
                                 + 1.15138 w_k1_MSFT_s110
                                 + 1.06345 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.12483 w_k1_ADM_s110
                                 + 1.20391 w_k1_AXP_s110
                                 + 0.954742 w_k1_DIS_s110
                                 + 1.08944 w_k1_FDX_s110
                                 + 1.19216 w_k1_IBM_s110
                                 + 1.07001 w_k1_INTC_s110
                                 + 0.990636 w_k1_KO_s110 + 1.5154 w_k1_MAT_s110
                                 + 0.955493 w_k1_MSFT_s110
                                 + 0.946426 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.21786 w_k1_ADM_s111
                                 + 1.37313 w_k1_AXP_s111
                                 + 1.02059 w_k1_DIS_s111
                                 + 1.06968 w_k1_FDX_s111
                                 + 0.905588 w_k1_IBM_s111
                                 + 0.952755 w_k1_INTC_s111
                                 + 1.00231 w_k1_KO_s111 + 1.04097 w_k1_MAT_s111
                                 + 1.01204 w_k1_MSFT_s111
                                 + 1.10885 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.820215 w_k1_ADM_s111
                                 + 0.768048 w_k1_AXP_s111
                                 + 0.776743 w_k1_DIS_s111
                                 + 0.815529 w_k1_FDX_s111
                                 + 1.15068 w_k1_IBM_s111
                                 + 0.959009 w_k1_INTC_s111
                                 + 0.90806 w_k1_KO_s111
                                 + 0.880818 w_k1_MAT_s111
                                 + 0.972825 w_k1_MSFT_s111
                                 + 0.956809 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.932657 w_k1_ADM_s111
                                 + 1.36314 w_k1_AXP_s111
                                 + 1.06238 w_k1_DIS_s111
                                 + 1.01596 w_k1_FDX_s111
                                 + 1.56697 w_k1_IBM_s111
                                 + 1.06326 w_k1_INTC_s111
                                 + 1.08644 w_k1_KO_s111 + 1.83371 w_k1_MAT_s111
                                 + 0.938929 w_k1_MSFT_s111
                                 + 1.17953 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.41161 w_k1_ADM_s111
                                 + 1.57449 w_k1_AXP_s111 + 1.3762 w_k1_DIS_s111
                                 + 1.40276 w_k1_FDX_s111
                                 + 1.37083 w_k1_IBM_s111
                                 + 1.2914 w_k1_INTC_s111 + 1.39366 w_k1_KO_s111
                                 + 1.50488 w_k1_MAT_s111
                                 + 1.2786 w_k1_MSFT_s111 + 1.20642 w_k1_PG_s111
                                 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.881055 w_k1_ADM_s111
                                 + 0.76722 w_k1_AXP_s111
                                 + 0.881158 w_k1_DIS_s111
                                 + 0.966272 w_k1_FDX_s111
                                 + 0.667931 w_k1_IBM_s111
                                 + 0.963598 w_k1_INTC_s111
                                 + 0.815913 w_k1_KO_s111
                                 + 0.693734 w_k1_MAT_s111
                                 + 0.932033 w_k1_MSFT_s111
                                 + 1.01407 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.17134 w_k1_ADM_s111
                                 + 1.11214 w_k1_AXP_s111
                                 + 1.10732 w_k1_DIS_s111 + 1.0723 w_k1_FDX_s111
                                 + 1.09313 w_k1_IBM_s111
                                 + 1.01801 w_k1_INTC_s111
                                 + 1.07103 w_k1_KO_s111 + 1.11015 w_k1_MAT_s111
                                 + 1.13292 w_k1_MSFT_s111
                                 + 1.10253 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.11412 w_k1_ADM_s111
                                 + 1.25573 w_k1_AXP_s111 + 1.3261 w_k1_DIS_s111
                                 + 1.23803 w_k1_FDX_s111
                                 + 0.849779 w_k1_IBM_s111
                                 + 1.72084 w_k1_INTC_s111
                                 + 0.967237 w_k1_KO_s111
                                 + 0.563937 w_k1_MAT_s111
                                 + 1.14294 w_k1_MSFT_s111
                                 + 1.14884 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.632673 w_k1_ADM_s111
                                 + 0.795906 w_k1_AXP_s111
                                 + 0.623154 w_k1_DIS_s111
                                 + 0.765125 w_k1_FDX_s111
                                 + 0.79242 w_k1_IBM_s111
                                 + 0.812262 w_k1_INTC_s111
                                 + 0.741974 w_k1_KO_s111
                                 + 0.922009 w_k1_MAT_s111
                                 + 0.725583 w_k1_MSFT_s111
                                 + 0.923552 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.29392 w_k1_ADM_s112
                                 + 1.01233 w_k1_AXP_s112 + 1.0093 w_k1_DIS_s112
                                 + 1.093 w_k1_FDX_s112 + 0.751694 w_k1_IBM_s112
                                 + 0.909906 w_k1_INTC_s112
                                 + 0.871489 w_k1_KO_s112
                                 + 0.795408 w_k1_MAT_s112
                                 + 0.954043 w_k1_MSFT_s112
                                 + 0.86302 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.02614 w_k1_ADM_s112
                                 + 1.29138 w_k1_AXP_s112
                                 + 1.62045 w_k1_DIS_s112
                                 + 1.37382 w_k1_FDX_s112
                                 + 1.00873 w_k1_IBM_s112
                                 + 1.31645 w_k1_INTC_s112
                                 + 1.30724 w_k1_KO_s112 + 1.047 w_k1_MAT_s112
                                 + 1.42339 w_k1_MSFT_s112
                                 + 1.16508 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.933049 w_k1_ADM_s112
                                 + 0.903147 w_k1_AXP_s112
                                 + 1.13622 w_k1_DIS_s112
                                 + 0.97359 w_k1_FDX_s112
                                 + 0.880827 w_k1_IBM_s112
                                 + 1.16254 w_k1_INTC_s112
                                 + 1.09954 w_k1_KO_s112
                                 + 0.910856 w_k1_MAT_s112
                                 + 0.975957 w_k1_MSFT_s112
                                 + 0.911294 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.926546 w_k1_ADM_s112
                                 + 1.4491 w_k1_AXP_s112 + 1.12942 w_k1_DIS_s112
                                 + 1.01141 w_k1_FDX_s112
                                 + 1.62734 w_k1_IBM_s112
                                 + 0.985913 w_k1_INTC_s112
                                 + 1.32593 w_k1_KO_s112 + 1.70825 w_k1_MAT_s112
                                 + 0.869885 w_k1_MSFT_s112
                                 + 1.03515 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.00455 w_k1_ADM_s112
                                 + 1.11113 w_k1_AXP_s112
                                 + 1.10689 w_k1_DIS_s112
                                 + 1.05576 w_k1_FDX_s112 + 1.0833 w_k1_IBM_s112
                                 + 1.06193 w_k1_INTC_s112
                                 + 0.965659 w_k1_KO_s112
                                 + 1.23081 w_k1_MAT_s112
                                 + 0.94044 w_k1_MSFT_s112
                                 + 0.971159 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.24259 w_k1_ADM_s112
                                 + 1.52068 w_k1_AXP_s112 + 1.4883 w_k1_DIS_s112
                                 + 1.28454 w_k1_FDX_s112
                                 + 1.24042 w_k1_IBM_s112
                                 + 1.11734 w_k1_INTC_s112
                                 + 1.17277 w_k1_KO_s112 + 1.3982 w_k1_MAT_s112
                                 + 1.04409 w_k1_MSFT_s112
                                 + 0.922566 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.782573 w_k1_ADM_s112
                                 + 0.992596 w_k1_AXP_s112
                                 + 1.17581 w_k1_DIS_s112
                                 + 1.01341 w_k1_FDX_s112
                                 + 1.31995 w_k1_IBM_s112
                                 + 1.04912 w_k1_INTC_s112 + 1.216 w_k1_KO_s112
                                 + 1.26043 w_k1_MAT_s112
                                 + 1.01991 w_k1_MSFT_s112
                                 + 1.04777 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.846524 w_k1_ADM_s112
                                 + 0.81669 w_k1_AXP_s112
                                 + 0.732796 w_k1_DIS_s112
                                 + 0.74294 w_k1_FDX_s112
                                 + 0.867545 w_k1_IBM_s112
                                 + 0.725069 w_k1_INTC_s112
                                 + 0.728142 w_k1_KO_s112
                                 + 0.879198 w_k1_MAT_s112
                                 + 0.716354 w_k1_MSFT_s112
                                 + 0.766571 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.811403 w_k1_ADM_s113
                                 + 0.689473 w_k1_AXP_s113
                                 + 0.569899 w_k1_DIS_s113
                                 + 0.636593 w_k1_FDX_s113
                                 + 0.861751 w_k1_IBM_s113
                                 + 0.64823 w_k1_INTC_s113
                                 + 0.842662 w_k1_KO_s113
                                 + 0.772721 w_k1_MAT_s113
                                 + 0.643059 w_k1_MSFT_s113
                                 + 0.822397 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.40252 w_k1_ADM_s113
                                 + 1.08164 w_k1_AXP_s113
                                 + 1.35621 w_k1_DIS_s113
                                 + 1.29716 w_k1_FDX_s113
                                 + 0.78929 w_k1_IBM_s113
                                 + 1.22804 w_k1_INTC_s113
                                 + 1.12765 w_k1_KO_s113 + 1.13845 w_k1_MAT_s113
                                 + 1.27268 w_k1_MSFT_s113
                                 + 1.22477 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.888362 w_k1_ADM_s113
                                 + 0.75759 w_k1_AXP_s113
                                 + 0.875962 w_k1_DIS_s113
                                 + 0.857031 w_k1_FDX_s113
                                 + 1.08914 w_k1_IBM_s113
                                 + 1.02094 w_k1_INTC_s113
                                 + 1.11539 w_k1_KO_s113
                                 + 0.719394 w_k1_MAT_s113
                                 + 0.867641 w_k1_MSFT_s113
                                 + 0.985301 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.06404 w_k1_ADM_s113
                                 + 1.06792 w_k1_AXP_s113
                                 + 1.29988 w_k1_DIS_s113
                                 + 0.947452 w_k1_FDX_s113
                                 + 1.16026 w_k1_IBM_s113
                                 + 0.85604 w_k1_INTC_s113
                                 + 1.40089 w_k1_KO_s113 + 1.4081 w_k1_MAT_s113
                                 + 1.13719 w_k1_MSFT_s113
                                 + 1.24984 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.2462 w_k1_ADM_s113
                                 + 1.05634 w_k1_AXP_s113
                                 + 1.06018 w_k1_DIS_s113
                                 + 1.05806 w_k1_FDX_s113
                                 + 0.936596 w_k1_IBM_s113
                                 + 0.998188 w_k1_INTC_s113
                                 + 1.12112 w_k1_KO_s113
                                 + 0.647471 w_k1_MAT_s113
                                 + 0.953473 w_k1_MSFT_s113
                                 + 1.16211 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.07948 w_k1_ADM_s113
                                 + 1.31318 w_k1_AXP_s113
                                 + 0.948396 w_k1_DIS_s113
                                 + 0.991833 w_k1_FDX_s113
                                 + 1.30529 w_k1_IBM_s113
                                 + 0.771178 w_k1_INTC_s113
                                 + 1.30591 w_k1_KO_s113 + 1.84156 w_k1_MAT_s113
                                 + 0.751239 w_k1_MSFT_s113
                                 + 1.07746 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.47993 w_k1_ADM_s113
                                 + 1.5667 w_k1_AXP_s113 + 1.71563 w_k1_DIS_s113
                                 + 1.34348 w_k1_FDX_s113
                                 + 1.29225 w_k1_IBM_s113
                                 + 1.26412 w_k1_INTC_s113
                                 + 1.51644 w_k1_KO_s113 + 1.05876 w_k1_MAT_s113
                                 + 1.433 w_k1_MSFT_s113 + 1.32852 w_k1_PG_s113
                                 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.966064 w_k1_ADM_s113
                                 + 0.831261 w_k1_AXP_s113
                                 + 1.09236 w_k1_DIS_s113
                                 + 0.98522 w_k1_FDX_s113 + 1.0841 w_k1_IBM_s113
                                 + 1.06097 w_k1_INTC_s113
                                 + 1.30772 w_k1_KO_s113 + 1.13683 w_k1_MAT_s113
                                 + 1.15396 w_k1_MSFT_s113
                                 + 1.10832 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.947099 w_k1_ADM_s114
                                 + 1.26445 w_k1_AXP_s114
                                 + 0.806461 w_k1_DIS_s114
                                 + 1.05245 w_k1_FDX_s114
                                 + 0.805698 w_k1_IBM_s114
                                 + 1.21944 w_k1_INTC_s114
                                 + 0.856712 w_k1_KO_s114 + 1.0559 w_k1_MAT_s114
                                 + 0.991595 w_k1_MSFT_s114
                                 + 0.847278 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.916452 w_k1_ADM_s114
                                 + 0.671397 w_k1_AXP_s114
                                 + 0.535303 w_k1_DIS_s114
                                 + 0.691023 w_k1_FDX_s114
                                 + 0.606747 w_k1_IBM_s114
                                 + 0.829386 w_k1_INTC_s114
                                 + 0.665281 w_k1_KO_s114
                                 + 0.979466 w_k1_MAT_s114
                                 + 0.836736 w_k1_MSFT_s114
                                 + 0.729027 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.04287 w_k1_ADM_s114
                                 + 0.976203 w_k1_AXP_s114
                                 + 1.08547 w_k1_DIS_s114
                                 + 1.00107 w_k1_FDX_s114
                                 + 0.868009 w_k1_IBM_s114
                                 + 1.01116 w_k1_INTC_s114
                                 + 1.10061 w_k1_KO_s114 + 1.28044 w_k1_MAT_s114
                                 + 1.06497 w_k1_MSFT_s114
                                 + 1.00328 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.33724 w_k1_ADM_s114
                                 + 1.12116 w_k1_AXP_s114
                                 + 0.987923 w_k1_DIS_s114
                                 + 0.950745 w_k1_FDX_s114
                                 + 1.16495 w_k1_IBM_s114
                                 + 1.03145 w_k1_INTC_s114
                                 + 1.08756 w_k1_KO_s114 + 1.33213 w_k1_MAT_s114
                                 + 1.04343 w_k1_MSFT_s114 + 1.024 w_k1_PG_s114
                                 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.866709 w_k1_ADM_s114
                                 + 0.914064 w_k1_AXP_s114
                                 + 0.716484 w_k1_DIS_s114
                                 + 0.77588 w_k1_FDX_s114
                                 + 0.981744 w_k1_IBM_s114
                                 + 0.956973 w_k1_INTC_s114
                                 + 0.882759 w_k1_KO_s114
                                 + 0.897541 w_k1_MAT_s114
                                 + 0.936935 w_k1_MSFT_s114
                                 + 0.883044 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.40913 w_k1_ADM_s114
                                 + 1.60726 w_k1_AXP_s114
                                 + 1.42194 w_k1_DIS_s114
                                 + 1.14992 w_k1_FDX_s114
                                 + 1.83398 w_k1_IBM_s114
                                 + 1.10902 w_k1_INTC_s114 + 1.4955 w_k1_KO_s114
                                 + 1.91795 w_k1_MAT_s114
                                 + 1.25592 w_k1_MSFT_s114
                                 + 1.17513 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.23112 w_k1_ADM_s114
                                 + 1.1823 w_k1_AXP_s114 + 1.43586 w_k1_DIS_s114
                                 + 1.27931 w_k1_FDX_s114
                                 + 0.976286 w_k1_IBM_s114
                                 + 1.21941 w_k1_INTC_s114
                                 + 1.07288 w_k1_KO_s114 + 1.05654 w_k1_MAT_s114
                                 + 1.25944 w_k1_MSFT_s114
                                 + 1.07172 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.40479 w_k1_ADM_s114
                                 + 0.888837 w_k1_AXP_s114
                                 + 0.918281 w_k1_DIS_s114
                                 + 0.943481 w_k1_FDX_s114
                                 + 0.690218 w_k1_IBM_s114
                                 + 1.13355 w_k1_INTC_s114
                                 + 0.985985 w_k1_KO_s114
                                 + 0.770356 w_k1_MAT_s114
                                 + 1.25063 w_k1_MSFT_s114
                                 + 0.950139 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.44691 w_k1_ADM_s115
                                 + 1.52754 w_k1_AXP_s115
                                 + 1.77881 w_k1_DIS_s115
                                 + 1.28191 w_k1_FDX_s115
                                 + 1.36539 w_k1_IBM_s115
                                 + 1.49592 w_k1_INTC_s115
                                 + 1.59516 w_k1_KO_s115 + 1.00548 w_k1_MAT_s115
                                 + 1.43589 w_k1_MSFT_s115
                                 + 1.18309 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.936265 w_k1_ADM_s115
                                 + 0.871049 w_k1_AXP_s115
                                 + 0.711829 w_k1_DIS_s115
                                 + 0.782133 w_k1_FDX_s115
                                 + 1.13436 w_k1_IBM_s115
                                 + 0.84886 w_k1_INTC_s115
                                 + 0.966959 w_k1_KO_s115
                                 + 1.28182 w_k1_MAT_s115
                                 + 0.856983 w_k1_MSFT_s115
                                 + 0.997035 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.932828 w_k1_ADM_s115
                                 + 1.35128 w_k1_AXP_s115
                                 + 1.23734 w_k1_DIS_s115
                                 + 1.15667 w_k1_FDX_s115
                                 + 0.946388 w_k1_IBM_s115
                                 + 1.19793 w_k1_INTC_s115
                                 + 1.15979 w_k1_KO_s115 + 1.22516 w_k1_MAT_s115
                                 + 1.31546 w_k1_MSFT_s115 + 1.1406 w_k1_PG_s115
                                 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.06853 w_k1_ADM_s115
                                 + 1.38505 w_k1_AXP_s115
                                 + 0.967972 w_k1_DIS_s115
                                 + 0.98758 w_k1_FDX_s115
                                 + 1.26349 w_k1_IBM_s115
                                 + 0.915453 w_k1_INTC_s115
                                 + 1.17225 w_k1_KO_s115 + 1.29335 w_k1_MAT_s115
                                 + 0.913787 w_k1_MSFT_s115
                                 + 1.15817 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.04398 w_k1_ADM_s115
                                 + 0.958293 w_k1_AXP_s115
                                 + 0.978302 w_k1_DIS_s115
                                 + 0.916572 w_k1_FDX_s115
                                 + 0.935892 w_k1_IBM_s115
                                 + 1.10686 w_k1_INTC_s115
                                 + 0.899339 w_k1_KO_s115
                                 + 0.622109 w_k1_MAT_s115
                                 + 0.984481 w_k1_MSFT_s115
                                 + 0.920762 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.992212 w_k1_ADM_s115
                                 + 1.30055 w_k1_AXP_s115
                                 + 0.968298 w_k1_DIS_s115
                                 + 1.02303 w_k1_FDX_s115
                                 + 1.85425 w_k1_IBM_s115
                                 + 1.17444 w_k1_INTC_s115
                                 + 1.43268 w_k1_KO_s115 + 1.7127 w_k1_MAT_s115
                                 + 0.996389 w_k1_MSFT_s115
                                 + 1.3125 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.870277 w_k1_ADM_s115
                                 + 1.14503 w_k1_AXP_s115
                                 + 0.733139 w_k1_DIS_s115
                                 + 0.918334 w_k1_FDX_s115
                                 + 0.644203 w_k1_IBM_s115
                                 + 0.873413 w_k1_INTC_s115
                                 + 0.824726 w_k1_KO_s115
                                 + 0.628794 w_k1_MAT_s115
                                 + 0.919185 w_k1_MSFT_s115
                                 + 0.880344 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.799306 w_k1_ADM_s115
                                 + 0.812154 w_k1_AXP_s115
                                 + 0.547389 w_k1_DIS_s115
                                 + 0.727062 w_k1_FDX_s115
                                 + 0.813156 w_k1_IBM_s115
                                 + 0.787713 w_k1_INTC_s115
                                 + 0.723727 w_k1_KO_s115
                                 + 0.603834 w_k1_MAT_s115
                                 + 0.759279 w_k1_MSFT_s115
                                 + 0.956447 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.08789 w_k1_ADM_s116
                                 + 1.30157 w_k1_AXP_s116
                                 + 1.32168 w_k1_DIS_s116
                                 + 1.30406 w_k1_FDX_s116
                                 + 1.14257 w_k1_IBM_s116
                                 + 1.43697 w_k1_INTC_s116
                                 + 1.12303 w_k1_KO_s116
                                 + 0.904882 w_k1_MAT_s116
                                 + 1.14981 w_k1_MSFT_s116
                                 + 1.06426 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.733441 w_k1_ADM_s116
                                 + 0.883884 w_k1_AXP_s116
                                 + 0.860325 w_k1_DIS_s116
                                 + 0.839842 w_k1_FDX_s116
                                 + 1.02845 w_k1_IBM_s116
                                 + 1.03386 w_k1_INTC_s116
                                 + 0.910061 w_k1_KO_s116 + 1.1832 w_k1_MAT_s116
                                 + 0.835944 w_k1_MSFT_s116
                                 + 0.879034 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.922308 w_k1_ADM_s116
                                 + 0.942843 w_k1_AXP_s116
                                 + 0.852239 w_k1_DIS_s116
                                 + 0.92565 w_k1_FDX_s116
                                 + 0.821455 w_k1_IBM_s116
                                 + 0.899325 w_k1_INTC_s116
                                 + 0.758197 w_k1_KO_s116
                                 + 0.85783 w_k1_MAT_s116
                                 + 0.810136 w_k1_MSFT_s116
                                 + 0.906961 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.03242 w_k1_ADM_s116
                                 + 0.974303 w_k1_AXP_s116
                                 + 1.07154 w_k1_DIS_s116
                                 + 1.19265 w_k1_FDX_s116
                                 + 0.814663 w_k1_IBM_s116
                                 + 1.09464 w_k1_INTC_s116
                                 + 0.900299 w_k1_KO_s116
                                 + 0.759671 w_k1_MAT_s116
                                 + 1.03233 w_k1_MSFT_s116
                                 + 1.00332 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.857097 w_k1_ADM_s116
                                 + 1.41289 w_k1_AXP_s116
                                 + 0.984522 w_k1_DIS_s116
                                 + 1.02031 w_k1_FDX_s116
                                 + 1.06269 w_k1_IBM_s116
                                 + 1.06998 w_k1_INTC_s116
                                 + 0.946578 w_k1_KO_s116
                                 + 1.71805 w_k1_MAT_s116
                                 + 0.992074 w_k1_MSFT_s116
                                 + 0.997947 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.993177 w_k1_ADM_s116
                                 + 0.886052 w_k1_AXP_s116
                                 + 0.985597 w_k1_DIS_s116
                                 + 0.873168 w_k1_FDX_s116
                                 + 1.43895 w_k1_IBM_s116
                                 + 0.912558 w_k1_INTC_s116
                                 + 1.07377 w_k1_KO_s116 + 1.09337 w_k1_MAT_s116
                                 + 0.914372 w_k1_MSFT_s116
                                 + 0.889883 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.785661 w_k1_ADM_s116
                                 + 0.574913 w_k1_AXP_s116
                                 + 0.585076 w_k1_DIS_s116
                                 + 0.727855 w_k1_FDX_s116
                                 + 0.677451 w_k1_IBM_s116
                                 + 0.765747 w_k1_INTC_s116
                                 + 0.677856 w_k1_KO_s116
                                 + 0.620089 w_k1_MAT_s116
                                 + 0.767185 w_k1_MSFT_s116
                                 + 0.767142 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.21664 w_k1_ADM_s116
                                 + 1.37392 w_k1_AXP_s116
                                 + 1.54123 w_k1_DIS_s116
                                 + 1.29224 w_k1_FDX_s116
                                 + 1.84607 w_k1_IBM_s116
                                 + 1.22682 w_k1_INTC_s116
                                 + 1.32905 w_k1_KO_s116 + 2.18782 w_k1_MAT_s116
                                 + 1.23833 w_k1_MSFT_s116
                                 + 1.14956 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.792694 w_k1_ADM_s117
                                 + 0.740687 w_k1_AXP_s117
                                 + 0.609317 w_k1_DIS_s117
                                 + 0.799221 w_k1_FDX_s117
                                 + 0.694022 w_k1_IBM_s117
                                 + 0.802033 w_k1_INTC_s117
                                 + 0.722032 w_k1_KO_s117
                                 + 0.78139 w_k1_MAT_s117
                                 + 0.833293 w_k1_MSFT_s117
                                 + 0.801288 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.937687 w_k1_ADM_s117
                                 + 0.960928 w_k1_AXP_s117
                                 + 0.839998 w_k1_DIS_s117
                                 + 0.873684 w_k1_FDX_s117
                                 + 1.36974 w_k1_IBM_s117
                                 + 1.09283 w_k1_INTC_s117
                                 + 1.24552 w_k1_KO_s117 + 1.32887 w_k1_MAT_s117
                                 + 0.995138 w_k1_MSFT_s117
                                 + 0.998008 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.95316 w_k1_ADM_s117
                                 + 1.10002 w_k1_AXP_s117
                                 + 1.45121 w_k1_DIS_s117
                                 + 1.22244 w_k1_FDX_s117
                                 + 1.27073 w_k1_IBM_s117
                                 + 1.2387 w_k1_INTC_s117 + 1.28751 w_k1_KO_s117
                                 + 1.34372 w_k1_MAT_s117
                                 + 1.42507 w_k1_MSFT_s117
                                 + 1.10059 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.03896 w_k1_ADM_s117
                                 + 0.720983 w_k1_AXP_s117
                                 + 1.0279 w_k1_DIS_s117 + 1.00855 w_k1_FDX_s117
                                 + 0.889367 w_k1_IBM_s117
                                 + 1.20515 w_k1_INTC_s117
                                 + 0.945566 w_k1_KO_s117
                                 + 0.585769 w_k1_MAT_s117
                                 + 1.08638 w_k1_MSFT_s117
                                 + 0.935326 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.847349 w_k1_ADM_s117
                                 + 1.04242 w_k1_AXP_s117
                                 + 0.814814 w_k1_DIS_s117
                                 + 0.939712 w_k1_FDX_s117
                                 + 1.13877 w_k1_IBM_s117
                                 + 0.921062 w_k1_INTC_s117
                                 + 0.971672 w_k1_KO_s117 + 1.0161 w_k1_MAT_s117
                                 + 0.910117 w_k1_MSFT_s117
                                 + 0.88751 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.1177 w_k1_ADM_s117
                                 + 1.07338 w_k1_AXP_s117
                                 + 1.07757 w_k1_DIS_s117
                                 + 1.02498 w_k1_FDX_s117
                                 + 0.92402 w_k1_IBM_s117
                                 + 1.06027 w_k1_INTC_s117
                                 + 1.08701 w_k1_KO_s117 + 1.15153 w_k1_MAT_s117
                                 + 1.08245 w_k1_MSFT_s117
                                 + 1.00683 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.30401 w_k1_ADM_s117
                                 + 1.31487 w_k1_AXP_s117
                                 + 1.43256 w_k1_DIS_s117
                                 + 1.24409 w_k1_FDX_s117
                                 + 1.13632 w_k1_IBM_s117
                                 + 1.24055 w_k1_INTC_s117
                                 + 1.46582 w_k1_KO_s117 + 1.73799 w_k1_MAT_s117
                                 + 1.45868 w_k1_MSFT_s117
                                 + 1.26423 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.773076 w_k1_ADM_s117
                                 + 0.798905 w_k1_AXP_s117
                                 + 0.728178 w_k1_DIS_s117
                                 + 0.821222 w_k1_FDX_s117
                                 + 0.873016 w_k1_IBM_s117
                                 + 0.906567 w_k1_INTC_s117
                                 + 0.859702 w_k1_KO_s117
                                 + 1.04233 w_k1_MAT_s117
                                 + 0.962156 w_k1_MSFT_s117
                                 + 0.852504 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0824809705316866 w_k1_ADM_s000
                                 + 0.0864157968470701 w_k1_ADM_s001
                                 + 0.0748458165041471 w_k1_ADM_s002
                                 + 0.0738582573052968 w_k1_ADM_s003
                                 + 0.144391427675921 w_k1_ADM_s004
                                 + 0.0338893633279055 w_k1_ADM_s005
                                 + 0.0606152396453042 w_k1_ADM_s006
                                 + 0.0328119566697851 w_k1_ADM_s007
                                 + 0.00792517901531303 w_k1_ADM_s010
                                 + 0.0119649256343193 w_k1_ADM_s011
                                 + 0.0263061354831394 w_k1_ADM_s012
                                 + 0.0233769913655366 w_k1_ADM_s013
                                 + 0.00898774415403951 w_k1_ADM_s014
                                 + 0.0276783810062361 w_k1_ADM_s015
                                 + 0.00988108989004419 w_k1_ADM_s016
                                 + 0.0194203022644431 w_k1_ADM_s017
                                 + 0.0275716303329849 w_k1_ADM_s100
                                 + 0.0398125932833233 w_k1_ADM_s101
                                 + 0.0257920589024351 w_k1_ADM_s102
                                 + 0.015320169749986 w_k1_ADM_s103
                                 + 0.016228640116312 w_k1_ADM_s104
                                 + 0.0118071938859694 w_k1_ADM_s105
                                 + 0.0196210782389321 w_k1_ADM_s106
                                 + 0.01666227343935 w_k1_ADM_s107
                                 + 0.00998281638870725 w_k1_ADM_s110
                                 + 0.0168551173523866 w_k1_ADM_s111
                                 + 0.0174135745962068 w_k1_ADM_s112
                                 + 0.0123182276344011 w_k1_ADM_s113
                                 + 0.025015510630375 w_k1_ADM_s114
                                 + 0.0132746212982069 w_k1_ADM_s115
                                 + 0.0203031303048738 w_k1_ADM_s116
                                 + 0.0161572939394197 w_k1_ADM_s117
                                 + 0.0661736257135755 w_k1_AXP_s000
                                 + 0.0867108395366702 w_k1_AXP_s001
                                 + 0.0847319935837803 w_k1_AXP_s002
                                 + 0.0665966193477923 w_k1_AXP_s003
                                 + 0.127776160474472 w_k1_AXP_s004
                                 + 0.0340955104318178 w_k1_AXP_s005
                                 + 0.0613848224348681 w_k1_AXP_s006
                                 + 0.035186008682049 w_k1_AXP_s007
                                 + 0.0083151842175719 w_k1_AXP_s010
                                 + 0.0115599177495432 w_k1_AXP_s011
                                 + 0.0235234117175687 w_k1_AXP_s012
                                 + 0.0217916281016961 w_k1_AXP_s013
                                 + 0.0102096820089309 w_k1_AXP_s014
                                 + 0.0274978883367415 w_k1_AXP_s015
                                 + 0.0102606296887962 w_k1_AXP_s016
                                 + 0.0168941912952965 w_k1_AXP_s017
                                 + 0.0285038296657544 w_k1_AXP_s100
                                 + 0.0409931737691211 w_k1_AXP_s101
                                 + 0.0274959493696576 w_k1_AXP_s102
                                 + 0.0162568719330204 w_k1_AXP_s103
                                 + 0.0176449350952393 w_k1_AXP_s104
                                 + 0.0124883943765559 w_k1_AXP_s105
                                 + 0.0192259115433402 w_k1_AXP_s106
                                 + 0.0157657361149103 w_k1_AXP_s107
                                 + 0.00889563780824765 w_k1_AXP_s110
                                 + 0.0191655002572883 w_k1_AXP_s111
                                 + 0.0183688872686682 w_k1_AXP_s112
                                 + 0.0111748687432777 w_k1_AXP_s113
                                 + 0.0240471889180788 w_k1_AXP_s114
                                 + 0.0152963292269201 w_k1_AXP_s115
                                 + 0.0220809696336613 w_k1_AXP_s116
                                 + 0.0166531284255085 w_k1_AXP_s117
                                 + 0.0741043015598276 w_k1_DIS_s000
                                 + 0.0895142709983048 w_k1_DIS_s001
                                 + 0.0917819135365077 w_k1_DIS_s002
                                 + 0.0699839051887872 w_k1_DIS_s003
                                 + 0.140460576804884 w_k1_DIS_s004
                                 + 0.033696892202822 w_k1_DIS_s005
                                 + 0.0588543516598069 w_k1_DIS_s006
                                 + 0.0337769670918649 w_k1_DIS_s007
                                 + 0.00900627271925754 w_k1_DIS_s010
                                 + 0.0125701626928601 w_k1_DIS_s011
                                 + 0.0251593887655097 w_k1_DIS_s012
                                 + 0.0239867192884695 w_k1_DIS_s013
                                 + 0.00922076033436208 w_k1_DIS_s014
                                 + 0.0277512517177583 w_k1_DIS_s015
                                 + 0.0101656638067001 w_k1_DIS_s016
                                 + 0.0173777078896287 w_k1_DIS_s017
                                 + 0.0304550203242557 w_k1_DIS_s100
                                 + 0.0409901533236379 w_k1_DIS_s101
                                 + 0.0275959461077544 w_k1_DIS_s102
                                 + 0.0168649022155921 w_k1_DIS_s103
                                 + 0.018600901286527 w_k1_DIS_s104
                                 + 0.0120798554678419 w_k1_DIS_s105
                                 + 0.0179800525085703 w_k1_DIS_s106
                                 + 0.0164880344635412 w_k1_DIS_s107
                                 + 0.00992715932038686 w_k1_DIS_s110
                                 + 0.0166792930131942 w_k1_DIS_s111
                                 + 0.0186233464755081 w_k1_DIS_s112
                                 + 0.0121178709195262 w_k1_DIS_s113
                                 + 0.0228891481417821 w_k1_DIS_s114
                                 + 0.0133689252660795 w_k1_DIS_s115
                                 + 0.0213514698322305 w_k1_DIS_s116
                                 + 0.017756873471235 w_k1_DIS_s117
                                 + 0.0728625511624082 w_k1_FDX_s000
                                 + 0.0881022537039008 w_k1_FDX_s001
                                 + 0.0824755041989769 w_k1_FDX_s002
                                 + 0.0658266560477959 w_k1_FDX_s003
                                 + 0.137665802061116 w_k1_FDX_s004
                                 + 0.0324569785619602 w_k1_FDX_s005
                                 + 0.057427128803636 w_k1_FDX_s006
                                 + 0.0357203257606607 w_k1_FDX_s007
                                 + 0.00863431738662599 w_k1_FDX_s010
                                 + 0.0126682912208334 w_k1_FDX_s011
                                 + 0.0251297851818057 w_k1_FDX_s012
                                 + 0.0241431423914526 w_k1_FDX_s013
                                 + 0.00930828304229803 w_k1_FDX_s014
                                 + 0.0283675110906492 w_k1_FDX_s015
                                 + 0.00964494991226863 w_k1_FDX_s016
                                 + 0.0177362218728798 w_k1_FDX_s017
                                 + 0.0304787657683352 w_k1_FDX_s100
                                 + 0.0404478004813371 w_k1_FDX_s101
                                 + 0.0249995304800821 w_k1_FDX_s102
                                 + 0.0153377008063683 w_k1_FDX_s103
                                 + 0.0175188406503313 w_k1_FDX_s104
                                 + 0.0122209609511645 w_k1_FDX_s105
                                 + 0.0191415599308375 w_k1_FDX_s106
                                 + 0.0156216393254832 w_k1_FDX_s107
                                 + 0.00940909742104247 w_k1_FDX_s110
                                 + 0.0173788917917814 w_k1_FDX_s111
                                 + 0.0175044100843555 w_k1_FDX_s112
                                 + 0.01118997072026 w_k1_FDX_s113
                                 + 0.021991491236969 w_k1_FDX_s114
                                 + 0.0128793407236781 w_k1_FDX_s115
                                 + 0.0217890332849759 w_k1_FDX_s116
                                 + 0.0168365198676531 w_k1_FDX_s117
                                 + 0.0729164695662841 w_k1_IBM_s000
                                 + 0.0944162839645828 w_k1_IBM_s001
                                 + 0.0843135654699152 w_k1_IBM_s002
                                 + 0.0687669839851728 w_k1_IBM_s003
                                 + 0.139084922283372 w_k1_IBM_s004
                                 + 0.0325218371384956 w_k1_IBM_s005
                                 + 0.0599439648416123 w_k1_IBM_s006
                                 + 0.0358090048227247 w_k1_IBM_s007
                                 + 0.00831709081656391 w_k1_IBM_s010
                                 + 0.0124485572707508 w_k1_IBM_s011
                                 + 0.0253254573609276 w_k1_IBM_s012
                                 + 0.022257994916249 w_k1_IBM_s013
                                 + 0.00885758225415422 w_k1_IBM_s014
                                 + 0.0276067288493637 w_k1_IBM_s015
                                 + 0.00971607026712988 w_k1_IBM_s016
                                 + 0.0176618547321152 w_k1_IBM_s017
                                 + 0.0320020152139388 w_k1_IBM_s100
                                 + 0.0413345776630042 w_k1_IBM_s101
                                 + 0.0262148736299758 w_k1_IBM_s102
                                 + 0.0161788520042745 w_k1_IBM_s103
                                 + 0.0177016636661281 w_k1_IBM_s104
                                 + 0.0120823990047872 w_k1_IBM_s105
                                 + 0.0191611199973506 w_k1_IBM_s106
                                 + 0.0157450207739465 w_k1_IBM_s107
                                 + 0.0100978918849121 w_k1_IBM_s110
                                 + 0.0189006447013584 w_k1_IBM_s111
                                 + 0.0176418231379806 w_k1_IBM_s112
                                 + 0.0116520316014457 w_k1_IBM_s113
                                 + 0.0228807853870921 w_k1_IBM_s114
                                 + 0.0142299294148437 w_k1_IBM_s115
                                 + 0.022193942875647 w_k1_IBM_s116
                                 + 0.0173967235087177 w_k1_IBM_s117
                                 + 0.0670933106356849 w_k1_INTC_s000
                                 + 0.0854368219326017 w_k1_INTC_s001
                                 + 0.0869921022565721 w_k1_INTC_s002
                                 + 0.0650420662276015 w_k1_INTC_s003
                                 + 0.150452671036214 w_k1_INTC_s004
                                 + 0.0338684110636884 w_k1_INTC_s005
                                 + 0.060366061530867 w_k1_INTC_s006
                                 + 0.0377178302726214 w_k1_INTC_s007
                                 + 0.00872718581915286 w_k1_INTC_s010
                                 + 0.0108278800722192 w_k1_INTC_s011
                                 + 0.0261510786309075 w_k1_INTC_s012
                                 + 0.0178276799845609 w_k1_INTC_s013
                                 + 0.0102188212184584 w_k1_INTC_s014
                                 + 0.0286442595442944 w_k1_INTC_s015
                                 + 0.0112983233284875 w_k1_INTC_s016
                                 + 0.0193164045863853 w_k1_INTC_s017
                                 + 0.0321605861945251 w_k1_INTC_s100
                                 + 0.0448055439033615 w_k1_INTC_s101
                                 + 0.0271098393924843 w_k1_INTC_s102
                                 + 0.0146327415842969 w_k1_INTC_s103
                                 + 0.0148766682230898 w_k1_INTC_s104
                                 + 0.0130363457131312 w_k1_INTC_s105
                                 + 0.0208910213171162 w_k1_INTC_s106
                                 + 0.0165384497135326 w_k1_INTC_s107
                                 + 0.00974977776233085 w_k1_INTC_s110
                                 + 0.0183014909333374 w_k1_INTC_s111
                                 + 0.0168088876470174 w_k1_INTC_s112
                                 + 0.0111768715013986 w_k1_INTC_s113
                                 + 0.0233956308680074 w_k1_INTC_s114
                                 + 0.0138688939735196 w_k1_INTC_s115
                                 + 0.0228795966642691 w_k1_INTC_s116
                                 + 0.0176328291227341 w_k1_INTC_s117
                                 + 0.0693260147581244 w_k1_KO_s000
                                 + 0.0829446630199945 w_k1_KO_s001
                                 + 0.0785864757113931 w_k1_KO_s002
                                 + 0.0659912189614285 w_k1_KO_s003
                                 + 0.138679860197615 w_k1_KO_s004
                                 + 0.029747801061425 w_k1_KO_s005
                                 + 0.0577393160852834 w_k1_KO_s006
                                 + 0.0338272744445341 w_k1_KO_s007
                                 + 0.00831745616819556 w_k1_KO_s010
                                 + 0.0118187647974362 w_k1_KO_s011
                                 + 0.0253485925909299 w_k1_KO_s012
                                 + 0.0204955093041636 w_k1_KO_s013
                                 + 0.00979629286451454 w_k1_KO_s014
                                 + 0.0298674353922282 w_k1_KO_s015
                                 + 0.0104784668672419 w_k1_KO_s016
                                 + 0.0178682082924647 w_k1_KO_s017
                                 + 0.029264354368947 w_k1_KO_s100
                                 + 0.0392083548134466 w_k1_KO_s101
                                 + 0.0271781209262044 w_k1_KO_s102
                                 + 0.0168718290584168 w_k1_KO_s103
                                 + 0.016712413592546 w_k1_KO_s104
                                 + 0.0131813246840254 w_k1_KO_s105
                                 + 0.0181305479233712 w_k1_KO_s106
                                 + 0.013656686092939 w_k1_KO_s107
                                 + 0.0112937528540117 w_k1_KO_s110
                                 + 0.0170770290806459 w_k1_KO_s111
                                 + 0.0172158083739952 w_k1_KO_s112
                                 + 0.0132843835483795 w_k1_KO_s113
                                 + 0.0231937758933668 w_k1_KO_s114
                                 + 0.0142604174624184 w_k1_KO_s115
                                 + 0.0200677876721053 w_k1_KO_s116
                                 + 0.0184696726768637 w_k1_KO_s117
                                 + 0.0564302476289664 w_k1_MAT_s000
                                 + 0.0737701940488083 w_k1_MAT_s001
                                 + 0.070362548536294 w_k1_MAT_s002
                                 + 0.0745187179977279 w_k1_MAT_s003
                                 + 0.142268715500205 w_k1_MAT_s004
                                 + 0.031371466034173 w_k1_MAT_s005
                                 + 0.0611363819832071 w_k1_MAT_s006
                                 + 0.0363935883693626 w_k1_MAT_s007
                                 + 0.00741431418122324 w_k1_MAT_s010
                                 + 0.0139688338487648 w_k1_MAT_s011
                                 + 0.0254758693086738 w_k1_MAT_s012
                                 + 0.0240705314469468 w_k1_MAT_s013
                                 + 0.0105636147626127 w_k1_MAT_s014
                                 + 0.0309891380500211 w_k1_MAT_s015
                                 + 0.00902114713408117 w_k1_MAT_s016
                                 + 0.0148279167970507 w_k1_MAT_s017
                                 + 0.0277871847316446 w_k1_MAT_s100
                                 + 0.0388346481504731 w_k1_MAT_s101
                                 + 0.0231128955599409 w_k1_MAT_s102
                                 + 0.0197841557518294 w_k1_MAT_s103
                                 + 0.0186333502268048 w_k1_MAT_s104
                                 + 0.0125695401102764 w_k1_MAT_s105
                                 + 0.0216668277502986 w_k1_MAT_s106
                                 + 0.0138451933742249 w_k1_MAT_s107
                                 + 0.0121222030029635 w_k1_MAT_s110
                                 + 0.0195467853834502 w_k1_MAT_s111
                                 + 0.0184492234104673 w_k1_MAT_s112
                                 + 0.0105635462264846 w_k1_MAT_s113
                                 + 0.0263801226522081 w_k1_MAT_s114
                                 + 0.013718290243321 w_k1_MAT_s115
                                 + 0.0226944563863054 w_k1_MAT_s116
                                 + 0.0199742755353526 w_k1_MAT_s117
                                 + 0.0674695788917148 w_k1_MSFT_s000
                                 + 0.0877378662225425 w_k1_MSFT_s001
                                 + 0.0849020156218446 w_k1_MSFT_s002
                                 + 0.0600643422848245 w_k1_MSFT_s003
                                 + 0.15417436713622 w_k1_MSFT_s004
                                 + 0.0325282306232119 w_k1_MSFT_s005
                                 + 0.0569022641862655 w_k1_MSFT_s006
                                 + 0.0344123617438988 w_k1_MSFT_s007
                                 + 0.00796484869569497 w_k1_MSFT_s010
                                 + 0.0135250739274974 w_k1_MSFT_s011
                                 + 0.0247349958570576 w_k1_MSFT_s012
                                 + 0.0214687789972294 w_k1_MSFT_s013
                                 + 0.00984636489978016 w_k1_MSFT_s014
                                 + 0.0298061418803215 w_k1_MSFT_s015
                                 + 0.0103419664802223 w_k1_MSFT_s016
                                 + 0.019094194952361 w_k1_MSFT_s017
                                 + 0.0310256069073047 w_k1_MSFT_s100
                                 + 0.0393903797603548 w_k1_MSFT_s101
                                 + 0.0252246331959749 w_k1_MSFT_s102
                                 + 0.0157707073455532 w_k1_MSFT_s103
                                 + 0.0162557281459362 w_k1_MSFT_s104
                                 + 0.0115754524760348 w_k1_MSFT_s105
                                 + 0.02038014104534 w_k1_MSFT_s106
                                 + 0.0168163272686197 w_k1_MSFT_s107
                                 + 0.00969935768546741 w_k1_MSFT_s110
                                 + 0.0171484353413236 w_k1_MSFT_s111
                                 + 0.0162723301230388 w_k1_MSFT_s112
                                 + 0.0113090343828568 w_k1_MSFT_s113
                                 + 0.0238736078858387 w_k1_MSFT_s114
                                 + 0.0137137272706474 w_k1_MSFT_s115
                                 + 0.0203566006078126 w_k1_MSFT_s116
                                 + 0.0188958392875765 w_k1_MSFT_s117
                                 + 0.0729124996445099 w_k1_PG_s000
                                 + 0.090629023999063 w_k1_PG_s001
                                 + 0.0795373947258146 w_k1_PG_s002
                                 + 0.0651264142784914 w_k1_PG_s003
                                 + 0.135087090727349 w_k1_PG_s004
                                 + 0.0346478989009302 w_k1_PG_s005
                                 + 0.0568150763514005 w_k1_PG_s006
                                 + 0.036050589328375 w_k1_PG_s007
                                 + 0.00761237010489122 w_k1_PG_s010
                                 + 0.0118964217603667 w_k1_PG_s011
                                 + 0.0221141936373504 w_k1_PG_s012
                                 + 0.0219173773674737 w_k1_PG_s013
                                 + 0.0090512630070468 w_k1_PG_s014
                                 + 0.0267091042195311 w_k1_PG_s015
                                 + 0.0102678464510066 w_k1_PG_s016
                                 + 0.0179431469363817 w_k1_PG_s017
                                 + 0.0278782845738153 w_k1_PG_s100
                                 + 0.0366938912321798 w_k1_PG_s101
                                 + 0.0257070816728243 w_k1_PG_s102
                                 + 0.0171224634865376 w_k1_PG_s103
                                 + 0.0170959850081221 w_k1_PG_s104
                                 + 0.0130434734318816 w_k1_PG_s105
                                 + 0.0197901659819549 w_k1_PG_s106
                                 + 0.0143480067149045 w_k1_PG_s107
                                 + 0.00893126605003603 w_k1_PG_s110
                                 + 0.0188883461803161 w_k1_PG_s111
                                 + 0.0158786767225852 w_k1_PG_s112
                                 + 0.0123284054605083 w_k1_PG_s113
                                 + 0.0218088297102008 w_k1_PG_s114
                                 + 0.0139652727115563 w_k1_PG_s115
                                 + 0.0205272759119499 w_k1_PG_s116
                                 + 0.0165360164592305 w_k1_PG_s117 >= 1.18
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

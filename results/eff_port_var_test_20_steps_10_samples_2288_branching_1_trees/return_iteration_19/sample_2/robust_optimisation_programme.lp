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
 w_k1_ADM_s0_return_constraint:  1.13843 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.883501 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.1068 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.820891 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.920751 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.37601 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.01746 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.06067 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.815206 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.873382 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.40113 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.700218 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.20012 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.960215 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.35673 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.26781 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.46052 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.10967 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.989729 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.68543 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.05669 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.0114 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.85093 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.33237 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.39948 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.01705 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.32955 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.15053 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.813618 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.980081 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.944196 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.764833 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.965652 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.14632 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.802685 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.947363 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.653805 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.21264 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.24439 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.917687 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.15914 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.976204 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.811406 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.24562 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.861546 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.34451 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.646448 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.819899 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.41844 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.597188 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.25579 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.940125 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.18702 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.01169 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.42434 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.00485 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.67907 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.7677 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.873592 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.926498 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.973743 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.64439 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.51772 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.20026 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.68177 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.05469 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.778239 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.00932 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.31123 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.681645 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.38569 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.09664 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.01394 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.84166 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.603663 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.55471 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.28473 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.85117 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.21865 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.922704 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.895435 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.49638 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.884035 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.00334 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.62489 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.726644 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.71959 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.54278 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.23338 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.03353 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.58778 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.21043 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.12289 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.08224 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.700958 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.95921 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.75774 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.977516 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.912388 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.59041 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.23967 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.14112 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.57574 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.14365 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.577872 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.762838 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.12858 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.645106 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.22975 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.07232 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.722207 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.77649 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.683288 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.65275 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.16732 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.914874 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.13405 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.89578 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.893845 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.2919 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.923083 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.28615 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.789535 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.851915 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.48881 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.739577 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.12387 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.07347 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.36034 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.12371 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.18778 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.945151 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.80765 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.43838 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.863594 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.913473 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.828267 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.3568 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.40988 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.990923 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.23809 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.07165 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.710686 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.93394 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.902249 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.849184 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.07494 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.12732 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.835243 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.949722 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.757374 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.36438 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.19052 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.913336 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.86897 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.21135 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.988763 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.09714 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.19974 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.899856 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.820091 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.995916 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.5022 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.561378 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.50046 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.897026 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.1163 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.797624 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.14287 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.71538 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.683083 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.66895 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.00604 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.20852 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.24617 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.22362 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.823445 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.53391 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.92116 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.902767 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.840844 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.714843 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.99737 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.516158 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.35229 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.816761 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.03407 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.71924 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.763324 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.34496 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.13723 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.913194 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.02605 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.982214 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.969879 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.21388 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.943197 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.14381 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.826671 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.963879 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.22381 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.770556 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.17684 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.09339 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.16969 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.12895 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.989072 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.895425 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.863123 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.23218 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.895936 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.8984 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.856394 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.31988 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.0785 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.994759 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.38613 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.976788 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.755521 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.859252 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.07877 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.890477 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.976412 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.17844 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.799583 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.92433 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.741933 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.38444 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.21562 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.889303 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.01648 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.0215 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.979415 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.20405 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.986324 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.04622 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.72599 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.921883 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.30506 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.605217 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.39385 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.936993 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.35679 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.968745 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.04364 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.35712 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.772791 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.59587 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.845346 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.988829 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.997506 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.41536 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.01302 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.19267 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.64208 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.10755 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.672964 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.766259 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.51196 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.692454 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.24558 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.08501 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.95735 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.826389 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.741562 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.53099 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.28917 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.960053 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.945794 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.11838 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.957166 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.867734 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.13811 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.07736 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.704954 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.968318 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.70846 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.503471 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.41984 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.813524 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.32371 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.839604 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.33265 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.35654 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.725501 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.80776 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.951911 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.21324 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.00595 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.37577 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.968365 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.43002 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.82236 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.827487 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.732595 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.726666 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 2.01822 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.755861 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.89081 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.10983 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.37843 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.73751 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.847942 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.71321 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.15897 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.910195 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.13164 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.907512 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.938577 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.22391 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.919878 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.04792 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.84281 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.826943 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.33734 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.6977 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.06071 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.08629 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.25687 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.13698 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.08272 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.01036 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.901173 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.21266 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.920346 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.982624 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.856777 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.3777 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.10558 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.04111 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.22466 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.04885 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.743979 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.813955 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.978293 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.864001 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.07133 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.13516 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.854296 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.888536 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.814587 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.39473 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.13114 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.935477 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.04167 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.946447 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.987374 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.15553 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.972304 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.01994 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.84341 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.967093 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.25066 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.736021 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.16122 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.951309 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.19301 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.04723 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.13263 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.17796 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.923945 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.31333 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.919788 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.990766 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.90715 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.20216 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.07248 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.06091 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.22215 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.972974 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.827609 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.807663 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.07627 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.858658 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.07962 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.10891 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.885493 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.878872 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.820406 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.22767 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.189000557688762 z_k1_s0000
                                 - 0.366999412174905 z_k1_s0001
                                 - 0.271000400665195 z_k1_s0002
                                 - 0.155999846876985 z_k1_s0003
                                 - 0.0809999023669123 z_k1_s0004
                                 - 0.112999887069869 z_k1_s0005
                                 - 0.294998975293671 z_k1_s0006
                                 - 0.0989999330372906 z_k1_s0007
                                 - 0.344000396901735 z_k1_s0010
                                 - 0.207000712263325 z_k1_s0011
                                 - 0.466999876015163 z_k1_s0012
                                 - 0.100999977288074 z_k1_s0013
                                 - 0.323999282152139 z_k1_s0014
                                 - 0.133999996425749 z_k1_s0015
                                 - 0.10900003998809 z_k1_s0016
                                 - 0.230000365626114 z_k1_s0017
                                 - 0.146999776104528 z_k1_s0020
                                 - 0.324999772203483 z_k1_s0021
                                 - 0.227999677784536 z_k1_s0022
                                 - 0.395998264548608 z_k1_s0023
                                 - 0.309000561463054 z_k1_s0024
                                 - 0.330999783105678 z_k1_s0025
                                 - 0.321999766752385 z_k1_s0026
                                 - 0.396999903029825 z_k1_s0027
                                 - 0.283000398713832 z_k1_s0030
                                 - 0.167999131265753 z_k1_s0031
                                 - 0.333000121257228 z_k1_s0032
                                 - 0.197999599991214 z_k1_s0033
                                 - 0.104999893740699 z_k1_s0034
                                 - 0.0939999865411229 z_k1_s0035
                                 - 0.288999857259502 z_k1_s0036
                                 - 0.117999885121924 z_k1_s0037
                                 - 0.436998369815308 z_k1_s0040
                                 - 0.199999127029356 z_k1_s0041
                                 - 0.220998980365204 z_k1_s0042
                                 - 0.191999108068442 z_k1_s0043
                                 - 0.251997791032331 z_k1_s0044
                                 - 0.352997609478407 z_k1_s0045
                                 - 0.236999018287033 z_k1_s0046
                                 - 0.352997609478407 z_k1_s0047
                                 - 0.136000116041454 z_k1_s0050
                                 - 0.476999330769183 z_k1_s0051
                                 - 0.172000183552469 z_k1_s0052
                                 - 0.392001291120562 z_k1_s0053
                                 - 0.286001439838521 z_k1_s0054
                                 - 0.147000171419859 z_k1_s0055
                                 - 0.405999996873359 z_k1_s0056
                                 - 0.4860015368994 z_k1_s0057
                                 - 0.30699867175044 z_k1_s0060
                                 - 0.323999175663395 z_k1_s0061
                                 - 0.302000391300391 z_k1_s0062
                                 - 0.327999916817696 z_k1_s0063
                                 - 0.214999353353449 z_k1_s0064
                                 - 0.5799989816678 z_k1_s0065
                                 - 0.147999639784469 z_k1_s0066
                                 - 0.441999871772671 z_k1_s0067
                                 - 0.400999229049701 z_k1_s0070
                                 - 0.249000184288482 z_k1_s0071
                                 - 0.253000197487505 z_k1_s0072
                                 - 0.356000446312256 z_k1_s0073
                                 - 0.580000092856317 z_k1_s0074
                                 - 0.794000616903866 z_k1_s0075
                                 - 0.352000433113233 z_k1_s0076
                                 - 0.656999345386369 z_k1_s0077
                                 - 0.196000131606154 z_k1_s0100
                                 - 0.257001121684146 z_k1_s0101
                                 - 0.111000021576923 z_k1_s0102
                                 - 0.41100059777432 z_k1_s0103
                                 - 0.121999984154792 z_k1_s0104
                                 - 0.167000030664378 z_k1_s0105
                                 - 0.393000659010534 z_k1_s0106
                                 - 0.339000842719174 z_k1_s0107
                                 - 0.303000429778214 z_k1_s0110
                                 - 0.298000722986891 z_k1_s0111
                                 - 0.813001153167288 z_k1_s0112
                                 - 0.172000544267237 z_k1_s0113
                                 - 0.412000584384899 z_k1_s0114
                                 - 0.314000445380724 z_k1_s0115
                                 - 0.218000609514095 z_k1_s0116
                                 - 0.473000670907906 z_k1_s0117
                                 - 0.0910001012869914 z_k1_s0120
                                 - 0.244999753311009 z_k1_s0121
                                 - 0.247999330012678 z_k1_s0122
                                 - 0.261000309053903 z_k1_s0123
                                 - 0.138000109081361 z_k1_s0124
                                 - 0.230000181802268 z_k1_s0125
                                 - 0.170999504000669 z_k1_s0126
                                 - 0.304000431112608 z_k1_s0127
                                 - 0.180000555066312 z_k1_s0130
                                 - 0.105000014855165 z_k1_s0131
                                 - 0.213999571289301 z_k1_s0132
                                 - 0.0980000226914926 z_k1_s0133
                                 - 0.25300066248758 z_k1_s0134
                                 - 0.164000005549009 z_k1_s0135
                                 - 0.143000029057992 z_k1_s0136
                                 - 0.167000191333553 z_k1_s0137
                                 - 1.06600204447683 z_k1_s0140
                                 - 0.666000877994277 z_k1_s0141
                                 - 0.405999493781111 z_k1_s0142
                                 - 1.15699733315551 z_k1_s0143
                                 - 1.44900106428551 z_k1_s0144
                                 - 0.597999803292931 z_k1_s0145
                                 - 0.330000023348838 z_k1_s0146
                                 - 0.587999461131112 z_k1_s0147
                                 - 0.392000275592074 z_k1_s0150
                                 - 0.185999693033799 z_k1_s0151
                                 - 0.505998096855249 z_k1_s0152
                                 - 0.118999878840837 z_k1_s0153
                                 - 0.165999721107593 z_k1_s0154
                                 - 0.19099968601535 z_k1_s0155
                                 - 0.212999563374349 z_k1_s0156
                                 - 0.520998534599039 z_k1_s0157
                                 - 0.197000116432651 z_k1_s0160
                                 - 0.166000169362082 z_k1_s0161
                                 - 0.219000984451133 z_k1_s0162
                                 - 0.284999817503833 z_k1_s0163
                                 - 0.511000202051117 z_k1_s0164
                                 - 0.370999630118928 z_k1_s0165
                                 - 0.220000300178718 z_k1_s0166
                                 - 0.126000095738665 z_k1_s0167
                                 - 0.504001190199509 z_k1_s0170
                                 - 0.149000507201459 z_k1_s0171
                                 - 0.168000553000253 z_k1_s0172
                                 - 0.327000590831341 z_k1_s0173
                                 - 0.202000597945363 z_k1_s0174
                                 - 0.279000031001628 z_k1_s0175
                                 - 0.178000552431131 z_k1_s0176
                                 - 0.537002454084778 z_k1_s0177
                                 - 0.212000026027671 z_k1_s1000
                                 - 0.314999955253368 z_k1_s1001
                                 - 0.354000690799119 z_k1_s1002
                                 - 0.346999905793387 z_k1_s1003
                                 - 0.341000091917157 z_k1_s1004
                                 - 0.352000163340299 z_k1_s1005
                                 - 0.720999858279927 z_k1_s1006
                                 - 0.894999058791211 z_k1_s1007
                                 - 0.364000615491366 z_k1_s1010
                                 - 0.344000425244469 z_k1_s1011
                                 - 0.539999442661895 z_k1_s1012
                                 - 0.193000270031368 z_k1_s1013
                                 - 0.652000508044522 z_k1_s1014
                                 - 0.328999570808753 z_k1_s1015
                                 - 0.184000042070156 z_k1_s1016
                                 - 0.241000157223489 z_k1_s1017
                                 - 0.498999186487428 z_k1_s1020
                                 - 0.262000185110544 z_k1_s1021
                                 - 0.878001063214804 z_k1_s1022
                                 - 0.331000015222823 z_k1_s1023
                                 - 0.434999886881934 z_k1_s1024
                                 - 0.247999948171772 z_k1_s1025
                                 - 0.276999940630567 z_k1_s1026
                                 - 0.74200117455288 z_k1_s1027
                                 - 0.261000031988322 z_k1_s1030
                                 - 0.530001641554668 z_k1_s1031
                                 - 0.595001842877411 z_k1_s1032
                                 - 0.450001393772832 z_k1_s1033
                                 - 0.219999905000131 z_k1_s1034
                                 - 0.577001010726577 z_k1_s1035
                                 - 0.368999978293842 z_k1_s1036
                                 - 0.879998843600603 z_k1_s1037
                                 - 0.457999201821434 z_k1_s1040
                                 - 0.475998446434639 z_k1_s1041
                                 - 0.225999947875141 z_k1_s1042
                                 - 0.600998358914264 z_k1_s1043
                                 - 0.170000013612264 z_k1_s1044
                                 - 0.772999246550037 z_k1_s1045
                                 - 0.230999811734351 z_k1_s1046
                                 - 0.38099838031015 z_k1_s1047
                                 - 1.1339964121181 z_k1_s1050
                                 - 1.9109991311618 z_k1_s1051
                                 - 1.01499518609341 z_k1_s1052
                                 - 1.11999744258574 z_k1_s1053
                                 - 2.50299977638563 z_k1_s1054
                                 - 0.722000348381099 z_k1_s1055
                                 - 0.832000097206484 z_k1_s1056
                                 - 0.748000289012553 z_k1_s1057
                                 - 0.522998826295943 z_k1_s1060
                                 - 0.683996039824345 z_k1_s1061
                                 - 1.40599695651774 z_k1_s1062
                                 - 0.387999517372896 z_k1_s1063
                                 - 1.34499878510359 z_k1_s1064
                                 - 0.698996154170972 z_k1_s1065
                                 - 0.352999250564099 z_k1_s1066
                                 - 0.335999120971254 z_k1_s1067
                                 - 1.2589992484941 z_k1_s1070
                                 - 0.516999573424188 z_k1_s1071
                                 - 1.68299707745103 z_k1_s1072
                                 - 0.688998643552066 z_k1_s1073
                                 - 2.02599806625917 z_k1_s1074
                                 - 0.852998787128413 z_k1_s1075
                                 - 1.17000261319495 z_k1_s1076
                                 - 0.488999421799131 z_k1_s1077
                                 - 0.116999931914635 z_k1_s1100
                                 - 0.151000118569982 z_k1_s1101
                                 - 0.175000220726724 z_k1_s1102
                                 - 0.109999923085566 z_k1_s1103
                                 - 0.25299992373671 z_k1_s1104
                                 - 0.229999930669736 z_k1_s1105
                                 - 0.0879999133084756 z_k1_s1106
                                 - 0.133999522042766 z_k1_s1107
                                 - 0.162999623689611 z_k1_s1110
                                 - 0.277999350967569 z_k1_s1111
                                 - 0.0799998666193668 z_k1_s1112
                                 - 0.0999998038242511 z_k1_s1113
                                 - 0.0989998010740155 z_k1_s1114
                                 - 0.217999185953428 z_k1_s1115
                                 - 0.132999541182541 z_k1_s1116
                                 - 0.106999823075901 z_k1_s1117
                                 - 0.158999698653091 z_k1_s1120
                                 - 0.615998159658476 z_k1_s1121
                                 - 0.195999684599935 z_k1_s1122
                                 - 0.363999120343269 z_k1_s1123
                                 - 0.35499915465344 z_k1_s1124
                                 - 0.164999751979505 z_k1_s1125
                                 - 0.262000271190494 z_k1_s1126
                                 - 0.168999787530448 z_k1_s1127
                                 - 0.882000557230965 z_k1_s1130
                                 - 0.442999529080489 z_k1_s1131
                                 - 0.845998883876279 z_k1_s1132
                                 - 0.419999807991401 z_k1_s1133
                                 - 0.462999616222839 z_k1_s1134
                                 - 0.775997315129342 z_k1_s1135
                                 - 0.605997838168083 z_k1_s1136
                                 - 0.618999284934195 z_k1_s1137
                                 - 0.756996584836445 z_k1_s1140
                                 - 0.893996894660944 z_k1_s1141
                                 - 0.371999238604132 z_k1_s1142
                                 - 0.560998364528203 z_k1_s1143
                                 - 0.995999276569644 z_k1_s1144
                                 - 0.288999380756493 z_k1_s1145
                                 - 0.46599900659558 z_k1_s1146
                                 - 0.57699671557687 z_k1_s1147
                                 - 0.121999915668934 z_k1_s1150
                                 - 0.169999386867 z_k1_s1151
                                 - 0.119999850294095 z_k1_s1152
                                 - 0.257000009474551 z_k1_s1153
                                 - 0.142999491505764 z_k1_s1154
                                 - 0.0999999369450226 z_k1_s1155
                                 - 0.192000352790011 z_k1_s1156
                                 - 0.129999313369081 z_k1_s1157
                                 - 0.134999895203661 z_k1_s1160
                                 - 0.0939999270306976 z_k1_s1161
                                 - 0.143999429817411 z_k1_s1162
                                 - 0.0739999425560811 z_k1_s1163
                                 - 0.133000125956114 z_k1_s1164
                                 - 0.223999367715877 z_k1_s1165
                                 - 0.204999840864819 z_k1_s1166
                                 - 0.136999664451209 z_k1_s1167
                                 - 0.365000179841677 z_k1_s1170
                                 - 0.291999462674096 z_k1_s1171
                                 - 0.0939998760155792 z_k1_s1172
                                 - 0.207999080686251 z_k1_s1173
                                 - 0.0899998740447116 z_k1_s1174
                                 - 0.124999891289804 z_k1_s1175
                                 - 0.428998848977068 z_k1_s1176
                                 - 0.0999998789718808 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.997382 w_k1_ADM_s000
                                 + 0.879568 w_k1_AXP_s000
                                 + 1.10874 w_k1_DIS_s000
                                 + 0.964932 w_k1_FDX_s000
                                 + 1.17949 w_k1_IBM_s000
                                 + 0.879269 w_k1_INTC_s000
                                 + 1.19808 w_k1_KO_s000
                                 + 0.972255 w_k1_MAT_s000
                                 + 1.06237 w_k1_MSFT_s000 + 1.0836 w_k1_PG_s000
                                 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.926788 w_k1_ADM_s000
                                 + 1.00904 w_k1_AXP_s000
                                 + 0.959157 w_k1_DIS_s000
                                 + 0.938382 w_k1_FDX_s000
                                 + 0.774431 w_k1_IBM_s000
                                 + 0.903409 w_k1_INTC_s000
                                 + 0.882361 w_k1_KO_s000
                                 + 0.700985 w_k1_MAT_s000
                                 + 0.85933 w_k1_MSFT_s000
                                 + 0.93693 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.91063 w_k1_ADM_s000
                                 + 1.17579 w_k1_AXP_s000
                                 + 0.991196 w_k1_DIS_s000
                                 + 1.04297 w_k1_FDX_s000
                                 + 1.09986 w_k1_IBM_s000
                                 + 1.0857 w_k1_INTC_s000
                                 + 0.959477 w_k1_KO_s000
                                 + 0.984295 w_k1_MAT_s000
                                 + 0.924231 w_k1_MSFT_s000
                                 + 1.0374 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.08773 w_k1_ADM_s000
                                 + 1.31098 w_k1_AXP_s000
                                 + 1.54799 w_k1_DIS_s000
                                 + 1.30434 w_k1_FDX_s000
                                 + 0.983873 w_k1_IBM_s000
                                 + 1.33763 w_k1_INTC_s000 + 1.2442 w_k1_KO_s000
                                 + 0.722526 w_k1_MAT_s000
                                 + 1.23139 w_k1_MSFT_s000
                                 + 1.14099 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.0008 w_k1_ADM_s000
                                 + 1.36709 w_k1_AXP_s000
                                 + 1.20243 w_k1_DIS_s000
                                 + 1.12786 w_k1_FDX_s000
                                 + 1.00538 w_k1_IBM_s000
                                 + 1.19612 w_k1_INTC_s000
                                 + 1.15846 w_k1_KO_s000 + 1.30369 w_k1_MAT_s000
                                 + 1.10571 w_k1_MSFT_s000
                                 + 1.04946 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.07582 w_k1_ADM_s000
                                 + 1.32088 w_k1_AXP_s000
                                 + 1.51944 w_k1_DIS_s000
                                 + 1.21739 w_k1_FDX_s000
                                 + 1.56739 w_k1_IBM_s000
                                 + 1.12882 w_k1_INTC_s000
                                 + 1.45137 w_k1_KO_s000 + 1.6159 w_k1_MAT_s000
                                 + 1.18505 w_k1_MSFT_s000 + 1.197 w_k1_PG_s000
                                 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.789583 w_k1_ADM_s000
                                 + 0.857018 w_k1_AXP_s000
                                 + 0.721098 w_k1_DIS_s000
                                 + 0.777495 w_k1_FDX_s000
                                 + 1.00473 w_k1_IBM_s000
                                 + 0.966322 w_k1_INTC_s000
                                 + 0.787667 w_k1_KO_s000
                                 + 0.65113 w_k1_MAT_s000
                                 + 0.836826 w_k1_MSFT_s000
                                 + 0.863295 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.089 w_k1_ADM_s000
                                 + 0.964038 w_k1_AXP_s000
                                 + 1.05228 w_k1_DIS_s000
                                 + 0.955779 w_k1_FDX_s000
                                 + 1.46972 w_k1_IBM_s000
                                 + 1.15504 w_k1_INTC_s000 + 1.2308 w_k1_KO_s000
                                 + 1.20572 w_k1_MAT_s000
                                 + 0.899372 w_k1_MSFT_s000
                                 + 1.03295 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.46912 w_k1_ADM_s001
                                 + 1.1733 w_k1_AXP_s001 + 1.53791 w_k1_DIS_s001
                                 + 1.33692 w_k1_FDX_s001
                                 + 0.991303 w_k1_IBM_s001
                                 + 1.33445 w_k1_INTC_s001
                                 + 1.06765 w_k1_KO_s001
                                 + 0.721017 w_k1_MAT_s001
                                 + 1.2278 w_k1_MSFT_s001 + 1.25944 w_k1_PG_s001
                                 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.960549 w_k1_ADM_s001
                                 + 0.712467 w_k1_AXP_s001
                                 + 0.872216 w_k1_DIS_s001
                                 + 0.837782 w_k1_FDX_s001
                                 + 0.733809 w_k1_IBM_s001
                                 + 0.752759 w_k1_INTC_s001
                                 + 0.76502 w_k1_KO_s001
                                 + 0.733621 w_k1_MAT_s001
                                 + 0.796025 w_k1_MSFT_s001
                                 + 0.979526 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.40725 w_k1_ADM_s001
                                 + 0.875032 w_k1_AXP_s001
                                 + 1.00369 w_k1_DIS_s001
                                 + 0.994562 w_k1_FDX_s001
                                 + 0.83396 w_k1_IBM_s001
                                 + 0.954842 w_k1_INTC_s001
                                 + 0.810793 w_k1_KO_s001
                                 + 0.884024 w_k1_MAT_s001
                                 + 0.952868 w_k1_MSFT_s001
                                 + 0.965227 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.809972 w_k1_ADM_s001
                                 + 0.715741 w_k1_AXP_s001
                                 + 0.633386 w_k1_DIS_s001
                                 + 0.732595 w_k1_FDX_s001
                                 + 0.727078 w_k1_IBM_s001
                                 + 0.828643 w_k1_INTC_s001
                                 + 0.657322 w_k1_KO_s001
                                 + 0.593593 w_k1_MAT_s001
                                 + 0.726507 w_k1_MSFT_s001
                                 + 0.762304 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.16408 w_k1_ADM_s001
                                 + 1.16265 w_k1_AXP_s001
                                 + 1.24713 w_k1_DIS_s001 + 1.038 w_k1_FDX_s001
                                 + 1.48394 w_k1_IBM_s001
                                 + 1.09813 w_k1_INTC_s001
                                 + 1.17448 w_k1_KO_s001 + 1.21143 w_k1_MAT_s001
                                 + 1.07367 w_k1_MSFT_s001
                                 + 1.07669 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.20133 w_k1_ADM_s001
                                 + 1.02794 w_k1_AXP_s001
                                 + 0.924021 w_k1_DIS_s001
                                 + 0.820555 w_k1_FDX_s001
                                 + 1.37068 w_k1_IBM_s001
                                 + 1.16279 w_k1_INTC_s001
                                 + 0.950272 w_k1_KO_s001
                                 + 1.03201 w_k1_MAT_s001
                                 + 0.946587 w_k1_MSFT_s001
                                 + 0.946712 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.968024 w_k1_ADM_s001
                                 + 0.764703 w_k1_AXP_s001
                                 + 0.648413 w_k1_DIS_s001
                                 + 0.744151 w_k1_FDX_s001
                                 + 0.74498 w_k1_IBM_s001
                                 + 0.98607 w_k1_INTC_s001
                                 + 0.667023 w_k1_KO_s001
                                 + 0.63089 w_k1_MAT_s001
                                 + 0.815507 w_k1_MSFT_s001
                                 + 0.825004 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.09554 w_k1_ADM_s001
                                 + 0.857029 w_k1_AXP_s001
                                 + 0.736798 w_k1_DIS_s001
                                 + 0.810465 w_k1_FDX_s001
                                 + 1.03489 w_k1_IBM_s001
                                 + 0.964348 w_k1_INTC_s001
                                 + 0.862395 w_k1_KO_s001
                                 + 0.965371 w_k1_MAT_s001
                                 + 0.857103 w_k1_MSFT_s001
                                 + 1.00027 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.805589 w_k1_ADM_s002
                                 + 0.845355 w_k1_AXP_s002
                                 + 0.596902 w_k1_DIS_s002
                                 + 0.64254 w_k1_FDX_s002
                                 + 1.02364 w_k1_IBM_s002
                                 + 0.81622 w_k1_INTC_s002
                                 + 0.797854 w_k1_KO_s002
                                 + 1.08753 w_k1_MAT_s002
                                 + 0.646736 w_k1_MSFT_s002
                                 + 0.80522 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.92746 w_k1_ADM_s002
                                 + 0.812531 w_k1_AXP_s002
                                 + 0.753758 w_k1_DIS_s002
                                 + 0.77252 w_k1_FDX_s002
                                 + 0.804624 w_k1_IBM_s002
                                 + 0.801343 w_k1_INTC_s002
                                 + 0.816057 w_k1_KO_s002
                                 + 0.81364 w_k1_MAT_s002
                                 + 0.705315 w_k1_MSFT_s002
                                 + 0.778367 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.12344 w_k1_ADM_s002
                                 + 1.06356 w_k1_AXP_s002
                                 + 1.36769 w_k1_DIS_s002
                                 + 1.06512 w_k1_FDX_s002 + 1.1996 w_k1_IBM_s002
                                 + 1.03648 w_k1_INTC_s002
                                 + 1.36166 w_k1_KO_s002 + 1.17902 w_k1_MAT_s002
                                 + 1.03697 w_k1_MSFT_s002
                                 + 1.08928 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.03528 w_k1_ADM_s002
                                 + 1.07629 w_k1_AXP_s002
                                 + 0.908814 w_k1_DIS_s002
                                 + 0.869672 w_k1_FDX_s002
                                 + 1.3604 w_k1_IBM_s002
                                 + 0.745672 w_k1_INTC_s002
                                 + 1.11581 w_k1_KO_s002 + 1.67382 w_k1_MAT_s002
                                 + 0.763589 w_k1_MSFT_s002
                                 + 1.07617 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.06813 w_k1_ADM_s002
                                 + 1.2584 w_k1_AXP_s002 + 1.22167 w_k1_DIS_s002
                                 + 1.2107 w_k1_FDX_s002
                                 + 0.887443 w_k1_IBM_s002
                                 + 1.23152 w_k1_INTC_s002
                                 + 0.966025 w_k1_KO_s002
                                 + 1.08697 w_k1_MAT_s002
                                 + 1.16555 w_k1_MSFT_s002
                                 + 1.02113 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.927229 w_k1_ADM_s002
                                 + 0.892848 w_k1_AXP_s002
                                 + 1.08773 w_k1_DIS_s002
                                 + 1.00203 w_k1_FDX_s002
                                 + 0.959824 w_k1_IBM_s002
                                 + 1.15925 w_k1_INTC_s002
                                 + 1.14259 w_k1_KO_s002
                                 + 0.665876 w_k1_MAT_s002
                                 + 0.943477 w_k1_MSFT_s002
                                 + 0.995273 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.09147 w_k1_ADM_s002
                                 + 0.958979 w_k1_AXP_s002
                                 + 0.922361 w_k1_DIS_s002
                                 + 0.983483 w_k1_FDX_s002
                                 + 0.801288 w_k1_IBM_s002
                                 + 0.928092 w_k1_INTC_s002
                                 + 0.845756 w_k1_KO_s002
                                 + 0.722723 w_k1_MAT_s002
                                 + 0.891729 w_k1_MSFT_s002
                                 + 0.861682 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.29521 w_k1_ADM_s002
                                 + 1.5302 w_k1_AXP_s002 + 1.25495 w_k1_DIS_s002
                                 + 1.2559 w_k1_FDX_s002 + 1.32404 w_k1_IBM_s002
                                 + 1.3578 w_k1_INTC_s002 + 1.34394 w_k1_KO_s002
                                 + 1.61621 w_k1_MAT_s002 + 1.103 w_k1_MSFT_s002
                                 + 1.21952 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.973888 w_k1_ADM_s003
                                 + 0.921065 w_k1_AXP_s003
                                 + 0.927317 w_k1_DIS_s003
                                 + 0.865095 w_k1_FDX_s003
                                 + 1.34807 w_k1_IBM_s003
                                 + 0.713523 w_k1_INTC_s003
                                 + 0.949382 w_k1_KO_s003
                                 + 1.45586 w_k1_MAT_s003
                                 + 0.955506 w_k1_MSFT_s003
                                 + 0.921097 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.993348 w_k1_ADM_s003
                                 + 1.08498 w_k1_AXP_s003
                                 + 0.777421 w_k1_DIS_s003
                                 + 0.927352 w_k1_FDX_s003
                                 + 0.987305 w_k1_IBM_s003
                                 + 0.716963 w_k1_INTC_s003
                                 + 0.875679 w_k1_KO_s003
                                 + 1.23301 w_k1_MAT_s003
                                 + 0.866452 w_k1_MSFT_s003
                                 + 0.878789 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.797642 w_k1_ADM_s003
                                 + 1.10495 w_k1_AXP_s003
                                 + 0.991449 w_k1_DIS_s003
                                 + 0.96606 w_k1_FDX_s003
                                 + 1.15566 w_k1_IBM_s003
                                 + 0.975034 w_k1_INTC_s003
                                 + 1.01658 w_k1_KO_s003 + 1.03588 w_k1_MAT_s003
                                 + 0.981324 w_k1_MSFT_s003
                                 + 0.901451 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.954296 w_k1_ADM_s003
                                 + 0.682613 w_k1_AXP_s003
                                 + 0.736432 w_k1_DIS_s003
                                 + 0.819116 w_k1_FDX_s003
                                 + 0.802005 w_k1_IBM_s003
                                 + 0.846052 w_k1_INTC_s003
                                 + 0.825348 w_k1_KO_s003
                                 + 0.639174 w_k1_MAT_s003
                                 + 0.868914 w_k1_MSFT_s003
                                 + 0.818398 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.26937 w_k1_ADM_s003
                                 + 1.55758 w_k1_AXP_s003
                                 + 1.16864 w_k1_DIS_s003
                                 + 1.23687 w_k1_FDX_s003
                                 + 1.07293 w_k1_IBM_s003
                                 + 1.16513 w_k1_INTC_s003
                                 + 1.04264 w_k1_KO_s003 + 1.02562 w_k1_MAT_s003
                                 + 1.10712 w_k1_MSFT_s003
                                 + 1.08597 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.550765 w_k1_ADM_s003
                                 + 0.608992 w_k1_AXP_s003
                                 + 0.719116 w_k1_DIS_s003
                                 + 0.726401 w_k1_FDX_s003
                                 + 0.83902 w_k1_IBM_s003
                                 + 0.832764 w_k1_INTC_s003
                                 + 0.742937 w_k1_KO_s003
                                 + 0.709092 w_k1_MAT_s003
                                 + 0.888688 w_k1_MSFT_s003
                                 + 0.754934 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.29167 w_k1_ADM_s003
                                 + 1.50903 w_k1_AXP_s003
                                 + 1.46976 w_k1_DIS_s003
                                 + 1.29924 w_k1_FDX_s003
                                 + 1.33072 w_k1_IBM_s003
                                 + 1.25827 w_k1_INTC_s003
                                 + 1.27968 w_k1_KO_s003 + 1.08385 w_k1_MAT_s003
                                 + 1.30613 w_k1_MSFT_s003
                                 + 1.14344 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.21622 w_k1_ADM_s003
                                 + 1.1102 w_k1_AXP_s003 + 1.02928 w_k1_DIS_s003
                                 + 1.1002 w_k1_FDX_s003
                                 + 0.854617 w_k1_IBM_s003
                                 + 0.830751 w_k1_INTC_s003
                                 + 0.821758 w_k1_KO_s003
                                 + 0.797136 w_k1_MAT_s003
                                 + 0.966079 w_k1_MSFT_s003
                                 + 0.948934 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.11352 w_k1_ADM_s004
                                 + 0.931184 w_k1_AXP_s004
                                 + 1.02193 w_k1_DIS_s004
                                 + 0.991263 w_k1_FDX_s004
                                 + 0.675676 w_k1_IBM_s004
                                 + 0.99284 w_k1_INTC_s004
                                 + 0.863966 w_k1_KO_s004
                                 + 0.533463 w_k1_MAT_s004
                                 + 0.944031 w_k1_MSFT_s004
                                 + 0.994696 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.12816 w_k1_ADM_s004
                                 + 1.33452 w_k1_AXP_s004
                                 + 0.832242 w_k1_DIS_s004
                                 + 1.08698 w_k1_FDX_s004
                                 + 0.746609 w_k1_IBM_s004
                                 + 1.12473 w_k1_INTC_s004
                                 + 0.874901 w_k1_KO_s004
                                 + 0.752837 w_k1_MAT_s004
                                 + 0.929307 w_k1_MSFT_s004
                                 + 1.05087 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.82498 w_k1_ADM_s004
                                 + 1.09992 w_k1_AXP_s004
                                 + 0.995293 w_k1_DIS_s004
                                 + 0.827692 w_k1_FDX_s004
                                 + 1.36011 w_k1_IBM_s004
                                 + 0.885802 w_k1_INTC_s004
                                 + 1.2228 w_k1_KO_s004 + 1.30477 w_k1_MAT_s004
                                 + 0.791186 w_k1_MSFT_s004
                                 + 1.00457 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.859542 w_k1_ADM_s004
                                 + 0.7862 w_k1_AXP_s004
                                 + 0.996501 w_k1_DIS_s004
                                 + 0.803829 w_k1_FDX_s004
                                 + 1.01419 w_k1_IBM_s004
                                 + 1.02017 w_k1_INTC_s004
                                 + 1.19152 w_k1_KO_s004
                                 + 0.927165 w_k1_MAT_s004
                                 + 0.97758 w_k1_MSFT_s004
                                 + 0.993531 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.15416 w_k1_ADM_s004
                                 + 1.56288 w_k1_AXP_s004 + 1.7736 w_k1_DIS_s004
                                 + 1.26464 w_k1_FDX_s004
                                 + 1.61292 w_k1_IBM_s004
                                 + 1.60535 w_k1_INTC_s004
                                 + 1.83915 w_k1_KO_s004 + 1.18597 w_k1_MAT_s004
                                 + 1.37238 w_k1_MSFT_s004
                                 + 1.10972 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.641054 w_k1_ADM_s004
                                 + 0.60394 w_k1_AXP_s004
                                 + 0.658094 w_k1_DIS_s004
                                 + 0.767847 w_k1_FDX_s004
                                 + 0.763417 w_k1_IBM_s004
                                 + 1.15678 w_k1_INTC_s004
                                 + 0.810146 w_k1_KO_s004
                                 + 0.525547 w_k1_MAT_s004
                                 + 0.814009 w_k1_MSFT_s004
                                 + 0.801202 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.25973 w_k1_ADM_s004
                                 + 1.4956 w_k1_AXP_s004 + 1.4258 w_k1_DIS_s004
                                 + 1.43046 w_k1_FDX_s004
                                 + 1.05109 w_k1_IBM_s004
                                 + 1.54411 w_k1_INTC_s004
                                 + 1.21727 w_k1_KO_s004 + 1.00799 w_k1_MAT_s004
                                 + 1.28607 w_k1_MSFT_s004
                                 + 1.17863 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.22365 w_k1_ADM_s004
                                 + 1.11588 w_k1_AXP_s004 + 1.2843 w_k1_DIS_s004
                                 + 1.0876 w_k1_FDX_s004 + 1.07044 w_k1_IBM_s004
                                 + 1.1134 w_k1_INTC_s004 + 1.49004 w_k1_KO_s004
                                 + 1.08007 w_k1_MAT_s004
                                 + 1.17955 w_k1_MSFT_s004
                                 + 1.20476 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.985009 w_k1_ADM_s005
                                 + 1.25725 w_k1_AXP_s005
                                 + 0.760172 w_k1_DIS_s005
                                 + 1.03587 w_k1_FDX_s005
                                 + 1.03722 w_k1_IBM_s005
                                 + 1.21211 w_k1_INTC_s005
                                 + 0.829351 w_k1_KO_s005
                                 + 1.26863 w_k1_MAT_s005
                                 + 0.946452 w_k1_MSFT_s005
                                 + 0.968368 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.29701 w_k1_ADM_s005
                                 + 1.42805 w_k1_AXP_s005
                                 + 1.40901 w_k1_DIS_s005 + 1.3345 w_k1_FDX_s005
                                 + 1.51482 w_k1_IBM_s005
                                 + 1.39908 w_k1_INTC_s005
                                 + 1.19333 w_k1_KO_s005 + 1.27921 w_k1_MAT_s005
                                 + 1.17793 w_k1_MSFT_s005
                                 + 1.29948 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.920989 w_k1_ADM_s005
                                 + 0.672106 w_k1_AXP_s005
                                 + 0.693451 w_k1_DIS_s005
                                 + 0.941452 w_k1_FDX_s005
                                 + 0.620485 w_k1_IBM_s005
                                 + 1.02204 w_k1_INTC_s005
                                 + 0.672224 w_k1_KO_s005
                                 + 0.544512 w_k1_MAT_s005
                                 + 1.04077 w_k1_MSFT_s005
                                 + 0.92823 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.198 w_k1_ADM_s005
                                 + 1.07959 w_k1_AXP_s005
                                 + 1.37209 w_k1_DIS_s005
                                 + 1.04747 w_k1_FDX_s005
                                 + 1.29279 w_k1_IBM_s005
                                 + 1.11828 w_k1_INTC_s005
                                 + 1.20856 w_k1_KO_s005 + 1.01324 w_k1_MAT_s005
                                 + 1.11025 w_k1_MSFT_s005
                                 + 1.06635 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.9059 w_k1_ADM_s005
                                 + 0.678063 w_k1_AXP_s005
                                 + 0.565137 w_k1_DIS_s005
                                 + 0.674004 w_k1_FDX_s005
                                 + 0.973525 w_k1_IBM_s005
                                 + 0.643053 w_k1_INTC_s005
                                 + 0.767163 w_k1_KO_s005
                                 + 0.840652 w_k1_MAT_s005
                                 + 0.706147 w_k1_MSFT_s005
                                 + 0.815656 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.97492 w_k1_ADM_s005
                                 + 1.05541 w_k1_AXP_s005
                                 + 0.645914 w_k1_DIS_s005
                                 + 0.915604 w_k1_FDX_s005
                                 + 0.927902 w_k1_IBM_s005
                                 + 1.00778 w_k1_INTC_s005
                                 + 0.75446 w_k1_KO_s005
                                 + 0.931364 w_k1_MAT_s005
                                 + 0.908376 w_k1_MSFT_s005
                                 + 0.962646 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.11278 w_k1_ADM_s005
                                 + 0.874438 w_k1_AXP_s005
                                 + 1.1144 w_k1_DIS_s005 + 1.12504 w_k1_FDX_s005
                                 + 0.858285 w_k1_IBM_s005
                                 + 1.01564 w_k1_INTC_s005
                                 + 0.986031 w_k1_KO_s005
                                 + 0.796199 w_k1_MAT_s005
                                 + 0.988668 w_k1_MSFT_s005
                                 + 0.98725 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.07238 w_k1_ADM_s005
                                 + 1.03822 w_k1_AXP_s005
                                 + 0.976495 w_k1_DIS_s005
                                 + 0.985264 w_k1_FDX_s005
                                 + 1.08532 w_k1_IBM_s005
                                 + 1.14625 w_k1_INTC_s005
                                 + 1.07935 w_k1_KO_s005 + 1.22024 w_k1_MAT_s005
                                 + 1.13809 w_k1_MSFT_s005
                                 + 0.999587 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.912204 w_k1_ADM_s006
                                 + 1.07775 w_k1_AXP_s006 + 1.1974 w_k1_DIS_s006
                                 + 1.11997 w_k1_FDX_s006
                                 + 0.927168 w_k1_IBM_s006
                                 + 1.15848 w_k1_INTC_s006
                                 + 0.887711 w_k1_KO_s006
                                 + 0.945648 w_k1_MAT_s006
                                 + 1.1712 w_k1_MSFT_s006
                                 + 0.930188 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.982656 w_k1_ADM_s006
                                 + 1.0825 w_k1_AXP_s006
                                 + 0.852773 w_k1_DIS_s006
                                 + 1.00579 w_k1_FDX_s006
                                 + 0.914877 w_k1_IBM_s006
                                 + 0.973148 w_k1_INTC_s006
                                 + 0.76656 w_k1_KO_s006 + 1.12959 w_k1_MAT_s006
                                 + 0.822902 w_k1_MSFT_s006
                                 + 0.896279 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.799585 w_k1_ADM_s006
                                 + 0.911287 w_k1_AXP_s006
                                 + 1.02212 w_k1_DIS_s006
                                 + 0.981702 w_k1_FDX_s006
                                 + 1.16333 w_k1_IBM_s006
                                 + 1.05727 w_k1_INTC_s006
                                 + 1.00404 w_k1_KO_s006
                                 + 0.961276 w_k1_MAT_s006
                                 + 1.04507 w_k1_MSFT_s006
                                 + 0.988551 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.960024 w_k1_ADM_s006
                                 + 1.12549 w_k1_AXP_s006
                                 + 0.911844 w_k1_DIS_s006
                                 + 0.913522 w_k1_FDX_s006
                                 + 1.55008 w_k1_IBM_s006
                                 + 0.899538 w_k1_INTC_s006 + 1.085 w_k1_KO_s006
                                 + 1.88613 w_k1_MAT_s006
                                 + 0.856534 w_k1_MSFT_s006
                                 + 1.06174 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.668757 w_k1_ADM_s006
                                 + 0.647392 w_k1_AXP_s006
                                 + 0.997727 w_k1_DIS_s006
                                 + 0.841004 w_k1_FDX_s006
                                 + 1.04603 w_k1_IBM_s006
                                 + 1.16765 w_k1_INTC_s006
                                 + 0.974929 w_k1_KO_s006
                                 + 0.634658 w_k1_MAT_s006
                                 + 0.95006 w_k1_MSFT_s006
                                 + 0.957635 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.803538 w_k1_ADM_s006
                                 + 0.761564 w_k1_AXP_s006
                                 + 0.697956 w_k1_DIS_s006
                                 + 0.806857 w_k1_FDX_s006
                                 + 0.718243 w_k1_IBM_s006
                                 + 0.780815 w_k1_INTC_s006
                                 + 0.718356 w_k1_KO_s006
                                 + 0.817296 w_k1_MAT_s006
                                 + 0.900668 w_k1_MSFT_s006
                                 + 0.811104 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.1713 w_k1_ADM_s006
                                 + 1.37949 w_k1_AXP_s006
                                 + 0.947257 w_k1_DIS_s006
                                 + 1.20406 w_k1_FDX_s006
                                 + 0.813693 w_k1_IBM_s006
                                 + 1.17533 w_k1_INTC_s006
                                 + 0.820611 w_k1_KO_s006
                                 + 0.869909 w_k1_MAT_s006
                                 + 1.03666 w_k1_MSFT_s006
                                 + 1.04952 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.26719 w_k1_ADM_s006
                                 + 1.27244 w_k1_AXP_s006
                                 + 1.50992 w_k1_DIS_s006
                                 + 1.27839 w_k1_FDX_s006
                                 + 1.39681 w_k1_IBM_s006
                                 + 0.830265 w_k1_INTC_s006
                                 + 1.13822 w_k1_KO_s006 + 1.24049 w_k1_MAT_s006
                                 + 1.0525 w_k1_MSFT_s006 + 1.10402 w_k1_PG_s006
                                 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.964477 w_k1_ADM_s007
                                 + 0.939471 w_k1_AXP_s007
                                 + 0.851759 w_k1_DIS_s007
                                 + 0.914795 w_k1_FDX_s007
                                 + 1.27212 w_k1_IBM_s007
                                 + 1.05926 w_k1_INTC_s007
                                 + 1.04319 w_k1_KO_s007 + 1.03555 w_k1_MAT_s007
                                 + 0.922787 w_k1_MSFT_s007
                                 + 1.05414 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.38932 w_k1_ADM_s007
                                 + 1.27175 w_k1_AXP_s007 + 1.9012 w_k1_DIS_s007
                                 + 1.32567 w_k1_FDX_s007
                                 + 1.51465 w_k1_IBM_s007
                                 + 1.17097 w_k1_INTC_s007
                                 + 1.67298 w_k1_KO_s007 + 1.50428 w_k1_MAT_s007
                                 + 1.14771 w_k1_MSFT_s007
                                 + 1.30245 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.12435 w_k1_ADM_s007
                                 + 1.02354 w_k1_AXP_s007
                                 + 1.20115 w_k1_DIS_s007
                                 + 1.15762 w_k1_FDX_s007
                                 + 0.734442 w_k1_IBM_s007
                                 + 1.04273 w_k1_INTC_s007
                                 + 0.914739 w_k1_KO_s007
                                 + 0.613979 w_k1_MAT_s007
                                 + 0.923565 w_k1_MSFT_s007
                                 + 1.07084 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.0648 w_k1_ADM_s007
                                 + 1.08191 w_k1_AXP_s007
                                 + 1.26541 w_k1_DIS_s007
                                 + 1.03114 w_k1_FDX_s007
                                 + 1.37914 w_k1_IBM_s007
                                 + 1.06975 w_k1_INTC_s007
                                 + 1.44731 w_k1_KO_s007 + 1.65121 w_k1_MAT_s007
                                 + 1.06958 w_k1_MSFT_s007
                                 + 1.19657 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.09987 w_k1_ADM_s007
                                 + 0.747381 w_k1_AXP_s007
                                 + 1.04691 w_k1_DIS_s007
                                 + 0.945332 w_k1_FDX_s007
                                 + 0.821722 w_k1_IBM_s007
                                 + 0.814068 w_k1_INTC_s007
                                 + 0.956913 w_k1_KO_s007
                                 + 0.839194 w_k1_MAT_s007
                                 + 0.930065 w_k1_MSFT_s007
                                 + 1.09743 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.905722 w_k1_ADM_s007
                                 + 0.754167 w_k1_AXP_s007
                                 + 0.665636 w_k1_DIS_s007
                                 + 0.838753 w_k1_FDX_s007
                                 + 0.71725 w_k1_IBM_s007
                                 + 0.737834 w_k1_INTC_s007
                                 + 0.770134 w_k1_KO_s007
                                 + 0.636709 w_k1_MAT_s007
                                 + 0.726977 w_k1_MSFT_s007
                                 + 0.893375 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.10586 w_k1_ADM_s007
                                 + 1.00219 w_k1_AXP_s007
                                 + 0.907778 w_k1_DIS_s007
                                 + 1.03841 w_k1_FDX_s007
                                 + 0.725458 w_k1_IBM_s007
                                 + 0.830409 w_k1_INTC_s007
                                 + 0.852199 w_k1_KO_s007
                                 + 0.910074 w_k1_MAT_s007
                                 + 0.958137 w_k1_MSFT_s007
                                 + 0.961247 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.06865 w_k1_ADM_s007
                                 + 1.33869 w_k1_AXP_s007
                                 + 1.33152 w_k1_DIS_s007
                                 + 1.23374 w_k1_FDX_s007
                                 + 1.05236 w_k1_IBM_s007
                                 + 1.20014 w_k1_INTC_s007
                                 + 1.07922 w_k1_KO_s007 + 1.02035 w_k1_MAT_s007
                                 + 1.17347 w_k1_MSFT_s007
                                 + 1.22618 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.5885 w_k1_ADM_s010
                                 + 1.42668 w_k1_AXP_s010
                                 + 1.73871 w_k1_DIS_s010
                                 + 1.27965 w_k1_FDX_s010
                                 + 1.30447 w_k1_IBM_s010
                                 + 1.10099 w_k1_INTC_s010
                                 + 1.30952 w_k1_KO_s010 + 1.18277 w_k1_MAT_s010
                                 + 1.47166 w_k1_MSFT_s010
                                 + 1.09426 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.01421 w_k1_ADM_s010
                                 + 0.872706 w_k1_AXP_s010
                                 + 0.984454 w_k1_DIS_s010
                                 + 1.02268 w_k1_FDX_s010
                                 + 0.746636 w_k1_IBM_s010
                                 + 0.930635 w_k1_INTC_s010
                                 + 0.831337 w_k1_KO_s010
                                 + 0.662148 w_k1_MAT_s010
                                 + 1.03207 w_k1_MSFT_s010
                                 + 0.926619 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.03513 w_k1_ADM_s010
                                 + 1.33733 w_k1_AXP_s010
                                 + 1.07107 w_k1_DIS_s010
                                 + 0.995824 w_k1_FDX_s010
                                 + 1.34097 w_k1_IBM_s010
                                 + 1.04947 w_k1_INTC_s010 + 1.2251 w_k1_KO_s010
                                 + 1.39981 w_k1_MAT_s010
                                 + 1.2281 w_k1_MSFT_s010 + 1.16971 w_k1_PG_s010
                                 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.823478 w_k1_ADM_s010
                                 + 0.879184 w_k1_AXP_s010
                                 + 0.732126 w_k1_DIS_s010
                                 + 0.762488 w_k1_FDX_s010
                                 + 0.922617 w_k1_IBM_s010
                                 + 1.01749 w_k1_INTC_s010
                                 + 0.860544 w_k1_KO_s010
                                 + 0.716342 w_k1_MAT_s010
                                 + 0.915883 w_k1_MSFT_s010
                                 + 0.933722 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.05996 w_k1_ADM_s010
                                 + 1.05472 w_k1_AXP_s010
                                 + 1.19768 w_k1_DIS_s010
                                 + 0.998624 w_k1_FDX_s010
                                 + 1.75892 w_k1_IBM_s010
                                 + 0.979688 w_k1_INTC_s010
                                 + 1.24924 w_k1_KO_s010 + 1.25729 w_k1_MAT_s010
                                 + 1.15835 w_k1_MSFT_s010
                                 + 1.18489 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.803343 w_k1_ADM_s010
                                 + 0.642702 w_k1_AXP_s010
                                 + 0.652674 w_k1_DIS_s010
                                 + 0.814792 w_k1_FDX_s010
                                 + 0.626172 w_k1_IBM_s010
                                 + 0.781173 w_k1_INTC_s010
                                 + 0.645612 w_k1_KO_s010
                                 + 0.658478 w_k1_MAT_s010
                                 + 0.857728 w_k1_MSFT_s010
                                 + 0.824652 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.10181 w_k1_ADM_s010
                                 + 1.09842 w_k1_AXP_s010
                                 + 1.26225 w_k1_DIS_s010
                                 + 1.17711 w_k1_FDX_s010
                                 + 0.937286 w_k1_IBM_s010
                                 + 1.19311 w_k1_INTC_s010
                                 + 0.902641 w_k1_KO_s010
                                 + 0.869885 w_k1_MAT_s010
                                 + 1.27638 w_k1_MSFT_s010
                                 + 1.05079 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.756799 w_k1_ADM_s010
                                 + 0.964738 w_k1_AXP_s010
                                 + 0.903779 w_k1_DIS_s010
                                 + 0.860044 w_k1_FDX_s010
                                 + 1.17627 w_k1_IBM_s010
                                 + 1.15974 w_k1_INTC_s010
                                 + 0.951071 w_k1_KO_s010
                                 + 0.88858 w_k1_MAT_s010
                                 + 0.97968 w_k1_MSFT_s010
                                 + 0.949592 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.11831 w_k1_ADM_s011
                                 + 0.918163 w_k1_AXP_s011
                                 + 1.06294 w_k1_DIS_s011
                                 + 1.13791 w_k1_FDX_s011
                                 + 0.747037 w_k1_IBM_s011
                                 + 0.898241 w_k1_INTC_s011
                                 + 0.98801 w_k1_KO_s011
                                 + 0.781903 w_k1_MAT_s011
                                 + 1.08089 w_k1_MSFT_s011
                                 + 1.00037 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.772299 w_k1_ADM_s011
                                 + 1.14622 w_k1_AXP_s011
                                 + 0.91285 w_k1_DIS_s011
                                 + 0.933287 w_k1_FDX_s011
                                 + 1.38916 w_k1_IBM_s011
                                 + 0.869045 w_k1_INTC_s011
                                 + 1.08665 w_k1_KO_s011 + 1.3186 w_k1_MAT_s011
                                 + 0.92759 w_k1_MSFT_s011
                                 + 0.970973 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.796831 w_k1_ADM_s011
                                 + 0.745298 w_k1_AXP_s011
                                 + 0.689855 w_k1_DIS_s011
                                 + 0.851144 w_k1_FDX_s011
                                 + 0.732879 w_k1_IBM_s011
                                 + 0.817183 w_k1_INTC_s011
                                 + 0.802314 w_k1_KO_s011
                                 + 0.737959 w_k1_MAT_s011
                                 + 0.89481 w_k1_MSFT_s011
                                 + 0.819196 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.3154 w_k1_ADM_s011
                                 + 1.59112 w_k1_AXP_s011
                                 + 1.54097 w_k1_DIS_s011
                                 + 1.31953 w_k1_FDX_s011
                                 + 1.66797 w_k1_IBM_s011
                                 + 1.19765 w_k1_INTC_s011
                                 + 1.50521 w_k1_KO_s011 + 1.68072 w_k1_MAT_s011
                                 + 1.24776 w_k1_MSFT_s011
                                 + 1.26891 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.22913 w_k1_ADM_s011
                                 + 1.29714 w_k1_AXP_s011
                                 + 1.39649 w_k1_DIS_s011 + 1.3016 w_k1_FDX_s011
                                 + 1.19542 w_k1_IBM_s011
                                 + 1.03776 w_k1_INTC_s011
                                 + 1.25625 w_k1_KO_s011 + 1.22955 w_k1_MAT_s011
                                 + 1.16519 w_k1_MSFT_s011
                                 + 1.10457 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.12056 w_k1_ADM_s011
                                 + 1.19277 w_k1_AXP_s011
                                 + 1.30612 w_k1_DIS_s011
                                 + 1.26672 w_k1_FDX_s011
                                 + 0.909856 w_k1_IBM_s011
                                 + 1.1654 w_k1_INTC_s011 + 1.07584 w_k1_KO_s011
                                 + 0.90067 w_k1_MAT_s011
                                 + 1.1714 w_k1_MSFT_s011
                                 + 0.951901 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.860464 w_k1_ADM_s011
                                 + 1.06445 w_k1_AXP_s011
                                 + 1.07677 w_k1_DIS_s011
                                 + 1.10792 w_k1_FDX_s011
                                 + 1.30171 w_k1_IBM_s011
                                 + 0.940468 w_k1_INTC_s011
                                 + 1.2819 w_k1_KO_s011 + 1.75445 w_k1_MAT_s011
                                 + 1.11785 w_k1_MSFT_s011
                                 + 1.17268 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.75145 w_k1_ADM_s011
                                 + 0.819594 w_k1_AXP_s011
                                 + 0.9225 w_k1_DIS_s011
                                 + 0.853805 w_k1_FDX_s011
                                 + 1.04369 w_k1_IBM_s011
                                 + 0.826342 w_k1_INTC_s011
                                 + 1.12643 w_k1_KO_s011 + 1.05143 w_k1_MAT_s011
                                 + 0.94188 w_k1_MSFT_s011
                                 + 0.969032 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.11357 w_k1_ADM_s012
                                 + 0.905995 w_k1_AXP_s012
                                 + 0.898292 w_k1_DIS_s012
                                 + 0.910088 w_k1_FDX_s012
                                 + 0.937464 w_k1_IBM_s012
                                 + 1.01515 w_k1_INTC_s012
                                 + 1.01925 w_k1_KO_s012
                                 + 0.950926 w_k1_MAT_s012
                                 + 1.05456 w_k1_MSFT_s012
                                 + 1.05441 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.913224 w_k1_ADM_s012
                                 + 0.711367 w_k1_AXP_s012
                                 + 0.747845 w_k1_DIS_s012
                                 + 0.899889 w_k1_FDX_s012
                                 + 0.7483 w_k1_IBM_s012
                                 + 0.879823 w_k1_INTC_s012
                                 + 0.806768 w_k1_KO_s012
                                 + 0.750763 w_k1_MAT_s012
                                 + 0.916417 w_k1_MSFT_s012
                                 + 0.831632 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.952932 w_k1_ADM_s012
                                 + 1.01384 w_k1_AXP_s012
                                 + 1.16744 w_k1_DIS_s012
                                 + 1.10828 w_k1_FDX_s012
                                 + 0.737973 w_k1_IBM_s012
                                 + 0.9921 w_k1_INTC_s012
                                 + 0.970647 w_k1_KO_s012
                                 + 0.933133 w_k1_MAT_s012
                                 + 0.978201 w_k1_MSFT_s012
                                 + 0.919389 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.08455 w_k1_ADM_s012
                                 + 0.976765 w_k1_AXP_s012
                                 + 1.02756 w_k1_DIS_s012
                                 + 0.96109 w_k1_FDX_s012
                                 + 1.20916 w_k1_IBM_s012
                                 + 1.04057 w_k1_INTC_s012
                                 + 1.09716 w_k1_KO_s012 + 1.26564 w_k1_MAT_s012
                                 + 0.933499 w_k1_MSFT_s012
                                 + 1.04195 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.599054 w_k1_ADM_s012
                                 + 0.539657 w_k1_AXP_s012
                                 + 0.728935 w_k1_DIS_s012
                                 + 0.770108 w_k1_FDX_s012
                                 + 0.678115 w_k1_IBM_s012
                                 + 1.07792 w_k1_INTC_s012
                                 + 0.875748 w_k1_KO_s012
                                 + 0.763147 w_k1_MAT_s012
                                 + 0.96305 w_k1_MSFT_s012
                                 + 0.889649 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.854701 w_k1_ADM_s012
                                 + 0.784053 w_k1_AXP_s012
                                 + 0.897805 w_k1_DIS_s012
                                 + 0.898074 w_k1_FDX_s012
                                 + 1.04893 w_k1_IBM_s012
                                 + 0.938342 w_k1_INTC_s012
                                 + 1.01855 w_k1_KO_s012
                                 + 0.870384 w_k1_MAT_s012
                                 + 0.889413 w_k1_MSFT_s012
                                 + 0.943373 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.33083 w_k1_ADM_s012
                                 + 1.11036 w_k1_AXP_s012
                                 + 1.37384 w_k1_DIS_s012
                                 + 1.12553 w_k1_FDX_s012
                                 + 1.56331 w_k1_IBM_s012
                                 + 1.22721 w_k1_INTC_s012
                                 + 1.65493 w_k1_KO_s012 + 1.26948 w_k1_MAT_s012
                                 + 1.24438 w_k1_MSFT_s012
                                 + 1.17873 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.11586 w_k1_ADM_s012
                                 + 0.998104 w_k1_AXP_s012
                                 + 1.45591 w_k1_DIS_s012
                                 + 1.18847 w_k1_FDX_s012
                                 + 1.07436 w_k1_IBM_s012
                                 + 1.1559 w_k1_INTC_s012 + 1.34601 w_k1_KO_s012
                                 + 0.923171 w_k1_MAT_s012
                                 + 1.14606 w_k1_MSFT_s012
                                 + 1.14457 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.09786 w_k1_ADM_s013
                                 + 1.04828 w_k1_AXP_s013
                                 + 1.21712 w_k1_DIS_s013
                                 + 1.06243 w_k1_FDX_s013
                                 + 1.31417 w_k1_IBM_s013
                                 + 1.27611 w_k1_INTC_s013
                                 + 1.21332 w_k1_KO_s013 + 1.83487 w_k1_MAT_s013
                                 + 1.14018 w_k1_MSFT_s013
                                 + 1.18308 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.1569 w_k1_ADM_s013
                                 + 1.34899 w_k1_AXP_s013
                                 + 1.52681 w_k1_DIS_s013
                                 + 1.15814 w_k1_FDX_s013 + 1.6374 w_k1_IBM_s013
                                 + 1.32778 w_k1_INTC_s013
                                 + 1.29532 w_k1_KO_s013 + 2.14978 w_k1_MAT_s013
                                 + 1.11982 w_k1_MSFT_s013
                                 + 1.03736 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.10755 w_k1_ADM_s013
                                 + 1.50866 w_k1_AXP_s013
                                 + 1.38286 w_k1_DIS_s013
                                 + 1.29214 w_k1_FDX_s013
                                 + 0.94634 w_k1_IBM_s013
                                 + 1.47632 w_k1_INTC_s013
                                 + 0.956717 w_k1_KO_s013
                                 + 1.23137 w_k1_MAT_s013
                                 + 1.3008 w_k1_MSFT_s013
                                 + 0.959068 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.61646 w_k1_ADM_s013
                                 + 0.790442 w_k1_AXP_s013
                                 + 0.70565 w_k1_DIS_s013
                                 + 0.904781 w_k1_FDX_s013
                                 + 0.584565 w_k1_IBM_s013
                                 + 0.996322 w_k1_INTC_s013
                                 + 0.579046 w_k1_KO_s013
                                 + 0.870579 w_k1_MAT_s013
                                 + 0.855476 w_k1_MSFT_s013
                                 + 0.727517 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.881944 w_k1_ADM_s013
                                 + 0.86401 w_k1_AXP_s013
                                 + 0.890183 w_k1_DIS_s013
                                 + 0.896551 w_k1_FDX_s013
                                 + 1.12954 w_k1_IBM_s013
                                 + 0.987621 w_k1_INTC_s013
                                 + 0.909759 w_k1_KO_s013
                                 + 1.29565 w_k1_MAT_s013
                                 + 0.784067 w_k1_MSFT_s013
                                 + 0.795095 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.882135 w_k1_ADM_s013
                                 + 1.09216 w_k1_AXP_s013
                                 + 0.816505 w_k1_DIS_s013
                                 + 0.885048 w_k1_FDX_s013
                                 + 0.932002 w_k1_IBM_s013
                                 + 0.977954 w_k1_INTC_s013
                                 + 0.834865 w_k1_KO_s013 + 1.0882 w_k1_MAT_s013
                                 + 0.852333 w_k1_MSFT_s013
                                 + 0.785869 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.74602 w_k1_ADM_s013
                                 + 0.894024 w_k1_AXP_s013
                                 + 0.852794 w_k1_DIS_s013
                                 + 0.898426 w_k1_FDX_s013
                                 + 0.720111 w_k1_IBM_s013
                                 + 0.915613 w_k1_INTC_s013
                                 + 0.645184 w_k1_KO_s013
                                 + 1.03975 w_k1_MAT_s013
                                 + 0.824859 w_k1_MSFT_s013
                                 + 0.79336 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.807731 w_k1_ADM_s013
                                 + 0.766715 w_k1_AXP_s013
                                 + 0.730242 w_k1_DIS_s013
                                 + 0.786538 w_k1_FDX_s013
                                 + 0.824233 w_k1_IBM_s013
                                 + 1.02038 w_k1_INTC_s013
                                 + 0.815081 w_k1_KO_s013
                                 + 0.978644 w_k1_MAT_s013
                                 + 0.830438 w_k1_MSFT_s013
                                 + 0.791147 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.971192 w_k1_ADM_s014
                                 + 1.03306 w_k1_AXP_s014
                                 + 0.883129 w_k1_DIS_s014
                                 + 0.875528 w_k1_FDX_s014
                                 + 1.19417 w_k1_IBM_s014
                                 + 0.704605 w_k1_INTC_s014
                                 + 0.894396 w_k1_KO_s014
                                 + 1.22625 w_k1_MAT_s014
                                 + 0.845738 w_k1_MSFT_s014
                                 + 0.925669 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.891296 w_k1_ADM_s014
                                 + 1.43924 w_k1_AXP_s014
                                 + 1.49742 w_k1_DIS_s014
                                 + 1.13707 w_k1_FDX_s014 + 1.5493 w_k1_IBM_s014
                                 + 1.20012 w_k1_INTC_s014
                                 + 1.46012 w_k1_KO_s014 + 1.66736 w_k1_MAT_s014
                                 + 1.12334 w_k1_MSFT_s014
                                 + 1.08121 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.06014 w_k1_ADM_s014
                                 + 1.54468 w_k1_AXP_s014
                                 + 1.57779 w_k1_DIS_s014
                                 + 1.34531 w_k1_FDX_s014
                                 + 1.07876 w_k1_IBM_s014
                                 + 0.942072 w_k1_INTC_s014
                                 + 1.13997 w_k1_KO_s014 + 1.32573 w_k1_MAT_s014
                                 + 1.07736 w_k1_MSFT_s014
                                 + 1.05638 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.766175 w_k1_ADM_s014
                                 + 0.689695 w_k1_AXP_s014
                                 + 0.648939 w_k1_DIS_s014
                                 + 0.682677 w_k1_FDX_s014
                                 + 0.87958 w_k1_IBM_s014
                                 + 0.659942 w_k1_INTC_s014
                                 + 0.740013 w_k1_KO_s014
                                 + 0.58286 w_k1_MAT_s014
                                 + 0.772278 w_k1_MSFT_s014
                                 + 0.77577 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.03957 w_k1_ADM_s014
                                 + 1.05878 w_k1_AXP_s014 + 0.9911 w_k1_DIS_s014
                                 + 1.09173 w_k1_FDX_s014
                                 + 0.902641 w_k1_IBM_s014
                                 + 0.839349 w_k1_INTC_s014
                                 + 0.923758 w_k1_KO_s014
                                 + 0.958256 w_k1_MAT_s014
                                 + 0.91262 w_k1_MSFT_s014
                                 + 0.955642 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.05325 w_k1_ADM_s014
                                 + 1.29452 w_k1_AXP_s014
                                 + 1.25292 w_k1_DIS_s014
                                 + 1.35633 w_k1_FDX_s014
                                 + 0.774197 w_k1_IBM_s014
                                 + 1.08047 w_k1_INTC_s014
                                 + 0.861973 w_k1_KO_s014
                                 + 0.973197 w_k1_MAT_s014
                                 + 1.18565 w_k1_MSFT_s014
                                 + 1.06288 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.991382 w_k1_ADM_s014
                                 + 1.6831 w_k1_AXP_s014 + 1.30303 w_k1_DIS_s014
                                 + 1.16049 w_k1_FDX_s014
                                 + 1.30578 w_k1_IBM_s014
                                 + 1.13517 w_k1_INTC_s014
                                 + 1.05425 w_k1_KO_s014 + 1.56226 w_k1_MAT_s014
                                 + 1.08343 w_k1_MSFT_s014 + 1.065 w_k1_PG_s014
                                 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.7637 w_k1_ADM_s014
                                 + 0.688214 w_k1_AXP_s014
                                 + 0.774729 w_k1_DIS_s014
                                 + 0.879282 w_k1_FDX_s014
                                 + 0.576919 w_k1_IBM_s014
                                 + 0.725741 w_k1_INTC_s014
                                 + 0.715273 w_k1_KO_s014
                                 + 0.526553 w_k1_MAT_s014
                                 + 0.771673 w_k1_MSFT_s014
                                 + 0.74483 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.843582 w_k1_ADM_s015
                                 + 0.870484 w_k1_AXP_s015
                                 + 0.804521 w_k1_DIS_s015
                                 + 0.935906 w_k1_FDX_s015
                                 + 1.08068 w_k1_IBM_s015
                                 + 1.01827 w_k1_INTC_s015 + 1.0601 w_k1_KO_s015
                                 + 1.23503 w_k1_MAT_s015
                                 + 1.03572 w_k1_MSFT_s015
                                 + 1.07707 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.698069 w_k1_ADM_s015
                                 + 0.682281 w_k1_AXP_s015
                                 + 0.765155 w_k1_DIS_s015
                                 + 0.820423 w_k1_FDX_s015
                                 + 1.36337 w_k1_IBM_s015
                                 + 0.978336 w_k1_INTC_s015
                                 + 1.02907 w_k1_KO_s015 + 1.25207 w_k1_MAT_s015
                                 + 0.970288 w_k1_MSFT_s015
                                 + 0.921442 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.24876 w_k1_ADM_s015
                                 + 1.12265 w_k1_AXP_s015 + 1.0125 w_k1_DIS_s015
                                 + 1.14804 w_k1_FDX_s015
                                 + 0.773213 w_k1_IBM_s015
                                 + 1.06914 w_k1_INTC_s015
                                 + 0.874302 w_k1_KO_s015
                                 + 0.849275 w_k1_MAT_s015
                                 + 1.38586 w_k1_MSFT_s015
                                 + 1.05645 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.4186 w_k1_ADM_s015
                                 + 1.26336 w_k1_AXP_s015
                                 + 1.70557 w_k1_DIS_s015
                                 + 1.43356 w_k1_FDX_s015
                                 + 0.890366 w_k1_IBM_s015
                                 + 1.18747 w_k1_INTC_s015
                                 + 1.32423 w_k1_KO_s015 + 0.88392 w_k1_MAT_s015
                                 + 1.38059 w_k1_MSFT_s015
                                 + 1.29611 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.872729 w_k1_ADM_s015
                                 + 1.56772 w_k1_AXP_s015
                                 + 1.41605 w_k1_DIS_s015
                                 + 1.27077 w_k1_FDX_s015 + 1.8583 w_k1_IBM_s015
                                 + 1.50533 w_k1_INTC_s015
                                 + 1.49075 w_k1_KO_s015 + 1.92752 w_k1_MAT_s015
                                 + 1.5549 w_k1_MSFT_s015 + 1.17402 w_k1_PG_s015
                                 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.875752 w_k1_ADM_s015
                                 + 1.16011 w_k1_AXP_s015
                                 + 1.27198 w_k1_DIS_s015
                                 + 1.25613 w_k1_FDX_s015
                                 + 1.01316 w_k1_IBM_s015
                                 + 1.38578 w_k1_INTC_s015
                                 + 1.08927 w_k1_KO_s015 + 1.0946 w_k1_MAT_s015
                                 + 1.52114 w_k1_MSFT_s015
                                 + 1.04516 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.03597 w_k1_ADM_s015
                                 + 1.06731 w_k1_AXP_s015
                                 + 1.36247 w_k1_DIS_s015
                                 + 1.24637 w_k1_FDX_s015
                                 + 1.27954 w_k1_IBM_s015
                                 + 1.17946 w_k1_INTC_s015
                                 + 1.28509 w_k1_KO_s015 + 1.14486 w_k1_MAT_s015
                                 + 1.23954 w_k1_MSFT_s015
                                 + 1.20668 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.815313 w_k1_ADM_s015
                                 + 0.750773 w_k1_AXP_s015
                                 + 0.693143 w_k1_DIS_s015
                                 + 0.878407 w_k1_FDX_s015
                                 + 0.747186 w_k1_IBM_s015
                                 + 0.969523 w_k1_INTC_s015
                                 + 0.708301 w_k1_KO_s015
                                 + 0.683984 w_k1_MAT_s015
                                 + 0.853947 w_k1_MSFT_s015
                                 + 0.827035 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.05246 w_k1_ADM_s016
                                 + 0.865163 w_k1_AXP_s016
                                 + 1.10556 w_k1_DIS_s016
                                 + 1.07075 w_k1_FDX_s016
                                 + 1.00898 w_k1_IBM_s016
                                 + 0.996592 w_k1_INTC_s016
                                 + 1.1988 w_k1_KO_s016 + 1.07737 w_k1_MAT_s016
                                 + 1.05074 w_k1_MSFT_s016
                                 + 1.13219 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.778371 w_k1_ADM_s016
                                 + 0.807044 w_k1_AXP_s016
                                 + 0.785699 w_k1_DIS_s016
                                 + 0.879417 w_k1_FDX_s016
                                 + 0.638521 w_k1_IBM_s016
                                 + 0.910594 w_k1_INTC_s016
                                 + 0.740678 w_k1_KO_s016
                                 + 0.68201 w_k1_MAT_s016
                                 + 0.783457 w_k1_MSFT_s016
                                 + 0.862424 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.14597 w_k1_ADM_s016
                                 + 1.01306 w_k1_AXP_s016
                                 + 1.34196 w_k1_DIS_s016
                                 + 1.26023 w_k1_FDX_s016
                                 + 0.913638 w_k1_IBM_s016
                                 + 1.07981 w_k1_INTC_s016
                                 + 1.31621 w_k1_KO_s016 + 1.20829 w_k1_MAT_s016
                                 + 1.28094 w_k1_MSFT_s016
                                 + 1.28293 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.16062 w_k1_ADM_s016
                                 + 1.26448 w_k1_AXP_s016 + 1.4018 w_k1_DIS_s016
                                 + 1.25169 w_k1_FDX_s016
                                 + 1.31915 w_k1_IBM_s016
                                 + 0.932831 w_k1_INTC_s016
                                 + 1.29172 w_k1_KO_s016 + 1.37366 w_k1_MAT_s016
                                 + 1.00578 w_k1_MSFT_s016
                                 + 1.13162 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.785399 w_k1_ADM_s016
                                 + 0.905914 w_k1_AXP_s016
                                 + 1.03196 w_k1_DIS_s016
                                 + 0.959268 w_k1_FDX_s016
                                 + 0.781024 w_k1_IBM_s016
                                 + 1.05632 w_k1_INTC_s016
                                 + 0.977571 w_k1_KO_s016
                                 + 0.807039 w_k1_MAT_s016
                                 + 0.928524 w_k1_MSFT_s016
                                 + 1.00337 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.968065 w_k1_ADM_s016
                                 + 1.04975 w_k1_AXP_s016
                                 + 1.04051 w_k1_DIS_s016
                                 + 0.852463 w_k1_FDX_s016
                                 + 1.26556 w_k1_IBM_s016
                                 + 0.995258 w_k1_INTC_s016
                                 + 1.14433 w_k1_KO_s016 + 1.383 w_k1_MAT_s016
                                 + 0.818309 w_k1_MSFT_s016
                                 + 1.11218 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.653154 w_k1_ADM_s016
                                 + 0.728993 w_k1_AXP_s016
                                 + 0.695427 w_k1_DIS_s016
                                 + 0.755858 w_k1_FDX_s016
                                 + 0.961984 w_k1_IBM_s016
                                 + 0.930493 w_k1_INTC_s016
                                 + 0.91733 w_k1_KO_s016
                                 + 0.925722 w_k1_MAT_s016
                                 + 0.734413 w_k1_MSFT_s016
                                 + 0.865137 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.25031 w_k1_ADM_s016
                                 + 1.4837 w_k1_AXP_s016 + 1.8213 w_k1_DIS_s016
                                 + 1.30851 w_k1_FDX_s016
                                 + 1.07806 w_k1_IBM_s016
                                 + 1.18326 w_k1_INTC_s016
                                 + 1.36429 w_k1_KO_s016
                                 + 0.818104 w_k1_MAT_s016
                                 + 1.19505 w_k1_MSFT_s016
                                 + 1.27146 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.00628 w_k1_ADM_s017
                                 + 1.28879 w_k1_AXP_s017
                                 + 1.09617 w_k1_DIS_s017
                                 + 1.09476 w_k1_FDX_s017
                                 + 1.10075 w_k1_IBM_s017
                                 + 1.20476 w_k1_INTC_s017
                                 + 1.29915 w_k1_KO_s017 + 1.17816 w_k1_MAT_s017
                                 + 1.01301 w_k1_MSFT_s017
                                 + 1.12816 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.840875 w_k1_ADM_s017
                                 + 0.635236 w_k1_AXP_s017
                                 + 0.551897 w_k1_DIS_s017
                                 + 0.754933 w_k1_FDX_s017
                                 + 0.720046 w_k1_IBM_s017
                                 + 0.76529 w_k1_INTC_s017
                                 + 0.69433 w_k1_KO_s017
                                 + 0.694033 w_k1_MAT_s017
                                 + 0.706224 w_k1_MSFT_s017
                                 + 0.815209 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.21763 w_k1_ADM_s017
                                 + 1.26164 w_k1_AXP_s017
                                 + 1.52054 w_k1_DIS_s017
                                 + 1.36726 w_k1_FDX_s017
                                 + 1.07983 w_k1_IBM_s017
                                 + 1.62515 w_k1_INTC_s017
                                 + 1.35228 w_k1_KO_s017 + 1.22413 w_k1_MAT_s017
                                 + 1.51852 w_k1_MSFT_s017
                                 + 1.32108 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.974123 w_k1_ADM_s017
                                 + 0.924019 w_k1_AXP_s017
                                 + 0.929801 w_k1_DIS_s017
                                 + 0.903059 w_k1_FDX_s017
                                 + 1.25501 w_k1_IBM_s017
                                 + 0.896949 w_k1_INTC_s017
                                 + 1.09964 w_k1_KO_s017 + 1.11972 w_k1_MAT_s017
                                 + 0.794527 w_k1_MSFT_s017
                                 + 0.962422 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.27134 w_k1_ADM_s017
                                 + 1.26041 w_k1_AXP_s017
                                 + 1.10337 w_k1_DIS_s017
                                 + 1.25387 w_k1_FDX_s017
                                 + 0.790804 w_k1_IBM_s017
                                 + 1.21454 w_k1_INTC_s017
                                 + 1.06032 w_k1_KO_s017
                                 + 0.938152 w_k1_MAT_s017
                                 + 1.11342 w_k1_MSFT_s017
                                 + 1.04162 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.723964 w_k1_ADM_s017
                                 + 0.647549 w_k1_AXP_s017
                                 + 0.733927 w_k1_DIS_s017
                                 + 0.743301 w_k1_FDX_s017
                                 + 0.967665 w_k1_IBM_s017
                                 + 0.79267 w_k1_INTC_s017
                                 + 0.905265 w_k1_KO_s017
                                 + 0.810429 w_k1_MAT_s017
                                 + 0.728648 w_k1_MSFT_s017
                                 + 0.947118 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.38725 w_k1_ADM_s017
                                 + 1.09956 w_k1_AXP_s017
                                 + 1.39995 w_k1_DIS_s017
                                 + 1.13707 w_k1_FDX_s017
                                 + 1.56583 w_k1_IBM_s017
                                 + 1.06873 w_k1_INTC_s017
                                 + 1.46678 w_k1_KO_s017 + 1.52917 w_k1_MAT_s017
                                 + 1.07345 w_k1_MSFT_s017
                                 + 1.13858 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.940007 w_k1_ADM_s017
                                 + 0.868581 w_k1_AXP_s017
                                 + 0.839874 w_k1_DIS_s017
                                 + 0.984984 w_k1_FDX_s017
                                 + 0.800188 w_k1_IBM_s017
                                 + 0.980025 w_k1_INTC_s017
                                 + 1.02466 w_k1_KO_s017
                                 + 0.810244 w_k1_MAT_s017
                                 + 0.956774 w_k1_MSFT_s017
                                 + 0.960483 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.808115 w_k1_ADM_s100
                                 + 0.771585 w_k1_AXP_s100
                                 + 0.71864 w_k1_DIS_s100
                                 + 0.801964 w_k1_FDX_s100
                                 + 1.24485 w_k1_IBM_s100
                                 + 1.07869 w_k1_INTC_s100
                                 + 0.935625 w_k1_KO_s100
                                 + 1.13695 w_k1_MAT_s100
                                 + 0.926537 w_k1_MSFT_s100
                                 + 1.02513 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.14833 w_k1_ADM_s100
                                 + 1.21712 w_k1_AXP_s100
                                 + 0.975135 w_k1_DIS_s100
                                 + 1.0672 w_k1_FDX_s100 + 1.11721 w_k1_IBM_s100
                                 + 1.16951 w_k1_INTC_s100
                                 + 1.08974 w_k1_KO_s100 + 1.15492 w_k1_MAT_s100
                                 + 0.973252 w_k1_MSFT_s100
                                 + 1.07049 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.918614 w_k1_ADM_s100
                                 + 0.85904 w_k1_AXP_s100
                                 + 0.590938 w_k1_DIS_s100
                                 + 0.790825 w_k1_FDX_s100
                                 + 1.00217 w_k1_IBM_s100
                                 + 0.909078 w_k1_INTC_s100
                                 + 0.826578 w_k1_KO_s100
                                 + 1.01324 w_k1_MAT_s100
                                 + 0.847446 w_k1_MSFT_s100
                                 + 0.941067 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.01514 w_k1_ADM_s100
                                 + 1.06887 w_k1_AXP_s100
                                 + 1.25938 w_k1_DIS_s100 + 1.2139 w_k1_FDX_s100
                                 + 1.10256 w_k1_IBM_s100
                                 + 1.18785 w_k1_INTC_s100
                                 + 1.25622 w_k1_KO_s100 + 1.55831 w_k1_MAT_s100
                                 + 1.25168 w_k1_MSFT_s100
                                 + 1.13184 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.03086 w_k1_ADM_s100
                                 + 1.16818 w_k1_AXP_s100
                                 + 1.03944 w_k1_DIS_s100
                                 + 0.90892 w_k1_FDX_s100
                                 + 1.54501 w_k1_IBM_s100
                                 + 0.875776 w_k1_INTC_s100
                                 + 1.20952 w_k1_KO_s100 + 1.93682 w_k1_MAT_s100
                                 + 0.942773 w_k1_MSFT_s100
                                 + 1.03298 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.46566 w_k1_ADM_s100
                                 + 1.6793 w_k1_AXP_s100 + 1.3198 w_k1_DIS_s100
                                 + 1.44028 w_k1_FDX_s100
                                 + 1.30515 w_k1_IBM_s100
                                 + 1.25429 w_k1_INTC_s100
                                 + 1.28539 w_k1_KO_s100 + 1.6245 w_k1_MAT_s100
                                 + 1.26157 w_k1_MSFT_s100 + 1.2875 w_k1_PG_s100
                                 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.20836 w_k1_ADM_s100
                                 + 1.03422 w_k1_AXP_s100
                                 + 1.10496 w_k1_DIS_s100
                                 + 1.09417 w_k1_FDX_s100
                                 + 0.89157 w_k1_IBM_s100
                                 + 1.02766 w_k1_INTC_s100
                                 + 1.03737 w_k1_KO_s100 + 1.1548 w_k1_MAT_s100
                                 + 1.08531 w_k1_MSFT_s100
                                 + 1.13795 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.973221 w_k1_ADM_s100
                                 + 0.675644 w_k1_AXP_s100
                                 + 0.794067 w_k1_DIS_s100
                                 + 0.921467 w_k1_FDX_s100
                                 + 0.697102 w_k1_IBM_s100
                                 + 0.869376 w_k1_INTC_s100
                                 + 0.777249 w_k1_KO_s100
                                 + 0.68151 w_k1_MAT_s100
                                 + 0.962233 w_k1_MSFT_s100
                                 + 0.865864 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.895873 w_k1_ADM_s101
                                 + 0.993665 w_k1_AXP_s101
                                 + 1.12213 w_k1_DIS_s101
                                 + 0.92205 w_k1_FDX_s101
                                 + 1.51361 w_k1_IBM_s101
                                 + 0.952818 w_k1_INTC_s101
                                 + 1.0463 w_k1_KO_s101 + 1.3704 w_k1_MAT_s101
                                 + 0.881907 w_k1_MSFT_s101
                                 + 0.919083 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.11096 w_k1_ADM_s101
                                 + 0.783792 w_k1_AXP_s101
                                 + 1.00849 w_k1_DIS_s101
                                 + 0.982988 w_k1_FDX_s101
                                 + 0.669472 w_k1_IBM_s101
                                 + 0.89614 w_k1_INTC_s101
                                 + 0.771189 w_k1_KO_s101
                                 + 0.414465 w_k1_MAT_s101
                                 + 0.930064 w_k1_MSFT_s101
                                 + 0.809112 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.964962 w_k1_ADM_s101
                                 + 1.16532 w_k1_AXP_s101 + 1.1574 w_k1_DIS_s101
                                 + 1.18891 w_k1_FDX_s101
                                 + 0.868593 w_k1_IBM_s101
                                 + 1.02879 w_k1_INTC_s101
                                 + 0.896875 w_k1_KO_s101
                                 + 1.10547 w_k1_MAT_s101
                                 + 1.00908 w_k1_MSFT_s101 + 1.0323 w_k1_PG_s101
                                 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.863416 w_k1_ADM_s101
                                 + 0.68269 w_k1_AXP_s101
                                 + 0.653835 w_k1_DIS_s101
                                 + 0.718129 w_k1_FDX_s101
                                 + 1.05142 w_k1_IBM_s101
                                 + 0.664485 w_k1_INTC_s101
                                 + 0.825493 w_k1_KO_s101 + 1.1729 w_k1_MAT_s101
                                 + 0.722924 w_k1_MSFT_s101
                                 + 0.879238 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.14367 w_k1_ADM_s101
                                 + 1.02285 w_k1_AXP_s101
                                 + 1.33048 w_k1_DIS_s101
                                 + 1.20573 w_k1_FDX_s101
                                 + 1.23054 w_k1_IBM_s101
                                 + 1.1849 w_k1_INTC_s101 + 1.06325 w_k1_KO_s101
                                 + 0.829416 w_k1_MAT_s101
                                 + 1.14886 w_k1_MSFT_s101
                                 + 1.06801 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.942691 w_k1_ADM_s101
                                 + 0.859765 w_k1_AXP_s101
                                 + 0.816742 w_k1_DIS_s101
                                 + 0.87866 w_k1_FDX_s101
                                 + 0.868049 w_k1_IBM_s101
                                 + 0.867498 w_k1_INTC_s101
                                 + 0.857837 w_k1_KO_s101
                                 + 0.877838 w_k1_MAT_s101
                                 + 0.870911 w_k1_MSFT_s101
                                 + 0.930782 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.842586 w_k1_ADM_s101
                                 + 0.626113 w_k1_AXP_s101
                                 + 0.602832 w_k1_DIS_s101
                                 + 0.715511 w_k1_FDX_s101
                                 + 0.728209 w_k1_IBM_s101
                                 + 0.84055 w_k1_INTC_s101
                                 + 0.716857 w_k1_KO_s101
                                 + 0.60635 w_k1_MAT_s101
                                 + 0.777504 w_k1_MSFT_s101
                                 + 0.725468 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.902939 w_k1_ADM_s101
                                 + 0.934208 w_k1_AXP_s101
                                 + 1.2181 w_k1_DIS_s101
                                 + 0.981375 w_k1_FDX_s101
                                 + 1.06089 w_k1_IBM_s101
                                 + 0.907031 w_k1_INTC_s101
                                 + 1.10176 w_k1_KO_s101
                                 + 0.907447 w_k1_MAT_s101
                                 + 0.954349 w_k1_MSFT_s101
                                 + 1.01952 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.946321 w_k1_ADM_s102
                                 + 0.976734 w_k1_AXP_s102
                                 + 1.21762 w_k1_DIS_s102
                                 + 1.12401 w_k1_FDX_s102
                                 + 0.64082 w_k1_IBM_s102
                                 + 1.29566 w_k1_INTC_s102
                                 + 0.911754 w_k1_KO_s102
                                 + 0.609887 w_k1_MAT_s102
                                 + 1.26095 w_k1_MSFT_s102
                                 + 1.00483 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.16674 w_k1_ADM_s102
                                 + 1.32547 w_k1_AXP_s102
                                 + 1.30151 w_k1_DIS_s102
                                 + 1.05297 w_k1_FDX_s102
                                 + 1.78088 w_k1_IBM_s102
                                 + 1.1999 w_k1_INTC_s102 + 1.44868 w_k1_KO_s102
                                 + 1.74895 w_k1_MAT_s102
                                 + 1.25116 w_k1_MSFT_s102
                                 + 1.22644 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.950573 w_k1_ADM_s102
                                 + 1.07299 w_k1_AXP_s102 + 1.0705 w_k1_DIS_s102
                                 + 0.958974 w_k1_FDX_s102
                                 + 1.04968 w_k1_IBM_s102
                                 + 1.25008 w_k1_INTC_s102
                                 + 1.05959 w_k1_KO_s102 + 1.05618 w_k1_MAT_s102
                                 + 1.06881 w_k1_MSFT_s102
                                 + 0.999437 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.610444 w_k1_ADM_s102
                                 + 0.804216 w_k1_AXP_s102
                                 + 0.708344 w_k1_DIS_s102
                                 + 0.733719 w_k1_FDX_s102
                                 + 1.20591 w_k1_IBM_s102
                                 + 0.947048 w_k1_INTC_s102
                                 + 0.880988 w_k1_KO_s102
                                 + 0.988784 w_k1_MAT_s102
                                 + 0.872066 w_k1_MSFT_s102
                                 + 0.842509 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.2432 w_k1_ADM_s102
                                 + 1.02019 w_k1_AXP_s102
                                 + 1.20659 w_k1_DIS_s102
                                 + 1.12904 w_k1_FDX_s102 + 1.1082 w_k1_IBM_s102
                                 + 1.20987 w_k1_INTC_s102
                                 + 1.07807 w_k1_KO_s102 + 1.22134 w_k1_MAT_s102
                                 + 1.27331 w_k1_MSFT_s102 + 1.0913 w_k1_PG_s102
                                 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.973271 w_k1_ADM_s102
                                 + 1.34578 w_k1_AXP_s102
                                 + 0.926882 w_k1_DIS_s102
                                 + 0.920916 w_k1_FDX_s102
                                 + 1.57786 w_k1_IBM_s102
                                 + 1.00008 w_k1_INTC_s102
                                 + 1.18847 w_k1_KO_s102 + 1.53771 w_k1_MAT_s102
                                 + 0.958756 w_k1_MSFT_s102
                                 + 1.12745 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.28998 w_k1_ADM_s102
                                 + 1.48384 w_k1_AXP_s102
                                 + 1.83298 w_k1_DIS_s102
                                 + 1.40981 w_k1_FDX_s102
                                 + 1.30976 w_k1_IBM_s102
                                 + 1.32841 w_k1_INTC_s102
                                 + 1.29675 w_k1_KO_s102 + 1.04479 w_k1_MAT_s102
                                 + 1.3869 w_k1_MSFT_s102 + 1.10113 w_k1_PG_s102
                                 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.695405 w_k1_ADM_s102
                                 + 0.641384 w_k1_AXP_s102
                                 + 0.770055 w_k1_DIS_s102
                                 + 0.803458 w_k1_FDX_s102
                                 + 0.660175 w_k1_IBM_s102
                                 + 1.00375 w_k1_INTC_s102
                                 + 0.681584 w_k1_KO_s102
                                 + 0.584824 w_k1_MAT_s102
                                 + 0.96553 w_k1_MSFT_s102
                                 + 0.797893 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.755965 w_k1_ADM_s103
                                 + 0.785926 w_k1_AXP_s103
                                 + 0.687911 w_k1_DIS_s103
                                 + 0.807722 w_k1_FDX_s103
                                 + 0.668552 w_k1_IBM_s103
                                 + 0.911335 w_k1_INTC_s103
                                 + 0.745976 w_k1_KO_s103
                                 + 0.675203 w_k1_MAT_s103
                                 + 0.777362 w_k1_MSFT_s103
                                 + 0.794734 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.0707 w_k1_ADM_s103
                                 + 0.994736 w_k1_AXP_s103
                                 + 0.972212 w_k1_DIS_s103
                                 + 1.02828 w_k1_FDX_s103
                                 + 0.864195 w_k1_IBM_s103
                                 + 1.20589 w_k1_INTC_s103
                                 + 0.89127 w_k1_KO_s103
                                 + 0.743021 w_k1_MAT_s103
                                 + 0.975713 w_k1_MSFT_s103
                                 + 1.00686 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.792655 w_k1_ADM_s103
                                 + 0.864477 w_k1_AXP_s103
                                 + 0.910898 w_k1_DIS_s103
                                 + 0.794724 w_k1_FDX_s103
                                 + 1.32851 w_k1_IBM_s103
                                 + 0.942697 w_k1_INTC_s103
                                 + 1.03558 w_k1_KO_s103 + 1.15122 w_k1_MAT_s103
                                 + 0.831266 w_k1_MSFT_s103
                                 + 0.966567 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.3929 w_k1_ADM_s103
                                 + 1.06447 w_k1_AXP_s103
                                 + 0.940693 w_k1_DIS_s103
                                 + 1.11315 w_k1_FDX_s103
                                 + 0.719339 w_k1_IBM_s103
                                 + 0.991167 w_k1_INTC_s103
                                 + 0.836521 w_k1_KO_s103
                                 + 0.736293 w_k1_MAT_s103
                                 + 0.988275 w_k1_MSFT_s103
                                 + 0.986869 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.07531 w_k1_ADM_s103
                                 + 1.44745 w_k1_AXP_s103
                                 + 1.67023 w_k1_DIS_s103
                                 + 1.17563 w_k1_FDX_s103
                                 + 1.72316 w_k1_IBM_s103
                                 + 1.38735 w_k1_INTC_s103
                                 + 1.66505 w_k1_KO_s103 + 1.83795 w_k1_MAT_s103
                                 + 1.39079 w_k1_MSFT_s103
                                 + 1.21792 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.49207 w_k1_ADM_s103
                                 + 1.65012 w_k1_AXP_s103
                                 + 1.34491 w_k1_DIS_s103
                                 + 1.32218 w_k1_FDX_s103
                                 + 1.03487 w_k1_IBM_s103
                                 + 1.23343 w_k1_INTC_s103
                                 + 1.25742 w_k1_KO_s103 + 1.34464 w_k1_MAT_s103
                                 + 1.21992 w_k1_MSFT_s103
                                 + 1.12374 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.16083 w_k1_ADM_s103
                                 + 1.2596 w_k1_AXP_s103 + 1.03947 w_k1_DIS_s103
                                 + 1.10672 w_k1_FDX_s103
                                 + 1.20559 w_k1_IBM_s103
                                 + 1.05655 w_k1_INTC_s103
                                 + 1.21591 w_k1_KO_s103 + 1.40666 w_k1_MAT_s103
                                 + 1.08009 w_k1_MSFT_s103
                                 + 1.07388 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.04224 w_k1_ADM_s103
                                 + 0.936096 w_k1_AXP_s103
                                 + 0.90633 w_k1_DIS_s103
                                 + 0.86466 w_k1_FDX_s103
                                 + 1.01608 w_k1_IBM_s103
                                 + 0.894354 w_k1_INTC_s103
                                 + 0.914331 w_k1_KO_s103
                                 + 1.01063 w_k1_MAT_s103
                                 + 0.875248 w_k1_MSFT_s103
                                 + 0.954954 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.1598 w_k1_ADM_s104
                                 + 1.44395 w_k1_AXP_s104
                                 + 1.78238 w_k1_DIS_s104
                                 + 1.44636 w_k1_FDX_s104
                                 + 1.19462 w_k1_IBM_s104
                                 + 1.04365 w_k1_INTC_s104
                                 + 1.31347 w_k1_KO_s104 + 1.53011 w_k1_MAT_s104
                                 + 1.29033 w_k1_MSFT_s104
                                 + 1.31313 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.970141 w_k1_ADM_s104
                                 + 1.09939 w_k1_AXP_s104
                                 + 1.30662 w_k1_DIS_s104
                                 + 1.05573 w_k1_FDX_s104 + 1.6653 w_k1_IBM_s104
                                 + 0.744735 w_k1_INTC_s104
                                 + 1.20333 w_k1_KO_s104 + 1.61986 w_k1_MAT_s104
                                 + 0.984576 w_k1_MSFT_s104
                                 + 1.11923 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.902912 w_k1_ADM_s104
                                 + 0.726288 w_k1_AXP_s104
                                 + 0.839672 w_k1_DIS_s104
                                 + 0.852815 w_k1_FDX_s104
                                 + 1.08362 w_k1_IBM_s104
                                 + 0.800673 w_k1_INTC_s104
                                 + 0.979974 w_k1_KO_s104
                                 + 0.952059 w_k1_MAT_s104
                                 + 1.04212 w_k1_MSFT_s104
                                 + 0.945272 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.16936 w_k1_ADM_s104
                                 + 1.04292 w_k1_AXP_s104
                                 + 1.26285 w_k1_DIS_s104
                                 + 1.20142 w_k1_FDX_s104
                                 + 0.759388 w_k1_IBM_s104
                                 + 1.08437 w_k1_INTC_s104
                                 + 1.12644 w_k1_KO_s104
                                 + 0.814788 w_k1_MAT_s104
                                 + 1.28636 w_k1_MSFT_s104
                                 + 1.23977 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.04084 w_k1_ADM_s104
                                 + 1.01096 w_k1_AXP_s104
                                 + 1.03919 w_k1_DIS_s104
                                 + 0.984548 w_k1_FDX_s104
                                 + 1.1548 w_k1_IBM_s104
                                 + 0.77652 w_k1_INTC_s104
                                 + 0.924799 w_k1_KO_s104 + 1.1336 w_k1_MAT_s104
                                 + 0.983081 w_k1_MSFT_s104
                                 + 0.981871 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.737879 w_k1_ADM_s104
                                 + 0.588251 w_k1_AXP_s104
                                 + 0.729372 w_k1_DIS_s104
                                 + 0.814463 w_k1_FDX_s104
                                 + 0.787776 w_k1_IBM_s104
                                 + 0.805688 w_k1_INTC_s104
                                 + 0.808537 w_k1_KO_s104
                                 + 0.818672 w_k1_MAT_s104
                                 + 0.963807 w_k1_MSFT_s104
                                 + 0.89589 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.935656 w_k1_ADM_s104
                                 + 0.70975 w_k1_AXP_s104
                                 + 0.725999 w_k1_DIS_s104
                                 + 0.75104 w_k1_FDX_s104 + 1.3163 w_k1_IBM_s104
                                 + 0.647929 w_k1_INTC_s104
                                 + 0.964635 w_k1_KO_s104
                                 + 1.19372 w_k1_MAT_s104
                                 + 0.798619 w_k1_MSFT_s104
                                 + 0.951957 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.05968 w_k1_ADM_s104
                                 + 1.03458 w_k1_AXP_s104
                                 + 1.00366 w_k1_DIS_s104
                                 + 1.04979 w_k1_FDX_s104
                                 + 0.920748 w_k1_IBM_s104
                                 + 0.933644 w_k1_INTC_s104
                                 + 0.817538 w_k1_KO_s104
                                 + 1.09562 w_k1_MAT_s104
                                 + 0.986351 w_k1_MSFT_s104
                                 + 1.03793 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.20151 w_k1_ADM_s105
                                 + 1.01809 w_k1_AXP_s105
                                 + 1.04283 w_k1_DIS_s105
                                 + 1.07302 w_k1_FDX_s105
                                 + 0.970254 w_k1_IBM_s105
                                 + 1.15479 w_k1_INTC_s105
                                 + 1.07198 w_k1_KO_s105 + 1.04236 w_k1_MAT_s105
                                 + 1.11124 w_k1_MSFT_s105
                                 + 1.22979 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.02475 w_k1_ADM_s105
                                 + 0.879926 w_k1_AXP_s105
                                 + 0.874349 w_k1_DIS_s105
                                 + 0.855247 w_k1_FDX_s105
                                 + 1.15228 w_k1_IBM_s105
                                 + 0.868138 w_k1_INTC_s105
                                 + 0.999829 w_k1_KO_s105
                                 + 1.29116 w_k1_MAT_s105
                                 + 0.980888 w_k1_MSFT_s105
                                 + 1.0561 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.21917 w_k1_ADM_s105
                                 + 1.52587 w_k1_AXP_s105
                                 + 1.50755 w_k1_DIS_s105
                                 + 1.27471 w_k1_FDX_s105
                                 + 1.47206 w_k1_IBM_s105
                                 + 1.43413 w_k1_INTC_s105
                                 + 1.32759 w_k1_KO_s105 + 1.38935 w_k1_MAT_s105
                                 + 1.19788 w_k1_MSFT_s105 + 1.1816 w_k1_PG_s105
                                 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.11851 w_k1_ADM_s105
                                 + 1.19019 w_k1_AXP_s105
                                 + 1.29312 w_k1_DIS_s105
                                 + 1.26811 w_k1_FDX_s105
                                 + 0.972974 w_k1_IBM_s105
                                 + 1.21295 w_k1_INTC_s105
                                 + 0.960336 w_k1_KO_s105
                                 + 1.02044 w_k1_MAT_s105
                                 + 1.24499 w_k1_MSFT_s105 + 1.0465 w_k1_PG_s105
                                 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.954504 w_k1_ADM_s105
                                 + 0.78283 w_k1_AXP_s105
                                 + 0.914468 w_k1_DIS_s105
                                 + 0.978323 w_k1_FDX_s105
                                 + 0.786727 w_k1_IBM_s105
                                 + 0.923802 w_k1_INTC_s105
                                 + 0.796243 w_k1_KO_s105
                                 + 0.628046 w_k1_MAT_s105
                                 + 0.955694 w_k1_MSFT_s105
                                 + 0.962053 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.703533 w_k1_ADM_s105
                                 + 0.687959 w_k1_AXP_s105
                                 + 0.548767 w_k1_DIS_s105
                                 + 0.705438 w_k1_FDX_s105
                                 + 0.655693 w_k1_IBM_s105
                                 + 0.775606 w_k1_INTC_s105
                                 + 0.63063 w_k1_KO_s105
                                 + 0.655158 w_k1_MAT_s105
                                 + 0.775621 w_k1_MSFT_s105
                                 + 0.798126 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.11218 w_k1_ADM_s105
                                 + 1.15572 w_k1_AXP_s105
                                 + 1.10302 w_k1_DIS_s105
                                 + 1.00651 w_k1_FDX_s105 + 1.7603 w_k1_IBM_s105
                                 + 1.26365 w_k1_INTC_s105
                                 + 1.14925 w_k1_KO_s105 + 1.50492 w_k1_MAT_s105
                                 + 1.07615 w_k1_MSFT_s105
                                 + 1.12723 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.01736 w_k1_ADM_s105
                                 + 0.663867 w_k1_AXP_s105
                                 + 0.875507 w_k1_DIS_s105
                                 + 0.9376 w_k1_FDX_s105
                                 + 0.591656 w_k1_IBM_s105
                                 + 1.00009 w_k1_INTC_s105
                                 + 0.670921 w_k1_KO_s105
                                 + 0.474906 w_k1_MAT_s105
                                 + 0.987554 w_k1_MSFT_s105
                                 + 0.848693 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.917958 w_k1_ADM_s106
                                 + 0.749666 w_k1_AXP_s106
                                 + 0.690066 w_k1_DIS_s106
                                 + 0.8255 w_k1_FDX_s106
                                 + 0.673675 w_k1_IBM_s106
                                 + 0.812888 w_k1_INTC_s106
                                 + 0.919139 w_k1_KO_s106
                                 + 0.70234 w_k1_MAT_s106
                                 + 0.841697 w_k1_MSFT_s106
                                 + 0.776967 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.1865 w_k1_ADM_s106
                                 + 1.28407 w_k1_AXP_s106 + 1.2134 w_k1_DIS_s106
                                 + 1.27364 w_k1_FDX_s106
                                 + 0.802005 w_k1_IBM_s106
                                 + 1.2644 w_k1_INTC_s106 + 1.09772 w_k1_KO_s106
                                 + 1.10644 w_k1_MAT_s106
                                 + 1.27246 w_k1_MSFT_s106
                                 + 1.05883 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.986995 w_k1_ADM_s106
                                 + 0.999768 w_k1_AXP_s106
                                 + 1.14149 w_k1_DIS_s106
                                 + 1.10284 w_k1_FDX_s106 + 1.0265 w_k1_IBM_s106
                                 + 1.10133 w_k1_INTC_s106
                                 + 1.20391 w_k1_KO_s106 + 1.35657 w_k1_MAT_s106
                                 + 1.09389 w_k1_MSFT_s106
                                 + 0.976055 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.998671 w_k1_ADM_s106
                                 + 0.688124 w_k1_AXP_s106
                                 + 0.899022 w_k1_DIS_s106
                                 + 0.985777 w_k1_FDX_s106
                                 + 0.679963 w_k1_IBM_s106
                                 + 0.99961 w_k1_INTC_s106
                                 + 0.927737 w_k1_KO_s106
                                 + 0.700198 w_k1_MAT_s106
                                 + 1.11288 w_k1_MSFT_s106
                                 + 0.916564 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.02601 w_k1_ADM_s106
                                 + 1.03536 w_k1_AXP_s106
                                 + 0.842294 w_k1_DIS_s106
                                 + 0.939509 w_k1_FDX_s106
                                 + 1.23717 w_k1_IBM_s106
                                 + 1.01723 w_k1_INTC_s106
                                 + 1.10221 w_k1_KO_s106 + 1.27951 w_k1_MAT_s106
                                 + 0.895826 w_k1_MSFT_s106
                                 + 1.01347 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.10182 w_k1_ADM_s106
                                 + 1.2894 w_k1_AXP_s106 + 1.63304 w_k1_DIS_s106
                                 + 1.20161 w_k1_FDX_s106
                                 + 1.67075 w_k1_IBM_s106
                                 + 1.32114 w_k1_INTC_s106
                                 + 1.67435 w_k1_KO_s106 + 2.05174 w_k1_MAT_s106
                                 + 1.26723 w_k1_MSFT_s106
                                 + 1.22931 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.722724 w_k1_ADM_s106
                                 + 0.562264 w_k1_AXP_s106
                                 + 0.622537 w_k1_DIS_s106
                                 + 0.778507 w_k1_FDX_s106
                                 + 0.476962 w_k1_IBM_s106
                                 + 0.801251 w_k1_INTC_s106
                                 + 0.686411 w_k1_KO_s106
                                 + 0.477206 w_k1_MAT_s106
                                 + 0.784821 w_k1_MSFT_s106
                                 + 0.795185 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.649952 w_k1_ADM_s106
                                 + 0.653534 w_k1_AXP_s106
                                 + 0.669542 w_k1_DIS_s106
                                 + 0.697302 w_k1_FDX_s106
                                 + 0.887748 w_k1_IBM_s106
                                 + 0.715306 w_k1_INTC_s106
                                 + 0.780703 w_k1_KO_s106
                                 + 0.807773 w_k1_MAT_s106
                                 + 0.781453 w_k1_MSFT_s106
                                 + 0.709472 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.991185 w_k1_ADM_s107
                                 + 1.20696 w_k1_AXP_s107
                                 + 1.15319 w_k1_DIS_s107
                                 + 1.24552 w_k1_FDX_s107
                                 + 0.798848 w_k1_IBM_s107
                                 + 1.14287 w_k1_INTC_s107
                                 + 0.891088 w_k1_KO_s107
                                 + 0.723191 w_k1_MAT_s107
                                 + 1.315 w_k1_MSFT_s107 + 1.10921 w_k1_PG_s107
                                 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.870727 w_k1_ADM_s107
                                 + 0.807263 w_k1_AXP_s107
                                 + 0.920862 w_k1_DIS_s107
                                 + 1.05266 w_k1_FDX_s107
                                 + 0.580615 w_k1_IBM_s107
                                 + 1.0816 w_k1_INTC_s107
                                 + 0.795438 w_k1_KO_s107
                                 + 0.509595 w_k1_MAT_s107
                                 + 1.28967 w_k1_MSFT_s107
                                 + 0.986885 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.871193 w_k1_ADM_s107
                                 + 0.800998 w_k1_AXP_s107
                                 + 0.813699 w_k1_DIS_s107
                                 + 0.873388 w_k1_FDX_s107
                                 + 1.01942 w_k1_IBM_s107
                                 + 1.01562 w_k1_INTC_s107
                                 + 0.974963 w_k1_KO_s107
                                 + 0.836154 w_k1_MAT_s107
                                 + 0.903904 w_k1_MSFT_s107
                                 + 0.920034 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.721685 w_k1_ADM_s107
                                 + 0.662952 w_k1_AXP_s107
                                 + 0.714604 w_k1_DIS_s107
                                 + 0.811025 w_k1_FDX_s107
                                 + 0.604472 w_k1_IBM_s107
                                 + 0.996394 w_k1_INTC_s107
                                 + 0.724216 w_k1_KO_s107
                                 + 0.584036 w_k1_MAT_s107
                                 + 0.885932 w_k1_MSFT_s107
                                 + 0.826385 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.02434 w_k1_ADM_s107
                                 + 1.12935 w_k1_AXP_s107
                                 + 1.25801 w_k1_DIS_s107
                                 + 1.07938 w_k1_FDX_s107
                                 + 1.27675 w_k1_IBM_s107
                                 + 1.18016 w_k1_INTC_s107
                                 + 1.18628 w_k1_KO_s107
                                 + 0.988315 w_k1_MAT_s107
                                 + 1.26213 w_k1_MSFT_s107
                                 + 1.19049 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.996557 w_k1_ADM_s107
                                 + 1.38836 w_k1_AXP_s107
                                 + 1.52484 w_k1_DIS_s107
                                 + 1.14728 w_k1_FDX_s107
                                 + 1.79101 w_k1_IBM_s107
                                 + 1.16537 w_k1_INTC_s107
                                 + 1.39177 w_k1_KO_s107 + 1.82126 w_k1_MAT_s107
                                 + 1.14109 w_k1_MSFT_s107
                                 + 1.25168 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.917271 w_k1_ADM_s107
                                 + 0.880634 w_k1_AXP_s107
                                 + 0.951184 w_k1_DIS_s107
                                 + 0.919324 w_k1_FDX_s107
                                 + 0.837237 w_k1_IBM_s107
                                 + 1.14053 w_k1_INTC_s107
                                 + 0.863273 w_k1_KO_s107
                                 + 0.775579 w_k1_MAT_s107
                                 + 1.04411 w_k1_MSFT_s107
                                 + 0.969251 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.642606 w_k1_ADM_s107
                                 + 0.445933 w_k1_AXP_s107
                                 + 0.535068 w_k1_DIS_s107
                                 + 0.691491 w_k1_FDX_s107
                                 + 0.621593 w_k1_IBM_s107
                                 + 0.758743 w_k1_INTC_s107
                                 + 0.57331 w_k1_KO_s107
                                 + 0.315827 w_k1_MAT_s107
                                 + 0.772617 w_k1_MSFT_s107
                                 + 0.737499 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.935168 w_k1_ADM_s110
                                 + 0.913973 w_k1_AXP_s110
                                 + 1.14524 w_k1_DIS_s110
                                 + 1.06665 w_k1_FDX_s110
                                 + 1.02671 w_k1_IBM_s110
                                 + 1.08617 w_k1_INTC_s110
                                 + 1.11501 w_k1_KO_s110 + 1.4595 w_k1_MAT_s110
                                 + 1.15872 w_k1_MSFT_s110 + 1.1771 w_k1_PG_s110
                                 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.06363 w_k1_ADM_s110
                                 + 1.11088 w_k1_AXP_s110
                                 + 1.22516 w_k1_DIS_s110
                                 + 1.25292 w_k1_FDX_s110
                                 + 0.91882 w_k1_IBM_s110
                                 + 1.08546 w_k1_INTC_s110
                                 + 1.02464 w_k1_KO_s110 + 1.14621 w_k1_MAT_s110
                                 + 1.16344 w_k1_MSFT_s110
                                 + 1.15151 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.834179 w_k1_ADM_s110
                                 + 1.02861 w_k1_AXP_s110
                                 + 0.871355 w_k1_DIS_s110
                                 + 1.05013 w_k1_FDX_s110
                                 + 1.19992 w_k1_IBM_s110
                                 + 1.08275 w_k1_INTC_s110
                                 + 0.877618 w_k1_KO_s110
                                 + 1.52241 w_k1_MAT_s110
                                 + 1.16022 w_k1_MSFT_s110
                                 + 0.947116 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.891951 w_k1_ADM_s110
                                 + 0.661591 w_k1_AXP_s110
                                 + 0.601872 w_k1_DIS_s110
                                 + 0.79574 w_k1_FDX_s110
                                 + 0.85548 w_k1_IBM_s110
                                 + 0.733257 w_k1_INTC_s110
                                 + 0.802397 w_k1_KO_s110
                                 + 1.05206 w_k1_MAT_s110
                                 + 0.772368 w_k1_MSFT_s110
                                 + 0.956948 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.736584 w_k1_ADM_s110
                                 + 0.603568 w_k1_AXP_s110
                                 + 0.708134 w_k1_DIS_s110
                                 + 0.877733 w_k1_FDX_s110
                                 + 0.699148 w_k1_IBM_s110
                                 + 0.942373 w_k1_INTC_s110
                                 + 0.750846 w_k1_KO_s110
                                 + 0.766699 w_k1_MAT_s110
                                 + 1.02051 w_k1_MSFT_s110
                                 + 0.899955 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.13685 w_k1_ADM_s110
                                 + 1.18444 w_k1_AXP_s110
                                 + 1.41947 w_k1_DIS_s110
                                 + 1.24437 w_k1_FDX_s110
                                 + 1.40265 w_k1_IBM_s110
                                 + 1.21662 w_k1_INTC_s110
                                 + 1.25122 w_k1_KO_s110 + 1.98592 w_k1_MAT_s110
                                 + 1.3878 w_k1_MSFT_s110 + 1.34285 w_k1_PG_s110
                                 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.10049 w_k1_ADM_s110
                                 + 0.975887 w_k1_AXP_s110
                                 + 0.866603 w_k1_DIS_s110
                                 + 1.08085 w_k1_FDX_s110
                                 + 0.878013 w_k1_IBM_s110
                                 + 1.04346 w_k1_INTC_s110
                                 + 0.751598 w_k1_KO_s110
                                 + 1.21178 w_k1_MAT_s110
                                 + 0.928738 w_k1_MSFT_s110
                                 + 0.842404 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.29354 w_k1_ADM_s110
                                 + 0.900659 w_k1_AXP_s110
                                 + 0.867289 w_k1_DIS_s110
                                 + 1.09866 w_k1_FDX_s110
                                 + 0.788122 w_k1_IBM_s110
                                 + 1.0545 w_k1_INTC_s110
                                 + 0.772666 w_k1_KO_s110
                                 + 0.88778 w_k1_MAT_s110
                                 + 1.10731 w_k1_MSFT_s110
                                 + 1.02175 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.30479 w_k1_ADM_s111
                                 + 1.25685 w_k1_AXP_s111
                                 + 1.44299 w_k1_DIS_s111
                                 + 1.34898 w_k1_FDX_s111
                                 + 0.823553 w_k1_IBM_s111
                                 + 1.15215 w_k1_INTC_s111
                                 + 1.01825 w_k1_KO_s111
                                 + 0.616194 w_k1_MAT_s111
                                 + 1.16325 w_k1_MSFT_s111
                                 + 1.08049 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.861259 w_k1_ADM_s111
                                 + 0.620711 w_k1_AXP_s111
                                 + 0.674751 w_k1_DIS_s111
                                 + 0.787343 w_k1_FDX_s111
                                 + 0.767327 w_k1_IBM_s111
                                 + 0.862688 w_k1_INTC_s111
                                 + 0.767892 w_k1_KO_s111
                                 + 0.464697 w_k1_MAT_s111
                                 + 0.822035 w_k1_MSFT_s111
                                 + 0.81974 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.806342 w_k1_ADM_s111
                                 + 1.17852 w_k1_AXP_s111
                                 + 0.908491 w_k1_DIS_s111
                                 + 0.859517 w_k1_FDX_s111
                                 + 1.33982 w_k1_IBM_s111
                                 + 0.9103 w_k1_INTC_s111 + 1.06299 w_k1_KO_s111
                                 + 1.38848 w_k1_MAT_s111
                                 + 0.93491 w_k1_MSFT_s111
                                 + 0.921131 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.12892 w_k1_ADM_s111
                                 + 1.66208 w_k1_AXP_s111
                                 + 1.42257 w_k1_DIS_s111
                                 + 1.14464 w_k1_FDX_s111
                                 + 1.44725 w_k1_IBM_s111
                                 + 0.890109 w_k1_INTC_s111
                                 + 1.26859 w_k1_KO_s111 + 1.56116 w_k1_MAT_s111
                                 + 1.13471 w_k1_MSFT_s111
                                 + 1.14932 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.980813 w_k1_ADM_s111
                                 + 0.908397 w_k1_AXP_s111
                                 + 0.693133 w_k1_DIS_s111
                                 + 0.917627 w_k1_FDX_s111
                                 + 0.820001 w_k1_IBM_s111
                                 + 0.742974 w_k1_INTC_s111
                                 + 0.708911 w_k1_KO_s111
                                 + 0.686646 w_k1_MAT_s111
                                 + 0.826204 w_k1_MSFT_s111
                                 + 0.804254 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.833728 w_k1_ADM_s111
                                 + 1.15903 w_k1_AXP_s111
                                 + 1.05869 w_k1_DIS_s111
                                 + 1.05001 w_k1_FDX_s111
                                 + 1.12708 w_k1_IBM_s111
                                 + 1.08068 w_k1_INTC_s111
                                 + 0.970873 w_k1_KO_s111
                                 + 1.01799 w_k1_MAT_s111
                                 + 1.05698 w_k1_MSFT_s111
                                 + 0.933112 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.26475 w_k1_ADM_s111
                                 + 1.15507 w_k1_AXP_s111
                                 + 1.54779 w_k1_DIS_s111
                                 + 1.25951 w_k1_FDX_s111
                                 + 1.21661 w_k1_IBM_s111
                                 + 1.20696 w_k1_INTC_s111
                                 + 1.31157 w_k1_KO_s111 + 1.18893 w_k1_MAT_s111
                                 + 1.3673 w_k1_MSFT_s111 + 1.07509 w_k1_PG_s111
                                 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.17836 w_k1_ADM_s111
                                 + 0.904594 w_k1_AXP_s111
                                 + 1.13796 w_k1_DIS_s111
                                 + 0.914024 w_k1_FDX_s111
                                 + 0.895619 w_k1_IBM_s111
                                 + 0.831976 w_k1_INTC_s111
                                 + 0.995071 w_k1_KO_s111
                                 + 0.673059 w_k1_MAT_s111
                                 + 1.05244 w_k1_MSFT_s111
                                 + 0.949884 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.56996 w_k1_ADM_s112
                                 + 1.43988 w_k1_AXP_s112
                                 + 1.12034 w_k1_DIS_s112
                                 + 1.35345 w_k1_FDX_s112
                                 + 1.03263 w_k1_IBM_s112
                                 + 1.10018 w_k1_INTC_s112
                                 + 1.01749 w_k1_KO_s112 + 1.35929 w_k1_MAT_s112
                                 + 1.07782 w_k1_MSFT_s112 + 1.1772 w_k1_PG_s112
                                 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.0704 w_k1_ADM_s112
                                 + 0.696503 w_k1_AXP_s112
                                 + 0.777884 w_k1_DIS_s112
                                 + 0.871542 w_k1_FDX_s112
                                 + 0.763713 w_k1_IBM_s112
                                 + 0.83324 w_k1_INTC_s112
                                 + 0.814606 w_k1_KO_s112
                                 + 0.659962 w_k1_MAT_s112
                                 + 0.816466 w_k1_MSFT_s112
                                 + 0.863891 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.21367 w_k1_ADM_s112
                                 + 1.17037 w_k1_AXP_s112
                                 + 1.70365 w_k1_DIS_s112
                                 + 1.29955 w_k1_FDX_s112
                                 + 1.29746 w_k1_IBM_s112
                                 + 1.39626 w_k1_INTC_s112
                                 + 1.53176 w_k1_KO_s112 + 1.54476 w_k1_MAT_s112
                                 + 1.49487 w_k1_MSFT_s112 + 1.4213 w_k1_PG_s112
                                 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.9449 w_k1_ADM_s112
                                 + 0.939223 w_k1_AXP_s112
                                 + 0.809596 w_k1_DIS_s112
                                 + 0.939854 w_k1_FDX_s112
                                 + 1.03434 w_k1_IBM_s112
                                 + 1.22758 w_k1_INTC_s112
                                 + 0.964686 w_k1_KO_s112
                                 + 1.09381 w_k1_MAT_s112
                                 + 0.827202 w_k1_MSFT_s112
                                 + 1.11046 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.974052 w_k1_ADM_s112
                                 + 0.907551 w_k1_AXP_s112
                                 + 1.21851 w_k1_DIS_s112
                                 + 1.07448 w_k1_FDX_s112
                                 + 1.12681 w_k1_IBM_s112
                                 + 1.17507 w_k1_INTC_s112 + 1.2141 w_k1_KO_s112
                                 + 1.03769 w_k1_MAT_s112
                                 + 1.12443 w_k1_MSFT_s112 + 1.1466 w_k1_PG_s112
                                 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.42417 w_k1_ADM_s112
                                 + 1.16058 w_k1_AXP_s112
                                 + 0.95065 w_k1_DIS_s112
                                 + 1.20964 w_k1_FDX_s112
                                 + 0.781128 w_k1_IBM_s112
                                 + 1.30584 w_k1_INTC_s112
                                 + 1.04162 w_k1_KO_s112
                                 + 0.879996 w_k1_MAT_s112
                                 + 1.09942 w_k1_MSFT_s112
                                 + 1.18626 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.933029 w_k1_ADM_s112
                                 + 0.799964 w_k1_AXP_s112
                                 + 0.838301 w_k1_DIS_s112
                                 + 0.879069 w_k1_FDX_s112
                                 + 1.25905 w_k1_IBM_s112
                                 + 0.998964 w_k1_INTC_s112
                                 + 1.07342 w_k1_KO_s112 + 1.22668 w_k1_MAT_s112
                                 + 1.0102 w_k1_MSFT_s112 + 1.03477 w_k1_PG_s112
                                 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.936728 w_k1_ADM_s112
                                 + 1.11755 w_k1_AXP_s112
                                 + 1.19774 w_k1_DIS_s112
                                 + 1.05588 w_k1_FDX_s112
                                 + 1.64016 w_k1_IBM_s112
                                 + 0.860168 w_k1_INTC_s112
                                 + 1.25852 w_k1_KO_s112 + 2.00375 w_k1_MAT_s112
                                 + 0.94045 w_k1_MSFT_s112
                                 + 1.07516 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.84035 w_k1_ADM_s113
                                 + 1.07505 w_k1_AXP_s113
                                 + 1.21941 w_k1_DIS_s113 + 1.0118 w_k1_FDX_s113
                                 + 1.16395 w_k1_IBM_s113
                                 + 0.950732 w_k1_INTC_s113
                                 + 1.01077 w_k1_KO_s113
                                 + 0.912024 w_k1_MAT_s113
                                 + 0.970798 w_k1_MSFT_s113
                                 + 1.03574 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.955543 w_k1_ADM_s113
                                 + 1.22821 w_k1_AXP_s113
                                 + 1.65343 w_k1_DIS_s113
                                 + 1.23106 w_k1_FDX_s113
                                 + 1.58138 w_k1_IBM_s113
                                 + 1.2505 w_k1_INTC_s113 + 1.41356 w_k1_KO_s113
                                 + 1.6094 w_k1_MAT_s113
                                 + 1.30188 w_k1_MSFT_s113
                                 + 1.06758 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.942992 w_k1_ADM_s113
                                 + 0.906758 w_k1_AXP_s113
                                 + 1.10394 w_k1_DIS_s113
                                 + 1.18512 w_k1_FDX_s113
                                 + 0.745743 w_k1_IBM_s113
                                 + 0.782815 w_k1_INTC_s113
                                 + 0.789082 w_k1_KO_s113
                                 + 0.783881 w_k1_MAT_s113
                                 + 1.14356 w_k1_MSFT_s113
                                 + 0.914171 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.983806 w_k1_ADM_s113
                                 + 0.806537 w_k1_AXP_s113
                                 + 0.940942 w_k1_DIS_s113
                                 + 0.93284 w_k1_FDX_s113 + 1.1333 w_k1_IBM_s113
                                 + 0.85144 w_k1_INTC_s113
                                 + 1.01143 w_k1_KO_s113 + 1.1435 w_k1_MAT_s113
                                 + 1.01335 w_k1_MSFT_s113
                                 + 0.965637 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.819089 w_k1_ADM_s113
                                 + 0.707293 w_k1_AXP_s113
                                 + 0.633552 w_k1_DIS_s113
                                 + 0.78058 w_k1_FDX_s113
                                 + 0.596283 w_k1_IBM_s113
                                 + 0.79238 w_k1_INTC_s113
                                 + 0.572326 w_k1_KO_s113
                                 + 0.388272 w_k1_MAT_s113
                                 + 0.750094 w_k1_MSFT_s113
                                 + 0.730533 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.762666 w_k1_ADM_s113
                                 + 0.668711 w_k1_AXP_s113
                                 + 0.760729 w_k1_DIS_s113
                                 + 0.786541 w_k1_FDX_s113
                                 + 0.934037 w_k1_IBM_s113
                                 + 0.76891 w_k1_INTC_s113
                                 + 0.804339 w_k1_KO_s113
                                 + 0.641212 w_k1_MAT_s113
                                 + 0.893858 w_k1_MSFT_s113
                                 + 0.747268 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.930233 w_k1_ADM_s113
                                 + 1.29294 w_k1_AXP_s113
                                 + 0.784384 w_k1_DIS_s113 + 1.027 w_k1_FDX_s113
                                 + 0.894387 w_k1_IBM_s113
                                 + 1.0044 w_k1_INTC_s113
                                 + 0.799698 w_k1_KO_s113
                                 + 1.01704 w_k1_MAT_s113
                                 + 0.863701 w_k1_MSFT_s113
                                 + 0.868688 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.998909 w_k1_ADM_s113
                                 + 1.42751 w_k1_AXP_s113
                                 + 1.35456 w_k1_DIS_s113
                                 + 1.27095 w_k1_FDX_s113
                                 + 1.28713 w_k1_IBM_s113
                                 + 1.02434 w_k1_INTC_s113
                                 + 1.11081 w_k1_KO_s113 + 1.36164 w_k1_MAT_s113
                                 + 1.06479 w_k1_MSFT_s113
                                 + 0.986974 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.879234 w_k1_ADM_s114
                                 + 0.831558 w_k1_AXP_s114
                                 + 0.680972 w_k1_DIS_s114
                                 + 0.865397 w_k1_FDX_s114
                                 + 0.645985 w_k1_IBM_s114
                                 + 0.912426 w_k1_INTC_s114
                                 + 0.712431 w_k1_KO_s114
                                 + 0.804723 w_k1_MAT_s114
                                 + 0.790848 w_k1_MSFT_s114
                                 + 0.925773 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.2296 w_k1_ADM_s114
                                 + 1.38306 w_k1_AXP_s114
                                 + 1.54974 w_k1_DIS_s114
                                 + 1.33742 w_k1_FDX_s114
                                 + 1.07918 w_k1_IBM_s114
                                 + 1.53736 w_k1_INTC_s114
                                 + 1.27452 w_k1_KO_s114 + 1.0744 w_k1_MAT_s114
                                 + 1.40839 w_k1_MSFT_s114
                                 + 1.22568 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.02025 w_k1_ADM_s114
                                 + 0.959297 w_k1_AXP_s114
                                 + 1.55086 w_k1_DIS_s114
                                 + 1.13177 w_k1_FDX_s114
                                 + 1.24814 w_k1_IBM_s114
                                 + 1.16634 w_k1_INTC_s114
                                 + 1.39728 w_k1_KO_s114 + 1.32866 w_k1_MAT_s114
                                 + 1.27437 w_k1_MSFT_s114
                                 + 1.27211 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.07263 w_k1_ADM_s114
                                 + 1.16946 w_k1_AXP_s114
                                 + 1.20557 w_k1_DIS_s114
                                 + 1.10022 w_k1_FDX_s114
                                 + 1.30592 w_k1_IBM_s114
                                 + 1.33981 w_k1_INTC_s114 + 1.1127 w_k1_KO_s114
                                 + 1.15468 w_k1_MAT_s114
                                 + 1.1045 w_k1_MSFT_s114 + 1.07653 w_k1_PG_s114
                                 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.739452 w_k1_ADM_s114
                                 + 0.900736 w_k1_AXP_s114
                                 + 0.86579 w_k1_DIS_s114
                                 + 0.882972 w_k1_FDX_s114
                                 + 0.979269 w_k1_IBM_s114
                                 + 1.03114 w_k1_INTC_s114
                                 + 0.886686 w_k1_KO_s114
                                 + 1.18105 w_k1_MAT_s114
                                 + 0.974122 w_k1_MSFT_s114
                                 + 1.01835 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.739381 w_k1_ADM_s114
                                 + 0.560211 w_k1_AXP_s114
                                 + 0.759331 w_k1_DIS_s114
                                 + 0.695633 w_k1_FDX_s114
                                 + 0.836114 w_k1_IBM_s114
                                 + 0.917107 w_k1_INTC_s114
                                 + 0.733862 w_k1_KO_s114
                                 + 0.484989 w_k1_MAT_s114
                                 + 0.762435 w_k1_MSFT_s114
                                 + 0.934104 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.05881 w_k1_ADM_s114
                                 + 1.00555 w_k1_AXP_s114 + 1.2152 w_k1_DIS_s114
                                 + 1.06669 w_k1_FDX_s114
                                 + 0.838996 w_k1_IBM_s114
                                 + 1.21697 w_k1_INTC_s114
                                 + 0.991318 w_k1_KO_s114
                                 + 0.883088 w_k1_MAT_s114
                                 + 1.10919 w_k1_MSFT_s114
                                 + 1.07288 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.958451 w_k1_ADM_s114
                                 + 0.828636 w_k1_AXP_s114
                                 + 1.11375 w_k1_DIS_s114
                                 + 0.91454 w_k1_FDX_s114
                                 + 1.48615 w_k1_IBM_s114
                                 + 1.11122 w_k1_INTC_s114
                                 + 1.16799 w_k1_KO_s114 + 1.1355 w_k1_MAT_s114
                                 + 0.985345 w_k1_MSFT_s114
                                 + 1.03894 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.2559 w_k1_ADM_s115
                                 + 1.19838 w_k1_AXP_s115
                                 + 0.927831 w_k1_DIS_s115
                                 + 1.16998 w_k1_FDX_s115
                                 + 0.717746 w_k1_IBM_s115
                                 + 1.16433 w_k1_INTC_s115
                                 + 0.936471 w_k1_KO_s115 + 1.0369 w_k1_MAT_s115
                                 + 1.23267 w_k1_MSFT_s115
                                 + 1.06388 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.06835 w_k1_ADM_s115
                                 + 1.01542 w_k1_AXP_s115
                                 + 0.875587 w_k1_DIS_s115
                                 + 0.942375 w_k1_FDX_s115
                                 + 1.00446 w_k1_IBM_s115
                                 + 0.867992 w_k1_INTC_s115
                                 + 0.920769 w_k1_KO_s115
                                 + 1.06081 w_k1_MAT_s115
                                 + 1.0629 w_k1_MSFT_s115 + 1.10891 w_k1_PG_s115
                                 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.923894 w_k1_ADM_s115
                                 + 0.930289 w_k1_AXP_s115
                                 + 1.01928 w_k1_DIS_s115
                                 + 0.978542 w_k1_FDX_s115
                                 + 0.875431 w_k1_IBM_s115
                                 + 1.02692 w_k1_INTC_s115
                                 + 0.94667 w_k1_KO_s115
                                 + 0.812516 w_k1_MAT_s115
                                 + 1.03711 w_k1_MSFT_s115
                                 + 0.951666 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.04315 w_k1_ADM_s115
                                 + 1.30021 w_k1_AXP_s115
                                 + 1.35672 w_k1_DIS_s115
                                 + 1.13241 w_k1_FDX_s115
                                 + 1.32115 w_k1_IBM_s115
                                 + 1.27579 w_k1_INTC_s115
                                 + 1.36648 w_k1_KO_s115 + 1.67317 w_k1_MAT_s115
                                 + 1.45794 w_k1_MSFT_s115
                                 + 1.28784 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.09647 w_k1_ADM_s115
                                 + 0.960401 w_k1_AXP_s115
                                 + 0.770772 w_k1_DIS_s115
                                 + 0.978965 w_k1_FDX_s115
                                 + 0.638713 w_k1_IBM_s115
                                 + 0.9742 w_k1_INTC_s115
                                 + 0.725974 w_k1_KO_s115
                                 + 1.10035 w_k1_MAT_s115
                                 + 1.00181 w_k1_MSFT_s115
                                 + 0.908664 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.1509 w_k1_ADM_s115
                                 + 1.15232 w_k1_AXP_s115
                                 + 1.38841 w_k1_DIS_s115 + 1.2029 w_k1_FDX_s115
                                 + 1.02871 w_k1_IBM_s115
                                 + 1.18345 w_k1_INTC_s115
                                 + 1.18034 w_k1_KO_s115
                                 + 0.940905 w_k1_MAT_s115
                                 + 1.20209 w_k1_MSFT_s115
                                 + 1.14199 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.861082 w_k1_ADM_s115
                                 + 0.814338 w_k1_AXP_s115
                                 + 0.66185 w_k1_DIS_s115
                                 + 0.722323 w_k1_FDX_s115
                                 + 0.81317 w_k1_IBM_s115
                                 + 0.958711 w_k1_INTC_s115
                                 + 0.703643 w_k1_KO_s115
                                 + 0.635993 w_k1_MAT_s115
                                 + 0.816954 w_k1_MSFT_s115
                                 + 0.907327 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.91684 w_k1_ADM_s115
                                 + 1.03578 w_k1_AXP_s115
                                 + 0.919051 w_k1_DIS_s115
                                 + 0.842933 w_k1_FDX_s115
                                 + 1.24546 w_k1_IBM_s115
                                 + 0.928111 w_k1_INTC_s115
                                 + 1.00707 w_k1_KO_s115 + 1.54566 w_k1_MAT_s115
                                 + 1.0123 w_k1_MSFT_s115 + 1.01542 w_k1_PG_s115
                                 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.11246 w_k1_ADM_s116
                                 + 0.957689 w_k1_AXP_s116
                                 + 1.12601 w_k1_DIS_s116
                                 + 0.978509 w_k1_FDX_s116
                                 + 1.14867 w_k1_IBM_s116
                                 + 1.13691 w_k1_INTC_s116
                                 + 1.19954 w_k1_KO_s116 + 1.13453 w_k1_MAT_s116
                                 + 1.0338 w_k1_MSFT_s116 + 1.04984 w_k1_PG_s116
                                 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.831989 w_k1_ADM_s116
                                 + 0.664219 w_k1_AXP_s116
                                 + 0.487109 w_k1_DIS_s116
                                 + 0.670201 w_k1_FDX_s116
                                 + 0.672256 w_k1_IBM_s116
                                 + 0.667117 w_k1_INTC_s116
                                 + 0.626586 w_k1_KO_s116
                                 + 0.597792 w_k1_MAT_s116
                                 + 0.673692 w_k1_MSFT_s116
                                 + 0.840973 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.04764 w_k1_ADM_s116
                                 + 1.12842 w_k1_AXP_s116
                                 + 0.791501 w_k1_DIS_s116
                                 + 0.960933 w_k1_FDX_s116
                                 + 0.997576 w_k1_IBM_s116
                                 + 0.982823 w_k1_INTC_s116
                                 + 0.838185 w_k1_KO_s116
                                 + 1.21053 w_k1_MAT_s116
                                 + 0.862555 w_k1_MSFT_s116
                                 + 1.00632 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.02786 w_k1_ADM_s116
                                 + 1.42269 w_k1_AXP_s116 + 1.7248 w_k1_DIS_s116
                                 + 1.22179 w_k1_FDX_s116
                                 + 1.71711 w_k1_IBM_s116
                                 + 1.43421 w_k1_INTC_s116
                                 + 1.49253 w_k1_KO_s116 + 1.61215 w_k1_MAT_s116
                                 + 1.26755 w_k1_MSFT_s116
                                 + 1.25864 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.914694 w_k1_ADM_s116
                                 + 0.836331 w_k1_AXP_s116
                                 + 0.768165 w_k1_DIS_s116
                                 + 0.806028 w_k1_FDX_s116
                                 + 0.945518 w_k1_IBM_s116
                                 + 0.948 w_k1_INTC_s116 + 0.982383 w_k1_KO_s116
                                 + 0.912093 w_k1_MAT_s116
                                 + 0.912618 w_k1_MSFT_s116
                                 + 0.924581 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.11521 w_k1_ADM_s116
                                 + 1.08923 w_k1_AXP_s116
                                 + 1.56487 w_k1_DIS_s116
                                 + 1.23188 w_k1_FDX_s116
                                 + 1.09214 w_k1_IBM_s116
                                 + 1.05037 w_k1_INTC_s116
                                 + 1.39207 w_k1_KO_s116 + 1.2488 w_k1_MAT_s116
                                 + 1.37639 w_k1_MSFT_s116
                                 + 1.19268 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.21821 w_k1_ADM_s116
                                 + 1.02609 w_k1_AXP_s116
                                 + 1.09337 w_k1_DIS_s116
                                 + 1.08134 w_k1_FDX_s116
                                 + 0.77376 w_k1_IBM_s116
                                 + 1.06685 w_k1_INTC_s116
                                 + 0.959842 w_k1_KO_s116
                                 + 0.77973 w_k1_MAT_s116
                                 + 1.02228 w_k1_MSFT_s116
                                 + 0.927371 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.83848 w_k1_ADM_s116
                                 + 0.845258 w_k1_AXP_s116
                                 + 0.669261 w_k1_DIS_s116
                                 + 0.825539 w_k1_FDX_s116
                                 + 0.70243 w_k1_IBM_s116
                                 + 0.996184 w_k1_INTC_s116
                                 + 0.750924 w_k1_KO_s116
                                 + 0.856456 w_k1_MAT_s116
                                 + 0.894603 w_k1_MSFT_s116
                                 + 0.847739 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.970431 w_k1_ADM_s117
                                 + 1.02907 w_k1_AXP_s117
                                 + 1.27854 w_k1_DIS_s117
                                 + 1.21576 w_k1_FDX_s117
                                 + 0.808197 w_k1_IBM_s117
                                 + 1.08173 w_k1_INTC_s117 + 0.9313 w_k1_KO_s117
                                 + 0.841256 w_k1_MAT_s117
                                 + 1.0879 w_k1_MSFT_s117 + 1.09675 w_k1_PG_s117
                                 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.921454 w_k1_ADM_s117
                                 + 1.17153 w_k1_AXP_s117
                                 + 1.10918 w_k1_DIS_s117 + 1.0521 w_k1_FDX_s117
                                 + 1.24591 w_k1_IBM_s117
                                 + 0.950361 w_k1_INTC_s117
                                 + 1.00877 w_k1_KO_s117 + 1.4233 w_k1_MAT_s117
                                 + 0.964808 w_k1_MSFT_s117
                                 + 0.978826 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.57269 w_k1_ADM_s117
                                 + 1.81617 w_k1_AXP_s117
                                 + 1.92698 w_k1_DIS_s117 + 1.6974 w_k1_FDX_s117
                                 + 1.1789 w_k1_IBM_s117
                                 + 1.18765 w_k1_INTC_s117
                                 + 1.07214 w_k1_KO_s117 + 1.95242 w_k1_MAT_s117
                                 + 1.40672 w_k1_MSFT_s117
                                 + 1.31571 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.873863 w_k1_ADM_s117
                                 + 0.858812 w_k1_AXP_s117
                                 + 0.662384 w_k1_DIS_s117
                                 + 0.800479 w_k1_FDX_s117
                                 + 1.04479 w_k1_IBM_s117
                                 + 0.755922 w_k1_INTC_s117
                                 + 0.760227 w_k1_KO_s117
                                 + 1.31521 w_k1_MAT_s117
                                 + 0.786904 w_k1_MSFT_s117
                                 + 0.946595 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.800358 w_k1_ADM_s117
                                 + 1.25718 w_k1_AXP_s117
                                 + 1.08381 w_k1_DIS_s117
                                 + 0.991811 w_k1_FDX_s117
                                 + 1.60237 w_k1_IBM_s117
                                 + 1.15246 w_k1_INTC_s117
                                 + 1.06513 w_k1_KO_s117 + 1.49349 w_k1_MAT_s117
                                 + 1.02736 w_k1_MSFT_s117
                                 + 1.15457 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.10867 w_k1_ADM_s117
                                 + 0.880227 w_k1_AXP_s117
                                 + 1.22792 w_k1_DIS_s117
                                 + 0.899012 w_k1_FDX_s117
                                 + 1.54734 w_k1_IBM_s117
                                 + 0.697725 w_k1_INTC_s117
                                 + 1.28729 w_k1_KO_s117 + 1.40909 w_k1_MAT_s117
                                 + 0.916435 w_k1_MSFT_s117
                                 + 1.1411 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.979743 w_k1_ADM_s117
                                 + 0.771574 w_k1_AXP_s117
                                 + 0.780464 w_k1_DIS_s117
                                 + 0.934684 w_k1_FDX_s117
                                 + 0.719902 w_k1_IBM_s117
                                 + 0.649795 w_k1_INTC_s117
                                 + 0.64292 w_k1_KO_s117
                                 + 0.671208 w_k1_MAT_s117
                                 + 0.825139 w_k1_MSFT_s117
                                 + 0.8703 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.30904 w_k1_ADM_s117
                                 + 1.15701 w_k1_AXP_s117
                                 + 1.56733 w_k1_DIS_s117
                                 + 1.20183 w_k1_FDX_s117
                                 + 2.04123 w_k1_IBM_s117
                                 + 1.25902 w_k1_INTC_s117 + 1.4895 w_k1_KO_s117
                                 + 2.11872 w_k1_MAT_s117
                                 + 1.25937 w_k1_MSFT_s117
                                 + 1.26865 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0148847255752396 w_k1_ADM_s000
                                 + 0.0233883347472769 w_k1_ADM_s001
                                 + 0.0258857543011757 w_k1_ADM_s002
                                 + 0.0159891928595459 w_k1_ADM_s003
                                 + 0.0230722507426466 w_k1_ADM_s004
                                 + 0.0275603118024633 w_k1_ADM_s005
                                 + 0.0249807126522018 w_k1_ADM_s006
                                 + 0.0384465959503324 w_k1_ADM_s007
                                 + 0.0197838968298073 w_k1_ADM_s010
                                 + 0.0284434475677541 w_k1_ADM_s011
                                 + 0.0169051311394263 w_k1_ADM_s012
                                 + 0.0122589292013671 w_k1_ADM_s013
                                 + 0.0585816791203087 w_k1_ADM_s014
                                 + 0.0221878757129179 w_k1_ADM_s015
                                 + 0.0198001356709777 w_k1_ADM_s016
                                 + 0.0236607317991196 w_k1_ADM_s017
                                 + 0.0382018295649633 w_k1_ADM_s100
                                 + 0.028244501550225 w_k1_ADM_s101
                                 + 0.0347003931022109 w_k1_ADM_s102
                                 + 0.0430622800942444 w_k1_ADM_s103
                                 + 0.0326701003771933 w_k1_ADM_s104
                                 + 0.103943782211159 w_k1_ADM_s105
                                 + 0.0569050256708409 w_k1_ADM_s106
                                 + 0.0797433811760408 w_k1_ADM_s107
                                 + 0.0123212678231264 w_k1_ADM_s110
                                 + 0.0120265655141954 w_k1_ADM_s111
                                 + 0.0247434532038416 w_k1_ADM_s112
                                 + 0.0452857058258998 w_k1_ADM_s113
                                 + 0.0474271341129287 w_k1_ADM_s114
                                 + 0.0127019643875075 w_k1_ADM_s115
                                 + 0.0119137511918892 w_k1_ADM_s116
                                 + 0.0171469513758914 w_k1_ADM_s117
                                 + 0.0166796074635634 w_k1_AXP_s000
                                 + 0.0182693786505363 w_k1_AXP_s001
                                 + 0.0265769876192078 w_k1_AXP_s002
                                 + 0.0173394694597301 w_k1_AXP_s003
                                 + 0.0242325001037987 w_k1_AXP_s004
                                 + 0.0259963885557072 w_k1_AXP_s005
                                 + 0.0267344412886652 w_k1_AXP_s006
                                 + 0.036020892764741 w_k1_AXP_s007
                                 + 0.020084385529388 w_k1_AXP_s010
                                 + 0.0302805182169142 w_k1_AXP_s011
                                 + 0.0151119809685873 w_k1_AXP_s012
                                 + 0.0138424723719516 w_k1_AXP_s013
                                 + 0.0675328230210051 w_k1_AXP_s014
                                 + 0.0228684240014786 w_k1_AXP_s015
                                 + 0.0208634847859511 w_k1_AXP_s016
                                 + 0.0235573533747045 w_k1_AXP_s017
                                 + 0.0356180390753134 w_k1_AXP_s100
                                 + 0.0268246108663576 w_k1_AXP_s101
                                 + 0.0370741190392775 w_k1_AXP_s102
                                 + 0.0428483235627914 w_k1_AXP_s103
                                 + 0.0316027635234054 w_k1_AXP_s104
                                 + 0.0963207050232473 w_k1_AXP_s105
                                 + 0.0565494498156467 w_k1_AXP_s106
                                 + 0.084625020540447 w_k1_AXP_s107
                                 + 0.0115914934941942 w_k1_AXP_s110
                                 + 0.0123092634570226 w_k1_AXP_s111
                                 + 0.0214138529641916 w_k1_AXP_s112
                                 + 0.0511169363401896 w_k1_AXP_s113
                                 + 0.0488459940016373 w_k1_AXP_s114
                                 + 0.0130818421347483 w_k1_AXP_s115
                                 + 0.0114086257576563 w_k1_AXP_s116
                                 + 0.0173692826571356 w_k1_AXP_s117
                                 + 0.0165765533125541 w_k1_DIS_s000
                                 + 0.0201031494858796 w_k1_DIS_s001
                                 + 0.0253718744992839 w_k1_DIS_s002
                                 + 0.0160552233410961 w_k1_DIS_s003
                                 + 0.0249483238001026 w_k1_DIS_s004
                                 + 0.0261621898979557 w_k1_DIS_s005
                                 + 0.0267856592493117 w_k1_DIS_s006
                                 + 0.0389940087407067 w_k1_DIS_s007
                                 + 0.0207114663133219 w_k1_DIS_s010
                                 + 0.0307655841905212 w_k1_DIS_s011
                                 + 0.0180729596787524 w_k1_DIS_s012
                                 + 0.0134750634519495 w_k1_DIS_s013
                                 + 0.064009940212385 w_k1_DIS_s014
                                 + 0.0230232043333384 w_k1_DIS_s015
                                 + 0.0233746363782675 w_k1_DIS_s016
                                 + 0.0232205428280441 w_k1_DIS_s017
                                 + 0.0343210055378909 w_k1_DIS_s100
                                 + 0.0304722749428869 w_k1_DIS_s101
                                 + 0.0395679953045005 w_k1_DIS_s102
                                 + 0.0398471652875225 w_k1_DIS_s103
                                 + 0.0367757910051185 w_k1_DIS_s104
                                 + 0.10089608144478 w_k1_DIS_s105
                                 + 0.0586370977500182 w_k1_DIS_s106
                                 + 0.0901371303630721 w_k1_DIS_s107
                                 + 0.0123579864826076 w_k1_DIS_s110
                                 + 0.01264753512613 w_k1_DIS_s111
                                 + 0.0229739585833095 w_k1_DIS_s112
                                 + 0.0533458115215033 w_k1_DIS_s113
                                 + 0.054448822506461 w_k1_DIS_s114
                                 + 0.0122864740613177 w_k1_DIS_s115
                                 + 0.0120793537682418 w_k1_DIS_s116
                                 + 0.0185204203606743 w_k1_DIS_s117
                                 + 0.0156578422438249 w_k1_FDX_s000
                                 + 0.0188556024742341 w_k1_FDX_s001
                                 + 0.0245380984693802 w_k1_FDX_s002
                                 + 0.0158795584874256 w_k1_FDX_s003
                                 + 0.0230050136291647 w_k1_FDX_s004
                                 + 0.0261293967047492 w_k1_FDX_s005
                                 + 0.0265785417831589 w_k1_FDX_s006
                                 + 0.0374723874220625 w_k1_FDX_s007
                                 + 0.0194962403248147 w_k1_FDX_s010
                                 + 0.03121236529417 w_k1_FDX_s011
                                 + 0.0169558080306103 w_k1_FDX_s012
                                 + 0.0130983148596266 w_k1_FDX_s013
                                 + 0.0631963697811403 w_k1_FDX_s014
                                 + 0.024449662842817 w_k1_FDX_s015
                                 + 0.0212725524949511 w_k1_FDX_s016
                                 + 0.0238124841071302 w_k1_FDX_s017
                                 + 0.0363788923450588 w_k1_FDX_s100
                                 + 0.0289776580466474 w_k1_FDX_s101
                                 + 0.0362780306084424 w_k1_FDX_s102
                                 + 0.0392040399263547 w_k1_FDX_s103
                                 + 0.0345015600633038 w_k1_FDX_s104
                                 + 0.100620917954539 w_k1_FDX_s105
                                 + 0.0584863541005019 w_k1_FDX_s106
                                 + 0.0872023851412035 w_k1_FDX_s107
                                 + 0.0133589938349898 w_k1_FDX_s110
                                 + 0.0120704997026519 w_k1_FDX_s111
                                 + 0.0233867343330937 w_k1_FDX_s112
                                 + 0.0521300884787838 w_k1_FDX_s113
                                 + 0.0499423113535551 w_k1_FDX_s114
                                 + 0.0121994331116628 w_k1_FDX_s115
                                 + 0.0114179891199821 w_k1_FDX_s116
                                 + 0.0179982002102767 w_k1_FDX_s117
                                 + 0.0165913360444683 w_k1_IBM_s000
                                 + 0.0193949622723421 w_k1_IBM_s001
                                 + 0.0259978214187853 w_k1_IBM_s002
                                 + 0.0176795032168612 w_k1_IBM_s003
                                 + 0.0224280366972047 w_k1_IBM_s004
                                 + 0.0276789124975933 w_k1_IBM_s005
                                 + 0.0282010305855943 w_k1_IBM_s006
                                 + 0.035569128350541 w_k1_IBM_s007
                                 + 0.0206187591089415 w_k1_IBM_s010
                                 + 0.030786950078064 w_k1_IBM_s011
                                 + 0.0169601575005559 w_k1_IBM_s012
                                 + 0.0134752775544023 w_k1_IBM_s013
                                 + 0.0630150346492012 w_k1_IBM_s014
                                 + 0.0233822580791656 w_k1_IBM_s015
                                 + 0.0209690704799084 w_k1_IBM_s016
                                 + 0.023920105485652 w_k1_IBM_s017
                                 + 0.036061754635635 w_k1_IBM_s100
                                 + 0.0293078330116233 w_k1_IBM_s101
                                 + 0.0383316607325282 w_k1_IBM_s102
                                 + 0.0406191429399303 w_k1_IBM_s103
                                 + 0.0350123904365255 w_k1_IBM_s104
                                 + 0.102358495486576 w_k1_IBM_s105
                                 + 0.0590647041766044 w_k1_IBM_s106
                                 + 0.0883603849203419 w_k1_IBM_s107
                                 + 0.0124532276072588 w_k1_IBM_s110
                                 + 0.0118398747917334 w_k1_IBM_s111
                                 + 0.024013949189219 w_k1_IBM_s112
                                 + 0.0517365681443615 w_k1_IBM_s113
                                 + 0.0511618073671295 w_k1_IBM_s114
                                 + 0.01215154128124 w_k1_IBM_s115
                                 + 0.0111422479485056 w_k1_IBM_s116
                                 + 0.0183751922652419 w_k1_IBM_s117
                                 + 0.0162448313169267 w_k1_INTC_s000
                                 + 0.0198536624817764 w_k1_INTC_s001
                                 + 0.0251416611953839 w_k1_INTC_s002
                                 + 0.0147686755638774 w_k1_INTC_s003
                                 + 0.0262231286956143 w_k1_INTC_s004
                                 + 0.027478526109212 w_k1_INTC_s005
                                 + 0.0253013665391177 w_k1_INTC_s006
                                 + 0.0349977225132528 w_k1_INTC_s007
                                 + 0.0210167230368733 w_k1_INTC_s010
                                 + 0.0279088738366877 w_k1_INTC_s011
                                 + 0.0175138270547805 w_k1_INTC_s012
                                 + 0.0149427842329605 w_k1_INTC_s013
                                 + 0.0536010311858631 w_k1_INTC_s014
                                 + 0.0253433634004883 w_k1_INTC_s015
                                 + 0.0211264331193951 w_k1_INTC_s016
                                 + 0.0247056114498169 w_k1_INTC_s017
                                 + 0.0359025938130168 w_k1_INTC_s100
                                 + 0.0277010841120581 w_k1_INTC_s101
                                 + 0.0425901805099873 w_k1_INTC_s102
                                 + 0.0407771976512143 w_k1_INTC_s103
                                 + 0.0292533317303545 w_k1_INTC_s104
                                 + 0.104543631581995 w_k1_INTC_s105
                                 + 0.0604112697137833 w_k1_INTC_s106
                                 + 0.0948436785913994 w_k1_INTC_s107
                                 + 0.0131249580566241 w_k1_INTC_s110
                                 + 0.011481493739957 w_k1_INTC_s111
                                 + 0.0244841633572205 w_k1_INTC_s112
                                 + 0.0461865600535455 w_k1_INTC_s113
                                 + 0.0575094239309343 w_k1_INTC_s114
                                 + 0.0130309623092428 w_k1_INTC_s115
                                 + 0.0118039637863415 w_k1_INTC_s116
                                 + 0.0153680653839295 w_k1_INTC_s117
                                 + 0.0161642728584271 w_k1_KO_s000
                                 + 0.0174958526668538 w_k1_KO_s001
                                 + 0.0261739615726063 w_k1_KO_s002
                                 + 0.0156383647706719 w_k1_KO_s003
                                 + 0.0261545319197231 w_k1_KO_s004
                                 + 0.025265989891006 w_k1_KO_s005
                                 + 0.0243078936811597 w_k1_KO_s006
                                 + 0.0375708061824562 w_k1_KO_s007
                                 + 0.0189736772102194 w_k1_KO_s010
                                 + 0.0320201618705647 w_k1_KO_s011
                                 + 0.0186478855311657 w_k1_KO_s012
                                 + 0.0121135794973666 w_k1_KO_s013
                                 + 0.0586735814973818 w_k1_KO_s014
                                 + 0.0230520625807845 w_k1_KO_s015
                                 + 0.0231330619426426 w_k1_KO_s016
                                 + 0.026230822079014 w_k1_KO_s017
                                 + 0.0357862264780695 w_k1_KO_s100
                                 + 0.0266266915636191 w_k1_KO_s101
                                 + 0.0368508374844915 w_k1_KO_s102
                                 + 0.0400480638630191 w_k1_KO_s103
                                 + 0.0338933957812977 w_k1_KO_s104
                                 + 0.0945570149156885 w_k1_KO_s105
                                 + 0.0644165346719284 w_k1_KO_s106
                                 + 0.0855392298270505 w_k1_KO_s107
                                 + 0.0117444573510564 w_k1_KO_s110
                                 + 0.0115408785460635 w_k1_KO_s111
                                 + 0.024117431915453 w_k1_KO_s112
                                 + 0.0467142649233623 w_k1_KO_s113
                                 + 0.050538386075943 w_k1_KO_s114
                                 + 0.0122343113087016 w_k1_KO_s115
                                 + 0.0119410576475291 w_k1_KO_s116
                                 + 0.0157492694524516 w_k1_KO_s117
                                 + 0.0142012026717411 w_k1_MAT_s000
                                 + 0.0169427668649886 w_k1_MAT_s001
                                 + 0.0278657647900207 w_k1_MAT_s002
                                 + 0.0167229701243824 w_k1_MAT_s003
                                 + 0.0195459032020722 w_k1_MAT_s004
                                 + 0.0256719735024255 w_k1_MAT_s005
                                 + 0.0285277966693842 w_k1_MAT_s006
                                 + 0.0351598471501512 w_k1_MAT_s007
                                 + 0.0175824173815175 w_k1_MAT_s010
                                 + 0.0318809133557261 w_k1_MAT_s011
                                 + 0.016354475818875 w_k1_MAT_s012
                                 + 0.0172321709411405 w_k1_MAT_s013
                                 + 0.0642590570062287 w_k1_MAT_s014
                                 + 0.023811800231887 w_k1_MAT_s015
                                 + 0.0221379522048629 w_k1_MAT_s016
                                 + 0.0239192203202671 w_k1_MAT_s017
                                 + 0.0417909576932734 w_k1_MAT_s100
                                 + 0.0262457705411865 w_k1_MAT_s101
                                 + 0.0365315464762236 w_k1_MAT_s102
                                 + 0.0417495945732399 w_k1_MAT_s103
                                 + 0.0369541493267912 w_k1_MAT_s104
                                 + 0.0985485860378371 w_k1_MAT_s105
                                 + 0.0689809549452834 w_k1_MAT_s106
                                 + 0.0760132760024947 w_k1_MAT_s107
                                 + 0.0160232273312621 w_k1_MAT_s110
                                 + 0.010168619101931 w_k1_MAT_s111
                                 + 0.0249718243989533 w_k1_MAT_s112
                                 + 0.047973281549272 w_k1_MAT_s113
                                 + 0.0509490129382056 w_k1_MAT_s114
                                 + 0.0140883081657725 w_k1_MAT_s115
                                 + 0.0118118607747563 w_k1_MAT_s116
                                 + 0.0199012034706072 w_k1_MAT_s117
                                 + 0.015180992485821 w_k1_MSFT_s000
                                 + 0.0186624312052856 w_k1_MSFT_s001
                                 + 0.0226057971605777 w_k1_MSFT_s002
                                 + 0.0160605013465485 w_k1_MSFT_s003
                                 + 0.0231530200140168 w_k1_MSFT_s004
                                 + 0.0259482464851308 w_k1_MSFT_s005
                                 + 0.0256801002378116 w_k1_MSFT_s006
                                 + 0.0349514159549572 w_k1_MSFT_s007
                                 + 0.0218472609901641 w_k1_MSFT_s010
                                 + 0.0308311052205738 w_k1_MSFT_s011
                                 + 0.0170538097947984 w_k1_MSFT_s012
                                 + 0.0127981171227942 w_k1_MSFT_s013
                                 + 0.0582331611598129 w_k1_MSFT_s014
                                 + 0.0270958520397316 w_k1_MSFT_s015
                                 + 0.0199443991730014 w_k1_MSFT_s016
                                 + 0.0226377797742055 w_k1_MSFT_s017
                                 + 0.0364659402731414 w_k1_MSFT_s100
                                 + 0.0273402994376459 w_k1_MSFT_s101
                                 + 0.0407634410745574 w_k1_MSFT_s102
                                 + 0.0383799008320475 w_k1_MSFT_s103
                                 + 0.0354067091140206 w_k1_MSFT_s104
                                 + 0.103309944208606 w_k1_MSFT_s105
                                 + 0.0591064958161246 w_k1_MSFT_s106
                                 + 0.095838586083106 w_k1_MSFT_s107
                                 + 0.0140673964593797 w_k1_MSFT_s110
                                 + 0.0121307393910059 w_k1_MSFT_s111
                                 + 0.0227259370841801 w_k1_MSFT_s112
                                 + 0.0504946372261829 w_k1_MSFT_s113
                                 + 0.0522744320191111 w_k1_MSFT_s114
                                 + 0.0138214262882476 w_k1_MSFT_s115
                                 + 0.0118271229269243 w_k1_MSFT_s116
                                 + 0.0166165100221005 w_k1_MSFT_s117
                                 + 0.0158498398806645 w_k1_PG_s000
                                 + 0.0195945814199923 w_k1_PG_s001
                                 + 0.0245242881399738 w_k1_PG_s002
                                 + 0.0149795099597919 w_k1_PG_s003
                                 + 0.0232470055691954 w_k1_PG_s004
                                 + 0.0259062779049154 w_k1_PG_s005
                                 + 0.0254238923830139 w_k1_PG_s006
                                 + 0.0393373042354282 w_k1_PG_s007
                                 + 0.0198336248301265 w_k1_PG_s010
                                 + 0.0294470226358276 w_k1_PG_s011
                                 + 0.0168891781902404 w_k1_PG_s012
                                 + 0.0117402898097151 w_k1_PG_s013
                                 + 0.0584304339168827 w_k1_PG_s014
                                 + 0.0236481684897275 w_k1_PG_s015
                                 + 0.0224555364113269 w_k1_PG_s016
                                 + 0.0241981868168513 w_k1_PG_s017
                                 + 0.0368127436926243 w_k1_PG_s100
                                 + 0.0272177680976596 w_k1_PG_s101
                                 + 0.036304875059413 w_k1_PG_s102
                                 + 0.0391322613644724 w_k1_PG_s103
                                 + 0.035676873062615 w_k1_PG_s104
                                 + 0.103411230384128 w_k1_PG_s105
                                 + 0.0560002566338958 w_k1_PG_s106
                                 + 0.0899878110484127 w_k1_PG_s107
                                 + 0.0133019792572617 w_k1_PG_s110
                                 + 0.0112029132212172 w_k1_PG_s111
                                 + 0.0245769699498808 w_k1_PG_s112
                                 + 0.04620889405275 w_k1_PG_s113
                                 + 0.0525737395670263 w_k1_PG_s114
                                 + 0.0131383038492281 w_k1_PG_s115
                                 + 0.011552083058369 w_k1_PG_s116
                                 + 0.0175346915519647 w_k1_PG_s117 >= 1.219
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

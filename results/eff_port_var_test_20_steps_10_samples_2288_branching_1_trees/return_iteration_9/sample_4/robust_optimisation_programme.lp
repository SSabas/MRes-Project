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
 w_k1_ADM_s0_return_constraint:  1.16088 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.932055 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.939739 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.07539 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.01236 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.12034 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.45311 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.851113 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.09808 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.18217 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.07701 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.781688 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.27711 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.79202 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.1942 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.80491 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.861372 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.10959 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.803214 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.37015 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.931809 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.07557 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.22495 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.3684 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.045 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.774548 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.16919 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.699449 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.907687 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.866653 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.09319 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.730354 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.888539 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.48366 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.860601 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.950619 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.10442 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.551932 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.25042 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.911161 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.875397 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.23035 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.894988 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.20102 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.59417 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.874702 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.15265 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.35595 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.860777 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.630945 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.23584 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.919947 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.08653 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.842976 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.707895 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.1748 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.842127 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.53107 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.30227 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.09447 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.25321 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.78484 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.10447 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.754374 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.34772 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.827063 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.08128 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.00298 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.2798 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.725124 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.34998 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.64529 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.905903 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.07654 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.16827 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.576325 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.30427 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.882787 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.907995 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.30711 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.888639 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.15824 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.72286 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.875392 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.833183 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.07511 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.851153 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.588772 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.39499 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.841176 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.06725 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.620434 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.570466 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.43908 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.827254 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.75144 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.27164 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.09559 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.42981 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 2.05656 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.987005 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.643477 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.21157 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.669609 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.12527 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.02089 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.50684 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.725129 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.57056 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 2.07205 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.922683 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.09073 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.12261 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.510776 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.19956 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.89669 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.955025 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.10031 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.00083 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.07455 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.55328 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.942167 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.917632 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.15845 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.05442 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.693513 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.19172 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.816563 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.1795 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.848491 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.658977 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.14296 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.841298 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.57242 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.08033 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.04622 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.28475 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.61953 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.09753 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.763552 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.16132 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.668151 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.992015 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.967464 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.37162 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.771652 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.18428 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.67462 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.979609 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.07215 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.14882 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.608735 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.18869 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.945273 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.80685 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.38174 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.815865 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.33877 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.07006 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.858573 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.09064 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.08068 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.529979 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.773989 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.17024 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.42301 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.70873 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.593563 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.9222 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.23792 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.901407 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.954961 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.60294 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.0951 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.81663 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.29921 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.747815 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.774246 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.26438 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.31612 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.50476 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.986985 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.874348 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.821447 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.93995 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.11121 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.840859 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.33239 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.892645 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.891696 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.10645 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.932826 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.976156 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.18452 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.972622 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.138 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.11197 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.00162 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.977607 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.990159 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.916833 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.750414 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.07242 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.980443 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.02329 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.886679 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.713569 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.02557 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.97463 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.39893 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.930812 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.872756 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.24527 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.68147 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.00814 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.799396 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.0942 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.828427 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.1191 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.15574 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.17538 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.77633 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.897673 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.32724 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.03492 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.955553 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.07139 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.672162 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.15721 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.922354 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.969169 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.29272 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.854456 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.22939 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.18972 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.81146 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.927968 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.01567 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.675669 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.671179 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.14733 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.95078 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.00981 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.66556 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.802709 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.37816 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.934322 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.16226 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.31971 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.05797 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.11705 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.61277 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.863148 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.783665 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.16227 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.930231 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.22949 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.07214 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.07509 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.774446 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.52474 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.35782 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.863596 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.09047 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.934996 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.623534 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.14289 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.838526 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.952199 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.4292 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.891123 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.49546 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.40361 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.663217 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.08307 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.2197 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.477725 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.658047 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.940004 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.05528 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.01618 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.680118 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.919636 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.54014 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.867932 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.22227 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.81839 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.2126 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.918912 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.52945 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.809173 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.796964 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.48134 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.1239 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.35839 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.793998 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.935768 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.800197 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.84354 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.35221 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.864255 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.45143 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.04602 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.684111 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.1659 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.911998 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.994104 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.12366 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.01756 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.10436 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.2404 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.975896 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.922142 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.02933 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.929907 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.696539 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.1094 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.926754 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.08186 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.768066 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.697521 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.21961 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.892109 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.30965 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.02164 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.00046 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.29955 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.48148 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.965588 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.749933 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.981207 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.693745 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.944809 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.956338 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.3233 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.828592 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.20696 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.56081 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.938354 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.02967 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.995514 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.612262 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.10696 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.93731 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.928036 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.09779 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.967328 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.12868 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.27344 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.02493 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.925456 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.01352 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.951444 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.81442 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.21144 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.02653 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.18623 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.825744 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.794044 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.26335 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.996524 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.31114 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.17185 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.04889 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.06575 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.27951 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.911344 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.788954 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.1485 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.863856 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.00658 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.93147 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.16346 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.781342 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.14643 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.35641 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.884488 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.06111 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.01043 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.683954 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.328000909929226 z_k1_s0000
                                 - 0.570999723718778 z_k1_s0001
                                 - 0.465999319192686 z_k1_s0002
                                 - 0.480999376982127 z_k1_s0003
                                 - 0.394999753122463 z_k1_s0004
                                 - 0.12200002785425 z_k1_s0005
                                 - 0.172999959038427 z_k1_s0006
                                 - 0.116999920157795 z_k1_s0007
                                 - 0.687999134381308 z_k1_s0010
                                 - 0.362999712935316 z_k1_s0011
                                 - 0.332999541880527 z_k1_s0012
                                 - 0.557000669449765 z_k1_s0013
                                 - 0.713000212176022 z_k1_s0014
                                 - 0.192999865923717 z_k1_s0015
                                 - 0.262999890951933 z_k1_s0016
                                 - 0.630997574848455 z_k1_s0017
                                 - 0.113000068573481 z_k1_s0020
                                 - 0.127000059281693 z_k1_s0021
                                 - 0.165000129775449 z_k1_s0022
                                 - 0.263000232232999 z_k1_s0023
                                 - 0.0860000625647774 z_k1_s0024
                                 - 0.321000480881421 z_k1_s0025
                                 - 0.275999912533354 z_k1_s0026
                                 - 0.32400040710458 z_k1_s0027
                                 - 0.279999943249325 z_k1_s0030
                                 - 0.276999993750955 z_k1_s0031
                                 - 0.2820001120483 z_k1_s0032
                                 - 0.624000428864377 z_k1_s0033
                                 - 0.462000726351632 z_k1_s0034
                                 - 0.487001317838358 z_k1_s0035
                                 - 0.437999308164121 z_k1_s0036
                                 - 0.186999990299379 z_k1_s0037
                                 - 0.0970000003406233 z_k1_s0040
                                 - 0.134000464289244 z_k1_s0041
                                 - 0.112999983831855 z_k1_s0042
                                 - 0.192999499587897 z_k1_s0043
                                 - 0.346000379448107 z_k1_s0044
                                 - 0.169999489844242 z_k1_s0045
                                 - 0.142999885927896 z_k1_s0046
                                 - 0.142999885927896 z_k1_s0047
                                 - 0.257000701594599 z_k1_s0050
                                 - 0.871001378890263 z_k1_s0051
                                 - 0.618003491044764 z_k1_s0052
                                 - 0.375001844288446 z_k1_s0053
                                 - 0.298000898808789 z_k1_s0054
                                 - 0.397999840814221 z_k1_s0055
                                 - 0.304000968169528 z_k1_s0056
                                 - 0.200000540819127 z_k1_s0057
                                 - 0.176000307436624 z_k1_s0060
                                 - 0.378999387685423 z_k1_s0061
                                 - 0.572999362464029 z_k1_s0062
                                 - 0.535999479882666 z_k1_s0063
                                 - 0.549999140237452 z_k1_s0064
                                 - 0.455001705044543 z_k1_s0065
                                 - 0.492001587625906 z_k1_s0066
                                 - 0.480000838463519 z_k1_s0067
                                 - 0.23500051814504 z_k1_s0070
                                 - 0.26200038064811 z_k1_s0071
                                 - 0.651001829428983 z_k1_s0072
                                 - 0.969001122778111 z_k1_s0073
                                 - 0.355000695338073 z_k1_s0074
                                 - 0.45999987020497 z_k1_s0075
                                 - 0.247000498524267 z_k1_s0076
                                 - 0.555000492792109 z_k1_s0077
                                 - 0.1290002100763 z_k1_s0100
                                 - 0.193000525450953 z_k1_s0101
                                 - 0.134000257114984 z_k1_s0102
                                 - 0.488000075127706 z_k1_s0103
                                 - 0.254000310841529 z_k1_s0104
                                 - 0.125000279965539 z_k1_s0105
                                 - 0.281000421490173 z_k1_s0106
                                 - 0.188000836812889 z_k1_s0107
                                 - 0.101999720015896 z_k1_s0110
                                 - 0.0920000654203885 z_k1_s0111
                                 - 0.195000207266077 z_k1_s0112
                                 - 0.0850000256133839 z_k1_s0113
                                 - 0.0560000271558819 z_k1_s0114
                                 - 0.0740000462881352 z_k1_s0115
                                 - 0.115999605429802 z_k1_s0116
                                 - 0.250999748921702 z_k1_s0117
                                 - 0.0970001064241805 z_k1_s0120
                                 - 0.44400117664349 z_k1_s0121
                                 - 0.219999450857993 z_k1_s0122
                                 - 0.12500005589704 z_k1_s0123
                                 - 0.405000828627206 z_k1_s0124
                                 - 0.353001074606369 z_k1_s0125
                                 - 0.157000150294781 z_k1_s0126
                                 - 0.32900020505708 z_k1_s0127
                                 - 0.189999917037127 z_k1_s0130
                                 - 0.136000230711045 z_k1_s0131
                                 - 0.0920000253338253 z_k1_s0132
                                 - 0.052000056836548 z_k1_s0133
                                 - 0.0550000789213657 z_k1_s0134
                                 - 0.0889996476486894 z_k1_s0135
                                 - 0.154999955714519 z_k1_s0136
                                 - 0.119999994391911 z_k1_s0137
                                 - 0.280000583881612 z_k1_s0140
                                 - 0.327999976431027 z_k1_s0141
                                 - 0.448999590235459 z_k1_s0142
                                 - 0.478000720654367 z_k1_s0143
                                 - 0.281000810310305 z_k1_s0144
                                 - 0.350000729852015 z_k1_s0145
                                 - 0.286000734450839 z_k1_s0146
                                 - 0.567999959185437 z_k1_s0147
                                 - 0.21900008078805 z_k1_s0150
                                 - 0.117000070125137 z_k1_s0151
                                 - 0.241999440843076 z_k1_s0152
                                 - 0.0649999764694877 z_k1_s0153
                                 - 0.175000693725941 z_k1_s0154
                                 - 0.308999593960949 z_k1_s0155
                                 - 0.120000053897567 z_k1_s0156
                                 - 0.159000405339451 z_k1_s0157
                                 - 0.229999665884908 z_k1_s0160
                                 - 0.216000699701781 z_k1_s0161
                                 - 0.286001080623832 z_k1_s0162
                                 - 0.15300017187172 z_k1_s0163
                                 - 0.240999778601291 z_k1_s0164
                                 - 0.194000474269014 z_k1_s0165
                                 - 0.227999813573032 z_k1_s0166
                                 - 0.301999899118835 z_k1_s0167
                                 - 0.356000311885516 z_k1_s0170
                                 - 0.170999566034898 z_k1_s0171
                                 - 0.216999593979679 z_k1_s0172
                                 - 0.237999439406967 z_k1_s0173
                                 - 0.817999110481612 z_k1_s0174
                                 - 0.402999874303908 z_k1_s0175
                                 - 0.225999527734231 z_k1_s0176
                                 - 0.319999477269427 z_k1_s0177
                                 - 0.517999005108969 z_k1_s1000
                                 - 0.176000164178198 z_k1_s1001
                                 - 0.286000310852094 z_k1_s1002
                                 - 0.363001153774185 z_k1_s1003
                                 - 0.405999124950226 z_k1_s1004
                                 - 0.543001842422599 z_k1_s1005
                                 - 0.36799960623587 z_k1_s1006
                                 - 0.865000176003334 z_k1_s1007
                                 - 0.25400037364008 z_k1_s1010
                                 - 0.439999667446175 z_k1_s1011
                                 - 0.205000238231089 z_k1_s1012
                                 - 0.253000339907206 z_k1_s1013
                                 - 0.184000136841468 z_k1_s1014
                                 - 0.341999396628193 z_k1_s1015
                                 - 0.85000135790985 z_k1_s1016
                                 - 0.507000410046912 z_k1_s1017
                                 - 0.534000847731072 z_k1_s1020
                                 - 0.493001040424195 z_k1_s1021
                                 - 0.299000340619733 z_k1_s1022
                                 - 0.229000274175993 z_k1_s1023
                                 - 0.148000035600974 z_k1_s1024
                                 - 0.414999541765831 z_k1_s1025
                                 - 0.251000312538324 z_k1_s1026
                                 - 0.690000786045538 z_k1_s1027
                                 - 1.45599732294925 z_k1_s1030
                                 - 1.19900535026752 z_k1_s1031
                                 - 1.89500649251456 z_k1_s1032
                                 - 1.63000011301314 z_k1_s1033
                                 - 0.826000831667441 z_k1_s1034
                                 - 0.792000712535779 z_k1_s1035
                                 - 0.794000307037291 z_k1_s1036
                                 - 1.42600340542658 z_k1_s1037
                                 - 0.707999558385281 z_k1_s1040
                                 - 0.216999937381064 z_k1_s1041
                                 - 0.805000697461782 z_k1_s1042
                                 - 0.337999836938724 z_k1_s1043
                                 - 0.539999011819443 z_k1_s1044
                                 - 0.239999902408557 z_k1_s1045
                                 - 0.623999285102362 z_k1_s1046
                                 - 0.371000054099844 z_k1_s1047
                                 - 0.367000749457671 z_k1_s1050
                                 - 0.12700013738452 z_k1_s1051
                                 - 0.24899994386331 z_k1_s1052
                                 - 0.230999963782832 z_k1_s1053
                                 - 0.496999874205481 z_k1_s1054
                                 - 0.259000079074705 z_k1_s1055
                                 - 0.348001019156085 z_k1_s1056
                                 - 0.554999605231442 z_k1_s1057
                                 - 0.508003670745346 z_k1_s1060
                                 - 0.718001055604847 z_k1_s1061
                                 - 0.399001290756595 z_k1_s1062
                                 - 0.471001701070483 z_k1_s1063
                                 - 0.788001788897383 z_k1_s1064
                                 - 0.711004834290081 z_k1_s1065
                                 - 0.77600276376566 z_k1_s1066
                                 - 0.444001487015048 z_k1_s1067
                                 - 0.376000327980681 z_k1_s1070
                                 - 0.709002432172494 z_k1_s1071
                                 - 0.636000041660671 z_k1_s1072
                                 - 0.47200039746723 z_k1_s1073
                                 - 1.27200008332134 z_k1_s1074
                                 - 0.891000958937825 z_k1_s1075
                                 - 0.473000558688007 z_k1_s1076
                                 - 1.87000002769564 z_k1_s1077
                                 - 0.323999283837248 z_k1_s1100
                                 - 0.262999404059266 z_k1_s1101
                                 - 0.391998373754669 z_k1_s1102
                                 - 0.309998496561365 z_k1_s1103
                                 - 0.232999610637121 z_k1_s1104
                                 - 0.132999904730471 z_k1_s1105
                                 - 0.443000058189738 z_k1_s1106
                                 - 0.268999741463215 z_k1_s1107
                                 - 0.642002610202626 z_k1_s1110
                                 - 0.655002584723806 z_k1_s1111
                                 - 0.914001303414433 z_k1_s1112
                                 - 0.24299994927313 z_k1_s1113
                                 - 0.903002872359183 z_k1_s1114
                                 - 0.395000076898004 z_k1_s1115
                                 - 0.635002237075554 z_k1_s1116
                                 - 0.642002610202626 z_k1_s1117
                                 - 0.237999608334049 z_k1_s1120
                                 - 0.120999760610251 z_k1_s1121
                                 - 0.211999908372652 z_k1_s1122
                                 - 0.175999095197886 z_k1_s1123
                                 - 0.059999891194319 z_k1_s1124
                                 - 0.211999908372652 z_k1_s1125
                                 - 0.162999245217188 z_k1_s1126
                                 - 0.148999816314046 z_k1_s1127
                                 - 0.177000235994595 z_k1_s1130
                                 - 0.266999247044874 z_k1_s1131
                                 - 0.205999652977982 z_k1_s1132
                                 - 0.187999553368511 z_k1_s1133
                                 - 0.311998752570014 z_k1_s1134
                                 - 0.0799998479099302 z_k1_s1135
                                 - 0.175999486962197 z_k1_s1136
                                 - 0.0809998534437897 z_k1_s1137
                                 - 0.206000188925336 z_k1_s1140
                                 - 0.651998842276197 z_k1_s1141
                                 - 0.332000052508778 z_k1_s1142
                                 - 0.712000969982733 z_k1_s1143
                                 - 0.551999482098894 z_k1_s1144
                                 - 0.530002050740038 z_k1_s1145
                                 - 0.90800206009039 z_k1_s1146
                                 - 0.293999960761382 z_k1_s1147
                                 - 0.870998841992366 z_k1_s1150
                                 - 0.707998113476042 z_k1_s1151
                                 - 0.379999325161292 z_k1_s1152
                                 - 0.317999726119872 z_k1_s1153
                                 - 0.530999149006321 z_k1_s1154
                                 - 0.469999343404531 z_k1_s1155
                                 - 1.36600091597769 z_k1_s1156
                                 - 0.995000295672361 z_k1_s1157
                                 - 0.127999350598769 z_k1_s1160
                                 - 0.449997636824207 z_k1_s1161
                                 - 0.159999188248461 z_k1_s1162
                                 - 0.321998286225439 z_k1_s1163
                                 - 0.418999036032189 z_k1_s1164
                                 - 0.422998855489224 z_k1_s1165
                                 - 0.270998024161417 z_k1_s1166
                                 - 0.390997735846116 z_k1_s1167
                                 - 0.158999731609387 z_k1_s1170
                                 - 0.113999848821774 z_k1_s1171
                                 - 0.239999010467719 z_k1_s1172
                                 - 0.240999287235988 z_k1_s1173
                                 - 0.221999494632135 z_k1_s1174
                                 - 0.149999791491819 z_k1_s1175
                                 - 0.266999195219972 z_k1_s1176
                                 - 0.429000303733495 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.15543 w_k1_ADM_s000
                                 + 0.934391 w_k1_AXP_s000
                                 + 0.925497 w_k1_DIS_s000
                                 + 1.12608 w_k1_FDX_s000
                                 + 0.592654 w_k1_IBM_s000
                                 + 1.02551 w_k1_INTC_s000
                                 + 0.860675 w_k1_KO_s000
                                 + 0.719441 w_k1_MAT_s000
                                 + 1.19692 w_k1_MSFT_s000
                                 + 1.03929 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.819355 w_k1_ADM_s000
                                 + 0.74887 w_k1_AXP_s000
                                 + 0.681168 w_k1_DIS_s000
                                 + 0.801504 w_k1_FDX_s000
                                 + 0.74665 w_k1_IBM_s000
                                 + 1.06735 w_k1_INTC_s000
                                 + 0.804669 w_k1_KO_s000
                                 + 0.627284 w_k1_MAT_s000
                                 + 0.892431 w_k1_MSFT_s000
                                 + 0.849461 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.895473 w_k1_ADM_s000
                                 + 0.983712 w_k1_AXP_s000
                                 + 1.0395 w_k1_DIS_s000
                                 + 0.960019 w_k1_FDX_s000
                                 + 1.35796 w_k1_IBM_s000
                                 + 1.03388 w_k1_INTC_s000
                                 + 1.20106 w_k1_KO_s000 + 1.16907 w_k1_MAT_s000
                                 + 1.12066 w_k1_MSFT_s000
                                 + 1.01357 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.987301 w_k1_ADM_s000
                                 + 0.969258 w_k1_AXP_s000
                                 + 1.11039 w_k1_DIS_s000
                                 + 0.978297 w_k1_FDX_s000
                                 + 0.824332 w_k1_IBM_s000
                                 + 1.26609 w_k1_INTC_s000
                                 + 0.918477 w_k1_KO_s000
                                 + 0.62398 w_k1_MAT_s000
                                 + 1.12623 w_k1_MSFT_s000
                                 + 0.987739 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.40466 w_k1_ADM_s000
                                 + 1.5646 w_k1_AXP_s000 + 1.60561 w_k1_DIS_s000
                                 + 1.33283 w_k1_FDX_s000
                                 + 1.43264 w_k1_IBM_s000
                                 + 1.78531 w_k1_INTC_s000
                                 + 1.56878 w_k1_KO_s000 + 1.20502 w_k1_MAT_s000
                                 + 1.45381 w_k1_MSFT_s000
                                 + 1.37601 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.631588 w_k1_ADM_s000
                                 + 0.516012 w_k1_AXP_s000
                                 + 0.512372 w_k1_DIS_s000
                                 + 0.62909 w_k1_FDX_s000
                                 + 0.758517 w_k1_IBM_s000
                                 + 0.88458 w_k1_INTC_s000
                                 + 0.641856 w_k1_KO_s000
                                 + 0.444006 w_k1_MAT_s000
                                 + 0.728485 w_k1_MSFT_s000
                                 + 0.712226 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.00239 w_k1_ADM_s000
                                 + 1.40384 w_k1_AXP_s000
                                 + 0.978541 w_k1_DIS_s000
                                 + 0.978092 w_k1_FDX_s000
                                 + 1.28881 w_k1_IBM_s000
                                 + 1.38607 w_k1_INTC_s000
                                 + 1.01874 w_k1_KO_s000 + 1.17605 w_k1_MAT_s000
                                 + 0.966908 w_k1_MSFT_s000
                                 + 1.0653 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.24901 w_k1_ADM_s000
                                 + 1.32575 w_k1_AXP_s000
                                 + 1.14338 w_k1_DIS_s000 + 1.1949 w_k1_FDX_s000
                                 + 1.20534 w_k1_IBM_s000
                                 + 1.22791 w_k1_INTC_s000 + 1.0607 w_k1_KO_s000
                                 + 0.872096 w_k1_MAT_s000
                                 + 1.13721 w_k1_MSFT_s000
                                 + 1.02845 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.04658 w_k1_ADM_s001
                                 + 1.32561 w_k1_AXP_s001
                                 + 1.15804 w_k1_DIS_s001
                                 + 1.10448 w_k1_FDX_s001
                                 + 1.49037 w_k1_IBM_s001
                                 + 1.16142 w_k1_INTC_s001
                                 + 1.23917 w_k1_KO_s001 + 1.26526 w_k1_MAT_s001
                                 + 1.01218 w_k1_MSFT_s001
                                 + 1.04733 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.14048 w_k1_ADM_s001
                                 + 1.0157 w_k1_AXP_s001
                                 + 0.937557 w_k1_DIS_s001
                                 + 0.943826 w_k1_FDX_s001
                                 + 1.12607 w_k1_IBM_s001
                                 + 0.90622 w_k1_INTC_s001
                                 + 1.14552 w_k1_KO_s001 + 1.55597 w_k1_MAT_s001
                                 + 0.982701 w_k1_MSFT_s001
                                 + 0.943035 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.02992 w_k1_ADM_s001
                                 + 1.38549 w_k1_AXP_s001
                                 + 1.02804 w_k1_DIS_s001
                                 + 1.10211 w_k1_FDX_s001
                                 + 1.06478 w_k1_IBM_s001
                                 + 1.04021 w_k1_INTC_s001
                                 + 1.24645 w_k1_KO_s001 + 1.37599 w_k1_MAT_s001
                                 + 0.868621 w_k1_MSFT_s001
                                 + 1.01134 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.0564 w_k1_ADM_s001
                                 + 0.846729 w_k1_AXP_s001
                                 + 0.793694 w_k1_DIS_s001
                                 + 0.905681 w_k1_FDX_s001
                                 + 0.737978 w_k1_IBM_s001
                                 + 0.876182 w_k1_INTC_s001
                                 + 0.883004 w_k1_KO_s001
                                 + 0.898838 w_k1_MAT_s001
                                 + 0.85023 w_k1_MSFT_s001
                                 + 0.759043 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.964716 w_k1_ADM_s001
                                 + 0.801588 w_k1_AXP_s001
                                 + 0.711626 w_k1_DIS_s001
                                 + 0.804445 w_k1_FDX_s001
                                 + 0.921201 w_k1_IBM_s001
                                 + 0.865717 w_k1_INTC_s001
                                 + 0.874261 w_k1_KO_s001
                                 + 0.666477 w_k1_MAT_s001
                                 + 0.770527 w_k1_MSFT_s001
                                 + 0.851826 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.11263 w_k1_ADM_s001
                                 + 1.0381 w_k1_AXP_s001 + 1.07963 w_k1_DIS_s001
                                 + 1.20742 w_k1_FDX_s001
                                 + 1.02866 w_k1_IBM_s001
                                 + 1.17853 w_k1_INTC_s001
                                 + 1.11483 w_k1_KO_s001 + 1.3503 w_k1_MAT_s001
                                 + 1.0247 w_k1_MSFT_s001
                                 + 0.987227 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.30291 w_k1_ADM_s001
                                 + 1.08979 w_k1_AXP_s001
                                 + 1.76334 w_k1_DIS_s001
                                 + 1.29811 w_k1_FDX_s001
                                 + 1.15092 w_k1_IBM_s001
                                 + 1.48924 w_k1_INTC_s001
                                 + 1.53946 w_k1_KO_s001 + 1.29793 w_k1_MAT_s001
                                 + 1.33447 w_k1_MSFT_s001
                                 + 1.13566 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.14611 w_k1_ADM_s001
                                 + 0.915433 w_k1_AXP_s001
                                 + 1.05946 w_k1_DIS_s001
                                 + 1.01097 w_k1_FDX_s001
                                 + 0.895785 w_k1_IBM_s001
                                 + 0.867883 w_k1_INTC_s001
                                 + 0.907621 w_k1_KO_s001
                                 + 1.11167 w_k1_MAT_s001
                                 + 0.874451 w_k1_MSFT_s001
                                 + 0.90928 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.00885 w_k1_ADM_s002
                                 + 1.1524 w_k1_AXP_s002
                                 + 0.898954 w_k1_DIS_s002
                                 + 1.1521 w_k1_FDX_s002
                                 + 0.755163 w_k1_IBM_s002
                                 + 0.784803 w_k1_INTC_s002
                                 + 0.762748 w_k1_KO_s002
                                 + 1.14296 w_k1_MAT_s002
                                 + 0.883743 w_k1_MSFT_s002
                                 + 1.04273 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.18471 w_k1_ADM_s002
                                 + 1.51596 w_k1_AXP_s002
                                 + 1.81261 w_k1_DIS_s002
                                 + 1.30547 w_k1_FDX_s002
                                 + 1.26266 w_k1_IBM_s002
                                 + 1.1944 w_k1_INTC_s002 + 1.27849 w_k1_KO_s002
                                 + 1.10975 w_k1_MAT_s002
                                 + 1.1018 w_k1_MSFT_s002 + 1.27502 w_k1_PG_s002
                                 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.979338 w_k1_ADM_s002
                                 + 0.946089 w_k1_AXP_s002
                                 + 0.888442 w_k1_DIS_s002
                                 + 0.952612 w_k1_FDX_s002
                                 + 0.601893 w_k1_IBM_s002
                                 + 0.787381 w_k1_INTC_s002
                                 + 0.674467 w_k1_KO_s002
                                 + 0.619501 w_k1_MAT_s002
                                 + 0.86218 w_k1_MSFT_s002
                                 + 0.998962 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.21565 w_k1_ADM_s002
                                 + 1.24181 w_k1_AXP_s002
                                 + 1.24727 w_k1_DIS_s002
                                 + 1.21547 w_k1_FDX_s002
                                 + 0.938614 w_k1_IBM_s002
                                 + 0.871265 w_k1_INTC_s002
                                 + 1.01788 w_k1_KO_s002 + 1.06183 w_k1_MAT_s002
                                 + 1.07977 w_k1_MSFT_s002
                                 + 1.19377 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.878154 w_k1_ADM_s002
                                 + 1.24039 w_k1_AXP_s002
                                 + 1.32205 w_k1_DIS_s002
                                 + 1.14538 w_k1_FDX_s002
                                 + 1.63023 w_k1_IBM_s002
                                 + 0.999925 w_k1_INTC_s002
                                 + 1.04783 w_k1_KO_s002 + 1.41116 w_k1_MAT_s002
                                 + 0.930166 w_k1_MSFT_s002
                                 + 1.1605 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.885256 w_k1_ADM_s002
                                 + 0.783892 w_k1_AXP_s002
                                 + 0.860566 w_k1_DIS_s002
                                 + 0.878728 w_k1_FDX_s002
                                 + 0.733805 w_k1_IBM_s002
                                 + 0.68582 w_k1_INTC_s002
                                 + 0.722437 w_k1_KO_s002
                                 + 0.586048 w_k1_MAT_s002
                                 + 0.772736 w_k1_MSFT_s002
                                 + 0.929434 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.847535 w_k1_ADM_s002
                                 + 1.03321 w_k1_AXP_s002
                                 + 1.04282 w_k1_DIS_s002
                                 + 0.944924 w_k1_FDX_s002
                                 + 1.1458 w_k1_IBM_s002
                                 + 0.971604 w_k1_INTC_s002
                                 + 0.932822 w_k1_KO_s002
                                 + 1.40029 w_k1_MAT_s002
                                 + 0.9237 w_k1_MSFT_s002 + 1.08245 w_k1_PG_s002
                                 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.890786 w_k1_ADM_s002
                                 + 0.825908 w_k1_AXP_s002
                                 + 0.877162 w_k1_DIS_s002
                                 + 0.912107 w_k1_FDX_s002
                                 + 0.99252 w_k1_IBM_s002
                                 + 0.946449 w_k1_INTC_s002
                                 + 0.908169 w_k1_KO_s002
                                 + 0.698987 w_k1_MAT_s002
                                 + 0.821481 w_k1_MSFT_s002
                                 + 1.05959 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.29335 w_k1_ADM_s003
                                 + 0.96072 w_k1_AXP_s003
                                 + 1.08307 w_k1_DIS_s003
                                 + 1.16137 w_k1_FDX_s003
                                 + 0.687096 w_k1_IBM_s003
                                 + 1.25616 w_k1_INTC_s003
                                 + 0.813143 w_k1_KO_s003
                                 + 0.527242 w_k1_MAT_s003
                                 + 1.04127 w_k1_MSFT_s003
                                 + 0.955582 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.41871 w_k1_ADM_s003
                                 + 1.44114 w_k1_AXP_s003
                                 + 1.59201 w_k1_DIS_s003
                                 + 1.26366 w_k1_FDX_s003
                                 + 1.58482 w_k1_IBM_s003
                                 + 1.35511 w_k1_INTC_s003 + 1.3761 w_k1_KO_s003
                                 + 1.60054 w_k1_MAT_s003
                                 + 1.40858 w_k1_MSFT_s003
                                 + 1.23517 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.735455 w_k1_ADM_s003
                                 + 0.731615 w_k1_AXP_s003
                                 + 0.638982 w_k1_DIS_s003
                                 + 0.704934 w_k1_FDX_s003
                                 + 0.759916 w_k1_IBM_s003
                                 + 0.797186 w_k1_INTC_s003
                                 + 0.655028 w_k1_KO_s003
                                 + 0.548132 w_k1_MAT_s003
                                 + 0.732391 w_k1_MSFT_s003
                                 + 0.9298 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.845594 w_k1_ADM_s003
                                 + 0.796095 w_k1_AXP_s003
                                 + 0.935816 w_k1_DIS_s003
                                 + 0.943782 w_k1_FDX_s003
                                 + 0.97882 w_k1_IBM_s003
                                 + 0.890222 w_k1_INTC_s003
                                 + 0.959237 w_k1_KO_s003
                                 + 0.876611 w_k1_MAT_s003
                                 + 0.903793 w_k1_MSFT_s003
                                 + 1.01868 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.96159 w_k1_ADM_s003
                                 + 0.972374 w_k1_AXP_s003
                                 + 1.21568 w_k1_DIS_s003
                                 + 0.891427 w_k1_FDX_s003
                                 + 1.52101 w_k1_IBM_s003
                                 + 1.13491 w_k1_INTC_s003 + 1.1579 w_k1_KO_s003
                                 + 1.30105 w_k1_MAT_s003
                                 + 1.06403 w_k1_MSFT_s003
                                 + 1.09909 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.29692 w_k1_ADM_s003
                                 + 1.28417 w_k1_AXP_s003
                                 + 1.27165 w_k1_DIS_s003
                                 + 1.14202 w_k1_FDX_s003
                                 + 1.01437 w_k1_IBM_s003
                                 + 1.32543 w_k1_INTC_s003 + 1.1579 w_k1_KO_s003
                                 + 1.07217 w_k1_MAT_s003
                                 + 1.14732 w_k1_MSFT_s003
                                 + 1.21096 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.811093 w_k1_ADM_s003
                                 + 0.862639 w_k1_AXP_s003
                                 + 0.752082 w_k1_DIS_s003
                                 + 0.88372 w_k1_FDX_s003
                                 + 0.637365 w_k1_IBM_s003
                                 + 0.986975 w_k1_INTC_s003
                                 + 0.798484 w_k1_KO_s003
                                 + 0.670466 w_k1_MAT_s003
                                 + 0.923475 w_k1_MSFT_s003
                                 + 0.997703 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.863006 w_k1_ADM_s003
                                 + 0.720409 w_k1_AXP_s003
                                 + 0.857833 w_k1_DIS_s003
                                 + 0.726143 w_k1_FDX_s003
                                 + 1.11674 w_k1_IBM_s003
                                 + 0.831488 w_k1_INTC_s003
                                 + 0.852555 w_k1_KO_s003
                                 + 0.806876 w_k1_MAT_s003
                                 + 0.798051 w_k1_MSFT_s003
                                 + 0.977206 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.24146 w_k1_ADM_s004
                                 + 1.28281 w_k1_AXP_s004
                                 + 1.50431 w_k1_DIS_s004
                                 + 1.22048 w_k1_FDX_s004
                                 + 1.73566 w_k1_IBM_s004
                                 + 1.05215 w_k1_INTC_s004
                                 + 1.41917 w_k1_KO_s004 + 1.4548 w_k1_MAT_s004
                                 + 1.10372 w_k1_MSFT_s004
                                 + 1.15166 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.778125 w_k1_ADM_s004
                                 + 0.868929 w_k1_AXP_s004
                                 + 0.637696 w_k1_DIS_s004
                                 + 0.824608 w_k1_FDX_s004
                                 + 0.757783 w_k1_IBM_s004
                                 + 0.959078 w_k1_INTC_s004
                                 + 0.684185 w_k1_KO_s004
                                 + 0.664885 w_k1_MAT_s004
                                 + 0.855015 w_k1_MSFT_s004
                                 + 0.773907 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.980971 w_k1_ADM_s004
                                 + 0.972133 w_k1_AXP_s004
                                 + 1.13174 w_k1_DIS_s004
                                 + 1.10204 w_k1_FDX_s004
                                 + 0.815054 w_k1_IBM_s004
                                 + 0.964432 w_k1_INTC_s004
                                 + 1.00505 w_k1_KO_s004 + 1.09902 w_k1_MAT_s004
                                 + 1.26801 w_k1_MSFT_s004
                                 + 1.06768 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.04144 w_k1_ADM_s004
                                 + 1.05573 w_k1_AXP_s004
                                 + 1.39648 w_k1_DIS_s004
                                 + 1.11528 w_k1_FDX_s004
                                 + 1.27534 w_k1_IBM_s004
                                 + 1.13426 w_k1_INTC_s004
                                 + 1.25023 w_k1_KO_s004 + 1.27669 w_k1_MAT_s004
                                 + 1.09526 w_k1_MSFT_s004
                                 + 1.01885 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.787768 w_k1_ADM_s004
                                 + 0.954083 w_k1_AXP_s004
                                 + 1.0499 w_k1_DIS_s004
                                 + 0.853215 w_k1_FDX_s004
                                 + 1.16142 w_k1_IBM_s004
                                 + 1.00876 w_k1_INTC_s004
                                 + 1.12971 w_k1_KO_s004 + 1.63027 w_k1_MAT_s004
                                 + 1.00275 w_k1_MSFT_s004
                                 + 0.909493 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.35571 w_k1_ADM_s004
                                 + 1.5327 w_k1_AXP_s004 + 1.50433 w_k1_DIS_s004
                                 + 1.34292 w_k1_FDX_s004
                                 + 0.919648 w_k1_IBM_s004
                                 + 1.28108 w_k1_INTC_s004
                                 + 1.25488 w_k1_KO_s004 + 1.30666 w_k1_MAT_s004
                                 + 1.13263 w_k1_MSFT_s004
                                 + 1.07169 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.884596 w_k1_ADM_s004
                                 + 0.919241 w_k1_AXP_s004
                                 + 0.908832 w_k1_DIS_s004
                                 + 1.0217 w_k1_FDX_s004
                                 + 0.787954 w_k1_IBM_s004
                                 + 1.06787 w_k1_INTC_s004
                                 + 0.82348 w_k1_KO_s004
                                 + 0.702868 w_k1_MAT_s004
                                 + 1.03482 w_k1_MSFT_s004 + 1.0191 w_k1_PG_s004
                                 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.722607 w_k1_ADM_s004
                                 + 0.740187 w_k1_AXP_s004
                                 + 0.710293 w_k1_DIS_s004
                                 + 0.649411 w_k1_FDX_s004
                                 + 0.974827 w_k1_IBM_s004
                                 + 0.725962 w_k1_INTC_s004
                                 + 0.867455 w_k1_KO_s004
                                 + 1.25871 w_k1_MAT_s004
                                 + 0.758513 w_k1_MSFT_s004
                                 + 0.78049 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.854503 w_k1_ADM_s005
                                 + 0.705221 w_k1_AXP_s005
                                 + 0.51988 w_k1_DIS_s005
                                 + 0.753443 w_k1_FDX_s005
                                 + 0.585422 w_k1_IBM_s005
                                 + 0.798411 w_k1_INTC_s005
                                 + 0.56744 w_k1_KO_s005
                                 + 0.454148 w_k1_MAT_s005
                                 + 0.720389 w_k1_MSFT_s005
                                 + 0.804028 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.00967 w_k1_ADM_s005
                                 + 1.14045 w_k1_AXP_s005
                                 + 1.21673 w_k1_DIS_s005
                                 + 1.17616 w_k1_FDX_s005
                                 + 0.862176 w_k1_IBM_s005
                                 + 1.20861 w_k1_INTC_s005 + 1.1581 w_k1_KO_s005
                                 + 0.857958 w_k1_MAT_s005
                                 + 1.20658 w_k1_MSFT_s005
                                 + 1.15235 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.06206 w_k1_ADM_s005
                                 + 1.25913 w_k1_AXP_s005
                                 + 1.32292 w_k1_DIS_s005 + 1.1142 w_k1_FDX_s005
                                 + 1.55093 w_k1_IBM_s005
                                 + 1.22553 w_k1_INTC_s005
                                 + 1.33929 w_k1_KO_s005 + 1.7632 w_k1_MAT_s005
                                 + 1.24376 w_k1_MSFT_s005
                                 + 1.11822 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.826087 w_k1_ADM_s005
                                 + 0.876365 w_k1_AXP_s005
                                 + 0.821589 w_k1_DIS_s005
                                 + 0.762746 w_k1_FDX_s005
                                 + 1.28614 w_k1_IBM_s005
                                 + 0.809442 w_k1_INTC_s005
                                 + 1.07028 w_k1_KO_s005 + 1.20539 w_k1_MAT_s005
                                 + 0.797189 w_k1_MSFT_s005
                                 + 0.990924 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.87682 w_k1_ADM_s005
                                 + 1.1574 w_k1_AXP_s005
                                 + 0.880856 w_k1_DIS_s005
                                 + 0.985153 w_k1_FDX_s005
                                 + 0.91168 w_k1_IBM_s005
                                 + 0.919909 w_k1_INTC_s005
                                 + 0.928089 w_k1_KO_s005
                                 + 1.00168 w_k1_MAT_s005
                                 + 0.929125 w_k1_MSFT_s005
                                 + 0.943309 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.778449 w_k1_ADM_s005
                                 + 0.854082 w_k1_AXP_s005
                                 + 0.63134 w_k1_DIS_s005
                                 + 0.762016 w_k1_FDX_s005
                                 + 0.859163 w_k1_IBM_s005
                                 + 0.903145 w_k1_INTC_s005
                                 + 0.732966 w_k1_KO_s005
                                 + 0.705374 w_k1_MAT_s005
                                 + 0.84776 w_k1_MSFT_s005
                                 + 0.900612 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.96058 w_k1_ADM_s005
                                 + 1.30367 w_k1_AXP_s005
                                 + 0.993595 w_k1_DIS_s005
                                 + 0.975841 w_k1_FDX_s005
                                 + 1.2177 w_k1_IBM_s005
                                 + 1.09334 w_k1_INTC_s005
                                 + 1.09263 w_k1_KO_s005 + 1.42262 w_k1_MAT_s005
                                 + 0.879912 w_k1_MSFT_s005
                                 + 1.02027 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.836639 w_k1_ADM_s005
                                 + 0.766078 w_k1_AXP_s005
                                 + 0.937173 w_k1_DIS_s005
                                 + 0.968518 w_k1_FDX_s005
                                 + 0.805129 w_k1_IBM_s005
                                 + 0.97583 w_k1_INTC_s005
                                 + 0.892406 w_k1_KO_s005
                                 + 0.72532 w_k1_MAT_s005
                                 + 1.18847 w_k1_MSFT_s005
                                 + 0.888671 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.844396 w_k1_ADM_s006
                                 + 1.1991 w_k1_AXP_s006
                                 + 0.781233 w_k1_DIS_s006
                                 + 0.96352 w_k1_FDX_s006
                                 + 0.917487 w_k1_IBM_s006
                                 + 0.995393 w_k1_INTC_s006
                                 + 0.89315 w_k1_KO_s006 + 1.2385 w_k1_MAT_s006
                                 + 0.833869 w_k1_MSFT_s006
                                 + 1.02252 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.794717 w_k1_ADM_s006
                                 + 0.745172 w_k1_AXP_s006
                                 + 0.703773 w_k1_DIS_s006
                                 + 0.777122 w_k1_FDX_s006
                                 + 0.943473 w_k1_IBM_s006
                                 + 0.960957 w_k1_INTC_s006
                                 + 0.898412 w_k1_KO_s006
                                 + 0.732209 w_k1_MAT_s006
                                 + 0.873019 w_k1_MSFT_s006
                                 + 0.838962 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.917098 w_k1_ADM_s006
                                 + 0.898588 w_k1_AXP_s006
                                 + 1.06517 w_k1_DIS_s006
                                 + 0.998555 w_k1_FDX_s006
                                 + 0.768744 w_k1_IBM_s006
                                 + 1.09259 w_k1_INTC_s006
                                 + 0.927022 w_k1_KO_s006
                                 + 0.888713 w_k1_MAT_s006
                                 + 1.27756 w_k1_MSFT_s006
                                 + 0.977369 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.966264 w_k1_ADM_s006
                                 + 1.40917 w_k1_AXP_s006 + 1.2072 w_k1_DIS_s006
                                 + 1.24563 w_k1_FDX_s006
                                 + 0.849385 w_k1_IBM_s006
                                 + 1.22724 w_k1_INTC_s006
                                 + 1.06593 w_k1_KO_s006 + 1.21362 w_k1_MAT_s006
                                 + 1.07954 w_k1_MSFT_s006
                                 + 1.04265 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.986609 w_k1_ADM_s006
                                 + 1.51128 w_k1_AXP_s006
                                 + 1.67847 w_k1_DIS_s006
                                 + 1.26388 w_k1_FDX_s006
                                 + 1.57503 w_k1_IBM_s006
                                 + 1.30072 w_k1_INTC_s006
                                 + 1.49744 w_k1_KO_s006 + 2.08363 w_k1_MAT_s006
                                 + 1.43762 w_k1_MSFT_s006
                                 + 1.15787 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.771088 w_k1_ADM_s006
                                 + 0.755459 w_k1_AXP_s006
                                 + 0.665038 w_k1_DIS_s006
                                 + 0.829705 w_k1_FDX_s006
                                 + 0.612363 w_k1_IBM_s006
                                 + 1.0323 w_k1_INTC_s006
                                 + 0.674353 w_k1_KO_s006
                                 + 0.816085 w_k1_MAT_s006
                                 + 1.00256 w_k1_MSFT_s006
                                 + 0.872783 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.824915 w_k1_ADM_s006
                                 + 0.935728 w_k1_AXP_s006
                                 + 0.847193 w_k1_DIS_s006
                                 + 0.882836 w_k1_FDX_s006
                                 + 1.19439 w_k1_IBM_s006
                                 + 1.01642 w_k1_INTC_s006
                                 + 0.980873 w_k1_KO_s006
                                 + 1.17249 w_k1_MAT_s006
                                 + 0.836595 w_k1_MSFT_s006
                                 + 0.974843 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.821518 w_k1_ADM_s006
                                 + 1.17707 w_k1_AXP_s006
                                 + 1.11108 w_k1_DIS_s006
                                 + 0.975382 w_k1_FDX_s006
                                 + 1.37078 w_k1_IBM_s006
                                 + 0.968807 w_k1_INTC_s006
                                 + 1.24104 w_k1_KO_s006 + 1.97976 w_k1_MAT_s006
                                 + 1.04769 w_k1_MSFT_s006
                                 + 1.06498 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.18821 w_k1_ADM_s007
                                 + 1.21841 w_k1_AXP_s007
                                 + 1.09368 w_k1_DIS_s007
                                 + 0.943227 w_k1_FDX_s007
                                 + 1.4243 w_k1_IBM_s007
                                 + 0.814238 w_k1_INTC_s007
                                 + 1.32547 w_k1_KO_s007 + 2.43217 w_k1_MAT_s007
                                 + 0.871742 w_k1_MSFT_s007
                                 + 1.11106 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.44778 w_k1_ADM_s007
                                 + 1.79328 w_k1_AXP_s007 + 2.1213 w_k1_DIS_s007
                                 + 1.51707 w_k1_FDX_s007
                                 + 1.47186 w_k1_IBM_s007
                                 + 1.43267 w_k1_INTC_s007
                                 + 1.63025 w_k1_KO_s007 + 1.62412 w_k1_MAT_s007
                                 + 1.31451 w_k1_MSFT_s007
                                 + 1.14471 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.30347 w_k1_ADM_s007
                                 + 1.09616 w_k1_AXP_s007
                                 + 1.28644 w_k1_DIS_s007 + 1.2497 w_k1_FDX_s007
                                 + 0.823897 w_k1_IBM_s007
                                 + 1.12089 w_k1_INTC_s007
                                 + 1.13294 w_k1_KO_s007 + 1.24621 w_k1_MAT_s007
                                 + 1.07633 w_k1_MSFT_s007
                                 + 1.06675 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.31641 w_k1_ADM_s007
                                 + 0.983353 w_k1_AXP_s007
                                 + 0.855697 w_k1_DIS_s007
                                 + 0.888492 w_k1_FDX_s007
                                 + 0.8915 w_k1_IBM_s007
                                 + 0.98932 w_k1_INTC_s007
                                 + 0.947967 w_k1_KO_s007
                                 + 0.902703 w_k1_MAT_s007
                                 + 0.819592 w_k1_MSFT_s007
                                 + 0.950706 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.812733 w_k1_ADM_s007
                                 + 0.835126 w_k1_AXP_s007
                                 + 0.954451 w_k1_DIS_s007
                                 + 0.798775 w_k1_FDX_s007
                                 + 1.4871 w_k1_IBM_s007
                                 + 0.746007 w_k1_INTC_s007
                                 + 1.14718 w_k1_KO_s007 + 1.67653 w_k1_MAT_s007
                                 + 0.748476 w_k1_MSFT_s007
                                 + 1.00669 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.22224 w_k1_ADM_s007
                                 + 1.31378 w_k1_AXP_s007 + 1.2501 w_k1_DIS_s007
                                 + 1.12268 w_k1_FDX_s007 + 1.1708 w_k1_IBM_s007
                                 + 0.902861 w_k1_INTC_s007
                                 + 1.39597 w_k1_KO_s007 + 1.65342 w_k1_MAT_s007
                                 + 0.834959 w_k1_MSFT_s007
                                 + 1.08681 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.31199 w_k1_ADM_s007
                                 + 0.934547 w_k1_AXP_s007
                                 + 1.62838 w_k1_DIS_s007
                                 + 1.18274 w_k1_FDX_s007
                                 + 1.31496 w_k1_IBM_s007 + 1.005 w_k1_INTC_s007
                                 + 1.56133 w_k1_KO_s007 + 1.10671 w_k1_MAT_s007
                                 + 1.09811 w_k1_MSFT_s007
                                 + 1.21186 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.854773 w_k1_ADM_s007
                                 + 0.642299 w_k1_AXP_s007
                                 + 0.68605 w_k1_DIS_s007
                                 + 0.769752 w_k1_FDX_s007
                                 + 0.718264 w_k1_IBM_s007
                                 + 0.670768 w_k1_INTC_s007
                                 + 0.83358 w_k1_KO_s007
                                 + 0.747402 w_k1_MAT_s007
                                 + 0.703605 w_k1_MSFT_s007
                                 + 0.754069 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.16926 w_k1_ADM_s010
                                 + 0.939467 w_k1_AXP_s010
                                 + 0.828506 w_k1_DIS_s010
                                 + 0.808403 w_k1_FDX_s010
                                 + 1.44838 w_k1_IBM_s010
                                 + 0.880049 w_k1_INTC_s010
                                 + 1.01702 w_k1_KO_s010 + 1.1196 w_k1_MAT_s010
                                 + 0.843623 w_k1_MSFT_s010
                                 + 0.857174 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.855313 w_k1_ADM_s010
                                 + 0.897754 w_k1_AXP_s010
                                 + 0.899793 w_k1_DIS_s010
                                 + 0.91596 w_k1_FDX_s010
                                 + 0.820203 w_k1_IBM_s010
                                 + 0.967204 w_k1_INTC_s010
                                 + 0.834345 w_k1_KO_s010
                                 + 0.613936 w_k1_MAT_s010
                                 + 0.958096 w_k1_MSFT_s010
                                 + 0.897371 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.914475 w_k1_ADM_s010
                                 + 1.06101 w_k1_AXP_s010
                                 + 0.720159 w_k1_DIS_s010
                                 + 0.883335 w_k1_FDX_s010 + 0.933 w_k1_IBM_s010
                                 + 0.837897 w_k1_INTC_s010
                                 + 0.806503 w_k1_KO_s010
                                 + 1.09517 w_k1_MAT_s010
                                 + 0.743743 w_k1_MSFT_s010
                                 + 0.808146 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.892918 w_k1_ADM_s010
                                 + 0.974652 w_k1_AXP_s010
                                 + 1.32884 w_k1_DIS_s010
                                 + 1.17776 w_k1_FDX_s010
                                 + 1.18547 w_k1_IBM_s010
                                 + 1.26076 w_k1_INTC_s010
                                 + 1.11131 w_k1_KO_s010
                                 + 0.911343 w_k1_MAT_s010
                                 + 1.22348 w_k1_MSFT_s010
                                 + 1.04862 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.29694 w_k1_ADM_s010
                                 + 1.35997 w_k1_AXP_s010
                                 + 1.20368 w_k1_DIS_s010
                                 + 1.24478 w_k1_FDX_s010
                                 + 0.868825 w_k1_IBM_s010
                                 + 1.30343 w_k1_INTC_s010
                                 + 0.921836 w_k1_KO_s010
                                 + 0.658886 w_k1_MAT_s010
                                 + 1.27278 w_k1_MSFT_s010
                                 + 1.01033 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.821013 w_k1_ADM_s010
                                 + 0.647288 w_k1_AXP_s010
                                 + 0.8752 w_k1_DIS_s010
                                 + 0.878642 w_k1_FDX_s010
                                 + 0.840051 w_k1_IBM_s010
                                 + 0.907613 w_k1_INTC_s010
                                 + 0.876821 w_k1_KO_s010
                                 + 0.419417 w_k1_MAT_s010
                                 + 1.04348 w_k1_MSFT_s010
                                 + 0.874216 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.843345 w_k1_ADM_s010
                                 + 0.988373 w_k1_AXP_s010
                                 + 0.844461 w_k1_DIS_s010
                                 + 0.893679 w_k1_FDX_s010
                                 + 1.11379 w_k1_IBM_s010 + 1.079 w_k1_INTC_s010
                                 + 0.927354 w_k1_KO_s010
                                 + 0.997215 w_k1_MAT_s010
                                 + 1.01282 w_k1_MSFT_s010
                                 + 0.938495 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.772022 w_k1_ADM_s010
                                 + 0.72949 w_k1_AXP_s010
                                 + 0.65633 w_k1_DIS_s010
                                 + 0.746126 w_k1_FDX_s010
                                 + 0.721193 w_k1_IBM_s010
                                 + 0.878727 w_k1_INTC_s010
                                 + 0.623028 w_k1_KO_s010
                                 + 0.676642 w_k1_MAT_s010
                                 + 0.810238 w_k1_MSFT_s010
                                 + 0.746095 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.886807 w_k1_ADM_s011
                                 + 0.60381 w_k1_AXP_s011
                                 + 0.589006 w_k1_DIS_s011
                                 + 0.690859 w_k1_FDX_s011
                                 + 0.824536 w_k1_IBM_s011
                                 + 0.644781 w_k1_INTC_s011
                                 + 0.770741 w_k1_KO_s011
                                 + 0.756602 w_k1_MAT_s011
                                 + 0.723884 w_k1_MSFT_s011
                                 + 0.878613 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.27406 w_k1_ADM_s011
                                 + 1.13848 w_k1_AXP_s011
                                 + 1.49328 w_k1_DIS_s011
                                 + 1.35293 w_k1_FDX_s011
                                 + 0.851605 w_k1_IBM_s011
                                 + 1.18061 w_k1_INTC_s011 + 1.0633 w_k1_KO_s011
                                 + 0.592633 w_k1_MAT_s011
                                 + 1.45389 w_k1_MSFT_s011 + 1.2517 w_k1_PG_s011
                                 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.05386 w_k1_ADM_s011
                                 + 0.933098 w_k1_AXP_s011
                                 + 1.10946 w_k1_DIS_s011
                                 + 0.970636 w_k1_FDX_s011
                                 + 1.1233 w_k1_IBM_s011
                                 + 1.06859 w_k1_INTC_s011
                                 + 0.950529 w_k1_KO_s011
                                 + 0.927227 w_k1_MAT_s011
                                 + 1.00858 w_k1_MSFT_s011
                                 + 1.03896 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.759054 w_k1_ADM_s011
                                 + 0.576791 w_k1_AXP_s011
                                 + 0.745691 w_k1_DIS_s011
                                 + 0.786428 w_k1_FDX_s011
                                 + 0.712568 w_k1_IBM_s011
                                 + 0.681839 w_k1_INTC_s011
                                 + 0.854123 w_k1_KO_s011
                                 + 0.728396 w_k1_MAT_s011
                                 + 0.814222 w_k1_MSFT_s011
                                 + 0.874113 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.44692 w_k1_ADM_s011
                                 + 1.58382 w_k1_AXP_s011 + 1.1669 w_k1_DIS_s011
                                 + 1.33072 w_k1_FDX_s011
                                 + 1.08118 w_k1_IBM_s011
                                 + 0.981921 w_k1_INTC_s011
                                 + 0.966635 w_k1_KO_s011
                                 + 1.24883 w_k1_MAT_s011
                                 + 0.912515 w_k1_MSFT_s011
                                 + 1.18562 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.05796 w_k1_ADM_s011
                                 + 0.996018 w_k1_AXP_s011
                                 + 1.18725 w_k1_DIS_s011
                                 + 1.09052 w_k1_FDX_s011
                                 + 0.774518 w_k1_IBM_s011
                                 + 1.0401 w_k1_INTC_s011
                                 + 0.900963 w_k1_KO_s011
                                 + 0.482639 w_k1_MAT_s011
                                 + 1.04888 w_k1_MSFT_s011
                                 + 0.978379 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.25998 w_k1_ADM_s011
                                 + 1.13778 w_k1_AXP_s011 + 1.4392 w_k1_DIS_s011
                                 + 1.14002 w_k1_FDX_s011
                                 + 1.46289 w_k1_IBM_s011
                                 + 0.989847 w_k1_INTC_s011
                                 + 1.34643 w_k1_KO_s011 + 1.43357 w_k1_MAT_s011
                                 + 1.23169 w_k1_MSFT_s011
                                 + 1.19218 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.852329 w_k1_ADM_s011
                                 + 0.740257 w_k1_AXP_s011
                                 + 0.910321 w_k1_DIS_s011
                                 + 0.859299 w_k1_FDX_s011
                                 + 0.992548 w_k1_IBM_s011
                                 + 0.795717 w_k1_INTC_s011
                                 + 0.943535 w_k1_KO_s011
                                 + 0.635882 w_k1_MAT_s011
                                 + 1.00994 w_k1_MSFT_s011
                                 + 0.942897 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.905827 w_k1_ADM_s012
                                 + 0.809459 w_k1_AXP_s012
                                 + 0.761454 w_k1_DIS_s012
                                 + 0.951031 w_k1_FDX_s012
                                 + 0.644429 w_k1_IBM_s012
                                 + 0.907489 w_k1_INTC_s012
                                 + 0.692425 w_k1_KO_s012
                                 + 0.719338 w_k1_MAT_s012
                                 + 0.889024 w_k1_MSFT_s012
                                 + 0.969488 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.01691 w_k1_ADM_s012
                                 + 1.29061 w_k1_AXP_s012
                                 + 1.37829 w_k1_DIS_s012
                                 + 1.24368 w_k1_FDX_s012
                                 + 0.92299 w_k1_IBM_s012
                                 + 1.0976 w_k1_INTC_s012
                                 + 0.979939 w_k1_KO_s012
                                 + 1.25841 w_k1_MAT_s012
                                 + 1.19143 w_k1_MSFT_s012
                                 + 1.13072 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.679122 w_k1_ADM_s012
                                 + 0.630033 w_k1_AXP_s012
                                 + 0.683861 w_k1_DIS_s012
                                 + 0.695447 w_k1_FDX_s012
                                 + 1.1204 w_k1_IBM_s012
                                 + 0.801854 w_k1_INTC_s012
                                 + 0.886443 w_k1_KO_s012
                                 + 0.936546 w_k1_MAT_s012
                                 + 0.850347 w_k1_MSFT_s012
                                 + 0.909604 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.22757 w_k1_ADM_s012
                                 + 1.73136 w_k1_AXP_s012 + 1.4914 w_k1_DIS_s012
                                 + 1.32177 w_k1_FDX_s012
                                 + 1.63437 w_k1_IBM_s012
                                 + 1.10899 w_k1_INTC_s012
                                 + 1.34615 w_k1_KO_s012 + 1.95674 w_k1_MAT_s012
                                 + 1.03336 w_k1_MSFT_s012
                                 + 1.22622 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.971525 w_k1_ADM_s012
                                 + 1.17087 w_k1_AXP_s012 + 1.2454 w_k1_DIS_s012
                                 + 1.16051 w_k1_FDX_s012
                                 + 1.31822 w_k1_IBM_s012
                                 + 1.04054 w_k1_INTC_s012
                                 + 1.11403 w_k1_KO_s012 + 1.4871 w_k1_MAT_s012
                                 + 1.23579 w_k1_MSFT_s012
                                 + 1.20954 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.781799 w_k1_ADM_s012
                                 + 1.01945 w_k1_AXP_s012
                                 + 0.918461 w_k1_DIS_s012
                                 + 0.991398 w_k1_FDX_s012
                                 + 1.01536 w_k1_IBM_s012
                                 + 1.00819 w_k1_INTC_s012
                                 + 0.927664 w_k1_KO_s012
                                 + 1.20336 w_k1_MAT_s012
                                 + 1.04781 w_k1_MSFT_s012
                                 + 1.02554 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.855464 w_k1_ADM_s012
                                 + 0.618098 w_k1_AXP_s012
                                 + 0.580575 w_k1_DIS_s012
                                 + 0.744425 w_k1_FDX_s012
                                 + 0.685407 w_k1_IBM_s012
                                 + 0.745411 w_k1_INTC_s012
                                 + 0.636565 w_k1_KO_s012
                                 + 0.482592 w_k1_MAT_s012
                                 + 0.720211 w_k1_MSFT_s012
                                 + 0.949891 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.00305 w_k1_ADM_s012
                                 + 1.00612 w_k1_AXP_s012
                                 + 0.949088 w_k1_DIS_s012
                                 + 1.08225 w_k1_FDX_s012
                                 + 0.692577 w_k1_IBM_s012
                                 + 0.961898 w_k1_INTC_s012
                                 + 0.852645 w_k1_KO_s012
                                 + 0.842209 w_k1_MAT_s012
                                 + 1.0456 w_k1_MSFT_s012 + 1.0735 w_k1_PG_s012
                                 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.14092 w_k1_ADM_s013
                                 + 0.971906 w_k1_AXP_s013
                                 + 0.949196 w_k1_DIS_s013
                                 + 1.0591 w_k1_FDX_s013
                                 + 0.876036 w_k1_IBM_s013
                                 + 1.01389 w_k1_INTC_s013
                                 + 0.92308 w_k1_KO_s013
                                 + 0.858759 w_k1_MAT_s013
                                 + 1.04309 w_k1_MSFT_s013
                                 + 0.849446 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.18928 w_k1_ADM_s013
                                 + 0.971401 w_k1_AXP_s013
                                 + 1.07552 w_k1_DIS_s013
                                 + 0.947679 w_k1_FDX_s013
                                 + 1.30333 w_k1_IBM_s013
                                 + 0.809339 w_k1_INTC_s013
                                 + 1.0171 w_k1_KO_s013 + 1.15634 w_k1_MAT_s013
                                 + 0.888317 w_k1_MSFT_s013
                                 + 1.00153 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.793906 w_k1_ADM_s013
                                 + 0.689451 w_k1_AXP_s013
                                 + 0.615826 w_k1_DIS_s013
                                 + 0.718192 w_k1_FDX_s013
                                 + 0.800622 w_k1_IBM_s013
                                 + 1.07771 w_k1_INTC_s013
                                 + 0.798442 w_k1_KO_s013
                                 + 0.707546 w_k1_MAT_s013
                                 + 0.867309 w_k1_MSFT_s013
                                 + 0.842347 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.14345 w_k1_ADM_s013
                                 + 1.21804 w_k1_AXP_s013 + 1.5587 w_k1_DIS_s013
                                 + 1.36905 w_k1_FDX_s013
                                 + 1.51172 w_k1_IBM_s013
                                 + 1.72311 w_k1_INTC_s013
                                 + 1.50496 w_k1_KO_s013 + 1.61993 w_k1_MAT_s013
                                 + 1.42625 w_k1_MSFT_s013
                                 + 1.22931 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.41572 w_k1_ADM_s013
                                 + 0.990723 w_k1_AXP_s013
                                 + 1.41562 w_k1_DIS_s013
                                 + 1.24092 w_k1_FDX_s013
                                 + 0.918043 w_k1_IBM_s013
                                 + 1.04457 w_k1_INTC_s013
                                 + 1.05873 w_k1_KO_s013
                                 + 0.814151 w_k1_MAT_s013
                                 + 1.32384 w_k1_MSFT_s013
                                 + 1.00879 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.910891 w_k1_ADM_s013
                                 + 0.896513 w_k1_AXP_s013
                                 + 0.687614 w_k1_DIS_s013
                                 + 0.853353 w_k1_FDX_s013
                                 + 0.743896 w_k1_IBM_s013
                                 + 0.878849 w_k1_INTC_s013
                                 + 0.768311 w_k1_KO_s013
                                 + 0.838021 w_k1_MAT_s013
                                 + 0.779131 w_k1_MSFT_s013
                                 + 0.803134 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.964925 w_k1_ADM_s013
                                 + 0.864804 w_k1_AXP_s013
                                 + 0.718111 w_k1_DIS_s013
                                 + 0.832166 w_k1_FDX_s013
                                 + 1.06394 w_k1_IBM_s013
                                 + 0.973679 w_k1_INTC_s013
                                 + 0.897459 w_k1_KO_s013
                                 + 0.966226 w_k1_MAT_s013
                                 + 0.786406 w_k1_MSFT_s013
                                 + 0.776434 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.4451 w_k1_ADM_s013
                                 + 1.53348 w_k1_AXP_s013 + 1.2631 w_k1_DIS_s013
                                 + 1.3355 w_k1_FDX_s013 + 1.12421 w_k1_IBM_s013
                                 + 1.18162 w_k1_INTC_s013
                                 + 1.01393 w_k1_KO_s013 + 1.06247 w_k1_MAT_s013
                                 + 1.11153 w_k1_MSFT_s013
                                 + 1.06274 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.26795 w_k1_ADM_s014
                                 + 1.26184 w_k1_AXP_s014
                                 + 1.02123 w_k1_DIS_s014
                                 + 0.973991 w_k1_FDX_s014
                                 + 1.66943 w_k1_IBM_s014
                                 + 0.935662 w_k1_INTC_s014
                                 + 1.25542 w_k1_KO_s014 + 1.93025 w_k1_MAT_s014
                                 + 1.0131 w_k1_MSFT_s014 + 1.01837 w_k1_PG_s014
                                 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.822587 w_k1_ADM_s014
                                 + 0.737382 w_k1_AXP_s014
                                 + 0.787633 w_k1_DIS_s014
                                 + 0.819347 w_k1_FDX_s014
                                 + 1.35793 w_k1_IBM_s014
                                 + 0.616312 w_k1_INTC_s014
                                 + 0.887541 w_k1_KO_s014
                                 + 1.38143 w_k1_MAT_s014
                                 + 0.742133 w_k1_MSFT_s014
                                 + 0.893679 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.954734 w_k1_ADM_s014
                                 + 0.671459 w_k1_AXP_s014
                                 + 0.793462 w_k1_DIS_s014
                                 + 0.923154 w_k1_FDX_s014
                                 + 0.629334 w_k1_IBM_s014
                                 + 0.955909 w_k1_INTC_s014
                                 + 0.752662 w_k1_KO_s014
                                 + 0.520824 w_k1_MAT_s014
                                 + 0.942858 w_k1_MSFT_s014
                                 + 0.824962 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.20962 w_k1_ADM_s014
                                 + 0.886976 w_k1_AXP_s014
                                 + 0.86417 w_k1_DIS_s014
                                 + 1.09864 w_k1_FDX_s014
                                 + 0.819136 w_k1_IBM_s014
                                 + 0.928354 w_k1_INTC_s014
                                 + 0.859393 w_k1_KO_s014
                                 + 0.700406 w_k1_MAT_s014
                                 + 0.931432 w_k1_MSFT_s014
                                 + 0.980809 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.23475 w_k1_ADM_s014
                                 + 1.42079 w_k1_AXP_s014
                                 + 1.58297 w_k1_DIS_s014
                                 + 1.36653 w_k1_FDX_s014
                                 + 1.36895 w_k1_IBM_s014
                                 + 1.37878 w_k1_INTC_s014
                                 + 1.31046 w_k1_KO_s014 + 1.57467 w_k1_MAT_s014
                                 + 1.43008 w_k1_MSFT_s014
                                 + 1.04869 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.12346 w_k1_ADM_s014
                                 + 0.980362 w_k1_AXP_s014
                                 + 0.983613 w_k1_DIS_s014
                                 + 1.08292 w_k1_FDX_s014 + 1.134 w_k1_IBM_s014
                                 + 0.908792 w_k1_INTC_s014
                                 + 1.02442 w_k1_KO_s014 + 1.27241 w_k1_MAT_s014
                                 + 1.04123 w_k1_MSFT_s014
                                 + 0.918438 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.3545 w_k1_ADM_s014
                                 + 1.08817 w_k1_AXP_s014
                                 + 1.42144 w_k1_DIS_s014
                                 + 1.34061 w_k1_FDX_s014
                                 + 0.973658 w_k1_IBM_s014
                                 + 0.983064 w_k1_INTC_s014
                                 + 0.93728 w_k1_KO_s014
                                 + 0.860032 w_k1_MAT_s014
                                 + 1.17861 w_k1_MSFT_s014
                                 + 0.962956 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.786623 w_k1_ADM_s014
                                 + 0.780705 w_k1_AXP_s014
                                 + 0.879284 w_k1_DIS_s014
                                 + 0.882286 w_k1_FDX_s014
                                 + 0.934946 w_k1_IBM_s014
                                 + 0.990458 w_k1_INTC_s014
                                 + 0.984949 w_k1_KO_s014
                                 + 1.02329 w_k1_MAT_s014
                                 + 1.00723 w_k1_MSFT_s014
                                 + 0.942209 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.970412 w_k1_ADM_s015
                                 + 0.652722 w_k1_AXP_s015
                                 + 0.758847 w_k1_DIS_s015
                                 + 0.842226 w_k1_FDX_s015
                                 + 0.665722 w_k1_IBM_s015
                                 + 0.696532 w_k1_INTC_s015
                                 + 0.847346 w_k1_KO_s015
                                 + 0.640848 w_k1_MAT_s015
                                 + 0.983623 w_k1_MSFT_s015
                                 + 0.868509 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.998364 w_k1_ADM_s015
                                 + 1.13631 w_k1_AXP_s015
                                 + 0.983609 w_k1_DIS_s015
                                 + 1.02733 w_k1_FDX_s015
                                 + 0.882931 w_k1_IBM_s015
                                 + 0.876719 w_k1_INTC_s015
                                 + 0.865104 w_k1_KO_s015
                                 + 0.765186 w_k1_MAT_s015
                                 + 0.932714 w_k1_MSFT_s015
                                 + 0.882783 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.92594 w_k1_ADM_s015
                                 + 0.77099 w_k1_AXP_s015
                                 + 0.855464 w_k1_DIS_s015
                                 + 0.803415 w_k1_FDX_s015
                                 + 1.00973 w_k1_IBM_s015
                                 + 0.725868 w_k1_INTC_s015
                                 + 0.99694 w_k1_KO_s015
                                 + 0.998362 w_k1_MAT_s015
                                 + 0.957388 w_k1_MSFT_s015
                                 + 0.942445 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.26277 w_k1_ADM_s015
                                 + 0.93361 w_k1_AXP_s015
                                 + 0.694889 w_k1_DIS_s015
                                 + 0.92223 w_k1_FDX_s015
                                 + 0.80787 w_k1_IBM_s015
                                 + 0.98013 w_k1_INTC_s015
                                 + 0.977373 w_k1_KO_s015
                                 + 0.899469 w_k1_MAT_s015
                                 + 1.06207 w_k1_MSFT_s015
                                 + 0.983352 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.17722 w_k1_ADM_s015
                                 + 1.56449 w_k1_AXP_s015
                                 + 1.66115 w_k1_DIS_s015
                                 + 1.32725 w_k1_FDX_s015
                                 + 1.52637 w_k1_IBM_s015
                                 + 1.18264 w_k1_INTC_s015
                                 + 1.46695 w_k1_KO_s015 + 1.52216 w_k1_MAT_s015
                                 + 1.32596 w_k1_MSFT_s015
                                 + 1.20708 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.2702 w_k1_ADM_s015
                                 + 1.17139 w_k1_AXP_s015
                                 + 1.08597 w_k1_DIS_s015
                                 + 1.09749 w_k1_FDX_s015
                                 + 1.25251 w_k1_IBM_s015
                                 + 1.12256 w_k1_INTC_s015
                                 + 1.18688 w_k1_KO_s015
                                 + 0.831265 w_k1_MAT_s015
                                 + 1.06667 w_k1_MSFT_s015
                                 + 0.986454 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.20489 w_k1_ADM_s015
                                 + 0.93071 w_k1_AXP_s015
                                 + 1.06205 w_k1_DIS_s015
                                 + 0.880845 w_k1_FDX_s015
                                 + 1.50834 w_k1_IBM_s015
                                 + 0.75208 w_k1_INTC_s015
                                 + 1.39374 w_k1_KO_s015 + 1.41564 w_k1_MAT_s015
                                 + 1.09444 w_k1_MSFT_s015
                                 + 1.06153 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.43357 w_k1_ADM_s015
                                 + 1.03118 w_k1_AXP_s015
                                 + 1.41102 w_k1_DIS_s015
                                 + 1.19154 w_k1_FDX_s015
                                 + 0.809041 w_k1_IBM_s015
                                 + 0.917081 w_k1_INTC_s015
                                 + 1.21989 w_k1_KO_s015
                                 + 0.554243 w_k1_MAT_s015
                                 + 1.18626 w_k1_MSFT_s015
                                 + 1.07424 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.61981 w_k1_ADM_s016
                                 + 1.17071 w_k1_AXP_s016
                                 + 1.08381 w_k1_DIS_s016
                                 + 1.20921 w_k1_FDX_s016
                                 + 0.729695 w_k1_IBM_s016
                                 + 1.22847 w_k1_INTC_s016
                                 + 0.839219 w_k1_KO_s016
                                 + 0.808156 w_k1_MAT_s016
                                 + 1.13319 w_k1_MSFT_s016
                                 + 1.06042 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.885825 w_k1_ADM_s016
                                 + 0.724628 w_k1_AXP_s016
                                 + 0.616083 w_k1_DIS_s016
                                 + 0.755825 w_k1_FDX_s016
                                 + 0.715043 w_k1_IBM_s016
                                 + 0.925441 w_k1_INTC_s016
                                 + 0.629736 w_k1_KO_s016
                                 + 0.493286 w_k1_MAT_s016
                                 + 0.728109 w_k1_MSFT_s016
                                 + 0.721689 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 2.07235 w_k1_ADM_s016
                                 + 1.79637 w_k1_AXP_s016
                                 + 1.54923 w_k1_DIS_s016
                                 + 1.54987 w_k1_FDX_s016
                                 + 1.19129 w_k1_IBM_s016
                                 + 1.61577 w_k1_INTC_s016 + 1.2733 w_k1_KO_s016
                                 + 1.32481 w_k1_MAT_s016
                                 + 1.42883 w_k1_MSFT_s016
                                 + 1.41315 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.715002 w_k1_ADM_s016
                                 + 0.760373 w_k1_AXP_s016
                                 + 0.625016 w_k1_DIS_s016
                                 + 0.677257 w_k1_FDX_s016
                                 + 1.15484 w_k1_IBM_s016
                                 + 0.87059 w_k1_INTC_s016
                                 + 0.945617 w_k1_KO_s016
                                 + 0.917977 w_k1_MAT_s016
                                 + 0.788391 w_k1_MSFT_s016
                                 + 0.875906 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.11987 w_k1_ADM_s016
                                 + 1.16683 w_k1_AXP_s016
                                 + 1.04909 w_k1_DIS_s016
                                 + 0.957231 w_k1_FDX_s016
                                 + 1.65487 w_k1_IBM_s016
                                 + 1.36246 w_k1_INTC_s016
                                 + 1.36897 w_k1_KO_s016 + 1.25977 w_k1_MAT_s016
                                 + 1.04633 w_k1_MSFT_s016
                                 + 1.06339 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.20655 w_k1_ADM_s016
                                 + 1.05982 w_k1_AXP_s016
                                 + 1.28094 w_k1_DIS_s016
                                 + 1.19635 w_k1_FDX_s016
                                 + 0.909129 w_k1_IBM_s016
                                 + 1.21974 w_k1_INTC_s016
                                 + 0.968767 w_k1_KO_s016
                                 + 0.840597 w_k1_MAT_s016
                                 + 1.20826 w_k1_MSFT_s016
                                 + 1.11967 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.05586 w_k1_ADM_s016
                                 + 0.900185 w_k1_AXP_s016
                                 + 1.03565 w_k1_DIS_s016
                                 + 0.995262 w_k1_FDX_s016
                                 + 1.14444 w_k1_IBM_s016
                                 + 0.921433 w_k1_INTC_s016
                                 + 1.09981 w_k1_KO_s016 + 0.93718 w_k1_MAT_s016
                                 + 0.956603 w_k1_MSFT_s016
                                 + 1.03348 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.833967 w_k1_ADM_s016
                                 + 0.910697 w_k1_AXP_s016
                                 + 0.809086 w_k1_DIS_s016
                                 + 0.886651 w_k1_FDX_s016
                                 + 0.900379 w_k1_IBM_s016
                                 + 1.06326 w_k1_INTC_s016
                                 + 0.982355 w_k1_KO_s016
                                 + 1.09638 w_k1_MAT_s016
                                 + 0.951254 w_k1_MSFT_s016
                                 + 0.950783 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.29585 w_k1_ADM_s017
                                 + 1.52151 w_k1_AXP_s017
                                 + 1.13124 w_k1_DIS_s017
                                 + 1.13409 w_k1_FDX_s017
                                 + 1.30766 w_k1_IBM_s017
                                 + 1.30612 w_k1_INTC_s017
                                 + 1.29232 w_k1_KO_s017 + 1.48013 w_k1_MAT_s017
                                 + 1.15988 w_k1_MSFT_s017
                                 + 1.12476 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.853371 w_k1_ADM_s017
                                 + 0.773582 w_k1_AXP_s017
                                 + 0.633654 w_k1_DIS_s017
                                 + 0.694144 w_k1_FDX_s017
                                 + 1.0394 w_k1_IBM_s017
                                 + 0.846487 w_k1_INTC_s017
                                 + 0.9167 w_k1_KO_s017 + 0.990496 w_k1_MAT_s017
                                 + 0.727133 w_k1_MSFT_s017
                                 + 0.868291 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.905903 w_k1_ADM_s017
                                 + 0.760799 w_k1_AXP_s017
                                 + 0.82558 w_k1_DIS_s017
                                 + 0.88395 w_k1_FDX_s017
                                 + 0.73254 w_k1_IBM_s017
                                 + 1.05121 w_k1_INTC_s017
                                 + 0.85168 w_k1_KO_s017
                                 + 0.616242 w_k1_MAT_s017
                                 + 0.870644 w_k1_MSFT_s017
                                 + 0.811012 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.06738 w_k1_ADM_s017
                                 + 1.00998 w_k1_AXP_s017
                                 + 0.813467 w_k1_DIS_s017
                                 + 0.918555 w_k1_FDX_s017
                                 + 1.47841 w_k1_IBM_s017
                                 + 0.901252 w_k1_INTC_s017
                                 + 1.0857 w_k1_KO_s017 + 1.51669 w_k1_MAT_s017
                                 + 0.943566 w_k1_MSFT_s017
                                 + 1.03044 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.0878 w_k1_ADM_s017
                                 + 0.901198 w_k1_AXP_s017
                                 + 0.851385 w_k1_DIS_s017
                                 + 0.933381 w_k1_FDX_s017
                                 + 1.01198 w_k1_IBM_s017
                                 + 0.962343 w_k1_INTC_s017
                                 + 0.969633 w_k1_KO_s017
                                 + 0.932469 w_k1_MAT_s017
                                 + 1.0093 w_k1_MSFT_s017 + 1.00637 w_k1_PG_s017
                                 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.21638 w_k1_ADM_s017
                                 + 1.09926 w_k1_AXP_s017
                                 + 1.12801 w_k1_DIS_s017
                                 + 1.25742 w_k1_FDX_s017
                                 + 0.792112 w_k1_IBM_s017
                                 + 1.21901 w_k1_INTC_s017
                                 + 1.03079 w_k1_KO_s017
                                 + 0.942417 w_k1_MAT_s017
                                 + 1.24062 w_k1_MSFT_s017
                                 + 1.16065 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.9485 w_k1_ADM_s017
                                 + 1.23275 w_k1_AXP_s017
                                 + 1.52558 w_k1_DIS_s017
                                 + 1.36031 w_k1_FDX_s017
                                 + 1.16919 w_k1_IBM_s017
                                 + 1.44056 w_k1_INTC_s017
                                 + 1.39322 w_k1_KO_s017 + 1.12945 w_k1_MAT_s017
                                 + 1.42988 w_k1_MSFT_s017
                                 + 1.28152 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.889549 w_k1_ADM_s017
                                 + 0.860847 w_k1_AXP_s017
                                 + 0.661986 w_k1_DIS_s017
                                 + 0.930926 w_k1_FDX_s017
                                 + 0.670389 w_k1_IBM_s017
                                 + 1.00586 w_k1_INTC_s017
                                 + 0.799763 w_k1_KO_s017
                                 + 0.659857 w_k1_MAT_s017
                                 + 0.988164 w_k1_MSFT_s017
                                 + 0.961194 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.04131 w_k1_ADM_s100
                                 + 1.07322 w_k1_AXP_s100
                                 + 0.982332 w_k1_DIS_s100
                                 + 0.952478 w_k1_FDX_s100
                                 + 1.18334 w_k1_IBM_s100
                                 + 0.909617 w_k1_INTC_s100
                                 + 1.0141 w_k1_KO_s100 + 1.08708 w_k1_MAT_s100
                                 + 0.866485 w_k1_MSFT_s100
                                 + 1.04475 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.0451 w_k1_ADM_s100
                                 + 0.753178 w_k1_AXP_s100
                                 + 1.03452 w_k1_DIS_s100
                                 + 0.937824 w_k1_FDX_s100
                                 + 1.06125 w_k1_IBM_s100
                                 + 0.935892 w_k1_INTC_s100
                                 + 1.00961 w_k1_KO_s100
                                 + 0.874776 w_k1_MAT_s100
                                 + 0.984531 w_k1_MSFT_s100
                                 + 1.07375 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.897234 w_k1_ADM_s100
                                 + 0.858127 w_k1_AXP_s100
                                 + 0.960854 w_k1_DIS_s100
                                 + 1.04893 w_k1_FDX_s100
                                 + 0.594264 w_k1_IBM_s100
                                 + 1.04615 w_k1_INTC_s100
                                 + 0.743552 w_k1_KO_s100
                                 + 0.523468 w_k1_MAT_s100
                                 + 0.91817 w_k1_MSFT_s100
                                 + 0.896994 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.684955 w_k1_ADM_s100
                                 + 0.613525 w_k1_AXP_s100
                                 + 0.646647 w_k1_DIS_s100
                                 + 0.732931 w_k1_FDX_s100
                                 + 0.879401 w_k1_IBM_s100
                                 + 0.757145 w_k1_INTC_s100
                                 + 0.716078 w_k1_KO_s100
                                 + 0.686637 w_k1_MAT_s100
                                 + 0.694546 w_k1_MSFT_s100
                                 + 0.834534 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.760016 w_k1_ADM_s100
                                 + 0.691983 w_k1_AXP_s100
                                 + 0.90627 w_k1_DIS_s100
                                 + 0.889731 w_k1_FDX_s100
                                 + 0.92676 w_k1_IBM_s100
                                 + 1.02014 w_k1_INTC_s100
                                 + 0.909743 w_k1_KO_s100
                                 + 0.557033 w_k1_MAT_s100
                                 + 0.900687 w_k1_MSFT_s100
                                 + 0.911635 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.08244 w_k1_ADM_s100
                                 + 1.05389 w_k1_AXP_s100
                                 + 1.07001 w_k1_DIS_s100
                                 + 1.18139 w_k1_FDX_s100
                                 + 0.915013 w_k1_IBM_s100
                                 + 1.12429 w_k1_INTC_s100
                                 + 1.01458 w_k1_KO_s100 + 1.09662 w_k1_MAT_s100
                                 + 1.1217 w_k1_MSFT_s100 + 1.12015 w_k1_PG_s100
                                 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.902938 w_k1_ADM_s100
                                 + 0.932792 w_k1_AXP_s100
                                 + 0.62717 w_k1_DIS_s100
                                 + 0.907393 w_k1_FDX_s100
                                 + 0.739009 w_k1_IBM_s100
                                 + 0.845246 w_k1_INTC_s100
                                 + 0.713937 w_k1_KO_s100
                                 + 0.64459 w_k1_MAT_s100
                                 + 0.844417 w_k1_MSFT_s100
                                 + 0.930881 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.19535 w_k1_ADM_s100
                                 + 1.30289 w_k1_AXP_s100
                                 + 1.38562 w_k1_DIS_s100
                                 + 1.27209 w_k1_FDX_s100
                                 + 1.25083 w_k1_IBM_s100
                                 + 1.2532 w_k1_INTC_s100 + 1.22195 w_k1_KO_s100
                                 + 1.00666 w_k1_MAT_s100
                                 + 1.10708 w_k1_MSFT_s100 + 1.1367 w_k1_PG_s100
                                 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.38549 w_k1_ADM_s101
                                 + 1.34682 w_k1_AXP_s101
                                 + 1.59588 w_k1_DIS_s101
                                 + 1.37869 w_k1_FDX_s101
                                 + 1.08586 w_k1_IBM_s101
                                 + 0.974923 w_k1_INTC_s101
                                 + 1.26579 w_k1_KO_s101 + 1.35401 w_k1_MAT_s101
                                 + 1.31851 w_k1_MSFT_s101
                                 + 1.24999 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.779289 w_k1_ADM_s101
                                 + 0.540664 w_k1_AXP_s101
                                 + 0.753438 w_k1_DIS_s101
                                 + 0.701954 w_k1_FDX_s101
                                 + 0.868861 w_k1_IBM_s101
                                 + 0.620034 w_k1_INTC_s101
                                 + 0.794159 w_k1_KO_s101
                                 + 0.847897 w_k1_MAT_s101
                                 + 0.807633 w_k1_MSFT_s101
                                 + 0.927225 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.33861 w_k1_ADM_s101
                                 + 1.48521 w_k1_AXP_s101
                                 + 1.20273 w_k1_DIS_s101
                                 + 1.28626 w_k1_FDX_s101
                                 + 1.09218 w_k1_IBM_s101
                                 + 0.864042 w_k1_INTC_s101
                                 + 1.17539 w_k1_KO_s101 + 1.18694 w_k1_MAT_s101
                                 + 1.13455 w_k1_MSFT_s101
                                 + 1.23446 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.31319 w_k1_ADM_s101
                                 + 1.10182 w_k1_AXP_s101
                                 + 0.958612 w_k1_DIS_s101
                                 + 1.08054 w_k1_FDX_s101
                                 + 0.72518 w_k1_IBM_s101
                                 + 0.810794 w_k1_INTC_s101
                                 + 0.84174 w_k1_KO_s101 + 0.72648 w_k1_MAT_s101
                                 + 0.951931 w_k1_MSFT_s101
                                 + 1.05141 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.87319 w_k1_ADM_s101
                                 + 0.997402 w_k1_AXP_s101
                                 + 1.2165 w_k1_DIS_s101 + 1.1622 w_k1_FDX_s101
                                 + 0.796145 w_k1_IBM_s101
                                 + 0.969436 w_k1_INTC_s101
                                 + 0.894543 w_k1_KO_s101
                                 + 0.886794 w_k1_MAT_s101
                                 + 1.13453 w_k1_MSFT_s101
                                 + 1.10756 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.967968 w_k1_ADM_s101
                                 + 1.00395 w_k1_AXP_s101
                                 + 1.22067 w_k1_DIS_s101
                                 + 1.06375 w_k1_FDX_s101
                                 + 1.05366 w_k1_IBM_s101
                                 + 0.9354 w_k1_INTC_s101 + 1.03948 w_k1_KO_s101
                                 + 0.883443 w_k1_MAT_s101
                                 + 1.00522 w_k1_MSFT_s101 + 1.0451 w_k1_PG_s101
                                 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.975216 w_k1_ADM_s101
                                 + 0.894548 w_k1_AXP_s101
                                 + 0.833886 w_k1_DIS_s101
                                 + 0.887918 w_k1_FDX_s101
                                 + 0.888459 w_k1_IBM_s101
                                 + 0.870654 w_k1_INTC_s101
                                 + 0.731354 w_k1_KO_s101
                                 + 0.786538 w_k1_MAT_s101
                                 + 0.878712 w_k1_MSFT_s101
                                 + 0.868936 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.26115 w_k1_ADM_s101
                                 + 1.10019 w_k1_AXP_s101
                                 + 1.12004 w_k1_DIS_s101
                                 + 0.937946 w_k1_FDX_s101
                                 + 1.59144 w_k1_IBM_s101
                                 + 0.769254 w_k1_INTC_s101
                                 + 1.25337 w_k1_KO_s101 + 1.37865 w_k1_MAT_s101
                                 + 1.0087 w_k1_MSFT_s101 + 1.07139 w_k1_PG_s101
                                 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.756578 w_k1_ADM_s102
                                 + 0.668745 w_k1_AXP_s102
                                 + 0.651669 w_k1_DIS_s102
                                 + 0.749435 w_k1_FDX_s102
                                 + 0.853617 w_k1_IBM_s102
                                 + 0.873671 w_k1_INTC_s102
                                 + 0.955706 w_k1_KO_s102
                                 + 0.98018 w_k1_MAT_s102
                                 + 0.805354 w_k1_MSFT_s102
                                 + 0.835603 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.886501 w_k1_ADM_s102
                                 + 0.803175 w_k1_AXP_s102
                                 + 0.803118 w_k1_DIS_s102
                                 + 0.997268 w_k1_FDX_s102
                                 + 0.680413 w_k1_IBM_s102
                                 + 0.956839 w_k1_INTC_s102
                                 + 0.877609 w_k1_KO_s102
                                 + 0.809434 w_k1_MAT_s102
                                 + 1.0182 w_k1_MSFT_s102
                                 + 0.964622 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.997912 w_k1_ADM_s102
                                 + 1.09365 w_k1_AXP_s102
                                 + 0.863348 w_k1_DIS_s102
                                 + 0.942821 w_k1_FDX_s102
                                 + 0.843431 w_k1_IBM_s102
                                 + 0.903876 w_k1_INTC_s102
                                 + 0.860581 w_k1_KO_s102
                                 + 1.03841 w_k1_MAT_s102
                                 + 0.93682 w_k1_MSFT_s102
                                 + 1.01876 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.06941 w_k1_ADM_s102
                                 + 1.5556 w_k1_AXP_s102 + 1.19999 w_k1_DIS_s102
                                 + 1.18006 w_k1_FDX_s102
                                 + 0.971057 w_k1_IBM_s102
                                 + 1.21508 w_k1_INTC_s102
                                 + 0.915946 w_k1_KO_s102 + 1.0706 w_k1_MAT_s102
                                 + 1.06419 w_k1_MSFT_s102
                                 + 1.13571 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.14827 w_k1_ADM_s102
                                 + 1.27493 w_k1_AXP_s102
                                 + 1.41798 w_k1_DIS_s102
                                 + 1.26801 w_k1_FDX_s102
                                 + 0.927804 w_k1_IBM_s102
                                 + 1.4069 w_k1_INTC_s102 + 1.23109 w_k1_KO_s102
                                 + 0.969347 w_k1_MAT_s102
                                 + 1.36962 w_k1_MSFT_s102
                                 + 1.23603 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.893782 w_k1_ADM_s102
                                 + 1.25828 w_k1_AXP_s102
                                 + 1.16611 w_k1_DIS_s102
                                 + 1.11496 w_k1_FDX_s102
                                 + 1.34559 w_k1_IBM_s102
                                 + 1.12284 w_k1_INTC_s102
                                 + 1.20275 w_k1_KO_s102 + 1.54417 w_k1_MAT_s102
                                 + 1.18422 w_k1_MSFT_s102
                                 + 1.16343 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.19824 w_k1_ADM_s102
                                 + 1.69619 w_k1_AXP_s102
                                 + 1.67796 w_k1_DIS_s102
                                 + 1.36902 w_k1_FDX_s102
                                 + 1.50842 w_k1_IBM_s102
                                 + 1.31197 w_k1_INTC_s102
                                 + 1.53121 w_k1_KO_s102 + 2.0638 w_k1_MAT_s102
                                 + 1.41583 w_k1_MSFT_s102
                                 + 1.26641 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.904189 w_k1_ADM_s102
                                 + 0.87766 w_k1_AXP_s102
                                 + 1.03869 w_k1_DIS_s102
                                 + 0.920435 w_k1_FDX_s102
                                 + 1.08368 w_k1_IBM_s102
                                 + 1.06613 w_k1_INTC_s102 + 1.084 w_k1_KO_s102
                                 + 0.886148 w_k1_MAT_s102
                                 + 1.05639 w_k1_MSFT_s102
                                 + 1.05048 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.0374 w_k1_ADM_s103
                                 + 0.9992 w_k1_AXP_s103
                                 + 0.960503 w_k1_DIS_s103
                                 + 1.01963 w_k1_FDX_s103
                                 + 1.13634 w_k1_IBM_s103 + 1.013 w_k1_INTC_s103
                                 + 1.11351 w_k1_KO_s103 + 1.00063 w_k1_MAT_s103
                                 + 1.04499 w_k1_MSFT_s103
                                 + 1.01283 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.01125 w_k1_ADM_s103
                                 + 0.736794 w_k1_AXP_s103
                                 + 0.772508 w_k1_DIS_s103
                                 + 0.84445 w_k1_FDX_s103
                                 + 0.707736 w_k1_IBM_s103
                                 + 0.844953 w_k1_INTC_s103
                                 + 0.782244 w_k1_KO_s103
                                 + 0.503809 w_k1_MAT_s103
                                 + 0.814153 w_k1_MSFT_s103
                                 + 0.903259 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.02352 w_k1_ADM_s103
                                 + 1.01069 w_k1_AXP_s103
                                 + 1.16757 w_k1_DIS_s103
                                 + 1.07536 w_k1_FDX_s103
                                 + 0.798149 w_k1_IBM_s103
                                 + 1.05753 w_k1_INTC_s103
                                 + 1.05332 w_k1_KO_s103
                                 + 0.913039 w_k1_MAT_s103
                                 + 1.1492 w_k1_MSFT_s103 + 1.11978 w_k1_PG_s103
                                 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.30721 w_k1_ADM_s103
                                 + 1.46894 w_k1_AXP_s103
                                 + 1.68323 w_k1_DIS_s103
                                 + 1.24273 w_k1_FDX_s103
                                 + 1.32418 w_k1_IBM_s103
                                 + 1.2523 w_k1_INTC_s103 + 1.36803 w_k1_KO_s103
                                 + 1.45256 w_k1_MAT_s103
                                 + 1.32051 w_k1_MSFT_s103
                                 + 1.21518 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.640563 w_k1_ADM_s103
                                 + 0.616559 w_k1_AXP_s103
                                 + 0.520276 w_k1_DIS_s103
                                 + 0.731306 w_k1_FDX_s103
                                 + 0.610022 w_k1_IBM_s103
                                 + 0.724078 w_k1_INTC_s103
                                 + 0.645091 w_k1_KO_s103
                                 + 0.527344 w_k1_MAT_s103
                                 + 0.69838 w_k1_MSFT_s103 + 0.7652 w_k1_PG_s103
                                 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.835502 w_k1_ADM_s103
                                 + 0.661994 w_k1_AXP_s103
                                 + 0.844795 w_k1_DIS_s103
                                 + 0.799966 w_k1_FDX_s103
                                 + 0.934326 w_k1_IBM_s103
                                 + 0.860977 w_k1_INTC_s103
                                 + 0.881014 w_k1_KO_s103
                                 + 0.702589 w_k1_MAT_s103
                                 + 0.913144 w_k1_MSFT_s103
                                 + 0.900897 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.26633 w_k1_ADM_s103
                                 + 1.27007 w_k1_AXP_s103
                                 + 1.27054 w_k1_DIS_s103
                                 + 1.33176 w_k1_FDX_s103
                                 + 0.849499 w_k1_IBM_s103
                                 + 1.34314 w_k1_INTC_s103
                                 + 1.17004 w_k1_KO_s103 + 1.13853 w_k1_MAT_s103
                                 + 1.35159 w_k1_MSFT_s103
                                 + 1.23552 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.848073 w_k1_ADM_s103
                                 + 0.792087 w_k1_AXP_s103
                                 + 0.8825 w_k1_DIS_s103
                                 + 0.817719 w_k1_FDX_s103
                                 + 1.30244 w_k1_IBM_s103
                                 + 0.71468 w_k1_INTC_s103 + 1.0963 w_k1_KO_s103
                                 + 1.39496 w_k1_MAT_s103
                                 + 0.840661 w_k1_MSFT_s103
                                 + 1.03342 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.27526 w_k1_ADM_s104
                                 + 1.46359 w_k1_AXP_s104
                                 + 1.69991 w_k1_DIS_s104
                                 + 1.34886 w_k1_FDX_s104
                                 + 1.29169 w_k1_IBM_s104
                                 + 1.24414 w_k1_INTC_s104
                                 + 1.36214 w_k1_KO_s104 + 1.00417 w_k1_MAT_s104
                                 + 1.38007 w_k1_MSFT_s104
                                 + 1.08736 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.833538 w_k1_ADM_s104
                                 + 0.732861 w_k1_AXP_s104
                                 + 1.0073 w_k1_DIS_s104
                                 + 0.900043 w_k1_FDX_s104
                                 + 0.77599 w_k1_IBM_s104
                                 + 0.774411 w_k1_INTC_s104
                                 + 0.910703 w_k1_KO_s104
                                 + 0.799431 w_k1_MAT_s104
                                 + 0.967156 w_k1_MSFT_s104
                                 + 0.801906 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.77634 w_k1_ADM_s104
                                 + 0.97902 w_k1_AXP_s104
                                 + 1.08415 w_k1_DIS_s104
                                 + 0.944653 w_k1_FDX_s104
                                 + 1.10729 w_k1_IBM_s104
                                 + 0.873267 w_k1_INTC_s104
                                 + 1.07637 w_k1_KO_s104
                                 + 0.947839 w_k1_MAT_s104
                                 + 1.04629 w_k1_MSFT_s104
                                 + 0.969177 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.15923 w_k1_ADM_s104
                                 + 0.941796 w_k1_AXP_s104
                                 + 1.07948 w_k1_DIS_s104
                                 + 1.12803 w_k1_FDX_s104
                                 + 0.664346 w_k1_IBM_s104
                                 + 0.874021 w_k1_INTC_s104
                                 + 0.883031 w_k1_KO_s104
                                 + 0.766421 w_k1_MAT_s104
                                 + 1.10806 w_k1_MSFT_s104
                                 + 0.942379 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.668788 w_k1_ADM_s104
                                 + 0.791952 w_k1_AXP_s104
                                 + 0.660082 w_k1_DIS_s104
                                 + 0.771739 w_k1_FDX_s104
                                 + 0.822303 w_k1_IBM_s104
                                 + 0.766391 w_k1_INTC_s104
                                 + 0.748151 w_k1_KO_s104
                                 + 0.899876 w_k1_MAT_s104
                                 + 0.81178 w_k1_MSFT_s104
                                 + 0.877319 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.02809 w_k1_ADM_s104
                                 + 1.09544 w_k1_AXP_s104
                                 + 0.856183 w_k1_DIS_s104
                                 + 0.939856 w_k1_FDX_s104
                                 + 0.800278 w_k1_IBM_s104
                                 + 0.931125 w_k1_INTC_s104
                                 + 0.77062 w_k1_KO_s104
                                 + 0.800923 w_k1_MAT_s104
                                 + 0.98475 w_k1_MSFT_s104
                                 + 0.854945 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.16802 w_k1_ADM_s104
                                 + 1.06235 w_k1_AXP_s104
                                 + 1.06062 w_k1_DIS_s104
                                 + 1.09233 w_k1_FDX_s104
                                 + 0.918428 w_k1_IBM_s104
                                 + 1.23484 w_k1_INTC_s104
                                 + 1.03193 w_k1_KO_s104
                                 + 0.656515 w_k1_MAT_s104
                                 + 1.1726 w_k1_MSFT_s104
                                 + 0.960025 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.770373 w_k1_ADM_s104
                                 + 0.949213 w_k1_AXP_s104
                                 + 1.18444 w_k1_DIS_s104
                                 + 0.953688 w_k1_FDX_s104
                                 + 1.38271 w_k1_IBM_s104
                                 + 0.906028 w_k1_INTC_s104
                                 + 1.20997 w_k1_KO_s104 + 1.3372 w_k1_MAT_s104
                                 + 1.04988 w_k1_MSFT_s104
                                 + 1.02061 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.89486 w_k1_ADM_s105
                                 + 0.760632 w_k1_AXP_s105
                                 + 1.01447 w_k1_DIS_s105
                                 + 0.953849 w_k1_FDX_s105
                                 + 1.01006 w_k1_IBM_s105
                                 + 1.0869 w_k1_INTC_s105 + 1.02638 w_k1_KO_s105
                                 + 0.998716 w_k1_MAT_s105
                                 + 1.05125 w_k1_MSFT_s105
                                 + 0.98562 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.18021 w_k1_ADM_s105
                                 + 0.950606 w_k1_AXP_s105
                                 + 1.27714 w_k1_DIS_s105
                                 + 1.33256 w_k1_FDX_s105
                                 + 0.779429 w_k1_IBM_s105
                                 + 0.995136 w_k1_INTC_s105
                                 + 0.939758 w_k1_KO_s105
                                 + 0.719939 w_k1_MAT_s105
                                 + 1.23475 w_k1_MSFT_s105
                                 + 1.06536 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.22918 w_k1_ADM_s105
                                 + 1.26247 w_k1_AXP_s105
                                 + 1.56758 w_k1_DIS_s105 + 1.1379 w_k1_FDX_s105
                                 + 1.71136 w_k1_IBM_s105
                                 + 0.984776 w_k1_INTC_s105
                                 + 1.38376 w_k1_KO_s105 + 1.77356 w_k1_MAT_s105
                                 + 1.07613 w_k1_MSFT_s105
                                 + 1.14709 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.01248 w_k1_ADM_s105
                                 + 0.810715 w_k1_AXP_s105
                                 + 0.90381 w_k1_DIS_s105
                                 + 0.872856 w_k1_FDX_s105
                                 + 1.38981 w_k1_IBM_s105
                                 + 0.714391 w_k1_INTC_s105
                                 + 1.14464 w_k1_KO_s105 + 1.39034 w_k1_MAT_s105
                                 + 0.953857 w_k1_MSFT_s105
                                 + 0.98969 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.784527 w_k1_ADM_s105
                                 + 0.725269 w_k1_AXP_s105
                                 + 0.7206 w_k1_DIS_s105
                                 + 0.828824 w_k1_FDX_s105
                                 + 0.67228 w_k1_IBM_s105
                                 + 0.910815 w_k1_INTC_s105
                                 + 0.789914 w_k1_KO_s105
                                 + 0.507422 w_k1_MAT_s105
                                 + 0.831254 w_k1_MSFT_s105
                                 + 0.844076 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.27958 w_k1_ADM_s105
                                 + 1.10188 w_k1_AXP_s105
                                 + 0.88441 w_k1_DIS_s105
                                 + 0.968951 w_k1_FDX_s105
                                 + 1.11136 w_k1_IBM_s105
                                 + 0.895068 w_k1_INTC_s105
                                 + 1.05943 w_k1_KO_s105 + 1.26719 w_k1_MAT_s105
                                 + 0.912863 w_k1_MSFT_s105
                                 + 1.0604 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.4093 w_k1_ADM_s105
                                 + 1.21827 w_k1_AXP_s105
                                 + 1.46468 w_k1_DIS_s105
                                 + 1.24304 w_k1_FDX_s105
                                 + 1.16476 w_k1_IBM_s105
                                 + 1.00814 w_k1_INTC_s105
                                 + 1.21429 w_k1_KO_s105 + 1.06534 w_k1_MAT_s105
                                 + 1.17415 w_k1_MSFT_s105
                                 + 1.12572 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.06507 w_k1_ADM_s105
                                 + 0.959005 w_k1_AXP_s105
                                 + 1.04295 w_k1_DIS_s105
                                 + 1.04861 w_k1_FDX_s105
                                 + 0.854232 w_k1_IBM_s105
                                 + 0.971912 w_k1_INTC_s105
                                 + 0.906713 w_k1_KO_s105
                                 + 0.775322 w_k1_MAT_s105
                                 + 0.951197 w_k1_MSFT_s105
                                 + 0.893228 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.10034 w_k1_ADM_s106
                                 + 1.26737 w_k1_AXP_s106
                                 + 1.32826 w_k1_DIS_s106
                                 + 1.06749 w_k1_FDX_s106
                                 + 1.42708 w_k1_IBM_s106
                                 + 1.13578 w_k1_INTC_s106
                                 + 1.33404 w_k1_KO_s106 + 1.27592 w_k1_MAT_s106
                                 + 1.27177 w_k1_MSFT_s106 + 1.1686 w_k1_PG_s106
                                 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.03841 w_k1_ADM_s106
                                 + 0.989567 w_k1_AXP_s106
                                 + 0.931691 w_k1_DIS_s106
                                 + 0.969278 w_k1_FDX_s106
                                 + 0.985767 w_k1_IBM_s106
                                 + 0.821125 w_k1_INTC_s106
                                 + 1.01225 w_k1_KO_s106 + 1.08605 w_k1_MAT_s106
                                 + 0.980618 w_k1_MSFT_s106
                                 + 1.08875 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.68948 w_k1_ADM_s106
                                 + 1.55794 w_k1_AXP_s106
                                 + 1.86973 w_k1_DIS_s106
                                 + 1.39108 w_k1_FDX_s106
                                 + 1.44678 w_k1_IBM_s106
                                 + 1.32462 w_k1_INTC_s106
                                 + 1.66215 w_k1_KO_s106 + 1.65644 w_k1_MAT_s106
                                 + 1.51277 w_k1_MSFT_s106
                                 + 1.43238 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.03973 w_k1_ADM_s106
                                 + 1.02617 w_k1_AXP_s106
                                 + 0.859041 w_k1_DIS_s106
                                 + 0.795952 w_k1_FDX_s106
                                 + 1.3329 w_k1_IBM_s106
                                 + 0.809714 w_k1_INTC_s106
                                 + 1.15499 w_k1_KO_s106 + 1.45577 w_k1_MAT_s106
                                 + 0.858986 w_k1_MSFT_s106
                                 + 1.09685 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.85024 w_k1_ADM_s106
                                 + 0.74536 w_k1_AXP_s106 + 0.809 w_k1_DIS_s106
                                 + 0.784981 w_k1_FDX_s106
                                 + 0.863371 w_k1_IBM_s106
                                 + 0.813903 w_k1_INTC_s106
                                 + 0.830845 w_k1_KO_s106
                                 + 0.820142 w_k1_MAT_s106
                                 + 0.871763 w_k1_MSFT_s106
                                 + 1.0004 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.19058 w_k1_ADM_s106
                                 + 1.16541 w_k1_AXP_s106
                                 + 1.33962 w_k1_DIS_s106
                                 + 1.16984 w_k1_FDX_s106
                                 + 0.888061 w_k1_IBM_s106
                                 + 0.824019 w_k1_INTC_s106
                                 + 1.05337 w_k1_KO_s106 + 0.93165 w_k1_MAT_s106
                                 + 1.20352 w_k1_MSFT_s106
                                 + 1.16027 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.971239 w_k1_ADM_s106
                                 + 0.657365 w_k1_AXP_s106
                                 + 0.900032 w_k1_DIS_s106
                                 + 0.908843 w_k1_FDX_s106
                                 + 0.659136 w_k1_IBM_s106
                                 + 0.792136 w_k1_INTC_s106
                                 + 0.860451 w_k1_KO_s106
                                 + 0.577301 w_k1_MAT_s106
                                 + 0.950391 w_k1_MSFT_s106
                                 + 1.0172 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.951482 w_k1_ADM_s106
                                 + 0.920255 w_k1_AXP_s106
                                 + 1.22763 w_k1_DIS_s106
                                 + 1.01596 w_k1_FDX_s106
                                 + 1.12462 w_k1_IBM_s106
                                 + 1.03672 w_k1_INTC_s106 + 1.16 w_k1_KO_s106
                                 + 0.866283 w_k1_MAT_s106
                                 + 1.19332 w_k1_MSFT_s106
                                 + 1.16075 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.07965 w_k1_ADM_s107
                                 + 1.05321 w_k1_AXP_s107
                                 + 0.842658 w_k1_DIS_s107
                                 + 0.929448 w_k1_FDX_s107
                                 + 0.776396 w_k1_IBM_s107
                                 + 0.715242 w_k1_INTC_s107
                                 + 0.755936 w_k1_KO_s107
                                 + 0.982381 w_k1_MAT_s107
                                 + 0.81265 w_k1_MSFT_s107
                                 + 1.07997 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.800164 w_k1_ADM_s107
                                 + 0.730383 w_k1_AXP_s107
                                 + 0.629487 w_k1_DIS_s107
                                 + 0.721845 w_k1_FDX_s107
                                 + 0.753179 w_k1_IBM_s107
                                 + 0.670878 w_k1_INTC_s107
                                 + 0.749841 w_k1_KO_s107
                                 + 0.808459 w_k1_MAT_s107
                                 + 0.789163 w_k1_MSFT_s107
                                 + 0.845481 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.02812 w_k1_ADM_s107
                                 + 0.969297 w_k1_AXP_s107
                                 + 0.880948 w_k1_DIS_s107
                                 + 1.05649 w_k1_FDX_s107
                                 + 0.734669 w_k1_IBM_s107
                                 + 0.78663 w_k1_INTC_s107
                                 + 0.697099 w_k1_KO_s107
                                 + 0.998294 w_k1_MAT_s107
                                 + 0.989961 w_k1_MSFT_s107
                                 + 1.01203 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.08313 w_k1_ADM_s107
                                 + 0.98854 w_k1_AXP_s107
                                 + 0.811334 w_k1_DIS_s107
                                 + 0.846185 w_k1_FDX_s107
                                 + 1.28888 w_k1_IBM_s107
                                 + 0.703726 w_k1_INTC_s107
                                 + 0.989226 w_k1_KO_s107
                                 + 1.33942 w_k1_MAT_s107
                                 + 0.923649 w_k1_MSFT_s107
                                 + 1.05126 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.09365 w_k1_ADM_s107
                                 + 1.29158 w_k1_AXP_s107
                                 + 1.41198 w_k1_DIS_s107
                                 + 1.19848 w_k1_FDX_s107
                                 + 1.44659 w_k1_IBM_s107
                                 + 1.11385 w_k1_INTC_s107
                                 + 1.31081 w_k1_KO_s107 + 1.65697 w_k1_MAT_s107
                                 + 1.08323 w_k1_MSFT_s107 + 1.2228 w_k1_PG_s107
                                 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.26045 w_k1_ADM_s107
                                 + 1.2501 w_k1_AXP_s107 + 1.4594 w_k1_DIS_s107
                                 + 1.27556 w_k1_FDX_s107
                                 + 0.904532 w_k1_IBM_s107
                                 + 1.03293 w_k1_INTC_s107
                                 + 0.94047 w_k1_KO_s107 + 1.01039 w_k1_MAT_s107
                                 + 1.23372 w_k1_MSFT_s107
                                 + 1.13503 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.947143 w_k1_ADM_s107
                                 + 0.643769 w_k1_AXP_s107
                                 + 0.698167 w_k1_DIS_s107
                                 + 0.770215 w_k1_FDX_s107
                                 + 0.908768 w_k1_IBM_s107
                                 + 0.730241 w_k1_INTC_s107
                                 + 0.843276 w_k1_KO_s107
                                 + 0.853407 w_k1_MAT_s107
                                 + 0.87846 w_k1_MSFT_s107 + 0.8493 w_k1_PG_s107
                                 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.15925 w_k1_ADM_s107
                                 + 0.909434 w_k1_AXP_s107
                                 + 1.00708 w_k1_DIS_s107
                                 + 0.987693 w_k1_FDX_s107
                                 + 1.07542 w_k1_IBM_s107
                                 + 0.764589 w_k1_INTC_s107
                                 + 0.977868 w_k1_KO_s107
                                 + 1.17641 w_k1_MAT_s107
                                 + 0.910869 w_k1_MSFT_s107
                                 + 1.11561 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.13945 w_k1_ADM_s110
                                 + 0.717987 w_k1_AXP_s110
                                 + 0.630047 w_k1_DIS_s110
                                 + 0.833862 w_k1_FDX_s110
                                 + 0.695703 w_k1_IBM_s110
                                 + 0.793464 w_k1_INTC_s110
                                 + 0.707857 w_k1_KO_s110
                                 + 0.613914 w_k1_MAT_s110
                                 + 0.687713 w_k1_MSFT_s110
                                 + 0.841189 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.17695 w_k1_ADM_s110
                                 + 1.23772 w_k1_AXP_s110
                                 + 0.913351 w_k1_DIS_s110
                                 + 1.01308 w_k1_FDX_s110
                                 + 1.16128 w_k1_IBM_s110
                                 + 0.934831 w_k1_INTC_s110
                                 + 1.11789 w_k1_KO_s110 + 1.20543 w_k1_MAT_s110
                                 + 0.914273 w_k1_MSFT_s110
                                 + 0.955623 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.01864 w_k1_ADM_s110
                                 + 0.85706 w_k1_AXP_s110 + 1.1069 w_k1_DIS_s110
                                 + 1.13359 w_k1_FDX_s110
                                 + 0.924939 w_k1_IBM_s110
                                 + 1.02612 w_k1_INTC_s110
                                 + 1.01096 w_k1_KO_s110
                                 + 0.868769 w_k1_MAT_s110
                                 + 1.04336 w_k1_MSFT_s110
                                 + 0.941341 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.38853 w_k1_ADM_s110
                                 + 1.15258 w_k1_AXP_s110
                                 + 1.19681 w_k1_DIS_s110
                                 + 1.15514 w_k1_FDX_s110
                                 + 1.56968 w_k1_IBM_s110
                                 + 1.18975 w_k1_INTC_s110
                                 + 1.39349 w_k1_KO_s110 + 1.58492 w_k1_MAT_s110
                                 + 1.13492 w_k1_MSFT_s110
                                 + 1.06406 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.752031 w_k1_ADM_s110
                                 + 0.753298 w_k1_AXP_s110
                                 + 0.892974 w_k1_DIS_s110
                                 + 0.780301 w_k1_FDX_s110
                                 + 1.3353 w_k1_IBM_s110
                                 + 0.794979 w_k1_INTC_s110
                                 + 1.05828 w_k1_KO_s110 + 1.44647 w_k1_MAT_s110
                                 + 0.79644 w_k1_MSFT_s110
                                 + 0.949032 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.826407 w_k1_ADM_s110
                                 + 0.655069 w_k1_AXP_s110
                                 + 0.687493 w_k1_DIS_s110
                                 + 0.726714 w_k1_FDX_s110
                                 + 0.981622 w_k1_IBM_s110
                                 + 0.655093 w_k1_INTC_s110
                                 + 0.877147 w_k1_KO_s110
                                 + 1.17863 w_k1_MAT_s110
                                 + 0.780444 w_k1_MSFT_s110
                                 + 0.826358 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.10819 w_k1_ADM_s110
                                 + 0.839402 w_k1_AXP_s110
                                 + 0.903839 w_k1_DIS_s110
                                 + 0.960057 w_k1_FDX_s110
                                 + 0.923995 w_k1_IBM_s110
                                 + 0.862559 w_k1_INTC_s110
                                 + 0.889423 w_k1_KO_s110
                                 + 0.744258 w_k1_MAT_s110
                                 + 0.898865 w_k1_MSFT_s110
                                 + 0.981644 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.5523 w_k1_ADM_s110
                                 + 1.49401 w_k1_AXP_s110 + 1.3103 w_k1_DIS_s110
                                 + 1.32803 w_k1_FDX_s110
                                 + 0.923399 w_k1_IBM_s110
                                 + 0.991598 w_k1_INTC_s110
                                 + 1.05592 w_k1_KO_s110 + 1.44134 w_k1_MAT_s110
                                 + 1.13654 w_k1_MSFT_s110
                                 + 1.01977 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.12692 w_k1_ADM_s111
                                 + 1.07139 w_k1_AXP_s111
                                 + 1.02219 w_k1_DIS_s111
                                 + 1.18653 w_k1_FDX_s111
                                 + 0.684731 w_k1_IBM_s111
                                 + 1.14048 w_k1_INTC_s111
                                 + 0.864008 w_k1_KO_s111
                                 + 0.595275 w_k1_MAT_s111
                                 + 0.94552 w_k1_MSFT_s111
                                 + 0.99778 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.977958 w_k1_ADM_s111
                                 + 1.21091 w_k1_AXP_s111
                                 + 0.88393 w_k1_DIS_s111
                                 + 0.973082 w_k1_FDX_s111
                                 + 1.45242 w_k1_IBM_s111
                                 + 0.938117 w_k1_INTC_s111
                                 + 1.15537 w_k1_KO_s111 + 1.65031 w_k1_MAT_s111
                                 + 0.8515 w_k1_MSFT_s111 + 1.09285 w_k1_PG_s111
                                 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.973406 w_k1_ADM_s111
                                 + 0.81096 w_k1_AXP_s111
                                 + 0.685202 w_k1_DIS_s111
                                 + 0.93594 w_k1_FDX_s111
                                 + 0.697338 w_k1_IBM_s111
                                 + 0.936891 w_k1_INTC_s111
                                 + 0.667348 w_k1_KO_s111
                                 + 0.513198 w_k1_MAT_s111
                                 + 0.873458 w_k1_MSFT_s111
                                 + 0.888669 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.777118 w_k1_ADM_s111
                                 + 0.761937 w_k1_AXP_s111
                                 + 1.06819 w_k1_DIS_s111
                                 + 0.922002 w_k1_FDX_s111
                                 + 0.969148 w_k1_IBM_s111
                                 + 0.966819 w_k1_INTC_s111
                                 + 1.00407 w_k1_KO_s111 + 0.99758 w_k1_MAT_s111
                                 + 0.991425 w_k1_MSFT_s111
                                 + 1.00352 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.20731 w_k1_ADM_s111
                                 + 1.0972 w_k1_AXP_s111 + 1.2632 w_k1_DIS_s111
                                 + 1.25378 w_k1_FDX_s111
                                 + 1.04724 w_k1_IBM_s111
                                 + 1.11522 w_k1_INTC_s111
                                 + 1.14412 w_k1_KO_s111 + 1.10324 w_k1_MAT_s111
                                 + 1.11863 w_k1_MSFT_s111
                                 + 1.30242 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.971274 w_k1_ADM_s111
                                 + 0.74793 w_k1_AXP_s111
                                 + 0.933309 w_k1_DIS_s111
                                 + 0.858722 w_k1_FDX_s111
                                 + 1.23378 w_k1_IBM_s111
                                 + 0.888446 w_k1_INTC_s111
                                 + 1.03928 w_k1_KO_s111 + 1.01412 w_k1_MAT_s111
                                 + 0.938867 w_k1_MSFT_s111
                                 + 1.10804 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.989437 w_k1_ADM_s111
                                 + 0.961637 w_k1_AXP_s111
                                 + 0.76503 w_k1_DIS_s111
                                 + 0.923143 w_k1_FDX_s111
                                 + 0.984321 w_k1_IBM_s111
                                 + 0.960267 w_k1_INTC_s111
                                 + 0.891659 w_k1_KO_s111
                                 + 0.863944 w_k1_MAT_s111
                                 + 0.814548 w_k1_MSFT_s111
                                 + 0.979683 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.41634 w_k1_ADM_s111
                                 + 1.59568 w_k1_AXP_s111
                                 + 1.40168 w_k1_DIS_s111 + 1.2409 w_k1_FDX_s111
                                 + 1.52041 w_k1_IBM_s111
                                 + 1.23157 w_k1_INTC_s111
                                 + 1.39023 w_k1_KO_s111 + 1.36887 w_k1_MAT_s111
                                 + 1.14004 w_k1_MSFT_s111 + 1.2057 w_k1_PG_s111
                                 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.30343 w_k1_ADM_s112
                                 + 1.39336 w_k1_AXP_s112
                                 + 1.18512 w_k1_DIS_s112
                                 + 1.22842 w_k1_FDX_s112
                                 + 1.49518 w_k1_IBM_s112
                                 + 1.34559 w_k1_INTC_s112
                                 + 1.47865 w_k1_KO_s112 + 1.95017 w_k1_MAT_s112
                                 + 1.13285 w_k1_MSFT_s112 + 1.4207 w_k1_PG_s112
                                 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.99176 w_k1_ADM_s112
                                 + 1.6081 w_k1_AXP_s112 + 1.45217 w_k1_DIS_s112
                                 + 1.50565 w_k1_FDX_s112
                                 + 0.926027 w_k1_IBM_s112
                                 + 1.72106 w_k1_INTC_s112
                                 + 1.24082 w_k1_KO_s112 + 1.19069 w_k1_MAT_s112
                                 + 1.35207 w_k1_MSFT_s112
                                 + 1.35072 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.30407 w_k1_ADM_s112
                                 + 0.938074 w_k1_AXP_s112
                                 + 0.623689 w_k1_DIS_s112
                                 + 0.814406 w_k1_FDX_s112
                                 + 0.939776 w_k1_IBM_s112
                                 + 0.852618 w_k1_INTC_s112
                                 + 0.75625 w_k1_KO_s112 + 1.18426 w_k1_MAT_s112
                                 + 0.764591 w_k1_MSFT_s112
                                 + 0.989334 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.35109 w_k1_ADM_s112
                                 + 0.85471 w_k1_AXP_s112
                                 + 1.14194 w_k1_DIS_s112
                                 + 1.05278 w_k1_FDX_s112
                                 + 0.925704 w_k1_IBM_s112
                                 + 1.44387 w_k1_INTC_s112
                                 + 1.11619 w_k1_KO_s112 + 1.00008 w_k1_MAT_s112
                                 + 1.15836 w_k1_MSFT_s112
                                 + 1.17614 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.652459 w_k1_ADM_s112
                                 + 0.675586 w_k1_AXP_s112
                                 + 0.51256 w_k1_DIS_s112
                                 + 0.593557 w_k1_FDX_s112
                                 + 1.20817 w_k1_IBM_s112
                                 + 0.884241 w_k1_INTC_s112
                                 + 0.836963 w_k1_KO_s112
                                 + 1.37404 w_k1_MAT_s112
                                 + 0.631325 w_k1_MSFT_s112
                                 + 0.904141 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.42277 w_k1_ADM_s112
                                 + 1.36864 w_k1_AXP_s112
                                 + 0.865281 w_k1_DIS_s112
                                 + 1.21004 w_k1_FDX_s112
                                 + 0.778452 w_k1_IBM_s112
                                 + 1.65021 w_k1_INTC_s112
                                 + 0.830391 w_k1_KO_s112
                                 + 0.91296 w_k1_MAT_s112
                                 + 1.13264 w_k1_MSFT_s112
                                 + 1.05566 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.983855 w_k1_ADM_s112
                                 + 0.742103 w_k1_AXP_s112
                                 + 0.644309 w_k1_DIS_s112
                                 + 0.781238 w_k1_FDX_s112
                                 + 0.694043 w_k1_IBM_s112
                                 + 1.04828 w_k1_INTC_s112
                                 + 0.681116 w_k1_KO_s112
                                 + 0.666635 w_k1_MAT_s112
                                 + 0.68807 w_k1_MSFT_s112
                                 + 0.870752 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.26879 w_k1_ADM_s112
                                 + 0.97756 w_k1_AXP_s112
                                 + 0.98555 w_k1_DIS_s112
                                 + 0.890682 w_k1_FDX_s112
                                 + 1.13335 w_k1_IBM_s112
                                 + 1.28413 w_k1_INTC_s112
                                 + 1.15034 w_k1_KO_s112 + 1.18404 w_k1_MAT_s112
                                 + 1.04139 w_k1_MSFT_s112
                                 + 1.02301 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.836201 w_k1_ADM_s113
                                 + 0.783846 w_k1_AXP_s113
                                 + 0.964201 w_k1_DIS_s113
                                 + 0.886432 w_k1_FDX_s113
                                 + 1.07131 w_k1_IBM_s113
                                 + 0.952583 w_k1_INTC_s113
                                 + 1.09741 w_k1_KO_s113 + 1.0558 w_k1_MAT_s113
                                 + 0.862067 w_k1_MSFT_s113
                                 + 0.942915 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.91823 w_k1_ADM_s113
                                 + 1.06514 w_k1_AXP_s113
                                 + 1.17895 w_k1_DIS_s113
                                 + 1.07329 w_k1_FDX_s113
                                 + 1.48726 w_k1_IBM_s113
                                 + 1.06417 w_k1_INTC_s113
                                 + 1.29131 w_k1_KO_s113 + 1.34796 w_k1_MAT_s113
                                 + 0.881368 w_k1_MSFT_s113
                                 + 1.13026 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.927707 w_k1_ADM_s113
                                 + 0.717872 w_k1_AXP_s113
                                 + 0.811933 w_k1_DIS_s113
                                 + 0.808191 w_k1_FDX_s113
                                 + 0.89036 w_k1_IBM_s113
                                 + 0.786249 w_k1_INTC_s113
                                 + 0.802838 w_k1_KO_s113
                                 + 0.829246 w_k1_MAT_s113
                                 + 0.737046 w_k1_MSFT_s113
                                 + 0.838642 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.930622 w_k1_ADM_s113
                                 + 1.4 w_k1_AXP_s113 + 1.52688 w_k1_DIS_s113
                                 + 1.33338 w_k1_FDX_s113
                                 + 1.15498 w_k1_IBM_s113
                                 + 1.2274 w_k1_INTC_s113 + 1.1898 w_k1_KO_s113
                                 + 1.07833 w_k1_MAT_s113
                                 + 1.03896 w_k1_MSFT_s113
                                 + 1.16201 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.897916 w_k1_ADM_s113
                                 + 0.903908 w_k1_AXP_s113
                                 + 0.776632 w_k1_DIS_s113
                                 + 0.905588 w_k1_FDX_s113
                                 + 0.813391 w_k1_IBM_s113
                                 + 0.925456 w_k1_INTC_s113
                                 + 0.790044 w_k1_KO_s113
                                 + 0.695681 w_k1_MAT_s113
                                 + 0.830438 w_k1_MSFT_s113
                                 + 0.838054 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.08303 w_k1_ADM_s113
                                 + 1.05342 w_k1_AXP_s113
                                 + 0.791714 w_k1_DIS_s113
                                 + 0.959563 w_k1_FDX_s113
                                 + 1.17254 w_k1_IBM_s113
                                 + 0.73737 w_k1_INTC_s113
                                 + 0.893086 w_k1_KO_s113
                                 + 1.19412 w_k1_MAT_s113
                                 + 0.825565 w_k1_MSFT_s113
                                 + 0.936814 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.97755 w_k1_ADM_s113
                                 + 1.09782 w_k1_AXP_s113
                                 + 1.13324 w_k1_DIS_s113
                                 + 1.17425 w_k1_FDX_s113
                                 + 0.800701 w_k1_IBM_s113
                                 + 1.25728 w_k1_INTC_s113
                                 + 1.03341 w_k1_KO_s113
                                 + 0.985003 w_k1_MAT_s113
                                 + 1.16256 w_k1_MSFT_s113
                                 + 1.01699 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.844423 w_k1_ADM_s113
                                 + 1.10293 w_k1_AXP_s113
                                 + 0.714371 w_k1_DIS_s113
                                 + 0.758618 w_k1_FDX_s113
                                 + 1.40782 w_k1_IBM_s113
                                 + 0.904279 w_k1_INTC_s113
                                 + 0.964215 w_k1_KO_s113
                                 + 1.23526 w_k1_MAT_s113
                                 + 0.773103 w_k1_MSFT_s113
                                 + 0.860629 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.14119 w_k1_ADM_s114
                                 + 1.61033 w_k1_AXP_s114
                                 + 1.68803 w_k1_DIS_s114
                                 + 1.31673 w_k1_FDX_s114
                                 + 1.55615 w_k1_IBM_s114
                                 + 1.26299 w_k1_INTC_s114
                                 + 1.49257 w_k1_KO_s114 + 1.31632 w_k1_MAT_s114
                                 + 1.38993 w_k1_MSFT_s114 + 1.2459 w_k1_PG_s114
                                 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.10252 w_k1_ADM_s114
                                 + 0.862762 w_k1_AXP_s114
                                 + 0.836668 w_k1_DIS_s114
                                 + 0.941927 w_k1_FDX_s114
                                 + 0.95809 w_k1_IBM_s114
                                 + 0.923716 w_k1_INTC_s114
                                 + 0.912725 w_k1_KO_s114
                                 + 0.743527 w_k1_MAT_s114
                                 + 0.912548 w_k1_MSFT_s114
                                 + 0.956215 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.11829 w_k1_ADM_s114
                                 + 1.27223 w_k1_AXP_s114
                                 + 1.05334 w_k1_DIS_s114
                                 + 1.00219 w_k1_FDX_s114
                                 + 1.63375 w_k1_IBM_s114
                                 + 1.01882 w_k1_INTC_s114 + 1.4139 w_k1_KO_s114
                                 + 1.68499 w_k1_MAT_s114
                                 + 1.00954 w_k1_MSFT_s114
                                 + 1.20338 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.908074 w_k1_ADM_s114
                                 + 0.765897 w_k1_AXP_s114
                                 + 0.818542 w_k1_DIS_s114
                                 + 0.847849 w_k1_FDX_s114
                                 + 0.725078 w_k1_IBM_s114
                                 + 0.740187 w_k1_INTC_s114
                                 + 0.844598 w_k1_KO_s114
                                 + 0.684687 w_k1_MAT_s114
                                 + 0.901541 w_k1_MSFT_s114
                                 + 0.931104 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.812353 w_k1_ADM_s114
                                 + 0.912457 w_k1_AXP_s114
                                 + 0.940637 w_k1_DIS_s114
                                 + 0.888701 w_k1_FDX_s114
                                 + 1.11246 w_k1_IBM_s114
                                 + 0.811774 w_k1_INTC_s114
                                 + 1.23771 w_k1_KO_s114 + 1.14937 w_k1_MAT_s114
                                 + 0.993944 w_k1_MSFT_s114 + 1.151 w_k1_PG_s114
                                 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.15043 w_k1_ADM_s114
                                 + 1.01061 w_k1_AXP_s114
                                 + 1.21349 w_k1_DIS_s114
                                 + 1.03768 w_k1_FDX_s114
                                 + 1.33916 w_k1_IBM_s114
                                 + 0.914209 w_k1_INTC_s114
                                 + 1.14829 w_k1_KO_s114 + 1.33368 w_k1_MAT_s114
                                 + 1.03711 w_k1_MSFT_s114
                                 + 1.12696 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.03275 w_k1_ADM_s114
                                 + 1.07001 w_k1_AXP_s114
                                 + 1.16252 w_k1_DIS_s114
                                 + 1.19598 w_k1_FDX_s114
                                 + 0.847716 w_k1_IBM_s114
                                 + 1.02553 w_k1_INTC_s114
                                 + 0.920805 w_k1_KO_s114
                                 + 0.986989 w_k1_MAT_s114
                                 + 1.22304 w_k1_MSFT_s114
                                 + 1.04177 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.66576 w_k1_ADM_s114
                                 + 1.28218 w_k1_AXP_s114
                                 + 1.54535 w_k1_DIS_s114
                                 + 1.42846 w_k1_FDX_s114 + 1.0406 w_k1_IBM_s114
                                 + 0.988726 w_k1_INTC_s114
                                 + 1.3265 w_k1_KO_s114 + 1.24023 w_k1_MAT_s114
                                 + 1.2958 w_k1_MSFT_s114 + 1.20364 w_k1_PG_s114
                                 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.16959 w_k1_ADM_s115
                                 + 1.16177 w_k1_AXP_s115
                                 + 1.29815 w_k1_DIS_s115
                                 + 1.08372 w_k1_FDX_s115
                                 + 1.08861 w_k1_IBM_s115
                                 + 1.23288 w_k1_INTC_s115
                                 + 1.23935 w_k1_KO_s115
                                 + 0.911449 w_k1_MAT_s115
                                 + 1.22447 w_k1_MSFT_s115
                                 + 1.05572 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.853553 w_k1_ADM_s115
                                 + 0.725388 w_k1_AXP_s115
                                 + 0.58213 w_k1_DIS_s115
                                 + 0.675239 w_k1_FDX_s115
                                 + 0.908374 w_k1_IBM_s115
                                 + 0.975282 w_k1_INTC_s115
                                 + 0.866072 w_k1_KO_s115
                                 + 0.74501 w_k1_MAT_s115
                                 + 0.76938 w_k1_MSFT_s115
                                 + 0.810651 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.03718 w_k1_ADM_s115
                                 + 0.956129 w_k1_AXP_s115
                                 + 1.18715 w_k1_DIS_s115
                                 + 0.882638 w_k1_FDX_s115
                                 + 1.46051 w_k1_IBM_s115
                                 + 0.881341 w_k1_INTC_s115
                                 + 1.31223 w_k1_KO_s115 + 1.07678 w_k1_MAT_s115
                                 + 0.914096 w_k1_MSFT_s115
                                 + 1.06487 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.824534 w_k1_ADM_s115
                                 + 0.821906 w_k1_AXP_s115
                                 + 0.753142 w_k1_DIS_s115
                                 + 0.721508 w_k1_FDX_s115
                                 + 1.37226 w_k1_IBM_s115
                                 + 0.784821 w_k1_INTC_s115
                                 + 0.905967 w_k1_KO_s115
                                 + 0.987574 w_k1_MAT_s115
                                 + 0.639223 w_k1_MSFT_s115
                                 + 0.871116 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.03051 w_k1_ADM_s115
                                 + 1.50892 w_k1_AXP_s115
                                 + 1.69743 w_k1_DIS_s115
                                 + 1.26269 w_k1_FDX_s115
                                 + 1.50438 w_k1_IBM_s115
                                 + 1.26789 w_k1_INTC_s115
                                 + 1.64257 w_k1_KO_s115 + 1.40799 w_k1_MAT_s115
                                 + 1.19196 w_k1_MSFT_s115
                                 + 1.24123 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.5047 w_k1_ADM_s115
                                 + 1.60553 w_k1_AXP_s115
                                 + 1.61737 w_k1_DIS_s115
                                 + 1.37087 w_k1_FDX_s115 + 1.0032 w_k1_IBM_s115
                                 + 1.42174 w_k1_INTC_s115
                                 + 1.31075 w_k1_KO_s115 + 1.13798 w_k1_MAT_s115
                                 + 1.29786 w_k1_MSFT_s115 + 1.1414 w_k1_PG_s115
                                 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.1835 w_k1_ADM_s115
                                 + 0.873965 w_k1_AXP_s115
                                 + 1.01848 w_k1_DIS_s115
                                 + 0.946588 w_k1_FDX_s115
                                 + 0.719302 w_k1_IBM_s115
                                 + 0.967298 w_k1_INTC_s115
                                 + 0.992298 w_k1_KO_s115
                                 + 0.583093 w_k1_MAT_s115
                                 + 1.01497 w_k1_MSFT_s115
                                 + 0.876946 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.03398 w_k1_ADM_s115
                                 + 0.836814 w_k1_AXP_s115
                                 + 0.944641 w_k1_DIS_s115
                                 + 0.883946 w_k1_FDX_s115
                                 + 0.965743 w_k1_IBM_s115
                                 + 0.926174 w_k1_INTC_s115
                                 + 1.01047 w_k1_KO_s115
                                 + 0.742373 w_k1_MAT_s115
                                 + 1.05675 w_k1_MSFT_s115
                                 + 0.980436 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.869302 w_k1_ADM_s116
                                 + 0.624161 w_k1_AXP_s116
                                 + 1.04292 w_k1_DIS_s116
                                 + 0.887665 w_k1_FDX_s116
                                 + 0.836863 w_k1_IBM_s116
                                 + 1.02526 w_k1_INTC_s116
                                 + 0.975343 w_k1_KO_s116
                                 + 0.493038 w_k1_MAT_s116
                                 + 1.11485 w_k1_MSFT_s116
                                 + 0.933972 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.0034 w_k1_ADM_s116
                                 + 0.956657 w_k1_AXP_s116
                                 + 0.980874 w_k1_DIS_s116
                                 + 1.13195 w_k1_FDX_s116
                                 + 0.756981 w_k1_IBM_s116
                                 + 1.04411 w_k1_INTC_s116
                                 + 0.909104 w_k1_KO_s116
                                 + 0.684798 w_k1_MAT_s116
                                 + 1.12944 w_k1_MSFT_s116
                                 + 0.920692 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.857477 w_k1_ADM_s116
                                 + 1.00166 w_k1_AXP_s116
                                 + 1.23658 w_k1_DIS_s116
                                 + 0.96376 w_k1_FDX_s116
                                 + 1.43496 w_k1_IBM_s116
                                 + 1.05048 w_k1_INTC_s116 + 1.3303 w_k1_KO_s116
                                 + 1.05338 w_k1_MAT_s116
                                 + 1.18771 w_k1_MSFT_s116
                                 + 1.05301 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.134 w_k1_ADM_s116
                                 + 1.2332 w_k1_AXP_s116 + 1.20716 w_k1_DIS_s116
                                 + 1.22494 w_k1_FDX_s116
                                 + 1.14293 w_k1_IBM_s116
                                 + 1.20273 w_k1_INTC_s116
                                 + 1.16447 w_k1_KO_s116 + 1.35091 w_k1_MAT_s116
                                 + 1.37425 w_k1_MSFT_s116
                                 + 1.08842 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.905371 w_k1_ADM_s116
                                 + 0.855079 w_k1_AXP_s116
                                 + 0.727261 w_k1_DIS_s116
                                 + 0.887779 w_k1_FDX_s116
                                 + 0.706676 w_k1_IBM_s116
                                 + 0.708496 w_k1_INTC_s116
                                 + 0.780367 w_k1_KO_s116
                                 + 0.685176 w_k1_MAT_s116
                                 + 0.811395 w_k1_MSFT_s116
                                 + 0.79075 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.900004 w_k1_ADM_s116
                                 + 0.740364 w_k1_AXP_s116
                                 + 0.720728 w_k1_DIS_s116
                                 + 0.824517 w_k1_FDX_s116
                                 + 0.981947 w_k1_IBM_s116
                                 + 0.754961 w_k1_INTC_s116
                                 + 0.853367 w_k1_KO_s116
                                 + 0.877256 w_k1_MAT_s116
                                 + 0.92603 w_k1_MSFT_s116
                                 + 0.907766 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.30189 w_k1_ADM_s116
                                 + 1.34815 w_k1_AXP_s116
                                 + 1.63697 w_k1_DIS_s116
                                 + 1.32972 w_k1_FDX_s116
                                 + 1.27958 w_k1_IBM_s116 + 1.334 w_k1_INTC_s116
                                 + 1.31628 w_k1_KO_s116
                                 + 0.985812 w_k1_MAT_s116
                                 + 1.35776 w_k1_MSFT_s116
                                 + 1.20614 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.938745 w_k1_ADM_s116
                                 + 1.06729 w_k1_AXP_s116 + 0.9821 w_k1_DIS_s116
                                 + 1.01086 w_k1_FDX_s116
                                 + 1.26758 w_k1_IBM_s116
                                 + 0.885019 w_k1_INTC_s116
                                 + 1.08434 w_k1_KO_s116 + 1.26676 w_k1_MAT_s116
                                 + 0.93012 w_k1_MSFT_s116
                                 + 0.981456 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.945456 w_k1_ADM_s117
                                 + 0.626865 w_k1_AXP_s117
                                 + 0.599083 w_k1_DIS_s117
                                 + 0.805449 w_k1_FDX_s117
                                 + 0.611579 w_k1_IBM_s117
                                 + 0.873825 w_k1_INTC_s117
                                 + 0.758473 w_k1_KO_s117
                                 + 0.53197 w_k1_MAT_s117
                                 + 0.850866 w_k1_MSFT_s117
                                 + 0.800335 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.26519 w_k1_ADM_s117
                                 + 1.02746 w_k1_AXP_s117
                                 + 0.721384 w_k1_DIS_s117
                                 + 0.986962 w_k1_FDX_s117
                                 + 0.578524 w_k1_IBM_s117
                                 + 0.925782 w_k1_INTC_s117
                                 + 0.823668 w_k1_KO_s117
                                 + 0.696758 w_k1_MAT_s117
                                 + 1.03526 w_k1_MSFT_s117
                                 + 1.02124 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.23217 w_k1_ADM_s117
                                 + 1.30151 w_k1_AXP_s117
                                 + 1.35577 w_k1_DIS_s117
                                 + 1.09977 w_k1_FDX_s117 + 1.6513 w_k1_IBM_s117
                                 + 0.895979 w_k1_INTC_s117
                                 + 1.57229 w_k1_KO_s117 + 1.50897 w_k1_MAT_s117
                                 + 1.12033 w_k1_MSFT_s117
                                 + 1.27146 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.898213 w_k1_ADM_s117
                                 + 0.778277 w_k1_AXP_s117
                                 + 0.832634 w_k1_DIS_s117
                                 + 0.90788 w_k1_FDX_s117
                                 + 0.936116 w_k1_IBM_s117
                                 + 0.757036 w_k1_INTC_s117
                                 + 1.02161 w_k1_KO_s117
                                 + 0.900712 w_k1_MAT_s117
                                 + 0.99742 w_k1_MSFT_s117
                                 + 0.961331 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.0159 w_k1_ADM_s117
                                 + 1.12229 w_k1_AXP_s117
                                 + 1.11443 w_k1_DIS_s117
                                 + 0.983385 w_k1_FDX_s117
                                 + 1.20671 w_k1_IBM_s117
                                 + 1.00948 w_k1_INTC_s117
                                 + 1.29316 w_k1_KO_s117 + 1.61258 w_k1_MAT_s117
                                 + 1.15723 w_k1_MSFT_s117
                                 + 1.19865 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.866257 w_k1_ADM_s117
                                 + 0.848759 w_k1_AXP_s117
                                 + 0.841743 w_k1_DIS_s117
                                 + 0.852494 w_k1_FDX_s117
                                 + 1.21898 w_k1_IBM_s117
                                 + 0.923757 w_k1_INTC_s117
                                 + 1.02811 w_k1_KO_s117 + 1.28871 w_k1_MAT_s117
                                 + 0.987981 w_k1_MSFT_s117
                                 + 0.938233 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.53411 w_k1_ADM_s117
                                 + 1.22531 w_k1_AXP_s117
                                 + 1.57698 w_k1_DIS_s117
                                 + 1.25552 w_k1_FDX_s117
                                 + 1.23442 w_k1_IBM_s117
                                 + 1.11037 w_k1_INTC_s117 + 1.5694 w_k1_KO_s117
                                 + 1.81703 w_k1_MAT_s117
                                 + 1.47674 w_k1_MSFT_s117
                                 + 1.20119 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.41311 w_k1_ADM_s117
                                 + 1.09446 w_k1_AXP_s117
                                 + 1.00073 w_k1_DIS_s117
                                 + 1.02364 w_k1_FDX_s117
                                 + 0.800095 w_k1_IBM_s117
                                 + 0.875375 w_k1_INTC_s117
                                 + 1.01302 w_k1_KO_s117
                                 + 0.880514 w_k1_MAT_s117
                                 + 1.08684 w_k1_MSFT_s117 + 1.0288 w_k1_PG_s117
                                 >= 1
 k1_return_constraint:           0.0269045615026662 w_k1_ADM_s000
                                 + 0.0403385641399164 w_k1_ADM_s001
                                 + 0.0162798900197673 w_k1_ADM_s002
                                 + 0.0308266731285602 w_k1_ADM_s003
                                 + 0.0126940571864506 w_k1_ADM_s004
                                 + 0.0309563399763071 w_k1_ADM_s005
                                 + 0.0318689405768705 w_k1_ADM_s006
                                 + 0.0443192564116974 w_k1_ADM_s007
                                 + 0.016883656807647 w_k1_ADM_s010
                                 + 0.00997098393040519 w_k1_ADM_s011
                                 + 0.0197598328703805 w_k1_ADM_s012
                                 + 0.0099292491076896 w_k1_ADM_s013
                                 + 0.0320607706064514 w_k1_ADM_s014
                                 + 0.0160651662388909 w_k1_ADM_s015
                                 + 0.0226253787424895 w_k1_ADM_s016
                                 + 0.0293690168619194 w_k1_ADM_s017
                                 + 0.034911755078842 w_k1_ADM_s100
                                 + 0.032615048066104 w_k1_ADM_s101
                                 + 0.0284984327637457 w_k1_ADM_s102
                                 + 0.101989204211573 w_k1_ADM_s103
                                 + 0.0372307343151935 w_k1_ADM_s104
                                 + 0.0282112212446476 w_k1_ADM_s105
                                 + 0.0516101788232628 w_k1_ADM_s106
                                 + 0.0726837087966324 w_k1_ADM_s107
                                 + 0.0270209708406199 w_k1_ADM_s110
                                 + 0.0545403097446825 w_k1_ADM_s111
                                 + 0.0175566332871473 w_k1_ADM_s112
                                 + 0.013664756782921 w_k1_ADM_s113
                                 + 0.0445737022343278 w_k1_ADM_s114
                                 + 0.061792351704472 w_k1_ADM_s115
                                 + 0.0254504642618459 w_k1_ADM_s116
                                 + 0.0217804540305628 w_k1_ADM_s117
                                 + 0.0273765427650182 w_k1_AXP_s000
                                 + 0.0384984972506811 w_k1_AXP_s001
                                 + 0.0171651313948448 w_k1_AXP_s002
                                 + 0.0295845818768665 w_k1_AXP_s003
                                 + 0.013824451476685 w_k1_AXP_s004
                                 + 0.0351572473809048 w_k1_AXP_s005
                                 + 0.0396397852081723 w_k1_AXP_s006
                                 + 0.0391075812694287 w_k1_AXP_s007
                                 + 0.0175348624010001 w_k1_AXP_s010
                                 + 0.00877495931913032 w_k1_AXP_s011
                                 + 0.0226870240946531 w_k1_AXP_s012
                                 + 0.00895881835579975 w_k1_AXP_s013
                                 + 0.0281766558259207 w_k1_AXP_s014
                                 + 0.0143454745518147 w_k1_AXP_s015
                                 + 0.0202296779337267 w_k1_AXP_s016
                                 + 0.0281366028076019 w_k1_AXP_s017
                                 + 0.0348009708774248 w_k1_AXP_s100
                                 + 0.0300825019634169 w_k1_AXP_s101
                                 + 0.0317851667539253 w_k1_AXP_s102
                                 + 0.0981942164688441 w_k1_AXP_s103
                                 + 0.0400731304740835 w_k1_AXP_s104
                                 + 0.0250356190834558 w_k1_AXP_s105
                                 + 0.0479395545183149 w_k1_AXP_s106
                                 + 0.0675878970238153 w_k1_AXP_s107
                                 + 0.0228779678344721 w_k1_AXP_s110
                                 + 0.0532863303614876 w_k1_AXP_s111
                                 + 0.0147280376840595 w_k1_AXP_s112
                                 + 0.0148305779593826 w_k1_AXP_s113
                                 + 0.0424978126104439 w_k1_AXP_s114
                                 + 0.0573246836755317 w_k1_AXP_s115
                                 + 0.0252184302653579 w_k1_AXP_s116
                                 + 0.0188986890798839 w_k1_AXP_s117
                                 + 0.0271080166196096 w_k1_DIS_s000
                                 + 0.0376952123040987 w_k1_DIS_s001
                                 + 0.0176836634248287 w_k1_DIS_s002
                                 + 0.0317915530672297 w_k1_DIS_s003
                                 + 0.0147931199471713 w_k1_DIS_s004
                                 + 0.0332230764913245 w_k1_DIS_s005
                                 + 0.0383751720073871 w_k1_DIS_s006
                                 + 0.0417628828763851 w_k1_DIS_s007
                                 + 0.0180133330617992 w_k1_DIS_s010
                                 + 0.0102582895004824 w_k1_DIS_s011
                                 + 0.0225470388945209 w_k1_DIS_s012
                                 + 0.00866262407134174 w_k1_DIS_s013
                                 + 0.0300867387802805 w_k1_DIS_s014
                                 + 0.0151152495521238 w_k1_DIS_s015
                                 + 0.0190286462906288 w_k1_DIS_s016
                                 + 0.0259146716097887 w_k1_DIS_s017
                                 + 0.0357878292359637 w_k1_DIS_s100
                                 + 0.0314392522565359 w_k1_DIS_s101
                                 + 0.0310853080011974 w_k1_DIS_s102
                                 + 0.106470326485369 w_k1_DIS_s103
                                 + 0.0432290602635743 w_k1_DIS_s104
                                 + 0.0280936295533 w_k1_DIS_s105
                                 + 0.0532781714116885 w_k1_DIS_s106
                                 + 0.0701621891964726 w_k1_DIS_s107
                                 + 0.0230162803470474 w_k1_DIS_s110
                                 + 0.0501607905498663 w_k1_DIS_s111
                                 + 0.0125703442241965 w_k1_DIS_s112
                                 + 0.0150271206084212 w_k1_DIS_s113
                                 + 0.0449803941921685 w_k1_DIS_s114
                                 + 0.0622610864787399 w_k1_DIS_s115
                                 + 0.0259864260120179 w_k1_DIS_s116
                                 + 0.0192756939823026 w_k1_DIS_s117
                                 + 0.0265717187413753 w_k1_FDX_s000
                                 + 0.0375988065989795 w_k1_FDX_s001
                                 + 0.0170972898518989 w_k1_FDX_s002
                                 + 0.0295379183498991 w_k1_FDX_s003
                                 + 0.013311405103801 w_k1_FDX_s004
                                 + 0.0327990246658747 w_k1_FDX_s005
                                 + 0.0367912798337254 w_k1_FDX_s006
                                 + 0.0381298692666768 w_k1_FDX_s007
                                 + 0.0179158049468301 w_k1_FDX_s010
                                 + 0.00954202284448433 w_k1_FDX_s011
                                 + 0.0225557215336352 w_k1_FDX_s012
                                 + 0.0090082233469324 w_k1_FDX_s013
                                 + 0.0312868300913491 w_k1_FDX_s014
                                 + 0.0142556657872538 w_k1_FDX_s015
                                 + 0.0194573454653654 w_k1_FDX_s016
                                 + 0.0280843726327111 w_k1_FDX_s017
                                 + 0.0366149423371138 w_k1_FDX_s100
                                 + 0.0300402522492166 w_k1_FDX_s101
                                 + 0.0307308968619332 w_k1_FDX_s102
                                 + 0.100216637744572 w_k1_FDX_s103
                                 + 0.0396975692323605 w_k1_FDX_s104
                                 + 0.0268170647370378 w_k1_FDX_s105
                                 + 0.0477484968534789 w_k1_FDX_s106
                                 + 0.068048798595438 w_k1_FDX_s107
                                 + 0.0240007498237282 w_k1_FDX_s110
                                 + 0.0533284450786095 w_k1_FDX_s111
                                 + 0.0138468347652719 w_k1_FDX_s112
                                 + 0.0148805076985194 w_k1_FDX_s113
                                 + 0.0436823289282381 w_k1_FDX_s114
                                 + 0.0547418991570809 w_k1_FDX_s115
                                 + 0.026479685234341 w_k1_FDX_s116
                                 + 0.0184387143402896 w_k1_FDX_s117
                                 + 0.0267245991131775 w_k1_IBM_s000
                                 + 0.0392303959651357 w_k1_IBM_s001
                                 + 0.0160542871243318 w_k1_IBM_s002
                                 + 0.031411655178103 w_k1_IBM_s003
                                 + 0.0141841229849898 w_k1_IBM_s004
                                 + 0.0348703178698287 w_k1_IBM_s005
                                 + 0.0380532210403294 w_k1_IBM_s006
                                 + 0.0391048410503998 w_k1_IBM_s007
                                 + 0.0182292140040404 w_k1_IBM_s010
                                 + 0.00978746626719951 w_k1_IBM_s011
                                 + 0.0215087239416694 w_k1_IBM_s012
                                 + 0.00912481484136641 w_k1_IBM_s013
                                 + 0.0317805376905064 w_k1_IBM_s014
                                 + 0.0150974140563751 w_k1_IBM_s015
                                 + 0.0194772157493532 w_k1_IBM_s016
                                 + 0.0277986624422298 w_k1_IBM_s017
                                 + 0.0351597306833266 w_k1_IBM_s100
                                 + 0.0313436930567211 w_k1_IBM_s101
                                 + 0.0308792319578186 w_k1_IBM_s102
                                 + 0.0994965021444865 w_k1_IBM_s103
                                 + 0.0392101714248064 w_k1_IBM_s104
                                 + 0.0271825646383326 w_k1_IBM_s105
                                 + 0.0496038674647628 w_k1_IBM_s106
                                 + 0.069884169618225 w_k1_IBM_s107
                                 + 0.0247940018921359 w_k1_IBM_s110
                                 + 0.0529796584765305 w_k1_IBM_s111
                                 + 0.0134957604020396 w_k1_IBM_s112
                                 + 0.0158980517606371 w_k1_IBM_s113
                                 + 0.0440340121355657 w_k1_IBM_s114
                                 + 0.0579648584876223 w_k1_IBM_s115
                                 + 0.0259921509980406 w_k1_IBM_s116
                                 + 0.0190867210073258 w_k1_IBM_s117
                                 + 0.0323317186465472 w_k1_INTC_s000
                                 + 0.0374645153105369 w_k1_INTC_s001
                                 + 0.014803871708209 w_k1_INTC_s002
                                 + 0.0326499365750733 w_k1_INTC_s003
                                 + 0.0138179992105302 w_k1_INTC_s004
                                 + 0.0347994544626648 w_k1_INTC_s005
                                 + 0.0397344653154925 w_k1_INTC_s006
                                 + 0.0355572108671827 w_k1_INTC_s007
                                 + 0.019406511446281 w_k1_INTC_s010
                                 + 0.0088721696211202 w_k1_INTC_s011
                                 + 0.0210119946662274 w_k1_INTC_s012
                                 + 0.00919843750875182 w_k1_INTC_s013
                                 + 0.0288634614822513 w_k1_INTC_s014
                                 + 0.0128438412170367 w_k1_INTC_s015
                                 + 0.0217392964054685 w_k1_INTC_s016
                                 + 0.0297823479959834 w_k1_INTC_s017
                                 + 0.0362967748493939 w_k1_INTC_s100
                                 + 0.0253105730147695 w_k1_INTC_s101
                                 + 0.0322591131581126 w_k1_INTC_s102
                                 + 0.0989887692728082 w_k1_INTC_s103
                                 + 0.0379129321465731 w_k1_INTC_s104
                                 + 0.0251025113411996 w_k1_INTC_s105
                                 + 0.0437869424651761 w_k1_INTC_s106
                                 + 0.0568938579023193 w_k1_INTC_s107
                                 + 0.021952102430896 w_k1_INTC_s110
                                 + 0.0519634270924 w_k1_INTC_s111
                                 + 0.0172847520791916 w_k1_INTC_s112
                                 + 0.0148771538434289 w_k1_INTC_s113
                                 + 0.0388220032694302 w_k1_INTC_s114
                                 + 0.0593315545210582 w_k1_INTC_s115
                                 + 0.0248019116222637 w_k1_INTC_s116
                                 + 0.016766276028348 w_k1_INTC_s117
                                 + 0.0274156592308444 w_k1_KO_s000
                                 + 0.0399136719764811 w_k1_KO_s001
                                 + 0.0150127068710265 w_k1_KO_s002
                                 + 0.0300015463730903 w_k1_KO_s003
                                 + 0.0143021755685667 w_k1_KO_s004
                                 + 0.0346251601241013 w_k1_KO_s005
                                 + 0.0380892700424294 w_k1_KO_s006
                                 + 0.0429242167295766 w_k1_KO_s007
                                 + 0.0166408194116639 w_k1_KO_s010
                                 + 0.00948208151447079 w_k1_KO_s011
                                 + 0.0202465580869482 w_k1_KO_s012
                                 + 0.00852812989249574 w_k1_KO_s013
                                 + 0.0294566899274941 w_k1_KO_s014
                                 + 0.0157624880100736 w_k1_KO_s015
                                 + 0.0190317855594743 w_k1_KO_s016
                                 + 0.028393898067015 w_k1_KO_s017
                                 + 0.0341557673132317 w_k1_KO_s100
                                 + 0.0290206512924815 w_k1_KO_s101
                                 + 0.0322371296485061 w_k1_KO_s102
                                 + 0.10508068684463 w_k1_KO_s103
                                 + 0.0400873199039114 w_k1_KO_s104
                                 + 0.0269777871221777 w_k1_KO_s105
                                 + 0.0519810862322257 w_k1_KO_s106
                                 + 0.0645893464218316 w_k1_KO_s107
                                 + 0.0239291998890811 w_k1_KO_s110
                                 + 0.0506780770317868 w_k1_KO_s111
                                 + 0.013675127520505 w_k1_KO_s112
                                 + 0.0150600723483556 w_k1_KO_s113
                                 + 0.0449122993351288 w_k1_KO_s114
                                 + 0.063285463665294 w_k1_KO_s115
                                 + 0.0259037223592053 w_k1_KO_s116
                                 + 0.0213296193368922 w_k1_KO_s117
                                 + 0.0227471921974366 w_k1_MAT_s000
                                 + 0.0417279433932459 w_k1_MAT_s001
                                 + 0.0157400568434324 w_k1_MAT_s002
                                 + 0.0286033992172822 w_k1_MAT_s003
                                 + 0.0166751127479566 w_k1_MAT_s004
                                 + 0.0356246871379224 w_k1_MAT_s005
                                 + 0.0469968299107299 w_k1_MAT_s006
                                 + 0.0472699518000794 w_k1_MAT_s007
                                 + 0.0148161870861692 w_k1_MAT_s010
                                 + 0.00805968201726509 w_k1_MAT_s011
                                 + 0.0245906165395593 w_k1_MAT_s012
                                 + 0.00866380671759275 w_k1_MAT_s013
                                 + 0.0327725090634245 w_k1_MAT_s014
                                 + 0.0131118243041757 w_k1_MAT_s015
                                 + 0.0182323748396946 w_k1_MAT_s016
                                 + 0.0279995856810334 w_k1_MAT_s017
                                 + 0.0304562064640314 w_k1_MAT_s100
                                 + 0.0297695778730728 w_k1_MAT_s101
                                 + 0.0339187188680703 w_k1_MAT_s102
                                 + 0.100441205360527 w_k1_MAT_s103
                                 + 0.0349040937819022 w_k1_MAT_s104
                                 + 0.0260218049253464 w_k1_MAT_s105
                                 + 0.0491584806316205 w_k1_MAT_s106
                                 + 0.0782116993081073 w_k1_MAT_s107
                                 + 0.0255902496949085 w_k1_MAT_s110
                                 + 0.0499883180231091 w_k1_MAT_s111
                                 + 0.0159636151894911 w_k1_MAT_s112
                                 + 0.0150632802502102 w_k1_MAT_s113
                                 + 0.0440497588489125 w_k1_MAT_s114
                                 + 0.0486221890414243 w_k1_MAT_s115
                                 + 0.0239541643882292 w_k1_MAT_s116
                                 + 0.0215742014819943 w_k1_MAT_s117
                                 + 0.0292956973391057 w_k1_MSFT_s000
                                 + 0.0346582302089522 w_k1_MSFT_s001
                                 + 0.0151517551699168 w_k1_MSFT_s002
                                 + 0.0305627968098024 w_k1_MSFT_s003
                                 + 0.013722478058721 w_k1_MSFT_s004
                                 + 0.0342314586223642 w_k1_MSFT_s005
                                 + 0.0394966323086905 w_k1_MSFT_s006
                                 + 0.033556656805796 w_k1_MSFT_s007
                                 + 0.0188111073748497 w_k1_MSFT_s010
                                 + 0.00998564414106103 w_k1_MSFT_s011
                                 + 0.0225892755494666 w_k1_MSFT_s012
                                 + 0.00870386039333687 w_k1_MSFT_s013
                                 + 0.0307113104420374 w_k1_MSFT_s014
                                 + 0.0150685620211931 w_k1_MSFT_s015
                                 + 0.0193912848645711 w_k1_MSFT_s016
                                 + 0.0291569501718484 w_k1_MSFT_s017
                                 + 0.033799667091928 w_k1_MSFT_s100
                                 + 0.0297453818647537 w_k1_MSFT_s101
                                 + 0.0323428094793978 w_k1_MSFT_s102
                                 + 0.103998708391433 w_k1_MSFT_s103
                                 + 0.041996574212651 w_k1_MSFT_s104
                                 + 0.0261700427830316 w_k1_MSFT_s105
                                 + 0.0516833087610466 w_k1_MSFT_s106
                                 + 0.0652660456460581 w_k1_MSFT_s107
                                 + 0.0221738609160176 w_k1_MSFT_s110
                                 + 0.0483414608232196 w_k1_MSFT_s111
                                 + 0.0134450235338084 w_k1_MSFT_s112
                                 + 0.0132743777879538 w_k1_MSFT_s113
                                 + 0.0444818647040922 w_k1_MSFT_s114
                                 + 0.0584269974828498 w_k1_MSFT_s115
                                 + 0.0274679263201028 w_k1_MSFT_s116
                                 + 0.0202820705042888 w_k1_MSFT_s117
                                 + 0.0270839546194779 w_k1_PG_s000
                                 + 0.0349276551191615 w_k1_PG_s001
                                 + 0.0179876235420214 w_k1_PG_s002
                                 + 0.0322481567955863 w_k1_PG_s003
                                 + 0.0128691312169526 w_k1_PG_s004
                                 + 0.0340044450126606 w_k1_PG_s005
                                 + 0.0364158244451671 w_k1_PG_s006
                                 + 0.037518523140654 w_k1_PG_s007
                                 + 0.0167367246487446 w_k1_PG_s010
                                 + 0.00995425967963407 w_k1_PG_s011
                                 + 0.023036675728736 w_k1_PG_s012
                                 + 0.00813861318086235 w_k1_PG_s013
                                 + 0.0284422340120467 w_k1_PG_s014
                                 + 0.013997202094466 w_k1_PG_s015
                                 + 0.0193422065329877 w_k1_PG_s016
                                 + 0.0285828177330622 w_k1_PG_s017
                                 + 0.035938130721493 w_k1_PG_s100
                                 + 0.0308755501039591 w_k1_PG_s101
                                 + 0.0319491658078414 w_k1_PG_s102
                                 + 0.104606546506992 w_k1_PG_s103
                                 + 0.0369921544940006 w_k1_PG_s104
                                 + 0.0259291008914358 w_k1_PG_s105
                                 + 0.0538151572964572 w_k1_PG_s106
                                 + 0.0719998804526797 w_k1_PG_s107
                                 + 0.0226294989170576 w_k1_PG_s110
                                 + 0.0542242520249227 w_k1_PG_s111
                                 + 0.0149070953914344 w_k1_PG_s112
                                 + 0.0144500938969165 w_k1_PG_s113
                                 + 0.0447501781660104 w_k1_PG_s114
                                 + 0.0554412728039265 w_k1_PG_s115
                                 + 0.0247872325112051 w_k1_PG_s116
                                 + 0.0194941020785434 w_k1_PG_s117 >= 1.12
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

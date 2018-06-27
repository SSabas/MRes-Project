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
 w_k1_ADM_s0_return_constraint:  1.18428 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.965302 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.898136 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.17363 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.21072 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.883457 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.38558 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.14664 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.8766 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.86005 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.06076 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.759367 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.22701 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.00548 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.14886 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.31757 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.11484 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.1274 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.702102 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.39676 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.731136 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.984976 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.931065 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.758934 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.02551 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.691213 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.86217 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.19555 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.945291 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.952485 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.737191 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.889656 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.770161 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.982493 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.09168 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.06463 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.876489 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.32551 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.26812 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.977809 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.856882 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.28784 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.21174 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.760105 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.62306 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.55872 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.13587 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.99365 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.08898 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.727119 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.18151 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.831933 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.981722 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.6472 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.26815 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.15008 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.77382 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.31215 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.10296 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.917901 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.03969 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.734255 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.29611 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.79128 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.917399 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.22551 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.48167 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.44175 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.589815 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.924549 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.2432 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.98634 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.06997 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.39623 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.780724 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.28878 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.3725 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.977289 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.843891 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.39751 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.20899 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.722721 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.63286 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.53211 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.28653 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.971823 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.13013 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.635813 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.28374 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.809942 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.903983 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.57689 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.859219 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.16234 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.600568 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.62123 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.890473 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.923584 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.975616 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.637815 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.17912 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.919174 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.915403 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.55218 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.66884 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.25726 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.485646 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.835504 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.03075 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.788868 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.20622 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.44654 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.765087 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.40357 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.27942 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.949139 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.891678 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.18061 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.15552 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.827787 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.4289 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.19578 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.00908 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.90611 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.17896 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.749948 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.20323 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.958244 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.01679 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.28718 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.00888 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.12543 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.732368 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.37478 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.842458 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.10566 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.08026 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.762433 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.14888 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.884525 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.861177 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.32411 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.30646 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.06663 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.62767 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.842256 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.931085 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.05333 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.25438 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.17898 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.892785 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.25868 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.04537 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.02777 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.860883 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.36949 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.17458 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.80807 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.14509 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.84494 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.44875 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.17289 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.683056 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.800189 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.00926 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.67498 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.775984 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.82165 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.07253 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.09828 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.743932 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.871041 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.32907 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.493367 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.698249 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.712952 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.11431 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.975569 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.39266 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.949809 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.62745 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 2.09734 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.70657 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.204 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.83445 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.589041 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.69227 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.47312 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.786854 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.05826 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.1693 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.934483 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.831815 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.29902 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.15657 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.866333 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.55954 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.39032 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.25183 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.14003 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.31994 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.846953 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.38367 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.05926 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.885351 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.13055 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.89073 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.940471 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.882228 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.00312 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.896633 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.1233 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.00575 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.790974 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.1164 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.977525 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.874184 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.11886 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.22624 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.1514 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.86696 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.832121 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.09852 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.956242 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.28732 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.12808 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.829696 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.26831 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.18296 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.01435 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.891414 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.32431 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.06836 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.743421 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.36112 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.42788 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.28875 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.990742 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.889675 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.698375 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.17676 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.739898 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.78353 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.36881 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.896 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.03792 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.690453 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.157 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.02322 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.703058 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.868837 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.72139 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.23368 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.02608 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.06739 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.22215 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.51484 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.45687 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.698003 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.13039 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.19004 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.69703 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.938579 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.37751 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.873512 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.33688 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.09985 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.07129 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.911305 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.41498 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.09587 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.695927 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.06674 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 2.18037 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.39744 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.17597 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.695616 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.890242 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.847421 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.60753 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.867356 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.69679 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.26921 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.16427 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.689292 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.13496 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.35551 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.437572 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.777818 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.67963 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.4167 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.99901 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.33139 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.922317 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 2.11001 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 2.42101 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.661888 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.12034 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.89959 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.671422 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.647116 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.87792 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.840398 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.947603 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.22244 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.982074 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.910049 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.25809 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.13874 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.742055 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.36468 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.24791 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.2012 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.986462 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.07696 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.759295 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.25049 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.936976 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.887872 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.1384 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.946715 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.984962 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.736942 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.13066 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.81633 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.05987 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.923538 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.764348 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.05306 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.866414 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.813137 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.17822 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.18449 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.03087 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.770545 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.02686 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.09474 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.958097 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.27862 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.2995 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.00781 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.36885 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.14553 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.01184 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.941776 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.156 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.1042 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.836589 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.2136 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.1639 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.13951 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.98976 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.947825 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.751304 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.0385 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.790891 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.91198 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.20831 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.984745 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.990265 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.768486 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.11284 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.973417 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.881549 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.88471 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.815619 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.9968 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.897299 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.967111 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.13599 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.20765 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.10709 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.728574 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.947731 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.11753 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.887634 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.06214 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.24792 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.892666 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.17095 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.137999804901683 z_k1_s0000
                                 - 0.0889998248192784 z_k1_s0001
                                 - 0.190999939737509 z_k1_s0002
                                 - 0.304999383717777 z_k1_s0003
                                 - 0.107999779449249 z_k1_s0004
                                 - 0.0689999024508415 z_k1_s0005
                                 - 0.181999236792865 z_k1_s0006
                                 - 0.336999202747364 z_k1_s0007
                                 - 0.108000209181598 z_k1_s0010
                                 - 0.107000050143031 z_k1_s0011
                                 - 0.0819999140751526 z_k1_s0012
                                 - 0.139999940420909 z_k1_s0013
                                 - 0.0829998945138918 z_k1_s0014
                                 - 0.0599998979233201 z_k1_s0015
                                 - 0.0920003435619434 z_k1_s0016
                                 - 0.220999427557751 z_k1_s0017
                                 - 0.0469999902991775 z_k1_s0020
                                 - 0.0770000991922384 z_k1_s0021
                                 - 0.0500000011884836 z_k1_s0022
                                 - 0.107999670815214 z_k1_s0023
                                 - 0.112999989463976 z_k1_s0024
                                 - 0.0449999830396401 z_k1_s0025
                                 - 0.0719997805434762 z_k1_s0026
                                 - 0.0890002629494303 z_k1_s0027
                                 - 0.623001903969506 z_k1_s0030
                                 - 0.195000138939868 z_k1_s0031
                                 - 0.219000049960078 z_k1_s0032
                                 - 0.515999810735842 z_k1_s0033
                                 - 0.205000197631695 z_k1_s0034
                                 - 0.513000396458752 z_k1_s0035
                                 - 0.193000242121589 z_k1_s0036
                                 - 0.409000590504359 z_k1_s0037
                                 - 0.0779999978676403 z_k1_s0040
                                 - 0.138000055858099 z_k1_s0041
                                 - 0.101999788503957 z_k1_s0042
                                 - 0.0660000056495375 z_k1_s0043
                                 - 0.0949999303183183 z_k1_s0044
                                 - 0.195999517595883 z_k1_s0045
                                 - 0.0969997513714373 z_k1_s0046
                                 - 0.196999525022386 z_k1_s0047
                                 - 0.259999206657485 z_k1_s0050
                                 - 0.44699917430671 z_k1_s0051
                                 - 0.160999841183698 z_k1_s0052
                                 - 0.239999584298875 z_k1_s0053
                                 - 0.422998804277267 z_k1_s0054
                                 - 0.25799883282207 z_k1_s0055
                                 - 0.174999782634503 z_k1_s0056
                                 - 0.0939998714836888 z_k1_s0057
                                 - 0.226999130003139 z_k1_s0060
                                 - 0.174000597032074 z_k1_s0061
                                 - 0.291999599871214 z_k1_s0062
                                 - 0.119999955588059 z_k1_s0063
                                 - 0.390999258286645 z_k1_s0064
                                 - 0.218000254327998 z_k1_s0065
                                 - 0.183999347782231 z_k1_s0066
                                 - 0.111999924188887 z_k1_s0067
                                 - 0.433998685357581 z_k1_s0070
                                 - 0.104999910508794 z_k1_s0071
                                 - 0.173999810020341 z_k1_s0072
                                 - 0.116999775241391 z_k1_s0073
                                 - 0.380000231859671 z_k1_s0074
                                 - 0.298999425616887 z_k1_s0075
                                 - 0.424999985106569 z_k1_s0076
                                 - 0.14999987215542 z_k1_s0077
                                 - 0.294999823328062 z_k1_s0100
                                 - 0.101999873096574 z_k1_s0101
                                 - 0.222998923073028 z_k1_s0102
                                 - 0.346999035292023 z_k1_s0103
                                 - 0.0909999153351389 z_k1_s0104
                                 - 0.271000062575637 z_k1_s0105
                                 - 0.201998930164937 z_k1_s0106
                                 - 0.0869999012764761 z_k1_s0107
                                 - 0.130000270925313 z_k1_s0110
                                 - 0.0740001445882472 z_k1_s0111
                                 - 0.0480000278031351 z_k1_s0112
                                 - 0.0690002786296957 z_k1_s0113
                                 - 0.0300000095519533 z_k1_s0114
                                 - 0.0650001354626563 z_k1_s0115
                                 - 0.0800003593388064 z_k1_s0116
                                 - 0.130000270925313 z_k1_s0117
                                 - 0.0790000261583589 z_k1_s0120
                                 - 0.0580000192048711 z_k1_s0121
                                 - 0.0990000327807282 z_k1_s0122
                                 - 0.0510000168870418 z_k1_s0123
                                 - 0.0530000175492787 z_k1_s0124
                                 - 0.0500000165559233 z_k1_s0125
                                 - 0.0350000115891463 z_k1_s0126
                                 - 0.075000024833885 z_k1_s0127
                                 - 0.199999679174767 z_k1_s0130
                                 - 0.0979997543158626 z_k1_s0131
                                 - 0.161999828611334 z_k1_s0132
                                 - 0.169999506099119 z_k1_s0133
                                 - 0.0859997956853199 z_k1_s0134
                                 - 0.0829997783777551 z_k1_s0135
                                 - 0.23199949512307 z_k1_s0136
                                 - 0.0749997690908211 z_k1_s0137
                                 - 0.0520001814756908 z_k1_s0140
                                 - 0.139000315096102 z_k1_s0141
                                 - 0.206000619934202 z_k1_s0142
                                 - 0.198000867461758 z_k1_s0143
                                 - 0.151000503306269 z_k1_s0144
                                 - 0.117999985728308 z_k1_s0145
                                 - 0.160000644463895 z_k1_s0146
                                 - 0.0950002217937558 z_k1_s0147
                                 - 0.153000150162627 z_k1_s0150
                                 - 0.0480001931691972 z_k1_s0151
                                 - 0.145000086134312 z_k1_s0152
                                 - 0.0820003220390163 z_k1_s0153
                                 - 0.113000785024517 z_k1_s0154
                                 - 0.0880003223100793 z_k1_s0155
                                 - 0.207000821104619 z_k1_s0156
                                 - 0.119000116793154 z_k1_s0157
                                 - 0.16400019589559 z_k1_s0160
                                 - 0.106000017825224 z_k1_s0161
                                 - 0.132000550525625 z_k1_s0162
                                 - 0.235999707727937 z_k1_s0163
                                 - 0.104000024502105 z_k1_s0164
                                 - 0.102000031178986 z_k1_s0165
                                 - 0.231000343919992 z_k1_s0166
                                 - 0.16400019589559 z_k1_s0167
                                 - 0.0819999887526182 z_k1_s0170
                                 - 0.0320000173790874 z_k1_s0171
                                 - 0.0330000431565785 z_k1_s0172
                                 - 0.108000233967056 z_k1_s0173
                                 - 0.0340000264340366 z_k1_s0174
                                 - 0.0410000368763421 z_k1_s0175
                                 - 0.0560001685385104 z_k1_s0176
                                 - 0.039000027821393 z_k1_s0177
                                 - 1.10399714930585 z_k1_s1000
                                 - 2.52099860935313 z_k1_s1001
                                 - 2.07800365909082 z_k1_s1002
                                 - 0.552000603854291 z_k1_s1003
                                 - 0.875000760525781 z_k1_s1004
                                 - 1.26900165840108 z_k1_s1005
                                 - 0.556000159747136 z_k1_s1006
                                 - 1.19099915788309 z_k1_s1007
                                 - 0.949996848044615 z_k1_s1010
                                 - 0.448999786606301 z_k1_s1011
                                 - 0.365999902447673 z_k1_s1012
                                 - 1.75500374957679 z_k1_s1013
                                 - 1.81400163934385 z_k1_s1014
                                 - 0.660999721382549 z_k1_s1015
                                 - 0.642999621563815 z_k1_s1016
                                 - 1.33900321532154 z_k1_s1017
                                 - 0.601001887090449 z_k1_s1020
                                 - 0.307000409526293 z_k1_s1021
                                 - 0.641997951770584 z_k1_s1022
                                 - 0.780999246575881 z_k1_s1023
                                 - 0.561000719203398 z_k1_s1024
                                 - 0.316000277500565 z_k1_s1025
                                 - 1.24399999965831 z_k1_s1026
                                 - 0.811998967287402 z_k1_s1027
                                 - 1.09599817501993 z_k1_s1030
                                 - 0.358999538355913 z_k1_s1031
                                 - 0.58199921977009 z_k1_s1032
                                 - 0.910996269312608 z_k1_s1033
                                 - 0.598998613801388 z_k1_s1034
                                 - 1.20199487104951 z_k1_s1035
                                 - 1.44299604575172 z_k1_s1036
                                 - 0.528998855459009 z_k1_s1037
                                 - 0.927999184747141 z_k1_s1040
                                 - 0.362000293257009 z_k1_s1041
                                 - 1.65600047554029 z_k1_s1042
                                 - 0.900998162863262 z_k1_s1043
                                 - 0.425999743722295 z_k1_s1044
                                 - 0.411000212675768 z_k1_s1045
                                 - 0.479999921489925 z_k1_s1046
                                 - 1.05600057367789 z_k1_s1047
                                 - 0.997002486058218 z_k1_s1050
                                 - 0.325000226747039 z_k1_s1051
                                 - 0.227000374546084 z_k1_s1052
                                 - 1.0680016724722 z_k1_s1053
                                 - 0.892000462554109 z_k1_s1054
                                 - 0.550999709277564 z_k1_s1055
                                 - 0.330000614019075 z_k1_s1056
                                 - 0.701001145424234 z_k1_s1057
                                 - 0.649000960519071 z_k1_s1060
                                 - 0.320001130592081 z_k1_s1061
                                 - 0.271000221632396 z_k1_s1062
                                 - 0.55600061205324 z_k1_s1063
                                 - 0.418999551280791 z_k1_s1064
                                 - 0.418000402910508 z_k1_s1065
                                 - 0.358999737044924 z_k1_s1066
                                 - 0.190000154213857 z_k1_s1067
                                 - 0.527000234191956 z_k1_s1070
                                 - 0.25500016466711 z_k1_s1071
                                 - 0.18800019630992 z_k1_s1072
                                 - 0.179000153043996 z_k1_s1073
                                 - 0.610999576806524 z_k1_s1074
                                 - 0.542000306301828 z_k1_s1075
                                 - 0.189000112683852 z_k1_s1076
                                 - 0.161999982886082 z_k1_s1077
                                 - 0.299000081483586 z_k1_s1100
                                 - 0.181999849772857 z_k1_s1101
                                 - 0.0969998041922658 z_k1_s1102
                                 - 0.105999910400677 z_k1_s1103
                                 - 0.304000140488259 z_k1_s1104
                                 - 0.144999796137817 z_k1_s1105
                                 - 0.480999548256902 z_k1_s1106
                                 - 0.300999339086373 z_k1_s1107
                                 - 1.19999849835178 z_k1_s1110
                                 - 0.34299856372145 z_k1_s1111
                                 - 0.454998572073781 z_k1_s1112
                                 - 1.18499566301952 z_k1_s1113
                                 - 0.645996249334309 z_k1_s1114
                                 - 0.578995836608845 z_k1_s1115
                                 - 0.560000041761654 z_k1_s1116
                                 - 0.600998881299919 z_k1_s1117
                                 - 0.546000701745921 z_k1_s1120
                                 - 0.189999795779174 z_k1_s1121
                                 - 0.351999461268854 z_k1_s1122
                                 - 0.203000025149225 z_k1_s1123
                                 - 0.522000439954989 z_k1_s1124
                                 - 0.200999828200055 z_k1_s1125
                                 - 0.481000868395109 z_k1_s1126
                                 - 0.131999863650792 z_k1_s1127
                                 - 0.407999420714704 z_k1_s1130
                                 - 0.667998339484687 z_k1_s1131
                                 - 0.173999875022309 z_k1_s1132
                                 - 0.504999346570824 z_k1_s1133
                                 - 0.204999696221863 z_k1_s1134
                                 - 0.287998522976298 z_k1_s1135
                                 - 0.191999879983217 z_k1_s1136
                                 - 0.153999898332378 z_k1_s1137
                                 - 0.518000944138543 z_k1_s1140
                                 - 0.370000674384674 z_k1_s1141
                                 - 0.298999860489469 z_k1_s1142
                                 - 0.293999786186995 z_k1_s1143
                                 - 0.273999790476999 z_k1_s1144
                                 - 0.493998537287367 z_k1_s1145
                                 - 0.573000555466161 z_k1_s1146
                                 - 0.193000033976438 z_k1_s1147
                                 - 0.224999957229328 z_k1_s1150
                                 - 0.227999978595046 z_k1_s1151
                                 - 0.210999857522643 z_k1_s1152
                                 - 0.295998817685118 z_k1_s1153
                                 - 0.180999918065384 z_k1_s1154
                                 - 0.649001057518044 z_k1_s1155
                                 - 0.361998739331072 z_k1_s1156
                                 - 0.589998717926118 z_k1_s1157
                                 - 0.707999873955738 z_k1_s1160
                                 - 0.449000118411118 z_k1_s1161
                                 - 0.76199702068737 z_k1_s1162
                                 - 1.06400055385297 z_k1_s1163
                                 - 0.674999810931556 z_k1_s1164
                                 - 0.854000152789994 z_k1_s1165
                                 - 1.21799936976639 z_k1_s1166
                                 - 1.66100095491799 z_k1_s1167
                                 - 0.383999344939402 z_k1_s1170
                                 - 1.11900100230298 z_k1_s1171
                                 - 0.888999182082513 z_k1_s1172
                                 - 0.346999692033313 z_k1_s1173
                                 - 0.765000236829856 z_k1_s1174
                                 - 1.30499607518749 z_k1_s1175
                                 - 1.12499523240529 z_k1_s1176
                                 - 0.755995511698479 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.09625 w_k1_ADM_s000
                                 + 0.925477 w_k1_AXP_s000
                                 + 0.936201 w_k1_DIS_s000
                                 + 1.0094 w_k1_FDX_s000 + 0.69329 w_k1_IBM_s000
                                 + 0.862193 w_k1_INTC_s000
                                 + 0.772613 w_k1_KO_s000
                                 + 0.97108 w_k1_MAT_s000
                                 + 0.978557 w_k1_MSFT_s000
                                 + 0.868422 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.71535 w_k1_ADM_s000
                                 + 1.47802 w_k1_AXP_s000
                                 + 1.63486 w_k1_DIS_s000
                                 + 1.57531 w_k1_FDX_s000
                                 + 1.04125 w_k1_IBM_s000
                                 + 1.3265 w_k1_INTC_s000 + 1.04304 w_k1_KO_s000
                                 + 0.937523 w_k1_MAT_s000
                                 + 1.25339 w_k1_MSFT_s000
                                 + 1.05524 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.02014 w_k1_ADM_s000
                                 + 0.954809 w_k1_AXP_s000
                                 + 1.58096 w_k1_DIS_s000
                                 + 1.13384 w_k1_FDX_s000
                                 + 1.37255 w_k1_IBM_s000
                                 + 1.21011 w_k1_INTC_s000 + 1.3255 w_k1_KO_s000
                                 + 1.47203 w_k1_MAT_s000
                                 + 1.34661 w_k1_MSFT_s000
                                 + 1.12384 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.885918 w_k1_ADM_s000
                                 + 0.816345 w_k1_AXP_s000
                                 + 0.726905 w_k1_DIS_s000
                                 + 0.795359 w_k1_FDX_s000
                                 + 0.904951 w_k1_IBM_s000
                                 + 0.622534 w_k1_INTC_s000
                                 + 0.697345 w_k1_KO_s000
                                 + 1.24857 w_k1_MAT_s000
                                 + 0.728846 w_k1_MSFT_s000
                                 + 0.820235 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.18192 w_k1_ADM_s000
                                 + 1.35402 w_k1_AXP_s000
                                 + 1.15682 w_k1_DIS_s000
                                 + 1.18094 w_k1_FDX_s000
                                 + 1.46061 w_k1_IBM_s000
                                 + 1.02051 w_k1_INTC_s000
                                 + 1.03135 w_k1_KO_s000 + 1.72763 w_k1_MAT_s000
                                 + 1.0671 w_k1_MSFT_s000 + 1.00239 w_k1_PG_s000
                                 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.765487 w_k1_ADM_s000
                                 + 0.64938 w_k1_AXP_s000
                                 + 1.31679 w_k1_DIS_s000
                                 + 0.998857 w_k1_FDX_s000
                                 + 1.00877 w_k1_IBM_s000
                                 + 0.929383 w_k1_INTC_s000
                                 + 1.01605 w_k1_KO_s000
                                 + 0.769744 w_k1_MAT_s000
                                 + 1.02301 w_k1_MSFT_s000
                                 + 0.92882 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.827897 w_k1_ADM_s000
                                 + 1.17595 w_k1_AXP_s000 + 1.2252 w_k1_DIS_s000
                                 + 1.16281 w_k1_FDX_s000
                                 + 0.837627 w_k1_IBM_s000
                                 + 0.9886 w_k1_INTC_s000
                                 + 0.830456 w_k1_KO_s000
                                 + 1.32919 w_k1_MAT_s000
                                 + 1.09171 w_k1_MSFT_s000
                                 + 0.988448 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.951322 w_k1_ADM_s000
                                 + 0.872341 w_k1_AXP_s000
                                 + 0.892537 w_k1_DIS_s000
                                 + 0.844344 w_k1_FDX_s000
                                 + 1.19173 w_k1_IBM_s000
                                 + 0.797166 w_k1_INTC_s000
                                 + 0.930427 w_k1_KO_s000
                                 + 1.89774 w_k1_MAT_s000
                                 + 0.891904 w_k1_MSFT_s000
                                 + 0.989539 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.844163 w_k1_ADM_s001
                                 + 0.775624 w_k1_AXP_s001
                                 + 0.477784 w_k1_DIS_s001
                                 + 0.697771 w_k1_FDX_s001
                                 + 0.90907 w_k1_IBM_s001
                                 + 0.742917 w_k1_INTC_s001
                                 + 0.667321 w_k1_KO_s001
                                 + 0.835812 w_k1_MAT_s001
                                 + 0.69614 w_k1_MSFT_s001
                                 + 0.909219 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.29501 w_k1_ADM_s001
                                 + 1.37799 w_k1_AXP_s001
                                 + 0.934845 w_k1_DIS_s001
                                 + 1.08761 w_k1_FDX_s001
                                 + 1.05294 w_k1_IBM_s001
                                 + 1.09795 w_k1_INTC_s001
                                 + 0.94893 w_k1_KO_s001 + 0.89243 w_k1_MAT_s001
                                 + 1.05181 w_k1_MSFT_s001
                                 + 1.08945 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.18588 w_k1_ADM_s001
                                 + 1.02968 w_k1_AXP_s001
                                 + 1.11568 w_k1_DIS_s001
                                 + 1.16562 w_k1_FDX_s001
                                 + 0.737441 w_k1_IBM_s001
                                 + 1.13115 w_k1_INTC_s001
                                 + 1.01205 w_k1_KO_s001
                                 + 0.575496 w_k1_MAT_s001
                                 + 1.22981 w_k1_MSFT_s001
                                 + 1.25202 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.769017 w_k1_ADM_s001
                                 + 0.700857 w_k1_AXP_s001
                                 + 0.81125 w_k1_DIS_s001
                                 + 0.830729 w_k1_FDX_s001
                                 + 0.942774 w_k1_IBM_s001
                                 + 0.932235 w_k1_INTC_s001
                                 + 0.921854 w_k1_KO_s001
                                 + 0.877264 w_k1_MAT_s001
                                 + 0.872649 w_k1_MSFT_s001
                                 + 0.950538 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.08491 w_k1_ADM_s001
                                 + 0.923469 w_k1_AXP_s001
                                 + 0.801304 w_k1_DIS_s001
                                 + 0.90522 w_k1_FDX_s001
                                 + 0.685102 w_k1_IBM_s001
                                 + 0.846441 w_k1_INTC_s001
                                 + 0.681045 w_k1_KO_s001
                                 + 0.522548 w_k1_MAT_s001
                                 + 0.786574 w_k1_MSFT_s001
                                 + 0.810765 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.972314 w_k1_ADM_s001
                                 + 0.882021 w_k1_AXP_s001
                                 + 1.28695 w_k1_DIS_s001
                                 + 1.07258 w_k1_FDX_s001
                                 + 0.943292 w_k1_IBM_s001
                                 + 1.07229 w_k1_INTC_s001
                                 + 1.22558 w_k1_KO_s001 + 1.12051 w_k1_MAT_s001
                                 + 1.06325 w_k1_MSFT_s001
                                 + 1.20017 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.928165 w_k1_ADM_s001
                                 + 0.977505 w_k1_AXP_s001
                                 + 0.881094 w_k1_DIS_s001
                                 + 0.870902 w_k1_FDX_s001
                                 + 1.09502 w_k1_IBM_s001
                                 + 0.926165 w_k1_INTC_s001
                                 + 0.86244 w_k1_KO_s001 + 0.83557 w_k1_MAT_s001
                                 + 0.917133 w_k1_MSFT_s001
                                 + 0.881515 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.18294 w_k1_ADM_s001
                                 + 1.33936 w_k1_AXP_s001
                                 + 1.49901 w_k1_DIS_s001
                                 + 1.25456 w_k1_FDX_s001
                                 + 1.30834 w_k1_IBM_s001
                                 + 1.30702 w_k1_INTC_s001
                                 + 1.33423 w_k1_KO_s001 + 1.11001 w_k1_MAT_s001
                                 + 1.28951 w_k1_MSFT_s001
                                 + 1.16075 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.841152 w_k1_ADM_s002
                                 + 0.750647 w_k1_AXP_s002
                                 + 0.745626 w_k1_DIS_s002
                                 + 0.819858 w_k1_FDX_s002
                                 + 1.40335 w_k1_IBM_s002
                                 + 0.857337 w_k1_INTC_s002
                                 + 0.97439 w_k1_KO_s002 + 1.33404 w_k1_MAT_s002
                                 + 0.789237 w_k1_MSFT_s002
                                 + 0.79272 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.25625 w_k1_ADM_s002
                                 + 0.901602 w_k1_AXP_s002
                                 + 1.03267 w_k1_DIS_s002
                                 + 0.971566 w_k1_FDX_s002
                                 + 1.03493 w_k1_IBM_s002
                                 + 0.92708 w_k1_INTC_s002 + 1.027 w_k1_KO_s002
                                 + 0.871073 w_k1_MAT_s002
                                 + 1.01028 w_k1_MSFT_s002
                                 + 0.985532 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.10455 w_k1_ADM_s002
                                 + 0.54992 w_k1_AXP_s002
                                 + 0.637038 w_k1_DIS_s002
                                 + 0.701666 w_k1_FDX_s002
                                 + 0.829986 w_k1_IBM_s002
                                 + 0.731662 w_k1_INTC_s002
                                 + 0.787154 w_k1_KO_s002
                                 + 0.602343 w_k1_MAT_s002
                                 + 0.74943 w_k1_MSFT_s002
                                 + 0.789059 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.95115 w_k1_ADM_s002
                                 + 0.944508 w_k1_AXP_s002
                                 + 0.768745 w_k1_DIS_s002
                                 + 0.889025 w_k1_FDX_s002
                                 + 0.750476 w_k1_IBM_s002
                                 + 0.852705 w_k1_INTC_s002
                                 + 0.757315 w_k1_KO_s002
                                 + 0.759137 w_k1_MAT_s002
                                 + 0.795877 w_k1_MSFT_s002
                                 + 0.792501 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.5341 w_k1_ADM_s002
                                 + 1.20463 w_k1_AXP_s002
                                 + 1.29683 w_k1_DIS_s002
                                 + 1.22379 w_k1_FDX_s002
                                 + 1.01575 w_k1_IBM_s002
                                 + 1.21076 w_k1_INTC_s002
                                 + 1.14563 w_k1_KO_s002 + 0.81844 w_k1_MAT_s002
                                 + 1.09296 w_k1_MSFT_s002
                                 + 1.02364 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.43694 w_k1_ADM_s002
                                 + 0.699365 w_k1_AXP_s002
                                 + 1.47185 w_k1_DIS_s002
                                 + 1.22489 w_k1_FDX_s002
                                 + 0.899975 w_k1_IBM_s002
                                 + 1.12615 w_k1_INTC_s002
                                 + 1.21741 w_k1_KO_s002 + 0.77208 w_k1_MAT_s002
                                 + 1.24262 w_k1_MSFT_s002 + 1.1158 w_k1_PG_s002
                                 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.18761 w_k1_ADM_s002
                                 + 1.24872 w_k1_AXP_s002
                                 + 1.19459 w_k1_DIS_s002
                                 + 1.21142 w_k1_FDX_s002
                                 + 1.49905 w_k1_IBM_s002
                                 + 1.19184 w_k1_INTC_s002
                                 + 1.20897 w_k1_KO_s002 + 1.12729 w_k1_MAT_s002
                                 + 1.12505 w_k1_MSFT_s002
                                 + 0.994833 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.16275 w_k1_ADM_s002
                                 + 0.958043 w_k1_AXP_s002
                                 + 0.907873 w_k1_DIS_s002
                                 + 0.948648 w_k1_FDX_s002
                                 + 1.09879 w_k1_IBM_s002
                                 + 0.819116 w_k1_INTC_s002
                                 + 0.95377 w_k1_KO_s002 + 1.25722 w_k1_MAT_s002
                                 + 0.857217 w_k1_MSFT_s002
                                 + 0.818359 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.29971 w_k1_ADM_s003
                                 + 0.973765 w_k1_AXP_s003
                                 + 1.16167 w_k1_DIS_s003
                                 + 1.14055 w_k1_FDX_s003
                                 + 0.833815 w_k1_IBM_s003
                                 + 1.01892 w_k1_INTC_s003
                                 + 1.03909 w_k1_KO_s003
                                 + 0.618626 w_k1_MAT_s003
                                 + 1.0029 w_k1_MSFT_s003 + 1.04858 w_k1_PG_s003
                                 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.97824 w_k1_ADM_s003
                                 + 0.705335 w_k1_AXP_s003
                                 + 0.725273 w_k1_DIS_s003
                                 + 0.753525 w_k1_FDX_s003
                                 + 1.07356 w_k1_IBM_s003
                                 + 0.822651 w_k1_INTC_s003
                                 + 0.791177 w_k1_KO_s003
                                 + 0.898685 w_k1_MAT_s003
                                 + 0.790932 w_k1_MSFT_s003
                                 + 0.863896 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.973336 w_k1_ADM_s003
                                 + 0.687349 w_k1_AXP_s003
                                 + 0.522684 w_k1_DIS_s003
                                 + 0.784085 w_k1_FDX_s003
                                 + 0.794077 w_k1_IBM_s003
                                 + 0.804204 w_k1_INTC_s003
                                 + 0.654677 w_k1_KO_s003
                                 + 0.743123 w_k1_MAT_s003
                                 + 0.702111 w_k1_MSFT_s003
                                 + 0.802901 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.33355 w_k1_ADM_s003
                                 + 1.1659 w_k1_AXP_s003 + 1.50679 w_k1_DIS_s003
                                 + 1.34648 w_k1_FDX_s003
                                 + 1.24033 w_k1_IBM_s003
                                 + 1.08761 w_k1_INTC_s003
                                 + 1.33227 w_k1_KO_s003
                                 + 0.912653 w_k1_MAT_s003
                                 + 1.12913 w_k1_MSFT_s003
                                 + 1.14292 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.01791 w_k1_ADM_s003
                                 + 0.873432 w_k1_AXP_s003
                                 + 0.694384 w_k1_DIS_s003
                                 + 0.824954 w_k1_FDX_s003
                                 + 0.849126 w_k1_IBM_s003
                                 + 0.833554 w_k1_INTC_s003
                                 + 0.773175 w_k1_KO_s003
                                 + 0.788937 w_k1_MAT_s003
                                 + 0.773919 w_k1_MSFT_s003
                                 + 0.833106 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.05937 w_k1_ADM_s003
                                 + 0.979588 w_k1_AXP_s003
                                 + 0.915537 w_k1_DIS_s003
                                 + 0.917039 w_k1_FDX_s003
                                 + 1.28631 w_k1_IBM_s003
                                 + 1.00113 w_k1_INTC_s003
                                 + 1.18909 w_k1_KO_s003 + 1.02082 w_k1_MAT_s003
                                 + 0.870711 w_k1_MSFT_s003
                                 + 0.961715 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.38293 w_k1_ADM_s003
                                 + 1.17084 w_k1_AXP_s003
                                 + 1.08388 w_k1_DIS_s003
                                 + 1.01155 w_k1_FDX_s003 + 1.6755 w_k1_IBM_s003
                                 + 0.884859 w_k1_INTC_s003
                                 + 1.31079 w_k1_KO_s003 + 1.75502 w_k1_MAT_s003
                                 + 0.996485 w_k1_MSFT_s003
                                 + 1.18741 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.09556 w_k1_ADM_s003
                                 + 0.75849 w_k1_AXP_s003
                                 + 0.969624 w_k1_DIS_s003
                                 + 0.982356 w_k1_FDX_s003
                                 + 0.793819 w_k1_IBM_s003
                                 + 0.768572 w_k1_INTC_s003
                                 + 0.922866 w_k1_KO_s003
                                 + 0.78421 w_k1_MAT_s003
                                 + 0.928402 w_k1_MSFT_s003
                                 + 0.909872 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.5203 w_k1_ADM_s004
                                 + 1.27108 w_k1_AXP_s004
                                 + 1.46243 w_k1_DIS_s004
                                 + 1.43532 w_k1_FDX_s004
                                 + 1.12253 w_k1_IBM_s004
                                 + 1.3367 w_k1_INTC_s004 + 1.15246 w_k1_KO_s004
                                 + 0.837226 w_k1_MAT_s004
                                 + 1.40566 w_k1_MSFT_s004
                                 + 1.24491 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.14898 w_k1_ADM_s004
                                 + 1.16594 w_k1_AXP_s004
                                 + 0.860317 w_k1_DIS_s004
                                 + 1.06921 w_k1_FDX_s004
                                 + 1.03798 w_k1_IBM_s004
                                 + 0.978204 w_k1_INTC_s004
                                 + 0.861363 w_k1_KO_s004
                                 + 1.12515 w_k1_MAT_s004
                                 + 0.981804 w_k1_MSFT_s004
                                 + 1.00373 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.06046 w_k1_ADM_s004
                                 + 0.843961 w_k1_AXP_s004
                                 + 0.790497 w_k1_DIS_s004
                                 + 0.982741 w_k1_FDX_s004
                                 + 0.816364 w_k1_IBM_s004
                                 + 0.727826 w_k1_INTC_s004
                                 + 0.781845 w_k1_KO_s004
                                 + 0.836067 w_k1_MAT_s004
                                 + 0.918672 w_k1_MSFT_s004
                                 + 0.98151 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.21933 w_k1_ADM_s004
                                 + 1.65917 w_k1_AXP_s004
                                 + 1.25109 w_k1_DIS_s004
                                 + 1.13559 w_k1_FDX_s004 + 1.3364 w_k1_IBM_s004
                                 + 1.06019 w_k1_INTC_s004
                                 + 1.25377 w_k1_KO_s004 + 1.5295 w_k1_MAT_s004
                                 + 1.25605 w_k1_MSFT_s004
                                 + 1.19053 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.37158 w_k1_ADM_s004
                                 + 0.870285 w_k1_AXP_s004
                                 + 1.33103 w_k1_DIS_s004
                                 + 1.10139 w_k1_FDX_s004
                                 + 1.04556 w_k1_IBM_s004
                                 + 0.952934 w_k1_INTC_s004
                                 + 1.24894 w_k1_KO_s004
                                 + 0.902588 w_k1_MAT_s004
                                 + 1.33546 w_k1_MSFT_s004
                                 + 1.25079 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.852536 w_k1_ADM_s004
                                 + 0.688981 w_k1_AXP_s004
                                 + 0.801459 w_k1_DIS_s004
                                 + 0.847474 w_k1_FDX_s004
                                 + 0.747016 w_k1_IBM_s004
                                 + 0.855154 w_k1_INTC_s004
                                 + 0.832693 w_k1_KO_s004
                                 + 0.804492 w_k1_MAT_s004
                                 + 1.05081 w_k1_MSFT_s004
                                 + 0.844444 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.06226 w_k1_ADM_s004
                                 + 1.15504 w_k1_AXP_s004
                                 + 1.09058 w_k1_DIS_s004
                                 + 1.15487 w_k1_FDX_s004
                                 + 0.943658 w_k1_IBM_s004
                                 + 0.939272 w_k1_INTC_s004
                                 + 1.07814 w_k1_KO_s004 + 0.85237 w_k1_MAT_s004
                                 + 1.19755 w_k1_MSFT_s004
                                 + 1.20485 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.05079 w_k1_ADM_s004
                                 + 0.847539 w_k1_AXP_s004
                                 + 0.817545 w_k1_DIS_s004
                                 + 0.887063 w_k1_FDX_s004
                                 + 1.2348 w_k1_IBM_s004
                                 + 0.912404 w_k1_INTC_s004
                                 + 1.15229 w_k1_KO_s004 + 1.09162 w_k1_MAT_s004
                                 + 0.892056 w_k1_MSFT_s004
                                 + 1.05258 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.946303 w_k1_ADM_s005
                                 + 1.01958 w_k1_AXP_s005
                                 + 0.911119 w_k1_DIS_s005
                                 + 0.920307 w_k1_FDX_s005 + 1.186 w_k1_IBM_s005
                                 + 1.00455 w_k1_INTC_s005
                                 + 1.11912 w_k1_KO_s005 + 1.08716 w_k1_MAT_s005
                                 + 0.891659 w_k1_MSFT_s005
                                 + 1.07052 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.18263 w_k1_ADM_s005
                                 + 0.940353 w_k1_AXP_s005
                                 + 1.05797 w_k1_DIS_s005
                                 + 0.984481 w_k1_FDX_s005
                                 + 0.785287 w_k1_IBM_s005
                                 + 1.04297 w_k1_INTC_s005
                                 + 0.920043 w_k1_KO_s005
                                 + 0.579692 w_k1_MAT_s005
                                 + 1.09658 w_k1_MSFT_s005 + 0.9892 w_k1_PG_s005
                                 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.16366 w_k1_ADM_s005
                                 + 1.00565 w_k1_AXP_s005
                                 + 0.884842 w_k1_DIS_s005
                                 + 0.883069 w_k1_FDX_s005
                                 + 1.68523 w_k1_IBM_s005
                                 + 1.04287 w_k1_INTC_s005
                                 + 1.21002 w_k1_KO_s005 + 1.28361 w_k1_MAT_s005
                                 + 0.87782 w_k1_MSFT_s005
                                 + 1.08497 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.31886 w_k1_ADM_s005
                                 + 1.22833 w_k1_AXP_s005
                                 + 1.32342 w_k1_DIS_s005 + 1.1101 w_k1_FDX_s005
                                 + 1.3365 w_k1_IBM_s005
                                 + 1.31611 w_k1_INTC_s005
                                 + 1.33533 w_k1_KO_s005 + 1.29239 w_k1_MAT_s005
                                 + 1.28202 w_k1_MSFT_s005
                                 + 1.24308 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.931727 w_k1_ADM_s005
                                 + 0.58416 w_k1_AXP_s005
                                 + 0.640592 w_k1_DIS_s005
                                 + 0.746658 w_k1_FDX_s005
                                 + 0.65597 w_k1_IBM_s005
                                 + 0.747077 w_k1_INTC_s005
                                 + 0.703992 w_k1_KO_s005
                                 + 0.537755 w_k1_MAT_s005
                                 + 0.791348 w_k1_MSFT_s005
                                 + 0.772016 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.2446 w_k1_ADM_s005
                                 + 1.46609 w_k1_AXP_s005
                                 + 1.17774 w_k1_DIS_s005
                                 + 1.14891 w_k1_FDX_s005
                                 + 0.955925 w_k1_IBM_s005
                                 + 1.23583 w_k1_INTC_s005
                                 + 1.10017 w_k1_KO_s005
                                 + 0.847206 w_k1_MAT_s005
                                 + 1.07366 w_k1_MSFT_s005 + 1.0619 w_k1_PG_s005
                                 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.47385 w_k1_ADM_s005
                                 + 1.62633 w_k1_AXP_s005
                                 + 1.75653 w_k1_DIS_s005 + 1.3546 w_k1_FDX_s005
                                 + 1.67034 w_k1_IBM_s005
                                 + 1.32108 w_k1_INTC_s005
                                 + 1.53685 w_k1_KO_s005 + 1.07712 w_k1_MAT_s005
                                 + 1.28746 w_k1_MSFT_s005
                                 + 1.47558 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.917066 w_k1_ADM_s005
                                 + 0.803992 w_k1_AXP_s005
                                 + 1.16335 w_k1_DIS_s005
                                 + 0.940162 w_k1_FDX_s005
                                 + 1.30399 w_k1_IBM_s005
                                 + 1.04121 w_k1_INTC_s005
                                 + 1.20257 w_k1_KO_s005
                                 + 0.882235 w_k1_MAT_s005
                                 + 0.923437 w_k1_MSFT_s005
                                 + 0.924763 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.988441 w_k1_ADM_s006
                                 + 0.839232 w_k1_AXP_s006
                                 + 0.932165 w_k1_DIS_s006
                                 + 0.950221 w_k1_FDX_s006
                                 + 0.74296 w_k1_IBM_s006
                                 + 0.839507 w_k1_INTC_s006
                                 + 0.87789 w_k1_KO_s006
                                 + 0.935724 w_k1_MAT_s006
                                 + 0.936978 w_k1_MSFT_s006
                                 + 0.996024 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.29458 w_k1_ADM_s006
                                 + 1.23793 w_k1_AXP_s006
                                 + 1.55541 w_k1_DIS_s006
                                 + 1.25778 w_k1_FDX_s006
                                 + 0.914773 w_k1_IBM_s006
                                 + 1.10431 w_k1_INTC_s006
                                 + 1.15493 w_k1_KO_s006 + 1.14947 w_k1_MAT_s006
                                 + 1.24601 w_k1_MSFT_s006
                                 + 1.14515 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.09146 w_k1_ADM_s006
                                 + 0.771446 w_k1_AXP_s006
                                 + 1.14257 w_k1_DIS_s006
                                 + 0.970764 w_k1_FDX_s006
                                 + 1.09182 w_k1_IBM_s006
                                 + 0.784162 w_k1_INTC_s006
                                 + 0.999972 w_k1_KO_s006
                                 + 1.02421 w_k1_MAT_s006
                                 + 1.01785 w_k1_MSFT_s006
                                 + 1.04971 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.06314 w_k1_ADM_s006
                                 + 1.58339 w_k1_AXP_s006
                                 + 1.36566 w_k1_DIS_s006
                                 + 1.34457 w_k1_FDX_s006
                                 + 1.29434 w_k1_IBM_s006
                                 + 1.00814 w_k1_INTC_s006
                                 + 1.29065 w_k1_KO_s006 + 2.58672 w_k1_MAT_s006
                                 + 1.10021 w_k1_MSFT_s006
                                 + 1.25097 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.04667 w_k1_ADM_s006
                                 + 1.42249 w_k1_AXP_s006
                                 + 1.10327 w_k1_DIS_s006
                                 + 1.12764 w_k1_FDX_s006 + 1.2056 w_k1_IBM_s006
                                 + 1.13224 w_k1_INTC_s006
                                 + 0.988561 w_k1_KO_s006
                                 + 1.78497 w_k1_MAT_s006
                                 + 0.944456 w_k1_MSFT_s006
                                 + 1.12479 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.902643 w_k1_ADM_s006
                                 + 0.997677 w_k1_AXP_s006
                                 + 0.714287 w_k1_DIS_s006
                                 + 0.814205 w_k1_FDX_s006
                                 + 1.01182 w_k1_IBM_s006
                                 + 0.744173 w_k1_INTC_s006
                                 + 0.65845 w_k1_KO_s006
                                 + 0.823937 w_k1_MAT_s006
                                 + 0.65851 w_k1_MSFT_s006
                                 + 0.799268 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.760046 w_k1_ADM_s006
                                 + 0.708425 w_k1_AXP_s006
                                 + 0.651384 w_k1_DIS_s006
                                 + 0.779098 w_k1_FDX_s006
                                 + 0.748083 w_k1_IBM_s006
                                 + 0.631308 w_k1_INTC_s006
                                 + 0.631784 w_k1_KO_s006
                                 + 0.795888 w_k1_MAT_s006
                                 + 0.705804 w_k1_MSFT_s006
                                 + 0.875804 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.16987 w_k1_ADM_s006
                                 + 1.14966 w_k1_AXP_s006
                                 + 1.02415 w_k1_DIS_s006
                                 + 1.20956 w_k1_FDX_s006
                                 + 0.728483 w_k1_IBM_s006
                                 + 0.993288 w_k1_INTC_s006
                                 + 0.86723 w_k1_KO_s006
                                 + 0.959691 w_k1_MAT_s006
                                 + 0.944629 w_k1_MSFT_s006
                                 + 1.05043 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.924824 w_k1_ADM_s007
                                 + 1.03557 w_k1_AXP_s007
                                 + 1.15977 w_k1_DIS_s007
                                 + 0.967103 w_k1_FDX_s007
                                 + 1.38553 w_k1_IBM_s007
                                 + 0.970312 w_k1_INTC_s007
                                 + 1.19824 w_k1_KO_s007 + 1.09836 w_k1_MAT_s007
                                 + 0.94899 w_k1_MSFT_s007
                                 + 1.07734 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.12301 w_k1_ADM_s007
                                 + 0.933136 w_k1_AXP_s007
                                 + 0.597269 w_k1_DIS_s007
                                 + 0.884051 w_k1_FDX_s007
                                 + 0.693692 w_k1_IBM_s007
                                 + 0.775282 w_k1_INTC_s007
                                 + 0.656242 w_k1_KO_s007
                                 + 0.891925 w_k1_MAT_s007
                                 + 0.760434 w_k1_MSFT_s007
                                 + 0.937158 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.947341 w_k1_ADM_s007
                                 + 0.632488 w_k1_AXP_s007
                                 + 0.943061 w_k1_DIS_s007
                                 + 1.00402 w_k1_FDX_s007
                                 + 0.676312 w_k1_IBM_s007
                                 + 0.814183 w_k1_INTC_s007
                                 + 0.873922 w_k1_KO_s007
                                 + 0.536454 w_k1_MAT_s007
                                 + 0.970022 w_k1_MSFT_s007
                                 + 1.05322 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.926752 w_k1_ADM_s007
                                 + 0.915463 w_k1_AXP_s007
                                 + 0.678737 w_k1_DIS_s007
                                 + 0.805578 w_k1_FDX_s007
                                 + 0.940113 w_k1_IBM_s007
                                 + 0.727148 w_k1_INTC_s007
                                 + 0.881511 w_k1_KO_s007
                                 + 1.20102 w_k1_MAT_s007
                                 + 0.730537 w_k1_MSFT_s007
                                 + 1.0172 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.20821 w_k1_ADM_s007
                                 + 1.57654 w_k1_AXP_s007
                                 + 1.20684 w_k1_DIS_s007
                                 + 1.18251 w_k1_FDX_s007
                                 + 1.34481 w_k1_IBM_s007
                                 + 1.06424 w_k1_INTC_s007 + 1.1561 w_k1_KO_s007
                                 + 1.92756 w_k1_MAT_s007
                                 + 1.11716 w_k1_MSFT_s007 + 1.1088 w_k1_PG_s007
                                 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.19452 w_k1_ADM_s007
                                 + 1.22848 w_k1_AXP_s007
                                 + 1.24331 w_k1_DIS_s007
                                 + 1.33133 w_k1_FDX_s007
                                 + 0.860572 w_k1_IBM_s007
                                 + 1.21896 w_k1_INTC_s007
                                 + 1.11555 w_k1_KO_s007 + 1.05011 w_k1_MAT_s007
                                 + 1.27446 w_k1_MSFT_s007
                                 + 1.24215 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.880017 w_k1_ADM_s007
                                 + 0.860077 w_k1_AXP_s007
                                 + 1.04367 w_k1_DIS_s007
                                 + 0.914542 w_k1_FDX_s007
                                 + 0.928645 w_k1_IBM_s007
                                 + 0.831658 w_k1_INTC_s007
                                 + 1.00108 w_k1_KO_s007 + 1.21766 w_k1_MAT_s007
                                 + 1.02561 w_k1_MSFT_s007
                                 + 1.06837 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.700724 w_k1_ADM_s007
                                 + 0.617466 w_k1_AXP_s007
                                 + 0.689117 w_k1_DIS_s007
                                 + 0.737831 w_k1_FDX_s007
                                 + 0.723463 w_k1_IBM_s007
                                 + 0.705377 w_k1_INTC_s007
                                 + 0.70583 w_k1_KO_s007
                                 + 0.568927 w_k1_MAT_s007
                                 + 0.790299 w_k1_MSFT_s007
                                 + 0.953551 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.936234 w_k1_ADM_s010
                                 + 1.17847 w_k1_AXP_s010
                                 + 1.30831 w_k1_DIS_s010
                                 + 1.20646 w_k1_FDX_s010
                                 + 0.895386 w_k1_IBM_s010
                                 + 1.26109 w_k1_INTC_s010
                                 + 1.22957 w_k1_KO_s010 + 1.12497 w_k1_MAT_s010
                                 + 1.33167 w_k1_MSFT_s010
                                 + 1.12783 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.710399 w_k1_ADM_s010
                                 + 0.898485 w_k1_AXP_s010
                                 + 0.942545 w_k1_DIS_s010
                                 + 0.737002 w_k1_FDX_s010
                                 + 1.43871 w_k1_IBM_s010
                                 + 0.924558 w_k1_INTC_s010
                                 + 1.12712 w_k1_KO_s010 + 1.06612 w_k1_MAT_s010
                                 + 0.884227 w_k1_MSFT_s010
                                 + 0.964191 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.846211 w_k1_ADM_s010
                                 + 0.958112 w_k1_AXP_s010
                                 + 1.2455 w_k1_DIS_s010
                                 + 0.992324 w_k1_FDX_s010
                                 + 1.22129 w_k1_IBM_s010
                                 + 1.01443 w_k1_INTC_s010
                                 + 1.21524 w_k1_KO_s010 + 1.17341 w_k1_MAT_s010
                                 + 1.2571 w_k1_MSFT_s010 + 1.13963 w_k1_PG_s010
                                 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.919046 w_k1_ADM_s010
                                 + 0.923881 w_k1_AXP_s010
                                 + 0.96115 w_k1_DIS_s010
                                 + 0.922711 w_k1_FDX_s010
                                 + 0.981365 w_k1_IBM_s010
                                 + 0.82011 w_k1_INTC_s010 + 1.0216 w_k1_KO_s010
                                 + 1.09074 w_k1_MAT_s010
                                 + 0.938591 w_k1_MSFT_s010
                                 + 0.949873 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.66156 w_k1_ADM_s010
                                 + 0.659805 w_k1_AXP_s010
                                 + 0.689178 w_k1_DIS_s010
                                 + 0.690939 w_k1_FDX_s010
                                 + 0.894431 w_k1_IBM_s010
                                 + 0.74537 w_k1_INTC_s010
                                 + 0.867336 w_k1_KO_s010
                                 + 0.524196 w_k1_MAT_s010
                                 + 0.693735 w_k1_MSFT_s010
                                 + 0.867605 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.705129 w_k1_ADM_s010
                                 + 0.895506 w_k1_AXP_s010
                                 + 0.66972 w_k1_DIS_s010
                                 + 0.823001 w_k1_FDX_s010
                                 + 0.750755 w_k1_IBM_s010
                                 + 0.883724 w_k1_INTC_s010
                                 + 0.769284 w_k1_KO_s010
                                 + 0.623345 w_k1_MAT_s010
                                 + 0.797931 w_k1_MSFT_s010
                                 + 0.919452 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.21863 w_k1_ADM_s010
                                 + 1.512 w_k1_AXP_s010 + 1.68342 w_k1_DIS_s010
                                 + 1.27068 w_k1_FDX_s010
                                 + 1.32098 w_k1_IBM_s010
                                 + 1.32846 w_k1_INTC_s010
                                 + 1.46798 w_k1_KO_s010 + 1.14042 w_k1_MAT_s010
                                 + 1.32443 w_k1_MSFT_s010
                                 + 1.12984 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.872298 w_k1_ADM_s010
                                 + 1.09922 w_k1_AXP_s010
                                 + 0.814374 w_k1_DIS_s010
                                 + 0.831035 w_k1_FDX_s010
                                 + 1.0882 w_k1_IBM_s010
                                 + 0.851942 w_k1_INTC_s010
                                 + 1.05546 w_k1_KO_s010 + 1.27699 w_k1_MAT_s010
                                 + 0.876899 w_k1_MSFT_s010
                                 + 1.0354 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.0128 w_k1_ADM_s011
                                 + 0.806335 w_k1_AXP_s011
                                 + 1.03615 w_k1_DIS_s011
                                 + 0.99184 w_k1_FDX_s011
                                 + 0.850358 w_k1_IBM_s011
                                 + 0.836538 w_k1_INTC_s011
                                 + 1.07665 w_k1_KO_s011 + 1.10336 w_k1_MAT_s011
                                 + 1.06105 w_k1_MSFT_s011 + 1.0216 w_k1_PG_s011
                                 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.764885 w_k1_ADM_s011
                                 + 0.904362 w_k1_AXP_s011
                                 + 1.17942 w_k1_DIS_s011
                                 + 0.929583 w_k1_FDX_s011
                                 + 1.6321 w_k1_IBM_s011
                                 + 0.885898 w_k1_INTC_s011
                                 + 1.42882 w_k1_KO_s011 + 1.60741 w_k1_MAT_s011
                                 + 1.01836 w_k1_MSFT_s011
                                 + 1.24008 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.12719 w_k1_ADM_s011
                                 + 0.940252 w_k1_AXP_s011
                                 + 1.36166 w_k1_DIS_s011
                                 + 1.06662 w_k1_FDX_s011
                                 + 1.10356 w_k1_IBM_s011
                                 + 1.0272 w_k1_INTC_s011 + 1.21385 w_k1_KO_s011
                                 + 1.26173 w_k1_MAT_s011
                                 + 1.10312 w_k1_MSFT_s011 + 1.0671 w_k1_PG_s011
                                 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.14919 w_k1_ADM_s011
                                 + 1.15381 w_k1_AXP_s011 + 1.3089 w_k1_DIS_s011
                                 + 1.15337 w_k1_FDX_s011
                                 + 1.18754 w_k1_IBM_s011
                                 + 1.16568 w_k1_INTC_s011
                                 + 1.15728 w_k1_KO_s011 + 1.42023 w_k1_MAT_s011
                                 + 1.09163 w_k1_MSFT_s011 + 1.1414 w_k1_PG_s011
                                 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.03624 w_k1_ADM_s011
                                 + 1.50839 w_k1_AXP_s011
                                 + 1.92551 w_k1_DIS_s011 + 1.3561 w_k1_FDX_s011
                                 + 1.15154 w_k1_IBM_s011
                                 + 1.36384 w_k1_INTC_s011
                                 + 1.35889 w_k1_KO_s011 + 2.09176 w_k1_MAT_s011
                                 + 1.45916 w_k1_MSFT_s011
                                 + 1.16975 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.68479 w_k1_ADM_s011
                                 + 1.21767 w_k1_AXP_s011
                                 + 1.43759 w_k1_DIS_s011
                                 + 1.33043 w_k1_FDX_s011
                                 + 0.826007 w_k1_IBM_s011
                                 + 0.759583 w_k1_INTC_s011
                                 + 1.09418 w_k1_KO_s011
                                 + 0.903674 w_k1_MAT_s011
                                 + 1.13656 w_k1_MSFT_s011
                                 + 1.12564 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.715846 w_k1_ADM_s011
                                 + 0.718985 w_k1_AXP_s011
                                 + 0.724651 w_k1_DIS_s011
                                 + 0.841765 w_k1_FDX_s011
                                 + 0.737765 w_k1_IBM_s011
                                 + 0.94895 w_k1_INTC_s011
                                 + 0.904828 w_k1_KO_s011
                                 + 0.674907 w_k1_MAT_s011
                                 + 0.86657 w_k1_MSFT_s011
                                 + 0.900991 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.737602 w_k1_ADM_s011
                                 + 0.774652 w_k1_AXP_s011
                                 + 0.760699 w_k1_DIS_s011
                                 + 0.737164 w_k1_FDX_s011
                                 + 1.01592 w_k1_IBM_s011
                                 + 0.738616 w_k1_INTC_s011
                                 + 0.931644 w_k1_KO_s011
                                 + 1.06905 w_k1_MAT_s011
                                 + 0.752764 w_k1_MSFT_s011
                                 + 0.939675 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.56664 w_k1_ADM_s012
                                 + 1.51305 w_k1_AXP_s012
                                 + 1.68426 w_k1_DIS_s012
                                 + 1.43487 w_k1_FDX_s012
                                 + 1.14057 w_k1_IBM_s012
                                 + 1.22853 w_k1_INTC_s012
                                 + 1.28267 w_k1_KO_s012 + 1.27471 w_k1_MAT_s012
                                 + 1.21573 w_k1_MSFT_s012
                                 + 1.24939 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.873609 w_k1_ADM_s012
                                 + 0.802859 w_k1_AXP_s012
                                 + 0.929435 w_k1_DIS_s012
                                 + 0.82254 w_k1_FDX_s012
                                 + 1.07743 w_k1_IBM_s012
                                 + 0.928451 w_k1_INTC_s012
                                 + 1.09586 w_k1_KO_s012
                                 + 0.904638 w_k1_MAT_s012
                                 + 0.830426 w_k1_MSFT_s012
                                 + 1.00068 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.03075 w_k1_ADM_s012
                                 + 0.876808 w_k1_AXP_s012
                                 + 0.964676 w_k1_DIS_s012
                                 + 1.08211 w_k1_FDX_s012
                                 + 0.683682 w_k1_IBM_s012
                                 + 0.869308 w_k1_INTC_s012
                                 + 0.857123 w_k1_KO_s012
                                 + 0.647004 w_k1_MAT_s012
                                 + 0.817586 w_k1_MSFT_s012
                                 + 0.954776 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.625612 w_k1_ADM_s012
                                 + 0.520639 w_k1_AXP_s012
                                 + 0.586732 w_k1_DIS_s012
                                 + 0.649449 w_k1_FDX_s012
                                 + 0.77701 w_k1_IBM_s012
                                 + 0.926025 w_k1_INTC_s012
                                 + 0.851839 w_k1_KO_s012
                                 + 0.813317 w_k1_MAT_s012
                                 + 0.725625 w_k1_MSFT_s012
                                 + 0.775141 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.88454 w_k1_ADM_s012
                                 + 0.758334 w_k1_AXP_s012
                                 + 0.75138 w_k1_DIS_s012
                                 + 0.793439 w_k1_FDX_s012
                                 + 0.761783 w_k1_IBM_s012
                                 + 0.941983 w_k1_INTC_s012
                                 + 1.01042 w_k1_KO_s012
                                 + 0.776127 w_k1_MAT_s012
                                 + 0.724724 w_k1_MSFT_s012
                                 + 0.949033 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.05356 w_k1_ADM_s012
                                 + 1.2697 w_k1_AXP_s012 + 1.03119 w_k1_DIS_s012
                                 + 1.11167 w_k1_FDX_s012
                                 + 0.975356 w_k1_IBM_s012
                                 + 1.03077 w_k1_INTC_s012
                                 + 1.23445 w_k1_KO_s012 + 1.49482 w_k1_MAT_s012
                                 + 0.972813 w_k1_MSFT_s012
                                 + 1.04899 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.975216 w_k1_ADM_s012
                                 + 0.856955 w_k1_AXP_s012
                                 + 1.02095 w_k1_DIS_s012
                                 + 0.907701 w_k1_FDX_s012
                                 + 1.55883 w_k1_IBM_s012
                                 + 0.888254 w_k1_INTC_s012
                                 + 1.5057 w_k1_KO_s012 + 1.53808 w_k1_MAT_s012
                                 + 0.977258 w_k1_MSFT_s012
                                 + 1.05978 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.975382 w_k1_ADM_s012
                                 + 0.897851 w_k1_AXP_s012
                                 + 1.27316 w_k1_DIS_s012
                                 + 1.03575 w_k1_FDX_s012
                                 + 0.994048 w_k1_IBM_s012
                                 + 1.02545 w_k1_INTC_s012 + 1.0554 w_k1_KO_s012
                                 + 1.10424 w_k1_MAT_s012
                                 + 1.09593 w_k1_MSFT_s012
                                 + 1.10587 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.847679 w_k1_ADM_s013
                                 + 0.669783 w_k1_AXP_s013
                                 + 0.770773 w_k1_DIS_s013
                                 + 0.892884 w_k1_FDX_s013
                                 + 0.722283 w_k1_IBM_s013
                                 + 0.973846 w_k1_INTC_s013
                                 + 0.809371 w_k1_KO_s013
                                 + 0.597866 w_k1_MAT_s013
                                 + 0.97687 w_k1_MSFT_s013
                                 + 0.960344 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.83235 w_k1_ADM_s013
                                 + 0.970322 w_k1_AXP_s013
                                 + 1.04019 w_k1_DIS_s013
                                 + 1.02568 w_k1_FDX_s013
                                 + 0.843906 w_k1_IBM_s013
                                 + 1.13577 w_k1_INTC_s013
                                 + 1.01864 w_k1_KO_s013 + 1.02812 w_k1_MAT_s013
                                 + 1.2164 w_k1_MSFT_s013 + 1.15308 w_k1_PG_s013
                                 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.7377 w_k1_ADM_s013
                                 + 0.764326 w_k1_AXP_s013
                                 + 1.10969 w_k1_DIS_s013
                                 + 0.891126 w_k1_FDX_s013
                                 + 1.29018 w_k1_IBM_s013
                                 + 1.16711 w_k1_INTC_s013
                                 + 1.08594 w_k1_KO_s013 + 1.01358 w_k1_MAT_s013
                                 + 1.02883 w_k1_MSFT_s013
                                 + 0.949797 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.02128 w_k1_ADM_s013
                                 + 0.802919 w_k1_AXP_s013
                                 + 0.768745 w_k1_DIS_s013
                                 + 0.831312 w_k1_FDX_s013
                                 + 0.99462 w_k1_IBM_s013
                                 + 0.959141 w_k1_INTC_s013
                                 + 1.02152 w_k1_KO_s013
                                 + 0.782946 w_k1_MAT_s013
                                 + 0.85788 w_k1_MSFT_s013
                                 + 0.979207 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.931195 w_k1_ADM_s013
                                 + 0.801286 w_k1_AXP_s013
                                 + 1.02945 w_k1_DIS_s013
                                 + 1.04212 w_k1_FDX_s013
                                 + 0.962762 w_k1_IBM_s013
                                 + 1.08339 w_k1_INTC_s013
                                 + 0.992376 w_k1_KO_s013
                                 + 0.863147 w_k1_MAT_s013
                                 + 1.22792 w_k1_MSFT_s013
                                 + 1.13171 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.15124 w_k1_ADM_s013
                                 + 0.918862 w_k1_AXP_s013
                                 + 1.36515 w_k1_DIS_s013
                                 + 1.13936 w_k1_FDX_s013
                                 + 0.979782 w_k1_IBM_s013
                                 + 1.29983 w_k1_INTC_s013
                                 + 1.12276 w_k1_KO_s013
                                 + 0.871904 w_k1_MAT_s013
                                 + 1.30186 w_k1_MSFT_s013
                                 + 1.23539 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.22302 w_k1_ADM_s013
                                 + 1.6155 w_k1_AXP_s013 + 1.34269 w_k1_DIS_s013
                                 + 1.30066 w_k1_FDX_s013
                                 + 1.26507 w_k1_IBM_s013
                                 + 1.21255 w_k1_INTC_s013
                                 + 1.43304 w_k1_KO_s013 + 1.72159 w_k1_MAT_s013
                                 + 1.20496 w_k1_MSFT_s013
                                 + 1.35155 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.501707 w_k1_ADM_s013
                                 + 0.665667 w_k1_AXP_s013
                                 + 0.563435 w_k1_DIS_s013
                                 + 0.634254 w_k1_FDX_s013
                                 + 0.721957 w_k1_IBM_s013
                                 + 0.806592 w_k1_INTC_s013
                                 + 0.791902 w_k1_KO_s013
                                 + 0.578871 w_k1_MAT_s013
                                 + 0.792517 w_k1_MSFT_s013
                                 + 0.866719 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.30267 w_k1_ADM_s014
                                 + 1.27253 w_k1_AXP_s014
                                 + 1.72118 w_k1_DIS_s014
                                 + 1.27659 w_k1_FDX_s014
                                 + 1.95966 w_k1_IBM_s014
                                 + 0.94861 w_k1_INTC_s014 + 1.5477 w_k1_KO_s014
                                 + 1.61639 w_k1_MAT_s014
                                 + 1.13953 w_k1_MSFT_s014
                                 + 1.16517 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.35561 w_k1_ADM_s014
                                 + 1.17585 w_k1_AXP_s014 + 1.0463 w_k1_DIS_s014
                                 + 1.28965 w_k1_FDX_s014
                                 + 0.780292 w_k1_IBM_s014
                                 + 0.923246 w_k1_INTC_s014
                                 + 0.832326 w_k1_KO_s014
                                 + 0.694021 w_k1_MAT_s014
                                 + 1.01819 w_k1_MSFT_s014
                                 + 0.958282 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.33741 w_k1_ADM_s014
                                 + 0.729301 w_k1_AXP_s014
                                 + 1.02881 w_k1_DIS_s014
                                 + 1.03783 w_k1_FDX_s014
                                 + 0.855178 w_k1_IBM_s014
                                 + 0.813129 w_k1_INTC_s014
                                 + 0.961901 w_k1_KO_s014
                                 + 0.550106 w_k1_MAT_s014
                                 + 1.06406 w_k1_MSFT_s014
                                 + 1.02818 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.823982 w_k1_ADM_s014
                                 + 0.607908 w_k1_AXP_s014
                                 + 0.747736 w_k1_DIS_s014
                                 + 0.777253 w_k1_FDX_s014
                                 + 1.06912 w_k1_IBM_s014
                                 + 0.742109 w_k1_INTC_s014
                                 + 0.989874 w_k1_KO_s014
                                 + 0.864025 w_k1_MAT_s014
                                 + 0.901347 w_k1_MSFT_s014
                                 + 0.960763 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.17696 w_k1_ADM_s014
                                 + 0.94504 w_k1_AXP_s014 + 1.1673 w_k1_DIS_s014
                                 + 0.929166 w_k1_FDX_s014
                                 + 1.27197 w_k1_IBM_s014
                                 + 0.820174 w_k1_INTC_s014
                                 + 1.2364 w_k1_KO_s014 + 1.22696 w_k1_MAT_s014
                                 + 1.03103 w_k1_MSFT_s014
                                 + 1.21087 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.872351 w_k1_ADM_s014
                                 + 0.594095 w_k1_AXP_s014
                                 + 0.692806 w_k1_DIS_s014
                                 + 0.783866 w_k1_FDX_s014
                                 + 0.6472 w_k1_IBM_s014
                                 + 0.579319 w_k1_INTC_s014
                                 + 0.740263 w_k1_KO_s014
                                 + 0.472961 w_k1_MAT_s014
                                 + 0.702193 w_k1_MSFT_s014
                                 + 0.900082 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.40757 w_k1_ADM_s014
                                 + 1.25665 w_k1_AXP_s014
                                 + 1.36745 w_k1_DIS_s014
                                 + 1.29828 w_k1_FDX_s014
                                 + 1.03877 w_k1_IBM_s014
                                 + 0.995514 w_k1_INTC_s014
                                 + 1.13366 w_k1_KO_s014 + 1.12313 w_k1_MAT_s014
                                 + 1.21773 w_k1_MSFT_s014 + 1.1862 w_k1_PG_s014
                                 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.13944 w_k1_ADM_s014
                                 + 1.13041 w_k1_AXP_s014
                                 + 1.00337 w_k1_DIS_s014
                                 + 0.980613 w_k1_FDX_s014
                                 + 1.50849 w_k1_IBM_s014
                                 + 0.785981 w_k1_INTC_s014
                                 + 1.2055 w_k1_KO_s014 + 1.59826 w_k1_MAT_s014
                                 + 0.862775 w_k1_MSFT_s014
                                 + 1.21066 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.809964 w_k1_ADM_s015
                                 + 0.801009 w_k1_AXP_s015
                                 + 0.755817 w_k1_DIS_s015
                                 + 0.858066 w_k1_FDX_s015
                                 + 0.936059 w_k1_IBM_s015
                                 + 0.982432 w_k1_INTC_s015
                                 + 0.778909 w_k1_KO_s015
                                 + 0.678793 w_k1_MAT_s015
                                 + 0.888889 w_k1_MSFT_s015
                                 + 0.865165 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.656375 w_k1_ADM_s015
                                 + 0.500146 w_k1_AXP_s015
                                 + 0.613857 w_k1_DIS_s015
                                 + 0.652559 w_k1_FDX_s015
                                 + 0.798272 w_k1_IBM_s015
                                 + 0.823681 w_k1_INTC_s015
                                 + 0.688767 w_k1_KO_s015
                                 + 0.579266 w_k1_MAT_s015
                                 + 0.758638 w_k1_MSFT_s015
                                 + 0.809012 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.787766 w_k1_ADM_s015
                                 + 0.93944 w_k1_AXP_s015
                                 + 1.20062 w_k1_DIS_s015
                                 + 1.00686 w_k1_FDX_s015
                                 + 1.02625 w_k1_IBM_s015
                                 + 1.11532 w_k1_INTC_s015
                                 + 1.08664 w_k1_KO_s015 + 1.24504 w_k1_MAT_s015
                                 + 1.31217 w_k1_MSFT_s015
                                 + 1.12683 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.642917 w_k1_ADM_s015
                                 + 0.608059 w_k1_AXP_s015
                                 + 0.622304 w_k1_DIS_s015
                                 + 0.80762 w_k1_FDX_s015
                                 + 0.694319 w_k1_IBM_s015
                                 + 0.895831 w_k1_INTC_s015
                                 + 0.661176 w_k1_KO_s015
                                 + 0.460868 w_k1_MAT_s015
                                 + 0.956288 w_k1_MSFT_s015
                                 + 0.840968 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.01484 w_k1_ADM_s015
                                 + 1.31031 w_k1_AXP_s015
                                 + 0.931586 w_k1_DIS_s015
                                 + 0.969947 w_k1_FDX_s015
                                 + 1.25911 w_k1_IBM_s015
                                 + 1.21654 w_k1_INTC_s015
                                 + 1.17121 w_k1_KO_s015 + 1.23235 w_k1_MAT_s015
                                 + 0.993859 w_k1_MSFT_s015
                                 + 1.16654 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.13977 w_k1_ADM_s015
                                 + 1.20448 w_k1_AXP_s015 + 1.0297 w_k1_DIS_s015
                                 + 1.20152 w_k1_FDX_s015
                                 + 0.860842 w_k1_IBM_s015
                                 + 1.35067 w_k1_INTC_s015
                                 + 1.14793 w_k1_KO_s015
                                 + 0.979521 w_k1_MAT_s015
                                 + 1.13344 w_k1_MSFT_s015
                                 + 1.13617 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.20216 w_k1_ADM_s015
                                 + 1.32716 w_k1_AXP_s015
                                 + 1.50126 w_k1_DIS_s015
                                 + 1.27717 w_k1_FDX_s015
                                 + 1.34435 w_k1_IBM_s015
                                 + 1.14697 w_k1_INTC_s015
                                 + 1.26507 w_k1_KO_s015 + 1.50958 w_k1_MAT_s015
                                 + 1.30635 w_k1_MSFT_s015
                                 + 1.27443 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.766183 w_k1_ADM_s015
                                 + 1.02323 w_k1_AXP_s015
                                 + 0.858158 w_k1_DIS_s015
                                 + 1.01334 w_k1_FDX_s015
                                 + 0.92061 w_k1_IBM_s015
                                 + 1.15844 w_k1_INTC_s015
                                 + 0.827326 w_k1_KO_s015
                                 + 0.64632 w_k1_MAT_s015
                                 + 1.10409 w_k1_MSFT_s015
                                 + 0.842973 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.47253 w_k1_ADM_s016
                                 + 1.08121 w_k1_AXP_s016
                                 + 1.01289 w_k1_DIS_s016
                                 + 1.16736 w_k1_FDX_s016
                                 + 0.960456 w_k1_IBM_s016
                                 + 0.966048 w_k1_INTC_s016
                                 + 0.923032 w_k1_KO_s016
                                 + 0.987192 w_k1_MAT_s016
                                 + 1.02727 w_k1_MSFT_s016
                                 + 1.07913 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.19114 w_k1_ADM_s016
                                 + 0.973059 w_k1_AXP_s016 + 1.009 w_k1_DIS_s016
                                 + 1.12899 w_k1_FDX_s016
                                 + 0.697075 w_k1_IBM_s016
                                 + 1.04003 w_k1_INTC_s016
                                 + 0.819759 w_k1_KO_s016
                                 + 0.849337 w_k1_MAT_s016
                                 + 1.10869 w_k1_MSFT_s016
                                 + 1.02798 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.14136 w_k1_ADM_s016
                                 + 0.588953 w_k1_AXP_s016
                                 + 0.51833 w_k1_DIS_s016
                                 + 0.722098 w_k1_FDX_s016
                                 + 0.716422 w_k1_IBM_s016
                                 + 0.808442 w_k1_INTC_s016
                                 + 0.692513 w_k1_KO_s016
                                 + 0.608457 w_k1_MAT_s016
                                 + 0.737099 w_k1_MSFT_s016
                                 + 0.803729 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.47182 w_k1_ADM_s016
                                 + 1.31868 w_k1_AXP_s016
                                 + 1.66733 w_k1_DIS_s016
                                 + 1.37232 w_k1_FDX_s016
                                 + 1.15427 w_k1_IBM_s016
                                 + 1.32397 w_k1_INTC_s016
                                 + 1.30003 w_k1_KO_s016 + 1.11211 w_k1_MAT_s016
                                 + 1.30982 w_k1_MSFT_s016
                                 + 1.17121 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.30386 w_k1_ADM_s016
                                 + 1.31842 w_k1_AXP_s016
                                 + 1.33771 w_k1_DIS_s016 + 1.1763 w_k1_FDX_s016
                                 + 1.7423 w_k1_IBM_s016
                                 + 1.23036 w_k1_INTC_s016
                                 + 1.44242 w_k1_KO_s016 + 2.13042 w_k1_MAT_s016
                                 + 1.18269 w_k1_MSFT_s016
                                 + 1.29283 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.29195 w_k1_ADM_s016
                                 + 0.99121 w_k1_AXP_s016
                                 + 0.898077 w_k1_DIS_s016
                                 + 0.992841 w_k1_FDX_s016
                                 + 0.928874 w_k1_IBM_s016
                                 + 0.973985 w_k1_INTC_s016
                                 + 0.843923 w_k1_KO_s016 + 1.1933 w_k1_MAT_s016
                                 + 0.945884 w_k1_MSFT_s016
                                 + 0.941007 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.03138 w_k1_ADM_s016
                                 + 0.575096 w_k1_AXP_s016
                                 + 0.832841 w_k1_DIS_s016
                                 + 0.905164 w_k1_FDX_s016
                                 + 0.894462 w_k1_IBM_s016
                                 + 0.87043 w_k1_INTC_s016
                                 + 0.957229 w_k1_KO_s016
                                 + 0.683082 w_k1_MAT_s016
                                 + 1.02191 w_k1_MSFT_s016
                                 + 0.854097 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.16186 w_k1_ADM_s016
                                 + 1.03608 w_k1_AXP_s016
                                 + 0.940351 w_k1_DIS_s016
                                 + 0.931222 w_k1_FDX_s016
                                 + 1.20381 w_k1_IBM_s016
                                 + 1.01044 w_k1_INTC_s016
                                 + 1.08631 w_k1_KO_s016 + 1.54751 w_k1_MAT_s016
                                 + 1.03644 w_k1_MSFT_s016
                                 + 1.13906 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.995969 w_k1_ADM_s017
                                 + 0.705875 w_k1_AXP_s017
                                 + 0.678833 w_k1_DIS_s017
                                 + 0.787439 w_k1_FDX_s017
                                 + 0.72512 w_k1_IBM_s017
                                 + 0.625726 w_k1_INTC_s017
                                 + 0.686292 w_k1_KO_s017
                                 + 0.554752 w_k1_MAT_s017
                                 + 0.646874 w_k1_MSFT_s017
                                 + 0.712813 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.21168 w_k1_ADM_s017
                                 + 1.36655 w_k1_AXP_s017
                                 + 1.22922 w_k1_DIS_s017
                                 + 1.24327 w_k1_FDX_s017
                                 + 1.22839 w_k1_IBM_s017
                                 + 0.825256 w_k1_INTC_s017
                                 + 1.04506 w_k1_KO_s017 + 1.82723 w_k1_MAT_s017
                                 + 0.984763 w_k1_MSFT_s017
                                 + 1.12985 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.68755 w_k1_ADM_s017
                                 + 1.36009 w_k1_AXP_s017
                                 + 1.75272 w_k1_DIS_s017 + 1.4758 w_k1_FDX_s017
                                 + 1.03761 w_k1_IBM_s017
                                 + 0.92226 w_k1_INTC_s017
                                 + 1.32388 w_k1_KO_s017 + 1.08069 w_k1_MAT_s017
                                 + 1.08861 w_k1_MSFT_s017
                                 + 1.21909 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.938139 w_k1_ADM_s017
                                 + 0.854924 w_k1_AXP_s017
                                 + 1.00594 w_k1_DIS_s017
                                 + 0.806695 w_k1_FDX_s017
                                 + 1.19184 w_k1_IBM_s017
                                 + 0.766111 w_k1_INTC_s017
                                 + 1.12546 w_k1_KO_s017 + 1.2833 w_k1_MAT_s017
                                 + 0.869361 w_k1_MSFT_s017
                                 + 1.02542 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.03249 w_k1_ADM_s017
                                 + 1.25719 w_k1_AXP_s017
                                 + 0.969909 w_k1_DIS_s017
                                 + 1.07757 w_k1_FDX_s017
                                 + 1.07806 w_k1_IBM_s017
                                 + 0.914151 w_k1_INTC_s017
                                 + 1.01994 w_k1_KO_s017 + 1.35484 w_k1_MAT_s017
                                 + 0.961507 w_k1_MSFT_s017
                                 + 1.14021 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.29113 w_k1_ADM_s017
                                 + 0.948644 w_k1_AXP_s017
                                 + 1.21991 w_k1_DIS_s017
                                 + 1.15883 w_k1_FDX_s017
                                 + 1.01013 w_k1_IBM_s017
                                 + 1.09324 w_k1_INTC_s017
                                 + 1.06542 w_k1_KO_s017 + 0.80648 w_k1_MAT_s017
                                 + 1.02473 w_k1_MSFT_s017
                                 + 1.09286 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.923205 w_k1_ADM_s017
                                 + 0.99065 w_k1_AXP_s017
                                 + 0.850789 w_k1_DIS_s017
                                 + 0.920361 w_k1_FDX_s017
                                 + 0.886804 w_k1_IBM_s017
                                 + 0.785135 w_k1_INTC_s017
                                 + 0.906001 w_k1_KO_s017
                                 + 1.20999 w_k1_MAT_s017
                                 + 0.839669 w_k1_MSFT_s017
                                 + 0.935565 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.44955 w_k1_ADM_s017
                                 + 1.18968 w_k1_AXP_s017
                                 + 1.34987 w_k1_DIS_s017
                                 + 1.08536 w_k1_FDX_s017
                                 + 1.59027 w_k1_IBM_s017
                                 + 0.830492 w_k1_INTC_s017
                                 + 1.34318 w_k1_KO_s017 + 1.42636 w_k1_MAT_s017
                                 + 0.963903 w_k1_MSFT_s017
                                 + 1.25428 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.12303 w_k1_ADM_s100
                                 + 1.00834 w_k1_AXP_s100
                                 + 1.06449 w_k1_DIS_s100
                                 + 1.12778 w_k1_FDX_s100
                                 + 0.872474 w_k1_IBM_s100
                                 + 1.04454 w_k1_INTC_s100
                                 + 0.938142 w_k1_KO_s100
                                 + 0.857307 w_k1_MAT_s100
                                 + 1.18045 w_k1_MSFT_s100
                                 + 1.07102 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.838272 w_k1_ADM_s100
                                 + 0.83073 w_k1_AXP_s100
                                 + 0.80121 w_k1_DIS_s100
                                 + 0.783694 w_k1_FDX_s100
                                 + 1.16752 w_k1_IBM_s100
                                 + 0.821354 w_k1_INTC_s100
                                 + 1.04157 w_k1_KO_s100 + 1.11527 w_k1_MAT_s100
                                 + 0.807078 w_k1_MSFT_s100
                                 + 0.951405 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.15284 w_k1_ADM_s100
                                 + 1.15102 w_k1_AXP_s100
                                 + 1.41363 w_k1_DIS_s100 + 1.1661 w_k1_FDX_s100
                                 + 1.33267 w_k1_IBM_s100
                                 + 1.03121 w_k1_INTC_s100
                                 + 1.21208 w_k1_KO_s100 + 1.26055 w_k1_MAT_s100
                                 + 1.18228 w_k1_MSFT_s100
                                 + 1.19764 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.942525 w_k1_ADM_s100
                                 + 1.01776 w_k1_AXP_s100
                                 + 0.650319 w_k1_DIS_s100
                                 + 0.923327 w_k1_FDX_s100
                                 + 1.03489 w_k1_IBM_s100
                                 + 0.873641 w_k1_INTC_s100
                                 + 0.930549 w_k1_KO_s100
                                 + 1.21692 w_k1_MAT_s100
                                 + 0.865624 w_k1_MSFT_s100
                                 + 0.919561 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.954669 w_k1_ADM_s100
                                 + 0.765552 w_k1_AXP_s100
                                 + 0.949585 w_k1_DIS_s100
                                 + 0.979909 w_k1_FDX_s100
                                 + 0.816093 w_k1_IBM_s100
                                 + 1.05637 w_k1_INTC_s100
                                 + 0.963355 w_k1_KO_s100
                                 + 0.728538 w_k1_MAT_s100
                                 + 1.03432 w_k1_MSFT_s100
                                 + 0.983749 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.10156 w_k1_ADM_s100
                                 + 0.752413 w_k1_AXP_s100
                                 + 0.757749 w_k1_DIS_s100
                                 + 0.941281 w_k1_FDX_s100
                                 + 0.642063 w_k1_IBM_s100
                                 + 0.916901 w_k1_INTC_s100
                                 + 0.673549 w_k1_KO_s100
                                 + 0.464804 w_k1_MAT_s100
                                 + 0.946705 w_k1_MSFT_s100
                                 + 0.895722 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.47743 w_k1_ADM_s100
                                 + 1.43843 w_k1_AXP_s100
                                 + 1.55927 w_k1_DIS_s100
                                 + 1.53557 w_k1_FDX_s100
                                 + 1.05555 w_k1_IBM_s100
                                 + 1.23844 w_k1_INTC_s100
                                 + 1.23217 w_k1_KO_s100 + 1.19413 w_k1_MAT_s100
                                 + 1.53011 w_k1_MSFT_s100
                                 + 1.27706 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.27841 w_k1_ADM_s100
                                 + 1.0524 w_k1_AXP_s100 + 1.02401 w_k1_DIS_s100
                                 + 0.980008 w_k1_FDX_s100
                                 + 1.14224 w_k1_IBM_s100
                                 + 1.11978 w_k1_INTC_s100
                                 + 1.17966 w_k1_KO_s100 + 1.06149 w_k1_MAT_s100
                                 + 1.03984 w_k1_MSFT_s100
                                 + 1.13175 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.09035 w_k1_ADM_s101
                                 + 1.05632 w_k1_AXP_s101 + 1.0913 w_k1_DIS_s101
                                 + 1.22435 w_k1_FDX_s101
                                 + 0.640736 w_k1_IBM_s101
                                 + 1.00036 w_k1_INTC_s101
                                 + 1.05513 w_k1_KO_s101
                                 + 0.824717 w_k1_MAT_s101
                                 + 1.07562 w_k1_MSFT_s101
                                 + 1.01413 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.1357 w_k1_ADM_s101
                                 + 1.46936 w_k1_AXP_s101
                                 + 1.61651 w_k1_DIS_s101
                                 + 1.44767 w_k1_FDX_s101
                                 + 1.10235 w_k1_IBM_s101
                                 + 1.50753 w_k1_INTC_s101
                                 + 1.22735 w_k1_KO_s101 + 1.29381 w_k1_MAT_s101
                                 + 1.36379 w_k1_MSFT_s101
                                 + 1.30628 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.03907 w_k1_ADM_s101
                                 + 1.46329 w_k1_AXP_s101
                                 + 1.25417 w_k1_DIS_s101
                                 + 1.16664 w_k1_FDX_s101
                                 + 1.71241 w_k1_IBM_s101
                                 + 1.16489 w_k1_INTC_s101
                                 + 1.33976 w_k1_KO_s101 + 2.27153 w_k1_MAT_s101
                                 + 1.11055 w_k1_MSFT_s101
                                 + 1.17218 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.830796 w_k1_ADM_s101
                                 + 0.914731 w_k1_AXP_s101
                                 + 0.853598 w_k1_DIS_s101
                                 + 0.817988 w_k1_FDX_s101
                                 + 1.36898 w_k1_IBM_s101
                                 + 0.804611 w_k1_INTC_s101
                                 + 1.10198 w_k1_KO_s101 + 1.47848 w_k1_MAT_s101
                                 + 0.704664 w_k1_MSFT_s101
                                 + 0.883559 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.996047 w_k1_ADM_s101
                                 + 0.963383 w_k1_AXP_s101
                                 + 0.87615 w_k1_DIS_s101
                                 + 0.991867 w_k1_FDX_s101
                                 + 0.919143 w_k1_IBM_s101
                                 + 0.925498 w_k1_INTC_s101
                                 + 0.939106 w_k1_KO_s101
                                 + 0.880344 w_k1_MAT_s101
                                 + 0.910068 w_k1_MSFT_s101
                                 + 0.91537 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.987503 w_k1_ADM_s101
                                 + 0.950203 w_k1_AXP_s101
                                 + 1.31389 w_k1_DIS_s101
                                 + 1.04962 w_k1_FDX_s101
                                 + 0.971979 w_k1_IBM_s101
                                 + 1.11604 w_k1_INTC_s101
                                 + 1.27854 w_k1_KO_s101
                                 + 0.822183 w_k1_MAT_s101
                                 + 1.13415 w_k1_MSFT_s101
                                 + 1.09818 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.948356 w_k1_ADM_s101
                                 + 1.22591 w_k1_AXP_s101
                                 + 1.30035 w_k1_DIS_s101
                                 + 1.15186 w_k1_FDX_s101
                                 + 1.34534 w_k1_IBM_s101
                                 + 1.30637 w_k1_INTC_s101 + 1.4473 w_k1_KO_s101
                                 + 1.31654 w_k1_MAT_s101
                                 + 1.19536 w_k1_MSFT_s101
                                 + 1.04371 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.812943 w_k1_ADM_s101
                                 + 0.645372 w_k1_AXP_s101
                                 + 0.662935 w_k1_DIS_s101
                                 + 0.871583 w_k1_FDX_s101
                                 + 0.656399 w_k1_IBM_s101
                                 + 0.812416 w_k1_INTC_s101
                                 + 0.716672 w_k1_KO_s101
                                 + 0.611397 w_k1_MAT_s101
                                 + 0.854269 w_k1_MSFT_s101
                                 + 0.780211 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.08747 w_k1_ADM_s102
                                 + 1.0764 w_k1_AXP_s102 + 1.03073 w_k1_DIS_s102
                                 + 0.912362 w_k1_FDX_s102 + 1.363 w_k1_IBM_s102
                                 + 0.790878 w_k1_INTC_s102
                                 + 1.02001 w_k1_KO_s102 + 1.24401 w_k1_MAT_s102
                                 + 0.921465 w_k1_MSFT_s102
                                 + 1.07661 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.73626 w_k1_ADM_s102
                                 + 1.66441 w_k1_AXP_s102
                                 + 1.87707 w_k1_DIS_s102
                                 + 1.60996 w_k1_FDX_s102
                                 + 1.09633 w_k1_IBM_s102
                                 + 1.20828 w_k1_INTC_s102
                                 + 1.33729 w_k1_KO_s102 + 1.46488 w_k1_MAT_s102
                                 + 1.5402 w_k1_MSFT_s102 + 1.25663 w_k1_PG_s102
                                 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.894241 w_k1_ADM_s102
                                 + 0.933855 w_k1_AXP_s102
                                 + 0.859521 w_k1_DIS_s102
                                 + 0.988883 w_k1_FDX_s102
                                 + 0.734911 w_k1_IBM_s102
                                 + 0.967031 w_k1_INTC_s102
                                 + 0.778467 w_k1_KO_s102
                                 + 0.827054 w_k1_MAT_s102
                                 + 0.911972 w_k1_MSFT_s102
                                 + 1.01026 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.916512 w_k1_ADM_s102
                                 + 0.816156 w_k1_AXP_s102
                                 + 1.09816 w_k1_DIS_s102
                                 + 0.991637 w_k1_FDX_s102
                                 + 1.07606 w_k1_IBM_s102
                                 + 0.890333 w_k1_INTC_s102
                                 + 1.03525 w_k1_KO_s102
                                 + 0.838886 w_k1_MAT_s102
                                 + 0.971328 w_k1_MSFT_s102
                                 + 0.970848 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.04558 w_k1_ADM_s102
                                 + 1.23603 w_k1_AXP_s102
                                 + 0.992941 w_k1_DIS_s102
                                 + 1.16065 w_k1_FDX_s102
                                 + 0.768774 w_k1_IBM_s102
                                 + 1.04487 w_k1_INTC_s102
                                 + 0.813282 w_k1_KO_s102
                                 + 1.13965 w_k1_MAT_s102
                                 + 1.02423 w_k1_MSFT_s102
                                 + 1.06124 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.05769 w_k1_ADM_s102
                                 + 1.67952 w_k1_AXP_s102
                                 + 1.19016 w_k1_DIS_s102
                                 + 1.26428 w_k1_FDX_s102 + 1.1767 w_k1_IBM_s102
                                 + 1.28393 w_k1_INTC_s102
                                 + 1.05296 w_k1_KO_s102 + 1.23198 w_k1_MAT_s102
                                 + 1.24264 w_k1_MSFT_s102
                                 + 1.25792 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.98329 w_k1_ADM_s102
                                 + 0.798932 w_k1_AXP_s102
                                 + 0.735059 w_k1_DIS_s102
                                 + 0.816718 w_k1_FDX_s102
                                 + 0.90929 w_k1_IBM_s102
                                 + 0.709514 w_k1_INTC_s102
                                 + 0.801417 w_k1_KO_s102
                                 + 0.996671 w_k1_MAT_s102
                                 + 0.824062 w_k1_MSFT_s102
                                 + 0.93579 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.19549 w_k1_ADM_s102
                                 + 1.33907 w_k1_AXP_s102
                                 + 1.43879 w_k1_DIS_s102
                                 + 1.15381 w_k1_FDX_s102
                                 + 1.31865 w_k1_IBM_s102
                                 + 0.959749 w_k1_INTC_s102
                                 + 1.16008 w_k1_KO_s102 + 1.33729 w_k1_MAT_s102
                                 + 1.23084 w_k1_MSFT_s102
                                 + 1.17666 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.23812 w_k1_ADM_s103
                                 + 0.913094 w_k1_AXP_s103
                                 + 1.21044 w_k1_DIS_s103
                                 + 1.04338 w_k1_FDX_s103
                                 + 0.990212 w_k1_IBM_s103
                                 + 1.18769 w_k1_INTC_s103
                                 + 1.15658 w_k1_KO_s103
                                 + 0.793888 w_k1_MAT_s103
                                 + 1.12666 w_k1_MSFT_s103
                                 + 1.06388 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.3719 w_k1_ADM_s103
                                 + 0.795517 w_k1_AXP_s103
                                 + 1.02257 w_k1_DIS_s103
                                 + 1.04937 w_k1_FDX_s103
                                 + 0.608471 w_k1_IBM_s103
                                 + 1.03338 w_k1_INTC_s103
                                 + 0.791332 w_k1_KO_s103
                                 + 0.463683 w_k1_MAT_s103
                                 + 1.12874 w_k1_MSFT_s103
                                 + 0.877644 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.61449 w_k1_ADM_s103
                                 + 1.13349 w_k1_AXP_s103
                                 + 1.10483 w_k1_DIS_s103
                                 + 1.18184 w_k1_FDX_s103
                                 + 0.915484 w_k1_IBM_s103
                                 + 0.919527 w_k1_INTC_s103
                                 + 0.929009 w_k1_KO_s103 + 1.0787 w_k1_MAT_s103
                                 + 1.11817 w_k1_MSFT_s103
                                 + 1.12464 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.00578 w_k1_ADM_s103
                                 + 0.698538 w_k1_AXP_s103
                                 + 0.832895 w_k1_DIS_s103
                                 + 0.768524 w_k1_FDX_s103
                                 + 1.25234 w_k1_IBM_s103
                                 + 0.86871 w_k1_INTC_s103
                                 + 1.08094 w_k1_KO_s103
                                 + 0.895638 w_k1_MAT_s103
                                 + 0.960781 w_k1_MSFT_s103
                                 + 0.905183 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.50762 w_k1_ADM_s103
                                 + 1.39319 w_k1_AXP_s103
                                 + 1.66742 w_k1_DIS_s103
                                 + 1.39472 w_k1_FDX_s103
                                 + 1.29681 w_k1_IBM_s103
                                 + 1.29979 w_k1_INTC_s103
                                 + 1.44462 w_k1_KO_s103 + 1.6478 w_k1_MAT_s103
                                 + 1.46662 w_k1_MSFT_s103
                                 + 1.25788 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.12821 w_k1_ADM_s103
                                 + 0.830874 w_k1_AXP_s103
                                 + 0.730749 w_k1_DIS_s103
                                 + 0.870979 w_k1_FDX_s103
                                 + 0.716325 w_k1_IBM_s103
                                 + 0.863214 w_k1_INTC_s103
                                 + 0.831185 w_k1_KO_s103
                                 + 0.804358 w_k1_MAT_s103
                                 + 0.854662 w_k1_MSFT_s103
                                 + 0.886926 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.17898 w_k1_ADM_s103
                                 + 0.920766 w_k1_AXP_s103
                                 + 0.912052 w_k1_DIS_s103
                                 + 0.878567 w_k1_FDX_s103
                                 + 1.0781 w_k1_IBM_s103
                                 + 1.01667 w_k1_INTC_s103 + 1.0771 w_k1_KO_s103
                                 + 0.980824 w_k1_MAT_s103
                                 + 0.919622 w_k1_MSFT_s103
                                 + 1.00443 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.42681 w_k1_ADM_s103
                                 + 1.27502 w_k1_AXP_s103
                                 + 1.22717 w_k1_DIS_s103 + 1.1631 w_k1_FDX_s103
                                 + 1.26753 w_k1_IBM_s103
                                 + 0.959445 w_k1_INTC_s103
                                 + 1.32873 w_k1_KO_s103 + 1.2288 w_k1_MAT_s103
                                 + 1.2139 w_k1_MSFT_s103 + 1.18383 w_k1_PG_s103
                                 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.644298 w_k1_ADM_s104
                                 + 0.703411 w_k1_AXP_s104
                                 + 0.813845 w_k1_DIS_s104
                                 + 0.808238 w_k1_FDX_s104
                                 + 1.08811 w_k1_IBM_s104
                                 + 1.01707 w_k1_INTC_s104
                                 + 0.990714 w_k1_KO_s104
                                 + 1.06166 w_k1_MAT_s104
                                 + 0.931995 w_k1_MSFT_s104
                                 + 0.836331 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.18621 w_k1_ADM_s104
                                 + 1.22125 w_k1_AXP_s104
                                 + 1.67813 w_k1_DIS_s104
                                 + 1.34393 w_k1_FDX_s104
                                 + 0.936136 w_k1_IBM_s104
                                 + 1.18532 w_k1_INTC_s104
                                 + 1.19249 w_k1_KO_s104 + 1.02342 w_k1_MAT_s104
                                 + 1.32024 w_k1_MSFT_s104
                                 + 1.07354 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.981354 w_k1_ADM_s104
                                 + 1.09753 w_k1_AXP_s104
                                 + 0.941023 w_k1_DIS_s104
                                 + 0.92858 w_k1_FDX_s104
                                 + 0.879548 w_k1_IBM_s104
                                 + 0.808663 w_k1_INTC_s104
                                 + 0.961365 w_k1_KO_s104 + 1.0273 w_k1_MAT_s104
                                 + 0.960008 w_k1_MSFT_s104
                                 + 0.971307 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.779035 w_k1_ADM_s104
                                 + 1.07335 w_k1_AXP_s104
                                 + 0.722168 w_k1_DIS_s104
                                 + 0.845437 w_k1_FDX_s104
                                 + 1.06708 w_k1_IBM_s104
                                 + 0.953286 w_k1_INTC_s104
                                 + 1.00169 w_k1_KO_s104 + 1.12018 w_k1_MAT_s104
                                 + 0.944268 w_k1_MSFT_s104
                                 + 0.988432 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.802477 w_k1_ADM_s104
                                 + 0.960958 w_k1_AXP_s104
                                 + 1.05013 w_k1_DIS_s104
                                 + 0.88823 w_k1_FDX_s104
                                 + 1.62156 w_k1_IBM_s104
                                 + 1.06452 w_k1_INTC_s104
                                 + 1.32585 w_k1_KO_s104 + 1.67224 w_k1_MAT_s104
                                 + 1.12703 w_k1_MSFT_s104
                                 + 1.03335 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.22403 w_k1_ADM_s104
                                 + 1.31778 w_k1_AXP_s104
                                 + 1.45297 w_k1_DIS_s104
                                 + 1.11328 w_k1_FDX_s104
                                 + 1.26525 w_k1_IBM_s104
                                 + 1.06473 w_k1_INTC_s104
                                 + 1.33242 w_k1_KO_s104 + 1.65305 w_k1_MAT_s104
                                 + 1.26166 w_k1_MSFT_s104 + 1.2052 w_k1_PG_s104
                                 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.08819 w_k1_ADM_s104
                                 + 1.66901 w_k1_AXP_s104 + 1.4403 w_k1_DIS_s104
                                 + 1.39891 w_k1_FDX_s104
                                 + 1.15158 w_k1_IBM_s104
                                 + 1.40126 w_k1_INTC_s104
                                 + 1.20587 w_k1_KO_s104 + 1.2042 w_k1_MAT_s104
                                 + 1.43745 w_k1_MSFT_s104
                                 + 1.09255 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.990408 w_k1_ADM_s104
                                 + 0.922629 w_k1_AXP_s104
                                 + 0.958437 w_k1_DIS_s104
                                 + 1.05363 w_k1_FDX_s104
                                 + 0.789469 w_k1_IBM_s104
                                 + 1.17188 w_k1_INTC_s104
                                 + 1.00718 w_k1_KO_s104
                                 + 0.732278 w_k1_MAT_s104
                                 + 1.18073 w_k1_MSFT_s104
                                 + 1.10011 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.915491 w_k1_ADM_s105
                                 + 0.694435 w_k1_AXP_s105
                                 + 0.867032 w_k1_DIS_s105
                                 + 0.85848 w_k1_FDX_s105
                                 + 0.865922 w_k1_IBM_s105
                                 + 0.986671 w_k1_INTC_s105
                                 + 0.93465 w_k1_KO_s105
                                 + 0.786776 w_k1_MAT_s105
                                 + 0.966752 w_k1_MSFT_s105
                                 + 0.871471 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.01987 w_k1_ADM_s105
                                 + 1.24797 w_k1_AXP_s105
                                 + 1.13992 w_k1_DIS_s105
                                 + 1.08865 w_k1_FDX_s105
                                 + 1.55807 w_k1_IBM_s105
                                 + 1.27546 w_k1_INTC_s105
                                 + 1.48246 w_k1_KO_s105 + 1.50726 w_k1_MAT_s105
                                 + 1.25123 w_k1_MSFT_s105
                                 + 1.13562 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.630116 w_k1_ADM_s105
                                 + 0.608158 w_k1_AXP_s105
                                 + 0.625354 w_k1_DIS_s105
                                 + 0.6346 w_k1_FDX_s105
                                 + 0.958081 w_k1_IBM_s105
                                 + 0.78713 w_k1_INTC_s105
                                 + 0.882984 w_k1_KO_s105
                                 + 1.07228 w_k1_MAT_s105
                                 + 0.776319 w_k1_MSFT_s105
                                 + 0.846483 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.02472 w_k1_ADM_s105
                                 + 1.14615 w_k1_AXP_s105
                                 + 1.08072 w_k1_DIS_s105
                                 + 1.13291 w_k1_FDX_s105
                                 + 1.02075 w_k1_IBM_s105
                                 + 1.36747 w_k1_INTC_s105
                                 + 1.15848 w_k1_KO_s105 + 1.06845 w_k1_MAT_s105
                                 + 1.08696 w_k1_MSFT_s105
                                 + 0.989812 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.26714 w_k1_ADM_s105
                                 + 1.21293 w_k1_AXP_s105
                                 + 1.61157 w_k1_DIS_s105
                                 + 1.34204 w_k1_FDX_s105
                                 + 1.16062 w_k1_IBM_s105
                                 + 1.31621 w_k1_INTC_s105
                                 + 1.32842 w_k1_KO_s105 + 1.38071 w_k1_MAT_s105
                                 + 1.32551 w_k1_MSFT_s105
                                 + 1.11085 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.838188 w_k1_ADM_s105
                                 + 0.867122 w_k1_AXP_s105
                                 + 0.978454 w_k1_DIS_s105
                                 + 0.889559 w_k1_FDX_s105
                                 + 1.32347 w_k1_IBM_s105
                                 + 0.979804 w_k1_INTC_s105
                                 + 1.21785 w_k1_KO_s105 + 1.49844 w_k1_MAT_s105
                                 + 1.136 w_k1_MSFT_s105 + 0.981196 w_k1_PG_s105
                                 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.03015 w_k1_ADM_s105
                                 + 0.849728 w_k1_AXP_s105
                                 + 1.1813 w_k1_DIS_s105
                                 + 0.959809 w_k1_FDX_s105
                                 + 1.17323 w_k1_IBM_s105
                                 + 1.11512 w_k1_INTC_s105
                                 + 1.17469 w_k1_KO_s105 + 1.12457 w_k1_MAT_s105
                                 + 1.11697 w_k1_MSFT_s105
                                 + 0.990416 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.904031 w_k1_ADM_s105
                                 + 0.803492 w_k1_AXP_s105
                                 + 0.668353 w_k1_DIS_s105
                                 + 0.818332 w_k1_FDX_s105
                                 + 0.676588 w_k1_IBM_s105
                                 + 0.916745 w_k1_INTC_s105
                                 + 0.819087 w_k1_KO_s105
                                 + 0.971532 w_k1_MAT_s105
                                 + 0.87419 w_k1_MSFT_s105
                                 + 0.839747 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.07681 w_k1_ADM_s106
                                 + 1.03614 w_k1_AXP_s106
                                 + 0.86945 w_k1_DIS_s106
                                 + 0.922475 w_k1_FDX_s106
                                 + 0.959607 w_k1_IBM_s106
                                 + 0.949984 w_k1_INTC_s106
                                 + 0.961711 w_k1_KO_s106
                                 + 0.817879 w_k1_MAT_s106
                                 + 1.08346 w_k1_MSFT_s106
                                 + 1.03507 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.44307 w_k1_ADM_s106
                                 + 1.28615 w_k1_AXP_s106
                                 + 1.20828 w_k1_DIS_s106
                                 + 1.06244 w_k1_FDX_s106
                                 + 1.42674 w_k1_IBM_s106
                                 + 1.06509 w_k1_INTC_s106
                                 + 1.42097 w_k1_KO_s106 + 1.3365 w_k1_MAT_s106
                                 + 1.23836 w_k1_MSFT_s106
                                 + 1.26341 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.38879 w_k1_ADM_s106
                                 + 0.780501 w_k1_AXP_s106
                                 + 1.0671 w_k1_DIS_s106
                                 + 0.940308 w_k1_FDX_s106
                                 + 1.01953 w_k1_IBM_s106
                                 + 1.01397 w_k1_INTC_s106
                                 + 1.03658 w_k1_KO_s106
                                 + 0.788945 w_k1_MAT_s106
                                 + 1.10925 w_k1_MSFT_s106
                                 + 1.09069 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.11922 w_k1_ADM_s106
                                 + 0.770537 w_k1_AXP_s106
                                 + 0.740397 w_k1_DIS_s106
                                 + 0.789634 w_k1_FDX_s106
                                 + 1.03875 w_k1_IBM_s106
                                 + 0.833874 w_k1_INTC_s106
                                 + 0.936956 w_k1_KO_s106
                                 + 0.686798 w_k1_MAT_s106
                                 + 0.899397 w_k1_MSFT_s106
                                 + 0.91114 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.1138 w_k1_ADM_s106
                                 + 0.858516 w_k1_AXP_s106
                                 + 0.730758 w_k1_DIS_s106
                                 + 0.854388 w_k1_FDX_s106
                                 + 0.678892 w_k1_IBM_s106
                                 + 0.895561 w_k1_INTC_s106
                                 + 0.806808 w_k1_KO_s106
                                 + 0.760618 w_k1_MAT_s106
                                 + 0.952678 w_k1_MSFT_s106
                                 + 0.916752 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.67506 w_k1_ADM_s106
                                 + 1.76542 w_k1_AXP_s106
                                 + 1.87726 w_k1_DIS_s106
                                 + 1.46877 w_k1_FDX_s106
                                 + 1.18104 w_k1_IBM_s106
                                 + 1.53893 w_k1_INTC_s106
                                 + 1.38776 w_k1_KO_s106 + 1.30986 w_k1_MAT_s106
                                 + 1.59045 w_k1_MSFT_s106
                                 + 1.29772 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.15131 w_k1_ADM_s106
                                 + 1.124 w_k1_AXP_s106 + 1.39163 w_k1_DIS_s106
                                 + 1.12422 w_k1_FDX_s106
                                 + 1.15054 w_k1_IBM_s106
                                 + 1.29282 w_k1_INTC_s106
                                 + 1.35212 w_k1_KO_s106 + 1.07793 w_k1_MAT_s106
                                 + 1.35639 w_k1_MSFT_s106
                                 + 1.24998 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.48261 w_k1_ADM_s106
                                 + 0.902877 w_k1_AXP_s106
                                 + 1.13435 w_k1_DIS_s106
                                 + 1.13188 w_k1_FDX_s106
                                 + 0.728881 w_k1_IBM_s106
                                 + 0.887678 w_k1_INTC_s106
                                 + 1.07292 w_k1_KO_s106 + 0.69911 w_k1_MAT_s106
                                 + 1.12746 w_k1_MSFT_s106
                                 + 0.975838 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.01437 w_k1_ADM_s107
                                 + 1.06886 w_k1_AXP_s107
                                 + 1.16376 w_k1_DIS_s107 + 1.0623 w_k1_FDX_s107
                                 + 1.06832 w_k1_IBM_s107
                                 + 1.15991 w_k1_INTC_s107
                                 + 1.19944 w_k1_KO_s107 + 1.28022 w_k1_MAT_s107
                                 + 1.14206 w_k1_MSFT_s107
                                 + 1.16798 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.17179 w_k1_ADM_s107
                                 + 1.20289 w_k1_AXP_s107
                                 + 1.74243 w_k1_DIS_s107
                                 + 1.20587 w_k1_FDX_s107 + 1.2223 w_k1_IBM_s107
                                 + 1.49714 w_k1_INTC_s107
                                 + 1.46169 w_k1_KO_s107 + 1.10308 w_k1_MAT_s107
                                 + 1.24637 w_k1_MSFT_s107
                                 + 1.22999 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.17619 w_k1_ADM_s107
                                 + 1.02632 w_k1_AXP_s107
                                 + 1.20707 w_k1_DIS_s107
                                 + 0.971458 w_k1_FDX_s107
                                 + 1.42218 w_k1_IBM_s107
                                 + 0.889988 w_k1_INTC_s107
                                 + 1.41064 w_k1_KO_s107 + 1.55056 w_k1_MAT_s107
                                 + 0.996502 w_k1_MSFT_s107
                                 + 1.19862 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.99037 w_k1_ADM_s107
                                 + 0.809765 w_k1_AXP_s107
                                 + 0.77934 w_k1_DIS_s107
                                 + 0.752082 w_k1_FDX_s107
                                 + 1.3059 w_k1_IBM_s107
                                 + 0.763745 w_k1_INTC_s107
                                 + 1.12928 w_k1_KO_s107 + 1.34827 w_k1_MAT_s107
                                 + 0.861418 w_k1_MSFT_s107
                                 + 1.04518 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.1491 w_k1_ADM_s107
                                 + 0.829301 w_k1_AXP_s107
                                 + 0.833269 w_k1_DIS_s107
                                 + 0.889066 w_k1_FDX_s107
                                 + 0.888954 w_k1_IBM_s107
                                 + 0.764653 w_k1_INTC_s107
                                 + 0.90161 w_k1_KO_s107
                                 + 0.966163 w_k1_MAT_s107
                                 + 0.907221 w_k1_MSFT_s107
                                 + 0.892145 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.770703 w_k1_ADM_s107
                                 + 0.851938 w_k1_AXP_s107
                                 + 0.892554 w_k1_DIS_s107
                                 + 0.953392 w_k1_FDX_s107
                                 + 0.660865 w_k1_IBM_s107
                                 + 0.928823 w_k1_INTC_s107
                                 + 0.952809 w_k1_KO_s107
                                 + 0.653526 w_k1_MAT_s107
                                 + 0.940965 w_k1_MSFT_s107
                                 + 1.01152 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.7142 w_k1_ADM_s107
                                 + 1.54886 w_k1_AXP_s107
                                 + 2.01267 w_k1_DIS_s107
                                 + 1.44762 w_k1_FDX_s107 + 1.5857 w_k1_IBM_s107
                                 + 1.19193 w_k1_INTC_s107
                                 + 1.87884 w_k1_KO_s107 + 1.27455 w_k1_MAT_s107
                                 + 1.15788 w_k1_MSFT_s107
                                 + 1.30194 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.689523 w_k1_ADM_s107
                                 + 0.393054 w_k1_AXP_s107
                                 + 0.592214 w_k1_DIS_s107
                                 + 0.637295 w_k1_FDX_s107
                                 + 0.83748 w_k1_IBM_s107
                                 + 0.658532 w_k1_INTC_s107
                                 + 0.859834 w_k1_KO_s107
                                 + 0.714521 w_k1_MAT_s107
                                 + 0.696547 w_k1_MSFT_s107
                                 + 0.791906 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.13981 w_k1_ADM_s110
                                 + 1.2876 w_k1_AXP_s110 + 1.74282 w_k1_DIS_s110
                                 + 1.30436 w_k1_FDX_s110
                                 + 1.44968 w_k1_IBM_s110
                                 + 0.877799 w_k1_INTC_s110
                                 + 1.33124 w_k1_KO_s110 + 1.5158 w_k1_MAT_s110
                                 + 1.16082 w_k1_MSFT_s110
                                 + 1.22181 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.974897 w_k1_ADM_s110
                                 + 0.735264 w_k1_AXP_s110
                                 + 0.841194 w_k1_DIS_s110
                                 + 0.961395 w_k1_FDX_s110
                                 + 0.530292 w_k1_IBM_s110
                                 + 0.846439 w_k1_INTC_s110
                                 + 0.807131 w_k1_KO_s110
                                 + 0.512942 w_k1_MAT_s110
                                 + 1.01424 w_k1_MSFT_s110
                                 + 0.885268 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.749736 w_k1_ADM_s110
                                 + 0.875291 w_k1_AXP_s110
                                 + 0.704066 w_k1_DIS_s110
                                 + 0.716303 w_k1_FDX_s110
                                 + 1.08681 w_k1_IBM_s110
                                 + 0.561611 w_k1_INTC_s110
                                 + 0.895041 w_k1_KO_s110
                                 + 0.965483 w_k1_MAT_s110
                                 + 0.696022 w_k1_MSFT_s110
                                 + 0.904073 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.661892 w_k1_ADM_s110
                                 + 0.647475 w_k1_AXP_s110
                                 + 0.602998 w_k1_DIS_s110
                                 + 0.736825 w_k1_FDX_s110
                                 + 0.616948 w_k1_IBM_s110
                                 + 0.628594 w_k1_INTC_s110
                                 + 0.677789 w_k1_KO_s110
                                 + 0.549356 w_k1_MAT_s110
                                 + 0.753351 w_k1_MSFT_s110
                                 + 0.908775 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.910732 w_k1_ADM_s110
                                 + 1.03599 w_k1_AXP_s110
                                 + 0.924924 w_k1_DIS_s110
                                 + 0.949788 w_k1_FDX_s110
                                 + 1.23555 w_k1_IBM_s110
                                 + 0.777412 w_k1_INTC_s110
                                 + 0.950869 w_k1_KO_s110
                                 + 1.20548 w_k1_MAT_s110
                                 + 0.859259 w_k1_MSFT_s110
                                 + 0.994839 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.857376 w_k1_ADM_s110
                                 + 0.871956 w_k1_AXP_s110
                                 + 1.11673 w_k1_DIS_s110
                                 + 0.893763 w_k1_FDX_s110
                                 + 1.45817 w_k1_IBM_s110
                                 + 0.882323 w_k1_INTC_s110
                                 + 1.18628 w_k1_KO_s110 + 1.11665 w_k1_MAT_s110
                                 + 0.991636 w_k1_MSFT_s110
                                 + 1.06495 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.07584 w_k1_ADM_s110
                                 + 1.00276 w_k1_AXP_s110 + 1.1014 w_k1_DIS_s110
                                 + 1.14372 w_k1_FDX_s110
                                 + 0.912537 w_k1_IBM_s110
                                 + 0.859664 w_k1_INTC_s110
                                 + 0.918074 w_k1_KO_s110
                                 + 0.69907 w_k1_MAT_s110
                                 + 0.985775 w_k1_MSFT_s110
                                 + 1.08568 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.762406 w_k1_ADM_s110
                                 + 0.733663 w_k1_AXP_s110
                                 + 1.00122 w_k1_DIS_s110
                                 + 0.969957 w_k1_FDX_s110
                                 + 0.765008 w_k1_IBM_s110
                                 + 0.900524 w_k1_INTC_s110
                                 + 0.912951 w_k1_KO_s110
                                 + 0.739238 w_k1_MAT_s110
                                 + 1.02858 w_k1_MSFT_s110
                                 + 0.953555 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.972792 w_k1_ADM_s111
                                 + 1.09032 w_k1_AXP_s111
                                 + 1.02863 w_k1_DIS_s111
                                 + 1.01645 w_k1_FDX_s111
                                 + 1.09249 w_k1_IBM_s111
                                 + 0.961214 w_k1_INTC_s111
                                 + 1.00594 w_k1_KO_s111 + 1.17236 w_k1_MAT_s111
                                 + 0.937652 w_k1_MSFT_s111
                                 + 0.951499 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.08556 w_k1_ADM_s111
                                 + 1.37507 w_k1_AXP_s111
                                 + 1.53022 w_k1_DIS_s111
                                 + 1.24915 w_k1_FDX_s111
                                 + 1.60403 w_k1_IBM_s111
                                 + 1.64685 w_k1_INTC_s111
                                 + 1.36661 w_k1_KO_s111 + 1.63744 w_k1_MAT_s111
                                 + 1.29563 w_k1_MSFT_s111
                                 + 1.26757 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.710816 w_k1_ADM_s111
                                 + 0.735365 w_k1_AXP_s111
                                 + 0.574892 w_k1_DIS_s111
                                 + 0.740644 w_k1_FDX_s111
                                 + 0.677716 w_k1_IBM_s111
                                 + 0.939283 w_k1_INTC_s111
                                 + 0.614275 w_k1_KO_s111
                                 + 0.646271 w_k1_MAT_s111
                                 + 0.673597 w_k1_MSFT_s111
                                 + 0.78593 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.898802 w_k1_ADM_s111
                                 + 0.924249 w_k1_AXP_s111
                                 + 0.815677 w_k1_DIS_s111
                                 + 0.893125 w_k1_FDX_s111
                                 + 0.858852 w_k1_IBM_s111
                                 + 1.08218 w_k1_INTC_s111
                                 + 0.834291 w_k1_KO_s111
                                 + 1.01301 w_k1_MAT_s111
                                 + 0.890954 w_k1_MSFT_s111
                                 + 0.931651 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.03272 w_k1_ADM_s111
                                 + 0.830626 w_k1_AXP_s111
                                 + 0.945057 w_k1_DIS_s111
                                 + 0.853752 w_k1_FDX_s111
                                 + 1.36059 w_k1_IBM_s111
                                 + 1.07958 w_k1_INTC_s111
                                 + 1.05028 w_k1_KO_s111 + 1.15289 w_k1_MAT_s111
                                 + 0.856844 w_k1_MSFT_s111
                                 + 1.00662 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.03779 w_k1_ADM_s111
                                 + 1.14597 w_k1_AXP_s111
                                 + 1.22717 w_k1_DIS_s111
                                 + 1.17058 w_k1_FDX_s111
                                 + 0.767311 w_k1_IBM_s111
                                 + 1.07962 w_k1_INTC_s111
                                 + 0.790706 w_k1_KO_s111
                                 + 1.00488 w_k1_MAT_s111
                                 + 1.07039 w_k1_MSFT_s111
                                 + 1.01917 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.39253 w_k1_ADM_s111
                                 + 1.28048 w_k1_AXP_s111
                                 + 1.54054 w_k1_DIS_s111 + 1.3097 w_k1_FDX_s111
                                 + 1.04661 w_k1_IBM_s111
                                 + 1.39074 w_k1_INTC_s111 + 1.144 w_k1_KO_s111
                                 + 1.37587 w_k1_MAT_s111
                                 + 1.28874 w_k1_MSFT_s111
                                 + 1.25955 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.919453 w_k1_ADM_s111
                                 + 0.783149 w_k1_AXP_s111
                                 + 0.85424 w_k1_DIS_s111
                                 + 0.965631 w_k1_FDX_s111
                                 + 0.710833 w_k1_IBM_s111
                                 + 1.10185 w_k1_INTC_s111
                                 + 0.856396 w_k1_KO_s111
                                 + 0.712782 w_k1_MAT_s111
                                 + 0.869116 w_k1_MSFT_s111
                                 + 0.952097 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.09434 w_k1_ADM_s112
                                 + 0.87937 w_k1_AXP_s112
                                 + 0.857392 w_k1_DIS_s112
                                 + 0.957546 w_k1_FDX_s112
                                 + 0.866957 w_k1_IBM_s112
                                 + 0.890071 w_k1_INTC_s112
                                 + 1.17678 w_k1_KO_s112
                                 + 0.992911 w_k1_MAT_s112
                                 + 1.04006 w_k1_MSFT_s112
                                 + 0.93087 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.720457 w_k1_ADM_s112
                                 + 0.512179 w_k1_AXP_s112
                                 + 0.573863 w_k1_DIS_s112
                                 + 0.678257 w_k1_FDX_s112
                                 + 0.717585 w_k1_IBM_s112
                                 + 0.637392 w_k1_INTC_s112
                                 + 0.819094 w_k1_KO_s112
                                 + 0.652866 w_k1_MAT_s112
                                 + 0.612461 w_k1_MSFT_s112
                                 + 0.738257 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.28924 w_k1_ADM_s112
                                 + 1.12901 w_k1_AXP_s112
                                 + 1.42522 w_k1_DIS_s112
                                 + 1.29681 w_k1_FDX_s112
                                 + 0.952137 w_k1_IBM_s112
                                 + 1.20313 w_k1_INTC_s112
                                 + 1.22933 w_k1_KO_s112
                                 + 0.895088 w_k1_MAT_s112
                                 + 1.21258 w_k1_MSFT_s112
                                 + 1.01422 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.990047 w_k1_ADM_s112
                                 + 0.706854 w_k1_AXP_s112
                                 + 0.804363 w_k1_DIS_s112
                                 + 0.828272 w_k1_FDX_s112
                                 + 0.83482 w_k1_IBM_s112
                                 + 0.879287 w_k1_INTC_s112
                                 + 0.933814 w_k1_KO_s112
                                 + 0.543986 w_k1_MAT_s112
                                 + 0.857994 w_k1_MSFT_s112
                                 + 0.803035 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.59667 w_k1_ADM_s112
                                 + 1.07521 w_k1_AXP_s112
                                 + 1.27898 w_k1_DIS_s112
                                 + 1.13445 w_k1_FDX_s112
                                 + 1.51716 w_k1_IBM_s112
                                 + 0.981388 w_k1_INTC_s112
                                 + 1.56567 w_k1_KO_s112 + 1.7192 w_k1_MAT_s112
                                 + 1.09917 w_k1_MSFT_s112
                                 + 1.19094 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.936934 w_k1_ADM_s112
                                 + 0.772655 w_k1_AXP_s112
                                 + 0.734139 w_k1_DIS_s112
                                 + 0.973035 w_k1_FDX_s112
                                 + 0.601278 w_k1_IBM_s112
                                 + 0.931894 w_k1_INTC_s112
                                 + 0.890527 w_k1_KO_s112
                                 + 0.677562 w_k1_MAT_s112
                                 + 1.00341 w_k1_MSFT_s112
                                 + 0.812828 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.34887 w_k1_ADM_s112
                                 + 0.822724 w_k1_AXP_s112
                                 + 1.03324 w_k1_DIS_s112
                                 + 0.966922 w_k1_FDX_s112
                                 + 1.11734 w_k1_IBM_s112
                                 + 1.01402 w_k1_INTC_s112
                                 + 1.22667 w_k1_KO_s112 + 1.01307 w_k1_MAT_s112
                                 + 1.04618 w_k1_MSFT_s112
                                 + 0.982328 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.29501 w_k1_ADM_s112
                                 + 0.699582 w_k1_AXP_s112
                                 + 1.2196 w_k1_DIS_s112 + 1.0381 w_k1_FDX_s112
                                 + 0.855423 w_k1_IBM_s112
                                 + 0.983182 w_k1_INTC_s112
                                 + 1.28275 w_k1_KO_s112
                                 + 0.839382 w_k1_MAT_s112
                                 + 1.14142 w_k1_MSFT_s112
                                 + 0.977748 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.14441 w_k1_ADM_s113
                                 + 0.912726 w_k1_AXP_s113
                                 + 1.19951 w_k1_DIS_s113
                                 + 1.07988 w_k1_FDX_s113
                                 + 0.768289 w_k1_IBM_s113
                                 + 1.13545 w_k1_INTC_s113
                                 + 1.04025 w_k1_KO_s113
                                 + 0.738757 w_k1_MAT_s113
                                 + 1.16414 w_k1_MSFT_s113
                                 + 1.02555 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.24557 w_k1_ADM_s113
                                 + 1.2913 w_k1_AXP_s113 + 1.3189 w_k1_DIS_s113
                                 + 1.17842 w_k1_FDX_s113
                                 + 1.26332 w_k1_IBM_s113
                                 + 1.38067 w_k1_INTC_s113
                                 + 1.39229 w_k1_KO_s113 + 1.6345 w_k1_MAT_s113
                                 + 1.33083 w_k1_MSFT_s113
                                 + 1.25094 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.915312 w_k1_ADM_s113
                                 + 0.869517 w_k1_AXP_s113
                                 + 0.931568 w_k1_DIS_s113
                                 + 0.891737 w_k1_FDX_s113
                                 + 1.41452 w_k1_IBM_s113
                                 + 0.995014 w_k1_INTC_s113
                                 + 1.42024 w_k1_KO_s113 + 1.30547 w_k1_MAT_s113
                                 + 1.04485 w_k1_MSFT_s113
                                 + 1.01579 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.99662 w_k1_ADM_s113
                                 + 1.06538 w_k1_AXP_s113
                                 + 0.79287 w_k1_DIS_s113
                                 + 0.912309 w_k1_FDX_s113
                                 + 0.886344 w_k1_IBM_s113
                                 + 1.03601 w_k1_INTC_s113
                                 + 0.950366 w_k1_KO_s113
                                 + 0.876209 w_k1_MAT_s113
                                 + 0.90634 w_k1_MSFT_s113
                                 + 0.914689 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.866925 w_k1_ADM_s113
                                 + 0.67245 w_k1_AXP_s113
                                 + 0.977595 w_k1_DIS_s113
                                 + 0.885603 w_k1_FDX_s113
                                 + 0.748634 w_k1_IBM_s113
                                 + 0.856418 w_k1_INTC_s113
                                 + 0.924158 w_k1_KO_s113
                                 + 0.488605 w_k1_MAT_s113
                                 + 0.985064 w_k1_MSFT_s113
                                 + 0.880137 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.922087 w_k1_ADM_s113
                                 + 0.965112 w_k1_AXP_s113
                                 + 1.12053 w_k1_DIS_s113
                                 + 0.935198 w_k1_FDX_s113
                                 + 1.06706 w_k1_IBM_s113
                                 + 0.994733 w_k1_INTC_s113
                                 + 1.08999 w_k1_KO_s113 + 1.42634 w_k1_MAT_s113
                                 + 1.08378 w_k1_MSFT_s113
                                 + 1.06139 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.650756 w_k1_ADM_s113
                                 + 0.567821 w_k1_AXP_s113
                                 + 0.651369 w_k1_DIS_s113
                                 + 0.673472 w_k1_FDX_s113
                                 + 0.760128 w_k1_IBM_s113
                                 + 0.831509 w_k1_INTC_s113
                                 + 0.786028 w_k1_KO_s113
                                 + 0.711034 w_k1_MAT_s113
                                 + 0.865368 w_k1_MSFT_s113
                                 + 0.775646 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.911826 w_k1_ADM_s113
                                 + 0.760549 w_k1_AXP_s113
                                 + 1.44238 w_k1_DIS_s113
                                 + 1.08679 w_k1_FDX_s113 + 1.0408 w_k1_IBM_s113
                                 + 1.14137 w_k1_INTC_s113
                                 + 1.39701 w_k1_KO_s113 + 1.18348 w_k1_MAT_s113
                                 + 1.32337 w_k1_MSFT_s113
                                 + 1.10539 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.914115 w_k1_ADM_s114
                                 + 0.796601 w_k1_AXP_s114
                                 + 0.762719 w_k1_DIS_s114
                                 + 0.860717 w_k1_FDX_s114
                                 + 0.799278 w_k1_IBM_s114
                                 + 0.878886 w_k1_INTC_s114
                                 + 0.883362 w_k1_KO_s114
                                 + 0.753364 w_k1_MAT_s114
                                 + 0.939234 w_k1_MSFT_s114
                                 + 0.895694 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.12078 w_k1_ADM_s114
                                 + 1.34076 w_k1_AXP_s114
                                 + 1.03145 w_k1_DIS_s114 + 1.2289 w_k1_FDX_s114
                                 + 0.805815 w_k1_IBM_s114
                                 + 0.908293 w_k1_INTC_s114
                                 + 0.898017 w_k1_KO_s114 + 1.0908 w_k1_MAT_s114
                                 + 0.979443 w_k1_MSFT_s114
                                 + 1.04448 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.01632 w_k1_ADM_s114
                                 + 0.820827 w_k1_AXP_s114
                                 + 1.04682 w_k1_DIS_s114
                                 + 1.03019 w_k1_FDX_s114
                                 + 0.804368 w_k1_IBM_s114
                                 + 0.960004 w_k1_INTC_s114
                                 + 0.985556 w_k1_KO_s114
                                 + 0.929485 w_k1_MAT_s114
                                 + 1.19419 w_k1_MSFT_s114
                                 + 0.981154 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.957205 w_k1_ADM_s114
                                 + 1.04205 w_k1_AXP_s114
                                 + 1.20266 w_k1_DIS_s114
                                 + 1.02402 w_k1_FDX_s114
                                 + 1.09465 w_k1_IBM_s114
                                 + 0.983864 w_k1_INTC_s114
                                 + 1.23594 w_k1_KO_s114 + 1.01932 w_k1_MAT_s114
                                 + 0.989686 w_k1_MSFT_s114
                                 + 1.08824 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.12495 w_k1_ADM_s114
                                 + 1.24768 w_k1_AXP_s114
                                 + 1.05828 w_k1_DIS_s114
                                 + 1.01946 w_k1_FDX_s114
                                 + 1.51761 w_k1_IBM_s114
                                 + 0.952398 w_k1_INTC_s114
                                 + 1.36329 w_k1_KO_s114 + 1.51996 w_k1_MAT_s114
                                 + 0.939244 w_k1_MSFT_s114
                                 + 1.01087 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.14346 w_k1_ADM_s114
                                 + 1.31756 w_k1_AXP_s114
                                 + 1.67025 w_k1_DIS_s114
                                 + 1.27623 w_k1_FDX_s114
                                 + 1.22848 w_k1_IBM_s114
                                 + 1.17909 w_k1_INTC_s114
                                 + 1.52729 w_k1_KO_s114 + 1.61312 w_k1_MAT_s114
                                 + 1.40658 w_k1_MSFT_s114
                                 + 1.33523 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.771161 w_k1_ADM_s114
                                 + 0.839529 w_k1_AXP_s114
                                 + 0.69178 w_k1_DIS_s114
                                 + 0.771598 w_k1_FDX_s114
                                 + 1.10376 w_k1_IBM_s114
                                 + 0.653775 w_k1_INTC_s114
                                 + 0.853685 w_k1_KO_s114
                                 + 1.33004 w_k1_MAT_s114
                                 + 0.810818 w_k1_MSFT_s114
                                 + 0.784615 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.09756 w_k1_ADM_s114
                                 + 1.00502 w_k1_AXP_s114
                                 + 0.99954 w_k1_DIS_s114
                                 + 0.934862 w_k1_FDX_s114
                                 + 1.26563 w_k1_IBM_s114
                                 + 1.07946 w_k1_INTC_s114
                                 + 1.16643 w_k1_KO_s114 + 1.06961 w_k1_MAT_s114
                                 + 1.03903 w_k1_MSFT_s114
                                 + 1.03496 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.850852 w_k1_ADM_s115
                                 + 0.639077 w_k1_AXP_s115
                                 + 0.515512 w_k1_DIS_s115
                                 + 0.690049 w_k1_FDX_s115
                                 + 0.811381 w_k1_IBM_s115
                                 + 0.853082 w_k1_INTC_s115
                                 + 0.735335 w_k1_KO_s115
                                 + 0.63184 w_k1_MAT_s115
                                 + 0.737043 w_k1_MSFT_s115
                                 + 0.779637 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.95436 w_k1_ADM_s115
                                 + 0.719307 w_k1_AXP_s115
                                 + 0.705939 w_k1_DIS_s115
                                 + 0.874344 w_k1_FDX_s115
                                 + 0.718939 w_k1_IBM_s115
                                 + 1.01744 w_k1_INTC_s115
                                 + 0.828301 w_k1_KO_s115
                                 + 0.531388 w_k1_MAT_s115
                                 + 0.951174 w_k1_MSFT_s115
                                 + 0.845655 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.03357 w_k1_ADM_s115
                                 + 0.89583 w_k1_AXP_s115
                                 + 0.718832 w_k1_DIS_s115
                                 + 0.801098 w_k1_FDX_s115
                                 + 0.933754 w_k1_IBM_s115
                                 + 0.864835 w_k1_INTC_s115
                                 + 0.822146 w_k1_KO_s115
                                 + 0.902581 w_k1_MAT_s115
                                 + 0.759514 w_k1_MSFT_s115
                                 + 0.843351 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.36267 w_k1_ADM_s115
                                 + 1.39506 w_k1_AXP_s115 + 1.4821 w_k1_DIS_s115
                                 + 1.20785 w_k1_FDX_s115
                                 + 1.61466 w_k1_IBM_s115
                                 + 1.19925 w_k1_INTC_s115
                                 + 1.51649 w_k1_KO_s115 + 1.76046 w_k1_MAT_s115
                                 + 1.3697 w_k1_MSFT_s115 + 1.29258 w_k1_PG_s115
                                 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.02838 w_k1_ADM_s115
                                 + 0.87003 w_k1_AXP_s115
                                 + 0.653032 w_k1_DIS_s115
                                 + 0.932247 w_k1_FDX_s115
                                 + 0.612074 w_k1_IBM_s115
                                 + 0.883742 w_k1_INTC_s115
                                 + 0.695572 w_k1_KO_s115
                                 + 0.483441 w_k1_MAT_s115
                                 + 0.845709 w_k1_MSFT_s115
                                 + 0.830306 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.10074 w_k1_ADM_s115
                                 + 1.07997 w_k1_AXP_s115
                                 + 0.999713 w_k1_DIS_s115
                                 + 0.999324 w_k1_FDX_s115
                                 + 1.19519 w_k1_IBM_s115
                                 + 0.962528 w_k1_INTC_s115
                                 + 1.13247 w_k1_KO_s115 + 1.20683 w_k1_MAT_s115
                                 + 0.906752 w_k1_MSFT_s115
                                 + 1.0635 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.33474 w_k1_ADM_s115
                                 + 1.27108 w_k1_AXP_s115
                                 + 1.27398 w_k1_DIS_s115
                                 + 1.35271 w_k1_FDX_s115
                                 + 1.05025 w_k1_IBM_s115
                                 + 1.36246 w_k1_INTC_s115
                                 + 1.02047 w_k1_KO_s115
                                 + 0.846459 w_k1_MAT_s115
                                 + 1.11399 w_k1_MSFT_s115
                                 + 1.09633 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.19568 w_k1_ADM_s115
                                 + 1.04243 w_k1_AXP_s115
                                 + 0.88756 w_k1_DIS_s115
                                 + 1.09544 w_k1_FDX_s115
                                 + 0.898909 w_k1_IBM_s115
                                 + 1.06743 w_k1_INTC_s115
                                 + 0.906216 w_k1_KO_s115
                                 + 0.965016 w_k1_MAT_s115
                                 + 0.96258 w_k1_MSFT_s115
                                 + 0.917484 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.838302 w_k1_ADM_s116
                                 + 0.814879 w_k1_AXP_s116
                                 + 0.849035 w_k1_DIS_s116
                                 + 0.834463 w_k1_FDX_s116
                                 + 0.908633 w_k1_IBM_s116
                                 + 1.03189 w_k1_INTC_s116
                                 + 0.91193 w_k1_KO_s116 + 1.01816 w_k1_MAT_s116
                                 + 0.921869 w_k1_MSFT_s116
                                 + 0.915603 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.726896 w_k1_ADM_s116
                                 + 0.605439 w_k1_AXP_s116
                                 + 0.597997 w_k1_DIS_s116
                                 + 0.661639 w_k1_FDX_s116
                                 + 0.681379 w_k1_IBM_s116
                                 + 0.976451 w_k1_INTC_s116
                                 + 0.730721 w_k1_KO_s116
                                 + 0.586473 w_k1_MAT_s116
                                 + 0.801363 w_k1_MSFT_s116
                                 + 0.819692 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.21089 w_k1_ADM_s116
                                 + 1.18505 w_k1_AXP_s116
                                 + 1.06195 w_k1_DIS_s116 + 1.0191 w_k1_FDX_s116
                                 + 0.999158 w_k1_IBM_s116
                                 + 1.20203 w_k1_INTC_s116
                                 + 1.01372 w_k1_KO_s116
                                 + 0.752838 w_k1_MAT_s116
                                 + 0.958899 w_k1_MSFT_s116
                                 + 1.02291 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.976979 w_k1_ADM_s116
                                 + 1.01856 w_k1_AXP_s116
                                 + 0.758086 w_k1_DIS_s116
                                 + 0.965644 w_k1_FDX_s116
                                 + 0.722507 w_k1_IBM_s116
                                 + 1.06169 w_k1_INTC_s116
                                 + 0.736316 w_k1_KO_s116
                                 + 0.767075 w_k1_MAT_s116
                                 + 0.89119 w_k1_MSFT_s116
                                 + 0.94342 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.960955 w_k1_ADM_s116
                                 + 0.923964 w_k1_AXP_s116
                                 + 0.620193 w_k1_DIS_s116
                                 + 0.788044 w_k1_FDX_s116
                                 + 1.04183 w_k1_IBM_s116
                                 + 0.916763 w_k1_INTC_s116
                                 + 0.859408 w_k1_KO_s116
                                 + 0.825275 w_k1_MAT_s116
                                 + 0.73918 w_k1_MSFT_s116
                                 + 0.839336 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.968688 w_k1_ADM_s116
                                 + 0.863858 w_k1_AXP_s116
                                 + 1.21392 w_k1_DIS_s116
                                 + 1.08803 w_k1_FDX_s116
                                 + 0.777432 w_k1_IBM_s116
                                 + 1.21936 w_k1_INTC_s116
                                 + 0.877456 w_k1_KO_s116
                                 + 0.661685 w_k1_MAT_s116
                                 + 1.156 w_k1_MSFT_s116 + 1.07037 w_k1_PG_s116
                                 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.09748 w_k1_ADM_s116
                                 + 1.60257 w_k1_AXP_s116
                                 + 1.67074 w_k1_DIS_s116
                                 + 1.32874 w_k1_FDX_s116
                                 + 1.28742 w_k1_IBM_s116
                                 + 1.56817 w_k1_INTC_s116
                                 + 1.34085 w_k1_KO_s116 + 1.43927 w_k1_MAT_s116
                                 + 1.38421 w_k1_MSFT_s116
                                 + 1.22495 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.03041 w_k1_ADM_s116
                                 + 1.13289 w_k1_AXP_s116 + 1.1619 w_k1_DIS_s116
                                 + 1.04561 w_k1_FDX_s116
                                 + 1.31556 w_k1_IBM_s116
                                 + 1.11118 w_k1_INTC_s116
                                 + 1.10395 w_k1_KO_s116 + 1.37895 w_k1_MAT_s116
                                 + 1.09129 w_k1_MSFT_s116
                                 + 1.07413 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.979078 w_k1_ADM_s117
                                 + 1.16262 w_k1_AXP_s117 + 1.7433 w_k1_DIS_s117
                                 + 1.26471 w_k1_FDX_s117
                                 + 1.38464 w_k1_IBM_s117
                                 + 1.16862 w_k1_INTC_s117
                                 + 1.24097 w_k1_KO_s117 + 1.52065 w_k1_MAT_s117
                                 + 1.47051 w_k1_MSFT_s117
                                 + 1.21691 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.97987 w_k1_ADM_s117
                                 + 0.994579 w_k1_AXP_s117
                                 + 1.15007 w_k1_DIS_s117
                                 + 1.22249 w_k1_FDX_s117
                                 + 0.838049 w_k1_IBM_s117
                                 + 1.19652 w_k1_INTC_s117
                                 + 1.06204 w_k1_KO_s117 + 1.13858 w_k1_MAT_s117
                                 + 1.13438 w_k1_MSFT_s117
                                 + 1.10911 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.849954 w_k1_ADM_s117
                                 + 1.01303 w_k1_AXP_s117
                                 + 0.943906 w_k1_DIS_s117
                                 + 0.860821 w_k1_FDX_s117
                                 + 1.36833 w_k1_IBM_s117
                                 + 0.964264 w_k1_INTC_s117
                                 + 1.05109 w_k1_KO_s117 + 1.38883 w_k1_MAT_s117
                                 + 1.02453 w_k1_MSFT_s117
                                 + 1.03434 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.870681 w_k1_ADM_s117
                                 + 0.771342 w_k1_AXP_s117
                                 + 0.866949 w_k1_DIS_s117
                                 + 0.877453 w_k1_FDX_s117
                                 + 0.766128 w_k1_IBM_s117
                                 + 1.00015 w_k1_INTC_s117
                                 + 0.792895 w_k1_KO_s117
                                 + 0.595422 w_k1_MAT_s117
                                 + 0.999729 w_k1_MSFT_s117
                                 + 0.834354 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.06582 w_k1_ADM_s117
                                 + 1.25229 w_k1_AXP_s117
                                 + 0.955757 w_k1_DIS_s117
                                 + 1.08108 w_k1_FDX_s117
                                 + 0.961554 w_k1_IBM_s117
                                 + 1.03574 w_k1_INTC_s117
                                 + 0.868541 w_k1_KO_s117
                                 + 1.26929 w_k1_MAT_s117
                                 + 0.981358 w_k1_MSFT_s117
                                 + 0.897187 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.708624 w_k1_ADM_s117
                                 + 0.75994 w_k1_AXP_s117
                                 + 0.685924 w_k1_DIS_s117
                                 + 0.767715 w_k1_FDX_s117
                                 + 0.767045 w_k1_IBM_s117
                                 + 0.759148 w_k1_INTC_s117
                                 + 0.689485 w_k1_KO_s117
                                 + 1.00736 w_k1_MAT_s117
                                 + 0.762999 w_k1_MSFT_s117
                                 + 0.789594 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.734844 w_k1_ADM_s117
                                 + 0.802395 w_k1_AXP_s117
                                 + 0.834209 w_k1_DIS_s117
                                 + 0.895131 w_k1_FDX_s117
                                 + 1.03394 w_k1_IBM_s117
                                 + 0.757269 w_k1_INTC_s117
                                 + 0.893974 w_k1_KO_s117
                                 + 1.24849 w_k1_MAT_s117
                                 + 0.942285 w_k1_MSFT_s117
                                 + 0.894244 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.18392 w_k1_ADM_s117
                                 + 1.73582 w_k1_AXP_s117
                                 + 1.41173 w_k1_DIS_s117
                                 + 1.35488 w_k1_FDX_s117
                                 + 1.29659 w_k1_IBM_s117
                                 + 1.19463 w_k1_INTC_s117
                                 + 1.15202 w_k1_KO_s117 + 1.53686 w_k1_MAT_s117
                                 + 1.1214 w_k1_MSFT_s117 + 1.24212 w_k1_PG_s117
                                 >= 1
 k1_return_constraint:           0.014207362635287 w_k1_ADM_s000
                                 + 0.0092984702582902 w_k1_ADM_s001
                                 + 0.00721225173240369 w_k1_ADM_s002
                                 + 0.0336887041369742 w_k1_ADM_s003
                                 + 0.0107322591949279 w_k1_ADM_s004
                                 + 0.023378998597293 w_k1_ADM_s005
                                 + 0.0177281249285816 w_k1_ADM_s006
                                 + 0.0208795187574943 w_k1_ADM_s007
                                 + 0.0142960539278678 w_k1_ADM_s010
                                 + 0.00615420641714959 w_k1_ADM_s011
                                 + 0.00515229344601895 w_k1_ADM_s012
                                 + 0.0104123316898295 w_k1_ADM_s013
                                 + 0.0130894385119958 w_k1_ADM_s014
                                 + 0.00877378096933493 w_k1_ADM_s015
                                 + 0.0156194012241499 w_k1_ADM_s016
                                 + 0.0047372487300748 w_k1_ADM_s017
                                 + 0.108462386896619 w_k1_ADM_s100
                                 + 0.075420036733928 w_k1_ADM_s101
                                 + 0.0559125099562611 w_k1_ADM_s102
                                 + 0.0842059173378585 w_k1_ADM_s103
                                 + 0.0576748229848401 w_k1_ADM_s104
                                 + 0.0504745173551114 w_k1_ADM_s105
                                 + 0.0402115772965516 w_k1_ADM_s106
                                 + 0.0278728778611885 w_k1_ADM_s107
                                 + 0.0180926338558343 w_k1_ADM_s110
                                 + 0.0552860761541158 w_k1_ADM_s111
                                 + 0.0323072324243773 w_k1_ADM_s112
                                 + 0.0267015917205585 w_k1_ADM_s113
                                 + 0.0300030782282657 w_k1_ADM_s114
                                 + 0.0311960953899226 w_k1_ADM_s115
                                 + 0.0740624095550009 w_k1_ADM_s116
                                 + 0.0599201356641158 w_k1_ADM_s117
                                 + 0.0138965359193611 w_k1_AXP_s000
                                 + 0.00929263751791685 w_k1_AXP_s001
                                 + 0.00576974634209554 w_k1_AXP_s002
                                 + 0.0271410954305632 w_k1_AXP_s003
                                 + 0.00952353426547744 w_k1_AXP_s004
                                 + 0.021276662168335 w_k1_AXP_s005
                                 + 0.0185397209140852 w_k1_AXP_s006
                                 + 0.0218912988533894 w_k1_AXP_s007
                                 + 0.0167731635274964 w_k1_AXP_s010
                                 + 0.00579116478433292 w_k1_AXP_s011
                                 + 0.00480462464089942 w_k1_AXP_s012
                                 + 0.0105926004747272 w_k1_AXP_s013
                                 + 0.0102147680864384 w_k1_AXP_s014
                                 + 0.00983189902716142 w_k1_AXP_s015
                                 + 0.0121039699792916 w_k1_AXP_s016
                                 + 0.00422339438807387 w_k1_AXP_s017
                                 + 0.0983894831711024 w_k1_AXP_s100
                                 + 0.0783224043780376 w_k1_AXP_s101
                                 + 0.0570018057110021 w_k1_AXP_s102
                                 + 0.0641876638396654 w_k1_AXP_s103
                                 + 0.0660585141695683 w_k1_AXP_s104
                                 + 0.0486346383010889 w_k1_AXP_s105
                                 + 0.033966868606855 w_k1_AXP_s106
                                 + 0.0253278571160151 w_k1_AXP_s107
                                 + 0.0181687952296738 w_k1_AXP_s110
                                 + 0.0559768414508684 w_k1_AXP_s111
                                 + 0.0232299213640464 w_k1_AXP_s112
                                 + 0.0256623889698706 w_k1_AXP_s113
                                 + 0.0312826805881949 w_k1_AXP_s114
                                 + 0.0284328914315268 w_k1_AXP_s115
                                 + 0.0803060039659792 w_k1_AXP_s116
                                 + 0.068923020367403 w_k1_AXP_s117
                                 + 0.0153793105945346 w_k1_DIS_s000
                                 + 0.009127561683411 w_k1_DIS_s001
                                 + 0.00609022369835197 w_k1_DIS_s002
                                 + 0.0297490772668848 w_k1_DIS_s003
                                 + 0.00946371104977897 w_k1_DIS_s004
                                 + 0.021614539190163 w_k1_DIS_s005
                                 + 0.018014034876892 w_k1_DIS_s006
                                 + 0.0218683201019001 w_k1_DIS_s007
                                 + 0.0174846276888392 w_k1_DIS_s010
                                 + 0.00685723493426341 w_k1_DIS_s011
                                 + 0.00534993093146035 w_k1_DIS_s012
                                 + 0.011221491114114 w_k1_DIS_s013
                                 + 0.0116705281589952 w_k1_DIS_s014
                                 + 0.00978991337013515 w_k1_DIS_s015
                                 + 0.0131230747486543 w_k1_DIS_s016
                                 + 0.0044476353229625 w_k1_DIS_s017
                                 + 0.103705697661513 w_k1_DIS_s100
                                 + 0.0790125310829022 w_k1_DIS_s101
                                 + 0.0562104568320273 w_k1_DIS_s102
                                 + 0.069379143982383 w_k1_DIS_s103
                                 + 0.0631971762404957 w_k1_DIS_s104
                                 + 0.0536606867653181 w_k1_DIS_s105
                                 + 0.0345777456232037 w_k1_DIS_s106
                                 + 0.0289327816811449 w_k1_DIS_s107
                                 + 0.0208065461076343 w_k1_DIS_s110
                                 + 0.0568449581097439 w_k1_DIS_s111
                                 + 0.0271529917443293 w_k1_DIS_s112
                                 + 0.0280322074873833 w_k1_DIS_s113
                                 + 0.0314767830791113 w_k1_DIS_s114
                                 + 0.0261916950083856 w_k1_DIS_s115
                                 + 0.0790561935655925 w_k1_DIS_s116
                                 + 0.067283433895112 w_k1_DIS_s117
                                 + 0.0143128277609495 w_k1_FDX_s000
                                 + 0.00900484668919212 w_k1_FDX_s001
                                 + 0.00609501886522898 w_k1_FDX_s002
                                 + 0.0296057106320842 w_k1_FDX_s003
                                 + 0.00992203878192756 w_k1_FDX_s004
                                 + 0.0202562148423861 w_k1_FDX_s005
                                 + 0.0177659230394996 w_k1_FDX_s006
                                 + 0.0212827431222116 w_k1_FDX_s007
                                 + 0.0158743128405799 w_k1_FDX_s010
                                 + 0.00618843551512054 w_k1_FDX_s011
                                 + 0.00508399559340427 w_k1_FDX_s012
                                 + 0.0109828789167612 w_k1_FDX_s013
                                 + 0.0114701985931371 w_k1_FDX_s014
                                 + 0.0097512880844963 w_k1_FDX_s015
                                 + 0.0131572350875171 w_k1_FDX_s016
                                 + 0.0041819828571124 w_k1_FDX_s017
                                 + 0.102264686559124 w_k1_FDX_s100
                                 + 0.0807643917036286 w_k1_FDX_s101
                                 + 0.0545544630919674 w_k1_FDX_s102
                                 + 0.0667361328933236 w_k1_FDX_s103
                                 + 0.0615606800834222 w_k1_FDX_s104
                                 + 0.0514135868335654 w_k1_FDX_s105
                                 + 0.0322311652527663 w_k1_FDX_s106
                                 + 0.0262138631315329 w_k1_FDX_s107
                                 + 0.0197297848408379 w_k1_FDX_s110
                                 + 0.0558659308519464 w_k1_FDX_s111
                                 + 0.0266618778470406 w_k1_FDX_s112
                                 + 0.0259120667442465 w_k1_FDX_s113
                                 + 0.0304197643471807 w_k1_FDX_s114
                                 + 0.0283445161823888 w_k1_FDX_s115
                                 + 0.0750814363107447 w_k1_FDX_s116
                                 + 0.0678355325669965 w_k1_FDX_s117
                                 + 0.0150792154097878 w_k1_IBM_s000
                                 + 0.00906648210125444 w_k1_IBM_s001
                                 + 0.00629200079968555 w_k1_IBM_s002
                                 + 0.0302471851285213 w_k1_IBM_s003
                                 + 0.00982802506135576 w_k1_IBM_s004
                                 + 0.0219044899571195 w_k1_IBM_s005
                                 + 0.017131566154076 w_k1_IBM_s006
                                 + 0.0217335907984779 w_k1_IBM_s007
                                 + 0.0167012364885284 w_k1_IBM_s010
                                 + 0.00645562026170919 w_k1_IBM_s011
                                 + 0.00478163105328582 w_k1_IBM_s012
                                 + 0.011170140303141 w_k1_IBM_s013
                                 + 0.0117616585004968 w_k1_IBM_s014
                                 + 0.00993144141018999 w_k1_IBM_s015
                                 + 0.0127837094071821 w_k1_IBM_s016
                                 + 0.00451479690275119 w_k1_IBM_s017
                                 + 0.10723232000698 w_k1_IBM_s100
                                 + 0.0818673726062986 w_k1_IBM_s101
                                 + 0.0547297222842699 w_k1_IBM_s102
                                 + 0.0684142171916196 w_k1_IBM_s103
                                 + 0.0636385591534092 w_k1_IBM_s104
                                 + 0.0530330768849476 w_k1_IBM_s105
                                 + 0.0326284446859771 w_k1_IBM_s106
                                 + 0.027125267850876 w_k1_IBM_s107
                                 + 0.0195702280585499 w_k1_IBM_s110
                                 + 0.0552377944584281 w_k1_IBM_s111
                                 + 0.0267749251772589 w_k1_IBM_s112
                                 + 0.0261809559869747 w_k1_IBM_s113
                                 + 0.0317392550900575 w_k1_IBM_s114
                                 + 0.0281844743013214 w_k1_IBM_s115
                                 + 0.0759973983319668 w_k1_IBM_s116
                                 + 0.0683174249815612 w_k1_IBM_s117
                                 + 0.0128095510004545 w_k1_INTC_s000
                                 + 0.00929632917981511 w_k1_INTC_s001
                                 + 0.0058656139878684 w_k1_INTC_s002
                                 + 0.0270211665435616 w_k1_INTC_s003
                                 + 0.00912456188014379 w_k1_INTC_s004
                                 + 0.0217507438344115 w_k1_INTC_s005
                                 + 0.0156501304781473 w_k1_INTC_s006
                                 + 0.0195740339693421 w_k1_INTC_s007
                                 + 0.0162690453709066 w_k1_INTC_s010
                                 + 0.00566269452109001 w_k1_INTC_s011
                                 + 0.00493654197457967 w_k1_INTC_s012
                                 + 0.0120106130691321 w_k1_INTC_s013
                                 + 0.00918260252327888 w_k1_INTC_s014
                                 + 0.0105663661462273 w_k1_INTC_s015
                                 + 0.012819325397052 w_k1_INTC_s016
                                 + 0.0034315346026106 w_k1_INTC_s017
                                 + 0.0995901330480327 w_k1_INTC_s100
                                 + 0.0781004451358774 w_k1_INTC_s101
                                 + 0.048162886191591 w_k1_INTC_s102
                                 + 0.0678999188666905 w_k1_INTC_s103
                                 + 0.0637218171156743 w_k1_INTC_s104
                                 + 0.0576193850873457 w_k1_INTC_s105
                                 + 0.0334708509589827 w_k1_INTC_s106
                                 + 0.0259965426571659 w_k1_INTC_s107
                                 + 0.0158644594490783 w_k1_INTC_s110
                                 + 0.0619159876305491 w_k1_INTC_s111
                                 + 0.0252620002756589 w_k1_INTC_s112
                                 + 0.0287933245107243 w_k1_INTC_s113
                                 + 0.0279400451064241 w_k1_INTC_s114
                                 + 0.0286900640897088 w_k1_INTC_s115
                                 + 0.0863043622210322 w_k1_INTC_s116
                                 + 0.0653002348162665 w_k1_INTC_s117
                                 + 0.0131149947472958 w_k1_KO_s000
                                 + 0.00889924063878052 w_k1_KO_s001
                                 + 0.00602193856815817 w_k1_KO_s002
                                 + 0.0303140023243444 w_k1_KO_s003
                                 + 0.00984693405291614 w_k1_KO_s004
                                 + 0.021811406377307 w_k1_KO_s005
                                 + 0.0159055456575781 w_k1_KO_s006
                                 + 0.0214833925901422 w_k1_KO_s007
                                 + 0.0177893927547404 w_k1_KO_s010
                                 + 0.00689204253230066 w_k1_KO_s011
                                 + 0.00540319214909611 w_k1_KO_s012
                                 + 0.0118167038747038 w_k1_KO_s013
                                 + 0.0116027951382415 w_k1_KO_s014
                                 + 0.00957701960331855 w_k1_KO_s015
                                 + 0.0127185786338251 w_k1_KO_s016
                                 + 0.00436436427213017 w_k1_KO_s017
                                 + 0.104816048393095 w_k1_KO_s100
                                 + 0.0841667499554481 w_k1_KO_s101
                                 + 0.0505981415085433 w_k1_KO_s102
                                 + 0.0719866538691726 w_k1_KO_s103
                                 + 0.0660044208766153 w_k1_KO_s104
                                 + 0.056691600828755 w_k1_KO_s105
                                 + 0.0348812700751101 w_k1_KO_s106
                                 + 0.0303387840463509 w_k1_KO_s107
                                 + 0.0188106845886879 w_k1_KO_s110
                                 + 0.0523562272961373 w_k1_KO_s111
                                 + 0.0317606652489497 w_k1_KO_s112
                                 + 0.0295094889593712 w_k1_KO_s113
                                 + 0.0329019965706681 w_k1_KO_s114
                                 + 0.0274160272401895 w_k1_KO_s115
                                 + 0.0732589664691649 w_k1_KO_s116
                                 + 0.0631535568956343 w_k1_KO_s117
                                 + 0.020005633075848 w_k1_MAT_s000
                                 + 0.00788551672023575 w_k1_MAT_s001
                                 + 0.005621611127582 w_k1_MAT_s002
                                 + 0.0253919656874183 w_k1_MAT_s003
                                 + 0.00947954195476886 w_k1_MAT_s004
                                 + 0.0177609027539945 w_k1_MAT_s005
                                 + 0.0215336080926298 w_k1_MAT_s006
                                 + 0.0245350079826359 w_k1_MAT_s007
                                 + 0.0163885489952761 w_k1_MAT_s010
                                 + 0.00735406356686506 w_k1_MAT_s011
                                 + 0.00511230357277757 w_k1_MAT_s012
                                 + 0.0110705024173826 w_k1_MAT_s013
                                 + 0.010375391346445 w_k1_MAT_s014
                                 + 0.00964833678627147 w_k1_MAT_s015
                                 + 0.0134956820194027 w_k1_MAT_s016
                                 + 0.00480738595572046 w_k1_MAT_s017
                                 + 0.102047052082061 w_k1_MAT_s100
                                 + 0.0859612213024274 w_k1_MAT_s101
                                 + 0.0573789473957151 w_k1_MAT_s102
                                 + 0.064995102706103 w_k1_MAT_s103
                                 + 0.0680926697175817 w_k1_MAT_s104
                                 + 0.0576817961665556 w_k1_MAT_s105
                                 + 0.0294017834865514 w_k1_MAT_s106
                                 + 0.0278998690188844 w_k1_MAT_s107
                                 + 0.0178560565452461 w_k1_MAT_s110
                                 + 0.0598838704073169 w_k1_MAT_s111
                                 + 0.0272337303617292 w_k1_MAT_s112
                                 + 0.0289261423541839 w_k1_MAT_s113
                                 + 0.0355333181341814 w_k1_MAT_s114
                                 + 0.0272137098789361 w_k1_MAT_s115
                                 + 0.0753961891579192 w_k1_MAT_s116
                                 + 0.08151293306487 w_k1_MAT_s117
                                 + 0.0141118765377336 w_k1_MSFT_s000
                                 + 0.00909180084114081 w_k1_MSFT_s001
                                 + 0.00575029985370198 w_k1_MSFT_s002
                                 + 0.0269280111840457 w_k1_MSFT_s003
                                 + 0.0104645766122101 w_k1_MSFT_s004
                                 + 0.0209486459890983 w_k1_MSFT_s005
                                 + 0.0160723941361709 w_k1_MSFT_s006
                                 + 0.0210597507357216 w_k1_MSFT_s007
                                 + 0.0171224832095258 w_k1_MSFT_s010
                                 + 0.00626404942773236 w_k1_MSFT_s011
                                 + 0.00465605223170437 w_k1_MSFT_s012
                                 + 0.0117973371194861 w_k1_MSFT_s013
                                 + 0.0111379521919944 w_k1_MSFT_s014
                                 + 0.010549479021313 w_k1_MSFT_s015
                                 + 0.0131792616292781 w_k1_MSFT_s016
                                 + 0.00373690542759653 w_k1_MSFT_s017
                                 + 0.104680540577257 w_k1_MSFT_s100
                                 + 0.0759034882582704 w_k1_MSFT_s101
                                 + 0.0536257711496784 w_k1_MSFT_s102
                                 + 0.0704104172467086 w_k1_MSFT_s103
                                 + 0.0671885976912733 w_k1_MSFT_s104
                                 + 0.0549730416281078 w_k1_MSFT_s105
                                 + 0.0366525708161301 w_k1_MSFT_s106
                                 + 0.0265722198599851 w_k1_MSFT_s107
                                 + 0.018678070428743 w_k1_MSFT_s110
                                 + 0.0534914408154616 w_k1_MSFT_s111
                                 + 0.0271457449916927 w_k1_MSFT_s112
                                 + 0.0288747981054307 w_k1_MSFT_s113
                                 + 0.031142818965475 w_k1_MSFT_s114
                                 + 0.026611294296286 w_k1_MSFT_s115
                                 + 0.0767617122900872 w_k1_MSFT_s116
                                 + 0.0674604852302762 w_k1_MSFT_s117
                                 + 0.0136429980750636 w_k1_PG_s000
                                 + 0.00927436097967245 w_k1_PG_s001
                                 + 0.00548530906280577 w_k1_PG_s002
                                 + 0.0285276471300261 w_k1_PG_s003
                                 + 0.0102287006251984 w_k1_PG_s004
                                 + 0.0213920920274386 w_k1_PG_s005
                                 + 0.017748127377316 w_k1_PG_s006
                                 + 0.022580740568989 w_k1_PG_s007
                                 + 0.0166122798785546 w_k1_PG_s010
                                 + 0.00657048995773309 w_k1_PG_s011
                                 + 0.0051357723405485 w_k1_PG_s012
                                 + 0.0120382827367365 w_k1_PG_s013
                                 + 0.0118968555180273 w_k1_PG_s014
                                 + 0.0099947840493026 w_k1_PG_s015
                                 + 0.012829811767768 w_k1_PG_s016
                                 + 0.00430464723160315 w_k1_PG_s017
                                 + 0.106326049259092 w_k1_PG_s100
                                 + 0.0763179639566468 w_k1_PG_s101
                                 + 0.055520759104285 w_k1_PG_s102
                                 + 0.0685542438046986 w_k1_PG_s103
                                 + 0.0628548154783764 w_k1_PG_s104
                                 + 0.0493422726207948 w_k1_PG_s105
                                 + 0.0343894370330419 w_k1_PG_s106
                                 + 0.0280930149376542 w_k1_PG_s107
                                 + 0.019765446481103 w_k1_PG_s110
                                 + 0.055560966985888 w_k1_PG_s111
                                 + 0.0255515808732043 w_k1_PG_s112
                                 + 0.026979747855359 w_k1_PG_s113
                                 + 0.0304964785961239 w_k1_PG_s114
                                 + 0.027074591104064 w_k1_PG_s115
                                 + 0.075563094571527 w_k1_PG_s116
                                 + 0.0657925855962667 w_k1_PG_s117 >= 1.229
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

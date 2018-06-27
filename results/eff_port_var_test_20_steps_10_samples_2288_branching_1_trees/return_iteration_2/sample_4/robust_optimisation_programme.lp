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
 w_k1_ADM_s0_return_constraint:  1.17344 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.878845 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.960351 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.15096 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.01353 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.991285 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.981742 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.880494 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.30198 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.756394 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.674993 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.99255 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.866781 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.945127 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.777897 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.690764 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.10495 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.716089 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.755259 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.15661 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.18685 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.23607 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.839886 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.912843 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.23553 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.954221 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.31784 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.0434 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.64299 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.3268 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.898165 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.19388 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.54061 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.01565 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.922251 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.06273 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.46987 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.16914 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.24273 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.883031 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.836501 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.26434 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.11847 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.01464 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.12745 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.10455 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.35401 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.788729 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.691638 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.22016 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.830267 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.39034 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.693922 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.1732 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.0722 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.834371 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.796032 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.3365 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.44729 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.974658 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.84159 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.713816 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.06346 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.906631 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.28517 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.12087 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.63646 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.25499 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.647245 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.878433 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.63623 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.958977 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.775021 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.23416 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.27359 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.17316 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.28485 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.886982 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.850682 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.37564 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.04885 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.965131 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.956377 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.0095 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.45972 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.777264 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.589228 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.15691 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.812488 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.3982 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.709333 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.30577 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.01497 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.774602 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.736473 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.22852 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.60677 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.19775 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.834005 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.681506 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.23636 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.868792 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.32886 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.00167 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.98471 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.44019 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.594597 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.811307 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.85652 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.10406 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.922654 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.23153 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.25078 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.11326 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.17339 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.911993 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.856651 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.18497 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.02007 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.0459 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.03353 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.936733 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.41918 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.808277 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.745615 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.09435 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.07851 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.15043 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.873964 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.07248 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.03916 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.821892 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.864641 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.18313 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.41189 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.14129 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.759536 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.745432 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.0815 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.943248 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.11725 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.985517 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.55435 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.24367 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.71279 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.982168 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.56129 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.02539 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.863119 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.04116 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.24811 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.11897 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.22217 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.909067 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.936134 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.13371 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.28669 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.78234 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.850934 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.60854 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.919459 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.07984 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.755514 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.20089 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.542655 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.906 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.599531 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.95998 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.12195 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.16574 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.832539 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.23689 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.21032 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.827202 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.31265 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.944034 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.27089 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.745568 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.82245 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.10984 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.29463 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.02809 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.855879 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.75787 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.30043 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.892236 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.16093 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.62398 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.946276 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.15595 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.15702 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.935422 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.870687 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.0606 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.08564 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.973301 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.02721 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.944247 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.12425 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.873896 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.731332 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.16758 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.927662 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.40284 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.886046 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.18545 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.20023 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.977844 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.79761 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.02889 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.26888 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.06089 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.80088 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.702065 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.979665 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.858062 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.18674 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.980856 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.34847 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.22441 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.678488 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.959636 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.37983 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.03868 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.93261 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.17535 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.20832 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.0138 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.20377 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.887472 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.92707 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.17986 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.11212 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.9333 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.989106 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.23977 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.09498 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.900865 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.637048 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.12537 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.689851 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.44077 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.744197 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.48953 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.03618 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.944416 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.763322 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.22648 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.2509 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.03304 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.03166 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.820495 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.30579 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.827049 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.502 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.03048 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.51448 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.19436 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.738952 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.714857 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.40053 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.0061 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.99001 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.35277 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.1277 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.01964 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.22484 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.946755 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.01369 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.31486 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.20346 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.858791 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.842024 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.34622 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.08461 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.991857 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.725676 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.26644 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.510039 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.78249 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.603057 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.95252 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.970918 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.15226 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.78047 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.74635 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.13344 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.629922 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.2647 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.849801 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.14635 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.739368 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.81651 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.31861 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.76483 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.24724 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.636568 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.503689 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.15541 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.976195 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.871595 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.39913 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.00742 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.21713 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.09041 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.932874 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.950518 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.31331 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.04942 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.983196 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.935439 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.931092 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.29167 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.783115 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.738093 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.16463 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.936354 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.27071 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.911704 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.01378 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.00696 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.799571 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.714202 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.969554 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.11861 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.09352 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.792832 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.787678 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.01617 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.955745 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.16284 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.998587 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.44652 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.2291 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.774729 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.869005 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.36567 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.19482 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.916127 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.06488 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.22955 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.01496 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.09961 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.939539 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.966464 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.12937 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.05197 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.999468 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.981896 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.977706 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.14713 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.891987 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.800114 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.13276 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.876412 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.23019 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.942057 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.19411 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.02536 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.87962 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.869331 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.17836 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.166 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.06932 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.931146 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.91165 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.07558 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.916678 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.17098 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.998759 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.33781 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.1526 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.806707 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.834114 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.31056 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.02219 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.91816 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.05623 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.04165 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.06776 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.550999029587144 z_k1_s0000
                                 - 1.15699799319114 z_k1_s0001
                                 - 0.375999780951405 z_k1_s0002
                                 - 0.461001457921862 z_k1_s0003
                                 - 0.544001203656468 z_k1_s0004
                                 - 0.318999617297716 z_k1_s0005
                                 - 0.26199987904378 z_k1_s0006
                                 - 0.584001542395769 z_k1_s0007
                                 - 0.109000023151651 z_k1_s0010
                                 - 0.190999859557558 z_k1_s0011
                                 - 0.263000136479636 z_k1_s0012
                                 - 0.126999926582242 z_k1_s0013
                                 - 0.177999487168988 z_k1_s0014
                                 - 0.20599977908305 z_k1_s0015
                                 - 0.113999885793529 z_k1_s0016
                                 - 0.469999689131148 z_k1_s0017
                                 - 0.396001041116984 z_k1_s0020
                                 - 0.213999914049459 z_k1_s0021
                                 - 0.348998888758333 z_k1_s0022
                                 - 0.551000601963776 z_k1_s0023
                                 - 0.326000729283249 z_k1_s0024
                                 - 0.545999901904622 z_k1_s0025
                                 - 0.419998708004271 z_k1_s0026
                                 - 0.360999303701046 z_k1_s0027
                                 - 0.391998165562622 z_k1_s0030
                                 - 0.169999792357462 z_k1_s0031
                                 - 0.206999777974631 z_k1_s0032
                                 - 0.26699970468903 z_k1_s0033
                                 - 0.177999905825877 z_k1_s0034
                                 - 0.823999979463031 z_k1_s0035
                                 - 0.491998043419952 z_k1_s0036
                                 - 0.550999111324191 z_k1_s0037
                                 - 0.381999594351121 z_k1_s0040
                                 - 0.865997449646835 z_k1_s0041
                                 - 0.192999559500298 z_k1_s0042
                                 - 0.232999708787012 z_k1_s0043
                                 - 0.395000645057662 z_k1_s0044
                                 - 0.327999666793608 z_k1_s0045
                                 - 0.180999831953764 z_k1_s0046
                                 - 0.306000550824333 z_k1_s0047
                                 - 0.205999395347961 z_k1_s0050
                                 - 0.207999344840805 z_k1_s0051
                                 - 0.570997074169532 z_k1_s0052
                                 - 0.786998187375301 z_k1_s0053
                                 - 0.319998815232583 z_k1_s0054
                                 - 0.416997679231237 z_k1_s0055
                                 - 0.187999521513435 z_k1_s0056
                                 - 0.586996670112283 z_k1_s0057
                                 - 0.137999944314105 z_k1_s0060
                                 - 0.277999555392255 z_k1_s0061
                                 - 0.121999947802325 z_k1_s0062
                                 - 0.305999805287813 z_k1_s0063
                                 - 0.209000402435024 z_k1_s0064
                                 - 0.589999999371846 z_k1_s0065
                                 - 0.213999569345136 z_k1_s0066
                                 - 0.190999919959378 z_k1_s0067
                                 - 0.29100090638956 z_k1_s0070
                                 - 0.144000078359675 z_k1_s0071
                                 - 0.110000064699753 z_k1_s0072
                                 - 0.347000026629169 z_k1_s0073
                                 - 0.157000073329642 z_k1_s0074
                                 - 0.197999628419414 z_k1_s0075
                                 - 0.0869999529297761 z_k1_s0076
                                 - 0.409000297609098 z_k1_s0077
                                 - 0.627999816204174 z_k1_s0100
                                 - 0.400000853021236 z_k1_s0101
                                 - 0.532999580211649 z_k1_s0102
                                 - 0.177999992154662 z_k1_s0103
                                 - 0.150999981335703 z_k1_s0104
                                 - 0.29199867214657 z_k1_s0105
                                 - 0.336001144057604 z_k1_s0106
                                 - 0.153999834093447 z_k1_s0107
                                 - 0.138999793049554 z_k1_s0110
                                 - 0.119999721016343 z_k1_s0111
                                 - 0.219999880912585 z_k1_s0112
                                 - 0.514998568774732 z_k1_s0113
                                 - 0.301000330974958 z_k1_s0114
                                 - 0.142999741493508 z_k1_s0115
                                 - 0.202000384564295 z_k1_s0116
                                 - 0.170999742800535 z_k1_s0117
                                 - 0.376998813749764 z_k1_s0120
                                 - 0.397997727623298 z_k1_s0121
                                 - 0.177999342740605 z_k1_s0122
                                 - 0.227998616869962 z_k1_s0123
                                 - 0.163999392161017 z_k1_s0124
                                 - 0.172999273648251 z_k1_s0125
                                 - 0.209999258693833 z_k1_s0126
                                 - 0.295998665969636 z_k1_s0127
                                 - 0.317999519344844 z_k1_s0130
                                 - 0.152999697828357 z_k1_s0131
                                 - 0.213998586580694 z_k1_s0132
                                 - 0.257998900904188 z_k1_s0133
                                 - 0.458999719883508 z_k1_s0134
                                 - 0.167999643639041 z_k1_s0135
                                 - 0.219999483622678 z_k1_s0136
                                 - 0.297998617199694 z_k1_s0137
                                 - 0.127000160590093 z_k1_s0140
                                 - 0.225000326982525 z_k1_s0141
                                 - 0.194000953186141 z_k1_s0142
                                 - 0.338001234375087 z_k1_s0143
                                 - 0.226000016781887 z_k1_s0144
                                 - 0.123000142790382 z_k1_s0145
                                 - 0.207000516584311 z_k1_s0146
                                 - 0.358000424372023 z_k1_s0147
                                 - 0.101999922237078 z_k1_s0150
                                 - 0.11799953532346 z_k1_s0151
                                 - 0.0510000127685151 z_k1_s0152
                                 - 0.192999432571333 z_k1_s0153
                                 - 0.0499999659318991 z_k1_s0154
                                 - 0.150999738032409 z_k1_s0155
                                 - 0.162000149935233 z_k1_s0156
                                 - 0.205999731547484 z_k1_s0157
                                 - 0.223000451827241 z_k1_s0160
                                 - 0.329999419924325 z_k1_s0161
                                 - 0.206000630073492 z_k1_s0162
                                 - 0.114999946215981 z_k1_s0163
                                 - 0.287000320782271 z_k1_s0164
                                 - 0.14499986566366 z_k1_s0165
                                 - 0.0709999215594536 z_k1_s0166
                                 - 0.152999925783002 z_k1_s0167
                                 - 0.246999845334202 z_k1_s0170
                                 - 0.375999841370201 z_k1_s0171
                                 - 0.167999900288844 z_k1_s0172
                                 - 0.71699829184839 z_k1_s0173
                                 - 0.450999917856164 z_k1_s0174
                                 - 0.510997946247089 z_k1_s0175
                                 - 0.656000804517077 z_k1_s0176
                                 - 0.261999860631394 z_k1_s0177
                                 - 0.22500018588212 z_k1_s1000
                                 - 0.171999765563638 z_k1_s1001
                                 - 0.116999955458859 z_k1_s1002
                                 - 0.441999635622573 z_k1_s1003
                                 - 0.43100016780107 z_k1_s1004
                                 - 0.123999937708551 z_k1_s1005
                                 - 0.132999839243953 z_k1_s1006
                                 - 0.120999794030279 z_k1_s1007
                                 - 0.511999923055856 z_k1_s1010
                                 - 0.997001543058383 z_k1_s1011
                                 - 1.43099729158655 z_k1_s1012
                                 - 0.999999200687515 z_k1_s1013
                                 - 0.334999663000349 z_k1_s1014
                                 - 1.39400079650552 z_k1_s1015
                                 - 0.44099974442515 z_k1_s1016
                                 - 0.813002133203677 z_k1_s1017
                                 - 0.253999374684689 z_k1_s1020
                                 - 0.177000105981265 z_k1_s1021
                                 - 0.349999490471159 z_k1_s1022
                                 - 0.267000859531321 z_k1_s1023
                                 - 0.318000598542763 z_k1_s1024
                                 - 0.0750001119581879 z_k1_s1025
                                 - 0.205999316791454 z_k1_s1026
                                 - 0.0730000952126311 z_k1_s1027
                                 - 1.6449925173248 z_k1_s1030
                                 - 0.500998814108284 z_k1_s1031
                                 - 0.455999104628079 z_k1_s1032
                                 - 2.0339978310803 z_k1_s1033
                                 - 0.453999265184178 z_k1_s1034
                                 - 1.60299422800666 z_k1_s1035
                                 - 0.633998933603107 z_k1_s1036
                                 - 0.977994571901108 z_k1_s1037
                                 - 0.175999888952837 z_k1_s1040
                                 - 0.394999673831681 z_k1_s1041
                                 - 0.269000440690179 z_k1_s1042
                                 - 0.26600088885331 z_k1_s1043
                                 - 0.245999362609266 z_k1_s1044
                                 - 0.373000703362183 z_k1_s1045
                                 - 0.400998777505419 z_k1_s1046
                                 - 0.130999840402426 z_k1_s1047
                                 - 0.435999411637946 z_k1_s1050
                                 - 0.236999397624329 z_k1_s1051
                                 - 0.171000008047883 z_k1_s1052
                                 - 0.133999886621746 z_k1_s1053
                                 - 0.283999890195328 z_k1_s1054
                                 - 0.199000014013617 z_k1_s1055
                                 - 0.183999954046266 z_k1_s1056
                                 - 0.342000413495719 z_k1_s1057
                                 - 0.163999996652721 z_k1_s1060
                                 - 0.203999961124134 z_k1_s1061
                                 - 0.333998778256769 z_k1_s1062
                                 - 0.124999829673196 z_k1_s1063
                                 - 0.314000456420219 z_k1_s1064
                                 - 0.534999584105119 z_k1_s1065
                                 - 0.171000002295212 z_k1_s1066
                                 - 0.524999237187447 z_k1_s1067
                                 - 0.830002191276083 z_k1_s1070
                                 - 0.997001531147042 z_k1_s1071
                                 - 0.838000642619526 z_k1_s1072
                                 - 0.62900038517361 z_k1_s1073
                                 - 0.211000322682871 z_k1_s1074
                                 - 0.27500060066105 z_k1_s1075
                                 - 0.24600078572794 z_k1_s1076
                                 - 0.498003124370026 z_k1_s1077
                                 - 1.13199880118956 z_k1_s1100
                                 - 1.95299997246867 z_k1_s1101
                                 - 0.929001088180591 z_k1_s1102
                                 - 1.23399773926305 z_k1_s1103
                                 - 1.42400218247925 z_k1_s1104
                                 - 0.822999727354863 z_k1_s1105
                                 - 0.630000105325394 z_k1_s1106
                                 - 0.725999834771129 z_k1_s1107
                                 - 0.405001531339653 z_k1_s1110
                                 - 0.224000738196168 z_k1_s1111
                                 - 0.264000671905594 z_k1_s1112
                                 - 0.310000954602528 z_k1_s1113
                                 - 0.223000625648084 z_k1_s1114
                                 - 0.747000866664949 z_k1_s1115
                                 - 0.519001309781418 z_k1_s1116
                                 - 0.571000636261882 z_k1_s1117
                                 - 0.155000678637404 z_k1_s1120
                                 - 0.467001098001198 z_k1_s1121
                                 - 0.370000823272681 z_k1_s1122
                                 - 0.678003118458501 z_k1_s1123
                                 - 0.529003258104935 z_k1_s1124
                                 - 0.688002060920313 z_k1_s1125
                                 - 0.511002982463548 z_k1_s1126
                                 - 0.234001040926221 z_k1_s1127
                                 - 0.286999208178478 z_k1_s1130
                                 - 0.461000224440787 z_k1_s1131
                                 - 0.674999349911696 z_k1_s1132
                                 - 0.25000014176009 z_k1_s1133
                                 - 0.16099999996865 z_k1_s1134
                                 - 0.339000011751387 z_k1_s1135
                                 - 0.218000023592346 z_k1_s1136
                                 - 0.327000035388519 z_k1_s1137
                                 - 0.267001180227831 z_k1_s1140
                                 - 0.280001298117472 z_k1_s1141
                                 - 0.570003050151684 z_k1_s1142
                                 - 0.907005833789627 z_k1_s1143
                                 - 0.416002198461456 z_k1_s1144
                                 - 0.293001416007113 z_k1_s1145
                                 - 0.884005655485013 z_k1_s1146
                                 - 0.318001370293662 z_k1_s1147
                                 - 0.327999067948852 z_k1_s1150
                                 - 0.572999513204137 z_k1_s1151
                                 - 0.23100026015487 z_k1_s1152
                                 - 0.492001395925254 z_k1_s1153
                                 - 0.195000103496886 z_k1_s1154
                                 - 0.371001552358624 z_k1_s1155
                                 - 0.175000078553606 z_k1_s1156
                                 - 0.428999445372596 z_k1_s1157
                                 - 0.651000521058442 z_k1_s1160
                                 - 0.44200088273216 z_k1_s1161
                                 - 0.476002862177561 z_k1_s1162
                                 - 0.339000942073475 z_k1_s1163
                                 - 0.407001350955984 z_k1_s1164
                                 - 0.291000569920921 z_k1_s1165
                                 - 0.448000396749985 z_k1_s1166
                                 - 0.49600005890088 z_k1_s1167
                                 - 0.179999976636647 z_k1_s1170
                                 - 0.438999623977527 z_k1_s1171
                                 - 0.540999757988378 z_k1_s1172
                                 - 0.167000150115766 z_k1_s1173
                                 - 0.333000177652925 z_k1_s1174
                                 - 0.165000199458721 z_k1_s1175
                                 - 0.445001537449856 z_k1_s1176
                                 - 0.675000280512639 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.06577 w_k1_ADM_s000
                                 + 1.13795 w_k1_AXP_s000
                                 + 1.49081 w_k1_DIS_s000
                                 + 1.13761 w_k1_FDX_s000
                                 + 1.47548 w_k1_IBM_s000
                                 + 1.19528 w_k1_INTC_s000
                                 + 1.33394 w_k1_KO_s000 + 1.23907 w_k1_MAT_s000
                                 + 1.14058 w_k1_MSFT_s000
                                 + 1.11299 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.991819 w_k1_ADM_s000
                                 + 0.737797 w_k1_AXP_s000
                                 + 0.937205 w_k1_DIS_s000
                                 + 0.928229 w_k1_FDX_s000
                                 + 0.866597 w_k1_IBM_s000
                                 + 0.912525 w_k1_INTC_s000
                                 + 0.812196 w_k1_KO_s000
                                 + 0.877719 w_k1_MAT_s000
                                 + 1.02141 w_k1_MSFT_s000
                                 + 0.875329 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.917744 w_k1_ADM_s000
                                 + 1.13966 w_k1_AXP_s000
                                 + 0.87989 w_k1_DIS_s000
                                 + 0.951096 w_k1_FDX_s000
                                 + 0.91935 w_k1_IBM_s000
                                 + 1.15032 w_k1_INTC_s000
                                 + 0.766996 w_k1_KO_s000 + 1.2733 w_k1_MAT_s000
                                 + 0.954901 w_k1_MSFT_s000
                                 + 0.905233 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.950083 w_k1_ADM_s000
                                 + 1.01485 w_k1_AXP_s000
                                 + 0.943482 w_k1_DIS_s000
                                 + 0.897739 w_k1_FDX_s000
                                 + 1.26036 w_k1_IBM_s000
                                 + 0.876344 w_k1_INTC_s000
                                 + 0.865438 w_k1_KO_s000
                                 + 1.26941 w_k1_MAT_s000
                                 + 0.833457 w_k1_MSFT_s000
                                 + 0.917547 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.933577 w_k1_ADM_s000
                                 + 0.623982 w_k1_AXP_s000
                                 + 0.698721 w_k1_DIS_s000
                                 + 0.771086 w_k1_FDX_s000
                                 + 0.963998 w_k1_IBM_s000
                                 + 0.873353 w_k1_INTC_s000
                                 + 0.761399 w_k1_KO_s000
                                 + 0.736835 w_k1_MAT_s000
                                 + 0.778004 w_k1_MSFT_s000
                                 + 0.856005 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.3432 w_k1_ADM_s000
                                 + 1.25423 w_k1_AXP_s000
                                 + 1.53695 w_k1_DIS_s000
                                 + 1.38214 w_k1_FDX_s000
                                 + 1.14449 w_k1_IBM_s000
                                 + 1.34389 w_k1_INTC_s000
                                 + 1.24937 w_k1_KO_s000
                                 + 0.980311 w_k1_MAT_s000
                                 + 1.25687 w_k1_MSFT_s000
                                 + 1.10681 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.844499 w_k1_ADM_s000
                                 + 1.14347 w_k1_AXP_s000
                                 + 0.963487 w_k1_DIS_s000
                                 + 0.954232 w_k1_FDX_s000
                                 + 1.64864 w_k1_IBM_s000
                                 + 1.0884 w_k1_INTC_s000 + 1.04653 w_k1_KO_s000
                                 + 1.30424 w_k1_MAT_s000 + 1.059 w_k1_MSFT_s000
                                 + 0.924417 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.13932 w_k1_ADM_s000
                                 + 1.06716 w_k1_AXP_s000
                                 + 1.12756 w_k1_DIS_s000 + 1.2292 w_k1_FDX_s000
                                 + 0.862518 w_k1_IBM_s000
                                 + 1.12346 w_k1_INTC_s000
                                 + 0.878878 w_k1_KO_s000
                                 + 1.06798 w_k1_MAT_s000
                                 + 1.17403 w_k1_MSFT_s000
                                 + 0.979547 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.28579 w_k1_ADM_s001
                                 + 1.90534 w_k1_AXP_s001
                                 + 1.22033 w_k1_DIS_s001
                                 + 1.37217 w_k1_FDX_s001
                                 + 1.21008 w_k1_IBM_s001
                                 + 1.16209 w_k1_INTC_s001
                                 + 1.09211 w_k1_KO_s001 + 1.92149 w_k1_MAT_s001
                                 + 1.24385 w_k1_MSFT_s001
                                 + 1.15689 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.690088 w_k1_ADM_s001
                                 + 0.867227 w_k1_AXP_s001
                                 + 0.793438 w_k1_DIS_s001
                                 + 0.748458 w_k1_FDX_s001
                                 + 1.2638 w_k1_IBM_s001
                                 + 0.82679 w_k1_INTC_s001
                                 + 0.886679 w_k1_KO_s001
                                 + 1.40134 w_k1_MAT_s001
                                 + 0.988513 w_k1_MSFT_s001
                                 + 0.99483 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.833223 w_k1_ADM_s001
                                 + 0.933421 w_k1_AXP_s001
                                 + 0.861044 w_k1_DIS_s001
                                 + 0.899686 w_k1_FDX_s001
                                 + 1.07862 w_k1_IBM_s001
                                 + 1.11493 w_k1_INTC_s001
                                 + 0.942722 w_k1_KO_s001
                                 + 1.26557 w_k1_MAT_s001
                                 + 0.979524 w_k1_MSFT_s001
                                 + 0.963846 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.04676 w_k1_ADM_s001
                                 + 1.22822 w_k1_AXP_s001
                                 + 1.33526 w_k1_DIS_s001
                                 + 1.15446 w_k1_FDX_s001
                                 + 1.61232 w_k1_IBM_s001
                                 + 1.00235 w_k1_INTC_s001
                                 + 1.30409 w_k1_KO_s001 + 1.73309 w_k1_MAT_s001
                                 + 1.15265 w_k1_MSFT_s001
                                 + 1.20048 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.25395 w_k1_ADM_s001
                                 + 1.39968 w_k1_AXP_s001
                                 + 1.32567 w_k1_DIS_s001
                                 + 1.22626 w_k1_FDX_s001
                                 + 0.960418 w_k1_IBM_s001
                                 + 1.38859 w_k1_INTC_s001
                                 + 1.33646 w_k1_KO_s001 + 1.22136 w_k1_MAT_s001
                                 + 1.2415 w_k1_MSFT_s001 + 1.29932 w_k1_PG_s001
                                 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.955127 w_k1_ADM_s001
                                 + 1.19539 w_k1_AXP_s001
                                 + 0.796451 w_k1_DIS_s001
                                 + 0.963235 w_k1_FDX_s001
                                 + 0.888898 w_k1_IBM_s001
                                 + 0.926173 w_k1_INTC_s001
                                 + 0.805764 w_k1_KO_s001
                                 + 1.24854 w_k1_MAT_s001
                                 + 0.911453 w_k1_MSFT_s001
                                 + 1.06857 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.05902 w_k1_ADM_s001
                                 + 0.956304 w_k1_AXP_s001
                                 + 1.29288 w_k1_DIS_s001
                                 + 1.12787 w_k1_FDX_s001
                                 + 1.18779 w_k1_IBM_s001
                                 + 0.995427 w_k1_INTC_s001
                                 + 1.18221 w_k1_KO_s001 + 1.06376 w_k1_MAT_s001
                                 + 1.16722 w_k1_MSFT_s001
                                 + 1.12902 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.929649 w_k1_ADM_s001
                                 + 0.686016 w_k1_AXP_s001
                                 + 0.774741 w_k1_DIS_s001
                                 + 0.86184 w_k1_FDX_s001
                                 + 0.782613 w_k1_IBM_s001
                                 + 0.888995 w_k1_INTC_s001
                                 + 0.84428 w_k1_KO_s001
                                 + 0.961613 w_k1_MAT_s001
                                 + 0.968397 w_k1_MSFT_s001
                                 + 0.944485 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.59781 w_k1_ADM_s002
                                 + 0.699154 w_k1_AXP_s002
                                 + 0.899664 w_k1_DIS_s002
                                 + 0.903956 w_k1_FDX_s002
                                 + 0.720089 w_k1_IBM_s002
                                 + 0.966264 w_k1_INTC_s002
                                 + 0.906466 w_k1_KO_s002
                                 + 0.598766 w_k1_MAT_s002
                                 + 0.903068 w_k1_MSFT_s002
                                 + 0.851013 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.866699 w_k1_ADM_s002
                                 + 0.864572 w_k1_AXP_s002
                                 + 0.805616 w_k1_DIS_s002
                                 + 1.00488 w_k1_FDX_s002
                                 + 0.540581 w_k1_IBM_s002
                                 + 1.08588 w_k1_INTC_s002
                                 + 0.735941 w_k1_KO_s002
                                 + 0.428351 w_k1_MAT_s002
                                 + 0.9246 w_k1_MSFT_s002
                                 + 0.829675 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.19 w_k1_ADM_s002
                                 + 1.20677 w_k1_AXP_s002 + 1.1814 w_k1_DIS_s002
                                 + 1.25373 w_k1_FDX_s002
                                 + 0.773646 w_k1_IBM_s002
                                 + 1.20746 w_k1_INTC_s002
                                 + 0.924603 w_k1_KO_s002
                                 + 0.634923 w_k1_MAT_s002
                                 + 1.16761 w_k1_MSFT_s002
                                 + 0.964653 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.908635 w_k1_ADM_s002
                                 + 1.01587 w_k1_AXP_s002
                                 + 0.97352 w_k1_DIS_s002
                                 + 0.942741 w_k1_FDX_s002
                                 + 1.10625 w_k1_IBM_s002
                                 + 1.24023 w_k1_INTC_s002
                                 + 1.08765 w_k1_KO_s002 + 1.07121 w_k1_MAT_s002
                                 + 1.0609 w_k1_MSFT_s002 + 1.00397 w_k1_PG_s002
                                 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.733136 w_k1_ADM_s002
                                 + 0.740949 w_k1_AXP_s002
                                 + 0.573022 w_k1_DIS_s002
                                 + 0.786937 w_k1_FDX_s002
                                 + 0.743404 w_k1_IBM_s002
                                 + 0.917303 w_k1_INTC_s002
                                 + 0.744596 w_k1_KO_s002
                                 + 0.528195 w_k1_MAT_s002
                                 + 0.673406 w_k1_MSFT_s002
                                 + 0.782416 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.82535 w_k1_ADM_s002
                                 + 1.302 w_k1_AXP_s002 + 1.2645 w_k1_DIS_s002
                                 + 1.02292 w_k1_FDX_s002
                                 + 1.44707 w_k1_IBM_s002
                                 + 1.13234 w_k1_INTC_s002 + 1.268 w_k1_KO_s002
                                 + 1.36612 w_k1_MAT_s002
                                 + 0.940483 w_k1_MSFT_s002
                                 + 1.0679 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.775474 w_k1_ADM_s002
                                 + 0.816163 w_k1_AXP_s002
                                 + 0.787565 w_k1_DIS_s002
                                 + 0.828661 w_k1_FDX_s002
                                 + 0.969306 w_k1_IBM_s002
                                 + 0.922984 w_k1_INTC_s002
                                 + 0.92277 w_k1_KO_s002
                                 + 0.958924 w_k1_MAT_s002
                                 + 0.822144 w_k1_MSFT_s002
                                 + 0.990362 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.22075 w_k1_ADM_s002
                                 + 1.65983 w_k1_AXP_s002
                                 + 1.72932 w_k1_DIS_s002
                                 + 1.41071 w_k1_FDX_s002
                                 + 1.29202 w_k1_IBM_s002
                                 + 1.35723 w_k1_INTC_s002
                                 + 1.39799 w_k1_KO_s002 + 1.39643 w_k1_MAT_s002
                                 + 1.28548 w_k1_MSFT_s002 + 1.1548 w_k1_PG_s002
                                 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.996504 w_k1_ADM_s003
                                 + 0.916953 w_k1_AXP_s003
                                 + 0.877202 w_k1_DIS_s003
                                 + 1.01442 w_k1_FDX_s003
                                 + 0.692883 w_k1_IBM_s003
                                 + 0.979278 w_k1_INTC_s003
                                 + 0.817365 w_k1_KO_s003
                                 + 0.651401 w_k1_MAT_s003
                                 + 1.02649 w_k1_MSFT_s003
                                 + 0.858079 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.863345 w_k1_ADM_s003
                                 + 0.699093 w_k1_AXP_s003
                                 + 0.825263 w_k1_DIS_s003
                                 + 0.850536 w_k1_FDX_s003
                                 + 0.89639 w_k1_IBM_s003
                                 + 0.786745 w_k1_INTC_s003
                                 + 0.909446 w_k1_KO_s003
                                 + 0.591899 w_k1_MAT_s003
                                 + 0.988163 w_k1_MSFT_s003
                                 + 0.840292 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.967751 w_k1_ADM_s003
                                 + 0.89874 w_k1_AXP_s003
                                 + 0.809828 w_k1_DIS_s003
                                 + 0.764979 w_k1_FDX_s003
                                 + 1.13359 w_k1_IBM_s003
                                 + 0.774038 w_k1_INTC_s003
                                 + 1.01683 w_k1_KO_s003 + 1.06298 w_k1_MAT_s003
                                 + 0.879541 w_k1_MSFT_s003
                                 + 0.848911 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.5024 w_k1_ADM_s003
                                 + 1.33917 w_k1_AXP_s003
                                 + 1.42665 w_k1_DIS_s003
                                 + 1.37554 w_k1_FDX_s003
                                 + 0.932437 w_k1_IBM_s003
                                 + 1.29151 w_k1_INTC_s003
                                 + 1.05438 w_k1_KO_s003
                                 + 0.618742 w_k1_MAT_s003
                                 + 1.42384 w_k1_MSFT_s003
                                 + 1.22488 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.679891 w_k1_ADM_s003
                                 + 0.734245 w_k1_AXP_s003
                                 + 0.641989 w_k1_DIS_s003
                                 + 0.759151 w_k1_FDX_s003
                                 + 0.776452 w_k1_IBM_s003
                                 + 0.815133 w_k1_INTC_s003
                                 + 0.731508 w_k1_KO_s003
                                 + 0.640311 w_k1_MAT_s003
                                 + 0.82591 w_k1_MSFT_s003
                                 + 0.834374 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.04214 w_k1_ADM_s003
                                 + 1.40424 w_k1_AXP_s003
                                 + 1.21077 w_k1_DIS_s003
                                 + 1.19114 w_k1_FDX_s003
                                 + 1.22921 w_k1_IBM_s003
                                 + 1.3714 w_k1_INTC_s003 + 1.20592 w_k1_KO_s003
                                 + 1.27583 w_k1_MAT_s003
                                 + 1.21058 w_k1_MSFT_s003
                                 + 1.10159 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.855288 w_k1_ADM_s003
                                 + 1.20477 w_k1_AXP_s003
                                 + 1.00665 w_k1_DIS_s003
                                 + 1.00746 w_k1_FDX_s003
                                 + 1.54639 w_k1_IBM_s003
                                 + 1.1174 w_k1_INTC_s003 + 1.20019 w_k1_KO_s003
                                 + 1.5183 w_k1_MAT_s003
                                 + 1.02206 w_k1_MSFT_s003 + 1.0767 w_k1_PG_s003
                                 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.992171 w_k1_ADM_s003
                                 + 1.25141 w_k1_AXP_s003
                                 + 0.98996 w_k1_DIS_s003
                                 + 1.04253 w_k1_FDX_s003
                                 + 0.918463 w_k1_IBM_s003
                                 + 1.03262 w_k1_INTC_s003
                                 + 1.05486 w_k1_KO_s003 + 1.01682 w_k1_MAT_s003
                                 + 1.09288 w_k1_MSFT_s003
                                 + 1.02774 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.887165 w_k1_ADM_s004
                                 + 0.894588 w_k1_AXP_s004
                                 + 1.03935 w_k1_DIS_s004
                                 + 0.968204 w_k1_FDX_s004
                                 + 1.58345 w_k1_IBM_s004
                                 + 1.07878 w_k1_INTC_s004
                                 + 1.39053 w_k1_KO_s004 + 1.16005 w_k1_MAT_s004
                                 + 0.97571 w_k1_MSFT_s004
                                 + 1.02183 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.05828 w_k1_ADM_s004
                                 + 0.812625 w_k1_AXP_s004
                                 + 0.836512 w_k1_DIS_s004
                                 + 0.907813 w_k1_FDX_s004
                                 + 0.727457 w_k1_IBM_s004
                                 + 1.07117 w_k1_INTC_s004
                                 + 0.859431 w_k1_KO_s004
                                 + 0.572017 w_k1_MAT_s004
                                 + 0.942083 w_k1_MSFT_s004
                                 + 0.893963 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.06275 w_k1_ADM_s004
                                 + 1.22708 w_k1_AXP_s004
                                 + 0.914415 w_k1_DIS_s004
                                 + 1.00383 w_k1_FDX_s004
                                 + 1.00023 w_k1_IBM_s004
                                 + 1.02274 w_k1_INTC_s004
                                 + 1.04287 w_k1_KO_s004 + 1.05914 w_k1_MAT_s004
                                 + 0.986685 w_k1_MSFT_s004
                                 + 1.04342 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.06919 w_k1_ADM_s004
                                 + 1.30912 w_k1_AXP_s004 + 1.449 w_k1_DIS_s004
                                 + 1.13677 w_k1_FDX_s004
                                 + 1.04431 w_k1_IBM_s004
                                 + 1.2532 w_k1_INTC_s004 + 1.32549 w_k1_KO_s004
                                 + 1.02728 w_k1_MAT_s004 + 1.202 w_k1_MSFT_s004
                                 + 1.08198 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.850061 w_k1_ADM_s004
                                 + 0.97431 w_k1_AXP_s004
                                 + 0.873348 w_k1_DIS_s004
                                 + 0.850885 w_k1_FDX_s004
                                 + 1.1155 w_k1_IBM_s004
                                 + 1.12793 w_k1_INTC_s004
                                 + 1.06344 w_k1_KO_s004
                                 + 0.810345 w_k1_MAT_s004
                                 + 0.916143 w_k1_MSFT_s004
                                 + 0.879634 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.03576 w_k1_ADM_s004
                                 + 1.01419 w_k1_AXP_s004
                                 + 1.23002 w_k1_DIS_s004
                                 + 1.10602 w_k1_FDX_s004
                                 + 0.872587 w_k1_IBM_s004
                                 + 1.21579 w_k1_INTC_s004
                                 + 1.11963 w_k1_KO_s004
                                 + 0.836184 w_k1_MAT_s004
                                 + 1.14489 w_k1_MSFT_s004
                                 + 0.923316 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.41524 w_k1_ADM_s004
                                 + 1.48802 w_k1_AXP_s004 + 1.8745 w_k1_DIS_s004
                                 + 1.40854 w_k1_FDX_s004
                                 + 1.42243 w_k1_IBM_s004
                                 + 1.21013 w_k1_INTC_s004
                                 + 1.56118 w_k1_KO_s004 + 1.34616 w_k1_MAT_s004
                                 + 1.23842 w_k1_MSFT_s004
                                 + 1.29384 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.34178 w_k1_ADM_s004
                                 + 1.33781 w_k1_AXP_s004
                                 + 1.46941 w_k1_DIS_s004
                                 + 1.31904 w_k1_FDX_s004
                                 + 0.938079 w_k1_IBM_s004
                                 + 1.36029 w_k1_INTC_s004
                                 + 1.15504 w_k1_KO_s004 + 0.83426 w_k1_MAT_s004
                                 + 1.31888 w_k1_MSFT_s004
                                 + 1.10907 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.792293 w_k1_ADM_s005
                                 + 0.550519 w_k1_AXP_s005
                                 + 0.774238 w_k1_DIS_s005
                                 + 0.837203 w_k1_FDX_s005
                                 + 0.623434 w_k1_IBM_s005
                                 + 0.884209 w_k1_INTC_s005
                                 + 0.699957 w_k1_KO_s005
                                 + 0.394879 w_k1_MAT_s005
                                 + 0.816619 w_k1_MSFT_s005
                                 + 0.833903 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.833917 w_k1_ADM_s005
                                 + 0.762321 w_k1_AXP_s005
                                 + 1.11988 w_k1_DIS_s005
                                 + 1.08546 w_k1_FDX_s005
                                 + 0.76854 w_k1_IBM_s005
                                 + 0.900827 w_k1_INTC_s005
                                 + 0.934326 w_k1_KO_s005
                                 + 0.840563 w_k1_MAT_s005
                                 + 1.10619 w_k1_MSFT_s005
                                 + 0.935574 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.959406 w_k1_ADM_s005
                                 + 1.0861 w_k1_AXP_s005 + 1.20081 w_k1_DIS_s005
                                 + 1.14635 w_k1_FDX_s005
                                 + 1.42174 w_k1_IBM_s005
                                 + 1.10861 w_k1_INTC_s005
                                 + 1.13416 w_k1_KO_s005 + 1.08951 w_k1_MAT_s005
                                 + 1.09483 w_k1_MSFT_s005
                                 + 1.01456 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.743737 w_k1_ADM_s005
                                 + 0.821681 w_k1_AXP_s005
                                 + 0.945402 w_k1_DIS_s005
                                 + 0.864366 w_k1_FDX_s005
                                 + 1.11488 w_k1_IBM_s005
                                 + 1.01315 w_k1_INTC_s005
                                 + 1.02029 w_k1_KO_s005
                                 + 0.974863 w_k1_MAT_s005
                                 + 0.96877 w_k1_MSFT_s005
                                 + 0.942799 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.841569 w_k1_ADM_s005
                                 + 1.19055 w_k1_AXP_s005
                                 + 0.875374 w_k1_DIS_s005
                                 + 1.10751 w_k1_FDX_s005
                                 + 0.749288 w_k1_IBM_s005
                                 + 0.963418 w_k1_INTC_s005
                                 + 0.795894 w_k1_KO_s005
                                 + 0.861256 w_k1_MAT_s005
                                 + 0.921452 w_k1_MSFT_s005
                                 + 0.894298 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.85874 w_k1_ADM_s005
                                 + 0.820355 w_k1_AXP_s005
                                 + 0.751278 w_k1_DIS_s005
                                 + 0.835087 w_k1_FDX_s005
                                 + 0.832097 w_k1_IBM_s005
                                 + 0.830693 w_k1_INTC_s005
                                 + 0.740243 w_k1_KO_s005
                                 + 0.693497 w_k1_MAT_s005
                                 + 0.830142 w_k1_MSFT_s005
                                 + 0.809877 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.940723 w_k1_ADM_s005
                                 + 0.95831 w_k1_AXP_s005
                                 + 1.36782 w_k1_DIS_s005
                                 + 1.12277 w_k1_FDX_s005
                                 + 0.954293 w_k1_IBM_s005
                                 + 1.10378 w_k1_INTC_s005
                                 + 1.00375 w_k1_KO_s005
                                 + 0.749161 w_k1_MAT_s005
                                 + 1.00897 w_k1_MSFT_s005
                                 + 1.01734 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.12875 w_k1_ADM_s005
                                 + 1.42844 w_k1_AXP_s005
                                 + 1.40009 w_k1_DIS_s005
                                 + 1.35296 w_k1_FDX_s005
                                 + 1.08429 w_k1_IBM_s005
                                 + 1.14948 w_k1_INTC_s005
                                 + 1.15183 w_k1_KO_s005 + 1.5274 w_k1_MAT_s005
                                 + 1.27977 w_k1_MSFT_s005 + 1.029 w_k1_PG_s005
                                 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.05976 w_k1_ADM_s006
                                 + 1.65677 w_k1_AXP_s006
                                 + 1.44626 w_k1_DIS_s006 + 1.4026 w_k1_FDX_s006
                                 + 1.18403 w_k1_IBM_s006
                                 + 1.64147 w_k1_INTC_s006
                                 + 1.04767 w_k1_KO_s006 + 1.39626 w_k1_MAT_s006
                                 + 1.30403 w_k1_MSFT_s006
                                 + 1.18488 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.09692 w_k1_ADM_s006
                                 + 0.983139 w_k1_AXP_s006
                                 + 0.994411 w_k1_DIS_s006
                                 + 1.03336 w_k1_FDX_s006
                                 + 0.98399 w_k1_IBM_s006
                                 + 1.15494 w_k1_INTC_s006
                                 + 1.05011 w_k1_KO_s006
                                 + 0.837209 w_k1_MAT_s006
                                 + 1.10294 w_k1_MSFT_s006
                                 + 1.01382 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.36883 w_k1_ADM_s006
                                 + 1.23731 w_k1_AXP_s006
                                 + 1.13592 w_k1_DIS_s006
                                 + 1.12027 w_k1_FDX_s006
                                 + 0.763461 w_k1_IBM_s006
                                 + 1.06185 w_k1_INTC_s006
                                 + 1.03694 w_k1_KO_s006 + 1.07765 w_k1_MAT_s006
                                 + 1.18562 w_k1_MSFT_s006
                                 + 1.08067 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.19945 w_k1_ADM_s006
                                 + 1.49561 w_k1_AXP_s006
                                 + 1.24546 w_k1_DIS_s006 + 1.0684 w_k1_FDX_s006
                                 + 1.52818 w_k1_IBM_s006
                                 + 1.13602 w_k1_INTC_s006
                                 + 1.29738 w_k1_KO_s006 + 2.14121 w_k1_MAT_s006
                                 + 1.17573 w_k1_MSFT_s006
                                 + 1.14276 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.24456 w_k1_ADM_s006
                                 + 1.22618 w_k1_AXP_s006
                                 + 1.53617 w_k1_DIS_s006
                                 + 1.28628 w_k1_FDX_s006
                                 + 1.02619 w_k1_IBM_s006
                                 + 1.36321 w_k1_INTC_s006
                                 + 1.12374 w_k1_KO_s006
                                 + 0.994004 w_k1_MAT_s006
                                 + 1.18859 w_k1_MSFT_s006
                                 + 1.14923 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.711316 w_k1_ADM_s006
                                 + 0.809966 w_k1_AXP_s006
                                 + 0.636951 w_k1_DIS_s006
                                 + 0.729996 w_k1_FDX_s006
                                 + 0.869425 w_k1_IBM_s006
                                 + 0.863171 w_k1_INTC_s006
                                 + 0.800874 w_k1_KO_s006
                                 + 1.06347 w_k1_MAT_s006
                                 + 0.828594 w_k1_MSFT_s006
                                 + 0.84887 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.185 w_k1_ADM_s006
                                 + 1.06602 w_k1_AXP_s006
                                 + 0.949456 w_k1_DIS_s006
                                 + 0.949433 w_k1_FDX_s006
                                 + 1.01216 w_k1_IBM_s006
                                 + 1.06128 w_k1_INTC_s006
                                 + 0.901157 w_k1_KO_s006
                                 + 0.859843 w_k1_MAT_s006
                                 + 0.904106 w_k1_MSFT_s006
                                 + 0.915107 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.870471 w_k1_ADM_s006
                                 + 0.94129 w_k1_AXP_s006
                                 + 0.682103 w_k1_DIS_s006
                                 + 0.919111 w_k1_FDX_s006
                                 + 0.639149 w_k1_IBM_s006
                                 + 0.972069 w_k1_INTC_s006
                                 + 0.720431 w_k1_KO_s006
                                 + 0.688419 w_k1_MAT_s006
                                 + 0.947718 w_k1_MSFT_s006
                                 + 0.863158 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.13491 w_k1_ADM_s007
                                 + 1.00671 w_k1_AXP_s007
                                 + 1.00873 w_k1_DIS_s007
                                 + 1.03305 w_k1_FDX_s007 + 1.2218 w_k1_IBM_s007
                                 + 0.73364 w_k1_INTC_s007
                                 + 1.12312 w_k1_KO_s007 + 1.1661 w_k1_MAT_s007
                                 + 0.793639 w_k1_MSFT_s007
                                 + 0.918456 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.1005 w_k1_ADM_s007
                                 + 0.813654 w_k1_AXP_s007
                                 + 0.558959 w_k1_DIS_s007
                                 + 0.764657 w_k1_FDX_s007
                                 + 1.09418 w_k1_IBM_s007
                                 + 0.88206 w_k1_INTC_s007
                                 + 0.863231 w_k1_KO_s007
                                 + 0.834519 w_k1_MAT_s007
                                 + 0.729067 w_k1_MSFT_s007
                                 + 0.795653 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.76267 w_k1_ADM_s007
                                 + 1.67153 w_k1_AXP_s007
                                 + 1.91735 w_k1_DIS_s007
                                 + 1.51218 w_k1_FDX_s007
                                 + 1.14696 w_k1_IBM_s007
                                 + 0.990634 w_k1_INTC_s007
                                 + 1.30995 w_k1_KO_s007 + 1.73286 w_k1_MAT_s007
                                 + 1.19562 w_k1_MSFT_s007
                                 + 1.14242 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.09238 w_k1_ADM_s007
                                 + 1.05222 w_k1_AXP_s007
                                 + 0.962556 w_k1_DIS_s007
                                 + 1.09031 w_k1_FDX_s007
                                 + 0.805564 w_k1_IBM_s007
                                 + 0.936672 w_k1_INTC_s007
                                 + 0.816553 w_k1_KO_s007
                                 + 0.78702 w_k1_MAT_s007
                                 + 0.907939 w_k1_MSFT_s007
                                 + 0.944949 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.32397 w_k1_ADM_s007
                                 + 1.40266 w_k1_AXP_s007
                                 + 1.42184 w_k1_DIS_s007
                                 + 1.32516 w_k1_FDX_s007
                                 + 1.39368 w_k1_IBM_s007
                                 + 0.9012 w_k1_INTC_s007 + 1.34007 w_k1_KO_s007
                                 + 1.94909 w_k1_MAT_s007
                                 + 1.04673 w_k1_MSFT_s007
                                 + 1.16913 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.36075 w_k1_ADM_s007
                                 + 1.47722 w_k1_AXP_s007
                                 + 1.28412 w_k1_DIS_s007
                                 + 1.31747 w_k1_FDX_s007
                                 + 0.79173 w_k1_IBM_s007
                                 + 1.05187 w_k1_INTC_s007
                                 + 1.01239 w_k1_KO_s007
                                 + 0.785749 w_k1_MAT_s007
                                 + 1.07765 w_k1_MSFT_s007
                                 + 1.03978 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.934365 w_k1_ADM_s007
                                 + 1.09245 w_k1_AXP_s007
                                 + 1.02565 w_k1_DIS_s007
                                 + 0.833326 w_k1_FDX_s007
                                 + 1.94984 w_k1_IBM_s007
                                 + 0.726831 w_k1_INTC_s007
                                 + 1.43106 w_k1_KO_s007 + 1.9935 w_k1_MAT_s007
                                 + 0.779086 w_k1_MSFT_s007
                                 + 1.14987 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.969385 w_k1_ADM_s007
                                 + 0.77389 w_k1_AXP_s007
                                 + 0.792468 w_k1_DIS_s007
                                 + 0.879976 w_k1_FDX_s007
                                 + 0.887484 w_k1_IBM_s007
                                 + 0.733996 w_k1_INTC_s007
                                 + 0.873587 w_k1_KO_s007
                                 + 1.03683 w_k1_MAT_s007
                                 + 0.839979 w_k1_MSFT_s007
                                 + 0.858635 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.00013 w_k1_ADM_s010
                                 + 0.79267 w_k1_AXP_s010
                                 + 0.936363 w_k1_DIS_s010
                                 + 0.937282 w_k1_FDX_s010
                                 + 1.09739 w_k1_IBM_s010
                                 + 0.894634 w_k1_INTC_s010
                                 + 0.979756 w_k1_KO_s010
                                 + 0.914401 w_k1_MAT_s010
                                 + 0.805175 w_k1_MSFT_s010
                                 + 0.826752 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.54321 w_k1_ADM_s010
                                 + 1.43186 w_k1_AXP_s010
                                 + 1.30646 w_k1_DIS_s010
                                 + 1.21071 w_k1_FDX_s010 + 1.1131 w_k1_IBM_s010
                                 + 1.21367 w_k1_INTC_s010
                                 + 1.15042 w_k1_KO_s010 + 1.05126 w_k1_MAT_s010
                                 + 1.06718 w_k1_MSFT_s010
                                 + 1.04011 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.33846 w_k1_ADM_s010
                                 + 1.12917 w_k1_AXP_s010
                                 + 1.20739 w_k1_DIS_s010
                                 + 1.14318 w_k1_FDX_s010
                                 + 1.03839 w_k1_IBM_s010
                                 + 1.08347 w_k1_INTC_s010
                                 + 1.06728 w_k1_KO_s010
                                 + 0.916379 w_k1_MAT_s010
                                 + 1.04164 w_k1_MSFT_s010
                                 + 1.07026 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.797817 w_k1_ADM_s010
                                 + 0.861245 w_k1_AXP_s010
                                 + 0.577331 w_k1_DIS_s010
                                 + 0.797334 w_k1_FDX_s010
                                 + 0.757522 w_k1_IBM_s010
                                 + 0.92273 w_k1_INTC_s010
                                 + 0.622788 w_k1_KO_s010
                                 + 0.585781 w_k1_MAT_s010
                                 + 0.753912 w_k1_MSFT_s010
                                 + 0.722018 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.00169 w_k1_ADM_s010
                                 + 0.998082 w_k1_AXP_s010
                                 + 1.00844 w_k1_DIS_s010
                                 + 0.826684 w_k1_FDX_s010
                                 + 1.60107 w_k1_IBM_s010
                                 + 1.04727 w_k1_INTC_s010
                                 + 1.17052 w_k1_KO_s010 + 1.68688 w_k1_MAT_s010
                                 + 0.916572 w_k1_MSFT_s010
                                 + 0.838677 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.04947 w_k1_ADM_s010
                                 + 1.00387 w_k1_AXP_s010
                                 + 0.864819 w_k1_DIS_s010
                                 + 1.00418 w_k1_FDX_s010
                                 + 0.699225 w_k1_IBM_s010
                                 + 0.954878 w_k1_INTC_s010
                                 + 0.917266 w_k1_KO_s010
                                 + 0.801248 w_k1_MAT_s010
                                 + 0.867664 w_k1_MSFT_s010
                                 + 0.810971 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.86359 w_k1_ADM_s010
                                 + 0.565739 w_k1_AXP_s010
                                 + 0.615545 w_k1_DIS_s010
                                 + 0.668357 w_k1_FDX_s010
                                 + 0.914388 w_k1_IBM_s010
                                 + 0.845433 w_k1_INTC_s010
                                 + 0.856694 w_k1_KO_s010
                                 + 0.560842 w_k1_MAT_s010
                                 + 0.750576 w_k1_MSFT_s010
                                 + 0.772645 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.90012 w_k1_ADM_s010
                                 + 1.47253 w_k1_AXP_s010
                                 + 1.84427 w_k1_DIS_s010
                                 + 1.56569 w_k1_FDX_s010
                                 + 1.12992 w_k1_IBM_s010
                                 + 1.31364 w_k1_INTC_s010
                                 + 1.21782 w_k1_KO_s010 + 1.08294 w_k1_MAT_s010
                                 + 1.33263 w_k1_MSFT_s010 + 1.2156 w_k1_PG_s010
                                 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.82289 w_k1_ADM_s011
                                 + 1.28665 w_k1_AXP_s011
                                 + 1.18614 w_k1_DIS_s011
                                 + 1.04313 w_k1_FDX_s011
                                 + 1.24184 w_k1_IBM_s011
                                 + 0.979569 w_k1_INTC_s011
                                 + 1.30516 w_k1_KO_s011 + 1.49376 w_k1_MAT_s011
                                 + 1.08607 w_k1_MSFT_s011
                                 + 1.19051 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.78929 w_k1_ADM_s011
                                 + 1.06648 w_k1_AXP_s011 + 1.4621 w_k1_DIS_s011
                                 + 1.27052 w_k1_FDX_s011
                                 + 1.01694 w_k1_IBM_s011
                                 + 1.15387 w_k1_INTC_s011
                                 + 1.26968 w_k1_KO_s011
                                 + 0.946261 w_k1_MAT_s011
                                 + 1.33657 w_k1_MSFT_s011
                                 + 1.04435 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 2.11821 w_k1_ADM_s011
                                 + 1.45928 w_k1_AXP_s011
                                 + 1.51785 w_k1_DIS_s011
                                 + 1.41654 w_k1_FDX_s011
                                 + 0.873244 w_k1_IBM_s011
                                 + 1.13926 w_k1_INTC_s011
                                 + 1.05859 w_k1_KO_s011
                                 + 0.928502 w_k1_MAT_s011
                                 + 1.27039 w_k1_MSFT_s011
                                 + 1.17214 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.33657 w_k1_ADM_s011
                                 + 1.18451 w_k1_AXP_s011
                                 + 0.882555 w_k1_DIS_s011
                                 + 0.96853 w_k1_FDX_s011
                                 + 0.857849 w_k1_IBM_s011
                                 + 0.917547 w_k1_INTC_s011
                                 + 0.836827 w_k1_KO_s011
                                 + 0.826159 w_k1_MAT_s011
                                 + 0.866495 w_k1_MSFT_s011
                                 + 0.920411 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.943808 w_k1_ADM_s011
                                 + 0.597406 w_k1_AXP_s011
                                 + 0.674644 w_k1_DIS_s011
                                 + 0.740414 w_k1_FDX_s011
                                 + 0.741951 w_k1_IBM_s011
                                 + 0.838031 w_k1_INTC_s011
                                 + 0.680139 w_k1_KO_s011
                                 + 0.609894 w_k1_MAT_s011
                                 + 0.855239 w_k1_MSFT_s011
                                 + 0.738069 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.495 w_k1_ADM_s011
                                 + 0.870333 w_k1_AXP_s011
                                 + 1.03157 w_k1_DIS_s011
                                 + 0.979508 w_k1_FDX_s011
                                 + 1.16584 w_k1_IBM_s011
                                 + 1.07012 w_k1_INTC_s011
                                 + 1.02224 w_k1_KO_s011
                                 + 0.840628 w_k1_MAT_s011
                                 + 0.963818 w_k1_MSFT_s011
                                 + 0.982483 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.60068 w_k1_ADM_s011
                                 + 1.82031 w_k1_AXP_s011
                                 + 1.40248 w_k1_DIS_s011
                                 + 1.24477 w_k1_FDX_s011
                                 + 1.61007 w_k1_IBM_s011
                                 + 1.30999 w_k1_INTC_s011
                                 + 1.35946 w_k1_KO_s011 + 1.64437 w_k1_MAT_s011
                                 + 1.34469 w_k1_MSFT_s011
                                 + 1.33316 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.958084 w_k1_ADM_s011
                                 + 0.808714 w_k1_AXP_s011
                                 + 0.640388 w_k1_DIS_s011
                                 + 0.737064 w_k1_FDX_s011
                                 + 1.27086 w_k1_IBM_s011
                                 + 0.791299 w_k1_INTC_s011
                                 + 0.975393 w_k1_KO_s011
                                 + 1.07862 w_k1_MAT_s011
                                 + 0.751818 w_k1_MSFT_s011
                                 + 0.85458 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.925031 w_k1_ADM_s012
                                 + 1.09499 w_k1_AXP_s012
                                 + 0.936281 w_k1_DIS_s012
                                 + 0.869603 w_k1_FDX_s012
                                 + 1.22496 w_k1_IBM_s012
                                 + 0.905595 w_k1_INTC_s012
                                 + 0.986262 w_k1_KO_s012
                                 + 1.49662 w_k1_MAT_s012
                                 + 0.953535 w_k1_MSFT_s012
                                 + 0.985947 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.961851 w_k1_ADM_s012
                                 + 1.06084 w_k1_AXP_s012
                                 + 1.16329 w_k1_DIS_s012
                                 + 1.06209 w_k1_FDX_s012
                                 + 1.12425 w_k1_IBM_s012
                                 + 0.882675 w_k1_INTC_s012
                                 + 1.00954 w_k1_KO_s012 + 1.34297 w_k1_MAT_s012
                                 + 1.0609 w_k1_MSFT_s012 + 1.18267 w_k1_PG_s012
                                 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.00298 w_k1_ADM_s012
                                 + 1.42257 w_k1_AXP_s012
                                 + 1.60023 w_k1_DIS_s012
                                 + 1.19909 w_k1_FDX_s012
                                 + 1.62969 w_k1_IBM_s012
                                 + 1.44629 w_k1_INTC_s012
                                 + 1.56446 w_k1_KO_s012 + 1.76633 w_k1_MAT_s012
                                 + 1.23037 w_k1_MSFT_s012
                                 + 1.17563 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.03896 w_k1_ADM_s012
                                 + 1.06288 w_k1_AXP_s012
                                 + 0.742464 w_k1_DIS_s012
                                 + 0.980069 w_k1_FDX_s012
                                 + 0.740535 w_k1_IBM_s012
                                 + 0.809758 w_k1_INTC_s012
                                 + 0.894651 w_k1_KO_s012
                                 + 0.85217 w_k1_MAT_s012
                                 + 0.777932 w_k1_MSFT_s012
                                 + 1.06323 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.4199 w_k1_ADM_s012
                                 + 1.45078 w_k1_AXP_s012
                                 + 1.31154 w_k1_DIS_s012
                                 + 1.39401 w_k1_FDX_s012
                                 + 0.87508 w_k1_IBM_s012
                                 + 0.951223 w_k1_INTC_s012
                                 + 1.03279 w_k1_KO_s012 + 1.00133 w_k1_MAT_s012
                                 + 1.05446 w_k1_MSFT_s012
                                 + 1.31681 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.802831 w_k1_ADM_s012
                                 + 0.716549 w_k1_AXP_s012
                                 + 0.707161 w_k1_DIS_s012
                                 + 0.796275 w_k1_FDX_s012
                                 + 0.704578 w_k1_IBM_s012
                                 + 0.619869 w_k1_INTC_s012
                                 + 0.782486 w_k1_KO_s012
                                 + 0.706615 w_k1_MAT_s012
                                 + 0.686766 w_k1_MSFT_s012
                                 + 0.876181 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.743409 w_k1_ADM_s012
                                 + 0.766952 w_k1_AXP_s012
                                 + 0.648526 w_k1_DIS_s012
                                 + 0.788116 w_k1_FDX_s012
                                 + 1.01098 w_k1_IBM_s012
                                 + 0.775728 w_k1_INTC_s012
                                 + 0.907497 w_k1_KO_s012
                                 + 0.991164 w_k1_MAT_s012
                                 + 0.883554 w_k1_MSFT_s012
                                 + 0.873527 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.807781 w_k1_ADM_s012
                                 + 0.929546 w_k1_AXP_s012
                                 + 0.95272 w_k1_DIS_s012
                                 + 1.06258 w_k1_FDX_s012
                                 + 0.72832 w_k1_IBM_s012
                                 + 0.795363 w_k1_INTC_s012
                                 + 0.78286 w_k1_KO_s012
                                 + 0.910663 w_k1_MAT_s012
                                 + 0.926244 w_k1_MSFT_s012
                                 + 0.969574 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.892153 w_k1_ADM_s013
                                 + 0.693881 w_k1_AXP_s013
                                 + 0.73761 w_k1_DIS_s013
                                 + 0.817405 w_k1_FDX_s013
                                 + 0.7126 w_k1_IBM_s013
                                 + 0.970331 w_k1_INTC_s013
                                 + 0.826898 w_k1_KO_s013
                                 + 0.473964 w_k1_MAT_s013
                                 + 0.892932 w_k1_MSFT_s013
                                 + 0.805692 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.82397 w_k1_ADM_s013
                                 + 1.70126 w_k1_AXP_s013
                                 + 1.53024 w_k1_DIS_s013
                                 + 1.48487 w_k1_FDX_s013
                                 + 0.933665 w_k1_IBM_s013
                                 + 1.24004 w_k1_INTC_s013
                                 + 1.09725 w_k1_KO_s013 + 1.18859 w_k1_MAT_s013
                                 + 1.22627 w_k1_MSFT_s013
                                 + 1.31555 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.13574 w_k1_ADM_s013
                                 + 0.744616 w_k1_AXP_s013
                                 + 1.08335 w_k1_DIS_s013
                                 + 1.03529 w_k1_FDX_s013
                                 + 0.770189 w_k1_IBM_s013
                                 + 0.961492 w_k1_INTC_s013
                                 + 0.960397 w_k1_KO_s013
                                 + 0.656231 w_k1_MAT_s013
                                 + 0.919931 w_k1_MSFT_s013
                                 + 1.02138 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.30374 w_k1_ADM_s013
                                 + 0.954935 w_k1_AXP_s013
                                 + 0.941801 w_k1_DIS_s013
                                 + 0.947563 w_k1_FDX_s013
                                 + 1.08117 w_k1_IBM_s013
                                 + 1.07238 w_k1_INTC_s013
                                 + 0.999008 w_k1_KO_s013
                                 + 0.833964 w_k1_MAT_s013
                                 + 1.01854 w_k1_MSFT_s013 + 1.0638 w_k1_PG_s013
                                 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.38555 w_k1_ADM_s013
                                 + 1.30009 w_k1_AXP_s013
                                 + 1.17125 w_k1_DIS_s013 + 1.0515 w_k1_FDX_s013
                                 + 1.43535 w_k1_IBM_s013
                                 + 0.83326 w_k1_INTC_s013 + 1.1668 w_k1_KO_s013
                                 + 1.42849 w_k1_MAT_s013
                                 + 1.0145 w_k1_MSFT_s013 + 1.11237 w_k1_PG_s013
                                 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.39806 w_k1_ADM_s013
                                 + 1.06448 w_k1_AXP_s013 + 1.5063 w_k1_DIS_s013
                                 + 1.24248 w_k1_FDX_s013
                                 + 0.994739 w_k1_IBM_s013
                                 + 1.04976 w_k1_INTC_s013
                                 + 1.24126 w_k1_KO_s013
                                 + 0.969218 w_k1_MAT_s013
                                 + 1.33586 w_k1_MSFT_s013
                                 + 1.21598 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.761737 w_k1_ADM_s013
                                 + 0.724956 w_k1_AXP_s013
                                 + 0.635309 w_k1_DIS_s013
                                 + 0.703163 w_k1_FDX_s013
                                 + 1.06783 w_k1_IBM_s013
                                 + 0.732914 w_k1_INTC_s013
                                 + 0.840289 w_k1_KO_s013
                                 + 0.823853 w_k1_MAT_s013
                                 + 0.855346 w_k1_MSFT_s013
                                 + 0.92815 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.11165 w_k1_ADM_s013
                                 + 1.11926 w_k1_AXP_s013
                                 + 0.95393 w_k1_DIS_s013
                                 + 1.05537 w_k1_FDX_s013
                                 + 0.876905 w_k1_IBM_s013
                                 + 0.956382 w_k1_INTC_s013
                                 + 0.866062 w_k1_KO_s013
                                 + 0.975137 w_k1_MAT_s013
                                 + 1.07042 w_k1_MSFT_s013
                                 + 1.11419 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.55655 w_k1_ADM_s014
                                 + 1.22499 w_k1_AXP_s014
                                 + 1.17351 w_k1_DIS_s014
                                 + 1.26714 w_k1_FDX_s014
                                 + 0.821194 w_k1_IBM_s014
                                 + 1.22943 w_k1_INTC_s014
                                 + 0.933579 w_k1_KO_s014
                                 + 0.918161 w_k1_MAT_s014
                                 + 1.40375 w_k1_MSFT_s014
                                 + 1.16274 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.02558 w_k1_ADM_s014
                                 + 0.887589 w_k1_AXP_s014
                                 + 0.874209 w_k1_DIS_s014
                                 + 0.805748 w_k1_FDX_s014
                                 + 1.06874 w_k1_IBM_s014
                                 + 0.972068 w_k1_INTC_s014
                                 + 0.966959 w_k1_KO_s014
                                 + 0.731889 w_k1_MAT_s014
                                 + 1.02577 w_k1_MSFT_s014
                                 + 0.952586 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.887875 w_k1_ADM_s014
                                 + 0.812838 w_k1_AXP_s014
                                 + 0.768584 w_k1_DIS_s014
                                 + 0.751261 w_k1_FDX_s014
                                 + 1.37247 w_k1_IBM_s014
                                 + 0.957699 w_k1_INTC_s014
                                 + 1.03848 w_k1_KO_s014
                                 + 0.902994 w_k1_MAT_s014
                                 + 0.920846 w_k1_MSFT_s014
                                 + 0.975486 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.975359 w_k1_ADM_s014
                                 + 0.711718 w_k1_AXP_s014
                                 + 0.658368 w_k1_DIS_s014
                                 + 0.822288 w_k1_FDX_s014
                                 + 0.722488 w_k1_IBM_s014
                                 + 0.905726 w_k1_INTC_s014
                                 + 0.798834 w_k1_KO_s014
                                 + 0.693324 w_k1_MAT_s014
                                 + 0.937099 w_k1_MSFT_s014
                                 + 0.886698 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.903806 w_k1_ADM_s014
                                 + 1.17274 w_k1_AXP_s014
                                 + 1.13684 w_k1_DIS_s014
                                 + 1.11473 w_k1_FDX_s014
                                 + 1.37299 w_k1_IBM_s014
                                 + 1.24702 w_k1_INTC_s014
                                 + 1.15573 w_k1_KO_s014 + 1.23533 w_k1_MAT_s014
                                 + 1.13291 w_k1_MSFT_s014
                                 + 1.05889 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.39022 w_k1_ADM_s014
                                 + 1.45913 w_k1_AXP_s014
                                 + 1.07386 w_k1_DIS_s014 + 1.1838 w_k1_FDX_s014
                                 + 0.77811 w_k1_IBM_s014
                                 + 1.37849 w_k1_INTC_s014
                                 + 0.951886 w_k1_KO_s014
                                 + 0.879875 w_k1_MAT_s014
                                 + 1.28009 w_k1_MSFT_s014
                                 + 1.07552 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.62819 w_k1_ADM_s014
                                 + 1.27883 w_k1_AXP_s014
                                 + 1.60273 w_k1_DIS_s014
                                 + 1.36076 w_k1_FDX_s014
                                 + 1.25812 w_k1_IBM_s014
                                 + 1.55205 w_k1_INTC_s014
                                 + 1.37483 w_k1_KO_s014
                                 + 0.841898 w_k1_MAT_s014
                                 + 1.52988 w_k1_MSFT_s014
                                 + 1.33777 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.22071 w_k1_ADM_s014
                                 + 0.95674 w_k1_AXP_s014
                                 + 0.931855 w_k1_DIS_s014
                                 + 1.02031 w_k1_FDX_s014
                                 + 0.874335 w_k1_IBM_s014
                                 + 1.20202 w_k1_INTC_s014
                                 + 1.05289 w_k1_KO_s014
                                 + 0.777448 w_k1_MAT_s014
                                 + 1.10215 w_k1_MSFT_s014
                                 + 0.967357 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.948908 w_k1_ADM_s015
                                 + 1.0496 w_k1_AXP_s015
                                 + 0.783152 w_k1_DIS_s015
                                 + 0.818286 w_k1_FDX_s015
                                 + 1.13617 w_k1_IBM_s015
                                 + 0.941621 w_k1_INTC_s015
                                 + 0.899574 w_k1_KO_s015
                                 + 0.966374 w_k1_MAT_s015
                                 + 0.886471 w_k1_MSFT_s015
                                 + 0.909129 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.69769 w_k1_ADM_s015
                                 + 1.49725 w_k1_AXP_s015
                                 + 1.31903 w_k1_DIS_s015
                                 + 1.30914 w_k1_FDX_s015
                                 + 0.966623 w_k1_IBM_s015
                                 + 1.15167 w_k1_INTC_s015
                                 + 1.16064 w_k1_KO_s015 + 1.06219 w_k1_MAT_s015
                                 + 1.13587 w_k1_MSFT_s015
                                 + 1.15285 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.859142 w_k1_ADM_s015
                                 + 0.61507 w_k1_AXP_s015
                                 + 0.662213 w_k1_DIS_s015
                                 + 0.84922 w_k1_FDX_s015
                                 + 0.676919 w_k1_IBM_s015
                                 + 1.11285 w_k1_INTC_s015
                                 + 0.651195 w_k1_KO_s015
                                 + 0.473701 w_k1_MAT_s015
                                 + 0.911786 w_k1_MSFT_s015
                                 + 0.854638 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.07457 w_k1_ADM_s015
                                 + 1.1671 w_k1_AXP_s015
                                 + 0.915977 w_k1_DIS_s015
                                 + 1.07168 w_k1_FDX_s015
                                 + 0.748648 w_k1_IBM_s015
                                 + 1.14497 w_k1_INTC_s015
                                 + 0.775226 w_k1_KO_s015
                                 + 0.65914 w_k1_MAT_s015
                                 + 1.0084 w_k1_MSFT_s015
                                 + 0.925113 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.29457 w_k1_ADM_s015
                                 + 1.0192 w_k1_AXP_s015 + 1.09576 w_k1_DIS_s015
                                 + 1.35116 w_k1_FDX_s015
                                 + 0.921869 w_k1_IBM_s015
                                 + 1.18017 w_k1_INTC_s015
                                 + 0.899449 w_k1_KO_s015
                                 + 0.668495 w_k1_MAT_s015
                                 + 1.1178 w_k1_MSFT_s015 + 1.06278 w_k1_PG_s015
                                 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.942397 w_k1_ADM_s015
                                 + 1.37437 w_k1_AXP_s015
                                 + 1.39357 w_k1_DIS_s015
                                 + 1.22805 w_k1_FDX_s015
                                 + 1.30083 w_k1_IBM_s015
                                 + 1.38101 w_k1_INTC_s015
                                 + 1.17945 w_k1_KO_s015 + 1.14875 w_k1_MAT_s015
                                 + 1.29888 w_k1_MSFT_s015
                                 + 1.06746 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.978923 w_k1_ADM_s015
                                 + 1.04541 w_k1_AXP_s015
                                 + 1.07266 w_k1_DIS_s015
                                 + 0.975909 w_k1_FDX_s015
                                 + 1.34493 w_k1_IBM_s015
                                 + 1.13507 w_k1_INTC_s015
                                 + 1.18676 w_k1_KO_s015 + 1.03398 w_k1_MAT_s015
                                 + 1.1331 w_k1_MSFT_s015 + 1.07408 w_k1_PG_s015
                                 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.809198 w_k1_ADM_s015
                                 + 0.853437 w_k1_AXP_s015
                                 + 0.803793 w_k1_DIS_s015
                                 + 0.811363 w_k1_FDX_s015
                                 + 0.911465 w_k1_IBM_s015
                                 + 0.860729 w_k1_INTC_s015
                                 + 0.850983 w_k1_KO_s015
                                 + 0.817907 w_k1_MAT_s015
                                 + 0.782676 w_k1_MSFT_s015
                                 + 0.879996 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.01237 w_k1_ADM_s016
                                 + 1.01826 w_k1_AXP_s016
                                 + 1.23063 w_k1_DIS_s016
                                 + 1.04242 w_k1_FDX_s016
                                 + 1.09062 w_k1_IBM_s016
                                 + 1.08955 w_k1_INTC_s016
                                 + 1.21649 w_k1_KO_s016 + 1.28845 w_k1_MAT_s016
                                 + 1.21011 w_k1_MSFT_s016
                                 + 1.12975 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.93892 w_k1_ADM_s016
                                 + 0.908947 w_k1_AXP_s016
                                 + 0.781223 w_k1_DIS_s016
                                 + 0.792719 w_k1_FDX_s016
                                 + 1.13135 w_k1_IBM_s016
                                 + 0.735852 w_k1_INTC_s016
                                 + 0.944928 w_k1_KO_s016
                                 + 1.05532 w_k1_MAT_s016
                                 + 0.793257 w_k1_MSFT_s016
                                 + 0.902973 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.23233 w_k1_ADM_s016
                                 + 0.820011 w_k1_AXP_s016
                                 + 0.86398 w_k1_DIS_s016
                                 + 0.988425 w_k1_FDX_s016
                                 + 0.67248 w_k1_IBM_s016
                                 + 1.0216 w_k1_INTC_s016
                                 + 0.852616 w_k1_KO_s016
                                 + 0.698757 w_k1_MAT_s016
                                 + 0.99443 w_k1_MSFT_s016
                                 + 0.952533 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.24736 w_k1_ADM_s016
                                 + 1.36621 w_k1_AXP_s016
                                 + 1.05819 w_k1_DIS_s016
                                 + 1.24445 w_k1_FDX_s016
                                 + 0.849098 w_k1_IBM_s016
                                 + 1.22116 w_k1_INTC_s016
                                 + 1.01454 w_k1_KO_s016 + 1.26309 w_k1_MAT_s016
                                 + 1.15504 w_k1_MSFT_s016 + 1.2779 w_k1_PG_s016
                                 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.98646 w_k1_ADM_s016
                                 + 0.857832 w_k1_AXP_s016
                                 + 0.79638 w_k1_DIS_s016
                                 + 0.786569 w_k1_FDX_s016
                                 + 0.798969 w_k1_IBM_s016
                                 + 1.00018 w_k1_INTC_s016
                                 + 0.89472 w_k1_KO_s016
                                 + 0.759632 w_k1_MAT_s016
                                 + 0.909671 w_k1_MSFT_s016
                                 + 0.854429 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.57241 w_k1_ADM_s016
                                 + 1.2777 w_k1_AXP_s016 + 1.67175 w_k1_DIS_s016
                                 + 1.38563 w_k1_FDX_s016
                                 + 0.840339 w_k1_IBM_s016
                                 + 1.07177 w_k1_INTC_s016
                                 + 1.26159 w_k1_KO_s016 + 1.17359 w_k1_MAT_s016
                                 + 1.33748 w_k1_MSFT_s016
                                 + 1.27678 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.17411 w_k1_ADM_s016
                                 + 1.00244 w_k1_AXP_s016 + 1.281 w_k1_DIS_s016
                                 + 1.0313 w_k1_FDX_s016 + 1.82091 w_k1_IBM_s016
                                 + 1.47862 w_k1_INTC_s016
                                 + 1.54742 w_k1_KO_s016 + 1.43998 w_k1_MAT_s016
                                 + 1.25277 w_k1_MSFT_s016
                                 + 1.13931 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.55871 w_k1_ADM_s016
                                 + 1.48617 w_k1_AXP_s016
                                 + 1.53723 w_k1_DIS_s016
                                 + 1.36044 w_k1_FDX_s016
                                 + 1.26274 w_k1_IBM_s016
                                 + 1.43033 w_k1_INTC_s016 + 1.4437 w_k1_KO_s016
                                 + 1.94314 w_k1_MAT_s016
                                 + 1.50934 w_k1_MSFT_s016
                                 + 1.20959 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.839654 w_k1_ADM_s017
                                 + 1.17701 w_k1_AXP_s017
                                 + 1.10745 w_k1_DIS_s017
                                 + 0.910503 w_k1_FDX_s017
                                 + 1.66814 w_k1_IBM_s017
                                 + 1.11382 w_k1_INTC_s017 + 1.2215 w_k1_KO_s017
                                 + 1.57974 w_k1_MAT_s017
                                 + 0.869319 w_k1_MSFT_s017
                                 + 1.10143 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.922313 w_k1_ADM_s017
                                 + 0.683378 w_k1_AXP_s017
                                 + 0.992506 w_k1_DIS_s017
                                 + 1.03613 w_k1_FDX_s017
                                 + 0.735635 w_k1_IBM_s017
                                 + 1.07283 w_k1_INTC_s017
                                 + 0.861368 w_k1_KO_s017
                                 + 0.744353 w_k1_MAT_s017
                                 + 1.03187 w_k1_MSFT_s017
                                 + 0.913604 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.11908 w_k1_ADM_s017
                                 + 1.30572 w_k1_AXP_s017
                                 + 0.915299 w_k1_DIS_s017
                                 + 1.18778 w_k1_FDX_s017
                                 + 0.556493 w_k1_IBM_s017
                                 + 0.833548 w_k1_INTC_s017
                                 + 0.753502 w_k1_KO_s017
                                 + 1.20647 w_k1_MAT_s017
                                 + 0.901791 w_k1_MSFT_s017
                                 + 0.94487 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.670884 w_k1_ADM_s017
                                 + 0.884618 w_k1_AXP_s017
                                 + 0.840577 w_k1_DIS_s017
                                 + 0.911295 w_k1_FDX_s017
                                 + 1.12648 w_k1_IBM_s017
                                 + 0.958017 w_k1_INTC_s017
                                 + 1.04415 w_k1_KO_s017 + 1.57505 w_k1_MAT_s017
                                 + 0.939067 w_k1_MSFT_s017
                                 + 1.01454 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.22471 w_k1_ADM_s017
                                 + 1.54996 w_k1_AXP_s017
                                 + 1.48204 w_k1_DIS_s017
                                 + 1.45003 w_k1_FDX_s017
                                 + 1.18454 w_k1_IBM_s017
                                 + 1.4254 w_k1_INTC_s017 + 1.29339 w_k1_KO_s017
                                 + 1.85308 w_k1_MAT_s017
                                 + 1.29719 w_k1_MSFT_s017
                                 + 1.24308 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.61799 w_k1_ADM_s017
                                 + 0.656257 w_k1_AXP_s017
                                 + 0.694811 w_k1_DIS_s017
                                 + 0.738807 w_k1_FDX_s017
                                 + 0.787554 w_k1_IBM_s017
                                 + 0.8786 w_k1_INTC_s017
                                 + 0.805893 w_k1_KO_s017
                                 + 0.67297 w_k1_MAT_s017
                                 + 0.770217 w_k1_MSFT_s017
                                 + 0.757947 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.91195 w_k1_ADM_s017
                                 + 1.22056 w_k1_AXP_s017
                                 + 1.28329 w_k1_DIS_s017
                                 + 1.12706 w_k1_FDX_s017
                                 + 1.07129 w_k1_IBM_s017
                                 + 1.17861 w_k1_INTC_s017
                                 + 1.16243 w_k1_KO_s017 + 1.20261 w_k1_MAT_s017
                                 + 1.18583 w_k1_MSFT_s017
                                 + 1.07723 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.883667 w_k1_ADM_s017
                                 + 1.26559 w_k1_AXP_s017
                                 + 0.860132 w_k1_DIS_s017
                                 + 1.05063 w_k1_FDX_s017
                                 + 0.944856 w_k1_IBM_s017
                                 + 1.20616 w_k1_INTC_s017
                                 + 0.89526 w_k1_KO_s017 + 1.04286 w_k1_MAT_s017
                                 + 0.930984 w_k1_MSFT_s017
                                 + 0.862168 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.44402 w_k1_ADM_s100
                                 + 1.33867 w_k1_AXP_s100
                                 + 1.19662 w_k1_DIS_s100
                                 + 1.27117 w_k1_FDX_s100
                                 + 1.28716 w_k1_IBM_s100 + 1.072 w_k1_INTC_s100
                                 + 1.28431 w_k1_KO_s100 + 1.45966 w_k1_MAT_s100
                                 + 1.18836 w_k1_MSFT_s100
                                 + 1.25344 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.948119 w_k1_ADM_s100
                                 + 0.95656 w_k1_AXP_s100
                                 + 0.915988 w_k1_DIS_s100
                                 + 0.901186 w_k1_FDX_s100
                                 + 1.33181 w_k1_IBM_s100
                                 + 0.859533 w_k1_INTC_s100
                                 + 1.0593 w_k1_KO_s100 + 1.32038 w_k1_MAT_s100
                                 + 0.822324 w_k1_MSFT_s100
                                 + 0.85558 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.980188 w_k1_ADM_s100
                                 + 0.75424 w_k1_AXP_s100
                                 + 0.801969 w_k1_DIS_s100
                                 + 0.883482 w_k1_FDX_s100 + 1.113 w_k1_IBM_s100
                                 + 0.822543 w_k1_INTC_s100
                                 + 0.997998 w_k1_KO_s100
                                 + 0.960072 w_k1_MAT_s100
                                 + 0.868091 w_k1_MSFT_s100
                                 + 0.961468 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.08941 w_k1_ADM_s100
                                 + 1.08105 w_k1_AXP_s100
                                 + 0.869619 w_k1_DIS_s100
                                 + 1.12514 w_k1_FDX_s100
                                 + 0.766387 w_k1_IBM_s100
                                 + 0.993062 w_k1_INTC_s100
                                 + 0.832241 w_k1_KO_s100
                                 + 0.826654 w_k1_MAT_s100
                                 + 1.0058 w_k1_MSFT_s100
                                 + 0.952164 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.865647 w_k1_ADM_s100
                                 + 0.774903 w_k1_AXP_s100
                                 + 0.684036 w_k1_DIS_s100
                                 + 0.865235 w_k1_FDX_s100 + 0.932 w_k1_IBM_s100
                                 + 0.936485 w_k1_INTC_s100
                                 + 0.997431 w_k1_KO_s100
                                 + 1.06374 w_k1_MAT_s100
                                 + 0.860322 w_k1_MSFT_s100
                                 + 0.866196 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.48337 w_k1_ADM_s100
                                 + 1.3914 w_k1_AXP_s100 + 1.44874 w_k1_DIS_s100
                                 + 1.18299 w_k1_FDX_s100 + 1.5721 w_k1_IBM_s100
                                 + 1.28573 w_k1_INTC_s100
                                 + 1.50375 w_k1_KO_s100 + 1.61408 w_k1_MAT_s100
                                 + 1.36186 w_k1_MSFT_s100
                                 + 1.27711 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.00555 w_k1_ADM_s100
                                 + 1.49725 w_k1_AXP_s100
                                 + 1.07765 w_k1_DIS_s100
                                 + 1.12452 w_k1_FDX_s100
                                 + 1.15742 w_k1_IBM_s100
                                 + 1.32166 w_k1_INTC_s100
                                 + 1.13562 w_k1_KO_s100 + 1.23472 w_k1_MAT_s100
                                 + 1.06475 w_k1_MSFT_s100 + 1.1796 w_k1_PG_s100
                                 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.39375 w_k1_ADM_s100
                                 + 1.42636 w_k1_AXP_s100
                                 + 1.50076 w_k1_DIS_s100
                                 + 1.28156 w_k1_FDX_s100
                                 + 0.989462 w_k1_IBM_s100
                                 + 1.09617 w_k1_INTC_s100 + 1.1743 w_k1_KO_s100
                                 + 0.93861 w_k1_MAT_s100
                                 + 1.15359 w_k1_MSFT_s100
                                 + 1.18834 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.20597 w_k1_ADM_s101
                                 + 1.04387 w_k1_AXP_s101
                                 + 1.32875 w_k1_DIS_s101
                                 + 1.18738 w_k1_FDX_s101
                                 + 0.816407 w_k1_IBM_s101
                                 + 1.0491 w_k1_INTC_s101 + 1.15333 w_k1_KO_s101
                                 + 0.85225 w_k1_MAT_s101
                                 + 1.15823 w_k1_MSFT_s101
                                 + 0.968725 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.828936 w_k1_ADM_s101
                                 + 0.764779 w_k1_AXP_s101
                                 + 0.845932 w_k1_DIS_s101
                                 + 0.891366 w_k1_FDX_s101
                                 + 0.828249 w_k1_IBM_s101
                                 + 1.03886 w_k1_INTC_s101
                                 + 0.965825 w_k1_KO_s101
                                 + 0.768153 w_k1_MAT_s101
                                 + 1.03122 w_k1_MSFT_s101
                                 + 0.888732 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.97439 w_k1_ADM_s101
                                 + 1.20757 w_k1_AXP_s101
                                 + 1.20911 w_k1_DIS_s101 + 1.117 w_k1_FDX_s101
                                 + 1.08818 w_k1_IBM_s101
                                 + 1.33322 w_k1_INTC_s101
                                 + 1.20605 w_k1_KO_s101 + 1.05982 w_k1_MAT_s101
                                 + 1.08358 w_k1_MSFT_s101
                                 + 1.01086 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.753667 w_k1_ADM_s101
                                 + 0.936798 w_k1_AXP_s101
                                 + 0.585822 w_k1_DIS_s101
                                 + 0.747292 w_k1_FDX_s101
                                 + 0.777487 w_k1_IBM_s101
                                 + 0.8643 w_k1_INTC_s101
                                 + 0.732841 w_k1_KO_s101
                                 + 0.737475 w_k1_MAT_s101
                                 + 0.717341 w_k1_MSFT_s101
                                 + 0.785694 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.06415 w_k1_ADM_s101
                                 + 1.05349 w_k1_AXP_s101
                                 + 1.78716 w_k1_DIS_s101
                                 + 1.22194 w_k1_FDX_s101
                                 + 1.06924 w_k1_IBM_s101
                                 + 1.21698 w_k1_INTC_s101
                                 + 1.24457 w_k1_KO_s101
                                 + 0.953194 w_k1_MAT_s101
                                 + 1.26666 w_k1_MSFT_s101
                                 + 1.07895 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.972693 w_k1_ADM_s101
                                 + 0.95568 w_k1_AXP_s101
                                 + 0.92666 w_k1_DIS_s101
                                 + 0.850645 w_k1_FDX_s101
                                 + 1.17623 w_k1_IBM_s101
                                 + 1.00272 w_k1_INTC_s101
                                 + 0.96883 w_k1_KO_s101 + 1.05682 w_k1_MAT_s101
                                 + 0.880666 w_k1_MSFT_s101
                                 + 0.868654 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.748101 w_k1_ADM_s101
                                 + 0.680137 w_k1_AXP_s101
                                 + 0.764492 w_k1_DIS_s101
                                 + 0.88367 w_k1_FDX_s101
                                 + 0.555758 w_k1_IBM_s101
                                 + 1.08329 w_k1_INTC_s101
                                 + 0.816871 w_k1_KO_s101
                                 + 0.645418 w_k1_MAT_s101
                                 + 0.96244 w_k1_MSFT_s101
                                 + 0.73895 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.2443 w_k1_ADM_s101
                                 + 1.47517 w_k1_AXP_s101
                                 + 1.67709 w_k1_DIS_s101
                                 + 1.38333 w_k1_FDX_s101
                                 + 1.38223 w_k1_IBM_s101
                                 + 1.48624 w_k1_INTC_s101
                                 + 1.53447 w_k1_KO_s101 + 1.21663 w_k1_MAT_s101
                                 + 1.28036 w_k1_MSFT_s101 + 1.2154 w_k1_PG_s101
                                 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.976652 w_k1_ADM_s102
                                 + 0.870291 w_k1_AXP_s102
                                 + 1.0177 w_k1_DIS_s102
                                 + 0.866847 w_k1_FDX_s102
                                 + 1.23835 w_k1_IBM_s102
                                 + 0.914081 w_k1_INTC_s102
                                 + 1.08079 w_k1_KO_s102
                                 + 0.989896 w_k1_MAT_s102
                                 + 0.984882 w_k1_MSFT_s102
                                 + 1.00507 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.652168 w_k1_ADM_s102
                                 + 0.759788 w_k1_AXP_s102
                                 + 0.838991 w_k1_DIS_s102
                                 + 0.762146 w_k1_FDX_s102
                                 + 0.844872 w_k1_IBM_s102
                                 + 0.931438 w_k1_INTC_s102
                                 + 0.761753 w_k1_KO_s102
                                 + 0.612717 w_k1_MAT_s102
                                 + 0.952302 w_k1_MSFT_s102
                                 + 0.926233 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.955502 w_k1_ADM_s102
                                 + 0.805349 w_k1_AXP_s102
                                 + 0.938334 w_k1_DIS_s102
                                 + 0.875943 w_k1_FDX_s102
                                 + 0.720613 w_k1_IBM_s102
                                 + 0.883763 w_k1_INTC_s102
                                 + 0.789343 w_k1_KO_s102
                                 + 0.719978 w_k1_MAT_s102
                                 + 0.906964 w_k1_MSFT_s102
                                 + 0.887631 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.876187 w_k1_ADM_s102
                                 + 0.932247 w_k1_AXP_s102
                                 + 1.16007 w_k1_DIS_s102
                                 + 0.930995 w_k1_FDX_s102
                                 + 0.990434 w_k1_IBM_s102
                                 + 0.932069 w_k1_INTC_s102
                                 + 1.08954 w_k1_KO_s102 + 1.12407 w_k1_MAT_s102
                                 + 1.10201 w_k1_MSFT_s102
                                 + 0.984905 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.04271 w_k1_ADM_s102
                                 + 1.13816 w_k1_AXP_s102
                                 + 1.12317 w_k1_DIS_s102
                                 + 0.986287 w_k1_FDX_s102
                                 + 1.08271 w_k1_IBM_s102
                                 + 1.0847 w_k1_INTC_s102 + 1.07591 w_k1_KO_s102
                                 + 0.969689 w_k1_MAT_s102
                                 + 1.01362 w_k1_MSFT_s102
                                 + 1.05611 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.33885 w_k1_ADM_s102
                                 + 1.65996 w_k1_AXP_s102
                                 + 1.56931 w_k1_DIS_s102
                                 + 1.36007 w_k1_FDX_s102
                                 + 1.52771 w_k1_IBM_s102
                                 + 1.39263 w_k1_INTC_s102
                                 + 1.37965 w_k1_KO_s102 + 1.69151 w_k1_MAT_s102
                                 + 1.23587 w_k1_MSFT_s102
                                 + 1.24754 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.0527 w_k1_ADM_s102
                                 + 1.3921 w_k1_AXP_s102 + 1.60976 w_k1_DIS_s102
                                 + 1.25636 w_k1_FDX_s102
                                 + 1.02208 w_k1_IBM_s102
                                 + 1.12159 w_k1_INTC_s102
                                 + 1.13464 w_k1_KO_s102 + 1.06385 w_k1_MAT_s102
                                 + 1.27485 w_k1_MSFT_s102
                                 + 1.07838 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.0743 w_k1_ADM_s102
                                 + 1.61504 w_k1_AXP_s102
                                 + 0.828611 w_k1_DIS_s102
                                 + 0.97339 w_k1_FDX_s102
                                 + 0.875795 w_k1_IBM_s102
                                 + 0.902939 w_k1_INTC_s102
                                 + 0.788117 w_k1_KO_s102 + 1.2531 w_k1_MAT_s102
                                 + 0.970196 w_k1_MSFT_s102
                                 + 1.00131 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.16113 w_k1_ADM_s103
                                 + 0.854743 w_k1_AXP_s103
                                 + 0.888677 w_k1_DIS_s103
                                 + 0.984295 w_k1_FDX_s103
                                 + 0.834281 w_k1_IBM_s103
                                 + 1.01874 w_k1_INTC_s103
                                 + 1.01858 w_k1_KO_s103
                                 + 0.829916 w_k1_MAT_s103
                                 + 0.935907 w_k1_MSFT_s103
                                 + 0.932298 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.686924 w_k1_ADM_s103
                                 + 0.514701 w_k1_AXP_s103
                                 + 0.430669 w_k1_DIS_s103
                                 + 0.594633 w_k1_FDX_s103
                                 + 0.637742 w_k1_IBM_s103
                                 + 0.766859 w_k1_INTC_s103
                                 + 0.650297 w_k1_KO_s103
                                 + 0.559559 w_k1_MAT_s103
                                 + 0.580903 w_k1_MSFT_s103
                                 + 0.645814 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.877691 w_k1_ADM_s103
                                 + 0.694331 w_k1_AXP_s103
                                 + 0.684812 w_k1_DIS_s103
                                 + 0.827543 w_k1_FDX_s103
                                 + 0.584506 w_k1_IBM_s103
                                 + 0.776741 w_k1_INTC_s103
                                 + 0.704786 w_k1_KO_s103
                                 + 0.663925 w_k1_MAT_s103
                                 + 0.803206 w_k1_MSFT_s103
                                 + 0.812209 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.0419 w_k1_ADM_s103
                                 + 1.101 w_k1_AXP_s103 + 1.37964 w_k1_DIS_s103
                                 + 1.1267 w_k1_FDX_s103 + 1.38857 w_k1_IBM_s103
                                 + 1.17412 w_k1_INTC_s103
                                 + 1.45718 w_k1_KO_s103 + 1.34704 w_k1_MAT_s103
                                 + 1.18844 w_k1_MSFT_s103
                                 + 1.19185 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.871747 w_k1_ADM_s103
                                 + 0.774935 w_k1_AXP_s103
                                 + 0.553131 w_k1_DIS_s103
                                 + 0.649487 w_k1_FDX_s103
                                 + 1.02964 w_k1_IBM_s103
                                 + 0.727888 w_k1_INTC_s103
                                 + 0.822736 w_k1_KO_s103
                                 + 1.23285 w_k1_MAT_s103
                                 + 0.61083 w_k1_MSFT_s103
                                 + 0.789914 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.35755 w_k1_ADM_s103
                                 + 1.39108 w_k1_AXP_s103
                                 + 1.36072 w_k1_DIS_s103
                                 + 1.27989 w_k1_FDX_s103
                                 + 1.02421 w_k1_IBM_s103
                                 + 1.31646 w_k1_INTC_s103
                                 + 1.20665 w_k1_KO_s103 + 1.14566 w_k1_MAT_s103
                                 + 1.23119 w_k1_MSFT_s103
                                 + 1.07615 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.918171 w_k1_ADM_s103
                                 + 0.954334 w_k1_AXP_s103
                                 + 0.687221 w_k1_DIS_s103
                                 + 0.796469 w_k1_FDX_s103
                                 + 0.84538 w_k1_IBM_s103
                                 + 1.04511 w_k1_INTC_s103
                                 + 0.912568 w_k1_KO_s103
                                 + 0.784447 w_k1_MAT_s103
                                 + 0.843578 w_k1_MSFT_s103
                                 + 0.881682 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.932312 w_k1_ADM_s103
                                 + 0.777025 w_k1_AXP_s103
                                 + 0.799512 w_k1_DIS_s103
                                 + 0.833858 w_k1_FDX_s103
                                 + 1.03723 w_k1_IBM_s103
                                 + 0.861217 w_k1_INTC_s103
                                 + 0.928008 w_k1_KO_s103
                                 + 0.955598 w_k1_MAT_s103
                                 + 0.866022 w_k1_MSFT_s103
                                 + 0.870709 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.08029 w_k1_ADM_s104
                                 + 1.16579 w_k1_AXP_s104
                                 + 1.19198 w_k1_DIS_s104
                                 + 1.04042 w_k1_FDX_s104
                                 + 1.61712 w_k1_IBM_s104
                                 + 1.02203 w_k1_INTC_s104
                                 + 1.20693 w_k1_KO_s104 + 1.73797 w_k1_MAT_s104
                                 + 1.18964 w_k1_MSFT_s104
                                 + 1.08044 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.61557 w_k1_ADM_s104
                                 + 0.668233 w_k1_AXP_s104
                                 + 0.70778 w_k1_DIS_s104
                                 + 0.706895 w_k1_FDX_s104
                                 + 0.839762 w_k1_IBM_s104
                                 + 0.705327 w_k1_INTC_s104
                                 + 0.8063 w_k1_KO_s104 + 0.732915 w_k1_MAT_s104
                                 + 0.665028 w_k1_MSFT_s104
                                 + 0.881703 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.877272 w_k1_ADM_s104
                                 + 1.11829 w_k1_AXP_s104
                                 + 0.886359 w_k1_DIS_s104
                                 + 0.968586 w_k1_FDX_s104
                                 + 0.867563 w_k1_IBM_s104
                                 + 0.983725 w_k1_INTC_s104
                                 + 0.929668 w_k1_KO_s104
                                 + 0.881138 w_k1_MAT_s104
                                 + 0.891824 w_k1_MSFT_s104
                                 + 1.08221 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.826593 w_k1_ADM_s104
                                 + 1.13389 w_k1_AXP_s104
                                 + 1.53575 w_k1_DIS_s104
                                 + 1.19864 w_k1_FDX_s104 + 0.8816 w_k1_IBM_s104
                                 + 0.895255 w_k1_INTC_s104
                                 + 1.11097 w_k1_KO_s104
                                 + 0.792693 w_k1_MAT_s104
                                 + 0.983086 w_k1_MSFT_s104
                                 + 1.10352 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.04757 w_k1_ADM_s104
                                 + 1.61261 w_k1_AXP_s104 + 1.4062 w_k1_DIS_s104
                                 + 1.30333 w_k1_FDX_s104
                                 + 1.18779 w_k1_IBM_s104
                                 + 1.14289 w_k1_INTC_s104
                                 + 1.27242 w_k1_KO_s104 + 1.40619 w_k1_MAT_s104
                                 + 1.27022 w_k1_MSFT_s104
                                 + 1.26995 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.83572 w_k1_ADM_s104
                                 + 0.876872 w_k1_AXP_s104
                                 + 1.02034 w_k1_DIS_s104 + 1.0276 w_k1_FDX_s104
                                 + 0.686552 w_k1_IBM_s104
                                 + 0.786633 w_k1_INTC_s104
                                 + 0.870175 w_k1_KO_s104
                                 + 0.670263 w_k1_MAT_s104
                                 + 0.864874 w_k1_MSFT_s104
                                 + 0.847628 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.892673 w_k1_ADM_s104
                                 + 1.12684 w_k1_AXP_s104
                                 + 0.850296 w_k1_DIS_s104
                                 + 0.908809 w_k1_FDX_s104
                                 + 1.14478 w_k1_IBM_s104
                                 + 0.898522 w_k1_INTC_s104
                                 + 0.920413 w_k1_KO_s104
                                 + 1.30576 w_k1_MAT_s104
                                 + 0.918575 w_k1_MSFT_s104
                                 + 0.939215 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.08211 w_k1_ADM_s104
                                 + 1.78446 w_k1_AXP_s104
                                 + 1.66682 w_k1_DIS_s104
                                 + 1.31453 w_k1_FDX_s104
                                 + 1.42078 w_k1_IBM_s104
                                 + 1.0061 w_k1_INTC_s104 + 1.46948 w_k1_KO_s104
                                 + 1.53562 w_k1_MAT_s104
                                 + 1.00772 w_k1_MSFT_s104
                                 + 1.15618 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.04811 w_k1_ADM_s105
                                 + 0.807372 w_k1_AXP_s105
                                 + 0.936249 w_k1_DIS_s105
                                 + 0.958867 w_k1_FDX_s105
                                 + 0.999646 w_k1_IBM_s105
                                 + 0.993791 w_k1_INTC_s105
                                 + 0.984352 w_k1_KO_s105
                                 + 0.927768 w_k1_MAT_s105
                                 + 0.971865 w_k1_MSFT_s105
                                 + 1.05281 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.924646 w_k1_ADM_s105
                                 + 1.1631 w_k1_AXP_s105 + 1.16927 w_k1_DIS_s105
                                 + 1.01415 w_k1_FDX_s105
                                 + 1.50899 w_k1_IBM_s105
                                 + 1.13286 w_k1_INTC_s105
                                 + 1.44772 w_k1_KO_s105 + 1.69744 w_k1_MAT_s105
                                 + 0.96445 w_k1_MSFT_s105 + 1.1356 w_k1_PG_s105
                                 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.02236 w_k1_ADM_s105
                                 + 0.74672 w_k1_AXP_s105
                                 + 0.651642 w_k1_DIS_s105
                                 + 0.811484 w_k1_FDX_s105
                                 + 0.731229 w_k1_IBM_s105
                                 + 0.822691 w_k1_INTC_s105
                                 + 0.885087 w_k1_KO_s105
                                 + 0.775967 w_k1_MAT_s105
                                 + 0.841311 w_k1_MSFT_s105
                                 + 0.878415 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.778372 w_k1_ADM_s105
                                 + 0.663038 w_k1_AXP_s105
                                 + 0.714417 w_k1_DIS_s105
                                 + 0.763678 w_k1_FDX_s105
                                 + 0.923734 w_k1_IBM_s105
                                 + 0.749106 w_k1_INTC_s105
                                 + 0.873175 w_k1_KO_s105
                                 + 0.762876 w_k1_MAT_s105
                                 + 0.81984 w_k1_MSFT_s105
                                 + 0.878058 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.904067 w_k1_ADM_s105
                                 + 0.893086 w_k1_AXP_s105
                                 + 0.956932 w_k1_DIS_s105
                                 + 0.959387 w_k1_FDX_s105
                                 + 0.719664 w_k1_IBM_s105
                                 + 1.05783 w_k1_INTC_s105
                                 + 0.912523 w_k1_KO_s105
                                 + 0.671725 w_k1_MAT_s105
                                 + 0.945315 w_k1_MSFT_s105
                                 + 0.926095 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.08173 w_k1_ADM_s105
                                 + 0.823436 w_k1_AXP_s105
                                 + 0.91267 w_k1_DIS_s105
                                 + 0.871607 w_k1_FDX_s105
                                 + 1.26148 w_k1_IBM_s105
                                 + 0.856648 w_k1_INTC_s105
                                 + 1.11084 w_k1_KO_s105
                                 + 0.994963 w_k1_MAT_s105
                                 + 0.847774 w_k1_MSFT_s105
                                 + 1.15415 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.12751 w_k1_ADM_s105
                                 + 1.48935 w_k1_AXP_s105
                                 + 1.46822 w_k1_DIS_s105
                                 + 1.20041 w_k1_FDX_s105
                                 + 1.35762 w_k1_IBM_s105
                                 + 1.35377 w_k1_INTC_s105
                                 + 1.58227 w_k1_KO_s105 + 1.88248 w_k1_MAT_s105
                                 + 1.35499 w_k1_MSFT_s105
                                 + 1.20165 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.00218 w_k1_ADM_s105
                                 + 1.20128 w_k1_AXP_s105
                                 + 1.38257 w_k1_DIS_s105
                                 + 1.22524 w_k1_FDX_s105
                                 + 0.993198 w_k1_IBM_s105
                                 + 1.05345 w_k1_INTC_s105
                                 + 1.31837 w_k1_KO_s105 + 1.13955 w_k1_MAT_s105
                                 + 1.19564 w_k1_MSFT_s105
                                 + 1.19975 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.16562 w_k1_ADM_s106
                                 + 1.26239 w_k1_AXP_s106
                                 + 1.64927 w_k1_DIS_s106
                                 + 1.28729 w_k1_FDX_s106
                                 + 1.19704 w_k1_IBM_s106
                                 + 1.06427 w_k1_INTC_s106
                                 + 1.32561 w_k1_KO_s106 + 1.16235 w_k1_MAT_s106
                                 + 1.09618 w_k1_MSFT_s106
                                 + 1.15522 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.707084 w_k1_ADM_s106
                                 + 0.974354 w_k1_AXP_s106
                                 + 0.793098 w_k1_DIS_s106
                                 + 0.857174 w_k1_FDX_s106
                                 + 1.33264 w_k1_IBM_s106
                                 + 0.962879 w_k1_INTC_s106
                                 + 0.993748 w_k1_KO_s106
                                 + 1.22238 w_k1_MAT_s106
                                 + 0.945422 w_k1_MSFT_s106
                                 + 1.01204 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.867135 w_k1_ADM_s106
                                 + 1.05742 w_k1_AXP_s106
                                 + 1.16886 w_k1_DIS_s106
                                 + 1.09392 w_k1_FDX_s106
                                 + 0.763191 w_k1_IBM_s106
                                 + 1.19426 w_k1_INTC_s106
                                 + 1.17926 w_k1_KO_s106
                                 + 0.714371 w_k1_MAT_s106
                                 + 1.25501 w_k1_MSFT_s106
                                 + 0.94703 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.37662 w_k1_ADM_s106
                                 + 1.84395 w_k1_AXP_s106
                                 + 1.54929 w_k1_DIS_s106
                                 + 1.55522 w_k1_FDX_s106
                                 + 1.16695 w_k1_IBM_s106
                                 + 1.34817 w_k1_INTC_s106
                                 + 1.10361 w_k1_KO_s106 + 1.15957 w_k1_MAT_s106
                                 + 1.30665 w_k1_MSFT_s106
                                 + 1.28739 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.09824 w_k1_ADM_s106
                                 + 1.32914 w_k1_AXP_s106
                                 + 1.19094 w_k1_DIS_s106
                                 + 1.11431 w_k1_FDX_s106
                                 + 1.27296 w_k1_IBM_s106
                                 + 0.998091 w_k1_INTC_s106
                                 + 1.13276 w_k1_KO_s106
                                 + 0.919055 w_k1_MAT_s106
                                 + 0.946904 w_k1_MSFT_s106
                                 + 1.08481 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.942645 w_k1_ADM_s106
                                 + 1.19134 w_k1_AXP_s106
                                 + 0.878625 w_k1_DIS_s106
                                 + 0.986172 w_k1_FDX_s106
                                 + 1.00885 w_k1_IBM_s106
                                 + 0.858595 w_k1_INTC_s106
                                 + 0.935903 w_k1_KO_s106
                                 + 0.994143 w_k1_MAT_s106
                                 + 0.896202 w_k1_MSFT_s106
                                 + 0.982181 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.06335 w_k1_ADM_s106
                                 + 1.1604 w_k1_AXP_s106 + 1.54229 w_k1_DIS_s106
                                 + 1.16111 w_k1_FDX_s106
                                 + 1.82554 w_k1_IBM_s106
                                 + 1.02327 w_k1_INTC_s106
                                 + 1.43946 w_k1_KO_s106 + 1.68237 w_k1_MAT_s106
                                 + 1.25109 w_k1_MSFT_s106
                                 + 1.16409 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.759479 w_k1_ADM_s106
                                 + 0.879973 w_k1_AXP_s106
                                 + 0.715188 w_k1_DIS_s106
                                 + 0.778438 w_k1_FDX_s106
                                 + 0.716773 w_k1_IBM_s106
                                 + 0.761168 w_k1_INTC_s106
                                 + 0.759973 w_k1_KO_s106
                                 + 0.734539 w_k1_MAT_s106
                                 + 0.81211 w_k1_MSFT_s106
                                 + 0.918965 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.11192 w_k1_ADM_s107
                                 + 0.963962 w_k1_AXP_s107
                                 + 0.958208 w_k1_DIS_s107
                                 + 1.09718 w_k1_FDX_s107
                                 + 0.690715 w_k1_IBM_s107
                                 + 0.807353 w_k1_INTC_s107
                                 + 0.973061 w_k1_KO_s107
                                 + 0.683105 w_k1_MAT_s107
                                 + 0.863726 w_k1_MSFT_s107
                                 + 0.976062 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.05669 w_k1_ADM_s107
                                 + 0.960799 w_k1_AXP_s107
                                 + 0.923574 w_k1_DIS_s107
                                 + 0.931279 w_k1_FDX_s107
                                 + 1.12245 w_k1_IBM_s107
                                 + 0.798725 w_k1_INTC_s107
                                 + 1.0319 w_k1_KO_s107 + 1.06342 w_k1_MAT_s107
                                 + 0.870397 w_k1_MSFT_s107
                                 + 1.01189 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.833177 w_k1_ADM_s107
                                 + 0.743894 w_k1_AXP_s107
                                 + 0.710882 w_k1_DIS_s107
                                 + 0.831957 w_k1_FDX_s107
                                 + 0.742722 w_k1_IBM_s107
                                 + 0.784644 w_k1_INTC_s107
                                 + 0.657258 w_k1_KO_s107
                                 + 0.518899 w_k1_MAT_s107
                                 + 0.815642 w_k1_MSFT_s107
                                 + 0.835051 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.19693 w_k1_ADM_s107
                                 + 1.48524 w_k1_AXP_s107
                                 + 1.12687 w_k1_DIS_s107
                                 + 1.24453 w_k1_FDX_s107
                                 + 1.09554 w_k1_IBM_s107
                                 + 1.09603 w_k1_INTC_s107
                                 + 1.11802 w_k1_KO_s107 + 1.21641 w_k1_MAT_s107
                                 + 1.08773 w_k1_MSFT_s107
                                 + 1.01671 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.93155 w_k1_ADM_s107
                                 + 0.701532 w_k1_AXP_s107
                                 + 0.795288 w_k1_DIS_s107
                                 + 0.759785 w_k1_FDX_s107
                                 + 1.01997 w_k1_IBM_s107
                                 + 0.693406 w_k1_INTC_s107
                                 + 1.01811 w_k1_KO_s107 + 1.02539 w_k1_MAT_s107
                                 + 0.841799 w_k1_MSFT_s107
                                 + 0.888459 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.24493 w_k1_ADM_s107
                                 + 1.34046 w_k1_AXP_s107
                                 + 1.83752 w_k1_DIS_s107
                                 + 1.25535 w_k1_FDX_s107
                                 + 1.49059 w_k1_IBM_s107
                                 + 1.0835 w_k1_INTC_s107 + 1.44266 w_k1_KO_s107
                                 + 1.62759 w_k1_MAT_s107
                                 + 1.36191 w_k1_MSFT_s107
                                 + 1.09463 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.781185 w_k1_ADM_s107
                                 + 1.14828 w_k1_AXP_s107
                                 + 1.03722 w_k1_DIS_s107
                                 + 1.00498 w_k1_FDX_s107
                                 + 1.50505 w_k1_IBM_s107
                                 + 1.00332 w_k1_INTC_s107
                                 + 1.21643 w_k1_KO_s107 + 1.2546 w_k1_MAT_s107
                                 + 1.01692 w_k1_MSFT_s107
                                 + 1.08077 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.946516 w_k1_ADM_s107
                                 + 1.20787 w_k1_AXP_s107
                                 + 1.49237 w_k1_DIS_s107
                                 + 1.28133 w_k1_FDX_s107
                                 + 0.863215 w_k1_IBM_s107
                                 + 1.36985 w_k1_INTC_s107 + 1.2586 w_k1_KO_s107
                                 + 0.680027 w_k1_MAT_s107
                                 + 1.27427 w_k1_MSFT_s107
                                 + 1.18509 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.992211 w_k1_ADM_s110
                                 + 1.17161 w_k1_AXP_s110
                                 + 0.878607 w_k1_DIS_s110
                                 + 0.994709 w_k1_FDX_s110
                                 + 0.933495 w_k1_IBM_s110
                                 + 1.03949 w_k1_INTC_s110
                                 + 0.864484 w_k1_KO_s110 + 1.21 w_k1_MAT_s110
                                 + 0.892343 w_k1_MSFT_s110
                                 + 0.916769 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.817097 w_k1_ADM_s110
                                 + 0.74675 w_k1_AXP_s110
                                 + 0.705089 w_k1_DIS_s110
                                 + 0.846386 w_k1_FDX_s110
                                 + 0.633739 w_k1_IBM_s110
                                 + 0.994208 w_k1_INTC_s110
                                 + 0.74521 w_k1_KO_s110
                                 + 0.639512 w_k1_MAT_s110
                                 + 0.843078 w_k1_MSFT_s110
                                 + 0.780697 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.09384 w_k1_ADM_s110
                                 + 1.30196 w_k1_AXP_s110
                                 + 1.18278 w_k1_DIS_s110
                                 + 1.06502 w_k1_FDX_s110
                                 + 1.20907 w_k1_IBM_s110
                                 + 1.2998 w_k1_INTC_s110 + 1.11674 w_k1_KO_s110
                                 + 1.34957 w_k1_MAT_s110
                                 + 1.10826 w_k1_MSFT_s110
                                 + 1.09976 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.927467 w_k1_ADM_s110
                                 + 0.672121 w_k1_AXP_s110
                                 + 0.935171 w_k1_DIS_s110
                                 + 0.835678 w_k1_FDX_s110
                                 + 0.882226 w_k1_IBM_s110
                                 + 1.19562 w_k1_INTC_s110
                                 + 1.08038 w_k1_KO_s110
                                 + 0.550724 w_k1_MAT_s110
                                 + 0.939284 w_k1_MSFT_s110
                                 + 0.926919 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.10016 w_k1_ADM_s110
                                 + 1.06596 w_k1_AXP_s110
                                 + 1.34509 w_k1_DIS_s110
                                 + 1.23763 w_k1_FDX_s110
                                 + 0.932071 w_k1_IBM_s110
                                 + 1.31836 w_k1_INTC_s110
                                 + 1.09396 w_k1_KO_s110 + 1.11834 w_k1_MAT_s110
                                 + 1.15811 w_k1_MSFT_s110
                                 + 1.03749 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.33849 w_k1_ADM_s110
                                 + 1.72678 w_k1_AXP_s110 + 1.6898 w_k1_DIS_s110
                                 + 1.43184 w_k1_FDX_s110
                                 + 1.26077 w_k1_IBM_s110
                                 + 1.54077 w_k1_INTC_s110
                                 + 1.42705 w_k1_KO_s110 + 1.51222 w_k1_MAT_s110
                                 + 1.22111 w_k1_MSFT_s110
                                 + 1.23247 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.695103 w_k1_ADM_s110
                                 + 0.775995 w_k1_AXP_s110
                                 + 0.73389 w_k1_DIS_s110
                                 + 0.759758 w_k1_FDX_s110
                                 + 1.18507 w_k1_IBM_s110
                                 + 1.1309 w_k1_INTC_s110
                                 + 0.916364 w_k1_KO_s110
                                 + 1.33017 w_k1_MAT_s110
                                 + 0.728407 w_k1_MSFT_s110
                                 + 0.855349 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.943761 w_k1_ADM_s110
                                 + 1.16716 w_k1_AXP_s110
                                 + 1.01331 w_k1_DIS_s110
                                 + 0.974793 w_k1_FDX_s110
                                 + 1.63587 w_k1_IBM_s110
                                 + 1.16445 w_k1_INTC_s110
                                 + 1.21088 w_k1_KO_s110 + 1.82087 w_k1_MAT_s110
                                 + 0.957112 w_k1_MSFT_s110
                                 + 0.969279 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.22244 w_k1_ADM_s111
                                 + 1.68615 w_k1_AXP_s111
                                 + 1.44718 w_k1_DIS_s111
                                 + 1.38749 w_k1_FDX_s111
                                 + 1.36997 w_k1_IBM_s111
                                 + 1.26883 w_k1_INTC_s111
                                 + 1.36764 w_k1_KO_s111 + 1.93254 w_k1_MAT_s111
                                 + 1.38278 w_k1_MSFT_s111
                                 + 1.17779 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.816384 w_k1_ADM_s111
                                 + 1.0791 w_k1_AXP_s111
                                 + 0.832318 w_k1_DIS_s111
                                 + 0.802343 w_k1_FDX_s111
                                 + 1.37435 w_k1_IBM_s111
                                 + 1.00386 w_k1_INTC_s111
                                 + 1.14936 w_k1_KO_s111 + 1.78039 w_k1_MAT_s111
                                 + 0.933674 w_k1_MSFT_s111
                                 + 1.06643 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.790687 w_k1_ADM_s111
                                 + 1.07364 w_k1_AXP_s111
                                 + 0.675034 w_k1_DIS_s111
                                 + 0.863552 w_k1_FDX_s111
                                 + 1.06202 w_k1_IBM_s111
                                 + 0.959618 w_k1_INTC_s111
                                 + 0.950601 w_k1_KO_s111
                                 + 1.54654 w_k1_MAT_s111
                                 + 0.785711 w_k1_MSFT_s111
                                 + 0.894196 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.753018 w_k1_ADM_s111
                                 + 0.654934 w_k1_AXP_s111
                                 + 0.799674 w_k1_DIS_s111
                                 + 0.781662 w_k1_FDX_s111
                                 + 0.8022 w_k1_IBM_s111
                                 + 0.85556 w_k1_INTC_s111
                                 + 0.983142 w_k1_KO_s111
                                 + 0.674957 w_k1_MAT_s111
                                 + 0.980055 w_k1_MSFT_s111
                                 + 0.856886 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.815072 w_k1_ADM_s111
                                 + 0.717797 w_k1_AXP_s111
                                 + 0.868479 w_k1_DIS_s111
                                 + 0.978104 w_k1_FDX_s111
                                 + 0.549693 w_k1_IBM_s111
                                 + 0.937388 w_k1_INTC_s111
                                 + 0.786004 w_k1_KO_s111
                                 + 0.566861 w_k1_MAT_s111
                                 + 1.15769 w_k1_MSFT_s111
                                 + 0.878392 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.880589 w_k1_ADM_s111
                                 + 1.0551 w_k1_AXP_s111
                                 + 0.973991 w_k1_DIS_s111
                                 + 0.990012 w_k1_FDX_s111
                                 + 0.981218 w_k1_IBM_s111
                                 + 1.14723 w_k1_INTC_s111
                                 + 1.11355 w_k1_KO_s111 + 1.32042 w_k1_MAT_s111
                                 + 1.07242 w_k1_MSFT_s111
                                 + 0.980594 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.00038 w_k1_ADM_s111
                                 + 1.09536 w_k1_AXP_s111
                                 + 1.20423 w_k1_DIS_s111
                                 + 1.02642 w_k1_FDX_s111
                                 + 1.21439 w_k1_IBM_s111
                                 + 0.969657 w_k1_INTC_s111
                                 + 1.4045 w_k1_KO_s111 + 1.68626 w_k1_MAT_s111
                                 + 1.17 w_k1_MSFT_s111 + 1.10508 w_k1_PG_s111
                                 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.13763 w_k1_ADM_s111
                                 + 1.07794 w_k1_AXP_s111
                                 + 1.32798 w_k1_DIS_s111
                                 + 1.14408 w_k1_FDX_s111
                                 + 0.764919 w_k1_IBM_s111
                                 + 1.20806 w_k1_INTC_s111
                                 + 1.19406 w_k1_KO_s111 + 1.00845 w_k1_MAT_s111
                                 + 1.30764 w_k1_MSFT_s111
                                 + 1.07604 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.996123 w_k1_ADM_s112
                                 + 0.818558 w_k1_AXP_s112
                                 + 0.763195 w_k1_DIS_s112
                                 + 0.950722 w_k1_FDX_s112
                                 + 0.419514 w_k1_IBM_s112
                                 + 0.886988 w_k1_INTC_s112
                                 + 0.688679 w_k1_KO_s112
                                 + 0.55072 w_k1_MAT_s112
                                 + 0.993581 w_k1_MSFT_s112
                                 + 0.790332 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.966477 w_k1_ADM_s112
                                 + 0.977025 w_k1_AXP_s112
                                 + 0.933956 w_k1_DIS_s112
                                 + 1.07697 w_k1_FDX_s112
                                 + 0.809455 w_k1_IBM_s112
                                 + 1.12615 w_k1_INTC_s112
                                 + 0.904777 w_k1_KO_s112
                                 + 1.10358 w_k1_MAT_s112
                                 + 1.22232 w_k1_MSFT_s112
                                 + 0.932291 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.895601 w_k1_ADM_s112
                                 + 0.87483 w_k1_AXP_s112
                                 + 0.634593 w_k1_DIS_s112
                                 + 0.741059 w_k1_FDX_s112
                                 + 0.86515 w_k1_IBM_s112
                                 + 0.882885 w_k1_INTC_s112
                                 + 0.873387 w_k1_KO_s112
                                 + 1.03348 w_k1_MAT_s112
                                 + 0.892239 w_k1_MSFT_s112
                                 + 0.878288 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.993914 w_k1_ADM_s112
                                 + 1.12448 w_k1_AXP_s112
                                 + 0.906533 w_k1_DIS_s112
                                 + 0.989219 w_k1_FDX_s112
                                 + 1.29716 w_k1_IBM_s112
                                 + 0.979086 w_k1_INTC_s112
                                 + 0.960792 w_k1_KO_s112
                                 + 1.68464 w_k1_MAT_s112
                                 + 1.0272 w_k1_MSFT_s112 + 0.90278 w_k1_PG_s112
                                 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.997581 w_k1_ADM_s112
                                 + 0.739617 w_k1_AXP_s112
                                 + 0.979735 w_k1_DIS_s112
                                 + 0.909064 w_k1_FDX_s112
                                 + 0.813705 w_k1_IBM_s112
                                 + 0.989857 w_k1_INTC_s112
                                 + 1.04802 w_k1_KO_s112
                                 + 0.746163 w_k1_MAT_s112
                                 + 1.04169 w_k1_MSFT_s112
                                 + 0.94499 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.05723 w_k1_ADM_s112
                                 + 1.3254 w_k1_AXP_s112 + 1.75715 w_k1_DIS_s112
                                 + 1.33744 w_k1_FDX_s112
                                 + 1.10769 w_k1_IBM_s112
                                 + 1.32607 w_k1_INTC_s112
                                 + 1.42502 w_k1_KO_s112 + 1.18913 w_k1_MAT_s112
                                 + 1.46477 w_k1_MSFT_s112
                                 + 1.09679 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.04691 w_k1_ADM_s112
                                 + 1.16312 w_k1_AXP_s112 + 1.2252 w_k1_DIS_s112
                                 + 1.05323 w_k1_FDX_s112
                                 + 1.48184 w_k1_IBM_s112
                                 + 1.08078 w_k1_INTC_s112
                                 + 1.30692 w_k1_KO_s112 + 1.41545 w_k1_MAT_s112
                                 + 1.16757 w_k1_MSFT_s112
                                 + 1.10873 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.781223 w_k1_ADM_s112
                                 + 0.686991 w_k1_AXP_s112
                                 + 0.985107 w_k1_DIS_s112
                                 + 0.867218 w_k1_FDX_s112
                                 + 1.02575 w_k1_IBM_s112
                                 + 0.907399 w_k1_INTC_s112
                                 + 0.985408 w_k1_KO_s112
                                 + 1.18273 w_k1_MAT_s112
                                 + 1.10882 w_k1_MSFT_s112
                                 + 0.947684 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.0354 w_k1_ADM_s113
                                 + 0.823484 w_k1_AXP_s113
                                 + 0.770399 w_k1_DIS_s113
                                 + 0.873896 w_k1_FDX_s113
                                 + 1.16223 w_k1_IBM_s113
                                 + 0.98002 w_k1_INTC_s113
                                 + 1.09587 w_k1_KO_s113 + 1.01356 w_k1_MAT_s113
                                 + 0.801502 w_k1_MSFT_s113
                                 + 0.974178 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.882389 w_k1_ADM_s113
                                 + 0.755898 w_k1_AXP_s113
                                 + 0.877397 w_k1_DIS_s113
                                 + 0.928208 w_k1_FDX_s113 + 0.715 w_k1_IBM_s113
                                 + 1.00599 w_k1_INTC_s113
                                 + 0.815947 w_k1_KO_s113
                                 + 0.571295 w_k1_MAT_s113
                                 + 0.897632 w_k1_MSFT_s113
                                 + 0.86489 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.95666 w_k1_ADM_s113
                                 + 0.988292 w_k1_AXP_s113
                                 + 1.07994 w_k1_DIS_s113
                                 + 1.16237 w_k1_FDX_s113
                                 + 0.975381 w_k1_IBM_s113
                                 + 1.17396 w_k1_INTC_s113
                                 + 0.953426 w_k1_KO_s113
                                 + 0.833734 w_k1_MAT_s113
                                 + 1.05668 w_k1_MSFT_s113
                                 + 0.978016 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.29256 w_k1_ADM_s113
                                 + 1.2239 w_k1_AXP_s113 + 1.02743 w_k1_DIS_s113
                                 + 1.05034 w_k1_FDX_s113
                                 + 1.00614 w_k1_IBM_s113
                                 + 0.906914 w_k1_INTC_s113
                                 + 1.02398 w_k1_KO_s113
                                 + 0.875103 w_k1_MAT_s113
                                 + 0.963497 w_k1_MSFT_s113
                                 + 1.12589 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.911324 w_k1_ADM_s113
                                 + 0.704184 w_k1_AXP_s113
                                 + 0.528878 w_k1_DIS_s113
                                 + 0.717573 w_k1_FDX_s113
                                 + 0.851495 w_k1_IBM_s113
                                 + 0.763615 w_k1_INTC_s113
                                 + 0.773544 w_k1_KO_s113 + 0.7806 w_k1_MAT_s113
                                 + 0.674403 w_k1_MSFT_s113
                                 + 0.843207 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.43371 w_k1_ADM_s113
                                 + 1.19418 w_k1_AXP_s113 + 1.7342 w_k1_DIS_s113
                                 + 1.33919 w_k1_FDX_s113
                                 + 1.25933 w_k1_IBM_s113
                                 + 1.13897 w_k1_INTC_s113
                                 + 1.44408 w_k1_KO_s113
                                 + 0.948781 w_k1_MAT_s113
                                 + 1.33408 w_k1_MSFT_s113 + 1.2062 w_k1_PG_s113
                                 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.08649 w_k1_ADM_s113
                                 + 0.86119 w_k1_AXP_s113 + 1.1335 w_k1_DIS_s113
                                 + 0.938524 w_k1_FDX_s113
                                 + 0.821913 w_k1_IBM_s113
                                 + 1.02099 w_k1_INTC_s113
                                 + 0.917836 w_k1_KO_s113
                                 + 0.658351 w_k1_MAT_s113
                                 + 1.01629 w_k1_MSFT_s113
                                 + 1.01806 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.12415 w_k1_ADM_s113
                                 + 1.19404 w_k1_AXP_s113
                                 + 1.22832 w_k1_DIS_s113
                                 + 1.05614 w_k1_FDX_s113
                                 + 1.41931 w_k1_IBM_s113
                                 + 1.14724 w_k1_INTC_s113
                                 + 1.19785 w_k1_KO_s113 + 1.39893 w_k1_MAT_s113
                                 + 0.963805 w_k1_MSFT_s113
                                 + 1.03194 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.1511 w_k1_ADM_s114
                                 + 1.7506 w_k1_AXP_s114 + 1.40253 w_k1_DIS_s114
                                 + 1.15828 w_k1_FDX_s114
                                 + 1.75692 w_k1_IBM_s114
                                 + 1.30619 w_k1_INTC_s114
                                 + 1.28754 w_k1_KO_s114 + 1.82131 w_k1_MAT_s114
                                 + 1.13294 w_k1_MSFT_s114
                                 + 1.17241 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.31795 w_k1_ADM_s114
                                 + 1.60068 w_k1_AXP_s114 + 1.3621 w_k1_DIS_s114
                                 + 1.37734 w_k1_FDX_s114
                                 + 0.993602 w_k1_IBM_s114
                                 + 1.44723 w_k1_INTC_s114
                                 + 1.11983 w_k1_KO_s114 + 1.25362 w_k1_MAT_s114
                                 + 1.26104 w_k1_MSFT_s114
                                 + 1.12953 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.16372 w_k1_ADM_s114
                                 + 0.988526 w_k1_AXP_s114
                                 + 0.965668 w_k1_DIS_s114
                                 + 1.05283 w_k1_FDX_s114
                                 + 0.763517 w_k1_IBM_s114
                                 + 1.15939 w_k1_INTC_s114 + 1.0232 w_k1_KO_s114
                                 + 0.788392 w_k1_MAT_s114
                                 + 1.0579 w_k1_MSFT_s114 + 1.00751 w_k1_PG_s114
                                 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.10355 w_k1_ADM_s114
                                 + 1.07216 w_k1_AXP_s114
                                 + 0.943392 w_k1_DIS_s114
                                 + 0.954895 w_k1_FDX_s114
                                 + 1.16498 w_k1_IBM_s114
                                 + 1.1234 w_k1_INTC_s114 + 1.1309 w_k1_KO_s114
                                 + 1.12661 w_k1_MAT_s114
                                 + 0.946793 w_k1_MSFT_s114
                                 + 0.93212 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.11518 w_k1_ADM_s114
                                 + 1.14736 w_k1_AXP_s114
                                 + 1.34451 w_k1_DIS_s114
                                 + 1.10767 w_k1_FDX_s114
                                 + 0.994862 w_k1_IBM_s114
                                 + 1.17757 w_k1_INTC_s114
                                 + 1.24097 w_k1_KO_s114 + 1.19677 w_k1_MAT_s114
                                 + 1.28802 w_k1_MSFT_s114
                                 + 1.04363 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.797916 w_k1_ADM_s114
                                 + 0.737853 w_k1_AXP_s114
                                 + 0.680635 w_k1_DIS_s114
                                 + 0.742183 w_k1_FDX_s114
                                 + 0.643047 w_k1_IBM_s114
                                 + 0.822853 w_k1_INTC_s114
                                 + 0.629648 w_k1_KO_s114
                                 + 0.567382 w_k1_MAT_s114
                                 + 0.904004 w_k1_MSFT_s114
                                 + 0.831027 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.995757 w_k1_ADM_s114
                                 + 0.973955 w_k1_AXP_s114
                                 + 0.696816 w_k1_DIS_s114
                                 + 0.761189 w_k1_FDX_s114
                                 + 0.958989 w_k1_IBM_s114
                                 + 0.805539 w_k1_INTC_s114
                                 + 0.929103 w_k1_KO_s114
                                 + 1.17903 w_k1_MAT_s114
                                 + 0.849816 w_k1_MSFT_s114
                                 + 0.844053 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.05845 w_k1_ADM_s114
                                 + 1.49292 w_k1_AXP_s114
                                 + 1.26368 w_k1_DIS_s114
                                 + 1.16434 w_k1_FDX_s114
                                 + 1.29508 w_k1_IBM_s114
                                 + 1.25732 w_k1_INTC_s114 + 1.2158 w_k1_KO_s114
                                 + 1.91463 w_k1_MAT_s114
                                 + 1.25155 w_k1_MSFT_s114
                                 + 1.17434 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.02142 w_k1_ADM_s115
                                 + 0.742601 w_k1_AXP_s115
                                 + 0.659124 w_k1_DIS_s115
                                 + 0.79326 w_k1_FDX_s115
                                 + 0.722288 w_k1_IBM_s115
                                 + 0.816816 w_k1_INTC_s115
                                 + 0.681771 w_k1_KO_s115
                                 + 0.572849 w_k1_MAT_s115
                                 + 0.870478 w_k1_MSFT_s115
                                 + 0.906232 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.99258 w_k1_ADM_s115
                                 + 0.974003 w_k1_AXP_s115
                                 + 1.16318 w_k1_DIS_s115
                                 + 1.09299 w_k1_FDX_s115
                                 + 1.04434 w_k1_IBM_s115
                                 + 0.979808 w_k1_INTC_s115
                                 + 1.03589 w_k1_KO_s115
                                 + 0.941891 w_k1_MAT_s115
                                 + 1.19657 w_k1_MSFT_s115
                                 + 1.05921 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.1968 w_k1_ADM_s115
                                 + 0.848989 w_k1_AXP_s115
                                 + 0.771656 w_k1_DIS_s115
                                 + 0.95636 w_k1_FDX_s115
                                 + 0.80656 w_k1_IBM_s115
                                 + 0.877561 w_k1_INTC_s115
                                 + 0.81278 w_k1_KO_s115
                                 + 0.797536 w_k1_MAT_s115
                                 + 0.943759 w_k1_MSFT_s115
                                 + 0.936888 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.08946 w_k1_ADM_s115
                                 + 1.49353 w_k1_AXP_s115
                                 + 1.74831 w_k1_DIS_s115
                                 + 1.35912 w_k1_FDX_s115
                                 + 1.38766 w_k1_IBM_s115
                                 + 1.26597 w_k1_INTC_s115
                                 + 1.67634 w_k1_KO_s115 + 1.76038 w_k1_MAT_s115
                                 + 1.48251 w_k1_MSFT_s115
                                 + 1.37463 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.13376 w_k1_ADM_s115
                                 + 0.798235 w_k1_AXP_s115
                                 + 1.0584 w_k1_DIS_s115 + 1.04419 w_k1_FDX_s115
                                 + 0.807068 w_k1_IBM_s115
                                 + 1.01833 w_k1_INTC_s115
                                 + 0.979857 w_k1_KO_s115
                                 + 0.655395 w_k1_MAT_s115
                                 + 1.14526 w_k1_MSFT_s115
                                 + 1.03632 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.32802 w_k1_ADM_s115
                                 + 1.04826 w_k1_AXP_s115
                                 + 0.854756 w_k1_DIS_s115
                                 + 0.980417 w_k1_FDX_s115
                                 + 0.989638 w_k1_IBM_s115
                                 + 0.872719 w_k1_INTC_s115
                                 + 0.940519 w_k1_KO_s115
                                 + 1.29186 w_k1_MAT_s115
                                 + 0.935481 w_k1_MSFT_s115
                                 + 0.983276 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.45279 w_k1_ADM_s115
                                 + 1.31046 w_k1_AXP_s115
                                 + 1.07471 w_k1_DIS_s115
                                 + 1.17782 w_k1_FDX_s115
                                 + 0.785861 w_k1_IBM_s115
                                 + 1.00339 w_k1_INTC_s115 + 1.0223 w_k1_KO_s115
                                 + 1.06962 w_k1_MAT_s115
                                 + 1.1888 w_k1_MSFT_s115 + 1.20095 w_k1_PG_s115
                                 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.752286 w_k1_ADM_s115
                                 + 0.791194 w_k1_AXP_s115
                                 + 0.640112 w_k1_DIS_s115
                                 + 0.77113 w_k1_FDX_s115
                                 + 1.13141 w_k1_IBM_s115
                                 + 0.884062 w_k1_INTC_s115
                                 + 0.906547 w_k1_KO_s115
                                 + 1.03085 w_k1_MAT_s115
                                 + 0.802878 w_k1_MSFT_s115
                                 + 0.991778 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.00421 w_k1_ADM_s116
                                 + 1.00846 w_k1_AXP_s116
                                 + 0.902636 w_k1_DIS_s116
                                 + 1.01951 w_k1_FDX_s116
                                 + 0.803026 w_k1_IBM_s116
                                 + 0.878188 w_k1_INTC_s116
                                 + 0.881676 w_k1_KO_s116
                                 + 0.822664 w_k1_MAT_s116
                                 + 0.964069 w_k1_MSFT_s116
                                 + 1.06359 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.01472 w_k1_ADM_s116
                                 + 1.45878 w_k1_AXP_s116
                                 + 1.64208 w_k1_DIS_s116
                                 + 1.17856 w_k1_FDX_s116
                                 + 1.47186 w_k1_IBM_s116
                                 + 1.17219 w_k1_INTC_s116
                                 + 1.37818 w_k1_KO_s116 + 1.36168 w_k1_MAT_s116
                                 + 1.22797 w_k1_MSFT_s116
                                 + 1.26427 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.12931 w_k1_ADM_s116
                                 + 1.41938 w_k1_AXP_s116
                                 + 1.39745 w_k1_DIS_s116
                                 + 1.32732 w_k1_FDX_s116
                                 + 0.896526 w_k1_IBM_s116
                                 + 1.08984 w_k1_INTC_s116
                                 + 0.952016 w_k1_KO_s116
                                 + 1.04277 w_k1_MAT_s116
                                 + 1.26004 w_k1_MSFT_s116
                                 + 1.16529 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.918438 w_k1_ADM_s116
                                 + 0.747831 w_k1_AXP_s116
                                 + 0.828466 w_k1_DIS_s116
                                 + 0.885846 w_k1_FDX_s116
                                 + 0.892634 w_k1_IBM_s116
                                 + 0.768551 w_k1_INTC_s116
                                 + 0.782074 w_k1_KO_s116
                                 + 0.845627 w_k1_MAT_s116
                                 + 0.942483 w_k1_MSFT_s116
                                 + 0.995314 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.94848 w_k1_ADM_s116
                                 + 1.16944 w_k1_AXP_s116
                                 + 0.826397 w_k1_DIS_s116
                                 + 0.900905 w_k1_FDX_s116
                                 + 1.41745 w_k1_IBM_s116
                                 + 1.02758 w_k1_INTC_s116
                                 + 1.07978 w_k1_KO_s116 + 1.3848 w_k1_MAT_s116
                                 + 0.872079 w_k1_MSFT_s116
                                 + 1.10572 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.943196 w_k1_ADM_s116
                                 + 1.02635 w_k1_AXP_s116
                                 + 1.16341 w_k1_DIS_s116
                                 + 0.896914 w_k1_FDX_s116
                                 + 1.35296 w_k1_IBM_s116
                                 + 1.10017 w_k1_INTC_s116
                                 + 1.15096 w_k1_KO_s116
                                 + 0.845337 w_k1_MAT_s116
                                 + 0.958424 w_k1_MSFT_s116
                                 + 1.17839 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.633011 w_k1_ADM_s116
                                 + 0.750106 w_k1_AXP_s116
                                 + 0.611279 w_k1_DIS_s116
                                 + 0.784433 w_k1_FDX_s116
                                 + 0.755831 w_k1_IBM_s116
                                 + 0.941531 w_k1_INTC_s116
                                 + 0.700695 w_k1_KO_s116
                                 + 0.623708 w_k1_MAT_s116
                                 + 0.791941 w_k1_MSFT_s116
                                 + 0.955743 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.780185 w_k1_ADM_s116
                                 + 1.0718 w_k1_AXP_s116 + 1.30104 w_k1_DIS_s116
                                 + 1.0968 w_k1_FDX_s116 + 1.0599 w_k1_IBM_s116
                                 + 1.15836 w_k1_INTC_s116
                                 + 1.07125 w_k1_KO_s116 + 1.17754 w_k1_MAT_s116
                                 + 1.07989 w_k1_MSFT_s116 + 1.0616 w_k1_PG_s116
                                 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.89299 w_k1_ADM_s117
                                 + 1.40244 w_k1_AXP_s117
                                 + 1.65805 w_k1_DIS_s117
                                 + 1.29043 w_k1_FDX_s117
                                 + 1.70394 w_k1_IBM_s117
                                 + 1.4309 w_k1_INTC_s117 + 1.64832 w_k1_KO_s117
                                 + 1.7018 w_k1_MAT_s117
                                 + 1.48876 w_k1_MSFT_s117
                                 + 1.25603 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.28555 w_k1_ADM_s117
                                 + 1.20788 w_k1_AXP_s117
                                 + 1.48212 w_k1_DIS_s117
                                 + 1.18611 w_k1_FDX_s117
                                 + 1.18101 w_k1_IBM_s117
                                 + 1.19921 w_k1_INTC_s117
                                 + 1.37619 w_k1_KO_s117 + 1.05495 w_k1_MAT_s117
                                 + 1.41938 w_k1_MSFT_s117
                                 + 1.11561 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.892279 w_k1_ADM_s117
                                 + 0.887177 w_k1_AXP_s117
                                 + 0.70887 w_k1_DIS_s117
                                 + 0.772631 w_k1_FDX_s117
                                 + 1.00014 w_k1_IBM_s117
                                 + 0.83202 w_k1_INTC_s117
                                 + 0.877945 w_k1_KO_s117
                                 + 0.701786 w_k1_MAT_s117
                                 + 0.871067 w_k1_MSFT_s117
                                 + 0.844379 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.979379 w_k1_ADM_s117
                                 + 0.995357 w_k1_AXP_s117
                                 + 1.04021 w_k1_DIS_s117
                                 + 1.15974 w_k1_FDX_s117 + 0.8044 w_k1_IBM_s117
                                 + 1.53243 w_k1_INTC_s117 + 1.0881 w_k1_KO_s117
                                 + 0.571806 w_k1_MAT_s117
                                 + 1.13993 w_k1_MSFT_s117
                                 + 0.996801 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.965654 w_k1_ADM_s117
                                 + 0.920758 w_k1_AXP_s117
                                 + 1.18447 w_k1_DIS_s117
                                 + 0.996123 w_k1_FDX_s117
                                 + 1.1729 w_k1_IBM_s117
                                 + 1.19633 w_k1_INTC_s117
                                 + 1.33265 w_k1_KO_s117
                                 + 0.806495 w_k1_MAT_s117
                                 + 1.30587 w_k1_MSFT_s117 + 1.0844 w_k1_PG_s117
                                 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.60709 w_k1_ADM_s117
                                 + 1.1442 w_k1_AXP_s117
                                 + 0.961083 w_k1_DIS_s117
                                 + 1.04219 w_k1_FDX_s117
                                 + 1.17453 w_k1_IBM_s117
                                 + 1.03013 w_k1_INTC_s117
                                 + 1.19837 w_k1_KO_s117
                                 + 0.937562 w_k1_MAT_s117
                                 + 1.05598 w_k1_MSFT_s117
                                 + 1.06622 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.05327 w_k1_ADM_s117
                                 + 1.13204 w_k1_AXP_s117
                                 + 0.978358 w_k1_DIS_s117
                                 + 1.00617 w_k1_FDX_s117
                                 + 1.08055 w_k1_IBM_s117
                                 + 1.05544 w_k1_INTC_s117
                                 + 1.25566 w_k1_KO_s117 + 1.17629 w_k1_MAT_s117
                                 + 1.11177 w_k1_MSFT_s117
                                 + 1.12488 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.987763 w_k1_ADM_s117
                                 + 0.742433 w_k1_AXP_s117
                                 + 0.889789 w_k1_DIS_s117
                                 + 0.903633 w_k1_FDX_s117
                                 + 0.823744 w_k1_IBM_s117
                                 + 0.882299 w_k1_INTC_s117
                                 + 0.93624 w_k1_KO_s117
                                 + 0.530474 w_k1_MAT_s117
                                 + 1.08227 w_k1_MSFT_s117
                                 + 0.888382 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.043408025902834 w_k1_ADM_s000
                                 + 0.0160165536470367 w_k1_ADM_s001
                                 + 0.0279420587126941 w_k1_ADM_s002
                                 + 0.0308609004357986 w_k1_ADM_s003
                                 + 0.0305184289110752 w_k1_ADM_s004
                                 + 0.029366253936801 w_k1_ADM_s005
                                 + 0.0208485810969858 w_k1_ADM_s006
                                 + 0.0201674147401387 w_k1_ADM_s007
                                 + 0.0314126122095829 w_k1_ADM_s010
                                 + 0.0260747444543982 w_k1_ADM_s011
                                 + 0.0191393067985281 w_k1_ADM_s012
                                 + 0.0251187387895495 w_k1_ADM_s013
                                 + 0.0207966737639244 w_k1_ADM_s014
                                 + 0.0108063322612437 w_k1_ADM_s015
                                 + 0.0176586691956859 w_k1_ADM_s016
                                 + 0.0292110825499435 w_k1_ADM_s017
                                 + 0.0194359503066551 w_k1_ADM_s100
                                 + 0.066458784791246 w_k1_ADM_s101
                                 + 0.0165914648473152 w_k1_ADM_s102
                                 + 0.0883947974320707 w_k1_ADM_s103
                                 + 0.019582846156949 w_k1_ADM_s104
                                 + 0.0197746819279699 w_k1_ADM_s105
                                 + 0.0222682737303645 w_k1_ADM_s106
                                 + 0.0462994377435442 w_k1_ADM_s107
                                 + 0.086709352519869 w_k1_ADM_s110
                                 + 0.0312848718495589 w_k1_ADM_s111
                                 + 0.0358387660565941 w_k1_ADM_s112
                                 + 0.0291002837474639 w_k1_ADM_s113
                                 + 0.0425517422622634 w_k1_ADM_s114
                                 + 0.0300699905674595 w_k1_ADM_s115
                                 + 0.0328221870850245 w_k1_ADM_s116
                                 + 0.030935535043486 w_k1_ADM_s117
                                 + 0.0403935617076876 w_k1_AXP_s000
                                 + 0.0170163429117775 w_k1_AXP_s001
                                 + 0.0333721708251902 w_k1_AXP_s002
                                 + 0.0359194608641858 w_k1_AXP_s003
                                 + 0.029835229362632 w_k1_AXP_s004
                                 + 0.0328050229507983 w_k1_AXP_s005
                                 + 0.0225262068535273 w_k1_AXP_s006
                                 + 0.0188337961124154 w_k1_AXP_s007
                                 + 0.0268638797226329 w_k1_AXP_s010
                                 + 0.0204815377087931 w_k1_AXP_s011
                                 + 0.0212866675744491 w_k1_AXP_s012
                                 + 0.0215526612681063 w_k1_AXP_s013
                                 + 0.0180527890592796 w_k1_AXP_s014
                                 + 0.011440059831245 w_k1_AXP_s015
                                 + 0.0158308273061123 w_k1_AXP_s016
                                 + 0.0356795228976465 w_k1_AXP_s017
                                 + 0.0191003895895571 w_k1_AXP_s100
                                 + 0.071461679295766 w_k1_AXP_s101
                                 + 0.0177742088602509 w_k1_AXP_s102
                                 + 0.0816664245798468 w_k1_AXP_s103
                                 + 0.0248096732081185 w_k1_AXP_s104
                                 + 0.0194658254894248 w_k1_AXP_s105
                                 + 0.0270460108550881 w_k1_AXP_s106
                                 + 0.0471626042815046 w_k1_AXP_s107
                                 + 0.0909888563070239 w_k1_AXP_s110
                                 + 0.0354331187028303 w_k1_AXP_s111
                                 + 0.0372748969251008 w_k1_AXP_s112
                                 + 0.0265427125520909 w_k1_AXP_s113
                                 + 0.044807515440462 w_k1_AXP_s114
                                 + 0.028459255904559 w_k1_AXP_s115
                                 + 0.0387272713274277 w_k1_AXP_s116
                                 + 0.0292919317326851 w_k1_AXP_s117
                                 + 0.0445288597572168 w_k1_DIS_s000
                                 + 0.0159214866801114 w_k1_DIS_s001
                                 + 0.0330966909281463 w_k1_DIS_s002
                                 + 0.031853908056913 w_k1_DIS_s003
                                 + 0.0317289043511406 w_k1_DIS_s004
                                 + 0.0349451302184714 w_k1_DIS_s005
                                 + 0.0202604834486698 w_k1_DIS_s006
                                 + 0.0180978248246339 w_k1_DIS_s007
                                 + 0.0275256570202866 w_k1_DIS_s010
                                 + 0.0187215618099295 w_k1_DIS_s011
                                 + 0.0202570818664639 w_k1_DIS_s012
                                 + 0.0215820363268467 w_k1_DIS_s013
                                 + 0.0177174971039539 w_k1_DIS_s014
                                 + 0.0105065117353002 w_k1_DIS_s015
                                 + 0.0162901821663776 w_k1_DIS_s016
                                 + 0.0349382514001862 w_k1_DIS_s017
                                 + 0.0170437329028721 w_k1_DIS_s100
                                 + 0.0743085137406545 w_k1_DIS_s101
                                 + 0.0191211806859702 w_k1_DIS_s102
                                 + 0.0844605325090532 w_k1_DIS_s103
                                 + 0.0242213569531181 w_k1_DIS_s104
                                 + 0.0208886473298259 w_k1_DIS_s105
                                 + 0.0249955549485361 w_k1_DIS_s106
                                 + 0.0469212765473445 w_k1_DIS_s107
                                 + 0.0912855337999179 w_k1_DIS_s110
                                 + 0.0350317694917524 w_k1_DIS_s111
                                 + 0.0398768909980972 w_k1_DIS_s112
                                 + 0.029332074490704 w_k1_DIS_s113
                                 + 0.0393855007753764 w_k1_DIS_s114
                                 + 0.029073032685193 w_k1_DIS_s115
                                 + 0.0385352508970382 w_k1_DIS_s116
                                 + 0.0309529905444348 w_k1_DIS_s117
                                 + 0.0430048917943462 w_k1_FDX_s000
                                 + 0.0162609724229533 w_k1_FDX_s001
                                 + 0.0320237111870676 w_k1_FDX_s002
                                 + 0.0325459057384368 w_k1_FDX_s003
                                 + 0.0297206988997281 w_k1_FDX_s004
                                 + 0.0344094971821665 w_k1_FDX_s005
                                 + 0.0202269447369053 w_k1_FDX_s006
                                 + 0.0185672520742699 w_k1_FDX_s007
                                 + 0.0270787107439117 w_k1_FDX_s010
                                 + 0.0184830311776372 w_k1_FDX_s011
                                 + 0.0203383814183003 w_k1_FDX_s012
                                 + 0.0211370856788583 w_k1_FDX_s013
                                 + 0.0181038625973986 w_k1_FDX_s014
                                 + 0.0106631802955232 w_k1_FDX_s015
                                 + 0.0154881595431903 w_k1_FDX_s016
                                 + 0.0351353214662642 w_k1_FDX_s017
                                 + 0.0186593268345135 w_k1_FDX_s100
                                 + 0.069518449042811 w_k1_FDX_s101
                                 + 0.0165574671055069 w_k1_FDX_s102
                                 + 0.0825312455860365 w_k1_FDX_s103
                                 + 0.0228227485539366 w_k1_FDX_s104
                                 + 0.0198533850222337 w_k1_FDX_s105
                                 + 0.0243047382429514 w_k1_FDX_s106
                                 + 0.0471000784034067 w_k1_FDX_s107
                                 + 0.0892676965959176 w_k1_FDX_s110
                                 + 0.0335559663168486 w_k1_FDX_s111
                                 + 0.037373876147551 w_k1_FDX_s112
                                 + 0.0284536644984328 w_k1_FDX_s113
                                 + 0.0388254168341807 w_k1_FDX_s114
                                 + 0.0288036594647112 w_k1_FDX_s115
                                 + 0.0363984804621397 w_k1_FDX_s116
                                 + 0.0292601957121124 w_k1_FDX_s117
                                 + 0.0456750550145912 w_k1_IBM_s000
                                 + 0.0171902859351286 w_k1_IBM_s001
                                 + 0.0318636220367824 w_k1_IBM_s002
                                 + 0.0332557880955862 w_k1_IBM_s003
                                 + 0.0294256478617836 w_k1_IBM_s004
                                 + 0.0338013579987702 w_k1_IBM_s005
                                 + 0.0206382392554965 w_k1_IBM_s006
                                 + 0.0182699063317635 w_k1_IBM_s007
                                 + 0.0274987923025545 w_k1_IBM_s010
                                 + 0.018611464100244 w_k1_IBM_s011
                                 + 0.0206147298606024 w_k1_IBM_s012
                                 + 0.0213539652090958 w_k1_IBM_s013
                                 + 0.0183466813849185 w_k1_IBM_s014
                                 + 0.0105712064810568 w_k1_IBM_s015
                                 + 0.0152636811940421 w_k1_IBM_s016
                                 + 0.0347678913105884 w_k1_IBM_s017
                                 + 0.0185793980280766 w_k1_IBM_s100
                                 + 0.0694514053460295 w_k1_IBM_s101
                                 + 0.0171410522049816 w_k1_IBM_s102
                                 + 0.0844241041897953 w_k1_IBM_s103
                                 + 0.0227765716502721 w_k1_IBM_s104
                                 + 0.020871903839717 w_k1_IBM_s105
                                 + 0.0249686345085503 w_k1_IBM_s106
                                 + 0.0442913032147313 w_k1_IBM_s107
                                 + 0.088054197846822 w_k1_IBM_s110
                                 + 0.033143439373645 w_k1_IBM_s111
                                 + 0.038324228921354 w_k1_IBM_s112
                                 + 0.0278038598713347 w_k1_IBM_s113
                                 + 0.0410102761338162 w_k1_IBM_s114
                                 + 0.0285179790633373 w_k1_IBM_s115
                                 + 0.037376257189228 w_k1_IBM_s116
                                 + 0.0312182995981298 w_k1_IBM_s117
                                 + 0.0439597216356518 w_k1_INTC_s000
                                 + 0.0167437540859846 w_k1_INTC_s001
                                 + 0.0351469959982274 w_k1_INTC_s002
                                 + 0.0341653863408517 w_k1_INTC_s003
                                 + 0.0330870222346145 w_k1_INTC_s004
                                 + 0.0333681894105465 w_k1_INTC_s005
                                 + 0.0223172401068581 w_k1_INTC_s006
                                 + 0.0148769880758014 w_k1_INTC_s007
                                 + 0.0271236166658706 w_k1_INTC_s010
                                 + 0.0180300132346692 w_k1_INTC_s011
                                 + 0.0179633883085561 w_k1_INTC_s012
                                 + 0.0198578795481192 w_k1_INTC_s013
                                 + 0.0206976427978535 w_k1_INTC_s014
                                 + 0.0113840777927952 w_k1_INTC_s015
                                 + 0.0160296522816651 w_k1_INTC_s016
                                 + 0.0368643074959092 w_k1_INTC_s017
                                 + 0.01795682754084 w_k1_INTC_s100
                                 + 0.0783654493268392 w_k1_INTC_s101
                                 + 0.0170156451281289 w_k1_INTC_s102
                                 + 0.0874796843590858 w_k1_INTC_s103
                                 + 0.0202791222253048 w_k1_INTC_s104
                                 + 0.0202311023680872 w_k1_INTC_s105
                                 + 0.0228571074548004 w_k1_INTC_s106
                                 + 0.0418664940430945 w_k1_INTC_s107
                                 + 0.105045560558331 w_k1_INTC_s110
                                 + 0.0351638388396803 w_k1_INTC_s111
                                 + 0.0385447915145639 w_k1_INTC_s112
                                 + 0.0287095366821802 w_k1_INTC_s113
                                 + 0.042766660578903 w_k1_INTC_s114
                                 + 0.0273213021588223 w_k1_INTC_s115
                                 + 0.0360384576767027 w_k1_INTC_s116
                                 + 0.031236276175049 w_k1_INTC_s117
                                 + 0.0396227466065641 w_k1_KO_s000
                                 + 0.0163741047181433 w_k1_KO_s001
                                 + 0.0326573635375372 w_k1_KO_s002
                                 + 0.0326261840927195 w_k1_KO_s003
                                 + 0.0320887313148263 w_k1_KO_s004
                                 + 0.03217287981247 w_k1_KO_s005
                                 + 0.0199784041300884 w_k1_KO_s006
                                 + 0.0177121619483382 w_k1_KO_s007
                                 + 0.0267515481456117 w_k1_KO_s010
                                 + 0.0178993811170641 w_k1_KO_s011
                                 + 0.0198311192709436 w_k1_KO_s012
                                 + 0.020811391868667 w_k1_KO_s013
                                 + 0.0184740646237593 w_k1_KO_s014
                                 + 0.0100216721802207 w_k1_KO_s015
                                 + 0.0164588298916366 w_k1_KO_s016
                                 + 0.0349306803181258 w_k1_KO_s017
                                 + 0.0186527045096122 w_k1_KO_s100
                                 + 0.0738737619215553 w_k1_KO_s101
                                 + 0.0171340971948195 w_k1_KO_s102
                                 + 0.090805656987488 w_k1_KO_s103
                                 + 0.0227568442146337 w_k1_KO_s104
                                 + 0.0226287519569409 w_k1_KO_s105
                                 + 0.0245347483156694 w_k1_KO_s106
                                 + 0.0462804965242604 w_k1_KO_s107
                                 + 0.0899330039880158 w_k1_KO_s110
                                 + 0.03784936602158 w_k1_KO_s111
                                 + 0.0393710065302528 w_k1_KO_s112
                                 + 0.0279609163073648 w_k1_KO_s113
                                 + 0.0417498094795121 w_k1_KO_s114
                                 + 0.0293751569572608 w_k1_KO_s115
                                 + 0.0352106811136827 w_k1_KO_s116
                                 + 0.0338976221879538 w_k1_KO_s117
                                 + 0.0444117652118314 w_k1_MAT_s000
                                 + 0.0207787233056427 w_k1_MAT_s001
                                 + 0.0296555425278043 w_k1_MAT_s002
                                 + 0.0321373799661119 w_k1_MAT_s003
                                 + 0.024755670112845 w_k1_MAT_s004
                                 + 0.0324771198738253 w_k1_MAT_s005
                                 + 0.0236279915980485 w_k1_MAT_s006
                                 + 0.0198230109909935 w_k1_MAT_s007
                                 + 0.0243134552694752 w_k1_MAT_s010
                                 + 0.0177131941637916 w_k1_MAT_s011
                                 + 0.0237158226479499 w_k1_MAT_s012
                                 + 0.0197851003751885 w_k1_MAT_s013
                                 + 0.0153081730951305 w_k1_MAT_s014
                                 + 0.00918159616061902 w_k1_MAT_s015
                                 + 0.0171250334717941 w_k1_MAT_s016
                                 + 0.0424383466271412 w_k1_MAT_s017
                                 + 0.0196964508594224 w_k1_MAT_s100
                                 + 0.0652255516972947 w_k1_MAT_s101
                                 + 0.0165785744252148 w_k1_MAT_s102
                                 + 0.0851627722907849 w_k1_MAT_s103
                                 + 0.0236397181862579 w_k1_MAT_s104
                                 + 0.0216658450226785 w_k1_MAT_s105
                                 + 0.0231730343186655 w_k1_MAT_s106
                                 + 0.0413840407404366 w_k1_MAT_s107
                                 + 0.0954906344506136 w_k1_MAT_s110
                                 + 0.0436277722938429 w_k1_MAT_s111
                                 + 0.0433822044122936 w_k1_MAT_s112
                                 + 0.0238408788588036 w_k1_MAT_s113
                                 + 0.0462376166608921 w_k1_MAT_s114
                                 + 0.0301443928395477 w_k1_MAT_s115
                                 + 0.0359353866928378 w_k1_MAT_s116
                                 + 0.0254938621012473 w_k1_MAT_s117
                                 + 0.0434076461650612 w_k1_MSFT_s000
                                 + 0.0172534164964444 w_k1_MSFT_s001
                                 + 0.0308992259541782 w_k1_MSFT_s002
                                 + 0.033821565794285 w_k1_MSFT_s003
                                 + 0.0302419015272728 w_k1_MSFT_s004
                                 + 0.0336781221765279 w_k1_MSFT_s005
                                 + 0.0210277023078887 w_k1_MSFT_s006
                                 + 0.015715516006457 w_k1_MSFT_s007
                                 + 0.0247109095952871 w_k1_MSFT_s010
                                 + 0.018325223945812 w_k1_MSFT_s011
                                 + 0.0192954331605385 w_k1_MSFT_s012
                                 + 0.0212845559845741 w_k1_MSFT_s013
                                 + 0.0202920573113705 w_k1_MSFT_s014
                                 + 0.0106238774813485 w_k1_MSFT_s015
                                 + 0.0164419793990038 w_k1_MSFT_s016
                                 + 0.034279498487523 w_k1_MSFT_s017
                                 + 0.0177581581071676 w_k1_MSFT_s100
                                 + 0.0700643305783595 w_k1_MSFT_s101
                                 + 0.0177885630659935 w_k1_MSFT_s102
                                 + 0.0824683652751478 w_k1_MSFT_s103
                                 + 0.0210889568173069 w_k1_MSFT_s104
                                 + 0.0200143315825459 w_k1_MSFT_s105
                                 + 0.0237223191056331 w_k1_MSFT_s106
                                 + 0.0438927221247415 w_k1_MSFT_s107
                                 + 0.0865319560075039 w_k1_MSFT_s110
                                 + 0.0369357696840812 w_k1_MSFT_s111
                                 + 0.0416632344837098 w_k1_MSFT_s112
                                 + 0.0269551922420926 w_k1_MSFT_s113
                                 + 0.0406727252410313 w_k1_MSFT_s114
                                 + 0.0304142059600053 w_k1_MSFT_s115
                                 + 0.0361391316382664 w_k1_MSFT_s116
                                 + 0.0338706333430901 w_k1_MSFT_s117
                                 + 0.0402236203853642 w_k1_PG_s000
                                 + 0.0174608521352719 w_k1_PG_s001
                                 + 0.0307537755030136 w_k1_PG_s002
                                 + 0.0313422859705 w_k1_PG_s003
                                 + 0.0284185314009927 w_k1_PG_s004
                                 + 0.0310684432514727 w_k1_PG_s005
                                 + 0.0202859934912311 w_k1_PG_s006
                                 + 0.0167605938068631 w_k1_PG_s007
                                 + 0.0244446678910564 w_k1_PG_s010
                                 + 0.0180076898155658 w_k1_PG_s011
                                 + 0.0213204450785301 w_k1_PG_s012
                                 + 0.0220160285346288 w_k1_PG_s013
                                 + 0.0184578212947548 w_k1_PG_s014
                                 + 0.0102050488953642 w_k1_PG_s015
                                 + 0.0158940858252184 w_k1_PG_s016
                                 + 0.0338221949226298 w_k1_PG_s017
                                 + 0.0179489933404677 w_k1_PG_s100
                                 + 0.0650063740310646 w_k1_PG_s101
                                 + 0.0171752189033903 w_k1_PG_s102
                                 + 0.0814597891130589 w_k1_PG_s103
                                 + 0.0227973817824969 w_k1_PG_s104
                                 + 0.0212013498638088 w_k1_PG_s105
                                 + 0.0242075516880367 w_k1_PG_s106
                                 + 0.045028115694466 w_k1_PG_s107
                                 + 0.0846225366790214 w_k1_PG_s110
                                 + 0.0333393570909912 w_k1_PG_s111
                                 + 0.0353775780384728 w_k1_PG_s112
                                 + 0.0272397528423633 w_k1_PG_s113
                                 + 0.0384626058766408 w_k1_PG_s114
                                 + 0.0299942930226765 w_k1_PG_s115
                                 + 0.0389096846021851 w_k1_PG_s116
                                 + 0.029763758264092 w_k1_PG_s117 >= 1.05
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

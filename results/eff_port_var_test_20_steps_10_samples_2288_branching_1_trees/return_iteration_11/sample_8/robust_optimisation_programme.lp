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
 w_k1_ADM_s0_return_constraint:  1.05389 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.836032 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.95811 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.04806 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.994682 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.980578 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.983519 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.24872 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.914295 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.900502 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.626924 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.996956 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.803842 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.74775 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.834358 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.05725 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.5243 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.760121 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.0688 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.03731 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.26321 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.03262 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.860677 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.82101 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.01654 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.854602 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.13533 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.13883 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.844285 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.17621 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.898656 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.856337 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.14898 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.829653 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.46652 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.735574 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.37636 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.10912 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.16128 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.777524 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.788963 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.12775 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.22694 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.839961 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.17237 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.24097 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.846603 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.81246 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.653573 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.09406 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.01862 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.69955 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.907895 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.810712 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.52942 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.653176 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.11623 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.00462 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.38926 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.20769 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.509515 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.61279 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.844247 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.788832 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.2345 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.901043 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.714638 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.09555 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.778336 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.10657 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.10145 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.822648 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.60945 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.709715 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.4038 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.4042 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.16822 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.745222 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.734961 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.17062 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.17585 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.844883 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.30525 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.45862 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.14209 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.950682 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.646918 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.985888 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.912245 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.690723 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.777697 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.795611 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.6623 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.685348 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.11627 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.842656 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.63557 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.21647 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.614339 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 2.05971 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.921507 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.819094 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.39631 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.868394 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.56754 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.02387 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.701409 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.878045 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.05779 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.06843 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.69109 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.63171 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.70267 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.21702 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.10523 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.842982 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.84038 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.10157 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.04912 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.955125 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.09063 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.34398 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.02759 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.01543 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.715253 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.08905 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.92681 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.854164 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.779281 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.963095 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.52371 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.74682 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.19575 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.819556 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.25391 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.05166 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.690374 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.59352 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.783139 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.871739 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.12095 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.0637 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.705526 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.14454 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.850951 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.885879 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.07427 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.846874 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.30123 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.716341 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.44832 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.0228 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.14431 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.863943 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.703732 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.13175 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.20932 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.748637 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.57824 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.866622 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.986253 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.688668 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.980108 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.87419 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.13142 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.665323 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.04282 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.604957 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.974633 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.759156 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.739892 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.55539 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.34114 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.47334 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.705697 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.50094 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.2845 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.858945 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.31766 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.659521 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.09716 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.02735 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.593629 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.08283 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.803403 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.17321 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.30192 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.899501 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.808103 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.36227 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.07454 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.841876 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.838913 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.13821 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.19408 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.966783 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.19011 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.21629 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.08772 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.931556 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.880644 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.09368 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.02995 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.865717 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.725737 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.938594 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.30968 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.796761 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.978341 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.88148 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.01276 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.917212 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.707503 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.31909 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.920438 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.80065 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.07396 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.929858 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.766917 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.907333 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.842772 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.961207 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.12024 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.953806 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.15262 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.734618 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.23837 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.03585 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.03204 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.867959 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.897333 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.1046 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.14446 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.778057 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.14761 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.11679 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.06451 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.760034 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.749154 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.893451 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.906342 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.650477 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.962961 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.838494 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.36391 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.750921 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.03704 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.18571 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.47593 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.36555 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.740902 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.73576 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.06495 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.868665 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.20088 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.857069 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.805561 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.979687 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.669888 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.880753 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.992774 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.12767 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.41682 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.815084 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.22711 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.39349 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.063 w_0_MAT - w_k1_MAT_s0 + 0.99 b_k1_MAT_s0
                                 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.886845 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.700308 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.40274 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.19194 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.669876 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.5894 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.08539 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.08115 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.82393 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.950306 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.11165 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.36123 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.605574 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.950784 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.468112 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.15238 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.585002 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.70478 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.06036 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.45358 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.36241 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.496303 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.48194 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.17323 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.800469 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.23698 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.647961 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.919247 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.09362 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.531358 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.24847 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.10424 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.02063 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.72594 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.873581 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.842697 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.55847 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.0749 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.846919 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.867603 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.10095 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.09811 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.924967 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.12606 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.25445 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.11293 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.964802 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.775543 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.08786 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.919822 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.866744 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.865172 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.90325 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.41726 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.759511 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.06462 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.881137 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.2909 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.06801 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.734726 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.54445 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.947724 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.883148 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.1531 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.995814 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.73451 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.04034 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.886715 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.908549 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.12896 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.03289 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.36841 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.747789 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.36106 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.07968 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.07904 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.87918 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.86636 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.09315 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.06047 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.93638 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.14509 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.07185 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.990184 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.908657 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.789216 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.939335 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.936068 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.880836 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.86762 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.880997 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.21921 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.78851 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.00109 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.960621 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.26838 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.03493 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.778503 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.28786 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.91847 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.929873 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.18686 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.943221 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.847813 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.01178 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.802424 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.891071 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.999864 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.964662 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.2054 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.781117 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.2118 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.08991 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.455999599024678 z_k1_s0000
                                 - 0.284000086969796 z_k1_s0001
                                 - 0.347999693992517 z_k1_s0002
                                 - 0.419999630680624 z_k1_s0003
                                 - 0.397999650025925 z_k1_s0004
                                 - 0.696999387105703 z_k1_s0005
                                 - 0.175000182816967 z_k1_s0006
                                 - 0.588999482073542 z_k1_s0007
                                 - 0.440001196448873 z_k1_s0010
                                 - 0.359000796129859 z_k1_s0011
                                 - 1.03600412002576 z_k1_s0012
                                 - 0.532999743787918 z_k1_s0013
                                 - 0.473000106664449 z_k1_s0014
                                 - 0.57799947163052 z_k1_s0015
                                 - 0.465001638718539 z_k1_s0016
                                 - 0.567001021828436 z_k1_s0017
                                 - 0.089000123368751 z_k1_s0020
                                 - 0.0990000453471854 z_k1_s0021
                                 - 0.168999929596587 z_k1_s0022
                                 - 0.136000057947645 z_k1_s0023
                                 - 0.299000206517315 z_k1_s0024
                                 - 0.0770000233144675 z_k1_s0025
                                 - 0.107000112050041 z_k1_s0026
                                 - 0.100000080585065 z_k1_s0027
                                 - 0.0750000320721081 z_k1_s0030
                                 - 0.0740000660445153 z_k1_s0031
                                 - 0.177000006890104 z_k1_s0032
                                 - 0.284000241846506 z_k1_s0033
                                 - 0.125000053453514 z_k1_s0034
                                 - 0.0960000711523293 z_k1_s0035
                                 - 0.132000460647326 z_k1_s0036
                                 - 0.112000065094366 z_k1_s0037
                                 - 0.48700006370135 z_k1_s0040
                                 - 0.132999667000952 z_k1_s0041
                                 - 0.178999625849235 z_k1_s0042
                                 - 0.479998551462638 z_k1_s0043
                                 - 0.347998832610677 z_k1_s0044
                                 - 0.26599886520296 z_k1_s0045
                                 - 0.153999750127113 z_k1_s0046
                                 - 0.296998195419552 z_k1_s0047
                                 - 0.0920002179333577 z_k1_s0050
                                 - 0.262999703451586 z_k1_s0051
                                 - 0.120000232034736 z_k1_s0052
                                 - 0.196000098709647 z_k1_s0053
                                 - 0.374999759857098 z_k1_s0054
                                 - 0.389000624909078 z_k1_s0055
                                 - 0.0800000402896519 z_k1_s0056
                                 - 0.201000530228396 z_k1_s0057
                                 - 0.406002112788772 z_k1_s0060
                                 - 0.530001960706478 z_k1_s0061
                                 - 0.580000448974285 z_k1_s0062
                                 - 0.819004013935037 z_k1_s0063
                                 - 0.381002868654868 z_k1_s0064
                                 - 0.470000177771565 z_k1_s0065
                                 - 0.229000989696354 z_k1_s0066
                                 - 0.182000612217844 z_k1_s0067
                                 - 0.202000530177239 z_k1_s0070
                                 - 0.310999820820011 z_k1_s0071
                                 - 0.511000507233504 z_k1_s0072
                                 - 0.35200092848708 z_k1_s0073
                                 - 0.107000145880691 z_k1_s0074
                                 - 0.15500035722004 z_k1_s0075
                                 - 0.443001625256135 z_k1_s0076
                                 - 0.249000703134437 z_k1_s0077
                                 - 1.04699669524229 z_k1_s0100
                                 - 2.25199899753556 z_k1_s0101
                                 - 0.875002470561136 z_k1_s0102
                                 - 0.942001853052471 z_k1_s0103
                                 - 0.665999368732249 z_k1_s0104
                                 - 0.686999941369482 z_k1_s0105
                                 - 0.677999581367863 z_k1_s0106
                                 - 0.873999846017689 z_k1_s0107
                                 - 0.849995261039928 z_k1_s0110
                                 - 0.407998043117769 z_k1_s0111
                                 - 0.33199861016651 z_k1_s0112
                                 - 0.512997893938432 z_k1_s0113
                                 - 0.460998003130901 z_k1_s0114
                                 - 0.640997747394894 z_k1_s0115
                                 - 1.31999600581193 z_k1_s0116
                                 - 0.771997278770486 z_k1_s0117
                                 - 0.496001472317262 z_k1_s0120
                                 - 0.420001854655599 z_k1_s0121
                                 - 0.335999808524884 z_k1_s0122
                                 - 0.442998157106703 z_k1_s0123
                                 - 1.02600112043093 z_k1_s0124
                                 - 0.650999314917038 z_k1_s0125
                                 - 0.243999940724003 z_k1_s0126
                                 - 0.572001089978924 z_k1_s0127
                                 - 0.771000945957525 z_k1_s0130
                                 - 1.38599961004505 z_k1_s0131
                                 - 0.60900064267252 z_k1_s0132
                                 - 1.37400356721504 z_k1_s0133
                                 - 0.375000289872084 z_k1_s0134
                                 - 1.20600524251503 z_k1_s0135
                                 - 0.961003528363001 z_k1_s0136
                                 - 0.993000650921106 z_k1_s0137
                                 - 0.936000187830621 z_k1_s0140
                                 - 0.75399884886566 z_k1_s0141
                                 - 0.792001511469878 z_k1_s0142
                                 - 0.593999668354776 z_k1_s0143
                                 - 0.318999648235253 z_k1_s0144
                                 - 1.28699732777184 z_k1_s0145
                                 - 0.84700104661454 z_k1_s0146
                                 - 0.331999325233426 z_k1_s0147
                                 - 0.38999989348851 z_k1_s0150
                                 - 1.13600057847092 z_k1_s0151
                                 - 0.271999932155062 z_k1_s0152
                                 - 0.338999775396969 z_k1_s0153
                                 - 0.316999839365295 z_k1_s0154
                                 - 0.589999806667241 z_k1_s0155
                                 - 0.672999793659409 z_k1_s0156
                                 - 0.468999321293245 z_k1_s0157
                                 - 0.730003134834464 z_k1_s0160
                                 - 0.461000961057426 z_k1_s0161
                                 - 0.685001497889183 z_k1_s0162
                                 - 0.473001397576168 z_k1_s0163
                                 - 0.813999164448959 z_k1_s0164
                                 - 0.898001049077369 z_k1_s0165
                                 - 1.05800101431334 z_k1_s0166
                                 - 0.736004524096618 z_k1_s0167
                                 - 0.502999256417424 z_k1_s0170
                                 - 1.07599627759948 z_k1_s0171
                                 - 0.934999003981264 z_k1_s0172
                                 - 0.54699949081623 z_k1_s0173
                                 - 0.291999436439883 z_k1_s0174
                                 - 1.01999782670804 z_k1_s0175
                                 - 1.10600078432112 z_k1_s0176
                                 - 0.497999501462713 z_k1_s0177
                                 - 0.0860000204460943 z_k1_s1000
                                 - 0.175999905340883 z_k1_s1001
                                 - 0.188000565886661 z_k1_s1002
                                 - 0.212000552978599 z_k1_s1003
                                 - 0.225000379236781 z_k1_s1004
                                 - 0.156000583097409 z_k1_s1005
                                 - 0.147000087688076 z_k1_s1006
                                 - 0.143999922551632 z_k1_s1007
                                 - 0.211999814882845 z_k1_s1010
                                 - 0.103000065323394 z_k1_s1011
                                 - 0.0740000075478438 z_k1_s1012
                                 - 0.0940000753689357 z_k1_s1013
                                 - 0.265999986409784 z_k1_s1014
                                 - 0.246000266288972 z_k1_s1015
                                 - 0.0860000714205176 z_k1_s1016
                                 - 0.0780000674720995 z_k1_s1017
                                 - 0.304000036829479 z_k1_s1020
                                 - 0.207000752923256 z_k1_s1021
                                 - 0.0889999816685029 z_k1_s1022
                                 - 0.257000079658467 z_k1_s1023
                                 - 0.166999324735374 z_k1_s1024
                                 - 0.186000248974582 z_k1_s1025
                                 - 0.195999786521672 z_k1_s1026
                                 - 0.233000206145594 z_k1_s1027
                                 - 0.521999438946777 z_k1_s1030
                                 - 0.161999705721669 z_k1_s1031
                                 - 0.135999666912201 z_k1_s1032
                                 - 0.156999564235517 z_k1_s1033
                                 - 0.448999712903996 z_k1_s1034
                                 - 0.541998511494553 z_k1_s1035
                                 - 0.3259985719997 z_k1_s1036
                                 - 0.194999544372594 z_k1_s1037
                                 - 0.644001227179495 z_k1_s1040
                                 - 0.315999443323186 z_k1_s1041
                                 - 1.17900036609048 z_k1_s1042
                                 - 0.353999373421548 z_k1_s1043
                                 - 1.13199727779417 z_k1_s1044
                                 - 0.24199950561595 z_k1_s1045
                                 - 0.539997776136284 z_k1_s1046
                                 - 0.268999628222447 z_k1_s1047
                                 - 0.266999951506437 z_k1_s1050
                                 - 0.440999970094886 z_k1_s1051
                                 - 0.395999426167008 z_k1_s1052
                                 - 0.334000463554705 z_k1_s1053
                                 - 0.496000634895627 z_k1_s1054
                                 - 0.178000066937594 z_k1_s1055
                                 - 0.551998929384398 z_k1_s1056
                                 - 0.314000221808981 z_k1_s1057
                                 - 0.102000017010868 z_k1_s1060
                                 - 0.156999518100019 z_k1_s1061
                                 - 0.0800000176751909 z_k1_s1062
                                 - 0.186000245519836 z_k1_s1063
                                 - 0.205999642188583 z_k1_s1064
                                 - 0.0530000034223133 z_k1_s1065
                                 - 0.113999544512107 z_k1_s1066
                                 - 0.206999667272025 z_k1_s1067
                                 - 0.578998731653424 z_k1_s1070
                                 - 0.385001622017638 z_k1_s1071
                                 - 0.796998210305328 z_k1_s1072
                                 - 0.390001360771184 z_k1_s1073
                                 - 0.55699988113782 z_k1_s1074
                                 - 0.340999935687425 z_k1_s1075
                                 - 0.402000733779695 z_k1_s1076
                                 - 0.172000070619731 z_k1_s1077
                                 - 0.0639999126041663 z_k1_s1100
                                 - 0.205000121169347 z_k1_s1101
                                 - 0.299000101012899 z_k1_s1102
                                 - 0.098999971150129 z_k1_s1103
                                 - 0.23000007770223 z_k1_s1104
                                 - 0.120999885139096 z_k1_s1105
                                 - 0.115999893832519 z_k1_s1106
                                 - 0.0600000150788459 z_k1_s1107
                                 - 0.082000034822708 z_k1_s1110
                                 - 0.0980000127759186 z_k1_s1111
                                 - 0.198999705636805 z_k1_s1112
                                 - 0.100000050332571 z_k1_s1113
                                 - 0.0730000270677766 z_k1_s1114
                                 - 0.125999678569029 z_k1_s1115
                                 - 0.192000380438544 z_k1_s1116
                                 - 0.205000355806778 z_k1_s1117
                                 - 0.281999789184639 z_k1_s1120
                                 - 0.472000718353225 z_k1_s1121
                                 - 0.219000397303818 z_k1_s1122
                                 - 0.213000260172015 z_k1_s1123
                                 - 0.16300031207547 z_k1_s1124
                                 - 0.155000299899991 z_k1_s1125
                                 - 0.427999371386146 z_k1_s1126
                                 - 0.627999675773106 z_k1_s1127
                                 - 0.181999337644096 z_k1_s1130
                                 - 0.141999523216676 z_k1_s1131
                                 - 0.082999705936564 z_k1_s1132
                                 - 0.413998989320479 z_k1_s1133
                                 - 0.105999617432264 z_k1_s1134
                                 - 0.158998989349986 z_k1_s1135
                                 - 0.216998538270406 z_k1_s1136
                                 - 0.152998835186535 z_k1_s1137
                                 - 0.194999072171531 z_k1_s1140
                                 - 0.165999601888818 z_k1_s1141
                                 - 0.138999730357375 z_k1_s1142
                                 - 0.0969997832976507 z_k1_s1143
                                 - 0.306999812395679 z_k1_s1144
                                 - 0.183000304466288 z_k1_s1145
                                 - 0.164000296936953 z_k1_s1146
                                 - 0.217999158601623 z_k1_s1147
                                 - 0.109000192585504 z_k1_s1150
                                 - 0.343000936521662 z_k1_s1151
                                 - 0.265999565464707 z_k1_s1152
                                 - 0.540000258312864 z_k1_s1153
                                 - 0.308000744223444 z_k1_s1154
                                 - 0.192000276178077 z_k1_s1155
                                 - 0.218000148170789 z_k1_s1156
                                 - 0.394001013584592 z_k1_s1157
                                 - 0.0579999499783769 z_k1_s1160
                                 - 0.256000517152732 z_k1_s1161
                                 - 0.229000601933479 z_k1_s1162
                                 - 0.13699956981621 z_k1_s1163
                                 - 0.148999532135878 z_k1_s1164
                                 - 0.154999513295712 z_k1_s1165
                                 - 0.119000022636652 z_k1_s1166
                                 - 0.217999315473969 z_k1_s1167
                                 - 0.481999488488158 z_k1_s1170
                                 - 0.114999925311462 z_k1_s1171
                                 - 0.429998838643371 z_k1_s1172
                                 - 0.322000805271507 z_k1_s1173
                                 - 0.188999855198851 z_k1_s1174
                                 - 0.360000207235627 z_k1_s1175
                                 - 0.434999827743294 z_k1_s1176
                                 - 0.202999835080253 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.996141 w_k1_ADM_s000
                                 + 1.06503 w_k1_AXP_s000
                                 + 1.43125 w_k1_DIS_s000
                                 + 1.25145 w_k1_FDX_s000
                                 + 0.989071 w_k1_IBM_s000
                                 + 1.03556 w_k1_INTC_s000
                                 + 1.16261 w_k1_KO_s000 + 1.06252 w_k1_MAT_s000
                                 + 1.29223 w_k1_MSFT_s000
                                 + 1.19082 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.737416 w_k1_ADM_s000
                                 + 1.19934 w_k1_AXP_s000
                                 + 0.900227 w_k1_DIS_s000
                                 + 0.803367 w_k1_FDX_s000
                                 + 1.49611 w_k1_IBM_s000
                                 + 1.02633 w_k1_INTC_s000
                                 + 1.21172 w_k1_KO_s000 + 1.63496 w_k1_MAT_s000
                                 + 0.83359 w_k1_MSFT_s000
                                 + 0.922334 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.15641 w_k1_ADM_s000
                                 + 1.45392 w_k1_AXP_s000
                                 + 1.64366 w_k1_DIS_s000
                                 + 1.34519 w_k1_FDX_s000
                                 + 1.48115 w_k1_IBM_s000
                                 + 1.12893 w_k1_INTC_s000
                                 + 1.50864 w_k1_KO_s000 + 2.00978 w_k1_MAT_s000
                                 + 1.25126 w_k1_MSFT_s000
                                 + 1.15718 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.763896 w_k1_ADM_s000
                                 + 0.790404 w_k1_AXP_s000
                                 + 0.71875 w_k1_DIS_s000
                                 + 0.792904 w_k1_FDX_s000
                                 + 0.68086 w_k1_IBM_s000
                                 + 0.736297 w_k1_INTC_s000
                                 + 0.817765 w_k1_KO_s000
                                 + 0.725242 w_k1_MAT_s000
                                 + 0.848404 w_k1_MSFT_s000
                                 + 0.777692 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.824588 w_k1_ADM_s000
                                 + 0.601939 w_k1_AXP_s000
                                 + 0.633782 w_k1_DIS_s000
                                 + 0.672904 w_k1_FDX_s000
                                 + 1.03092 w_k1_IBM_s000
                                 + 0.612141 w_k1_INTC_s000
                                 + 0.937359 w_k1_KO_s000
                                 + 0.894161 w_k1_MAT_s000
                                 + 0.671534 w_k1_MSFT_s000
                                 + 0.885363 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.934585 w_k1_ADM_s000
                                 + 0.972601 w_k1_AXP_s000
                                 + 0.847942 w_k1_DIS_s000
                                 + 0.84296 w_k1_FDX_s000
                                 + 1.03655 w_k1_IBM_s000
                                 + 0.85083 w_k1_INTC_s000
                                 + 1.12856 w_k1_KO_s000 + 1.12542 w_k1_MAT_s000
                                 + 0.887663 w_k1_MSFT_s000
                                 + 0.911491 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.52473 w_k1_ADM_s000
                                 + 1.31374 w_k1_AXP_s000
                                 + 1.09687 w_k1_DIS_s000
                                 + 1.28814 w_k1_FDX_s000
                                 + 0.801814 w_k1_IBM_s000
                                 + 1.12491 w_k1_INTC_s000
                                 + 1.00271 w_k1_KO_s000
                                 + 0.876274 w_k1_MAT_s000
                                 + 1.07215 w_k1_MSFT_s000
                                 + 0.979179 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.17214 w_k1_ADM_s000
                                 + 1.08649 w_k1_AXP_s000
                                 + 0.950377 w_k1_DIS_s000
                                 + 1.00503 w_k1_FDX_s000
                                 + 0.870919 w_k1_IBM_s000
                                 + 0.929849 w_k1_INTC_s000
                                 + 1.02194 w_k1_KO_s000 + 1.07304 w_k1_MAT_s000
                                 + 1.03983 w_k1_MSFT_s000
                                 + 1.10684 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.08108 w_k1_ADM_s001
                                 + 0.832087 w_k1_AXP_s001
                                 + 1.2477 w_k1_DIS_s001 + 1.03551 w_k1_FDX_s001
                                 + 1.11325 w_k1_IBM_s001
                                 + 1.0459 w_k1_INTC_s001 + 1.08518 w_k1_KO_s001
                                 + 0.980176 w_k1_MAT_s001
                                 + 1.11868 w_k1_MSFT_s001
                                 + 1.18508 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.367 w_k1_ADM_s001
                                 + 1.3773 w_k1_AXP_s001 + 1.84987 w_k1_DIS_s001
                                 + 1.34969 w_k1_FDX_s001 + 1.6748 w_k1_IBM_s001
                                 + 1.24007 w_k1_INTC_s001
                                 + 1.39626 w_k1_KO_s001 + 1.58869 w_k1_MAT_s001
                                 + 1.37111 w_k1_MSFT_s001
                                 + 1.44116 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.0693 w_k1_ADM_s001
                                 + 1.05732 w_k1_AXP_s001
                                 + 0.862356 w_k1_DIS_s001
                                 + 1.06528 w_k1_FDX_s001
                                 + 0.76827 w_k1_IBM_s001
                                 + 1.00465 w_k1_INTC_s001
                                 + 0.815363 w_k1_KO_s001
                                 + 0.95727 w_k1_MAT_s001
                                 + 1.1068 w_k1_MSFT_s001 + 0.99858 w_k1_PG_s001
                                 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.788298 w_k1_ADM_s001
                                 + 0.686368 w_k1_AXP_s001
                                 + 0.832035 w_k1_DIS_s001
                                 + 0.843668 w_k1_FDX_s001
                                 + 0.836516 w_k1_IBM_s001
                                 + 0.832769 w_k1_INTC_s001
                                 + 0.862479 w_k1_KO_s001
                                 + 0.643815 w_k1_MAT_s001
                                 + 0.905005 w_k1_MSFT_s001
                                 + 1.00199 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.15715 w_k1_ADM_s001
                                 + 1.46025 w_k1_AXP_s001
                                 + 1.27823 w_k1_DIS_s001
                                 + 1.31856 w_k1_FDX_s001
                                 + 0.945532 w_k1_IBM_s001
                                 + 1.33102 w_k1_INTC_s001
                                 + 0.981742 w_k1_KO_s001
                                 + 1.28276 w_k1_MAT_s001
                                 + 1.17474 w_k1_MSFT_s001
                                 + 1.20835 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.895766 w_k1_ADM_s001
                                 + 1.41182 w_k1_AXP_s001
                                 + 1.19306 w_k1_DIS_s001
                                 + 1.09381 w_k1_FDX_s001
                                 + 1.59332 w_k1_IBM_s001
                                 + 1.15477 w_k1_INTC_s001
                                 + 1.15616 w_k1_KO_s001 + 1.38094 w_k1_MAT_s001
                                 + 0.982567 w_k1_MSFT_s001
                                 + 1.15295 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.777637 w_k1_ADM_s001
                                 + 0.607075 w_k1_AXP_s001
                                 + 0.603566 w_k1_DIS_s001
                                 + 0.723694 w_k1_FDX_s001
                                 + 0.852167 w_k1_IBM_s001
                                 + 0.81938 w_k1_INTC_s001
                                 + 0.664168 w_k1_KO_s001
                                 + 0.416511 w_k1_MAT_s001
                                 + 0.725261 w_k1_MSFT_s001
                                 + 0.849899 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.883053 w_k1_ADM_s001
                                 + 1.04065 w_k1_AXP_s001
                                 + 0.796491 w_k1_DIS_s001
                                 + 0.807745 w_k1_FDX_s001
                                 + 1.28109 w_k1_IBM_s001
                                 + 0.748095 w_k1_INTC_s001
                                 + 0.802733 w_k1_KO_s001
                                 + 1.35468 w_k1_MAT_s001
                                 + 0.799849 w_k1_MSFT_s001
                                 + 0.951149 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.08912 w_k1_ADM_s002
                                 + 0.773446 w_k1_AXP_s002
                                 + 0.898334 w_k1_DIS_s002
                                 + 1.07659 w_k1_FDX_s002
                                 + 0.737364 w_k1_IBM_s002
                                 + 1.08211 w_k1_INTC_s002
                                 + 0.748974 w_k1_KO_s002
                                 + 0.372802 w_k1_MAT_s002
                                 + 0.917043 w_k1_MSFT_s002
                                 + 0.927347 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.933771 w_k1_ADM_s002
                                 + 0.591806 w_k1_AXP_s002
                                 + 0.629984 w_k1_DIS_s002
                                 + 0.72575 w_k1_FDX_s002
                                 + 0.924653 w_k1_IBM_s002
                                 + 0.779835 w_k1_INTC_s002
                                 + 0.734162 w_k1_KO_s002
                                 + 0.692608 w_k1_MAT_s002
                                 + 0.751747 w_k1_MSFT_s002
                                 + 0.824041 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.939651 w_k1_ADM_s002
                                 + 0.719794 w_k1_AXP_s002
                                 + 0.840475 w_k1_DIS_s002
                                 + 0.886104 w_k1_FDX_s002
                                 + 0.77474 w_k1_IBM_s002
                                 + 0.778388 w_k1_INTC_s002
                                 + 0.762846 w_k1_KO_s002
                                 + 0.509384 w_k1_MAT_s002
                                 + 0.853269 w_k1_MSFT_s002
                                 + 0.925359 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.99485 w_k1_ADM_s002
                                 + 1.06055 w_k1_AXP_s002
                                 + 0.934826 w_k1_DIS_s002
                                 + 0.973897 w_k1_FDX_s002
                                 + 1.18513 w_k1_IBM_s002
                                 + 1.19584 w_k1_INTC_s002
                                 + 1.00177 w_k1_KO_s002
                                 + 0.917246 w_k1_MAT_s002
                                 + 0.984679 w_k1_MSFT_s002
                                 + 1.10165 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.20736 w_k1_ADM_s002
                                 + 1.24982 w_k1_AXP_s002
                                 + 1.39566 w_k1_DIS_s002
                                 + 1.22568 w_k1_FDX_s002
                                 + 1.17414 w_k1_IBM_s002
                                 + 1.27732 w_k1_INTC_s002
                                 + 1.29428 w_k1_KO_s002 + 1.13191 w_k1_MAT_s002
                                 + 1.19621 w_k1_MSFT_s002
                                 + 1.32959 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.865296 w_k1_ADM_s002
                                 + 0.842894 w_k1_AXP_s002
                                 + 0.947886 w_k1_DIS_s002
                                 + 0.939874 w_k1_FDX_s002
                                 + 0.897046 w_k1_IBM_s002
                                 + 1.01751 w_k1_INTC_s002
                                 + 0.811512 w_k1_KO_s002
                                 + 0.940188 w_k1_MAT_s002
                                 + 0.907339 w_k1_MSFT_s002
                                 + 1.00588 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.951316 w_k1_ADM_s002
                                 + 0.90997 w_k1_AXP_s002
                                 + 1.28835 w_k1_DIS_s002
                                 + 1.15502 w_k1_FDX_s002
                                 + 0.836313 w_k1_IBM_s002
                                 + 0.953462 w_k1_INTC_s002
                                 + 0.947007 w_k1_KO_s002
                                 + 0.533361 w_k1_MAT_s002
                                 + 1.13159 w_k1_MSFT_s002
                                 + 1.00944 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.10686 w_k1_ADM_s002
                                 + 1.12667 w_k1_AXP_s002
                                 + 1.32796 w_k1_DIS_s002
                                 + 1.13367 w_k1_FDX_s002
                                 + 1.58236 w_k1_IBM_s002
                                 + 1.13686 w_k1_INTC_s002
                                 + 1.33263 w_k1_KO_s002 + 1.45122 w_k1_MAT_s002
                                 + 1.18646 w_k1_MSFT_s002
                                 + 1.23532 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.18318 w_k1_ADM_s003
                                 + 0.900798 w_k1_AXP_s003
                                 + 0.92095 w_k1_DIS_s003
                                 + 0.982765 w_k1_FDX_s003
                                 + 0.611548 w_k1_IBM_s003
                                 + 0.976849 w_k1_INTC_s003
                                 + 0.749684 w_k1_KO_s003
                                 + 0.342917 w_k1_MAT_s003
                                 + 0.925456 w_k1_MSFT_s003
                                 + 0.923063 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.32895 w_k1_ADM_s003
                                 + 1.30171 w_k1_AXP_s003 + 1.4559 w_k1_DIS_s003
                                 + 1.32549 w_k1_FDX_s003
                                 + 1.49639 w_k1_IBM_s003
                                 + 1.34343 w_k1_INTC_s003
                                 + 1.57643 w_k1_KO_s003 + 1.27048 w_k1_MAT_s003
                                 + 1.44652 w_k1_MSFT_s003 + 1.2834 w_k1_PG_s003
                                 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.94608 w_k1_ADM_s003
                                 + 1.08876 w_k1_AXP_s003 + 1.0895 w_k1_DIS_s003
                                 + 1.03628 w_k1_FDX_s003
                                 + 1.41068 w_k1_IBM_s003
                                 + 1.06679 w_k1_INTC_s003
                                 + 1.26521 w_k1_KO_s003 + 1.28638 w_k1_MAT_s003
                                 + 1.09153 w_k1_MSFT_s003
                                 + 1.07939 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.957372 w_k1_ADM_s003
                                 + 0.71907 w_k1_AXP_s003
                                 + 0.702849 w_k1_DIS_s003
                                 + 0.823771 w_k1_FDX_s003
                                 + 0.924725 w_k1_IBM_s003
                                 + 0.952337 w_k1_INTC_s003
                                 + 0.836488 w_k1_KO_s003
                                 + 0.57665 w_k1_MAT_s003
                                 + 0.933938 w_k1_MSFT_s003
                                 + 0.921799 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.0187 w_k1_ADM_s003
                                 + 1.07281 w_k1_AXP_s003
                                 + 1.09111 w_k1_DIS_s003
                                 + 1.07106 w_k1_FDX_s003
                                 + 1.15972 w_k1_IBM_s003
                                 + 1.22813 w_k1_INTC_s003
                                 + 1.22309 w_k1_KO_s003
                                 + 0.796284 w_k1_MAT_s003
                                 + 1.05657 w_k1_MSFT_s003
                                 + 1.08151 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.34612 w_k1_ADM_s003
                                 + 0.999157 w_k1_AXP_s003
                                 + 1.03487 w_k1_DIS_s003
                                 + 1.15586 w_k1_FDX_s003
                                 + 0.695096 w_k1_IBM_s003
                                 + 1.10409 w_k1_INTC_s003
                                 + 0.859753 w_k1_KO_s003
                                 + 0.692585 w_k1_MAT_s003
                                 + 1.08678 w_k1_MSFT_s003
                                 + 1.10129 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.20892 w_k1_ADM_s003
                                 + 0.957074 w_k1_AXP_s003
                                 + 0.824473 w_k1_DIS_s003
                                 + 0.991777 w_k1_FDX_s003
                                 + 0.914827 w_k1_IBM_s003
                                 + 1.03698 w_k1_INTC_s003
                                 + 0.958366 w_k1_KO_s003
                                 + 0.770791 w_k1_MAT_s003
                                 + 0.94456 w_k1_MSFT_s003
                                 + 1.02684 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.02075 w_k1_ADM_s003
                                 + 1.23811 w_k1_AXP_s003
                                 + 1.42226 w_k1_DIS_s003
                                 + 1.24537 w_k1_FDX_s003
                                 + 1.04804 w_k1_IBM_s003
                                 + 1.61499 w_k1_INTC_s003 + 1.1566 w_k1_KO_s003
                                 + 0.868665 w_k1_MAT_s003
                                 + 1.33671 w_k1_MSFT_s003
                                 + 1.13272 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.28067 w_k1_ADM_s004
                                 + 1.03028 w_k1_AXP_s004
                                 + 0.82657 w_k1_DIS_s004
                                 + 0.898786 w_k1_FDX_s004
                                 + 1.04797 w_k1_IBM_s004
                                 + 0.833531 w_k1_INTC_s004
                                 + 1.04494 w_k1_KO_s004 + 1.1742 w_k1_MAT_s004
                                 + 0.870943 w_k1_MSFT_s004
                                 + 1.04328 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.16315 w_k1_ADM_s004
                                 + 1.09987 w_k1_AXP_s004
                                 + 0.971676 w_k1_DIS_s004
                                 + 0.92493 w_k1_FDX_s004 + 1.6925 w_k1_IBM_s004
                                 + 0.841028 w_k1_INTC_s004
                                 + 1.12205 w_k1_KO_s004 + 1.79733 w_k1_MAT_s004
                                 + 0.901341 w_k1_MSFT_s004
                                 + 1.07866 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.30559 w_k1_ADM_s004
                                 + 1.45784 w_k1_AXP_s004
                                 + 1.81725 w_k1_DIS_s004
                                 + 1.42571 w_k1_FDX_s004
                                 + 1.35032 w_k1_IBM_s004
                                 + 1.08456 w_k1_INTC_s004 + 1.3858 w_k1_KO_s004
                                 + 1.48365 w_k1_MAT_s004
                                 + 1.47917 w_k1_MSFT_s004
                                 + 1.21091 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.29709 w_k1_ADM_s004
                                 + 1.05455 w_k1_AXP_s004
                                 + 1.02386 w_k1_DIS_s004
                                 + 1.05809 w_k1_FDX_s004
                                 + 0.758689 w_k1_IBM_s004
                                 + 0.951704 w_k1_INTC_s004
                                 + 0.950157 w_k1_KO_s004
                                 + 0.771205 w_k1_MAT_s004
                                 + 0.992566 w_k1_MSFT_s004
                                 + 1.04494 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.946206 w_k1_ADM_s004
                                 + 0.867656 w_k1_AXP_s004
                                 + 0.610638 w_k1_DIS_s004
                                 + 0.864347 w_k1_FDX_s004
                                 + 0.633314 w_k1_IBM_s004
                                 + 0.793686 w_k1_INTC_s004
                                 + 0.67784 w_k1_KO_s004
                                 + 0.544722 w_k1_MAT_s004
                                 + 0.814902 w_k1_MSFT_s004
                                 + 0.886542 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.09871 w_k1_ADM_s004
                                 + 1.22281 w_k1_AXP_s004
                                 + 1.24723 w_k1_DIS_s004
                                 + 1.05302 w_k1_FDX_s004
                                 + 1.30117 w_k1_IBM_s004
                                 + 0.928004 w_k1_INTC_s004
                                 + 1.36293 w_k1_KO_s004 + 1.33222 w_k1_MAT_s004
                                 + 1.15767 w_k1_MSFT_s004
                                 + 1.29625 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.59679 w_k1_ADM_s004
                                 + 1.24501 w_k1_AXP_s004
                                 + 1.62356 w_k1_DIS_s004
                                 + 1.28948 w_k1_FDX_s004
                                 + 1.01162 w_k1_IBM_s004
                                 + 1.03423 w_k1_INTC_s004
                                 + 1.21114 w_k1_KO_s004
                                 + 0.702918 w_k1_MAT_s004
                                 + 1.19296 w_k1_MSFT_s004
                                 + 1.20746 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.934247 w_k1_ADM_s004
                                 + 0.728966 w_k1_AXP_s004
                                 + 0.739504 w_k1_DIS_s004
                                 + 0.818922 w_k1_FDX_s004
                                 + 0.859553 w_k1_IBM_s004
                                 + 0.816834 w_k1_INTC_s004
                                 + 0.880102 w_k1_KO_s004
                                 + 0.908055 w_k1_MAT_s004
                                 + 0.906531 w_k1_MSFT_s004
                                 + 1.00037 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.972221 w_k1_ADM_s005
                                 + 0.692313 w_k1_AXP_s005
                                 + 0.839585 w_k1_DIS_s005
                                 + 0.878753 w_k1_FDX_s005
                                 + 0.623478 w_k1_IBM_s005
                                 + 0.825375 w_k1_INTC_s005
                                 + 0.757242 w_k1_KO_s005
                                 + 0.566306 w_k1_MAT_s005
                                 + 0.857426 w_k1_MSFT_s005
                                 + 0.900778 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.06704 w_k1_ADM_s005
                                 + 1.30434 w_k1_AXP_s005
                                 + 1.36121 w_k1_DIS_s005
                                 + 1.26478 w_k1_FDX_s005
                                 + 1.06826 w_k1_IBM_s005
                                 + 1.25398 w_k1_INTC_s005
                                 + 1.05041 w_k1_KO_s005
                                 + 0.926119 w_k1_MAT_s005
                                 + 1.04454 w_k1_MSFT_s005
                                 + 1.09428 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.06364 w_k1_ADM_s005
                                 + 1.54713 w_k1_AXP_s005
                                 + 1.82002 w_k1_DIS_s005
                                 + 1.27034 w_k1_FDX_s005
                                 + 1.38668 w_k1_IBM_s005
                                 + 1.13599 w_k1_INTC_s005
                                 + 1.39639 w_k1_KO_s005 + 2.00386 w_k1_MAT_s005
                                 + 1.27603 w_k1_MSFT_s005
                                 + 1.28445 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.808741 w_k1_ADM_s005
                                 + 0.815372 w_k1_AXP_s005
                                 + 0.825291 w_k1_DIS_s005
                                 + 0.792698 w_k1_FDX_s005
                                 + 0.884251 w_k1_IBM_s005
                                 + 0.760514 w_k1_INTC_s005
                                 + 0.880596 w_k1_KO_s005
                                 + 0.788756 w_k1_MAT_s005
                                 + 0.728802 w_k1_MSFT_s005
                                 + 0.986422 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.88566 w_k1_ADM_s005
                                 + 0.900173 w_k1_AXP_s005
                                 + 0.946123 w_k1_DIS_s005
                                 + 0.873746 w_k1_FDX_s005
                                 + 1.02336 w_k1_IBM_s005
                                 + 0.892125 w_k1_INTC_s005
                                 + 0.909106 w_k1_KO_s005
                                 + 1.33087 w_k1_MAT_s005
                                 + 0.87263 w_k1_MSFT_s005
                                 + 0.999873 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.02965 w_k1_ADM_s005
                                 + 0.991933 w_k1_AXP_s005
                                 + 1.0098 w_k1_DIS_s005 + 1.06643 w_k1_FDX_s005
                                 + 0.787488 w_k1_IBM_s005
                                 + 1.03053 w_k1_INTC_s005
                                 + 0.849313 w_k1_KO_s005
                                 + 0.802619 w_k1_MAT_s005
                                 + 1.0388 w_k1_MSFT_s005
                                 + 0.989016 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.752323 w_k1_ADM_s005
                                 + 0.717357 w_k1_AXP_s005
                                 + 0.488192 w_k1_DIS_s005
                                 + 0.713583 w_k1_FDX_s005
                                 + 0.5981 w_k1_IBM_s005
                                 + 0.789837 w_k1_INTC_s005
                                 + 0.587895 w_k1_KO_s005
                                 + 0.554385 w_k1_MAT_s005
                                 + 0.618576 w_k1_MSFT_s005
                                 + 0.787846 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.06041 w_k1_ADM_s005
                                 + 1.02927 w_k1_AXP_s005
                                 + 1.12838 w_k1_DIS_s005
                                 + 0.987127 w_k1_FDX_s005
                                 + 1.43292 w_k1_IBM_s005
                                 + 0.869501 w_k1_INTC_s005
                                 + 1.06995 w_k1_KO_s005 + 1.62329 w_k1_MAT_s005
                                 + 0.835156 w_k1_MSFT_s005
                                 + 1.11555 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.23813 w_k1_ADM_s006
                                 + 1.3666 w_k1_AXP_s006 + 1.19646 w_k1_DIS_s006
                                 + 1.16733 w_k1_FDX_s006
                                 + 1.19568 w_k1_IBM_s006
                                 + 1.17973 w_k1_INTC_s006
                                 + 1.27658 w_k1_KO_s006
                                 + 0.988145 w_k1_MAT_s006
                                 + 1.08522 w_k1_MSFT_s006
                                 + 1.13698 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.46158 w_k1_ADM_s006
                                 + 1.10145 w_k1_AXP_s006
                                 + 0.881579 w_k1_DIS_s006
                                 + 0.959859 w_k1_FDX_s006
                                 + 1.06775 w_k1_IBM_s006
                                 + 0.893698 w_k1_INTC_s006
                                 + 0.936552 w_k1_KO_s006 + 1.0526 w_k1_MAT_s006
                                 + 0.957346 w_k1_MSFT_s006
                                 + 1.08437 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.01994 w_k1_ADM_s006
                                 + 0.771092 w_k1_AXP_s006
                                 + 0.706944 w_k1_DIS_s006
                                 + 0.851618 w_k1_FDX_s006
                                 + 0.70169 w_k1_IBM_s006
                                 + 0.834225 w_k1_INTC_s006
                                 + 0.805128 w_k1_KO_s006
                                 + 0.628386 w_k1_MAT_s006
                                 + 1.00556 w_k1_MSFT_s006
                                 + 0.899564 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.906516 w_k1_ADM_s006
                                 + 0.716064 w_k1_AXP_s006
                                 + 0.762051 w_k1_DIS_s006
                                 + 0.835681 w_k1_FDX_s006
                                 + 0.943877 w_k1_IBM_s006
                                 + 0.875714 w_k1_INTC_s006
                                 + 0.950467 w_k1_KO_s006
                                 + 0.884188 w_k1_MAT_s006
                                 + 0.922667 w_k1_MSFT_s006
                                 + 0.981155 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.43742 w_k1_ADM_s006
                                 + 1.12462 w_k1_AXP_s006
                                 + 1.29719 w_k1_DIS_s006
                                 + 1.29012 w_k1_FDX_s006
                                 + 0.916527 w_k1_IBM_s006
                                 + 1.08077 w_k1_INTC_s006
                                 + 1.13949 w_k1_KO_s006 + 1.0532 w_k1_MAT_s006
                                 + 1.31377 w_k1_MSFT_s006
                                 + 1.24165 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.13085 w_k1_ADM_s006
                                 + 0.935662 w_k1_AXP_s006
                                 + 1.12189 w_k1_DIS_s006
                                 + 0.976373 w_k1_FDX_s006
                                 + 1.39985 w_k1_IBM_s006
                                 + 0.984489 w_k1_INTC_s006
                                 + 1.19317 w_k1_KO_s006 + 1.3292 w_k1_MAT_s006
                                 + 1.03714 w_k1_MSFT_s006
                                 + 1.06318 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.899784 w_k1_ADM_s006
                                 + 0.699956 w_k1_AXP_s006
                                 + 0.725565 w_k1_DIS_s006
                                 + 0.748488 w_k1_FDX_s006
                                 + 1.26334 w_k1_IBM_s006
                                 + 0.871309 w_k1_INTC_s006
                                 + 0.921516 w_k1_KO_s006
                                 + 0.85246 w_k1_MAT_s006
                                 + 0.73234 w_k1_MSFT_s006
                                 + 0.885173 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.22268 w_k1_ADM_s006
                                 + 1.67581 w_k1_AXP_s006 + 1.7131 w_k1_DIS_s006
                                 + 1.41798 w_k1_FDX_s006
                                 + 1.42043 w_k1_IBM_s006
                                 + 1.39698 w_k1_INTC_s006
                                 + 1.51089 w_k1_KO_s006 + 1.38652 w_k1_MAT_s006
                                 + 1.341 w_k1_MSFT_s006 + 1.55483 w_k1_PG_s006
                                 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.926997 w_k1_ADM_s007
                                 + 0.671734 w_k1_AXP_s007
                                 + 0.583691 w_k1_DIS_s007
                                 + 0.727937 w_k1_FDX_s007
                                 + 0.604231 w_k1_IBM_s007
                                 + 0.881626 w_k1_INTC_s007
                                 + 0.756309 w_k1_KO_s007
                                 + 0.715849 w_k1_MAT_s007
                                 + 0.795597 w_k1_MSFT_s007
                                 + 0.844696 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.37652 w_k1_ADM_s007
                                 + 1.02884 w_k1_AXP_s007
                                 + 1.02092 w_k1_DIS_s007
                                 + 1.11764 w_k1_FDX_s007
                                 + 0.761492 w_k1_IBM_s007
                                 + 1.30641 w_k1_INTC_s007
                                 + 0.976586 w_k1_KO_s007
                                 + 0.647941 w_k1_MAT_s007
                                 + 1.06824 w_k1_MSFT_s007
                                 + 0.92106 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.00151 w_k1_ADM_s007
                                 + 0.88981 w_k1_AXP_s007
                                 + 0.966575 w_k1_DIS_s007
                                 + 0.880379 w_k1_FDX_s007
                                 + 1.15674 w_k1_IBM_s007
                                 + 1.02021 w_k1_INTC_s007
                                 + 1.03711 w_k1_KO_s007 + 1.16308 w_k1_MAT_s007
                                 + 0.883556 w_k1_MSFT_s007
                                 + 0.860968 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.953846 w_k1_ADM_s007
                                 + 0.961281 w_k1_AXP_s007
                                 + 0.68722 w_k1_DIS_s007
                                 + 0.86382 w_k1_FDX_s007
                                 + 0.872551 w_k1_IBM_s007
                                 + 0.956068 w_k1_INTC_s007
                                 + 0.836213 w_k1_KO_s007
                                 + 0.962189 w_k1_MAT_s007
                                 + 0.85998 w_k1_MSFT_s007
                                 + 0.82193 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.40124 w_k1_ADM_s007
                                 + 1.7474 w_k1_AXP_s007 + 1.83547 w_k1_DIS_s007
                                 + 1.48096 w_k1_FDX_s007
                                 + 1.39073 w_k1_IBM_s007
                                 + 1.41077 w_k1_INTC_s007
                                 + 1.56911 w_k1_KO_s007 + 2.61588 w_k1_MAT_s007
                                 + 1.48592 w_k1_MSFT_s007
                                 + 1.36284 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.42559 w_k1_ADM_s007
                                 + 1.3089 w_k1_AXP_s007 + 1.36019 w_k1_DIS_s007
                                 + 1.17659 w_k1_FDX_s007
                                 + 1.68992 w_k1_IBM_s007
                                 + 1.18053 w_k1_INTC_s007
                                 + 1.62105 w_k1_KO_s007 + 1.11219 w_k1_MAT_s007
                                 + 1.16881 w_k1_MSFT_s007
                                 + 1.10212 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.29895 w_k1_ADM_s007
                                 + 1.42908 w_k1_AXP_s007
                                 + 1.18991 w_k1_DIS_s007
                                 + 1.14348 w_k1_FDX_s007 + 1.0951 w_k1_IBM_s007
                                 + 1.04942 w_k1_INTC_s007
                                 + 1.22921 w_k1_KO_s007 + 2.11801 w_k1_MAT_s007
                                 + 1.08594 w_k1_MSFT_s007
                                 + 1.08475 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.21383 w_k1_ADM_s007
                                 + 1.09202 w_k1_AXP_s007
                                 + 1.34881 w_k1_DIS_s007
                                 + 1.02415 w_k1_FDX_s007
                                 + 1.16611 w_k1_IBM_s007
                                 + 1.11104 w_k1_INTC_s007
                                 + 1.44834 w_k1_KO_s007 + 1.6541 w_k1_MAT_s007
                                 + 1.12619 w_k1_MSFT_s007
                                 + 1.06265 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.01803 w_k1_ADM_s010
                                 + 0.709042 w_k1_AXP_s010
                                 + 0.841344 w_k1_DIS_s010
                                 + 0.92807 w_k1_FDX_s010
                                 + 0.888659 w_k1_IBM_s010
                                 + 1.16667 w_k1_INTC_s010
                                 + 0.813382 w_k1_KO_s010
                                 + 0.794022 w_k1_MAT_s010
                                 + 0.900837 w_k1_MSFT_s010
                                 + 0.937603 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.30654 w_k1_ADM_s010
                                 + 1.05977 w_k1_AXP_s010
                                 + 1.14667 w_k1_DIS_s010 + 1.1727 w_k1_FDX_s010
                                 + 0.855368 w_k1_IBM_s010
                                 + 1.23426 w_k1_INTC_s010 + 1.0784 w_k1_KO_s010
                                 + 0.94211 w_k1_MAT_s010
                                 + 1.08793 w_k1_MSFT_s010
                                 + 1.13847 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.671598 w_k1_ADM_s010
                                 + 0.614986 w_k1_AXP_s010
                                 + 0.610845 w_k1_DIS_s010
                                 + 0.702882 w_k1_FDX_s010
                                 + 0.930803 w_k1_IBM_s010
                                 + 0.836669 w_k1_INTC_s010
                                 + 0.783675 w_k1_KO_s010
                                 + 0.898243 w_k1_MAT_s010
                                 + 0.681258 w_k1_MSFT_s010
                                 + 0.857997 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.852611 w_k1_ADM_s010
                                 + 0.75704 w_k1_AXP_s010
                                 + 0.869763 w_k1_DIS_s010
                                 + 0.835169 w_k1_FDX_s010
                                 + 1.38951 w_k1_IBM_s010
                                 + 0.983325 w_k1_INTC_s010
                                 + 1.1756 w_k1_KO_s010 + 1.19955 w_k1_MAT_s010
                                 + 0.796931 w_k1_MSFT_s010
                                 + 1.00276 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.10526 w_k1_ADM_s010
                                 + 0.660616 w_k1_AXP_s010
                                 + 0.572287 w_k1_DIS_s010
                                 + 0.759402 w_k1_FDX_s010
                                 + 0.646347 w_k1_IBM_s010
                                 + 0.757677 w_k1_INTC_s010
                                 + 0.706531 w_k1_KO_s010
                                 + 0.746561 w_k1_MAT_s010
                                 + 0.745972 w_k1_MSFT_s010
                                 + 0.827475 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.05962 w_k1_ADM_s010
                                 + 1.17193 w_k1_AXP_s010
                                 + 0.933388 w_k1_DIS_s010
                                 + 0.981394 w_k1_FDX_s010
                                 + 1.16162 w_k1_IBM_s010
                                 + 0.97623 w_k1_INTC_s010 + 1.0708 w_k1_KO_s010
                                 + 1.69396 w_k1_MAT_s010
                                 + 0.866024 w_k1_MSFT_s010
                                 + 1.06765 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.977783 w_k1_ADM_s010
                                 + 0.850741 w_k1_AXP_s010
                                 + 1.12874 w_k1_DIS_s010
                                 + 1.11807 w_k1_FDX_s010
                                 + 1.10529 w_k1_IBM_s010
                                 + 1.28819 w_k1_INTC_s010
                                 + 1.07488 w_k1_KO_s010
                                 + 0.788301 w_k1_MAT_s010
                                 + 1.02787 w_k1_MSFT_s010
                                 + 1.03962 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.36849 w_k1_ADM_s010
                                 + 1.31659 w_k1_AXP_s010
                                 + 1.60793 w_k1_DIS_s010
                                 + 1.36702 w_k1_FDX_s010
                                 + 1.42568 w_k1_IBM_s010
                                 + 1.13602 w_k1_INTC_s010
                                 + 1.36051 w_k1_KO_s010 + 1.49155 w_k1_MAT_s010
                                 + 1.12049 w_k1_MSFT_s010
                                 + 1.25377 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.869715 w_k1_ADM_s011
                                 + 0.88381 w_k1_AXP_s011
                                 + 0.615263 w_k1_DIS_s011
                                 + 0.73868 w_k1_FDX_s011
                                 + 0.87772 w_k1_IBM_s011
                                 + 0.775169 w_k1_INTC_s011
                                 + 0.673377 w_k1_KO_s011
                                 + 0.898681 w_k1_MAT_s011
                                 + 0.719104 w_k1_MSFT_s011
                                 + 0.832011 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.916751 w_k1_ADM_s011
                                 + 1.22342 w_k1_AXP_s011
                                 + 1.02996 w_k1_DIS_s011
                                 + 0.950627 w_k1_FDX_s011
                                 + 1.59221 w_k1_IBM_s011
                                 + 1.04269 w_k1_INTC_s011 + 1.1708 w_k1_KO_s011
                                 + 1.70995 w_k1_MAT_s011
                                 + 0.983849 w_k1_MSFT_s011
                                 + 0.997942 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.38243 w_k1_ADM_s011
                                 + 1.47316 w_k1_AXP_s011
                                 + 1.40724 w_k1_DIS_s011
                                 + 1.35925 w_k1_FDX_s011
                                 + 1.25575 w_k1_IBM_s011
                                 + 1.11252 w_k1_INTC_s011
                                 + 1.45858 w_k1_KO_s011 + 1.50551 w_k1_MAT_s011
                                 + 1.28793 w_k1_MSFT_s011
                                 + 1.23616 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.0762 w_k1_ADM_s011
                                 + 1.1345 w_k1_AXP_s011 + 1.30172 w_k1_DIS_s011
                                 + 1.04406 w_k1_FDX_s011 + 1.3883 w_k1_IBM_s011
                                 + 1.05436 w_k1_INTC_s011
                                 + 1.26278 w_k1_KO_s011 + 1.65231 w_k1_MAT_s011
                                 + 1.28017 w_k1_MSFT_s011
                                 + 1.09031 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.987663 w_k1_ADM_s011
                                 + 1.32611 w_k1_AXP_s011
                                 + 1.66369 w_k1_DIS_s011 + 1.1916 w_k1_FDX_s011
                                 + 1.25714 w_k1_IBM_s011
                                 + 1.08133 w_k1_INTC_s011
                                 + 1.20324 w_k1_KO_s011 + 1.88668 w_k1_MAT_s011
                                 + 1.27521 w_k1_MSFT_s011
                                 + 1.27733 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.16453 w_k1_ADM_s011
                                 + 1.25857 w_k1_AXP_s011
                                 + 1.16574 w_k1_DIS_s011
                                 + 1.18931 w_k1_FDX_s011
                                 + 0.930087 w_k1_IBM_s011
                                 + 1.16917 w_k1_INTC_s011
                                 + 1.04027 w_k1_KO_s011 + 1.54623 w_k1_MAT_s011
                                 + 1.18031 w_k1_MSFT_s011 + 1.1062 w_k1_PG_s011
                                 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.882924 w_k1_ADM_s011
                                 + 0.876139 w_k1_AXP_s011
                                 + 0.986858 w_k1_DIS_s011
                                 + 0.896155 w_k1_FDX_s011
                                 + 1.00682 w_k1_IBM_s011
                                 + 0.886547 w_k1_INTC_s011
                                 + 1.05655 w_k1_KO_s011 + 1.10175 w_k1_MAT_s011
                                 + 1.0124 w_k1_MSFT_s011
                                 + 0.961942 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.990911 w_k1_ADM_s011
                                 + 1.07568 w_k1_AXP_s011
                                 + 0.858971 w_k1_DIS_s011
                                 + 1.03422 w_k1_FDX_s011
                                 + 0.806149 w_k1_IBM_s011
                                 + 0.91526 w_k1_INTC_s011
                                 + 0.866246 w_k1_KO_s011
                                 + 1.29867 w_k1_MAT_s011
                                 + 1.03985 w_k1_MSFT_s011
                                 + 1.02703 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.29368 w_k1_ADM_s012
                                 + 1.71553 w_k1_AXP_s012
                                 + 1.66891 w_k1_DIS_s012
                                 + 1.41984 w_k1_FDX_s012
                                 + 1.04476 w_k1_IBM_s012
                                 + 1.16291 w_k1_INTC_s012
                                 + 1.18356 w_k1_KO_s012 + 1.01223 w_k1_MAT_s012
                                 + 1.3554 w_k1_MSFT_s012 + 1.29299 w_k1_PG_s012
                                 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.993644 w_k1_ADM_s012
                                 + 1.00017 w_k1_AXP_s012
                                 + 1.18343 w_k1_DIS_s012
                                 + 0.948961 w_k1_FDX_s012
                                 + 1.39297 w_k1_IBM_s012
                                 + 0.990798 w_k1_INTC_s012
                                 + 1.24724 w_k1_KO_s012 + 1.39581 w_k1_MAT_s012
                                 + 0.981646 w_k1_MSFT_s012
                                 + 1.12335 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.803177 w_k1_ADM_s012
                                 + 0.704754 w_k1_AXP_s012
                                 + 0.618197 w_k1_DIS_s012
                                 + 0.725615 w_k1_FDX_s012
                                 + 0.582258 w_k1_IBM_s012
                                 + 0.910267 w_k1_INTC_s012
                                 + 0.692804 w_k1_KO_s012
                                 + 0.598224 w_k1_MAT_s012
                                 + 0.761188 w_k1_MSFT_s012
                                 + 0.722649 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.27844 w_k1_ADM_s012
                                 + 1.09613 w_k1_AXP_s012
                                 + 1.31974 w_k1_DIS_s012
                                 + 1.09402 w_k1_FDX_s012
                                 + 1.05667 w_k1_IBM_s012
                                 + 0.906649 w_k1_INTC_s012
                                 + 1.23187 w_k1_KO_s012 + 1.04372 w_k1_MAT_s012
                                 + 1.05197 w_k1_MSFT_s012
                                 + 1.10819 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.876613 w_k1_ADM_s012
                                 + 0.720061 w_k1_AXP_s012
                                 + 0.815932 w_k1_DIS_s012
                                 + 0.835269 w_k1_FDX_s012
                                 + 0.942209 w_k1_IBM_s012
                                 + 0.850201 w_k1_INTC_s012
                                 + 0.904254 w_k1_KO_s012
                                 + 0.686523 w_k1_MAT_s012
                                 + 0.85801 w_k1_MSFT_s012
                                 + 0.927227 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.85775 w_k1_ADM_s012
                                 + 0.909117 w_k1_AXP_s012
                                 + 0.768684 w_k1_DIS_s012
                                 + 0.814562 w_k1_FDX_s012
                                 + 0.83323 w_k1_IBM_s012
                                 + 0.932723 w_k1_INTC_s012
                                 + 0.861653 w_k1_KO_s012
                                 + 0.974602 w_k1_MAT_s012
                                 + 0.883328 w_k1_MSFT_s012
                                 + 0.915692 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.31716 w_k1_ADM_s012
                                 + 1.4159 w_k1_AXP_s012 + 1.72184 w_k1_DIS_s012
                                 + 1.25298 w_k1_FDX_s012
                                 + 1.53005 w_k1_IBM_s012
                                 + 1.18877 w_k1_INTC_s012
                                 + 1.60339 w_k1_KO_s012 + 1.58638 w_k1_MAT_s012
                                 + 1.27111 w_k1_MSFT_s012
                                 + 1.27892 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.17436 w_k1_ADM_s012
                                 + 0.949632 w_k1_AXP_s012
                                 + 1.0488 w_k1_DIS_s012 + 1.10155 w_k1_FDX_s012
                                 + 0.742152 w_k1_IBM_s012
                                 + 0.994929 w_k1_INTC_s012
                                 + 0.834873 w_k1_KO_s012
                                 + 0.53635 w_k1_MAT_s012
                                 + 1.02671 w_k1_MSFT_s012
                                 + 1.01641 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.15795 w_k1_ADM_s013
                                 + 1.14863 w_k1_AXP_s013
                                 + 0.879369 w_k1_DIS_s013
                                 + 1.09585 w_k1_FDX_s013
                                 + 0.93608 w_k1_IBM_s013
                                 + 1.02816 w_k1_INTC_s013
                                 + 1.05684 w_k1_KO_s013 + 1.16801 w_k1_MAT_s013
                                 + 1.11043 w_k1_MSFT_s013
                                 + 0.972354 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.827194 w_k1_ADM_s013
                                 + 0.676568 w_k1_AXP_s013
                                 + 0.554669 w_k1_DIS_s013
                                 + 0.769228 w_k1_FDX_s013
                                 + 0.697992 w_k1_IBM_s013
                                 + 0.744694 w_k1_INTC_s013
                                 + 0.78097 w_k1_KO_s013
                                 + 0.735737 w_k1_MAT_s013
                                 + 0.739869 w_k1_MSFT_s013
                                 + 0.844802 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.888878 w_k1_ADM_s013
                                 + 0.871699 w_k1_AXP_s013
                                 + 0.678458 w_k1_DIS_s013
                                 + 0.791501 w_k1_FDX_s013
                                 + 1.0414 w_k1_IBM_s013
                                 + 0.724095 w_k1_INTC_s013
                                 + 0.967521 w_k1_KO_s013
                                 + 1.15855 w_k1_MAT_s013
                                 + 0.927033 w_k1_MSFT_s013
                                 + 0.931444 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.16236 w_k1_ADM_s013
                                 + 0.973233 w_k1_AXP_s013
                                 + 0.987183 w_k1_DIS_s013
                                 + 1.00425 w_k1_FDX_s013
                                 + 0.836625 w_k1_IBM_s013
                                 + 0.863912 w_k1_INTC_s013
                                 + 0.99709 w_k1_KO_s013
                                 + 0.996289 w_k1_MAT_s013
                                 + 1.04638 w_k1_MSFT_s013
                                 + 0.991525 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.08364 w_k1_ADM_s013
                                 + 1.37834 w_k1_AXP_s013
                                 + 1.57856 w_k1_DIS_s013
                                 + 1.13487 w_k1_FDX_s013
                                 + 1.72875 w_k1_IBM_s013
                                 + 1.08973 w_k1_INTC_s013
                                 + 1.72863 w_k1_KO_s013 + 2.0518 w_k1_MAT_s013
                                 + 1.40563 w_k1_MSFT_s013
                                 + 1.24772 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.32872 w_k1_ADM_s013
                                 + 1.25449 w_k1_AXP_s013
                                 + 1.35162 w_k1_DIS_s013
                                 + 1.14361 w_k1_FDX_s013
                                 + 1.22976 w_k1_IBM_s013
                                 + 1.08775 w_k1_INTC_s013
                                 + 1.31992 w_k1_KO_s013 + 1.14045 w_k1_MAT_s013
                                 + 1.24116 w_k1_MSFT_s013
                                 + 1.04787 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.01767 w_k1_ADM_s013
                                 + 1.12326 w_k1_AXP_s013
                                 + 0.954897 w_k1_DIS_s013
                                 + 0.934766 w_k1_FDX_s013
                                 + 1.20624 w_k1_IBM_s013
                                 + 0.979425 w_k1_INTC_s013
                                 + 1.34662 w_k1_KO_s013 + 1.35221 w_k1_MAT_s013
                                 + 1.11069 w_k1_MSFT_s013
                                 + 1.02039 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.41544 w_k1_ADM_s013
                                 + 1.71018 w_k1_AXP_s013
                                 + 1.63776 w_k1_DIS_s013
                                 + 1.42166 w_k1_FDX_s013
                                 + 1.15041 w_k1_IBM_s013
                                 + 1.40791 w_k1_INTC_s013
                                 + 1.80157 w_k1_KO_s013 + 1.60587 w_k1_MAT_s013
                                 + 1.59332 w_k1_MSFT_s013
                                 + 1.40962 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.07105 w_k1_ADM_s014
                                 + 1.13131 w_k1_AXP_s014
                                 + 1.33409 w_k1_DIS_s014 + 1.0525 w_k1_FDX_s014
                                 + 1.20015 w_k1_IBM_s014
                                 + 1.21389 w_k1_INTC_s014
                                 + 1.31275 w_k1_KO_s014 + 1.22942 w_k1_MAT_s014
                                 + 1.27778 w_k1_MSFT_s014
                                 + 1.11179 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.984531 w_k1_ADM_s014
                                 + 0.91776 w_k1_AXP_s014
                                 + 0.741935 w_k1_DIS_s014
                                 + 0.835885 w_k1_FDX_s014
                                 + 1.1713 w_k1_IBM_s014
                                 + 0.922441 w_k1_INTC_s014
                                 + 0.983758 w_k1_KO_s014 + 1.2651 w_k1_MAT_s014
                                 + 0.875109 w_k1_MSFT_s014
                                 + 0.903023 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.20742 w_k1_ADM_s014
                                 + 1.40317 w_k1_AXP_s014
                                 + 1.55386 w_k1_DIS_s014
                                 + 1.35538 w_k1_FDX_s014
                                 + 1.01288 w_k1_IBM_s014
                                 + 1.2392 w_k1_INTC_s014 + 1.08764 w_k1_KO_s014
                                 + 1.43216 w_k1_MAT_s014
                                 + 1.38116 w_k1_MSFT_s014
                                 + 1.05318 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.30019 w_k1_ADM_s014
                                 + 0.994514 w_k1_AXP_s014
                                 + 1.10419 w_k1_DIS_s014
                                 + 1.19738 w_k1_FDX_s014
                                 + 0.832699 w_k1_IBM_s014
                                 + 1.17889 w_k1_INTC_s014
                                 + 0.994575 w_k1_KO_s014
                                 + 0.772181 w_k1_MAT_s014
                                 + 1.21384 w_k1_MSFT_s014
                                 + 0.987607 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.940487 w_k1_ADM_s014
                                 + 0.555429 w_k1_AXP_s014
                                 + 0.720486 w_k1_DIS_s014
                                 + 0.68059 w_k1_FDX_s014
                                 + 0.795132 w_k1_IBM_s014
                                 + 0.765499 w_k1_INTC_s014
                                 + 0.866996 w_k1_KO_s014
                                 + 0.768617 w_k1_MAT_s014
                                 + 0.770172 w_k1_MSFT_s014
                                 + 0.815839 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.10546 w_k1_ADM_s014
                                 + 0.926523 w_k1_AXP_s014
                                 + 0.99024 w_k1_DIS_s014
                                 + 0.90826 w_k1_FDX_s014
                                 + 0.87271 w_k1_IBM_s014
                                 + 1.05794 w_k1_INTC_s014
                                 + 1.01051 w_k1_KO_s014
                                 + 0.989066 w_k1_MAT_s014
                                 + 1.07531 w_k1_MSFT_s014
                                 + 1.04766 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.867761 w_k1_ADM_s014
                                 + 0.764443 w_k1_AXP_s014
                                 + 0.753798 w_k1_DIS_s014
                                 + 0.839729 w_k1_FDX_s014
                                 + 0.658126 w_k1_IBM_s014
                                 + 1.12697 w_k1_INTC_s014
                                 + 0.695594 w_k1_KO_s014
                                 + 0.583827 w_k1_MAT_s014
                                 + 0.963288 w_k1_MSFT_s014
                                 + 0.865828 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.06329 w_k1_ADM_s014
                                 + 1.33326 w_k1_AXP_s014
                                 + 1.14844 w_k1_DIS_s014
                                 + 0.977986 w_k1_FDX_s014
                                 + 1.62862 w_k1_IBM_s014
                                 + 1.06153 w_k1_INTC_s014
                                 + 1.28982 w_k1_KO_s014 + 1.9795 w_k1_MAT_s014
                                 + 1.06207 w_k1_MSFT_s014
                                 + 1.22603 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.06189 w_k1_ADM_s015
                                 + 0.929062 w_k1_AXP_s015
                                 + 0.813583 w_k1_DIS_s015
                                 + 0.954894 w_k1_FDX_s015
                                 + 0.690871 w_k1_IBM_s015
                                 + 0.731783 w_k1_INTC_s015
                                 + 0.683309 w_k1_KO_s015
                                 + 0.86722 w_k1_MAT_s015
                                 + 0.839571 w_k1_MSFT_s015
                                 + 0.860887 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.03575 w_k1_ADM_s015
                                 + 1.01871 w_k1_AXP_s015 + 1.4612 w_k1_DIS_s015
                                 + 1.15268 w_k1_FDX_s015
                                 + 1.34495 w_k1_IBM_s015
                                 + 0.879863 w_k1_INTC_s015
                                 + 1.32117 w_k1_KO_s015 + 1.42511 w_k1_MAT_s015
                                 + 1.12966 w_k1_MSFT_s015
                                 + 1.13907 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.29948 w_k1_ADM_s015
                                 + 1.13749 w_k1_AXP_s015
                                 + 1.34692 w_k1_DIS_s015 + 1.27 w_k1_FDX_s015
                                 + 0.79303 w_k1_IBM_s015
                                 + 1.27687 w_k1_INTC_s015
                                 + 1.10559 w_k1_KO_s015
                                 + 0.921146 w_k1_MAT_s015
                                 + 1.3166 w_k1_MSFT_s015
                                 + 0.997159 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.850027 w_k1_ADM_s015
                                 + 0.684295 w_k1_AXP_s015
                                 + 0.731175 w_k1_DIS_s015
                                 + 0.790544 w_k1_FDX_s015
                                 + 0.891531 w_k1_IBM_s015
                                 + 0.71013 w_k1_INTC_s015
                                 + 0.916491 w_k1_KO_s015
                                 + 1.05117 w_k1_MAT_s015
                                 + 0.904588 w_k1_MSFT_s015
                                 + 0.945697 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.834465 w_k1_ADM_s015
                                 + 0.591144 w_k1_AXP_s015
                                 + 0.744365 w_k1_DIS_s015
                                 + 0.872125 w_k1_FDX_s015
                                 + 0.580306 w_k1_IBM_s015
                                 + 0.69783 w_k1_INTC_s015
                                 + 0.733835 w_k1_KO_s015
                                 + 0.701317 w_k1_MAT_s015
                                 + 0.753151 w_k1_MSFT_s015
                                 + 0.812445 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.884869 w_k1_ADM_s015
                                 + 0.8188 w_k1_AXP_s015
                                 + 0.773239 w_k1_DIS_s015
                                 + 0.758946 w_k1_FDX_s015
                                 + 1.25303 w_k1_IBM_s015
                                 + 0.988369 w_k1_INTC_s015
                                 + 0.989563 w_k1_KO_s015
                                 + 1.13631 w_k1_MAT_s015
                                 + 0.812574 w_k1_MSFT_s015
                                 + 0.86017 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.980165 w_k1_ADM_s015
                                 + 1.10106 w_k1_AXP_s015
                                 + 0.929604 w_k1_DIS_s015
                                 + 1.07262 w_k1_FDX_s015
                                 + 0.94385 w_k1_IBM_s015
                                 + 0.936712 w_k1_INTC_s015
                                 + 0.894594 w_k1_KO_s015
                                 + 1.48279 w_k1_MAT_s015
                                 + 0.961798 w_k1_MSFT_s015
                                 + 0.989671 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.909297 w_k1_ADM_s015
                                 + 0.59642 w_k1_AXP_s015
                                 + 0.869609 w_k1_DIS_s015
                                 + 0.848537 w_k1_FDX_s015
                                 + 0.838774 w_k1_IBM_s015
                                 + 1.08294 w_k1_INTC_s015
                                 + 0.954673 w_k1_KO_s015
                                 + 0.786515 w_k1_MAT_s015
                                 + 0.900606 w_k1_MSFT_s015
                                 + 0.903411 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.06298 w_k1_ADM_s016
                                 + 1.15443 w_k1_AXP_s016
                                 + 0.827116 w_k1_DIS_s016
                                 + 0.902437 w_k1_FDX_s016
                                 + 1.22872 w_k1_IBM_s016
                                 + 0.996325 w_k1_INTC_s016
                                 + 0.940508 w_k1_KO_s016
                                 + 1.23679 w_k1_MAT_s016
                                 + 0.886732 w_k1_MSFT_s016
                                 + 0.885765 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.37159 w_k1_ADM_s016
                                 + 1.76567 w_k1_AXP_s016
                                 + 1.63758 w_k1_DIS_s016
                                 + 1.43551 w_k1_FDX_s016
                                 + 1.73003 w_k1_IBM_s016
                                 + 1.3686 w_k1_INTC_s016 + 1.81485 w_k1_KO_s016
                                 + 1.52198 w_k1_MAT_s016
                                 + 1.29508 w_k1_MSFT_s016
                                 + 1.33169 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.982895 w_k1_ADM_s016
                                 + 0.895993 w_k1_AXP_s016
                                 + 0.720667 w_k1_DIS_s016
                                 + 0.887933 w_k1_FDX_s016
                                 + 0.760975 w_k1_IBM_s016
                                 + 1.03347 w_k1_INTC_s016
                                 + 0.805935 w_k1_KO_s016
                                 + 0.759272 w_k1_MAT_s016
                                 + 0.805734 w_k1_MSFT_s016
                                 + 0.852759 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.30685 w_k1_ADM_s016
                                 + 1.10895 w_k1_AXP_s016
                                 + 1.60138 w_k1_DIS_s016
                                 + 1.37116 w_k1_FDX_s016
                                 + 1.05054 w_k1_IBM_s016
                                 + 1.25028 w_k1_INTC_s016 + 1.2557 w_k1_KO_s016
                                 + 1.05566 w_k1_MAT_s016
                                 + 1.24662 w_k1_MSFT_s016
                                 + 1.14245 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.15689 w_k1_ADM_s016
                                 + 0.838124 w_k1_AXP_s016
                                 + 0.924965 w_k1_DIS_s016
                                 + 0.995213 w_k1_FDX_s016
                                 + 0.774146 w_k1_IBM_s016
                                 + 1.08384 w_k1_INTC_s016
                                 + 0.909016 w_k1_KO_s016
                                 + 0.572292 w_k1_MAT_s016
                                 + 0.921375 w_k1_MSFT_s016
                                 + 0.868313 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.26255 w_k1_ADM_s016
                                 + 1.14376 w_k1_AXP_s016
                                 + 1.24313 w_k1_DIS_s016
                                 + 1.05801 w_k1_FDX_s016
                                 + 1.39454 w_k1_IBM_s016
                                 + 1.35471 w_k1_INTC_s016
                                 + 1.34792 w_k1_KO_s016
                                 + 0.952794 w_k1_MAT_s016
                                 + 1.03958 w_k1_MSFT_s016
                                 + 0.923982 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.09156 w_k1_ADM_s016
                                 + 1.10688 w_k1_AXP_s016 + 1.0788 w_k1_DIS_s016
                                 + 1.13157 w_k1_FDX_s016
                                 + 0.995997 w_k1_IBM_s016
                                 + 1.24389 w_k1_INTC_s016 + 1.1142 w_k1_KO_s016
                                 + 0.884717 w_k1_MAT_s016
                                 + 1.06533 w_k1_MSFT_s016
                                 + 0.887728 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.738919 w_k1_ADM_s016
                                 + 0.582672 w_k1_AXP_s016
                                 + 0.57168 w_k1_DIS_s016
                                 + 0.719942 w_k1_FDX_s016
                                 + 0.73637 w_k1_IBM_s016
                                 + 0.8379 w_k1_INTC_s016
                                 + 0.749711 w_k1_KO_s016
                                 + 0.667162 w_k1_MAT_s016
                                 + 0.75664 w_k1_MSFT_s016
                                 + 0.718471 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.96344 w_k1_ADM_s017
                                 + 1.00344 w_k1_AXP_s017
                                 + 0.718065 w_k1_DIS_s017
                                 + 0.954216 w_k1_FDX_s017
                                 + 0.800599 w_k1_IBM_s017
                                 + 0.971066 w_k1_INTC_s017
                                 + 0.828675 w_k1_KO_s017
                                 + 0.984749 w_k1_MAT_s017
                                 + 1.00607 w_k1_MSFT_s017
                                 + 1.01395 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.968645 w_k1_ADM_s017
                                 + 1.21016 w_k1_AXP_s017
                                 + 1.14416 w_k1_DIS_s017
                                 + 1.07158 w_k1_FDX_s017
                                 + 1.37498 w_k1_IBM_s017
                                 + 1.19674 w_k1_INTC_s017
                                 + 1.19313 w_k1_KO_s017 + 1.67089 w_k1_MAT_s017
                                 + 1.0785 w_k1_MSFT_s017 + 1.08755 w_k1_PG_s017
                                 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.06365 w_k1_ADM_s017
                                 + 0.836469 w_k1_AXP_s017
                                 + 0.916038 w_k1_DIS_s017
                                 + 0.848831 w_k1_FDX_s017
                                 + 1.26122 w_k1_IBM_s017
                                 + 0.911198 w_k1_INTC_s017
                                 + 1.02507 w_k1_KO_s017 + 1.00691 w_k1_MAT_s017
                                 + 0.916847 w_k1_MSFT_s017
                                 + 1.00561 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.17298 w_k1_ADM_s017
                                 + 1.38614 w_k1_AXP_s017
                                 + 1.74587 w_k1_DIS_s017
                                 + 1.17521 w_k1_FDX_s017 + 1.4776 w_k1_IBM_s017
                                 + 1.51203 w_k1_INTC_s017
                                 + 1.42035 w_k1_KO_s017 + 1.28675 w_k1_MAT_s017
                                 + 1.28216 w_k1_MSFT_s017
                                 + 1.27414 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.808864 w_k1_ADM_s017
                                 + 0.565952 w_k1_AXP_s017
                                 + 0.579317 w_k1_DIS_s017
                                 + 0.766122 w_k1_FDX_s017
                                 + 0.553444 w_k1_IBM_s017
                                 + 0.953478 w_k1_INTC_s017
                                 + 0.682353 w_k1_KO_s017
                                 + 0.469844 w_k1_MAT_s017
                                 + 0.859904 w_k1_MSFT_s017
                                 + 0.856479 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.05936 w_k1_ADM_s017
                                 + 0.665708 w_k1_AXP_s017
                                 + 0.740377 w_k1_DIS_s017
                                 + 0.808788 w_k1_FDX_s017
                                 + 0.751709 w_k1_IBM_s017
                                 + 0.821772 w_k1_INTC_s017
                                 + 0.727822 w_k1_KO_s017
                                 + 0.546998 w_k1_MAT_s017
                                 + 0.864008 w_k1_MSFT_s017
                                 + 0.871242 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.979932 w_k1_ADM_s017
                                 + 0.990348 w_k1_AXP_s017
                                 + 0.998025 w_k1_DIS_s017
                                 + 1.02738 w_k1_FDX_s017
                                 + 0.770282 w_k1_IBM_s017
                                 + 1.25912 w_k1_INTC_s017
                                 + 0.976217 w_k1_KO_s017 + 0.6636 w_k1_MAT_s017
                                 + 1.00509 w_k1_MSFT_s017
                                 + 0.983635 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.37915 w_k1_ADM_s017
                                 + 1.14717 w_k1_AXP_s017
                                 + 1.35758 w_k1_DIS_s017
                                 + 1.17947 w_k1_FDX_s017
                                 + 0.881972 w_k1_IBM_s017
                                 + 1.20175 w_k1_INTC_s017
                                 + 1.10109 w_k1_KO_s017
                                 + 0.991621 w_k1_MAT_s017
                                 + 1.21814 w_k1_MSFT_s017
                                 + 1.09091 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.779022 w_k1_ADM_s100
                                 + 0.958352 w_k1_AXP_s100
                                 + 0.718655 w_k1_DIS_s100
                                 + 0.910002 w_k1_FDX_s100
                                 + 0.667101 w_k1_IBM_s100
                                 + 0.827026 w_k1_INTC_s100
                                 + 0.654673 w_k1_KO_s100
                                 + 0.880726 w_k1_MAT_s100
                                 + 0.889339 w_k1_MSFT_s100
                                 + 0.872289 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.27342 w_k1_ADM_s100
                                 + 1.5963 w_k1_AXP_s100 + 1.36465 w_k1_DIS_s100
                                 + 1.23063 w_k1_FDX_s100
                                 + 1.36045 w_k1_IBM_s100
                                 + 1.14412 w_k1_INTC_s100
                                 + 1.19267 w_k1_KO_s100 + 1.50414 w_k1_MAT_s100
                                 + 1.21123 w_k1_MSFT_s100
                                 + 1.08372 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.973485 w_k1_ADM_s100
                                 + 1.03514 w_k1_AXP_s100
                                 + 0.818273 w_k1_DIS_s100
                                 + 0.879532 w_k1_FDX_s100
                                 + 1.39633 w_k1_IBM_s100
                                 + 0.785192 w_k1_INTC_s100
                                 + 1.02715 w_k1_KO_s100 + 2.00218 w_k1_MAT_s100
                                 + 0.842024 w_k1_MSFT_s100
                                 + 0.976549 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.963495 w_k1_ADM_s100
                                 + 0.981638 w_k1_AXP_s100
                                 + 0.974915 w_k1_DIS_s100
                                 + 1.04637 w_k1_FDX_s100
                                 + 0.769471 w_k1_IBM_s100
                                 + 0.908916 w_k1_INTC_s100
                                 + 0.882795 w_k1_KO_s100
                                 + 0.851573 w_k1_MAT_s100
                                 + 1.08141 w_k1_MSFT_s100
                                 + 0.961298 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.700279 w_k1_ADM_s100
                                 + 0.761576 w_k1_AXP_s100
                                 + 0.749791 w_k1_DIS_s100
                                 + 0.814169 w_k1_FDX_s100
                                 + 1.01387 w_k1_IBM_s100
                                 + 0.96724 w_k1_INTC_s100
                                 + 0.788324 w_k1_KO_s100
                                 + 0.802486 w_k1_MAT_s100
                                 + 0.829598 w_k1_MSFT_s100
                                 + 0.915543 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.16814 w_k1_ADM_s100
                                 + 1.21478 w_k1_AXP_s100
                                 + 1.04381 w_k1_DIS_s100
                                 + 1.14551 w_k1_FDX_s100
                                 + 0.996337 w_k1_IBM_s100
                                 + 1.10722 w_k1_INTC_s100
                                 + 1.05581 w_k1_KO_s100 + 1.11486 w_k1_MAT_s100
                                 + 1.15063 w_k1_MSFT_s100
                                 + 1.13508 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.861764 w_k1_ADM_s100
                                 + 1.30253 w_k1_AXP_s100
                                 + 1.38696 w_k1_DIS_s100
                                 + 1.36648 w_k1_FDX_s100
                                 + 0.804364 w_k1_IBM_s100
                                 + 1.11077 w_k1_INTC_s100
                                 + 0.976386 w_k1_KO_s100
                                 + 1.41009 w_k1_MAT_s100
                                 + 1.24138 w_k1_MSFT_s100
                                 + 1.12511 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.639398 w_k1_ADM_s100
                                 + 0.746553 w_k1_AXP_s100
                                 + 0.611412 w_k1_DIS_s100
                                 + 0.762643 w_k1_FDX_s100
                                 + 0.701497 w_k1_IBM_s100
                                 + 0.746441 w_k1_INTC_s100
                                 + 0.638441 w_k1_KO_s100
                                 + 0.963065 w_k1_MAT_s100
                                 + 0.781737 w_k1_MSFT_s100
                                 + 0.826017 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.928898 w_k1_ADM_s101
                                 + 1.16877 w_k1_AXP_s101
                                 + 0.924282 w_k1_DIS_s101
                                 + 0.977153 w_k1_FDX_s101
                                 + 0.949981 w_k1_IBM_s101
                                 + 0.927299 w_k1_INTC_s101
                                 + 0.80533 w_k1_KO_s101 + 1.34687 w_k1_MAT_s101
                                 + 0.962369 w_k1_MSFT_s101
                                 + 1.02084 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.854251 w_k1_ADM_s101
                                 + 1.81735 w_k1_AXP_s101
                                 + 1.71886 w_k1_DIS_s101
                                 + 1.32355 w_k1_FDX_s101
                                 + 1.17764 w_k1_IBM_s101
                                 + 1.40299 w_k1_INTC_s101
                                 + 1.24709 w_k1_KO_s101 + 1.86839 w_k1_MAT_s101
                                 + 1.15951 w_k1_MSFT_s101
                                 + 1.28621 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.29402 w_k1_ADM_s101
                                 + 0.949476 w_k1_AXP_s101
                                 + 1.62303 w_k1_DIS_s101
                                 + 1.35693 w_k1_FDX_s101
                                 + 0.820295 w_k1_IBM_s101
                                 + 1.20786 w_k1_INTC_s101 + 1.0376 w_k1_KO_s101
                                 + 0.867506 w_k1_MAT_s101
                                 + 1.269 w_k1_MSFT_s101 + 1.17819 w_k1_PG_s101
                                 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.847425 w_k1_ADM_s101
                                 + 1.35655 w_k1_AXP_s101
                                 + 1.12135 w_k1_DIS_s101
                                 + 1.12502 w_k1_FDX_s101
                                 + 1.20202 w_k1_IBM_s101
                                 + 1.02007 w_k1_INTC_s101
                                 + 0.996805 w_k1_KO_s101
                                 + 1.48893 w_k1_MAT_s101
                                 + 0.943603 w_k1_MSFT_s101
                                 + 1.28312 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.939992 w_k1_ADM_s101
                                 + 1.21002 w_k1_AXP_s101
                                 + 0.886543 w_k1_DIS_s101
                                 + 0.90688 w_k1_FDX_s101 + 1.6001 w_k1_IBM_s101
                                 + 0.788146 w_k1_INTC_s101
                                 + 0.97544 w_k1_KO_s101 + 2.26363 w_k1_MAT_s101
                                 + 0.924664 w_k1_MSFT_s101
                                 + 1.19123 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.822878 w_k1_ADM_s101
                                 + 0.829324 w_k1_AXP_s101
                                 + 0.825499 w_k1_DIS_s101
                                 + 0.925404 w_k1_FDX_s101
                                 + 0.770617 w_k1_IBM_s101
                                 + 0.979406 w_k1_INTC_s101
                                 + 0.758013 w_k1_KO_s101
                                 + 0.680039 w_k1_MAT_s101
                                 + 0.878479 w_k1_MSFT_s101
                                 + 1.04015 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.985822 w_k1_ADM_s101
                                 + 1.30683 w_k1_AXP_s101
                                 + 0.947566 w_k1_DIS_s101
                                 + 1.24061 w_k1_FDX_s101
                                 + 0.599112 w_k1_IBM_s101
                                 + 0.935862 w_k1_INTC_s101
                                 + 0.623454 w_k1_KO_s101
                                 + 0.783913 w_k1_MAT_s101
                                 + 0.922156 w_k1_MSFT_s101
                                 + 1.11462 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.736962 w_k1_ADM_s101
                                 + 0.727956 w_k1_AXP_s101
                                 + 0.739944 w_k1_DIS_s101
                                 + 0.962029 w_k1_FDX_s101
                                 + 0.408982 w_k1_IBM_s101
                                 + 0.712918 w_k1_INTC_s101
                                 + 0.484266 w_k1_KO_s101
                                 + 0.457572 w_k1_MAT_s101
                                 + 0.850456 w_k1_MSFT_s101
                                 + 0.946418 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.721404 w_k1_ADM_s102
                                 + 0.712767 w_k1_AXP_s102
                                 + 0.692059 w_k1_DIS_s102
                                 + 0.821362 w_k1_FDX_s102
                                 + 0.816355 w_k1_IBM_s102
                                 + 0.749278 w_k1_INTC_s102
                                 + 0.742454 w_k1_KO_s102
                                 + 0.85857 w_k1_MAT_s102
                                 + 0.729958 w_k1_MSFT_s102
                                 + 0.960604 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.989911 w_k1_ADM_s102
                                 + 0.923561 w_k1_AXP_s102
                                 + 0.930223 w_k1_DIS_s102
                                 + 0.940113 w_k1_FDX_s102
                                 + 0.984175 w_k1_IBM_s102
                                 + 0.772522 w_k1_INTC_s102
                                 + 0.883487 w_k1_KO_s102
                                 + 0.883221 w_k1_MAT_s102
                                 + 0.864889 w_k1_MSFT_s102
                                 + 0.935515 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.08445 w_k1_ADM_s102
                                 + 1.14162 w_k1_AXP_s102
                                 + 1.80528 w_k1_DIS_s102 + 1.4458 w_k1_FDX_s102
                                 + 1.17179 w_k1_IBM_s102
                                 + 1.14851 w_k1_INTC_s102
                                 + 1.42845 w_k1_KO_s102 + 1.20888 w_k1_MAT_s102
                                 + 1.44966 w_k1_MSFT_s102
                                 + 1.36984 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.822989 w_k1_ADM_s102
                                 + 0.862344 w_k1_AXP_s102
                                 + 0.936564 w_k1_DIS_s102
                                 + 1.07707 w_k1_FDX_s102
                                 + 0.709982 w_k1_IBM_s102
                                 + 0.941005 w_k1_INTC_s102
                                 + 0.862499 w_k1_KO_s102
                                 + 0.844167 w_k1_MAT_s102
                                 + 0.942981 w_k1_MSFT_s102
                                 + 1.09579 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.847457 w_k1_ADM_s102
                                 + 0.98725 w_k1_AXP_s102
                                 + 1.01643 w_k1_DIS_s102
                                 + 0.931284 w_k1_FDX_s102
                                 + 1.67973 w_k1_IBM_s102
                                 + 1.05382 w_k1_INTC_s102
                                 + 1.26311 w_k1_KO_s102 + 1.61321 w_k1_MAT_s102
                                 + 0.974713 w_k1_MSFT_s102
                                 + 1.16796 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.862801 w_k1_ADM_s102
                                 + 1.23087 w_k1_AXP_s102
                                 + 1.21484 w_k1_DIS_s102
                                 + 1.17342 w_k1_FDX_s102
                                 + 1.23967 w_k1_IBM_s102
                                 + 1.01655 w_k1_INTC_s102
                                 + 1.09933 w_k1_KO_s102 + 1.46998 w_k1_MAT_s102
                                 + 0.961524 w_k1_MSFT_s102
                                 + 1.16147 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.0112 w_k1_ADM_s102
                                 + 1.2991 w_k1_AXP_s102 + 1.0287 w_k1_DIS_s102
                                 + 1.17209 w_k1_FDX_s102
                                 + 0.857932 w_k1_IBM_s102
                                 + 0.946856 w_k1_INTC_s102
                                 + 0.88085 w_k1_KO_s102 + 1.16885 w_k1_MAT_s102
                                 + 1.00092 w_k1_MSFT_s102
                                 + 1.08576 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.991035 w_k1_ADM_s102
                                 + 0.980572 w_k1_AXP_s102
                                 + 0.853111 w_k1_DIS_s102
                                 + 0.944353 w_k1_FDX_s102
                                 + 1.17085 w_k1_IBM_s102
                                 + 0.886624 w_k1_INTC_s102
                                 + 0.845089 w_k1_KO_s102
                                 + 0.934238 w_k1_MAT_s102
                                 + 0.81478 w_k1_MSFT_s102
                                 + 0.885868 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.12309 w_k1_ADM_s103
                                 + 0.784451 w_k1_AXP_s103
                                 + 1.04985 w_k1_DIS_s103
                                 + 0.97239 w_k1_FDX_s103
                                 + 0.960668 w_k1_IBM_s103
                                 + 0.970431 w_k1_INTC_s103
                                 + 1.00595 w_k1_KO_s103 + 1.05326 w_k1_MAT_s103
                                 + 1.01926 w_k1_MSFT_s103
                                 + 1.03976 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.16858 w_k1_ADM_s103
                                 + 0.922761 w_k1_AXP_s103
                                 + 0.775254 w_k1_DIS_s103
                                 + 0.937235 w_k1_FDX_s103
                                 + 0.72523 w_k1_IBM_s103
                                 + 0.798502 w_k1_INTC_s103
                                 + 0.704771 w_k1_KO_s103
                                 + 0.867093 w_k1_MAT_s103
                                 + 0.799183 w_k1_MSFT_s103
                                 + 0.910473 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.850685 w_k1_ADM_s103
                                 + 1.33476 w_k1_AXP_s103
                                 + 0.953429 w_k1_DIS_s103
                                 + 1.01016 w_k1_FDX_s103
                                 + 1.07552 w_k1_IBM_s103
                                 + 1.17593 w_k1_INTC_s103
                                 + 0.980395 w_k1_KO_s103
                                 + 1.23762 w_k1_MAT_s103
                                 + 1.06669 w_k1_MSFT_s103
                                 + 0.960486 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.89009 w_k1_ADM_s103
                                 + 0.711563 w_k1_AXP_s103
                                 + 0.818351 w_k1_DIS_s103
                                 + 0.807628 w_k1_FDX_s103
                                 + 0.780779 w_k1_IBM_s103
                                 + 1.05162 w_k1_INTC_s103
                                 + 0.797631 w_k1_KO_s103
                                 + 0.541445 w_k1_MAT_s103
                                 + 0.889033 w_k1_MSFT_s103
                                 + 0.87697 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.19659 w_k1_ADM_s103
                                 + 0.960711 w_k1_AXP_s103
                                 + 0.943854 w_k1_DIS_s103
                                 + 0.909702 w_k1_FDX_s103
                                 + 1.33734 w_k1_IBM_s103
                                 + 0.939764 w_k1_INTC_s103
                                 + 1.05424 w_k1_KO_s103 + 1.43524 w_k1_MAT_s103
                                 + 0.997787 w_k1_MSFT_s103
                                 + 0.90176 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.318 w_k1_ADM_s103
                                 + 1.41042 w_k1_AXP_s103
                                 + 1.49723 w_k1_DIS_s103
                                 + 1.34387 w_k1_FDX_s103
                                 + 1.16534 w_k1_IBM_s103
                                 + 1.10694 w_k1_INTC_s103
                                 + 1.30472 w_k1_KO_s103 + 1.55699 w_k1_MAT_s103
                                 + 1.33951 w_k1_MSFT_s103
                                 + 1.14589 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.25822 w_k1_ADM_s103
                                 + 1.19318 w_k1_AXP_s103
                                 + 1.09555 w_k1_DIS_s103
                                 + 1.11775 w_k1_FDX_s103
                                 + 0.939396 w_k1_IBM_s103
                                 + 0.913061 w_k1_INTC_s103
                                 + 0.816941 w_k1_KO_s103
                                 + 1.15006 w_k1_MAT_s103
                                 + 1.07388 w_k1_MSFT_s103
                                 + 1.06372 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.823119 w_k1_ADM_s103
                                 + 0.768773 w_k1_AXP_s103
                                 + 0.626767 w_k1_DIS_s103
                                 + 0.686769 w_k1_FDX_s103
                                 + 0.846238 w_k1_IBM_s103
                                 + 1.01138 w_k1_INTC_s103
                                 + 0.766908 w_k1_KO_s103
                                 + 0.844061 w_k1_MAT_s103
                                 + 0.788687 w_k1_MSFT_s103
                                 + 0.840517 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.891071 w_k1_ADM_s104
                                 + 0.97966 w_k1_AXP_s104
                                 + 1.09728 w_k1_DIS_s104
                                 + 1.17947 w_k1_FDX_s104
                                 + 1.08914 w_k1_IBM_s104
                                 + 1.23745 w_k1_INTC_s104 + 1.1107 w_k1_KO_s104
                                 + 1.33839 w_k1_MAT_s104
                                 + 1.09797 w_k1_MSFT_s104 + 1.1135 w_k1_PG_s104
                                 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.932887 w_k1_ADM_s104
                                 + 0.914696 w_k1_AXP_s104
                                 + 0.791816 w_k1_DIS_s104
                                 + 0.905454 w_k1_FDX_s104
                                 + 1.18986 w_k1_IBM_s104
                                 + 0.825502 w_k1_INTC_s104
                                 + 0.857247 w_k1_KO_s104
                                 + 1.05019 w_k1_MAT_s104
                                 + 0.818663 w_k1_MSFT_s104
                                 + 0.928099 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.878434 w_k1_ADM_s104
                                 + 0.694456 w_k1_AXP_s104
                                 + 0.688313 w_k1_DIS_s104
                                 + 0.862485 w_k1_FDX_s104
                                 + 0.852485 w_k1_IBM_s104
                                 + 0.956492 w_k1_INTC_s104
                                 + 0.89601 w_k1_KO_s104
                                 + 0.773296 w_k1_MAT_s104
                                 + 0.858308 w_k1_MSFT_s104
                                 + 0.895985 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.76522 w_k1_ADM_s104
                                 + 1.54281 w_k1_AXP_s104
                                 + 1.52493 w_k1_DIS_s104
                                 + 1.34732 w_k1_FDX_s104
                                 + 1.50787 w_k1_IBM_s104 + 1.099 w_k1_INTC_s104
                                 + 1.52792 w_k1_KO_s104 + 1.87394 w_k1_MAT_s104
                                 + 1.21788 w_k1_MSFT_s104 + 1.2556 w_k1_PG_s104
                                 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.38366 w_k1_ADM_s104
                                 + 1.02966 w_k1_AXP_s104
                                 + 1.14817 w_k1_DIS_s104 + 1.282 w_k1_FDX_s104
                                 + 0.661581 w_k1_IBM_s104
                                 + 1.01987 w_k1_INTC_s104
                                 + 0.893172 w_k1_KO_s104
                                 + 0.737772 w_k1_MAT_s104
                                 + 1.11619 w_k1_MSFT_s104
                                 + 1.08601 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.1894 w_k1_ADM_s104
                                 + 1.05939 w_k1_AXP_s104
                                 + 1.23255 w_k1_DIS_s104
                                 + 1.10664 w_k1_FDX_s104
                                 + 1.27346 w_k1_IBM_s104
                                 + 0.927657 w_k1_INTC_s104
                                 + 1.40751 w_k1_KO_s104 + 1.44028 w_k1_MAT_s104
                                 + 1.18735 w_k1_MSFT_s104
                                 + 1.08724 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.906724 w_k1_ADM_s104
                                 + 0.833415 w_k1_AXP_s104
                                 + 0.845625 w_k1_DIS_s104
                                 + 0.767242 w_k1_FDX_s104
                                 + 1.65014 w_k1_IBM_s104
                                 + 0.932123 w_k1_INTC_s104
                                 + 1.23872 w_k1_KO_s104 + 1.78205 w_k1_MAT_s104
                                 + 0.866184 w_k1_MSFT_s104
                                 + 0.961595 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.17864 w_k1_ADM_s104
                                 + 0.734253 w_k1_AXP_s104
                                 + 1.2892 w_k1_DIS_s104 + 1.01881 w_k1_FDX_s104
                                 + 0.864158 w_k1_IBM_s104
                                 + 0.994569 w_k1_INTC_s104
                                 + 1.21424 w_k1_KO_s104 + 0.60476 w_k1_MAT_s104
                                 + 1.17912 w_k1_MSFT_s104
                                 + 0.977651 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.790791 w_k1_ADM_s105
                                 + 1.05695 w_k1_AXP_s105
                                 + 0.765985 w_k1_DIS_s105
                                 + 0.799045 w_k1_FDX_s105
                                 + 1.21795 w_k1_IBM_s105
                                 + 0.770904 w_k1_INTC_s105
                                 + 0.84458 w_k1_KO_s105 + 1.2976 w_k1_MAT_s105
                                 + 0.717226 w_k1_MSFT_s105
                                 + 0.969826 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.17986 w_k1_ADM_s105
                                 + 1.04317 w_k1_AXP_s105
                                 + 1.14968 w_k1_DIS_s105
                                 + 1.07192 w_k1_FDX_s105
                                 + 0.849364 w_k1_IBM_s105
                                 + 0.846889 w_k1_INTC_s105
                                 + 0.870461 w_k1_KO_s105
                                 + 0.928598 w_k1_MAT_s105
                                 + 1.09064 w_k1_MSFT_s105
                                 + 1.09229 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.767311 w_k1_ADM_s105
                                 + 0.620996 w_k1_AXP_s105
                                 + 0.640738 w_k1_DIS_s105
                                 + 0.713651 w_k1_FDX_s105
                                 + 0.828689 w_k1_IBM_s105
                                 + 0.829707 w_k1_INTC_s105
                                 + 0.704493 w_k1_KO_s105
                                 + 0.583261 w_k1_MAT_s105
                                 + 0.735856 w_k1_MSFT_s105
                                 + 0.877335 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.02607 w_k1_ADM_s105
                                 + 0.856577 w_k1_AXP_s105
                                 + 1.04836 w_k1_DIS_s105
                                 + 0.918703 w_k1_FDX_s105
                                 + 1.18834 w_k1_IBM_s105
                                 + 1.0667 w_k1_INTC_s105
                                 + 0.991902 w_k1_KO_s105
                                 + 0.863698 w_k1_MAT_s105
                                 + 1.02213 w_k1_MSFT_s105
                                 + 1.01724 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.07141 w_k1_ADM_s105
                                 + 1.23729 w_k1_AXP_s105
                                 + 1.29905 w_k1_DIS_s105
                                 + 1.05827 w_k1_FDX_s105
                                 + 1.31708 w_k1_IBM_s105
                                 + 1.05771 w_k1_INTC_s105
                                 + 1.21118 w_k1_KO_s105 + 2.01723 w_k1_MAT_s105
                                 + 1.18023 w_k1_MSFT_s105
                                 + 1.13564 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.61944 w_k1_ADM_s105
                                 + 1.50637 w_k1_AXP_s105
                                 + 1.95191 w_k1_DIS_s105
                                 + 1.43499 w_k1_FDX_s105
                                 + 1.46754 w_k1_IBM_s105
                                 + 1.25493 w_k1_INTC_s105
                                 + 1.34358 w_k1_KO_s105 + 1.31557 w_k1_MAT_s105
                                 + 1.46548 w_k1_MSFT_s105
                                 + 1.25516 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.892983 w_k1_ADM_s105
                                 + 0.816736 w_k1_AXP_s105
                                 + 0.949116 w_k1_DIS_s105
                                 + 0.928254 w_k1_FDX_s105
                                 + 0.807312 w_k1_IBM_s105
                                 + 0.900443 w_k1_INTC_s105
                                 + 0.762477 w_k1_KO_s105
                                 + 0.678124 w_k1_MAT_s105
                                 + 0.88957 w_k1_MSFT_s105
                                 + 0.932534 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.23623 w_k1_ADM_s105
                                 + 1.15846 w_k1_AXP_s105
                                 + 1.60659 w_k1_DIS_s105 + 1.4397 w_k1_FDX_s105
                                 + 0.810167 w_k1_IBM_s105
                                 + 1.35901 w_k1_INTC_s105
                                 + 1.02321 w_k1_KO_s105 + 1.10839 w_k1_MAT_s105
                                 + 1.36041 w_k1_MSFT_s105
                                 + 1.29032 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.13972 w_k1_ADM_s106
                                 + 0.810254 w_k1_AXP_s106
                                 + 1.41301 w_k1_DIS_s106
                                 + 1.18191 w_k1_FDX_s106
                                 + 0.97674 w_k1_IBM_s106
                                 + 1.11752 w_k1_INTC_s106
                                 + 1.14567 w_k1_KO_s106 + 0.80242 w_k1_MAT_s106
                                 + 1.28453 w_k1_MSFT_s106
                                 + 1.03276 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.688009 w_k1_ADM_s106
                                 + 0.71784 w_k1_AXP_s106
                                 + 0.856366 w_k1_DIS_s106
                                 + 0.80214 w_k1_FDX_s106
                                 + 0.873861 w_k1_IBM_s106
                                 + 0.859334 w_k1_INTC_s106
                                 + 0.891668 w_k1_KO_s106
                                 + 0.893907 w_k1_MAT_s106
                                 + 0.864425 w_k1_MSFT_s106
                                 + 0.890461 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.0601 w_k1_ADM_s106
                                 + 0.775988 w_k1_AXP_s106
                                 + 0.880913 w_k1_DIS_s106
                                 + 1.03417 w_k1_FDX_s106
                                 + 0.651281 w_k1_IBM_s106
                                 + 0.923066 w_k1_INTC_s106
                                 + 0.804141 w_k1_KO_s106
                                 + 0.563276 w_k1_MAT_s106
                                 + 0.892609 w_k1_MSFT_s106
                                 + 1.00742 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.12623 w_k1_ADM_s106
                                 + 1.05608 w_k1_AXP_s106
                                 + 1.04048 w_k1_DIS_s106
                                 + 1.03015 w_k1_FDX_s106
                                 + 1.12584 w_k1_IBM_s106
                                 + 0.899543 w_k1_INTC_s106
                                 + 1.22673 w_k1_KO_s106 + 1.31699 w_k1_MAT_s106
                                 + 0.942571 w_k1_MSFT_s106
                                 + 1.1549 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.39636 w_k1_ADM_s106
                                 + 1.31514 w_k1_AXP_s106
                                 + 1.30947 w_k1_DIS_s106
                                 + 1.23806 w_k1_FDX_s106
                                 + 0.903177 w_k1_IBM_s106
                                 + 1.13734 w_k1_INTC_s106
                                 + 1.07651 w_k1_KO_s106
                                 + 0.928176 w_k1_MAT_s106
                                 + 1.18251 w_k1_MSFT_s106
                                 + 1.23323 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.652491 w_k1_ADM_s106
                                 + 0.660423 w_k1_AXP_s106
                                 + 0.637929 w_k1_DIS_s106
                                 + 0.665162 w_k1_FDX_s106
                                 + 0.70107 w_k1_IBM_s106
                                 + 0.762225 w_k1_INTC_s106
                                 + 0.752368 w_k1_KO_s106
                                 + 0.667739 w_k1_MAT_s106
                                 + 0.719101 w_k1_MSFT_s106
                                 + 0.76061 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.929589 w_k1_ADM_s106
                                 + 0.947949 w_k1_AXP_s106
                                 + 0.895689 w_k1_DIS_s106
                                 + 1.00939 w_k1_FDX_s106
                                 + 0.857741 w_k1_IBM_s106
                                 + 0.801341 w_k1_INTC_s106
                                 + 0.843432 w_k1_KO_s106
                                 + 0.857471 w_k1_MAT_s106
                                 + 0.846985 w_k1_MSFT_s106
                                 + 0.996067 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.13645 w_k1_ADM_s106
                                 + 0.881232 w_k1_AXP_s106
                                 + 1.17241 w_k1_DIS_s106
                                 + 0.90815 w_k1_FDX_s106
                                 + 1.41917 w_k1_IBM_s106
                                 + 0.732416 w_k1_INTC_s106
                                 + 1.1951 w_k1_KO_s106 + 1.44612 w_k1_MAT_s106
                                 + 0.836545 w_k1_MSFT_s106
                                 + 0.994555 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.964606 w_k1_ADM_s107
                                 + 0.877627 w_k1_AXP_s107
                                 + 1.08331 w_k1_DIS_s107
                                 + 0.965964 w_k1_FDX_s107
                                 + 1.18756 w_k1_IBM_s107
                                 + 1.05484 w_k1_INTC_s107
                                 + 1.16238 w_k1_KO_s107
                                 + 0.951764 w_k1_MAT_s107
                                 + 0.946943 w_k1_MSFT_s107
                                 + 1.05335 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.99867 w_k1_ADM_s107
                                 + 1.4628 w_k1_AXP_s107 + 1.10691 w_k1_DIS_s107
                                 + 1.03422 w_k1_FDX_s107
                                 + 1.49753 w_k1_IBM_s107
                                 + 1.26998 w_k1_INTC_s107
                                 + 1.29436 w_k1_KO_s107 + 1.67289 w_k1_MAT_s107
                                 + 0.910611 w_k1_MSFT_s107
                                 + 1.22234 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.03725 w_k1_ADM_s107
                                 + 0.934482 w_k1_AXP_s107
                                 + 0.836428 w_k1_DIS_s107
                                 + 0.920313 w_k1_FDX_s107
                                 + 0.781815 w_k1_IBM_s107
                                 + 0.932114 w_k1_INTC_s107
                                 + 0.857872 w_k1_KO_s107
                                 + 0.831847 w_k1_MAT_s107
                                 + 0.789265 w_k1_MSFT_s107
                                 + 1.02271 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.31888 w_k1_ADM_s107
                                 + 1.18316 w_k1_AXP_s107
                                 + 1.10395 w_k1_DIS_s107
                                 + 1.15075 w_k1_FDX_s107
                                 + 0.935072 w_k1_IBM_s107
                                 + 1.22721 w_k1_INTC_s107
                                 + 0.998555 w_k1_KO_s107
                                 + 0.786478 w_k1_MAT_s107
                                 + 0.956354 w_k1_MSFT_s107
                                 + 1.09151 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.863451 w_k1_ADM_s107
                                 + 0.672726 w_k1_AXP_s107
                                 + 0.77491 w_k1_DIS_s107
                                 + 0.732256 w_k1_FDX_s107
                                 + 0.998298 w_k1_IBM_s107
                                 + 0.767441 w_k1_INTC_s107
                                 + 0.946158 w_k1_KO_s107
                                 + 0.878488 w_k1_MAT_s107
                                 + 0.756164 w_k1_MSFT_s107
                                 + 0.942056 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.20932 w_k1_ADM_s107
                                 + 1.41926 w_k1_AXP_s107
                                 + 1.68013 w_k1_DIS_s107
                                 + 1.41969 w_k1_FDX_s107
                                 + 0.944368 w_k1_IBM_s107
                                 + 1.34079 w_k1_INTC_s107
                                 + 1.24535 w_k1_KO_s107
                                 + 0.986806 w_k1_MAT_s107
                                 + 1.18303 w_k1_MSFT_s107 + 1.3939 w_k1_PG_s107
                                 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.15479 w_k1_ADM_s107
                                 + 0.902082 w_k1_AXP_s107
                                 + 1.24513 w_k1_DIS_s107
                                 + 1.09223 w_k1_FDX_s107
                                 + 0.690859 w_k1_IBM_s107
                                 + 1.09362 w_k1_INTC_s107
                                 + 0.889984 w_k1_KO_s107
                                 + 0.959304 w_k1_MAT_s107
                                 + 1.11583 w_k1_MSFT_s107
                                 + 1.09823 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.23868 w_k1_ADM_s107
                                 + 1.58275 w_k1_AXP_s107
                                 + 1.62765 w_k1_DIS_s107
                                 + 1.31796 w_k1_FDX_s107
                                 + 1.33326 w_k1_IBM_s107
                                 + 1.23973 w_k1_INTC_s107
                                 + 1.40341 w_k1_KO_s107 + 1.84197 w_k1_MAT_s107
                                 + 1.12292 w_k1_MSFT_s107
                                 + 1.26008 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.30869 w_k1_ADM_s110
                                 + 1.35548 w_k1_AXP_s110
                                 + 1.08743 w_k1_DIS_s110
                                 + 1.07913 w_k1_FDX_s110
                                 + 0.813858 w_k1_IBM_s110
                                 + 1.04644 w_k1_INTC_s110
                                 + 1.03693 w_k1_KO_s110 + 1.03657 w_k1_MAT_s110
                                 + 1.0801 w_k1_MSFT_s110 + 1.06842 w_k1_PG_s110
                                 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.04986 w_k1_ADM_s110
                                 + 0.916126 w_k1_AXP_s110
                                 + 1.00169 w_k1_DIS_s110
                                 + 1.03622 w_k1_FDX_s110 + 1.0109 w_k1_IBM_s110
                                 + 0.927266 w_k1_INTC_s110
                                 + 1.11672 w_k1_KO_s110 + 1.58342 w_k1_MAT_s110
                                 + 1.01856 w_k1_MSFT_s110
                                 + 1.09075 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.667753 w_k1_ADM_s110
                                 + 0.643537 w_k1_AXP_s110
                                 + 0.804122 w_k1_DIS_s110
                                 + 0.826046 w_k1_FDX_s110
                                 + 0.74956 w_k1_IBM_s110
                                 + 0.866946 w_k1_INTC_s110
                                 + 0.817539 w_k1_KO_s110
                                 + 0.640896 w_k1_MAT_s110
                                 + 0.834368 w_k1_MSFT_s110
                                 + 0.843724 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.787587 w_k1_ADM_s110
                                 + 1.05917 w_k1_AXP_s110
                                 + 1.11623 w_k1_DIS_s110
                                 + 1.12105 w_k1_FDX_s110
                                 + 0.891453 w_k1_IBM_s110
                                 + 1.18899 w_k1_INTC_s110
                                 + 1.12869 w_k1_KO_s110
                                 + 0.882288 w_k1_MAT_s110
                                 + 1.05278 w_k1_MSFT_s110
                                 + 1.00108 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.936792 w_k1_ADM_s110
                                 + 1.04754 w_k1_AXP_s110
                                 + 1.33827 w_k1_DIS_s110
                                 + 0.998104 w_k1_FDX_s110
                                 + 1.35158 w_k1_IBM_s110
                                 + 1.26486 w_k1_INTC_s110
                                 + 1.33609 w_k1_KO_s110 + 1.37286 w_k1_MAT_s110
                                 + 1.07622 w_k1_MSFT_s110
                                 + 1.14712 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.735353 w_k1_ADM_s110
                                 + 0.704599 w_k1_AXP_s110
                                 + 1.06739 w_k1_DIS_s110
                                 + 0.755236 w_k1_FDX_s110
                                 + 1.19825 w_k1_IBM_s110
                                 + 0.840217 w_k1_INTC_s110
                                 + 1.19938 w_k1_KO_s110 + 1.4653 w_k1_MAT_s110
                                 + 0.896585 w_k1_MSFT_s110
                                 + 1.0044 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.23133 w_k1_ADM_s110
                                 + 1.38493 w_k1_AXP_s110
                                 + 1.69094 w_k1_DIS_s110
                                 + 1.36747 w_k1_FDX_s110
                                 + 1.01662 w_k1_IBM_s110
                                 + 1.34595 w_k1_INTC_s110
                                 + 1.43306 w_k1_KO_s110 + 1.04941 w_k1_MAT_s110
                                 + 1.41754 w_k1_MSFT_s110
                                 + 1.29166 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.16578 w_k1_ADM_s110
                                 + 1.63832 w_k1_AXP_s110
                                 + 1.78657 w_k1_DIS_s110
                                 + 1.35982 w_k1_FDX_s110
                                 + 1.70507 w_k1_IBM_s110
                                 + 1.68396 w_k1_INTC_s110
                                 + 1.74232 w_k1_KO_s110 + 2.08709 w_k1_MAT_s110
                                 + 1.21567 w_k1_MSFT_s110
                                 + 1.42716 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.74523 w_k1_ADM_s111
                                 + 0.788976 w_k1_AXP_s111
                                 + 0.783727 w_k1_DIS_s111
                                 + 0.797149 w_k1_FDX_s111
                                 + 1.0144 w_k1_IBM_s111
                                 + 0.838208 w_k1_INTC_s111
                                 + 1.08342 w_k1_KO_s111 + 1.13193 w_k1_MAT_s111
                                 + 0.828899 w_k1_MSFT_s111
                                 + 0.936102 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.81885 w_k1_ADM_s111
                                 + 0.622704 w_k1_AXP_s111
                                 + 0.688286 w_k1_DIS_s111
                                 + 0.784459 w_k1_FDX_s111
                                 + 0.735869 w_k1_IBM_s111
                                 + 0.691425 w_k1_INTC_s111
                                 + 0.856205 w_k1_KO_s111
                                 + 0.574141 w_k1_MAT_s111
                                 + 0.8444 w_k1_MSFT_s111
                                 + 0.896661 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.13376 w_k1_ADM_s111
                                 + 0.930246 w_k1_AXP_s111
                                 + 0.96299 w_k1_DIS_s111
                                 + 0.928103 w_k1_FDX_s111
                                 + 1.07815 w_k1_IBM_s111
                                 + 1.04267 w_k1_INTC_s111
                                 + 1.10864 w_k1_KO_s111
                                 + 0.722689 w_k1_MAT_s111
                                 + 1.01104 w_k1_MSFT_s111
                                 + 1.02615 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.06777 w_k1_ADM_s111
                                 + 0.912421 w_k1_AXP_s111
                                 + 0.779829 w_k1_DIS_s111
                                 + 0.987149 w_k1_FDX_s111
                                 + 0.64205 w_k1_IBM_s111
                                 + 0.991687 w_k1_INTC_s111
                                 + 0.8387 w_k1_KO_s111 + 0.707329 w_k1_MAT_s111
                                 + 0.901442 w_k1_MSFT_s111
                                 + 0.920796 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.62214 w_k1_ADM_s111
                                 + 1.63005 w_k1_AXP_s111
                                 + 1.42733 w_k1_DIS_s111 + 1.323 w_k1_FDX_s111
                                 + 1.39062 w_k1_IBM_s111
                                 + 0.982684 w_k1_INTC_s111
                                 + 1.5702 w_k1_KO_s111 + 1.5268 w_k1_MAT_s111
                                 + 1.14974 w_k1_MSFT_s111
                                 + 1.38393 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.841886 w_k1_ADM_s111
                                 + 0.887109 w_k1_AXP_s111
                                 + 0.856823 w_k1_DIS_s111
                                 + 0.993279 w_k1_FDX_s111
                                 + 0.809972 w_k1_IBM_s111
                                 + 0.957424 w_k1_INTC_s111
                                 + 1.03354 w_k1_KO_s111
                                 + 0.931108 w_k1_MAT_s111
                                 + 0.97557 w_k1_MSFT_s111
                                 + 1.02918 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.06121 w_k1_ADM_s111
                                 + 1.01148 w_k1_AXP_s111
                                 + 0.752072 w_k1_DIS_s111
                                 + 0.892316 w_k1_FDX_s111
                                 + 1.36732 w_k1_IBM_s111
                                 + 0.872802 w_k1_INTC_s111
                                 + 1.08838 w_k1_KO_s111 + 1.41962 w_k1_MAT_s111
                                 + 0.881175 w_k1_MSFT_s111
                                 + 1.00942 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.25259 w_k1_ADM_s111
                                 + 1.36977 w_k1_AXP_s111
                                 + 1.15072 w_k1_DIS_s111 + 1.2506 w_k1_FDX_s111
                                 + 0.988988 w_k1_IBM_s111
                                 + 1.13113 w_k1_INTC_s111
                                 + 1.17747 w_k1_KO_s111
                                 + 0.986927 w_k1_MAT_s111
                                 + 1.20755 w_k1_MSFT_s111
                                 + 1.08768 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.990818 w_k1_ADM_s112
                                 + 1.14544 w_k1_AXP_s112
                                 + 1.10483 w_k1_DIS_s112
                                 + 1.03629 w_k1_FDX_s112
                                 + 1.16113 w_k1_IBM_s112
                                 + 1.08106 w_k1_INTC_s112
                                 + 1.14899 w_k1_KO_s112
                                 + 0.994168 w_k1_MAT_s112
                                 + 1.03093 w_k1_MSFT_s112
                                 + 1.07331 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.948675 w_k1_ADM_s112
                                 + 0.878511 w_k1_AXP_s112
                                 + 0.943895 w_k1_DIS_s112
                                 + 0.840404 w_k1_FDX_s112
                                 + 1.22827 w_k1_IBM_s112
                                 + 0.72855 w_k1_INTC_s112 + 1.0225 w_k1_KO_s112
                                 + 1.10087 w_k1_MAT_s112
                                 + 0.862254 w_k1_MSFT_s112
                                 + 0.934053 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.29591 w_k1_ADM_s112
                                 + 1.43977 w_k1_AXP_s112
                                 + 1.56974 w_k1_DIS_s112
                                 + 1.39048 w_k1_FDX_s112
                                 + 0.971462 w_k1_IBM_s112
                                 + 0.987165 w_k1_INTC_s112
                                 + 1.06643 w_k1_KO_s112
                                 + 0.875632 w_k1_MAT_s112
                                 + 1.00926 w_k1_MSFT_s112 + 1.0706 w_k1_PG_s112
                                 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.3358 w_k1_ADM_s112
                                 + 1.47622 w_k1_AXP_s112
                                 + 1.64462 w_k1_DIS_s112
                                 + 1.25245 w_k1_FDX_s112
                                 + 1.57272 w_k1_IBM_s112
                                 + 1.00145 w_k1_INTC_s112
                                 + 1.51861 w_k1_KO_s112 + 1.3889 w_k1_MAT_s112
                                 + 1.22689 w_k1_MSFT_s112
                                 + 1.12443 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.852342 w_k1_ADM_s112
                                 + 0.889602 w_k1_AXP_s112
                                 + 1.23688 w_k1_DIS_s112
                                 + 1.02634 w_k1_FDX_s112
                                 + 0.873418 w_k1_IBM_s112
                                 + 1.0131 w_k1_INTC_s112
                                 + 0.880399 w_k1_KO_s112
                                 + 0.636745 w_k1_MAT_s112
                                 + 1.12857 w_k1_MSFT_s112
                                 + 1.05361 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.876901 w_k1_ADM_s112
                                 + 1.22529 w_k1_AXP_s112
                                 + 1.38882 w_k1_DIS_s112
                                 + 1.13728 w_k1_FDX_s112
                                 + 1.04908 w_k1_IBM_s112
                                 + 1.15476 w_k1_INTC_s112
                                 + 1.20416 w_k1_KO_s112
                                 + 0.977986 w_k1_MAT_s112
                                 + 1.18171 w_k1_MSFT_s112
                                 + 1.01159 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.00036 w_k1_ADM_s112
                                 + 1.20748 w_k1_AXP_s112
                                 + 0.885485 w_k1_DIS_s112
                                 + 1.11506 w_k1_FDX_s112
                                 + 0.803989 w_k1_IBM_s112
                                 + 1.12052 w_k1_INTC_s112
                                 + 0.927414 w_k1_KO_s112
                                 + 0.911697 w_k1_MAT_s112
                                 + 0.931201 w_k1_MSFT_s112
                                 + 0.949126 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.906375 w_k1_ADM_s112
                                 + 0.801999 w_k1_AXP_s112
                                 + 0.76111 w_k1_DIS_s112
                                 + 0.808848 w_k1_FDX_s112
                                 + 0.755324 w_k1_IBM_s112
                                 + 0.681237 w_k1_INTC_s112
                                 + 0.724278 w_k1_KO_s112
                                 + 0.614481 w_k1_MAT_s112
                                 + 0.808046 w_k1_MSFT_s112
                                 + 0.869651 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.01732 w_k1_ADM_s113
                                 + 1.15106 w_k1_AXP_s113
                                 + 0.923901 w_k1_DIS_s113
                                 + 0.959237 w_k1_FDX_s113
                                 + 1.03761 w_k1_IBM_s113
                                 + 0.943173 w_k1_INTC_s113
                                 + 0.867171 w_k1_KO_s113
                                 + 0.793797 w_k1_MAT_s113
                                 + 0.881934 w_k1_MSFT_s113
                                 + 0.971235 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.63707 w_k1_ADM_s113
                                 + 0.826842 w_k1_AXP_s113
                                 + 0.639164 w_k1_DIS_s113
                                 + 0.7319 w_k1_FDX_s113 + 1.34604 w_k1_IBM_s113
                                 + 0.82776 w_k1_INTC_s113
                                 + 0.998606 w_k1_KO_s113
                                 + 1.48019 w_k1_MAT_s113
                                 + 0.794239 w_k1_MSFT_s113
                                 + 0.905735 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.608 w_k1_ADM_s113
                                 + 0.688934 w_k1_AXP_s113
                                 + 0.502839 w_k1_DIS_s113
                                 + 0.714861 w_k1_FDX_s113
                                 + 0.721754 w_k1_IBM_s113
                                 + 0.73065 w_k1_INTC_s113
                                 + 0.635725 w_k1_KO_s113
                                 + 0.680663 w_k1_MAT_s113
                                 + 0.606397 w_k1_MSFT_s113
                                 + 0.845801 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.984762 w_k1_ADM_s113
                                 + 0.928578 w_k1_AXP_s113
                                 + 0.752407 w_k1_DIS_s113
                                 + 0.84878 w_k1_FDX_s113
                                 + 0.899168 w_k1_IBM_s113
                                 + 0.787051 w_k1_INTC_s113
                                 + 0.783184 w_k1_KO_s113
                                 + 0.638193 w_k1_MAT_s113
                                 + 0.820624 w_k1_MSFT_s113
                                 + 0.8569 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.26128 w_k1_ADM_s113
                                 + 1.19554 w_k1_AXP_s113
                                 + 1.62378 w_k1_DIS_s113
                                 + 1.41484 w_k1_FDX_s113
                                 + 0.929694 w_k1_IBM_s113
                                 + 1.5268 w_k1_INTC_s113 + 1.22386 w_k1_KO_s113
                                 + 0.829231 w_k1_MAT_s113
                                 + 1.45753 w_k1_MSFT_s113
                                 + 1.12701 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.07855 w_k1_ADM_s113
                                 + 1.22156 w_k1_AXP_s113
                                 + 1.06139 w_k1_DIS_s113
                                 + 1.05313 w_k1_FDX_s113
                                 + 1.46555 w_k1_IBM_s113
                                 + 0.928439 w_k1_INTC_s113
                                 + 1.17695 w_k1_KO_s113 + 1.61061 w_k1_MAT_s113
                                 + 0.998112 w_k1_MSFT_s113
                                 + 1.13175 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.0742 w_k1_ADM_s113
                                 + 1.01075 w_k1_AXP_s113
                                 + 1.00646 w_k1_DIS_s113
                                 + 1.09253 w_k1_FDX_s113
                                 + 0.843166 w_k1_IBM_s113
                                 + 0.974658 w_k1_INTC_s113
                                 + 0.97603 w_k1_KO_s113
                                 + 0.804909 w_k1_MAT_s113
                                 + 0.970355 w_k1_MSFT_s113
                                 + 1.06247 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.25985 w_k1_ADM_s113
                                 + 1.37147 w_k1_AXP_s113
                                 + 1.17243 w_k1_DIS_s113
                                 + 1.23889 w_k1_FDX_s113
                                 + 1.00628 w_k1_IBM_s113
                                 + 1.15449 w_k1_INTC_s113
                                 + 1.13588 w_k1_KO_s113 + 1.32467 w_k1_MAT_s113
                                 + 1.16749 w_k1_MSFT_s113
                                 + 1.19722 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.796177 w_k1_ADM_s114
                                 + 1.08355 w_k1_AXP_s114
                                 + 0.912813 w_k1_DIS_s114
                                 + 1.01655 w_k1_FDX_s114
                                 + 0.69687 w_k1_IBM_s114
                                 + 0.941787 w_k1_INTC_s114
                                 + 0.869012 w_k1_KO_s114
                                 + 0.824856 w_k1_MAT_s114
                                 + 0.943076 w_k1_MSFT_s114
                                 + 0.996341 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.793106 w_k1_ADM_s114
                                 + 1.23537 w_k1_AXP_s114
                                 + 0.92352 w_k1_DIS_s114
                                 + 0.989857 w_k1_FDX_s114
                                 + 1.10986 w_k1_IBM_s114
                                 + 1.24718 w_k1_INTC_s114 + 1.0773 w_k1_KO_s114
                                 + 1.12827 w_k1_MAT_s114
                                 + 1.02803 w_k1_MSFT_s114
                                 + 1.02395 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.16318 w_k1_ADM_s114
                                 + 1.34321 w_k1_AXP_s114
                                 + 1.64378 w_k1_DIS_s114
                                 + 1.26252 w_k1_FDX_s114
                                 + 1.65879 w_k1_IBM_s114
                                 + 1.28207 w_k1_INTC_s114
                                 + 1.53231 w_k1_KO_s114 + 1.59613 w_k1_MAT_s114
                                 + 1.10976 w_k1_MSFT_s114
                                 + 1.15914 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.15853 w_k1_ADM_s114
                                 + 1.64586 w_k1_AXP_s114
                                 + 1.26915 w_k1_DIS_s114 + 1.4416 w_k1_FDX_s114
                                 + 1.17539 w_k1_IBM_s114
                                 + 1.23579 w_k1_INTC_s114
                                 + 1.07362 w_k1_KO_s114 + 1.2149 w_k1_MAT_s114
                                 + 1.15706 w_k1_MSFT_s114 + 1.2669 w_k1_PG_s114
                                 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.72463 w_k1_ADM_s114
                                 + 0.838977 w_k1_AXP_s114
                                 + 0.658052 w_k1_DIS_s114
                                 + 0.820082 w_k1_FDX_s114
                                 + 0.774702 w_k1_IBM_s114
                                 + 0.999743 w_k1_INTC_s114
                                 + 0.703822 w_k1_KO_s114
                                 + 0.662121 w_k1_MAT_s114
                                 + 0.701239 w_k1_MSFT_s114
                                 + 0.784664 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.984652 w_k1_ADM_s114
                                 + 1.23276 w_k1_AXP_s114
                                 + 1.24605 w_k1_DIS_s114
                                 + 1.29381 w_k1_FDX_s114
                                 + 0.97245 w_k1_IBM_s114
                                 + 1.45279 w_k1_INTC_s114
                                 + 1.06362 w_k1_KO_s114
                                 + 0.770085 w_k1_MAT_s114
                                 + 1.13088 w_k1_MSFT_s114
                                 + 1.08131 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.04168 w_k1_ADM_s114
                                 + 0.919104 w_k1_AXP_s114
                                 + 1.21183 w_k1_DIS_s114
                                 + 1.06332 w_k1_FDX_s114
                                 + 1.10582 w_k1_IBM_s114
                                 + 1.07811 w_k1_INTC_s114
                                 + 1.10597 w_k1_KO_s114 + 0.87181 w_k1_MAT_s114
                                 + 1.02242 w_k1_MSFT_s114
                                 + 1.05184 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.809737 w_k1_ADM_s114
                                 + 0.698206 w_k1_AXP_s114
                                 + 0.761094 w_k1_DIS_s114
                                 + 0.841385 w_k1_FDX_s114
                                 + 1.16485 w_k1_IBM_s114
                                 + 0.902717 w_k1_INTC_s114
                                 + 0.992472 w_k1_KO_s114
                                 + 1.03383 w_k1_MAT_s114
                                 + 0.884515 w_k1_MSFT_s114
                                 + 0.990904 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.55558 w_k1_ADM_s115
                                 + 1.15276 w_k1_AXP_s115
                                 + 1.24958 w_k1_DIS_s115
                                 + 1.31674 w_k1_FDX_s115
                                 + 0.731093 w_k1_IBM_s115
                                 + 1.14712 w_k1_INTC_s115
                                 + 0.950686 w_k1_KO_s115
                                 + 0.852697 w_k1_MAT_s115
                                 + 1.34559 w_k1_MSFT_s115
                                 + 1.17279 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.746468 w_k1_ADM_s115
                                 + 0.712811 w_k1_AXP_s115
                                 + 0.538742 w_k1_DIS_s115
                                 + 0.696732 w_k1_FDX_s115
                                 + 0.890027 w_k1_IBM_s115
                                 + 0.911586 w_k1_INTC_s115
                                 + 0.792561 w_k1_KO_s115
                                 + 0.834864 w_k1_MAT_s115
                                 + 0.823749 w_k1_MSFT_s115
                                 + 0.928909 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.926142 w_k1_ADM_s115
                                 + 0.8222 w_k1_AXP_s115
                                 + 0.896018 w_k1_DIS_s115
                                 + 0.847866 w_k1_FDX_s115
                                 + 0.653348 w_k1_IBM_s115
                                 + 0.908314 w_k1_INTC_s115
                                 + 0.746157 w_k1_KO_s115
                                 + 0.538995 w_k1_MAT_s115
                                 + 1.00796 w_k1_MSFT_s115
                                 + 0.968054 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.21397 w_k1_ADM_s115
                                 + 1.33342 w_k1_AXP_s115
                                 + 1.20722 w_k1_DIS_s115
                                 + 1.14134 w_k1_FDX_s115
                                 + 1.02554 w_k1_IBM_s115
                                 + 1.1918 w_k1_INTC_s115
                                 + 0.914871 w_k1_KO_s115
                                 + 1.21961 w_k1_MAT_s115
                                 + 1.22941 w_k1_MSFT_s115
                                 + 1.04604 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.83548 w_k1_ADM_s115
                                 + 1.58857 w_k1_AXP_s115
                                 + 1.63993 w_k1_DIS_s115
                                 + 1.35457 w_k1_FDX_s115
                                 + 1.41668 w_k1_IBM_s115
                                 + 1.28995 w_k1_INTC_s115
                                 + 1.45901 w_k1_KO_s115 + 1.18489 w_k1_MAT_s115
                                 + 1.33453 w_k1_MSFT_s115 + 1.363 w_k1_PG_s115
                                 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.08348 w_k1_ADM_s115
                                 + 0.902765 w_k1_AXP_s115
                                 + 0.917619 w_k1_DIS_s115
                                 + 0.85369 w_k1_FDX_s115
                                 + 1.15392 w_k1_IBM_s115
                                 + 0.919429 w_k1_INTC_s115
                                 + 1.05827 w_k1_KO_s115 + 1.25777 w_k1_MAT_s115
                                 + 0.999327 w_k1_MSFT_s115
                                 + 1.01994 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.14688 w_k1_ADM_s115
                                 + 0.905826 w_k1_AXP_s115
                                 + 0.808597 w_k1_DIS_s115
                                 + 0.988839 w_k1_FDX_s115
                                 + 0.766572 w_k1_IBM_s115
                                 + 1.02017 w_k1_INTC_s115
                                 + 0.847094 w_k1_KO_s115
                                 + 0.618917 w_k1_MAT_s115
                                 + 0.972535 w_k1_MSFT_s115
                                 + 1.01793 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.12271 w_k1_ADM_s115
                                 + 1.1596 w_k1_AXP_s115
                                 + 0.978119 w_k1_DIS_s115
                                 + 0.894314 w_k1_FDX_s115
                                 + 1.40736 w_k1_IBM_s115
                                 + 1.25815 w_k1_INTC_s115
                                 + 1.10612 w_k1_KO_s115 + 1.15107 w_k1_MAT_s115
                                 + 0.982893 w_k1_MSFT_s115
                                 + 1.08318 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.66762 w_k1_ADM_s116
                                 + 0.530308 w_k1_AXP_s116
                                 + 0.528379 w_k1_DIS_s116
                                 + 0.628004 w_k1_FDX_s116
                                 + 0.858278 w_k1_IBM_s116
                                 + 0.689432 w_k1_INTC_s116
                                 + 0.7623 w_k1_KO_s116 + 1.0915 w_k1_MAT_s116
                                 + 0.695762 w_k1_MSFT_s116
                                 + 0.734891 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.807699 w_k1_ADM_s116
                                 + 0.717469 w_k1_AXP_s116
                                 + 0.755124 w_k1_DIS_s116
                                 + 0.838757 w_k1_FDX_s116
                                 + 0.945521 w_k1_IBM_s116
                                 + 0.871553 w_k1_INTC_s116
                                 + 0.815341 w_k1_KO_s116
                                 + 1.07046 w_k1_MAT_s116
                                 + 0.873417 w_k1_MSFT_s116
                                 + 0.897945 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.37738 w_k1_ADM_s116
                                 + 1.54976 w_k1_AXP_s116
                                 + 1.55693 w_k1_DIS_s116
                                 + 1.42231 w_k1_FDX_s116
                                 + 1.40866 w_k1_IBM_s116
                                 + 1.74552 w_k1_INTC_s116
                                 + 1.32581 w_k1_KO_s116 + 1.23766 w_k1_MAT_s116
                                 + 1.50482 w_k1_MSFT_s116
                                 + 1.25408 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.0251 w_k1_ADM_s116
                                 + 0.955179 w_k1_AXP_s116
                                 + 0.997657 w_k1_DIS_s116
                                 + 1.13403 w_k1_FDX_s116
                                 + 0.759139 w_k1_IBM_s116
                                 + 0.947561 w_k1_INTC_s116
                                 + 0.812443 w_k1_KO_s116
                                 + 0.969908 w_k1_MAT_s116
                                 + 1.00893 w_k1_MSFT_s116
                                 + 0.899461 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.96177 w_k1_ADM_s116
                                 + 1.05759 w_k1_AXP_s116
                                 + 0.886304 w_k1_DIS_s116
                                 + 0.909341 w_k1_FDX_s116
                                 + 1.45423 w_k1_IBM_s116
                                 + 1.19686 w_k1_INTC_s116 + 1.0587 w_k1_KO_s116
                                 + 1.73797 w_k1_MAT_s116
                                 + 1.07916 w_k1_MSFT_s116
                                 + 1.04653 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.03967 w_k1_ADM_s116
                                 + 1.12347 w_k1_AXP_s116
                                 + 0.843904 w_k1_DIS_s116
                                 + 0.976878 w_k1_FDX_s116
                                 + 0.977775 w_k1_IBM_s116
                                 + 0.844327 w_k1_INTC_s116
                                 + 0.781788 w_k1_KO_s116
                                 + 1.33918 w_k1_MAT_s116
                                 + 0.837251 w_k1_MSFT_s116
                                 + 0.887226 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.679879 w_k1_ADM_s116
                                 + 0.667337 w_k1_AXP_s116
                                 + 0.699332 w_k1_DIS_s116
                                 + 0.713127 w_k1_FDX_s116
                                 + 1.00652 w_k1_IBM_s116
                                 + 0.81021 w_k1_INTC_s116
                                 + 0.868149 w_k1_KO_s116
                                 + 1.00368 w_k1_MAT_s116
                                 + 0.85648 w_k1_MSFT_s116
                                 + 0.878887 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.09182 w_k1_ADM_s116
                                 + 1.11418 w_k1_AXP_s116
                                 + 1.28297 w_k1_DIS_s116
                                 + 1.04864 w_k1_FDX_s116
                                 + 1.25119 w_k1_IBM_s116
                                 + 1.02305 w_k1_INTC_s116
                                 + 1.33528 w_k1_KO_s116 + 1.60771 w_k1_MAT_s116
                                 + 1.14074 w_k1_MSFT_s116
                                 + 1.12456 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.72875 w_k1_ADM_s117
                                 + 0.662621 w_k1_AXP_s117
                                 + 0.648216 w_k1_DIS_s117
                                 + 0.774694 w_k1_FDX_s117
                                 + 0.834389 w_k1_IBM_s117
                                 + 0.865591 w_k1_INTC_s117
                                 + 0.762254 w_k1_KO_s117
                                 + 1.01782 w_k1_MAT_s117
                                 + 0.801982 w_k1_MSFT_s117
                                 + 0.831121 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.37434 w_k1_ADM_s117
                                 + 1.62522 w_k1_AXP_s117
                                 + 1.61721 w_k1_DIS_s117
                                 + 1.47579 w_k1_FDX_s117 + 1.2525 w_k1_IBM_s117
                                 + 1.04305 w_k1_INTC_s117 + 1.2245 w_k1_KO_s117
                                 + 1.66017 w_k1_MAT_s117
                                 + 1.10425 w_k1_MSFT_s117
                                 + 1.27825 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.07028 w_k1_ADM_s117
                                 + 1.10267 w_k1_AXP_s117
                                 + 1.16541 w_k1_DIS_s117
                                 + 1.15395 w_k1_FDX_s117
                                 + 0.960226 w_k1_IBM_s117
                                 + 1.23513 w_k1_INTC_s117
                                 + 0.983567 w_k1_KO_s117
                                 + 1.20704 w_k1_MAT_s117
                                 + 1.24825 w_k1_MSFT_s117
                                 + 1.07992 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.965372 w_k1_ADM_s117
                                 + 1.17047 w_k1_AXP_s117
                                 + 1.24202 w_k1_DIS_s117
                                 + 0.979813 w_k1_FDX_s117
                                 + 1.70964 w_k1_IBM_s117
                                 + 0.860868 w_k1_INTC_s117
                                 + 1.1412 w_k1_KO_s117 + 1.73051 w_k1_MAT_s117
                                 + 0.917617 w_k1_MSFT_s117
                                 + 1.11358 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.830328 w_k1_ADM_s117
                                 + 0.749864 w_k1_AXP_s117
                                 + 0.797031 w_k1_DIS_s117
                                 + 0.818422 w_k1_FDX_s117
                                 + 1.25044 w_k1_IBM_s117
                                 + 0.923862 w_k1_INTC_s117
                                 + 0.815082 w_k1_KO_s117
                                 + 1.22993 w_k1_MAT_s117
                                 + 0.824549 w_k1_MSFT_s117
                                 + 0.935298 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.992629 w_k1_ADM_s117
                                 + 0.995278 w_k1_AXP_s117
                                 + 0.958381 w_k1_DIS_s117
                                 + 0.94019 w_k1_FDX_s117 + 1.0447 w_k1_IBM_s117
                                 + 0.942921 w_k1_INTC_s117
                                 + 0.941644 w_k1_KO_s117
                                 + 0.915718 w_k1_MAT_s117
                                 + 0.942097 w_k1_MSFT_s117
                                 + 1.0902 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.943899 w_k1_ADM_s117
                                 + 0.825973 w_k1_AXP_s117
                                 + 0.831512 w_k1_DIS_s117
                                 + 1.09478 w_k1_FDX_s117
                                 + 0.684939 w_k1_IBM_s117
                                 + 0.97645 w_k1_INTC_s117
                                 + 0.751469 w_k1_KO_s117
                                 + 0.714051 w_k1_MAT_s117
                                 + 0.965696 w_k1_MSFT_s117
                                 + 0.931867 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.15201 w_k1_ADM_s117
                                 + 1.03414 w_k1_AXP_s117
                                 + 1.33326 w_k1_DIS_s117
                                 + 1.17471 w_k1_FDX_s117
                                 + 1.30717 w_k1_IBM_s117
                                 + 1.15038 w_k1_INTC_s117
                                 + 1.14584 w_k1_KO_s117 + 1.11824 w_k1_MAT_s117
                                 + 1.16592 w_k1_MSFT_s117
                                 + 1.16807 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0332374125799121 w_k1_ADM_s000
                                 + 0.0442177120012672 w_k1_ADM_s001
                                 + 0.0112358150550788 w_k1_ADM_s002
                                 + 0.0115689820533611 w_k1_ADM_s003
                                 + 0.0277959550755971 w_k1_ADM_s004
                                 + 0.0166221173340533 w_k1_ADM_s005
                                 + 0.0411907034181402 w_k1_ADM_s006
                                 + 0.027114594827363 w_k1_ADM_s007
                                 + 0.0872206972115876 w_k1_ADM_s010
                                 + 0.0525654794106991 w_k1_ADM_s011
                                 + 0.0434613613426398 w_k1_ADM_s012
                                 + 0.0857001106844917 w_k1_ADM_s013
                                 + 0.0628417426149292 w_k1_ADM_s014
                                 + 0.0410507534845395 w_k1_ADM_s015
                                 + 0.0647390749613011 w_k1_ADM_s016
                                 + 0.0625035347412204 w_k1_ADM_s017
                                 + 0.0123694111881849 w_k1_ADM_s100
                                 + 0.0105505953096514 w_k1_ADM_s101
                                 + 0.014633557245604 w_k1_ADM_s102
                                 + 0.0285330694261606 w_k1_ADM_s103
                                 + 0.0519001960247481 w_k1_ADM_s104
                                 + 0.0307880457749942 w_k1_ADM_s105
                                 + 0.0118200479613961 w_k1_ADM_s106
                                 + 0.0385464623016929 w_k1_ADM_s107
                                 + 0.0109382760574561 w_k1_ADM_s110
                                 + 0.0115877890512357 w_k1_ADM_s111
                                 + 0.0256772488827819 w_k1_ADM_s112
                                 + 0.014648084661166 w_k1_ADM_s113
                                 + 0.0131097862569238 w_k1_ADM_s114
                                 + 0.0279322177894464 w_k1_ADM_s115
                                 + 0.0132472564943949 w_k1_ADM_s116
                                 + 0.0243910810289161 w_k1_ADM_s117
                                 + 0.0345151962012109 w_k1_AXP_s000
                                 + 0.0470086301937405 w_k1_AXP_s001
                                 + 0.0104193888555227 w_k1_AXP_s002
                                 + 0.0105583614617291 w_k1_AXP_s003
                                 + 0.0245060621926199 w_k1_AXP_s004
                                 + 0.0173990255189915 w_k1_AXP_s005
                                 + 0.0350583995857757 w_k1_AXP_s006
                                 + 0.0254357537860411 w_k1_AXP_s007
                                 + 0.0735280013998607 w_k1_AXP_s010
                                 + 0.0572646707103149 w_k1_AXP_s011
                                 + 0.0421264743188987 w_k1_AXP_s012
                                 + 0.084988747583588 w_k1_AXP_s013
                                 + 0.0591268982522967 w_k1_AXP_s014
                                 + 0.0375218040722391 w_k1_AXP_s015
                                 + 0.0610426383918247 w_k1_AXP_s016
                                 + 0.0585806691633851 w_k1_AXP_s017
                                 + 0.0142591852063667 w_k1_AXP_s100
                                 + 0.0132779054461092 w_k1_AXP_s101
                                 + 0.0160799466299804 w_k1_AXP_s102
                                 + 0.0258690185329693 w_k1_AXP_s103
                                 + 0.0435436352449871 w_k1_AXP_s104
                                 + 0.0297067939320024 w_k1_AXP_s105
                                 + 0.010502579245586 w_k1_AXP_s106
                                 + 0.0377108639258967 w_k1_AXP_s107
                                 + 0.0115697360712647 w_k1_AXP_s110
                                 + 0.0110785891068116 w_k1_AXP_s111
                                 + 0.0272279862172124 w_k1_AXP_s112
                                 + 0.015186331733845 w_k1_AXP_s113
                                 + 0.0154881895037315 w_k1_AXP_s114
                                 + 0.0262586353887702 w_k1_AXP_s115
                                 + 0.0135420667127907 w_k1_AXP_s116
                                 + 0.0242657509947563 w_k1_AXP_s117
                                 + 0.0337716596377104 w_k1_DIS_s000
                                 + 0.0457643447221467 w_k1_DIS_s001
                                 + 0.0117243646109383 w_k1_DIS_s002
                                 + 0.0107311919950883 w_k1_DIS_s003
                                 + 0.0236243287767822 w_k1_DIS_s004
                                 + 0.0182886885587762 w_k1_DIS_s005
                                 + 0.0348661651411777 w_k1_DIS_s006
                                 + 0.0244144641783541 w_k1_DIS_s007
                                 + 0.0800998942180146 w_k1_DIS_s010
                                 + 0.0555814001617433 w_k1_DIS_s011
                                 + 0.0447478457819999 w_k1_DIS_s012
                                 + 0.079823170658146 w_k1_DIS_s013
                                 + 0.0621869460334758 w_k1_DIS_s014
                                 + 0.0431709541491213 w_k1_DIS_s015
                                 + 0.0604121952107087 w_k1_DIS_s016
                                 + 0.0610801644511041 w_k1_DIS_s017
                                 + 0.0128596567193226 w_k1_DIS_s100
                                 + 0.0117660134732085 w_k1_DIS_s101
                                 + 0.0160041341775145 w_k1_DIS_s102
                                 + 0.0264641242902428 w_k1_DIS_s103
                                 + 0.0470964037051197 w_k1_DIS_s104
                                 + 0.0333556177740397 w_k1_DIS_s105
                                 + 0.0119093589295333 w_k1_DIS_s106
                                 + 0.0393561897112197 w_k1_DIS_s107
                                 + 0.0136618065216263 w_k1_DIS_s110
                                 + 0.00993786014643321 w_k1_DIS_s111
                                 + 0.0272500231578922 w_k1_DIS_s112
                                 + 0.013508023492628 w_k1_DIS_s113
                                 + 0.0147760159352221 w_k1_DIS_s114
                                 + 0.0245416988680097 w_k1_DIS_s115
                                 + 0.0134294527327912 w_k1_DIS_s116
                                 + 0.0252749043543474 w_k1_DIS_s117
                                 + 0.0327270713641297 w_k1_FDX_s000
                                 + 0.0454388647930163 w_k1_FDX_s001
                                 + 0.0112567074159229 w_k1_FDX_s002
                                 + 0.0110440822584695 w_k1_FDX_s003
                                 + 0.0234649927112953 w_k1_FDX_s004
                                 + 0.017192884092017 w_k1_FDX_s005
                                 + 0.0354094313941618 w_k1_FDX_s006
                                 + 0.0235098143218836 w_k1_FDX_s007
                                 + 0.0814716481400584 w_k1_FDX_s010
                                 + 0.0529560444489748 w_k1_FDX_s011
                                 + 0.0415434453556476 w_k1_FDX_s012
                                 + 0.0788771775865695 w_k1_FDX_s013
                                 + 0.0582208105446129 w_k1_FDX_s014
                                 + 0.0413936565130753 w_k1_FDX_s015
                                 + 0.0606462877716074 w_k1_FDX_s016
                                 + 0.0584180919306001 w_k1_FDX_s017
                                 + 0.0135461626091657 w_k1_FDX_s100
                                 + 0.0120025736021976 w_k1_FDX_s101
                                 + 0.0167332547112435 w_k1_FDX_s102
                                 + 0.025587335186342 w_k1_FDX_s103
                                 + 0.0494691898193665 w_k1_FDX_s104
                                 + 0.0302030652002913 w_k1_FDX_s105
                                 + 0.0111418245810803 w_k1_FDX_s106
                                 + 0.0369749736636283 w_k1_FDX_s107
                                 + 0.0120062426847728 w_k1_FDX_s110
                                 + 0.0107508073825085 w_k1_FDX_s111
                                 + 0.0258896644370989 w_k1_FDX_s112
                                 + 0.0143328250894978 w_k1_FDX_s113
                                 + 0.0152420566378515 w_k1_FDX_s114
                                 + 0.0237340547933791 w_k1_FDX_s115
                                 + 0.0133258982406907 w_k1_FDX_s116
                                 + 0.025626608683946 w_k1_FDX_s117
                                 + 0.034633809952479 w_k1_IBM_s000
                                 + 0.0482369280978072 w_k1_IBM_s001
                                 + 0.0111713730764712 w_k1_IBM_s002
                                 + 0.0111874403288615 w_k1_IBM_s003
                                 + 0.0231891698214678 w_k1_IBM_s004
                                 + 0.0170398616235575 w_k1_IBM_s005
                                 + 0.037863321714541 w_k1_IBM_s006
                                 + 0.0244335215309905 w_k1_IBM_s007
                                 + 0.0820401665779816 w_k1_IBM_s010
                                 + 0.056518445500162 w_k1_IBM_s011
                                 + 0.0407397743196305 w_k1_IBM_s012
                                 + 0.0790581174203732 w_k1_IBM_s013
                                 + 0.0577828131596449 w_k1_IBM_s014
                                 + 0.0426707616787996 w_k1_IBM_s015
                                 + 0.0619088091536181 w_k1_IBM_s016
                                 + 0.0608916094976399 w_k1_IBM_s017
                                 + 0.0132525625818779 w_k1_IBM_s100
                                 + 0.0119500746370096 w_k1_IBM_s101
                                 + 0.0169070711974934 w_k1_IBM_s102
                                 + 0.025911431285294 w_k1_IBM_s103
                                 + 0.0479688244899161 w_k1_IBM_s104
                                 + 0.0303935128069187 w_k1_IBM_s105
                                 + 0.011130929127732 w_k1_IBM_s106
                                 + 0.0363705965468946 w_k1_IBM_s107
                                 + 0.0124777745552643 w_k1_IBM_s110
                                 + 0.0110289293303468 w_k1_IBM_s111
                                 + 0.0257834820558741 w_k1_IBM_s112
                                 + 0.0148063501851308 w_k1_IBM_s113
                                 + 0.0151579240258296 w_k1_IBM_s114
                                 + 0.0249205705041076 w_k1_IBM_s115
                                 + 0.0147918835543942 w_k1_IBM_s116
                                 + 0.0268534283608273 w_k1_IBM_s117
                                 + 0.0304700444399283 w_k1_INTC_s000
                                 + 0.0449228329542626 w_k1_INTC_s001
                                 + 0.0114366732031762 w_k1_INTC_s002
                                 + 0.0120923324774071 w_k1_INTC_s003
                                 + 0.0209365766756232 w_k1_INTC_s004
                                 + 0.0166449155927662 w_k1_INTC_s005
                                 + 0.0348196703436213 w_k1_INTC_s006
                                 + 0.0251772652081684 w_k1_INTC_s007
                                 + 0.0870098927359266 w_k1_INTC_s010
                                 + 0.0511928804469815 w_k1_INTC_s011
                                 + 0.0403910493819566 w_k1_INTC_s012
                                 + 0.0751261811194772 w_k1_INTC_s013
                                 + 0.0642616114007804 w_k1_INTC_s014
                                 + 0.0381561725744513 w_k1_INTC_s015
                                 + 0.0668906990202896 w_k1_INTC_s016
                                 + 0.0656286142485979 w_k1_INTC_s017
                                 + 0.0127392356556958 w_k1_INTC_s100
                                 + 0.0111303639702321 w_k1_INTC_s101
                                 + 0.0148898183427855 w_k1_INTC_s102
                                 + 0.0247774031645561 w_k1_INTC_s103
                                 + 0.0472439285800051 w_k1_INTC_s104
                                 + 0.0293592661360011 w_k1_INTC_s105
                                 + 0.0100771453602688 w_k1_INTC_s106
                                 + 0.0385874568135599 w_k1_INTC_s107
                                 + 0.0128374036944265 w_k1_INTC_s110
                                 + 0.0103498396037733 w_k1_INTC_s111
                                 + 0.023297556864323 w_k1_INTC_s112
                                 + 0.0137327650542461 w_k1_INTC_s113
                                 + 0.0163514155352314 w_k1_INTC_s114
                                 + 0.0261484035813952 w_k1_INTC_s115
                                 + 0.0142128668243171 w_k1_INTC_s116
                                 + 0.0251781381785224 w_k1_INTC_s117
                                 + 0.0367981628181684 w_k1_KO_s000
                                 + 0.0417977304278282 w_k1_KO_s001
                                 + 0.0108857194074887 w_k1_KO_s002
                                 + 0.011258539144864 w_k1_KO_s003
                                 + 0.0240857911328845 w_k1_KO_s004
                                 + 0.0161947793489719 w_k1_KO_s005
                                 + 0.037410295149721 w_k1_KO_s006
                                 + 0.0260514241259628 w_k1_KO_s007
                                 + 0.0819734138221182 w_k1_KO_s010
                                 + 0.0546698467975593 w_k1_KO_s011
                                 + 0.0424686522771857 w_k1_KO_s012
                                 + 0.0917960882764238 w_k1_KO_s013
                                 + 0.0601715969462555 w_k1_KO_s014
                                 + 0.0424502109189445 w_k1_KO_s015
                                 + 0.0635022317610571 w_k1_KO_s016
                                 + 0.0600565909171243 w_k1_KO_s017
                                 + 0.0122401396467837 w_k1_KO_s100
                                 + 0.0100699061113944 w_k1_KO_s101
                                 + 0.0154234934700254 w_k1_KO_s102
                                 + 0.0249421720830559 w_k1_KO_s103
                                 + 0.0493068172905757 w_k1_KO_s104
                                 + 0.0280172867909788 w_k1_KO_s105
                                 + 0.0115452552876495 w_k1_KO_s106
                                 + 0.0379534250625654 w_k1_KO_s107
                                 + 0.013746753658914 w_k1_KO_s110
                                 + 0.0117243914784111 w_k1_KO_s111
                                 + 0.0254557802215078 w_k1_KO_s112
                                 + 0.0137907644853984 w_k1_KO_s113
                                 + 0.014738729641241 w_k1_KO_s114
                                 + 0.0234102545319148 w_k1_KO_s115
                                 + 0.0134117914986471 w_k1_KO_s116
                                 + 0.0235116006504589 w_k1_KO_s117
                                 + 0.0387850260303948 w_k1_MAT_s000
                                 + 0.0470321993053108 w_k1_MAT_s001
                                 + 0.00925606642078591 w_k1_MAT_s002
                                 + 0.00876251453681612 w_k1_MAT_s003
                                 + 0.023685013650605 w_k1_MAT_s004
                                 + 0.0187265741261615 w_k1_MAT_s005
                                 + 0.0352124483487376 w_k1_MAT_s006
                                 + 0.0308158062802834 w_k1_MAT_s007
                                 + 0.0836795414623316 w_k1_MAT_s010
                                 + 0.0712675058766662 w_k1_MAT_s011
                                 + 0.0378438755715525 w_k1_MAT_s012
                                 + 0.0903365315449714 w_k1_MAT_s013
                                 + 0.0636737803315824 w_k1_MAT_s014
                                 + 0.0482357195731014 w_k1_MAT_s015
                                 + 0.0537244688757373 w_k1_MAT_s016
                                 + 0.0586141071548631 w_k1_MAT_s017
                                 + 0.015978589749999 w_k1_MAT_s100
                                 + 0.0155465801067219 w_k1_MAT_s101
                                 + 0.017579676314117 w_k1_MAT_s102
                                 + 0.0297141051521667 w_k1_MAT_s103
                                 + 0.05077560306463 w_k1_MAT_s104
                                 + 0.0323249471761411 w_k1_MAT_s105
                                 + 0.0113590413062028 w_k1_MAT_s106
                                 + 0.0369312194886912 w_k1_MAT_s107
                                 + 0.0140993205498123 w_k1_MAT_s110
                                 + 0.0106729567489431 w_k1_MAT_s111
                                 + 0.0231904378314634 w_k1_MAT_s112
                                 + 0.0139668346277446 w_k1_MAT_s113
                                 + 0.0140039318309973 w_k1_MAT_s114
                                 + 0.0237615135551756 w_k1_MAT_s115
                                 + 0.0169009371002516 w_k1_MAT_s116
                                 + 0.0285748941635802 w_k1_MAT_s117
                                 + 0.0330382009950652 w_k1_MSFT_s000
                                 + 0.0452780501914801 w_k1_MSFT_s001
                                 + 0.0110141724280871 w_k1_MSFT_s002
                                 + 0.0114568746653207 w_k1_MSFT_s003
                                 + 0.0232970593210817 w_k1_MSFT_s004
                                 + 0.0159824904247341 w_k1_MSFT_s005
                                 + 0.0368666562681618 w_k1_MSFT_s006
                                 + 0.0234879969813273 w_k1_MSFT_s007
                                 + 0.0750798237866678 w_k1_MSFT_s010
                                 + 0.0558052958420686 w_k1_MSFT_s011
                                 + 0.0415914847771098 w_k1_MSFT_s012
                                 + 0.0855739620164112 w_k1_MSFT_s013
                                 + 0.0646885302480456 w_k1_MSFT_s014
                                 + 0.0406333796076895 w_k1_MSFT_s015
                                 + 0.0575348535798585 w_k1_MSFT_s016
                                 + 0.0607574755291892 w_k1_MSFT_s017
                                 + 0.013384318958697 w_k1_MSFT_s100
                                 + 0.0111376420489448 w_k1_MSFT_s101
                                 + 0.0149994996911541 w_k1_MSFT_s102
                                 + 0.0262406329207126 w_k1_MSFT_s103
                                 + 0.0474464771264524 w_k1_MSFT_s104
                                 + 0.0306972305894037 w_k1_MSFT_s105
                                 + 0.0106489299669356 w_k1_MSFT_s106
                                 + 0.0336719087192149 w_k1_MSFT_s107
                                 + 0.0122502460840775 w_k1_MSFT_s110
                                 + 0.0106564853784132 w_k1_MSFT_s111
                                 + 0.024531915154901 w_k1_MSFT_s112
                                 + 0.0136574789080829 w_k1_MSFT_s113
                                 + 0.0140377476086495 w_k1_MSFT_s114
                                 + 0.0256339579270534 w_k1_MSFT_s115
                                 + 0.0138794653779077 w_k1_MSFT_s116
                                 + 0.0249751696277888 w_k1_MSFT_s117
                                 + 0.0334525254065039 w_k1_PG_s000
                                 + 0.0477986866505226 w_k1_PG_s001
                                 + 0.0117686692415422 w_k1_PG_s002
                                 + 0.0112036540865173 w_k1_PG_s003
                                 + 0.0250623420368807 w_k1_PG_s004
                                 + 0.0176507402537996 w_k1_PG_s005
                                 + 0.0382010442131905 w_k1_PG_s006
                                 + 0.0224815342536007 w_k1_PG_s007
                                 + 0.0832608184084261 w_k1_PG_s010
                                 + 0.0544463617650739 w_k1_PG_s011
                                 + 0.0428776491809869 w_k1_PG_s012
                                 + 0.0796217402097838 w_k1_PG_s013
                                 + 0.0589125866297628 w_k1_PG_s014
                                 + 0.0407634086099957 w_k1_PG_s015
                                 + 0.0538960112823195 w_k1_PG_s016
                                 + 0.060873441768557 w_k1_PG_s017
                                 + 0.0132054727819905 w_k1_PG_s100
                                 + 0.0129911945889974 w_k1_PG_s101
                                 + 0.0171950820825327 w_k1_PG_s102
                                 + 0.0249519246711802 w_k1_PG_s103
                                 + 0.0478594121331106 w_k1_PG_s104
                                 + 0.0313444140125662 w_k1_PG_s105
                                 + 0.0115432979653875 w_k1_PG_s106
                                 + 0.0397954762682615 w_k1_PG_s107
                                 + 0.0126419513929112 w_k1_PG_s110
                                 + 0.0110840346801429 w_k1_PG_s111
                                 + 0.0249841397819211 w_k1_PG_s112
                                 + 0.0144347247054436 w_k1_PG_s113
                                 + 0.0147556010642799 w_k1_PG_s114
                                 + 0.0253313149114139 w_k1_PG_s115
                                 + 0.0132610183220739 w_k1_PG_s116
                                 + 0.0258225996830105 w_k1_PG_s117 >= 1.14
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

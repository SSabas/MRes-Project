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
 w_k1_ADM_s0_return_constraint:  1.12574 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.92674 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.08821 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.86727 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.749207 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.966643 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.912645 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.25618 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.845881 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.62819 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.855319 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.19535 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.02864 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.922324 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.841818 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.09547 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.908052 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.07748 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.34887 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.05923 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.46364 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.0024 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.25153 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.931431 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.25246 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.1086 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.12519 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.84772 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.876699 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.09136 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.79773 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.04197 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.26739 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.4425 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.935838 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.17064 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.994981 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.948123 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.18945 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.820649 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.17024 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.85049 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.735445 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.13602 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.883818 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.38143 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.701592 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.75478 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.89251 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.09912 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.21458 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.720495 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.711276 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.971401 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.773591 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.3279 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.40244 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.15234 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.71694 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.996234 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.30477 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.1267 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.53061 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.17868 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.914018 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.787267 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.885452 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.07412 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.709996 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.08312 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.60493 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.36065 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.753456 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.981894 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.10601 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.794796 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.26286 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.819511 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.31912 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.789643 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.631885 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.14488 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.968924 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.18956 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.631852 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.78412 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.73797 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.05891 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.29159 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.582159 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.746953 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.977025 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.812071 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.57742 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.23624 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.32117 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.37867 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.880624 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.40319 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.15642 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.83838 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.3628 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.07784 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.719826 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.939447 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.05578 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.591908 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.95216 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.69565 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.56309 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.782092 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.1648 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.975633 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.891145 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.1358 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.871059 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.21984 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.873921 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.747499 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.08001 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.993276 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.07447 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.82154 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.55226 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.802484 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.05952 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.23127 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.811396 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.77933 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.041 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.952742 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.17487 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.32892 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.13948 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.3373 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.85377 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.31245 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.855731 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.20413 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.1154 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.13234 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.839703 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.936797 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.904006 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.765288 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.10787 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.33383 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.40526 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.780554 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.01388 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.932481 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.901676 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.12572 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.908009 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.10161 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.991127 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.905609 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.04881 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.715892 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.80646 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.679541 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.21587 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.07988 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.12802 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.873449 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.446838 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.852684 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.796404 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.500947 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.73128 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.743571 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.05946 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.11969 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.29101 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.849788 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.64858 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.74962 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.08572 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.584986 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.703027 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.938414 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.31941 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.651718 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.856052 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.76865 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.994491 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.30937 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.32417 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.81142 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.80047 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.12322 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.859477 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.10066 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.828253 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.800938 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.05769 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.999623 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.986958 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.87176 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.50011 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.907974 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.0607 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.18459 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.806744 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.903945 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.16672 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.955465 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.32776 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.22454 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.14023 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.35025 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.994954 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.32769 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.02165 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.24321 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.09345 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.0778 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.01874 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.96773 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.00302 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.893035 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.18491 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.48547 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.35247 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.89488 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.12417 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.11795 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.965441 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.1204 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.819906 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.22185 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.998263 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.838874 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.0833 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.903277 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.24374 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.672857 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.31526 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.976105 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.04645 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.958516 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.556128 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.798676 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.737567 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.611061 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.58823 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.883476 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.05253 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.05067 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.965421 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.05984 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.31541 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.74591 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.2214 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.910008 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.772438 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.975093 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.12839 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.703387 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.989532 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.80519 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.26237 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.09072 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.43409 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.38169 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.965968 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.18521 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.889512 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.3062 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.07605 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.896384 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.961002 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.664095 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.92257 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.588057 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.29266 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.857942 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.943806 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.752533 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.370082 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.637301 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.783983 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.443557 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.61037 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.572986 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.08781 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.42809 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.07273 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.03061 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.50861 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.50981 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.04619 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.506477 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.628351 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.926304 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.3711 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.651801 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.27994 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 2.39681 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.25922 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.2938 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.39023 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 2.14338 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.884722 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.13036 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.88259 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.2154 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.857936 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.755625 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.03553 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.994958 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.00862 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.769933 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.30033 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.801248 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.03096 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.09213 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.737548 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.84103 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.05184 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.979818 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.3305 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.26342 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.23023 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.16718 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.0154 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.29439 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.83005 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.36515 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.32278 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.20083 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.888539 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.952928 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.00184 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.732913 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.10661 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.43344 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.387 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.877247 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.15859 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.997652 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.987864 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.06649 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.929781 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.16891 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.951532 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.826248 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.03519 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.855057 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.17533 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.826933 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.20303 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.848618 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.01077 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.09471 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.717742 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.901174 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.943511 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.836364 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.29899 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.958652 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.05539 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.26387 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.98613 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.16198 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.01448 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.22553 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.11117 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.979565 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.811976 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.927991 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.04904 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.813735 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.02825 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.3662 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.28305 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.938019 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.15312 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.00108 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.951878 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.469001878728012 z_k1_s0000
                                 - 0.641999807859179 z_k1_s0001
                                 - 0.326000772799282 z_k1_s0002
                                 - 0.810999894499937 z_k1_s0003
                                 - 0.557000558603482 z_k1_s0004
                                 - 0.525001038648765 z_k1_s0005
                                 - 0.500002986812091 z_k1_s0006
                                 - 0.365000502369682 z_k1_s0007
                                 - 0.485999457566346 z_k1_s0010
                                 - 0.347000577056764 z_k1_s0011
                                 - 0.194000336488822 z_k1_s0012
                                 - 0.678999882008483 z_k1_s0013
                                 - 0.194000336488822 z_k1_s0014
                                 - 0.730002430202356 z_k1_s0015
                                 - 0.590999477485254 z_k1_s0016
                                 - 0.481001748336344 z_k1_s0017
                                 - 0.43599987491276 z_k1_s0020
                                 - 0.231999986812275 z_k1_s0021
                                 - 0.400001063882842 z_k1_s0022
                                 - 0.328999705464059 z_k1_s0023
                                 - 0.20099994678134 z_k1_s0024
                                 - 0.247999350647833 z_k1_s0025
                                 - 0.395000353684557 z_k1_s0026
                                 - 0.182999814066642 z_k1_s0027
                                 - 0.196000376402789 z_k1_s0030
                                 - 0.223999734175268 z_k1_s0031
                                 - 0.202999809846289 z_k1_s0032
                                 - 0.143999945112564 z_k1_s0033
                                 - 0.204000192909212 z_k1_s0034
                                 - 0.09699996753832 z_k1_s0035
                                 - 0.117999891867299 z_k1_s0036
                                 - 0.437998885718269 z_k1_s0037
                                 - 0.772999605390251 z_k1_s0040
                                 - 0.284999378802945 z_k1_s0041
                                 - 0.54899730301758 z_k1_s0042
                                 - 0.25199932200185 z_k1_s0043
                                 - 0.733996663041095 z_k1_s0044
                                 - 0.328999032838717 z_k1_s0045
                                 - 0.661000562699015 z_k1_s0046
                                 - 0.633999174455784 z_k1_s0047
                                 - 0.358000078327163 z_k1_s0050
                                 - 0.934999520479737 z_k1_s0051
                                 - 0.423000068803309 z_k1_s0052
                                 - 0.503000742128219 z_k1_s0053
                                 - 0.553001668956599 z_k1_s0054
                                 - 0.46099866823158 z_k1_s0055
                                 - 0.404998601704393 z_k1_s0056
                                 - 0.410001932789228 z_k1_s0057
                                 - 0.248999477742434 z_k1_s0060
                                 - 0.36999872645345 z_k1_s0061
                                 - 0.365999628182189 z_k1_s0062
                                 - 0.396998699024267 z_k1_s0063
                                 - 0.389000502481745 z_k1_s0064
                                 - 0.226999589858654 z_k1_s0065
                                 - 0.53300035619277 z_k1_s0066
                                 - 0.161999835424903 z_k1_s0067
                                 - 0.394999477418152 z_k1_s0070
                                 - 0.183000330433987 z_k1_s0071
                                 - 0.22100038211974 z_k1_s0072
                                 - 0.352999909449478 z_k1_s0073
                                 - 0.110000019174605 z_k1_s0074
                                 - 0.0889999253904006 z_k1_s0075
                                 - 0.0949999743144494 z_k1_s0076
                                 - 0.102999936279892 z_k1_s0077
                                 - 0.701001499465469 z_k1_s0100
                                 - 1.29699710068415 z_k1_s0101
                                 - 0.600997573085503 z_k1_s0102
                                 - 0.323999430227386 z_k1_s0103
                                 - 0.478999174874108 z_k1_s0104
                                 - 0.535997303785727 z_k1_s0105
                                 - 0.508998830889964 z_k1_s0106
                                 - 0.62599728643205 z_k1_s0107
                                 - 0.793997073293587 z_k1_s0110
                                 - 0.490999665255923 z_k1_s0111
                                 - 0.389999846710704 z_k1_s0112
                                 - 0.570000090976874 z_k1_s0113
                                 - 0.243999967098943 z_k1_s0114
                                 - 0.958997638470575 z_k1_s0115
                                 - 1.04099881904856 z_k1_s0116
                                 - 0.629999831132598 z_k1_s0117
                                 - 0.24099970348432 z_k1_s0120
                                 - 0.714000489202959 z_k1_s0121
                                 - 0.631999197648528 z_k1_s0122
                                 - 0.299999581359778 z_k1_s0123
                                 - 0.241999808200759 z_k1_s0124
                                 - 0.610998988201555 z_k1_s0125
                                 - 0.171000000126862 z_k1_s0126
                                 - 0.404999302195817 z_k1_s0127
                                 - 0.206000067427525 z_k1_s0130
                                 - 0.919000691873452 z_k1_s0131
                                 - 0.428998302399224 z_k1_s0132
                                 - 0.282000002358028 z_k1_s0133
                                 - 0.534999197444837 z_k1_s0134
                                 - 0.813998741271193 z_k1_s0135
                                 - 0.576000227642029 z_k1_s0136
                                 - 0.267000126500741 z_k1_s0137
                                 - 0.0910000516353567 z_k1_s0140
                                 - 0.0830000424190581 z_k1_s0141
                                 - 0.306000246123342 z_k1_s0142
                                 - 0.192000008391007 z_k1_s0143
                                 - 0.0970000585475806 z_k1_s0144
                                 - 0.08600009907521 z_k1_s0145
                                 - 0.104000119811882 z_k1_s0146
                                 - 0.105000067763879 z_k1_s0147
                                 - 0.377001270844967 z_k1_s0150
                                 - 0.796003206070384 z_k1_s0151
                                 - 0.298001051295136 z_k1_s0152
                                 - 0.272001212299867 z_k1_s0153
                                 - 0.572002867520034 z_k1_s0154
                                 - 0.523001291206495 z_k1_s0155
                                 - 0.356001339256248 z_k1_s0156
                                 - 0.812003230252552 z_k1_s0157
                                 - 0.124000049918618 z_k1_s0160
                                 - 0.295000155387069 z_k1_s0161
                                 - 0.111000076430406 z_k1_s0162
                                 - 0.20999957173265 z_k1_s0163
                                 - 0.108000082548511 z_k1_s0164
                                 - 0.197000355245166 z_k1_s0165
                                 - 0.359000024867496 z_k1_s0166
                                 - 0.110000078469774 z_k1_s0167
                                 - 0.668000968137801 z_k1_s0170
                                 - 0.676998854738791 z_k1_s0171
                                 - 0.738001229254013 z_k1_s0172
                                 - 0.824998575727482 z_k1_s0173
                                 - 0.359999819226469 z_k1_s0174
                                 - 0.503998001637328 z_k1_s0175
                                 - 0.237999918195271 z_k1_s0176
                                 - 0.838000909705852 z_k1_s0177
                                 - 0.20999985444842 z_k1_s1000
                                 - 0.0580000075714681 z_k1_s1001
                                 - 0.0870000113572022 z_k1_s1002
                                 - 0.0739999606532528 z_k1_s1003
                                 - 0.108000331202042 z_k1_s1004
                                 - 0.0569999843788568 z_k1_s1005
                                 - 0.139999902965614 z_k1_s1006
                                 - 0.102000359246372 z_k1_s1007
                                 - 0.0960000621313114 z_k1_s1010
                                 - 0.128999660989169 z_k1_s1011
                                 - 0.0850001958453588 z_k1_s1012
                                 - 0.106000124936627 z_k1_s1013
                                 - 0.0429999996629979 z_k1_s1014
                                 - 0.103999841975704 z_k1_s1015
                                 - 0.0309999918965839 z_k1_s1016
                                 - 0.0820001094037992 z_k1_s1017
                                 - 0.0529999929316147 z_k1_s1020
                                 - 0.0819999890640076 z_k1_s1021
                                 - 0.0759999898642022 z_k1_s1022
                                 - 0.15799997892821 z_k1_s1023
                                 - 0.134999981995622 z_k1_s1024
                                 - 0.201999973060116 z_k1_s1025
                                 - 0.041999994398638 z_k1_s1026
                                 - 0.0519999930649804 z_k1_s1027
                                 - 0.158999128144797 z_k1_s1030
                                 - 0.303999729343727 z_k1_s1031
                                 - 0.140999808787823 z_k1_s1032
                                 - 0.15199911967309 z_k1_s1033
                                 - 0.186999162031622 z_k1_s1034
                                 - 0.267000110278292 z_k1_s1035
                                 - 0.15199911967309 z_k1_s1036
                                 - 0.127999729197617 z_k1_s1037
                                 - 0.0419999749422825 z_k1_s1040
                                 - 0.0599999866032593 z_k1_s1041
                                 - 0.146999990697984 z_k1_s1042
                                 - 0.145000006824541 z_k1_s1043
                                 - 0.0849997066213014 z_k1_s1044
                                 - 0.0609999785399808 z_k1_s1045
                                 - 0.0960000099252129 z_k1_s1046
                                 - 0.148000374634681 z_k1_s1047
                                 - 0.114000117385837 z_k1_s1050
                                 - 0.359001491242753 z_k1_s1051
                                 - 0.269000725621522 z_k1_s1052
                                 - 0.323000332593203 z_k1_s1053
                                 - 0.387000734968125 z_k1_s1054
                                 - 0.386001294728594 z_k1_s1055
                                 - 0.137000242011174 z_k1_s1056
                                 - 0.15600026157548 z_k1_s1057
                                 - 0.180999585876498 z_k1_s1060
                                 - 0.368999732864251 z_k1_s1061
                                 - 0.0750000178919221 z_k1_s1062
                                 - 0.284000629513421 z_k1_s1063
                                 - 0.10900000743226 z_k1_s1064
                                 - 0.248999564957173 z_k1_s1065
                                 - 0.277999402418047 z_k1_s1066
                                 - 0.19600004211489 z_k1_s1067
                                 - 0.0659999919912174 z_k1_s1070
                                 - 0.067000012021388 z_k1_s1071
                                 - 0.087000202624793 z_k1_s1072
                                 - 0.0940002028359827 z_k1_s1073
                                 - 0.0610000318403688 z_k1_s1074
                                 - 0.187000105641786 z_k1_s1075
                                 - 0.0579999717498571 z_k1_s1076
                                 - 0.0800002024136032 z_k1_s1077
                                 - 0.649997158885772 z_k1_s1100
                                 - 0.660000918710915 z_k1_s1101
                                 - 0.882996121901768 z_k1_s1102
                                 - 1.32099678640987 z_k1_s1103
                                 - 0.796001400384383 z_k1_s1104
                                 - 0.748997598192103 z_k1_s1105
                                 - 0.449999749303789 z_k1_s1106
                                 - 0.520999309089906 z_k1_s1107
                                 - 0.7480002908547 z_k1_s1110
                                 - 1.3230015851899 z_k1_s1111
                                 - 0.420999872805479 z_k1_s1112
                                 - 0.681000679485878 z_k1_s1113
                                 - 0.855998121203518 z_k1_s1114
                                 - 0.44200052259617 z_k1_s1115
                                 - 1.74700375539538 z_k1_s1116
                                 - 1.49800057010847 z_k1_s1117
                                 - 0.569001796276814 z_k1_s1120
                                 - 0.237000078601126 z_k1_s1121
                                 - 0.593000535337473 z_k1_s1122
                                 - 0.652999135489417 z_k1_s1123
                                 - 0.622999835413445 z_k1_s1124
                                 - 0.358999685743826 z_k1_s1125
                                 - 0.210000007532633 z_k1_s1126
                                 - 0.260999869161962 z_k1_s1127
                                 - 0.543999676321937 z_k1_s1130
                                 - 0.502003316841145 z_k1_s1131
                                 - 0.859004691525983 z_k1_s1132
                                 - 0.21800062734528 z_k1_s1133
                                 - 0.734003654082825 z_k1_s1134
                                 - 0.280000661747629 z_k1_s1135
                                 - 0.454001607230991 z_k1_s1136
                                 - 1.02600208844993 z_k1_s1137
                                 - 0.587000655417517 z_k1_s1140
                                 - 0.297999938660048 z_k1_s1141
                                 - 0.117999893410365 z_k1_s1142
                                 - 0.346000433806572 z_k1_s1143
                                 - 0.578999179393267 z_k1_s1144
                                 - 0.447001301489903 z_k1_s1145
                                 - 0.174999889159433 z_k1_s1146
                                 - 0.237000040998752 z_k1_s1147
                                 - 0.298999902569843 z_k1_s1150
                                 - 0.317999904511739 z_k1_s1151
                                 - 0.465998914006454 z_k1_s1152
                                 - 0.218999930961858 z_k1_s1153
                                 - 0.455000229345435 z_k1_s1154
                                 - 0.462000339766141 z_k1_s1155
                                 - 0.392999747619217 z_k1_s1156
                                 - 0.862000197806063 z_k1_s1157
                                 - 0.457999250591587 z_k1_s1160
                                 - 0.26299936500145 z_k1_s1161
                                 - 0.265998882549383 z_k1_s1162
                                 - 0.394998967093497 z_k1_s1163
                                 - 0.14199986861696 z_k1_s1164
                                 - 0.321000452586317 z_k1_s1165
                                 - 0.133999905356825 z_k1_s1166
                                 - 0.103999938981406 z_k1_s1167
                                 - 1.49099592378593 z_k1_s1170
                                 - 1.1639942935446 z_k1_s1171
                                 - 0.687998336696189 z_k1_s1172
                                 - 0.390998893334958 z_k1_s1173
                                 - 1.24399627989989 z_k1_s1174
                                 - 0.621998860948379 z_k1_s1175
                                 - 1.09799409679835 z_k1_s1176
                                 - 0.511999013703595 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.45702 w_k1_ADM_s000
                                 + 1.42454 w_k1_AXP_s000
                                 + 1.06395 w_k1_DIS_s000
                                 + 1.25012 w_k1_FDX_s000
                                 + 1.14793 w_k1_IBM_s000
                                 + 1.24468 w_k1_INTC_s000
                                 + 1.01604 w_k1_KO_s000 + 1.11444 w_k1_MAT_s000
                                 + 1.14047 w_k1_MSFT_s000
                                 + 1.21456 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.974597 w_k1_ADM_s000
                                 + 0.628182 w_k1_AXP_s000
                                 + 0.753368 w_k1_DIS_s000
                                 + 0.836306 w_k1_FDX_s000
                                 + 0.841589 w_k1_IBM_s000
                                 + 0.86118 w_k1_INTC_s000
                                 + 0.811401 w_k1_KO_s000
                                 + 0.63037 w_k1_MAT_s000
                                 + 0.898223 w_k1_MSFT_s000
                                 + 0.959363 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.21246 w_k1_ADM_s000
                                 + 1.19455 w_k1_AXP_s000
                                 + 1.31459 w_k1_DIS_s000
                                 + 1.08988 w_k1_FDX_s000
                                 + 0.970231 w_k1_IBM_s000
                                 + 1.23531 w_k1_INTC_s000
                                 + 1.13803 w_k1_KO_s000
                                 + 0.941113 w_k1_MAT_s000
                                 + 1.19017 w_k1_MSFT_s000 + 1.1871 w_k1_PG_s000
                                 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.28729 w_k1_ADM_s000
                                 + 0.952442 w_k1_AXP_s000
                                 + 0.663435 w_k1_DIS_s000
                                 + 0.89242 w_k1_FDX_s000
                                 + 0.910805 w_k1_IBM_s000
                                 + 0.868781 w_k1_INTC_s000
                                 + 0.73966 w_k1_KO_s000 + 0.7832 w_k1_MAT_s000
                                 + 0.785423 w_k1_MSFT_s000
                                 + 0.871622 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.04366 w_k1_ADM_s000
                                 + 1.04174 w_k1_AXP_s000
                                 + 1.09114 w_k1_DIS_s000
                                 + 0.936886 w_k1_FDX_s000
                                 + 1.47935 w_k1_IBM_s000
                                 + 0.943552 w_k1_INTC_s000
                                 + 1.25907 w_k1_KO_s000 + 1.17919 w_k1_MAT_s000
                                 + 0.915108 w_k1_MSFT_s000
                                 + 1.22755 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.49686 w_k1_ADM_s000
                                 + 1.29096 w_k1_AXP_s000
                                 + 1.62706 w_k1_DIS_s000
                                 + 1.34408 w_k1_FDX_s000
                                 + 1.31738 w_k1_IBM_s000
                                 + 1.28753 w_k1_INTC_s000
                                 + 1.30435 w_k1_KO_s000 + 1.17472 w_k1_MAT_s000
                                 + 1.32305 w_k1_MSFT_s000
                                 + 1.30425 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.968738 w_k1_ADM_s000
                                 + 0.968441 w_k1_AXP_s000
                                 + 1.01854 w_k1_DIS_s000
                                 + 1.00758 w_k1_FDX_s000
                                 + 0.941481 w_k1_IBM_s000
                                 + 1.04346 w_k1_INTC_s000
                                 + 0.966637 w_k1_KO_s000
                                 + 1.02833 w_k1_MAT_s000
                                 + 1.05182 w_k1_MSFT_s000
                                 + 1.02363 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.20874 w_k1_ADM_s000
                                 + 1.1273 w_k1_AXP_s000
                                 + 0.866699 w_k1_DIS_s000
                                 + 1.11174 w_k1_FDX_s000
                                 + 0.893533 w_k1_IBM_s000
                                 + 1.05618 w_k1_INTC_s000
                                 + 0.87082 w_k1_KO_s000
                                 + 0.866173 w_k1_MAT_s000
                                 + 0.852656 w_k1_MSFT_s000
                                 + 1.13149 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.18666 w_k1_ADM_s001
                                 + 1.34405 w_k1_AXP_s001
                                 + 1.67091 w_k1_DIS_s001
                                 + 1.45588 w_k1_FDX_s001
                                 + 1.11182 w_k1_IBM_s001
                                 + 1.29686 w_k1_INTC_s001
                                 + 1.40168 w_k1_KO_s001 + 1.22436 w_k1_MAT_s001
                                 + 1.29067 w_k1_MSFT_s001
                                 + 1.22571 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.935379 w_k1_ADM_s001
                                 + 1.22249 w_k1_AXP_s001
                                 + 0.974542 w_k1_DIS_s001
                                 + 1.05495 w_k1_FDX_s001
                                 + 0.693109 w_k1_IBM_s001
                                 + 0.998112 w_k1_INTC_s001
                                 + 0.82821 w_k1_KO_s001
                                 + 0.709468 w_k1_MAT_s001
                                 + 0.975349 w_k1_MSFT_s001
                                 + 0.928358 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.674821 w_k1_ADM_s001
                                 + 0.759069 w_k1_AXP_s001
                                 + 0.781512 w_k1_DIS_s001
                                 + 0.644979 w_k1_FDX_s001
                                 + 1.27276 w_k1_IBM_s001
                                 + 0.829711 w_k1_INTC_s001
                                 + 1.04587 w_k1_KO_s001
                                 + 0.947573 w_k1_MAT_s001
                                 + 0.854775 w_k1_MSFT_s001
                                 + 0.848585 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.992622 w_k1_ADM_s001
                                 + 1.00524 w_k1_AXP_s001
                                 + 1.15442 w_k1_DIS_s001
                                 + 1.03664 w_k1_FDX_s001
                                 + 1.04589 w_k1_IBM_s001
                                 + 1.01168 w_k1_INTC_s001 + 1.3013 w_k1_KO_s001
                                 + 1.09024 w_k1_MAT_s001
                                 + 1.09795 w_k1_MSFT_s001
                                 + 1.10808 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.743247 w_k1_ADM_s001
                                 + 0.617936 w_k1_AXP_s001
                                 + 0.905666 w_k1_DIS_s001
                                 + 0.826252 w_k1_FDX_s001
                                 + 0.700469 w_k1_IBM_s001
                                 + 0.932079 w_k1_INTC_s001
                                 + 0.820171 w_k1_KO_s001
                                 + 0.606274 w_k1_MAT_s001
                                 + 0.969444 w_k1_MSFT_s001
                                 + 0.849034 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.678757 w_k1_ADM_s001
                                 + 0.665714 w_k1_AXP_s001
                                 + 0.71506 w_k1_DIS_s001
                                 + 0.791508 w_k1_FDX_s001
                                 + 0.750975 w_k1_IBM_s001
                                 + 1.05923 w_k1_INTC_s001
                                 + 0.824813 w_k1_KO_s001
                                 + 0.612402 w_k1_MAT_s001
                                 + 0.876211 w_k1_MSFT_s001
                                 + 0.754742 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.717466 w_k1_ADM_s001
                                 + 0.833937 w_k1_AXP_s001
                                 + 0.766153 w_k1_DIS_s001
                                 + 0.803826 w_k1_FDX_s001
                                 + 0.997756 w_k1_IBM_s001
                                 + 0.990285 w_k1_INTC_s001
                                 + 0.844057 w_k1_KO_s001
                                 + 0.892762 w_k1_MAT_s001
                                 + 0.829366 w_k1_MSFT_s001
                                 + 0.825866 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.26562 w_k1_ADM_s001
                                 + 1.35689 w_k1_AXP_s001
                                 + 1.35534 w_k1_DIS_s001
                                 + 1.07802 w_k1_FDX_s001
                                 + 1.60018 w_k1_IBM_s001
                                 + 1.13814 w_k1_INTC_s001
                                 + 1.49556 w_k1_KO_s001 + 1.74461 w_k1_MAT_s001
                                 + 1.24293 w_k1_MSFT_s001
                                 + 1.15473 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.967672 w_k1_ADM_s002
                                 + 0.971607 w_k1_AXP_s002
                                 + 1.01369 w_k1_DIS_s002
                                 + 0.894173 w_k1_FDX_s002
                                 + 1.33707 w_k1_IBM_s002
                                 + 0.782071 w_k1_INTC_s002
                                 + 1.13342 w_k1_KO_s002 + 1.18321 w_k1_MAT_s002
                                 + 1.01933 w_k1_MSFT_s002
                                 + 0.940229 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.657803 w_k1_ADM_s002
                                 + 0.70236 w_k1_AXP_s002
                                 + 0.700434 w_k1_DIS_s002
                                 + 0.712803 w_k1_FDX_s002
                                 + 1.03943 w_k1_IBM_s002
                                 + 0.860114 w_k1_INTC_s002
                                 + 0.902473 w_k1_KO_s002
                                 + 0.814281 w_k1_MAT_s002
                                 + 0.832145 w_k1_MSFT_s002
                                 + 0.816688 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.759689 w_k1_ADM_s002
                                 + 0.701204 w_k1_AXP_s002
                                 + 0.726092 w_k1_DIS_s002
                                 + 0.774208 w_k1_FDX_s002
                                 + 0.768512 w_k1_IBM_s002
                                 + 0.785696 w_k1_INTC_s002
                                 + 0.751974 w_k1_KO_s002
                                 + 0.720737 w_k1_MAT_s002
                                 + 0.80659 w_k1_MSFT_s002
                                 + 0.887142 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.995227 w_k1_ADM_s002
                                 + 1.01802 w_k1_AXP_s002
                                 + 0.934955 w_k1_DIS_s002
                                 + 0.909777 w_k1_FDX_s002
                                 + 0.877667 w_k1_IBM_s002
                                 + 1.05237 w_k1_INTC_s002
                                 + 1.04615 w_k1_KO_s002
                                 + 0.775191 w_k1_MAT_s002
                                 + 0.95827 w_k1_MSFT_s002
                                 + 0.912987 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.13931 w_k1_ADM_s002
                                 + 1.23485 w_k1_AXP_s002
                                 + 0.914401 w_k1_DIS_s002
                                 + 1.07406 w_k1_FDX_s002
                                 + 0.896531 w_k1_IBM_s002
                                 + 0.999912 w_k1_INTC_s002
                                 + 0.934451 w_k1_KO_s002
                                 + 1.10363 w_k1_MAT_s002
                                 + 1.03452 w_k1_MSFT_s002
                                 + 1.04498 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.25974 w_k1_ADM_s002
                                 + 1.37715 w_k1_AXP_s002
                                 + 1.35854 w_k1_DIS_s002
                                 + 1.15436 w_k1_FDX_s002
                                 + 1.02706 w_k1_IBM_s002
                                 + 1.51903 w_k1_INTC_s002
                                 + 1.33269 w_k1_KO_s002
                                 + 0.986966 w_k1_MAT_s002
                                 + 1.26131 w_k1_MSFT_s002
                                 + 1.13337 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.44162 w_k1_ADM_s002
                                 + 1.48915 w_k1_AXP_s002
                                 + 1.59412 w_k1_DIS_s002
                                 + 1.23197 w_k1_FDX_s002
                                 + 1.29733 w_k1_IBM_s002
                                 + 1.13638 w_k1_INTC_s002
                                 + 1.37843 w_k1_KO_s002 + 1.19938 w_k1_MAT_s002
                                 + 1.3137 w_k1_MSFT_s002 + 1.15 w_k1_PG_s002
                                 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.19461 w_k1_ADM_s002
                                 + 1.07204 w_k1_AXP_s002
                                 + 1.21373 w_k1_DIS_s002
                                 + 1.13348 w_k1_FDX_s002
                                 + 0.810636 w_k1_IBM_s002
                                 + 1.09166 w_k1_INTC_s002 + 1.0358 w_k1_KO_s002
                                 + 0.757328 w_k1_MAT_s002
                                 + 1.26414 w_k1_MSFT_s002
                                 + 1.00578 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.945438 w_k1_ADM_s003
                                 + 1.15264 w_k1_AXP_s003
                                 + 1.03386 w_k1_DIS_s003
                                 + 1.15476 w_k1_FDX_s003
                                 + 0.612307 w_k1_IBM_s003
                                 + 1.09788 w_k1_INTC_s003
                                 + 0.854442 w_k1_KO_s003
                                 + 0.585157 w_k1_MAT_s003
                                 + 1.17904 w_k1_MSFT_s003
                                 + 1.12998 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.755019 w_k1_ADM_s003
                                 + 0.779087 w_k1_AXP_s003
                                 + 0.609105 w_k1_DIS_s003
                                 + 0.752197 w_k1_FDX_s003
                                 + 0.73883 w_k1_IBM_s003
                                 + 0.62171 w_k1_INTC_s003
                                 + 0.949533 w_k1_KO_s003
                                 + 0.759416 w_k1_MAT_s003
                                 + 0.850161 w_k1_MSFT_s003
                                 + 0.943727 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.963171 w_k1_ADM_s003
                                 + 1.29279 w_k1_AXP_s003
                                 + 1.37182 w_k1_DIS_s003
                                 + 1.07339 w_k1_FDX_s003
                                 + 1.82176 w_k1_IBM_s003
                                 + 1.02759 w_k1_INTC_s003
                                 + 1.44614 w_k1_KO_s003 + 1.55051 w_k1_MAT_s003
                                 + 1.26344 w_k1_MSFT_s003
                                 + 1.22193 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.730794 w_k1_ADM_s003
                                 + 0.673796 w_k1_AXP_s003
                                 + 0.787121 w_k1_DIS_s003
                                 + 0.926557 w_k1_FDX_s003
                                 + 0.669495 w_k1_IBM_s003
                                 + 0.822451 w_k1_INTC_s003
                                 + 0.802752 w_k1_KO_s003
                                 + 0.57977 w_k1_MAT_s003
                                 + 0.938884 w_k1_MSFT_s003
                                 + 0.886456 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.751921 w_k1_ADM_s003
                                 + 0.77285 w_k1_AXP_s003
                                 + 0.868882 w_k1_DIS_s003
                                 + 0.753755 w_k1_FDX_s003
                                 + 1.21763 w_k1_IBM_s003
                                 + 0.825405 w_k1_INTC_s003
                                 + 1.0379 w_k1_KO_s003 + 0.814658 w_k1_MAT_s003
                                 + 0.905384 w_k1_MSFT_s003
                                 + 1.06664 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.929512 w_k1_ADM_s003
                                 + 1.41658 w_k1_AXP_s003
                                 + 0.86385 w_k1_DIS_s003
                                 + 0.960742 w_k1_FDX_s003
                                 + 1.18569 w_k1_IBM_s003
                                 + 1.01176 w_k1_INTC_s003
                                 + 0.974491 w_k1_KO_s003
                                 + 1.87369 w_k1_MAT_s003
                                 + 0.979991 w_k1_MSFT_s003
                                 + 1.04229 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.66734 w_k1_ADM_s003
                                 + 1.62985 w_k1_AXP_s003
                                 + 1.82427 w_k1_DIS_s003
                                 + 1.39725 w_k1_FDX_s003
                                 + 1.31911 w_k1_IBM_s003
                                 + 1.08167 w_k1_INTC_s003
                                 + 1.22611 w_k1_KO_s003 + 1.43829 w_k1_MAT_s003
                                 + 1.32579 w_k1_MSFT_s003
                                 + 1.33236 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.11339 w_k1_ADM_s003
                                 + 1.02354 w_k1_AXP_s003
                                 + 1.15048 w_k1_DIS_s003
                                 + 1.04092 w_k1_FDX_s003
                                 + 1.00464 w_k1_IBM_s003
                                 + 1.1419 w_k1_INTC_s003 + 1.14003 w_k1_KO_s003
                                 + 0.8148 w_k1_MAT_s003
                                 + 1.07488 w_k1_MSFT_s003
                                 + 1.15607 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.889805 w_k1_ADM_s004
                                 + 1.02558 w_k1_AXP_s004
                                 + 0.919008 w_k1_DIS_s004
                                 + 0.986528 w_k1_FDX_s004
                                 + 0.644944 w_k1_IBM_s004
                                 + 1.02938 w_k1_INTC_s004
                                 + 0.713424 w_k1_KO_s004
                                 + 0.804926 w_k1_MAT_s004
                                 + 0.90289 w_k1_MSFT_s004
                                 + 0.975645 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.698939 w_k1_ADM_s004
                                 + 0.474165 w_k1_AXP_s004
                                 + 0.605362 w_k1_DIS_s004
                                 + 0.675504 w_k1_FDX_s004
                                 + 0.717165 w_k1_IBM_s004
                                 + 0.901104 w_k1_INTC_s004
                                 + 0.657699 w_k1_KO_s004
                                 + 0.51162 w_k1_MAT_s004
                                 + 0.772627 w_k1_MSFT_s004
                                 + 0.771956 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.09379 w_k1_ADM_s004
                                 + 0.910525 w_k1_AXP_s004
                                 + 1.08394 w_k1_DIS_s004
                                 + 0.987052 w_k1_FDX_s004
                                 + 1.06918 w_k1_IBM_s004
                                 + 1.15027 w_k1_INTC_s004
                                 + 0.988522 w_k1_KO_s004
                                 + 1.19592 w_k1_MAT_s004
                                 + 1.12386 w_k1_MSFT_s004
                                 + 1.07468 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.874006 w_k1_ADM_s004
                                 + 0.69858 w_k1_AXP_s004
                                 + 0.736473 w_k1_DIS_s004
                                 + 0.754178 w_k1_FDX_s004
                                 + 0.984952 w_k1_IBM_s004
                                 + 0.768319 w_k1_INTC_s004
                                 + 0.776305 w_k1_KO_s004
                                 + 0.764248 w_k1_MAT_s004
                                 + 0.826414 w_k1_MSFT_s004
                                 + 0.929889 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.02786 w_k1_ADM_s004
                                 + 0.9649 w_k1_AXP_s004 + 0.87252 w_k1_DIS_s004
                                 + 0.868374 w_k1_FDX_s004
                                 + 1.17997 w_k1_IBM_s004
                                 + 1.07538 w_k1_INTC_s004
                                 + 0.937952 w_k1_KO_s004
                                 + 0.948929 w_k1_MAT_s004
                                 + 0.997775 w_k1_MSFT_s004
                                 + 1.05658 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.15786 w_k1_ADM_s004
                                 + 0.776611 w_k1_AXP_s004
                                 + 0.873103 w_k1_DIS_s004
                                 + 0.939944 w_k1_FDX_s004
                                 + 0.811847 w_k1_IBM_s004
                                 + 0.95471 w_k1_INTC_s004
                                 + 0.786437 w_k1_KO_s004
                                 + 0.865812 w_k1_MAT_s004
                                 + 1.07161 w_k1_MSFT_s004
                                 + 0.985696 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.21397 w_k1_ADM_s004
                                 + 1.06246 w_k1_AXP_s004
                                 + 1.37333 w_k1_DIS_s004
                                 + 1.13464 w_k1_FDX_s004
                                 + 1.15682 w_k1_IBM_s004
                                 + 1.10528 w_k1_INTC_s004
                                 + 1.20497 w_k1_KO_s004 + 1.40686 w_k1_MAT_s004
                                 + 1.31563 w_k1_MSFT_s004
                                 + 1.21096 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.2359 w_k1_ADM_s004
                                 + 1.43141 w_k1_AXP_s004
                                 + 1.66647 w_k1_DIS_s004
                                 + 1.39884 w_k1_FDX_s004
                                 + 1.12895 w_k1_IBM_s004
                                 + 1.23506 w_k1_INTC_s004 + 1.0134 w_k1_KO_s004
                                 + 1.16769 w_k1_MAT_s004
                                 + 1.26612 w_k1_MSFT_s004
                                 + 1.28678 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.01679 w_k1_ADM_s005
                                 + 1.17819 w_k1_AXP_s005
                                 + 0.956993 w_k1_DIS_s005
                                 + 0.958982 w_k1_FDX_s005
                                 + 1.05592 w_k1_IBM_s005
                                 + 0.978235 w_k1_INTC_s005
                                 + 0.893403 w_k1_KO_s005
                                 + 1.46251 w_k1_MAT_s005
                                 + 0.921254 w_k1_MSFT_s005
                                 + 1.05846 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.02541 w_k1_ADM_s005
                                 + 1.43231 w_k1_AXP_s005
                                 + 1.40565 w_k1_DIS_s005
                                 + 1.20088 w_k1_FDX_s005
                                 + 1.30688 w_k1_IBM_s005
                                 + 1.00071 w_k1_INTC_s005
                                 + 1.46453 w_k1_KO_s005 + 1.69185 w_k1_MAT_s005
                                 + 1.10431 w_k1_MSFT_s005
                                 + 1.35705 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.06862 w_k1_ADM_s005
                                 + 1.14304 w_k1_AXP_s005
                                 + 0.997041 w_k1_DIS_s005
                                 + 1.03767 w_k1_FDX_s005
                                 + 0.750635 w_k1_IBM_s005
                                 + 0.792476 w_k1_INTC_s005
                                 + 0.897192 w_k1_KO_s005
                                 + 0.658631 w_k1_MAT_s005
                                 + 0.950895 w_k1_MSFT_s005
                                 + 0.993465 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.851322 w_k1_ADM_s005
                                 + 0.934994 w_k1_AXP_s005
                                 + 0.618239 w_k1_DIS_s005
                                 + 0.761049 w_k1_FDX_s005
                                 + 0.763914 w_k1_IBM_s005
                                 + 0.828447 w_k1_INTC_s005
                                 + 0.679907 w_k1_KO_s005
                                 + 0.976103 w_k1_MAT_s005
                                 + 0.736479 w_k1_MSFT_s005
                                 + 0.818537 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.782958 w_k1_ADM_s005
                                 + 0.813908 w_k1_AXP_s005
                                 + 0.782576 w_k1_DIS_s005
                                 + 0.800786 w_k1_FDX_s005
                                 + 1.10333 w_k1_IBM_s005
                                 + 0.872515 w_k1_INTC_s005
                                 + 0.867253 w_k1_KO_s005
                                 + 1.05156 w_k1_MAT_s005
                                 + 0.779919 w_k1_MSFT_s005
                                 + 0.836039 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.958645 w_k1_ADM_s005
                                 + 0.978214 w_k1_AXP_s005
                                 + 1.21273 w_k1_DIS_s005
                                 + 0.937496 w_k1_FDX_s005
                                 + 1.39249 w_k1_IBM_s005
                                 + 0.948116 w_k1_INTC_s005
                                 + 1.29809 w_k1_KO_s005 + 1.34588 w_k1_MAT_s005
                                 + 1.02391 w_k1_MSFT_s005
                                 + 1.12936 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.980009 w_k1_ADM_s005
                                 + 0.851812 w_k1_AXP_s005
                                 + 0.862136 w_k1_DIS_s005
                                 + 0.97726 w_k1_FDX_s005
                                 + 0.77984 w_k1_IBM_s005
                                 + 1.06596 w_k1_INTC_s005
                                 + 0.860471 w_k1_KO_s005
                                 + 0.53732 w_k1_MAT_s005
                                 + 0.90309 w_k1_MSFT_s005
                                 + 0.996731 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.3489 w_k1_ADM_s005
                                 + 1.43812 w_k1_AXP_s005
                                 + 1.15105 w_k1_DIS_s005 + 1.2254 w_k1_FDX_s005
                                 + 0.938171 w_k1_IBM_s005
                                 + 1.32184 w_k1_INTC_s005
                                 + 1.15513 w_k1_KO_s005 + 1.4031 w_k1_MAT_s005
                                 + 1.02438 w_k1_MSFT_s005
                                 + 1.13043 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.58962 w_k1_ADM_s006
                                 + 0.988713 w_k1_AXP_s006
                                 + 1.02344 w_k1_DIS_s006
                                 + 1.08647 w_k1_FDX_s006
                                 + 0.721939 w_k1_IBM_s006
                                 + 0.929366 w_k1_INTC_s006
                                 + 0.910484 w_k1_KO_s006
                                 + 0.692136 w_k1_MAT_s006
                                 + 1.11705 w_k1_MSFT_s006
                                 + 0.879639 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.22147 w_k1_ADM_s006
                                 + 0.924016 w_k1_AXP_s006
                                 + 1.05445 w_k1_DIS_s006
                                 + 1.05998 w_k1_FDX_s006 + 1.0857 w_k1_IBM_s006
                                 + 0.937488 w_k1_INTC_s006
                                 + 1.00133 w_k1_KO_s006
                                 + 0.858926 w_k1_MAT_s006
                                 + 1.03577 w_k1_MSFT_s006
                                 + 0.988206 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.31788 w_k1_ADM_s006
                                 + 1.04225 w_k1_AXP_s006 + 1.1491 w_k1_DIS_s006
                                 + 0.979706 w_k1_FDX_s006
                                 + 1.51124 w_k1_IBM_s006
                                 + 0.880003 w_k1_INTC_s006
                                 + 1.42116 w_k1_KO_s006 + 1.60549 w_k1_MAT_s006
                                 + 1.05427 w_k1_MSFT_s006
                                 + 1.20336 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.20348 w_k1_ADM_s006
                                 + 0.812904 w_k1_AXP_s006
                                 + 0.860132 w_k1_DIS_s006
                                 + 0.954303 w_k1_FDX_s006
                                 + 0.78676 w_k1_IBM_s006
                                 + 1.04923 w_k1_INTC_s006 + 1.0094 w_k1_KO_s006
                                 + 0.787544 w_k1_MAT_s006
                                 + 1.09528 w_k1_MSFT_s006
                                 + 1.00779 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.45651 w_k1_ADM_s006
                                 + 1.16971 w_k1_AXP_s006
                                 + 1.34297 w_k1_DIS_s006
                                 + 1.24351 w_k1_FDX_s006
                                 + 1.23179 w_k1_IBM_s006
                                 + 1.12514 w_k1_INTC_s006
                                 + 1.26891 w_k1_KO_s006 + 1.37811 w_k1_MAT_s006
                                 + 1.33098 w_k1_MSFT_s006
                                 + 1.03406 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.46126 w_k1_ADM_s006
                                 + 1.49761 w_k1_AXP_s006
                                 + 0.915633 w_k1_DIS_s006
                                 + 1.1827 w_k1_FDX_s006
                                 + 0.745131 w_k1_IBM_s006
                                 + 1.19207 w_k1_INTC_s006
                                 + 0.894498 w_k1_KO_s006
                                 + 1.04064 w_k1_MAT_s006
                                 + 1.17658 w_k1_MSFT_s006 + 1.1326 w_k1_PG_s006
                                 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.11161 w_k1_ADM_s006
                                 + 0.856574 w_k1_AXP_s006
                                 + 0.665321 w_k1_DIS_s006
                                 + 0.828679 w_k1_FDX_s006
                                 + 0.837325 w_k1_IBM_s006
                                 + 0.909632 w_k1_INTC_s006
                                 + 0.933684 w_k1_KO_s006
                                 + 0.826999 w_k1_MAT_s006
                                 + 0.890747 w_k1_MSFT_s006
                                 + 0.764198 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.76004 w_k1_ADM_s006
                                 + 1.59606 w_k1_AXP_s006 + 1.6532 w_k1_DIS_s006
                                 + 1.5295 w_k1_FDX_s006 + 1.09137 w_k1_IBM_s006
                                 + 1.74588 w_k1_INTC_s006
                                 + 1.21496 w_k1_KO_s006 + 1.10051 w_k1_MAT_s006
                                 + 1.6269 w_k1_MSFT_s006 + 1.15902 w_k1_PG_s006
                                 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.94596 w_k1_ADM_s007
                                 + 0.916128 w_k1_AXP_s007
                                 + 0.956795 w_k1_DIS_s007
                                 + 0.999356 w_k1_FDX_s007
                                 + 0.881769 w_k1_IBM_s007
                                 + 1.09707 w_k1_INTC_s007
                                 + 0.968477 w_k1_KO_s007
                                 + 0.818863 w_k1_MAT_s007
                                 + 1.05926 w_k1_MSFT_s007
                                 + 1.02881 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.03853 w_k1_ADM_s007
                                 + 1.171 w_k1_AXP_s007 + 1.28075 w_k1_DIS_s007
                                 + 1.06689 w_k1_FDX_s007
                                 + 1.59086 w_k1_IBM_s007
                                 + 1.02589 w_k1_INTC_s007
                                 + 1.38829 w_k1_KO_s007 + 1.35142 w_k1_MAT_s007
                                 + 0.954695 w_k1_MSFT_s007
                                 + 1.15551 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.914145 w_k1_ADM_s007
                                 + 0.617521 w_k1_AXP_s007
                                 + 0.666133 w_k1_DIS_s007
                                 + 0.733917 w_k1_FDX_s007
                                 + 0.883459 w_k1_IBM_s007
                                 + 0.734931 w_k1_INTC_s007
                                 + 0.740534 w_k1_KO_s007
                                 + 0.824293 w_k1_MAT_s007
                                 + 0.766005 w_k1_MSFT_s007
                                 + 0.885007 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.781473 w_k1_ADM_s007
                                 + 0.859462 w_k1_AXP_s007
                                 + 0.864553 w_k1_DIS_s007
                                 + 0.8853 w_k1_FDX_s007 + 1.16753 w_k1_IBM_s007
                                 + 0.899628 w_k1_INTC_s007
                                 + 1.18147 w_k1_KO_s007 + 1.39082 w_k1_MAT_s007
                                 + 0.967897 w_k1_MSFT_s007
                                 + 1.0146 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.27689 w_k1_ADM_s007
                                 + 1.22166 w_k1_AXP_s007
                                 + 1.39488 w_k1_DIS_s007 + 1.1698 w_k1_FDX_s007
                                 + 1.03567 w_k1_IBM_s007
                                 + 1.04569 w_k1_INTC_s007
                                 + 1.05924 w_k1_KO_s007 + 1.05514 w_k1_MAT_s007
                                 + 1.13864 w_k1_MSFT_s007
                                 + 1.10071 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.10625 w_k1_ADM_s007
                                 + 1.44828 w_k1_AXP_s007
                                 + 1.26735 w_k1_DIS_s007
                                 + 1.41635 w_k1_FDX_s007
                                 + 0.815838 w_k1_IBM_s007
                                 + 0.988754 w_k1_INTC_s007
                                 + 0.829967 w_k1_KO_s007
                                 + 0.787811 w_k1_MAT_s007
                                 + 1.14327 w_k1_MSFT_s007
                                 + 1.07322 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.38757 w_k1_ADM_s007
                                 + 1.40274 w_k1_AXP_s007
                                 + 1.78713 w_k1_DIS_s007
                                 + 1.49683 w_k1_FDX_s007
                                 + 1.32448 w_k1_IBM_s007
                                 + 1.37747 w_k1_INTC_s007
                                 + 1.61485 w_k1_KO_s007 + 1.25065 w_k1_MAT_s007
                                 + 1.43177 w_k1_MSFT_s007
                                 + 1.40645 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.29122 w_k1_ADM_s007
                                 + 1.21359 w_k1_AXP_s007 + 1.5828 w_k1_DIS_s007
                                 + 1.29968 w_k1_FDX_s007
                                 + 0.914146 w_k1_IBM_s007
                                 + 1.21044 w_k1_INTC_s007
                                 + 1.43461 w_k1_KO_s007 + 1.10607 w_k1_MAT_s007
                                 + 1.32459 w_k1_MSFT_s007
                                 + 1.26027 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.12476 w_k1_ADM_s010
                                 + 1.35297 w_k1_AXP_s010
                                 + 1.68597 w_k1_DIS_s010
                                 + 1.23316 w_k1_FDX_s010
                                 + 1.48489 w_k1_IBM_s010
                                 + 1.16308 w_k1_INTC_s010
                                 + 1.37176 w_k1_KO_s010 + 1.65298 w_k1_MAT_s010
                                 + 1.5159 w_k1_MSFT_s010 + 1.17391 w_k1_PG_s010
                                 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.0696 w_k1_ADM_s010
                                 + 1.25785 w_k1_AXP_s010 + 1.0231 w_k1_DIS_s010
                                 + 1.0339 w_k1_FDX_s010 + 1.10331 w_k1_IBM_s010
                                 + 1.02448 w_k1_INTC_s010
                                 + 0.992259 w_k1_KO_s010
                                 + 1.20077 w_k1_MAT_s010
                                 + 1.14322 w_k1_MSFT_s010
                                 + 1.00167 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.796916 w_k1_ADM_s010
                                 + 0.76252 w_k1_AXP_s010
                                 + 0.676291 w_k1_DIS_s010
                                 + 0.807197 w_k1_FDX_s010
                                 + 1.15744 w_k1_IBM_s010
                                 + 0.794278 w_k1_INTC_s010
                                 + 0.85825 w_k1_KO_s010 + 1.23235 w_k1_MAT_s010
                                 + 0.785759 w_k1_MSFT_s010
                                 + 0.826459 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.910485 w_k1_ADM_s010
                                 + 0.847857 w_k1_AXP_s010
                                 + 0.7431 w_k1_DIS_s010
                                 + 0.828251 w_k1_FDX_s010
                                 + 0.83448 w_k1_IBM_s010
                                 + 0.734152 w_k1_INTC_s010
                                 + 0.763271 w_k1_KO_s010
                                 + 0.796619 w_k1_MAT_s010
                                 + 0.81185 w_k1_MSFT_s010
                                 + 0.79207 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.723096 w_k1_ADM_s010
                                 + 0.573119 w_k1_AXP_s010
                                 + 0.523822 w_k1_DIS_s010
                                 + 0.71771 w_k1_FDX_s010
                                 + 0.644929 w_k1_IBM_s010
                                 + 0.764451 w_k1_INTC_s010
                                 + 0.64944 w_k1_KO_s010
                                 + 0.705508 w_k1_MAT_s010
                                 + 0.769983 w_k1_MSFT_s010
                                 + 0.780113 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.03112 w_k1_ADM_s010
                                 + 1.0906 w_k1_AXP_s010 + 1.23029 w_k1_DIS_s010
                                 + 1.14984 w_k1_FDX_s010
                                 + 0.916238 w_k1_IBM_s010
                                 + 1.05288 w_k1_INTC_s010 + 1.0023 w_k1_KO_s010
                                 + 1.01505 w_k1_MAT_s010
                                 + 1.2311 w_k1_MSFT_s010 + 1.0446 w_k1_PG_s010
                                 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.9788 w_k1_ADM_s010
                                 + 0.814276 w_k1_AXP_s010
                                 + 0.837479 w_k1_DIS_s010
                                 + 0.853259 w_k1_FDX_s010
                                 + 0.680219 w_k1_IBM_s010
                                 + 0.862759 w_k1_INTC_s010
                                 + 0.797888 w_k1_KO_s010
                                 + 0.740611 w_k1_MAT_s010
                                 + 0.911988 w_k1_MSFT_s010
                                 + 0.882065 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.04067 w_k1_ADM_s010
                                 + 1.04353 w_k1_AXP_s010
                                 + 1.01602 w_k1_DIS_s010
                                 + 1.10226 w_k1_FDX_s010
                                 + 0.714799 w_k1_IBM_s010
                                 + 0.871968 w_k1_INTC_s010
                                 + 0.763962 w_k1_KO_s010
                                 + 0.941586 w_k1_MAT_s010
                                 + 1.008 w_k1_MSFT_s010 + 0.941729 w_k1_PG_s010
                                 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.43994 w_k1_ADM_s011
                                 + 1.36029 w_k1_AXP_s011
                                 + 1.47102 w_k1_DIS_s011 + 1.3268 w_k1_FDX_s011
                                 + 0.967762 w_k1_IBM_s011
                                 + 0.984205 w_k1_INTC_s011
                                 + 1.16713 w_k1_KO_s011
                                 + 0.939275 w_k1_MAT_s011
                                 + 1.24122 w_k1_MSFT_s011
                                 + 1.10255 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.14317 w_k1_ADM_s011
                                 + 1.65472 w_k1_AXP_s011
                                 + 1.09036 w_k1_DIS_s011
                                 + 1.15461 w_k1_FDX_s011
                                 + 1.40789 w_k1_IBM_s011
                                 + 1.08349 w_k1_INTC_s011 + 1.2685 w_k1_KO_s011
                                 + 1.72786 w_k1_MAT_s011
                                 + 1.02157 w_k1_MSFT_s011
                                 + 1.18601 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.88777 w_k1_ADM_s011
                                 + 0.975825 w_k1_AXP_s011
                                 + 0.783587 w_k1_DIS_s011
                                 + 1.06219 w_k1_FDX_s011
                                 + 0.674869 w_k1_IBM_s011
                                 + 1.05329 w_k1_INTC_s011
                                 + 0.873244 w_k1_KO_s011
                                 + 0.788438 w_k1_MAT_s011
                                 + 0.919216 w_k1_MSFT_s011
                                 + 1.01814 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.898459 w_k1_ADM_s011
                                 + 0.792221 w_k1_AXP_s011
                                 + 0.749576 w_k1_DIS_s011
                                 + 0.909172 w_k1_FDX_s011
                                 + 0.876694 w_k1_IBM_s011
                                 + 0.89789 w_k1_INTC_s011
                                 + 1.05619 w_k1_KO_s011
                                 + 0.829244 w_k1_MAT_s011
                                 + 0.992446 w_k1_MSFT_s011
                                 + 1.06123 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.10012 w_k1_ADM_s011
                                 + 0.901711 w_k1_AXP_s011
                                 + 1.25174 w_k1_DIS_s011
                                 + 1.01834 w_k1_FDX_s011
                                 + 0.94846 w_k1_IBM_s011
                                 + 0.922505 w_k1_INTC_s011
                                 + 1.33559 w_k1_KO_s011
                                 + 0.568177 w_k1_MAT_s011
                                 + 1.11061 w_k1_MSFT_s011
                                 + 1.13501 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.844281 w_k1_ADM_s011
                                 + 0.740388 w_k1_AXP_s011
                                 + 0.718164 w_k1_DIS_s011
                                 + 0.773713 w_k1_FDX_s011
                                 + 0.87071 w_k1_IBM_s011
                                 + 0.682758 w_k1_INTC_s011
                                 + 1.08665 w_k1_KO_s011
                                 + 0.814986 w_k1_MAT_s011
                                 + 0.757435 w_k1_MSFT_s011
                                 + 0.979078 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.06384 w_k1_ADM_s011
                                 + 1.15761 w_k1_AXP_s011
                                 + 0.958593 w_k1_DIS_s011
                                 + 0.99159 w_k1_FDX_s011
                                 + 0.946188 w_k1_IBM_s011
                                 + 0.879436 w_k1_INTC_s011
                                 + 1.23701 w_k1_KO_s011 + 1.00873 w_k1_MAT_s011
                                 + 0.96329 w_k1_MSFT_s011
                                 + 0.970597 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.828238 w_k1_ADM_s011
                                 + 1.13617 w_k1_AXP_s011
                                 + 0.964155 w_k1_DIS_s011
                                 + 0.929858 w_k1_FDX_s011
                                 + 1.3422 w_k1_IBM_s011
                                 + 0.916959 w_k1_INTC_s011
                                 + 1.35205 w_k1_KO_s011 + 1.05658 w_k1_MAT_s011
                                 + 0.949518 w_k1_MSFT_s011
                                 + 1.12808 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.36738 w_k1_ADM_s012
                                 + 1.10759 w_k1_AXP_s012
                                 + 1.87647 w_k1_DIS_s012
                                 + 1.38031 w_k1_FDX_s012
                                 + 0.998319 w_k1_IBM_s012
                                 + 1.52366 w_k1_INTC_s012
                                 + 1.50128 w_k1_KO_s012
                                 + 0.928581 w_k1_MAT_s012
                                 + 1.48753 w_k1_MSFT_s012
                                 + 1.29128 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.838265 w_k1_ADM_s012
                                 + 0.797504 w_k1_AXP_s012
                                 + 0.869217 w_k1_DIS_s012
                                 + 0.885846 w_k1_FDX_s012
                                 + 1.07604 w_k1_IBM_s012
                                 + 1.14414 w_k1_INTC_s012
                                 + 1.13303 w_k1_KO_s012
                                 + 0.831253 w_k1_MAT_s012
                                 + 0.983898 w_k1_MSFT_s012
                                 + 0.97275 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.11987 w_k1_ADM_s012
                                 + 1.26131 w_k1_AXP_s012
                                 + 1.29445 w_k1_DIS_s012
                                 + 1.10126 w_k1_FDX_s012
                                 + 1.48332 w_k1_IBM_s012
                                 + 0.918627 w_k1_INTC_s012
                                 + 1.27859 w_k1_KO_s012 + 1.48642 w_k1_MAT_s012
                                 + 0.958386 w_k1_MSFT_s012
                                 + 1.19288 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.714139 w_k1_ADM_s012
                                 + 0.655689 w_k1_AXP_s012
                                 + 0.51306 w_k1_DIS_s012
                                 + 0.646204 w_k1_FDX_s012
                                 + 0.935105 w_k1_IBM_s012
                                 + 0.779399 w_k1_INTC_s012
                                 + 0.740938 w_k1_KO_s012
                                 + 0.565644 w_k1_MAT_s012
                                 + 0.639415 w_k1_MSFT_s012
                                 + 0.739786 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.819667 w_k1_ADM_s012
                                 + 0.658205 w_k1_AXP_s012
                                 + 0.550569 w_k1_DIS_s012
                                 + 0.790537 w_k1_FDX_s012
                                 + 0.647173 w_k1_IBM_s012
                                 + 0.784989 w_k1_INTC_s012
                                 + 0.663153 w_k1_KO_s012
                                 + 0.417554 w_k1_MAT_s012
                                 + 0.7081 w_k1_MSFT_s012
                                 + 0.835453 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.05258 w_k1_ADM_s012
                                 + 0.907834 w_k1_AXP_s012
                                 + 0.82668 w_k1_DIS_s012
                                 + 0.94948 w_k1_FDX_s012
                                 + 0.825232 w_k1_IBM_s012
                                 + 1.12695 w_k1_INTC_s012
                                 + 0.989605 w_k1_KO_s012
                                 + 0.692009 w_k1_MAT_s012
                                 + 0.972324 w_k1_MSFT_s012
                                 + 1.04848 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.02232 w_k1_ADM_s012
                                 + 1.20208 w_k1_AXP_s012
                                 + 0.818834 w_k1_DIS_s012
                                 + 0.96921 w_k1_FDX_s012
                                 + 0.87516 w_k1_IBM_s012
                                 + 0.997509 w_k1_INTC_s012
                                 + 0.864009 w_k1_KO_s012
                                 + 1.08928 w_k1_MAT_s012
                                 + 0.818527 w_k1_MSFT_s012
                                 + 0.969915 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.32806 w_k1_ADM_s012
                                 + 1.14206 w_k1_AXP_s012
                                 + 1.22689 w_k1_DIS_s012 + 1.3059 w_k1_FDX_s012
                                 + 1.02163 w_k1_IBM_s012
                                 + 1.16574 w_k1_INTC_s012
                                 + 1.13272 w_k1_KO_s012 + 1.00706 w_k1_MAT_s012
                                 + 1.07825 w_k1_MSFT_s012
                                 + 1.08128 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.11795 w_k1_ADM_s013
                                 + 1.18668 w_k1_AXP_s013
                                 + 1.80627 w_k1_DIS_s013
                                 + 1.26354 w_k1_FDX_s013
                                 + 1.04308 w_k1_IBM_s013
                                 + 1.11817 w_k1_INTC_s013
                                 + 1.11107 w_k1_KO_s013 + 1.00011 w_k1_MAT_s013
                                 + 1.17095 w_k1_MSFT_s013
                                 + 1.11377 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.40392 w_k1_ADM_s013
                                 + 1.05187 w_k1_AXP_s013
                                 + 1.18579 w_k1_DIS_s013
                                 + 1.12354 w_k1_FDX_s013
                                 + 0.69888 w_k1_IBM_s013
                                 + 0.931508 w_k1_INTC_s013
                                 + 0.805316 w_k1_KO_s013
                                 + 0.688086 w_k1_MAT_s013
                                 + 0.964262 w_k1_MSFT_s013
                                 + 0.97684 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.09493 w_k1_ADM_s013
                                 + 1.02597 w_k1_AXP_s013
                                 + 0.799094 w_k1_DIS_s013
                                 + 0.87313 w_k1_FDX_s013 + 1.0081 w_k1_IBM_s013
                                 + 0.758208 w_k1_INTC_s013
                                 + 0.872163 w_k1_KO_s013
                                 + 0.982001 w_k1_MAT_s013
                                 + 0.876499 w_k1_MSFT_s013
                                 + 0.893503 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.33578 w_k1_ADM_s013
                                 + 1.34025 w_k1_AXP_s013
                                 + 1.59462 w_k1_DIS_s013
                                 + 1.24426 w_k1_FDX_s013 + 1.6797 w_k1_IBM_s013
                                 + 1.18894 w_k1_INTC_s013
                                 + 1.36843 w_k1_KO_s013 + 1.5254 w_k1_MAT_s013
                                 + 1.36932 w_k1_MSFT_s013
                                 + 1.23181 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.19312 w_k1_ADM_s013
                                 + 1.48159 w_k1_AXP_s013
                                 + 1.23861 w_k1_DIS_s013
                                 + 1.22054 w_k1_FDX_s013
                                 + 1.15805 w_k1_IBM_s013
                                 + 1.48219 w_k1_INTC_s013
                                 + 1.13702 w_k1_KO_s013 + 1.1728 w_k1_MAT_s013
                                 + 1.20745 w_k1_MSFT_s013
                                 + 1.03357 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.789363 w_k1_ADM_s013
                                 + 0.778148 w_k1_AXP_s013
                                 + 0.663177 w_k1_DIS_s013
                                 + 0.775667 w_k1_FDX_s013
                                 + 0.747953 w_k1_IBM_s013
                                 + 0.848366 w_k1_INTC_s013
                                 + 0.646122 w_k1_KO_s013
                                 + 0.739118 w_k1_MAT_s013
                                 + 0.683625 w_k1_MSFT_s013
                                 + 0.779418 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.01215 w_k1_ADM_s013
                                 + 1.11082 w_k1_AXP_s013
                                 + 0.944038 w_k1_DIS_s013
                                 + 0.880821 w_k1_FDX_s013
                                 + 1.49977 w_k1_IBM_s013
                                 + 0.923976 w_k1_INTC_s013
                                 + 0.933582 w_k1_KO_s013
                                 + 1.78695 w_k1_MAT_s013
                                 + 0.806367 w_k1_MSFT_s013
                                 + 1.02924 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.19981 w_k1_ADM_s013
                                 + 0.899273 w_k1_AXP_s013
                                 + 0.811638 w_k1_DIS_s013
                                 + 0.958871 w_k1_FDX_s013
                                 + 1.09903 w_k1_IBM_s013
                                 + 0.849976 w_k1_INTC_s013
                                 + 0.765848 w_k1_KO_s013
                                 + 1.04431 w_k1_MAT_s013
                                 + 0.803552 w_k1_MSFT_s013
                                 + 0.888834 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.767162 w_k1_ADM_s014
                                 + 0.988583 w_k1_AXP_s014
                                 + 0.991734 w_k1_DIS_s014
                                 + 0.906103 w_k1_FDX_s014
                                 + 1.60843 w_k1_IBM_s014
                                 + 0.870919 w_k1_INTC_s014
                                 + 1.12903 w_k1_KO_s014 + 1.59663 w_k1_MAT_s014
                                 + 0.845713 w_k1_MSFT_s014
                                 + 1.01508 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.05579 w_k1_ADM_s014
                                 + 1.01971 w_k1_AXP_s014
                                 + 0.97798 w_k1_DIS_s014 + 1.074 w_k1_FDX_s014
                                 + 0.959601 w_k1_IBM_s014
                                 + 0.986217 w_k1_INTC_s014
                                 + 0.995319 w_k1_KO_s014
                                 + 1.25553 w_k1_MAT_s014
                                 + 1.01711 w_k1_MSFT_s014
                                 + 1.05768 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.752854 w_k1_ADM_s014
                                 + 0.691092 w_k1_AXP_s014
                                 + 0.775487 w_k1_DIS_s014
                                 + 0.818473 w_k1_FDX_s014
                                 + 0.92179 w_k1_IBM_s014
                                 + 0.969193 w_k1_INTC_s014
                                 + 0.897314 w_k1_KO_s014
                                 + 1.17018 w_k1_MAT_s014
                                 + 0.828736 w_k1_MSFT_s014
                                 + 0.978366 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.991291 w_k1_ADM_s014
                                 + 1.23786 w_k1_AXP_s014 + 1.3301 w_k1_DIS_s014
                                 + 1.24087 w_k1_FDX_s014
                                 + 1.14571 w_k1_IBM_s014
                                 + 1.41215 w_k1_INTC_s014
                                 + 1.38794 w_k1_KO_s014 + 2.50042 w_k1_MAT_s014
                                 + 1.23388 w_k1_MSFT_s014
                                 + 1.29976 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.33674 w_k1_ADM_s014
                                 + 1.40331 w_k1_AXP_s014
                                 + 1.03342 w_k1_DIS_s014 + 1.1166 w_k1_FDX_s014
                                 + 1.09648 w_k1_IBM_s014
                                 + 0.999928 w_k1_INTC_s014
                                 + 1.11163 w_k1_KO_s014 + 1.59817 w_k1_MAT_s014
                                 + 0.965911 w_k1_MSFT_s014
                                 + 1.18275 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.828376 w_k1_ADM_s014
                                 + 0.655523 w_k1_AXP_s014
                                 + 0.933399 w_k1_DIS_s014
                                 + 1.04758 w_k1_FDX_s014
                                 + 0.667478 w_k1_IBM_s014
                                 + 0.920451 w_k1_INTC_s014
                                 + 0.779576 w_k1_KO_s014
                                 + 0.672042 w_k1_MAT_s014
                                 + 0.982523 w_k1_MSFT_s014
                                 + 0.955585 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.847217 w_k1_ADM_s014
                                 + 0.8941 w_k1_AXP_s014 + 1.18066 w_k1_DIS_s014
                                 + 1.01352 w_k1_FDX_s014
                                 + 1.37938 w_k1_IBM_s014
                                 + 1.00249 w_k1_INTC_s014
                                 + 1.09762 w_k1_KO_s014 + 1.29954 w_k1_MAT_s014
                                 + 0.934581 w_k1_MSFT_s014
                                 + 1.03706 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.24752 w_k1_ADM_s014
                                 + 1.09445 w_k1_AXP_s014
                                 + 1.35347 w_k1_DIS_s014
                                 + 1.28106 w_k1_FDX_s014
                                 + 0.765844 w_k1_IBM_s014
                                 + 1.02173 w_k1_INTC_s014
                                 + 1.12558 w_k1_KO_s014 + 1.30682 w_k1_MAT_s014
                                 + 1.15686 w_k1_MSFT_s014
                                 + 1.22163 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.03437 w_k1_ADM_s015
                                 + 0.916427 w_k1_AXP_s015
                                 + 0.935632 w_k1_DIS_s015
                                 + 1.05997 w_k1_FDX_s015
                                 + 0.70957 w_k1_IBM_s015
                                 + 1.04237 w_k1_INTC_s015
                                 + 0.832948 w_k1_KO_s015
                                 + 0.564488 w_k1_MAT_s015
                                 + 0.929733 w_k1_MSFT_s015
                                 + 0.981128 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.934029 w_k1_ADM_s015
                                 + 0.931867 w_k1_AXP_s015
                                 + 0.83901 w_k1_DIS_s015
                                 + 0.854039 w_k1_FDX_s015
                                 + 1.34763 w_k1_IBM_s015
                                 + 1.0125 w_k1_INTC_s015
                                 + 0.927431 w_k1_KO_s015
                                 + 0.810142 w_k1_MAT_s015
                                 + 0.816148 w_k1_MSFT_s015
                                 + 1.02938 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.976472 w_k1_ADM_s015
                                 + 0.676206 w_k1_AXP_s015
                                 + 0.613087 w_k1_DIS_s015
                                 + 0.683457 w_k1_FDX_s015
                                 + 0.809291 w_k1_IBM_s015
                                 + 0.759262 w_k1_INTC_s015
                                 + 0.756092 w_k1_KO_s015
                                 + 0.716921 w_k1_MAT_s015
                                 + 0.730967 w_k1_MSFT_s015
                                 + 0.856489 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.21586 w_k1_ADM_s015
                                 + 1.25321 w_k1_AXP_s015 + 1.346 w_k1_DIS_s015
                                 + 1.02844 w_k1_FDX_s015
                                 + 1.63834 w_k1_IBM_s015
                                 + 1.04855 w_k1_INTC_s015
                                 + 1.42631 w_k1_KO_s015 + 1.81796 w_k1_MAT_s015
                                 + 1.00739 w_k1_MSFT_s015
                                 + 1.15668 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.10872 w_k1_ADM_s015
                                 + 0.698555 w_k1_AXP_s015
                                 + 0.785222 w_k1_DIS_s015
                                 + 0.855872 w_k1_FDX_s015
                                 + 0.769011 w_k1_IBM_s015
                                 + 0.967818 w_k1_INTC_s015
                                 + 0.968181 w_k1_KO_s015
                                 + 0.642232 w_k1_MAT_s015
                                 + 0.935766 w_k1_MSFT_s015
                                 + 1.00623 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.49212 w_k1_ADM_s015
                                 + 1.29489 w_k1_AXP_s015
                                 + 1.59828 w_k1_DIS_s015 + 1.3085 w_k1_FDX_s015
                                 + 1.11036 w_k1_IBM_s015
                                 + 1.32373 w_k1_INTC_s015
                                 + 1.30705 w_k1_KO_s015
                                 + 0.785272 w_k1_MAT_s015
                                 + 1.21779 w_k1_MSFT_s015
                                 + 1.28116 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.09636 w_k1_ADM_s015
                                 + 0.701229 w_k1_AXP_s015
                                 + 1.22377 w_k1_DIS_s015
                                 + 0.928112 w_k1_FDX_s015
                                 + 1.01446 w_k1_IBM_s015
                                 + 1.12723 w_k1_INTC_s015
                                 + 1.19621 w_k1_KO_s015 + 0.61772 w_k1_MAT_s015
                                 + 1.00153 w_k1_MSFT_s015 + 1.0496 w_k1_PG_s015
                                 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.01699 w_k1_ADM_s015
                                 + 1.09465 w_k1_AXP_s015
                                 + 1.03189 w_k1_DIS_s015
                                 + 1.02206 w_k1_FDX_s015
                                 + 0.943974 w_k1_IBM_s015
                                 + 1.09259 w_k1_INTC_s015
                                 + 1.03769 w_k1_KO_s015
                                 + 0.922998 w_k1_MAT_s015
                                 + 1.02757 w_k1_MSFT_s015
                                 + 0.984929 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.00084 w_k1_ADM_s016
                                 + 1.29265 w_k1_AXP_s016
                                 + 1.23962 w_k1_DIS_s016
                                 + 1.15347 w_k1_FDX_s016
                                 + 1.39835 w_k1_IBM_s016
                                 + 1.00903 w_k1_INTC_s016
                                 + 1.43229 w_k1_KO_s016 + 1.69448 w_k1_MAT_s016
                                 + 1.15896 w_k1_MSFT_s016
                                 + 1.05423 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.870013 w_k1_ADM_s016
                                 + 1.05975 w_k1_AXP_s016
                                 + 1.09174 w_k1_DIS_s016
                                 + 1.02811 w_k1_FDX_s016
                                 + 1.34447 w_k1_IBM_s016
                                 + 1.05689 w_k1_INTC_s016
                                 + 1.27231 w_k1_KO_s016 + 1.19636 w_k1_MAT_s016
                                 + 1.04742 w_k1_MSFT_s016
                                 + 1.06776 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.25808 w_k1_ADM_s016
                                 + 1.20373 w_k1_AXP_s016
                                 + 1.74021 w_k1_DIS_s016
                                 + 1.47634 w_k1_FDX_s016
                                 + 1.08293 w_k1_IBM_s016
                                 + 1.50441 w_k1_INTC_s016
                                 + 1.45666 w_k1_KO_s016
                                 + 0.874085 w_k1_MAT_s016
                                 + 1.44007 w_k1_MSFT_s016
                                 + 1.10147 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.852182 w_k1_ADM_s016
                                 + 0.880113 w_k1_AXP_s016
                                 + 1.14512 w_k1_DIS_s016
                                 + 1.10941 w_k1_FDX_s016
                                 + 0.741559 w_k1_IBM_s016
                                 + 0.987667 w_k1_INTC_s016
                                 + 0.896972 w_k1_KO_s016
                                 + 0.550642 w_k1_MAT_s016
                                 + 1.05407 w_k1_MSFT_s016
                                 + 0.884593 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.815855 w_k1_ADM_s016
                                 + 0.824184 w_k1_AXP_s016
                                 + 0.648511 w_k1_DIS_s016
                                 + 0.852022 w_k1_FDX_s016
                                 + 1.24986 w_k1_IBM_s016
                                 + 0.655358 w_k1_INTC_s016
                                 + 0.969629 w_k1_KO_s016 + 1.0524 w_k1_MAT_s016
                                 + 0.877335 w_k1_MSFT_s016
                                 + 0.864736 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.13068 w_k1_ADM_s016
                                 + 1.02612 w_k1_AXP_s016 + 1.2451 w_k1_DIS_s016
                                 + 1.15805 w_k1_FDX_s016
                                 + 0.967389 w_k1_IBM_s016
                                 + 1.21032 w_k1_INTC_s016
                                 + 1.27082 w_k1_KO_s016
                                 + 0.765235 w_k1_MAT_s016
                                 + 1.15761 w_k1_MSFT_s016
                                 + 0.938186 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.693076 w_k1_ADM_s016
                                 + 0.852506 w_k1_AXP_s016
                                 + 0.752141 w_k1_DIS_s016
                                 + 0.841482 w_k1_FDX_s016
                                 + 0.953068 w_k1_IBM_s016
                                 + 0.915964 w_k1_INTC_s016
                                 + 0.866709 w_k1_KO_s016
                                 + 0.947727 w_k1_MAT_s016
                                 + 0.903547 w_k1_MSFT_s016
                                 + 0.796035 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.918364 w_k1_ADM_s016
                                 + 0.737045 w_k1_AXP_s016
                                 + 0.760359 w_k1_DIS_s016
                                 + 0.907964 w_k1_FDX_s016
                                 + 0.696652 w_k1_IBM_s016
                                 + 0.873197 w_k1_INTC_s016
                                 + 0.799916 w_k1_KO_s016
                                 + 0.48327 w_k1_MAT_s016
                                 + 0.911197 w_k1_MSFT_s016
                                 + 0.737986 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.727059 w_k1_ADM_s017
                                 + 0.670962 w_k1_AXP_s017
                                 + 0.56819 w_k1_DIS_s017
                                 + 0.780274 w_k1_FDX_s017
                                 + 0.707505 w_k1_IBM_s017
                                 + 0.774245 w_k1_INTC_s017
                                 + 0.628854 w_k1_KO_s017
                                 + 0.62819 w_k1_MAT_s017
                                 + 0.657164 w_k1_MSFT_s017
                                 + 0.792678 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.40832 w_k1_ADM_s017
                                 + 1.44452 w_k1_AXP_s017
                                 + 1.52494 w_k1_DIS_s017
                                 + 1.29793 w_k1_FDX_s017
                                 + 1.26708 w_k1_IBM_s017
                                 + 1.18502 w_k1_INTC_s017
                                 + 1.34095 w_k1_KO_s017 + 1.30141 w_k1_MAT_s017
                                 + 1.03734 w_k1_MSFT_s017
                                 + 1.22539 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.763234 w_k1_ADM_s017
                                 + 0.998693 w_k1_AXP_s017
                                 + 1.10038 w_k1_DIS_s017
                                 + 0.897917 w_k1_FDX_s017
                                 + 1.53976 w_k1_IBM_s017
                                 + 0.85057 w_k1_INTC_s017 + 1.1027 w_k1_KO_s017
                                 + 1.70083 w_k1_MAT_s017
                                 + 0.751345 w_k1_MSFT_s017
                                 + 0.915914 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.08571 w_k1_ADM_s017
                                 + 1.00592 w_k1_AXP_s017
                                 + 1.12203 w_k1_DIS_s017
                                 + 1.20187 w_k1_FDX_s017
                                 + 0.796437 w_k1_IBM_s017
                                 + 1.1403 w_k1_INTC_s017
                                 + 0.923071 w_k1_KO_s017
                                 + 0.877198 w_k1_MAT_s017
                                 + 1.0037 w_k1_MSFT_s017 + 1.10336 w_k1_PG_s017
                                 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.04362 w_k1_ADM_s017
                                 + 1.05344 w_k1_AXP_s017
                                 + 1.07026 w_k1_DIS_s017
                                 + 0.994367 w_k1_FDX_s017
                                 + 1.10487 w_k1_IBM_s017
                                 + 0.91214 w_k1_INTC_s017
                                 + 1.06311 w_k1_KO_s017 + 1.10664 w_k1_MAT_s017
                                 + 0.925871 w_k1_MSFT_s017
                                 + 0.948532 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.626719 w_k1_ADM_s017
                                 + 0.817946 w_k1_AXP_s017
                                 + 0.662611 w_k1_DIS_s017
                                 + 0.751064 w_k1_FDX_s017
                                 + 1.00413 w_k1_IBM_s017
                                 + 0.790572 w_k1_INTC_s017
                                 + 0.861032 w_k1_KO_s017
                                 + 1.08649 w_k1_MAT_s017
                                 + 0.696127 w_k1_MSFT_s017
                                 + 0.919251 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.987208 w_k1_ADM_s017
                                 + 1.17404 w_k1_AXP_s017
                                 + 0.70834 w_k1_DIS_s017
                                 + 1.01633 w_k1_FDX_s017
                                 + 0.65884 w_k1_IBM_s017
                                 + 0.90868 w_k1_INTC_s017
                                 + 0.659946 w_k1_KO_s017
                                 + 0.902688 w_k1_MAT_s017
                                 + 0.768771 w_k1_MSFT_s017
                                 + 0.88226 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.05626 w_k1_ADM_s017
                                 + 0.962688 w_k1_AXP_s017
                                 + 0.91767 w_k1_DIS_s017
                                 + 0.964934 w_k1_FDX_s017
                                 + 0.94502 w_k1_IBM_s017
                                 + 0.993505 w_k1_INTC_s017
                                 + 0.939888 w_k1_KO_s017
                                 + 1.06978 w_k1_MAT_s017
                                 + 0.912506 w_k1_MSFT_s017
                                 + 1.08367 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.04558 w_k1_ADM_s100
                                 + 1.0118 w_k1_AXP_s100 + 1.0639 w_k1_DIS_s100
                                 + 1.04206 w_k1_FDX_s100
                                 + 1.14031 w_k1_IBM_s100
                                 + 0.994907 w_k1_INTC_s100
                                 + 1.14644 w_k1_KO_s100 + 0.72709 w_k1_MAT_s100
                                 + 0.984335 w_k1_MSFT_s100
                                 + 0.972071 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.850931 w_k1_ADM_s100
                                 + 0.629614 w_k1_AXP_s100
                                 + 0.562569 w_k1_DIS_s100
                                 + 0.788791 w_k1_FDX_s100
                                 + 0.694969 w_k1_IBM_s100
                                 + 0.950112 w_k1_INTC_s100
                                 + 0.67021 w_k1_KO_s100
                                 + 0.284767 w_k1_MAT_s100
                                 + 0.776057 w_k1_MSFT_s100
                                 + 0.793623 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.41491 w_k1_ADM_s100
                                 + 0.992656 w_k1_AXP_s100
                                 + 1.38176 w_k1_DIS_s100
                                 + 1.24771 w_k1_FDX_s100
                                 + 0.929779 w_k1_IBM_s100
                                 + 1.06833 w_k1_INTC_s100 + 1.1705 w_k1_KO_s100
                                 + 0.902027 w_k1_MAT_s100
                                 + 1.32386 w_k1_MSFT_s100
                                 + 1.06996 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.941456 w_k1_ADM_s100
                                 + 0.867028 w_k1_AXP_s100
                                 + 0.575879 w_k1_DIS_s100
                                 + 0.666522 w_k1_FDX_s100
                                 + 0.960498 w_k1_IBM_s100
                                 + 0.80274 w_k1_INTC_s100
                                 + 0.785202 w_k1_KO_s100
                                 + 0.761971 w_k1_MAT_s100
                                 + 0.800628 w_k1_MSFT_s100
                                 + 0.769142 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.868739 w_k1_ADM_s100
                                 + 0.997441 w_k1_AXP_s100
                                 + 0.715788 w_k1_DIS_s100
                                 + 0.86628 w_k1_FDX_s100
                                 + 0.856638 w_k1_IBM_s100
                                 + 0.887087 w_k1_INTC_s100
                                 + 0.812837 w_k1_KO_s100
                                 + 0.535586 w_k1_MAT_s100
                                 + 0.828172 w_k1_MSFT_s100
                                 + 0.867554 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.933054 w_k1_ADM_s100
                                 + 1.36131 w_k1_AXP_s100
                                 + 0.879308 w_k1_DIS_s100
                                 + 1.0968 w_k1_FDX_s100
                                 + 0.981717 w_k1_IBM_s100
                                 + 1.03132 w_k1_INTC_s100
                                 + 0.907044 w_k1_KO_s100
                                 + 1.06006 w_k1_MAT_s100
                                 + 0.985441 w_k1_MSFT_s100
                                 + 0.897815 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.764719 w_k1_ADM_s100
                                 + 0.7871 w_k1_AXP_s100 + 0.91775 w_k1_DIS_s100
                                 + 0.956721 w_k1_FDX_s100
                                 + 0.779546 w_k1_IBM_s100
                                 + 0.935363 w_k1_INTC_s100
                                 + 0.861481 w_k1_KO_s100
                                 + 0.593786 w_k1_MAT_s100
                                 + 0.936939 w_k1_MSFT_s100
                                 + 0.854875 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.32937 w_k1_ADM_s100
                                 + 1.68484 w_k1_AXP_s100
                                 + 1.39205 w_k1_DIS_s100
                                 + 1.32842 w_k1_FDX_s100 + 1.4626 w_k1_IBM_s100
                                 + 1.17337 w_k1_INTC_s100
                                 + 1.19762 w_k1_KO_s100 + 1.61722 w_k1_MAT_s100
                                 + 1.0637 w_k1_MSFT_s100 + 1.15268 w_k1_PG_s100
                                 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.849931 w_k1_ADM_s101
                                 + 1.31272 w_k1_AXP_s101
                                 + 1.08588 w_k1_DIS_s101
                                 + 1.26273 w_k1_FDX_s101
                                 + 1.61594 w_k1_IBM_s101
                                 + 1.31869 w_k1_INTC_s101
                                 + 1.17162 w_k1_KO_s101 + 1.87427 w_k1_MAT_s101
                                 + 1.46315 w_k1_MSFT_s101
                                 + 1.18605 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.12486 w_k1_ADM_s101
                                 + 1.0658 w_k1_AXP_s101 + 1.14761 w_k1_DIS_s101
                                 + 1.00593 w_k1_FDX_s101
                                 + 0.958046 w_k1_IBM_s101
                                 + 0.982291 w_k1_INTC_s101
                                 + 1.00714 w_k1_KO_s101 + 1.03105 w_k1_MAT_s101
                                 + 1.21764 w_k1_MSFT_s101
                                 + 1.12037 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.926854 w_k1_ADM_s101
                                 + 0.648921 w_k1_AXP_s101
                                 + 0.708559 w_k1_DIS_s101
                                 + 0.840314 w_k1_FDX_s101
                                 + 0.993049 w_k1_IBM_s101
                                 + 0.748009 w_k1_INTC_s101
                                 + 0.812407 w_k1_KO_s101
                                 + 0.769158 w_k1_MAT_s101
                                 + 1.00372 w_k1_MSFT_s101
                                 + 0.992754 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.986377 w_k1_ADM_s101
                                 + 0.990932 w_k1_AXP_s101
                                 + 1.16201 w_k1_DIS_s101
                                 + 1.05981 w_k1_FDX_s101
                                 + 1.24176 w_k1_IBM_s101 + 1.204 w_k1_INTC_s101
                                 + 1.08801 w_k1_KO_s101 + 1.1686 w_k1_MAT_s101
                                 + 1.29658 w_k1_MSFT_s101
                                 + 1.06261 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.29537 w_k1_ADM_s101
                                 + 1.62323 w_k1_AXP_s101
                                 + 1.02795 w_k1_DIS_s101
                                 + 1.36302 w_k1_FDX_s101
                                 + 0.922661 w_k1_IBM_s101
                                 + 1.17408 w_k1_INTC_s101
                                 + 1.02696 w_k1_KO_s101 + 1.33478 w_k1_MAT_s101
                                 + 1.20579 w_k1_MSFT_s101
                                 + 1.17471 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.12656 w_k1_ADM_s101
                                 + 0.859993 w_k1_AXP_s101
                                 + 0.741807 w_k1_DIS_s101
                                 + 0.886214 w_k1_FDX_s101
                                 + 0.817215 w_k1_IBM_s101
                                 + 0.913788 w_k1_INTC_s101
                                 + 0.845709 w_k1_KO_s101
                                 + 0.824744 w_k1_MAT_s101
                                 + 1.09259 w_k1_MSFT_s101
                                 + 0.952908 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.11933 w_k1_ADM_s101
                                 + 0.941127 w_k1_AXP_s101
                                 + 0.892682 w_k1_DIS_s101
                                 + 0.986499 w_k1_FDX_s101
                                 + 1.50972 w_k1_IBM_s101
                                 + 0.847685 w_k1_INTC_s101
                                 + 1.04768 w_k1_KO_s101 + 1.38769 w_k1_MAT_s101
                                 + 1.13256 w_k1_MSFT_s101
                                 + 0.96682 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.932996 w_k1_ADM_s101
                                 + 0.856903 w_k1_AXP_s101
                                 + 0.869106 w_k1_DIS_s101
                                 + 1.07745 w_k1_FDX_s101
                                 + 0.695645 w_k1_IBM_s101
                                 + 1.13884 w_k1_INTC_s101
                                 + 0.751431 w_k1_KO_s101
                                 + 0.619583 w_k1_MAT_s101
                                 + 1.09831 w_k1_MSFT_s101
                                 + 0.824333 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.901906 w_k1_ADM_s102
                                 + 1.36895 w_k1_AXP_s102
                                 + 1.30892 w_k1_DIS_s102
                                 + 1.31172 w_k1_FDX_s102
                                 + 1.05014 w_k1_IBM_s102
                                 + 1.09171 w_k1_INTC_s102
                                 + 1.00018 w_k1_KO_s102 + 1.70677 w_k1_MAT_s102
                                 + 1.021 w_k1_MSFT_s102 + 1.2545 w_k1_PG_s102
                                 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.84182 w_k1_ADM_s102
                                 + 0.798302 w_k1_AXP_s102
                                 + 0.77573 w_k1_DIS_s102
                                 + 0.802702 w_k1_FDX_s102
                                 + 1.06768 w_k1_IBM_s102
                                 + 0.759677 w_k1_INTC_s102
                                 + 0.869211 w_k1_KO_s102
                                 + 1.37877 w_k1_MAT_s102
                                 + 0.834887 w_k1_MSFT_s102
                                 + 0.970188 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.34034 w_k1_ADM_s102
                                 + 1.23299 w_k1_AXP_s102
                                 + 1.44418 w_k1_DIS_s102
                                 + 1.29705 w_k1_FDX_s102
                                 + 1.16807 w_k1_IBM_s102
                                 + 1.06763 w_k1_INTC_s102
                                 + 1.12834 w_k1_KO_s102 + 1.71852 w_k1_MAT_s102
                                 + 1.16701 w_k1_MSFT_s102
                                 + 1.17831 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.08268 w_k1_ADM_s102
                                 + 1.14877 w_k1_AXP_s102
                                 + 1.16797 w_k1_DIS_s102
                                 + 1.31187 w_k1_FDX_s102
                                 + 0.788487 w_k1_IBM_s102
                                 + 1.30437 w_k1_INTC_s102
                                 + 0.902541 w_k1_KO_s102
                                 + 0.735938 w_k1_MAT_s102
                                 + 1.18868 w_k1_MSFT_s102
                                 + 1.12406 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.818639 w_k1_ADM_s102
                                 + 1.21105 w_k1_AXP_s102
                                 + 1.08264 w_k1_DIS_s102
                                 + 1.05088 w_k1_FDX_s102
                                 + 1.63198 w_k1_IBM_s102
                                 + 1.03743 w_k1_INTC_s102
                                 + 1.05713 w_k1_KO_s102 + 2.22458 w_k1_MAT_s102
                                 + 0.834391 w_k1_MSFT_s102
                                 + 1.11503 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.861942 w_k1_ADM_s102
                                 + 0.648716 w_k1_AXP_s102
                                 + 0.785314 w_k1_DIS_s102
                                 + 0.906633 w_k1_FDX_s102
                                 + 0.711226 w_k1_IBM_s102
                                 + 0.78608 w_k1_INTC_s102
                                 + 0.723992 w_k1_KO_s102
                                 + 0.731163 w_k1_MAT_s102
                                 + 0.818436 w_k1_MSFT_s102
                                 + 0.991967 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.770424 w_k1_ADM_s102
                                 + 0.778753 w_k1_AXP_s102
                                 + 0.542954 w_k1_DIS_s102
                                 + 0.757713 w_k1_FDX_s102
                                 + 0.856606 w_k1_IBM_s102
                                 + 0.917467 w_k1_INTC_s102
                                 + 0.685569 w_k1_KO_s102
                                 + 1.13467 w_k1_MAT_s102
                                 + 0.69795 w_k1_MSFT_s102
                                 + 0.780313 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.8071 w_k1_ADM_s102
                                 + 1.22694 w_k1_AXP_s102
                                 + 0.92155 w_k1_DIS_s102
                                 + 1.04745 w_k1_FDX_s102
                                 + 1.05091 w_k1_IBM_s102
                                 + 0.955259 w_k1_INTC_s102
                                 + 0.91248 w_k1_KO_s102 + 1.07888 w_k1_MAT_s102
                                 + 0.836016 w_k1_MSFT_s102
                                 + 0.903083 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.773444 w_k1_ADM_s103
                                 + 0.922458 w_k1_AXP_s103
                                 + 0.826067 w_k1_DIS_s103
                                 + 0.918426 w_k1_FDX_s103
                                 + 1.15328 w_k1_IBM_s103
                                 + 0.980706 w_k1_INTC_s103
                                 + 0.903008 w_k1_KO_s103
                                 + 1.14239 w_k1_MAT_s103
                                 + 0.835671 w_k1_MSFT_s103
                                 + 0.947444 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.738689 w_k1_ADM_s103
                                 + 0.905911 w_k1_AXP_s103
                                 + 1.0388 w_k1_DIS_s103
                                 + 0.989049 w_k1_FDX_s103
                                 + 1.06519 w_k1_IBM_s103
                                 + 1.22797 w_k1_INTC_s103
                                 + 1.00552 w_k1_KO_s103 + 1.07818 w_k1_MAT_s103
                                 + 0.885267 w_k1_MSFT_s103
                                 + 0.972519 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.640067 w_k1_ADM_s103
                                 + 0.660797 w_k1_AXP_s103
                                 + 0.55193 w_k1_DIS_s103
                                 + 0.746141 w_k1_FDX_s103
                                 + 0.790978 w_k1_IBM_s103
                                 + 0.868383 w_k1_INTC_s103
                                 + 0.641723 w_k1_KO_s103
                                 + 0.715152 w_k1_MAT_s103
                                 + 0.623667 w_k1_MSFT_s103
                                 + 0.786691 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.37594 w_k1_ADM_s103
                                 + 1.44105 w_k1_AXP_s103
                                 + 1.70596 w_k1_DIS_s103
                                 + 1.45146 w_k1_FDX_s103
                                 + 1.12609 w_k1_IBM_s103
                                 + 1.7018 w_k1_INTC_s103 + 1.18853 w_k1_KO_s103
                                 + 1.0581 w_k1_MAT_s103 + 1.4058 w_k1_MSFT_s103
                                 + 1.14902 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.2125 w_k1_ADM_s103
                                 + 1.26722 w_k1_AXP_s103
                                 + 1.09804 w_k1_DIS_s103
                                 + 1.25464 w_k1_FDX_s103
                                 + 0.943072 w_k1_IBM_s103
                                 + 1.27363 w_k1_INTC_s103
                                 + 1.09557 w_k1_KO_s103 + 1.02666 w_k1_MAT_s103
                                 + 0.996228 w_k1_MSFT_s103
                                 + 1.04832 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.90408 w_k1_ADM_s103
                                 + 0.719147 w_k1_AXP_s103
                                 + 0.844762 w_k1_DIS_s103
                                 + 1.0297 w_k1_FDX_s103
                                 + 0.721548 w_k1_IBM_s103
                                 + 1.00638 w_k1_INTC_s103
                                 + 0.796564 w_k1_KO_s103
                                 + 0.633632 w_k1_MAT_s103
                                 + 0.844717 w_k1_MSFT_s103
                                 + 0.916616 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.17432 w_k1_ADM_s103
                                 + 1.06795 w_k1_AXP_s103
                                 + 1.10669 w_k1_DIS_s103
                                 + 1.24559 w_k1_FDX_s103
                                 + 1.13062 w_k1_IBM_s103
                                 + 1.46596 w_k1_INTC_s103
                                 + 1.08132 w_k1_KO_s103 + 1.03708 w_k1_MAT_s103
                                 + 1.15182 w_k1_MSFT_s103
                                 + 1.10316 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.910905 w_k1_ADM_s103
                                 + 1.21568 w_k1_AXP_s103
                                 + 1.28271 w_k1_DIS_s103 + 1.0641 w_k1_FDX_s103
                                 + 1.6077 w_k1_IBM_s103
                                 + 1.25992 w_k1_INTC_s103
                                 + 1.44169 w_k1_KO_s103 + 1.85413 w_k1_MAT_s103
                                 + 1.1315 w_k1_MSFT_s103 + 1.06659 w_k1_PG_s103
                                 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.95449 w_k1_ADM_s104
                                 + 0.660529 w_k1_AXP_s104
                                 + 0.657957 w_k1_DIS_s104
                                 + 0.853345 w_k1_FDX_s104
                                 + 0.474398 w_k1_IBM_s104
                                 + 0.858575 w_k1_INTC_s104
                                 + 0.657842 w_k1_KO_s104
                                 + 0.449528 w_k1_MAT_s104
                                 + 0.815479 w_k1_MSFT_s104
                                 + 0.792676 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.660844 w_k1_ADM_s104
                                 + 0.471546 w_k1_AXP_s104
                                 + 0.693864 w_k1_DIS_s104
                                 + 0.653651 w_k1_FDX_s104
                                 + 0.776352 w_k1_IBM_s104
                                 + 0.694119 w_k1_INTC_s104
                                 + 0.750621 w_k1_KO_s104
                                 + 0.641624 w_k1_MAT_s104
                                 + 0.703513 w_k1_MSFT_s104
                                 + 0.769432 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.1108 w_k1_ADM_s104
                                 + 1.18487 w_k1_AXP_s104
                                 + 0.902472 w_k1_DIS_s104
                                 + 1.09826 w_k1_FDX_s104
                                 + 0.682492 w_k1_IBM_s104
                                 + 1.16029 w_k1_INTC_s104
                                 + 0.792701 w_k1_KO_s104
                                 + 0.73974 w_k1_MAT_s104
                                 + 0.970474 w_k1_MSFT_s104
                                 + 1.02891 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.962453 w_k1_ADM_s104
                                 + 0.924219 w_k1_AXP_s104
                                 + 1.03318 w_k1_DIS_s104
                                 + 0.903232 w_k1_FDX_s104
                                 + 1.07146 w_k1_IBM_s104
                                 + 0.914886 w_k1_INTC_s104
                                 + 0.980963 w_k1_KO_s104
                                 + 1.03653 w_k1_MAT_s104
                                 + 0.93456 w_k1_MSFT_s104
                                 + 1.02803 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.44092 w_k1_ADM_s104
                                 + 1.84461 w_k1_AXP_s104
                                 + 1.79386 w_k1_DIS_s104
                                 + 1.38131 w_k1_FDX_s104
                                 + 1.28733 w_k1_IBM_s104
                                 + 1.53836 w_k1_INTC_s104
                                 + 1.56454 w_k1_KO_s104 + 1.50562 w_k1_MAT_s104
                                 + 1.26511 w_k1_MSFT_s104
                                 + 1.32558 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.936194 w_k1_ADM_s104
                                 + 1.0551 w_k1_AXP_s104
                                 + 0.761788 w_k1_DIS_s104
                                 + 0.905953 w_k1_FDX_s104
                                 + 0.873512 w_k1_IBM_s104
                                 + 0.83567 w_k1_INTC_s104
                                 + 0.735985 w_k1_KO_s104
                                 + 1.09417 w_k1_MAT_s104
                                 + 0.719245 w_k1_MSFT_s104
                                 + 0.807771 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.801066 w_k1_ADM_s104
                                 + 0.877746 w_k1_AXP_s104
                                 + 0.823324 w_k1_DIS_s104
                                 + 0.870764 w_k1_FDX_s104
                                 + 0.802617 w_k1_IBM_s104
                                 + 0.912391 w_k1_INTC_s104
                                 + 0.803951 w_k1_KO_s104
                                 + 0.825573 w_k1_MAT_s104
                                 + 0.86968 w_k1_MSFT_s104
                                 + 0.909872 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.01926 w_k1_ADM_s104
                                 + 1.12342 w_k1_AXP_s104
                                 + 1.34396 w_k1_DIS_s104 + 1.212 w_k1_FDX_s104
                                 + 1.08505 w_k1_IBM_s104
                                 + 1.19825 w_k1_INTC_s104
                                 + 1.22992 w_k1_KO_s104 + 1.30107 w_k1_MAT_s104
                                 + 1.2939 w_k1_MSFT_s104 + 1.18119 w_k1_PG_s104
                                 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.857055 w_k1_ADM_s105
                                 + 0.913631 w_k1_AXP_s105
                                 + 0.943915 w_k1_DIS_s105
                                 + 0.810993 w_k1_FDX_s105
                                 + 1.43221 w_k1_IBM_s105
                                 + 0.694137 w_k1_INTC_s105
                                 + 1.09534 w_k1_KO_s105 + 1.49362 w_k1_MAT_s105
                                 + 0.775396 w_k1_MSFT_s105
                                 + 1.07445 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.767302 w_k1_ADM_s105
                                 + 0.761499 w_k1_AXP_s105
                                 + 0.757124 w_k1_DIS_s105
                                 + 0.81509 w_k1_FDX_s105
                                 + 0.946428 w_k1_IBM_s105
                                 + 0.784344 w_k1_INTC_s105
                                 + 0.826693 w_k1_KO_s105
                                 + 1.04542 w_k1_MAT_s105
                                 + 0.909437 w_k1_MSFT_s105
                                 + 0.951772 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.05264 w_k1_ADM_s105
                                 + 0.9622 w_k1_AXP_s105 + 1.00144 w_k1_DIS_s105
                                 + 1.1056 w_k1_FDX_s105
                                 + 0.857057 w_k1_IBM_s105
                                 + 1.09211 w_k1_INTC_s105
                                 + 0.938648 w_k1_KO_s105
                                 + 0.759642 w_k1_MAT_s105
                                 + 1.05693 w_k1_MSFT_s105
                                 + 1.08367 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.09926 w_k1_ADM_s105
                                 + 1.08619 w_k1_AXP_s105
                                 + 1.04567 w_k1_DIS_s105
                                 + 0.979218 w_k1_FDX_s105
                                 + 1.15029 w_k1_IBM_s105
                                 + 0.965436 w_k1_INTC_s105
                                 + 1.00906 w_k1_KO_s105
                                 + 0.926507 w_k1_MAT_s105
                                 + 0.982261 w_k1_MSFT_s105
                                 + 1.06451 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.23238 w_k1_ADM_s105
                                 + 1.30052 w_k1_AXP_s105
                                 + 1.51842 w_k1_DIS_s105
                                 + 1.25697 w_k1_FDX_s105 + 1.0939 w_k1_IBM_s105
                                 + 1.11731 w_k1_INTC_s105
                                 + 1.15449 w_k1_KO_s105
                                 + 0.860555 w_k1_MAT_s105
                                 + 1.17193 w_k1_MSFT_s105
                                 + 1.14961 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.722928 w_k1_ADM_s105
                                 + 0.774733 w_k1_AXP_s105
                                 + 0.687866 w_k1_DIS_s105
                                 + 0.843248 w_k1_FDX_s105
                                 + 0.584051 w_k1_IBM_s105
                                 + 0.882924 w_k1_INTC_s105
                                 + 0.736408 w_k1_KO_s105
                                 + 0.571649 w_k1_MAT_s105
                                 + 0.876972 w_k1_MSFT_s105
                                 + 0.883333 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.05987 w_k1_ADM_s105
                                 + 1.01257 w_k1_AXP_s105
                                 + 1.11952 w_k1_DIS_s105
                                 + 1.23171 w_k1_FDX_s105
                                 + 0.628615 w_k1_IBM_s105
                                 + 0.97301 w_k1_INTC_s105
                                 + 0.876574 w_k1_KO_s105
                                 + 0.819353 w_k1_MAT_s105
                                 + 1.09011 w_k1_MSFT_s105
                                 + 1.11393 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.06377 w_k1_ADM_s105
                                 + 1.36725 w_k1_AXP_s105
                                 + 1.39354 w_k1_DIS_s105
                                 + 1.15085 w_k1_FDX_s105
                                 + 1.56117 w_k1_IBM_s105
                                 + 0.923164 w_k1_INTC_s105
                                 + 1.32292 w_k1_KO_s105 + 1.57593 w_k1_MAT_s105
                                 + 1.16557 w_k1_MSFT_s105
                                 + 1.31679 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.942903 w_k1_ADM_s106
                                 + 1.2101 w_k1_AXP_s106
                                 + 0.851374 w_k1_DIS_s106
                                 + 0.872496 w_k1_FDX_s106
                                 + 1.0605 w_k1_IBM_s106
                                 + 1.07127 w_k1_INTC_s106
                                 + 1.01832 w_k1_KO_s106 + 1.09881 w_k1_MAT_s106
                                 + 0.986247 w_k1_MSFT_s106
                                 + 0.885589 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.26915 w_k1_ADM_s106
                                 + 0.893575 w_k1_AXP_s106
                                 + 0.866028 w_k1_DIS_s106
                                 + 0.864729 w_k1_FDX_s106
                                 + 1.00012 w_k1_IBM_s106 + 1.024 w_k1_INTC_s106
                                 + 0.943292 w_k1_KO_s106
                                 + 0.616432 w_k1_MAT_s106
                                 + 1.07651 w_k1_MSFT_s106
                                 + 1.06306 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.782594 w_k1_ADM_s106
                                 + 0.81418 w_k1_AXP_s106
                                 + 0.796572 w_k1_DIS_s106
                                 + 0.703432 w_k1_FDX_s106
                                 + 1.48057 w_k1_IBM_s106
                                 + 1.11492 w_k1_INTC_s106
                                 + 1.07149 w_k1_KO_s106
                                 + 0.908784 w_k1_MAT_s106
                                 + 0.899381 w_k1_MSFT_s106
                                 + 0.924495 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.962654 w_k1_ADM_s106
                                 + 0.714087 w_k1_AXP_s106
                                 + 0.664801 w_k1_DIS_s106
                                 + 0.72837 w_k1_FDX_s106
                                 + 0.811893 w_k1_IBM_s106
                                 + 0.830762 w_k1_INTC_s106
                                 + 0.725298 w_k1_KO_s106
                                 + 0.567885 w_k1_MAT_s106
                                 + 0.685743 w_k1_MSFT_s106
                                 + 0.874042 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.49848 w_k1_ADM_s106
                                 + 1.2106 w_k1_AXP_s106 + 1.60935 w_k1_DIS_s106
                                 + 1.23835 w_k1_FDX_s106
                                 + 1.44673 w_k1_IBM_s106
                                 + 1.22763 w_k1_INTC_s106
                                 + 1.42281 w_k1_KO_s106 + 1.22448 w_k1_MAT_s106
                                 + 1.35888 w_k1_MSFT_s106
                                 + 1.29849 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.11939 w_k1_ADM_s106
                                 + 1.39195 w_k1_AXP_s106
                                 + 1.33609 w_k1_DIS_s106
                                 + 1.04039 w_k1_FDX_s106 + 1.3567 w_k1_IBM_s106
                                 + 1.11386 w_k1_INTC_s106 + 1.2516 w_k1_KO_s106
                                 + 1.25746 w_k1_MAT_s106
                                 + 1.2158 w_k1_MSFT_s106 + 1.20221 w_k1_PG_s106
                                 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.43426 w_k1_ADM_s106
                                 + 1.29025 w_k1_AXP_s106
                                 + 1.30744 w_k1_DIS_s106
                                 + 1.23329 w_k1_FDX_s106
                                 + 0.855116 w_k1_IBM_s106
                                 + 1.11399 w_k1_INTC_s106
                                 + 1.09959 w_k1_KO_s106 + 1.03397 w_k1_MAT_s106
                                 + 1.33158 w_k1_MSFT_s106
                                 + 1.14634 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.33439 w_k1_ADM_s106
                                 + 1.00447 w_k1_AXP_s106
                                 + 0.979972 w_k1_DIS_s106
                                 + 1.0149 w_k1_FDX_s106
                                 + 0.780834 w_k1_IBM_s106
                                 + 1.03866 w_k1_INTC_s106
                                 + 0.93764 w_k1_KO_s106
                                 + 0.747334 w_k1_MAT_s106
                                 + 1.22191 w_k1_MSFT_s106 + 1.0359 w_k1_PG_s106
                                 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.948536 w_k1_ADM_s107
                                 + 1.01199 w_k1_AXP_s107
                                 + 0.876837 w_k1_DIS_s107
                                 + 0.998892 w_k1_FDX_s107
                                 + 1.14747 w_k1_IBM_s107
                                 + 0.877948 w_k1_INTC_s107
                                 + 0.932805 w_k1_KO_s107
                                 + 0.959744 w_k1_MAT_s107
                                 + 0.87933 w_k1_MSFT_s107
                                 + 0.86299 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.870781 w_k1_ADM_s107
                                 + 0.887914 w_k1_AXP_s107
                                 + 0.774241 w_k1_DIS_s107
                                 + 0.848694 w_k1_FDX_s107
                                 + 0.910019 w_k1_IBM_s107
                                 + 1.06689 w_k1_INTC_s107
                                 + 0.95601 w_k1_KO_s107
                                 + 0.784189 w_k1_MAT_s107
                                 + 0.947219 w_k1_MSFT_s107
                                 + 0.962604 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.665604 w_k1_ADM_s107
                                 + 0.612349 w_k1_AXP_s107
                                 + 0.69844 w_k1_DIS_s107
                                 + 0.835349 w_k1_FDX_s107
                                 + 0.664998 w_k1_IBM_s107
                                 + 0.85259 w_k1_INTC_s107
                                 + 0.724765 w_k1_KO_s107
                                 + 0.466404 w_k1_MAT_s107
                                 + 0.922756 w_k1_MSFT_s107
                                 + 0.803945 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.31175 w_k1_ADM_s107
                                 + 1.44932 w_k1_AXP_s107 + 1.5274 w_k1_DIS_s107
                                 + 1.42095 w_k1_FDX_s107
                                 + 1.08378 w_k1_IBM_s107
                                 + 1.18613 w_k1_INTC_s107
                                 + 1.41404 w_k1_KO_s107
                                 + 0.889875 w_k1_MAT_s107
                                 + 1.20234 w_k1_MSFT_s107
                                 + 1.10948 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.11702 w_k1_ADM_s107
                                 + 1.23837 w_k1_AXP_s107
                                 + 1.28905 w_k1_DIS_s107
                                 + 0.998501 w_k1_FDX_s107
                                 + 1.38077 w_k1_IBM_s107
                                 + 0.898086 w_k1_INTC_s107
                                 + 1.23085 w_k1_KO_s107 + 1.24366 w_k1_MAT_s107
                                 + 0.894647 w_k1_MSFT_s107
                                 + 1.01322 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.06563 w_k1_ADM_s107
                                 + 0.841877 w_k1_AXP_s107
                                 + 0.989183 w_k1_DIS_s107
                                 + 1.00648 w_k1_FDX_s107
                                 + 0.871937 w_k1_IBM_s107
                                 + 1.02578 w_k1_INTC_s107
                                 + 0.968725 w_k1_KO_s107
                                 + 0.622017 w_k1_MAT_s107
                                 + 1.02387 w_k1_MSFT_s107
                                 + 0.858104 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.02204 w_k1_ADM_s107
                                 + 1.4218 w_k1_AXP_s107 + 1.59934 w_k1_DIS_s107
                                 + 1.34761 w_k1_FDX_s107
                                 + 1.41483 w_k1_IBM_s107
                                 + 1.30381 w_k1_INTC_s107
                                 + 1.33594 w_k1_KO_s107 + 1.1942 w_k1_MAT_s107
                                 + 1.36514 w_k1_MSFT_s107
                                 + 1.21532 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.966332 w_k1_ADM_s107
                                 + 0.8068 w_k1_AXP_s107 + 1.00492 w_k1_DIS_s107
                                 + 1.0456 w_k1_FDX_s107 + 1.24081 w_k1_IBM_s107
                                 + 1.02236 w_k1_INTC_s107
                                 + 1.01908 w_k1_KO_s107
                                 + 0.837347 w_k1_MAT_s107
                                 + 1.01087 w_k1_MSFT_s107
                                 + 0.927912 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.943524 w_k1_ADM_s110
                                 + 0.734231 w_k1_AXP_s110
                                 + 0.700931 w_k1_DIS_s110
                                 + 0.750192 w_k1_FDX_s110
                                 + 1.14374 w_k1_IBM_s110
                                 + 0.83329 w_k1_INTC_s110
                                 + 1.05818 w_k1_KO_s110 + 1.01195 w_k1_MAT_s110
                                 + 0.905137 w_k1_MSFT_s110
                                 + 0.836288 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.13913 w_k1_ADM_s110
                                 + 1.28138 w_k1_AXP_s110
                                 + 1.26007 w_k1_DIS_s110 + 1.2121 w_k1_FDX_s110
                                 + 1.39451 w_k1_IBM_s110
                                 + 1.0222 w_k1_INTC_s110 + 1.14917 w_k1_KO_s110
                                 + 1.62314 w_k1_MAT_s110
                                 + 1.16001 w_k1_MSFT_s110
                                 + 1.15671 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.813239 w_k1_ADM_s110
                                 + 0.657144 w_k1_AXP_s110
                                 + 0.708444 w_k1_DIS_s110
                                 + 0.769438 w_k1_FDX_s110
                                 + 0.695827 w_k1_IBM_s110
                                 + 0.893214 w_k1_INTC_s110
                                 + 0.708621 w_k1_KO_s110
                                 + 0.523622 w_k1_MAT_s110
                                 + 0.901105 w_k1_MSFT_s110
                                 + 0.787789 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.01161 w_k1_ADM_s110
                                 + 1.04707 w_k1_AXP_s110
                                 + 1.01377 w_k1_DIS_s110 + 1.0592 w_k1_FDX_s110
                                 + 0.970513 w_k1_IBM_s110
                                 + 1.12476 w_k1_INTC_s110 + 1.0724 w_k1_KO_s110
                                 + 0.754639 w_k1_MAT_s110
                                 + 1.12533 w_k1_MSFT_s110 + 1.0607 w_k1_PG_s110
                                 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.1443 w_k1_ADM_s110
                                 + 0.778603 w_k1_AXP_s110
                                 + 0.970586 w_k1_DIS_s110
                                 + 1.06456 w_k1_FDX_s110
                                 + 0.618927 w_k1_IBM_s110
                                 + 0.86015 w_k1_INTC_s110
                                 + 0.81238 w_k1_KO_s110
                                 + 0.423077 w_k1_MAT_s110
                                 + 1.2321 w_k1_MSFT_s110
                                 + 0.913444 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.90075 w_k1_ADM_s110
                                 + 1.02609 w_k1_AXP_s110
                                 + 0.890608 w_k1_DIS_s110
                                 + 0.885624 w_k1_FDX_s110
                                 + 1.62832 w_k1_IBM_s110
                                 + 1.08788 w_k1_INTC_s110
                                 + 1.11557 w_k1_KO_s110 + 1.4596 w_k1_MAT_s110
                                 + 1.08793 w_k1_MSFT_s110
                                 + 1.01526 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.40985 w_k1_ADM_s110
                                 + 1.42801 w_k1_AXP_s110
                                 + 1.64314 w_k1_DIS_s110 + 1.3847 w_k1_FDX_s110
                                 + 1.28894 w_k1_IBM_s110
                                 + 1.26887 w_k1_INTC_s110
                                 + 1.40458 w_k1_KO_s110 + 1.33722 w_k1_MAT_s110
                                 + 1.48245 w_k1_MSFT_s110
                                 + 1.21038 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.3144 w_k1_ADM_s110
                                 + 1.31678 w_k1_AXP_s110
                                 + 1.31623 w_k1_DIS_s110
                                 + 1.28743 w_k1_FDX_s110 + 0.7069 w_k1_IBM_s110
                                 + 1.09743 w_k1_INTC_s110
                                 + 0.957326 w_k1_KO_s110
                                 + 1.14335 w_k1_MAT_s110
                                 + 1.40718 w_k1_MSFT_s110
                                 + 1.01061 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.893374 w_k1_ADM_s111
                                 + 0.992748 w_k1_AXP_s111
                                 + 1.08404 w_k1_DIS_s111
                                 + 0.938662 w_k1_FDX_s111
                                 + 1.40971 w_k1_IBM_s111
                                 + 0.944417 w_k1_INTC_s111
                                 + 1.15983 w_k1_KO_s111
                                 + 0.942691 w_k1_MAT_s111
                                 + 0.899522 w_k1_MSFT_s111
                                 + 0.989302 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.26941 w_k1_ADM_s111
                                 + 1.18672 w_k1_AXP_s111
                                 + 0.95806 w_k1_DIS_s111
                                 + 0.982401 w_k1_FDX_s111
                                 + 0.952147 w_k1_IBM_s111
                                 + 0.798672 w_k1_INTC_s111
                                 + 0.841262 w_k1_KO_s111
                                 + 0.897306 w_k1_MAT_s111
                                 + 0.84984 w_k1_MSFT_s111
                                 + 0.897602 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.31289 w_k1_ADM_s111
                                 + 1.28467 w_k1_AXP_s111
                                 + 1.91172 w_k1_DIS_s111
                                 + 1.43957 w_k1_FDX_s111
                                 + 1.07198 w_k1_IBM_s111
                                 + 1.23032 w_k1_INTC_s111
                                 + 1.41183 w_k1_KO_s111
                                 + 0.977401 w_k1_MAT_s111
                                 + 1.40489 w_k1_MSFT_s111
                                 + 1.29801 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.72767 w_k1_ADM_s111
                                 + 0.740128 w_k1_AXP_s111
                                 + 0.609075 w_k1_DIS_s111
                                 + 0.736445 w_k1_FDX_s111
                                 + 0.721938 w_k1_IBM_s111
                                 + 0.696899 w_k1_INTC_s111
                                 + 0.695382 w_k1_KO_s111
                                 + 0.518651 w_k1_MAT_s111
                                 + 0.624143 w_k1_MSFT_s111
                                 + 0.822863 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.02303 w_k1_ADM_s111
                                 + 0.767293 w_k1_AXP_s111
                                 + 0.768101 w_k1_DIS_s111
                                 + 0.862907 w_k1_FDX_s111
                                 + 0.990948 w_k1_IBM_s111
                                 + 0.844323 w_k1_INTC_s111
                                 + 0.9307 w_k1_KO_s111 + 0.626623 w_k1_MAT_s111
                                 + 0.839569 w_k1_MSFT_s111
                                 + 0.845209 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.857938 w_k1_ADM_s111
                                 + 1.10412 w_k1_AXP_s111
                                 + 0.777349 w_k1_DIS_s111
                                 + 0.7309 w_k1_FDX_s111 + 1.38877 w_k1_IBM_s111
                                 + 0.603395 w_k1_INTC_s111
                                 + 1.10351 w_k1_KO_s111 + 1.19836 w_k1_MAT_s111
                                 + 0.805868 w_k1_MSFT_s111
                                 + 0.92932 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.16851 w_k1_ADM_s111
                                 + 0.837515 w_k1_AXP_s111
                                 + 1.04896 w_k1_DIS_s111
                                 + 1.04219 w_k1_FDX_s111
                                 + 0.672047 w_k1_IBM_s111
                                 + 0.749884 w_k1_INTC_s111
                                 + 0.856007 w_k1_KO_s111
                                 + 0.33162 w_k1_MAT_s111
                                 + 0.944884 w_k1_MSFT_s111
                                 + 0.896946 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.29322 w_k1_ADM_s111
                                 + 1.33161 w_k1_AXP_s111
                                 + 1.41975 w_k1_DIS_s111
                                 + 1.10943 w_k1_FDX_s111
                                 + 1.33744 w_k1_IBM_s111
                                 + 1.17329 w_k1_INTC_s111
                                 + 1.38409 w_k1_KO_s111 + 1.39049 w_k1_MAT_s111
                                 + 1.09479 w_k1_MSFT_s111
                                 + 1.09113 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.967392 w_k1_ADM_s112
                                 + 1.0177 w_k1_AXP_s112 + 1.01442 w_k1_DIS_s112
                                 + 0.806429 w_k1_FDX_s112
                                 + 1.42609 w_k1_IBM_s112
                                 + 0.635118 w_k1_INTC_s112
                                 + 1.07696 w_k1_KO_s112 + 1.17317 w_k1_MAT_s112
                                 + 0.714688 w_k1_MSFT_s112
                                 + 0.960251 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.0571 w_k1_ADM_s112
                                 + 1.25228 w_k1_AXP_s112
                                 + 0.813951 w_k1_DIS_s112
                                 + 0.912125 w_k1_FDX_s112
                                 + 0.96952 w_k1_IBM_s112
                                 + 0.737951 w_k1_INTC_s112
                                 + 0.694349 w_k1_KO_s112
                                 + 1.18625 w_k1_MAT_s112
                                 + 0.744276 w_k1_MSFT_s112 + 0.868 w_k1_PG_s112
                                 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.48671 w_k1_ADM_s112
                                 + 1.37788 w_k1_AXP_s112
                                 + 1.29066 w_k1_DIS_s112
                                 + 1.12093 w_k1_FDX_s112
                                 + 1.42018 w_k1_IBM_s112
                                 + 1.02717 w_k1_INTC_s112
                                 + 1.23674 w_k1_KO_s112 + 1.15705 w_k1_MAT_s112
                                 + 1.0824 w_k1_MSFT_s112 + 1.13544 w_k1_PG_s112
                                 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.18074 w_k1_ADM_s112
                                 + 1.3501 w_k1_AXP_s112 + 1.45752 w_k1_DIS_s112
                                 + 1.31775 w_k1_FDX_s112
                                 + 0.892016 w_k1_IBM_s112
                                 + 1.07337 w_k1_INTC_s112
                                 + 0.985844 w_k1_KO_s112
                                 + 0.801745 w_k1_MAT_s112
                                 + 1.11102 w_k1_MSFT_s112
                                 + 1.05934 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.919059 w_k1_ADM_s112
                                 + 0.771488 w_k1_AXP_s112
                                 + 0.818916 w_k1_DIS_s112
                                 + 0.797221 w_k1_FDX_s112
                                 + 0.874027 w_k1_IBM_s112
                                 + 0.800933 w_k1_INTC_s112
                                 + 0.809764 w_k1_KO_s112
                                 + 0.739168 w_k1_MAT_s112
                                 + 0.876971 w_k1_MSFT_s112
                                 + 0.854078 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.44373 w_k1_ADM_s112
                                 + 1.06523 w_k1_AXP_s112
                                 + 0.945481 w_k1_DIS_s112
                                 + 1.06636 w_k1_FDX_s112
                                 + 0.717606 w_k1_IBM_s112
                                 + 0.740876 w_k1_INTC_s112
                                 + 0.708574 w_k1_KO_s112
                                 + 0.657952 w_k1_MAT_s112
                                 + 0.845215 w_k1_MSFT_s112
                                 + 0.95872 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.924453 w_k1_ADM_s112
                                 + 0.78644 w_k1_AXP_s112
                                 + 0.725428 w_k1_DIS_s112
                                 + 0.929608 w_k1_FDX_s112
                                 + 0.66333 w_k1_IBM_s112
                                 + 0.824278 w_k1_INTC_s112
                                 + 0.65211 w_k1_KO_s112 + 0.54845 w_k1_MAT_s112
                                 + 0.860477 w_k1_MSFT_s112
                                 + 0.830803 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.818584 w_k1_ADM_s112
                                 + 0.691799 w_k1_AXP_s112
                                 + 0.603882 w_k1_DIS_s112
                                 + 0.746696 w_k1_FDX_s112
                                 + 0.593308 w_k1_IBM_s112
                                 + 0.783889 w_k1_INTC_s112
                                 + 0.737828 w_k1_KO_s112
                                 + 0.563846 w_k1_MAT_s112
                                 + 0.854283 w_k1_MSFT_s112
                                 + 0.81486 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.93817 w_k1_ADM_s113
                                 + 0.9847 w_k1_AXP_s113 + 0.89317 w_k1_DIS_s113
                                 + 0.930885 w_k1_FDX_s113
                                 + 0.861403 w_k1_IBM_s113
                                 + 0.96294 w_k1_INTC_s113
                                 + 1.01982 w_k1_KO_s113
                                 + 0.880184 w_k1_MAT_s113
                                 + 0.969236 w_k1_MSFT_s113
                                 + 0.903064 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.20408 w_k1_ADM_s113
                                 + 1.20428 w_k1_AXP_s113
                                 + 1.02098 w_k1_DIS_s113
                                 + 1.24229 w_k1_FDX_s113
                                 + 0.76623 w_k1_IBM_s113
                                 + 1.02989 w_k1_INTC_s113
                                 + 0.911381 w_k1_KO_s113
                                 + 0.859391 w_k1_MAT_s113
                                 + 1.12851 w_k1_MSFT_s113
                                 + 1.01377 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.741505 w_k1_ADM_s113
                                 + 0.667531 w_k1_AXP_s113
                                 + 0.71671 w_k1_DIS_s113
                                 + 0.862253 w_k1_FDX_s113
                                 + 0.598088 w_k1_IBM_s113
                                 + 0.718464 w_k1_INTC_s113
                                 + 0.776638 w_k1_KO_s113
                                 + 0.618481 w_k1_MAT_s113
                                 + 0.844426 w_k1_MSFT_s113
                                 + 0.775871 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.952975 w_k1_ADM_s113
                                 + 1.17206 w_k1_AXP_s113
                                 + 0.756543 w_k1_DIS_s113
                                 + 1.00419 w_k1_FDX_s113
                                 + 0.80163 w_k1_IBM_s113
                                 + 0.874651 w_k1_INTC_s113
                                 + 0.736096 w_k1_KO_s113
                                 + 1.02416 w_k1_MAT_s113
                                 + 0.741617 w_k1_MSFT_s113
                                 + 0.687636 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.21581 w_k1_ADM_s113
                                 + 1.17849 w_k1_AXP_s113
                                 + 1.26382 w_k1_DIS_s113
                                 + 1.15583 w_k1_FDX_s113
                                 + 1.05831 w_k1_IBM_s113
                                 + 1.12933 w_k1_INTC_s113
                                 + 1.09942 w_k1_KO_s113
                                 + 0.929433 w_k1_MAT_s113
                                 + 1.04751 w_k1_MSFT_s113
                                 + 0.962089 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.06761 w_k1_ADM_s113
                                 + 1.66289 w_k1_AXP_s113
                                 + 1.80235 w_k1_DIS_s113
                                 + 1.52566 w_k1_FDX_s113
                                 + 1.21688 w_k1_IBM_s113
                                 + 1.4465 w_k1_INTC_s113 + 1.33759 w_k1_KO_s113
                                 + 0.978672 w_k1_MAT_s113
                                 + 1.29064 w_k1_MSFT_s113
                                 + 1.21243 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.2654 w_k1_ADM_s113
                                 + 1.18124 w_k1_AXP_s113
                                 + 1.35933 w_k1_DIS_s113
                                 + 1.16271 w_k1_FDX_s113
                                 + 1.52945 w_k1_IBM_s113
                                 + 1.02061 w_k1_INTC_s113
                                 + 1.42488 w_k1_KO_s113 + 1.4602 w_k1_MAT_s113
                                 + 1.11583 w_k1_MSFT_s113
                                 + 1.09945 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.818395 w_k1_ADM_s113
                                 + 0.840004 w_k1_AXP_s113
                                 + 0.762246 w_k1_DIS_s113
                                 + 0.814151 w_k1_FDX_s113
                                 + 1.19551 w_k1_IBM_s113
                                 + 0.877179 w_k1_INTC_s113
                                 + 1.00781 w_k1_KO_s113 + 1.21881 w_k1_MAT_s113
                                 + 0.899851 w_k1_MSFT_s113
                                 + 0.833594 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.737577 w_k1_ADM_s114
                                 + 0.758267 w_k1_AXP_s114
                                 + 0.682972 w_k1_DIS_s114
                                 + 0.829275 w_k1_FDX_s114
                                 + 0.871665 w_k1_IBM_s114
                                 + 0.904358 w_k1_INTC_s114
                                 + 0.791299 w_k1_KO_s114
                                 + 1.00213 w_k1_MAT_s114
                                 + 0.855122 w_k1_MSFT_s114
                                 + 0.840206 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.818312 w_k1_ADM_s114
                                 + 1.06475 w_k1_AXP_s114
                                 + 0.92046 w_k1_DIS_s114
                                 + 0.881758 w_k1_FDX_s114
                                 + 1.1207 w_k1_IBM_s114
                                 + 0.948086 w_k1_INTC_s114
                                 + 0.986634 w_k1_KO_s114
                                 + 1.19802 w_k1_MAT_s114 + 1.003 w_k1_MSFT_s114
                                 + 0.93487 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.10635 w_k1_ADM_s114
                                 + 1.76882 w_k1_AXP_s114
                                 + 1.41542 w_k1_DIS_s114
                                 + 1.32496 w_k1_FDX_s114
                                 + 1.17448 w_k1_IBM_s114
                                 + 1.1803 w_k1_INTC_s114 + 1.43056 w_k1_KO_s114
                                 + 2.00745 w_k1_MAT_s114
                                 + 1.32294 w_k1_MSFT_s114
                                 + 1.25924 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.02187 w_k1_ADM_s114
                                 + 1.36182 w_k1_AXP_s114 + 1.2472 w_k1_DIS_s114
                                 + 1.21859 w_k1_FDX_s114
                                 + 1.22233 w_k1_IBM_s114
                                 + 1.1125 w_k1_INTC_s114 + 1.09591 w_k1_KO_s114
                                 + 1.30689 w_k1_MAT_s114
                                 + 1.10313 w_k1_MSFT_s114
                                 + 1.13188 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.86948 w_k1_ADM_s114
                                 + 1.13799 w_k1_AXP_s114
                                 + 1.07648 w_k1_DIS_s114
                                 + 1.10527 w_k1_FDX_s114
                                 + 0.929416 w_k1_IBM_s114
                                 + 0.936458 w_k1_INTC_s114
                                 + 0.912624 w_k1_KO_s114
                                 + 0.923572 w_k1_MAT_s114
                                 + 1.01423 w_k1_MSFT_s114
                                 + 0.935266 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.952018 w_k1_ADM_s114
                                 + 1.25941 w_k1_AXP_s114
                                 + 1.05438 w_k1_DIS_s114
                                 + 0.944056 w_k1_FDX_s114
                                 + 1.55844 w_k1_IBM_s114
                                 + 0.989917 w_k1_INTC_s114
                                 + 1.35607 w_k1_KO_s114 + 1.66888 w_k1_MAT_s114
                                 + 0.919226 w_k1_MSFT_s114
                                 + 1.1432 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.991719 w_k1_ADM_s114
                                 + 0.820643 w_k1_AXP_s114
                                 + 0.777061 w_k1_DIS_s114
                                 + 0.973371 w_k1_FDX_s114
                                 + 0.663751 w_k1_IBM_s114
                                 + 0.931138 w_k1_INTC_s114
                                 + 0.762599 w_k1_KO_s114
                                 + 0.652476 w_k1_MAT_s114
                                 + 0.982721 w_k1_MSFT_s114
                                 + 0.850685 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.992153 w_k1_ADM_s114
                                 + 1.28662 w_k1_AXP_s114
                                 + 0.972296 w_k1_DIS_s114
                                 + 1.06562 w_k1_FDX_s114
                                 + 0.711257 w_k1_IBM_s114
                                 + 0.976982 w_k1_INTC_s114
                                 + 0.755511 w_k1_KO_s114
                                 + 0.856089 w_k1_MAT_s114
                                 + 0.99424 w_k1_MSFT_s114
                                 + 0.92418 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.04124 w_k1_ADM_s115
                                 + 1.25901 w_k1_AXP_s115
                                 + 0.999881 w_k1_DIS_s115
                                 + 0.99997 w_k1_FDX_s115
                                 + 1.26251 w_k1_IBM_s115
                                 + 0.988547 w_k1_INTC_s115
                                 + 0.942996 w_k1_KO_s115
                                 + 1.42774 w_k1_MAT_s115
                                 + 0.938604 w_k1_MSFT_s115
                                 + 1.1061 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.01314 w_k1_ADM_s115
                                 + 1.13488 w_k1_AXP_s115
                                 + 1.40606 w_k1_DIS_s115
                                 + 1.16058 w_k1_FDX_s115
                                 + 1.55717 w_k1_IBM_s115
                                 + 1.13951 w_k1_INTC_s115
                                 + 1.22872 w_k1_KO_s115 + 1.3506 w_k1_MAT_s115
                                 + 1.07474 w_k1_MSFT_s115
                                 + 1.14045 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.24582 w_k1_ADM_s115
                                 + 1.07924 w_k1_AXP_s115
                                 + 0.986613 w_k1_DIS_s115
                                 + 1.14733 w_k1_FDX_s115
                                 + 0.816577 w_k1_IBM_s115
                                 + 1.00331 w_k1_INTC_s115
                                 + 0.851656 w_k1_KO_s115
                                 + 0.764588 w_k1_MAT_s115
                                 + 0.948911 w_k1_MSFT_s115 + 1.061 w_k1_PG_s115
                                 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.09763 w_k1_ADM_s115
                                 + 1.92236 w_k1_AXP_s115
                                 + 1.68764 w_k1_DIS_s115
                                 + 1.37802 w_k1_FDX_s115
                                 + 1.31557 w_k1_IBM_s115
                                 + 1.3416 w_k1_INTC_s115 + 1.36926 w_k1_KO_s115
                                 + 1.71958 w_k1_MAT_s115
                                 + 1.20805 w_k1_MSFT_s115
                                 + 1.28971 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.840453 w_k1_ADM_s115
                                 + 1.10662 w_k1_AXP_s115
                                 + 0.906114 w_k1_DIS_s115
                                 + 0.911648 w_k1_FDX_s115
                                 + 1.1166 w_k1_IBM_s115
                                 + 0.987653 w_k1_INTC_s115
                                 + 0.886641 w_k1_KO_s115
                                 + 1.03766 w_k1_MAT_s115
                                 + 0.885515 w_k1_MSFT_s115
                                 + 0.885453 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.874111 w_k1_ADM_s115
                                 + 0.992239 w_k1_AXP_s115
                                 + 0.961058 w_k1_DIS_s115
                                 + 0.924703 w_k1_FDX_s115
                                 + 1.0035 w_k1_IBM_s115
                                 + 0.792821 w_k1_INTC_s115
                                 + 0.895329 w_k1_KO_s115
                                 + 1.04595 w_k1_MAT_s115
                                 + 0.888624 w_k1_MSFT_s115
                                 + 0.935677 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.07291 w_k1_ADM_s115
                                 + 1.32225 w_k1_AXP_s115
                                 + 1.36685 w_k1_DIS_s115
                                 + 1.20938 w_k1_FDX_s115
                                 + 1.01134 w_k1_IBM_s115
                                 + 1.11289 w_k1_INTC_s115
                                 + 0.918643 w_k1_KO_s115
                                 + 0.933597 w_k1_MAT_s115
                                 + 1.1218 w_k1_MSFT_s115 + 1.02104 w_k1_PG_s115
                                 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.790561 w_k1_ADM_s115
                                 + 0.906468 w_k1_AXP_s115
                                 + 0.695805 w_k1_DIS_s115
                                 + 0.862785 w_k1_FDX_s115
                                 + 0.756294 w_k1_IBM_s115
                                 + 0.837329 w_k1_INTC_s115
                                 + 0.697907 w_k1_KO_s115
                                 + 0.729786 w_k1_MAT_s115
                                 + 0.869254 w_k1_MSFT_s115
                                 + 0.747976 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.09467 w_k1_ADM_s116
                                 + 1.12169 w_k1_AXP_s116
                                 + 0.958368 w_k1_DIS_s116
                                 + 0.863928 w_k1_FDX_s116
                                 + 1.26596 w_k1_IBM_s116
                                 + 0.89034 w_k1_INTC_s116
                                 + 1.14954 w_k1_KO_s116 + 1.33088 w_k1_MAT_s116
                                 + 0.901101 w_k1_MSFT_s116
                                 + 1.11297 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.14986 w_k1_ADM_s116
                                 + 1.2066 w_k1_AXP_s116 + 1.41181 w_k1_DIS_s116
                                 + 1.0304 w_k1_FDX_s116 + 1.59639 w_k1_IBM_s116
                                 + 0.981252 w_k1_INTC_s116
                                 + 1.47962 w_k1_KO_s116 + 1.71583 w_k1_MAT_s116
                                 + 1.06472 w_k1_MSFT_s116 + 1.3595 w_k1_PG_s116
                                 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.761906 w_k1_ADM_s116
                                 + 0.963676 w_k1_AXP_s116
                                 + 0.703331 w_k1_DIS_s116
                                 + 0.766254 w_k1_FDX_s116
                                 + 0.865526 w_k1_IBM_s116
                                 + 0.874656 w_k1_INTC_s116
                                 + 0.797478 w_k1_KO_s116
                                 + 0.943627 w_k1_MAT_s116
                                 + 0.801963 w_k1_MSFT_s116
                                 + 0.981405 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.16568 w_k1_ADM_s116
                                 + 1.06043 w_k1_AXP_s116
                                 + 0.989801 w_k1_DIS_s116
                                 + 1.03814 w_k1_FDX_s116
                                 + 0.76517 w_k1_IBM_s116
                                 + 0.830644 w_k1_INTC_s116
                                 + 0.990724 w_k1_KO_s116
                                 + 0.684064 w_k1_MAT_s116
                                 + 0.922696 w_k1_MSFT_s116
                                 + 0.918893 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.11346 w_k1_ADM_s116
                                 + 0.866522 w_k1_AXP_s116
                                 + 0.696862 w_k1_DIS_s116
                                 + 0.835959 w_k1_FDX_s116
                                 + 0.579531 w_k1_IBM_s116
                                 + 0.834758 w_k1_INTC_s116
                                 + 0.67124 w_k1_KO_s116
                                 + 0.582602 w_k1_MAT_s116
                                 + 0.932315 w_k1_MSFT_s116
                                 + 0.887612 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.950449 w_k1_ADM_s116
                                 + 1.0367 w_k1_AXP_s116 + 1.15078 w_k1_DIS_s116
                                 + 1.11839 w_k1_FDX_s116
                                 + 0.979452 w_k1_IBM_s116
                                 + 0.857762 w_k1_INTC_s116
                                 + 1.07217 w_k1_KO_s116 + 0.91672 w_k1_MAT_s116
                                 + 1.09467 w_k1_MSFT_s116 + 1.0934 w_k1_PG_s116
                                 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.973298 w_k1_ADM_s116
                                 + 1.24185 w_k1_AXP_s116
                                 + 1.39056 w_k1_DIS_s116
                                 + 1.24268 w_k1_FDX_s116
                                 + 1.12656 w_k1_IBM_s116
                                 + 1.16752 w_k1_INTC_s116
                                 + 1.07061 w_k1_KO_s116
                                 + 0.873099 w_k1_MAT_s116
                                 + 1.23209 w_k1_MSFT_s116
                                 + 1.14478 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.707155 w_k1_ADM_s116
                                 + 0.700844 w_k1_AXP_s116
                                 + 0.585467 w_k1_DIS_s116
                                 + 0.683421 w_k1_FDX_s116
                                 + 0.762018 w_k1_IBM_s116
                                 + 0.621501 w_k1_INTC_s116
                                 + 0.662396 w_k1_KO_s116
                                 + 0.65943 w_k1_MAT_s116
                                 + 0.731883 w_k1_MSFT_s116
                                 + 0.863282 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.10277 w_k1_ADM_s117
                                 + 0.849228 w_k1_AXP_s117
                                 + 0.73589 w_k1_DIS_s117
                                 + 0.913754 w_k1_FDX_s117
                                 + 0.760887 w_k1_IBM_s117
                                 + 0.980575 w_k1_INTC_s117
                                 + 0.76335 w_k1_KO_s117
                                 + 0.712357 w_k1_MAT_s117
                                 + 1.03966 w_k1_MSFT_s117
                                 + 0.912401 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.877888 w_k1_ADM_s117
                                 + 0.856008 w_k1_AXP_s117
                                 + 0.768427 w_k1_DIS_s117
                                 + 0.887719 w_k1_FDX_s117
                                 + 0.824931 w_k1_IBM_s117
                                 + 0.804489 w_k1_INTC_s117
                                 + 0.830905 w_k1_KO_s117
                                 + 0.726774 w_k1_MAT_s117
                                 + 0.829061 w_k1_MSFT_s117
                                 + 0.889276 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.845397 w_k1_ADM_s117
                                 + 0.758396 w_k1_AXP_s117
                                 + 0.571429 w_k1_DIS_s117
                                 + 0.685712 w_k1_FDX_s117
                                 + 1.0519 w_k1_IBM_s117
                                 + 0.80414 w_k1_INTC_s117
                                 + 0.822293 w_k1_KO_s117 + 1.0881 w_k1_MAT_s117
                                 + 0.732157 w_k1_MSFT_s117
                                 + 0.920387 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.23506 w_k1_ADM_s117
                                 + 0.969867 w_k1_AXP_s117
                                 + 1.10532 w_k1_DIS_s117
                                 + 1.33633 w_k1_FDX_s117
                                 + 0.636131 w_k1_IBM_s117
                                 + 1.17631 w_k1_INTC_s117
                                 + 0.81051 w_k1_KO_s117
                                 + 0.541722 w_k1_MAT_s117
                                 + 1.11931 w_k1_MSFT_s117 + 1.0299 w_k1_PG_s117
                                 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.845738 w_k1_ADM_s117
                                 + 0.993748 w_k1_AXP_s117
                                 + 1.02798 w_k1_DIS_s117
                                 + 0.91806 w_k1_FDX_s117
                                 + 1.51099 w_k1_IBM_s117
                                 + 0.960106 w_k1_INTC_s117
                                 + 1.10453 w_k1_KO_s117 + 1.31126 w_k1_MAT_s117
                                 + 0.990139 w_k1_MSFT_s117
                                 + 0.928294 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.1672 w_k1_ADM_s117
                                 + 1.34449 w_k1_AXP_s117
                                 + 1.19662 w_k1_DIS_s117
                                 + 1.15375 w_k1_FDX_s117
                                 + 1.34916 w_k1_IBM_s117
                                 + 1.0312 w_k1_INTC_s117 + 1.17363 w_k1_KO_s117
                                 + 1.94238 w_k1_MAT_s117
                                 + 1.08137 w_k1_MSFT_s117
                                 + 1.05965 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.25541 w_k1_ADM_s117
                                 + 1.15587 w_k1_AXP_s117
                                 + 1.12194 w_k1_DIS_s117
                                 + 1.21958 w_k1_FDX_s117
                                 + 0.958632 w_k1_IBM_s117
                                 + 1.17204 w_k1_INTC_s117
                                 + 1.14183 w_k1_KO_s117
                                 + 0.850414 w_k1_MAT_s117
                                 + 1.07836 w_k1_MSFT_s117
                                 + 1.08502 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.44582 w_k1_ADM_s117
                                 + 1.14188 w_k1_AXP_s117
                                 + 1.61362 w_k1_DIS_s117
                                 + 1.32497 w_k1_FDX_s117
                                 + 1.15289 w_k1_IBM_s117
                                 + 1.44204 w_k1_INTC_s117
                                 + 1.41406 w_k1_KO_s117 + 0.86925 w_k1_MAT_s117
                                 + 1.44492 w_k1_MSFT_s117 + 1.1994 w_k1_PG_s117
                                 >= 1
 k1_return_constraint:           0.0504102601364007 w_k1_ADM_s000
                                 + 0.0337867078248526 w_k1_ADM_s001
                                 + 0.0253529072548506 w_k1_ADM_s002
                                 + 0.0158315226685568 w_k1_ADM_s003
                                 + 0.0442912783723426 w_k1_ADM_s004
                                 + 0.0402787572007897 w_k1_ADM_s005
                                 + 0.0358378373733573 w_k1_ADM_s006
                                 + 0.0154532011540396 w_k1_ADM_s007
                                 + 0.0499837232759479 w_k1_ADM_s010
                                 + 0.0527029482150702 w_k1_ADM_s011
                                 + 0.0340422280232901 w_k1_ADM_s012
                                 + 0.0455112108415754 w_k1_ADM_s013
                                 + 0.0099814841638905 w_k1_ADM_s014
                                 + 0.0438583045898718 w_k1_ADM_s015
                                 + 0.0136005419365826 w_k1_ADM_s016
                                 + 0.0470975437591747 w_k1_ADM_s017
                                 + 0.00851355511803236 w_k1_ADM_s100
                                 + 0.00694106810764792 w_k1_ADM_s101
                                 + 0.00748714995147042 w_k1_ADM_s102
                                 + 0.014101463142754 w_k1_ADM_s103
                                 + 0.00789921265440044 w_k1_ADM_s104
                                 + 0.0207852315677641 w_k1_ADM_s105
                                 + 0.020733958427564 w_k1_ADM_s106
                                 + 0.00706154379676703 w_k1_ADM_s107
                                 + 0.0632429283560617 w_k1_ADM_s110
                                 + 0.086295021993297 w_k1_ADM_s111
                                 + 0.0395228026981824 w_k1_ADM_s112
                                 + 0.0456503085383729 w_k1_ADM_s113
                                 + 0.0249860426188916 w_k1_ADM_s114
                                 + 0.0334380372742131 w_k1_ADM_s115
                                 + 0.0213405067965971 w_k1_ADM_s116
                                 + 0.0762740419961839 w_k1_ADM_s117
                                 + 0.0438695280593472 w_k1_AXP_s000
                                 + 0.0365860521250062 w_k1_AXP_s001
                                 + 0.0257611365474622 w_k1_AXP_s002
                                 + 0.0169559713229788 w_k1_AXP_s003
                                 + 0.0417736418565358 w_k1_AXP_s004
                                 + 0.0455047243211346 w_k1_AXP_s005
                                 + 0.0280234904472416 w_k1_AXP_s006
                                 + 0.0153756524533759 w_k1_AXP_s007
                                 + 0.052396343241164 w_k1_AXP_s010
                                 + 0.0557557623635452 w_k1_AXP_s011
                                 + 0.032122607636418 w_k1_AXP_s012
                                 + 0.0433521484034172 w_k1_AXP_s013
                                 + 0.010241405561762 w_k1_AXP_s014
                                 + 0.0384495040563834 w_k1_AXP_s015
                                 + 0.0146963518187143 w_k1_AXP_s016
                                 + 0.0487069708290818 w_k1_AXP_s017
                                 + 0.00866883406643248 w_k1_AXP_s100
                                 + 0.0068238546281042 w_k1_AXP_s101
                                 + 0.00804268794738067 w_k1_AXP_s102
                                 + 0.0148119287567402 w_k1_AXP_s103
                                 + 0.00835905187189324 w_k1_AXP_s104
                                 + 0.0214156937688169 w_k1_AXP_s105
                                 + 0.0184673645197058 w_k1_AXP_s106
                                 + 0.00695772662129277 w_k1_AXP_s107
                                 + 0.0600334538595998 w_k1_AXP_s110
                                 + 0.0796019918420912 w_k1_AXP_s111
                                 + 0.0378332728363528 w_k1_AXP_s112
                                 + 0.0469790809525043 w_k1_AXP_s113
                                 + 0.0311270415495477 w_k1_AXP_s114
                                 + 0.0392420359472474 w_k1_AXP_s115
                                 + 0.0220139640207424 w_k1_AXP_s116
                                 + 0.0708984595798116 w_k1_AXP_s117
                                 + 0.0423685212309105 w_k1_DIS_s000
                                 + 0.0388810460351745 w_k1_DIS_s001
                                 + 0.025750088023708 w_k1_DIS_s002
                                 + 0.0171111902552413 w_k1_DIS_s003
                                 + 0.0455558076663891 w_k1_DIS_s004
                                 + 0.0420253808055914 w_k1_DIS_s005
                                 + 0.0275972141852879 w_k1_DIS_s006
                                 + 0.0166375046188667 w_k1_DIS_s007
                                 + 0.0512868248157633 w_k1_DIS_s010
                                 + 0.0503566293482825 w_k1_DIS_s011
                                 + 0.0332010766737199 w_k1_DIS_s012
                                 + 0.0421728053618919 w_k1_DIS_s013
                                 + 0.0110951608087428 w_k1_DIS_s014
                                 + 0.0412801681681559 w_k1_DIS_s015
                                 + 0.0157839713532828 w_k1_DIS_s016
                                 + 0.0480653023145407 w_k1_DIS_s017
                                 + 0.00816776404811366 w_k1_DIS_s100
                                 + 0.00655976289640306 w_k1_DIS_s101
                                 + 0.00802793948934761 w_k1_DIS_s102
                                 + 0.0154755183971942 w_k1_DIS_s103
                                 + 0.00828632811911148 w_k1_DIS_s104
                                 + 0.0221046884834941 w_k1_DIS_s105
                                 + 0.0178585634150594 w_k1_DIS_s106
                                 + 0.00750850617388777 w_k1_DIS_s107
                                 + 0.0611680930975692 w_k1_DIS_s110
                                 + 0.0825839654550792 w_k1_DIS_s111
                                 + 0.0364679954760163 w_k1_DIS_s112
                                 + 0.0461051532187991 w_k1_DIS_s113
                                 + 0.0273476323497653 w_k1_DIS_s114
                                 + 0.0356869185036819 w_k1_DIS_s115
                                 + 0.0210387124592415 w_k1_DIS_s116
                                 + 0.0689812964993527 w_k1_DIS_s117
                                 + 0.0433933925738647 w_k1_FDX_s000
                                 + 0.03634315910041 w_k1_FDX_s001
                                 + 0.0236045173968552 w_k1_FDX_s002
                                 + 0.0161390270298932 w_k1_FDX_s003
                                 + 0.042705305745429 w_k1_FDX_s004
                                 + 0.0406110625235019 w_k1_FDX_s005
                                 + 0.0284181417407053 w_k1_FDX_s006
                                 + 0.015954919130657 w_k1_FDX_s007
                                 + 0.0504330184153815 w_k1_FDX_s010
                                 + 0.0516138876854552 w_k1_FDX_s011
                                 + 0.0332106976450405 w_k1_FDX_s012
                                 + 0.0406602763015741 w_k1_FDX_s013
                                 + 0.0109861729260725 w_k1_FDX_s014
                                 + 0.0389706921472491 w_k1_FDX_s015
                                 + 0.015652953029459 w_k1_FDX_s016
                                 + 0.0484113452093355 w_k1_FDX_s017
                                 + 0.00847971992419515 w_k1_FDX_s100
                                 + 0.00704462088795018 w_k1_FDX_s101
                                 + 0.0085249388230649 w_k1_FDX_s102
                                 + 0.0160760567966036 w_k1_FDX_s103
                                 + 0.00803116240819493 w_k1_FDX_s104
                                 + 0.0215898736735646 w_k1_FDX_s105
                                 + 0.016724322781234 w_k1_FDX_s106
                                 + 0.00739964505377253 w_k1_FDX_s107
                                 + 0.0617080664145458 w_k1_FDX_s110
                                 + 0.0765375743067482 w_k1_FDX_s111
                                 + 0.0346983164911487 w_k1_FDX_s112
                                 + 0.0472839034958533 w_k1_FDX_s113
                                 + 0.0281236309032716 w_k1_FDX_s114
                                 + 0.03565516105327 w_k1_FDX_s115
                                 + 0.0199586436673463 w_k1_FDX_s116
                                 + 0.0726714499050141 w_k1_FDX_s117
                                 + 0.0444614806822438 w_k1_IBM_s000
                                 + 0.0378139564376985 w_k1_IBM_s001
                                 + 0.0251597284362964 w_k1_IBM_s002
                                 + 0.0171082898722512 w_k1_IBM_s003
                                 + 0.0415172016888768 w_k1_IBM_s004
                                 + 0.0425428498137572 w_k1_IBM_s005
                                 + 0.0271833366979054 w_k1_IBM_s006
                                 + 0.0165332474746415 w_k1_IBM_s007
                                 + 0.0503160412979724 w_k1_IBM_s010
                                 + 0.0511958758720207 w_k1_IBM_s011
                                 + 0.0345111928388576 w_k1_IBM_s012
                                 + 0.0414899197804806 w_k1_IBM_s013
                                 + 0.0111568958873437 w_k1_IBM_s014
                                 + 0.0417528189406796 w_k1_IBM_s015
                                 + 0.0159029088410111 w_k1_IBM_s016
                                 + 0.0497639389668443 w_k1_IBM_s017
                                 + 0.0083853787023092 w_k1_IBM_s100
                                 + 0.00723262951659398 w_k1_IBM_s101
                                 + 0.0081117106181754 w_k1_IBM_s102
                                 + 0.0153652590025689 w_k1_IBM_s103
                                 + 0.00722539770718475 w_k1_IBM_s104
                                 + 0.0208358206061752 w_k1_IBM_s105
                                 + 0.0178889149765472 w_k1_IBM_s106
                                 + 0.00725039271589399 w_k1_IBM_s107
                                 + 0.0622085434462626 w_k1_IBM_s110
                                 + 0.0789674039981563 w_k1_IBM_s111
                                 + 0.0356216888598023 w_k1_IBM_s112
                                 + 0.045802683228522 w_k1_IBM_s113
                                 + 0.0292663178182723 w_k1_IBM_s114
                                 + 0.0356235655207933 w_k1_IBM_s115
                                 + 0.0215903521606951 w_k1_IBM_s116
                                 + 0.0742881910153032 w_k1_IBM_s117
                                 + 0.0435268089251222 w_k1_INTC_s000
                                 + 0.0391128287151601 w_k1_INTC_s001
                                 + 0.0242738274210231 w_k1_INTC_s002
                                 + 0.0157579257845951 w_k1_INTC_s003
                                 + 0.0449467712934255 w_k1_INTC_s004
                                 + 0.0393105016393017 w_k1_INTC_s005
                                 + 0.027928506095945 w_k1_INTC_s006
                                 + 0.0155962872981962 w_k1_INTC_s007
                                 + 0.0477479755739968 w_k1_INTC_s010
                                 + 0.0460905978893801 w_k1_INTC_s011
                                 + 0.0351973945143984 w_k1_INTC_s012
                                 + 0.0398964402200657 w_k1_INTC_s013
                                 + 0.0111650858627883 w_k1_INTC_s014
                                 + 0.0424478497564014 w_k1_INTC_s015
                                 + 0.0154539678195002 w_k1_INTC_s016
                                 + 0.046635619766097 w_k1_INTC_s017
                                 + 0.00821610083392659 w_k1_INTC_s100
                                 + 0.00709697069770348 w_k1_INTC_s101
                                 + 0.00794432670049869 w_k1_INTC_s102
                                 + 0.0180131324742968 w_k1_INTC_s103
                                 + 0.00827595349343911 w_k1_INTC_s104
                                 + 0.0201685292597568 w_k1_INTC_s105
                                 + 0.0181573939632879 w_k1_INTC_s106
                                 + 0.00719112360207295 w_k1_INTC_s107
                                 + 0.0613304948048044 w_k1_INTC_s110
                                 + 0.0681270082881229 w_k1_INTC_s111
                                 + 0.0298894846079321 w_k1_INTC_s112
                                 + 0.0444598502788342 w_k1_INTC_s113
                                 + 0.0271678548712901 w_k1_INTC_s114
                                 + 0.0339410037131027 w_k1_INTC_s115
                                 + 0.0184156589843302 w_k1_INTC_s116
                                 + 0.0727262640052772 w_k1_INTC_s117
                                 + 0.041555657191132 w_k1_KO_s000
                                 + 0.0403451991113753 w_k1_KO_s001
                                 + 0.0260088054502978 w_k1_KO_s002
                                 + 0.0173959859484478 w_k1_KO_s003
                                 + 0.0386341578446567 w_k1_KO_s004
                                 + 0.0431078439709622 w_k1_KO_s005
                                 + 0.029092099149878 w_k1_KO_s006
                                 + 0.0170888140611482 w_k1_KO_s007
                                 + 0.0474434191907724 w_k1_KO_s010
                                 + 0.0599962022249601 w_k1_KO_s011
                                 + 0.0357276785428337 w_k1_KO_s012
                                 + 0.0360549240532487 w_k1_KO_s013
                                 + 0.0113362648011598 w_k1_KO_s014
                                 + 0.0417138590393045 w_k1_KO_s015
                                 + 0.0165720016520396 w_k1_KO_s016
                                 + 0.0466459481907161 w_k1_KO_s017
                                 + 0.00821815934298352 w_k1_KO_s100
                                 + 0.00652988639132322 w_k1_KO_s101
                                 + 0.00717841856264462 w_k1_KO_s102
                                 + 0.0148684259201981 w_k1_KO_s103
                                 + 0.00768521713734121 w_k1_KO_s104
                                 + 0.0205758625323302 w_k1_KO_s105
                                 + 0.0177493385052714 w_k1_KO_s106
                                 + 0.00736837207759203 w_k1_KO_s107
                                 + 0.0610165549869921 w_k1_KO_s110
                                 + 0.0790172450264763 w_k1_KO_s111
                                 + 0.0324287250232171 w_k1_KO_s112
                                 + 0.0470232046760712 w_k1_KO_s113
                                 + 0.0275358564649535 w_k1_KO_s114
                                 + 0.0314911377112327 w_k1_KO_s115
                                 + 0.0217092462181647 w_k1_KO_s116
                                 + 0.0706971280529115 w_k1_KO_s117
                                 + 0.0397321091566229 w_k1_MAT_s000
                                 + 0.0369678105742686 w_k1_MAT_s001
                                 + 0.0232706867648256 w_k1_MAT_s002
                                 + 0.0155757787261483 w_k1_MAT_s003
                                 + 0.0426877747275966 w_k1_MAT_s004
                                 + 0.0486989000815937 w_k1_MAT_s005
                                 + 0.0278178925123924 w_k1_MAT_s006
                                 + 0.0166280656352241 w_k1_MAT_s007
                                 + 0.0556328235563661 w_k1_MAT_s010
                                 + 0.0501025691621975 w_k1_MAT_s011
                                 + 0.0304440226902734 w_k1_MAT_s012
                                 + 0.0422701649235976 w_k1_MAT_s013
                                 + 0.0157284525109783 w_k1_MAT_s014
                                 + 0.0331326217872068 w_k1_MAT_s015
                                 + 0.0143350468467943 w_k1_MAT_s016
                                 + 0.053368816111919 w_k1_MAT_s017
                                 + 0.00670421337781016 w_k1_MAT_s100
                                 + 0.00739178501310995 w_k1_MAT_s101
                                 + 0.010021746663442 w_k1_MAT_s102
                                 + 0.0152719976446047 w_k1_MAT_s103
                                 + 0.00782951694755139 w_k1_MAT_s104
                                 + 0.0196097623579538 w_k1_MAT_s105
                                 + 0.0153628177168083 w_k1_MAT_s106
                                 + 0.00568541481808026 w_k1_MAT_s107
                                 + 0.0581571040897647 w_k1_MAT_s110
                                 + 0.063853178377381 w_k1_MAT_s111
                                 + 0.0311739158181002 w_k1_MAT_s112
                                 + 0.045344533248502 w_k1_MAT_s113
                                 + 0.0323213956421002 w_k1_MAT_s114
                                 + 0.0354061372596952 w_k1_MAT_s115
                                 + 0.021445853495276 w_k1_MAT_s116
                                 + 0.0708667126863825 w_k1_MAT_s117
                                 + 0.0417796717725196 w_k1_MSFT_s000
                                 + 0.0379276114108318 w_k1_MSFT_s001
                                 + 0.0254638466289249 w_k1_MSFT_s002
                                 + 0.0172020237078374 w_k1_MSFT_s003
                                 + 0.0450065708453115 w_k1_MSFT_s004
                                 + 0.0382408208077057 w_k1_MSFT_s005
                                 + 0.0300522667734128 w_k1_MSFT_s006
                                 + 0.0160352376333092 w_k1_MSFT_s007
                                 + 0.0540457298081092 w_k1_MSFT_s010
                                 + 0.0500965109654599 w_k1_MSFT_s011
                                 + 0.0320062901669012 w_k1_MSFT_s012
                                 + 0.0377100878205429 w_k1_MSFT_s013
                                 + 0.0104873590340097 w_k1_MSFT_s014
                                 + 0.0385510994125729 w_k1_MSFT_s015
                                 + 0.01581309259879 w_k1_MSFT_s016
                                 + 0.0415561708698734 w_k1_MSFT_s017
                                 + 0.00811425917251223 w_k1_MSFT_s100
                                 + 0.00810940724598079 w_k1_MSFT_s101
                                 + 0.00749831422998149 w_k1_MSFT_s102
                                 + 0.0143534983195188 w_k1_MSFT_s103
                                 + 0.00781026625271991 w_k1_MSFT_s104
                                 + 0.0213969469831159 w_k1_MSFT_s105
                                 + 0.0189847169268062 w_k1_MSFT_s106
                                 + 0.00720884799893369 w_k1_MSFT_s107
                                 + 0.0683202589672842 w_k1_MSFT_s110
                                 + 0.0717925820751634 w_k1_MSFT_s111
                                 + 0.0320386374099643 w_k1_MSFT_s112
                                 + 0.04540912540721 w_k1_MSFT_s113
                                 + 0.0274438589338796 w_k1_MSFT_s114
                                 + 0.0333278243884998 w_k1_MSFT_s115
                                 + 0.0199550339907383 w_k1_MSFT_s116
                                 + 0.072846935345745 w_k1_MSFT_s117
                                 + 0.0457271376711704 w_k1_PG_s000
                                 + 0.0359403702083974 w_k1_PG_s001
                                 + 0.0238406543768725 w_k1_PG_s002
                                 + 0.0179084484512266 w_k1_PG_s003
                                 + 0.0451458992362926 w_k1_PG_s004
                                 + 0.0432984790841146 w_k1_PG_s005
                                 + 0.0267961369506068 w_k1_PG_s006
                                 + 0.0165159378209161 w_k1_PG_s007
                                 + 0.0484747195720114 w_k1_PG_s010
                                 + 0.0539668405943905 w_k1_PG_s011
                                 + 0.0342814956168587 w_k1_PG_s012
                                 + 0.0387540068020529 w_k1_PG_s013
                                 + 0.0116212680594125 w_k1_PG_s014
                                 + 0.0417816858267162 w_k1_PG_s015
                                 + 0.01398910981259 w_k1_PG_s016
                                 + 0.0486817965361708 w_k1_PG_s017
                                 + 0.00782295669865536 w_k1_PG_s100
                                 + 0.00702590920430573 w_k1_PG_s101
                                 + 0.00843836689461064 w_k1_PG_s102
                                 + 0.0147683570645682 w_k1_PG_s103
                                 + 0.00803884607588059 w_k1_PG_s104
                                 + 0.0224341595445806 w_k1_PG_s105
                                 + 0.0183272918327021 w_k1_PG_s106
                                 + 0.00662680200003978 w_k1_PG_s107
                                 + 0.0596253822294908 w_k1_PG_s110
                                 + 0.0737009799376949 w_k1_PG_s111
                                 + 0.0338058286592534 w_k1_PG_s112
                                 + 0.0421664363048121 w_k1_PG_s113
                                 + 0.027324444451134 w_k1_PG_s114
                                 + 0.0335144636430386 w_k1_PG_s115
                                 + 0.0221150585106706 w_k1_PG_s116
                                 + 0.0705074539420843 w_k1_PG_s117 >= 1.15
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

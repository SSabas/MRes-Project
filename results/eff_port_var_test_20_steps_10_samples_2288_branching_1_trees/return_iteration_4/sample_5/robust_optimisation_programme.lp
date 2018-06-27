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
 w_k1_ADM_s0_return_constraint:  1.16217 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.971037 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.86423 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.04241 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.33966 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.02716 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.17999 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.00643 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.22102 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.0222 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.02733 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.706175 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.917002 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.48922 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.0514 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.993201 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.15723 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.23046 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.870299 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.42408 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.25492 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.1132 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.831494 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.993729 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.16738 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.20153 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.944358 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.778321 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.937483 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.813376 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.690845 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.858879 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.718129 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.972449 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.995806 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.940571 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.05478 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.37342 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.21492 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.960468 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.811568 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.19074 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.47068 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.918085 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.981603 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.12229 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.4605 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.12959 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.02941 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.619133 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.77344 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.56873 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.3058 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.00786 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.20589 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.03251 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.691382 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.75366 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.27435 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.956127 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.02203 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.10998 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.20546 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.37623 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.791432 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.700055 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.790815 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.764156 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.566263 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.765852 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.865993 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.12543 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.40047 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.03682 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.3132 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.65052 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.29379 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.967781 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.724846 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.28863 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.517 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.876568 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.23249 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.35533 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.48727 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.966556 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.984757 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.621363 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.846181 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.67487 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.30834 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.875797 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.43698 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.2203 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.677084 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.70575 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.02744 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.907612 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.952499 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.26686 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.00263 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.61587 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.965097 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.644106 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.971571 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.827045 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.49359 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.730208 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.808256 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.1194 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.38917 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.00693 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.2956 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.93169 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.26043 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.950777 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.84631 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.14529 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.32503 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.926445 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.07073 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.18016 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.27768 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.03076 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.935375 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.747155 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.83974 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.55302 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.08333 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.841811 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.30896 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.03087 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.743232 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.45258 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.18298 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.947743 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.913872 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.05493 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.12922 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.30496 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.08228 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.783762 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.938603 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.864595 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.646383 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.884034 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.790696 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.13813 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.10392 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.00078 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.15695 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.6178 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.919448 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.03936 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.824481 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.26366 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.3785 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.888688 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.07803 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.852311 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.45262 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.814021 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.37018 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.654138 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.971867 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.898775 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.66743 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.35453 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.780629 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.12789 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.755592 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.42202 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.00172 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.756705 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.62615 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.940791 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.902435 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.16702 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.527057 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.622586 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.752435 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.999369 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.684279 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.728243 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.4123 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.877876 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.88767 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.09337 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.26881 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.17121 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.20114 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 1.03171 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.750314 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.06161 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.11929 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.929951 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.13587 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.3269 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.20181 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.21535 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.10936 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.966879 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.929304 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.56972 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.26016 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.906066 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.11583 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.15785 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.790117 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.34223 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.17186 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.931601 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.888827 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.15114 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.00637 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.16756 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.04366 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.792624 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.01561 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.905086 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.621165 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.970143 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.748721 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.16323 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.943328 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.965064 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.09189 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.1422 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.08733 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.993547 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.787038 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.19016 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.34507 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.916959 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.23713 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.985001 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.31928 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.893319 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.16299 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.695656 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.979217 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.2034 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.3711 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.971911 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.14602 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.09721 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.737425 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.50031 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.983092 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.947788 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.07115 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.06985 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.909902 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.24629 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.745029 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.66317 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.83865 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.913382 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.603539 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.744046 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.02359 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.938998 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.47229 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.966645 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.16045 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.34215 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.039 w_0_MAT - w_k1_MAT_s0 + 0.99 b_k1_MAT_s0
                                 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.01595 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.792142 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.23197 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.42428 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.936102 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.06607 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.890908 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.51068 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.990974 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.13643 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.405383 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.69927 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.11597 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.69814 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.11059 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.895196 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.08115 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.741223 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.48756 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.979539 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.900476 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.61818 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.00317 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.03256 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.26055 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.625696 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.614528 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.689452 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.849687 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.558139 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.678235 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.16276 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.837001 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.9161 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.849272 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.32535 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.33393 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.21305 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.97959 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.797163 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.18921 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.26664 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.909505 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.14014 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.25927 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.15756 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.0472 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.973797 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.800167 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.933505 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.46045 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.01677 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.8123 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.2155 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.0774 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.784544 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.30695 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.938671 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.919913 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.884367 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.05974 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.940329 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.18449 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.05846 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.742493 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.915521 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.853457 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.679736 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.874253 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.854679 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.18211 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.15709 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.02359 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.12194 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.47828 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.12483 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.979188 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.836318 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.12451 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.2128 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.962119 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.131 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.05113 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.2317 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.02786 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.05811 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.779824 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.993469 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.22058 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.15783 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.949217 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.01874 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.03316 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.841549 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.32253 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.976908 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.965332 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.06139 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.05543 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.06903 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.16317 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.865269 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.790263 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.909928 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.983106 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.782125 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.872102 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.9739 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.996168 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.17016 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.01973 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.12666 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.23728 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.0870001364759803 z_k1_s0000
                                 - 0.194000081304828 z_k1_s0001
                                 - 0.261999935865103 z_k1_s0002
                                 - 0.0660000986852214 z_k1_s0003
                                 - 0.203999563680824 z_k1_s0004
                                 - 0.273000491279866 z_k1_s0005
                                 - 0.206999970794206 z_k1_s0006
                                 - 0.113000056054883 z_k1_s0007
                                 - 0.143999649569634 z_k1_s0010
                                 - 0.0740000000511041 z_k1_s0011
                                 - 0.0510000302892815 z_k1_s0012
                                 - 0.0650000219530019 z_k1_s0013
                                 - 0.0839997955822867 z_k1_s0014
                                 - 0.106000256711101 z_k1_s0015
                                 - 0.0570000156880162 z_k1_s0016
                                 - 0.19099979252645 z_k1_s0017
                                 - 0.113000308182438 z_k1_s0020
                                 - 0.0450000402035145 z_k1_s0021
                                 - 0.182000337107783 z_k1_s0022
                                 - 0.0879997405130842 z_k1_s0023
                                 - 0.0710001234189593 z_k1_s0024
                                 - 0.111000295488934 z_k1_s0025
                                 - 0.0470000528970188 z_k1_s0026
                                 - 0.161000612826539 z_k1_s0027
                                 - 0.0830002846073077 z_k1_s0030
                                 - 0.13800050660021 z_k1_s0031
                                 - 0.226000901388982 z_k1_s0032
                                 - 0.0780002734261639 z_k1_s0033
                                 - 0.0570002264653599 z_k1_s0034
                                 - 0.12100017158336 z_k1_s0035
                                 - 0.196000834302119 z_k1_s0036
                                 - 0.0910003024971378 z_k1_s0037
                                 - 0.0810002721348243 z_k1_s0040
                                 - 0.0540002117566477 z_k1_s0041
                                 - 0.116999988637882 z_k1_s0042
                                 - 0.0650002363551641 z_k1_s0043
                                 - 0.146000872491166 z_k1_s0044
                                 - 0.245001002877519 z_k1_s0045
                                 - 0.124000641293545 z_k1_s0046
                                 - 0.078000265426138 z_k1_s0047
                                 - 0.0720002030086069 z_k1_s0050
                                 - 0.230001144334654 z_k1_s0051
                                 - 0.200000447245753 z_k1_s0052
                                 - 0.187000355174575 z_k1_s0053
                                 - 0.10800019951257 z_k1_s0054
                                 - 0.0930002709694788 z_k1_s0055
                                 - 0.0770002141897507 z_k1_s0056
                                 - 0.0830002486071912 z_k1_s0057
                                 - 0.135999435595707 z_k1_s0060
                                 - 0.101999912196321 z_k1_s0061
                                 - 0.202000247133313 z_k1_s0062
                                 - 0.264999008785605 z_k1_s0063
                                 - 0.291000223147678 z_k1_s0064
                                 - 0.111999811490204 z_k1_s0065
                                 - 0.165000015846773 z_k1_s0066
                                 - 0.0689999090269661 z_k1_s0067
                                 - 0.0350000103226554 z_k1_s0070
                                 - 0.082999817050926 z_k1_s0071
                                 - 0.0309999714286481 z_k1_s0072
                                 - 0.0530001853456885 z_k1_s0073
                                 - 0.0689999009218395 z_k1_s0074
                                 - 0.0619999428572962 z_k1_s0075
                                 - 0.064000182304239 z_k1_s0076
                                 - 0.0430000001106944 z_k1_s0077
                                 - 0.102000287359518 z_k1_s0100
                                 - 0.0440000285237144 z_k1_s0101
                                 - 0.175000174901136 z_k1_s0102
                                 - 0.0739996915353401 z_k1_s0103
                                 - 0.0780002595102197 z_k1_s0104
                                 - 0.124000031221377 z_k1_s0105
                                 - 0.0430000217299937 z_k1_s0106
                                 - 0.0359999741739485 z_k1_s0107
                                 - 0.233000731824224 z_k1_s0110
                                 - 0.163000682435385 z_k1_s0111
                                 - 0.11300023104235 z_k1_s0112
                                 - 0.145000858878713 z_k1_s0113
                                 - 0.198000707129805 z_k1_s0114
                                 - 0.16400030485431 z_k1_s0115
                                 - 0.187000888512016 z_k1_s0116
                                 - 0.12100025560112 z_k1_s0117
                                 - 0.231999994521126 z_k1_s0120
                                 - 0.0549998843219593 z_k1_s0121
                                 - 0.146999339115488 z_k1_s0122
                                 - 0.223000085595625 z_k1_s0123
                                 - 0.150000135422971 z_k1_s0124
                                 - 0.1059999055658 z_k1_s0125
                                 - 0.0839998526371382 z_k1_s0126
                                 - 0.243000020985457 z_k1_s0127
                                 - 0.112000343995186 z_k1_s0130
                                 - 0.0510000201835707 z_k1_s0131
                                 - 0.0899997078534772 z_k1_s0132
                                 - 0.189999560135055 z_k1_s0133
                                 - 0.0990000766386827 z_k1_s0134
                                 - 0.0549999983548395 z_k1_s0135
                                 - 0.146999655493965 z_k1_s0136
                                 - 0.0519999550264092 z_k1_s0137
                                 - 0.180000200041855 z_k1_s0140
                                 - 0.438999734099035 z_k1_s0141
                                 - 0.180000200041855 z_k1_s0142
                                 - 0.140000155588109 z_k1_s0143
                                 - 0.1650001833717 z_k1_s0144
                                 - 0.419000155272646 z_k1_s0145
                                 - 0.121000223152677 z_k1_s0146
                                 - 0.572999794338985 z_k1_s0147
                                 - 0.155999598964435 z_k1_s0150
                                 - 0.0920000207534855 z_k1_s0151
                                 - 0.0630000667203421 z_k1_s0152
                                 - 0.0750000573099224 z_k1_s0153
                                 - 0.159999874527738 z_k1_s0154
                                 - 0.0650000187019207 z_k1_s0155
                                 - 0.213999878630867 z_k1_s0156
                                 - 0.103999732642957 z_k1_s0157
                                 - 0.118000212970814 z_k1_s0160
                                 - 0.0529999548395073 z_k1_s0161
                                 - 0.0599999501598189 z_k1_s0162
                                 - 0.0479999873678408 z_k1_s0163
                                 - 0.125000208291125 z_k1_s0164
                                 - 0.105999637279158 z_k1_s0165
                                 - 0.0559999625624929 z_k1_s0166
                                 - 0.114999728548079 z_k1_s0167
                                 - 0.222999910207788 z_k1_s0170
                                 - 0.278000800326812 z_k1_s0171
                                 - 0.229999949344786 z_k1_s0172
                                 - 0.400999877263316 z_k1_s0173
                                 - 0.21599987107079 z_k1_s0174
                                 - 0.366998864655388 z_k1_s0175
                                 - 0.347998792697807 z_k1_s0176
                                 - 0.335998999777123 z_k1_s0177
                                 - 0.833998877597183 z_k1_s1000
                                 - 0.199999496696606 z_k1_s1001
                                 - 0.204999747768572 z_k1_s1002
                                 - 0.757998327217738 z_k1_s1003
                                 - 0.396999114909569 z_k1_s1004
                                 - 0.351000751360756 z_k1_s1005
                                 - 0.272999488193569 z_k1_s1006
                                 - 0.384000095079674 z_k1_s1007
                                 - 0.119999867085685 z_k1_s1010
                                 - 0.126999857939017 z_k1_s1011
                                 - 0.145999952540846 z_k1_s1012
                                 - 0.24699939062491 z_k1_s1013
                                 - 0.339999316876298 z_k1_s1014
                                 - 0.136999916529448 z_k1_s1015
                                 - 0.297000519576543 z_k1_s1016
                                 - 0.257999471794373 z_k1_s1017
                                 - 0.235999450654439 z_k1_s1020
                                 - 0.291999011234145 z_k1_s1021
                                 - 0.34199927014315 z_k1_s1022
                                 - 0.665998030279985 z_k1_s1023
                                 - 0.219999576203094 z_k1_s1024
                                 - 0.339998634588179 z_k1_s1025
                                 - 0.35899946237198 z_k1_s1026
                                 - 0.14999973472933 z_k1_s1027
                                 - 0.403999382461524 z_k1_s1030
                                 - 0.378999177254149 z_k1_s1031
                                 - 0.626997978964527 z_k1_s1032
                                 - 0.980996951011302 z_k1_s1033
                                 - 0.38799938351259 z_k1_s1034
                                 - 0.35699935599908 z_k1_s1035
                                 - 1.18799791256155 z_k1_s1036
                                 - 0.403999382461524 z_k1_s1037
                                 - 0.116000063749314 z_k1_s1040
                                 - 0.345999784797082 z_k1_s1041
                                 - 0.263000486686816 z_k1_s1042
                                 - 0.187999780779915 z_k1_s1043
                                 - 0.0899999622882589 z_k1_s1044
                                 - 0.104999977069636 z_k1_s1045
                                 - 0.0760000243323089 z_k1_s1046
                                 - 0.0799999524340075 z_k1_s1047
                                 - 0.249000696484322 z_k1_s1050
                                 - 0.192000638820741 z_k1_s1051
                                 - 0.189000539659216 z_k1_s1052
                                 - 0.172000586545706 z_k1_s1053
                                 - 0.57500264441063 z_k1_s1054
                                 - 0.524002556769426 z_k1_s1055
                                 - 0.19100045356645 z_k1_s1056
                                 - 0.19100045356645 z_k1_s1057
                                 - 0.104000047279607 z_k1_s1060
                                 - 0.17200040496255 z_k1_s1061
                                 - 0.0820000009704456 z_k1_s1062
                                 - 0.20700039818164 z_k1_s1063
                                 - 0.0810000685927828 z_k1_s1064
                                 - 0.0820000009704456 z_k1_s1065
                                 - 0.181999728739178 z_k1_s1066
                                 - 0.270000621976091 z_k1_s1067
                                 - 0.91000200838613 z_k1_s1070
                                 - 0.566002911859944 z_k1_s1071
                                 - 0.847001247143191 z_k1_s1072
                                 - 0.352000635698624 z_k1_s1073
                                 - 0.517000205112642 z_k1_s1074
                                 - 0.329000433308445 z_k1_s1075
                                 - 0.953000111199546 z_k1_s1076
                                 - 0.284000161449562 z_k1_s1077
                                 - 0.518000237452636 z_k1_s1100
                                 - 0.654999471841581 z_k1_s1101
                                 - 0.764997523387335 z_k1_s1102
                                 - 0.610000268936501 z_k1_s1103
                                 - 1.26800079503852 z_k1_s1104
                                 - 0.546000295182512 z_k1_s1105
                                 - 0.560997814684013 z_k1_s1106
                                 - 0.613997983283427 z_k1_s1107
                                 - 2.60300385500016 z_k1_s1110
                                 - 1.49099910372239 z_k1_s1111
                                 - 1.56300421512456 z_k1_s1112
                                 - 1.69600381529209 z_k1_s1113
                                 - 1.79899674337265 z_k1_s1114
                                 - 0.795001021898957 z_k1_s1115
                                 - 1.86300280095327 z_k1_s1116
                                 - 1.81699802122319 z_k1_s1117
                                 - 0.963997147805471 z_k1_s1120
                                 - 0.467000193668918 z_k1_s1121
                                 - 0.608999988922518 z_k1_s1122
                                 - 1.48600227216993 z_k1_s1123
                                 - 0.931999893439675 z_k1_s1124
                                 - 0.660000604320247 z_k1_s1125
                                 - 0.348000088741481 z_k1_s1126
                                 - 1.18900279882618 z_k1_s1127
                                 - 3.0970107779092 z_k1_s1130
                                 - 0.97300110192007 z_k1_s1131
                                 - 1.13700121511141 z_k1_s1132
                                 - 2.01900063208241 z_k1_s1133
                                 - 1.42299565224785 z_k1_s1134
                                 - 0.760001021119228 z_k1_s1135
                                 - 1.58800288368983 z_k1_s1136
                                 - 2.57601031900483 z_k1_s1137
                                 - 0.742999662839159 z_k1_s1140
                                 - 0.759999280799184 z_k1_s1141
                                 - 1.58400112991798 z_k1_s1142
                                 - 0.372999998039558 z_k1_s1143
                                 - 0.591002077079049 z_k1_s1144
                                 - 0.497000149559392 z_k1_s1145
                                 - 0.581000966279195 z_k1_s1146
                                 - 0.547000027559348 z_k1_s1147
                                 - 0.567001305510186 z_k1_s1150
                                 - 0.652000771327274 z_k1_s1151
                                 - 0.309000047500136 z_k1_s1152
                                 - 0.805000699798197 z_k1_s1153
                                 - 0.817999152217008 z_k1_s1154
                                 - 0.225000051869035 z_k1_s1155
                                 - 0.663997973559768 z_k1_s1156
                                 - 0.409999936294656 z_k1_s1157
                                 - 1.18900252763386 z_k1_s1160
                                 - 0.742999608462424 z_k1_s1161
                                 - 0.315999937715883 z_k1_s1162
                                 - 1.27699625104032 z_k1_s1163
                                 - 0.918996736372192 z_k1_s1164
                                 - 0.614999869355914 z_k1_s1165
                                 - 1.00500211142476 z_k1_s1166
                                 - 1.38299681459546 z_k1_s1167
                                 - 0.32700109142431 z_k1_s1170
                                 - 0.238000330579642 z_k1_s1171
                                 - 0.581000704994323 z_k1_s1172
                                 - 0.176000310701814 z_k1_s1173
                                 - 0.197000256408763 z_k1_s1174
                                 - 0.820001483417578 z_k1_s1175
                                 - 0.41999891413897 z_k1_s1176
                                 - 0.393999251616336 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.20167 w_k1_ADM_s000
                                 + 1.30441 w_k1_AXP_s000 + 1.1812 w_k1_DIS_s000
                                 + 1.23153 w_k1_FDX_s000
                                 + 1.02984 w_k1_IBM_s000
                                 + 1.25125 w_k1_INTC_s000
                                 + 1.03901 w_k1_KO_s000 + 1.03974 w_k1_MAT_s000
                                 + 1.19529 w_k1_MSFT_s000
                                 + 1.08876 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.21072 w_k1_ADM_s000
                                 + 1.34869 w_k1_AXP_s000
                                 + 1.31237 w_k1_DIS_s000
                                 + 1.09294 w_k1_FDX_s000
                                 + 1.51747 w_k1_IBM_s000
                                 + 1.2994 w_k1_INTC_s000 + 1.38814 w_k1_KO_s000
                                 + 1.87814 w_k1_MAT_s000
                                 + 1.22649 w_k1_MSFT_s000
                                 + 1.06741 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.743344 w_k1_ADM_s000
                                 + 0.695598 w_k1_AXP_s000
                                 + 0.6869 w_k1_DIS_s000
                                 + 0.762914 w_k1_FDX_s000
                                 + 0.946837 w_k1_IBM_s000
                                 + 0.814327 w_k1_INTC_s000
                                 + 0.645468 w_k1_KO_s000
                                 + 0.761956 w_k1_MAT_s000
                                 + 0.743986 w_k1_MSFT_s000
                                 + 0.81807 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.920515 w_k1_ADM_s000
                                 + 0.759958 w_k1_AXP_s000
                                 + 0.673589 w_k1_DIS_s000
                                 + 0.906679 w_k1_FDX_s000
                                 + 0.514535 w_k1_IBM_s000
                                 + 0.761041 w_k1_INTC_s000
                                 + 0.605237 w_k1_KO_s000
                                 + 0.515143 w_k1_MAT_s000
                                 + 0.767563 w_k1_MSFT_s000
                                 + 0.742948 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.995417 w_k1_ADM_s000
                                 + 0.980383 w_k1_AXP_s000
                                 + 0.943175 w_k1_DIS_s000
                                 + 1.10714 w_k1_FDX_s000
                                 + 0.772834 w_k1_IBM_s000
                                 + 1.23859 w_k1_INTC_s000
                                 + 0.749133 w_k1_KO_s000
                                 + 0.593515 w_k1_MAT_s000
                                 + 1.07134 w_k1_MSFT_s000
                                 + 0.863035 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.873872 w_k1_ADM_s000
                                 + 0.925696 w_k1_AXP_s000
                                 + 0.908312 w_k1_DIS_s000
                                 + 0.934992 w_k1_FDX_s000
                                 + 0.943988 w_k1_IBM_s000
                                 + 1.12478 w_k1_INTC_s000
                                 + 0.844495 w_k1_KO_s000
                                 + 0.645295 w_k1_MAT_s000
                                 + 0.982246 w_k1_MSFT_s000
                                 + 0.898908 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.894905 w_k1_ADM_s000
                                 + 1.14235 w_k1_AXP_s000
                                 + 1.09753 w_k1_DIS_s000
                                 + 1.08906 w_k1_FDX_s000 + 1.111 w_k1_IBM_s000
                                 + 1.10657 w_k1_INTC_s000
                                 + 0.95886 w_k1_KO_s000 + 1.33866 w_k1_MAT_s000
                                 + 0.974154 w_k1_MSFT_s000
                                 + 1.00592 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.26141 w_k1_ADM_s000
                                 + 1.19438 w_k1_AXP_s000
                                 + 1.55105 w_k1_DIS_s000
                                 + 1.27066 w_k1_FDX_s000
                                 + 1.08641 w_k1_IBM_s000
                                 + 1.17058 w_k1_INTC_s000
                                 + 0.943259 w_k1_KO_s000
                                 + 1.08693 w_k1_MAT_s000
                                 + 1.28231 w_k1_MSFT_s000
                                 + 1.02672 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.02653 w_k1_ADM_s001
                                 + 0.768278 w_k1_AXP_s001
                                 + 0.712223 w_k1_DIS_s001
                                 + 0.851459 w_k1_FDX_s001
                                 + 0.940734 w_k1_IBM_s001
                                 + 0.799809 w_k1_INTC_s001
                                 + 0.877823 w_k1_KO_s001
                                 + 0.60274 w_k1_MAT_s001
                                 + 0.755521 w_k1_MSFT_s001
                                 + 0.893743 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.60009 w_k1_ADM_s001
                                 + 1.23722 w_k1_AXP_s001
                                 + 1.52734 w_k1_DIS_s001
                                 + 1.44741 w_k1_FDX_s001 + 1.2353 w_k1_IBM_s001
                                 + 1.06061 w_k1_INTC_s001
                                 + 1.41569 w_k1_KO_s001 + 1.43215 w_k1_MAT_s001
                                 + 1.24176 w_k1_MSFT_s001
                                 + 1.40063 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.33934 w_k1_ADM_s001
                                 + 0.884034 w_k1_AXP_s001
                                 + 1.51894 w_k1_DIS_s001
                                 + 1.06162 w_k1_FDX_s001
                                 + 1.57212 w_k1_IBM_s001
                                 + 1.01529 w_k1_INTC_s001
                                 + 1.58334 w_k1_KO_s001 + 1.03265 w_k1_MAT_s001
                                 + 1.16609 w_k1_MSFT_s001
                                 + 1.05255 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.969479 w_k1_ADM_s001
                                 + 0.667493 w_k1_AXP_s001
                                 + 0.6988 w_k1_DIS_s001
                                 + 0.746711 w_k1_FDX_s001
                                 + 0.741464 w_k1_IBM_s001
                                 + 0.584755 w_k1_INTC_s001
                                 + 0.822641 w_k1_KO_s001
                                 + 0.784132 w_k1_MAT_s001
                                 + 0.681738 w_k1_MSFT_s001
                                 + 0.859854 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.32882 w_k1_ADM_s001
                                 + 1.43214 w_k1_AXP_s001 + 1.3478 w_k1_DIS_s001
                                 + 1.15012 w_k1_FDX_s001
                                 + 1.19647 w_k1_IBM_s001
                                 + 1.18782 w_k1_INTC_s001
                                 + 1.19619 w_k1_KO_s001 + 1.17811 w_k1_MAT_s001
                                 + 1.1404 w_k1_MSFT_s001 + 1.11681 w_k1_PG_s001
                                 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.26382 w_k1_ADM_s001
                                 + 0.872895 w_k1_AXP_s001
                                 + 1.10607 w_k1_DIS_s001
                                 + 0.929861 w_k1_FDX_s001
                                 + 0.76698 w_k1_IBM_s001
                                 + 0.791463 w_k1_INTC_s001
                                 + 1.12044 w_k1_KO_s001
                                 + 0.819034 w_k1_MAT_s001
                                 + 0.946905 w_k1_MSFT_s001
                                 + 1.04478 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.39267 w_k1_ADM_s001
                                 + 1.2119 w_k1_AXP_s001 + 1.08664 w_k1_DIS_s001
                                 + 1.30306 w_k1_FDX_s001
                                 + 0.751345 w_k1_IBM_s001
                                 + 1.12131 w_k1_INTC_s001
                                 + 1.11778 w_k1_KO_s001 + 1.08285 w_k1_MAT_s001
                                 + 1.11207 w_k1_MSFT_s001
                                 + 1.18995 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.25652 w_k1_ADM_s001
                                 + 0.885575 w_k1_AXP_s001
                                 + 1.01273 w_k1_DIS_s001
                                 + 1.09417 w_k1_FDX_s001
                                 + 0.816994 w_k1_IBM_s001
                                 + 0.954337 w_k1_INTC_s001
                                 + 1.06351 w_k1_KO_s001
                                 + 0.577014 w_k1_MAT_s001
                                 + 1.01751 w_k1_MSFT_s001
                                 + 1.17686 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.941385 w_k1_ADM_s002
                                 + 0.809802 w_k1_AXP_s002
                                 + 0.821509 w_k1_DIS_s002
                                 + 0.798838 w_k1_FDX_s002
                                 + 0.809178 w_k1_IBM_s002
                                 + 1.02949 w_k1_INTC_s002
                                 + 0.824656 w_k1_KO_s002
                                 + 0.58331 w_k1_MAT_s002
                                 + 0.812602 w_k1_MSFT_s002
                                 + 0.911598 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.37232 w_k1_ADM_s002
                                 + 1.54323 w_k1_AXP_s002
                                 + 1.62185 w_k1_DIS_s002
                                 + 1.36049 w_k1_FDX_s002
                                 + 1.08523 w_k1_IBM_s002
                                 + 1.61266 w_k1_INTC_s002
                                 + 1.39465 w_k1_KO_s002 + 1.27022 w_k1_MAT_s002
                                 + 1.3624 w_k1_MSFT_s002 + 1.13276 w_k1_PG_s002
                                 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.798895 w_k1_ADM_s002
                                 + 1.00312 w_k1_AXP_s002
                                 + 1.15728 w_k1_DIS_s002
                                 + 1.06262 w_k1_FDX_s002
                                 + 0.894988 w_k1_IBM_s002
                                 + 1.24097 w_k1_INTC_s002
                                 + 0.991513 w_k1_KO_s002
                                 + 1.08164 w_k1_MAT_s002
                                 + 1.23251 w_k1_MSFT_s002
                                 + 1.05808 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.987548 w_k1_ADM_s002
                                 + 0.782608 w_k1_AXP_s002
                                 + 0.834126 w_k1_DIS_s002
                                 + 0.810771 w_k1_FDX_s002
                                 + 1.09161 w_k1_IBM_s002
                                 + 0.980275 w_k1_INTC_s002
                                 + 1.04194 w_k1_KO_s002 + 1.21894 w_k1_MAT_s002
                                 + 0.869846 w_k1_MSFT_s002
                                 + 1.02141 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.31431 w_k1_ADM_s002
                                 + 1.2606 w_k1_AXP_s002 + 1.37046 w_k1_DIS_s002
                                 + 1.27138 w_k1_FDX_s002
                                 + 0.94263 w_k1_IBM_s002
                                 + 1.28482 w_k1_INTC_s002 + 1.1679 w_k1_KO_s002
                                 + 1.18932 w_k1_MAT_s002
                                 + 1.36609 w_k1_MSFT_s002 + 1.1088 w_k1_PG_s002
                                 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.067 w_k1_ADM_s002
                                 + 0.926578 w_k1_AXP_s002
                                 + 0.779815 w_k1_DIS_s002
                                 + 0.923889 w_k1_FDX_s002
                                 + 0.757017 w_k1_IBM_s002
                                 + 1.07057 w_k1_INTC_s002 + 0.8677 w_k1_KO_s002
                                 + 0.658143 w_k1_MAT_s002
                                 + 0.982981 w_k1_MSFT_s002
                                 + 0.937893 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.10901 w_k1_ADM_s002
                                 + 0.88387 w_k1_AXP_s002
                                 + 0.860182 w_k1_DIS_s002
                                 + 0.765792 w_k1_FDX_s002
                                 + 1.33994 w_k1_IBM_s002
                                 + 0.776662 w_k1_INTC_s002
                                 + 1.11342 w_k1_KO_s002 + 1.11614 w_k1_MAT_s002
                                 + 0.800098 w_k1_MSFT_s002
                                 + 0.873816 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.00882 w_k1_ADM_s002
                                 + 1.26552 w_k1_AXP_s002
                                 + 1.32196 w_k1_DIS_s002
                                 + 1.06471 w_k1_FDX_s002
                                 + 1.45641 w_k1_IBM_s002
                                 + 1.10356 w_k1_INTC_s002 + 1.3587 w_k1_KO_s002
                                 + 1.80761 w_k1_MAT_s002
                                 + 1.12132 w_k1_MSFT_s002
                                 + 1.06375 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.919106 w_k1_ADM_s003
                                 + 1.4131 w_k1_AXP_s003 + 1.03354 w_k1_DIS_s003
                                 + 0.973198 w_k1_FDX_s003
                                 + 1.27686 w_k1_IBM_s003
                                 + 1.03774 w_k1_INTC_s003
                                 + 1.15945 w_k1_KO_s003
                                 + 0.973307 w_k1_MAT_s003
                                 + 0.982537 w_k1_MSFT_s003
                                 + 1.00392 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.13369 w_k1_ADM_s003
                                 + 1.1048 w_k1_AXP_s003 + 1.22303 w_k1_DIS_s003
                                 + 1.20615 w_k1_FDX_s003
                                 + 0.977202 w_k1_IBM_s003
                                 + 1.04486 w_k1_INTC_s003
                                 + 1.12498 w_k1_KO_s003 + 1.03832 w_k1_MAT_s003
                                 + 1.10077 w_k1_MSFT_s003
                                 + 1.02383 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.903656 w_k1_ADM_s003
                                 + 0.892036 w_k1_AXP_s003
                                 + 0.939483 w_k1_DIS_s003
                                 + 0.906964 w_k1_FDX_s003
                                 + 1.21878 w_k1_IBM_s003
                                 + 0.885812 w_k1_INTC_s003
                                 + 1.08474 w_k1_KO_s003 + 1.03716 w_k1_MAT_s003
                                 + 0.878131 w_k1_MSFT_s003
                                 + 0.941277 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.86711 w_k1_ADM_s003
                                 + 0.59501 w_k1_AXP_s003
                                 + 0.585804 w_k1_DIS_s003
                                 + 0.726489 w_k1_FDX_s003
                                 + 0.737037 w_k1_IBM_s003
                                 + 0.652707 w_k1_INTC_s003
                                 + 0.731407 w_k1_KO_s003
                                 + 0.693281 w_k1_MAT_s003
                                 + 0.775397 w_k1_MSFT_s003
                                 + 0.755652 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.828488 w_k1_ADM_s003
                                 + 0.697094 w_k1_AXP_s003
                                 + 0.659289 w_k1_DIS_s003
                                 + 0.849527 w_k1_FDX_s003
                                 + 0.582075 w_k1_IBM_s003
                                 + 0.757687 w_k1_INTC_s003
                                 + 0.80979 w_k1_KO_s003
                                 + 0.818776 w_k1_MAT_s003
                                 + 0.853084 w_k1_MSFT_s003
                                 + 0.842418 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.908237 w_k1_ADM_s003
                                 + 0.727865 w_k1_AXP_s003
                                 + 0.919727 w_k1_DIS_s003
                                 + 1.04659 w_k1_FDX_s003
                                 + 0.654475 w_k1_IBM_s003
                                 + 0.922121 w_k1_INTC_s003
                                 + 0.866639 w_k1_KO_s003
                                 + 0.637553 w_k1_MAT_s003
                                 + 1.04171 w_k1_MSFT_s003
                                 + 0.948671 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.29193 w_k1_ADM_s003
                                 + 1.07054 w_k1_AXP_s003
                                 + 0.850358 w_k1_DIS_s003
                                 + 1.01623 w_k1_FDX_s003
                                 + 0.940704 w_k1_IBM_s003
                                 + 0.975611 w_k1_INTC_s003
                                 + 0.933221 w_k1_KO_s003
                                 + 0.867917 w_k1_MAT_s003
                                 + 0.822194 w_k1_MSFT_s003
                                 + 0.866259 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.23685 w_k1_ADM_s003
                                 + 1.63601 w_k1_AXP_s003
                                 + 1.55692 w_k1_DIS_s003 + 1.2693 w_k1_FDX_s003
                                 + 1.5141 w_k1_IBM_s003
                                 + 1.23578 w_k1_INTC_s003
                                 + 1.44325 w_k1_KO_s003 + 1.70545 w_k1_MAT_s003
                                 + 1.20008 w_k1_MSFT_s003
                                 + 1.09654 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.778534 w_k1_ADM_s004
                                 + 0.818697 w_k1_AXP_s004
                                 + 0.762597 w_k1_DIS_s004
                                 + 0.951987 w_k1_FDX_s004
                                 + 0.544664 w_k1_IBM_s004
                                 + 0.781293 w_k1_INTC_s004
                                 + 0.646171 w_k1_KO_s004
                                 + 0.676662 w_k1_MAT_s004
                                 + 0.897897 w_k1_MSFT_s004
                                 + 0.787688 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.46966 w_k1_ADM_s004
                                 + 1.46119 w_k1_AXP_s004
                                 + 1.65332 w_k1_DIS_s004
                                 + 1.46371 w_k1_FDX_s004
                                 + 1.13871 w_k1_IBM_s004
                                 + 0.981053 w_k1_INTC_s004 + 1.158 w_k1_KO_s004
                                 + 1.05465 w_k1_MAT_s004
                                 + 1.07497 w_k1_MSFT_s004
                                 + 1.14766 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.38465 w_k1_ADM_s004
                                 + 1.03374 w_k1_AXP_s004
                                 + 1.11036 w_k1_DIS_s004 + 1.2087 w_k1_FDX_s004
                                 + 0.767823 w_k1_IBM_s004
                                 + 0.888136 w_k1_INTC_s004
                                 + 0.913576 w_k1_KO_s004
                                 + 1.06496 w_k1_MAT_s004
                                 + 1.05806 w_k1_MSFT_s004
                                 + 0.966442 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.897021 w_k1_ADM_s004
                                 + 0.791431 w_k1_AXP_s004
                                 + 0.554383 w_k1_DIS_s004
                                 + 0.771747 w_k1_FDX_s004
                                 + 0.769464 w_k1_IBM_s004
                                 + 0.783762 w_k1_INTC_s004
                                 + 0.667472 w_k1_KO_s004
                                 + 0.768148 w_k1_MAT_s004
                                 + 0.657635 w_k1_MSFT_s004
                                 + 0.791242 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.753187 w_k1_ADM_s004
                                 + 0.80732 w_k1_AXP_s004
                                 + 0.830279 w_k1_DIS_s004
                                 + 0.841895 w_k1_FDX_s004
                                 + 0.94752 w_k1_IBM_s004
                                 + 0.951671 w_k1_INTC_s004
                                 + 0.803387 w_k1_KO_s004
                                 + 0.994398 w_k1_MAT_s004
                                 + 0.890531 w_k1_MSFT_s004
                                 + 0.822441 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.975528 w_k1_ADM_s004
                                 + 1.42242 w_k1_AXP_s004
                                 + 1.15278 w_k1_DIS_s004
                                 + 1.06383 w_k1_FDX_s004
                                 + 1.44377 w_k1_IBM_s004
                                 + 1.05848 w_k1_INTC_s004 + 1.1285 w_k1_KO_s004
                                 + 1.70248 w_k1_MAT_s004
                                 + 1.2189 w_k1_MSFT_s004 + 1.10253 w_k1_PG_s004
                                 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.97539 w_k1_ADM_s004
                                 + 0.888228 w_k1_AXP_s004
                                 + 0.830326 w_k1_DIS_s004
                                 + 0.92681 w_k1_FDX_s004
                                 + 0.840545 w_k1_IBM_s004
                                 + 0.918847 w_k1_INTC_s004
                                 + 0.776304 w_k1_KO_s004
                                 + 0.902478 w_k1_MAT_s004
                                 + 0.894098 w_k1_MSFT_s004
                                 + 0.835081 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.699253 w_k1_ADM_s004
                                 + 0.808697 w_k1_AXP_s004
                                 + 0.881538 w_k1_DIS_s004
                                 + 0.803149 w_k1_FDX_s004
                                 + 1.28147 w_k1_IBM_s004
                                 + 0.832403 w_k1_INTC_s004
                                 + 0.947227 w_k1_KO_s004
                                 + 1.08553 w_k1_MAT_s004
                                 + 0.805718 w_k1_MSFT_s004
                                 + 0.869082 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.715206 w_k1_ADM_s005
                                 + 0.596964 w_k1_AXP_s005
                                 + 0.572514 w_k1_DIS_s005
                                 + 0.632582 w_k1_FDX_s005
                                 + 0.943221 w_k1_IBM_s005
                                 + 0.700768 w_k1_INTC_s005
                                 + 0.775997 w_k1_KO_s005
                                 + 1.28193 w_k1_MAT_s005
                                 + 0.721014 w_k1_MSFT_s005
                                 + 0.852011 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.26281 w_k1_ADM_s005
                                 + 1.29708 w_k1_AXP_s005
                                 + 1.07907 w_k1_DIS_s005
                                 + 0.990421 w_k1_FDX_s005
                                 + 1.20029 w_k1_IBM_s005
                                 + 1.04538 w_k1_INTC_s005
                                 + 1.22676 w_k1_KO_s005 + 2.20891 w_k1_MAT_s005
                                 + 1.07949 w_k1_MSFT_s005
                                 + 1.07087 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.878798 w_k1_ADM_s005
                                 + 0.744772 w_k1_AXP_s005
                                 + 0.740716 w_k1_DIS_s005
                                 + 0.86648 w_k1_FDX_s005
                                 + 0.713521 w_k1_IBM_s005
                                 + 0.768705 w_k1_INTC_s005
                                 + 0.731801 w_k1_KO_s005
                                 + 0.687408 w_k1_MAT_s005
                                 + 0.945464 w_k1_MSFT_s005
                                 + 0.854118 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.17224 w_k1_ADM_s005
                                 + 0.980867 w_k1_AXP_s005
                                 + 1.24035 w_k1_DIS_s005
                                 + 1.18814 w_k1_FDX_s005
                                 + 0.86245 w_k1_IBM_s005
                                 + 0.906376 w_k1_INTC_s005
                                 + 1.06647 w_k1_KO_s005 + 1.23446 w_k1_MAT_s005
                                 + 1.19053 w_k1_MSFT_s005
                                 + 1.11641 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.14389 w_k1_ADM_s005
                                 + 1.3921 w_k1_AXP_s005 + 1.40079 w_k1_DIS_s005
                                 + 1.23 w_k1_FDX_s005 + 1.21302 w_k1_IBM_s005
                                 + 0.982174 w_k1_INTC_s005
                                 + 1.25052 w_k1_KO_s005 + 2.8191 w_k1_MAT_s005
                                 + 1.21362 w_k1_MSFT_s005
                                 + 1.33394 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.994393 w_k1_ADM_s005
                                 + 0.969171 w_k1_AXP_s005
                                 + 0.798132 w_k1_DIS_s005
                                 + 0.77628 w_k1_FDX_s005
                                 + 1.09877 w_k1_IBM_s005
                                 + 0.719126 w_k1_INTC_s005
                                 + 1.03621 w_k1_KO_s005 + 1.86729 w_k1_MAT_s005
                                 + 0.893216 w_k1_MSFT_s005
                                 + 1.00922 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.858984 w_k1_ADM_s005
                                 + 0.84892 w_k1_AXP_s005
                                 + 0.831553 w_k1_DIS_s005
                                 + 0.750587 w_k1_FDX_s005
                                 + 1.53439 w_k1_IBM_s005
                                 + 0.84089 w_k1_INTC_s005
                                 + 1.13279 w_k1_KO_s005 + 1.90835 w_k1_MAT_s005
                                 + 0.840145 w_k1_MSFT_s005
                                 + 1.00931 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.25378 w_k1_ADM_s005
                                 + 1.52243 w_k1_AXP_s005
                                 + 1.64494 w_k1_DIS_s005 + 1.4445 w_k1_FDX_s005
                                 + 1.14146 w_k1_IBM_s005
                                 + 1.19514 w_k1_INTC_s005
                                 + 1.30333 w_k1_KO_s005 + 1.73813 w_k1_MAT_s005
                                 + 1.47663 w_k1_MSFT_s005
                                 + 1.30263 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.926587 w_k1_ADM_s006
                                 + 1.36064 w_k1_AXP_s006
                                 + 1.00186 w_k1_DIS_s006
                                 + 1.00475 w_k1_FDX_s006
                                 + 1.40293 w_k1_IBM_s006
                                 + 1.01747 w_k1_INTC_s006
                                 + 1.05849 w_k1_KO_s006 + 2.60396 w_k1_MAT_s006
                                 + 1.0339 w_k1_MSFT_s006 + 1.09495 w_k1_PG_s006
                                 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.975804 w_k1_ADM_s006
                                 + 0.958245 w_k1_AXP_s006
                                 + 0.932907 w_k1_DIS_s006
                                 + 0.859803 w_k1_FDX_s006
                                 + 1.27854 w_k1_IBM_s006
                                 + 0.830892 w_k1_INTC_s006
                                 + 0.929424 w_k1_KO_s006
                                 + 1.15808 w_k1_MAT_s006
                                 + 0.877027 w_k1_MSFT_s006
                                 + 0.975156 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.846524 w_k1_ADM_s006
                                 + 0.986374 w_k1_AXP_s006
                                 + 1.04231 w_k1_DIS_s006
                                 + 0.928314 w_k1_FDX_s006
                                 + 0.932439 w_k1_IBM_s006
                                 + 1.23542 w_k1_INTC_s006
                                 + 0.98601 w_k1_KO_s006
                                 + 0.930408 w_k1_MAT_s006
                                 + 0.993449 w_k1_MSFT_s006
                                 + 0.938943 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.963049 w_k1_ADM_s006
                                 + 1.00689 w_k1_AXP_s006
                                 + 1.03449 w_k1_DIS_s006
                                 + 1.08889 w_k1_FDX_s006
                                 + 0.804827 w_k1_IBM_s006
                                 + 1.09335 w_k1_INTC_s006
                                 + 0.936914 w_k1_KO_s006
                                 + 1.17363 w_k1_MAT_s006
                                 + 1.09189 w_k1_MSFT_s006
                                 + 0.930221 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.87093 w_k1_ADM_s006
                                 + 0.86786 w_k1_AXP_s006
                                 + 0.726163 w_k1_DIS_s006
                                 + 0.748749 w_k1_FDX_s006
                                 + 0.770337 w_k1_IBM_s006
                                 + 0.948108 w_k1_INTC_s006
                                 + 0.752903 w_k1_KO_s006
                                 + 0.874293 w_k1_MAT_s006
                                 + 0.775352 w_k1_MSFT_s006
                                 + 0.802612 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.04611 w_k1_ADM_s006
                                 + 0.932667 w_k1_AXP_s006
                                 + 1.40196 w_k1_DIS_s006
                                 + 1.06788 w_k1_FDX_s006
                                 + 1.41092 w_k1_IBM_s006
                                 + 1.16955 w_k1_INTC_s006
                                 + 1.22789 w_k1_KO_s006 + 1.23074 w_k1_MAT_s006
                                 + 1.21624 w_k1_MSFT_s006 + 1.1132 w_k1_PG_s006
                                 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.17727 w_k1_ADM_s006
                                 + 1.44066 w_k1_AXP_s006
                                 + 1.53231 w_k1_DIS_s006
                                 + 1.38721 w_k1_FDX_s006
                                 + 1.08259 w_k1_IBM_s006
                                 + 1.41528 w_k1_INTC_s006
                                 + 1.36415 w_k1_KO_s006 + 1.29559 w_k1_MAT_s006
                                 + 1.33097 w_k1_MSFT_s006 + 1.1972 w_k1_PG_s006
                                 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.817479 w_k1_ADM_s006
                                 + 0.71883 w_k1_AXP_s006
                                 + 0.653148 w_k1_DIS_s006
                                 + 0.680395 w_k1_FDX_s006
                                 + 1.31044 w_k1_IBM_s006
                                 + 0.807216 w_k1_INTC_s006
                                 + 1.01306 w_k1_KO_s006 + 1.59512 w_k1_MAT_s006
                                 + 0.859541 w_k1_MSFT_s006
                                 + 0.877519 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.898907 w_k1_ADM_s007
                                 + 0.67125 w_k1_AXP_s007
                                 + 0.918041 w_k1_DIS_s007
                                 + 0.922883 w_k1_FDX_s007
                                 + 0.669878 w_k1_IBM_s007
                                 + 0.798346 w_k1_INTC_s007
                                 + 0.765486 w_k1_KO_s007
                                 + 0.352862 w_k1_MAT_s007
                                 + 0.97351 w_k1_MSFT_s007
                                 + 0.85871 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.586117 w_k1_ADM_s007
                                 + 0.644966 w_k1_AXP_s007
                                 + 0.665003 w_k1_DIS_s007
                                 + 0.769379 w_k1_FDX_s007
                                 + 0.82994 w_k1_IBM_s007
                                 + 0.922411 w_k1_INTC_s007
                                 + 0.730032 w_k1_KO_s007
                                 + 0.636672 w_k1_MAT_s007
                                 + 0.855379 w_k1_MSFT_s007
                                 + 0.860825 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.0932 w_k1_ADM_s007
                                 + 1.6987 w_k1_AXP_s007 + 1.90035 w_k1_DIS_s007
                                 + 1.27359 w_k1_FDX_s007
                                 + 1.48564 w_k1_IBM_s007
                                 + 1.30545 w_k1_INTC_s007
                                 + 1.67589 w_k1_KO_s007 + 1.5421 w_k1_MAT_s007
                                 + 1.31027 w_k1_MSFT_s007
                                 + 1.23848 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.902192 w_k1_ADM_s007
                                 + 0.848046 w_k1_AXP_s007
                                 + 0.782638 w_k1_DIS_s007
                                 + 0.910608 w_k1_FDX_s007
                                 + 0.874543 w_k1_IBM_s007
                                 + 0.853061 w_k1_INTC_s007
                                 + 0.842477 w_k1_KO_s007
                                 + 0.745573 w_k1_MAT_s007
                                 + 0.84103 w_k1_MSFT_s007
                                 + 0.812639 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.903847 w_k1_ADM_s007
                                 + 0.77789 w_k1_AXP_s007
                                 + 1.22781 w_k1_DIS_s007
                                 + 0.882334 w_k1_FDX_s007
                                 + 1.13847 w_k1_IBM_s007
                                 + 0.871144 w_k1_INTC_s007
                                 + 1.05309 w_k1_KO_s007
                                 + 0.983517 w_k1_MAT_s007
                                 + 1.05665 w_k1_MSFT_s007
                                 + 1.00267 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.985214 w_k1_ADM_s007
                                 + 1.14522 w_k1_AXP_s007 + 1.0817 w_k1_DIS_s007
                                 + 0.920288 w_k1_FDX_s007
                                 + 1.38694 w_k1_IBM_s007
                                 + 0.865456 w_k1_INTC_s007
                                 + 1.1379 w_k1_KO_s007 + 1.09752 w_k1_MAT_s007
                                 + 0.974474 w_k1_MSFT_s007
                                 + 1.28988 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.02088 w_k1_ADM_s007
                                 + 1.1272 w_k1_AXP_s007 + 1.47909 w_k1_DIS_s007
                                 + 1.1981 w_k1_FDX_s007 + 1.45641 w_k1_IBM_s007
                                 + 1.08797 w_k1_INTC_s007
                                 + 1.42383 w_k1_KO_s007 + 1.92153 w_k1_MAT_s007
                                 + 1.36779 w_k1_MSFT_s007
                                 + 1.30222 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.40217 w_k1_ADM_s007
                                 + 1.18681 w_k1_AXP_s007
                                 + 1.06305 w_k1_DIS_s007
                                 + 1.18619 w_k1_FDX_s007
                                 + 0.766632 w_k1_IBM_s007
                                 + 0.806989 w_k1_INTC_s007
                                 + 0.880697 w_k1_KO_s007
                                 + 0.557482 w_k1_MAT_s007
                                 + 1.15186 w_k1_MSFT_s007
                                 + 1.04003 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.27463 w_k1_ADM_s010
                                 + 1.15577 w_k1_AXP_s010 + 1.0902 w_k1_DIS_s010
                                 + 1.16623 w_k1_FDX_s010
                                 + 0.835645 w_k1_IBM_s010
                                 + 0.815998 w_k1_INTC_s010
                                 + 0.896064 w_k1_KO_s010
                                 + 0.987166 w_k1_MAT_s010
                                 + 1.03879 w_k1_MSFT_s010
                                 + 0.972824 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.744406 w_k1_ADM_s010
                                 + 0.662549 w_k1_AXP_s010
                                 + 0.616539 w_k1_DIS_s010
                                 + 0.664778 w_k1_FDX_s010
                                 + 0.935394 w_k1_IBM_s010
                                 + 0.737858 w_k1_INTC_s010
                                 + 0.879966 w_k1_KO_s010
                                 + 1.02149 w_k1_MAT_s010
                                 + 0.869807 w_k1_MSFT_s010
                                 + 0.828689 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.18962 w_k1_ADM_s010
                                 + 0.869659 w_k1_AXP_s010
                                 + 0.971333 w_k1_DIS_s010
                                 + 0.933818 w_k1_FDX_s010
                                 + 0.972748 w_k1_IBM_s010
                                 + 0.904721 w_k1_INTC_s010
                                 + 1.10795 w_k1_KO_s010 + 1.13478 w_k1_MAT_s010
                                 + 0.924319 w_k1_MSFT_s010
                                 + 0.933589 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.13681 w_k1_ADM_s010
                                 + 1.05675 w_k1_AXP_s010
                                 + 1.38423 w_k1_DIS_s010
                                 + 1.22213 w_k1_FDX_s010
                                 + 0.874466 w_k1_IBM_s010
                                 + 0.795988 w_k1_INTC_s010
                                 + 0.981896 w_k1_KO_s010
                                 + 0.869069 w_k1_MAT_s010
                                 + 1.02757 w_k1_MSFT_s010
                                 + 0.970226 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.83022 w_k1_ADM_s010
                                 + 0.658921 w_k1_AXP_s010
                                 + 0.714191 w_k1_DIS_s010
                                 + 0.859499 w_k1_FDX_s010
                                 + 0.670832 w_k1_IBM_s010
                                 + 0.688704 w_k1_INTC_s010
                                 + 0.714042 w_k1_KO_s010
                                 + 0.652067 w_k1_MAT_s010
                                 + 0.861992 w_k1_MSFT_s010
                                 + 0.854018 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.37329 w_k1_ADM_s010
                                 + 1.30802 w_k1_AXP_s010 + 1.2827 w_k1_DIS_s010
                                 + 1.17173 w_k1_FDX_s010 + 1.2596 w_k1_IBM_s010
                                 + 0.938339 w_k1_INTC_s010
                                 + 1.13865 w_k1_KO_s010 + 1.34393 w_k1_MAT_s010
                                 + 0.977717 w_k1_MSFT_s010
                                 + 0.985158 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.17731 w_k1_ADM_s010
                                 + 0.957446 w_k1_AXP_s010
                                 + 0.951623 w_k1_DIS_s010
                                 + 0.845241 w_k1_FDX_s010
                                 + 1.43401 w_k1_IBM_s010
                                 + 0.793879 w_k1_INTC_s010
                                 + 1.16899 w_k1_KO_s010 + 1.49595 w_k1_MAT_s010
                                 + 0.968568 w_k1_MSFT_s010
                                 + 1.05919 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.27401 w_k1_ADM_s010
                                 + 1.62605 w_k1_AXP_s010
                                 + 1.57049 w_k1_DIS_s010
                                 + 1.16072 w_k1_FDX_s010
                                 + 1.72882 w_k1_IBM_s010
                                 + 1.06493 w_k1_INTC_s010
                                 + 1.59341 w_k1_KO_s010 + 2.40797 w_k1_MAT_s010
                                 + 1.34767 w_k1_MSFT_s010
                                 + 1.23351 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.951813 w_k1_ADM_s011
                                 + 0.80071 w_k1_AXP_s011
                                 + 0.935791 w_k1_DIS_s011
                                 + 0.975578 w_k1_FDX_s011
                                 + 0.780924 w_k1_IBM_s011
                                 + 1.25662 w_k1_INTC_s011
                                 + 1.02946 w_k1_KO_s011
                                 + 0.880943 w_k1_MAT_s011
                                 + 1.10351 w_k1_MSFT_s011
                                 + 1.00312 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.848375 w_k1_ADM_s011
                                 + 0.872258 w_k1_AXP_s011
                                 + 0.770818 w_k1_DIS_s011
                                 + 0.83495 w_k1_FDX_s011
                                 + 0.99481 w_k1_IBM_s011
                                 + 0.892904 w_k1_INTC_s011
                                 + 0.943591 w_k1_KO_s011
                                 + 1.34751 w_k1_MAT_s011
                                 + 1.0059 w_k1_MSFT_s011
                                 + 0.923089 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.833766 w_k1_ADM_s011
                                 + 0.721632 w_k1_AXP_s011
                                 + 0.941295 w_k1_DIS_s011
                                 + 0.779049 w_k1_FDX_s011
                                 + 1.16188 w_k1_IBM_s011
                                 + 0.97699 w_k1_INTC_s011
                                 + 1.09878 w_k1_KO_s011 + 1.15242 w_k1_MAT_s011
                                 + 0.986837 w_k1_MSFT_s011
                                 + 0.905175 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.05268 w_k1_ADM_s011
                                 + 0.984593 w_k1_AXP_s011
                                 + 1.43686 w_k1_DIS_s011
                                 + 1.29486 w_k1_FDX_s011
                                 + 0.85764 w_k1_IBM_s011
                                 + 1.35745 w_k1_INTC_s011 + 1.3327 w_k1_KO_s011
                                 + 1.08527 w_k1_MAT_s011
                                 + 1.2878 w_k1_MSFT_s011 + 1.188 w_k1_PG_s011
                                 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.873355 w_k1_ADM_s011
                                 + 0.924262 w_k1_AXP_s011
                                 + 1.31448 w_k1_DIS_s011
                                 + 1.14068 w_k1_FDX_s011
                                 + 1.12483 w_k1_IBM_s011
                                 + 1.24585 w_k1_INTC_s011 + 1.3259 w_k1_KO_s011
                                 + 1.28074 w_k1_MAT_s011
                                 + 1.19393 w_k1_MSFT_s011
                                 + 0.951379 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.659454 w_k1_ADM_s011
                                 + 0.716443 w_k1_AXP_s011
                                 + 0.639985 w_k1_DIS_s011
                                 + 0.777584 w_k1_FDX_s011
                                 + 0.734941 w_k1_IBM_s011
                                 + 0.897258 w_k1_INTC_s011
                                 + 0.740175 w_k1_KO_s011
                                 + 0.824009 w_k1_MAT_s011
                                 + 0.851294 w_k1_MSFT_s011
                                 + 0.705029 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.11628 w_k1_ADM_s011
                                 + 1.24683 w_k1_AXP_s011
                                 + 1.10688 w_k1_DIS_s011
                                 + 1.12209 w_k1_FDX_s011
                                 + 0.987604 w_k1_IBM_s011
                                 + 0.908963 w_k1_INTC_s011
                                 + 1.12563 w_k1_KO_s011 + 1.70326 w_k1_MAT_s011
                                 + 1.09783 w_k1_MSFT_s011
                                 + 1.19399 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.09212 w_k1_ADM_s011
                                 + 1.53779 w_k1_AXP_s011
                                 + 1.46865 w_k1_DIS_s011
                                 + 1.24654 w_k1_FDX_s011
                                 + 1.66906 w_k1_IBM_s011
                                 + 1.1675 w_k1_INTC_s011 + 1.30452 w_k1_KO_s011
                                 + 2.70059 w_k1_MAT_s011
                                 + 1.12791 w_k1_MSFT_s011
                                 + 1.10724 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.877169 w_k1_ADM_s012
                                 + 0.886143 w_k1_AXP_s012
                                 + 0.825881 w_k1_DIS_s012
                                 + 0.884792 w_k1_FDX_s012
                                 + 0.806767 w_k1_IBM_s012
                                 + 1.01164 w_k1_INTC_s012
                                 + 0.780452 w_k1_KO_s012
                                 + 0.763708 w_k1_MAT_s012
                                 + 0.986248 w_k1_MSFT_s012
                                 + 0.98527 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.16641 w_k1_ADM_s012
                                 + 1.82122 w_k1_AXP_s012
                                 + 0.917306 w_k1_DIS_s012
                                 + 1.08302 w_k1_FDX_s012
                                 + 1.02878 w_k1_IBM_s012
                                 + 1.25856 w_k1_INTC_s012
                                 + 0.902255 w_k1_KO_s012
                                 + 1.18247 w_k1_MAT_s012
                                 + 0.938308 w_k1_MSFT_s012
                                 + 1.07414 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.967741 w_k1_ADM_s012
                                 + 0.706646 w_k1_AXP_s012
                                 + 0.622443 w_k1_DIS_s012
                                 + 0.740926 w_k1_FDX_s012
                                 + 0.836321 w_k1_IBM_s012
                                 + 0.826537 w_k1_INTC_s012
                                 + 0.687681 w_k1_KO_s012
                                 + 1.01936 w_k1_MAT_s012
                                 + 0.738062 w_k1_MSFT_s012
                                 + 0.797482 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.35057 w_k1_ADM_s012
                                 + 0.884498 w_k1_AXP_s012
                                 + 1.06563 w_k1_DIS_s012
                                 + 1.00879 w_k1_FDX_s012
                                 + 0.890814 w_k1_IBM_s012
                                 + 0.925628 w_k1_INTC_s012
                                 + 0.895498 w_k1_KO_s012
                                 + 0.956092 w_k1_MAT_s012
                                 + 1.02612 w_k1_MSFT_s012
                                 + 1.08883 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.3181 w_k1_ADM_s012
                                 + 1.22427 w_k1_AXP_s012
                                 + 0.936842 w_k1_DIS_s012
                                 + 1.06217 w_k1_FDX_s012
                                 + 1.06489 w_k1_IBM_s012 + 1.219 w_k1_INTC_s012
                                 + 0.952101 w_k1_KO_s012
                                 + 1.06741 w_k1_MAT_s012
                                 + 1.06649 w_k1_MSFT_s012
                                 + 1.05473 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.762705 w_k1_ADM_s012
                                 + 0.924875 w_k1_AXP_s012
                                 + 0.743716 w_k1_DIS_s012
                                 + 0.763412 w_k1_FDX_s012
                                 + 1.34022 w_k1_IBM_s012
                                 + 1.00832 w_k1_INTC_s012
                                 + 1.01852 w_k1_KO_s012 + 1.44579 w_k1_MAT_s012
                                 + 0.834824 w_k1_MSFT_s012
                                 + 0.959387 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.42648 w_k1_ADM_s012
                                 + 1.67011 w_k1_AXP_s012
                                 + 1.78543 w_k1_DIS_s012
                                 + 1.54847 w_k1_FDX_s012
                                 + 0.932704 w_k1_IBM_s012
                                 + 1.46757 w_k1_INTC_s012
                                 + 1.21989 w_k1_KO_s012 + 1.56929 w_k1_MAT_s012
                                 + 1.25014 w_k1_MSFT_s012
                                 + 1.46454 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.42944 w_k1_ADM_s012
                                 + 1.16037 w_k1_AXP_s012
                                 + 1.40981 w_k1_DIS_s012
                                 + 1.16365 w_k1_FDX_s012 + 1.3643 w_k1_IBM_s012
                                 + 1.40238 w_k1_INTC_s012
                                 + 1.28372 w_k1_KO_s012 + 1.21037 w_k1_MAT_s012
                                 + 1.2482 w_k1_MSFT_s012 + 1.31792 w_k1_PG_s012
                                 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.76825 w_k1_ADM_s013
                                 + 0.991073 w_k1_AXP_s013
                                 + 0.988611 w_k1_DIS_s013
                                 + 0.916301 w_k1_FDX_s013
                                 + 1.22646 w_k1_IBM_s013
                                 + 1.04126 w_k1_INTC_s013
                                 + 1.12846 w_k1_KO_s013
                                 + 0.981395 w_k1_MAT_s013
                                 + 0.862588 w_k1_MSFT_s013
                                 + 0.930022 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.25605 w_k1_ADM_s013
                                 + 1.39696 w_k1_AXP_s013
                                 + 1.60775 w_k1_DIS_s013
                                 + 1.50284 w_k1_FDX_s013
                                 + 1.17531 w_k1_IBM_s013
                                 + 1.18343 w_k1_INTC_s013
                                 + 1.41212 w_k1_KO_s013 + 1.8175 w_k1_MAT_s013
                                 + 1.39173 w_k1_MSFT_s013
                                 + 1.40674 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.18925 w_k1_ADM_s013
                                 + 1.01819 w_k1_AXP_s013 + 1.4571 w_k1_DIS_s013
                                 + 1.15998 w_k1_FDX_s013
                                 + 1.24586 w_k1_IBM_s013
                                 + 1.00701 w_k1_INTC_s013
                                 + 1.53667 w_k1_KO_s013 + 1.2232 w_k1_MAT_s013
                                 + 1.28018 w_k1_MSFT_s013
                                 + 1.15358 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.02568 w_k1_ADM_s013
                                 + 1.04383 w_k1_AXP_s013
                                 + 0.980494 w_k1_DIS_s013
                                 + 1.02659 w_k1_FDX_s013
                                 + 0.914881 w_k1_IBM_s013
                                 + 0.98187 w_k1_INTC_s013
                                 + 0.972078 w_k1_KO_s013
                                 + 1.22616 w_k1_MAT_s013 + 1.01 w_k1_MSFT_s013
                                 + 0.981059 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.952358 w_k1_ADM_s013
                                 + 0.821067 w_k1_AXP_s013
                                 + 0.622383 w_k1_DIS_s013
                                 + 0.742316 w_k1_FDX_s013
                                 + 0.718507 w_k1_IBM_s013
                                 + 0.605927 w_k1_INTC_s013
                                 + 0.693899 w_k1_KO_s013
                                 + 1.10426 w_k1_MAT_s013
                                 + 0.633515 w_k1_MSFT_s013
                                 + 0.840255 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.02846 w_k1_ADM_s013
                                 + 0.956068 w_k1_AXP_s013
                                 + 1.20041 w_k1_DIS_s013 + 1.1909 w_k1_FDX_s013
                                 + 0.807652 w_k1_IBM_s013
                                 + 0.999004 w_k1_INTC_s013
                                 + 1.0424 w_k1_KO_s013 + 0.708652 w_k1_MAT_s013
                                 + 1.01962 w_k1_MSFT_s013
                                 + 0.986711 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.750675 w_k1_ADM_s013
                                 + 0.84005 w_k1_AXP_s013
                                 + 0.800263 w_k1_DIS_s013
                                 + 0.866176 w_k1_FDX_s013
                                 + 0.749724 w_k1_IBM_s013
                                 + 0.970554 w_k1_INTC_s013
                                 + 0.954507 w_k1_KO_s013
                                 + 0.694997 w_k1_MAT_s013
                                 + 0.931881 w_k1_MSFT_s013
                                 + 0.9097 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.781786 w_k1_ADM_s013
                                 + 1.30964 w_k1_AXP_s013
                                 + 1.11831 w_k1_DIS_s013
                                 + 1.15223 w_k1_FDX_s013
                                 + 0.908525 w_k1_IBM_s013
                                 + 0.972598 w_k1_INTC_s013
                                 + 1.07752 w_k1_KO_s013
                                 + 0.985043 w_k1_MAT_s013
                                 + 0.893629 w_k1_MSFT_s013
                                 + 1.07358 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.10135 w_k1_ADM_s014
                                 + 1.38757 w_k1_AXP_s014 + 1.1587 w_k1_DIS_s014
                                 + 1.21254 w_k1_FDX_s014
                                 + 1.20761 w_k1_IBM_s014
                                 + 1.01233 w_k1_INTC_s014
                                 + 1.08766 w_k1_KO_s014 + 1.38807 w_k1_MAT_s014
                                 + 1.08719 w_k1_MSFT_s014
                                 + 1.08991 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.791215 w_k1_ADM_s014
                                 + 0.792427 w_k1_AXP_s014
                                 + 0.755886 w_k1_DIS_s014
                                 + 0.857676 w_k1_FDX_s014
                                 + 0.566697 w_k1_IBM_s014
                                 + 0.785249 w_k1_INTC_s014
                                 + 0.683835 w_k1_KO_s014
                                 + 0.651964 w_k1_MAT_s014
                                 + 0.823173 w_k1_MSFT_s014
                                 + 0.797908 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.04228 w_k1_ADM_s014
                                 + 1.19193 w_k1_AXP_s014
                                 + 1.64166 w_k1_DIS_s014
                                 + 1.40472 w_k1_FDX_s014
                                 + 1.29135 w_k1_IBM_s014
                                 + 1.08971 w_k1_INTC_s014
                                 + 1.42451 w_k1_KO_s014 + 1.07705 w_k1_MAT_s014
                                 + 1.34938 w_k1_MSFT_s014 + 1.1756 w_k1_PG_s014
                                 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.838284 w_k1_ADM_s014
                                 + 1.29207 w_k1_AXP_s014
                                 + 0.964401 w_k1_DIS_s014
                                 + 0.964161 w_k1_FDX_s014
                                 + 1.5895 w_k1_IBM_s014
                                 + 0.813783 w_k1_INTC_s014
                                 + 1.12072 w_k1_KO_s014 + 1.95059 w_k1_MAT_s014
                                 + 0.935826 w_k1_MSFT_s014
                                 + 0.984279 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.52899 w_k1_ADM_s014
                                 + 1.30763 w_k1_AXP_s014 + 1.0286 w_k1_DIS_s014
                                 + 1.27007 w_k1_FDX_s014
                                 + 0.896974 w_k1_IBM_s014
                                 + 1.01515 w_k1_INTC_s014
                                 + 1.05729 w_k1_KO_s014 + 1.01994 w_k1_MAT_s014
                                 + 1.02725 w_k1_MSFT_s014
                                 + 0.989688 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.07004 w_k1_ADM_s014
                                 + 0.969114 w_k1_AXP_s014
                                 + 0.860595 w_k1_DIS_s014
                                 + 0.881492 w_k1_FDX_s014
                                 + 1.16657 w_k1_IBM_s014
                                 + 0.896346 w_k1_INTC_s014
                                 + 1.03516 w_k1_KO_s014 + 1.03537 w_k1_MAT_s014
                                 + 0.785012 w_k1_MSFT_s014
                                 + 0.8973 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.42739 w_k1_ADM_s014
                                 + 1.64422 w_k1_AXP_s014
                                 + 1.54939 w_k1_DIS_s014
                                 + 1.32107 w_k1_FDX_s014
                                 + 1.79881 w_k1_IBM_s014
                                 + 1.31248 w_k1_INTC_s014
                                 + 1.62754 w_k1_KO_s014 + 1.8383 w_k1_MAT_s014
                                 + 1.174 w_k1_MSFT_s014 + 1.21869 w_k1_PG_s014
                                 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.971421 w_k1_ADM_s014
                                 + 1.09488 w_k1_AXP_s014
                                 + 0.878691 w_k1_DIS_s014
                                 + 1.06209 w_k1_FDX_s014
                                 + 0.848601 w_k1_IBM_s014
                                 + 1.03989 w_k1_INTC_s014
                                 + 0.961496 w_k1_KO_s014
                                 + 0.704532 w_k1_MAT_s014
                                 + 0.903263 w_k1_MSFT_s014
                                 + 0.860339 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.1353 w_k1_ADM_s015
                                 + 1.10542 w_k1_AXP_s015
                                 + 1.17553 w_k1_DIS_s015
                                 + 0.974206 w_k1_FDX_s015
                                 + 1.50215 w_k1_IBM_s015
                                 + 1.05672 w_k1_INTC_s015
                                 + 1.22909 w_k1_KO_s015 + 1.59203 w_k1_MAT_s015
                                 + 1.01842 w_k1_MSFT_s015 + 1.0547 w_k1_PG_s015
                                 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.51554 w_k1_ADM_s015
                                 + 1.55424 w_k1_AXP_s015
                                 + 1.37992 w_k1_DIS_s015
                                 + 1.24712 w_k1_FDX_s015
                                 + 1.08689 w_k1_IBM_s015
                                 + 1.06663 w_k1_INTC_s015
                                 + 0.985563 w_k1_KO_s015
                                 + 1.30333 w_k1_MAT_s015
                                 + 1.01681 w_k1_MSFT_s015
                                 + 1.00158 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.1214 w_k1_ADM_s015
                                 + 1.43839 w_k1_AXP_s015
                                 + 2.19722 w_k1_DIS_s015
                                 + 1.36438 w_k1_FDX_s015
                                 + 1.44374 w_k1_IBM_s015
                                 + 1.46368 w_k1_INTC_s015
                                 + 1.49465 w_k1_KO_s015 + 1.04881 w_k1_MAT_s015
                                 + 1.46378 w_k1_MSFT_s015 + 1.2747 w_k1_PG_s015
                                 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.19097 w_k1_ADM_s015
                                 + 1.06145 w_k1_AXP_s015
                                 + 0.922827 w_k1_DIS_s015
                                 + 0.933907 w_k1_FDX_s015
                                 + 1.06393 w_k1_IBM_s015
                                 + 0.883674 w_k1_INTC_s015
                                 + 0.949423 w_k1_KO_s015
                                 + 1.61985 w_k1_MAT_s015
                                 + 0.893281 w_k1_MSFT_s015
                                 + 0.891615 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.71036 w_k1_ADM_s015
                                 + 0.646261 w_k1_AXP_s015
                                 + 0.733767 w_k1_DIS_s015
                                 + 0.646061 w_k1_FDX_s015
                                 + 0.811956 w_k1_IBM_s015
                                 + 0.769527 w_k1_INTC_s015
                                 + 0.845294 w_k1_KO_s015
                                 + 0.922638 w_k1_MAT_s015
                                 + 0.792402 w_k1_MSFT_s015
                                 + 0.827503 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.1857 w_k1_ADM_s015
                                 + 0.830575 w_k1_AXP_s015
                                 + 1.23666 w_k1_DIS_s015
                                 + 1.09548 w_k1_FDX_s015
                                 + 0.569971 w_k1_IBM_s015
                                 + 0.848465 w_k1_INTC_s015
                                 + 0.887981 w_k1_KO_s015
                                 + 0.819949 w_k1_MAT_s015
                                 + 0.966084 w_k1_MSFT_s015
                                 + 0.871986 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.15062 w_k1_ADM_s015
                                 + 0.886255 w_k1_AXP_s015
                                 + 0.790444 w_k1_DIS_s015
                                 + 0.897384 w_k1_FDX_s015
                                 + 0.839326 w_k1_IBM_s015
                                 + 0.790896 w_k1_INTC_s015
                                 + 0.742179 w_k1_KO_s015
                                 + 0.858438 w_k1_MAT_s015
                                 + 0.812045 w_k1_MSFT_s015
                                 + 0.891617 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.987093 w_k1_ADM_s015
                                 + 1.08131 w_k1_AXP_s015
                                 + 1.48525 w_k1_DIS_s015
                                 + 1.18497 w_k1_FDX_s015
                                 + 1.04936 w_k1_IBM_s015
                                 + 0.966744 w_k1_INTC_s015
                                 + 1.19191 w_k1_KO_s015 + 1.23894 w_k1_MAT_s015
                                 + 1.27476 w_k1_MSFT_s015
                                 + 1.16979 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.957232 w_k1_ADM_s016
                                 + 0.949302 w_k1_AXP_s016
                                 + 1.00317 w_k1_DIS_s016
                                 + 0.934128 w_k1_FDX_s016
                                 + 1.23869 w_k1_IBM_s016
                                 + 0.88671 w_k1_INTC_s016
                                 + 1.10192 w_k1_KO_s016 + 1.59215 w_k1_MAT_s016
                                 + 1.1177 w_k1_MSFT_s016 + 1.20947 w_k1_PG_s016
                                 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.15557 w_k1_ADM_s016
                                 + 1.16137 w_k1_AXP_s016
                                 + 0.984632 w_k1_DIS_s016
                                 + 0.979392 w_k1_FDX_s016
                                 + 0.842029 w_k1_IBM_s016
                                 + 0.96623 w_k1_INTC_s016
                                 + 0.844351 w_k1_KO_s016
                                 + 0.888565 w_k1_MAT_s016
                                 + 1.01836 w_k1_MSFT_s016
                                 + 0.981356 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.00717 w_k1_ADM_s016
                                 + 0.999963 w_k1_AXP_s016
                                 + 1.58156 w_k1_DIS_s016
                                 + 1.19958 w_k1_FDX_s016 + 1.013 w_k1_IBM_s016
                                 + 1.3345 w_k1_INTC_s016 + 1.3855 w_k1_KO_s016
                                 + 1.05375 w_k1_MAT_s016
                                 + 1.53363 w_k1_MSFT_s016
                                 + 1.36659 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.805985 w_k1_ADM_s016
                                 + 0.58113 w_k1_AXP_s016
                                 + 0.637414 w_k1_DIS_s016
                                 + 0.774883 w_k1_FDX_s016
                                 + 0.677976 w_k1_IBM_s016
                                 + 0.920871 w_k1_INTC_s016
                                 + 0.713023 w_k1_KO_s016
                                 + 0.685407 w_k1_MAT_s016
                                 + 0.848187 w_k1_MSFT_s016
                                 + 0.823396 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.910094 w_k1_ADM_s016
                                 + 0.689992 w_k1_AXP_s016
                                 + 0.722557 w_k1_DIS_s016
                                 + 0.881323 w_k1_FDX_s016
                                 + 0.79504 w_k1_IBM_s016
                                 + 0.657775 w_k1_INTC_s016
                                 + 0.84034 w_k1_KO_s016
                                 + 0.952759 w_k1_MAT_s016
                                 + 0.849257 w_k1_MSFT_s016
                                 + 0.855917 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.0348 w_k1_ADM_s016
                                 + 0.951613 w_k1_AXP_s016
                                 + 0.864436 w_k1_DIS_s016
                                 + 0.865258 w_k1_FDX_s016
                                 + 1.10253 w_k1_IBM_s016
                                 + 0.733476 w_k1_INTC_s016
                                 + 1.11929 w_k1_KO_s016 + 1.37371 w_k1_MAT_s016
                                 + 0.94388 w_k1_MSFT_s016
                                 + 1.03115 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.94357 w_k1_ADM_s016
                                 + 0.913428 w_k1_AXP_s016
                                 + 1.12307 w_k1_DIS_s016
                                 + 1.07915 w_k1_FDX_s016
                                 + 0.796928 w_k1_IBM_s016
                                 + 0.917706 w_k1_INTC_s016
                                 + 0.949424 w_k1_KO_s016
                                 + 1.03132 w_k1_MAT_s016
                                 + 1.16313 w_k1_MSFT_s016
                                 + 0.943995 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.20342 w_k1_ADM_s016
                                 + 1.44099 w_k1_AXP_s016
                                 + 1.24047 w_k1_DIS_s016
                                 + 1.14719 w_k1_FDX_s016
                                 + 1.37631 w_k1_IBM_s016
                                 + 1.06148 w_k1_INTC_s016
                                 + 1.53398 w_k1_KO_s016 + 2.23974 w_k1_MAT_s016
                                 + 1.19014 w_k1_MSFT_s016
                                 + 1.29714 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.708668 w_k1_ADM_s017
                                 + 0.638652 w_k1_AXP_s017
                                 + 0.615675 w_k1_DIS_s017
                                 + 0.744983 w_k1_FDX_s017
                                 + 0.749227 w_k1_IBM_s017
                                 + 0.899709 w_k1_INTC_s017
                                 + 0.694731 w_k1_KO_s017
                                 + 0.627621 w_k1_MAT_s017
                                 + 0.770868 w_k1_MSFT_s017
                                 + 0.727146 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.01841 w_k1_ADM_s017
                                 + 0.863109 w_k1_AXP_s017
                                 + 1.19074 w_k1_DIS_s017
                                 + 1.11953 w_k1_FDX_s017
                                 + 0.789938 w_k1_IBM_s017
                                 + 1.00654 w_k1_INTC_s017
                                 + 0.889686 w_k1_KO_s017
                                 + 0.599979 w_k1_MAT_s017
                                 + 1.11387 w_k1_MSFT_s017
                                 + 1.08384 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.830301 w_k1_ADM_s017
                                 + 0.917787 w_k1_AXP_s017
                                 + 0.751198 w_k1_DIS_s017
                                 + 0.886977 w_k1_FDX_s017
                                 + 1.09464 w_k1_IBM_s017
                                 + 1.06898 w_k1_INTC_s017
                                 + 0.818542 w_k1_KO_s017
                                 + 0.863404 w_k1_MAT_s017
                                 + 0.839884 w_k1_MSFT_s017
                                 + 0.926532 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.08761 w_k1_ADM_s017
                                 + 1.03002 w_k1_AXP_s017
                                 + 0.868332 w_k1_DIS_s017
                                 + 1.06075 w_k1_FDX_s017
                                 + 0.742707 w_k1_IBM_s017
                                 + 1.1137 w_k1_INTC_s017
                                 + 0.688721 w_k1_KO_s017
                                 + 0.592466 w_k1_MAT_s017
                                 + 1.0444 w_k1_MSFT_s017
                                 + 0.887756 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.882539 w_k1_ADM_s017
                                 + 1.04959 w_k1_AXP_s017
                                 + 0.774772 w_k1_DIS_s017
                                 + 0.958464 w_k1_FDX_s017
                                 + 0.883571 w_k1_IBM_s017
                                 + 0.850215 w_k1_INTC_s017
                                 + 0.716575 w_k1_KO_s017
                                 + 1.02596 w_k1_MAT_s017
                                 + 0.915304 w_k1_MSFT_s017
                                 + 0.838062 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.25773 w_k1_ADM_s017
                                 + 1.48555 w_k1_AXP_s017 + 1.4951 w_k1_DIS_s017
                                 + 1.36882 w_k1_FDX_s017
                                 + 1.21248 w_k1_IBM_s017
                                 + 1.4687 w_k1_INTC_s017 + 1.09917 w_k1_KO_s017
                                 + 0.947737 w_k1_MAT_s017
                                 + 1.33387 w_k1_MSFT_s017
                                 + 1.12369 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.02551 w_k1_ADM_s017
                                 + 0.860863 w_k1_AXP_s017
                                 + 0.966701 w_k1_DIS_s017
                                 + 0.943771 w_k1_FDX_s017
                                 + 1.3503 w_k1_IBM_s017
                                 + 1.03423 w_k1_INTC_s017 + 1.0903 w_k1_KO_s017
                                 + 1.06515 w_k1_MAT_s017
                                 + 0.964652 w_k1_MSFT_s017
                                 + 1.06435 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.10903 w_k1_ADM_s017
                                 + 1.60536 w_k1_AXP_s017
                                 + 1.09893 w_k1_DIS_s017
                                 + 1.21566 w_k1_FDX_s017
                                 + 1.46497 w_k1_IBM_s017
                                 + 1.01155 w_k1_INTC_s017
                                 + 1.07195 w_k1_KO_s017 + 1.35609 w_k1_MAT_s017
                                 + 0.934254 w_k1_MSFT_s017
                                 + 1.0812 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.800275 w_k1_ADM_s100
                                 + 0.691635 w_k1_AXP_s100
                                 + 0.735768 w_k1_DIS_s100
                                 + 0.771269 w_k1_FDX_s100
                                 + 0.783448 w_k1_IBM_s100
                                 + 1.00092 w_k1_INTC_s100
                                 + 0.814083 w_k1_KO_s100
                                 + 0.664038 w_k1_MAT_s100
                                 + 0.917282 w_k1_MSFT_s100
                                 + 0.804981 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.708775 w_k1_ADM_s100
                                 + 0.748706 w_k1_AXP_s100
                                 + 0.652597 w_k1_DIS_s100
                                 + 0.715277 w_k1_FDX_s100
                                 + 1.1876 w_k1_IBM_s100
                                 + 1.03022 w_k1_INTC_s100
                                 + 0.964486 w_k1_KO_s100 + 1.1545 w_k1_MAT_s100
                                 + 0.811797 w_k1_MSFT_s100
                                 + 0.91737 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.820041 w_k1_ADM_s100
                                 + 1.04606 w_k1_AXP_s100
                                 + 0.938303 w_k1_DIS_s100
                                 + 0.887744 w_k1_FDX_s100
                                 + 1.09155 w_k1_IBM_s100
                                 + 1.00593 w_k1_INTC_s100
                                 + 1.13269 w_k1_KO_s100 + 1.63652 w_k1_MAT_s100
                                 + 0.976818 w_k1_MSFT_s100
                                 + 0.984626 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.26265 w_k1_ADM_s100
                                 + 1.28436 w_k1_AXP_s100
                                 + 1.20661 w_k1_DIS_s100
                                 + 1.07761 w_k1_FDX_s100
                                 + 0.997331 w_k1_IBM_s100
                                 + 1.24926 w_k1_INTC_s100
                                 + 1.22039 w_k1_KO_s100
                                 + 0.920933 w_k1_MAT_s100
                                 + 1.0756 w_k1_MSFT_s100
                                 + 0.973223 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.29684 w_k1_ADM_s100
                                 + 1.74062 w_k1_AXP_s100
                                 + 1.81867 w_k1_DIS_s100
                                 + 1.46425 w_k1_FDX_s100
                                 + 1.39705 w_k1_IBM_s100
                                 + 1.66731 w_k1_INTC_s100
                                 + 1.60306 w_k1_KO_s100 + 1.43834 w_k1_MAT_s100
                                 + 1.4051 w_k1_MSFT_s100 + 1.02145 w_k1_PG_s100
                                 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.16863 w_k1_ADM_s100
                                 + 1.349 w_k1_AXP_s100 + 1.45498 w_k1_DIS_s100
                                 + 1.05303 w_k1_FDX_s100
                                 + 1.62258 w_k1_IBM_s100
                                 + 1.44028 w_k1_INTC_s100
                                 + 1.66725 w_k1_KO_s100 + 1.77137 w_k1_MAT_s100
                                 + 1.24709 w_k1_MSFT_s100
                                 + 1.26091 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.936102 w_k1_ADM_s100
                                 + 0.941077 w_k1_AXP_s100
                                 + 1.15575 w_k1_DIS_s100
                                 + 0.879923 w_k1_FDX_s100
                                 + 1.10278 w_k1_IBM_s100
                                 + 1.16395 w_k1_INTC_s100
                                 + 1.12434 w_k1_KO_s100
                                 + 0.783454 w_k1_MAT_s100
                                 + 1.07657 w_k1_MSFT_s100
                                 + 1.03863 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.05911 w_k1_ADM_s100
                                 + 0.865512 w_k1_AXP_s100
                                 + 0.952271 w_k1_DIS_s100
                                 + 1.00949 w_k1_FDX_s100
                                 + 0.647609 w_k1_IBM_s100
                                 + 1.08539 w_k1_INTC_s100
                                 + 0.873501 w_k1_KO_s100
                                 + 0.690144 w_k1_MAT_s100
                                 + 0.921785 w_k1_MSFT_s100
                                 + 0.86624 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.902562 w_k1_ADM_s101
                                 + 0.818764 w_k1_AXP_s101
                                 + 1.10784 w_k1_DIS_s101
                                 + 1.06239 w_k1_FDX_s101
                                 + 0.802083 w_k1_IBM_s101
                                 + 0.919121 w_k1_INTC_s101
                                 + 0.966745 w_k1_KO_s101
                                 + 0.982991 w_k1_MAT_s101
                                 + 0.97823 w_k1_MSFT_s101
                                 + 0.955642 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.20705 w_k1_ADM_s101
                                 + 1.40657 w_k1_AXP_s101
                                 + 1.66652 w_k1_DIS_s101 + 1.5917 w_k1_FDX_s101
                                 + 1.1667 w_k1_IBM_s101
                                 + 1.24868 w_k1_INTC_s101
                                 + 1.22784 w_k1_KO_s101 + 1.28661 w_k1_MAT_s101
                                 + 1.46173 w_k1_MSFT_s101
                                 + 1.18619 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.35496 w_k1_ADM_s101
                                 + 1.1462 w_k1_AXP_s101 + 1.2255 w_k1_DIS_s101
                                 + 1.26287 w_k1_FDX_s101
                                 + 0.800031 w_k1_IBM_s101
                                 + 1.18107 w_k1_INTC_s101
                                 + 1.10824 w_k1_KO_s101
                                 + 0.804069 w_k1_MAT_s101
                                 + 1.25003 w_k1_MSFT_s101
                                 + 1.07921 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.858599 w_k1_ADM_s101
                                 + 0.654984 w_k1_AXP_s101
                                 + 0.665518 w_k1_DIS_s101
                                 + 0.814089 w_k1_FDX_s101
                                 + 0.846216 w_k1_IBM_s101
                                 + 0.856606 w_k1_INTC_s101
                                 + 0.891095 w_k1_KO_s101
                                 + 0.652695 w_k1_MAT_s101
                                 + 0.82289 w_k1_MSFT_s101
                                 + 0.85443 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.904147 w_k1_ADM_s101
                                 + 0.788684 w_k1_AXP_s101
                                 + 0.836333 w_k1_DIS_s101
                                 + 1.00478 w_k1_FDX_s101
                                 + 0.698931 w_k1_IBM_s101
                                 + 0.815871 w_k1_INTC_s101
                                 + 0.862037 w_k1_KO_s101
                                 + 0.72497 w_k1_MAT_s101
                                 + 0.958774 w_k1_MSFT_s101
                                 + 0.901745 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.916316 w_k1_ADM_s101
                                 + 1.20569 w_k1_AXP_s101
                                 + 1.37781 w_k1_DIS_s101
                                 + 1.09593 w_k1_FDX_s101
                                 + 1.62705 w_k1_IBM_s101
                                 + 1.04073 w_k1_INTC_s101
                                 + 1.25882 w_k1_KO_s101 + 1.79144 w_k1_MAT_s101
                                 + 1.22941 w_k1_MSFT_s101
                                 + 1.07763 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.04072 w_k1_ADM_s101
                                 + 1.18452 w_k1_AXP_s101
                                 + 1.25709 w_k1_DIS_s101
                                 + 1.12319 w_k1_FDX_s101
                                 + 1.18497 w_k1_IBM_s101
                                 + 1.25266 w_k1_INTC_s101
                                 + 1.27086 w_k1_KO_s101 + 1.27336 w_k1_MAT_s101
                                 + 1.13264 w_k1_MSFT_s101
                                 + 1.10844 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.819107 w_k1_ADM_s101
                                 + 0.774838 w_k1_AXP_s101
                                 + 0.88618 w_k1_DIS_s101
                                 + 0.874466 w_k1_FDX_s101
                                 + 1.1933 w_k1_IBM_s101
                                 + 0.984867 w_k1_INTC_s101
                                 + 1.03191 w_k1_KO_s101
                                 + 0.957025 w_k1_MAT_s101
                                 + 0.922022 w_k1_MSFT_s101
                                 + 0.964741 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.763821 w_k1_ADM_s102
                                 + 0.686423 w_k1_AXP_s102
                                 + 0.66493 w_k1_DIS_s102
                                 + 0.745154 w_k1_FDX_s102
                                 + 0.671178 w_k1_IBM_s102
                                 + 0.804639 w_k1_INTC_s102
                                 + 0.862138 w_k1_KO_s102
                                 + 0.699332 w_k1_MAT_s102
                                 + 0.915472 w_k1_MSFT_s102
                                 + 0.814692 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.00058 w_k1_ADM_s102
                                 + 0.90286 w_k1_AXP_s102
                                 + 0.96975 w_k1_DIS_s102
                                 + 1.09849 w_k1_FDX_s102
                                 + 0.683198 w_k1_IBM_s102
                                 + 1.11568 w_k1_INTC_s102
                                 + 0.945972 w_k1_KO_s102
                                 + 0.845758 w_k1_MAT_s102
                                 + 1.15877 w_k1_MSFT_s102
                                 + 0.96141 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.825575 w_k1_ADM_s102
                                 + 0.737291 w_k1_AXP_s102
                                 + 0.813043 w_k1_DIS_s102
                                 + 0.864251 w_k1_FDX_s102
                                 + 0.890761 w_k1_IBM_s102
                                 + 0.960582 w_k1_INTC_s102
                                 + 1.03722 w_k1_KO_s102
                                 + 0.739188 w_k1_MAT_s102
                                 + 1.02919 w_k1_MSFT_s102
                                 + 0.853377 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.916212 w_k1_ADM_s102
                                 + 1.08851 w_k1_AXP_s102
                                 + 1.17757 w_k1_DIS_s102
                                 + 1.14149 w_k1_FDX_s102 + 1.0527 w_k1_IBM_s102
                                 + 1.14635 w_k1_INTC_s102
                                 + 1.27117 w_k1_KO_s102 + 1.36971 w_k1_MAT_s102
                                 + 1.21596 w_k1_MSFT_s102
                                 + 1.00882 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.04773 w_k1_ADM_s102
                                 + 1.02845 w_k1_AXP_s102
                                 + 0.858081 w_k1_DIS_s102
                                 + 0.978463 w_k1_FDX_s102
                                 + 0.714526 w_k1_IBM_s102
                                 + 0.794468 w_k1_INTC_s102
                                 + 0.907032 w_k1_KO_s102
                                 + 0.783175 w_k1_MAT_s102
                                 + 0.955628 w_k1_MSFT_s102
                                 + 0.966054 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.770023 w_k1_ADM_s102
                                 + 0.969263 w_k1_AXP_s102
                                 + 0.882107 w_k1_DIS_s102
                                 + 0.861819 w_k1_FDX_s102
                                 + 1.16653 w_k1_IBM_s102
                                 + 0.889506 w_k1_INTC_s102
                                 + 0.934454 w_k1_KO_s102
                                 + 1.38553 w_k1_MAT_s102
                                 + 0.958618 w_k1_MSFT_s102
                                 + 0.797725 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.22375 w_k1_ADM_s102
                                 + 1.24305 w_k1_AXP_s102
                                 + 1.45972 w_k1_DIS_s102 + 1.1401 w_k1_FDX_s102
                                 + 1.58847 w_k1_IBM_s102
                                 + 1.38726 w_k1_INTC_s102
                                 + 1.72526 w_k1_KO_s102 + 1.71793 w_k1_MAT_s102
                                 + 1.37892 w_k1_MSFT_s102
                                 + 1.28241 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.850071 w_k1_ADM_s102
                                 + 1.03529 w_k1_AXP_s102
                                 + 0.924283 w_k1_DIS_s102
                                 + 0.956071 w_k1_FDX_s102
                                 + 0.835585 w_k1_IBM_s102
                                 + 1.06651 w_k1_INTC_s102
                                 + 0.969805 w_k1_KO_s102
                                 + 1.05816 w_k1_MAT_s102
                                 + 1.11338 w_k1_MSFT_s102
                                 + 0.96493 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.08141 w_k1_ADM_s103
                                 + 1.19184 w_k1_AXP_s103
                                 + 1.17988 w_k1_DIS_s103 + 1.1461 w_k1_FDX_s103
                                 + 1.47576 w_k1_IBM_s103
                                 + 1.37819 w_k1_INTC_s103
                                 + 1.40361 w_k1_KO_s103 + 1.79245 w_k1_MAT_s103
                                 + 1.27483 w_k1_MSFT_s103
                                 + 1.24835 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.890615 w_k1_ADM_s103
                                 + 1.0226 w_k1_AXP_s103
                                 + 0.946793 w_k1_DIS_s103
                                 + 0.883768 w_k1_FDX_s103
                                 + 1.46235 w_k1_IBM_s103
                                 + 1.03758 w_k1_INTC_s103
                                 + 1.23988 w_k1_KO_s103 + 1.50409 w_k1_MAT_s103
                                 + 0.984513 w_k1_MSFT_s103
                                 + 1.10258 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.75368 w_k1_ADM_s103
                                 + 0.802222 w_k1_AXP_s103
                                 + 0.74097 w_k1_DIS_s103
                                 + 0.771007 w_k1_FDX_s103
                                 + 1.18202 w_k1_IBM_s103
                                 + 1.09849 w_k1_INTC_s103
                                 + 1.15992 w_k1_KO_s103 + 1.00087 w_k1_MAT_s103
                                 + 0.991379 w_k1_MSFT_s103
                                 + 0.929045 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.823126 w_k1_ADM_s103
                                 + 0.686358 w_k1_AXP_s103
                                 + 0.61103 w_k1_DIS_s103
                                 + 0.812799 w_k1_FDX_s103
                                 + 0.727317 w_k1_IBM_s103
                                 + 0.92583 w_k1_INTC_s103
                                 + 0.840568 w_k1_KO_s103
                                 + 0.749657 w_k1_MAT_s103
                                 + 0.932123 w_k1_MSFT_s103
                                 + 0.8597 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.12169 w_k1_ADM_s103
                                 + 1.36924 w_k1_AXP_s103
                                 + 1.86833 w_k1_DIS_s103
                                 + 1.29995 w_k1_FDX_s103
                                 + 1.82539 w_k1_IBM_s103
                                 + 1.54731 w_k1_INTC_s103
                                 + 1.71512 w_k1_KO_s103 + 1.88752 w_k1_MAT_s103
                                 + 1.45937 w_k1_MSFT_s103
                                 + 1.28761 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.01584 w_k1_ADM_s103
                                 + 1.07494 w_k1_AXP_s103 + 1.4697 w_k1_DIS_s103
                                 + 1.26981 w_k1_FDX_s103
                                 + 0.834613 w_k1_IBM_s103
                                 + 1.2779 w_k1_INTC_s103 + 1.13692 w_k1_KO_s103
                                 + 0.93341 w_k1_MAT_s103
                                 + 1.41087 w_k1_MSFT_s103
                                 + 1.14298 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.898359 w_k1_ADM_s103
                                 + 0.833926 w_k1_AXP_s103
                                 + 0.997663 w_k1_DIS_s103
                                 + 1.01847 w_k1_FDX_s103
                                 + 0.858192 w_k1_IBM_s103
                                 + 1.09858 w_k1_INTC_s103
                                 + 0.971418 w_k1_KO_s103
                                 + 0.778103 w_k1_MAT_s103
                                 + 1.13582 w_k1_MSFT_s103
                                 + 0.974734 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.30622 w_k1_ADM_s103
                                 + 1.44164 w_k1_AXP_s103
                                 + 1.27821 w_k1_DIS_s103
                                 + 1.41335 w_k1_FDX_s103
                                 + 0.996717 w_k1_IBM_s103
                                 + 1.20205 w_k1_INTC_s103
                                 + 1.38698 w_k1_KO_s103 + 1.27314 w_k1_MAT_s103
                                 + 1.37442 w_k1_MSFT_s103
                                 + 1.16456 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.954489 w_k1_ADM_s104
                                 + 1.09734 w_k1_AXP_s104
                                 + 1.29035 w_k1_DIS_s104
                                 + 1.11038 w_k1_FDX_s104
                                 + 1.46975 w_k1_IBM_s104
                                 + 1.29569 w_k1_INTC_s104
                                 + 1.36516 w_k1_KO_s104 + 1.23597 w_k1_MAT_s104
                                 + 1.10047 w_k1_MSFT_s104
                                 + 1.17136 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.998265 w_k1_ADM_s104
                                 + 0.915131 w_k1_AXP_s104
                                 + 0.697635 w_k1_DIS_s104
                                 + 0.899995 w_k1_FDX_s104
                                 + 0.845334 w_k1_IBM_s104
                                 + 0.800572 w_k1_INTC_s104
                                 + 0.777441 w_k1_KO_s104
                                 + 0.613004 w_k1_MAT_s104
                                 + 0.807016 w_k1_MSFT_s104
                                 + 0.948549 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.671166 w_k1_ADM_s104
                                 + 0.641586 w_k1_AXP_s104
                                 + 0.81571 w_k1_DIS_s104
                                 + 0.804098 w_k1_FDX_s104
                                 + 0.960937 w_k1_IBM_s104
                                 + 0.705679 w_k1_INTC_s104
                                 + 1.01542 w_k1_KO_s104
                                 + 0.915267 w_k1_MAT_s104
                                 + 0.895992 w_k1_MSFT_s104
                                 + 0.922025 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.12866 w_k1_ADM_s104
                                 + 0.95128 w_k1_AXP_s104
                                 + 0.981458 w_k1_DIS_s104
                                 + 1.04753 w_k1_FDX_s104
                                 + 0.880436 w_k1_IBM_s104
                                 + 0.911233 w_k1_INTC_s104
                                 + 1.01426 w_k1_KO_s104 + 0.73136 w_k1_MAT_s104
                                 + 0.921901 w_k1_MSFT_s104
                                 + 1.11114 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.919944 w_k1_ADM_s104
                                 + 1.03749 w_k1_AXP_s104
                                 + 1.31341 w_k1_DIS_s104
                                 + 1.01959 w_k1_FDX_s104
                                 + 1.11792 w_k1_IBM_s104
                                 + 1.12225 w_k1_INTC_s104
                                 + 1.19967 w_k1_KO_s104 + 0.95087 w_k1_MAT_s104
                                 + 1.11278 w_k1_MSFT_s104
                                 + 1.21978 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.14023 w_k1_ADM_s104
                                 + 1.42216 w_k1_AXP_s104
                                 + 1.46331 w_k1_DIS_s104
                                 + 1.09879 w_k1_FDX_s104 + 1.2821 w_k1_IBM_s104
                                 + 1.19095 w_k1_INTC_s104
                                 + 1.44084 w_k1_KO_s104 + 1.28422 w_k1_MAT_s104
                                 + 1.25789 w_k1_MSFT_s104
                                 + 1.30338 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.38126 w_k1_ADM_s104
                                 + 1.71778 w_k1_AXP_s104
                                 + 1.61951 w_k1_DIS_s104
                                 + 1.37806 w_k1_FDX_s104
                                 + 1.21393 w_k1_IBM_s104
                                 + 1.34197 w_k1_INTC_s104
                                 + 1.42185 w_k1_KO_s104 + 1.52086 w_k1_MAT_s104
                                 + 1.44229 w_k1_MSFT_s104
                                 + 1.49348 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.03152 w_k1_ADM_s104
                                 + 1.34142 w_k1_AXP_s104
                                 + 0.995131 w_k1_DIS_s104
                                 + 1.13857 w_k1_FDX_s104
                                 + 0.981201 w_k1_IBM_s104
                                 + 1.22822 w_k1_INTC_s104
                                 + 1.04279 w_k1_KO_s104 + 1.08242 w_k1_MAT_s104
                                 + 0.911766 w_k1_MSFT_s104
                                 + 1.23239 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.994753 w_k1_ADM_s105
                                 + 1.09729 w_k1_AXP_s105
                                 + 1.18202 w_k1_DIS_s105
                                 + 1.15574 w_k1_FDX_s105
                                 + 0.74619 w_k1_IBM_s105
                                 + 1.05885 w_k1_INTC_s105
                                 + 0.949387 w_k1_KO_s105
                                 + 0.707971 w_k1_MAT_s105
                                 + 1.15412 w_k1_MSFT_s105
                                 + 0.986045 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.11837 w_k1_ADM_s105
                                 + 0.890803 w_k1_AXP_s105
                                 + 0.814422 w_k1_DIS_s105
                                 + 0.898308 w_k1_FDX_s105
                                 + 0.612213 w_k1_IBM_s105
                                 + 0.841029 w_k1_INTC_s105
                                 + 0.706537 w_k1_KO_s105
                                 + 0.542073 w_k1_MAT_s105
                                 + 0.91527 w_k1_MSFT_s105
                                 + 0.813707 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.18015 w_k1_ADM_s105
                                 + 1.334 w_k1_AXP_s105 + 1.5803 w_k1_DIS_s105
                                 + 1.35369 w_k1_FDX_s105 + 1.562 w_k1_IBM_s105
                                 + 1.40997 w_k1_INTC_s105
                                 + 1.55348 w_k1_KO_s105 + 2.28372 w_k1_MAT_s105
                                 + 1.39564 w_k1_MSFT_s105
                                 + 1.20335 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.732462 w_k1_ADM_s105
                                 + 0.499714 w_k1_AXP_s105
                                 + 0.612214 w_k1_DIS_s105
                                 + 0.69873 w_k1_FDX_s105
                                 + 0.76494 w_k1_IBM_s105
                                 + 0.743723 w_k1_INTC_s105
                                 + 0.760055 w_k1_KO_s105
                                 + 0.658298 w_k1_MAT_s105
                                 + 0.773548 w_k1_MSFT_s105
                                 + 0.849016 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.03253 w_k1_ADM_s105
                                 + 1.04295 w_k1_AXP_s105
                                 + 1.03366 w_k1_DIS_s105
                                 + 0.966292 w_k1_FDX_s105
                                 + 0.984692 w_k1_IBM_s105
                                 + 1.04086 w_k1_INTC_s105
                                 + 1.02882 w_k1_KO_s105 + 1.11276 w_k1_MAT_s105
                                 + 0.988842 w_k1_MSFT_s105
                                 + 1.01786 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.00032 w_k1_ADM_s105
                                 + 0.898576 w_k1_AXP_s105
                                 + 0.871553 w_k1_DIS_s105
                                 + 0.811584 w_k1_FDX_s105
                                 + 1.30073 w_k1_IBM_s105
                                 + 0.961293 w_k1_INTC_s105
                                 + 1.07744 w_k1_KO_s105 + 1.05793 w_k1_MAT_s105
                                 + 0.886213 w_k1_MSFT_s105
                                 + 0.883804 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.27518 w_k1_ADM_s105
                                 + 1.28767 w_k1_AXP_s105
                                 + 1.52376 w_k1_DIS_s105
                                 + 1.39533 w_k1_FDX_s105
                                 + 0.882284 w_k1_IBM_s105
                                 + 1.22473 w_k1_INTC_s105
                                 + 1.19265 w_k1_KO_s105 + 1.31277 w_k1_MAT_s105
                                 + 1.46733 w_k1_MSFT_s105
                                 + 1.29883 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.857808 w_k1_ADM_s105
                                 + 1.48404 w_k1_AXP_s105
                                 + 1.20978 w_k1_DIS_s105
                                 + 1.08499 w_k1_FDX_s105
                                 + 1.52671 w_k1_IBM_s105
                                 + 1.37808 w_k1_INTC_s105 + 1.4167 w_k1_KO_s105
                                 + 2.02714 w_k1_MAT_s105
                                 + 1.02637 w_k1_MSFT_s105
                                 + 1.12251 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.892917 w_k1_ADM_s106
                                 + 0.984482 w_k1_AXP_s106
                                 + 0.745535 w_k1_DIS_s106
                                 + 0.917136 w_k1_FDX_s106
                                 + 0.898346 w_k1_IBM_s106
                                 + 0.913055 w_k1_INTC_s106
                                 + 0.913825 w_k1_KO_s106
                                 + 0.924113 w_k1_MAT_s106
                                 + 0.964793 w_k1_MSFT_s106
                                 + 0.852728 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.802676 w_k1_ADM_s106
                                 + 0.819536 w_k1_AXP_s106
                                 + 0.947137 w_k1_DIS_s106
                                 + 0.916285 w_k1_FDX_s106
                                 + 0.7772 w_k1_IBM_s106
                                 + 0.983065 w_k1_INTC_s106
                                 + 0.94707 w_k1_KO_s106
                                 + 0.952521 w_k1_MAT_s106
                                 + 1.14736 w_k1_MSFT_s106
                                 + 0.944061 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.675476 w_k1_ADM_s106
                                 + 0.64413 w_k1_AXP_s106
                                 + 0.634137 w_k1_DIS_s106
                                 + 0.724821 w_k1_FDX_s106
                                 + 0.849927 w_k1_IBM_s106
                                 + 0.70488 w_k1_INTC_s106
                                 + 0.846812 w_k1_KO_s106
                                 + 0.920536 w_k1_MAT_s106
                                 + 0.80733 w_k1_MSFT_s106
                                 + 0.859445 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.05019 w_k1_ADM_s106
                                 + 1.42328 w_k1_AXP_s106
                                 + 1.12107 w_k1_DIS_s106
                                 + 1.13627 w_k1_FDX_s106
                                 + 1.05273 w_k1_IBM_s106
                                 + 1.07884 w_k1_INTC_s106
                                 + 1.14701 w_k1_KO_s106 + 1.40172 w_k1_MAT_s106
                                 + 1.22843 w_k1_MSFT_s106
                                 + 1.12896 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.852006 w_k1_ADM_s106
                                 + 0.630857 w_k1_AXP_s106
                                 + 0.889298 w_k1_DIS_s106
                                 + 0.753268 w_k1_FDX_s106
                                 + 1.02335 w_k1_IBM_s106
                                 + 0.744783 w_k1_INTC_s106
                                 + 1.10044 w_k1_KO_s106
                                 + 0.825431 w_k1_MAT_s106
                                 + 0.925238 w_k1_MSFT_s106
                                 + 0.993748 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.787219 w_k1_ADM_s106
                                 + 0.91228 w_k1_AXP_s106
                                 + 1.01762 w_k1_DIS_s106
                                 + 0.881094 w_k1_FDX_s106
                                 + 1.13788 w_k1_IBM_s106
                                 + 1.0222 w_k1_INTC_s106 + 1.18295 w_k1_KO_s106
                                 + 1.30537 w_k1_MAT_s106
                                 + 1.13486 w_k1_MSFT_s106
                                 + 0.952601 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.96556 w_k1_ADM_s106
                                 + 1.00958 w_k1_AXP_s106
                                 + 1.44157 w_k1_DIS_s106
                                 + 1.17239 w_k1_FDX_s106
                                 + 1.05449 w_k1_IBM_s106
                                 + 1.03853 w_k1_INTC_s106
                                 + 1.27582 w_k1_KO_s106 + 1.00038 w_k1_MAT_s106
                                 + 1.19716 w_k1_MSFT_s106
                                 + 1.16287 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.08711 w_k1_ADM_s106
                                 + 1.0313 w_k1_AXP_s106 + 1.36278 w_k1_DIS_s106
                                 + 1.03406 w_k1_FDX_s106
                                 + 1.35851 w_k1_IBM_s106
                                 + 1.02404 w_k1_INTC_s106
                                 + 1.36202 w_k1_KO_s106 + 1.34978 w_k1_MAT_s106
                                 + 1.15833 w_k1_MSFT_s106
                                 + 1.09457 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.623495 w_k1_ADM_s107
                                 + 0.961922 w_k1_AXP_s107
                                 + 0.71137 w_k1_DIS_s107
                                 + 0.863983 w_k1_FDX_s107
                                 + 0.710129 w_k1_IBM_s107
                                 + 0.812686 w_k1_INTC_s107
                                 + 0.820841 w_k1_KO_s107 + 1.0142 w_k1_MAT_s107
                                 + 0.814847 w_k1_MSFT_s107
                                 + 0.832876 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.24191 w_k1_ADM_s107
                                 + 1.49328 w_k1_AXP_s107
                                 + 1.66797 w_k1_DIS_s107
                                 + 1.22241 w_k1_FDX_s107 + 1.521 w_k1_IBM_s107
                                 + 1.2382 w_k1_INTC_s107 + 1.58894 w_k1_KO_s107
                                 + 2.06928 w_k1_MAT_s107
                                 + 1.25469 w_k1_MSFT_s107
                                 + 1.29452 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.917756 w_k1_ADM_s107
                                 + 1.0999 w_k1_AXP_s107
                                 + 0.874572 w_k1_DIS_s107
                                 + 0.857709 w_k1_FDX_s107
                                 + 1.23704 w_k1_IBM_s107
                                 + 1.16946 w_k1_INTC_s107
                                 + 1.07331 w_k1_KO_s107 + 1.40735 w_k1_MAT_s107
                                 + 0.962812 w_k1_MSFT_s107
                                 + 0.892909 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.29785 w_k1_ADM_s107
                                 + 1.49573 w_k1_AXP_s107
                                 + 0.898882 w_k1_DIS_s107
                                 + 1.04356 w_k1_FDX_s107
                                 + 1.04508 w_k1_IBM_s107
                                 + 1.04016 w_k1_INTC_s107
                                 + 1.02685 w_k1_KO_s107 + 1.6901 w_k1_MAT_s107
                                 + 1.07315 w_k1_MSFT_s107
                                 + 1.05243 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.09816 w_k1_ADM_s107
                                 + 1.19978 w_k1_AXP_s107
                                 + 1.28909 w_k1_DIS_s107
                                 + 1.06003 w_k1_FDX_s107
                                 + 1.22506 w_k1_IBM_s107
                                 + 1.00247 w_k1_INTC_s107
                                 + 1.03043 w_k1_KO_s107 + 1.25127 w_k1_MAT_s107
                                 + 1.03105 w_k1_MSFT_s107
                                 + 0.989953 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.2604 w_k1_ADM_s107
                                 + 1.46245 w_k1_AXP_s107
                                 + 1.37745 w_k1_DIS_s107
                                 + 1.33229 w_k1_FDX_s107
                                 + 0.890736 w_k1_IBM_s107
                                 + 1.10988 w_k1_INTC_s107
                                 + 1.12045 w_k1_KO_s107 + 1.19297 w_k1_MAT_s107
                                 + 1.1749 w_k1_MSFT_s107 + 1.0439 w_k1_PG_s107
                                 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.05618 w_k1_ADM_s107
                                 + 0.959028 w_k1_AXP_s107
                                 + 1.05219 w_k1_DIS_s107
                                 + 0.924126 w_k1_FDX_s107
                                 + 0.838543 w_k1_IBM_s107
                                 + 0.899498 w_k1_INTC_s107
                                 + 1.02663 w_k1_KO_s107
                                 + 0.902481 w_k1_MAT_s107
                                 + 0.92293 w_k1_MSFT_s107
                                 + 0.897628 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.20649 w_k1_ADM_s107
                                 + 1.35701 w_k1_AXP_s107
                                 + 1.41495 w_k1_DIS_s107
                                 + 1.17639 w_k1_FDX_s107 + 1.0562 w_k1_IBM_s107
                                 + 1.30444 w_k1_INTC_s107
                                 + 1.13164 w_k1_KO_s107 + 1.04772 w_k1_MAT_s107
                                 + 1.31307 w_k1_MSFT_s107
                                 + 1.10945 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.662928 w_k1_ADM_s110
                                 + 0.742157 w_k1_AXP_s110
                                 + 0.636191 w_k1_DIS_s110
                                 + 0.746427 w_k1_FDX_s110
                                 + 0.736113 w_k1_IBM_s110
                                 + 0.985222 w_k1_INTC_s110
                                 + 0.769067 w_k1_KO_s110
                                 + 0.722254 w_k1_MAT_s110
                                 + 0.80714 w_k1_MSFT_s110
                                 + 0.72516 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.80241 w_k1_ADM_s110
                                 + 0.806162 w_k1_AXP_s110
                                 + 0.781545 w_k1_DIS_s110
                                 + 0.960982 w_k1_FDX_s110
                                 + 0.562961 w_k1_IBM_s110
                                 + 1.04396 w_k1_INTC_s110
                                 + 0.713402 w_k1_KO_s110
                                 + 0.446275 w_k1_MAT_s110
                                 + 1.00242 w_k1_MSFT_s110
                                 + 0.815606 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.909277 w_k1_ADM_s110
                                 + 1.30335 w_k1_AXP_s110
                                 + 1.31705 w_k1_DIS_s110
                                 + 1.09784 w_k1_FDX_s110 + 1.7049 w_k1_IBM_s110
                                 + 1.11313 w_k1_INTC_s110
                                 + 1.39861 w_k1_KO_s110 + 1.82701 w_k1_MAT_s110
                                 + 1.06728 w_k1_MSFT_s110
                                 + 1.07416 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.03533 w_k1_ADM_s110
                                 + 1.20217 w_k1_AXP_s110 + 1.5561 w_k1_DIS_s110
                                 + 1.15033 w_k1_FDX_s110
                                 + 1.28821 w_k1_IBM_s110
                                 + 1.31084 w_k1_INTC_s110
                                 + 1.33574 w_k1_KO_s110 + 1.03679 w_k1_MAT_s110
                                 + 1.15201 w_k1_MSFT_s110
                                 + 1.02838 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.778282 w_k1_ADM_s110
                                 + 0.809055 w_k1_AXP_s110
                                 + 0.86553 w_k1_DIS_s110
                                 + 0.905852 w_k1_FDX_s110
                                 + 0.966596 w_k1_IBM_s110
                                 + 1.02249 w_k1_INTC_s110
                                 + 0.982904 w_k1_KO_s110
                                 + 1.00566 w_k1_MAT_s110
                                 + 0.94683 w_k1_MSFT_s110
                                 + 0.876762 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.05361 w_k1_ADM_s110
                                 + 1.01175 w_k1_AXP_s110
                                 + 1.24048 w_k1_DIS_s110
                                 + 1.24218 w_k1_FDX_s110
                                 + 0.702775 w_k1_IBM_s110
                                 + 1.16296 w_k1_INTC_s110
                                 + 0.932047 w_k1_KO_s110
                                 + 0.618569 w_k1_MAT_s110
                                 + 1.25554 w_k1_MSFT_s110
                                 + 0.980499 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.844565 w_k1_ADM_s110
                                 + 0.991437 w_k1_AXP_s110
                                 + 0.851663 w_k1_DIS_s110
                                 + 0.838376 w_k1_FDX_s110
                                 + 1.26202 w_k1_IBM_s110
                                 + 1.12067 w_k1_INTC_s110
                                 + 1.03362 w_k1_KO_s110
                                 + 0.952128 w_k1_MAT_s110
                                 + 0.851911 w_k1_MSFT_s110
                                 + 0.885984 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.06515 w_k1_ADM_s110
                                 + 1.26468 w_k1_AXP_s110
                                 + 1.52379 w_k1_DIS_s110
                                 + 1.30201 w_k1_FDX_s110
                                 + 0.89637 w_k1_IBM_s110
                                 + 1.24368 w_k1_INTC_s110
                                 + 1.23742 w_k1_KO_s110 + 1.10146 w_k1_MAT_s110
                                 + 1.30495 w_k1_MSFT_s110 + 1.0574 w_k1_PG_s110
                                 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.810681 w_k1_ADM_s111
                                 + 0.83075 w_k1_AXP_s111
                                 + 0.725832 w_k1_DIS_s111
                                 + 0.850136 w_k1_FDX_s111
                                 + 0.813261 w_k1_IBM_s111
                                 + 0.79437 w_k1_INTC_s111
                                 + 0.882932 w_k1_KO_s111
                                 + 0.692103 w_k1_MAT_s111
                                 + 0.928666 w_k1_MSFT_s111
                                 + 0.897378 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.798744 w_k1_ADM_s111
                                 + 1.09431 w_k1_AXP_s111
                                 + 1.06034 w_k1_DIS_s111
                                 + 0.908822 w_k1_FDX_s111
                                 + 1.66943 w_k1_IBM_s111
                                 + 0.818986 w_k1_INTC_s111
                                 + 1.17311 w_k1_KO_s111 + 1.39406 w_k1_MAT_s111
                                 + 0.919004 w_k1_MSFT_s111
                                 + 1.00671 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.29984 w_k1_ADM_s111
                                 + 1.52555 w_k1_AXP_s111
                                 + 1.25499 w_k1_DIS_s111 + 1.2895 w_k1_FDX_s111
                                 + 0.878901 w_k1_IBM_s111
                                 + 0.972444 w_k1_INTC_s111
                                 + 1.11825 w_k1_KO_s111 + 1.6128 w_k1_MAT_s111
                                 + 1.20497 w_k1_MSFT_s111
                                 + 1.06378 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.05224 w_k1_ADM_s111
                                 + 1.19291 w_k1_AXP_s111
                                 + 1.57422 w_k1_DIS_s111
                                 + 1.33279 w_k1_FDX_s111
                                 + 1.06453 w_k1_IBM_s111
                                 + 1.15448 w_k1_INTC_s111 + 1.3874 w_k1_KO_s111
                                 + 0.988528 w_k1_MAT_s111
                                 + 1.30372 w_k1_MSFT_s111
                                 + 1.25321 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.05103 w_k1_ADM_s111
                                 + 1.23662 w_k1_AXP_s111
                                 + 1.24838 w_k1_DIS_s111
                                 + 0.967311 w_k1_FDX_s111
                                 + 1.36481 w_k1_IBM_s111
                                 + 0.811455 w_k1_INTC_s111
                                 + 1.26823 w_k1_KO_s111 + 1.59298 w_k1_MAT_s111
                                 + 1.0793 w_k1_MSFT_s111 + 1.05043 w_k1_PG_s111
                                 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.566881 w_k1_ADM_s111
                                 + 0.64138 w_k1_AXP_s111
                                 + 0.59689 w_k1_DIS_s111
                                 + 0.70868 w_k1_FDX_s111
                                 + 0.849559 w_k1_IBM_s111
                                 + 0.677942 w_k1_INTC_s111
                                 + 0.789946 w_k1_KO_s111
                                 + 0.798052 w_k1_MAT_s111
                                 + 0.761248 w_k1_MSFT_s111
                                 + 0.779717 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.01247 w_k1_ADM_s111
                                 + 1.19708 w_k1_AXP_s111
                                 + 1.10726 w_k1_DIS_s111
                                 + 0.991008 w_k1_FDX_s111
                                 + 1.07459 w_k1_IBM_s111
                                 + 0.778901 w_k1_INTC_s111
                                 + 1.15477 w_k1_KO_s111 + 1.00807 w_k1_MAT_s111
                                 + 1.03007 w_k1_MSFT_s111 + 1.1054 w_k1_PG_s111
                                 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.05422 w_k1_ADM_s111
                                 + 0.933147 w_k1_AXP_s111
                                 + 1.04225 w_k1_DIS_s111
                                 + 1.06822 w_k1_FDX_s111
                                 + 0.647884 w_k1_IBM_s111
                                 + 0.880133 w_k1_INTC_s111
                                 + 0.934261 w_k1_KO_s111
                                 + 0.931455 w_k1_MAT_s111
                                 + 1.1851 w_k1_MSFT_s111 + 1.01206 w_k1_PG_s111
                                 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.37163 w_k1_ADM_s112
                                 + 1.28386 w_k1_AXP_s112
                                 + 1.12895 w_k1_DIS_s112 + 1.0938 w_k1_FDX_s112
                                 + 1.03903 w_k1_IBM_s112
                                 + 1.14615 w_k1_INTC_s112
                                 + 1.08876 w_k1_KO_s112 + 1.29377 w_k1_MAT_s112
                                 + 1.15229 w_k1_MSFT_s112
                                 + 1.00878 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.39647 w_k1_ADM_s112
                                 + 0.977924 w_k1_AXP_s112
                                 + 1.46733 w_k1_DIS_s112
                                 + 1.33597 w_k1_FDX_s112
                                 + 0.94316 w_k1_IBM_s112
                                 + 1.36519 w_k1_INTC_s112
                                 + 1.16677 w_k1_KO_s112
                                 + 0.817698 w_k1_MAT_s112
                                 + 1.19334 w_k1_MSFT_s112
                                 + 1.03189 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.27933 w_k1_ADM_s112
                                 + 1.3171 w_k1_AXP_s112 + 1.33811 w_k1_DIS_s112
                                 + 1.19976 w_k1_FDX_s112
                                 + 1.28285 w_k1_IBM_s112
                                 + 1.36603 w_k1_INTC_s112
                                 + 1.31854 w_k1_KO_s112 + 1.47626 w_k1_MAT_s112
                                 + 1.34921 w_k1_MSFT_s112
                                 + 1.19241 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.976158 w_k1_ADM_s112
                                 + 0.919579 w_k1_AXP_s112
                                 + 0.886249 w_k1_DIS_s112
                                 + 1.03978 w_k1_FDX_s112
                                 + 0.738972 w_k1_IBM_s112
                                 + 1.17818 w_k1_INTC_s112
                                 + 0.874087 w_k1_KO_s112
                                 + 0.790429 w_k1_MAT_s112
                                 + 1.02097 w_k1_MSFT_s112
                                 + 0.940425 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.951428 w_k1_ADM_s112
                                 + 0.961471 w_k1_AXP_s112
                                 + 0.783552 w_k1_DIS_s112
                                 + 0.876217 w_k1_FDX_s112
                                 + 1.31632 w_k1_IBM_s112
                                 + 0.85642 w_k1_INTC_s112
                                 + 1.00543 w_k1_KO_s112 + 1.32582 w_k1_MAT_s112
                                 + 0.927747 w_k1_MSFT_s112
                                 + 0.897081 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.626068 w_k1_ADM_s112
                                 + 0.606527 w_k1_AXP_s112
                                 + 0.623141 w_k1_DIS_s112
                                 + 0.722566 w_k1_FDX_s112
                                 + 0.654858 w_k1_IBM_s112
                                 + 0.965504 w_k1_INTC_s112
                                 + 0.730896 w_k1_KO_s112
                                 + 0.679804 w_k1_MAT_s112
                                 + 0.790955 w_k1_MSFT_s112
                                 + 0.789269 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.39445 w_k1_ADM_s112
                                 + 1.23273 w_k1_AXP_s112
                                 + 1.79492 w_k1_DIS_s112
                                 + 1.16484 w_k1_FDX_s112
                                 + 1.71444 w_k1_IBM_s112
                                 + 1.19283 w_k1_INTC_s112 + 1.5436 w_k1_KO_s112
                                 + 1.94443 w_k1_MAT_s112
                                 + 1.35286 w_k1_MSFT_s112
                                 + 1.10691 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.809856 w_k1_ADM_s112
                                 + 0.840686 w_k1_AXP_s112
                                 + 0.747399 w_k1_DIS_s112
                                 + 0.817192 w_k1_FDX_s112
                                 + 0.941845 w_k1_IBM_s112
                                 + 0.984579 w_k1_INTC_s112
                                 + 0.833085 w_k1_KO_s112
                                 + 0.785082 w_k1_MAT_s112
                                 + 0.839299 w_k1_MSFT_s112
                                 + 0.852823 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.910419 w_k1_ADM_s113
                                 + 0.832579 w_k1_AXP_s113
                                 + 0.775225 w_k1_DIS_s113
                                 + 0.899503 w_k1_FDX_s113
                                 + 0.667132 w_k1_IBM_s113
                                 + 0.735912 w_k1_INTC_s113
                                 + 0.805652 w_k1_KO_s113
                                 + 0.681316 w_k1_MAT_s113
                                 + 0.818663 w_k1_MSFT_s113
                                 + 0.933825 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.12428 w_k1_ADM_s113
                                 + 0.996831 w_k1_AXP_s113
                                 + 1.19999 w_k1_DIS_s113
                                 + 1.12552 w_k1_FDX_s113
                                 + 0.895399 w_k1_IBM_s113
                                 + 0.828347 w_k1_INTC_s113
                                 + 1.14575 w_k1_KO_s113 + 1.32762 w_k1_MAT_s113
                                 + 1.0557 w_k1_MSFT_s113 + 1.08924 w_k1_PG_s113
                                 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.30912 w_k1_ADM_s113
                                 + 1.24587 w_k1_AXP_s113
                                 + 1.68045 w_k1_DIS_s113
                                 + 1.44752 w_k1_FDX_s113 + 1.2358 w_k1_IBM_s113
                                 + 1.04379 w_k1_INTC_s113
                                 + 1.36611 w_k1_KO_s113 + 1.54132 w_k1_MAT_s113
                                 + 1.26645 w_k1_MSFT_s113
                                 + 1.15094 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.01592 w_k1_ADM_s113
                                 + 1.06412 w_k1_AXP_s113
                                 + 0.874062 w_k1_DIS_s113
                                 + 0.994674 w_k1_FDX_s113
                                 + 1.03866 w_k1_IBM_s113
                                 + 0.828533 w_k1_INTC_s113
                                 + 0.989903 w_k1_KO_s113
                                 + 1.41032 w_k1_MAT_s113
                                 + 0.800125 w_k1_MSFT_s113
                                 + 1.05048 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.17982 w_k1_ADM_s113
                                 + 1.25283 w_k1_AXP_s113
                                 + 1.27895 w_k1_DIS_s113
                                 + 1.15092 w_k1_FDX_s113
                                 + 1.20005 w_k1_IBM_s113
                                 + 1.02221 w_k1_INTC_s113 + 1.0831 w_k1_KO_s113
                                 + 1.64692 w_k1_MAT_s113
                                 + 1.0881 w_k1_MSFT_s113 + 1.0982 w_k1_PG_s113
                                 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.33215 w_k1_ADM_s113
                                 + 1.34525 w_k1_AXP_s113
                                 + 1.20682 w_k1_DIS_s113 + 1.3723 w_k1_FDX_s113
                                 + 0.701268 w_k1_IBM_s113
                                 + 0.853591 w_k1_INTC_s113
                                 + 1.03482 w_k1_KO_s113
                                 + 0.924844 w_k1_MAT_s113
                                 + 1.09363 w_k1_MSFT_s113
                                 + 1.12114 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.980679 w_k1_ADM_s113
                                 + 1.13589 w_k1_AXP_s113
                                 + 1.35494 w_k1_DIS_s113
                                 + 1.04045 w_k1_FDX_s113
                                 + 1.54854 w_k1_IBM_s113
                                 + 0.912511 w_k1_INTC_s113
                                 + 1.50119 w_k1_KO_s113 + 1.85196 w_k1_MAT_s113
                                 + 1.07949 w_k1_MSFT_s113
                                 + 1.20025 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.897408 w_k1_ADM_s113
                                 + 0.785951 w_k1_AXP_s113
                                 + 0.93755 w_k1_DIS_s113
                                 + 0.839109 w_k1_FDX_s113
                                 + 1.0761 w_k1_IBM_s113
                                 + 0.793568 w_k1_INTC_s113
                                 + 1.17891 w_k1_KO_s113
                                 + 0.853394 w_k1_MAT_s113
                                 + 0.848065 w_k1_MSFT_s113
                                 + 0.999918 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.655904 w_k1_ADM_s114
                                 + 0.7746 w_k1_AXP_s114
                                 + 0.647958 w_k1_DIS_s114
                                 + 0.780897 w_k1_FDX_s114
                                 + 0.818762 w_k1_IBM_s114
                                 + 0.825948 w_k1_INTC_s114
                                 + 0.707083 w_k1_KO_s114
                                 + 0.839326 w_k1_MAT_s114
                                 + 0.826635 w_k1_MSFT_s114
                                 + 0.884937 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.36179 w_k1_ADM_s114
                                 + 1.16435 w_k1_AXP_s114
                                 + 1.25309 w_k1_DIS_s114
                                 + 1.23194 w_k1_FDX_s114 + 1.0518 w_k1_IBM_s114
                                 + 1.48779 w_k1_INTC_s114
                                 + 1.10207 w_k1_KO_s114
                                 + 0.777355 w_k1_MAT_s114
                                 + 1.18751 w_k1_MSFT_s114 + 1.0092 w_k1_PG_s114
                                 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.0441 w_k1_ADM_s114
                                 + 0.964076 w_k1_AXP_s114
                                 + 1.12291 w_k1_DIS_s114
                                 + 1.02399 w_k1_FDX_s114
                                 + 1.32185 w_k1_IBM_s114
                                 + 1.05749 w_k1_INTC_s114
                                 + 1.14643 w_k1_KO_s114 + 1.32119 w_k1_MAT_s114
                                 + 0.911696 w_k1_MSFT_s114
                                 + 1.05203 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.10172 w_k1_ADM_s114
                                 + 1.15623 w_k1_AXP_s114
                                 + 1.32372 w_k1_DIS_s114
                                 + 1.31891 w_k1_FDX_s114
                                 + 0.776101 w_k1_IBM_s114
                                 + 1.055 w_k1_INTC_s114 + 0.974591 w_k1_KO_s114
                                 + 0.93715 w_k1_MAT_s114
                                 + 1.2671 w_k1_MSFT_s114 + 1.09411 w_k1_PG_s114
                                 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.36246 w_k1_ADM_s114
                                 + 1.33371 w_k1_AXP_s114 + 1.8565 w_k1_DIS_s114
                                 + 1.37338 w_k1_FDX_s114
                                 + 1.35584 w_k1_IBM_s114
                                 + 1.28871 w_k1_INTC_s114
                                 + 1.45117 w_k1_KO_s114 + 1.17265 w_k1_MAT_s114
                                 + 1.339 w_k1_MSFT_s114 + 1.24894 w_k1_PG_s114
                                 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.714907 w_k1_ADM_s114
                                 + 0.657232 w_k1_AXP_s114
                                 + 0.72204 w_k1_DIS_s114
                                 + 0.761011 w_k1_FDX_s114
                                 + 1.07434 w_k1_IBM_s114
                                 + 1.00198 w_k1_INTC_s114
                                 + 0.861343 w_k1_KO_s114
                                 + 0.961445 w_k1_MAT_s114
                                 + 0.863258 w_k1_MSFT_s114
                                 + 0.905372 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.15554 w_k1_ADM_s114
                                 + 0.775489 w_k1_AXP_s114
                                 + 1.00595 w_k1_DIS_s114 + 1.0027 w_k1_FDX_s114
                                 + 0.817155 w_k1_IBM_s114
                                 + 1.18726 w_k1_INTC_s114
                                 + 0.998961 w_k1_KO_s114
                                 + 0.810644 w_k1_MAT_s114
                                 + 1.03369 w_k1_MSFT_s114
                                 + 0.966736 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.987677 w_k1_ADM_s114
                                 + 0.730572 w_k1_AXP_s114
                                 + 0.793941 w_k1_DIS_s114
                                 + 0.884384 w_k1_FDX_s114
                                 + 0.594913 w_k1_IBM_s114
                                 + 0.979284 w_k1_INTC_s114
                                 + 0.807995 w_k1_KO_s114
                                 + 0.647796 w_k1_MAT_s114
                                 + 0.882304 w_k1_MSFT_s114
                                 + 0.937778 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.43186 w_k1_ADM_s115
                                 + 1.07403 w_k1_AXP_s115
                                 + 1.25558 w_k1_DIS_s115
                                 + 1.22767 w_k1_FDX_s115
                                 + 0.902735 w_k1_IBM_s115
                                 + 1.15895 w_k1_INTC_s115
                                 + 1.02703 w_k1_KO_s115 + 1.04721 w_k1_MAT_s115
                                 + 1.18674 w_k1_MSFT_s115
                                 + 1.05859 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.14378 w_k1_ADM_s115
                                 + 0.857234 w_k1_AXP_s115
                                 + 0.956425 w_k1_DIS_s115
                                 + 0.980731 w_k1_FDX_s115
                                 + 0.853995 w_k1_IBM_s115
                                 + 0.860823 w_k1_INTC_s115
                                 + 1.06004 w_k1_KO_s115 + 1.1682 w_k1_MAT_s115
                                 + 0.946143 w_k1_MSFT_s115
                                 + 0.921997 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.830908 w_k1_ADM_s115
                                 + 0.680562 w_k1_AXP_s115
                                 + 0.584023 w_k1_DIS_s115
                                 + 0.734273 w_k1_FDX_s115
                                 + 0.717121 w_k1_IBM_s115
                                 + 0.668966 w_k1_INTC_s115
                                 + 0.754394 w_k1_KO_s115
                                 + 0.944769 w_k1_MAT_s115
                                 + 0.69949 w_k1_MSFT_s115
                                 + 0.864772 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.19298 w_k1_ADM_s115
                                 + 1.04733 w_k1_AXP_s115
                                 + 1.04488 w_k1_DIS_s115
                                 + 1.02437 w_k1_FDX_s115
                                 + 1.15184 w_k1_IBM_s115
                                 + 1.10996 w_k1_INTC_s115 + 1.2298 w_k1_KO_s115
                                 + 1.2014 w_k1_MAT_s115
                                 + 1.03628 w_k1_MSFT_s115
                                 + 1.01154 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.938762 w_k1_ADM_s115
                                 + 0.914837 w_k1_AXP_s115
                                 + 0.771803 w_k1_DIS_s115
                                 + 0.857605 w_k1_FDX_s115
                                 + 1.08696 w_k1_IBM_s115
                                 + 0.75065 w_k1_INTC_s115
                                 + 1.05188 w_k1_KO_s115 + 1.35263 w_k1_MAT_s115
                                 + 0.769749 w_k1_MSFT_s115
                                 + 0.958603 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.57383 w_k1_ADM_s115
                                 + 1.22886 w_k1_AXP_s115
                                 + 1.82508 w_k1_DIS_s115
                                 + 1.34784 w_k1_FDX_s115 + 1.2161 w_k1_IBM_s115
                                 + 1.086 w_k1_INTC_s115 + 1.50933 w_k1_KO_s115
                                 + 1.41709 w_k1_MAT_s115
                                 + 1.39304 w_k1_MSFT_s115 + 1.2313 w_k1_PG_s115
                                 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.18314 w_k1_ADM_s115
                                 + 1.27542 w_k1_AXP_s115
                                 + 1.33356 w_k1_DIS_s115
                                 + 1.10736 w_k1_FDX_s115
                                 + 1.41828 w_k1_IBM_s115
                                 + 1.0013 w_k1_INTC_s115 + 1.21301 w_k1_KO_s115
                                 + 1.81684 w_k1_MAT_s115
                                 + 1.13129 w_k1_MSFT_s115
                                 + 1.10129 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.00475 w_k1_ADM_s115
                                 + 0.79186 w_k1_AXP_s115
                                 + 0.804132 w_k1_DIS_s115
                                 + 0.947682 w_k1_FDX_s115
                                 + 0.746255 w_k1_IBM_s115
                                 + 0.844301 w_k1_INTC_s115
                                 + 0.862467 w_k1_KO_s115
                                 + 0.66566 w_k1_MAT_s115
                                 + 0.837542 w_k1_MSFT_s115
                                 + 0.817665 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.832322 w_k1_ADM_s116
                                 + 0.924576 w_k1_AXP_s116
                                 + 0.689013 w_k1_DIS_s116
                                 + 0.805545 w_k1_FDX_s116
                                 + 0.834215 w_k1_IBM_s116
                                 + 0.868987 w_k1_INTC_s116
                                 + 0.746062 w_k1_KO_s116
                                 + 0.88844 w_k1_MAT_s116
                                 + 0.833125 w_k1_MSFT_s116
                                 + 0.833835 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.0941 w_k1_ADM_s116
                                 + 1.12397 w_k1_AXP_s116
                                 + 0.897576 w_k1_DIS_s116
                                 + 0.963618 w_k1_FDX_s116
                                 + 1.03528 w_k1_IBM_s116
                                 + 0.886324 w_k1_INTC_s116
                                 + 1.03683 w_k1_KO_s116 + 1.24092 w_k1_MAT_s116
                                 + 0.976218 w_k1_MSFT_s116
                                 + 0.977014 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.31012 w_k1_ADM_s116
                                 + 1.87957 w_k1_AXP_s116
                                 + 1.13174 w_k1_DIS_s116
                                 + 1.27202 w_k1_FDX_s116
                                 + 1.11586 w_k1_IBM_s116
                                 + 1.20157 w_k1_INTC_s116
                                 + 1.11643 w_k1_KO_s116 + 1.41296 w_k1_MAT_s116
                                 + 1.14961 w_k1_MSFT_s116
                                 + 1.18523 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.01612 w_k1_ADM_s116
                                 + 0.932714 w_k1_AXP_s116
                                 + 1.00347 w_k1_DIS_s116
                                 + 1.13115 w_k1_FDX_s116
                                 + 0.761273 w_k1_IBM_s116
                                 + 0.914617 w_k1_INTC_s116
                                 + 1.01318 w_k1_KO_s116
                                 + 0.705201 w_k1_MAT_s116
                                 + 0.993061 w_k1_MSFT_s116
                                 + 0.962086 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.812322 w_k1_ADM_s116
                                 + 0.841228 w_k1_AXP_s116
                                 + 0.793922 w_k1_DIS_s116
                                 + 0.803938 w_k1_FDX_s116
                                 + 1.07351 w_k1_IBM_s116
                                 + 0.842707 w_k1_INTC_s116
                                 + 0.940708 w_k1_KO_s116
                                 + 0.997699 w_k1_MAT_s116
                                 + 0.896797 w_k1_MSFT_s116
                                 + 0.886512 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.30482 w_k1_ADM_s116
                                 + 1.54611 w_k1_AXP_s116
                                 + 1.63165 w_k1_DIS_s116
                                 + 1.43864 w_k1_FDX_s116
                                 + 1.14252 w_k1_IBM_s116
                                 + 1.23164 w_k1_INTC_s116
                                 + 1.41114 w_k1_KO_s116 + 1.26535 w_k1_MAT_s116
                                 + 1.31444 w_k1_MSFT_s116
                                 + 1.09475 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.09959 w_k1_ADM_s116
                                 + 0.955904 w_k1_AXP_s116
                                 + 1.30626 w_k1_DIS_s116
                                 + 1.11049 w_k1_FDX_s116
                                 + 1.00992 w_k1_IBM_s116
                                 + 1.20554 w_k1_INTC_s116
                                 + 1.19114 w_k1_KO_s116
                                 + 0.821019 w_k1_MAT_s116
                                 + 1.14755 w_k1_MSFT_s116 + 1.0228 w_k1_PG_s116
                                 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.16465 w_k1_ADM_s116
                                 + 1.20679 w_k1_AXP_s116
                                 + 1.12046 w_k1_DIS_s116
                                 + 1.03844 w_k1_FDX_s116
                                 + 1.45588 w_k1_IBM_s116
                                 + 0.985368 w_k1_INTC_s116
                                 + 1.20769 w_k1_KO_s116 + 1.63432 w_k1_MAT_s116
                                 + 1.12148 w_k1_MSFT_s116 + 1.2138 w_k1_PG_s116
                                 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.11718 w_k1_ADM_s117
                                 + 0.993195 w_k1_AXP_s117
                                 + 1.1905 w_k1_DIS_s117 + 1.00595 w_k1_FDX_s117
                                 + 1.55329 w_k1_IBM_s117
                                 + 1.20367 w_k1_INTC_s117
                                 + 1.44305 w_k1_KO_s117 + 1.12415 w_k1_MAT_s117
                                 + 0.982348 w_k1_MSFT_s117
                                 + 1.12464 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.22539 w_k1_ADM_s117
                                 + 1.20149 w_k1_AXP_s117
                                 + 1.65874 w_k1_DIS_s117
                                 + 1.27689 w_k1_FDX_s117
                                 + 1.03323 w_k1_IBM_s117
                                 + 1.47975 w_k1_INTC_s117
                                 + 1.46223 w_k1_KO_s117 + 1.0256 w_k1_MAT_s117
                                 + 1.50059 w_k1_MSFT_s117
                                 + 1.29637 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.909132 w_k1_ADM_s117
                                 + 0.714713 w_k1_AXP_s117
                                 + 0.705041 w_k1_DIS_s117
                                 + 0.764883 w_k1_FDX_s117
                                 + 0.861196 w_k1_IBM_s117
                                 + 0.954257 w_k1_INTC_s117
                                 + 0.909792 w_k1_KO_s117
                                 + 0.714878 w_k1_MAT_s117
                                 + 0.86246 w_k1_MSFT_s117
                                 + 0.857835 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.20317 w_k1_ADM_s117
                                 + 1.75067 w_k1_AXP_s117
                                 + 1.54592 w_k1_DIS_s117
                                 + 1.28936 w_k1_FDX_s117 + 1.5947 w_k1_IBM_s117
                                 + 1.64327 w_k1_INTC_s117 + 1.6281 w_k1_KO_s117
                                 + 1.91873 w_k1_MAT_s117
                                 + 1.25493 w_k1_MSFT_s117
                                 + 1.37906 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.865113 w_k1_ADM_s117
                                 + 0.707582 w_k1_AXP_s117
                                 + 0.665617 w_k1_DIS_s117
                                 + 0.816665 w_k1_FDX_s117
                                 + 0.531309 w_k1_IBM_s117
                                 + 0.807956 w_k1_INTC_s117
                                 + 0.651447 w_k1_KO_s117
                                 + 0.558562 w_k1_MAT_s117
                                 + 0.692696 w_k1_MSFT_s117
                                 + 0.797745 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.991489 w_k1_ADM_s117
                                 + 1.11561 w_k1_AXP_s117
                                 + 0.870108 w_k1_DIS_s117
                                 + 0.989676 w_k1_FDX_s117
                                 + 1.02828 w_k1_IBM_s117
                                 + 1.16157 w_k1_INTC_s117
                                 + 1.04958 w_k1_KO_s117 + 1.03043 w_k1_MAT_s117
                                 + 0.845431 w_k1_MSFT_s117
                                 + 1.09035 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.955853 w_k1_ADM_s117
                                 + 0.980744 w_k1_AXP_s117
                                 + 0.92254 w_k1_DIS_s117
                                 + 1.03594 w_k1_FDX_s117
                                 + 0.630345 w_k1_IBM_s117
                                 + 1.09725 w_k1_INTC_s117
                                 + 0.820458 w_k1_KO_s117
                                 + 0.706639 w_k1_MAT_s117
                                 + 1.04094 w_k1_MSFT_s117
                                 + 1.05028 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.917333 w_k1_ADM_s117
                                 + 1.09512 w_k1_AXP_s117
                                 + 1.21242 w_k1_DIS_s117
                                 + 1.22815 w_k1_FDX_s117
                                 + 0.88187 w_k1_IBM_s117
                                 + 1.53594 w_k1_INTC_s117
                                 + 1.12111 w_k1_KO_s117
                                 + 0.665456 w_k1_MAT_s117
                                 + 1.29451 w_k1_MSFT_s117
                                 + 1.16272 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0136435222217138 w_k1_ADM_s000
                                 + 0.00962512281160951 w_k1_ADM_s001
                                 + 0.00826732031740806 w_k1_ADM_s002
                                 + 0.0102749166001371 w_k1_ADM_s003
                                 + 0.00887196230954916 w_k1_ADM_s004
                                 + 0.0112313718201362 w_k1_ADM_s005
                                 + 0.012730128113988 w_k1_ADM_s006
                                 + 0.00410893251449415 w_k1_ADM_s007
                                 + 0.00786607869917338 w_k1_ADM_s010
                                 + 0.0122888179986866 w_k1_ADM_s011
                                 + 0.0145683002756678 w_k1_ADM_s012
                                 + 0.00753864357553637 w_k1_ADM_s013
                                 + 0.0228052599212182 w_k1_ADM_s014
                                 + 0.0101612513072003 w_k1_ADM_s015
                                 + 0.00687999379522898 w_k1_ADM_s016
                                 + 0.0244997536475956 w_k1_ADM_s017
                                 + 0.0352166592890044 w_k1_ADM_s100
                                 + 0.0162486815599491 w_k1_ADM_s101
                                 + 0.0242411408298187 w_k1_ADM_s102
                                 + 0.0444730120190216 w_k1_ADM_s103
                                 + 0.0123484147094658 w_k1_ADM_s104
                                 + 0.0233673421824178 w_k1_ADM_s105
                                 + 0.0110651932553562 w_k1_ADM_s106
                                 + 0.0483609498372378 w_k1_ADM_s107
                                 + 0.0488605385614986 w_k1_ADM_s110
                                 + 0.132606114311666 w_k1_ADM_s111
                                 + 0.0695220995747519 w_k1_ADM_s112
                                 + 0.140134847262024 w_k1_ADM_s113
                                 + 0.0595924778769406 w_k1_ADM_s114
                                 + 0.0509428062804617 w_k1_ADM_s115
                                 + 0.0777891101066791 w_k1_ADM_s116
                                 + 0.0314326260234061 w_k1_ADM_s117
                                 + 0.0143167834266366 w_k1_AXP_s000
                                 + 0.00741708384871487 w_k1_AXP_s001
                                 + 0.00850035386366863 w_k1_AXP_s002
                                 + 0.0100427298843252 w_k1_AXP_s003
                                 + 0.00957193192682802 w_k1_AXP_s004
                                 + 0.0110589795763178 w_k1_AXP_s005
                                 + 0.0139317393091326 w_k1_AXP_s006
                                 + 0.00422483806692499 w_k1_AXP_s007
                                 + 0.00690729184799952 w_k1_AXP_s010
                                 + 0.0127278877948045 w_k1_AXP_s011
                                 + 0.0131080794186069 w_k1_AXP_s012
                                 + 0.00797667264566189 w_k1_AXP_s013
                                 + 0.0244121086732879 w_k1_AXP_s014
                                 + 0.00945166148530327 w_k1_AXP_s015
                                 + 0.00665447527739685 w_k1_AXP_s016
                                 + 0.0261737829553778 w_k1_AXP_s017
                                 + 0.0366834324719972 w_k1_AXP_s100
                                 + 0.0159105385659388 w_k1_AXP_s101
                                 + 0.0256008234579495 w_k1_AXP_s102
                                 + 0.045335150412172 w_k1_AXP_s103
                                 + 0.0127207018698296 w_k1_AXP_s104
                                 + 0.0238229153691359 w_k1_AXP_s105
                                 + 0.0117888623451837 w_k1_AXP_s106
                                 + 0.0557944440221345 w_k1_AXP_s107
                                 + 0.0555385987363672 w_k1_AXP_s110
                                 + 0.148619444271098 w_k1_AXP_s111
                                 + 0.0658790565683478 w_k1_AXP_s112
                                 + 0.137470159528243 w_k1_AXP_s113
                                 + 0.0538385650748313 w_k1_AXP_s114
                                 + 0.0441765701375992 w_k1_AXP_s115
                                 + 0.0807306425389839 w_k1_AXP_s116
                                 + 0.0323168165281436 w_k1_AXP_s117
                                 + 0.0142462346992621 w_k1_DIS_s000
                                 + 0.00824299369200637 w_k1_DIS_s001
                                 + 0.00886969801262713 w_k1_DIS_s002
                                 + 0.00969797566289353 w_k1_DIS_s003
                                 + 0.00892372197020116 w_k1_DIS_s004
                                 + 0.01095570160664 w_k1_DIS_s005
                                 + 0.013823257064431 w_k1_DIS_s006
                                 + 0.0047987461287958 w_k1_DIS_s007
                                 + 0.00722963817273269 w_k1_DIS_s010
                                 + 0.0140831660791042 w_k1_DIS_s011
                                 + 0.012831103713304 w_k1_DIS_s012
                                 + 0.0081358102528982 w_k1_DIS_s013
                                 + 0.0219219032852876 w_k1_DIS_s014
                                 + 0.0103937803572819 w_k1_DIS_s015
                                 + 0.00683544379482664 w_k1_DIS_s016
                                 + 0.0241099948645041 w_k1_DIS_s017
                                 + 0.0376500976420112 w_k1_DIS_s100
                                 + 0.0176299679703721 w_k1_DIS_s101
                                 + 0.0265378210091683 w_k1_DIS_s102
                                 + 0.0485072027266444 w_k1_DIS_s103
                                 + 0.0126465585695265 w_k1_DIS_s104
                                 + 0.0242783268106661 w_k1_DIS_s105
                                 + 0.0131029963037059 w_k1_DIS_s106
                                 + 0.0517281927245072 w_k1_DIS_s107
                                 + 0.0598640030104741 w_k1_DIS_s110
                                 + 0.14778701653228 w_k1_DIS_s111
                                 + 0.0656026061628156 w_k1_DIS_s112
                                 + 0.145477890185701 w_k1_DIS_s113
                                 + 0.0618101135550589 w_k1_DIS_s114
                                 + 0.0461425010357202 w_k1_DIS_s115
                                 + 0.0772065694539986 w_k1_DIS_s116
                                 + 0.0317556212735435 w_k1_DIS_s117
                                 + 0.0142902744117814 w_k1_FDX_s000
                                 + 0.00810833054276968 w_k1_FDX_s001
                                 + 0.00816467196619358 w_k1_FDX_s002
                                 + 0.0099861545695189 w_k1_FDX_s003
                                 + 0.00908861170663284 w_k1_FDX_s004
                                 + 0.0105154669994906 w_k1_FDX_s005
                                 + 0.0131374504137576 w_k1_FDX_s006
                                 + 0.00429526427700746 w_k1_FDX_s007
                                 + 0.0069252857545321 w_k1_FDX_s010
                                 + 0.0135323928213817 w_k1_FDX_s011
                                 + 0.0125179907253497 w_k1_FDX_s012
                                 + 0.00804952727798853 w_k1_FDX_s013
                                 + 0.0232994373365295 w_k1_FDX_s014
                                 + 0.00912562450444924 w_k1_FDX_s015
                                 + 0.00665545640053479 w_k1_FDX_s016
                                 + 0.025530018619691 w_k1_FDX_s017
                                 + 0.0336388945364381 w_k1_FDX_s100
                                 + 0.017660573914728 w_k1_FDX_s101
                                 + 0.026134016433336 w_k1_FDX_s102
                                 + 0.0481737812072218 w_k1_FDX_s103
                                 + 0.012515699285367 w_k1_FDX_s104
                                 + 0.0229092067647687 w_k1_FDX_s105
                                 + 0.0117346321722912 w_k1_FDX_s106
                                 + 0.0477307510618189 w_k1_FDX_s107
                                 + 0.0565424887708703 w_k1_FDX_s110
                                 + 0.139346659021299 w_k1_FDX_s111
                                 + 0.0662462374411131 w_k1_FDX_s112
                                 + 0.140294767200807 w_k1_FDX_s113
                                 + 0.058866552208353 w_k1_FDX_s114
                                 + 0.0451565540840757 w_k1_FDX_s115
                                 + 0.0769597920512815 w_k1_FDX_s116
                                 + 0.0319557517136535 w_k1_FDX_s117
                                 + 0.0143412446298872 w_k1_IBM_s000
                                 + 0.00735926583892201 w_k1_IBM_s001
                                 + 0.00847638381887508 w_k1_IBM_s002
                                 + 0.0100830090153359 w_k1_IBM_s003
                                 + 0.00941706101293171 w_k1_IBM_s004
                                 + 0.0109404528462662 w_k1_IBM_s005
                                 + 0.0137407854246793 w_k1_IBM_s006
                                 + 0.00475456543267761 w_k1_IBM_s007
                                 + 0.0069375012313097 w_k1_IBM_s010
                                 + 0.0132964901300725 w_k1_IBM_s011
                                 + 0.0127701171794325 w_k1_IBM_s012
                                 + 0.0075626433481318 w_k1_IBM_s013
                                 + 0.022618218750198 w_k1_IBM_s014
                                 + 0.00960789032801296 w_k1_IBM_s015
                                 + 0.00703267131978144 w_k1_IBM_s016
                                 + 0.0253423474030979 w_k1_IBM_s017
                                 + 0.0354455156508373 w_k1_IBM_s100
                                 + 0.0169058916412514 w_k1_IBM_s101
                                 + 0.0261303794684628 w_k1_IBM_s102
                                 + 0.050334653412778 w_k1_IBM_s103
                                 + 0.0128721312075152 w_k1_IBM_s104
                                 + 0.0243804138465951 w_k1_IBM_s105
                                 + 0.012496293054305 w_k1_IBM_s106
                                 + 0.0494825383628149 w_k1_IBM_s107
                                 + 0.0570781827882037 w_k1_IBM_s110
                                 + 0.140950702053631 w_k1_IBM_s111
                                 + 0.0669694595468316 w_k1_IBM_s112
                                 + 0.139112615244481 w_k1_IBM_s113
                                 + 0.0592644136741048 w_k1_IBM_s114
                                 + 0.0462793422132992 w_k1_IBM_s115
                                 + 0.0780350180775969 w_k1_IBM_s116
                                 + 0.0309491925563448 w_k1_IBM_s117
                                 + 0.0154559790474946 w_k1_INTC_s000
                                 + 0.00713311039323708 w_k1_INTC_s001
                                 + 0.00925256312364142 w_k1_INTC_s002
                                 + 0.00939871613187748 w_k1_INTC_s003
                                 + 0.00848257362559119 w_k1_INTC_s004
                                 + 0.00951027540295928 w_k1_INTC_s005
                                 + 0.0145852614043857 w_k1_INTC_s006
                                 + 0.00408281243964551 w_k1_INTC_s007
                                 + 0.00575474426003818 w_k1_INTC_s010
                                 + 0.0145064330613471 w_k1_INTC_s011
                                 + 0.0138562266787878 w_k1_INTC_s012
                                 + 0.00762340099680598 w_k1_INTC_s013
                                 + 0.0213475764765886 w_k1_INTC_s014
                                 + 0.0086953264651196 w_k1_INTC_s015
                                 + 0.00613545460527869 w_k1_INTC_s016
                                 + 0.0259536100343106 w_k1_INTC_s017
                                 + 0.040959695985154 w_k1_INTC_s100
                                 + 0.0169900524682935 w_k1_INTC_s101
                                 + 0.027428722637742 w_k1_INTC_s102
                                 + 0.0539432029992138 w_k1_INTC_s103
                                 + 0.0121050286159882 w_k1_INTC_s104
                                 + 0.0241889350868313 w_k1_INTC_s105
                                 + 0.0115481730182634 w_k1_INTC_s106
                                 + 0.0490815281852931 w_k1_INTC_s107
                                 + 0.0616909799334442 w_k1_INTC_s110
                                 + 0.118158562691689 w_k1_INTC_s111
                                 + 0.0734630844197732 w_k1_INTC_s112
                                 + 0.115413489619204 w_k1_INTC_s113
                                 + 0.0629806084713273 w_k1_INTC_s114
                                 + 0.0418801817260165 w_k1_INTC_s115
                                 + 0.0734565912217366 w_k1_INTC_s116
                                 + 0.037670820945296 w_k1_INTC_s117
                                 + 0.0125719422788562 w_k1_KO_s000
                                 + 0.00851479645466701 w_k1_KO_s001
                                 + 0.00878410170172048 w_k1_KO_s002
                                 + 0.0101895461591901 w_k1_KO_s003
                                 + 0.00829071477393015 w_k1_KO_s004
                                 + 0.0111064494836093 w_k1_KO_s005
                                 + 0.0133781496296563 w_k1_KO_s006
                                 + 0.00456196710917037 w_k1_KO_s007
                                 + 0.00701186170595991 w_k1_KO_s010
                                 + 0.0146333507561753 w_k1_KO_s011
                                 + 0.0119666643345209 w_k1_KO_s012
                                 + 0.00843771544341824 w_k1_KO_s013
                                 + 0.0226535041813423 w_k1_KO_s014
                                 + 0.00923529445482829 w_k1_KO_s015
                                 + 0.00745394386584568 w_k1_KO_s016
                                 + 0.021644714868287 w_k1_KO_s017
                                 + 0.0389308388115601 w_k1_KO_s100
                                 + 0.0176307622776644 w_k1_KO_s101
                                 + 0.0296310982705156 w_k1_KO_s102
                                 + 0.0537456004417269 w_k1_KO_s103
                                 + 0.0133583188428073 w_k1_KO_s104
                                 + 0.0245093461264226 w_k1_KO_s105
                                 + 0.013508871359674 w_k1_KO_s106
                                 + 0.0511798389039305 w_k1_KO_s107
                                 + 0.058452427035005 w_k1_KO_s110
                                 + 0.149066861650254 w_k1_KO_s111
                                 + 0.0664349525227763 w_k1_KO_s112
                                 + 0.149103016742137 w_k1_KO_s113
                                 + 0.0585050637201069 w_k1_KO_s114
                                 + 0.0485565576436703 w_k1_KO_s115
                                 + 0.0790374089545025 w_k1_KO_s116
                                 + 0.034103264006935 w_k1_KO_s117
                                 + 0.0138561713145999 w_k1_MAT_s000
                                 + 0.00654118062048261 w_k1_MAT_s001
                                 + 0.00928180300213956 w_k1_MAT_s002
                                 + 0.009616720082858 w_k1_MAT_s003
                                 + 0.0104516300079173 w_k1_MAT_s004
                                 + 0.0173800783314931 w_k1_MAT_s005
                                 + 0.016873121927539 w_k1_MAT_s006
                                 + 0.00435373145497239 w_k1_MAT_s007
                                 + 0.00777055838701497 w_k1_MAT_s010
                                 + 0.0174650258814999 w_k1_MAT_s011
                                 + 0.013445751213529 w_k1_MAT_s012
                                 + 0.00847350812311972 w_k1_MAT_s013
                                 + 0.022312588983775 w_k1_MAT_s014
                                 + 0.0106930009254036 w_k1_MAT_s015
                                 + 0.00911123627362408 w_k1_MAT_s016
                                 + 0.021386571667361 w_k1_MAT_s017
                                 + 0.0348992803683475 w_k1_MAT_s100
                                 + 0.0167698434099816 w_k1_MAT_s101
                                 + 0.0299586452808952 w_k1_MAT_s102
                                 + 0.0516146814291559 w_k1_MAT_s103
                                 + 0.0115628331720463 w_k1_MAT_s104
                                 + 0.0265733788127917 w_k1_MAT_s105
                                 + 0.0134599298151192 w_k1_MAT_s106
                                 + 0.0607809765800466 w_k1_MAT_s107
                                 + 0.055198904189278 w_k1_MAT_s110
                                 + 0.151481565584701 w_k1_MAT_s111
                                 + 0.0699713898531391 w_k1_MAT_s112
                                 + 0.161874503570381 w_k1_MAT_s113
                                 + 0.0565293784857805 w_k1_MAT_s114
                                 + 0.0551909239746467 w_k1_MAT_s115
                                 + 0.0810585217712994 w_k1_MAT_s116
                                 + 0.0287870087855496 w_k1_MAT_s117
                                 + 0.0142077055541989 w_k1_MSFT_s000
                                 + 0.00758366355895008 w_k1_MSFT_s001
                                 + 0.00878237384536859 w_k1_MSFT_s002
                                 + 0.00937428666751998 w_k1_MSFT_s003
                                 + 0.00899682916020444 w_k1_MSFT_s004
                                 + 0.0111331237958861 w_k1_MSFT_s005
                                 + 0.0136085778231483 w_k1_MSFT_s006
                                 + 0.00460657106109914 w_k1_MSFT_s007
                                 + 0.0066066132947512 w_k1_MSFT_s010
                                 + 0.0143710998261659 w_k1_MSFT_s011
                                 + 0.0127452494767363 w_k1_MSFT_s012
                                 + 0.00776955778198454 w_k1_MSFT_s013
                                 + 0.0208901196461161 w_k1_MSFT_s014
                                 + 0.00907565945807373 w_k1_MSFT_s015
                                 + 0.0072680181381428 w_k1_MSFT_s016
                                 + 0.0243037825801115 w_k1_MSFT_s017
                                 + 0.035863431642526 w_k1_MSFT_s100
                                 + 0.0175747205144559 w_k1_MSFT_s101
                                 + 0.0291442458648157 w_k1_MSFT_s102
                                 + 0.0539869356209845 w_k1_MSFT_s103
                                 + 0.0123062936136673 w_k1_MSFT_s104
                                 + 0.0236919900521645 w_k1_MSFT_s105
                                 + 0.0131680694045154 w_k1_MSFT_s106
                                 + 0.0481698320217283 w_k1_MSFT_s107
                                 + 0.0575913952977009 w_k1_MSFT_s110
                                 + 0.145012403746663 w_k1_MSFT_s111
                                 + 0.0686234204266504 w_k1_MSFT_s112
                                 + 0.12896392377434 w_k1_MSFT_s113
                                 + 0.0573703840135658 w_k1_MSFT_s114
                                 + 0.0437777219932787 w_k1_MSFT_s115
                                 + 0.0768415657012341 w_k1_MSFT_s116
                                 + 0.0317727332679564 w_k1_MSFT_s117
                                 + 0.0131087474145699 w_k1_PG_s000
                                 + 0.00839081854045762 w_k1_PG_s001
                                 + 0.00831604969224337 w_k1_PG_s002
                                 + 0.00928665546443791 w_k1_PG_s003
                                 + 0.00851818710899467 w_k1_PG_s004
                                 + 0.0111099855855297 w_k1_PG_s005
                                 + 0.0130087798077197 w_k1_PG_s006
                                 + 0.0046018626307654 w_k1_PG_s007
                                 + 0.00649590099736757 w_k1_PG_s010
                                 + 0.0131999003510361 w_k1_PG_s011
                                 + 0.0135087902657708 w_k1_PG_s012
                                 + 0.0079313530486113 w_k1_PG_s013
                                 + 0.0207557623899709 w_k1_PG_s014
                                 + 0.00905398778473635 w_k1_PG_s015
                                 + 0.00734573632559282 w_k1_PG_s016
                                 + 0.0235964311198114 w_k1_PG_s017
                                 + 0.0325865308053481 w_k1_PG_s100
                                 + 0.0166626900096748 w_k1_PG_s101
                                 + 0.0252560470384618 w_k1_PG_s102
                                 + 0.0488418044637088 w_k1_PG_s103
                                 + 0.0137419336652905 w_k1_PG_s104
                                 + 0.0228608799052484 w_k1_PG_s105
                                 + 0.0122101583143487 w_k1_PG_s106
                                 + 0.0464314500712983 w_k1_PG_s107
                                 + 0.0515226057956725 w_k1_PG_s110
                                 + 0.140328939408939 w_k1_PG_s111
                                 + 0.063342163617149 w_k1_PG_s112
                                 + 0.142780386898703 w_k1_PG_s113
                                 + 0.0576175419891454 w_k1_PG_s114
                                 + 0.0441054559550315 w_k1_PG_s115
                                 + 0.0751503619830265 w_k1_PG_s116
                                 + 0.0336788458063261 w_k1_PG_s117 >= 1.07
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

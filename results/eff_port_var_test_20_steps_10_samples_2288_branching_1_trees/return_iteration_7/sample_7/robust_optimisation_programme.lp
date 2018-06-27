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
 w_k1_ADM_s0_return_constraint:  1.15529 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.90427 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.852471 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.02906 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.973563 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.00282 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.01904 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.45706 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.18818 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.751962 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.739269 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.31255 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.14905 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.00936 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.770411 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.886349 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.856422 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.823006 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.42596 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.16263 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.16593 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.88527 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.03201 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.32153 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.929793 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.829961 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.40802 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.986125 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.10837 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.09496 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.978289 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.805944 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.13073 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.66529 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.14084 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.07957 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.745807 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.32889 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.23549 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.86474 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.72696 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.08853 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.19766 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.930694 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.817403 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.53519 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.03439 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.964452 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.710231 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.52322 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.15181 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.25411 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.643184 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.932696 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.900468 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.706562 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.48393 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.31211 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.09868 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.942777 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.669104 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.2359 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.938669 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.586972 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.26189 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.902138 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.926005 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.91316 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.12197 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.840533 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.02242 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.719002 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.34604 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.64155 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.646535 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.51688 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.30528 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.825718 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.76429 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.11062 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.12999 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.995994 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.847374 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.8142 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.914154 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.804404 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.587906 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.40949 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.02589 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.17316 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.677579 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.892788 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.822761 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.596719 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.68902 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.16689 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.993953 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.882787 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.802486 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.31121 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.04984 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.640517 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.57276 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.63957 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.937674 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.29522 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.896198 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.880917 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.684966 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.638829 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.29665 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.65621 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.479618 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.73941 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.17501 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.892205 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.888763 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.05565 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.997721 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.03803 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.915365 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.42993 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.13239 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.795931 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.68676 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.34558 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.02751 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.05563 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.712443 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.997682 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.85431 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.751199 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.3705 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.22848 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.14139 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.862469 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.894325 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.25077 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.858796 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.714778 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.46716 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.857908 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.948845 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.16389 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.929725 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.867383 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.936656 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.650944 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.31007 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.24193 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.618498 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.36808 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.14715 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.951588 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.686015 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.16096 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.25577 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.754358 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.755995 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.35835 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.656325 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.24002 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.757497 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.894126 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.840162 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.2991 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.796655 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.73016 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.873065 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.542968 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.3321 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.873788 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.653238 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.27983 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.651657 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.36429 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.67817 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.841973 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.948068 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.806396 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.00189 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.947797 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.23817 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.816562 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.66103 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.14651 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.903781 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.88183 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.682721 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.30377 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.09687 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.925102 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.886082 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.02464 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.971307 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.906952 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.972737 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.4071 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.968377 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.836598 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.788454 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.17924 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.985708 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.24829 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.865107 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.909437 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.954707 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.830851 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.30261 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.07152 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.29343 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.871557 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.00546 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.23665 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.03064 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.956564 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.60045 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.994624 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.969391 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.16535 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.08647 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.965955 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.995826 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.829466 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.28161 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.42362 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.728022 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.43365 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.14702 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.887408 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.782637 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.20802 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.20565 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.885705 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.868218 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.45048 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.757731 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.05394 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.731961 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.17489 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.05898 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.10234 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.694705 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.793184 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.835218 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.587039 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.41623 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.956786 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.735782 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.98875 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.725983 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.21313 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.17934 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.743019 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.20779 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.762327 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.961071 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.07681 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.972695 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.95398 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.726434 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.923556 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.19372 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.63739 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.69782 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.51435 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.13704 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.840727 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.737619 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.25414 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.48708 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.818102 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.77195 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.75026 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.68661 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.15562 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.681265 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.06522 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.794058 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.10948 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.747948 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.828257 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.803318 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.502774 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.21337 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.08199 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.672408 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.35199 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.569855 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.46774 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.63494 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.647657 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.945903 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.782675 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.926529 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.906072 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.38608 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.838717 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.654397 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.09464 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.24231 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 2.02673 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.732579 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.38617 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.12466 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.924462 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.912113 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.11569 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.9827 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.02872 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.856257 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.42579 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.03401 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.71767 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.749754 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.3145 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.00167 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.01307 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.759091 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.988096 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.89158 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.737947 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.19293 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.08347 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.03484 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.856558 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.964581 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.20866 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.949627 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.76966 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.43678 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.88213 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.888492 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.23462 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.905271 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.897352 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.845504 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.715414 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.13014 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.18701 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.625622 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.21991 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.05525 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.942728 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.904035 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.09175 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.06876 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.919069 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.880639 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.31799 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.949722 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.950566 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.788861 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.23244 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.04131 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.17112 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.816795 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.98511 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.967118 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.740539 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.25012 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.10057 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.96419 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.986829 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.790869 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.21356 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.945941 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.739015 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.17907 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.877419 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.987205 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.0616 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.996109 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.907652 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.872301 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.833023 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.08946 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.35862 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.785951 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.32963 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.131000091542601 z_k1_s0000
                                 - 0.199999787087963 z_k1_s0001
                                 - 0.228000285280243 z_k1_s0002
                                 - 0.153000090122275 z_k1_s0003
                                 - 0.167999689153897 z_k1_s0004
                                 - 0.0539999865137471 z_k1_s0005
                                 - 0.0509999567074298 z_k1_s0006
                                 - 0.114999492575603 z_k1_s0007
                                 - 0.0979999660837432 z_k1_s0010
                                 - 0.0780003414551564 z_k1_s0011
                                 - 0.0679999981397842 z_k1_s0012
                                 - 0.0350000865015372 z_k1_s0013
                                 - 0.0430000956532956 z_k1_s0014
                                 - 0.0870002787382362 z_k1_s0015
                                 - 0.0330000709385706 z_k1_s0016
                                 - 0.0890000288006634 z_k1_s0017
                                 - 0.050000113078566 z_k1_s0020
                                 - 0.0880002471143845 z_k1_s0021
                                 - 0.139000759247416 z_k1_s0022
                                 - 0.0780001643785358 z_k1_s0023
                                 - 0.234000593335833 z_k1_s0024
                                 - 0.152000736543726 z_k1_s0025
                                 - 0.153000734797288 z_k1_s0026
                                 - 0.107999811384727 z_k1_s0027
                                 - 0.251999539240161 z_k1_s0030
                                 - 0.526999929760926 z_k1_s0031
                                 - 0.241000810020408 z_k1_s0032
                                 - 0.172000185780212 z_k1_s0033
                                 - 0.469000393142459 z_k1_s0034
                                 - 0.119999994990732 z_k1_s0035
                                 - 0.249999185563352 z_k1_s0036
                                 - 0.11300004352298 z_k1_s0037
                                 - 0.13600034249237 z_k1_s0040
                                 - 0.179000075202964 z_k1_s0041
                                 - 0.123000341142444 z_k1_s0042
                                 - 0.182000018898956 z_k1_s0043
                                 - 0.253000649042454 z_k1_s0044
                                 - 0.0810002461962991 z_k1_s0045
                                 - 0.22300121208254 z_k1_s0046
                                 - 0.295001332790109 z_k1_s0047
                                 - 0.050000063329612 z_k1_s0050
                                 - 0.162000570864405 z_k1_s0051
                                 - 0.129000236132491 z_k1_s0052
                                 - 0.132000355926414 z_k1_s0053
                                 - 0.225000137533068 z_k1_s0054
                                 - 0.10800038057627 z_k1_s0055
                                 - 0.081000088831954 z_k1_s0056
                                 - 0.0960000980008252 z_k1_s0057
                                 - 0.214000689842227 z_k1_s0060
                                 - 0.0670001321355432 z_k1_s0061
                                 - 0.217000501680883 z_k1_s0062
                                 - 0.127000098513447 z_k1_s0063
                                 - 0.0490000515020559 z_k1_s0064
                                 - 0.066000094055862 z_k1_s0065
                                 - 0.13299972219076 z_k1_s0066
                                 - 0.135000604751154 z_k1_s0067
                                 - 0.240999428912344 z_k1_s0070
                                 - 0.288999135467475 z_k1_s0071
                                 - 0.320998939837562 z_k1_s0072
                                 - 0.345001199115456 z_k1_s0073
                                 - 0.213000081288583 z_k1_s0074
                                 - 0.194000137293835 z_k1_s0075
                                 - 0.355000536481026 z_k1_s0076
                                 - 0.447999667181552 z_k1_s0077
                                 - 0.257999858251883 z_k1_s0100
                                 - 0.63700031872143 z_k1_s0101
                                 - 0.233999827428483 z_k1_s0102
                                 - 0.305999919898684 z_k1_s0103
                                 - 0.284999656378374 z_k1_s0104
                                 - 0.691999732275515 z_k1_s0105
                                 - 0.198999913894266 z_k1_s0106
                                 - 0.542998568430918 z_k1_s0107
                                 - 0.916001358221532 z_k1_s0110
                                 - 0.368999898336123 z_k1_s0111
                                 - 0.264999838533251 z_k1_s0112
                                 - 0.919998735602161 z_k1_s0113
                                 - 0.443000976182496 z_k1_s0114
                                 - 0.730998663605599 z_k1_s0115
                                 - 0.405000033164051 z_k1_s0116
                                 - 0.19000005219178 z_k1_s0117
                                 - 0.258999751317975 z_k1_s0120
                                 - 0.19899998639791 z_k1_s0121
                                 - 0.709998629420206 z_k1_s0122
                                 - 0.252999835305945 z_k1_s0123
                                 - 0.434001283267282 z_k1_s0124
                                 - 0.462998407726412 z_k1_s0125
                                 - 0.475998578552338 z_k1_s0126
                                 - 0.229999905259928 z_k1_s0127
                                 - 0.14699993930366 z_k1_s0130
                                 - 0.325999978194443 z_k1_s0131
                                 - 0.132999945084264 z_k1_s0132
                                 - 0.336000256065267 z_k1_s0133
                                 - 0.351999516259312 z_k1_s0134
                                 - 0.233999790581406 z_k1_s0135
                                 - 0.208999095904345 z_k1_s0136
                                 - 0.237000268742411 z_k1_s0137
                                 - 0.945996080774714 z_k1_s0140
                                 - 0.815000491956341 z_k1_s0141
                                 - 1.30500023495595 z_k1_s0142
                                 - 0.748996735749073 z_k1_s0143
                                 - 1.02199431523998 z_k1_s0144
                                 - 1.32899529995537 z_k1_s0145
                                 - 0.448998816797807 z_k1_s0146
                                 - 0.31599948063936 z_k1_s0147
                                 - 0.347999835809958 z_k1_s0150
                                 - 0.594998724231766 z_k1_s0151
                                 - 0.440000555069555 z_k1_s0152
                                 - 0.362999939557727 z_k1_s0153
                                 - 0.760998877094408 z_k1_s0154
                                 - 0.5560003621063 z_k1_s0155
                                 - 0.463999228146977 z_k1_s0156
                                 - 0.620000970643339 z_k1_s0157
                                 - 0.149999945742668 z_k1_s0160
                                 - 0.383999891629201 z_k1_s0161
                                 - 0.131999891197607 z_k1_s0162
                                 - 0.461998855992365 z_k1_s0163
                                 - 0.104999809380016 z_k1_s0164
                                 - 0.154999854894177 z_k1_s0165
                                 - 0.246000109449783 z_k1_s0166
                                 - 0.273999982297861 z_k1_s0167
                                 - 2.01699960525606 z_k1_s0170
                                 - 1.11899646649844 z_k1_s0171
                                 - 0.870996362252907 z_k1_s0172
                                 - 0.743999456737336 z_k1_s0173
                                 - 0.740999179347509 z_k1_s0174
                                 - 0.922998317012385 z_k1_s0175
                                 - 1.60600012281629 z_k1_s0176
                                 - 0.538999619082548 z_k1_s0177
                                 - 0.32699912394192 z_k1_s1000
                                 - 0.375998892886368 z_k1_s1001
                                 - 0.520000567688754 z_k1_s1002
                                 - 0.625997079118783 z_k1_s1003
                                 - 0.354999154509776 z_k1_s1004
                                 - 0.80499755908402 z_k1_s1005
                                 - 0.508997994736249 z_k1_s1006
                                 - 0.275999163114521 z_k1_s1007
                                 - 1.05100085034475 z_k1_s1010
                                 - 0.303000355010666 z_k1_s1011
                                 - 1.08899773031299 z_k1_s1012
                                 - 0.968999562725487 z_k1_s1013
                                 - 0.95200265513519 z_k1_s1014
                                 - 0.754000403713457 z_k1_s1015
                                 - 0.311999924529598 z_k1_s1016
                                 - 0.430999963371366 z_k1_s1017
                                 - 0.255999471962363 z_k1_s1020
                                 - 0.170000148733125 z_k1_s1021
                                 - 0.442001116006573 z_k1_s1022
                                 - 0.249000332670942 z_k1_s1023
                                 - 0.469000247560699 z_k1_s1024
                                 - 0.160000152601773 z_k1_s1025
                                 - 0.184000143317019 z_k1_s1026
                                 - 0.21499991682408 z_k1_s1027
                                 - 0.404999174155685 z_k1_s1030
                                 - 0.835998841934149 z_k1_s1031
                                 - 1.06499949728029 z_k1_s1032
                                 - 0.398999135596438 z_k1_s1033
                                 - 0.995996604853619 z_k1_s1034
                                 - 0.452998992830592 z_k1_s1035
                                 - 0.361999224347128 z_k1_s1036
                                 - 0.381999189611595 z_k1_s1037
                                 - 0.282000128987223 z_k1_s1040
                                 - 0.316999910529579 z_k1_s1041
                                 - 0.2670001221262 z_k1_s1042
                                 - 1.24599939758879 z_k1_s1043
                                 - 0.40599901337153 z_k1_s1044
                                 - 0.40599901337153 z_k1_s1045
                                 - 0.340999921507215 z_k1_s1046
                                 - 0.252000115265178 z_k1_s1047
                                 - 0.131000080817679 z_k1_s1050
                                 - 0.148000126825359 z_k1_s1051
                                 - 0.146000192048069 z_k1_s1052
                                 - 0.247000350605325 z_k1_s1053
                                 - 0.238000343907162 z_k1_s1054
                                 - 0.162000120566925 z_k1_s1055
                                 - 0.229000337209 z_k1_s1056
                                 - 0.200000982737936 z_k1_s1057
                                 - 0.24100081274471 z_k1_s1060
                                 - 0.189000623413847 z_k1_s1061
                                 - 0.444000684467598 z_k1_s1062
                                 - 0.33700186137363 z_k1_s1063
                                 - 0.807001630702743 z_k1_s1064
                                 - 0.185000790096791 z_k1_s1065
                                 - 0.315001263423949 z_k1_s1066
                                 - 0.848003035764641 z_k1_s1067
                                 - 0.410000310883907 z_k1_s1070
                                 - 0.338001379685206 z_k1_s1071
                                 - 0.122000137187423 z_k1_s1072
                                 - 0.153000084098129 z_k1_s1073
                                 - 0.531999662971034 z_k1_s1074
                                 - 0.32599981315143 z_k1_s1075
                                 - 0.597000733526185 z_k1_s1076
                                 - 0.139000175609449 z_k1_s1077
                                 - 0.273999902911053 z_k1_s1100
                                 - 0.468000334388481 z_k1_s1101
                                 - 0.739999783263452 z_k1_s1102
                                 - 0.84300042660993 z_k1_s1103
                                 - 0.466000503352721 z_k1_s1104
                                 - 1.26100249310312 z_k1_s1105
                                 - 0.909997881309502 z_k1_s1106
                                 - 1.26799878672667 z_k1_s1107
                                 - 0.8119975949542 z_k1_s1110
                                 - 0.688997389284568 z_k1_s1111
                                 - 0.860001739039292 z_k1_s1112
                                 - 0.593999292448325 z_k1_s1113
                                 - 0.848998860918233 z_k1_s1114
                                 - 0.468999068383272 z_k1_s1115
                                 - 0.966000862699098 z_k1_s1116
                                 - 0.931998698679448 z_k1_s1117
                                 - 0.204000544464758 z_k1_s1120
                                 - 0.14599996528978 z_k1_s1121
                                 - 0.148999951193689 z_k1_s1122
                                 - 0.307999053800968 z_k1_s1123
                                 - 0.211000010574229 z_k1_s1124
                                 - 0.189000614945214 z_k1_s1125
                                 - 0.132999926172912 z_k1_s1126
                                 - 0.163000236111686 z_k1_s1127
                                 - 0.305000216444072 z_k1_s1130
                                 - 0.258000280241094 z_k1_s1131
                                 - 0.123999827975948 z_k1_s1132
                                 - 0.352000152647051 z_k1_s1133
                                 - 0.226000583272783 z_k1_s1134
                                 - 0.120999780129014 z_k1_s1135
                                 - 0.240000341759055 z_k1_s1136
                                 - 0.11699977443303 z_k1_s1137
                                 - 0.894000331653131 z_k1_s1140
                                 - 0.358999152878319 z_k1_s1141
                                 - 0.704000309601475 z_k1_s1142
                                 - 0.533000289754983 z_k1_s1143
                                 - 0.668000761211833 z_k1_s1144
                                 - 0.350999493791254 z_k1_s1145
                                 - 0.38799927019124 z_k1_s1146
                                 - 0.432999138677431 z_k1_s1147
                                 - 0.0939999920852492 z_k1_s1150
                                 - 0.187999984170498 z_k1_s1151
                                 - 0.236999727103344 z_k1_s1152
                                 - 0.224000471692298 z_k1_s1153
                                 - 0.176999394900138 z_k1_s1154
                                 - 0.259999518214717 z_k1_s1155
                                 - 0.203999400291642 z_k1_s1156
                                 - 0.0570000273931692 z_k1_s1157
                                 - 0.379999707052721 z_k1_s1160
                                 - 0.232999907876996 z_k1_s1161
                                 - 0.344999359034997 z_k1_s1162
                                 - 0.336000490972354 z_k1_s1163
                                 - 0.302999891412995 z_k1_s1164
                                 - 0.454999774234083 z_k1_s1165
                                 - 0.130999911510271 z_k1_s1166
                                 - 0.191999839484543 z_k1_s1167
                                 - 0.228000344387176 z_k1_s1170
                                 - 0.437002099243567 z_k1_s1171
                                 - 0.355000522444931 z_k1_s1172
                                 - 0.653999872319964 z_k1_s1173
                                 - 0.556000371575264 z_k1_s1174
                                 - 0.866001115259866 z_k1_s1175
                                 - 0.500002339008594 z_k1_s1176
                                 - 0.329000510716597 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.750543 w_k1_ADM_s000
                                 + 0.720815 w_k1_AXP_s000
                                 + 0.649065 w_k1_DIS_s000
                                 + 0.754904 w_k1_FDX_s000
                                 + 0.661212 w_k1_IBM_s000
                                 + 0.889658 w_k1_INTC_s000
                                 + 0.753715 w_k1_KO_s000
                                 + 0.454157 w_k1_MAT_s000
                                 + 1.01765 w_k1_MSFT_s000
                                 + 0.778202 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.994756 w_k1_ADM_s000
                                 + 1.11875 w_k1_AXP_s000
                                 + 0.904318 w_k1_DIS_s000
                                 + 1.08413 w_k1_FDX_s000
                                 + 0.728626 w_k1_IBM_s000
                                 + 0.929332 w_k1_INTC_s000
                                 + 0.964875 w_k1_KO_s000
                                 + 1.01126 w_k1_MAT_s000
                                 + 1.09773 w_k1_MSFT_s000
                                 + 1.05839 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.897535 w_k1_ADM_s000
                                 + 1.03064 w_k1_AXP_s000 + 1.0581 w_k1_DIS_s000
                                 + 1.05062 w_k1_FDX_s000
                                 + 1.16686 w_k1_IBM_s000
                                 + 0.920746 w_k1_INTC_s000
                                 + 1.09987 w_k1_KO_s000 + 1.0874 w_k1_MAT_s000
                                 + 1.08778 w_k1_MSFT_s000
                                 + 1.20013 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.99275 w_k1_ADM_s000
                                 + 0.937421 w_k1_AXP_s000
                                 + 0.668447 w_k1_DIS_s000
                                 + 0.805683 w_k1_FDX_s000
                                 + 0.825552 w_k1_IBM_s000
                                 + 0.899959 w_k1_INTC_s000
                                 + 0.855382 w_k1_KO_s000
                                 + 0.564514 w_k1_MAT_s000
                                 + 0.973278 w_k1_MSFT_s000
                                 + 0.972799 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.926129 w_k1_ADM_s000
                                 + 0.985024 w_k1_AXP_s000
                                 + 0.946903 w_k1_DIS_s000
                                 + 0.879204 w_k1_FDX_s000
                                 + 1.08705 w_k1_IBM_s000
                                 + 0.916947 w_k1_INTC_s000
                                 + 0.967453 w_k1_KO_s000
                                 + 0.792703 w_k1_MAT_s000
                                 + 0.980742 w_k1_MSFT_s000
                                 + 1.07681 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.692589 w_k1_ADM_s000
                                 + 1.01277 w_k1_AXP_s000
                                 + 1.10723 w_k1_DIS_s000
                                 + 0.855165 w_k1_FDX_s000 + 1.522 w_k1_IBM_s000
                                 + 0.991765 w_k1_INTC_s000
                                 + 1.03243 w_k1_KO_s000 + 1.25966 w_k1_MAT_s000
                                 + 1.04304 w_k1_MSFT_s000
                                 + 1.07211 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.29068 w_k1_ADM_s000
                                 + 1.75923 w_k1_AXP_s000
                                 + 1.75433 w_k1_DIS_s000
                                 + 1.48158 w_k1_FDX_s000
                                 + 1.40221 w_k1_IBM_s000
                                 + 1.34185 w_k1_INTC_s000
                                 + 1.47796 w_k1_KO_s000 + 1.56978 w_k1_MAT_s000
                                 + 1.6553 w_k1_MSFT_s000 + 1.45775 w_k1_PG_s000
                                 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.17038 w_k1_ADM_s000
                                 + 1.09227 w_k1_AXP_s000
                                 + 1.41082 w_k1_DIS_s000
                                 + 1.23543 w_k1_FDX_s000
                                 + 1.15264 w_k1_IBM_s000
                                 + 1.26805 w_k1_INTC_s000
                                 + 1.35899 w_k1_KO_s000 + 0.75931 w_k1_MAT_s000
                                 + 1.5532 w_k1_MSFT_s000 + 1.37025 w_k1_PG_s000
                                 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.987456 w_k1_ADM_s001
                                 + 0.646629 w_k1_AXP_s001
                                 + 0.990767 w_k1_DIS_s001
                                 + 0.922507 w_k1_FDX_s001
                                 + 0.753179 w_k1_IBM_s001
                                 + 0.906344 w_k1_INTC_s001
                                 + 0.888513 w_k1_KO_s001
                                 + 0.536712 w_k1_MAT_s001
                                 + 1.00846 w_k1_MSFT_s001
                                 + 0.932668 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.813382 w_k1_ADM_s001
                                 + 1.42801 w_k1_AXP_s001
                                 + 0.883602 w_k1_DIS_s001
                                 + 1.0453 w_k1_FDX_s001
                                 + 0.914766 w_k1_IBM_s001
                                 + 1.0224 w_k1_INTC_s001
                                 + 0.892437 w_k1_KO_s001
                                 + 0.925394 w_k1_MAT_s001
                                 + 0.861462 w_k1_MSFT_s001
                                 + 0.883443 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.1329 w_k1_ADM_s001
                                 + 0.820951 w_k1_AXP_s001
                                 + 1.09521 w_k1_DIS_s001
                                 + 1.00381 w_k1_FDX_s001
                                 + 1.08794 w_k1_IBM_s001
                                 + 0.80939 w_k1_INTC_s001
                                 + 1.08221 w_k1_KO_s001
                                 + 0.636538 w_k1_MAT_s001
                                 + 0.924447 w_k1_MSFT_s001
                                 + 1.08726 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.737231 w_k1_ADM_s001
                                 + 0.897903 w_k1_AXP_s001
                                 + 1.01636 w_k1_DIS_s001
                                 + 0.811505 w_k1_FDX_s001
                                 + 1.4084 w_k1_IBM_s001
                                 + 0.733559 w_k1_INTC_s001
                                 + 1.07955 w_k1_KO_s001 + 1.03838 w_k1_MAT_s001
                                 + 0.822462 w_k1_MSFT_s001
                                 + 0.842233 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.10671 w_k1_ADM_s001
                                 + 1.0596 w_k1_AXP_s001 + 0.80334 w_k1_DIS_s001
                                 + 0.932677 w_k1_FDX_s001
                                 + 0.611926 w_k1_IBM_s001
                                 + 0.915605 w_k1_INTC_s001
                                 + 0.817829 w_k1_KO_s001
                                 + 0.699704 w_k1_MAT_s001
                                 + 0.927181 w_k1_MSFT_s001
                                 + 1.02446 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.21557 w_k1_ADM_s001
                                 + 1.32185 w_k1_AXP_s001
                                 + 1.49496 w_k1_DIS_s001
                                 + 1.19889 w_k1_FDX_s001
                                 + 1.26886 w_k1_IBM_s001
                                 + 1.21588 w_k1_INTC_s001
                                 + 1.26346 w_k1_KO_s001 + 1.02379 w_k1_MAT_s001
                                 + 1.06593 w_k1_MSFT_s001 + 1.2556 w_k1_PG_s001
                                 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.688506 w_k1_ADM_s001
                                 + 0.654963 w_k1_AXP_s001
                                 + 0.640797 w_k1_DIS_s001
                                 + 0.744657 w_k1_FDX_s001
                                 + 0.625827 w_k1_IBM_s001
                                 + 0.950161 w_k1_INTC_s001
                                 + 0.688036 w_k1_KO_s001
                                 + 0.342807 w_k1_MAT_s001
                                 + 0.809107 w_k1_MSFT_s001
                                 + 0.823725 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.22355 w_k1_ADM_s001
                                 + 1.46918 w_k1_AXP_s001
                                 + 1.48983 w_k1_DIS_s001
                                 + 1.42278 w_k1_FDX_s001
                                 + 1.07494 w_k1_IBM_s001
                                 + 0.8488 w_k1_INTC_s001 + 1.1434 w_k1_KO_s001
                                 + 1.33554 w_k1_MAT_s001
                                 + 1.10362 w_k1_MSFT_s001
                                 + 1.16147 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.905305 w_k1_ADM_s002
                                 + 0.780878 w_k1_AXP_s002
                                 + 0.641003 w_k1_DIS_s002
                                 + 0.67913 w_k1_FDX_s002
                                 + 0.848101 w_k1_IBM_s002
                                 + 0.772343 w_k1_INTC_s002
                                 + 0.824115 w_k1_KO_s002
                                 + 0.758999 w_k1_MAT_s002
                                 + 0.728255 w_k1_MSFT_s002
                                 + 0.783161 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.714224 w_k1_ADM_s002
                                 + 0.853358 w_k1_AXP_s002
                                 + 0.780686 w_k1_DIS_s002
                                 + 0.818509 w_k1_FDX_s002
                                 + 1.1902 w_k1_IBM_s002
                                 + 0.925788 w_k1_INTC_s002
                                 + 0.920533 w_k1_KO_s002
                                 + 0.873479 w_k1_MAT_s002
                                 + 0.725788 w_k1_MSFT_s002
                                 + 0.992627 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.97653 w_k1_ADM_s002
                                 + 1.52183 w_k1_AXP_s002
                                 + 1.28748 w_k1_DIS_s002
                                 + 1.09819 w_k1_FDX_s002
                                 + 1.63951 w_k1_IBM_s002
                                 + 1.14116 w_k1_INTC_s002
                                 + 1.28242 w_k1_KO_s002 + 1.87926 w_k1_MAT_s002
                                 + 1.00137 w_k1_MSFT_s002
                                 + 1.03617 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.34926 w_k1_ADM_s002
                                 + 1.69914 w_k1_AXP_s002
                                 + 1.80361 w_k1_DIS_s002
                                 + 1.42621 w_k1_FDX_s002
                                 + 1.26851 w_k1_IBM_s002
                                 + 1.36606 w_k1_INTC_s002
                                 + 1.65218 w_k1_KO_s002 + 1.57688 w_k1_MAT_s002
                                 + 1.39691 w_k1_MSFT_s002
                                 + 1.26738 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.906668 w_k1_ADM_s002
                                 + 1.22367 w_k1_AXP_s002
                                 + 1.35353 w_k1_DIS_s002
                                 + 1.16013 w_k1_FDX_s002 + 1.0097 w_k1_IBM_s002
                                 + 1.10895 w_k1_INTC_s002
                                 + 1.19531 w_k1_KO_s002 + 1.13165 w_k1_MAT_s002
                                 + 1.24572 w_k1_MSFT_s002
                                 + 1.12421 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.942208 w_k1_ADM_s002
                                 + 0.79975 w_k1_AXP_s002
                                 + 0.908785 w_k1_DIS_s002
                                 + 0.935794 w_k1_FDX_s002
                                 + 0.655283 w_k1_IBM_s002
                                 + 0.923725 w_k1_INTC_s002
                                 + 0.957746 w_k1_KO_s002
                                 + 0.639181 w_k1_MAT_s002
                                 + 0.908262 w_k1_MSFT_s002
                                 + 0.946993 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.05485 w_k1_ADM_s002
                                 + 1.00398 w_k1_AXP_s002
                                 + 0.878896 w_k1_DIS_s002
                                 + 0.911192 w_k1_FDX_s002
                                 + 1.06543 w_k1_IBM_s002
                                 + 1.00122 w_k1_INTC_s002
                                 + 1.05985 w_k1_KO_s002 + 1.12152 w_k1_MAT_s002
                                 + 0.902186 w_k1_MSFT_s002
                                 + 0.925777 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.877581 w_k1_ADM_s002
                                 + 1.04423 w_k1_AXP_s002
                                 + 0.943104 w_k1_DIS_s002
                                 + 0.996782 w_k1_FDX_s002
                                 + 1.02345 w_k1_IBM_s002
                                 + 1.15662 w_k1_INTC_s002
                                 + 0.881062 w_k1_KO_s002
                                 + 0.86048 w_k1_MAT_s002
                                 + 0.994618 w_k1_MSFT_s002
                                 + 0.971547 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.10741 w_k1_ADM_s003
                                 + 0.826689 w_k1_AXP_s003
                                 + 0.740466 w_k1_DIS_s003
                                 + 0.799286 w_k1_FDX_s003
                                 + 0.966107 w_k1_IBM_s003
                                 + 0.908416 w_k1_INTC_s003
                                 + 0.775845 w_k1_KO_s003
                                 + 0.77253 w_k1_MAT_s003
                                 + 0.996763 w_k1_MSFT_s003
                                 + 0.957217 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.47039 w_k1_ADM_s003
                                 + 1.06364 w_k1_AXP_s003
                                 + 1.04585 w_k1_DIS_s003
                                 + 1.10263 w_k1_FDX_s003
                                 + 0.871422 w_k1_IBM_s003
                                 + 1.14984 w_k1_INTC_s003
                                 + 0.894455 w_k1_KO_s003
                                 + 0.738091 w_k1_MAT_s003
                                 + 1.14765 w_k1_MSFT_s003
                                 + 1.01353 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.28832 w_k1_ADM_s003
                                 + 0.943197 w_k1_AXP_s003
                                 + 0.820395 w_k1_DIS_s003
                                 + 0.967684 w_k1_FDX_s003
                                 + 0.651274 w_k1_IBM_s003
                                 + 1.05874 w_k1_INTC_s003
                                 + 0.754432 w_k1_KO_s003
                                 + 0.70923 w_k1_MAT_s003
                                 + 1.05075 w_k1_MSFT_s003
                                 + 0.848035 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.947595 w_k1_ADM_s003
                                 + 1.11583 w_k1_AXP_s003
                                 + 1.04623 w_k1_DIS_s003
                                 + 0.917104 w_k1_FDX_s003
                                 + 1.68715 w_k1_IBM_s003
                                 + 1.0208 w_k1_INTC_s003 + 1.15266 w_k1_KO_s003
                                 + 1.6787 w_k1_MAT_s003
                                 + 0.965786 w_k1_MSFT_s003
                                 + 1.00048 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.45485 w_k1_ADM_s003
                                 + 1.49795 w_k1_AXP_s003
                                 + 1.53141 w_k1_DIS_s003
                                 + 1.25282 w_k1_FDX_s003
                                 + 1.19167 w_k1_IBM_s003
                                 + 1.50629 w_k1_INTC_s003
                                 + 1.35041 w_k1_KO_s003 + 1.20331 w_k1_MAT_s003
                                 + 1.55189 w_k1_MSFT_s003 + 1.2415 w_k1_PG_s003
                                 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.932178 w_k1_ADM_s003
                                 + 0.641172 w_k1_AXP_s003
                                 + 0.561901 w_k1_DIS_s003
                                 + 0.680487 w_k1_FDX_s003
                                 + 0.715709 w_k1_IBM_s003
                                 + 0.649921 w_k1_INTC_s003
                                 + 0.600131 w_k1_KO_s003
                                 + 0.760305 w_k1_MAT_s003
                                 + 0.805054 w_k1_MSFT_s003
                                 + 0.791997 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.18097 w_k1_ADM_s003
                                 + 1.04626 w_k1_AXP_s003
                                 + 1.10074 w_k1_DIS_s003
                                 + 1.03189 w_k1_FDX_s003
                                 + 1.10536 w_k1_IBM_s003
                                 + 0.978982 w_k1_INTC_s003
                                 + 1.10059 w_k1_KO_s003 + 1.10703 w_k1_MAT_s003
                                 + 1.20955 w_k1_MSFT_s003
                                 + 1.03359 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.01656 w_k1_ADM_s003
                                 + 0.73403 w_k1_AXP_s003
                                 + 0.596688 w_k1_DIS_s003
                                 + 0.714872 w_k1_FDX_s003
                                 + 1.14503 w_k1_IBM_s003
                                 + 0.759218 w_k1_INTC_s003
                                 + 0.775426 w_k1_KO_s003
                                 + 1.14793 w_k1_MAT_s003
                                 + 0.949273 w_k1_MSFT_s003
                                 + 0.850248 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.777678 w_k1_ADM_s004
                                 + 0.733769 w_k1_AXP_s004
                                 + 0.593226 w_k1_DIS_s004
                                 + 0.764568 w_k1_FDX_s004
                                 + 0.696932 w_k1_IBM_s004
                                 + 0.828406 w_k1_INTC_s004
                                 + 0.693222 w_k1_KO_s004
                                 + 0.658583 w_k1_MAT_s004
                                 + 0.764812 w_k1_MSFT_s004
                                 + 0.817482 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.21143 w_k1_ADM_s004
                                 + 1.2456 w_k1_AXP_s004 + 0.95022 w_k1_DIS_s004
                                 + 1.14112 w_k1_FDX_s004
                                 + 0.905333 w_k1_IBM_s004
                                 + 0.982727 w_k1_INTC_s004
                                 + 0.895283 w_k1_KO_s004 + 1.0717 w_k1_MAT_s004
                                 + 0.928422 w_k1_MSFT_s004
                                 + 1.00768 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.852623 w_k1_ADM_s004
                                 + 0.743426 w_k1_AXP_s004
                                 + 0.626536 w_k1_DIS_s004
                                 + 0.753314 w_k1_FDX_s004
                                 + 0.950853 w_k1_IBM_s004
                                 + 0.841038 w_k1_INTC_s004
                                 + 0.757449 w_k1_KO_s004
                                 + 0.777408 w_k1_MAT_s004
                                 + 0.776449 w_k1_MSFT_s004
                                 + 0.904267 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.781381 w_k1_ADM_s004
                                 + 0.902717 w_k1_AXP_s004
                                 + 0.868841 w_k1_DIS_s004
                                 + 0.801733 w_k1_FDX_s004
                                 + 1.31197 w_k1_IBM_s004
                                 + 0.961994 w_k1_INTC_s004
                                 + 1.0146 w_k1_KO_s004 + 1.1353 w_k1_MAT_s004
                                 + 0.866048 w_k1_MSFT_s004
                                 + 0.973139 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.27953 w_k1_ADM_s004
                                 + 1.61603 w_k1_AXP_s004 + 1.3281 w_k1_DIS_s004
                                 + 1.27903 w_k1_FDX_s004
                                 + 1.48821 w_k1_IBM_s004
                                 + 1.23445 w_k1_INTC_s004
                                 + 1.22646 w_k1_KO_s004 + 1.86852 w_k1_MAT_s004
                                 + 1.09355 w_k1_MSFT_s004
                                 + 1.12715 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.916324 w_k1_ADM_s004
                                 + 0.847397 w_k1_AXP_s004
                                 + 0.995977 w_k1_DIS_s004
                                 + 1.12671 w_k1_FDX_s004
                                 + 0.587273 w_k1_IBM_s004
                                 + 1.13517 w_k1_INTC_s004
                                 + 0.817315 w_k1_KO_s004
                                 + 0.631106 w_k1_MAT_s004
                                 + 1.07518 w_k1_MSFT_s004
                                 + 0.998894 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.58856 w_k1_ADM_s004
                                 + 1.24009 w_k1_AXP_s004
                                 + 1.28664 w_k1_DIS_s004
                                 + 1.27935 w_k1_FDX_s004
                                 + 0.966601 w_k1_IBM_s004
                                 + 0.984338 w_k1_INTC_s004
                                 + 1.15934 w_k1_KO_s004
                                 + 0.948703 w_k1_MAT_s004
                                 + 1.12053 w_k1_MSFT_s004
                                 + 1.17511 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.992965 w_k1_ADM_s004
                                 + 0.808571 w_k1_AXP_s004
                                 + 0.980844 w_k1_DIS_s004
                                 + 1.04917 w_k1_FDX_s004
                                 + 0.985471 w_k1_IBM_s004
                                 + 1.04268 w_k1_INTC_s004
                                 + 1.01507 w_k1_KO_s004
                                 + 0.690892 w_k1_MAT_s004
                                 + 1.0381 w_k1_MSFT_s004 + 0.99423 w_k1_PG_s004
                                 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.839184 w_k1_ADM_s005
                                 + 0.587062 w_k1_AXP_s005
                                 + 0.686424 w_k1_DIS_s005
                                 + 0.700726 w_k1_FDX_s005
                                 + 0.746146 w_k1_IBM_s005
                                 + 0.731869 w_k1_INTC_s005
                                 + 0.785366 w_k1_KO_s005
                                 + 0.614784 w_k1_MAT_s005
                                 + 0.703427 w_k1_MSFT_s005
                                 + 0.747546 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.22756 w_k1_ADM_s005
                                 + 1.22569 w_k1_AXP_s005
                                 + 1.22569 w_k1_DIS_s005
                                 + 1.04302 w_k1_FDX_s005
                                 + 1.67709 w_k1_IBM_s005
                                 + 1.31456 w_k1_INTC_s005
                                 + 1.28882 w_k1_KO_s005 + 1.05315 w_k1_MAT_s005
                                 + 0.881838 w_k1_MSFT_s005
                                 + 1.02251 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.25074 w_k1_ADM_s005
                                 + 1.44934 w_k1_AXP_s005
                                 + 1.36371 w_k1_DIS_s005
                                 + 1.27967 w_k1_FDX_s005
                                 + 1.18991 w_k1_IBM_s005
                                 + 1.16919 w_k1_INTC_s005
                                 + 1.09169 w_k1_KO_s005 + 1.15357 w_k1_MAT_s005
                                 + 1.16462 w_k1_MSFT_s005
                                 + 1.20522 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.909235 w_k1_ADM_s005
                                 + 1.01604 w_k1_AXP_s005
                                 + 1.04157 w_k1_DIS_s005
                                 + 1.13789 w_k1_FDX_s005
                                 + 0.866081 w_k1_IBM_s005
                                 + 1.27308 w_k1_INTC_s005
                                 + 0.868692 w_k1_KO_s005
                                 + 0.760824 w_k1_MAT_s005
                                 + 1.05149 w_k1_MSFT_s005
                                 + 0.943202 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.964243 w_k1_ADM_s005
                                 + 1.00588 w_k1_AXP_s005
                                 + 0.936137 w_k1_DIS_s005
                                 + 0.9672 w_k1_FDX_s005 + 1.03022 w_k1_IBM_s005
                                 + 0.868547 w_k1_INTC_s005
                                 + 0.835332 w_k1_KO_s005
                                 + 1.00656 w_k1_MAT_s005
                                 + 0.84628 w_k1_MSFT_s005
                                 + 0.961358 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.858903 w_k1_ADM_s005
                                 + 0.786988 w_k1_AXP_s005
                                 + 0.755848 w_k1_DIS_s005
                                 + 0.861066 w_k1_FDX_s005
                                 + 0.77063 w_k1_IBM_s005
                                 + 0.821946 w_k1_INTC_s005
                                 + 0.706139 w_k1_KO_s005
                                 + 0.618687 w_k1_MAT_s005
                                 + 0.823502 w_k1_MSFT_s005
                                 + 0.929143 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.880281 w_k1_ADM_s005
                                 + 0.700359 w_k1_AXP_s005
                                 + 0.673209 w_k1_DIS_s005
                                 + 0.812741 w_k1_FDX_s005
                                 + 1.17806 w_k1_IBM_s005
                                 + 0.928988 w_k1_INTC_s005
                                 + 0.866989 w_k1_KO_s005
                                 + 0.862186 w_k1_MAT_s005
                                 + 0.891493 w_k1_MSFT_s005
                                 + 0.783536 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.50949 w_k1_ADM_s005
                                 + 1.21686 w_k1_AXP_s005
                                 + 1.42017 w_k1_DIS_s005
                                 + 1.50097 w_k1_FDX_s005
                                 + 0.888746 w_k1_IBM_s005
                                 + 1.29891 w_k1_INTC_s005
                                 + 0.84592 w_k1_KO_s005
                                 + 0.790343 w_k1_MAT_s005
                                 + 1.21521 w_k1_MSFT_s005
                                 + 1.01344 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.976918 w_k1_ADM_s006
                                 + 0.792329 w_k1_AXP_s006
                                 + 1.01672 w_k1_DIS_s006
                                 + 0.898042 w_k1_FDX_s006
                                 + 1.03944 w_k1_IBM_s006
                                 + 1.08336 w_k1_INTC_s006
                                 + 0.986125 w_k1_KO_s006
                                 + 0.997102 w_k1_MAT_s006
                                 + 1.1163 w_k1_MSFT_s006
                                 + 0.952674 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.15785 w_k1_ADM_s006
                                 + 1.57854 w_k1_AXP_s006
                                 + 1.60732 w_k1_DIS_s006
                                 + 1.43802 w_k1_FDX_s006
                                 + 1.39223 w_k1_IBM_s006
                                 + 1.27249 w_k1_INTC_s006
                                 + 1.06327 w_k1_KO_s006
                                 + 0.961134 w_k1_MAT_s006
                                 + 1.23089 w_k1_MSFT_s006
                                 + 0.97089 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.737942 w_k1_ADM_s006
                                 + 1.01107 w_k1_AXP_s006
                                 + 0.68917 w_k1_DIS_s006
                                 + 0.880793 w_k1_FDX_s006
                                 + 1.06081 w_k1_IBM_s006
                                 + 1.0114 w_k1_INTC_s006
                                 + 0.815122 w_k1_KO_s006
                                 + 1.23945 w_k1_MAT_s006
                                 + 0.783988 w_k1_MSFT_s006
                                 + 0.948163 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.99296 w_k1_ADM_s006
                                 + 1.05284 w_k1_AXP_s006
                                 + 1.31712 w_k1_DIS_s006 + 1.1231 w_k1_FDX_s006
                                 + 1.2223 w_k1_IBM_s006
                                 + 1.11592 w_k1_INTC_s006
                                 + 1.10949 w_k1_KO_s006 + 1.68623 w_k1_MAT_s006
                                 + 1.16172 w_k1_MSFT_s006
                                 + 1.11395 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.03117 w_k1_ADM_s006
                                 + 0.925493 w_k1_AXP_s006
                                 + 1.52516 w_k1_DIS_s006
                                 + 1.24555 w_k1_FDX_s006
                                 + 0.913207 w_k1_IBM_s006
                                 + 1.39826 w_k1_INTC_s006
                                 + 0.969353 w_k1_KO_s006
                                 + 0.725523 w_k1_MAT_s006
                                 + 1.39348 w_k1_MSFT_s006
                                 + 1.14879 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.05805 w_k1_ADM_s006
                                 + 1.22332 w_k1_AXP_s006
                                 + 1.04754 w_k1_DIS_s006
                                 + 1.00401 w_k1_FDX_s006
                                 + 1.70377 w_k1_IBM_s006
                                 + 1.36382 w_k1_INTC_s006
                                 + 1.10937 w_k1_KO_s006 + 1.5146 w_k1_MAT_s006
                                 + 1.12672 w_k1_MSFT_s006
                                 + 1.07052 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.79714 w_k1_ADM_s006
                                 + 0.684324 w_k1_AXP_s006
                                 + 0.794884 w_k1_DIS_s006
                                 + 0.837613 w_k1_FDX_s006
                                 + 0.686609 w_k1_IBM_s006
                                 + 1.05797 w_k1_INTC_s006
                                 + 0.788799 w_k1_KO_s006 + 0.66 w_k1_MAT_s006
                                 + 0.933803 w_k1_MSFT_s006
                                 + 0.789467 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.877435 w_k1_ADM_s006
                                 + 1.02907 w_k1_AXP_s006
                                 + 1.10291 w_k1_DIS_s006
                                 + 1.10554 w_k1_FDX_s006
                                 + 0.80385 w_k1_IBM_s006
                                 + 1.24331 w_k1_INTC_s006
                                 + 0.870188 w_k1_KO_s006
                                 + 0.942884 w_k1_MAT_s006
                                 + 1.26045 w_k1_MSFT_s006
                                 + 0.93673 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.38133 w_k1_ADM_s007
                                 + 1.02475 w_k1_AXP_s007
                                 + 1.40757 w_k1_DIS_s007
                                 + 1.15032 w_k1_FDX_s007 + 1.1048 w_k1_IBM_s007
                                 + 0.839699 w_k1_INTC_s007
                                 + 1.45653 w_k1_KO_s007 + 1.16823 w_k1_MAT_s007
                                 + 1.14334 w_k1_MSFT_s007
                                 + 1.09641 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.935727 w_k1_ADM_s007
                                 + 0.682964 w_k1_AXP_s007
                                 + 0.690123 w_k1_DIS_s007
                                 + 0.854339 w_k1_FDX_s007
                                 + 0.668673 w_k1_IBM_s007
                                 + 0.675018 w_k1_INTC_s007
                                 + 0.738808 w_k1_KO_s007
                                 + 0.664425 w_k1_MAT_s007
                                 + 0.836645 w_k1_MSFT_s007
                                 + 0.783502 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.16337 w_k1_ADM_s007
                                 + 0.956644 w_k1_AXP_s007
                                 + 1.03015 w_k1_DIS_s007
                                 + 1.06907 w_k1_FDX_s007
                                 + 0.969312 w_k1_IBM_s007
                                 + 1.00244 w_k1_INTC_s007
                                 + 1.12088 w_k1_KO_s007 + 0.99548 w_k1_MAT_s007
                                 + 0.976103 w_k1_MSFT_s007
                                 + 0.897846 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.11483 w_k1_ADM_s007
                                 + 1.04664 w_k1_AXP_s007
                                 + 1.04492 w_k1_DIS_s007
                                 + 0.938736 w_k1_FDX_s007
                                 + 1.35295 w_k1_IBM_s007
                                 + 0.927171 w_k1_INTC_s007
                                 + 1.28618 w_k1_KO_s007 + 1.53623 w_k1_MAT_s007
                                 + 1.05231 w_k1_MSFT_s007
                                 + 1.09309 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.38849 w_k1_ADM_s007
                                 + 1.4459 w_k1_AXP_s007 + 1.6399 w_k1_DIS_s007
                                 + 1.44586 w_k1_FDX_s007
                                 + 1.28725 w_k1_IBM_s007
                                 + 1.20254 w_k1_INTC_s007
                                 + 1.47345 w_k1_KO_s007 + 1.84159 w_k1_MAT_s007
                                 + 1.61342 w_k1_MSFT_s007
                                 + 1.17057 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.95141 w_k1_ADM_s007
                                 + 1.21979 w_k1_AXP_s007
                                 + 0.856923 w_k1_DIS_s007
                                 + 0.941691 w_k1_FDX_s007
                                 + 1.11581 w_k1_IBM_s007
                                 + 0.799495 w_k1_INTC_s007
                                 + 1.06802 w_k1_KO_s007 + 1.6034 w_k1_MAT_s007
                                 + 0.849727 w_k1_MSFT_s007
                                 + 0.92807 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.2646 w_k1_ADM_s007
                                 + 1.19366 w_k1_AXP_s007
                                 + 1.08511 w_k1_DIS_s007
                                 + 1.24735 w_k1_FDX_s007
                                 + 0.690899 w_k1_IBM_s007
                                 + 1.18102 w_k1_INTC_s007
                                 + 1.10788 w_k1_KO_s007
                                 + 0.947282 w_k1_MAT_s007
                                 + 1.36053 w_k1_MSFT_s007
                                 + 1.03267 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.992792 w_k1_ADM_s007
                                 + 0.773965 w_k1_AXP_s007
                                 + 0.867768 w_k1_DIS_s007
                                 + 0.883801 w_k1_FDX_s007
                                 + 0.801707 w_k1_IBM_s007
                                 + 0.763424 w_k1_INTC_s007
                                 + 0.922935 w_k1_KO_s007
                                 + 0.984448 w_k1_MAT_s007
                                 + 1.03657 w_k1_MSFT_s007
                                 + 0.896773 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.942569 w_k1_ADM_s010
                                 + 0.946471 w_k1_AXP_s010
                                 + 1.33492 w_k1_DIS_s010
                                 + 1.16315 w_k1_FDX_s010
                                 + 1.17483 w_k1_IBM_s010
                                 + 0.994771 w_k1_INTC_s010
                                 + 1.24617 w_k1_KO_s010 + 1.23078 w_k1_MAT_s010
                                 + 1.17465 w_k1_MSFT_s010 + 1.2574 w_k1_PG_s010
                                 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.12106 w_k1_ADM_s010
                                 + 1.21042 w_k1_AXP_s010
                                 + 1.13405 w_k1_DIS_s010
                                 + 1.14954 w_k1_FDX_s010
                                 + 0.880977 w_k1_IBM_s010
                                 + 0.966155 w_k1_INTC_s010
                                 + 0.9717 w_k1_KO_s010 + 0.772755 w_k1_MAT_s010
                                 + 0.956801 w_k1_MSFT_s010
                                 + 1.05201 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.51842 w_k1_ADM_s010
                                 + 1.33633 w_k1_AXP_s010
                                 + 1.21439 w_k1_DIS_s010
                                 + 1.31233 w_k1_FDX_s010
                                 + 1.19368 w_k1_IBM_s010
                                 + 1.00839 w_k1_INTC_s010
                                 + 1.32607 w_k1_KO_s010 + 1.06101 w_k1_MAT_s010
                                 + 1.1428 w_k1_MSFT_s010 + 1.18337 w_k1_PG_s010
                                 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.975103 w_k1_ADM_s010
                                 + 1.00245 w_k1_AXP_s010
                                 + 0.983954 w_k1_DIS_s010
                                 + 0.998242 w_k1_FDX_s010
                                 + 1.51125 w_k1_IBM_s010
                                 + 0.893058 w_k1_INTC_s010
                                 + 1.08895 w_k1_KO_s010 + 1.48416 w_k1_MAT_s010
                                 + 1.02256 w_k1_MSFT_s010
                                 + 1.01301 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.744446 w_k1_ADM_s010
                                 + 0.666772 w_k1_AXP_s010
                                 + 0.544894 w_k1_DIS_s010
                                 + 0.659555 w_k1_FDX_s010
                                 + 1.01476 w_k1_IBM_s010
                                 + 0.881559 w_k1_INTC_s010
                                 + 0.947045 w_k1_KO_s010
                                 + 0.728835 w_k1_MAT_s010
                                 + 0.740613 w_k1_MSFT_s010
                                 + 0.760257 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.905874 w_k1_ADM_s010
                                 + 0.743627 w_k1_AXP_s010
                                 + 0.856758 w_k1_DIS_s010
                                 + 0.926736 w_k1_FDX_s010
                                 + 0.689254 w_k1_IBM_s010
                                 + 0.884115 w_k1_INTC_s010
                                 + 0.825242 w_k1_KO_s010
                                 + 0.598423 w_k1_MAT_s010
                                 + 1.02192 w_k1_MSFT_s010
                                 + 0.897405 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.18413 w_k1_ADM_s010
                                 + 1.48989 w_k1_AXP_s010
                                 + 1.74769 w_k1_DIS_s010
                                 + 1.55326 w_k1_FDX_s010
                                 + 1.20837 w_k1_IBM_s010
                                 + 1.50865 w_k1_INTC_s010 + 1.5521 w_k1_KO_s010
                                 + 1.20143 w_k1_MAT_s010
                                 + 1.5084 w_k1_MSFT_s010 + 1.15864 w_k1_PG_s010
                                 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.06243 w_k1_ADM_s010
                                 + 0.92059 w_k1_AXP_s010
                                 + 0.777031 w_k1_DIS_s010
                                 + 0.904552 w_k1_FDX_s010
                                 + 1.06585 w_k1_IBM_s010
                                 + 0.812764 w_k1_INTC_s010
                                 + 0.943837 w_k1_KO_s010
                                 + 0.859732 w_k1_MAT_s010
                                 + 0.801258 w_k1_MSFT_s010
                                 + 0.925216 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.998993 w_k1_ADM_s011
                                 + 0.899856 w_k1_AXP_s011
                                 + 0.802607 w_k1_DIS_s011
                                 + 0.840352 w_k1_FDX_s011
                                 + 1.03462 w_k1_IBM_s011
                                 + 0.877808 w_k1_INTC_s011
                                 + 0.919064 w_k1_KO_s011
                                 + 0.846244 w_k1_MAT_s011
                                 + 0.780202 w_k1_MSFT_s011
                                 + 0.894919 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.45839 w_k1_ADM_s011
                                 + 1.00462 w_k1_AXP_s011
                                 + 1.09146 w_k1_DIS_s011
                                 + 1.13757 w_k1_FDX_s011
                                 + 0.961181 w_k1_IBM_s011
                                 + 1.23768 w_k1_INTC_s011
                                 + 0.994839 w_k1_KO_s011
                                 + 0.786433 w_k1_MAT_s011
                                 + 0.941664 w_k1_MSFT_s011
                                 + 1.01384 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.45032 w_k1_ADM_s011
                                 + 1.13233 w_k1_AXP_s011
                                 + 1.38112 w_k1_DIS_s011
                                 + 1.29933 w_k1_FDX_s011
                                 + 0.897451 w_k1_IBM_s011
                                 + 1.27862 w_k1_INTC_s011
                                 + 1.10458 w_k1_KO_s011
                                 + 0.919459 w_k1_MAT_s011
                                 + 1.18861 w_k1_MSFT_s011
                                 + 1.06817 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.07826 w_k1_ADM_s011
                                 + 0.74562 w_k1_AXP_s011
                                 + 0.828187 w_k1_DIS_s011
                                 + 0.901568 w_k1_FDX_s011
                                 + 0.736471 w_k1_IBM_s011
                                 + 1.00539 w_k1_INTC_s011
                                 + 0.832971 w_k1_KO_s011
                                 + 0.566231 w_k1_MAT_s011
                                 + 0.851028 w_k1_MSFT_s011
                                 + 0.793057 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.28661 w_k1_ADM_s011
                                 + 1.28743 w_k1_AXP_s011
                                 + 0.868652 w_k1_DIS_s011
                                 + 1.08796 w_k1_FDX_s011
                                 + 0.979751 w_k1_IBM_s011
                                 + 0.873243 w_k1_INTC_s011
                                 + 0.969745 w_k1_KO_s011
                                 + 1.06926 w_k1_MAT_s011
                                 + 0.842887 w_k1_MSFT_s011
                                 + 1.0186 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.23959 w_k1_ADM_s011
                                 + 1.2313 w_k1_AXP_s011 + 1.10719 w_k1_DIS_s011
                                 + 1.07347 w_k1_FDX_s011
                                 + 1.29716 w_k1_IBM_s011
                                 + 1.10224 w_k1_INTC_s011
                                 + 1.25447 w_k1_KO_s011 + 1.51222 w_k1_MAT_s011
                                 + 1.00798 w_k1_MSFT_s011
                                 + 1.02463 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.50039 w_k1_ADM_s011
                                 + 1.53327 w_k1_AXP_s011
                                 + 1.63081 w_k1_DIS_s011
                                 + 1.26381 w_k1_FDX_s011
                                 + 1.55251 w_k1_IBM_s011
                                 + 1.41135 w_k1_INTC_s011
                                 + 1.39375 w_k1_KO_s011 + 1.13616 w_k1_MAT_s011
                                 + 1.14512 w_k1_MSFT_s011
                                 + 1.28516 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.1449 w_k1_ADM_s011
                                 + 0.762323 w_k1_AXP_s011
                                 + 0.971286 w_k1_DIS_s011
                                 + 0.953431 w_k1_FDX_s011
                                 + 1.03391 w_k1_IBM_s011
                                 + 1.18335 w_k1_INTC_s011
                                 + 1.18481 w_k1_KO_s011
                                 + 0.751678 w_k1_MAT_s011
                                 + 1.02016 w_k1_MSFT_s011
                                 + 1.05316 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.891767 w_k1_ADM_s012
                                 + 0.742199 w_k1_AXP_s012
                                 + 0.707839 w_k1_DIS_s012
                                 + 0.863766 w_k1_FDX_s012
                                 + 0.93516 w_k1_IBM_s012 + 1.063 w_k1_INTC_s012
                                 + 0.794688 w_k1_KO_s012
                                 + 0.874074 w_k1_MAT_s012
                                 + 0.9062 w_k1_MSFT_s012
                                 + 0.853978 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.923307 w_k1_ADM_s012
                                 + 1.08166 w_k1_AXP_s012
                                 + 0.976525 w_k1_DIS_s012
                                 + 0.900867 w_k1_FDX_s012
                                 + 1.63886 w_k1_IBM_s012
                                 + 0.970015 w_k1_INTC_s012
                                 + 1.09048 w_k1_KO_s012 + 1.91478 w_k1_MAT_s012
                                 + 0.90348 w_k1_MSFT_s012
                                 + 0.966219 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.2078 w_k1_ADM_s012
                                 + 1.30458 w_k1_AXP_s012
                                 + 1.07098 w_k1_DIS_s012 + 1.118 w_k1_FDX_s012
                                 + 1.29156 w_k1_IBM_s012
                                 + 1.20321 w_k1_INTC_s012
                                 + 1.04136 w_k1_KO_s012 + 1.71684 w_k1_MAT_s012
                                 + 1.05376 w_k1_MSFT_s012
                                 + 1.03576 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.708779 w_k1_ADM_s012
                                 + 0.586232 w_k1_AXP_s012
                                 + 0.59746 w_k1_DIS_s012
                                 + 0.703173 w_k1_FDX_s012
                                 + 0.797991 w_k1_IBM_s012
                                 + 0.748496 w_k1_INTC_s012
                                 + 0.718147 w_k1_KO_s012
                                 + 0.895642 w_k1_MAT_s012
                                 + 0.736093 w_k1_MSFT_s012
                                 + 0.796914 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.834196 w_k1_ADM_s012
                                 + 0.733433 w_k1_AXP_s012
                                 + 1.10946 w_k1_DIS_s012
                                 + 1.07294 w_k1_FDX_s012
                                 + 1.07063 w_k1_IBM_s012
                                 + 1.16292 w_k1_INTC_s012
                                 + 1.08831 w_k1_KO_s012
                                 + 0.934466 w_k1_MAT_s012
                                 + 1.05557 w_k1_MSFT_s012
                                 + 0.993279 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.14053 w_k1_ADM_s012
                                 + 0.913108 w_k1_AXP_s012
                                 + 0.736011 w_k1_DIS_s012
                                 + 0.970607 w_k1_FDX_s012
                                 + 0.758753 w_k1_IBM_s012
                                 + 0.914763 w_k1_INTC_s012
                                 + 0.725876 w_k1_KO_s012
                                 + 0.939069 w_k1_MAT_s012
                                 + 0.813434 w_k1_MSFT_s012
                                 + 0.863992 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.20092 w_k1_ADM_s012
                                 + 1.13405 w_k1_AXP_s012
                                 + 1.35489 w_k1_DIS_s012
                                 + 1.25201 w_k1_FDX_s012
                                 + 0.97872 w_k1_IBM_s012
                                 + 1.06703 w_k1_INTC_s012
                                 + 1.07501 w_k1_KO_s012 + 1.31947 w_k1_MAT_s012
                                 + 1.24338 w_k1_MSFT_s012
                                 + 1.02347 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.85001 w_k1_ADM_s012
                                 + 0.854542 w_k1_AXP_s012
                                 + 0.778062 w_k1_DIS_s012
                                 + 0.830673 w_k1_FDX_s012
                                 + 1.17971 w_k1_IBM_s012
                                 + 0.935262 w_k1_INTC_s012
                                 + 0.904115 w_k1_KO_s012
                                 + 0.921254 w_k1_MAT_s012
                                 + 0.766899 w_k1_MSFT_s012
                                 + 0.923615 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.5495 w_k1_ADM_s013
                                 + 1.56371 w_k1_AXP_s013
                                 + 1.70807 w_k1_DIS_s013
                                 + 1.44486 w_k1_FDX_s013
                                 + 1.25568 w_k1_IBM_s013
                                 + 1.45081 w_k1_INTC_s013
                                 + 1.27088 w_k1_KO_s013 + 1.06544 w_k1_MAT_s013
                                 + 1.33194 w_k1_MSFT_s013
                                 + 1.34369 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.866947 w_k1_ADM_s013
                                 + 0.856548 w_k1_AXP_s013
                                 + 0.741862 w_k1_DIS_s013
                                 + 0.863679 w_k1_FDX_s013
                                 + 0.906211 w_k1_IBM_s013
                                 + 0.804157 w_k1_INTC_s013
                                 + 0.828607 w_k1_KO_s013
                                 + 0.950186 w_k1_MAT_s013
                                 + 0.769163 w_k1_MSFT_s013
                                 + 0.919882 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.759505 w_k1_ADM_s013
                                 + 0.590066 w_k1_AXP_s013
                                 + 0.487343 w_k1_DIS_s013
                                 + 0.715955 w_k1_FDX_s013
                                 + 0.657889 w_k1_IBM_s013
                                 + 0.914782 w_k1_INTC_s013
                                 + 0.597955 w_k1_KO_s013
                                 + 0.375829 w_k1_MAT_s013
                                 + 0.728165 w_k1_MSFT_s013
                                 + 0.760077 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.11126 w_k1_ADM_s013
                                 + 0.874552 w_k1_AXP_s013
                                 + 1.02049 w_k1_DIS_s013
                                 + 0.960461 w_k1_FDX_s013
                                 + 1.28862 w_k1_IBM_s013
                                 + 0.860816 w_k1_INTC_s013
                                 + 1.12694 w_k1_KO_s013 + 1.27885 w_k1_MAT_s013
                                 + 0.939262 w_k1_MSFT_s013
                                 + 1.1923 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.934669 w_k1_ADM_s013
                                 + 0.790287 w_k1_AXP_s013
                                 + 1.22109 w_k1_DIS_s013
                                 + 1.04858 w_k1_FDX_s013
                                 + 1.10355 w_k1_IBM_s013
                                 + 1.26009 w_k1_INTC_s013
                                 + 1.17613 w_k1_KO_s013 + 1.01059 w_k1_MAT_s013
                                 + 1.15027 w_k1_MSFT_s013
                                 + 1.14488 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.2923 w_k1_ADM_s013
                                 + 1.16096 w_k1_AXP_s013
                                 + 1.22529 w_k1_DIS_s013
                                 + 1.25676 w_k1_FDX_s013
                                 + 0.91989 w_k1_IBM_s013
                                 + 1.2859 w_k1_INTC_s013 + 1.08873 w_k1_KO_s013
                                 + 1.07893 w_k1_MAT_s013
                                 + 1.1958 w_k1_MSFT_s013 + 1.08957 w_k1_PG_s013
                                 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.06968 w_k1_ADM_s013
                                 + 1.04442 w_k1_AXP_s013
                                 + 0.881059 w_k1_DIS_s013
                                 + 1.10949 w_k1_FDX_s013
                                 + 0.752109 w_k1_IBM_s013
                                 + 1.16333 w_k1_INTC_s013
                                 + 0.975434 w_k1_KO_s013
                                 + 0.842426 w_k1_MAT_s013
                                 + 1.07753 w_k1_MSFT_s013
                                 + 1.04324 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.888995 w_k1_ADM_s013
                                 + 0.668518 w_k1_AXP_s013
                                 + 0.637422 w_k1_DIS_s013
                                 + 0.701236 w_k1_FDX_s013
                                 + 1.18723 w_k1_IBM_s013
                                 + 0.657507 w_k1_INTC_s013
                                 + 0.956817 w_k1_KO_s013
                                 + 1.19159 w_k1_MAT_s013
                                 + 0.748702 w_k1_MSFT_s013
                                 + 0.907093 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.813151 w_k1_ADM_s014
                                 + 1.1908 w_k1_AXP_s014
                                 + 0.997672 w_k1_DIS_s014
                                 + 0.976898 w_k1_FDX_s014
                                 + 1.69869 w_k1_IBM_s014
                                 + 0.957196 w_k1_INTC_s014
                                 + 1.07674 w_k1_KO_s014 + 2.21149 w_k1_MAT_s014
                                 + 0.962851 w_k1_MSFT_s014
                                 + 1.02597 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.947877 w_k1_ADM_s014
                                 + 0.916903 w_k1_AXP_s014
                                 + 0.932641 w_k1_DIS_s014
                                 + 0.907095 w_k1_FDX_s014
                                 + 1.04117 w_k1_IBM_s014
                                 + 1.03248 w_k1_INTC_s014
                                 + 0.899757 w_k1_KO_s014
                                 + 1.02719 w_k1_MAT_s014
                                 + 0.91841 w_k1_MSFT_s014
                                 + 1.02859 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.910962 w_k1_ADM_s014
                                 + 0.91642 w_k1_AXP_s014
                                 + 1.07098 w_k1_DIS_s014
                                 + 1.05158 w_k1_FDX_s014
                                 + 0.706617 w_k1_IBM_s014
                                 + 0.915582 w_k1_INTC_s014
                                 + 0.79057 w_k1_KO_s014
                                 + 0.726332 w_k1_MAT_s014
                                 + 1.0201 w_k1_MSFT_s014 + 0.9444 w_k1_PG_s014
                                 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.807118 w_k1_ADM_s014
                                 + 0.64486 w_k1_AXP_s014
                                 + 0.661511 w_k1_DIS_s014
                                 + 0.868305 w_k1_FDX_s014
                                 + 0.571293 w_k1_IBM_s014
                                 + 0.869264 w_k1_INTC_s014
                                 + 0.635773 w_k1_KO_s014
                                 + 0.599196 w_k1_MAT_s014
                                 + 0.937453 w_k1_MSFT_s014
                                 + 0.873272 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.2718 w_k1_ADM_s014
                                 + 1.22305 w_k1_AXP_s014
                                 + 1.27241 w_k1_DIS_s014
                                 + 1.21435 w_k1_FDX_s014 + 1.0498 w_k1_IBM_s014
                                 + 1.22427 w_k1_INTC_s014
                                 + 1.03698 w_k1_KO_s014 + 1.30199 w_k1_MAT_s014
                                 + 1.16223 w_k1_MSFT_s014
                                 + 1.13279 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.827697 w_k1_ADM_s014
                                 + 0.790144 w_k1_AXP_s014
                                 + 0.668991 w_k1_DIS_s014
                                 + 0.767409 w_k1_FDX_s014
                                 + 1.05292 w_k1_IBM_s014
                                 + 0.795422 w_k1_INTC_s014
                                 + 0.796751 w_k1_KO_s014
                                 + 1.14609 w_k1_MAT_s014
                                 + 0.792435 w_k1_MSFT_s014
                                 + 0.940902 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.14511 w_k1_ADM_s014
                                 + 1.41416 w_k1_AXP_s014
                                 + 1.92171 w_k1_DIS_s014
                                 + 1.35345 w_k1_FDX_s014
                                 + 1.83336 w_k1_IBM_s014
                                 + 1.39697 w_k1_INTC_s014
                                 + 1.51484 w_k1_KO_s014 + 1.85403 w_k1_MAT_s014
                                 + 1.58846 w_k1_MSFT_s014 + 1.5219 w_k1_PG_s014
                                 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.33294 w_k1_ADM_s014
                                 + 1.24361 w_k1_AXP_s014
                                 + 1.25038 w_k1_DIS_s014
                                 + 1.49791 w_k1_FDX_s014
                                 + 0.607533 w_k1_IBM_s014
                                 + 0.978651 w_k1_INTC_s014
                                 + 0.825002 w_k1_KO_s014
                                 + 1.03006 w_k1_MAT_s014
                                 + 1.38145 w_k1_MSFT_s014 + 1.2036 w_k1_PG_s014
                                 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.874688 w_k1_ADM_s015
                                 + 1.34666 w_k1_AXP_s015
                                 + 1.14189 w_k1_DIS_s015
                                 + 0.97263 w_k1_FDX_s015
                                 + 1.60354 w_k1_IBM_s015
                                 + 1.15461 w_k1_INTC_s015
                                 + 1.20276 w_k1_KO_s015 + 1.77128 w_k1_MAT_s015
                                 + 1.0612 w_k1_MSFT_s015 + 1.16647 w_k1_PG_s015
                                 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.824761 w_k1_ADM_s015
                                 + 0.594715 w_k1_AXP_s015
                                 + 0.689981 w_k1_DIS_s015
                                 + 0.789435 w_k1_FDX_s015
                                 + 0.71562 w_k1_IBM_s015
                                 + 0.903789 w_k1_INTC_s015
                                 + 0.727526 w_k1_KO_s015
                                 + 0.593856 w_k1_MAT_s015
                                 + 0.897352 w_k1_MSFT_s015
                                 + 0.802488 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.22199 w_k1_ADM_s015
                                 + 0.994701 w_k1_AXP_s015
                                 + 1.25494 w_k1_DIS_s015
                                 + 1.22099 w_k1_FDX_s015
                                 + 0.745785 w_k1_IBM_s015
                                 + 1.27871 w_k1_INTC_s015
                                 + 0.891364 w_k1_KO_s015
                                 + 0.695517 w_k1_MAT_s015
                                 + 1.19975 w_k1_MSFT_s015
                                 + 1.01247 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.59293 w_k1_ADM_s015
                                 + 1.48107 w_k1_AXP_s015
                                 + 1.67506 w_k1_DIS_s015
                                 + 1.37066 w_k1_FDX_s015
                                 + 0.979517 w_k1_IBM_s015
                                 + 1.41903 w_k1_INTC_s015
                                 + 1.17658 w_k1_KO_s015
                                 + 0.907366 w_k1_MAT_s015
                                 + 1.26248 w_k1_MSFT_s015
                                 + 1.24348 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.822733 w_k1_ADM_s015
                                 + 0.892488 w_k1_AXP_s015
                                 + 0.984839 w_k1_DIS_s015
                                 + 1.02745 w_k1_FDX_s015
                                 + 0.803048 w_k1_IBM_s015
                                 + 0.959627 w_k1_INTC_s015
                                 + 1.03093 w_k1_KO_s015
                                 + 0.644678 w_k1_MAT_s015
                                 + 1.00077 w_k1_MSFT_s015
                                 + 1.00959 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.908372 w_k1_ADM_s015
                                 + 0.928272 w_k1_AXP_s015
                                 + 0.794377 w_k1_DIS_s015
                                 + 0.802404 w_k1_FDX_s015
                                 + 1.15601 w_k1_IBM_s015
                                 + 1.05782 w_k1_INTC_s015
                                 + 0.978526 w_k1_KO_s015
                                 + 1.05834 w_k1_MAT_s015
                                 + 0.882039 w_k1_MSFT_s015
                                 + 0.887921 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.19314 w_k1_ADM_s015
                                 + 1.11606 w_k1_AXP_s015
                                 + 1.50408 w_k1_DIS_s015 + 1.1443 w_k1_FDX_s015
                                 + 1.38229 w_k1_IBM_s015
                                 + 1.12775 w_k1_INTC_s015 + 1.331 w_k1_KO_s015
                                 + 1.10398 w_k1_MAT_s015
                                 + 1.25472 w_k1_MSFT_s015
                                 + 1.10001 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.967575 w_k1_ADM_s015
                                 + 0.834756 w_k1_AXP_s015
                                 + 1.13888 w_k1_DIS_s015
                                 + 0.890664 w_k1_FDX_s015
                                 + 1.15844 w_k1_IBM_s015
                                 + 0.91889 w_k1_INTC_s015
                                 + 1.07041 w_k1_KO_s015
                                 + 0.852422 w_k1_MAT_s015
                                 + 1.0629 w_k1_MSFT_s015
                                 + 0.916753 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.878812 w_k1_ADM_s016
                                 + 1.19359 w_k1_AXP_s016
                                 + 1.18584 w_k1_DIS_s016
                                 + 1.07514 w_k1_FDX_s016
                                 + 1.29483 w_k1_IBM_s016
                                 + 1.04443 w_k1_INTC_s016
                                 + 1.46344 w_k1_KO_s016 + 1.18542 w_k1_MAT_s016
                                 + 1.13981 w_k1_MSFT_s016
                                 + 1.17325 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.839706 w_k1_ADM_s016
                                 + 0.980055 w_k1_AXP_s016
                                 + 1.04963 w_k1_DIS_s016
                                 + 1.10648 w_k1_FDX_s016
                                 + 0.735654 w_k1_IBM_s016
                                 + 0.8801 w_k1_INTC_s016
                                 + 0.878334 w_k1_KO_s016
                                 + 0.807676 w_k1_MAT_s016
                                 + 1.05871 w_k1_MSFT_s016
                                 + 0.854054 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.00201 w_k1_ADM_s016
                                 + 1.55889 w_k1_AXP_s016
                                 + 1.18926 w_k1_DIS_s016
                                 + 1.11133 w_k1_FDX_s016
                                 + 1.79121 w_k1_IBM_s016
                                 + 1.02515 w_k1_INTC_s016
                                 + 1.48945 w_k1_KO_s016 + 2.06701 w_k1_MAT_s016
                                 + 1.00606 w_k1_MSFT_s016
                                 + 1.20486 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.879983 w_k1_ADM_s016
                                 + 0.872289 w_k1_AXP_s016
                                 + 0.761789 w_k1_DIS_s016
                                 + 0.844095 w_k1_FDX_s016
                                 + 0.790469 w_k1_IBM_s016
                                 + 0.784442 w_k1_INTC_s016
                                 + 0.800659 w_k1_KO_s016
                                 + 0.581365 w_k1_MAT_s016
                                 + 0.865036 w_k1_MSFT_s016
                                 + 0.817911 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.27508 w_k1_ADM_s016
                                 + 1.29602 w_k1_AXP_s016
                                 + 1.17942 w_k1_DIS_s016 + 1.2522 w_k1_FDX_s016
                                 + 0.968082 w_k1_IBM_s016
                                 + 0.794708 w_k1_INTC_s016
                                 + 1.14887 w_k1_KO_s016 + 1.12356 w_k1_MAT_s016
                                 + 0.969622 w_k1_MSFT_s016
                                 + 1.18352 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.36593 w_k1_ADM_s016
                                 + 1.34004 w_k1_AXP_s016
                                 + 1.58146 w_k1_DIS_s016
                                 + 1.37623 w_k1_FDX_s016
                                 + 1.16908 w_k1_IBM_s016
                                 + 1.09355 w_k1_INTC_s016
                                 + 1.32443 w_k1_KO_s016 + 1.1092 w_k1_MAT_s016
                                 + 1.29149 w_k1_MSFT_s016
                                 + 1.08095 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.07023 w_k1_ADM_s016
                                 + 1.07808 w_k1_AXP_s016
                                 + 1.10324 w_k1_DIS_s016
                                 + 1.06754 w_k1_FDX_s016
                                 + 1.16645 w_k1_IBM_s016
                                 + 0.911593 w_k1_INTC_s016
                                 + 1.03714 w_k1_KO_s016 + 1.27083 w_k1_MAT_s016
                                 + 0.969315 w_k1_MSFT_s016
                                 + 1.01004 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.807638 w_k1_ADM_s016
                                 + 0.792319 w_k1_AXP_s016
                                 + 0.968615 w_k1_DIS_s016
                                 + 0.911282 w_k1_FDX_s016
                                 + 1.20451 w_k1_IBM_s016
                                 + 0.979752 w_k1_INTC_s016
                                 + 1.1461 w_k1_KO_s016 + 1.09215 w_k1_MAT_s016
                                 + 1.03616 w_k1_MSFT_s016
                                 + 0.954913 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.789032 w_k1_ADM_s017
                                 + 0.784395 w_k1_AXP_s017
                                 + 0.715402 w_k1_DIS_s017
                                 + 0.853262 w_k1_FDX_s017
                                 + 0.68497 w_k1_IBM_s017
                                 + 1.06492 w_k1_INTC_s017
                                 + 0.803569 w_k1_KO_s017
                                 + 0.765302 w_k1_MAT_s017
                                 + 0.842629 w_k1_MSFT_s017
                                 + 0.794439 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.13295 w_k1_ADM_s017
                                 + 0.986009 w_k1_AXP_s017
                                 + 1.08725 w_k1_DIS_s017
                                 + 0.983054 w_k1_FDX_s017
                                 + 1.03141 w_k1_IBM_s017
                                 + 1.19049 w_k1_INTC_s017
                                 + 1.13676 w_k1_KO_s017
                                 + 0.948073 w_k1_MAT_s017
                                 + 1.10799 w_k1_MSFT_s017
                                 + 0.955371 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.43264 w_k1_ADM_s017
                                 + 1.38027 w_k1_AXP_s017
                                 + 1.33772 w_k1_DIS_s017
                                 + 1.31906 w_k1_FDX_s017
                                 + 0.987863 w_k1_IBM_s017
                                 + 1.38065 w_k1_INTC_s017
                                 + 1.16933 w_k1_KO_s017 + 1.21049 w_k1_MAT_s017
                                 + 1.28531 w_k1_MSFT_s017 + 1.0742 w_k1_PG_s017
                                 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.01031 w_k1_ADM_s017
                                 + 0.858077 w_k1_AXP_s017
                                 + 0.984964 w_k1_DIS_s017
                                 + 0.856463 w_k1_FDX_s017
                                 + 1.5771 w_k1_IBM_s017
                                 + 1.09457 w_k1_INTC_s017
                                 + 1.22332 w_k1_KO_s017 + 1.3334 w_k1_MAT_s017
                                 + 0.991685 w_k1_MSFT_s017
                                 + 1.07438 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.09343 w_k1_ADM_s017
                                 + 0.910354 w_k1_AXP_s017
                                 + 1.21785 w_k1_DIS_s017 + 1.1878 w_k1_FDX_s017
                                 + 0.78436 w_k1_IBM_s017
                                 + 1.30403 w_k1_INTC_s017
                                 + 0.93853 w_k1_KO_s017
                                 + 0.443826 w_k1_MAT_s017
                                 + 1.22582 w_k1_MSFT_s017
                                 + 1.08481 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.20556 w_k1_ADM_s017
                                 + 1.5686 w_k1_AXP_s017 + 1.50056 w_k1_DIS_s017
                                 + 1.24527 w_k1_FDX_s017
                                 + 1.47796 w_k1_IBM_s017
                                 + 1.27621 w_k1_INTC_s017
                                 + 1.45881 w_k1_KO_s017 + 1.88701 w_k1_MAT_s017
                                 + 1.22231 w_k1_MSFT_s017
                                 + 1.18175 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.906422 w_k1_ADM_s017
                                 + 0.753146 w_k1_AXP_s017
                                 + 0.693273 w_k1_DIS_s017
                                 + 0.75479 w_k1_FDX_s017
                                 + 1.00999 w_k1_IBM_s017
                                 + 0.97872 w_k1_INTC_s017
                                 + 0.839768 w_k1_KO_s017
                                 + 0.841069 w_k1_MAT_s017
                                 + 0.854194 w_k1_MSFT_s017
                                 + 0.882232 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.25221 w_k1_ADM_s017
                                 + 0.96276 w_k1_AXP_s017
                                 + 1.35087 w_k1_DIS_s017
                                 + 1.07353 w_k1_FDX_s017
                                 + 1.29497 w_k1_IBM_s017
                                 + 1.06785 w_k1_INTC_s017
                                 + 1.25305 w_k1_KO_s017 + 1.05852 w_k1_MAT_s017
                                 + 1.08321 w_k1_MSFT_s017
                                 + 1.08717 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.850032 w_k1_ADM_s100
                                 + 1.17899 w_k1_AXP_s100
                                 + 1.08788 w_k1_DIS_s100
                                 + 1.01967 w_k1_FDX_s100
                                 + 1.55259 w_k1_IBM_s100
                                 + 0.905476 w_k1_INTC_s100
                                 + 1.29438 w_k1_KO_s100 + 1.5065 w_k1_MAT_s100
                                 + 0.963467 w_k1_MSFT_s100
                                 + 1.17434 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.693601 w_k1_ADM_s100
                                 + 0.856229 w_k1_AXP_s100
                                 + 0.793885 w_k1_DIS_s100
                                 + 0.796853 w_k1_FDX_s100
                                 + 1.12891 w_k1_IBM_s100
                                 + 0.873721 w_k1_INTC_s100
                                 + 1.01844 w_k1_KO_s100 + 1.27011 w_k1_MAT_s100
                                 + 0.926123 w_k1_MSFT_s100
                                 + 1.05781 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.02741 w_k1_ADM_s100
                                 + 0.949991 w_k1_AXP_s100
                                 + 0.862559 w_k1_DIS_s100
                                 + 1.01284 w_k1_FDX_s100
                                 + 0.979326 w_k1_IBM_s100
                                 + 1.02084 w_k1_INTC_s100
                                 + 0.918488 w_k1_KO_s100
                                 + 0.981604 w_k1_MAT_s100
                                 + 0.939111 w_k1_MSFT_s100
                                 + 1.03862 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.03894 w_k1_ADM_s100
                                 + 1.03749 w_k1_AXP_s100
                                 + 1.15459 w_k1_DIS_s100
                                 + 1.04513 w_k1_FDX_s100
                                 + 0.957142 w_k1_IBM_s100
                                 + 0.968649 w_k1_INTC_s100
                                 + 1.17515 w_k1_KO_s100
                                 + 0.904034 w_k1_MAT_s100
                                 + 1.04495 w_k1_MSFT_s100
                                 + 1.19397 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.887361 w_k1_ADM_s100
                                 + 0.894214 w_k1_AXP_s100
                                 + 0.703534 w_k1_DIS_s100
                                 + 0.947542 w_k1_FDX_s100
                                 + 0.601284 w_k1_IBM_s100
                                 + 0.885978 w_k1_INTC_s100
                                 + 0.802755 w_k1_KO_s100
                                 + 0.602315 w_k1_MAT_s100
                                 + 0.848568 w_k1_MSFT_s100
                                 + 0.929453 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.03376 w_k1_ADM_s100
                                 + 1.308 w_k1_AXP_s100 + 1.4241 w_k1_DIS_s100
                                 + 1.27413 w_k1_FDX_s100
                                 + 1.27701 w_k1_IBM_s100
                                 + 1.15125 w_k1_INTC_s100
                                 + 1.24077 w_k1_KO_s100 + 1.22233 w_k1_MAT_s100
                                 + 1.19653 w_k1_MSFT_s100
                                 + 1.19447 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.842748 w_k1_ADM_s100
                                 + 0.719154 w_k1_AXP_s100
                                 + 0.62473 w_k1_DIS_s100
                                 + 0.761791 w_k1_FDX_s100
                                 + 0.83832 w_k1_IBM_s100
                                 + 0.727447 w_k1_INTC_s100
                                 + 0.878172 w_k1_KO_s100
                                 + 0.736771 w_k1_MAT_s100
                                 + 0.875901 w_k1_MSFT_s100
                                 + 0.944458 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.37378 w_k1_ADM_s100
                                 + 1.80058 w_k1_AXP_s100
                                 + 1.37029 w_k1_DIS_s100
                                 + 1.43004 w_k1_FDX_s100
                                 + 1.08211 w_k1_IBM_s100
                                 + 1.09998 w_k1_INTC_s100
                                 + 1.23241 w_k1_KO_s100 + 1.11159 w_k1_MAT_s100
                                 + 1.07909 w_k1_MSFT_s100
                                 + 1.33118 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.21028 w_k1_ADM_s101
                                 + 1.33031 w_k1_AXP_s101
                                 + 1.33515 w_k1_DIS_s101
                                 + 1.14865 w_k1_FDX_s101
                                 + 0.989007 w_k1_IBM_s101
                                 + 1.04746 w_k1_INTC_s101
                                 + 1.07524 w_k1_KO_s101 + 1.0706 w_k1_MAT_s101
                                 + 1.07057 w_k1_MSFT_s101
                                 + 1.12234 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.769219 w_k1_ADM_s101
                                 + 0.931741 w_k1_AXP_s101
                                 + 1.01341 w_k1_DIS_s101
                                 + 0.830896 w_k1_FDX_s101
                                 + 1.23132 w_k1_IBM_s101
                                 + 0.934458 w_k1_INTC_s101
                                 + 1.1505 w_k1_KO_s101 + 1.11674 w_k1_MAT_s101
                                 + 0.903718 w_k1_MSFT_s101
                                 + 0.98254 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.868916 w_k1_ADM_s101
                                 + 0.858372 w_k1_AXP_s101
                                 + 0.837782 w_k1_DIS_s101
                                 + 0.869107 w_k1_FDX_s101
                                 + 0.931378 w_k1_IBM_s101
                                 + 0.838625 w_k1_INTC_s101
                                 + 0.947717 w_k1_KO_s101
                                 + 0.856313 w_k1_MAT_s101
                                 + 0.926716 w_k1_MSFT_s101
                                 + 0.810687 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.959419 w_k1_ADM_s101
                                 + 1.01245 w_k1_AXP_s101
                                 + 0.902833 w_k1_DIS_s101
                                 + 1.0101 w_k1_FDX_s101
                                 + 0.715415 w_k1_IBM_s101
                                 + 1.1349 w_k1_INTC_s101
                                 + 0.892077 w_k1_KO_s101
                                 + 0.64288 w_k1_MAT_s101
                                 + 0.981817 w_k1_MSFT_s101
                                 + 0.869053 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.17983 w_k1_ADM_s101
                                 + 1.43604 w_k1_AXP_s101
                                 + 1.77454 w_k1_DIS_s101
                                 + 1.29331 w_k1_FDX_s101
                                 + 1.39707 w_k1_IBM_s101
                                 + 1.35146 w_k1_INTC_s101
                                 + 1.55782 w_k1_KO_s101 + 1.4118 w_k1_MAT_s101
                                 + 1.22768 w_k1_MSFT_s101
                                 + 1.01189 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.752518 w_k1_ADM_s101
                                 + 0.650218 w_k1_AXP_s101
                                 + 0.609114 w_k1_DIS_s101
                                 + 0.669713 w_k1_FDX_s101
                                 + 0.774466 w_k1_IBM_s101
                                 + 0.688468 w_k1_INTC_s101
                                 + 0.716878 w_k1_KO_s101
                                 + 0.538052 w_k1_MAT_s101
                                 + 0.717716 w_k1_MSFT_s101
                                 + 0.834959 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.76367 w_k1_ADM_s101
                                 + 0.924362 w_k1_AXP_s101
                                 + 0.659409 w_k1_DIS_s101
                                 + 0.771574 w_k1_FDX_s101
                                 + 1.18445 w_k1_IBM_s101
                                 + 0.762355 w_k1_INTC_s101
                                 + 0.833185 w_k1_KO_s101
                                 + 0.986015 w_k1_MAT_s101
                                 + 0.710702 w_k1_MSFT_s101
                                 + 0.796582 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.12436 w_k1_ADM_s101
                                 + 1.1369 w_k1_AXP_s101 + 1.14032 w_k1_DIS_s101
                                 + 1.06352 w_k1_FDX_s101
                                 + 1.72321 w_k1_IBM_s101
                                 + 0.908631 w_k1_INTC_s101
                                 + 1.21662 w_k1_KO_s101 + 1.62119 w_k1_MAT_s101
                                 + 1.07612 w_k1_MSFT_s101 + 1.1429 w_k1_PG_s101
                                 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.63527 w_k1_ADM_s102
                                 + 1.39953 w_k1_AXP_s102
                                 + 1.99703 w_k1_DIS_s102
                                 + 1.55975 w_k1_FDX_s102
                                 + 1.55552 w_k1_IBM_s102
                                 + 1.27364 w_k1_INTC_s102
                                 + 1.85553 w_k1_KO_s102 + 1.5554 w_k1_MAT_s102
                                 + 1.45743 w_k1_MSFT_s102
                                 + 1.57949 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.89228 w_k1_ADM_s102
                                 + 0.750717 w_k1_AXP_s102
                                 + 0.625642 w_k1_DIS_s102
                                 + 0.834303 w_k1_FDX_s102
                                 + 1.17269 w_k1_IBM_s102
                                 + 0.753631 w_k1_INTC_s102
                                 + 0.817796 w_k1_KO_s102
                                 + 0.700044 w_k1_MAT_s102
                                 + 0.706765 w_k1_MSFT_s102
                                 + 0.893007 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.994131 w_k1_ADM_s102
                                 + 0.819771 w_k1_AXP_s102
                                 + 0.673399 w_k1_DIS_s102
                                 + 0.870428 w_k1_FDX_s102
                                 + 1.00215 w_k1_IBM_s102
                                 + 0.867476 w_k1_INTC_s102
                                 + 0.849194 w_k1_KO_s102 + 0.8809 w_k1_MAT_s102
                                 + 0.795117 w_k1_MSFT_s102
                                 + 0.907194 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.00756 w_k1_ADM_s102
                                 + 0.842846 w_k1_AXP_s102
                                 + 0.88706 w_k1_DIS_s102
                                 + 0.969005 w_k1_FDX_s102
                                 + 0.772913 w_k1_IBM_s102
                                 + 1.04119 w_k1_INTC_s102
                                 + 0.81868 w_k1_KO_s102
                                 + 0.694852 w_k1_MAT_s102
                                 + 0.97965 w_k1_MSFT_s102
                                 + 0.962823 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.07973 w_k1_ADM_s102
                                 + 1.21616 w_k1_AXP_s102
                                 + 1.09673 w_k1_DIS_s102
                                 + 1.07946 w_k1_FDX_s102
                                 + 1.20461 w_k1_IBM_s102
                                 + 1.09659 w_k1_INTC_s102
                                 + 1.17475 w_k1_KO_s102 + 1.12907 w_k1_MAT_s102
                                 + 1.08719 w_k1_MSFT_s102
                                 + 1.18185 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.783694 w_k1_ADM_s102
                                 + 0.628651 w_k1_AXP_s102
                                 + 0.499714 w_k1_DIS_s102
                                 + 0.666315 w_k1_FDX_s102
                                 + 0.749101 w_k1_IBM_s102
                                 + 0.702239 w_k1_INTC_s102
                                 + 0.70904 w_k1_KO_s102 + 0.64779 w_k1_MAT_s102
                                 + 0.720586 w_k1_MSFT_s102
                                 + 0.808178 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.19192 w_k1_ADM_s102
                                 + 0.805846 w_k1_AXP_s102
                                 + 1.12694 w_k1_DIS_s102
                                 + 1.17047 w_k1_FDX_s102 + 0.8343 w_k1_IBM_s102
                                 + 1.14374 w_k1_INTC_s102
                                 + 1.13132 w_k1_KO_s102 + 0.63424 w_k1_MAT_s102
                                 + 1.19556 w_k1_MSFT_s102 + 1.2124 w_k1_PG_s102
                                 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.866852 w_k1_ADM_s102
                                 + 0.566292 w_k1_AXP_s102
                                 + 0.528288 w_k1_DIS_s102
                                 + 0.754156 w_k1_FDX_s102
                                 + 0.588399 w_k1_IBM_s102
                                 + 0.699169 w_k1_INTC_s102
                                 + 0.697116 w_k1_KO_s102
                                 + 0.586601 w_k1_MAT_s102
                                 + 0.752874 w_k1_MSFT_s102
                                 + 0.850953 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.26562 w_k1_ADM_s103
                                 + 1.4509 w_k1_AXP_s103 + 1.18118 w_k1_DIS_s103
                                 + 1.22145 w_k1_FDX_s103
                                 + 0.98575 w_k1_IBM_s103
                                 + 0.929511 w_k1_INTC_s103
                                 + 1.06452 w_k1_KO_s103 + 1.20961 w_k1_MAT_s103
                                 + 1.11898 w_k1_MSFT_s103
                                 + 1.23443 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.838345 w_k1_ADM_s103
                                 + 0.982875 w_k1_AXP_s103
                                 + 0.721143 w_k1_DIS_s103
                                 + 0.866153 w_k1_FDX_s103
                                 + 1.02234 w_k1_IBM_s103
                                 + 0.81798 w_k1_INTC_s103
                                 + 0.898209 w_k1_KO_s103
                                 + 1.02974 w_k1_MAT_s103
                                 + 0.889258 w_k1_MSFT_s103
                                 + 0.993734 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.915111 w_k1_ADM_s103
                                 + 0.784881 w_k1_AXP_s103
                                 + 0.788978 w_k1_DIS_s103
                                 + 0.85798 w_k1_FDX_s103
                                 + 0.669002 w_k1_IBM_s103
                                 + 0.654652 w_k1_INTC_s103
                                 + 0.647033 w_k1_KO_s103
                                 + 0.70747 w_k1_MAT_s103
                                 + 0.809684 w_k1_MSFT_s103
                                 + 0.933057 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.21454 w_k1_ADM_s103
                                 + 1.24176 w_k1_AXP_s103
                                 + 1.50902 w_k1_DIS_s103
                                 + 1.31413 w_k1_FDX_s103
                                 + 0.917193 w_k1_IBM_s103
                                 + 1.02305 w_k1_INTC_s103
                                 + 1.06352 w_k1_KO_s103 + 1.02256 w_k1_MAT_s103
                                 + 1.35342 w_k1_MSFT_s103
                                 + 1.02088 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.16109 w_k1_ADM_s103
                                 + 1.09023 w_k1_AXP_s103
                                 + 1.08342 w_k1_DIS_s103
                                 + 1.07647 w_k1_FDX_s103
                                 + 1.13574 w_k1_IBM_s103
                                 + 0.951572 w_k1_INTC_s103
                                 + 1.1615 w_k1_KO_s103 + 1.08731 w_k1_MAT_s103
                                 + 1.09631 w_k1_MSFT_s103
                                 + 1.18358 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.92242 w_k1_ADM_s103
                                 + 1.12446 w_k1_AXP_s103
                                 + 0.96206 w_k1_DIS_s103
                                 + 0.914042 w_k1_FDX_s103
                                 + 1.64898 w_k1_IBM_s103
                                 + 0.954855 w_k1_INTC_s103
                                 + 1.26138 w_k1_KO_s103 + 1.59911 w_k1_MAT_s103
                                 + 0.998859 w_k1_MSFT_s103
                                 + 1.08974 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.17408 w_k1_ADM_s103
                                 + 1.33937 w_k1_AXP_s103
                                 + 1.83658 w_k1_DIS_s103
                                 + 1.37852 w_k1_FDX_s103
                                 + 1.48555 w_k1_IBM_s103
                                 + 1.23394 w_k1_INTC_s103 + 1.5219 w_k1_KO_s103
                                 + 1.4961 w_k1_MAT_s103 + 1.5773 w_k1_MSFT_s103
                                 + 1.25337 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.607518 w_k1_ADM_s103
                                 + 0.752884 w_k1_AXP_s103
                                 + 0.611526 w_k1_DIS_s103
                                 + 0.69599 w_k1_FDX_s103
                                 + 0.877867 w_k1_IBM_s103
                                 + 0.78265 w_k1_INTC_s103
                                 + 0.726904 w_k1_KO_s103
                                 + 0.811265 w_k1_MAT_s103
                                 + 0.80206 w_k1_MSFT_s103
                                 + 0.803518 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.732432 w_k1_ADM_s104
                                 + 0.825767 w_k1_AXP_s104
                                 + 1.06702 w_k1_DIS_s104
                                 + 1.07056 w_k1_FDX_s104
                                 + 0.758733 w_k1_IBM_s104
                                 + 1.02998 w_k1_INTC_s104
                                 + 0.904563 w_k1_KO_s104
                                 + 0.959668 w_k1_MAT_s104
                                 + 1.08975 w_k1_MSFT_s104
                                 + 0.932442 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.13311 w_k1_ADM_s104
                                 + 1.34812 w_k1_AXP_s104
                                 + 1.16803 w_k1_DIS_s104
                                 + 1.04913 w_k1_FDX_s104
                                 + 1.39648 w_k1_IBM_s104
                                 + 0.790858 w_k1_INTC_s104
                                 + 1.13452 w_k1_KO_s104 + 1.86267 w_k1_MAT_s104
                                 + 0.948377 w_k1_MSFT_s104
                                 + 1.02124 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.0362 w_k1_ADM_s104
                                 + 1.28721 w_k1_AXP_s104
                                 + 1.11441 w_k1_DIS_s104
                                 + 1.14669 w_k1_FDX_s104
                                 + 1.04663 w_k1_IBM_s104
                                 + 0.919708 w_k1_INTC_s104
                                 + 1.02755 w_k1_KO_s104 + 1.19541 w_k1_MAT_s104
                                 + 0.918818 w_k1_MSFT_s104
                                 + 1.07122 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.925075 w_k1_ADM_s104
                                 + 0.839801 w_k1_AXP_s104
                                 + 0.737947 w_k1_DIS_s104
                                 + 0.854964 w_k1_FDX_s104
                                 + 0.783148 w_k1_IBM_s104
                                 + 0.624499 w_k1_INTC_s104
                                 + 0.751609 w_k1_KO_s104
                                 + 0.707235 w_k1_MAT_s104
                                 + 0.760786 w_k1_MSFT_s104
                                 + 0.88885 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.30042 w_k1_ADM_s104
                                 + 1.68292 w_k1_AXP_s104
                                 + 1.90205 w_k1_DIS_s104
                                 + 1.39731 w_k1_FDX_s104
                                 + 1.46589 w_k1_IBM_s104
                                 + 1.29524 w_k1_INTC_s104
                                 + 1.57509 w_k1_KO_s104 + 2.06305 w_k1_MAT_s104
                                 + 1.27703 w_k1_MSFT_s104
                                 + 1.21991 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.806562 w_k1_ADM_s104
                                 + 0.791641 w_k1_AXP_s104
                                 + 0.69099 w_k1_DIS_s104
                                 + 0.726749 w_k1_FDX_s104
                                 + 1.01474 w_k1_IBM_s104
                                 + 0.720904 w_k1_INTC_s104
                                 + 0.83911 w_k1_KO_s104 + 1.0798 w_k1_MAT_s104
                                 + 0.805217 w_k1_MSFT_s104
                                 + 0.885006 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.12092 w_k1_ADM_s104
                                 + 1.18739 w_k1_AXP_s104
                                 + 1.29393 w_k1_DIS_s104
                                 + 1.13916 w_k1_FDX_s104
                                 + 0.856714 w_k1_IBM_s104
                                 + 1.01021 w_k1_INTC_s104
                                 + 1.06049 w_k1_KO_s104
                                 + 0.844177 w_k1_MAT_s104
                                 + 1.01957 w_k1_MSFT_s104
                                 + 1.19117 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.07838 w_k1_ADM_s104
                                 + 0.964582 w_k1_AXP_s104
                                 + 1.17689 w_k1_DIS_s104
                                 + 0.928536 w_k1_FDX_s104
                                 + 1.26757 w_k1_IBM_s104
                                 + 0.699705 w_k1_INTC_s104
                                 + 1.15697 w_k1_KO_s104 + 1.28149 w_k1_MAT_s104
                                 + 0.955727 w_k1_MSFT_s104
                                 + 1.04489 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.92793 w_k1_ADM_s105
                                 + 1.66377 w_k1_AXP_s105
                                 + 1.62026 w_k1_DIS_s105
                                 + 1.25488 w_k1_FDX_s105
                                 + 1.53874 w_k1_IBM_s105
                                 + 0.984101 w_k1_INTC_s105
                                 + 1.21068 w_k1_KO_s105 + 1.42086 w_k1_MAT_s105
                                 + 1.07829 w_k1_MSFT_s105
                                 + 1.06006 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.892275 w_k1_ADM_s105
                                 + 0.994775 w_k1_AXP_s105
                                 + 1.06138 w_k1_DIS_s105
                                 + 1.06623 w_k1_FDX_s105
                                 + 0.758866 w_k1_IBM_s105
                                 + 0.997344 w_k1_INTC_s105
                                 + 0.848878 w_k1_KO_s105
                                 + 0.902646 w_k1_MAT_s105
                                 + 0.93427 w_k1_MSFT_s105
                                 + 0.944287 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.901467 w_k1_ADM_s105
                                 + 0.839658 w_k1_AXP_s105
                                 + 0.722252 w_k1_DIS_s105
                                 + 0.85633 w_k1_FDX_s105
                                 + 0.659143 w_k1_IBM_s105
                                 + 0.734299 w_k1_INTC_s105
                                 + 0.657083 w_k1_KO_s105
                                 + 0.657397 w_k1_MAT_s105
                                 + 0.744787 w_k1_MSFT_s105
                                 + 0.887762 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.03981 w_k1_ADM_s105
                                 + 1.09957 w_k1_AXP_s105
                                 + 1.16452 w_k1_DIS_s105
                                 + 0.991456 w_k1_FDX_s105
                                 + 1.27596 w_k1_IBM_s105
                                 + 0.845157 w_k1_INTC_s105
                                 + 1.0442 w_k1_KO_s105 + 1.58253 w_k1_MAT_s105
                                 + 1.01973 w_k1_MSFT_s105
                                 + 1.10096 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.00954 w_k1_ADM_s105
                                 + 1.30634 w_k1_AXP_s105
                                 + 1.46274 w_k1_DIS_s105
                                 + 1.25437 w_k1_FDX_s105
                                 + 0.936555 w_k1_IBM_s105
                                 + 0.923666 w_k1_INTC_s105
                                 + 1.03138 w_k1_KO_s105 + 1.16013 w_k1_MAT_s105
                                 + 0.937015 w_k1_MSFT_s105
                                 + 1.08088 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.862422 w_k1_ADM_s105
                                 + 0.868834 w_k1_AXP_s105
                                 + 1.01749 w_k1_DIS_s105
                                 + 0.96383 w_k1_FDX_s105
                                 + 0.854164 w_k1_IBM_s105
                                 + 1.09835 w_k1_INTC_s105
                                 + 1.06748 w_k1_KO_s105 + 1.11881 w_k1_MAT_s105
                                 + 0.9706 w_k1_MSFT_s105
                                 + 0.989208 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.680825 w_k1_ADM_s105
                                 + 0.776929 w_k1_AXP_s105
                                 + 0.672927 w_k1_DIS_s105
                                 + 0.755562 w_k1_FDX_s105
                                 + 0.862955 w_k1_IBM_s105
                                 + 0.758086 w_k1_INTC_s105
                                 + 0.749489 w_k1_KO_s105
                                 + 0.740309 w_k1_MAT_s105
                                 + 0.733352 w_k1_MSFT_s105
                                 + 0.849547 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.888038 w_k1_ADM_s105
                                 + 0.949414 w_k1_AXP_s105
                                 + 0.993908 w_k1_DIS_s105
                                 + 0.78767 w_k1_FDX_s105 + 1.1863 w_k1_IBM_s105
                                 + 0.892571 w_k1_INTC_s105
                                 + 1.03901 w_k1_KO_s105 + 1.16113 w_k1_MAT_s105
                                 + 0.888489 w_k1_MSFT_s105
                                 + 0.963455 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.99306 w_k1_ADM_s106
                                 + 1.05004 w_k1_AXP_s106 + 1.0401 w_k1_DIS_s106
                                 + 0.905803 w_k1_FDX_s106
                                 + 1.39394 w_k1_IBM_s106
                                 + 1.09062 w_k1_INTC_s106
                                 + 1.19494 w_k1_KO_s106 + 1.27937 w_k1_MAT_s106
                                 + 0.914409 w_k1_MSFT_s106
                                 + 1.0161 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.06962 w_k1_ADM_s106
                                 + 1.39449 w_k1_AXP_s106
                                 + 1.27972 w_k1_DIS_s106
                                 + 1.04079 w_k1_FDX_s106
                                 + 1.80352 w_k1_IBM_s106
                                 + 1.46905 w_k1_INTC_s106
                                 + 1.43989 w_k1_KO_s106 + 1.68576 w_k1_MAT_s106
                                 + 1.20977 w_k1_MSFT_s106
                                 + 1.17861 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.09159 w_k1_ADM_s106
                                 + 1.25371 w_k1_AXP_s106 + 1.0645 w_k1_DIS_s106
                                 + 1.22202 w_k1_FDX_s106
                                 + 1.02683 w_k1_IBM_s106
                                 + 1.13625 w_k1_INTC_s106
                                 + 0.956591 w_k1_KO_s106
                                 + 1.26834 w_k1_MAT_s106
                                 + 1.11945 w_k1_MSFT_s106 + 1.0638 w_k1_PG_s106
                                 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.10689 w_k1_ADM_s106
                                 + 1.11291 w_k1_AXP_s106
                                 + 1.49156 w_k1_DIS_s106
                                 + 1.29879 w_k1_FDX_s106
                                 + 1.29586 w_k1_IBM_s106
                                 + 1.19089 w_k1_INTC_s106
                                 + 1.33069 w_k1_KO_s106 + 1.48509 w_k1_MAT_s106
                                 + 1.44963 w_k1_MSFT_s106
                                 + 1.22858 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.851763 w_k1_ADM_s106
                                 + 0.930105 w_k1_AXP_s106
                                 + 0.789811 w_k1_DIS_s106
                                 + 0.955566 w_k1_FDX_s106
                                 + 0.772521 w_k1_IBM_s106
                                 + 1.15803 w_k1_INTC_s106
                                 + 0.775404 w_k1_KO_s106
                                 + 0.689231 w_k1_MAT_s106
                                 + 1.03635 w_k1_MSFT_s106
                                 + 0.806399 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.0927 w_k1_ADM_s106
                                 + 1.46123 w_k1_AXP_s106
                                 + 0.982779 w_k1_DIS_s106
                                 + 1.09875 w_k1_FDX_s106
                                 + 1.21043 w_k1_IBM_s106
                                 + 1.17347 w_k1_INTC_s106
                                 + 1.07978 w_k1_KO_s106 + 1.76028 w_k1_MAT_s106
                                 + 1.05783 w_k1_MSFT_s106
                                 + 1.07506 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.07022 w_k1_ADM_s106
                                 + 0.874205 w_k1_AXP_s106
                                 + 1.37882 w_k1_DIS_s106
                                 + 1.09025 w_k1_FDX_s106
                                 + 0.88807 w_k1_IBM_s106
                                 + 1.01646 w_k1_INTC_s106
                                 + 1.09517 w_k1_KO_s106
                                 + 0.566127 w_k1_MAT_s106
                                 + 1.1877 w_k1_MSFT_s106 + 0.96946 w_k1_PG_s106
                                 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.852061 w_k1_ADM_s106
                                 + 0.707013 w_k1_AXP_s106
                                 + 0.787982 w_k1_DIS_s106
                                 + 0.837368 w_k1_FDX_s106
                                 + 1.03706 w_k1_IBM_s106
                                 + 1.02651 w_k1_INTC_s106
                                 + 1.01252 w_k1_KO_s106 + 0.98443 w_k1_MAT_s106
                                 + 1.04384 w_k1_MSFT_s106
                                 + 0.934155 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.834731 w_k1_ADM_s107
                                 + 0.817469 w_k1_AXP_s107
                                 + 0.708316 w_k1_DIS_s107
                                 + 0.851931 w_k1_FDX_s107
                                 + 0.920167 w_k1_IBM_s107
                                 + 0.889113 w_k1_INTC_s107
                                 + 0.773396 w_k1_KO_s107
                                 + 0.787299 w_k1_MAT_s107
                                 + 0.803458 w_k1_MSFT_s107
                                 + 0.799402 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.887735 w_k1_ADM_s107
                                 + 1.01478 w_k1_AXP_s107
                                 + 1.05766 w_k1_DIS_s107
                                 + 0.989682 w_k1_FDX_s107
                                 + 1.26504 w_k1_IBM_s107
                                 + 0.946118 w_k1_INTC_s107
                                 + 1.16798 w_k1_KO_s107 + 1.17852 w_k1_MAT_s107
                                 + 0.94128 w_k1_MSFT_s107
                                 + 1.04695 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.15498 w_k1_ADM_s107
                                 + 0.956023 w_k1_AXP_s107
                                 + 0.854227 w_k1_DIS_s107
                                 + 0.974789 w_k1_FDX_s107
                                 + 0.649786 w_k1_IBM_s107
                                 + 0.812931 w_k1_INTC_s107
                                 + 0.766695 w_k1_KO_s107
                                 + 0.456728 w_k1_MAT_s107
                                 + 0.832268 w_k1_MSFT_s107
                                 + 0.816061 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.07079 w_k1_ADM_s107
                                 + 1.3837 w_k1_AXP_s107 + 1.31201 w_k1_DIS_s107
                                 + 1.1249 w_k1_FDX_s107 + 1.74993 w_k1_IBM_s107
                                 + 1.27668 w_k1_INTC_s107
                                 + 1.36482 w_k1_KO_s107 + 1.76895 w_k1_MAT_s107
                                 + 1.22136 w_k1_MSFT_s107
                                 + 1.17262 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.945797 w_k1_ADM_s107
                                 + 1.11548 w_k1_AXP_s107
                                 + 0.869086 w_k1_DIS_s107
                                 + 0.973282 w_k1_FDX_s107
                                 + 0.953167 w_k1_IBM_s107
                                 + 0.901957 w_k1_INTC_s107
                                 + 0.837942 w_k1_KO_s107
                                 + 1.05716 w_k1_MAT_s107
                                 + 0.934869 w_k1_MSFT_s107
                                 + 0.945953 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.626548 w_k1_ADM_s107
                                 + 0.712068 w_k1_AXP_s107
                                 + 0.679298 w_k1_DIS_s107
                                 + 0.784122 w_k1_FDX_s107
                                 + 0.733297 w_k1_IBM_s107
                                 + 1.00741 w_k1_INTC_s107
                                 + 0.752662 w_k1_KO_s107
                                 + 0.777363 w_k1_MAT_s107
                                 + 0.888324 w_k1_MSFT_s107
                                 + 0.877132 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.41364 w_k1_ADM_s107
                                 + 1.44945 w_k1_AXP_s107
                                 + 1.29359 w_k1_DIS_s107
                                 + 1.28083 w_k1_FDX_s107
                                 + 1.17201 w_k1_IBM_s107
                                 + 1.24272 w_k1_INTC_s107
                                 + 1.17667 w_k1_KO_s107
                                 + 0.929642 w_k1_MAT_s107
                                 + 1.13941 w_k1_MSFT_s107
                                 + 1.04198 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.05102 w_k1_ADM_s107
                                 + 1.01457 w_k1_AXP_s107
                                 + 1.18881 w_k1_DIS_s107
                                 + 0.893514 w_k1_FDX_s107
                                 + 1.1404 w_k1_IBM_s107
                                 + 1.03319 w_k1_INTC_s107
                                 + 1.04362 w_k1_KO_s107
                                 + 0.776686 w_k1_MAT_s107
                                 + 0.992393 w_k1_MSFT_s107
                                 + 0.943537 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.11064 w_k1_ADM_s110
                                 + 1.20355 w_k1_AXP_s110
                                 + 1.95923 w_k1_DIS_s110
                                 + 1.23904 w_k1_FDX_s110
                                 + 1.59459 w_k1_IBM_s110
                                 + 1.28861 w_k1_INTC_s110 + 1.6743 w_k1_KO_s110
                                 + 1.32097 w_k1_MAT_s110
                                 + 1.08428 w_k1_MSFT_s110
                                 + 1.27623 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.56191 w_k1_ADM_s110
                                 + 1.5181 w_k1_AXP_s110 + 1.56986 w_k1_DIS_s110
                                 + 1.42235 w_k1_FDX_s110
                                 + 1.25133 w_k1_IBM_s110
                                 + 1.23588 w_k1_INTC_s110
                                 + 1.32213 w_k1_KO_s110 + 1.3531 w_k1_MAT_s110
                                 + 1.23629 w_k1_MSFT_s110 + 1.116 w_k1_PG_s110
                                 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.17814 w_k1_ADM_s110
                                 + 1.17811 w_k1_AXP_s110
                                 + 1.36109 w_k1_DIS_s110
                                 + 1.19519 w_k1_FDX_s110
                                 + 0.884306 w_k1_IBM_s110
                                 + 1.33051 w_k1_INTC_s110
                                 + 1.30603 w_k1_KO_s110
                                 + 0.793043 w_k1_MAT_s110
                                 + 1.24363 w_k1_MSFT_s110
                                 + 1.11928 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.14686 w_k1_ADM_s110
                                 + 1.20076 w_k1_AXP_s110
                                 + 0.978141 w_k1_DIS_s110
                                 + 1.06808 w_k1_FDX_s110
                                 + 0.869371 w_k1_IBM_s110
                                 + 1.10172 w_k1_INTC_s110
                                 + 1.06532 w_k1_KO_s110
                                 + 0.823934 w_k1_MAT_s110
                                 + 0.97703 w_k1_MSFT_s110
                                 + 1.08445 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.02044 w_k1_ADM_s110
                                 + 0.982285 w_k1_AXP_s110
                                 + 0.830589 w_k1_DIS_s110
                                 + 0.888338 w_k1_FDX_s110
                                 + 0.79839 w_k1_IBM_s110
                                 + 0.817896 w_k1_INTC_s110
                                 + 0.872811 w_k1_KO_s110
                                 + 0.966288 w_k1_MAT_s110
                                 + 0.802991 w_k1_MSFT_s110
                                 + 0.948755 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.924793 w_k1_ADM_s110
                                 + 0.735532 w_k1_AXP_s110
                                 + 0.955726 w_k1_DIS_s110
                                 + 0.932507 w_k1_FDX_s110
                                 + 0.815696 w_k1_IBM_s110
                                 + 0.977534 w_k1_INTC_s110
                                 + 1.0436 w_k1_KO_s110 + 0.586313 w_k1_MAT_s110
                                 + 0.909357 w_k1_MSFT_s110
                                 + 0.914287 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.61102 w_k1_ADM_s110
                                 + 0.590239 w_k1_AXP_s110
                                 + 0.683364 w_k1_DIS_s110
                                 + 0.729797 w_k1_FDX_s110
                                 + 0.851096 w_k1_IBM_s110
                                 + 0.840412 w_k1_INTC_s110
                                 + 0.845864 w_k1_KO_s110
                                 + 0.909548 w_k1_MAT_s110
                                 + 0.821857 w_k1_MSFT_s110
                                 + 0.867794 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.935319 w_k1_ADM_s110
                                 + 1.05353 w_k1_AXP_s110
                                 + 1.09944 w_k1_DIS_s110
                                 + 0.91445 w_k1_FDX_s110
                                 + 1.40392 w_k1_IBM_s110
                                 + 1.00765 w_k1_INTC_s110
                                 + 1.22349 w_k1_KO_s110 + 1.28961 w_k1_MAT_s110
                                 + 0.905461 w_k1_MSFT_s110
                                 + 1.0667 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.836691 w_k1_ADM_s111
                                 + 0.666382 w_k1_AXP_s111
                                 + 0.624951 w_k1_DIS_s111
                                 + 0.804421 w_k1_FDX_s111
                                 + 0.697405 w_k1_IBM_s111
                                 + 0.879644 w_k1_INTC_s111
                                 + 0.666125 w_k1_KO_s111
                                 + 0.521101 w_k1_MAT_s111
                                 + 0.812302 w_k1_MSFT_s111
                                 + 0.897071 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.19021 w_k1_ADM_s111
                                 + 0.806156 w_k1_AXP_s111
                                 + 0.830885 w_k1_DIS_s111
                                 + 0.960334 w_k1_FDX_s111
                                 + 0.708733 w_k1_IBM_s111
                                 + 0.94849 w_k1_INTC_s111
                                 + 0.734488 w_k1_KO_s111
                                 + 0.759675 w_k1_MAT_s111
                                 + 0.941013 w_k1_MSFT_s111
                                 + 1.04282 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.20423 w_k1_ADM_s111
                                 + 1.02532 w_k1_AXP_s111
                                 + 1.08109 w_k1_DIS_s111 + 1.0542 w_k1_FDX_s111
                                 + 1.03776 w_k1_IBM_s111
                                 + 0.928393 w_k1_INTC_s111
                                 + 1.06892 w_k1_KO_s111 + 1.13928 w_k1_MAT_s111
                                 + 1.03373 w_k1_MSFT_s111
                                 + 1.09838 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.57265 w_k1_ADM_s111
                                 + 1.47005 w_k1_AXP_s111
                                 + 1.44169 w_k1_DIS_s111
                                 + 1.44932 w_k1_FDX_s111
                                 + 1.33297 w_k1_IBM_s111
                                 + 1.22013 w_k1_INTC_s111 + 1.0729 w_k1_KO_s111
                                 + 1.34324 w_k1_MAT_s111
                                 + 1.31567 w_k1_MSFT_s111 + 1.2645 w_k1_PG_s111
                                 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.1713 w_k1_ADM_s111
                                 + 1.19145 w_k1_AXP_s111
                                 + 1.22924 w_k1_DIS_s111
                                 + 1.31747 w_k1_FDX_s111
                                 + 0.867261 w_k1_IBM_s111
                                 + 1.24191 w_k1_INTC_s111
                                 + 1.06183 w_k1_KO_s111 + 0.84364 w_k1_MAT_s111
                                 + 1.14716 w_k1_MSFT_s111
                                 + 1.04355 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.17121 w_k1_ADM_s111
                                 + 1.32021 w_k1_AXP_s111
                                 + 1.21542 w_k1_DIS_s111
                                 + 1.15378 w_k1_FDX_s111
                                 + 1.49081 w_k1_IBM_s111
                                 + 1.12985 w_k1_INTC_s111
                                 + 1.13455 w_k1_KO_s111 + 1.67352 w_k1_MAT_s111
                                 + 1.11345 w_k1_MSFT_s111
                                 + 1.24736 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.977109 w_k1_ADM_s111
                                 + 0.721592 w_k1_AXP_s111
                                 + 0.686383 w_k1_DIS_s111
                                 + 0.816786 w_k1_FDX_s111
                                 + 1.03176 w_k1_IBM_s111
                                 + 1.01369 w_k1_INTC_s111
                                 + 0.867053 w_k1_KO_s111
                                 + 0.65611 w_k1_MAT_s111
                                 + 0.846139 w_k1_MSFT_s111
                                 + 0.999689 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.17012 w_k1_ADM_s111
                                 + 0.992293 w_k1_AXP_s111
                                 + 0.817827 w_k1_DIS_s111
                                 + 0.893983 w_k1_FDX_s111
                                 + 1.23624 w_k1_IBM_s111
                                 + 0.945612 w_k1_INTC_s111
                                 + 0.890605 w_k1_KO_s111
                                 + 1.01747 w_k1_MAT_s111
                                 + 0.86783 w_k1_MSFT_s111
                                 + 1.08922 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.83193 w_k1_ADM_s112
                                 + 0.888218 w_k1_AXP_s112
                                 + 1.18854 w_k1_DIS_s112
                                 + 1.13451 w_k1_FDX_s112
                                 + 0.882486 w_k1_IBM_s112
                                 + 1.20695 w_k1_INTC_s112 + 1.0713 w_k1_KO_s112
                                 + 0.777555 w_k1_MAT_s112
                                 + 1.0602 w_k1_MSFT_s112 + 1.15241 w_k1_PG_s112
                                 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.940371 w_k1_ADM_s112
                                 + 0.978982 w_k1_AXP_s112
                                 + 0.97329 w_k1_DIS_s112
                                 + 0.957443 w_k1_FDX_s112
                                 + 1.3217 w_k1_IBM_s112
                                 + 0.963275 w_k1_INTC_s112
                                 + 1.07864 w_k1_KO_s112 + 1.18906 w_k1_MAT_s112
                                 + 0.89429 w_k1_MSFT_s112
                                 + 1.05691 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.698407 w_k1_ADM_s112
                                 + 0.611352 w_k1_AXP_s112
                                 + 0.535719 w_k1_DIS_s112
                                 + 0.729924 w_k1_FDX_s112
                                 + 0.648502 w_k1_IBM_s112
                                 + 0.916447 w_k1_INTC_s112
                                 + 0.703369 w_k1_KO_s112
                                 + 0.933505 w_k1_MAT_s112
                                 + 0.815566 w_k1_MSFT_s112
                                 + 0.824365 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.830987 w_k1_ADM_s112
                                 + 0.830069 w_k1_AXP_s112
                                 + 0.714948 w_k1_DIS_s112
                                 + 0.84153 w_k1_FDX_s112
                                 + 0.886632 w_k1_IBM_s112
                                 + 0.847259 w_k1_INTC_s112
                                 + 0.87416 w_k1_KO_s112 + 0.89508 w_k1_MAT_s112
                                 + 0.867858 w_k1_MSFT_s112
                                 + 0.956334 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.15459 w_k1_ADM_s112
                                 + 1.1811 w_k1_AXP_s112 + 1.57847 w_k1_DIS_s112
                                 + 1.38619 w_k1_FDX_s112
                                 + 1.03861 w_k1_IBM_s112
                                 + 1.18344 w_k1_INTC_s112
                                 + 1.20271 w_k1_KO_s112 + 1.21881 w_k1_MAT_s112
                                 + 1.38817 w_k1_MSFT_s112 + 1.1562 w_k1_PG_s112
                                 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.861447 w_k1_ADM_s112
                                 + 0.742515 w_k1_AXP_s112
                                 + 1.03562 w_k1_DIS_s112
                                 + 0.977581 w_k1_FDX_s112
                                 + 0.637834 w_k1_IBM_s112
                                 + 0.983189 w_k1_INTC_s112
                                 + 0.923829 w_k1_KO_s112
                                 + 0.736081 w_k1_MAT_s112
                                 + 1.0843 w_k1_MSFT_s112 + 1.01654 w_k1_PG_s112
                                 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.00523 w_k1_ADM_s112
                                 + 1.06337 w_k1_AXP_s112
                                 + 0.923619 w_k1_DIS_s112
                                 + 1.03726 w_k1_FDX_s112
                                 + 0.856445 w_k1_IBM_s112
                                 + 0.746129 w_k1_INTC_s112
                                 + 0.799631 w_k1_KO_s112
                                 + 1.31417 w_k1_MAT_s112
                                 + 0.991233 w_k1_MSFT_s112
                                 + 0.910186 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.36516 w_k1_ADM_s112
                                 + 1.44359 w_k1_AXP_s112
                                 + 1.26814 w_k1_DIS_s112
                                 + 1.16412 w_k1_FDX_s112
                                 + 1.49667 w_k1_IBM_s112
                                 + 1.20396 w_k1_INTC_s112
                                 + 1.40479 w_k1_KO_s112 + 2.32253 w_k1_MAT_s112
                                 + 1.07887 w_k1_MSFT_s112
                                 + 1.27746 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.16561 w_k1_ADM_s113
                                 + 1.15014 w_k1_AXP_s113
                                 + 0.881813 w_k1_DIS_s113
                                 + 0.929408 w_k1_FDX_s113
                                 + 1.06926 w_k1_IBM_s113
                                 + 1.18916 w_k1_INTC_s113
                                 + 0.917366 w_k1_KO_s113
                                 + 0.831124 w_k1_MAT_s113
                                 + 0.965041 w_k1_MSFT_s113
                                 + 1.02288 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.40769 w_k1_ADM_s113
                                 + 1.10538 w_k1_AXP_s113 + 1.0254 w_k1_DIS_s113
                                 + 1.19868 w_k1_FDX_s113
                                 + 0.926614 w_k1_IBM_s113
                                 + 1.04094 w_k1_INTC_s113
                                 + 0.996472 w_k1_KO_s113
                                 + 0.762469 w_k1_MAT_s113
                                 + 1.05996 w_k1_MSFT_s113
                                 + 1.03018 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.21263 w_k1_ADM_s113
                                 + 1.02089 w_k1_AXP_s113
                                 + 1.27431 w_k1_DIS_s113
                                 + 1.24399 w_k1_FDX_s113
                                 + 0.846066 w_k1_IBM_s113
                                 + 1.16051 w_k1_INTC_s113
                                 + 0.892062 w_k1_KO_s113
                                 + 0.77494 w_k1_MAT_s113
                                 + 1.13342 w_k1_MSFT_s113
                                 + 1.05368 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.37129 w_k1_ADM_s113
                                 + 1.39948 w_k1_AXP_s113
                                 + 1.44966 w_k1_DIS_s113
                                 + 1.23411 w_k1_FDX_s113
                                 + 1.37807 w_k1_IBM_s113
                                 + 1.42366 w_k1_INTC_s113
                                 + 1.27807 w_k1_KO_s113 + 1.36064 w_k1_MAT_s113
                                 + 1.23461 w_k1_MSFT_s113
                                 + 1.21274 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.952537 w_k1_ADM_s113
                                 + 0.775358 w_k1_AXP_s113
                                 + 0.636386 w_k1_DIS_s113
                                 + 0.766345 w_k1_FDX_s113
                                 + 0.786713 w_k1_IBM_s113
                                 + 0.845184 w_k1_INTC_s113
                                 + 0.76282 w_k1_KO_s113
                                 + 0.767209 w_k1_MAT_s113
                                 + 0.828553 w_k1_MSFT_s113
                                 + 0.87115 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.06495 w_k1_ADM_s113
                                 + 0.882937 w_k1_AXP_s113
                                 + 0.682418 w_k1_DIS_s113
                                 + 0.856778 w_k1_FDX_s113
                                 + 0.574915 w_k1_IBM_s113
                                 + 0.669681 w_k1_INTC_s113
                                 + 0.685369 w_k1_KO_s113
                                 + 0.663302 w_k1_MAT_s113
                                 + 0.858357 w_k1_MSFT_s113
                                 + 0.929391 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.06882 w_k1_ADM_s113
                                 + 0.90882 w_k1_AXP_s113
                                 + 0.798677 w_k1_DIS_s113
                                 + 0.837113 w_k1_FDX_s113
                                 + 1.23559 w_k1_IBM_s113
                                 + 0.926264 w_k1_INTC_s113
                                 + 1.07832 w_k1_KO_s113 + 1.01194 w_k1_MAT_s113
                                 + 0.840947 w_k1_MSFT_s113
                                 + 0.962009 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.943364 w_k1_ADM_s113
                                 + 0.631761 w_k1_AXP_s113
                                 + 0.92067 w_k1_DIS_s113
                                 + 0.943382 w_k1_FDX_s113
                                 + 0.735663 w_k1_IBM_s113
                                 + 0.839438 w_k1_INTC_s113
                                 + 0.841652 w_k1_KO_s113
                                 + 0.688219 w_k1_MAT_s113
                                 + 0.987982 w_k1_MSFT_s113
                                 + 1.06111 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.23703 w_k1_ADM_s114
                                 + 1.3281 w_k1_AXP_s114 + 1.21801 w_k1_DIS_s114
                                 + 1.17453 w_k1_FDX_s114
                                 + 1.26501 w_k1_IBM_s114
                                 + 1.22151 w_k1_INTC_s114
                                 + 1.10601 w_k1_KO_s114 + 1.07593 w_k1_MAT_s114
                                 + 1.14048 w_k1_MSFT_s114
                                 + 1.23655 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.176 w_k1_ADM_s114
                                 + 0.993567 w_k1_AXP_s114
                                 + 1.05788 w_k1_DIS_s114
                                 + 0.989422 w_k1_FDX_s114
                                 + 1.21411 w_k1_IBM_s114
                                 + 1.1785 w_k1_INTC_s114 + 1.08206 w_k1_KO_s114
                                 + 0.932769 w_k1_MAT_s114
                                 + 1.04672 w_k1_MSFT_s114
                                 + 1.07766 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.914624 w_k1_ADM_s114
                                 + 0.731866 w_k1_AXP_s114
                                 + 0.869004 w_k1_DIS_s114
                                 + 0.849304 w_k1_FDX_s114
                                 + 0.745056 w_k1_IBM_s114
                                 + 0.896997 w_k1_INTC_s114
                                 + 0.725314 w_k1_KO_s114
                                 + 0.607489 w_k1_MAT_s114
                                 + 0.854326 w_k1_MSFT_s114
                                 + 0.93689 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.08575 w_k1_ADM_s114
                                 + 1.07927 w_k1_AXP_s114
                                 + 0.917295 w_k1_DIS_s114
                                 + 0.991926 w_k1_FDX_s114
                                 + 0.952875 w_k1_IBM_s114
                                 + 0.930738 w_k1_INTC_s114
                                 + 0.73665 w_k1_KO_s114
                                 + 0.770011 w_k1_MAT_s114
                                 + 0.869606 w_k1_MSFT_s114
                                 + 0.968519 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.03971 w_k1_ADM_s114
                                 + 0.783145 w_k1_AXP_s114
                                 + 0.71392 w_k1_DIS_s114
                                 + 0.74123 w_k1_FDX_s114
                                 + 0.963656 w_k1_IBM_s114
                                 + 0.785997 w_k1_INTC_s114
                                 + 0.6985 w_k1_KO_s114 + 0.970554 w_k1_MAT_s114
                                 + 0.827094 w_k1_MSFT_s114
                                 + 0.931884 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.02124 w_k1_ADM_s114
                                 + 0.721935 w_k1_AXP_s114
                                 + 1.18543 w_k1_DIS_s114
                                 + 0.982262 w_k1_FDX_s114
                                 + 0.79089 w_k1_IBM_s114
                                 + 0.980423 w_k1_INTC_s114
                                 + 0.84181 w_k1_KO_s114
                                 + 0.401567 w_k1_MAT_s114
                                 + 1.06682 w_k1_MSFT_s114
                                 + 1.06205 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.993697 w_k1_ADM_s114
                                 + 1.19373 w_k1_AXP_s114
                                 + 0.851895 w_k1_DIS_s114
                                 + 0.928853 w_k1_FDX_s114
                                 + 1.52493 w_k1_IBM_s114
                                 + 0.910116 w_k1_INTC_s114
                                 + 0.972696 w_k1_KO_s114
                                 + 1.24552 w_k1_MAT_s114
                                 + 0.887381 w_k1_MSFT_s114
                                 + 1.04484 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.66123 w_k1_ADM_s114
                                 + 1.49727 w_k1_AXP_s114
                                 + 1.62128 w_k1_DIS_s114 + 1.3863 w_k1_FDX_s114
                                 + 1.04723 w_k1_IBM_s114
                                 + 1.39397 w_k1_INTC_s114
                                 + 1.12913 w_k1_KO_s114 + 0.87736 w_k1_MAT_s114
                                 + 1.36972 w_k1_MSFT_s114
                                 + 1.19595 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.41158 w_k1_ADM_s115
                                 + 1.39136 w_k1_AXP_s115
                                 + 1.04743 w_k1_DIS_s115
                                 + 1.01711 w_k1_FDX_s115
                                 + 1.61771 w_k1_IBM_s115
                                 + 1.14185 w_k1_INTC_s115
                                 + 1.26605 w_k1_KO_s115 + 1.77281 w_k1_MAT_s115
                                 + 0.936104 w_k1_MSFT_s115
                                 + 0.974279 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.982849 w_k1_ADM_s115
                                 + 0.653209 w_k1_AXP_s115
                                 + 0.704042 w_k1_DIS_s115
                                 + 0.791948 w_k1_FDX_s115
                                 + 1.05828 w_k1_IBM_s115
                                 + 0.768898 w_k1_INTC_s115
                                 + 0.922418 w_k1_KO_s115
                                 + 1.20338 w_k1_MAT_s115
                                 + 0.763103 w_k1_MSFT_s115
                                 + 0.831379 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.33233 w_k1_ADM_s115
                                 + 1.26176 w_k1_AXP_s115
                                 + 1.42165 w_k1_DIS_s115
                                 + 1.32061 w_k1_FDX_s115
                                 + 1.33223 w_k1_IBM_s115
                                 + 1.18487 w_k1_INTC_s115
                                 + 1.29679 w_k1_KO_s115 + 1.53112 w_k1_MAT_s115
                                 + 1.2747 w_k1_MSFT_s115 + 1.1794 w_k1_PG_s115
                                 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.27947 w_k1_ADM_s115
                                 + 0.928044 w_k1_AXP_s115
                                 + 0.801726 w_k1_DIS_s115
                                 + 0.944853 w_k1_FDX_s115
                                 + 0.777608 w_k1_IBM_s115
                                 + 0.965325 w_k1_INTC_s115
                                 + 0.806281 w_k1_KO_s115
                                 + 0.790896 w_k1_MAT_s115
                                 + 1.05254 w_k1_MSFT_s115 + 0.9658 w_k1_PG_s115
                                 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.02316 w_k1_ADM_s115
                                 + 0.744877 w_k1_AXP_s115
                                 + 0.615462 w_k1_DIS_s115
                                 + 0.73599 w_k1_FDX_s115
                                 + 0.695521 w_k1_IBM_s115
                                 + 0.768073 w_k1_INTC_s115
                                 + 0.713102 w_k1_KO_s115
                                 + 0.692479 w_k1_MAT_s115
                                 + 0.755779 w_k1_MSFT_s115
                                 + 0.762805 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.37868 w_k1_ADM_s115
                                 + 1.03842 w_k1_AXP_s115
                                 + 1.14727 w_k1_DIS_s115
                                 + 1.14163 w_k1_FDX_s115
                                 + 0.822474 w_k1_IBM_s115
                                 + 0.952143 w_k1_INTC_s115
                                 + 0.98707 w_k1_KO_s115 + 1.10496 w_k1_MAT_s115
                                 + 1.17975 w_k1_MSFT_s115 + 1.0646 w_k1_PG_s115
                                 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.949755 w_k1_ADM_s115
                                 + 0.617562 w_k1_AXP_s115
                                 + 0.963866 w_k1_DIS_s115
                                 + 0.913862 w_k1_FDX_s115
                                 + 0.969744 w_k1_IBM_s115
                                 + 0.894649 w_k1_INTC_s115
                                 + 0.976917 w_k1_KO_s115
                                 + 0.731211 w_k1_MAT_s115
                                 + 0.998709 w_k1_MSFT_s115
                                 + 0.877524 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.42518 w_k1_ADM_s115
                                 + 1.70964 w_k1_AXP_s115
                                 + 1.44935 w_k1_DIS_s115
                                 + 1.59286 w_k1_FDX_s115
                                 + 0.93548 w_k1_IBM_s115
                                 + 1.46693 w_k1_INTC_s115
                                 + 1.14686 w_k1_KO_s115 + 1.18707 w_k1_MAT_s115
                                 + 1.39287 w_k1_MSFT_s115 + 1.0944 w_k1_PG_s115
                                 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.930969 w_k1_ADM_s116
                                 + 0.990973 w_k1_AXP_s116
                                 + 0.929306 w_k1_DIS_s116
                                 + 1.05642 w_k1_FDX_s116
                                 + 0.97218 w_k1_IBM_s116
                                 + 1.28686 w_k1_INTC_s116
                                 + 0.929089 w_k1_KO_s116
                                 + 0.861704 w_k1_MAT_s116
                                 + 1.13761 w_k1_MSFT_s116
                                 + 1.03202 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.74764 w_k1_ADM_s116
                                 + 1.30252 w_k1_AXP_s116
                                 + 1.29783 w_k1_DIS_s116 + 1.1296 w_k1_FDX_s116
                                 + 1.734 w_k1_IBM_s116 + 1.28538 w_k1_INTC_s116
                                 + 1.315 w_k1_KO_s116 + 1.72368 w_k1_MAT_s116
                                 + 1.18119 w_k1_MSFT_s116
                                 + 1.10551 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.770064 w_k1_ADM_s116
                                 + 0.745251 w_k1_AXP_s116
                                 + 0.99524 w_k1_DIS_s116
                                 + 0.886682 w_k1_FDX_s116
                                 + 1.1622 w_k1_IBM_s116
                                 + 1.10299 w_k1_INTC_s116
                                 + 0.997489 w_k1_KO_s116
                                 + 0.777753 w_k1_MAT_s116
                                 + 0.986445 w_k1_MSFT_s116
                                 + 0.985321 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.740527 w_k1_ADM_s116
                                 + 0.738017 w_k1_AXP_s116
                                 + 0.786211 w_k1_DIS_s116
                                 + 0.84274 w_k1_FDX_s116
                                 + 1.17446 w_k1_IBM_s116
                                 + 0.868518 w_k1_INTC_s116
                                 + 0.876392 w_k1_KO_s116
                                 + 1.15757 w_k1_MAT_s116
                                 + 0.892371 w_k1_MSFT_s116
                                 + 0.894743 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.10169 w_k1_ADM_s116
                                 + 0.789564 w_k1_AXP_s116
                                 + 0.85651 w_k1_DIS_s116
                                 + 0.978652 w_k1_FDX_s116
                                 + 0.692682 w_k1_IBM_s116
                                 + 1.0233 w_k1_INTC_s116
                                 + 0.701954 w_k1_KO_s116
                                 + 0.515489 w_k1_MAT_s116
                                 + 0.965714 w_k1_MSFT_s116
                                 + 0.898045 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.2623 w_k1_ADM_s116
                                 + 1.24314 w_k1_AXP_s116
                                 + 1.54345 w_k1_DIS_s116
                                 + 1.25922 w_k1_FDX_s116
                                 + 1.10121 w_k1_IBM_s116
                                 + 1.33423 w_k1_INTC_s116
                                 + 1.18724 w_k1_KO_s116
                                 + 0.862797 w_k1_MAT_s116
                                 + 1.20738 w_k1_MSFT_s116
                                 + 1.09911 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.603181 w_k1_ADM_s116
                                 + 0.593798 w_k1_AXP_s116
                                 + 0.49286 w_k1_DIS_s116
                                 + 0.614857 w_k1_FDX_s116
                                 + 0.793134 w_k1_IBM_s116
                                 + 0.644529 w_k1_INTC_s116
                                 + 0.626329 w_k1_KO_s116
                                 + 0.784867 w_k1_MAT_s116
                                 + 0.592249 w_k1_MSFT_s116
                                 + 0.71665 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.733172 w_k1_ADM_s116
                                 + 0.727402 w_k1_AXP_s116
                                 + 0.786962 w_k1_DIS_s116
                                 + 0.810349 w_k1_FDX_s116
                                 + 0.729878 w_k1_IBM_s116
                                 + 0.955258 w_k1_INTC_s116
                                 + 0.72509 w_k1_KO_s116
                                 + 0.658784 w_k1_MAT_s116
                                 + 0.835976 w_k1_MSFT_s116
                                 + 0.794914 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.30511 w_k1_ADM_s117
                                 + 1.61502 w_k1_AXP_s117
                                 + 2.02169 w_k1_DIS_s117
                                 + 1.54379 w_k1_FDX_s117
                                 + 1.41206 w_k1_IBM_s117
                                 + 1.29177 w_k1_INTC_s117 + 1.5053 w_k1_KO_s117
                                 + 1.83876 w_k1_MAT_s117
                                 + 1.58216 w_k1_MSFT_s117
                                 + 1.28558 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.967787 w_k1_ADM_s117
                                 + 1.03158 w_k1_AXP_s117
                                 + 0.825943 w_k1_DIS_s117
                                 + 0.915096 w_k1_FDX_s117
                                 + 0.809747 w_k1_IBM_s117
                                 + 0.851217 w_k1_INTC_s117
                                 + 0.863415 w_k1_KO_s117
                                 + 1.22515 w_k1_MAT_s117
                                 + 1.0095 w_k1_MSFT_s117
                                 + 0.946987 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.17513 w_k1_ADM_s117
                                 + 1.23593 w_k1_AXP_s117
                                 + 1.13527 w_k1_DIS_s117
                                 + 1.23074 w_k1_FDX_s117
                                 + 0.948535 w_k1_IBM_s117
                                 + 1.09092 w_k1_INTC_s117
                                 + 1.06604 w_k1_KO_s117
                                 + 0.915517 w_k1_MAT_s117
                                 + 1.26444 w_k1_MSFT_s117
                                 + 1.18718 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.738984 w_k1_ADM_s117
                                 + 0.66237 w_k1_AXP_s117
                                 + 0.599888 w_k1_DIS_s117
                                 + 0.714936 w_k1_FDX_s117
                                 + 0.998578 w_k1_IBM_s117
                                 + 0.809982 w_k1_INTC_s117
                                 + 0.724111 w_k1_KO_s117
                                 + 0.998605 w_k1_MAT_s117
                                 + 0.809385 w_k1_MSFT_s117
                                 + 0.831261 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.03283 w_k1_ADM_s117
                                 + 0.725901 w_k1_AXP_s117
                                 + 0.670046 w_k1_DIS_s117
                                 + 0.814437 w_k1_FDX_s117
                                 + 0.837873 w_k1_IBM_s117
                                 + 0.842888 w_k1_INTC_s117
                                 + 0.849761 w_k1_KO_s117
                                 + 0.752167 w_k1_MAT_s117
                                 + 0.803624 w_k1_MSFT_s117
                                 + 0.800414 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.974094 w_k1_ADM_s117
                                 + 1.15145 w_k1_AXP_s117
                                 + 1.09951 w_k1_DIS_s117 + 1.0016 w_k1_FDX_s117
                                 + 1.33634 w_k1_IBM_s117
                                 + 0.904806 w_k1_INTC_s117
                                 + 1.03579 w_k1_KO_s117 + 1.63969 w_k1_MAT_s117
                                 + 1.01999 w_k1_MSFT_s117
                                 + 1.12513 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.994105 w_k1_ADM_s117
                                 + 0.976732 w_k1_AXP_s117
                                 + 1.04938 w_k1_DIS_s117
                                 + 1.00951 w_k1_FDX_s117
                                 + 0.819902 w_k1_IBM_s117
                                 + 1.05075 w_k1_INTC_s117
                                 + 0.929188 w_k1_KO_s117
                                 + 0.945827 w_k1_MAT_s117
                                 + 1.13441 w_k1_MSFT_s117
                                 + 1.01621 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.32588 w_k1_ADM_s117
                                 + 1.45649 w_k1_AXP_s117
                                 + 1.37562 w_k1_DIS_s117
                                 + 1.37175 w_k1_FDX_s117
                                 + 1.15903 w_k1_IBM_s117
                                 + 0.992907 w_k1_INTC_s117
                                 + 1.18984 w_k1_KO_s117 + 1.77397 w_k1_MAT_s117
                                 + 1.27965 w_k1_MSFT_s117
                                 + 1.27789 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0104720817644878 w_k1_ADM_s000
                                 + 0.00548015371673823 w_k1_ADM_s001
                                 + 0.00960646721594622 w_k1_ADM_s002
                                 + 0.0273173435579326 w_k1_ADM_s003
                                 + 0.0161481600602973 w_k1_ADM_s004
                                 + 0.0104812045849836 w_k1_ADM_s005
                                 + 0.00917709467743887 w_k1_ADM_s006
                                 + 0.027354085761273 w_k1_ADM_s007
                                 + 0.0326256086032067 w_k1_ADM_s010
                                 + 0.0513085428795856 w_k1_ADM_s011
                                 + 0.0310880696744242 w_k1_ADM_s012
                                 + 0.0205045404894162 w_k1_ADM_s013
                                 + 0.0667023142752854 w_k1_ADM_s014
                                 + 0.0419570032350745 w_k1_ADM_s015
                                 + 0.0182325671916944 w_k1_ADM_s016
                                 + 0.0891235373887426 w_k1_ADM_s017
                                 + 0.0367868596985135 w_k1_ADM_s100
                                 + 0.057944673521309 w_k1_ADM_s101
                                 + 0.0229807713148317 w_k1_ADM_s102
                                 + 0.0490400813093138 w_k1_ADM_s103
                                 + 0.0350446823587326 w_k1_ADM_s104
                                 + 0.0135556447030072 w_k1_ADM_s105
                                 + 0.0324837327152329 w_k1_ADM_s106
                                 + 0.0264448860047987 w_k1_ADM_s107
                                 + 0.0625761856325542 w_k1_ADM_s110
                                 + 0.0704740276281691 w_k1_ADM_s111
                                 + 0.0142966355801895 w_k1_ADM_s112
                                 + 0.0206277909142826 w_k1_ADM_s113
                                 + 0.0490853554223546 w_k1_ADM_s114
                                 + 0.0173436742131991 w_k1_ADM_s115
                                 + 0.0217040074972024 w_k1_ADM_s116
                                 + 0.0397204806218208 w_k1_ADM_s117
                                 + 0.011320927460383 w_k1_AXP_s000
                                 + 0.00574941344392854 w_k1_AXP_s001
                                 + 0.011324968437874 w_k1_AXP_s002
                                 + 0.0231208700622441 w_k1_AXP_s003
                                 + 0.0157105859190262 w_k1_AXP_s004
                                 + 0.010338644230712 w_k1_AXP_s005
                                 + 0.00984463236061581 w_k1_AXP_s006
                                 + 0.0242761618296842 w_k1_AXP_s007
                                 + 0.0313566442833557 w_k1_AXP_s010
                                 + 0.0441723793237692 w_k1_AXP_s011
                                 + 0.0295947547345643 w_k1_AXP_s012
                                 + 0.0180799516593247 w_k1_AXP_s013
                                 + 0.0688067896981915 w_k1_AXP_s014
                                 + 0.0402849188055178 w_k1_AXP_s015
                                 + 0.0199024032798837 w_k1_AXP_s016
                                 + 0.0837695053181398 w_k1_AXP_s017
                                 + 0.0408431996036764 w_k1_AXP_s100
                                 + 0.0623208682202279 w_k1_AXP_s101
                                 + 0.0199910168330843 w_k1_AXP_s102
                                 + 0.0510835130556479 w_k1_AXP_s103
                                 + 0.037029410837019 w_k1_AXP_s104
                                 + 0.0157882655341945 w_k1_AXP_s105
                                 + 0.0334417061847854 w_k1_AXP_s106
                                 + 0.0283841787811547 w_k1_AXP_s107
                                 + 0.0618253021913974 w_k1_AXP_s110
                                 + 0.0610411683574909 w_k1_AXP_s111
                                 + 0.0143716172481263 w_k1_AXP_s112
                                 + 0.0182928820492879 w_k1_AXP_s113
                                 + 0.0452251992996941 w_k1_AXP_s114
                                 + 0.013857734625238 w_k1_AXP_s115
                                 + 0.0220745645967192 w_k1_AXP_s116
                                 + 0.0405928588119479 w_k1_AXP_s117
                                 + 0.0107999468786647 w_k1_DIS_s000
                                 + 0.00594410248896675 w_k1_DIS_s001
                                 + 0.0111158295777739 w_k1_DIS_s002
                                 + 0.0224369746400322 w_k1_DIS_s003
                                 + 0.0147891872176515 w_k1_DIS_s004
                                 + 0.0102941951930018 w_k1_DIS_s005
                                 + 0.0104057756449407 w_k1_DIS_s006
                                 + 0.0251936064766088 w_k1_DIS_s007
                                 + 0.0316994406736637 w_k1_DIS_s010
                                 + 0.0430505589009102 w_k1_DIS_s011
                                 + 0.029353690856127 w_k1_DIS_s012
                                 + 0.0195238508815022 w_k1_DIS_s013
                                 + 0.0704444203520986 w_k1_DIS_s014
                                 + 0.0456327025987704 w_k1_DIS_s015
                                 + 0.0199562417832622 w_k1_DIS_s016
                                 + 0.0868651153813835 w_k1_DIS_s017
                                 + 0.0391787303481404 w_k1_DIS_s100
                                 + 0.0634334728383726 w_k1_DIS_s101
                                 + 0.0205137915849681 w_k1_DIS_s102
                                 + 0.0493694831091205 w_k1_DIS_s103
                                 + 0.036787724324021 w_k1_DIS_s104
                                 + 0.0162827355694011 w_k1_DIS_s105
                                 + 0.0338956487469676 w_k1_DIS_s106
                                 + 0.0257417722878408 w_k1_DIS_s107
                                 + 0.0671147955905491 w_k1_DIS_s110
                                 + 0.0590495290212419 w_k1_DIS_s111
                                 + 0.015429269074807 w_k1_DIS_s112
                                 + 0.0172759837572184 w_k1_DIS_s113
                                 + 0.044949088688433 w_k1_DIS_s114
                                 + 0.0143380308527969 w_k1_DIS_s115
                                 + 0.0244050775489932 w_k1_DIS_s116
                                 + 0.0391922780789468 w_k1_DIS_s117
                                 + 0.0109004872975383 w_k1_FDX_s000
                                 + 0.00564211394156242 w_k1_FDX_s001
                                 + 0.0103065722842963 w_k1_FDX_s002
                                 + 0.0218144405832929 w_k1_FDX_s003
                                 + 0.0155647773235141 w_k1_FDX_s004
                                 + 0.0103982674599341 w_k1_FDX_s005
                                 + 0.0101024310449968 w_k1_FDX_s006
                                 + 0.025205740315822 w_k1_FDX_s007
                                 + 0.0327443990665689 w_k1_FDX_s010
                                 + 0.0432295795713772 w_k1_FDX_s011
                                 + 0.0307672534485883 w_k1_FDX_s012
                                 + 0.0197314754202984 w_k1_FDX_s013
                                 + 0.0702807228977199 w_k1_FDX_s014
                                 + 0.0415416574516818 w_k1_FDX_s015
                                 + 0.0197992792550621 w_k1_FDX_s016
                                 + 0.0842753357109529 w_k1_FDX_s017
                                 + 0.0395846082822697 w_k1_FDX_s100
                                 + 0.0581954245553234 w_k1_FDX_s101
                                 + 0.0215752715917834 w_k1_FDX_s102
                                 + 0.0490798550630456 w_k1_FDX_s103
                                 + 0.034907336811784 w_k1_FDX_s104
                                 + 0.0147734567515611 w_k1_FDX_s105
                                 + 0.0342321525385912 w_k1_FDX_s106
                                 + 0.0263710441652757 w_k1_FDX_s107
                                 + 0.0620348417322425 w_k1_FDX_s110
                                 + 0.0636422367169358 w_k1_FDX_s111
                                 + 0.0154413323968899 w_k1_FDX_s112
                                 + 0.0176953812537399 w_k1_FDX_s113
                                 + 0.043324156548039 w_k1_FDX_s114
                                 + 0.0147343966891834 w_k1_FDX_s115
                                 + 0.0235931111785249 w_k1_FDX_s116
                                 + 0.0393264272387868 w_k1_FDX_s117
                                 + 0.0109357552666429 w_k1_IBM_s000
                                 + 0.00521463693178312 w_k1_IBM_s001
                                 + 0.0108339804089968 w_k1_IBM_s002
                                 + 0.0220035184113346 w_k1_IBM_s003
                                 + 0.015429272804709 w_k1_IBM_s004
                                 + 0.0107258919860166 w_k1_IBM_s005
                                 + 0.0105818420627824 w_k1_IBM_s006
                                 + 0.0233250521735177 w_k1_IBM_s007
                                 + 0.0319144228109888 w_k1_IBM_s010
                                 + 0.0442522820677248 w_k1_IBM_s011
                                 + 0.0324039548833721 w_k1_IBM_s012
                                 + 0.0204275256447322 w_k1_IBM_s013
                                 + 0.0729291203444519 w_k1_IBM_s014
                                 + 0.0428101109186652 w_k1_IBM_s015
                                 + 0.0197818898700127 w_k1_IBM_s016
                                 + 0.0883491209743929 w_k1_IBM_s017
                                 + 0.0400739255015489 w_k1_IBM_s100
                                 + 0.0614625672892167 w_k1_IBM_s101
                                 + 0.0219781250373342 w_k1_IBM_s102
                                 + 0.0508364089335142 w_k1_IBM_s103
                                 + 0.0353059963836765 w_k1_IBM_s104
                                 + 0.0152143813543646 w_k1_IBM_s105
                                 + 0.0357595694555616 w_k1_IBM_s106
                                 + 0.0275621327743505 w_k1_IBM_s107
                                 + 0.0636511598077247 w_k1_IBM_s110
                                 + 0.0632321223457829 w_k1_IBM_s111
                                 + 0.014402667863926 w_k1_IBM_s112
                                 + 0.0178516070471888 w_k1_IBM_s113
                                 + 0.0456563247309715 w_k1_IBM_s114
                                 + 0.0142904327562143 w_k1_IBM_s115
                                 + 0.0252399716198623 w_k1_IBM_s116
                                 + 0.040800156196033 w_k1_IBM_s117
                                 + 0.010718971464164 w_k1_INTC_s000
                                 + 0.00501333983654246 w_k1_INTC_s001
                                 + 0.0106326579696586 w_k1_INTC_s002
                                 + 0.0238059848857568 w_k1_INTC_s003
                                 + 0.0149846885117876 w_k1_INTC_s004
                                 + 0.0105256303907619 w_k1_INTC_s005
                                 + 0.0114537742108965 w_k1_INTC_s006
                                 + 0.0221162357095083 w_k1_INTC_s007
                                 + 0.0298593284631968 w_k1_INTC_s010
                                 + 0.0451358556374722 w_k1_INTC_s011
                                 + 0.0316325404253876 w_k1_INTC_s012
                                 + 0.0202974015368545 w_k1_INTC_s013
                                 + 0.0683768616072311 w_k1_INTC_s014
                                 + 0.0442870950033281 w_k1_INTC_s015
                                 + 0.0173800203127072 w_k1_INTC_s016
                                 + 0.0978861796137545 w_k1_INTC_s017
                                 + 0.0367695431618208 w_k1_INTC_s100
                                 + 0.0583217407020121 w_k1_INTC_s101
                                 + 0.020842794954415 w_k1_INTC_s102
                                 + 0.0429164945987283 w_k1_INTC_s103
                                 + 0.0290420368038346 w_k1_INTC_s104
                                 + 0.0134236890423536 w_k1_INTC_s105
                                 + 0.0378861435265955 w_k1_INTC_s106
                                 + 0.0267261027143308 w_k1_INTC_s107
                                 + 0.0650108368544425 w_k1_INTC_s110
                                 + 0.0633576523812622 w_k1_INTC_s111
                                 + 0.0151537214660804 w_k1_INTC_s112
                                 + 0.0186884951072169 w_k1_INTC_s113
                                 + 0.0446856736032536 w_k1_INTC_s114
                                 + 0.0139856191194661 w_k1_INTC_s115
                                 + 0.0264582992433653 w_k1_INTC_s116
                                 + 0.0368776632940545 w_k1_INTC_s117
                                 + 0.0112329779120171 w_k1_KO_s000
                                 + 0.00537615223103391 w_k1_KO_s001
                                 + 0.0111193434515891 w_k1_KO_s002
                                 + 0.0211509507522814 w_k1_KO_s003
                                 + 0.0146683697865963 w_k1_KO_s004
                                 + 0.00919201530995 w_k1_KO_s005
                                 + 0.00943160708375206 w_k1_KO_s006
                                 + 0.0269588626902763 w_k1_KO_s007
                                 + 0.0324634830581591 w_k1_KO_s010
                                 + 0.044042017838348 w_k1_KO_s011
                                 + 0.0287194171007412 w_k1_KO_s012
                                 + 0.0201451575083648 w_k1_KO_s013
                                 + 0.0631930772549777 w_k1_KO_s014
                                 + 0.0428057204236141 w_k1_KO_s015
                                 + 0.020183923834424 w_k1_KO_s016
                                 + 0.0888745048920432 w_k1_KO_s017
                                 + 0.0409037427077155 w_k1_KO_s100
                                 + 0.0618305585785302 w_k1_KO_s101
                                 + 0.0221568390307765 w_k1_KO_s102
                                 + 0.0485232206967378 w_k1_KO_s103
                                 + 0.0345893635608327 w_k1_KO_s104
                                 + 0.0143592217800794 w_k1_KO_s105
                                 + 0.0346240609751591 w_k1_KO_s106
                                 + 0.0255291467118406 w_k1_KO_s107
                                 + 0.0698587330154308 w_k1_KO_s110
                                 + 0.057047391127277 w_k1_KO_s111
                                 + 0.0151377762286193 w_k1_KO_s112
                                 + 0.0170998077936445 w_k1_KO_s113
                                 + 0.0390887894532916 w_k1_KO_s114
                                 + 0.0142788734030212 w_k1_KO_s115
                                 + 0.0227220175953961 w_k1_KO_s116
                                 + 0.0379805321239202 w_k1_KO_s117
                                 + 0.00964619053705349 w_k1_MAT_s000
                                 + 0.00453739886495748 w_k1_MAT_s001
                                 + 0.011255196857277 w_k1_MAT_s002
                                 + 0.021053750231856 w_k1_MAT_s003
                                 + 0.0152288045866265 w_k1_MAT_s004
                                 + 0.00889594699269406 w_k1_MAT_s005
                                 + 0.0111147340244675 w_k1_MAT_s006
                                 + 0.0280374686456142 w_k1_MAT_s007
                                 + 0.0283995923697784 w_k1_MAT_s010
                                 + 0.0401202036081332 w_k1_MAT_s011
                                 + 0.0373327854009304 w_k1_MAT_s012
                                 + 0.0201272940222808 w_k1_MAT_s013
                                 + 0.083376108098952 w_k1_MAT_s014
                                 + 0.0372493537337438 w_k1_MAT_s015
                                 + 0.0193116831680646 w_k1_MAT_s016
                                 + 0.086427659810882 w_k1_MAT_s017
                                 + 0.0392614826818205 w_k1_MAT_s100
                                 + 0.0577514568909794 w_k1_MAT_s101
                                 + 0.0195556635053082 w_k1_MAT_s102
                                 + 0.0517105014893482 w_k1_MAT_s103
                                 + 0.0394827564069972 w_k1_MAT_s104
                                 + 0.0166570689599813 w_k1_MAT_s105
                                 + 0.0358555490365099 w_k1_MAT_s106
                                 + 0.0252629000093537 w_k1_MAT_s107
                                 + 0.0592916878324884 w_k1_MAT_s110
                                 + 0.0580742550319068 w_k1_MAT_s111
                                 + 0.0169664649048659 w_k1_MAT_s112
                                 + 0.0160228462120782 w_k1_MAT_s113
                                 + 0.0378727118769981 w_k1_MAT_s114
                                 + 0.0155960270577362 w_k1_MAT_s115
                                 + 0.0216440240463475 w_k1_MAT_s116
                                 + 0.0482745883939942 w_k1_MAT_s117
                                 + 0.0123390971875427 w_k1_MSFT_s000
                                 + 0.00515199890357259 w_k1_MSFT_s001
                                 + 0.0102344214504096 w_k1_MSFT_s002
                                 + 0.025094401599691 w_k1_MSFT_s003
                                 + 0.0144320568221945 w_k1_MSFT_s004
                                 + 0.00935285900296824 w_k1_MSFT_s005
                                 + 0.0107602437605323 w_k1_MSFT_s006
                                 + 0.0264958784948309 w_k1_MSFT_s007
                                 + 0.0314635659030371 w_k1_MSFT_s010
                                 + 0.0392790204196848 w_k1_MSFT_s011
                                 + 0.0295186883717614 w_k1_MSFT_s012
                                 + 0.0194633720207138 w_k1_MSFT_s013
                                 + 0.0708342683736576 w_k1_MSFT_s014
                                 + 0.0438257854344492 w_k1_MSFT_s015
                                 + 0.019343117218712 w_k1_MSFT_s016
                                 + 0.0878893982410529 w_k1_MSFT_s017
                                 + 0.0381385242999906 w_k1_MSFT_s100
                                 + 0.057550281438585 w_k1_MSFT_s101
                                 + 0.0209566455062969 w_k1_MSFT_s102
                                 + 0.0502070190256168 w_k1_MSFT_s103
                                 + 0.0323511531234686 w_k1_MSFT_s104
                                 + 0.0136602460590486 w_k1_MSFT_s105
                                 + 0.0372592718608682 w_k1_MSFT_s106
                                 + 0.025410918193437 w_k1_MSFT_s107
                                 + 0.0603650661787634 w_k1_MSFT_s110
                                 + 0.0610079139807256 w_k1_MSFT_s111
                                 + 0.0154119365524726 w_k1_MSFT_s112
                                 + 0.0175147019249559 w_k1_MSFT_s113
                                 + 0.0432465057218877 w_k1_MSFT_s114
                                 + 0.0149296665163811 w_k1_MSFT_s115
                                 + 0.0242772930373398 w_k1_MSFT_s116
                                 + 0.0409844206841391 w_k1_MSFT_s117
                                 + 0.0120681153734994 w_k1_PG_s000
                                 + 0.00547565468717157 w_k1_PG_s001
                                 + 0.0102297471943625 w_k1_PG_s002
                                 + 0.0218358669288927 w_k1_PG_s003
                                 + 0.0150131956455947 w_k1_PG_s004
                                 + 0.00960411520107751 w_k1_PG_s005
                                 + 0.00974549455919188 w_k1_PG_s006
                                 + 0.0235372018148686 w_k1_PG_s007
                                 + 0.0315206630893046 w_k1_PG_s010
                                 + 0.0412736444383096 w_k1_PG_s011
                                 + 0.0288117785726811 w_k1_PG_s012
                                 + 0.0209008102453518 w_k1_PG_s013
                                 + 0.0716721260458869 w_k1_PG_s014
                                 + 0.0412105402616274 w_k1_PG_s015
                                 + 0.0184279197818995 w_k1_PG_s016
                                 + 0.0830384784836443 w_k1_PG_s017
                                 + 0.0420888051988549 w_k1_PG_s100
                                 + 0.0553624331497533 w_k1_PG_s101
                                 + 0.0228651708041502 w_k1_PG_s102
                                 + 0.0516489405390329 w_k1_PG_s103
                                 + 0.0350429928444429 w_k1_PG_s104
                                 + 0.0148491359321335 w_k1_PG_s105
                                 + 0.0330120028368696 w_k1_PG_s106
                                 + 0.025030023038603 w_k1_PG_s107
                                 + 0.063517375405026 w_k1_PG_s110
                                 + 0.0659447606567999 w_k1_PG_s111
                                 + 0.0157214725241372 w_k1_PG_s112
                                 + 0.0179967478815385 w_k1_PG_s113
                                 + 0.0458666760600228 w_k1_PG_s114
                                 + 0.0139694530107341 w_k1_PG_s115
                                 + 0.0230902701289771 w_k1_PG_s116
                                 + 0.0401997025358431 w_k1_PG_s117 >= 1.1
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

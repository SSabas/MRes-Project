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
 w_k1_ADM_s0_return_constraint:  1.18254 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.878445 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.811558 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.16487 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.946185 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.19863 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.991427 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.0372 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.0018 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.858693 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.83799 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.3528 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.32178 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.13015 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.989225 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.75746 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.89891 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.917119 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.20248 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.918499 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.941591 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.16067 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.29765 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.943942 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.972631 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.18875 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.800293 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.830953 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.30545 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.838009 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.17774 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.03283 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.07384 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.27792 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.36224 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.20963 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.966605 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.944724 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.3149 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.910338 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.813139 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.17904 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.92342 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.39942 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.887273 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.02632 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.996649 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.655751 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.715458 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.38432 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.18904 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.925732 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.983738 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.751198 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.920612 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.27687 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.41899 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.746908 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.994802 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.11533 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.4124 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.988085 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.36375 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.04969 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.788237 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.56181 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.21179 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.755563 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.972722 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.792627 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.840647 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.19837 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.27082 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.38544 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.620835 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.82327 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.37001 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.905845 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.773373 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.25282 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.824783 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.40952 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.757522 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.993825 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.06272 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.574232 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.727902 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.90239 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.32306 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.993182 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.806334 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.862523 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.859479 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.39836 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.63024 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.695278 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.32795 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.01404 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.67047 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.05303 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.46728 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.13093 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.740617 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.615493 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.51085 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.761283 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.17941 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.89552 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.0916 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.35119 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.5689 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.18899 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.715224 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.858138 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.23339 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.937089 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.833054 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.11731 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.877042 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.22958 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.888857 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.94941 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.03515 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.740701 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.728726 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.50321 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.19004 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.13661 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.772542 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.821947 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.875264 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.12732 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.42125 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.844227 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.1939 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.13724 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.38129 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.963221 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.14378 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.12725 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.740956 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.798409 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.40481 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.864013 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.01367 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.959027 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.15742 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.08227 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.37776 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.22179 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.776739 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.99165 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.24396 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.96335 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.809224 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.17123 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.932756 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.20009 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.803651 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.35729 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.969161 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.621758 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.13048 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.32992 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.38627 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.655634 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.54082 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.871651 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.15852 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.84658 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.0434 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.659423 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.957401 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.719181 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.42146 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.30764 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 2.02749 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.914432 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.31529 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.495285 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.829877 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.759645 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.13841 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.97514 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.652698 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.71912 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.870497 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.11871 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.878815 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.591464 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.11494 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.94787 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.928006 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.14528 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.929389 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.21858 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.800371 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.04089 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.14335 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.624653 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.828398 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.43101 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.917098 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.01144 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.793427 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.887103 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.908279 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.12623 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.19616 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.798027 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.13687 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.10807 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.14913 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.957714 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.06673 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.09413 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.763843 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.837203 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.24959 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.884159 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.922264 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.969952 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.1191 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.951659 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.44283 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.10641 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.721387 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.966283 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.21106 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.947684 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.846958 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.0875 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.893938 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.21721 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.717496 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.16772 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.07583 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.619377 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.94829 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.52949 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.27353 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.827033 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.07189 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.01696 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.08961 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.5274 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.47675 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.754436 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.33186 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.902958 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.43443 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.15053 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.69179 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.03635 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.953913 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.668121 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.16862 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.791194 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.10219 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.975096 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.926507 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.37614 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.16611 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.13962 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.884217 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.80591 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.31888 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.952471 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.895144 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.27067 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.998439 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.21077 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.826011 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.14202 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.743308 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.488173 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.754423 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.04063 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.33244 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.572999 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.43578 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.822064 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.15361 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.63628 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.26767 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.512697 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.931398 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.629144 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.68046 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.24199 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 2.03469 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.727207 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.28142 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.365808 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.00484 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.655329 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.11117 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.903463 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.643112 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.79194 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.761181 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.63657 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.963937 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.558893 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.15911 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.982754 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.885679 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.1776 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.862941 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.14282 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.80328 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.968999 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.06101 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.692117 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.758612 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.46175 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.09515 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.990071 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.755102 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.891633 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.821138 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.1339 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.34341 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.834529 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.11491 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.07247 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.23803 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.974591 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.07195 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.04369 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.738987 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.85466 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.44317 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.922005 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.15365 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.02469 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.18731 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.972349 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.39392 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.17016 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.864968 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.930699 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.11486 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.934981 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.891185 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.11372 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.942791 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.18702 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.893438 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.11514 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.988537 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.823125 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.886467 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.29154 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.14274 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.911217 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.93423 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.904703 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.98814 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.1686 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.3023 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.897648 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.06599 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.01064 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.22263 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.11058 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.19453 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.05147 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.841387 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.759295 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.21073 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.887532 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.10158 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.985092 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.02682 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.09519 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.13571 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.08079 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.899771 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.87254 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.096999634975727 z_k1_s0000
                                 - 0.117999696589013 z_k1_s0001
                                 - 0.0679999772716172 z_k1_s0002
                                 - 0.0649999582125775 z_k1_s0003
                                 - 0.11899987047534 z_k1_s0004
                                 - 0.0620000109535297 z_k1_s0005
                                 - 0.155999842270193 z_k1_s0006
                                 - 0.0329999942494607 z_k1_s0007
                                 - 0.180999702441518 z_k1_s0010
                                 - 0.300999635279403 z_k1_s0011
                                 - 0.254998784294148 z_k1_s0012
                                 - 0.433999320565814 z_k1_s0013
                                 - 0.118999799503545 z_k1_s0014
                                 - 0.350999699003424 z_k1_s0015
                                 - 0.383999256841794 z_k1_s0016
                                 - 0.175999916168882 z_k1_s0017
                                 - 0.179999920468044 z_k1_s0020
                                 - 0.154999960389132 z_k1_s0021
                                 - 0.117000021069185 z_k1_s0022
                                 - 0.317999492203764 z_k1_s0023
                                 - 0.234999624741776 z_k1_s0024
                                 - 0.139999984341785 z_k1_s0025
                                 - 0.53599914409188 z_k1_s0026
                                 - 0.397999364456284 z_k1_s0027
                                 - 0.0509999441178277 z_k1_s0030
                                 - 0.0929998043914594 z_k1_s0031
                                 - 0.0509999441178277 z_k1_s0032
                                 - 0.0919997557421574 z_k1_s0033
                                 - 0.074000060053128 z_k1_s0034
                                 - 0.0579999306634197 z_k1_s0035
                                 - 0.122000035222811 z_k1_s0036
                                 - 0.048999905819144 z_k1_s0037
                                 - 0.149000314850753 z_k1_s0040
                                 - 0.0739999582896611 z_k1_s0041
                                 - 0.189000038710062 z_k1_s0042
                                 - 0.0759999871326211 z_k1_s0043
                                 - 0.0380000362163051 z_k1_s0044
                                 - 0.0819999883615119 z_k1_s0045
                                 - 0.167000318537425 z_k1_s0046
                                 - 0.078000015975581 z_k1_s0047
                                 - 0.162000119768921 z_k1_s0050
                                 - 0.058000001788877 z_k1_s0051
                                 - 0.0470000176909849 z_k1_s0052
                                 - 0.0380000249927101 z_k1_s0053
                                 - 0.145000161472177 z_k1_s0054
                                 - 0.0599999931884943 z_k1_s0055
                                 - 0.0519999647900313 z_k1_s0056
                                 - 0.0660002813873148 z_k1_s0057
                                 - 0.0959998375359089 z_k1_s0060
                                 - 0.15699902365504 z_k1_s0061
                                 - 0.242000145553914 z_k1_s0062
                                 - 0.172999045377587 z_k1_s0063
                                 - 0.203999178964826 z_k1_s0064
                                 - 0.166999586230456 z_k1_s0065
                                 - 0.234999312552064 z_k1_s0066
                                 - 0.189998976957989 z_k1_s0067
                                 - 0.193999990616937 z_k1_s0070
                                 - 0.223999450619071 z_k1_s0071
                                 - 0.165999861414375 z_k1_s0072
                                 - 0.094000049308255 z_k1_s0073
                                 - 0.223999450619071 z_k1_s0074
                                 - 0.227999695219641 z_k1_s0075
                                 - 0.130000192811529 z_k1_s0076
                                 - 0.323000359728537 z_k1_s0077
                                 - 0.12899982244031 z_k1_s0100
                                 - 0.131000359547671 z_k1_s0101
                                 - 0.142999766193539 z_k1_s0102
                                 - 0.105999846702894 z_k1_s0103
                                 - 0.0709999873198227 z_k1_s0104
                                 - 0.114000087133188 z_k1_s0105
                                 - 0.134999525763244 z_k1_s0106
                                 - 0.124999702225163 z_k1_s0107
                                 - 0.103999937593638 z_k1_s0110
                                 - 0.195000048488074 z_k1_s0111
                                 - 0.218000577017449 z_k1_s0112
                                 - 0.0519999687968188 z_k1_s0113
                                 - 0.233000173362676 z_k1_s0114
                                 - 0.150999555475369 z_k1_s0115
                                 - 0.129999988192048 z_k1_s0116
                                 - 0.240999781546793 z_k1_s0117
                                 - 0.262999571704298 z_k1_s0120
                                 - 0.397998665880228 z_k1_s0121
                                 - 0.24399982526459 z_k1_s0122
                                 - 0.457000135895897 z_k1_s0123
                                 - 0.407999607373682 z_k1_s0124
                                 - 0.778000402667233 z_k1_s0125
                                 - 0.258999875906201 z_k1_s0126
                                 - 0.596999700818555 z_k1_s0127
                                 - 0.250000070536646 z_k1_s0130
                                 - 0.0769999838292628 z_k1_s0131
                                 - 0.142000033748811 z_k1_s0132
                                 - 0.279999510560683 z_k1_s0133
                                 - 0.0890000756390779 z_k1_s0134
                                 - 0.0939999823097507 z_k1_s0135
                                 - 0.365999800097457 z_k1_s0136
                                 - 0.281000597195518 z_k1_s0137
                                 - 0.20299978411924 z_k1_s0140
                                 - 0.117999808368877 z_k1_s0141
                                 - 0.288999246191822 z_k1_s0142
                                 - 0.254000440928868 z_k1_s0143
                                 - 0.369000111932559 z_k1_s0144
                                 - 0.205999921484425 z_k1_s0145
                                 - 0.521998725564391 z_k1_s0146
                                 - 0.136999909082394 z_k1_s0147
                                 - 0.697000330027538 z_k1_s0150
                                 - 0.458001901111576 z_k1_s0151
                                 - 0.398001080921237 z_k1_s0152
                                 - 0.300000591945247 z_k1_s0153
                                 - 0.421001023160184 z_k1_s0154
                                 - 0.323999325018453 z_k1_s0155
                                 - 0.225000364208789 z_k1_s0156
                                 - 0.367000603989465 z_k1_s0157
                                 - 0.108999978550475 z_k1_s0160
                                 - 0.167999785347588 z_k1_s0161
                                 - 0.101999941161009 z_k1_s0162
                                 - 0.181000172499334 z_k1_s0163
                                 - 0.0740000140030639 z_k1_s0164
                                 - 0.241000334123384 z_k1_s0165
                                 - 0.177999997618194 z_k1_s0166
                                 - 0.0590000291970307 z_k1_s0167
                                 - 0.189000104391344 z_k1_s0170
                                 - 0.475001247027065 z_k1_s0171
                                 - 0.168000177192409 z_k1_s0172
                                 - 0.474001105807892 z_k1_s0173
                                 - 0.353999351525615 z_k1_s0174
                                 - 0.260001268941845 z_k1_s0175
                                 - 0.318999472860723 z_k1_s0176
                                 - 0.293000865168391 z_k1_s0177
                                 - 1.05799957231877 z_k1_s1000
                                 - 0.452002387753302 z_k1_s1001
                                 - 0.862002331553699 z_k1_s1002
                                 - 0.226000421275824 z_k1_s1003
                                 - 0.276000442688098 z_k1_s1004
                                 - 0.357000299677791 z_k1_s1005
                                 - 0.214000385232846 z_k1_s1006
                                 - 0.418000225362658 z_k1_s1007
                                 - 1.10700456200567 z_k1_s1010
                                 - 0.548000789934422 z_k1_s1011
                                 - 1.03899833332408 z_k1_s1012
                                 - 0.57600104403505 z_k1_s1013
                                 - 0.369000568400424 z_k1_s1014
                                 - 1.39199936869652 z_k1_s1015
                                 - 1.21899843639719 z_k1_s1016
                                 - 0.892001463052653 z_k1_s1017
                                 - 0.560000093035623 z_k1_s1020
                                 - 0.18500011664934 z_k1_s1021
                                 - 0.206000120138176 z_k1_s1022
                                 - 0.242000254990574 z_k1_s1023
                                 - 0.400000496025805 z_k1_s1024
                                 - 0.554999447847265 z_k1_s1025
                                 - 0.554001123011105 z_k1_s1026
                                 - 0.305000265457082 z_k1_s1027
                                 - 1.03400080918764 z_k1_s1030
                                 - 1.64199543448653 z_k1_s1031
                                 - 0.759000159794904 z_k1_s1032
                                 - 2.09499956726667 z_k1_s1033
                                 - 1.89799639355126 z_k1_s1034
                                 - 1.19999953567528 z_k1_s1035
                                 - 1.60400512343627 z_k1_s1036
                                 - 1.10500108403528 z_k1_s1037
                                 - 0.277000272514732 z_k1_s1040
                                 - 0.21900011534439 z_k1_s1041
                                 - 0.256000394179112 z_k1_s1042
                                 - 0.333000504910321 z_k1_s1043
                                 - 0.439001195377246 z_k1_s1044
                                 - 0.199000199396147 z_k1_s1045
                                 - 0.838999514342111 z_k1_s1046
                                 - 0.779001771099561 z_k1_s1047
                                 - 0.557001512962828 z_k1_s1050
                                 - 0.239000188012268 z_k1_s1051
                                 - 0.451001071312641 z_k1_s1052
                                 - 0.278000363430747 z_k1_s1053
                                 - 0.489001488130917 z_k1_s1054
                                 - 0.465999162817574 z_k1_s1055
                                 - 0.7640018907603 z_k1_s1056
                                 - 0.181000146685028 z_k1_s1057
                                 - 0.942001555193668 z_k1_s1060
                                 - 0.351000358163154 z_k1_s1061
                                 - 0.564002154218373 z_k1_s1062
                                 - 0.330000239042241 z_k1_s1063
                                 - 0.323000046935206 z_k1_s1064
                                 - 0.50200123647068 z_k1_s1065
                                 - 1.02800091090741 z_k1_s1066
                                 - 0.531998140927771 z_k1_s1067
                                 - 1.36599647438373 z_k1_s1070
                                 - 0.90600047994975 z_k1_s1071
                                 - 2.72800522507577 z_k1_s1072
                                 - 1.09099983084595 z_k1_s1073
                                 - 1.64100001312097 z_k1_s1074
                                 - 0.614000274526655 z_k1_s1075
                                 - 0.945999532057098 z_k1_s1076
                                 - 2.20000532589973 z_k1_s1077
                                 - 0.331997742074023 z_k1_s1100
                                 - 0.584998601887922 z_k1_s1101
                                 - 0.580997567625084 z_k1_s1102
                                 - 0.161999659568826 z_k1_s1103
                                 - 0.420998728986786 z_k1_s1104
                                 - 0.160999552902671 z_k1_s1105
                                 - 0.163999569102027 z_k1_s1106
                                 - 0.631999362009692 z_k1_s1107
                                 - 0.480000600003967 z_k1_s1110
                                 - 0.204999780952098 z_k1_s1111
                                 - 0.198999831651999 z_k1_s1112
                                 - 0.283999462603106 z_k1_s1113
                                 - 0.292000791802051 z_k1_s1114
                                 - 0.433999359104593 z_k1_s1115
                                 - 0.156999953751504 z_k1_s1116
                                 - 0.276999172553287 z_k1_s1117
                                 - 0.078999928026495 z_k1_s1120
                                 - 0.27700047609768 z_k1_s1121
                                 - 0.342999754388427 z_k1_s1122
                                 - 0.0759999867950873 z_k1_s1123
                                 - 0.16000019767378 z_k1_s1124
                                 - 0.207999542976192 z_k1_s1125
                                 - 0.113999908792659 z_k1_s1126
                                 - 0.171000077388905 z_k1_s1127
                                 - 0.335998955946243 z_k1_s1130
                                 - 0.494999809953159 z_k1_s1131
                                 - 0.442998533593634 z_k1_s1132
                                 - 0.609999003257308 z_k1_s1133
                                 - 0.486997677902673 z_k1_s1134
                                 - 0.608996639756226 z_k1_s1135
                                 - 0.391999201336238 z_k1_s1136
                                 - 0.82199678674156 z_k1_s1137
                                 - 0.606999436311015 z_k1_s1140
                                 - 0.243000030936679 z_k1_s1141
                                 - 0.148000201975528 z_k1_s1142
                                 - 0.401998952378581 z_k1_s1143
                                 - 0.268000217252901 z_k1_s1144
                                 - 0.450999965850739 z_k1_s1145
                                 - 0.198000025207664 z_k1_s1146
                                 - 0.429999139076737 z_k1_s1147
                                 - 0.118999776298224 z_k1_s1150
                                 - 0.300000421809294 z_k1_s1151
                                 - 0.320000038330479 z_k1_s1152
                                 - 0.0799999066827611 z_k1_s1153
                                 - 0.327999473339518 z_k1_s1154
                                 - 0.282999307168266 z_k1_s1155
                                 - 0.342000028103217 z_k1_s1156
                                 - 0.285999867045595 z_k1_s1157
                                 - 0.333000461441597 z_k1_s1160
                                 - 1.17500294665122 z_k1_s1161
                                 - 0.368000377288467 z_k1_s1162
                                 - 0.4700006399599 z_k1_s1163
                                 - 0.702002283121552 z_k1_s1164
                                 - 0.417000582694438 z_k1_s1165
                                 - 0.770999054322044 z_k1_s1166
                                 - 1.15100383549427 z_k1_s1167
                                 - 0.536997043654134 z_k1_s1170
                                 - 0.990997609711528 z_k1_s1171
                                 - 0.262999068394787 z_k1_s1172
                                 - 0.200999283071319 z_k1_s1173
                                 - 0.331998906338358 z_k1_s1174
                                 - 0.861995464174008 z_k1_s1175
                                 - 0.827998055322856 z_k1_s1176
                                 - 0.31799880109815 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.775385 w_k1_ADM_s000
                                 + 0.684632 w_k1_AXP_s000
                                 + 0.613618 w_k1_DIS_s000
                                 + 0.765788 w_k1_FDX_s000
                                 + 0.799958 w_k1_IBM_s000
                                 + 1.11805 w_k1_INTC_s000
                                 + 0.953553 w_k1_KO_s000
                                 + 0.483933 w_k1_MAT_s000
                                 + 0.954522 w_k1_MSFT_s000
                                 + 0.915875 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.18755 w_k1_ADM_s000
                                 + 0.95457 w_k1_AXP_s000
                                 + 0.983414 w_k1_DIS_s000
                                 + 1.07463 w_k1_FDX_s000
                                 + 0.838519 w_k1_IBM_s000
                                 + 1.17725 w_k1_INTC_s000
                                 + 0.996567 w_k1_KO_s000
                                 + 0.539342 w_k1_MAT_s000
                                 + 1.15535 w_k1_MSFT_s000 + 1.0492 w_k1_PG_s000
                                 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.21983 w_k1_ADM_s000
                                 + 1.6077 w_k1_AXP_s000 + 1.38453 w_k1_DIS_s000
                                 + 1.46616 w_k1_FDX_s000
                                 + 1.15455 w_k1_IBM_s000
                                 + 1.27867 w_k1_INTC_s000
                                 + 1.29299 w_k1_KO_s000 + 1.1673 w_k1_MAT_s000
                                 + 1.3621 w_k1_MSFT_s000 + 1.33283 w_k1_PG_s000
                                 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.1712 w_k1_ADM_s000
                                 + 1.13506 w_k1_AXP_s000
                                 + 1.58685 w_k1_DIS_s000
                                 + 1.20243 w_k1_FDX_s000
                                 + 1.45279 w_k1_IBM_s000
                                 + 1.19717 w_k1_INTC_s000
                                 + 1.63918 w_k1_KO_s000 + 1.00351 w_k1_MAT_s000
                                 + 1.32022 w_k1_MSFT_s000
                                 + 1.20209 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.00119 w_k1_ADM_s000
                                 + 0.777024 w_k1_AXP_s000
                                 + 0.825294 w_k1_DIS_s000
                                 + 0.874405 w_k1_FDX_s000
                                 + 0.817121 w_k1_IBM_s000
                                 + 1.07659 w_k1_INTC_s000
                                 + 0.965446 w_k1_KO_s000
                                 + 0.570391 w_k1_MAT_s000
                                 + 0.993623 w_k1_MSFT_s000
                                 + 0.912845 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.16411 w_k1_ADM_s000
                                 + 1.40116 w_k1_AXP_s000
                                 + 0.943521 w_k1_DIS_s000
                                 + 1.03324 w_k1_FDX_s000
                                 + 1.10764 w_k1_IBM_s000 + 1.152 w_k1_INTC_s000
                                 + 1.1137 w_k1_KO_s000 + 1.05262 w_k1_MAT_s000
                                 + 1.06399 w_k1_MSFT_s000 + 1.1262 w_k1_PG_s000
                                 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.21576 w_k1_ADM_s000
                                 + 0.865456 w_k1_AXP_s000
                                 + 0.835646 w_k1_DIS_s000
                                 + 0.910636 w_k1_FDX_s000
                                 + 1.25215 w_k1_IBM_s000
                                 + 1.0801 w_k1_INTC_s000 + 1.24953 w_k1_KO_s000
                                 + 0.885913 w_k1_MAT_s000
                                 + 1.06577 w_k1_MSFT_s000
                                 + 0.962726 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.29702 w_k1_ADM_s000
                                 + 1.35903 w_k1_AXP_s000
                                 + 1.39345 w_k1_DIS_s000
                                 + 1.33302 w_k1_FDX_s000
                                 + 0.862083 w_k1_IBM_s000
                                 + 1.12399 w_k1_INTC_s000
                                 + 1.24915 w_k1_KO_s000 + 1.03365 w_k1_MAT_s000
                                 + 1.4811 w_k1_MSFT_s000 + 1.23514 w_k1_PG_s000
                                 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.31828 w_k1_ADM_s001
                                 + 1.21282 w_k1_AXP_s001
                                 + 1.28519 w_k1_DIS_s001
                                 + 1.11683 w_k1_FDX_s001
                                 + 1.56304 w_k1_IBM_s001
                                 + 0.952628 w_k1_INTC_s001
                                 + 1.30125 w_k1_KO_s001 + 1.73088 w_k1_MAT_s001
                                 + 1.1043 w_k1_MSFT_s001
                                 + 0.989421 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.897219 w_k1_ADM_s001
                                 + 0.863626 w_k1_AXP_s001
                                 + 0.786897 w_k1_DIS_s001
                                 + 0.887613 w_k1_FDX_s001
                                 + 0.687871 w_k1_IBM_s001
                                 + 0.823926 w_k1_INTC_s001
                                 + 0.796846 w_k1_KO_s001
                                 + 0.799816 w_k1_MAT_s001
                                 + 0.812038 w_k1_MSFT_s001
                                 + 0.810041 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.16829 w_k1_ADM_s001
                                 + 1.41042 w_k1_AXP_s001 + 1.4676 w_k1_DIS_s001
                                 + 1.31198 w_k1_FDX_s001
                                 + 0.993788 w_k1_IBM_s001
                                 + 1.05284 w_k1_INTC_s001
                                 + 1.04132 w_k1_KO_s001 + 1.19325 w_k1_MAT_s001
                                 + 1.14045 w_k1_MSFT_s001
                                 + 1.04181 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.837921 w_k1_ADM_s001
                                 + 0.834567 w_k1_AXP_s001
                                 + 1.18699 w_k1_DIS_s001
                                 + 1.13793 w_k1_FDX_s001
                                 + 0.753335 w_k1_IBM_s001
                                 + 0.981376 w_k1_INTC_s001
                                 + 0.986023 w_k1_KO_s001
                                 + 0.791962 w_k1_MAT_s001
                                 + 1.19892 w_k1_MSFT_s001
                                 + 0.93203 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.824032 w_k1_ADM_s001
                                 + 0.991113 w_k1_AXP_s001
                                 + 0.799887 w_k1_DIS_s001
                                 + 0.754034 w_k1_FDX_s001 + 1.638 w_k1_IBM_s001
                                 + 0.893111 w_k1_INTC_s001 + 1.057 w_k1_KO_s001
                                 + 1.52398 w_k1_MAT_s001
                                 + 0.880136 w_k1_MSFT_s001
                                 + 0.918924 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.13863 w_k1_ADM_s001
                                 + 1.1578 w_k1_AXP_s001 + 1.23602 w_k1_DIS_s001
                                 + 1.01444 w_k1_FDX_s001
                                 + 1.06453 w_k1_IBM_s001
                                 + 1.02053 w_k1_INTC_s001
                                 + 1.03413 w_k1_KO_s001 + 1.27778 w_k1_MAT_s001
                                 + 1.01533 w_k1_MSFT_s001
                                 + 0.965725 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.04488 w_k1_ADM_s001
                                 + 0.793094 w_k1_AXP_s001
                                 + 0.741755 w_k1_DIS_s001
                                 + 0.77047 w_k1_FDX_s001
                                 + 1.10016 w_k1_IBM_s001
                                 + 0.9271 w_k1_INTC_s001
                                 + 0.928832 w_k1_KO_s001
                                 + 1.19828 w_k1_MAT_s001
                                 + 0.846077 w_k1_MSFT_s001
                                 + 0.878272 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.3382 w_k1_ADM_s001
                                 + 1.27415 w_k1_AXP_s001
                                 + 1.53658 w_k1_DIS_s001
                                 + 1.35778 w_k1_FDX_s001
                                 + 1.23459 w_k1_IBM_s001
                                 + 1.07142 w_k1_INTC_s001
                                 + 1.32927 w_k1_KO_s001 + 1.26026 w_k1_MAT_s001
                                 + 1.27789 w_k1_MSFT_s001
                                 + 1.11488 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.807589 w_k1_ADM_s002
                                 + 0.778227 w_k1_AXP_s002
                                 + 0.694125 w_k1_DIS_s002
                                 + 0.745123 w_k1_FDX_s002
                                 + 0.823721 w_k1_IBM_s002
                                 + 0.785467 w_k1_INTC_s002
                                 + 0.789906 w_k1_KO_s002
                                 + 0.925382 w_k1_MAT_s002
                                 + 0.794918 w_k1_MSFT_s002
                                 + 0.94046 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.717814 w_k1_ADM_s002
                                 + 0.515152 w_k1_AXP_s002
                                 + 0.696203 w_k1_DIS_s002
                                 + 0.747872 w_k1_FDX_s002
                                 + 0.675288 w_k1_IBM_s002
                                 + 0.626387 w_k1_INTC_s002
                                 + 0.686216 w_k1_KO_s002
                                 + 0.538599 w_k1_MAT_s002
                                 + 0.822128 w_k1_MSFT_s002
                                 + 0.750661 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.28131 w_k1_ADM_s002
                                 + 1.69189 w_k1_AXP_s002
                                 + 1.22582 w_k1_DIS_s002
                                 + 1.23143 w_k1_FDX_s002
                                 + 1.13906 w_k1_IBM_s002
                                 + 1.01425 w_k1_INTC_s002
                                 + 1.02134 w_k1_KO_s002 + 1.99149 w_k1_MAT_s002
                                 + 1.04112 w_k1_MSFT_s002 + 1.1052 w_k1_PG_s002
                                 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.998031 w_k1_ADM_s002
                                 + 0.865461 w_k1_AXP_s002
                                 + 0.73074 w_k1_DIS_s002
                                 + 0.788664 w_k1_FDX_s002
                                 + 1.11711 w_k1_IBM_s002
                                 + 0.889192 w_k1_INTC_s002
                                 + 0.847827 w_k1_KO_s002
                                 + 1.31658 w_k1_MAT_s002
                                 + 0.872745 w_k1_MSFT_s002
                                 + 0.905999 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.23547 w_k1_ADM_s002
                                 + 1.04032 w_k1_AXP_s002
                                 + 1.02275 w_k1_DIS_s002
                                 + 1.14358 w_k1_FDX_s002
                                 + 0.796742 w_k1_IBM_s002
                                 + 0.932124 w_k1_INTC_s002
                                 + 0.889985 w_k1_KO_s002
                                 + 0.876092 w_k1_MAT_s002
                                 + 0.996568 w_k1_MSFT_s002
                                 + 1.00858 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.08021 w_k1_ADM_s002
                                 + 0.961317 w_k1_AXP_s002
                                 + 0.641456 w_k1_DIS_s002
                                 + 0.887778 w_k1_FDX_s002
                                 + 0.70366 w_k1_IBM_s002
                                 + 0.749811 w_k1_INTC_s002
                                 + 0.68484 w_k1_KO_s002
                                 + 0.961542 w_k1_MAT_s002
                                 + 0.795935 w_k1_MSFT_s002
                                 + 0.895469 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.01894 w_k1_ADM_s002
                                 + 0.906796 w_k1_AXP_s002
                                 + 1.08517 w_k1_DIS_s002
                                 + 0.975354 w_k1_FDX_s002
                                 + 0.952576 w_k1_IBM_s002
                                 + 1.05437 w_k1_INTC_s002
                                 + 0.995961 w_k1_KO_s002
                                 + 1.04692 w_k1_MAT_s002
                                 + 0.966523 w_k1_MSFT_s002
                                 + 1.04016 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.33013 w_k1_ADM_s002
                                 + 1.29093 w_k1_AXP_s002
                                 + 1.44441 w_k1_DIS_s002
                                 + 1.22743 w_k1_FDX_s002
                                 + 1.28791 w_k1_IBM_s002
                                 + 1.17068 w_k1_INTC_s002
                                 + 1.19838 w_k1_KO_s002 + 1.59971 w_k1_MAT_s002
                                 + 1.28516 w_k1_MSFT_s002
                                 + 1.19842 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.11775 w_k1_ADM_s003
                                 + 1.0868 w_k1_AXP_s003 + 1.08851 w_k1_DIS_s003
                                 + 1.18277 w_k1_FDX_s003
                                 + 0.852178 w_k1_IBM_s003
                                 + 1.22014 w_k1_INTC_s003
                                 + 0.867178 w_k1_KO_s003
                                 + 0.920657 w_k1_MAT_s003
                                 + 1.05748 w_k1_MSFT_s003
                                 + 1.01303 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.21644 w_k1_ADM_s003
                                 + 1.12043 w_k1_AXP_s003
                                 + 1.22677 w_k1_DIS_s003 + 1.1214 w_k1_FDX_s003
                                 + 1.06644 w_k1_IBM_s003
                                 + 1.30278 w_k1_INTC_s003
                                 + 1.05448 w_k1_KO_s003 + 1.05585 w_k1_MAT_s003
                                 + 1.15081 w_k1_MSFT_s003
                                 + 1.04071 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.718817 w_k1_ADM_s003
                                 + 0.950168 w_k1_AXP_s003
                                 + 0.756546 w_k1_DIS_s003
                                 + 0.783689 w_k1_FDX_s003
                                 + 1.42198 w_k1_IBM_s003
                                 + 1.15633 w_k1_INTC_s003
                                 + 1.09521 w_k1_KO_s003 + 1.80383 w_k1_MAT_s003
                                 + 0.968382 w_k1_MSFT_s003
                                 + 1.04328 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.10978 w_k1_ADM_s003
                                 + 0.667153 w_k1_AXP_s003
                                 + 1.06975 w_k1_DIS_s003
                                 + 0.943447 w_k1_FDX_s003
                                 + 0.72922 w_k1_IBM_s003 + 1.008 w_k1_INTC_s003
                                 + 1.02825 w_k1_KO_s003
                                 + 0.588842 w_k1_MAT_s003
                                 + 1.12762 w_k1_MSFT_s003
                                 + 0.895169 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.847314 w_k1_ADM_s003
                                 + 1.11603 w_k1_AXP_s003
                                 + 0.941232 w_k1_DIS_s003
                                 + 0.919653 w_k1_FDX_s003
                                 + 1.20314 w_k1_IBM_s003
                                 + 1.29254 w_k1_INTC_s003
                                 + 0.997258 w_k1_KO_s003
                                 + 1.23122 w_k1_MAT_s003
                                 + 0.997577 w_k1_MSFT_s003
                                 + 0.853163 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.26143 w_k1_ADM_s003
                                 + 1.26386 w_k1_AXP_s003
                                 + 1.79219 w_k1_DIS_s003
                                 + 1.41771 w_k1_FDX_s003
                                 + 1.11905 w_k1_IBM_s003
                                 + 1.47368 w_k1_INTC_s003 + 1.4491 w_k1_KO_s003
                                 + 1.41456 w_k1_MAT_s003
                                 + 1.43411 w_k1_MSFT_s003
                                 + 1.34473 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.818642 w_k1_ADM_s003
                                 + 0.685155 w_k1_AXP_s003
                                 + 0.812398 w_k1_DIS_s003
                                 + 0.762108 w_k1_FDX_s003
                                 + 0.95826 w_k1_IBM_s003
                                 + 1.09539 w_k1_INTC_s003
                                 + 1.01209 w_k1_KO_s003
                                 + 0.849969 w_k1_MAT_s003
                                 + 0.929622 w_k1_MSFT_s003
                                 + 0.827291 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.38466 w_k1_ADM_s003
                                 + 1.16218 w_k1_AXP_s003
                                 + 1.26074 w_k1_DIS_s003
                                 + 0.938981 w_k1_FDX_s003
                                 + 1.59564 w_k1_IBM_s003
                                 + 1.21415 w_k1_INTC_s003
                                 + 1.57502 w_k1_KO_s003 + 2.22921 w_k1_MAT_s003
                                 + 1.14411 w_k1_MSFT_s003
                                 + 1.17457 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.09882 w_k1_ADM_s004
                                 + 1.1209 w_k1_AXP_s004
                                 + 0.950871 w_k1_DIS_s004
                                 + 0.936297 w_k1_FDX_s004
                                 + 1.06617 w_k1_IBM_s004
                                 + 0.87238 w_k1_INTC_s004
                                 + 0.912555 w_k1_KO_s004
                                 + 1.50471 w_k1_MAT_s004
                                 + 0.959276 w_k1_MSFT_s004
                                 + 0.995924 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.33499 w_k1_ADM_s004
                                 + 1.43722 w_k1_AXP_s004
                                 + 1.31945 w_k1_DIS_s004
                                 + 1.17147 w_k1_FDX_s004 + 1.5487 w_k1_IBM_s004
                                 + 1.11823 w_k1_INTC_s004
                                 + 1.26103 w_k1_KO_s004 + 1.97601 w_k1_MAT_s004
                                 + 1.1756 w_k1_MSFT_s004 + 1.08192 w_k1_PG_s004
                                 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.958319 w_k1_ADM_s004
                                 + 1.07644 w_k1_AXP_s004
                                 + 0.872919 w_k1_DIS_s004
                                 + 0.991999 w_k1_FDX_s004
                                 + 0.862195 w_k1_IBM_s004
                                 + 0.814228 w_k1_INTC_s004
                                 + 0.848523 w_k1_KO_s004
                                 + 1.21302 w_k1_MAT_s004
                                 + 0.991113 w_k1_MSFT_s004
                                 + 0.9994 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.36095 w_k1_ADM_s004
                                 + 1.43114 w_k1_AXP_s004
                                 + 1.73673 w_k1_DIS_s004
                                 + 1.41596 w_k1_FDX_s004
                                 + 1.25282 w_k1_IBM_s004
                                 + 1.41275 w_k1_INTC_s004
                                 + 1.33699 w_k1_KO_s004 + 1.52165 w_k1_MAT_s004
                                 + 1.50665 w_k1_MSFT_s004
                                 + 1.19726 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.993012 w_k1_ADM_s004
                                 + 1.18795 w_k1_AXP_s004
                                 + 0.881257 w_k1_DIS_s004
                                 + 0.911587 w_k1_FDX_s004
                                 + 1.48803 w_k1_IBM_s004
                                 + 0.866699 w_k1_INTC_s004
                                 + 0.913357 w_k1_KO_s004
                                 + 1.86761 w_k1_MAT_s004
                                 + 0.888139 w_k1_MSFT_s004
                                 + 0.936276 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.29546 w_k1_ADM_s004
                                 + 1.02025 w_k1_AXP_s004 + 1.131 w_k1_DIS_s004
                                 + 1.10351 w_k1_FDX_s004
                                 + 0.799595 w_k1_IBM_s004
                                 + 1.07689 w_k1_INTC_s004
                                 + 1.05638 w_k1_KO_s004
                                 + 0.819993 w_k1_MAT_s004
                                 + 1.15416 w_k1_MSFT_s004
                                 + 1.07809 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.929949 w_k1_ADM_s004
                                 + 0.665955 w_k1_AXP_s004
                                 + 0.686104 w_k1_DIS_s004
                                 + 0.871664 w_k1_FDX_s004
                                 + 0.739454 w_k1_IBM_s004
                                 + 0.716509 w_k1_INTC_s004
                                 + 0.706692 w_k1_KO_s004
                                 + 0.726105 w_k1_MAT_s004
                                 + 0.855238 w_k1_MSFT_s004
                                 + 0.792788 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.838019 w_k1_ADM_s004
                                 + 1.17653 w_k1_AXP_s004
                                 + 1.21946 w_k1_DIS_s004
                                 + 1.11622 w_k1_FDX_s004
                                 + 1.15199 w_k1_IBM_s004
                                 + 1.08619 w_k1_INTC_s004
                                 + 1.07444 w_k1_KO_s004 + 1.49746 w_k1_MAT_s004
                                 + 1.36332 w_k1_MSFT_s004
                                 + 1.16801 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.943499 w_k1_ADM_s005
                                 + 0.862851 w_k1_AXP_s005
                                 + 0.830522 w_k1_DIS_s005
                                 + 0.92672 w_k1_FDX_s005
                                 + 0.668456 w_k1_IBM_s005
                                 + 0.714913 w_k1_INTC_s005
                                 + 0.721346 w_k1_KO_s005
                                 + 1.03398 w_k1_MAT_s005
                                 + 0.913136 w_k1_MSFT_s005
                                 + 0.907217 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.13195 w_k1_ADM_s005
                                 + 1.62827 w_k1_AXP_s005
                                 + 1.34638 w_k1_DIS_s005
                                 + 1.03164 w_k1_FDX_s005
                                 + 2.07553 w_k1_IBM_s005
                                 + 1.20521 w_k1_INTC_s005
                                 + 1.38537 w_k1_KO_s005 + 1.80768 w_k1_MAT_s005
                                 + 1.07706 w_k1_MSFT_s005
                                 + 1.16733 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.38895 w_k1_ADM_s005
                                 + 1.10229 w_k1_AXP_s005
                                 + 1.54589 w_k1_DIS_s005
                                 + 1.13205 w_k1_FDX_s005
                                 + 1.20984 w_k1_IBM_s005
                                 + 1.28778 w_k1_INTC_s005
                                 + 1.06427 w_k1_KO_s005
                                 + 0.620922 w_k1_MAT_s005
                                 + 1.15518 w_k1_MSFT_s005
                                 + 1.07156 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.188 w_k1_ADM_s005
                                 + 1.36844 w_k1_AXP_s005
                                 + 1.51023 w_k1_DIS_s005 + 1.4324 w_k1_FDX_s005
                                 + 0.921816 w_k1_IBM_s005
                                 + 1.26684 w_k1_INTC_s005
                                 + 1.13628 w_k1_KO_s005 + 1.11151 w_k1_MAT_s005
                                 + 1.28077 w_k1_MSFT_s005
                                 + 1.17821 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.806263 w_k1_ADM_s005
                                 + 1.16449 w_k1_AXP_s005
                                 + 0.972169 w_k1_DIS_s005
                                 + 0.861095 w_k1_FDX_s005
                                 + 1.26488 w_k1_IBM_s005
                                 + 0.978667 w_k1_INTC_s005
                                 + 1.06551 w_k1_KO_s005 + 1.8457 w_k1_MAT_s005
                                 + 0.822404 w_k1_MSFT_s005
                                 + 1.07694 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.40788 w_k1_ADM_s005
                                 + 1.18807 w_k1_AXP_s005 + 1.1252 w_k1_DIS_s005
                                 + 1.16925 w_k1_FDX_s005
                                 + 0.678593 w_k1_IBM_s005
                                 + 0.760087 w_k1_INTC_s005
                                 + 0.854913 w_k1_KO_s005
                                 + 0.945375 w_k1_MAT_s005
                                 + 1.05515 w_k1_MSFT_s005
                                 + 1.06996 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.61053 w_k1_ADM_s005
                                 + 0.828611 w_k1_AXP_s005
                                 + 0.652215 w_k1_DIS_s005
                                 + 0.645371 w_k1_FDX_s005
                                 + 0.952094 w_k1_IBM_s005
                                 + 0.713043 w_k1_INTC_s005
                                 + 0.769795 w_k1_KO_s005
                                 + 1.22634 w_k1_MAT_s005
                                 + 0.709346 w_k1_MSFT_s005
                                 + 0.745757 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.15251 w_k1_ADM_s005
                                 + 1.38959 w_k1_AXP_s005
                                 + 0.995936 w_k1_DIS_s005
                                 + 1.01044 w_k1_FDX_s005
                                 + 1.00876 w_k1_IBM_s005
                                 + 1.11696 w_k1_INTC_s005
                                 + 0.973602 w_k1_KO_s005
                                 + 1.37105 w_k1_MAT_s005
                                 + 1.00731 w_k1_MSFT_s005 + 1.2228 w_k1_PG_s005
                                 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.691858 w_k1_ADM_s006
                                 + 0.784645 w_k1_AXP_s006
                                 + 0.697771 w_k1_DIS_s006
                                 + 0.790388 w_k1_FDX_s006
                                 + 0.868039 w_k1_IBM_s006
                                 + 1.06664 w_k1_INTC_s006
                                 + 0.763976 w_k1_KO_s006
                                 + 0.75612 w_k1_MAT_s006
                                 + 0.827313 w_k1_MSFT_s006
                                 + 0.82874 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.811964 w_k1_ADM_s006
                                 + 1.52581 w_k1_AXP_s006
                                 + 1.45259 w_k1_DIS_s006
                                 + 1.13527 w_k1_FDX_s006
                                 + 1.64475 w_k1_IBM_s006
                                 + 1.41918 w_k1_INTC_s006
                                 + 1.43285 w_k1_KO_s006 + 1.90757 w_k1_MAT_s006
                                 + 1.16419 w_k1_MSFT_s006
                                 + 1.18099 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.23611 w_k1_ADM_s006
                                 + 1.3075 w_k1_AXP_s006 + 1.32192 w_k1_DIS_s006
                                 + 1.29088 w_k1_FDX_s006
                                 + 0.747333 w_k1_IBM_s006
                                 + 1.30513 w_k1_INTC_s006
                                 + 0.978786 w_k1_KO_s006
                                 + 0.777772 w_k1_MAT_s006
                                 + 1.35857 w_k1_MSFT_s006
                                 + 1.21241 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.854398 w_k1_ADM_s006
                                 + 0.8623 w_k1_AXP_s006
                                 + 0.862392 w_k1_DIS_s006
                                 + 0.886659 w_k1_FDX_s006
                                 + 0.688834 w_k1_IBM_s006
                                 + 1.1084 w_k1_INTC_s006 + 0.85308 w_k1_KO_s006
                                 + 0.625076 w_k1_MAT_s006
                                 + 0.958784 w_k1_MSFT_s006
                                 + 0.894532 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.941258 w_k1_ADM_s006
                                 + 1.10139 w_k1_AXP_s006
                                 + 1.05439 w_k1_DIS_s006
                                 + 0.924749 w_k1_FDX_s006
                                 + 1.47219 w_k1_IBM_s006
                                 + 1.21944 w_k1_INTC_s006
                                 + 1.22345 w_k1_KO_s006 + 1.79809 w_k1_MAT_s006
                                 + 0.953127 w_k1_MSFT_s006
                                 + 1.15646 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.800474 w_k1_ADM_s006
                                 + 1.24243 w_k1_AXP_s006
                                 + 0.967669 w_k1_DIS_s006
                                 + 1.09464 w_k1_FDX_s006
                                 + 0.914097 w_k1_IBM_s006
                                 + 1.18096 w_k1_INTC_s006
                                 + 0.890817 w_k1_KO_s006 + 1.0931 w_k1_MAT_s006
                                 + 0.950428 w_k1_MSFT_s006
                                 + 1.03878 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.871189 w_k1_ADM_s006
                                 + 1.13346 w_k1_AXP_s006
                                 + 0.903647 w_k1_DIS_s006
                                 + 0.90509 w_k1_FDX_s006 + 1.1188 w_k1_IBM_s006
                                 + 1.13204 w_k1_INTC_s006
                                 + 1.00505 w_k1_KO_s006 + 1.17239 w_k1_MAT_s006
                                 + 1.00826 w_k1_MSFT_s006
                                 + 1.09355 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.952765 w_k1_ADM_s006
                                 + 0.923063 w_k1_AXP_s006
                                 + 0.982751 w_k1_DIS_s006
                                 + 1.17046 w_k1_FDX_s006
                                 + 0.660563 w_k1_IBM_s006
                                 + 1.33169 w_k1_INTC_s006
                                 + 0.825629 w_k1_KO_s006
                                 + 0.672653 w_k1_MAT_s006
                                 + 1.23372 w_k1_MSFT_s006
                                 + 1.00993 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.89295 w_k1_ADM_s007
                                 + 0.871762 w_k1_AXP_s007
                                 + 0.922321 w_k1_DIS_s007
                                 + 0.846008 w_k1_FDX_s007
                                 + 1.13417 w_k1_IBM_s007
                                 + 0.909616 w_k1_INTC_s007
                                 + 1.14289 w_k1_KO_s007 + 1.37528 w_k1_MAT_s007
                                 + 0.976519 w_k1_MSFT_s007
                                 + 1.08335 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.899223 w_k1_ADM_s007
                                 + 0.80783 w_k1_AXP_s007
                                 + 0.792265 w_k1_DIS_s007
                                 + 0.888867 w_k1_FDX_s007
                                 + 0.773606 w_k1_IBM_s007
                                 + 0.893608 w_k1_INTC_s007
                                 + 0.869088 w_k1_KO_s007
                                 + 0.980355 w_k1_MAT_s007
                                 + 0.862582 w_k1_MSFT_s007
                                 + 0.904203 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.19739 w_k1_ADM_s007
                                 + 1.04277 w_k1_AXP_s007
                                 + 0.982209 w_k1_DIS_s007
                                 + 0.996123 w_k1_FDX_s007
                                 + 0.864216 w_k1_IBM_s007
                                 + 0.89988 w_k1_INTC_s007
                                 + 0.992364 w_k1_KO_s007
                                 + 0.883103 w_k1_MAT_s007
                                 + 1.11253 w_k1_MSFT_s007
                                 + 1.14925 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.96742 w_k1_ADM_s007
                                 + 0.755398 w_k1_AXP_s007
                                 + 0.985323 w_k1_DIS_s007
                                 + 0.881896 w_k1_FDX_s007
                                 + 0.694896 w_k1_IBM_s007
                                 + 1.07196 w_k1_INTC_s007
                                 + 1.04064 w_k1_KO_s007
                                 + 0.689355 w_k1_MAT_s007
                                 + 1.11262 w_k1_MSFT_s007
                                 + 0.976489 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.11742 w_k1_ADM_s007
                                 + 1.20586 w_k1_AXP_s007
                                 + 1.10809 w_k1_DIS_s007
                                 + 1.12382 w_k1_FDX_s007
                                 + 0.985291 w_k1_IBM_s007
                                 + 0.882818 w_k1_INTC_s007
                                 + 1.10986 w_k1_KO_s007
                                 + 0.937068 w_k1_MAT_s007
                                 + 1.11616 w_k1_MSFT_s007
                                 + 1.16071 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.701203 w_k1_ADM_s007
                                 + 0.719648 w_k1_AXP_s007
                                 + 0.596045 w_k1_DIS_s007
                                 + 0.702562 w_k1_FDX_s007
                                 + 0.834765 w_k1_IBM_s007
                                 + 0.787614 w_k1_INTC_s007
                                 + 0.806083 w_k1_KO_s007
                                 + 0.991489 w_k1_MAT_s007
                                 + 0.776985 w_k1_MSFT_s007
                                 + 0.805313 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.52178 w_k1_ADM_s007
                                 + 1.5677 w_k1_AXP_s007 + 1.30032 w_k1_DIS_s007
                                 + 1.24654 w_k1_FDX_s007
                                 + 1.42533 w_k1_IBM_s007
                                 + 1.15234 w_k1_INTC_s007
                                 + 1.24705 w_k1_KO_s007 + 1.55993 w_k1_MAT_s007
                                 + 1.27674 w_k1_MSFT_s007
                                 + 1.27679 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.11089 w_k1_ADM_s007
                                 + 1.24124 w_k1_AXP_s007
                                 + 1.44728 w_k1_DIS_s007
                                 + 1.29028 w_k1_FDX_s007
                                 + 1.19801 w_k1_IBM_s007
                                 + 1.44873 w_k1_INTC_s007
                                 + 1.39231 w_k1_KO_s007 + 1.20401 w_k1_MAT_s007
                                 + 1.41438 w_k1_MSFT_s007
                                 + 1.21674 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.24889 w_k1_ADM_s010
                                 + 1.48737 w_k1_AXP_s010
                                 + 1.37378 w_k1_DIS_s010
                                 + 1.41276 w_k1_FDX_s010
                                 + 0.931224 w_k1_IBM_s010
                                 + 1.43423 w_k1_INTC_s010
                                 + 1.33684 w_k1_KO_s010
                                 + 0.682431 w_k1_MAT_s010
                                 + 1.3098 w_k1_MSFT_s010 + 1.29512 w_k1_PG_s010
                                 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.778638 w_k1_ADM_s010
                                 + 0.802261 w_k1_AXP_s010
                                 + 0.940773 w_k1_DIS_s010
                                 + 0.939218 w_k1_FDX_s010
                                 + 0.845251 w_k1_IBM_s010
                                 + 1.06687 w_k1_INTC_s010
                                 + 1.00079 w_k1_KO_s010
                                 + 0.734278 w_k1_MAT_s010
                                 + 0.928117 w_k1_MSFT_s010
                                 + 0.968179 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.870446 w_k1_ADM_s010
                                 + 0.899896 w_k1_AXP_s010
                                 + 0.779674 w_k1_DIS_s010
                                 + 0.928369 w_k1_FDX_s010
                                 + 0.659034 w_k1_IBM_s010
                                 + 1.0247 w_k1_INTC_s010
                                 + 0.817239 w_k1_KO_s010
                                 + 0.986832 w_k1_MAT_s010
                                 + 0.978408 w_k1_MSFT_s010
                                 + 0.894351 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.727498 w_k1_ADM_s010
                                 + 0.628352 w_k1_AXP_s010
                                 + 0.771812 w_k1_DIS_s010
                                 + 0.787374 w_k1_FDX_s010
                                 + 1.17932 w_k1_IBM_s010
                                 + 0.847079 w_k1_INTC_s010
                                 + 1.04402 w_k1_KO_s010
                                 + 0.923167 w_k1_MAT_s010
                                 + 0.931466 w_k1_MSFT_s010
                                 + 1.01664 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.517888 w_k1_ADM_s010
                                 + 0.971429 w_k1_AXP_s010
                                 + 1.20738 w_k1_DIS_s010
                                 + 0.948265 w_k1_FDX_s010
                                 + 1.60966 w_k1_IBM_s010
                                 + 1.05324 w_k1_INTC_s010
                                 + 1.45016 w_k1_KO_s010 + 1.37953 w_k1_MAT_s010
                                 + 0.910739 w_k1_MSFT_s010
                                 + 1.13776 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.883815 w_k1_ADM_s010
                                 + 1.01585 w_k1_AXP_s010
                                 + 1.09869 w_k1_DIS_s010 + 1.1309 w_k1_FDX_s010
                                 + 1.061 w_k1_IBM_s010 + 1.09715 w_k1_INTC_s010
                                 + 1.23006 w_k1_KO_s010 + 1.19607 w_k1_MAT_s010
                                 + 1.24431 w_k1_MSFT_s010
                                 + 1.10953 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.932123 w_k1_ADM_s010
                                 + 1.32079 w_k1_AXP_s010
                                 + 1.37685 w_k1_DIS_s010
                                 + 1.22744 w_k1_FDX_s010
                                 + 1.25606 w_k1_IBM_s010
                                 + 1.15097 w_k1_INTC_s010
                                 + 1.49012 w_k1_KO_s010 + 1.60088 w_k1_MAT_s010
                                 + 1.21642 w_k1_MSFT_s010
                                 + 1.15369 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.87339 w_k1_ADM_s010
                                 + 1.0248 w_k1_AXP_s010
                                 + 0.668398 w_k1_DIS_s010
                                 + 0.952355 w_k1_FDX_s010
                                 + 0.940889 w_k1_IBM_s010
                                 + 1.06492 w_k1_INTC_s010
                                 + 0.890296 w_k1_KO_s010
                                 + 1.00285 w_k1_MAT_s010
                                 + 0.917798 w_k1_MSFT_s010
                                 + 0.955235 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.730569 w_k1_ADM_s011
                                 + 0.682159 w_k1_AXP_s011
                                 + 0.591724 w_k1_DIS_s011
                                 + 0.660089 w_k1_FDX_s011
                                 + 1.14849 w_k1_IBM_s011
                                 + 0.717891 w_k1_INTC_s011
                                 + 0.721257 w_k1_KO_s011
                                 + 0.987962 w_k1_MAT_s011
                                 + 0.644103 w_k1_MSFT_s011
                                 + 0.832504 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.2754 w_k1_ADM_s011
                                 + 0.920008 w_k1_AXP_s011
                                 + 0.947555 w_k1_DIS_s011 + 1.004 w_k1_FDX_s011
                                 + 0.904431 w_k1_IBM_s011
                                 + 0.897091 w_k1_INTC_s011
                                 + 0.915252 w_k1_KO_s011
                                 + 0.69491 w_k1_MAT_s011
                                 + 0.938339 w_k1_MSFT_s011
                                 + 0.986641 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.875893 w_k1_ADM_s011
                                 + 1.05932 w_k1_AXP_s011
                                 + 1.12772 w_k1_DIS_s011
                                 + 0.995843 w_k1_FDX_s011
                                 + 1.32795 w_k1_IBM_s011
                                 + 1.09248 w_k1_INTC_s011
                                 + 1.08249 w_k1_KO_s011
                                 + 0.893963 w_k1_MAT_s011
                                 + 1.00538 w_k1_MSFT_s011
                                 + 1.06243 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.33279 w_k1_ADM_s011
                                 + 1.17596 w_k1_AXP_s011
                                 + 0.940797 w_k1_DIS_s011
                                 + 1.05933 w_k1_FDX_s011
                                 + 1.11973 w_k1_IBM_s011
                                 + 0.839623 w_k1_INTC_s011
                                 + 0.89665 w_k1_KO_s011 + 1.17966 w_k1_MAT_s011
                                 + 0.987697 w_k1_MSFT_s011
                                 + 1.16699 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.975716 w_k1_ADM_s011
                                 + 0.831925 w_k1_AXP_s011
                                 + 0.82856 w_k1_DIS_s011
                                 + 1.03146 w_k1_FDX_s011
                                 + 0.694328 w_k1_IBM_s011
                                 + 1.10648 w_k1_INTC_s011
                                 + 0.773255 w_k1_KO_s011
                                 + 0.740321 w_k1_MAT_s011
                                 + 0.925663 w_k1_MSFT_s011
                                 + 0.915165 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.784891 w_k1_ADM_s011
                                 + 0.750021 w_k1_AXP_s011
                                 + 0.938515 w_k1_DIS_s011
                                 + 0.882427 w_k1_FDX_s011
                                 + 0.998457 w_k1_IBM_s011
                                 + 1.03443 w_k1_INTC_s011
                                 + 1.05159 w_k1_KO_s011
                                 + 0.748899 w_k1_MAT_s011
                                 + 0.958646 w_k1_MSFT_s011
                                 + 0.940286 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.900396 w_k1_ADM_s011
                                 + 0.675191 w_k1_AXP_s011
                                 + 0.578715 w_k1_DIS_s011
                                 + 0.763492 w_k1_FDX_s011
                                 + 0.653613 w_k1_IBM_s011
                                 + 0.733978 w_k1_INTC_s011
                                 + 0.711829 w_k1_KO_s011
                                 + 0.59736 w_k1_MAT_s011
                                 + 0.747444 w_k1_MSFT_s011
                                 + 0.793426 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.36894 w_k1_ADM_s011
                                 + 1.14857 w_k1_AXP_s011 + 1.46 w_k1_DIS_s011
                                 + 1.44446 w_k1_FDX_s011
                                 + 0.995951 w_k1_IBM_s011
                                 + 1.10146 w_k1_INTC_s011
                                 + 1.13375 w_k1_KO_s011 + 1.11423 w_k1_MAT_s011
                                 + 1.23078 w_k1_MSFT_s011
                                 + 1.17857 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.756197 w_k1_ADM_s012
                                 + 0.687955 w_k1_AXP_s012
                                 + 0.752098 w_k1_DIS_s012
                                 + 0.704829 w_k1_FDX_s012
                                 + 1.09989 w_k1_IBM_s012
                                 + 0.744846 w_k1_INTC_s012
                                 + 0.935039 w_k1_KO_s012
                                 + 0.860106 w_k1_MAT_s012
                                 + 0.721487 w_k1_MSFT_s012
                                 + 0.828609 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.15032 w_k1_ADM_s012
                                 + 1.39429 w_k1_AXP_s012 + 1.6431 w_k1_DIS_s012
                                 + 1.32413 w_k1_FDX_s012
                                 + 1.25586 w_k1_IBM_s012
                                 + 1.30772 w_k1_INTC_s012
                                 + 1.24136 w_k1_KO_s012 + 1.16682 w_k1_MAT_s012
                                 + 1.37362 w_k1_MSFT_s012
                                 + 1.19157 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.15195 w_k1_ADM_s012
                                 + 0.957102 w_k1_AXP_s012
                                 + 1.44569 w_k1_DIS_s012
                                 + 1.23284 w_k1_FDX_s012
                                 + 0.798373 w_k1_IBM_s012
                                 + 1.07872 w_k1_INTC_s012
                                 + 1.08857 w_k1_KO_s012
                                 + 0.651688 w_k1_MAT_s012
                                 + 1.28007 w_k1_MSFT_s012
                                 + 1.12945 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.707848 w_k1_ADM_s012
                                 + 0.71186 w_k1_AXP_s012
                                 + 0.615814 w_k1_DIS_s012
                                 + 0.78476 w_k1_FDX_s012
                                 + 0.706938 w_k1_IBM_s012
                                 + 0.816486 w_k1_INTC_s012
                                 + 0.710345 w_k1_KO_s012
                                 + 0.671415 w_k1_MAT_s012
                                 + 0.893845 w_k1_MSFT_s012
                                 + 0.86601 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.28698 w_k1_ADM_s012
                                 + 1.46125 w_k1_AXP_s012
                                 + 1.18958 w_k1_DIS_s012
                                 + 1.16709 w_k1_FDX_s012
                                 + 1.16074 w_k1_IBM_s012
                                 + 0.986262 w_k1_INTC_s012
                                 + 1.21933 w_k1_KO_s012 + 1.39345 w_k1_MAT_s012
                                 + 1.11083 w_k1_MSFT_s012
                                 + 1.09426 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.883102 w_k1_ADM_s012
                                 + 1.02409 w_k1_AXP_s012
                                 + 1.04245 w_k1_DIS_s012
                                 + 1.05138 w_k1_FDX_s012
                                 + 1.00478 w_k1_IBM_s012
                                 + 0.883492 w_k1_INTC_s012
                                 + 1.01721 w_k1_KO_s012 + 1.08542 w_k1_MAT_s012
                                 + 1.11007 w_k1_MSFT_s012
                                 + 0.982459 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.07026 w_k1_ADM_s012
                                 + 0.939093 w_k1_AXP_s012
                                 + 1.21843 w_k1_DIS_s012
                                 + 1.03019 w_k1_FDX_s012
                                 + 1.58211 w_k1_IBM_s012
                                 + 1.14652 w_k1_INTC_s012
                                 + 1.37055 w_k1_KO_s012 + 1.22087 w_k1_MAT_s012
                                 + 1.24959 w_k1_MSFT_s012
                                 + 1.02736 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.938021 w_k1_ADM_s012
                                 + 0.925356 w_k1_AXP_s012
                                 + 0.779862 w_k1_DIS_s012
                                 + 0.930826 w_k1_FDX_s012
                                 + 0.855928 w_k1_IBM_s012
                                 + 1.18791 w_k1_INTC_s012
                                 + 0.939068 w_k1_KO_s012
                                 + 0.74019 w_k1_MAT_s012
                                 + 1.02874 w_k1_MSFT_s012
                                 + 0.892013 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.19311 w_k1_ADM_s013
                                 + 0.98902 w_k1_AXP_s013
                                 + 0.850473 w_k1_DIS_s013
                                 + 0.908392 w_k1_FDX_s013
                                 + 0.811925 w_k1_IBM_s013
                                 + 0.960519 w_k1_INTC_s013
                                 + 0.905037 w_k1_KO_s013
                                 + 0.901557 w_k1_MAT_s013
                                 + 0.866628 w_k1_MSFT_s013
                                 + 0.959891 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.30255 w_k1_ADM_s013
                                 + 1.28341 w_k1_AXP_s013
                                 + 1.00356 w_k1_DIS_s013
                                 + 1.03472 w_k1_FDX_s013
                                 + 1.00501 w_k1_IBM_s013
                                 + 1.0882 w_k1_INTC_s013
                                 + 0.958676 w_k1_KO_s013
                                 + 0.913478 w_k1_MAT_s013
                                 + 0.902827 w_k1_MSFT_s013
                                 + 1.05049 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.916698 w_k1_ADM_s013
                                 + 0.95654 w_k1_AXP_s013
                                 + 1.09599 w_k1_DIS_s013
                                 + 0.903092 w_k1_FDX_s013
                                 + 1.03529 w_k1_IBM_s013
                                 + 1.14112 w_k1_INTC_s013
                                 + 1.07194 w_k1_KO_s013 + 1.13916 w_k1_MAT_s013
                                 + 1.1402 w_k1_MSFT_s013 + 1.00386 w_k1_PG_s013
                                 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.34177 w_k1_ADM_s013
                                 + 0.893359 w_k1_AXP_s013
                                 + 0.783953 w_k1_DIS_s013
                                 + 0.724872 w_k1_FDX_s013
                                 + 1.26591 w_k1_IBM_s013
                                 + 0.752685 w_k1_INTC_s013
                                 + 1.1373 w_k1_KO_s013 + 1.31576 w_k1_MAT_s013
                                 + 0.740083 w_k1_MSFT_s013
                                 + 1.02106 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.06776 w_k1_ADM_s013
                                 + 0.991367 w_k1_AXP_s013
                                 + 0.984547 w_k1_DIS_s013
                                 + 1.01671 w_k1_FDX_s013
                                 + 1.19633 w_k1_IBM_s013
                                 + 0.989486 w_k1_INTC_s013
                                 + 1.23959 w_k1_KO_s013 + 1.42948 w_k1_MAT_s013
                                 + 0.933992 w_k1_MSFT_s013
                                 + 1.08939 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.13501 w_k1_ADM_s013
                                 + 1.24808 w_k1_AXP_s013
                                 + 1.08073 w_k1_DIS_s013
                                 + 1.19665 w_k1_FDX_s013
                                 + 0.745771 w_k1_IBM_s013
                                 + 1.29127 w_k1_INTC_s013
                                 + 0.908102 w_k1_KO_s013
                                 + 0.917521 w_k1_MAT_s013
                                 + 1.14893 w_k1_MSFT_s013
                                 + 0.995338 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.25563 w_k1_ADM_s013
                                 + 1.45976 w_k1_AXP_s013
                                 + 1.46506 w_k1_DIS_s013
                                 + 1.30812 w_k1_FDX_s013
                                 + 1.23252 w_k1_IBM_s013
                                 + 1.23609 w_k1_INTC_s013
                                 + 1.36631 w_k1_KO_s013 + 1.58898 w_k1_MAT_s013
                                 + 1.34524 w_k1_MSFT_s013
                                 + 1.35512 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.22613 w_k1_ADM_s013
                                 + 0.789573 w_k1_AXP_s013
                                 + 0.746537 w_k1_DIS_s013
                                 + 0.81988 w_k1_FDX_s013
                                 + 0.61637 w_k1_IBM_s013
                                 + 0.705394 w_k1_INTC_s013
                                 + 0.827222 w_k1_KO_s013
                                 + 0.832807 w_k1_MAT_s013
                                 + 0.787782 w_k1_MSFT_s013
                                 + 0.841908 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.00447 w_k1_ADM_s014
                                 + 0.824945 w_k1_AXP_s014
                                 + 0.744125 w_k1_DIS_s014
                                 + 0.82191 w_k1_FDX_s014
                                 + 1.02888 w_k1_IBM_s014
                                 + 0.924052 w_k1_INTC_s014
                                 + 0.80825 w_k1_KO_s014 + 1.00704 w_k1_MAT_s014
                                 + 0.834157 w_k1_MSFT_s014
                                 + 0.880995 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.956099 w_k1_ADM_s014
                                 + 0.91635 w_k1_AXP_s014
                                 + 0.93446 w_k1_DIS_s014
                                 + 0.943005 w_k1_FDX_s014
                                 + 1.24738 w_k1_IBM_s014
                                 + 0.934984 w_k1_INTC_s014
                                 + 0.972067 w_k1_KO_s014 + 1.2154 w_k1_MAT_s014
                                 + 0.97342 w_k1_MSFT_s014
                                 + 0.986933 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.12661 w_k1_ADM_s014
                                 + 1.07833 w_k1_AXP_s014
                                 + 1.09096 w_k1_DIS_s014
                                 + 0.932754 w_k1_FDX_s014
                                 + 1.68012 w_k1_IBM_s014
                                 + 1.05316 w_k1_INTC_s014 + 1.2898 w_k1_KO_s014
                                 + 1.70947 w_k1_MAT_s014
                                 + 1.02308 w_k1_MSFT_s014
                                 + 1.14393 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.684729 w_k1_ADM_s014
                                 + 0.755971 w_k1_AXP_s014
                                 + 0.577112 w_k1_DIS_s014
                                 + 0.689007 w_k1_FDX_s014
                                 + 0.884938 w_k1_IBM_s014
                                 + 0.946341 w_k1_INTC_s014
                                 + 0.791306 w_k1_KO_s014
                                 + 0.747537 w_k1_MAT_s014
                                 + 0.752121 w_k1_MSFT_s014
                                 + 0.781291 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.24525 w_k1_ADM_s014
                                 + 1.42271 w_k1_AXP_s014
                                 + 1.59641 w_k1_DIS_s014
                                 + 1.37293 w_k1_FDX_s014
                                 + 1.12069 w_k1_IBM_s014
                                 + 1.25108 w_k1_INTC_s014
                                 + 1.15956 w_k1_KO_s014 + 1.11937 w_k1_MAT_s014
                                 + 1.26956 w_k1_MSFT_s014
                                 + 1.15806 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.798998 w_k1_ADM_s014
                                 + 1.06572 w_k1_AXP_s014
                                 + 0.979611 w_k1_DIS_s014
                                 + 1.00219 w_k1_FDX_s014
                                 + 0.747371 w_k1_IBM_s014
                                 + 1.17755 w_k1_INTC_s014
                                 + 0.813542 w_k1_KO_s014
                                 + 0.668881 w_k1_MAT_s014
                                 + 0.896398 w_k1_MSFT_s014
                                 + 0.869005 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.15489 w_k1_ADM_s014
                                 + 1.20376 w_k1_AXP_s014 + 1.0456 w_k1_DIS_s014
                                 + 1.0756 w_k1_FDX_s014 + 1.11127 w_k1_IBM_s014
                                 + 0.97472 w_k1_INTC_s014
                                 + 0.91029 w_k1_KO_s014 + 1.13759 w_k1_MAT_s014
                                 + 0.928221 w_k1_MSFT_s014
                                 + 0.915968 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.952236 w_k1_ADM_s014
                                 + 0.884731 w_k1_AXP_s014
                                 + 0.648962 w_k1_DIS_s014
                                 + 0.880676 w_k1_FDX_s014
                                 + 0.778309 w_k1_IBM_s014
                                 + 0.848528 w_k1_INTC_s014
                                 + 0.741327 w_k1_KO_s014
                                 + 0.588947 w_k1_MAT_s014
                                 + 0.837241 w_k1_MSFT_s014
                                 + 0.90399 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.751735 w_k1_ADM_s015
                                 + 0.864334 w_k1_AXP_s015
                                 + 0.698487 w_k1_DIS_s015
                                 + 0.838214 w_k1_FDX_s015
                                 + 0.951934 w_k1_IBM_s015
                                 + 0.963157 w_k1_INTC_s015
                                 + 0.864802 w_k1_KO_s015
                                 + 1.03392 w_k1_MAT_s015
                                 + 0.82365 w_k1_MSFT_s015
                                 + 0.774048 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.731061 w_k1_ADM_s015
                                 + 0.70355 w_k1_AXP_s015
                                 + 0.713024 w_k1_DIS_s015
                                 + 0.851692 w_k1_FDX_s015
                                 + 0.585603 w_k1_IBM_s015
                                 + 0.892469 w_k1_INTC_s015
                                 + 0.734133 w_k1_KO_s015
                                 + 0.76125 w_k1_MAT_s015
                                 + 0.898479 w_k1_MSFT_s015
                                 + 0.73332 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.794593 w_k1_ADM_s015
                                 + 1.13845 w_k1_AXP_s015
                                 + 0.937408 w_k1_DIS_s015
                                 + 1.07056 w_k1_FDX_s015
                                 + 0.995093 w_k1_IBM_s015
                                 + 0.996659 w_k1_INTC_s015
                                 + 0.927018 w_k1_KO_s015
                                 + 1.46075 w_k1_MAT_s015
                                 + 0.980424 w_k1_MSFT_s015
                                 + 0.915161 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.844304 w_k1_ADM_s015
                                 + 1.01634 w_k1_AXP_s015
                                 + 0.889464 w_k1_DIS_s015
                                 + 0.848162 w_k1_FDX_s015
                                 + 1.47874 w_k1_IBM_s015
                                 + 1.01661 w_k1_INTC_s015
                                 + 1.12026 w_k1_KO_s015 + 1.63899 w_k1_MAT_s015
                                 + 0.886792 w_k1_MSFT_s015
                                 + 1.00535 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.771932 w_k1_ADM_s015
                                 + 1.47882 w_k1_AXP_s015
                                 + 1.14852 w_k1_DIS_s015
                                 + 1.16268 w_k1_FDX_s015 + 1.3639 w_k1_IBM_s015
                                 + 1.29886 w_k1_INTC_s015
                                 + 1.17843 w_k1_KO_s015 + 1.98451 w_k1_MAT_s015
                                 + 1.0419 w_k1_MSFT_s015 + 1.0877 w_k1_PG_s015
                                 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.948805 w_k1_ADM_s015
                                 + 1.14083 w_k1_AXP_s015
                                 + 1.37719 w_k1_DIS_s015
                                 + 1.27991 w_k1_FDX_s015
                                 + 0.998139 w_k1_IBM_s015
                                 + 1.25501 w_k1_INTC_s015
                                 + 1.27857 w_k1_KO_s015 + 1.06841 w_k1_MAT_s015
                                 + 1.27504 w_k1_MSFT_s015
                                 + 1.21009 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.21071 w_k1_ADM_s015
                                 + 1.88727 w_k1_AXP_s015
                                 + 1.68982 w_k1_DIS_s015
                                 + 1.51606 w_k1_FDX_s015 + 1.3521 w_k1_IBM_s015
                                 + 1.5321 w_k1_INTC_s015 + 1.55665 w_k1_KO_s015
                                 + 2.1234 w_k1_MAT_s015
                                 + 1.28755 w_k1_MSFT_s015
                                 + 1.20558 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.794866 w_k1_ADM_s015
                                 + 0.915503 w_k1_AXP_s015
                                 + 1.16505 w_k1_DIS_s015
                                 + 1.20413 w_k1_FDX_s015
                                 + 0.82502 w_k1_IBM_s015
                                 + 1.38529 w_k1_INTC_s015
                                 + 0.957199 w_k1_KO_s015
                                 + 0.779257 w_k1_MAT_s015
                                 + 1.14634 w_k1_MSFT_s015
                                 + 0.960727 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.11368 w_k1_ADM_s016
                                 + 1.03519 w_k1_AXP_s016
                                 + 0.921076 w_k1_DIS_s016
                                 + 0.889373 w_k1_FDX_s016
                                 + 1.22351 w_k1_IBM_s016
                                 + 1.02337 w_k1_INTC_s016
                                 + 0.901779 w_k1_KO_s016
                                 + 0.878885 w_k1_MAT_s016
                                 + 1.24983 w_k1_MSFT_s016
                                 + 1.07367 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.39139 w_k1_ADM_s016
                                 + 1.54016 w_k1_AXP_s016 + 1.3014 w_k1_DIS_s016
                                 + 1.24651 w_k1_FDX_s016
                                 + 1.05786 w_k1_IBM_s016
                                 + 1.28973 w_k1_INTC_s016 + 1.0954 w_k1_KO_s016
                                 + 1.28019 w_k1_MAT_s016
                                 + 1.3786 w_k1_MSFT_s016 + 1.20835 w_k1_PG_s016
                                 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.21754 w_k1_ADM_s016
                                 + 1.03456 w_k1_AXP_s016
                                 + 1.41507 w_k1_DIS_s016
                                 + 1.07653 w_k1_FDX_s016
                                 + 0.954378 w_k1_IBM_s016
                                 + 1.43502 w_k1_INTC_s016
                                 + 1.21433 w_k1_KO_s016
                                 + 0.827072 w_k1_MAT_s016
                                 + 1.36627 w_k1_MSFT_s016
                                 + 1.16563 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.933325 w_k1_ADM_s016
                                 + 0.831985 w_k1_AXP_s016
                                 + 0.786949 w_k1_DIS_s016
                                 + 0.80368 w_k1_FDX_s016
                                 + 1.11571 w_k1_IBM_s016
                                 + 0.932691 w_k1_INTC_s016
                                 + 0.854427 w_k1_KO_s016
                                 + 0.88824 w_k1_MAT_s016
                                 + 0.911161 w_k1_MSFT_s016
                                 + 0.84788 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.824469 w_k1_ADM_s016
                                 + 0.679301 w_k1_AXP_s016
                                 + 0.553172 w_k1_DIS_s016
                                 + 0.647099 w_k1_FDX_s016
                                 + 0.718939 w_k1_IBM_s016
                                 + 0.741926 w_k1_INTC_s016
                                 + 0.776795 w_k1_KO_s016
                                 + 0.698395 w_k1_MAT_s016
                                 + 0.823692 w_k1_MSFT_s016
                                 + 0.889613 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.18799 w_k1_ADM_s016
                                 + 0.913325 w_k1_AXP_s016
                                 + 0.909704 w_k1_DIS_s016
                                 + 0.911554 w_k1_FDX_s016
                                 + 0.695181 w_k1_IBM_s016
                                 + 1.14625 w_k1_INTC_s016
                                 + 0.845643 w_k1_KO_s016
                                 + 0.573298 w_k1_MAT_s016
                                 + 1.18582 w_k1_MSFT_s016
                                 + 0.944835 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.07193 w_k1_ADM_s016
                                 + 1.20262 w_k1_AXP_s016
                                 + 0.987062 w_k1_DIS_s016
                                 + 1.00944 w_k1_FDX_s016
                                 + 1.05678 w_k1_IBM_s016
                                 + 0.989575 w_k1_INTC_s016
                                 + 0.906143 w_k1_KO_s016
                                 + 0.82412 w_k1_MAT_s016
                                 + 0.989754 w_k1_MSFT_s016
                                 + 1.0509 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.935139 w_k1_ADM_s016
                                 + 0.970745 w_k1_AXP_s016
                                 + 1.45358 w_k1_DIS_s016
                                 + 0.999993 w_k1_FDX_s016
                                 + 1.37216 w_k1_IBM_s016
                                 + 1.28305 w_k1_INTC_s016
                                 + 1.28098 w_k1_KO_s016
                                 + 0.969214 w_k1_MAT_s016
                                 + 1.39149 w_k1_MSFT_s016
                                 + 1.12541 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.86761 w_k1_ADM_s017
                                 + 0.982994 w_k1_AXP_s017
                                 + 0.77161 w_k1_DIS_s017
                                 + 0.807932 w_k1_FDX_s017
                                 + 1.06787 w_k1_IBM_s017
                                 + 0.798329 w_k1_INTC_s017
                                 + 1.19152 w_k1_KO_s017 + 1.39255 w_k1_MAT_s017
                                 + 0.85977 w_k1_MSFT_s017
                                 + 1.05778 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.18162 w_k1_ADM_s017
                                 + 1.31515 w_k1_AXP_s017
                                 + 1.47447 w_k1_DIS_s017
                                 + 1.22624 w_k1_FDX_s017
                                 + 1.24482 w_k1_IBM_s017
                                 + 1.15124 w_k1_INTC_s017
                                 + 1.47123 w_k1_KO_s017 + 1.53582 w_k1_MAT_s017
                                 + 1.25901 w_k1_MSFT_s017
                                 + 1.31117 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.921083 w_k1_ADM_s017
                                 + 0.909648 w_k1_AXP_s017
                                 + 1.27374 w_k1_DIS_s017 + 1.2434 w_k1_FDX_s017
                                 + 0.72158 w_k1_IBM_s017
                                 + 1.0386 w_k1_INTC_s017
                                 + 0.940233 w_k1_KO_s017
                                 + 1.02678 w_k1_MAT_s017
                                 + 1.09784 w_k1_MSFT_s017
                                 + 0.995537 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.998394 w_k1_ADM_s017
                                 + 0.842313 w_k1_AXP_s017
                                 + 1.10608 w_k1_DIS_s017
                                 + 0.999645 w_k1_FDX_s017
                                 + 0.974032 w_k1_IBM_s017
                                 + 0.88842 w_k1_INTC_s017
                                 + 1.06437 w_k1_KO_s017 + 1.21457 w_k1_MAT_s017
                                 + 1.03348 w_k1_MSFT_s017
                                 + 0.967561 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.01659 w_k1_ADM_s017
                                 + 1.42884 w_k1_AXP_s017
                                 + 1.11845 w_k1_DIS_s017
                                 + 1.25322 w_k1_FDX_s017
                                 + 1.14945 w_k1_IBM_s017
                                 + 1.16749 w_k1_INTC_s017
                                 + 1.09022 w_k1_KO_s017 + 1.52994 w_k1_MAT_s017
                                 + 1.08634 w_k1_MSFT_s017
                                 + 1.01817 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.12612 w_k1_ADM_s017
                                 + 0.787195 w_k1_AXP_s017
                                 + 0.782428 w_k1_DIS_s017
                                 + 1.0375 w_k1_FDX_s017
                                 + 0.720371 w_k1_IBM_s017
                                 + 0.90428 w_k1_INTC_s017
                                 + 0.835004 w_k1_KO_s017
                                 + 0.535142 w_k1_MAT_s017
                                 + 0.932547 w_k1_MSFT_s017
                                 + 0.891007 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.779062 w_k1_ADM_s017
                                 + 0.801541 w_k1_AXP_s017
                                 + 0.745461 w_k1_DIS_s017
                                 + 0.830534 w_k1_FDX_s017
                                 + 0.69827 w_k1_IBM_s017
                                 + 0.794429 w_k1_INTC_s017
                                 + 0.75286 w_k1_KO_s017
                                 + 0.807977 w_k1_MAT_s017
                                 + 0.766976 w_k1_MSFT_s017
                                 + 0.861611 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.859574 w_k1_ADM_s017
                                 + 0.865021 w_k1_AXP_s017
                                 + 0.744587 w_k1_DIS_s017
                                 + 0.807081 w_k1_FDX_s017
                                 + 1.29567 w_k1_IBM_s017
                                 + 0.699666 w_k1_INTC_s017
                                 + 0.994395 w_k1_KO_s017
                                 + 1.31238 w_k1_MAT_s017
                                 + 0.782088 w_k1_MSFT_s017
                                 + 0.949919 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.878626 w_k1_ADM_s100
                                 + 1.22923 w_k1_AXP_s100
                                 + 0.980622 w_k1_DIS_s100
                                 + 1.05146 w_k1_FDX_s100
                                 + 0.926842 w_k1_IBM_s100
                                 + 1.15122 w_k1_INTC_s100
                                 + 1.07708 w_k1_KO_s100 + 0.8976 w_k1_MAT_s100
                                 + 1.1141 w_k1_MSFT_s100 + 1.10391 w_k1_PG_s100
                                 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.875839 w_k1_ADM_s100
                                 + 0.872246 w_k1_AXP_s100
                                 + 0.672832 w_k1_DIS_s100
                                 + 0.814447 w_k1_FDX_s100
                                 + 0.780751 w_k1_IBM_s100
                                 + 0.932294 w_k1_INTC_s100
                                 + 0.716853 w_k1_KO_s100
                                 + 0.573435 w_k1_MAT_s100
                                 + 0.816797 w_k1_MSFT_s100
                                 + 0.857965 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.58507 w_k1_ADM_s100
                                 + 1.40637 w_k1_AXP_s100
                                 + 1.32054 w_k1_DIS_s100
                                 + 1.16899 w_k1_FDX_s100
                                 + 1.24747 w_k1_IBM_s100
                                 + 1.21571 w_k1_INTC_s100 + 1.3837 w_k1_KO_s100
                                 + 1.03569 w_k1_MAT_s100
                                 + 1.22484 w_k1_MSFT_s100
                                 + 1.21343 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.672443 w_k1_ADM_s100
                                 + 0.717018 w_k1_AXP_s100
                                 + 0.597048 w_k1_DIS_s100
                                 + 0.761276 w_k1_FDX_s100
                                 + 0.588612 w_k1_IBM_s100
                                 + 0.980955 w_k1_INTC_s100
                                 + 0.636227 w_k1_KO_s100
                                 + 0.389807 w_k1_MAT_s100
                                 + 0.887811 w_k1_MSFT_s100
                                 + 0.830567 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.31483 w_k1_ADM_s100
                                 + 1.22324 w_k1_AXP_s100 + 1.5506 w_k1_DIS_s100
                                 + 1.15233 w_k1_FDX_s100
                                 + 1.40164 w_k1_IBM_s100
                                 + 1.22268 w_k1_INTC_s100
                                 + 1.21192 w_k1_KO_s100 + 1.33115 w_k1_MAT_s100
                                 + 1.25584 w_k1_MSFT_s100
                                 + 1.21842 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.890913 w_k1_ADM_s100
                                 + 0.888475 w_k1_AXP_s100
                                 + 0.77216 w_k1_DIS_s100
                                 + 0.856953 w_k1_FDX_s100
                                 + 0.881992 w_k1_IBM_s100
                                 + 0.793626 w_k1_INTC_s100
                                 + 0.936618 w_k1_KO_s100
                                 + 0.877864 w_k1_MAT_s100
                                 + 1.01733 w_k1_MSFT_s100 + 0.9714 w_k1_PG_s100
                                 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.25875 w_k1_ADM_s100
                                 + 1.47279 w_k1_AXP_s100
                                 + 1.05931 w_k1_DIS_s100
                                 + 1.23278 w_k1_FDX_s100
                                 + 0.693188 w_k1_IBM_s100
                                 + 1.22688 w_k1_INTC_s100
                                 + 0.951773 w_k1_KO_s100
                                 + 0.726404 w_k1_MAT_s100
                                 + 1.05796 w_k1_MSFT_s100
                                 + 1.07237 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.03316 w_k1_ADM_s100
                                 + 0.835554 w_k1_AXP_s100
                                 + 0.745665 w_k1_DIS_s100
                                 + 0.795304 w_k1_FDX_s100
                                 + 1.23221 w_k1_IBM_s100
                                 + 0.913077 w_k1_INTC_s100
                                 + 0.953698 w_k1_KO_s100
                                 + 0.809554 w_k1_MAT_s100
                                 + 0.759354 w_k1_MSFT_s100
                                 + 0.997385 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.29232 w_k1_ADM_s101
                                 + 1.25867 w_k1_AXP_s101
                                 + 1.06821 w_k1_DIS_s101
                                 + 1.05758 w_k1_FDX_s101
                                 + 1.07212 w_k1_IBM_s101
                                 + 1.02698 w_k1_INTC_s101
                                 + 1.08471 w_k1_KO_s101 + 1.10234 w_k1_MAT_s101
                                 + 1.13735 w_k1_MSFT_s101
                                 + 0.946676 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.49813 w_k1_ADM_s101
                                 + 1.56971 w_k1_AXP_s101
                                 + 1.75203 w_k1_DIS_s101
                                 + 1.34676 w_k1_FDX_s101
                                 + 1.41447 w_k1_IBM_s101
                                 + 1.26577 w_k1_INTC_s101
                                 + 1.31758 w_k1_KO_s101 + 1.30091 w_k1_MAT_s101
                                 + 1.12654 w_k1_MSFT_s101
                                 + 1.14636 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.994992 w_k1_ADM_s101
                                 + 0.964397 w_k1_AXP_s101
                                 + 0.90754 w_k1_DIS_s101
                                 + 0.973844 w_k1_FDX_s101
                                 + 0.710864 w_k1_IBM_s101
                                 + 0.968214 w_k1_INTC_s101
                                 + 0.847274 w_k1_KO_s101
                                 + 0.727732 w_k1_MAT_s101
                                 + 0.965232 w_k1_MSFT_s101
                                 + 0.900775 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.987659 w_k1_ADM_s101
                                 + 1.1987 w_k1_AXP_s101 + 1.03126 w_k1_DIS_s101
                                 + 0.91552 w_k1_FDX_s101
                                 + 1.60555 w_k1_IBM_s101
                                 + 1.23801 w_k1_INTC_s101
                                 + 1.20585 w_k1_KO_s101 + 1.39688 w_k1_MAT_s101
                                 + 1.06554 w_k1_MSFT_s101
                                 + 1.05707 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.15841 w_k1_ADM_s101
                                 + 0.847155 w_k1_AXP_s101
                                 + 1.34388 w_k1_DIS_s101
                                 + 1.04625 w_k1_FDX_s101
                                 + 0.941517 w_k1_IBM_s101
                                 + 1.14758 w_k1_INTC_s101 + 1.0886 w_k1_KO_s101
                                 + 0.782078 w_k1_MAT_s101
                                 + 1.32435 w_k1_MSFT_s101 + 0.9916 w_k1_PG_s101
                                 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.857972 w_k1_ADM_s101
                                 + 0.708552 w_k1_AXP_s101
                                 + 0.741397 w_k1_DIS_s101
                                 + 0.796133 w_k1_FDX_s101
                                 + 0.822177 w_k1_IBM_s101
                                 + 0.90141 w_k1_INTC_s101
                                 + 0.828594 w_k1_KO_s101
                                 + 0.739713 w_k1_MAT_s101
                                 + 0.843546 w_k1_MSFT_s101
                                 + 0.854315 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.17756 w_k1_ADM_s101
                                 + 1.08778 w_k1_AXP_s101
                                 + 1.17745 w_k1_DIS_s101
                                 + 1.01084 w_k1_FDX_s101
                                 + 1.20761 w_k1_IBM_s101
                                 + 1.05856 w_k1_INTC_s101
                                 + 1.12876 w_k1_KO_s101 + 1.05209 w_k1_MAT_s101
                                 + 1.18386 w_k1_MSFT_s101
                                 + 1.02781 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.3111 w_k1_ADM_s101
                                 + 1.09453 w_k1_AXP_s101
                                 + 1.19797 w_k1_DIS_s101
                                 + 1.18458 w_k1_FDX_s101 + 0.849 w_k1_IBM_s101
                                 + 1.20116 w_k1_INTC_s101
                                 + 0.997258 w_k1_KO_s101
                                 + 0.802615 w_k1_MAT_s101
                                 + 1.22571 w_k1_MSFT_s101
                                 + 0.994116 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.986294 w_k1_ADM_s102
                                 + 0.954787 w_k1_AXP_s102
                                 + 0.789737 w_k1_DIS_s102
                                 + 0.804391 w_k1_FDX_s102
                                 + 1.39123 w_k1_IBM_s102
                                 + 0.753553 w_k1_INTC_s102
                                 + 0.922272 w_k1_KO_s102
                                 + 1.14822 w_k1_MAT_s102
                                 + 0.747778 w_k1_MSFT_s102
                                 + 0.959112 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.1645 w_k1_ADM_s102
                                 + 0.772265 w_k1_AXP_s102
                                 + 1.03166 w_k1_DIS_s102
                                 + 0.922952 w_k1_FDX_s102
                                 + 0.92127 w_k1_IBM_s102
                                 + 1.02634 w_k1_INTC_s102
                                 + 1.07998 w_k1_KO_s102
                                 + 0.677909 w_k1_MAT_s102
                                 + 1.07266 w_k1_MSFT_s102 + 1.1116 w_k1_PG_s102
                                 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.63094 w_k1_ADM_s102
                                 + 1.2976 w_k1_AXP_s102 + 1.65307 w_k1_DIS_s102
                                 + 1.39596 w_k1_FDX_s102
                                 + 0.860909 w_k1_IBM_s102
                                 + 1.14859 w_k1_INTC_s102
                                 + 0.948596 w_k1_KO_s102
                                 + 0.791182 w_k1_MAT_s102
                                 + 1.37241 w_k1_MSFT_s102
                                 + 1.15368 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.1821 w_k1_ADM_s102
                                 + 0.6986 w_k1_AXP_s102
                                 + 0.559873 w_k1_DIS_s102
                                 + 0.77223 w_k1_FDX_s102
                                 + 0.666402 w_k1_IBM_s102
                                 + 0.948045 w_k1_INTC_s102
                                 + 0.659326 w_k1_KO_s102
                                 + 0.385783 w_k1_MAT_s102
                                 + 0.88184 w_k1_MSFT_s102
                                 + 0.811932 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.45533 w_k1_ADM_s102
                                 + 1.67582 w_k1_AXP_s102
                                 + 1.52055 w_k1_DIS_s102
                                 + 1.32545 w_k1_FDX_s102
                                 + 1.42259 w_k1_IBM_s102
                                 + 1.48466 w_k1_INTC_s102
                                 + 1.26856 w_k1_KO_s102 + 1.62731 w_k1_MAT_s102
                                 + 1.35615 w_k1_MSFT_s102
                                 + 1.21182 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.18499 w_k1_ADM_s102
                                 + 0.801738 w_k1_AXP_s102
                                 + 0.738366 w_k1_DIS_s102
                                 + 0.885645 w_k1_FDX_s102
                                 + 0.843532 w_k1_IBM_s102
                                 + 0.870667 w_k1_INTC_s102
                                 + 0.736841 w_k1_KO_s102
                                 + 0.94004 w_k1_MAT_s102
                                 + 0.914148 w_k1_MSFT_s102
                                 + 0.973384 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.982074 w_k1_ADM_s102
                                 + 1.0599 w_k1_AXP_s102 + 1.12192 w_k1_DIS_s102
                                 + 0.948349 w_k1_FDX_s102 + 1.128 w_k1_IBM_s102
                                 + 1.22425 w_k1_INTC_s102
                                 + 1.00433 w_k1_KO_s102 + 1.0447 w_k1_MAT_s102
                                 + 1.099 w_k1_MSFT_s102 + 1.02657 w_k1_PG_s102
                                 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.14175 w_k1_ADM_s102
                                 + 1.18046 w_k1_AXP_s102
                                 + 1.02637 w_k1_DIS_s102
                                 + 1.08736 w_k1_FDX_s102
                                 + 0.814469 w_k1_IBM_s102
                                 + 1.11597 w_k1_INTC_s102
                                 + 0.829563 w_k1_KO_s102
                                 + 1.15505 w_k1_MAT_s102
                                 + 1.12914 w_k1_MSFT_s102 + 1.1243 w_k1_PG_s102
                                 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.694088 w_k1_ADM_s103
                                 + 0.699174 w_k1_AXP_s103
                                 + 0.583235 w_k1_DIS_s103
                                 + 0.75084 w_k1_FDX_s103
                                 + 0.724165 w_k1_IBM_s103
                                 + 0.828486 w_k1_INTC_s103
                                 + 0.703603 w_k1_KO_s103
                                 + 0.706895 w_k1_MAT_s103
                                 + 0.832109 w_k1_MSFT_s103
                                 + 0.735279 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.16548 w_k1_ADM_s103
                                 + 1.36419 w_k1_AXP_s103
                                 + 1.18505 w_k1_DIS_s103
                                 + 1.27902 w_k1_FDX_s103
                                 + 0.739363 w_k1_IBM_s103
                                 + 1.04347 w_k1_INTC_s103
                                 + 0.898977 w_k1_KO_s103
                                 + 1.21686 w_k1_MAT_s103
                                 + 1.30319 w_k1_MSFT_s103 + 1.0493 w_k1_PG_s103
                                 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.942706 w_k1_ADM_s103
                                 + 1.15904 w_k1_AXP_s103
                                 + 1.00198 w_k1_DIS_s103
                                 + 0.914176 w_k1_FDX_s103
                                 + 1.49637 w_k1_IBM_s103
                                 + 0.962045 w_k1_INTC_s103
                                 + 1.0874 w_k1_KO_s103 + 1.27273 w_k1_MAT_s103
                                 + 0.932933 w_k1_MSFT_s103
                                 + 0.958021 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.900234 w_k1_ADM_s103
                                 + 0.815303 w_k1_AXP_s103
                                 + 0.929722 w_k1_DIS_s103
                                 + 0.978748 w_k1_FDX_s103
                                 + 0.774536 w_k1_IBM_s103
                                 + 0.996334 w_k1_INTC_s103
                                 + 0.894949 w_k1_KO_s103
                                 + 0.784947 w_k1_MAT_s103
                                 + 0.981559 w_k1_MSFT_s103
                                 + 0.98086 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.32089 w_k1_ADM_s103
                                 + 1.4138 w_k1_AXP_s103 + 1.38911 w_k1_DIS_s103
                                 + 1.25159 w_k1_FDX_s103
                                 + 1.42615 w_k1_IBM_s103
                                 + 0.929791 w_k1_INTC_s103
                                 + 1.27304 w_k1_KO_s103 + 1.8091 w_k1_MAT_s103
                                 + 1.25924 w_k1_MSFT_s103
                                 + 1.11326 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.01818 w_k1_ADM_s103
                                 + 0.954974 w_k1_AXP_s103
                                 + 1.16399 w_k1_DIS_s103
                                 + 1.03887 w_k1_FDX_s103
                                 + 1.11237 w_k1_IBM_s103
                                 + 1.00834 w_k1_INTC_s103
                                 + 1.15693 w_k1_KO_s103 + 1.17207 w_k1_MAT_s103
                                 + 1.20416 w_k1_MSFT_s103
                                 + 1.01907 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.799872 w_k1_ADM_s103
                                 + 0.882033 w_k1_AXP_s103
                                 + 0.743564 w_k1_DIS_s103
                                 + 0.861032 w_k1_FDX_s103
                                 + 0.965141 w_k1_IBM_s103
                                 + 0.741973 w_k1_INTC_s103
                                 + 0.85428 w_k1_KO_s103 + 1.03887 w_k1_MAT_s103
                                 + 0.844368 w_k1_MSFT_s103
                                 + 0.926594 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.04371 w_k1_ADM_s103
                                 + 1.16145 w_k1_AXP_s103
                                 + 0.813627 w_k1_DIS_s103
                                 + 0.939719 w_k1_FDX_s103
                                 + 1.11989 w_k1_IBM_s103
                                 + 1.06727 w_k1_INTC_s103
                                 + 0.971779 w_k1_KO_s103
                                 + 1.55644 w_k1_MAT_s103
                                 + 0.951665 w_k1_MSFT_s103
                                 + 1.05001 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.00072 w_k1_ADM_s104
                                 + 0.748196 w_k1_AXP_s104
                                 + 0.740825 w_k1_DIS_s104
                                 + 0.87795 w_k1_FDX_s104
                                 + 0.568918 w_k1_IBM_s104
                                 + 0.838541 w_k1_INTC_s104
                                 + 0.779525 w_k1_KO_s104
                                 + 0.489252 w_k1_MAT_s104
                                 + 0.876166 w_k1_MSFT_s104
                                 + 0.944476 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.40226 w_k1_ADM_s104
                                 + 0.925119 w_k1_AXP_s104
                                 + 1.15262 w_k1_DIS_s104
                                 + 1.16168 w_k1_FDX_s104
                                 + 0.729006 w_k1_IBM_s104
                                 + 0.990243 w_k1_INTC_s104
                                 + 0.895738 w_k1_KO_s104
                                 + 0.748324 w_k1_MAT_s104
                                 + 1.21187 w_k1_MSFT_s104
                                 + 1.09982 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.88009 w_k1_ADM_s104
                                 + 1.20597 w_k1_AXP_s104
                                 + 1.01791 w_k1_DIS_s104
                                 + 1.13137 w_k1_FDX_s104
                                 + 0.853746 w_k1_IBM_s104
                                 + 1.15176 w_k1_INTC_s104
                                 + 0.945431 w_k1_KO_s104
                                 + 0.939475 w_k1_MAT_s104
                                 + 1.15106 w_k1_MSFT_s104
                                 + 1.14403 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.912439 w_k1_ADM_s104
                                 + 0.763335 w_k1_AXP_s104
                                 + 0.831579 w_k1_DIS_s104
                                 + 0.884881 w_k1_FDX_s104
                                 + 0.751179 w_k1_IBM_s104
                                 + 0.993594 w_k1_INTC_s104
                                 + 0.891767 w_k1_KO_s104
                                 + 0.565605 w_k1_MAT_s104
                                 + 1.03306 w_k1_MSFT_s104
                                 + 0.925231 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.969518 w_k1_ADM_s104
                                 + 1.28043 w_k1_AXP_s104
                                 + 1.45209 w_k1_DIS_s104
                                 + 1.30041 w_k1_FDX_s104
                                 + 1.04275 w_k1_IBM_s104
                                 + 1.37468 w_k1_INTC_s104
                                 + 1.37288 w_k1_KO_s104
                                 + 0.963436 w_k1_MAT_s104
                                 + 1.3121 w_k1_MSFT_s104 + 1.3291 w_k1_PG_s104
                                 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.862217 w_k1_ADM_s104
                                 + 0.87932 w_k1_AXP_s104
                                 + 0.747981 w_k1_DIS_s104
                                 + 0.775362 w_k1_FDX_s104
                                 + 0.876055 w_k1_IBM_s104
                                 + 0.822633 w_k1_INTC_s104
                                 + 0.778901 w_k1_KO_s104
                                 + 0.82963 w_k1_MAT_s104
                                 + 0.902803 w_k1_MSFT_s104
                                 + 1.0496 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.22852 w_k1_ADM_s104
                                 + 0.915755 w_k1_AXP_s104
                                 + 0.887246 w_k1_DIS_s104
                                 + 0.946044 w_k1_FDX_s104
                                 + 1.04073 w_k1_IBM_s104
                                 + 0.867509 w_k1_INTC_s104
                                 + 0.89523 w_k1_KO_s104
                                 + 0.683451 w_k1_MAT_s104
                                 + 1.01445 w_k1_MSFT_s104 + 1.1079 w_k1_PG_s104
                                 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.07714 w_k1_ADM_s104
                                 + 1.19914 w_k1_AXP_s104
                                 + 1.19336 w_k1_DIS_s104
                                 + 1.03714 w_k1_FDX_s104
                                 + 1.46644 w_k1_IBM_s104
                                 + 1.09198 w_k1_INTC_s104
                                 + 1.04086 w_k1_KO_s104 + 1.10717 w_k1_MAT_s104
                                 + 1.01166 w_k1_MSFT_s104
                                 + 1.13704 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.06566 w_k1_ADM_s105
                                 + 1.1024 w_k1_AXP_s105 + 1.07105 w_k1_DIS_s105
                                 + 1.06512 w_k1_FDX_s105 + 0.8604 w_k1_IBM_s105
                                 + 0.867535 w_k1_INTC_s105
                                 + 0.819403 w_k1_KO_s105
                                 + 1.10972 w_k1_MAT_s105
                                 + 0.899804 w_k1_MSFT_s105
                                 + 1.02717 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.01388 w_k1_ADM_s105
                                 + 0.974914 w_k1_AXP_s105
                                 + 0.749301 w_k1_DIS_s105
                                 + 0.877057 w_k1_FDX_s105
                                 + 0.994819 w_k1_IBM_s105
                                 + 0.846402 w_k1_INTC_s105
                                 + 0.820761 w_k1_KO_s105
                                 + 1.12591 w_k1_MAT_s105
                                 + 0.802924 w_k1_MSFT_s105
                                 + 0.91725 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.119 w_k1_ADM_s105
                                 + 1.24975 w_k1_AXP_s105 + 1.2562 w_k1_DIS_s105
                                 + 1.23316 w_k1_FDX_s105
                                 + 1.37594 w_k1_IBM_s105
                                 + 0.969214 w_k1_INTC_s105
                                 + 1.12617 w_k1_KO_s105 + 1.52433 w_k1_MAT_s105
                                 + 1.11408 w_k1_MSFT_s105
                                 + 1.11826 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.755725 w_k1_ADM_s105
                                 + 0.624202 w_k1_AXP_s105
                                 + 0.741264 w_k1_DIS_s105
                                 + 0.78959 w_k1_FDX_s105
                                 + 0.749963 w_k1_IBM_s105
                                 + 0.767609 w_k1_INTC_s105
                                 + 0.761631 w_k1_KO_s105 + 0.6028 w_k1_MAT_s105
                                 + 0.81582 w_k1_MSFT_s105
                                 + 0.951216 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.973151 w_k1_ADM_s105
                                 + 0.897917 w_k1_AXP_s105
                                 + 0.628106 w_k1_DIS_s105
                                 + 0.8644 w_k1_FDX_s105
                                 + 0.660809 w_k1_IBM_s105
                                 + 0.816538 w_k1_INTC_s105
                                 + 0.594727 w_k1_KO_s105
                                 + 0.434662 w_k1_MAT_s105
                                 + 0.678233 w_k1_MSFT_s105
                                 + 0.798754 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.25999 w_k1_ADM_s105
                                 + 1.34334 w_k1_AXP_s105
                                 + 1.38496 w_k1_DIS_s105
                                 + 1.40036 w_k1_FDX_s105
                                 + 0.829801 w_k1_IBM_s105
                                 + 1.20699 w_k1_INTC_s105
                                 + 1.04194 w_k1_KO_s105 + 1.22902 w_k1_MAT_s105
                                 + 1.30014 w_k1_MSFT_s105
                                 + 1.26393 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.791604 w_k1_ADM_s105
                                 + 0.874793 w_k1_AXP_s105
                                 + 0.747218 w_k1_DIS_s105
                                 + 0.717207 w_k1_FDX_s105
                                 + 1.2245 w_k1_IBM_s105
                                 + 0.906588 w_k1_INTC_s105
                                 + 0.868547 w_k1_KO_s105
                                 + 1.07415 w_k1_MAT_s105
                                 + 0.785196 w_k1_MSFT_s105
                                 + 0.992723 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.996803 w_k1_ADM_s105
                                 + 0.956431 w_k1_AXP_s105
                                 + 1.69169 w_k1_DIS_s105
                                 + 1.16266 w_k1_FDX_s105
                                 + 1.25202 w_k1_IBM_s105
                                 + 1.07257 w_k1_INTC_s105
                                 + 1.29977 w_k1_KO_s105 + 1.1099 w_k1_MAT_s105
                                 + 1.26339 w_k1_MSFT_s105
                                 + 1.15932 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.851976 w_k1_ADM_s106
                                 + 0.990194 w_k1_AXP_s106
                                 + 0.834292 w_k1_DIS_s106
                                 + 1.02952 w_k1_FDX_s106
                                 + 0.762748 w_k1_IBM_s106
                                 + 0.961665 w_k1_INTC_s106
                                 + 0.828237 w_k1_KO_s106
                                 + 0.612581 w_k1_MAT_s106
                                 + 0.894341 w_k1_MSFT_s106
                                 + 0.986727 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.854717 w_k1_ADM_s106
                                 + 0.701802 w_k1_AXP_s106
                                 + 1.10892 w_k1_DIS_s106
                                 + 0.947433 w_k1_FDX_s106
                                 + 0.901527 w_k1_IBM_s106
                                 + 1.08075 w_k1_INTC_s106
                                 + 1.01152 w_k1_KO_s106
                                 + 0.611421 w_k1_MAT_s106
                                 + 1.01279 w_k1_MSFT_s106
                                 + 0.96626 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.901665 w_k1_ADM_s106
                                 + 0.882039 w_k1_AXP_s106
                                 + 0.733819 w_k1_DIS_s106
                                 + 0.812834 w_k1_FDX_s106
                                 + 1.10996 w_k1_IBM_s106
                                 + 0.829235 w_k1_INTC_s106
                                 + 0.898329 w_k1_KO_s106
                                 + 1.02556 w_k1_MAT_s106
                                 + 0.715923 w_k1_MSFT_s106
                                 + 0.835506 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.882032 w_k1_ADM_s106
                                 + 0.746017 w_k1_AXP_s106
                                 + 0.741749 w_k1_DIS_s106
                                 + 0.781421 w_k1_FDX_s106
                                 + 0.855667 w_k1_IBM_s106
                                 + 0.896544 w_k1_INTC_s106
                                 + 0.76776 w_k1_KO_s106
                                 + 0.637728 w_k1_MAT_s106
                                 + 0.792942 w_k1_MSFT_s106
                                 + 0.849227 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.736694 w_k1_ADM_s106
                                 + 0.614805 w_k1_AXP_s106
                                 + 0.532118 w_k1_DIS_s106
                                 + 0.771622 w_k1_FDX_s106
                                 + 0.644224 w_k1_IBM_s106
                                 + 0.868439 w_k1_INTC_s106
                                 + 0.648043 w_k1_KO_s106
                                 + 0.45381 w_k1_MAT_s106
                                 + 0.690105 w_k1_MSFT_s106
                                 + 0.70683 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.35739 w_k1_ADM_s106
                                 + 1.49557 w_k1_AXP_s106
                                 + 1.74417 w_k1_DIS_s106
                                 + 1.38433 w_k1_FDX_s106
                                 + 1.51125 w_k1_IBM_s106
                                 + 1.20933 w_k1_INTC_s106
                                 + 1.42484 w_k1_KO_s106 + 1.10417 w_k1_MAT_s106
                                 + 1.18466 w_k1_MSFT_s106
                                 + 1.17962 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.00506 w_k1_ADM_s106
                                 + 1.37248 w_k1_AXP_s106
                                 + 1.22238 w_k1_DIS_s106
                                 + 1.03337 w_k1_FDX_s106
                                 + 1.30808 w_k1_IBM_s106
                                 + 0.942866 w_k1_INTC_s106
                                 + 1.10514 w_k1_KO_s106 + 1.11831 w_k1_MAT_s106
                                 + 0.913889 w_k1_MSFT_s106
                                 + 1.01686 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.25113 w_k1_ADM_s106
                                 + 1.31805 w_k1_AXP_s106
                                 + 1.18863 w_k1_DIS_s106
                                 + 1.28313 w_k1_FDX_s106
                                 + 0.829835 w_k1_IBM_s106
                                 + 1.08642 w_k1_INTC_s106
                                 + 0.960754 w_k1_KO_s106
                                 + 0.874693 w_k1_MAT_s106
                                 + 1.07508 w_k1_MSFT_s106
                                 + 1.00506 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.12016 w_k1_ADM_s107
                                 + 1.18986 w_k1_AXP_s107
                                 + 0.791174 w_k1_DIS_s107
                                 + 0.953226 w_k1_FDX_s107
                                 + 0.960214 w_k1_IBM_s107
                                 + 0.897728 w_k1_INTC_s107
                                 + 0.912612 w_k1_KO_s107
                                 + 1.14237 w_k1_MAT_s107
                                 + 0.778815 w_k1_MSFT_s107
                                 + 0.915416 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.959475 w_k1_ADM_s107
                                 + 0.700457 w_k1_AXP_s107
                                 + 0.893198 w_k1_DIS_s107
                                 + 0.814913 w_k1_FDX_s107
                                 + 0.920521 w_k1_IBM_s107
                                 + 0.921434 w_k1_INTC_s107
                                 + 0.945196 w_k1_KO_s107
                                 + 0.692908 w_k1_MAT_s107
                                 + 0.812716 w_k1_MSFT_s107
                                 + 0.844348 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.29182 w_k1_ADM_s107
                                 + 1.28192 w_k1_AXP_s107
                                 + 1.16524 w_k1_DIS_s107 + 1.0705 w_k1_FDX_s107
                                 + 1.40712 w_k1_IBM_s107
                                 + 1.03428 w_k1_INTC_s107
                                 + 1.32562 w_k1_KO_s107 + 1.45014 w_k1_MAT_s107
                                 + 0.932458 w_k1_MSFT_s107
                                 + 1.09607 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.01359 w_k1_ADM_s107
                                 + 0.79714 w_k1_AXP_s107
                                 + 1.06751 w_k1_DIS_s107
                                 + 0.95242 w_k1_FDX_s107
                                 + 0.924947 w_k1_IBM_s107
                                 + 1.07828 w_k1_INTC_s107
                                 + 1.09323 w_k1_KO_s107
                                 + 0.771464 w_k1_MAT_s107
                                 + 1.01845 w_k1_MSFT_s107
                                 + 0.991808 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.927812 w_k1_ADM_s107
                                 + 0.706418 w_k1_AXP_s107
                                 + 0.559704 w_k1_DIS_s107
                                 + 0.710834 w_k1_FDX_s107
                                 + 0.967132 w_k1_IBM_s107
                                 + 0.637403 w_k1_INTC_s107
                                 + 0.772736 w_k1_KO_s107 + 0.9007 w_k1_MAT_s107
                                 + 0.699888 w_k1_MSFT_s107
                                 + 0.740379 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.928758 w_k1_ADM_s107
                                 + 0.79816 w_k1_AXP_s107
                                 + 0.668719 w_k1_DIS_s107
                                 + 0.94435 w_k1_FDX_s107
                                 + 0.577186 w_k1_IBM_s107
                                 + 0.87216 w_k1_INTC_s107
                                 + 0.742618 w_k1_KO_s107
                                 + 0.600507 w_k1_MAT_s107
                                 + 0.81878 w_k1_MSFT_s107
                                 + 0.88508 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.14484 w_k1_ADM_s107
                                 + 0.928395 w_k1_AXP_s107
                                 + 0.970182 w_k1_DIS_s107
                                 + 1.09947 w_k1_FDX_s107
                                 + 0.684333 w_k1_IBM_s107
                                 + 0.821644 w_k1_INTC_s107
                                 + 0.780657 w_k1_KO_s107
                                 + 0.62736 w_k1_MAT_s107
                                 + 0.968765 w_k1_MSFT_s107
                                 + 0.979986 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.31503 w_k1_ADM_s107
                                 + 1.4214 w_k1_AXP_s107 + 1.30767 w_k1_DIS_s107
                                 + 1.3782 w_k1_FDX_s107
                                 + 0.841503 w_k1_IBM_s107
                                 + 1.21972 w_k1_INTC_s107
                                 + 1.27496 w_k1_KO_s107 + 1.21768 w_k1_MAT_s107
                                 + 1.21233 w_k1_MSFT_s107
                                 + 1.18618 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.26953 w_k1_ADM_s110
                                 + 1.40898 w_k1_AXP_s110
                                 + 1.40922 w_k1_DIS_s110
                                 + 1.35121 w_k1_FDX_s110
                                 + 0.954094 w_k1_IBM_s110
                                 + 1.19616 w_k1_INTC_s110
                                 + 1.15008 w_k1_KO_s110 + 1.31691 w_k1_MAT_s110
                                 + 1.02595 w_k1_MSFT_s110
                                 + 1.19882 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.21571 w_k1_ADM_s110
                                 + 1.06665 w_k1_AXP_s110
                                 + 1.19995 w_k1_DIS_s110
                                 + 1.20272 w_k1_FDX_s110
                                 + 1.01866 w_k1_IBM_s110
                                 + 1.29667 w_k1_INTC_s110 + 1.1618 w_k1_KO_s110
                                 + 0.980796 w_k1_MAT_s110
                                 + 1.05111 w_k1_MSFT_s110
                                 + 1.02114 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.971003 w_k1_ADM_s110
                                 + 0.942268 w_k1_AXP_s110
                                 + 0.900051 w_k1_DIS_s110
                                 + 0.958078 w_k1_FDX_s110
                                 + 0.720857 w_k1_IBM_s110
                                 + 1.12596 w_k1_INTC_s110
                                 + 0.878504 w_k1_KO_s110
                                 + 0.878534 w_k1_MAT_s110
                                 + 0.965386 w_k1_MSFT_s110
                                 + 0.95918 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.843773 w_k1_ADM_s110
                                 + 0.805483 w_k1_AXP_s110
                                 + 0.673119 w_k1_DIS_s110
                                 + 0.785087 w_k1_FDX_s110
                                 + 0.806006 w_k1_IBM_s110
                                 + 0.952755 w_k1_INTC_s110
                                 + 0.781983 w_k1_KO_s110
                                 + 0.76553 w_k1_MAT_s110
                                 + 0.731492 w_k1_MSFT_s110
                                 + 0.797912 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.02309 w_k1_ADM_s110
                                 + 1.06063 w_k1_AXP_s110
                                 + 1.05476 w_k1_DIS_s110
                                 + 1.01186 w_k1_FDX_s110
                                 + 1.40026 w_k1_IBM_s110
                                 + 1.1933 w_k1_INTC_s110 + 1.2269 w_k1_KO_s110
                                 + 1.28069 w_k1_MAT_s110
                                 + 0.914208 w_k1_MSFT_s110
                                 + 1.00159 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.672044 w_k1_ADM_s110
                                 + 0.710517 w_k1_AXP_s110
                                 + 0.591326 w_k1_DIS_s110
                                 + 0.72703 w_k1_FDX_s110
                                 + 0.983823 w_k1_IBM_s110
                                 + 0.976667 w_k1_INTC_s110
                                 + 0.864029 w_k1_KO_s110
                                 + 1.14823 w_k1_MAT_s110
                                 + 0.720415 w_k1_MSFT_s110
                                 + 0.835399 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.1423 w_k1_ADM_s110
                                 + 1.32903 w_k1_AXP_s110
                                 + 1.59709 w_k1_DIS_s110 + 1.2563 w_k1_FDX_s110
                                 + 1.66248 w_k1_IBM_s110
                                 + 1.14758 w_k1_INTC_s110
                                 + 1.46954 w_k1_KO_s110 + 1.87295 w_k1_MAT_s110
                                 + 1.10189 w_k1_MSFT_s110
                                 + 1.09905 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.906049 w_k1_ADM_s110
                                 + 0.834197 w_k1_AXP_s110
                                 + 0.884775 w_k1_DIS_s110
                                 + 0.943124 w_k1_FDX_s110
                                 + 1.05348 w_k1_IBM_s110
                                 + 0.998087 w_k1_INTC_s110
                                 + 0.910354 w_k1_KO_s110
                                 + 1.13587 w_k1_MAT_s110
                                 + 0.855448 w_k1_MSFT_s110
                                 + 0.908833 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.04109 w_k1_ADM_s111
                                 + 1.15221 w_k1_AXP_s111
                                 + 0.893426 w_k1_DIS_s111
                                 + 0.941566 w_k1_FDX_s111
                                 + 0.949134 w_k1_IBM_s111
                                 + 0.822275 w_k1_INTC_s111
                                 + 0.918979 w_k1_KO_s111
                                 + 1.20013 w_k1_MAT_s111
                                 + 0.735691 w_k1_MSFT_s111
                                 + 0.982196 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.0507 w_k1_ADM_s111
                                 + 1.01855 w_k1_AXP_s111
                                 + 1.25283 w_k1_DIS_s111
                                 + 1.04501 w_k1_FDX_s111
                                 + 1.19422 w_k1_IBM_s111
                                 + 1.0368 w_k1_INTC_s111 + 1.24585 w_k1_KO_s111
                                 + 1.20155 w_k1_MAT_s111
                                 + 1.09046 w_k1_MSFT_s111
                                 + 1.15997 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.30693 w_k1_ADM_s111
                                 + 1.49479 w_k1_AXP_s111
                                 + 1.36812 w_k1_DIS_s111
                                 + 1.20506 w_k1_FDX_s111 + 1.444 w_k1_IBM_s111
                                 + 1.28356 w_k1_INTC_s111
                                 + 1.27858 w_k1_KO_s111 + 1.74438 w_k1_MAT_s111
                                 + 1.1955 w_k1_MSFT_s111 + 1.13429 w_k1_PG_s111
                                 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.08193 w_k1_ADM_s111
                                 + 1.0127 w_k1_AXP_s111 + 1.00046 w_k1_DIS_s111
                                 + 1.07379 w_k1_FDX_s111
                                 + 0.769116 w_k1_IBM_s111
                                 + 1.06892 w_k1_INTC_s111
                                 + 0.972799 w_k1_KO_s111
                                 + 0.927806 w_k1_MAT_s111
                                 + 1.00062 w_k1_MSFT_s111
                                 + 0.955085 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.18691 w_k1_ADM_s111
                                 + 1.18971 w_k1_AXP_s111
                                 + 1.32273 w_k1_DIS_s111
                                 + 1.23001 w_k1_FDX_s111
                                 + 1.07601 w_k1_IBM_s111
                                 + 0.966313 w_k1_INTC_s111
                                 + 1.11847 w_k1_KO_s111 + 1.23774 w_k1_MAT_s111
                                 + 1.04186 w_k1_MSFT_s111
                                 + 1.09165 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.840311 w_k1_ADM_s111
                                 + 0.686939 w_k1_AXP_s111
                                 + 0.811867 w_k1_DIS_s111
                                 + 0.862347 w_k1_FDX_s111
                                 + 0.713298 w_k1_IBM_s111
                                 + 0.733479 w_k1_INTC_s111
                                 + 0.699831 w_k1_KO_s111
                                 + 0.601423 w_k1_MAT_s111
                                 + 0.819758 w_k1_MSFT_s111
                                 + 0.877364 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.05193 w_k1_ADM_s111
                                 + 1.142 w_k1_AXP_s111 + 1.54693 w_k1_DIS_s111
                                 + 1.44 w_k1_FDX_s111 + 0.919282 w_k1_IBM_s111
                                 + 1.19149 w_k1_INTC_s111
                                 + 1.08992 w_k1_KO_s111 + 1.20212 w_k1_MAT_s111
                                 + 1.20936 w_k1_MSFT_s111
                                 + 1.12198 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.922609 w_k1_ADM_s111
                                 + 0.999569 w_k1_AXP_s111
                                 + 0.855767 w_k1_DIS_s111
                                 + 0.815963 w_k1_FDX_s111
                                 + 1.30584 w_k1_IBM_s111
                                 + 1.05204 w_k1_INTC_s111
                                 + 1.08697 w_k1_KO_s111 + 1.6224 w_k1_MAT_s111
                                 + 0.868772 w_k1_MSFT_s111
                                 + 0.984135 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.05473 w_k1_ADM_s112
                                 + 0.707971 w_k1_AXP_s112
                                 + 1.12434 w_k1_DIS_s112
                                 + 0.875627 w_k1_FDX_s112
                                 + 0.845661 w_k1_IBM_s112
                                 + 0.884238 w_k1_INTC_s112
                                 + 1.02296 w_k1_KO_s112
                                 + 0.617484 w_k1_MAT_s112
                                 + 1.03706 w_k1_MSFT_s112
                                 + 0.917897 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.30292 w_k1_ADM_s112
                                 + 1.20747 w_k1_AXP_s112
                                 + 1.30016 w_k1_DIS_s112
                                 + 1.19782 w_k1_FDX_s112
                                 + 0.914088 w_k1_IBM_s112
                                 + 0.964428 w_k1_INTC_s112
                                 + 1.09258 w_k1_KO_s112 + 1.56867 w_k1_MAT_s112
                                 + 1.19447 w_k1_MSFT_s112
                                 + 1.13091 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.03051 w_k1_ADM_s112
                                 + 0.880812 w_k1_AXP_s112
                                 + 0.980495 w_k1_DIS_s112
                                 + 0.958719 w_k1_FDX_s112
                                 + 1.05603 w_k1_IBM_s112
                                 + 0.833775 w_k1_INTC_s112
                                 + 1.04398 w_k1_KO_s112 + 1.00129 w_k1_MAT_s112
                                 + 1.05121 w_k1_MSFT_s112
                                 + 0.941163 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.682088 w_k1_ADM_s112
                                 + 0.58344 w_k1_AXP_s112
                                 + 0.816739 w_k1_DIS_s112
                                 + 0.82561 w_k1_FDX_s112
                                 + 0.642513 w_k1_IBM_s112
                                 + 0.768485 w_k1_INTC_s112
                                 + 0.860391 w_k1_KO_s112
                                 + 0.757027 w_k1_MAT_s112
                                 + 0.907637 w_k1_MSFT_s112
                                 + 0.871882 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.860849 w_k1_ADM_s112
                                 + 0.709066 w_k1_AXP_s112
                                 + 0.713227 w_k1_DIS_s112
                                 + 0.758128 w_k1_FDX_s112
                                 + 0.922011 w_k1_IBM_s112
                                 + 0.799594 w_k1_INTC_s112
                                 + 0.868011 w_k1_KO_s112
                                 + 1.09146 w_k1_MAT_s112
                                 + 0.758728 w_k1_MSFT_s112
                                 + 0.81504 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.35818 w_k1_ADM_s112
                                 + 1.17716 w_k1_AXP_s112
                                 + 1.42681 w_k1_DIS_s112
                                 + 1.12786 w_k1_FDX_s112
                                 + 1.44684 w_k1_IBM_s112
                                 + 1.05983 w_k1_INTC_s112
                                 + 1.36149 w_k1_KO_s112 + 1.36572 w_k1_MAT_s112
                                 + 1.13539 w_k1_MSFT_s112
                                 + 1.04605 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.895594 w_k1_ADM_s112
                                 + 0.717734 w_k1_AXP_s112
                                 + 0.85832 w_k1_DIS_s112
                                 + 0.774385 w_k1_FDX_s112
                                 + 1.29722 w_k1_IBM_s112
                                 + 0.768909 w_k1_INTC_s112
                                 + 1.01064 w_k1_KO_s112
                                 + 0.920007 w_k1_MAT_s112
                                 + 0.824671 w_k1_MSFT_s112
                                 + 0.953677 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.04006 w_k1_ADM_s112
                                 + 0.805075 w_k1_AXP_s112
                                 + 0.886297 w_k1_DIS_s112
                                 + 0.929719 w_k1_FDX_s112
                                 + 0.692429 w_k1_IBM_s112
                                 + 0.75086 w_k1_INTC_s112
                                 + 0.753784 w_k1_KO_s112
                                 + 0.553087 w_k1_MAT_s112
                                 + 0.85259 w_k1_MSFT_s112
                                 + 0.857626 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.858045 w_k1_ADM_s113
                                 + 0.737733 w_k1_AXP_s113
                                 + 0.895947 w_k1_DIS_s113
                                 + 0.907142 w_k1_FDX_s113
                                 + 1.04975 w_k1_IBM_s113
                                 + 0.980162 w_k1_INTC_s113
                                 + 0.958172 w_k1_KO_s113
                                 + 0.852933 w_k1_MAT_s113
                                 + 1.1686 w_k1_MSFT_s113
                                 + 0.897058 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.02274 w_k1_ADM_s113
                                 + 1.03802 w_k1_AXP_s113 + 1.0347 w_k1_DIS_s113
                                 + 1.00009 w_k1_FDX_s113
                                 + 1.09068 w_k1_IBM_s113
                                 + 1.15385 w_k1_INTC_s113 + 1.0361 w_k1_KO_s113
                                 + 1.12665 w_k1_MAT_s113
                                 + 1.0706 w_k1_MSFT_s113
                                 + 0.992301 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.29727 w_k1_ADM_s113
                                 + 1.29367 w_k1_AXP_s113
                                 + 1.51639 w_k1_DIS_s113
                                 + 1.24412 w_k1_FDX_s113
                                 + 1.48886 w_k1_IBM_s113
                                 + 1.29715 w_k1_INTC_s113
                                 + 1.33652 w_k1_KO_s113 + 1.30511 w_k1_MAT_s113
                                 + 1.26586 w_k1_MSFT_s113
                                 + 1.37489 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.608662 w_k1_ADM_s113
                                 + 0.641291 w_k1_AXP_s113
                                 + 0.619834 w_k1_DIS_s113
                                 + 0.727994 w_k1_FDX_s113
                                 + 0.77215 w_k1_IBM_s113
                                 + 0.90012 w_k1_INTC_s113
                                 + 0.74113 w_k1_KO_s113
                                 + 0.636961 w_k1_MAT_s113
                                 + 0.893777 w_k1_MSFT_s113
                                 + 0.793698 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.996016 w_k1_ADM_s113
                                 + 1.54733 w_k1_AXP_s113 + 1.5036 w_k1_DIS_s113
                                 + 1.41428 w_k1_FDX_s113
                                 + 1.00605 w_k1_IBM_s113
                                 + 1.4183 w_k1_INTC_s113 + 1.21044 w_k1_KO_s113
                                 + 1.40288 w_k1_MAT_s113
                                 + 1.56004 w_k1_MSFT_s113
                                 + 1.28028 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.696772 w_k1_ADM_s113
                                 + 0.899871 w_k1_AXP_s113
                                 + 0.772728 w_k1_DIS_s113
                                 + 0.829352 w_k1_FDX_s113
                                 + 0.964874 w_k1_IBM_s113
                                 + 1.00538 w_k1_INTC_s113
                                 + 0.856805 w_k1_KO_s113
                                 + 1.07989 w_k1_MAT_s113
                                 + 0.924152 w_k1_MSFT_s113
                                 + 0.948459 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.838884 w_k1_ADM_s113
                                 + 1.14873 w_k1_AXP_s113
                                 + 0.905343 w_k1_DIS_s113
                                 + 0.949424 w_k1_FDX_s113
                                 + 1.62575 w_k1_IBM_s113
                                 + 1.32222 w_k1_INTC_s113
                                 + 1.20532 w_k1_KO_s113 + 1.53124 w_k1_MAT_s113
                                 + 1.15943 w_k1_MSFT_s113
                                 + 1.05717 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.02536 w_k1_ADM_s113
                                 + 0.928955 w_k1_AXP_s113
                                 + 0.957928 w_k1_DIS_s113
                                 + 1.05015 w_k1_FDX_s113
                                 + 0.739161 w_k1_IBM_s113
                                 + 1.28188 w_k1_INTC_s113
                                 + 0.862512 w_k1_KO_s113
                                 + 0.653134 w_k1_MAT_s113
                                 + 1.09899 w_k1_MSFT_s113
                                 + 0.964626 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.977664 w_k1_ADM_s114
                                 + 1.39213 w_k1_AXP_s114
                                 + 1.29332 w_k1_DIS_s114
                                 + 1.18502 w_k1_FDX_s114
                                 + 1.46355 w_k1_IBM_s114
                                 + 1.00977 w_k1_INTC_s114
                                 + 1.24095 w_k1_KO_s114 + 1.92486 w_k1_MAT_s114
                                 + 1.19829 w_k1_MSFT_s114
                                 + 1.19177 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.1331 w_k1_ADM_s114
                                 + 1.06151 w_k1_AXP_s114
                                 + 0.932066 w_k1_DIS_s114
                                 + 1.14221 w_k1_FDX_s114
                                 + 0.740857 w_k1_IBM_s114
                                 + 0.899304 w_k1_INTC_s114
                                 + 0.789538 w_k1_KO_s114
                                 + 1.06374 w_k1_MAT_s114
                                 + 0.970295 w_k1_MSFT_s114
                                 + 0.938045 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.17354 w_k1_ADM_s114
                                 + 0.662545 w_k1_AXP_s114
                                 + 0.910528 w_k1_DIS_s114
                                 + 1.00429 w_k1_FDX_s114
                                 + 0.715533 w_k1_IBM_s114
                                 + 0.840745 w_k1_INTC_s114
                                 + 0.876573 w_k1_KO_s114
                                 + 0.538632 w_k1_MAT_s114
                                 + 0.945525 w_k1_MSFT_s114
                                 + 0.911332 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.834588 w_k1_ADM_s114
                                 + 0.800058 w_k1_AXP_s114
                                 + 0.782221 w_k1_DIS_s114
                                 + 0.832126 w_k1_FDX_s114
                                 + 1.15052 w_k1_IBM_s114
                                 + 0.653778 w_k1_INTC_s114
                                 + 0.926552 w_k1_KO_s114 + 1.5986 w_k1_MAT_s114
                                 + 0.856464 w_k1_MSFT_s114
                                 + 0.965106 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.82537 w_k1_ADM_s114
                                 + 0.930001 w_k1_AXP_s114
                                 + 0.875859 w_k1_DIS_s114
                                 + 0.960727 w_k1_FDX_s114
                                 + 0.952818 w_k1_IBM_s114
                                 + 0.872115 w_k1_INTC_s114
                                 + 0.866491 w_k1_KO_s114
                                 + 1.58235 w_k1_MAT_s114
                                 + 0.987811 w_k1_MSFT_s114
                                 + 0.88712 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.741987 w_k1_ADM_s114
                                 + 0.740633 w_k1_AXP_s114
                                 + 0.653626 w_k1_DIS_s114
                                 + 0.839183 w_k1_FDX_s114
                                 + 0.682308 w_k1_IBM_s114
                                 + 0.66399 w_k1_INTC_s114
                                 + 0.689798 w_k1_KO_s114
                                 + 0.851958 w_k1_MAT_s114
                                 + 0.735158 w_k1_MSFT_s114
                                 + 0.84573 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.3265 w_k1_ADM_s114
                                 + 1.64987 w_k1_AXP_s114 + 1.6164 w_k1_DIS_s114
                                 + 1.61452 w_k1_FDX_s114
                                 + 1.14846 w_k1_IBM_s114
                                 + 1.44356 w_k1_INTC_s114
                                 + 1.29479 w_k1_KO_s114 + 1.90339 w_k1_MAT_s114
                                 + 1.52721 w_k1_MSFT_s114
                                 + 1.37332 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.20208 w_k1_ADM_s114
                                 + 1.19867 w_k1_AXP_s114
                                 + 1.14884 w_k1_DIS_s114
                                 + 1.16454 w_k1_FDX_s114
                                 + 1.04579 w_k1_IBM_s114
                                 + 0.936912 w_k1_INTC_s114
                                 + 1.14718 w_k1_KO_s114 + 1.20868 w_k1_MAT_s114
                                 + 1.1008 w_k1_MSFT_s114 + 1.18693 w_k1_PG_s114
                                 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.31859 w_k1_ADM_s115
                                 + 1.42014 w_k1_AXP_s115
                                 + 1.67959 w_k1_DIS_s115
                                 + 1.44041 w_k1_FDX_s115
                                 + 1.17925 w_k1_IBM_s115
                                 + 1.25233 w_k1_INTC_s115
                                 + 1.05583 w_k1_KO_s115
                                 + 0.936322 w_k1_MAT_s115
                                 + 1.29202 w_k1_MSFT_s115
                                 + 1.06236 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.02175 w_k1_ADM_s115
                                 + 0.89842 w_k1_AXP_s115
                                 + 0.902426 w_k1_DIS_s115
                                 + 0.885017 w_k1_FDX_s115
                                 + 1.15468 w_k1_IBM_s115
                                 + 0.92642 w_k1_INTC_s115
                                 + 1.02037 w_k1_KO_s115
                                 + 0.892787 w_k1_MAT_s115
                                 + 0.797152 w_k1_MSFT_s115
                                 + 1.12077 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.16481 w_k1_ADM_s115
                                 + 1.28333 w_k1_AXP_s115
                                 + 1.20694 w_k1_DIS_s115
                                 + 1.21981 w_k1_FDX_s115
                                 + 1.01665 w_k1_IBM_s115
                                 + 1.16225 w_k1_INTC_s115
                                 + 1.04922 w_k1_KO_s115 + 0.75837 w_k1_MAT_s115
                                 + 1.0641 w_k1_MSFT_s115 + 1.2807 w_k1_PG_s115
                                 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.12244 w_k1_ADM_s115
                                 + 1.49435 w_k1_AXP_s115
                                 + 1.06719 w_k1_DIS_s115
                                 + 1.01344 w_k1_FDX_s115
                                 + 1.30269 w_k1_IBM_s115
                                 + 0.937958 w_k1_INTC_s115
                                 + 1.1535 w_k1_KO_s115 + 1.3873 w_k1_MAT_s115
                                 + 1.01906 w_k1_MSFT_s115
                                 + 1.31844 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.02333 w_k1_ADM_s115
                                 + 0.877339 w_k1_AXP_s115
                                 + 0.920156 w_k1_DIS_s115
                                 + 0.97294 w_k1_FDX_s115
                                 + 0.737708 w_k1_IBM_s115
                                 + 1.12144 w_k1_INTC_s115
                                 + 0.778591 w_k1_KO_s115
                                 + 0.42646 w_k1_MAT_s115
                                 + 0.992679 w_k1_MSFT_s115
                                 + 0.92042 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.09233 w_k1_ADM_s115
                                 + 1.18933 w_k1_AXP_s115
                                 + 1.28909 w_k1_DIS_s115
                                 + 1.04696 w_k1_FDX_s115
                                 + 1.50775 w_k1_IBM_s115
                                 + 1.07262 w_k1_INTC_s115
                                 + 1.33933 w_k1_KO_s115 + 1.31911 w_k1_MAT_s115
                                 + 1.09348 w_k1_MSFT_s115
                                 + 1.07192 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.916322 w_k1_ADM_s115
                                 + 0.797798 w_k1_AXP_s115
                                 + 0.592207 w_k1_DIS_s115
                                 + 0.790218 w_k1_FDX_s115
                                 + 0.78133 w_k1_IBM_s115
                                 + 0.776264 w_k1_INTC_s115
                                 + 0.708092 w_k1_KO_s115
                                 + 0.70821 w_k1_MAT_s115
                                 + 0.74932 w_k1_MSFT_s115
                                 + 0.79243 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.987405 w_k1_ADM_s115
                                 + 0.917579 w_k1_AXP_s115
                                 + 0.725789 w_k1_DIS_s115
                                 + 0.825379 w_k1_FDX_s115
                                 + 1.00666 w_k1_IBM_s115
                                 + 0.695865 w_k1_INTC_s115
                                 + 0.865163 w_k1_KO_s115
                                 + 0.83586 w_k1_MAT_s115
                                 + 0.791146 w_k1_MSFT_s115
                                 + 0.921451 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.13951 w_k1_ADM_s116
                                 + 1.37157 w_k1_AXP_s116
                                 + 1.57354 w_k1_DIS_s116 + 1.3433 w_k1_FDX_s116
                                 + 0.897451 w_k1_IBM_s116
                                 + 1.52875 w_k1_INTC_s116
                                 + 1.08753 w_k1_KO_s116
                                 + 0.893454 w_k1_MAT_s116
                                 + 1.38918 w_k1_MSFT_s116
                                 + 1.05493 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.768653 w_k1_ADM_s116
                                 + 1.08584 w_k1_AXP_s116
                                 + 0.949283 w_k1_DIS_s116
                                 + 0.938531 w_k1_FDX_s116
                                 + 0.927001 w_k1_IBM_s116
                                 + 1.11313 w_k1_INTC_s116
                                 + 0.929829 w_k1_KO_s116
                                 + 0.807086 w_k1_MAT_s116
                                 + 0.95613 w_k1_MSFT_s116
                                 + 0.94286 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.86889 w_k1_ADM_s116
                                 + 1.62171 w_k1_AXP_s116
                                 + 1.02941 w_k1_DIS_s116
                                 + 0.955311 w_k1_FDX_s116
                                 + 1.90021 w_k1_IBM_s116
                                 + 1.20855 w_k1_INTC_s116 + 1.118 w_k1_KO_s116
                                 + 1.79236 w_k1_MAT_s116
                                 + 0.82494 w_k1_MSFT_s116
                                 + 0.964815 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.849888 w_k1_ADM_s116
                                 + 0.912924 w_k1_AXP_s116
                                 + 0.716015 w_k1_DIS_s116
                                 + 0.909726 w_k1_FDX_s116
                                 + 0.555912 w_k1_IBM_s116
                                 + 0.928075 w_k1_INTC_s116
                                 + 0.667659 w_k1_KO_s116
                                 + 0.363668 w_k1_MAT_s116
                                 + 0.854874 w_k1_MSFT_s116
                                 + 0.802116 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.827408 w_k1_ADM_s116
                                 + 0.713697 w_k1_AXP_s116
                                 + 0.752069 w_k1_DIS_s116
                                 + 0.782208 w_k1_FDX_s116
                                 + 0.71715 w_k1_IBM_s116
                                 + 0.73789 w_k1_INTC_s116
                                 + 0.652194 w_k1_KO_s116
                                 + 0.532071 w_k1_MAT_s116
                                 + 0.790826 w_k1_MSFT_s116
                                 + 0.839413 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.30474 w_k1_ADM_s116
                                 + 1.20447 w_k1_AXP_s116
                                 + 1.10233 w_k1_DIS_s116
                                 + 1.21826 w_k1_FDX_s116
                                 + 0.747475 w_k1_IBM_s116
                                 + 1.23887 w_k1_INTC_s116
                                 + 0.852482 w_k1_KO_s116
                                 + 0.605127 w_k1_MAT_s116
                                 + 1.03148 w_k1_MSFT_s116
                                 + 1.03101 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.858387 w_k1_ADM_s116
                                 + 1.3124 w_k1_AXP_s116
                                 + 0.831915 w_k1_DIS_s116
                                 + 0.881742 w_k1_FDX_s116
                                 + 1.26882 w_k1_IBM_s116
                                 + 0.978925 w_k1_INTC_s116
                                 + 1.0919 w_k1_KO_s116 + 1.15521 w_k1_MAT_s116
                                 + 0.777223 w_k1_MSFT_s116
                                 + 0.932503 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.85959 w_k1_ADM_s116
                                 + 1.39144 w_k1_AXP_s116 + 1.4602 w_k1_DIS_s116
                                 + 1.08318 w_k1_FDX_s116
                                 + 1.32624 w_k1_IBM_s116
                                 + 1.19589 w_k1_INTC_s116
                                 + 1.23574 w_k1_KO_s116 + 1.25091 w_k1_MAT_s116
                                 + 1.06298 w_k1_MSFT_s116 + 1.1421 w_k1_PG_s116
                                 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.72274 w_k1_ADM_s117
                                 + 1.44254 w_k1_AXP_s117
                                 + 1.53643 w_k1_DIS_s117
                                 + 1.46012 w_k1_FDX_s117
                                 + 1.19504 w_k1_IBM_s117
                                 + 1.2957 w_k1_INTC_s117 + 1.14072 w_k1_KO_s117
                                 + 1.0523 w_k1_MAT_s117
                                 + 1.32181 w_k1_MSFT_s117
                                 + 1.26636 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.19741 w_k1_ADM_s117
                                 + 0.877423 w_k1_AXP_s117
                                 + 0.901685 w_k1_DIS_s117
                                 + 1.06027 w_k1_FDX_s117
                                 + 0.739426 w_k1_IBM_s117
                                 + 0.922478 w_k1_INTC_s117
                                 + 0.769371 w_k1_KO_s117
                                 + 0.758876 w_k1_MAT_s117
                                 + 1.01011 w_k1_MSFT_s117
                                 + 0.985501 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.920797 w_k1_ADM_s117
                                 + 0.955838 w_k1_AXP_s117
                                 + 0.60195 w_k1_DIS_s117
                                 + 0.749503 w_k1_FDX_s117
                                 + 1.06859 w_k1_IBM_s117
                                 + 0.855758 w_k1_INTC_s117
                                 + 0.727426 w_k1_KO_s117
                                 + 0.952659 w_k1_MAT_s117
                                 + 0.757175 w_k1_MSFT_s117
                                 + 0.815098 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.852453 w_k1_ADM_s117
                                 + 0.611435 w_k1_AXP_s117
                                 + 0.534271 w_k1_DIS_s117
                                 + 0.688082 w_k1_FDX_s117
                                 + 0.599198 w_k1_IBM_s117
                                 + 0.63949 w_k1_INTC_s117
                                 + 0.635035 w_k1_KO_s117
                                 + 0.773022 w_k1_MAT_s117
                                 + 0.859643 w_k1_MSFT_s117
                                 + 0.770807 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.03936 w_k1_ADM_s117
                                 + 0.895435 w_k1_AXP_s117
                                 + 0.827804 w_k1_DIS_s117
                                 + 0.793525 w_k1_FDX_s117
                                 + 1.01575 w_k1_IBM_s117
                                 + 0.822808 w_k1_INTC_s117
                                 + 0.944315 w_k1_KO_s117
                                 + 1.45561 w_k1_MAT_s117
                                 + 0.841601 w_k1_MSFT_s117
                                 + 0.995196 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.12444 w_k1_ADM_s117
                                 + 0.997553 w_k1_AXP_s117
                                 + 0.998843 w_k1_DIS_s117
                                 + 0.88609 w_k1_FDX_s117
                                 + 0.995512 w_k1_IBM_s117
                                 + 0.888952 w_k1_INTC_s117
                                 + 1.06739 w_k1_KO_s117 + 1.09381 w_k1_MAT_s117
                                 + 1.00421 w_k1_MSFT_s117
                                 + 1.02577 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.12524 w_k1_ADM_s117
                                 + 1.44117 w_k1_AXP_s117
                                 + 1.16564 w_k1_DIS_s117
                                 + 1.10429 w_k1_FDX_s117
                                 + 1.31848 w_k1_IBM_s117
                                 + 1.07092 w_k1_INTC_s117
                                 + 1.09417 w_k1_KO_s117 + 1.82161 w_k1_MAT_s117
                                 + 1.2003 w_k1_MSFT_s117 + 1.13615 w_k1_PG_s117
                                 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.960722 w_k1_ADM_s117
                                 + 0.849977 w_k1_AXP_s117
                                 + 1.34486 w_k1_DIS_s117
                                 + 1.03368 w_k1_FDX_s117
                                 + 1.41167 w_k1_IBM_s117
                                 + 0.925675 w_k1_INTC_s117
                                 + 1.24822 w_k1_KO_s117 + 1.51245 w_k1_MAT_s117
                                 + 1.09441 w_k1_MSFT_s117
                                 + 1.04518 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.00798195298835909 w_k1_ADM_s000
                                 + 0.0230471521525802 w_k1_ADM_s001
                                 + 0.0224101997752819 w_k1_ADM_s002
                                 + 0.00612479687197147 w_k1_ADM_s003
                                 + 0.00911697712363579 w_k1_ADM_s004
                                 + 0.00638119335694221 w_k1_ADM_s005
                                 + 0.0135229252650074 w_k1_ADM_s006
                                 + 0.016311869507996 w_k1_ADM_s007
                                 + 0.00837231378487876 w_k1_ADM_s010
                                 + 0.0137775830963598 w_k1_ADM_s011
                                 + 0.0331060405320952 w_k1_ADM_s012
                                 + 0.0191026523523874 w_k1_ADM_s013
                                 + 0.0217363586884716 w_k1_ADM_s014
                                 + 0.026248502780221 w_k1_ADM_s015
                                 + 0.0124155878203648 w_k1_ADM_s016
                                 + 0.0250627244407427 w_k1_ADM_s017
                                 + 0.0422595765626938 w_k1_ADM_s100
                                 + 0.0808096555158926 w_k1_ADM_s101
                                 + 0.0352190537451255 w_k1_ADM_s102
                                 + 0.113980636887346 w_k1_ADM_s103
                                 + 0.0358046341601099 w_k1_ADM_s104
                                 + 0.0339888878066249 w_k1_ADM_s105
                                 + 0.0452034554786742 w_k1_ADM_s106
                                 + 0.130982259287908 w_k1_ADM_s107
                                 + 0.0313238853461121 w_k1_ADM_s110
                                 + 0.0241445146924597 w_k1_ADM_s111
                                 + 0.0154972121454352 w_k1_ADM_s112
                                 + 0.0382160401987921 w_k1_ADM_s113
                                 + 0.0271334834016507 w_k1_ADM_s114
                                 + 0.0216653186864084 w_k1_ADM_s115
                                 + 0.0477795324501867 w_k1_ADM_s116
                                 + 0.0507678146884988 w_k1_ADM_s117
                                 + 0.00721347112874538 w_k1_AXP_s000
                                 + 0.0225445556602752 w_k1_AXP_s001
                                 + 0.0207198703585172 w_k1_AXP_s002
                                 + 0.00565888662114114 w_k1_AXP_s003
                                 + 0.00917369183299648 w_k1_AXP_s004
                                 + 0.0071296653573495 w_k1_AXP_s005
                                 + 0.0165438003806071 w_k1_AXP_s006
                                 + 0.0163309619171725 w_k1_AXP_s007
                                 + 0.00983441232297616 w_k1_AXP_s010
                                 + 0.012140998005313 w_k1_AXP_s011
                                 + 0.0348330435195854 w_k1_AXP_s012
                                 + 0.016937400026725 w_k1_AXP_s013
                                 + 0.0227333394380425 w_k1_AXP_s014
                                 + 0.0343551472731852 w_k1_AXP_s015
                                 + 0.01169416978995 w_k1_AXP_s016
                                 + 0.0258218536746641 w_k1_AXP_s017
                                 + 0.0438836268851896 w_k1_AXP_s100
                                 + 0.0754724417288627 w_k1_AXP_s101
                                 + 0.0317643644015019 w_k1_AXP_s102
                                 + 0.120782558288124 w_k1_AXP_s103
                                 + 0.0341231673936822 w_k1_AXP_s104
                                 + 0.0349074579082858 w_k1_AXP_s105
                                 + 0.0498422559867376 w_k1_AXP_s106
                                 + 0.122813742186672 w_k1_AXP_s107
                                 + 0.0307579944553971 w_k1_AXP_s110
                                 + 0.0244863431320549 w_k1_AXP_s111
                                 + 0.0131464809489149 w_k1_AXP_s112
                                 + 0.0424144180398833 w_k1_AXP_s113
                                 + 0.0294801565715106 w_k1_AXP_s114
                                 + 0.0212835693737528 w_k1_AXP_s115
                                 + 0.0637515694962582 w_k1_AXP_s116
                                 + 0.0463919353756071 w_k1_AXP_s117
                                 + 0.00705909145477013 w_k1_DIS_s000
                                 + 0.0244316414746495 w_k1_DIS_s001
                                 + 0.0209532394634677 w_k1_DIS_s002
                                 + 0.00641090707769715 w_k1_DIS_s003
                                 + 0.00872219776575456 w_k1_DIS_s004
                                 + 0.00650804208565515 w_k1_DIS_s005
                                 + 0.0153991206487118 w_k1_DIS_s006
                                 + 0.0163268759947332 w_k1_DIS_s007
                                 + 0.00974162499153013 w_k1_DIS_s010
                                 + 0.0130294015104926 w_k1_DIS_s011
                                 + 0.0356345354104102 w_k1_DIS_s012
                                 + 0.0160023185536332 w_k1_DIS_s013
                                 + 0.0214878149560048 w_k1_DIS_s014
                                 + 0.0319086203550365 w_k1_DIS_s015
                                 + 0.0112743914617031 w_k1_DIS_s016
                                 + 0.0263981029686327 w_k1_DIS_s017
                                 + 0.0385686950955244 w_k1_DIS_s100
                                 + 0.0771138389270992 w_k1_DIS_s101
                                 + 0.0306173358010232 w_k1_DIS_s102
                                 + 0.113822347317542 w_k1_DIS_s103
                                 + 0.0345548035854689 w_k1_DIS_s104
                                 + 0.033778858297132 w_k1_DIS_s105
                                 + 0.0477019486209175 w_k1_DIS_s106
                                 + 0.113571533526625 w_k1_DIS_s107
                                 + 0.0316215456935391 w_k1_DIS_s110
                                 + 0.0246056329803872 w_k1_DIS_s111
                                 + 0.0150764696167919 w_k1_DIS_s112
                                 + 0.0420821616328788 w_k1_DIS_s113
                                 + 0.0280431008588234 w_k1_DIS_s114
                                 + 0.0201892759001359 w_k1_DIS_s115
                                 + 0.0566447853115969 w_k1_DIS_s116
                                 + 0.0451296610361617 w_k1_DIS_s117
                                 + 0.00733107614604937 w_k1_FDX_s000
                                 + 0.0227835834188728 w_k1_FDX_s001
                                 + 0.0204924934464202 w_k1_FDX_s002
                                 + 0.00580644815648681 w_k1_FDX_s003
                                 + 0.00879059559773048 w_k1_FDX_s004
                                 + 0.00612863978344406 w_k1_FDX_s005
                                 + 0.0152643190383112 w_k1_FDX_s006
                                 + 0.0160221614013715 w_k1_FDX_s007
                                 + 0.0100249916891383 w_k1_FDX_s010
                                 + 0.0135755367804653 w_k1_FDX_s011
                                 + 0.0348848812060159 w_k1_FDX_s012
                                 + 0.0155010516090628 w_k1_FDX_s013
                                 + 0.0211787199267797 w_k1_FDX_s014
                                 + 0.0334205506418819 w_k1_FDX_s015
                                 + 0.0106787735307473 w_k1_FDX_s016
                                 + 0.0263269474053192 w_k1_FDX_s017
                                 + 0.0388052573535067 w_k1_FDX_s100
                                 + 0.0723107714792872 w_k1_FDX_s101
                                 + 0.0297439769021819 w_k1_FDX_s102
                                 + 0.116624976116941 w_k1_FDX_s103
                                 + 0.0340874051950474 w_k1_FDX_s104
                                 + 0.0341220163604054 w_k1_FDX_s105
                                 + 0.0470776539702588 w_k1_FDX_s106
                                 + 0.11818289784988 w_k1_FDX_s107
                                 + 0.031811427056758 w_k1_FDX_s110
                                 + 0.0239646453460171 w_k1_FDX_s111
                                 + 0.0139571474560628 w_k1_FDX_s112
                                 + 0.0422427910722826 w_k1_FDX_s113
                                 + 0.029363851876244 w_k1_FDX_s114
                                 + 0.0203005391796433 w_k1_FDX_s115
                                 + 0.0531291633542998 w_k1_FDX_s116
                                 + 0.044405417047162 w_k1_FDX_s117
                                 + 0.00739176206467059 w_k1_IBM_s000
                                 + 0.0227864004510496 w_k1_IBM_s001
                                 + 0.0205036352663124 w_k1_IBM_s002
                                 + 0.00631279922327725 w_k1_IBM_s003
                                 + 0.00867088859875093 w_k1_IBM_s004
                                 + 0.00660772906164818 w_k1_IBM_s005
                                 + 0.0148298096348661 w_k1_IBM_s006
                                 + 0.015853779269082 w_k1_IBM_s007
                                 + 0.00972523643882084 w_k1_IBM_s010
                                 + 0.0128106553173753 w_k1_IBM_s011
                                 + 0.0348303060046722 w_k1_IBM_s012
                                 + 0.0158271078937834 w_k1_IBM_s013
                                 + 0.0232058396331168 w_k1_IBM_s014
                                 + 0.0327598205292421 w_k1_IBM_s015
                                 + 0.0110017768587822 w_k1_IBM_s016
                                 + 0.025726185283586 w_k1_IBM_s017
                                 + 0.0390697958989482 w_k1_IBM_s100
                                 + 0.073466315128514 w_k1_IBM_s101
                                 + 0.0319866382609969 w_k1_IBM_s102
                                 + 0.115484573333593 w_k1_IBM_s103
                                 + 0.0343358007586628 w_k1_IBM_s104
                                 + 0.0341800570497699 w_k1_IBM_s105
                                 + 0.0469624810742107 w_k1_IBM_s106
                                 + 0.114335349059823 w_k1_IBM_s107
                                 + 0.0314840802565858 w_k1_IBM_s110
                                 + 0.0233599449328681 w_k1_IBM_s111
                                 + 0.0144581135475988 w_k1_IBM_s112
                                 + 0.0434560452991483 w_k1_IBM_s113
                                 + 0.0287696938645062 w_k1_IBM_s114
                                 + 0.0214005763963899 w_k1_IBM_s115
                                 + 0.0566854229113952 w_k1_IBM_s116
                                 + 0.0451194184969304 w_k1_IBM_s117
                                 + 0.00817256291951277 w_k1_INTC_s000
                                 + 0.0212387754126859 w_k1_INTC_s001
                                 + 0.0199499746556955 w_k1_INTC_s002
                                 + 0.00709346105825061 w_k1_INTC_s003
                                 + 0.00799611602810887 w_k1_INTC_s004
                                 + 0.005926937812665 w_k1_INTC_s005
                                 + 0.0179783388252157 w_k1_INTC_s006
                                 + 0.0162184867696118 w_k1_INTC_s007
                                 + 0.010494481827972 w_k1_INTC_s010
                                 + 0.0130629230058684 w_k1_INTC_s011
                                 + 0.0344858879164128 w_k1_INTC_s012
                                 + 0.0155678022443413 w_k1_INTC_s013
                                 + 0.0217191849094793 w_k1_INTC_s014
                                 + 0.0358829731317692 w_k1_INTC_s015
                                 + 0.0122640436114143 w_k1_INTC_s016
                                 + 0.0240015157236551 w_k1_INTC_s017
                                 + 0.0417403460498457 w_k1_INTC_s100
                                 + 0.0758962157520537 w_k1_INTC_s101
                                 + 0.031735910413382 w_k1_INTC_s102
                                 + 0.107317502749762 w_k1_INTC_s103
                                 + 0.0342054201239311 w_k1_INTC_s104
                                 + 0.0318453557810128 w_k1_INTC_s105
                                 + 0.0448361514351714 w_k1_INTC_s106
                                 + 0.11101187617412 w_k1_INTC_s107
                                 + 0.0344281226953991 w_k1_INTC_s110
                                 + 0.0224520885717398 w_k1_INTC_s111
                                 + 0.0124582325125697 w_k1_INTC_s112
                                 + 0.0489918906326214 w_k1_INTC_s113
                                 + 0.0244059196809752 w_k1_INTC_s114
                                 + 0.0200979203117996 w_k1_INTC_s115
                                 + 0.0586378238799415 w_k1_INTC_s116
                                 + 0.0418408831638455 w_k1_INTC_s117
                                 + 0.00824644605554423 w_k1_KO_s000
                                 + 0.0224822917740167 w_k1_KO_s001
                                 + 0.0195346427921952 w_k1_KO_s002
                                 + 0.00651242559638822 w_k1_KO_s003
                                 + 0.00814424125419865 w_k1_KO_s004
                                 + 0.00600490193181559 w_k1_KO_s005
                                 + 0.0147414959336118 w_k1_KO_s006
                                 + 0.0172317646148565 w_k1_KO_s007
                                 + 0.0108672743535516 w_k1_KO_s010
                                 + 0.012408235195244 w_k1_KO_s011
                                 + 0.0353468467755275 w_k1_KO_s012
                                 + 0.0169894054845446 w_k1_KO_s013
                                 + 0.0202472090529509 w_k1_KO_s014
                                 + 0.0325594989903847 w_k1_KO_s015
                                 + 0.0105898827513432 w_k1_KO_s016
                                 + 0.0272106516768772 w_k1_KO_s017
                                 + 0.0407129877398744 w_k1_KO_s100
                                 + 0.0731830954584854 w_k1_KO_s101
                                 + 0.0279702464234445 w_k1_KO_s102
                                 + 0.111525238794891 w_k1_KO_s103
                                 + 0.0327071166679247 w_k1_KO_s104
                                 + 0.0304740398192846 w_k1_KO_s105
                                 + 0.044670585288401 w_k1_KO_s106
                                 + 0.121794322246812 w_k1_KO_s107
                                 + 0.0317054370308539 w_k1_KO_s110
                                 + 0.0232974855773612 w_k1_KO_s111
                                 + 0.0147311475428023 w_k1_KO_s112
                                 + 0.0417171846115952 w_k1_KO_s113
                                 + 0.0274029313898269 w_k1_KO_s114
                                 + 0.0198379618367517 w_k1_KO_s115
                                 + 0.0525744690065946 w_k1_KO_s116
                                 + 0.042304620111128 w_k1_KO_s117
                                 + 0.00560639637216755 w_k1_MAT_s000
                                 + 0.0251381935220493 w_k1_MAT_s001
                                 + 0.0244005659690317 w_k1_MAT_s002
                                 + 0.00677397912234445 w_k1_MAT_s003
                                 + 0.0109160346302433 w_k1_MAT_s004
                                 + 0.0082237969404321 w_k1_MAT_s005
                                 + 0.0162110287051151 w_k1_MAT_s006
                                 + 0.0172544643900842 w_k1_MAT_s007
                                 + 0.00998969154494255 w_k1_MAT_s010
                                 + 0.011262465200421 w_k1_MAT_s011
                                 + 0.0332753145431666 w_k1_MAT_s012
                                 + 0.0185495643188265 w_k1_MAT_s013
                                 + 0.0225709944313467 w_k1_MAT_s014
                                 + 0.0408777299992984 w_k1_MAT_s015
                                 + 0.00949726241576405 w_k1_MAT_s016
                                 + 0.0306392333504845 w_k1_MAT_s017
                                 + 0.0336435833695629 w_k1_MAT_s100
                                 + 0.068106088647804 w_k1_MAT_s101
                                 + 0.0312846174029319 w_k1_MAT_s102
                                 + 0.135658438862956 w_k1_MAT_s103
                                 + 0.0275220723962401 w_k1_MAT_s104
                                 + 0.0354907885244596 w_k1_MAT_s105
                                 + 0.0389636379152759 w_k1_MAT_s106
                                 + 0.121050498899441 w_k1_MAT_s107
                                 + 0.0339448263884874 w_k1_MAT_s110
                                 + 0.0269358283717285 w_k1_MAT_s111
                                 + 0.0154244137995065 w_k1_MAT_s112
                                 + 0.0428895111471621 w_k1_MAT_s113
                                 + 0.0385413675802477 w_k1_MAT_s114
                                 + 0.0172737044498945 w_k1_MAT_s115
                                 + 0.0503268376523589 w_k1_MAT_s116
                                 + 0.0513841859940187 w_k1_MAT_s117
                                 + 0.00806700892238441 w_k1_MSFT_s000
                                 + 0.022663629462292 w_k1_MSFT_s001
                                 + 0.0204503087050883 w_k1_MSFT_s002
                                 + 0.00640558955714115 w_k1_MSFT_s003
                                 + 0.00909306973911081 w_k1_MSFT_s004
                                 + 0.00599286811814637 w_k1_MSFT_s005
                                 + 0.015813456143262 w_k1_MSFT_s006
                                 + 0.017219219965881 w_k1_MSFT_s007
                                 + 0.0101464950228961 w_k1_MSFT_s010
                                 + 0.0127471663724468 w_k1_MSFT_s011
                                 + 0.0371192808021437 w_k1_MSFT_s012
                                 + 0.0156015558364875 w_k1_MSFT_s013
                                 + 0.0202326329845583 w_k1_MSFT_s014
                                 + 0.0320399710534588 w_k1_MSFT_s015
                                 + 0.0127712610107102 w_k1_MSFT_s016
                                 + 0.0253568037635024 w_k1_MSFT_s017
                                 + 0.0405798561513123 w_k1_MSFT_s100
                                 + 0.0769238304464107 w_k1_MSFT_s101
                                 + 0.0311636764629829 w_k1_MSFT_s102
                                 + 0.120056812601238 w_k1_MSFT_s103
                                 + 0.0354165864042188 w_k1_MSFT_s104
                                 + 0.0318842680813459 w_k1_MSFT_s105
                                 + 0.0419243744742311 w_k1_MSFT_s106
                                 + 0.106898897595302 w_k1_MSFT_s107
                                 + 0.0285711831107212 w_k1_MSFT_s110
                                 + 0.0218927300433856 w_k1_MSFT_s111
                                 + 0.0143970420171427 w_k1_MSFT_s112
                                 + 0.0470897558204944 w_k1_MSFT_s113
                                 + 0.0281939905757049 w_k1_MSFT_s114
                                 + 0.0193252015638006 w_k1_MSFT_s115
                                 + 0.0509944438037308 w_k1_MSFT_s116
                                 + 0.0456965081449259 w_k1_MSFT_s117
                                 + 0.00750810597965893 w_k1_PG_s000
                                 + 0.0207486316028559 w_k1_PG_s001
                                 + 0.0209992763126275 w_k1_PG_s002
                                 + 0.00583624561378751 w_k1_PG_s003
                                 + 0.00855815944274816 w_k1_PG_s004
                                 + 0.0064964830157836 w_k1_PG_s005
                                 + 0.0157139164656723 w_k1_PG_s006
                                 + 0.0169787614472282 w_k1_PG_s007
                                 + 0.0101197687087513 w_k1_PG_s010
                                 + 0.0131366605835766 w_k1_PG_s011
                                 + 0.0337294812735156 w_k1_PG_s012
                                 + 0.0167237283153689 w_k1_PG_s013
                                 + 0.0203266277373637 w_k1_PG_s014
                                 + 0.0301504881236414 w_k1_PG_s015
                                 + 0.0113938951077366 w_k1_PG_s016
                                 + 0.0259387830585554 w_k1_PG_s017
                                 + 0.041188958159347 w_k1_PG_s100
                                 + 0.0691571659725991 w_k1_PG_s101
                                 + 0.0311348382347527 w_k1_PG_s102
                                 + 0.112476349615201 w_k1_PG_s103
                                 + 0.0371108617116587 w_k1_PG_s104
                                 + 0.0350799457379816 w_k1_PG_s105
                                 + 0.0442062815102075 w_k1_PG_s106
                                 + 0.113826516957103 w_k1_PG_s107
                                 + 0.0299270630982168 w_k1_PG_s110
                                 + 0.0235450887788033 w_k1_PG_s111
                                 + 0.0137821579011639 w_k1_PG_s112
                                 + 0.0429425959967521 w_k1_PG_s113
                                 + 0.028756664298443 w_k1_PG_s114
                                 + 0.0211774703950588 w_k1_PG_s115
                                 + 0.0524392479731998 w_k1_PG_s116
                                 + 0.0451367142897189 w_k1_PG_s117 >= 1.209
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

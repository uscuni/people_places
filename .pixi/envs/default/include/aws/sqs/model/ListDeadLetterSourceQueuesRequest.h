﻿/**
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0.
 */

#pragma once
#include <aws/sqs/SQS_EXPORTS.h>
#include <aws/sqs/SQSRequest.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <utility>

namespace Aws
{
namespace SQS
{
namespace Model
{

  /**
   * <p/><p><h3>See Also:</h3>   <a
   * href="http://docs.aws.amazon.com/goto/WebAPI/sqs-2012-11-05/ListDeadLetterSourceQueuesRequest">AWS
   * API Reference</a></p>
   */
  class ListDeadLetterSourceQueuesRequest : public SQSRequest
  {
  public:
    AWS_SQS_API ListDeadLetterSourceQueuesRequest();

    // Service request name is the Operation name which will send this request out,
    // each operation should has unique request name, so that we can get operation's name from this request.
    // Note: this is not true for response, multiple operations may have the same response name,
    // so we can not get operation's name from response.
    inline virtual const char* GetServiceRequestName() const override { return "ListDeadLetterSourceQueues"; }

    AWS_SQS_API Aws::String SerializePayload() const override;

    AWS_SQS_API Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const override;


    ///@{
    /**
     * <p>The URL of a dead-letter queue.</p> <p>Queue URLs and names are
     * case-sensitive.</p>
     */
    inline const Aws::String& GetQueueUrl() const{ return m_queueUrl; }
    inline bool QueueUrlHasBeenSet() const { return m_queueUrlHasBeenSet; }
    inline void SetQueueUrl(const Aws::String& value) { m_queueUrlHasBeenSet = true; m_queueUrl = value; }
    inline void SetQueueUrl(Aws::String&& value) { m_queueUrlHasBeenSet = true; m_queueUrl = std::move(value); }
    inline void SetQueueUrl(const char* value) { m_queueUrlHasBeenSet = true; m_queueUrl.assign(value); }
    inline ListDeadLetterSourceQueuesRequest& WithQueueUrl(const Aws::String& value) { SetQueueUrl(value); return *this;}
    inline ListDeadLetterSourceQueuesRequest& WithQueueUrl(Aws::String&& value) { SetQueueUrl(std::move(value)); return *this;}
    inline ListDeadLetterSourceQueuesRequest& WithQueueUrl(const char* value) { SetQueueUrl(value); return *this;}
    ///@}

    ///@{
    /**
     * <p>Pagination token to request the next set of results.</p>
     */
    inline const Aws::String& GetNextToken() const{ return m_nextToken; }
    inline bool NextTokenHasBeenSet() const { return m_nextTokenHasBeenSet; }
    inline void SetNextToken(const Aws::String& value) { m_nextTokenHasBeenSet = true; m_nextToken = value; }
    inline void SetNextToken(Aws::String&& value) { m_nextTokenHasBeenSet = true; m_nextToken = std::move(value); }
    inline void SetNextToken(const char* value) { m_nextTokenHasBeenSet = true; m_nextToken.assign(value); }
    inline ListDeadLetterSourceQueuesRequest& WithNextToken(const Aws::String& value) { SetNextToken(value); return *this;}
    inline ListDeadLetterSourceQueuesRequest& WithNextToken(Aws::String&& value) { SetNextToken(std::move(value)); return *this;}
    inline ListDeadLetterSourceQueuesRequest& WithNextToken(const char* value) { SetNextToken(value); return *this;}
    ///@}

    ///@{
    /**
     * <p>Maximum number of results to include in the response. Value range is 1 to
     * 1000. You must set <code>MaxResults</code> to receive a value for
     * <code>NextToken</code> in the response.</p>
     */
    inline int GetMaxResults() const{ return m_maxResults; }
    inline bool MaxResultsHasBeenSet() const { return m_maxResultsHasBeenSet; }
    inline void SetMaxResults(int value) { m_maxResultsHasBeenSet = true; m_maxResults = value; }
    inline ListDeadLetterSourceQueuesRequest& WithMaxResults(int value) { SetMaxResults(value); return *this;}
    ///@}
  private:

    Aws::String m_queueUrl;
    bool m_queueUrlHasBeenSet = false;

    Aws::String m_nextToken;
    bool m_nextTokenHasBeenSet = false;

    int m_maxResults;
    bool m_maxResultsHasBeenSet = false;
  };

} // namespace Model
} // namespace SQS
} // namespace Aws
